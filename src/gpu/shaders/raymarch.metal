/**
 * Unified SDF Raymarching Compute Shader
 *
 * Sphere-traces through a sparse voxel SDF atlas and produces outputs
 * based on the selected mode:
 *   0 - Depth: normalized [0,1] between near/far clip
 *   1 - RGB:   Phong-shaded with material palette
 *   2 - Material: material ID as float
 *   3 - Distance: raw hit distance for LiDAR/ToF
 *
 * Thread grid: (pixel_x, pixel_y, drone_id) for cameras
 *              (ray_idx, drone_id, 1)       for LiDAR/ToF
 *
 * Buffer Layout:
 *   0: sdf_data      - int8  contiguous atlas [brick_count * 512]
 *   1: material_data  - uint8 contiguous atlas [brick_count * 512]
 *   2: brick_indices  - int32 grid [grid_total]
 *   3: pos_x          - float [drone_count]
 *   4: pos_y          - float [drone_count]
 *   5: pos_z          - float [drone_count]
 *   6: quat_w         - float [drone_count]
 *   7: quat_x         - float [drone_count]
 *   8: quat_y         - float [drone_count]
 *   9: quat_z         - float [drone_count]
 *  10: ray_directions  - float4 [ray_count] (xyz = dir, w = 0)
 *  11: output          - float [drone_count * rays_per_drone * floats_per_ray]
 *  12: drone_indices   - uint32 [drone_count] maps thread drone_id -> actual index
 * Constants:
 *  16: WorldParams
 *  17: RaymarchParams
 */

#include <metal_stdlib>
using namespace metal;

#include "sdf_types.h"

/* ============================================================================
 * Material Palette (constant memory)
 * ============================================================================ */

constant float PALETTE_R[16] = MATERIAL_PALETTE_R;
constant float PALETTE_G[16] = MATERIAL_PALETTE_G;
constant float PALETTE_B[16] = MATERIAL_PALETTE_B;

/* Normalized light direction (precomputed from [0.5, 0.3, 1.0]) */
constant float3 LIGHT_DIR = float3(0.43193421f, 0.25916053f, 0.86386843f);

/* ============================================================================
 * Quaternion Rotation
 * ============================================================================ */

inline float3 quat_rotate(float4 q, float3 v) {
    /* q = (w, x, y, z), v = vector
     * result = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v) */
    float3 u = q.xyz;
    float s = q.w;
    return v + 2.0f * cross(u, cross(u, v) + s * v);
}

/* ============================================================================
 * SDF Query (trilinear interpolation with cross-brick boundaries)
 * ============================================================================ */

/**
 * Sample a single dequantized SDF voxel, handling cross-brick boundaries.
 * Voxel coordinates may be outside [0, GPU_BRICK_MASK] — wraps into neighbor.
 */
inline float sdf_sample_voxel(int bx, int by, int bz,
                              int vx, int vy, int vz,
                              constant WorldParams& wp,
                              device const char* sdf_data,
                              device const int* brick_indices) {
    if (vx >= GPU_BRICK_SIZE) { bx++; vx -= GPU_BRICK_SIZE; }
    else if (vx < 0)          { bx--; vx += GPU_BRICK_SIZE; }
    if (vy >= GPU_BRICK_SIZE) { by++; vy -= GPU_BRICK_SIZE; }
    else if (vy < 0)          { by--; vy += GPU_BRICK_SIZE; }
    if (vz >= GPU_BRICK_SIZE) { bz++; vz -= GPU_BRICK_SIZE; }
    else if (vz < 0)          { bz--; vz += GPU_BRICK_SIZE; }

    if (bx < 0 || bx >= (int)wp.grid_x ||
        by < 0 || by >= (int)wp.grid_y ||
        bz < 0 || bz >= (int)wp.grid_z) {
        return wp.sdf_scale;
    }

    uint grid_idx = (uint)bx + (uint)by * wp.stride_y + (uint)bz * wp.stride_z;
    int atlas_idx = brick_indices[grid_idx];

    if (atlas_idx == GPU_BRICK_EMPTY_INDEX || atlas_idx == GPU_BRICK_UNIFORM_OUTSIDE)
        return wp.sdf_scale;
    if (atlas_idx == GPU_BRICK_UNIFORM_INSIDE)
        return -wp.sdf_scale;

    device const char* sdf = sdf_data + (long)atlas_idx * GPU_BRICK_VOXELS;
    int idx = vx + (vy << GPU_BRICK_SHIFT) + (vz << (GPU_BRICK_SHIFT * 2));
    return float(sdf[idx]) * wp.sdf_scale_div_127;
}

inline float sdf_query(float3 pos,
                       constant WorldParams& wp,
                       device const char* sdf_data,
                       device const int* brick_indices) {
    /* World position -> brick coordinates */
    float rel_x = pos.x - wp.world_min_x;
    float rel_y = pos.y - wp.world_min_y;
    float rel_z = pos.z - wp.world_min_z;

    int bx = (int)floor(rel_x * wp.inv_brick_size);
    int by = (int)floor(rel_y * wp.inv_brick_size);
    int bz = (int)floor(rel_z * wp.inv_brick_size);

    /* Bounds check */
    if (bx < 0 || bx >= (int)wp.grid_x ||
        by < 0 || by >= (int)wp.grid_y ||
        bz < 0 || bz >= (int)wp.grid_z) {
        return wp.sdf_scale;
    }

    /* Get atlas index from grid */
    uint grid_idx = (uint)bx + (uint)by * wp.stride_y + (uint)bz * wp.stride_z;
    int atlas_idx = brick_indices[grid_idx];

    /* Sentinel handling */
    if (atlas_idx == GPU_BRICK_EMPTY_INDEX || atlas_idx == GPU_BRICK_UNIFORM_OUTSIDE) {
        return wp.sdf_scale;
    }
    if (atlas_idx == GPU_BRICK_UNIFORM_INSIDE) {
        return -wp.sdf_scale;
    }

    /* Compute local voxel coordinates */
    float brick_ox = wp.world_min_x + float(bx) * wp.brick_size_world;
    float brick_oy = wp.world_min_y + float(by) * wp.brick_size_world;
    float brick_oz = wp.world_min_z + float(bz) * wp.brick_size_world;

    float local_x = (pos.x - brick_ox) * wp.inv_voxel_size;
    float local_y = (pos.y - brick_oy) * wp.inv_voxel_size;
    float local_z = (pos.z - brick_oz) * wp.inv_voxel_size;

    int x0 = (int)floor(local_x);
    int y0 = (int)floor(local_y);
    int z0 = (int)floor(local_z);

    float fx = local_x - float(x0);
    float fy = local_y - float(y0);
    float fz = local_z - float(z0);

    float c000, c100, c010, c110, c001, c101, c011, c111;

    /* Check if interpolation crosses brick boundary */
    if (x0 < 0 || x0 > GPU_BRICK_MASK - 1 ||
        y0 < 0 || y0 > GPU_BRICK_MASK - 1 ||
        z0 < 0 || z0 > GPU_BRICK_MASK - 1) {
        /* Slow path: cross-brick interpolation */
        int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
        c000 = sdf_sample_voxel(bx, by, bz, x0, y0, z0, wp, sdf_data, brick_indices);
        c100 = sdf_sample_voxel(bx, by, bz, x1, y0, z0, wp, sdf_data, brick_indices);
        c010 = sdf_sample_voxel(bx, by, bz, x0, y1, z0, wp, sdf_data, brick_indices);
        c110 = sdf_sample_voxel(bx, by, bz, x1, y1, z0, wp, sdf_data, brick_indices);
        c001 = sdf_sample_voxel(bx, by, bz, x0, y0, z1, wp, sdf_data, brick_indices);
        c101 = sdf_sample_voxel(bx, by, bz, x1, y0, z1, wp, sdf_data, brick_indices);
        c011 = sdf_sample_voxel(bx, by, bz, x0, y1, z1, wp, sdf_data, brick_indices);
        c111 = sdf_sample_voxel(bx, by, bz, x1, y1, z1, wp, sdf_data, brick_indices);
    } else {
        /* Fast path: all 8 corners within same brick */
        int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;

        int idx_y0 = y0 << GPU_BRICK_SHIFT;
        int idx_y1 = y1 << GPU_BRICK_SHIFT;
        int idx_z0 = z0 << (GPU_BRICK_SHIFT * 2);
        int idx_z1 = z1 << (GPU_BRICK_SHIFT * 2);

        device const char* sdf = sdf_data + (long)atlas_idx * GPU_BRICK_VOXELS;

        float scale = wp.sdf_scale_div_127;
        c000 = float(sdf[x0 + idx_y0 + idx_z0]) * scale;
        c100 = float(sdf[x1 + idx_y0 + idx_z0]) * scale;
        c010 = float(sdf[x0 + idx_y1 + idx_z0]) * scale;
        c110 = float(sdf[x1 + idx_y1 + idx_z0]) * scale;
        c001 = float(sdf[x0 + idx_y0 + idx_z1]) * scale;
        c101 = float(sdf[x1 + idx_y0 + idx_z1]) * scale;
        c011 = float(sdf[x0 + idx_y1 + idx_z1]) * scale;
        c111 = float(sdf[x1 + idx_y1 + idx_z1]) * scale;
    }

    /* Trilinear interpolation */
    float c00 = c000 + fx * (c100 - c000);
    float c01 = c001 + fx * (c101 - c001);
    float c10 = c010 + fx * (c110 - c010);
    float c11 = c011 + fx * (c111 - c011);

    float c_0 = c00 + fy * (c10 - c00);
    float c_1 = c01 + fy * (c11 - c01);

    return c_0 + fz * (c_1 - c_0);
}

/* ============================================================================
 * SDF Gradient (central differences for normal computation)
 * ============================================================================ */

inline float3 sdf_gradient(float3 pos,
                           constant WorldParams& wp,
                           device const char* sdf_data,
                           device const int* brick_indices) {
    float eps = RAYMARCH_NORMAL_EPSILON;

    float dx = sdf_query(pos + float3(eps, 0, 0), wp, sdf_data, brick_indices) -
               sdf_query(pos - float3(eps, 0, 0), wp, sdf_data, brick_indices);
    float dy = sdf_query(pos + float3(0, eps, 0), wp, sdf_data, brick_indices) -
               sdf_query(pos - float3(0, eps, 0), wp, sdf_data, brick_indices);
    float dz = sdf_query(pos + float3(0, 0, eps), wp, sdf_data, brick_indices) -
               sdf_query(pos - float3(0, 0, eps), wp, sdf_data, brick_indices);

    float inv_eps2 = 0.5f / eps;
    return float3(dx, dy, dz) * inv_eps2;
}

/* ============================================================================
 * Material Query (nearest-neighbor, no interpolation)
 * ============================================================================ */

inline uint material_query(float3 pos,
                          constant WorldParams& wp,
                          device const uchar* material_data,
                          device const int* brick_indices) {
    float rel_x = pos.x - wp.world_min_x;
    float rel_y = pos.y - wp.world_min_y;
    float rel_z = pos.z - wp.world_min_z;

    int bx = (int)floor(rel_x * wp.inv_brick_size);
    int by = (int)floor(rel_y * wp.inv_brick_size);
    int bz = (int)floor(rel_z * wp.inv_brick_size);

    if (bx < 0 || bx >= (int)wp.grid_x ||
        by < 0 || by >= (int)wp.grid_y ||
        bz < 0 || bz >= (int)wp.grid_z) {
        return 0;
    }

    uint grid_idx = (uint)bx + (uint)by * wp.stride_y + (uint)bz * wp.stride_z;
    int atlas_idx = brick_indices[grid_idx];

    if (atlas_idx < 0) return 0;

    float brick_ox = wp.world_min_x + float(bx) * wp.brick_size_world;
    float brick_oy = wp.world_min_y + float(by) * wp.brick_size_world;
    float brick_oz = wp.world_min_z + float(bz) * wp.brick_size_world;

    int vx = (int)floor((pos.x - brick_ox) * wp.inv_voxel_size);
    int vy = (int)floor((pos.y - brick_oy) * wp.inv_voxel_size);
    int vz = (int)floor((pos.z - brick_oz) * wp.inv_voxel_size);

    vx = clamp(vx, 0, GPU_BRICK_MASK);
    vy = clamp(vy, 0, GPU_BRICK_MASK);
    vz = clamp(vz, 0, GPU_BRICK_MASK);

    uint voxel_idx = (uint)vx + ((uint)vy << GPU_BRICK_SHIFT) +
                     ((uint)vz << (GPU_BRICK_SHIFT * 2));

    device const uchar* mat = material_data + (long)atlas_idx * GPU_BRICK_VOXELS;
    return (uint)mat[voxel_idx];
}

/* ============================================================================
 * World Bounds Check
 * ============================================================================ */

inline bool world_contains(float3 pos, constant WorldParams& wp) {
    float max_x = wp.world_min_x + float(wp.grid_x) * wp.brick_size_world;
    float max_y = wp.world_min_y + float(wp.grid_y) * wp.brick_size_world;
    float max_z = wp.world_min_z + float(wp.grid_z) * wp.brick_size_world;
    return pos.x >= wp.world_min_x && pos.x < max_x &&
           pos.y >= wp.world_min_y && pos.y < max_y &&
           pos.z >= wp.world_min_z && pos.z < max_z;
}

/* ============================================================================
 * Unified Raymarch Kernel
 * ============================================================================ */

kernel void raymarch_unified(
    device const char*     sdf_data       [[buffer(0)]],
    device const uchar*    material_data  [[buffer(1)]],
    device const int*      brick_indices  [[buffer(2)]],
    device const float*    pos_x          [[buffer(3)]],
    device const float*    pos_y          [[buffer(4)]],
    device const float*    pos_z          [[buffer(5)]],
    device const float*    quat_w         [[buffer(6)]],
    device const float*    quat_x         [[buffer(7)]],
    device const float*    quat_y         [[buffer(8)]],
    device const float*    quat_z         [[buffer(9)]],
    device const float4*   ray_directions [[buffer(10)]],
    device float*          output         [[buffer(11)]],
    device const uint*     drone_indices  [[buffer(12)]],
    constant WorldParams&  world_params   [[buffer(16)]],
    constant RaymarchParams& ray_params   [[buffer(17)]],
    uint3                  gid            [[thread_position_in_grid]])
{
    /* Thread mapping:
     * gid.x = pixel_x (or ray_idx for LiDAR)
     * gid.y = pixel_y (or drone_idx for LiDAR 1D)
     * gid.z = drone_idx (for cameras)
     */
    uint pixel_x = gid.x;
    uint pixel_y = gid.y;
    uint drone_thread = gid.z;

    /* Bounds check */
    if (pixel_x >= ray_params.image_width ||
        pixel_y >= ray_params.image_height ||
        drone_thread >= ray_params.drone_count) {
        return;
    }

    /* Get actual drone index */
    uint drone_idx = drone_indices[drone_thread];

    /* Get drone pose */
    float3 drone_pos = float3(pos_x[drone_idx], pos_y[drone_idx], pos_z[drone_idx]);
    /* float4 layout: (x, y, z, w) so that q.xyz = vector part, q.w = scalar part */
    float4 q = float4(quat_x[drone_idx], quat_y[drone_idx],
                       quat_z[drone_idx], quat_w[drone_idx]);

    /* Get ray direction */
    uint ray_idx = pixel_y * ray_params.image_width + pixel_x;
    float3 local_dir = ray_directions[ray_idx].xyz;
    float3 world_dir = quat_rotate(q, local_dir);

    /* Sensor position (no offset for now - offset applied by caller if needed) */
    float3 origin = drone_pos;

    /* ---- Sphere tracing ---- */
    float t = 0.0f;
    bool hit = false;
    float3 hit_pos;

    for (uint step = 0; step < ray_params.max_steps; step++) {
        float3 pos = origin + world_dir * t;

        /* Per-thread bounds check (preserves original accuracy) */
        if (!world_contains(pos, world_params) && t > 0.0f) {
            break;
        }

        float dist = sdf_query(pos, world_params, sdf_data, brick_indices);

        /* Hit detection */
        if (dist < ray_params.hit_dist) {
            hit = true;
            hit_pos = pos;
            break;
        }

        /* Per-thread max distance check */
        if (t > ray_params.max_distance) {
            break;
        }

        /* Step forward */
        t += max(dist, ray_params.epsilon);

        /* SIMD-group early exit: when all threads in the group are done,
         * skip remaining iterations. Helps when most rays are sky/miss. */
        if (simd_all(t > ray_params.max_distance)) break;
    }

    /* ---- Output based on mode ---- */
    uint pixels_per_drone = ray_params.image_width * ray_params.image_height;
    uint out_idx;

    if (ray_params.output_mode == OUTPUT_MODE_DEPTH) {
        out_idx = drone_thread * pixels_per_drone + ray_idx;

        float depth = 1.0f;
        if (hit && t >= ray_params.near_clip) {
            depth = (t - ray_params.near_clip) * ray_params.inv_depth_range;
            depth = clamp(depth, 0.0f, 1.0f);
        }
        output[out_idx] = depth;
    }
    else if (ray_params.output_mode == OUTPUT_MODE_RGB) {
        out_idx = drone_thread * pixels_per_drone * 3 + ray_idx * 3;

        if (hit && t >= ray_params.near_clip) {
            /* Get material and normal */
            uint mat_id = material_query(hit_pos, world_params, material_data, brick_indices);
            mat_id = min(mat_id, 15u);

            float3 normal = normalize(sdf_gradient(hit_pos, world_params, sdf_data, brick_indices));

            /* Phong shading */
            float ndotl = max(dot(normal, LIGHT_DIR), 0.0f);
            float lighting = AMBIENT + DIFFUSE * ndotl;

            output[out_idx + 0] = PALETTE_R[mat_id] * lighting;
            output[out_idx + 1] = PALETTE_G[mat_id] * lighting;
            output[out_idx + 2] = PALETTE_B[mat_id] * lighting;
        } else {
            output[out_idx + 0] = SKY_COLOR_R;
            output[out_idx + 1] = SKY_COLOR_G;
            output[out_idx + 2] = SKY_COLOR_B;
        }
    }
    else if (ray_params.output_mode == OUTPUT_MODE_MATERIAL) {
        out_idx = drone_thread * pixels_per_drone + ray_idx;

        if (hit && t >= ray_params.near_clip) {
            uint mat_id = material_query(hit_pos, world_params, material_data, brick_indices);
            output[out_idx] = float(mat_id);
        } else {
            output[out_idx] = 0.0f;
        }
    }
    else if (ray_params.output_mode == OUTPUT_MODE_DEPTH_NORMAL) {
        out_idx = drone_thread * pixels_per_drone * 4 + ray_idx * 4;

        if (hit && t >= ray_params.near_clip) {
            float depth = (t - ray_params.near_clip) * ray_params.inv_depth_range;
            depth = clamp(depth, 0.0f, 1.0f);
            float3 normal = normalize(sdf_gradient(hit_pos, world_params, sdf_data, brick_indices));
            output[out_idx + 0] = depth;
            output[out_idx + 1] = normal.x;
            output[out_idx + 2] = normal.y;
            output[out_idx + 3] = normal.z;
        } else {
            output[out_idx + 0] = 1.0f;
            output[out_idx + 1] = 0.0f;
            output[out_idx + 2] = 0.0f;
            output[out_idx + 3] = 0.0f;
        }
    }
    else { /* OUTPUT_MODE_DISTANCE */
        out_idx = drone_thread * pixels_per_drone + ray_idx;

        if (hit) {
            output[out_idx] = t;
        } else {
            output[out_idx] = ray_params.max_distance;
        }
    }
}

/* ============================================================================
 * FP16 Output Kernel Variant
 *
 * Identical raymarching logic but writes half-precision output.
 * Reduces GPU memory bandwidth by 50% for large camera outputs.
 * ============================================================================ */

kernel void raymarch_unified_fp16(
    device const char*     sdf_data       [[buffer(0)]],
    device const uchar*    material_data  [[buffer(1)]],
    device const int*      brick_indices  [[buffer(2)]],
    device const float*    pos_x          [[buffer(3)]],
    device const float*    pos_y          [[buffer(4)]],
    device const float*    pos_z          [[buffer(5)]],
    device const float*    quat_w         [[buffer(6)]],
    device const float*    quat_x         [[buffer(7)]],
    device const float*    quat_y         [[buffer(8)]],
    device const float*    quat_z         [[buffer(9)]],
    device const float4*   ray_directions [[buffer(10)]],
    device half*           output_fp16    [[buffer(11)]],
    device const uint*     drone_indices  [[buffer(12)]],
    constant WorldParams&  world_params   [[buffer(16)]],
    constant RaymarchParams& ray_params   [[buffer(17)]],
    uint3                  gid            [[thread_position_in_grid]])
{
    uint pixel_x = gid.x;
    uint pixel_y = gid.y;
    uint drone_thread = gid.z;

    if (pixel_x >= ray_params.image_width ||
        pixel_y >= ray_params.image_height ||
        drone_thread >= ray_params.drone_count) {
        return;
    }

    uint drone_idx = drone_indices[drone_thread];
    float3 drone_pos = float3(pos_x[drone_idx], pos_y[drone_idx], pos_z[drone_idx]);
    float4 q = float4(quat_x[drone_idx], quat_y[drone_idx],
                       quat_z[drone_idx], quat_w[drone_idx]);

    uint ray_idx = pixel_y * ray_params.image_width + pixel_x;
    float3 local_dir = ray_directions[ray_idx].xyz;
    float3 world_dir = quat_rotate(q, local_dir);
    float3 origin = drone_pos;

    /* Sphere tracing (identical to FP32 variant) */
    float t = 0.0f;
    bool hit = false;
    float3 hit_pos;

    for (uint step = 0; step < ray_params.max_steps; step++) {
        float3 pos = origin + world_dir * t;

        if (!world_contains(pos, world_params) && t > 0.0f) break;

        float dist = sdf_query(pos, world_params, sdf_data, brick_indices);

        if (dist < ray_params.hit_dist) {
            hit = true;
            hit_pos = pos;
            break;
        }

        if (t > ray_params.max_distance) break;

        t += max(dist, ray_params.epsilon);

        if (simd_all(t > ray_params.max_distance)) break;
    }

    /* Output in half precision */
    uint pixels_per_drone = ray_params.image_width * ray_params.image_height;
    uint out_idx;

    if (ray_params.output_mode == OUTPUT_MODE_DEPTH) {
        out_idx = drone_thread * pixels_per_drone + ray_idx;
        float depth = 1.0f;
        if (hit && t >= ray_params.near_clip) {
            depth = (t - ray_params.near_clip) * ray_params.inv_depth_range;
            depth = clamp(depth, 0.0f, 1.0f);
        }
        output_fp16[out_idx] = half(depth);
    }
    else if (ray_params.output_mode == OUTPUT_MODE_RGB) {
        out_idx = drone_thread * pixels_per_drone * 3 + ray_idx * 3;
        if (hit && t >= ray_params.near_clip) {
            uint mat_id = material_query(hit_pos, world_params, material_data, brick_indices);
            mat_id = min(mat_id, 15u);
            float3 normal = normalize(sdf_gradient(hit_pos, world_params, sdf_data, brick_indices));
            float ndotl = max(dot(normal, LIGHT_DIR), 0.0f);
            float lighting = AMBIENT + DIFFUSE * ndotl;
            output_fp16[out_idx + 0] = half(PALETTE_R[mat_id] * lighting);
            output_fp16[out_idx + 1] = half(PALETTE_G[mat_id] * lighting);
            output_fp16[out_idx + 2] = half(PALETTE_B[mat_id] * lighting);
        } else {
            output_fp16[out_idx + 0] = half(SKY_COLOR_R);
            output_fp16[out_idx + 1] = half(SKY_COLOR_G);
            output_fp16[out_idx + 2] = half(SKY_COLOR_B);
        }
    }
    else if (ray_params.output_mode == OUTPUT_MODE_MATERIAL) {
        out_idx = drone_thread * pixels_per_drone + ray_idx;
        if (hit && t >= ray_params.near_clip) {
            uint mat_id = material_query(hit_pos, world_params, material_data, brick_indices);
            output_fp16[out_idx] = half(float(mat_id));
        } else {
            output_fp16[out_idx] = half(0.0f);
        }
    }
    else { /* OUTPUT_MODE_DISTANCE */
        out_idx = drone_thread * pixels_per_drone + ray_idx;
        output_fp16[out_idx] = hit ? half(t) : half(ray_params.max_distance);
    }
}

/* ============================================================================
 * Tiled Raymarch Kernel (Threadgroup Brick Caching)
 *
 * Pre-caches the SDF brick at the threadgroup center before the march loop.
 * This avoids threadgroup barriers inside the loop while still benefiting
 * from shared memory for the common case where nearby pixels access the
 * same brick.
 * ============================================================================ */

#define TILE_CACHE_SLOTS 4

kernel void raymarch_unified_tiled(
    device const char*     sdf_data       [[buffer(0)]],
    device const uchar*    material_data  [[buffer(1)]],
    device const int*      brick_indices  [[buffer(2)]],
    device const float*    pos_x          [[buffer(3)]],
    device const float*    pos_y          [[buffer(4)]],
    device const float*    pos_z          [[buffer(5)]],
    device const float*    quat_w         [[buffer(6)]],
    device const float*    quat_x         [[buffer(7)]],
    device const float*    quat_y         [[buffer(8)]],
    device const float*    quat_z         [[buffer(9)]],
    device const float4*   ray_directions [[buffer(10)]],
    device float*          output         [[buffer(11)]],
    device const uint*     drone_indices  [[buffer(12)]],
    constant WorldParams&  world_params   [[buffer(16)]],
    constant RaymarchParams& ray_params   [[buffer(17)]],
    uint3                  gid            [[thread_position_in_grid]],
    uint3                  lid            [[thread_position_in_threadgroup]],
    uint                   tid_flat       [[thread_index_in_threadgroup]])
{
    /* Threadgroup shared memory for brick caching */
    threadgroup char tile_sdf[TILE_CACHE_SLOTS * GPU_BRICK_VOXELS];
    threadgroup int  tile_atlas_idx[TILE_CACHE_SLOTS];
    threadgroup int  tile_valid_count;

    uint pixel_x = gid.x;
    uint pixel_y = gid.y;
    uint drone_thread = gid.z;

    if (pixel_x >= ray_params.image_width ||
        pixel_y >= ray_params.image_height ||
        drone_thread >= ray_params.drone_count) {
        return;
    }

    uint drone_idx = drone_indices[drone_thread];
    float3 drone_pos = float3(pos_x[drone_idx], pos_y[drone_idx], pos_z[drone_idx]);
    float4 q = float4(quat_x[drone_idx], quat_y[drone_idx],
                       quat_z[drone_idx], quat_w[drone_idx]);

    /* Pre-cache: Load brick at drone position into shared memory.
     * Most initial ray steps will be near the drone, so this covers
     * the common case for the first few march steps. */
    if (tid_flat == 0) {
        tile_valid_count = 0;
        for (int i = 0; i < TILE_CACHE_SLOTS; i++) {
            tile_atlas_idx[i] = -1;
        }

        /* Cache brick at drone position */
        float rel_x = drone_pos.x - world_params.world_min_x;
        float rel_y = drone_pos.y - world_params.world_min_y;
        float rel_z = drone_pos.z - world_params.world_min_z;
        int bx = (int)floor(rel_x * world_params.inv_brick_size);
        int by = (int)floor(rel_y * world_params.inv_brick_size);
        int bz = (int)floor(rel_z * world_params.inv_brick_size);

        if (bx >= 0 && bx < (int)world_params.grid_x &&
            by >= 0 && by < (int)world_params.grid_y &&
            bz >= 0 && bz < (int)world_params.grid_z) {
            uint grid_idx = (uint)bx + (uint)by * world_params.stride_y +
                            (uint)bz * world_params.stride_z;
            int aidx = brick_indices[grid_idx];
            if (aidx >= 0) {
                tile_atlas_idx[0] = aidx;
                tile_valid_count = 1;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Cooperative brick load */
    if (tile_valid_count > 0) {
        int aidx = tile_atlas_idx[0];
        device const char* src = sdf_data + (long)aidx * GPU_BRICK_VOXELS;
        for (uint vi = tid_flat; vi < GPU_BRICK_VOXELS; vi += 64) {
            tile_sdf[vi] = src[vi];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint ray_idx = pixel_y * ray_params.image_width + pixel_x;
    float3 local_dir = ray_directions[ray_idx].xyz;
    float3 world_dir = quat_rotate(q, local_dir);
    float3 origin = drone_pos;

    /* Sphere tracing with tile-cached SDF for the pre-loaded brick */
    float t = 0.0f;
    bool hit = false;
    float3 hit_pos;

    for (uint step = 0; step < ray_params.max_steps; step++) {
        float3 pos = origin + world_dir * t;

        if (!world_contains(pos, world_params) && t > 0.0f) break;

        /* SDF query - check tile cache, fallback to global */
        float dist;
        {
            float rel_x = pos.x - world_params.world_min_x;
            float rel_y = pos.y - world_params.world_min_y;
            float rel_z = pos.z - world_params.world_min_z;

            int bx = (int)floor(rel_x * world_params.inv_brick_size);
            int by = (int)floor(rel_y * world_params.inv_brick_size);
            int bz = (int)floor(rel_z * world_params.inv_brick_size);

            if (bx < 0 || bx >= (int)world_params.grid_x ||
                by < 0 || by >= (int)world_params.grid_y ||
                bz < 0 || bz >= (int)world_params.grid_z) {
                dist = world_params.sdf_scale;
            } else {
                uint grid_idx = (uint)bx + (uint)by * world_params.stride_y +
                                (uint)bz * world_params.stride_z;
                int atlas_idx = brick_indices[grid_idx];

                if (atlas_idx == GPU_BRICK_EMPTY_INDEX ||
                    atlas_idx == GPU_BRICK_UNIFORM_OUTSIDE) {
                    dist = world_params.sdf_scale;
                } else if (atlas_idx == GPU_BRICK_UNIFORM_INSIDE) {
                    dist = -world_params.sdf_scale;
                } else {
                    /* Check if this brick is in our tile cache */
                    bool cached = false;
                    for (int ci = 0; ci < tile_valid_count; ci++) {
                        if (tile_atlas_idx[ci] == atlas_idx) {
                            /* Trilinear from shared memory */
                            threadgroup char* sdf = tile_sdf + ci * GPU_BRICK_VOXELS;
                            float brick_ox = world_params.world_min_x +
                                             float(bx) * world_params.brick_size_world;
                            float brick_oy = world_params.world_min_y +
                                             float(by) * world_params.brick_size_world;
                            float brick_oz = world_params.world_min_z +
                                             float(bz) * world_params.brick_size_world;
                            float lx = (pos.x - brick_ox) * world_params.inv_voxel_size;
                            float ly = (pos.y - brick_oy) * world_params.inv_voxel_size;
                            float lz = (pos.z - brick_oz) * world_params.inv_voxel_size;
                            int x0 = clamp((int)floor(lx), 0, GPU_BRICK_MASK - 1);
                            int y0 = clamp((int)floor(ly), 0, GPU_BRICK_MASK - 1);
                            int z0 = clamp((int)floor(lz), 0, GPU_BRICK_MASK - 1);
                            float fx = lx - float(x0), fy = ly - float(y0), fz = lz - float(z0);
                            int x1 = x0+1, y1 = y0+1, z1 = z0+1;
                            int iy0 = y0<<GPU_BRICK_SHIFT, iy1 = y1<<GPU_BRICK_SHIFT;
                            int iz0 = z0<<(GPU_BRICK_SHIFT*2), iz1 = z1<<(GPU_BRICK_SHIFT*2);
                            float s = world_params.sdf_scale_div_127;
                            float c000 = float(sdf[x0+iy0+iz0])*s;
                            float c100 = float(sdf[x1+iy0+iz0])*s;
                            float c010 = float(sdf[x0+iy1+iz0])*s;
                            float c110 = float(sdf[x1+iy1+iz0])*s;
                            float c001 = float(sdf[x0+iy0+iz1])*s;
                            float c101 = float(sdf[x1+iy0+iz1])*s;
                            float c011 = float(sdf[x0+iy1+iz1])*s;
                            float c111 = float(sdf[x1+iy1+iz1])*s;
                            float c00=c000+fx*(c100-c000), c01=c001+fx*(c101-c001);
                            float c10=c010+fx*(c110-c010), c11=c011+fx*(c111-c011);
                            float c_0=c00+fy*(c10-c00), c_1=c01+fy*(c11-c01);
                            dist = c_0 + fz * (c_1 - c_0);
                            cached = true;
                            break;
                        }
                    }
                    if (!cached) {
                        dist = sdf_query(pos, world_params, sdf_data, brick_indices);
                    }
                }
            }
        }

        if (dist < ray_params.hit_dist) {
            hit = true;
            hit_pos = pos;
            break;
        }

        if (t > ray_params.max_distance) break;

        t += max(dist, ray_params.epsilon);
        if (simd_all(t > ray_params.max_distance)) break;
    }

    /* Output (same as standard kernel) */
    uint pixels_per_drone = ray_params.image_width * ray_params.image_height;
    uint out_idx;

    if (ray_params.output_mode == OUTPUT_MODE_DEPTH) {
        out_idx = drone_thread * pixels_per_drone + ray_idx;
        float depth = 1.0f;
        if (hit && t >= ray_params.near_clip) {
            depth = (t - ray_params.near_clip) * ray_params.inv_depth_range;
            depth = clamp(depth, 0.0f, 1.0f);
        }
        output[out_idx] = depth;
    }
    else if (ray_params.output_mode == OUTPUT_MODE_RGB) {
        out_idx = drone_thread * pixels_per_drone * 3 + ray_idx * 3;
        if (hit && t >= ray_params.near_clip) {
            uint mat_id = material_query(hit_pos, world_params, material_data, brick_indices);
            mat_id = min(mat_id, 15u);
            float3 normal = normalize(sdf_gradient(hit_pos, world_params, sdf_data, brick_indices));
            float ndotl = max(dot(normal, LIGHT_DIR), 0.0f);
            float lighting = AMBIENT + DIFFUSE * ndotl;
            output[out_idx + 0] = PALETTE_R[mat_id] * lighting;
            output[out_idx + 1] = PALETTE_G[mat_id] * lighting;
            output[out_idx + 2] = PALETTE_B[mat_id] * lighting;
        } else {
            output[out_idx + 0] = SKY_COLOR_R;
            output[out_idx + 1] = SKY_COLOR_G;
            output[out_idx + 2] = SKY_COLOR_B;
        }
    }
    else if (ray_params.output_mode == OUTPUT_MODE_MATERIAL) {
        out_idx = drone_thread * pixels_per_drone + ray_idx;
        if (hit && t >= ray_params.near_clip) {
            uint mat_id = material_query(hit_pos, world_params, material_data, brick_indices);
            output[out_idx] = float(mat_id);
        } else {
            output[out_idx] = 0.0f;
        }
    }
    else if (ray_params.output_mode == OUTPUT_MODE_DEPTH_NORMAL) {
        out_idx = drone_thread * pixels_per_drone * 4 + ray_idx * 4;
        if (hit && t >= ray_params.near_clip) {
            float depth = (t - ray_params.near_clip) * ray_params.inv_depth_range;
            depth = clamp(depth, 0.0f, 1.0f);
            float3 normal = normalize(sdf_gradient(hit_pos, world_params, sdf_data, brick_indices));
            output[out_idx + 0] = depth;
            output[out_idx + 1] = normal.x;
            output[out_idx + 2] = normal.y;
            output[out_idx + 3] = normal.z;
        } else {
            output[out_idx + 0] = 1.0f;
            output[out_idx + 1] = 0.0f;
            output[out_idx + 2] = 0.0f;
            output[out_idx + 3] = 0.0f;
        }
    }
    else { /* OUTPUT_MODE_DISTANCE */
        out_idx = drone_thread * pixels_per_drone + ray_idx;
        output[out_idx] = hit ? t : ray_params.max_distance;
    }
}
