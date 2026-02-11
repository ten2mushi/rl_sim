/**
 * GPU Voxelization Shaders
 *
 * Phase 3 of the three-phase sparse voxelization pipeline.
 * For each surface brick, computes per-voxel SDF using:
 *   1. BVH-accelerated closest_point query (unsigned distance + face)
 *   2. BVH-accelerated inside/outside via ray-casting parity
 *   3. Sign * distance → int8 quantization
 *
 * Dispatch: grid=(8, 8, num_surface_bricks), group=(8, 8, 1)
 * Each thread handles a column of 8 voxels (vz=0..7).
 */

#include <metal_stdlib>
using namespace metal;

#include "sdf_types.h"

/* ============================================================================
 * BVH Traversal Stack
 * ============================================================================ */

#define BVH_STACK_SIZE 32

/* ============================================================================
 * Triangle Geometry Functions
 * ============================================================================ */

static inline float3 get_triangle_vertex(const device float* vx,
                                          const device float* vy,
                                          const device float* vz,
                                          uint idx) {
    return float3(vx[idx], vy[idx], vz[idx]);
}

static inline void get_triangle(const device uint* face_v,
                                 const device float* vx,
                                 const device float* vy,
                                 const device float* vz,
                                 uint face_idx,
                                 thread float3& v0,
                                 thread float3& v1,
                                 thread float3& v2) {
    uint base = face_idx * 3;
    uint i0 = face_v[base + 0];
    uint i1 = face_v[base + 1];
    uint i2 = face_v[base + 2];
    v0 = get_triangle_vertex(vx, vy, vz, i0);
    v1 = get_triangle_vertex(vx, vy, vz, i1);
    v2 = get_triangle_vertex(vx, vy, vz, i2);
}

/* ============================================================================
 * Closest Point on Triangle (Voronoi region method)
 * ============================================================================ */

static float3 closest_point_on_triangle(float3 p, float3 a, float3 b, float3 c) {
    float3 ab = b - a;
    float3 ac = c - a;
    float3 ap = p - a;

    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;

    float3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        return a + ab * v;
    }

    float3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        return a + ac * w;
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return a + ab * v + ac * w;
}

/* ============================================================================
 * BVH Closest Point Query
 * Returns unsigned distance and closest face index
 * ============================================================================ */

static float bvh_closest_point_gpu(float3 point,
                                    const device GpuBVHNode* nodes,
                                    const device uint* bvh_face_indices,
                                    const device uint* face_v,
                                    const device float* vx,
                                    const device float* vy,
                                    const device float* vz,
                                    uint node_count,
                                    thread uint& out_face) {
    float best_dist_sq = FLT_MAX;
    out_face = UINT_MAX;

    uint stack[BVH_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0; /* Root */

    while (sp > 0) {
        uint idx = stack[--sp];
        if (idx >= node_count) continue;

        const device GpuBVHNode& node = nodes[idx];

        /* AABB distance check: can this subtree contain a closer point? */
        float3 clamped = clamp(point,
                               float3(node.bbox_min_x, node.bbox_min_y, node.bbox_min_z),
                               float3(node.bbox_max_x, node.bbox_max_y, node.bbox_max_z));
        float dist_sq = length_squared(clamped - point);
        if (dist_sq >= best_dist_sq) continue;

        /* Leaf node? */
        if (node.left == node.right) {
            for (uint i = 0; i < node.face_count; i++) {
                uint f = bvh_face_indices[node.face_start + i];
                float3 v0, v1, v2;
                get_triangle(face_v, vx, vy, vz, f, v0, v1, v2);

                float3 cp = closest_point_on_triangle(point, v0, v1, v2);
                float d2 = length_squared(cp - point);
                if (d2 < best_dist_sq) {
                    best_dist_sq = d2;
                    out_face = f;
                }
            }
        } else {
            if (sp < BVH_STACK_SIZE - 1) {
                stack[sp++] = node.right;
                stack[sp++] = node.left;
            }
        }
    }

    return sqrt(best_dist_sq);
}

/* ============================================================================
 * Ray-Triangle Intersection (Moller-Trumbore)
 * ============================================================================ */

static bool ray_triangle_intersect(float3 origin, float3 dir,
                                    float3 v0, float3 v1, float3 v2,
                                    thread float& t) {
    const float EPS = 1e-8f;

    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 h = cross(dir, e2);
    float a = dot(e1, h);

    if (a > -EPS && a < EPS) return false;

    float f = 1.0f / a;
    float3 s = origin - v0;
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;

    float3 q = cross(s, e1);
    float v = f * dot(dir, q);
    if (v < 0.0f || u + v > 1.0f) return false;

    t = f * dot(e2, q);
    return t > EPS;
}

/* ============================================================================
 * Ray-AABB Intersection
 * ============================================================================ */

static bool ray_aabb_intersect(float3 origin, float3 inv_dir,
                                float3 box_min, float3 box_max) {
    float3 t1 = (box_min - origin) * inv_dir;
    float3 t2 = (box_max - origin) * inv_dir;

    float3 tmin_v = min(t1, t2);
    float3 tmax_v = max(t1, t2);

    float tmin = max(max(tmin_v.x, tmin_v.y), tmin_v.z);
    float tmax = min(min(tmax_v.x, tmax_v.y), tmax_v.z);

    return tmax >= 0.0f && tmin <= tmax;
}

/* ============================================================================
 * Inside/Outside via Ray-Casting Parity
 * Returns +1.0 (outside) or -1.0 (inside)
 * ============================================================================ */

static float bvh_inside_outside_gpu(float3 point,
                                     const device GpuBVHNode* nodes,
                                     const device uint* bvh_face_indices,
                                     const device uint* face_v,
                                     const device float* vx,
                                     const device float* vy,
                                     const device float* vz,
                                     uint node_count,
                                     constant VoxelizeParams& params) {
    /* Select ray direction aligned with mesh's dominant normal.
     * Small off-axis perturbation avoids edge/vertex degeneracies. */
    float ax = abs(params.avg_normal_x);
    float ay = abs(params.avg_normal_y);
    float az = abs(params.avg_normal_z);

    float3 dir;
    if (ay >= ax && ay >= az) {
        dir = float3(0.0013f, (params.avg_normal_y >= 0 ? 1.0f : -1.0f), 0.0027f);
    } else if (ax >= ay && ax >= az) {
        dir = float3((params.avg_normal_x >= 0 ? 1.0f : -1.0f), 0.0013f, 0.0027f);
    } else {
        dir = float3(0.0013f, 0.0027f, (params.avg_normal_z >= 0 ? 1.0f : -1.0f));
    }

    float3 inv_dir = 1.0f / dir;

    /* Count ray crossings */
    uint crossings = 0;

    uint stack[BVH_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        uint idx = stack[--sp];
        if (idx >= node_count) continue;

        const device GpuBVHNode& node = nodes[idx];

        float3 box_min = float3(node.bbox_min_x, node.bbox_min_y, node.bbox_min_z);
        float3 box_max = float3(node.bbox_max_x, node.bbox_max_y, node.bbox_max_z);

        if (!ray_aabb_intersect(point, inv_dir, box_min, box_max)) continue;

        if (node.left == node.right) {
            /* Leaf: test all triangles */
            for (uint i = 0; i < node.face_count; i++) {
                uint f = bvh_face_indices[node.face_start + i];
                float3 v0, v1, v2;
                get_triangle(face_v, vx, vy, vz, f, v0, v1, v2);

                float t;
                if (ray_triangle_intersect(point, dir, v0, v1, v2, t)) {
                    crossings++;
                }
            }
        } else {
            if (sp < BVH_STACK_SIZE - 1) {
                stack[sp++] = node.right;
                stack[sp++] = node.left;
            }
        }
    }

    return (crossings & 1) ? -1.0f : 1.0f;
}

/* ============================================================================
 * Robust Inside/Outside via 3-Ray Majority Voting
 * Returns +1.0 (outside) or -1.0 (inside)
 * More reliable for complex/symmetric surfaces (gyroid)
 * ============================================================================ */

static float bvh_inside_outside_robust_gpu(float3 point,
                                            const device GpuBVHNode* nodes,
                                            const device uint* bvh_face_indices,
                                            const device uint* face_v,
                                            const device float* vx,
                                            const device float* vy,
                                            const device float* vz,
                                            uint node_count) {
    /* Three non-coplanar directions with irrational-looking components (matches CPU) */
    const float3 ROBUST_DIRS[3] = {
        float3( 0.8017f,  0.2673f,  0.5345f),  /* ~(3,1,2)/sqrt(14) */
        float3( 0.2357f,  0.9428f, -0.2357f),  /* ~(1,4,-1)/sqrt(18) */
        float3(-0.3015f, -0.3015f,  0.9045f)   /* ~(-1,-1,3)/sqrt(11) */
    };

    int inside_votes = 0;

    for (int r = 0; r < 3; r++) {
        float3 dir = ROBUST_DIRS[r];
        float3 inv_dir = 1.0f / dir;

        uint crossings = 0;
        uint stack[BVH_STACK_SIZE];
        int sp = 0;
        stack[sp++] = 0;

        while (sp > 0) {
            uint idx = stack[--sp];
            if (idx >= node_count) continue;

            const device GpuBVHNode& node = nodes[idx];

            float3 box_min = float3(node.bbox_min_x, node.bbox_min_y, node.bbox_min_z);
            float3 box_max = float3(node.bbox_max_x, node.bbox_max_y, node.bbox_max_z);

            if (!ray_aabb_intersect(point, inv_dir, box_min, box_max)) continue;

            if (node.left == node.right) {
                for (uint i = 0; i < node.face_count; i++) {
                    uint f = bvh_face_indices[node.face_start + i];
                    float3 v0, v1, v2;
                    get_triangle(face_v, vx, vy, vz, f, v0, v1, v2);

                    float t;
                    if (ray_triangle_intersect(point, dir, v0, v1, v2, t)) {
                        crossings++;
                    }
                }
            } else {
                if (sp < BVH_STACK_SIZE - 1) {
                    stack[sp++] = node.right;
                    stack[sp++] = node.left;
                }
            }
        }

        if (crossings & 1) inside_votes++;
    }

    return (inside_votes >= 2) ? -1.0f : 1.0f;
}

/* ============================================================================
 * SDF Quantization (matches CPU sdf_quantize)
 * ============================================================================ */

static inline char sdf_quantize_gpu(float sdf, float inv_sdf_scale) {
    float val = sdf * inv_sdf_scale * 127.0f;
    val = clamp(val, -127.0f, 127.0f);
    return (char)(int)(val);
}

/* ============================================================================
 * Main Voxelization Kernel
 *
 * Dispatch: grid=(8, 8, num_surface_bricks), group=(8, 8, 1)
 * Each thread: vx = gid.x, vy = gid.y, brick = gid.z
 * Loops over vz=0..7 to fill one voxel column.
 * ============================================================================ */

kernel void sdf_voxelize_surface(
    /* BVH data */
    const device GpuBVHNode*   bvh_nodes       [[buffer(0)]],
    const device uint*         bvh_face_indices [[buffer(1)]],
    /* Triangle mesh data */
    const device float*        vertices_x       [[buffer(2)]],
    const device float*        vertices_y       [[buffer(3)]],
    const device float*        vertices_z       [[buffer(4)]],
    const device uint*         face_v           [[buffer(5)]],
    const device uchar*        face_mat         [[buffer(6)]],
    /* Brick list: packed (bx, by, bz) per brick */
    const device uint*         brick_list       [[buffer(7)]],
    /* Output: SDF and material per voxel */
    device char*               output_sdf       [[buffer(8)]],
    device uchar*              output_mat       [[buffer(9)]],
    /* Parameters */
    constant VoxelizeParams&   params           [[buffer(16)]],
    /* Thread position */
    uint3 gid [[thread_position_in_grid]]
) {
    uint voxel_x = gid.x;
    uint voxel_y = gid.y;
    uint brick_idx = gid.z;

    if (voxel_x >= 8 || voxel_y >= 8 || brick_idx >= params.num_surface_bricks) return;

    /* Get brick coordinates from brick list */
    uint brick_base = brick_idx * 3;
    uint bx = brick_list[brick_base + 0];
    uint by = brick_list[brick_base + 1];
    uint bz = brick_list[brick_base + 2];

    /* Brick origin in world space */
    float3 brick_origin = float3(
        params.world_min_x + float(bx) * params.brick_size_world,
        params.world_min_y + float(by) * params.brick_size_world,
        params.world_min_z + float(bz) * params.brick_size_world
    );

    /* Output base offset for this brick */
    uint brick_out_base = brick_idx * GPU_BRICK_VOXELS;

    /* Process 8 voxels along Z axis */
    for (uint vz = 0; vz < 8; vz++) {
        uint voxel_idx = voxel_x + (voxel_y << GPU_BRICK_SHIFT) +
                         (vz << (GPU_BRICK_SHIFT * 2));

        /* Voxel center in world space */
        float3 voxel_center = float3(
            brick_origin.x + (float(voxel_x) + 0.5f) * params.voxel_size,
            brick_origin.y + (float(voxel_y) + 0.5f) * params.voxel_size,
            brick_origin.z + (float(vz) + 0.5f) * params.voxel_size
        );

        /* Find closest point on mesh */
        uint closest_face;
        float distance = bvh_closest_point_gpu(
            voxel_center,
            bvh_nodes, bvh_face_indices,
            face_v, vertices_x, vertices_y, vertices_z,
            params.node_count, closest_face);

        /* Compute signed distance */
        float signed_dist;
        if (params.shell_mode) {
            signed_dist = distance - params.shell_half_thickness;
        } else {
            /* Near surface: use 3-ray majority voting for robustness
             * against complex/symmetric geometry (gyroid).
             * Far voxels: single ray (faster, sign errors harmless). */
            float sign;
            if (distance < params.voxel_size * 1.5f) {
                sign = bvh_inside_outside_robust_gpu(
                    voxel_center,
                    bvh_nodes, bvh_face_indices,
                    face_v, vertices_x, vertices_y, vertices_z,
                    params.node_count);
            } else {
                sign = bvh_inside_outside_gpu(
                    voxel_center,
                    bvh_nodes, bvh_face_indices,
                    face_v, vertices_x, vertices_y, vertices_z,
                    params.node_count, params);
            }

            /* Closed mesh bbox hack: force positive outside mesh bbox
             * to prevent spurious surfaces in padding regions.
             * Skip for open meshes (terrain) to avoid floor artifact. */
            if (params.normal_coherence <= 0.3f && distance > params.close_threshold) {
                bool outside_bbox =
                    voxel_center.x < params.mesh_bbox_min_x ||
                    voxel_center.x > params.mesh_bbox_max_x ||
                    voxel_center.y < params.mesh_bbox_min_y ||
                    voxel_center.y > params.mesh_bbox_max_y ||
                    voxel_center.z < params.mesh_bbox_min_z ||
                    voxel_center.z > params.mesh_bbox_max_z;
                if (outside_bbox) {
                    sign = 1.0f;
                }
            }

            signed_dist = sign * distance;
        }

        /* Quantize and store */
        output_sdf[brick_out_base + voxel_idx] =
            sdf_quantize_gpu(signed_dist, params.inv_sdf_scale);

        /* Material from closest face */
        if (params.preserve_materials && closest_face != UINT_MAX && signed_dist < 0.0f) {
            output_mat[brick_out_base + voxel_idx] = face_mat[closest_face];
        } else {
            output_mat[brick_out_base + voxel_idx] = 0;
        }
    }
}
