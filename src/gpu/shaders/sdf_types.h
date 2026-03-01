/**
 * Shared C/Metal Types for SDF Raymarching
 *
 * This header is included by both C code (gpu_sensor_dispatch.c) and
 * Metal shaders (raymarch.metal). Uses only scalar types for ABI compatibility.
 *
 * No Vec3/float3 - use explicit float fields instead.
 */

#ifndef SDF_TYPES_H
#define SDF_TYPES_H

/* ============================================================================
 * Output Modes
 * ============================================================================ */

#define OUTPUT_MODE_DEPTH    0   /* Normalized depth [0,1] */
#define OUTPUT_MODE_RGB      1   /* Phong-shaded RGB [0,1]^3 */
#define OUTPUT_MODE_MATERIAL 2   /* Material ID as float */
#define OUTPUT_MODE_DISTANCE 3   /* Raw hit distance for LiDAR/ToF */
#define OUTPUT_MODE_DEPTH_NORMAL 4 /* float4(depth, nx, ny, nz) per pixel */

/* ============================================================================
 * Brick Constants (must match world_brick_map.h)
 * ============================================================================ */

#define GPU_BRICK_SIZE        8
#define GPU_BRICK_VOXELS      512
#define GPU_BRICK_SHIFT       3
#define GPU_BRICK_MASK        7

#define GPU_BRICK_EMPTY_INDEX    (-1)
#define GPU_BRICK_UNIFORM_OUTSIDE (-2)
#define GPU_BRICK_UNIFORM_INSIDE  (-3)

/* ============================================================================
 * Raymarch Constants (defaults, overridable via RaymarchParams)
 * ============================================================================ */

#ifndef RAYMARCH_MAX_STEPS
#define RAYMARCH_MAX_STEPS   128
#endif
#ifndef RAYMARCH_EPSILON
#define RAYMARCH_EPSILON     0.001f
#endif
#ifndef RAYMARCH_HIT_DIST
#define RAYMARCH_HIT_DIST    0.01f
#endif

/* ============================================================================
 * WorldParams - SDF grid parameters
 * ============================================================================ */

typedef struct WorldParams {
    float world_min_x;
    float world_min_y;
    float world_min_z;
    float voxel_size;
    float inv_voxel_size;
    float brick_size_world;
    float inv_brick_size;
    float sdf_scale;
    float sdf_scale_div_127;
    unsigned int grid_x;
    unsigned int grid_y;
    unsigned int grid_z;
    unsigned int stride_y;
    unsigned int stride_z;
    float _padding[2];  /* Pad to 16-byte boundary */
} WorldParams;

/* ============================================================================
 * RaymarchParams - Per-dispatch parameters
 * ============================================================================ */

/* Output precision modes */
#define OUTPUT_PRECISION_FP32 0
#define OUTPUT_PRECISION_FP16 1

typedef struct RaymarchParams {
    unsigned int max_steps;
    float epsilon;
    float hit_dist;
    float max_distance;
    float near_clip;
    float far_clip;
    float inv_depth_range;
    unsigned int output_mode;      /* OUTPUT_MODE_* */
    unsigned int image_width;
    unsigned int image_height;
    unsigned int agent_count;
    unsigned int output_precision; /* OUTPUT_PRECISION_FP32 or FP16 */
} RaymarchParams;

/* ============================================================================
 * Material Palette (shared between C and Metal)
 * ============================================================================ */

/* 16 material colors, RGB float [0,1] each
 * Must match MATERIAL_COLORS[16][3] in sensor_camera.c */
#define MATERIAL_PALETTE_R { 0.5f, 0.4f, 0.2f, 0.5f, 0.3f, 0.7f, 0.1f, 0.1f, \
                             0.9f, 0.9f, 0.6f, 0.1f, 0.9f, 0.1f, 0.8f, 0.3f }
#define MATERIAL_PALETTE_G { 0.5f, 0.3f, 0.6f, 0.5f, 0.2f, 0.1f, 0.7f, 0.1f, \
                             0.9f, 0.5f, 0.1f, 0.6f, 0.9f, 0.1f, 0.6f, 0.3f }
#define MATERIAL_PALETTE_B { 0.5f, 0.2f, 0.2f, 0.5f, 0.1f, 0.1f, 0.1f, 0.7f, \
                             0.1f, 0.1f, 0.6f, 0.6f, 0.9f, 0.1f, 0.4f, 0.8f }

/* Phong shading constants */
#define AMBIENT     0.3f
#define DIFFUSE     0.7f

/* Sky color */
#define SKY_COLOR_R  0.5f
#define SKY_COLOR_G  0.7f
#define SKY_COLOR_B  0.9f

/* ============================================================================
 * GPU BVH Node (for voxelization shaders)
 * ============================================================================ */

/**
 * BVH node for GPU traversal. Matches CPU BVHNode fields with explicit floats.
 * Leaf detection: left == right == node_idx (self-reference sentinel).
 * Padded to 48 bytes for alignment.
 */
typedef struct GpuBVHNode {
    float bbox_min_x, bbox_min_y, bbox_min_z;
    float bbox_max_x, bbox_max_y, bbox_max_z;
    unsigned int left, right;
    unsigned int face_start, face_count;
    unsigned int _pad0, _pad1;
} GpuBVHNode;

/* ============================================================================
 * VoxelizeParams - Per-dispatch parameters for GPU voxelization
 * ============================================================================ */

typedef struct VoxelizeParams {
    /* Row 0: world origin + voxel size */
    float world_min_x, world_min_y, world_min_z;
    float voxel_size;
    /* Row 1: scales */
    float brick_size_world, sdf_scale, inv_sdf_scale;
    float normal_coherence;
    /* Row 2: ray direction for inside/outside */
    float avg_normal_x, avg_normal_y, avg_normal_z;
    float shell_half_thickness;
    /* Row 3: grid dimensions */
    unsigned int grid_x, grid_y, grid_z;
    unsigned int node_count;
    /* Row 4: flags */
    unsigned int shell_mode;
    unsigned int preserve_materials;
    unsigned int vertex_count;
    unsigned int face_count;
    /* Row 5: mesh bbox (for closed mesh hack) */
    float close_threshold;
    float mesh_bbox_min_x, mesh_bbox_min_y, mesh_bbox_min_z;
    /* Row 6: mesh bbox max */
    float mesh_bbox_max_x, mesh_bbox_max_y, mesh_bbox_max_z;
    unsigned int num_surface_bricks;
} VoxelizeParams;

#endif /* SDF_TYPES_H */
