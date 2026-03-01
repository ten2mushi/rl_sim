/**
 * GPU Voxelization Driver
 *
 * Orchestrates GPU-accelerated Phase 3 voxelization:
 * 1. Upload BVH + triangle data to GPU
 * 2. Upload surface brick list
 * 3. Dispatch voxelization kernel
 * 4. Read back SDF + material data to CPU WorldBrickMap
 *
 * Phases 1-2 (coarse/fine classification) remain on CPU since they're
 * O(bricks) queries, while Phase 3 is O(bricks * 512) - the bottleneck.
 */

#include "gpu_hal.h"
#include "sdf_types.h"
#include "obj_io.h"
#include "world_brick_map.h"

#include <string.h>
#include <stdio.h>

/* ============================================================================
 * Build VoxelizeParams from World + Options
 * ============================================================================ */

static VoxelizeParams build_voxelize_params(const WorldBrickMap* world,
                                             const MeshBVH* bvh,
                                             const TriangleMesh* mesh,
                                             const VoxelizeOptions* options,
                                             uint32_t num_surface_bricks) {
    VoxelizeParams p = {0};

    p.world_min_x = world->world_min.x;
    p.world_min_y = world->world_min.y;
    p.world_min_z = world->world_min.z;
    p.voxel_size = world->voxel_size;
    p.brick_size_world = world->brick_size_world;
    p.sdf_scale = world->sdf_scale;
    p.inv_sdf_scale = world->inv_sdf_scale;
    p.normal_coherence = bvh->normal_coherence;
    p.avg_normal_x = bvh->avg_normal.x;
    p.avg_normal_y = bvh->avg_normal.y;
    p.avg_normal_z = bvh->avg_normal.z;

    p.grid_x = world->grid_x;
    p.grid_y = world->grid_y;
    p.grid_z = world->grid_z;
    p.node_count = bvh->node_count;
    p.vertex_count = mesh->vertex_count;
    p.face_count = mesh->face_count;

    p.shell_mode = options && options->shell_mode;
    p.preserve_materials = options ? options->preserve_materials : 1;
    p.close_threshold = world->voxel_size * 3.0f;

    if (p.shell_mode) {
        float thickness = (options->shell_thickness > 0.0f)
            ? options->shell_thickness : 2.0f * world->voxel_size;
        p.shell_half_thickness = thickness * 0.5f;
    }

    p.mesh_bbox_min_x = mesh->bbox_min.x;
    p.mesh_bbox_min_y = mesh->bbox_min.y;
    p.mesh_bbox_min_z = mesh->bbox_min.z;
    p.mesh_bbox_max_x = mesh->bbox_max.x;
    p.mesh_bbox_max_y = mesh->bbox_max.y;
    p.mesh_bbox_max_z = mesh->bbox_max.z;
    p.num_surface_bricks = num_surface_bricks;

    return p;
}

/* ============================================================================
 * GPU Voxelization Entry Point
 * ============================================================================ */

GpuResult gpu_voxelize_surface_bricks(GpuDevice* device,
                                       const MeshBVH* bvh,
                                       const TriangleMesh* mesh,
                                       const uint32_t* surface_brick_list,
                                       uint32_t num_surface_bricks,
                                       WorldBrickMap* world,
                                       const void* opts) {
    if (!device || !bvh || !mesh || !world || num_surface_bricks == 0) {
        return GPU_ERROR_INVALID_ARG;
    }

    const VoxelizeOptions* options = (const VoxelizeOptions*)opts;

    /* Init all resources to zero/NULL for uniform cleanup */
    GpuLinearBVH gpu_bvh = {0};
    GpuTriangleData gpu_tris = {0};
    GpuBuffer* brick_list_buf = NULL;
    GpuBuffer* output_sdf = NULL;
    GpuBuffer* output_mat = NULL;
    GpuKernel* kernel = NULL;
    GpuCommandQueue* queue = NULL;
    GpuResult result = GPU_ERROR_NO_MEMORY;

    /* ====================================================================
     * Step 1: Upload BVH and triangle data to GPU
     * ==================================================================== */

    gpu_bvh = gpu_linear_bvh_create(device, bvh);
    if (!gpu_bvh.nodes) {
        fprintf(stderr, "gpu_voxelize: failed to create GPU BVH\n");
        goto cleanup;
    }

    gpu_tris = gpu_triangle_data_create(device, mesh);
    if (!gpu_tris.vertices_x) {
        fprintf(stderr, "gpu_voxelize: failed to create GPU triangle data\n");
        goto cleanup;
    }

    /* ====================================================================
     * Step 2: Upload brick list (packed bx,by,bz per brick)
     * ==================================================================== */

    size_t brick_list_bytes = num_surface_bricks * 3 * sizeof(uint32_t);
    brick_list_buf = gpu_buffer_create(device, brick_list_bytes, GPU_MEMORY_SHARED);
    if (!brick_list_buf) goto cleanup;
    gpu_buffer_upload(brick_list_buf, surface_brick_list, brick_list_bytes, 0);

    /* ====================================================================
     * Step 3: Allocate output buffers
     * ==================================================================== */

    size_t total_voxels = (size_t)num_surface_bricks * BRICK_VOXELS;
    size_t sdf_bytes = total_voxels * sizeof(int8_t);
    size_t mat_bytes = total_voxels * sizeof(uint8_t);

    output_sdf = gpu_buffer_create(device, sdf_bytes, GPU_MEMORY_SHARED);
    output_mat = gpu_buffer_create(device, mat_bytes, GPU_MEMORY_SHARED);
    if (!output_sdf || !output_mat) goto cleanup;

    /* Clear output (default: outside, material 0) */
    memset(gpu_buffer_map(output_sdf), 127, sdf_bytes);
    memset(gpu_buffer_map(output_mat), 0, mat_bytes);

    /* ====================================================================
     * Step 4: Create kernel and bind buffers
     * ==================================================================== */

    kernel = gpu_kernel_create(device, "sdf_voxelize_surface");
    if (!kernel) {
        fprintf(stderr, "gpu_voxelize: failed to create kernel\n");
        result = GPU_ERROR_COMPILE;
        goto cleanup;
    }

    /* Bind buffers matching shader layout */
    gpu_kernel_set_buffer(kernel, 0, gpu_bvh.nodes);
    gpu_kernel_set_buffer(kernel, 1, gpu_bvh.face_indices);
    gpu_kernel_set_buffer(kernel, 2, gpu_tris.vertices_x);
    gpu_kernel_set_buffer(kernel, 3, gpu_tris.vertices_y);
    gpu_kernel_set_buffer(kernel, 4, gpu_tris.vertices_z);
    gpu_kernel_set_buffer(kernel, 5, gpu_tris.face_v);
    gpu_kernel_set_buffer(kernel, 6, gpu_tris.face_mat);
    gpu_kernel_set_buffer(kernel, 7, brick_list_buf);
    gpu_kernel_set_buffer(kernel, 8, output_sdf);
    gpu_kernel_set_buffer(kernel, 9, output_mat);

    VoxelizeParams params = build_voxelize_params(world, bvh, mesh, options,
                                                   num_surface_bricks);
    gpu_kernel_set_constant(kernel, 0, &params, sizeof(VoxelizeParams));

    /* ====================================================================
     * Step 5: Dispatch
     * ==================================================================== */

    queue = gpu_queue_create(device);
    if (!queue) goto cleanup;

    /* Grid: (8, 8, num_surface_bricks), Group: (8, 8, 1) */
    result = gpu_queue_dispatch(queue, kernel,
                                 8, 8, num_surface_bricks,
                                 8, 8, 1);
    if (result != GPU_SUCCESS) {
        fprintf(stderr, "gpu_voxelize: dispatch failed: %s\n",
                gpu_error_string(result));
        goto cleanup;
    }

    result = gpu_queue_wait(queue);
    if (result != GPU_SUCCESS) {
        fprintf(stderr, "gpu_voxelize: wait failed: %s\n",
                gpu_error_string(result));
        goto cleanup;
    }

    /* ====================================================================
     * Step 6: Read back results to CPU WorldBrickMap
     * ==================================================================== */

    {
        const int8_t* sdf_data = (const int8_t*)gpu_buffer_map(output_sdf);
        const uint8_t* mat_data = (const uint8_t*)gpu_buffer_map(output_mat);

        if (!sdf_data || !mat_data) {
            result = GPU_ERROR_NO_MEMORY;
            goto cleanup;
        }

        for (uint32_t i = 0; i < num_surface_bricks; i++) {
            uint32_t bx = surface_brick_list[i * 3 + 0];
            uint32_t by = surface_brick_list[i * 3 + 1];
            uint32_t bz = surface_brick_list[i * 3 + 2];

            /* Allocate brick in WorldBrickMap atlas */
            int32_t atlas_idx = world_alloc_brick(world, (int32_t)bx,
                                                   (int32_t)by, (int32_t)bz);
            if (atlas_idx == BRICK_EMPTY_INDEX) {
                continue; /* Atlas full */
            }

            /* Copy GPU output to atlas pages */
            int8_t* dst_sdf = world_brick_sdf(world, atlas_idx);
            uint8_t* dst_mat = world_brick_material(world, atlas_idx);

            if (dst_sdf && dst_mat) {
                memcpy(dst_sdf, sdf_data + (size_t)i * BRICK_VOXELS,
                       BRICK_VOXELS * sizeof(int8_t));
                memcpy(dst_mat, mat_data + (size_t)i * BRICK_VOXELS,
                       BRICK_VOXELS * sizeof(uint8_t));
            }
        }

        result = GPU_SUCCESS;
    }

cleanup:
    gpu_queue_destroy(queue);
    gpu_kernel_destroy(kernel);
    gpu_buffer_destroy(output_sdf);
    gpu_buffer_destroy(output_mat);
    gpu_buffer_destroy(brick_list_buf);
    gpu_linear_bvh_destroy(&gpu_bvh);
    gpu_triangle_data_destroy(&gpu_tris);

    return result;
}
