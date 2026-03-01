/**
 * GPU Common Implementation
 *
 * Backend-independent functions: error strings, SDF atlas flattening,
 * drone pose upload, ray table creation, sensor output management.
 *
 * These functions use the GPU HAL API and work with any backend.
 */

#include "gpu_hal.h"
#include "world_brick_map.h"
#include "drone_state.h"
#include <string.h>

/* ============================================================================
 * Section 1: Error Strings
 * ============================================================================ */

const char* gpu_error_string(GpuResult result) {
    switch (result) {
        case GPU_SUCCESS:           return "Success";
        case GPU_ERROR_NO_DEVICE:   return "No GPU device available";
        case GPU_ERROR_NO_MEMORY:   return "GPU memory allocation failed";
        case GPU_ERROR_INVALID_ARG: return "Invalid argument";
        case GPU_ERROR_COMPILE:     return "Shader compilation failed";
        case GPU_ERROR_DISPATCH:    return "Dispatch failed";
        case GPU_ERROR_TIMEOUT:     return "Operation timed out";
        case GPU_ERROR_NOT_READY:   return "Operation not yet complete";
        case GPU_ERROR_BACKEND:     return "No GPU backend available";
        default:                    return "Unknown GPU error";
    }
}

/* ============================================================================
 * Section 2: SDF Atlas Flattening
 * ============================================================================ */

#if GPU_AVAILABLE

GpuSdfAtlas gpu_sdf_atlas_upload(GpuDevice* device,
                                  const struct WorldBrickMap* world) {
    GpuSdfAtlas atlas = {0};

    if (device == NULL || world == NULL) return atlas;

    /* Count active bricks (those with atlas_idx >= 0) */
    uint32_t active_count = world->atlas_count;
    if (active_count == 0) {
        /* Even with no bricks, create the index buffer */
        atlas.grid_total = world->grid_total;
        atlas.brick_indices = gpu_buffer_create(device,
            world->grid_total * sizeof(int32_t), GPU_MEMORY_SHARED);
        if (atlas.brick_indices == NULL) return atlas;

        /* Upload brick indices as-is (all sentinels) */
        gpu_buffer_upload(atlas.brick_indices, world->brick_indices,
                          world->grid_total * sizeof(int32_t), 0);
        atlas.needs_full_sync = false;
        return atlas;
    }

    /* Create contiguous SDF buffer: active_count * BRICK_VOXELS bytes */
    size_t sdf_size = (size_t)active_count * BRICK_VOXELS;
    atlas.sdf_data = gpu_buffer_create(device, sdf_size, GPU_MEMORY_SHARED);
    if (atlas.sdf_data == NULL) return atlas;

    /* Create material buffer */
    atlas.material_data = gpu_buffer_create(device, sdf_size, GPU_MEMORY_SHARED);
    if (atlas.material_data == NULL) {
        gpu_buffer_destroy(atlas.sdf_data);
        atlas.sdf_data = NULL;
        return atlas;
    }

    /* Flatten demand-paged SDF data into contiguous buffer.
     * The CPU atlas uses pages of 64 bricks each. We iterate all allocated
     * bricks and copy them contiguously. The atlas indices are the same since
     * the demand-paged allocator already uses sequential indices. */
    int8_t* sdf_dst = (int8_t*)gpu_buffer_map(atlas.sdf_data);
    uint8_t* mat_dst = (uint8_t*)gpu_buffer_map(atlas.material_data);

    if (sdf_dst == NULL || mat_dst == NULL) {
        gpu_buffer_destroy(atlas.sdf_data);
        gpu_buffer_destroy(atlas.material_data);
        atlas.sdf_data = NULL;
        atlas.material_data = NULL;
        return atlas;
    }

    for (uint32_t i = 0; i < active_count; i++) {
        uint32_t page = i / ATLAS_PAGE_BRICKS;
        uint32_t offset = i % ATLAS_PAGE_BRICKS;
        size_t dst_off = (size_t)i * BRICK_VOXELS;
        size_t src_off = (size_t)offset * BRICK_VOXELS;

        /* Copy SDF data */
        if (page < world->page_count && world->sdf_pages[page] != NULL) {
            memcpy(sdf_dst + dst_off, world->sdf_pages[page] + src_off,
                   BRICK_VOXELS);
        } else {
            /* Page not allocated - fill with +127 (outside) */
            memset(sdf_dst + dst_off, 127, BRICK_VOXELS);
        }

        /* Copy material data */
        if (page < world->page_count && world->material_pages[page] != NULL) {
            memcpy(mat_dst + dst_off, world->material_pages[page] + src_off,
                   BRICK_VOXELS);
        } else {
            memset(mat_dst + dst_off, 0, BRICK_VOXELS);
        }
    }

    /* Create and upload brick indices.
     * Since the demand-paged allocator assigns sequential indices,
     * the brick_indices values already point to the correct flat offset.
     * We just need to copy the grid as-is. */
    atlas.grid_total = world->grid_total;
    atlas.brick_indices = gpu_buffer_create(device,
        world->grid_total * sizeof(int32_t), GPU_MEMORY_SHARED);
    if (atlas.brick_indices == NULL) {
        gpu_buffer_destroy(atlas.sdf_data);
        gpu_buffer_destroy(atlas.material_data);
        atlas.sdf_data = NULL;
        atlas.material_data = NULL;
        return atlas;
    }

    gpu_buffer_upload(atlas.brick_indices, world->brick_indices,
                      world->grid_total * sizeof(int32_t), 0);

    atlas.brick_count = active_count;
    atlas.needs_full_sync = false;

    /* Upload user-defined feature channels */
    atlas.channel_count = 0;
    for (uint32_t ch = 0; ch < world->feature_channel_count &&
                           ch < GPU_MAX_FEATURE_CHANNELS; ch++) {
        const VoxelChannel* fc = &world->feature_channels[ch];
        if (fc->pages == NULL) continue;

        size_t ch_buf_size = (size_t)active_count * fc->bytes_per_brick;
        atlas.channel_data[ch] = gpu_buffer_create(device, ch_buf_size,
                                                    GPU_MEMORY_SHARED);
        if (atlas.channel_data[ch] == NULL) continue;

        uint8_t* ch_dst = (uint8_t*)gpu_buffer_map(atlas.channel_data[ch]);
        if (ch_dst == NULL) {
            gpu_buffer_destroy(atlas.channel_data[ch]);
            atlas.channel_data[ch] = NULL;
            continue;
        }

        for (uint32_t i = 0; i < active_count; i++) {
            uint32_t page = i / ATLAS_PAGE_BRICKS;
            uint32_t offset = i % ATLAS_PAGE_BRICKS;
            size_t brick_bytes = fc->bytes_per_brick;

            if (page < fc->page_count && fc->pages[page] != NULL) {
                memcpy(ch_dst + (size_t)i * brick_bytes,
                       (uint8_t*)fc->pages[page] + (size_t)offset * brick_bytes,
                       brick_bytes);
            } else {
                memset(ch_dst + (size_t)i * brick_bytes, 0, brick_bytes);
            }
        }
        atlas.channel_count = ch + 1;
    }

    return atlas;
}

GpuResult gpu_sdf_atlas_sync_dirty(GpuSdfAtlas* atlas,
                                    const struct WorldBrickMap* world) {
    if (atlas == NULL || world == NULL) return GPU_ERROR_INVALID_ARG;

    if (atlas->needs_full_sync || atlas->sdf_data == NULL) {
        /* Full re-upload needed - caller should do full upload */
        return GPU_ERROR_INVALID_ARG;
    }

    /* Page-level incremental sync: only upload pages marked dirty */
    bool dirty_pages[MAX_ATLAS_PAGES];
    uint32_t dirty_count = world_get_dirty_pages(world, dirty_pages,
                                                   MAX_ATLAS_PAGES);

    if (dirty_count == 0) {
        return GPU_SUCCESS; /* Nothing changed */
    }

    int8_t* sdf_dst = (int8_t*)gpu_buffer_map(atlas->sdf_data);
    uint8_t* mat_dst = (uint8_t*)gpu_buffer_map(atlas->material_data);
    if (sdf_dst == NULL) return GPU_ERROR_NO_MEMORY;

    uint32_t active_count = atlas->brick_count;
    if (active_count > world->atlas_count) {
        active_count = world->atlas_count;
    }

    /* Only re-upload dirty pages */
    for (uint32_t page = 0; page < world->page_count; page++) {
        if (!dirty_pages[page]) continue;

        /* Calculate brick range for this page */
        uint32_t brick_start = page * ATLAS_PAGE_BRICKS;
        uint32_t brick_end = brick_start + ATLAS_PAGE_BRICKS;
        if (brick_end > active_count) brick_end = active_count;
        if (brick_start >= active_count) continue;

        uint32_t bricks_in_page = brick_end - brick_start;

        /* Copy SDF data for this page */
        if (world->sdf_pages[page] != NULL) {
            memcpy(sdf_dst + (size_t)brick_start * BRICK_VOXELS,
                   world->sdf_pages[page],
                   (size_t)bricks_in_page * BRICK_VOXELS);
        }

        /* Copy material data for this page */
        if (mat_dst != NULL && world->material_pages[page] != NULL) {
            memcpy(mat_dst + (size_t)brick_start * BRICK_VOXELS,
                   world->material_pages[page],
                   (size_t)bricks_in_page * BRICK_VOXELS);
        }
    }

    /* Re-upload brick indices in case topology changed */
    if (atlas->brick_indices != NULL) {
        gpu_buffer_upload(atlas->brick_indices, world->brick_indices,
                          world->grid_total * sizeof(int32_t), 0);
    }

    /* Clear dirty flags after successful sync */
    world_clear_dirty_pages((struct WorldBrickMap*)world);

    return GPU_SUCCESS;
}

void gpu_sdf_atlas_destroy(GpuSdfAtlas* atlas) {
    if (atlas == NULL) return;
    gpu_buffer_destroy(atlas->sdf_data);
    gpu_buffer_destroy(atlas->material_data);
    gpu_buffer_destroy(atlas->brick_indices);
    for (uint32_t ch = 0; ch < atlas->channel_count; ch++) {
        gpu_buffer_destroy(atlas->channel_data[ch]);
    }
    memset(atlas, 0, sizeof(GpuSdfAtlas));
}

/* ============================================================================
 * Section 3: Drone Poses
 * ============================================================================ */

GpuDronePoses gpu_agent_poses_create(GpuDevice* device, uint32_t max_agents) {
    GpuDronePoses poses = {0};
    if (device == NULL || max_agents == 0) return poses;

    size_t float_size = max_agents * sizeof(float);

    poses.pos_x  = gpu_buffer_create(device, float_size, GPU_MEMORY_SHARED);
    poses.pos_y  = gpu_buffer_create(device, float_size, GPU_MEMORY_SHARED);
    poses.pos_z  = gpu_buffer_create(device, float_size, GPU_MEMORY_SHARED);
    poses.quat_w = gpu_buffer_create(device, float_size, GPU_MEMORY_SHARED);
    poses.quat_x = gpu_buffer_create(device, float_size, GPU_MEMORY_SHARED);
    poses.quat_y = gpu_buffer_create(device, float_size, GPU_MEMORY_SHARED);
    poses.quat_z = gpu_buffer_create(device, float_size, GPU_MEMORY_SHARED);

    if (poses.pos_x == NULL || poses.pos_y == NULL || poses.pos_z == NULL ||
        poses.quat_w == NULL || poses.quat_x == NULL ||
        poses.quat_y == NULL || poses.quat_z == NULL) {
        gpu_agent_poses_destroy(&poses);
        memset(&poses, 0, sizeof(poses));
        return poses;
    }

    poses.max_agents = max_agents;
    return poses;
}

GpuResult gpu_agent_poses_upload(GpuDronePoses* poses,
                                  const struct RigidBodyStateSOA* agents,
                                  uint32_t agent_count) {
    if (poses == NULL || agents == NULL || agent_count == 0) {
        return GPU_ERROR_INVALID_ARG;
    }
    if (agent_count > poses->max_agents) {
        agent_count = poses->max_agents;
    }

    size_t copy_size = agent_count * sizeof(float);

    /* Direct memcpy from SoA arrays to GPU buffers (zero-copy on shared) */
    GpuResult r;
    r = gpu_buffer_upload(poses->pos_x,  agents->pos_x,  copy_size, 0); if (r != GPU_SUCCESS) return r;
    r = gpu_buffer_upload(poses->pos_y,  agents->pos_y,  copy_size, 0); if (r != GPU_SUCCESS) return r;
    r = gpu_buffer_upload(poses->pos_z,  agents->pos_z,  copy_size, 0); if (r != GPU_SUCCESS) return r;
    r = gpu_buffer_upload(poses->quat_w, agents->quat_w, copy_size, 0); if (r != GPU_SUCCESS) return r;
    r = gpu_buffer_upload(poses->quat_x, agents->quat_x, copy_size, 0); if (r != GPU_SUCCESS) return r;
    r = gpu_buffer_upload(poses->quat_y, agents->quat_y, copy_size, 0); if (r != GPU_SUCCESS) return r;
    r = gpu_buffer_upload(poses->quat_z, agents->quat_z, copy_size, 0); if (r != GPU_SUCCESS) return r;

    return GPU_SUCCESS;
}

void gpu_agent_poses_destroy(GpuDronePoses* poses) {
    if (poses == NULL) return;
    gpu_buffer_destroy(poses->pos_x);
    gpu_buffer_destroy(poses->pos_y);
    gpu_buffer_destroy(poses->pos_z);
    gpu_buffer_destroy(poses->quat_w);
    gpu_buffer_destroy(poses->quat_x);
    gpu_buffer_destroy(poses->quat_y);
    gpu_buffer_destroy(poses->quat_z);
    memset(poses, 0, sizeof(GpuDronePoses));
}

/* ============================================================================
 * Section 4: Ray Table
 * ============================================================================ */

GpuRayTable gpu_ray_table_create(GpuDevice* device, const Vec3* directions,
                                  uint32_t count) {
    GpuRayTable table = {0};
    if (device == NULL || directions == NULL || count == 0) return table;

    /* Convert Vec3 (12 bytes) -> float4 (16 bytes) with zero w-padding */
    size_t float4_size = count * 4 * sizeof(float);
    table.rays = gpu_buffer_create(device, float4_size, GPU_MEMORY_SHARED);
    if (table.rays == NULL) return table;

    float* dst = (float*)gpu_buffer_map(table.rays);
    if (dst == NULL) {
        gpu_buffer_destroy(table.rays);
        table.rays = NULL;
        return table;
    }

    for (uint32_t i = 0; i < count; i++) {
        dst[i * 4 + 0] = directions[i].x;
        dst[i * 4 + 1] = directions[i].y;
        dst[i * 4 + 2] = directions[i].z;
        dst[i * 4 + 3] = 0.0f;
    }

    table.ray_count = count;
    return table;
}

void gpu_ray_table_destroy(GpuRayTable* table) {
    if (table == NULL) return;
    gpu_buffer_destroy(table->rays);
    memset(table, 0, sizeof(GpuRayTable));
}

/* ============================================================================
 * Section 5: Sensor Output
 * ============================================================================ */

GpuSensorOutput gpu_sensor_output_create(GpuDevice* device,
                                          uint32_t total_floats) {
    GpuSensorOutput output = {0};
    if (device == NULL || total_floats == 0) return output;

    size_t size = total_floats * sizeof(float);
    output.buffer = gpu_buffer_create(device, size, GPU_MEMORY_SHARED);
    if (output.buffer == NULL) return output;

    /* Zero-init output buffer */
    void* ptr = gpu_buffer_map(output.buffer);
    if (ptr != NULL) {
        memset(ptr, 0, size);
    }

    output.total_floats = total_floats;
    return output;
}

void gpu_sensor_output_destroy(GpuSensorOutput* output) {
    if (output == NULL) return;
    gpu_buffer_destroy(output->buffer);
    memset(output, 0, sizeof(GpuSensorOutput));
}

#endif /* GPU_AVAILABLE */
