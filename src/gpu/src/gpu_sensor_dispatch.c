/**
 * GPU Sensor Dispatch
 *
 * Dispatches GPU compute kernels for each sensor type and manages
 * GPU vtable registration.
 *
 * Implements: sensor_implementations_register_gpu(),
 * gpu_sensors_dispatch(), gpu_sensors_wait(), gpu_sensors_scatter_results()
 */

#include "gpu_hal.h"

#if GPU_AVAILABLE

#include "sensor_system.h"
#include "sensor_implementations.h"
#include "world_brick_map.h"
#include <string.h>

/* Forward declarations for sensor context */
struct GpuSensorContext;

/* ============================================================================
 * Shared Types (must match sdf_types.h)
 * ============================================================================ */

#include "sdf_types.h"

/* ============================================================================
 * Section 1: World/Raymarch Params Helpers
 * ============================================================================ */

WorldParams gpu_world_params_from_world(const WorldBrickMap* world) {
    WorldParams wp = {0};
    if (world == NULL) return wp;

    wp.world_min_x = world->world_min.x;
    wp.world_min_y = world->world_min.y;
    wp.world_min_z = world->world_min.z;
    wp.voxel_size = world->voxel_size;
    wp.inv_voxel_size = world->inv_voxel_size;
    wp.brick_size_world = world->brick_size_world;
    wp.inv_brick_size = world->inv_brick_size;
    wp.grid_x = world->grid_x;
    wp.grid_y = world->grid_y;
    wp.grid_z = world->grid_z;
    wp.stride_y = world->stride_y;
    wp.stride_z = world->stride_z;
    wp.sdf_scale = world->sdf_scale;
    wp.sdf_scale_div_127 = world->sdf_scale_div_127;

    return wp;
}

RaymarchParams gpu_raymarch_params_for_depth(float near_clip, float far_clip) {
    RaymarchParams rp = {0};
    rp.max_steps = RAYMARCH_MAX_STEPS;
    rp.epsilon = RAYMARCH_EPSILON;
    rp.hit_dist = RAYMARCH_HIT_DIST;
    rp.max_distance = far_clip;
    rp.near_clip = near_clip;
    rp.far_clip = far_clip;
    rp.inv_depth_range = 1.0f / (far_clip - near_clip);
    rp.output_mode = OUTPUT_MODE_DEPTH;
    rp.output_precision = OUTPUT_PRECISION_FP32;
    return rp;
}

RaymarchParams gpu_raymarch_params_for_rgb(float near_clip, float far_clip) {
    RaymarchParams rp = gpu_raymarch_params_for_depth(near_clip, far_clip);
    rp.output_mode = OUTPUT_MODE_RGB;
    return rp;
}

RaymarchParams gpu_raymarch_params_for_material(float near_clip, float far_clip) {
    RaymarchParams rp = gpu_raymarch_params_for_depth(near_clip, far_clip);
    rp.output_mode = OUTPUT_MODE_MATERIAL;
    return rp;
}

RaymarchParams gpu_raymarch_params_for_distance(float max_range) {
    RaymarchParams rp = {0};
    rp.max_steps = RAYMARCH_MAX_STEPS;
    rp.epsilon = RAYMARCH_EPSILON;
    rp.hit_dist = RAYMARCH_HIT_DIST;
    rp.max_distance = max_range;
    rp.near_clip = 0.0f;
    rp.far_clip = max_range;
    rp.inv_depth_range = 1.0f / max_range;
    rp.output_mode = OUTPUT_MODE_DISTANCE;
    rp.output_precision = OUTPUT_PRECISION_FP32;
    return rp;
}

RaymarchParams gpu_raymarch_params_for_depth_normal(float near_clip, float far_clip) {
    RaymarchParams rp = gpu_raymarch_params_for_depth(near_clip, far_clip);
    rp.output_mode = OUTPUT_MODE_DEPTH_NORMAL;
    return rp;
}

/* ============================================================================
 * Section 2: Per-Sensor GPU Dispatch Functions
 * ============================================================================ */

/* Forward declarations for context accessors */
GpuDevice* gpu_sensor_context_device(struct GpuSensorContext* ctx);
GpuCommandQueue* gpu_sensor_context_queue(struct GpuSensorContext* ctx);
GpuSdfAtlas* gpu_sensor_context_atlas(struct GpuSensorContext* ctx);
GpuDronePoses* gpu_sensor_context_poses(struct GpuSensorContext* ctx);

/* Forward declarations for sensor slot access */
typedef struct GpuSensorSlot GpuSensorSlot;
GpuSensorSlot* gpu_sensor_context_find_slot(struct GpuSensorContext* ctx,
                                             SensorType type);

/* Slot field accessors */
GpuKernel* gpu_slot_kernel(GpuSensorSlot* slot);
GpuBuffer* gpu_slot_ray_buffer(GpuSensorSlot* slot);
GpuBuffer* gpu_slot_output_buffer(GpuSensorSlot* slot);
GpuBuffer* gpu_slot_agent_indices(GpuSensorSlot* slot);
uint32_t gpu_slot_image_width(GpuSensorSlot* slot);
uint32_t gpu_slot_image_height(GpuSensorSlot* slot);
uint32_t gpu_slot_floats_per_pixel(GpuSensorSlot* slot);
uint32_t gpu_slot_output_mode(GpuSensorSlot* slot);
void gpu_slot_set_dispatch_info(GpuSensorSlot* slot, uint32_t sensor_idx,
                                 uint32_t agent_count);
uint32_t gpu_slot_dispatch_sensor_idx(GpuSensorSlot* slot);
uint32_t gpu_slot_dispatch_agent_count(GpuSensorSlot* slot);
bool gpu_slot_is_initialized(GpuSensorSlot* slot);
uint32_t gpu_sensor_context_slot_count(struct GpuSensorContext* ctx);
GpuSensorSlot* gpu_sensor_context_slot_at(struct GpuSensorContext* ctx, uint32_t i);

/**
 * Dispatch the unified raymarch kernel for a sensor.
 */
GpuResult gpu_dispatch_raymarch(struct GpuSensorContext* ctx,
                                GpuKernel* kernel,
                                GpuBuffer* ray_table_buf,
                                GpuBuffer* output_buf,
                                GpuBuffer* agent_idx_buf,
                                const WorldBrickMap* world,
                                RaymarchParams rp) {
    GpuCommandQueue* queue = gpu_sensor_context_queue(ctx);
    GpuSdfAtlas* atlas = gpu_sensor_context_atlas(ctx);
    GpuDronePoses* poses = gpu_sensor_context_poses(ctx);

    if (queue == NULL || atlas == NULL || poses == NULL || kernel == NULL) {
        return GPU_ERROR_INVALID_ARG;
    }

    /* Build WorldParams constant */
    WorldParams wp = gpu_world_params_from_world(world);

    /* Bind all buffers */
    gpu_kernel_set_buffer(kernel, 0, atlas->sdf_data);
    gpu_kernel_set_buffer(kernel, 1, atlas->material_data);
    gpu_kernel_set_buffer(kernel, 2, atlas->brick_indices);
    gpu_kernel_set_buffer(kernel, 3, poses->pos_x);
    gpu_kernel_set_buffer(kernel, 4, poses->pos_y);
    gpu_kernel_set_buffer(kernel, 5, poses->pos_z);
    gpu_kernel_set_buffer(kernel, 6, poses->quat_w);
    gpu_kernel_set_buffer(kernel, 7, poses->quat_x);
    gpu_kernel_set_buffer(kernel, 8, poses->quat_y);
    gpu_kernel_set_buffer(kernel, 9, poses->quat_z);
    gpu_kernel_set_buffer(kernel, 10, ray_table_buf);
    gpu_kernel_set_buffer(kernel, 11, output_buf);
    gpu_kernel_set_buffer(kernel, 12, agent_idx_buf);

    /* Set constants */
    gpu_kernel_set_constant(kernel, 0, &wp, sizeof(WorldParams));
    gpu_kernel_set_constant(kernel, 1, &rp, sizeof(RaymarchParams));

    /* Dispatch: grid = (width, height, agent_count), group = (8, 8, 1) */
    uint32_t group_x = 8;
    uint32_t group_y = 8;
    uint32_t group_z = 1;

    /* Round up grid to multiple of group size */
    uint32_t grid_x = ((rp.image_width + group_x - 1) / group_x) * group_x;
    uint32_t grid_y = ((rp.image_height + group_y - 1) / group_y) * group_y;
    uint32_t grid_z = rp.agent_count;

    return gpu_queue_dispatch(queue, kernel,
                              grid_x, grid_y, grid_z,
                              group_x, group_y, group_z);
}

/* ============================================================================
 * Section 3: Build RaymarchParams from Sensor
 * ============================================================================ */

static RaymarchParams build_raymarch_params(const Sensor* sensor,
                                             uint32_t width, uint32_t height,
                                             uint32_t output_mode,
                                             uint32_t agent_count) {
    RaymarchParams rp;

    switch (sensor->type) {
        case SENSOR_TYPE_CAMERA_DEPTH:
        case SENSOR_TYPE_CAMERA_RGB:
        case SENSOR_TYPE_CAMERA_SEGMENTATION: {
            CameraImpl* cam = (CameraImpl*)sensor->impl;
            if (output_mode == OUTPUT_MODE_DEPTH) {
                rp = gpu_raymarch_params_for_depth(cam->near_clip, cam->far_clip);
            } else if (output_mode == OUTPUT_MODE_RGB) {
                rp = gpu_raymarch_params_for_rgb(cam->near_clip, cam->far_clip);
            } else {
                rp = gpu_raymarch_params_for_material(cam->near_clip, cam->far_clip);
            }
            break;
        }
        case SENSOR_TYPE_LIDAR_3D: {
            LiDAR3DImpl* impl = (LiDAR3DImpl*)sensor->impl;
            rp = gpu_raymarch_params_for_distance(impl->max_range);
            break;
        }
        case SENSOR_TYPE_LIDAR_2D: {
            LiDAR2DImpl* impl = (LiDAR2DImpl*)sensor->impl;
            rp = gpu_raymarch_params_for_distance(impl->max_range);
            break;
        }
        case SENSOR_TYPE_TOF: {
            ToFImpl* impl = (ToFImpl*)sensor->impl;
            rp = gpu_raymarch_params_for_distance(impl->max_range);
            break;
        }
        default:
            memset(&rp, 0, sizeof(rp));
            break;
    }

    rp.image_width = width;
    rp.image_height = height;
    rp.agent_count = agent_count;
    return rp;
}

/* ============================================================================
 * Section 4: GPU Vtable Registration
 * ============================================================================ */

/** Marker function - never called, just non-NULL to indicate GPU support */
static int32_t gpu_batch_sample_marker(Sensor* sensor, const SensorContext* ctx,
                                        float* output) {
    (void)sensor; (void)ctx; (void)output;
    return 0;
}

void sensor_implementations_register_gpu(SensorRegistry* registry) {
    if (registry == NULL) return;

    /* Static storage for GPU vtable copies */
    static SensorVTable gpu_vtables[SENSOR_TYPE_COUNT];

    /* Copy CPU vtables */
    for (int t = 0; t < SENSOR_TYPE_COUNT; t++) {
        const SensorVTable* cpu = sensor_registry_get(registry, (SensorType)t);
        if (cpu != NULL) {
            gpu_vtables[t] = *cpu;
        }
    }

    /* Set batch_sample_gpu marker for GPU-capable types */
    gpu_vtables[SENSOR_TYPE_CAMERA_DEPTH].batch_sample_gpu = gpu_batch_sample_marker;
    gpu_vtables[SENSOR_TYPE_CAMERA_RGB].batch_sample_gpu = gpu_batch_sample_marker;
    gpu_vtables[SENSOR_TYPE_CAMERA_SEGMENTATION].batch_sample_gpu = gpu_batch_sample_marker;
    gpu_vtables[SENSOR_TYPE_LIDAR_3D].batch_sample_gpu = gpu_batch_sample_marker;
    gpu_vtables[SENSOR_TYPE_LIDAR_2D].batch_sample_gpu = gpu_batch_sample_marker;
    gpu_vtables[SENSOR_TYPE_TOF].batch_sample_gpu = gpu_batch_sample_marker;

    /* Re-register all (preserves CPU vtable for non-GPU types) */
    for (int t = 0; t < SENSOR_TYPE_COUNT; t++) {
        if (gpu_vtables[t].name != NULL) {
            sensor_registry_register(registry, (SensorType)t, &gpu_vtables[t]);
        }
    }
}

/* ============================================================================
 * Section 5: Batch GPU Dispatch / Wait / Scatter
 * ============================================================================ */

int32_t gpu_sensors_dispatch(struct GpuSensorContext* gpu_ctx,
                              SensorSystem* sys,
                              const struct WorldBrickMap* world,
                              uint32_t agent_count) {
    if (gpu_ctx == NULL || sys == NULL) return GPU_ERROR_INVALID_ARG;

    for (uint32_t s = 0; s < sys->sensor_count; s++) {
        Sensor* sensor = &sys->sensors[s];
        if (sensor->vtable->batch_sample_gpu == NULL) continue;

        /* Find the GPU slot for this sensor type */
        GpuSensorSlot* slot = gpu_sensor_context_find_slot(gpu_ctx, sensor->type);
        if (slot == NULL || !gpu_slot_is_initialized(slot)) continue;

        /* Build drone list for this sensor */
        uint32_t* indices = (uint32_t*)gpu_buffer_map(gpu_slot_agent_indices(slot));
        if (indices == NULL) continue;

        uint32_t count = 0;
        for (uint32_t d = 0; d < agent_count && d < sys->max_agents; d++) {
            uint32_t attach_count = sys->attachment_counts[d];
            uint32_t base_idx = d * MAX_SENSORS_PER_DRONE;
            for (uint32_t a = 0; a < attach_count; a++) {
                if (sys->attachments[base_idx + a].sensor_idx == s) {
                    indices[count++] = d;
                    break;
                }
            }
        }

        if (count == 0) continue;

        /* Store dispatch info for scatter phase */
        gpu_slot_set_dispatch_info(slot, s, count);

        /* Build RaymarchParams */
        uint32_t width = gpu_slot_image_width(slot);
        uint32_t height = gpu_slot_image_height(slot);

        RaymarchParams rp = build_raymarch_params(sensor, width, height,
                                                   gpu_slot_output_mode(slot), count);

        /* Dispatch (async, returns immediately) */
        GpuResult r = gpu_dispatch_raymarch(gpu_ctx,
                                             gpu_slot_kernel(slot),
                                             gpu_slot_ray_buffer(slot),
                                             gpu_slot_output_buffer(slot),
                                             gpu_slot_agent_indices(slot),
                                             world, rp);
        if (r != GPU_SUCCESS) return r;
    }

    return GPU_SUCCESS;
}

int32_t gpu_sensors_wait(struct GpuSensorContext* gpu_ctx) {
    if (gpu_ctx == NULL) return GPU_ERROR_INVALID_ARG;
    GpuCommandQueue* queue = gpu_sensor_context_queue(gpu_ctx);
    if (queue == NULL) return GPU_ERROR_INVALID_ARG;
    return gpu_queue_wait(queue);
}

int32_t gpu_sensors_scatter_results(struct GpuSensorContext* gpu_ctx,
                                     SensorSystem* sys,
                                     uint32_t agent_count) {
    if (gpu_ctx == NULL || sys == NULL) return GPU_ERROR_INVALID_ARG;
    (void)agent_count;

    uint32_t num_slots = gpu_sensor_context_slot_count(gpu_ctx);

    for (uint32_t i = 0; i < num_slots; i++) {
        GpuSensorSlot* slot = gpu_sensor_context_slot_at(gpu_ctx, i);
        if (!gpu_slot_is_initialized(slot)) continue;

        uint32_t count = gpu_slot_dispatch_agent_count(slot);
        if (count == 0) continue;

        uint32_t s = gpu_slot_dispatch_sensor_idx(slot);
        if (s >= sys->sensor_count) continue;

        Sensor* sensor = &sys->sensors[s];
        uint32_t output_size = (uint32_t)sensor->output_size;

        /* Map GPU output and drone list */
        float* gpu_output = (float*)gpu_buffer_map(gpu_slot_output_buffer(slot));
        uint32_t* drone_list = (uint32_t*)gpu_buffer_map(gpu_slot_agent_indices(slot));
        if (gpu_output == NULL || drone_list == NULL) continue;

        /* Scatter to per-drone observation buffers */
        for (uint32_t j = 0; j < count; j++) {
            uint32_t agent_idx = drone_list[j];
            if (agent_idx >= sys->max_agents) continue;

            uint32_t base_idx = agent_idx * MAX_SENSORS_PER_DRONE;
            uint32_t attach_count = sys->attachment_counts[agent_idx];

            for (uint32_t a = 0; a < attach_count; a++) {
                if (sys->attachments[base_idx + a].sensor_idx == s) {
                    float* dst = sys->observation_buffer +
                                 agent_idx * sys->obs_dim +
                                 sys->attachments[base_idx + a].output_offset;
                    memcpy(dst, gpu_output + j * output_size,
                           output_size * sizeof(float));
                    break;
                }
            }
        }

        /* Clear dispatch info for next frame */
        gpu_slot_set_dispatch_info(slot, 0, 0);
    }

    return GPU_SUCCESS;
}

#endif /* GPU_AVAILABLE */
