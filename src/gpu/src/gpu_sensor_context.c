/**
 * GPU Sensor Context
 *
 * Manages per-frame GPU state for sensor dispatch:
 * - SDF atlas sync
 * - Drone pose upload
 * - Per-sensor ray table and output buffer management
 * - Lazy initialization per sensor type
 */

#include "gpu_hal.h"

#if GPU_AVAILABLE

#include "sensor_system.h"
#include "sensor_implementations.h"
#include "world_brick_map.h"
#include "sdf_types.h"
#include <string.h>
#include <stdlib.h>

/* ============================================================================
 * Section 1: GpuSensorContext Structure
 * ============================================================================ */

/** Per-sensor-type GPU resources */
typedef struct GpuSensorSlot {
    GpuRayTable ray_table;
    GpuSensorOutput output;
    GpuBuffer* drone_indices;   /* uint32 [max_drones] - maps thread index -> drone */
    GpuKernel* kernel;
    bool initialized;
    SensorType type;
    uint32_t output_mode;
    uint32_t image_width;
    uint32_t image_height;
    uint32_t floats_per_pixel;  /* 1 for depth/seg/distance, 3 for RGB */

    /* Dispatch state (set by gpu_sensors_dispatch, read by scatter) */
    uint32_t dispatch_sensor_idx;
    uint32_t dispatch_drone_count;
} GpuSensorSlot;

#define GPU_MAX_SENSOR_SLOTS 16

struct GpuSensorContext {
    GpuDevice* device;
    GpuCommandQueue* queue;
    GpuSdfAtlas atlas;
    GpuDronePoses poses;
    GpuEvent* fence;
    uint64_t fence_value;

    GpuSensorSlot slots[GPU_MAX_SENSOR_SLOTS];
    uint32_t slot_count;

    uint32_t max_drones;
    bool atlas_valid;
    const struct WorldBrickMap* last_world;  /* detect world pointer changes */
};

/* ============================================================================
 * Section 2: Lifecycle
 * ============================================================================ */

struct GpuSensorContext* gpu_sensor_context_create(uint32_t max_drones) {
    if (!gpu_is_available() || max_drones == 0) return NULL;

    GpuDevice* device = gpu_device_create();
    if (device == NULL) return NULL;

    struct GpuSensorContext* ctx = calloc(1, sizeof(struct GpuSensorContext));
    if (ctx == NULL) {
        gpu_device_destroy(device);
        return NULL;
    }

    ctx->device = device;
    ctx->queue = gpu_queue_create(device);
    if (ctx->queue == NULL) {
        gpu_device_destroy(device);
        free(ctx);
        return NULL;
    }

    ctx->fence = gpu_event_create(device);
    ctx->fence_value = 0;
    ctx->max_drones = max_drones;

    /* Create drone pose buffers */
    ctx->poses = gpu_drone_poses_create(device, max_drones);
    if (ctx->poses.pos_x == NULL) {
        gpu_queue_destroy(ctx->queue);
        gpu_event_destroy(ctx->fence);
        gpu_device_destroy(device);
        free(ctx);
        return NULL;
    }

    return ctx;
}

void gpu_sensor_context_destroy(struct GpuSensorContext* ctx) {
    if (ctx == NULL) return;

    /* Destroy per-sensor slots */
    for (uint32_t i = 0; i < ctx->slot_count; i++) {
        gpu_ray_table_destroy(&ctx->slots[i].ray_table);
        gpu_sensor_output_destroy(&ctx->slots[i].output);
        gpu_buffer_destroy(ctx->slots[i].drone_indices);
        gpu_kernel_destroy(ctx->slots[i].kernel);
    }

    gpu_sdf_atlas_destroy(&ctx->atlas);
    gpu_drone_poses_destroy(&ctx->poses);
    gpu_event_destroy(ctx->fence);
    gpu_queue_destroy(ctx->queue);
    gpu_device_destroy(ctx->device);
    free(ctx);
}

/* ============================================================================
 * Section 3: Per-Frame Sync
 * ============================================================================ */

GpuResult gpu_sensor_context_sync_frame(struct GpuSensorContext* ctx,
                                         const struct WorldBrickMap* world,
                                         const struct DroneStateSOA* drones,
                                         uint32_t drone_count) {
    if (ctx == NULL) return GPU_ERROR_INVALID_ARG;

    /* Sync SDF atlas */
    if (world != NULL) {
        bool need_full_upload = !ctx->atlas_valid || ctx->atlas.sdf_data == NULL;

        /* Detect world change (different world pointer = different scene) */
        if (!need_full_upload && world != ctx->last_world) {
            need_full_upload = true;
        }

        /* Detect world topology change (different grid or more bricks) */
        if (!need_full_upload) {
            if (ctx->atlas.grid_total != world->grid_total ||
                ctx->atlas.brick_count < world->atlas_count) {
                need_full_upload = true;
            }
        }

        if (need_full_upload) {
            gpu_sdf_atlas_destroy(&ctx->atlas);
            ctx->atlas = gpu_sdf_atlas_upload(ctx->device, world);
            ctx->atlas_valid = (ctx->atlas.sdf_data != NULL ||
                                ctx->atlas.brick_count == 0);
        } else {
            /* Incremental sync (same world, data may have changed) */
            gpu_sdf_atlas_sync_dirty(&ctx->atlas, world);
        }

        ctx->last_world = world;
    }

    /* Upload drone poses */
    if (drones != NULL && drone_count > 0) {
        GpuResult r = gpu_drone_poses_upload(&ctx->poses, drones, drone_count);
        if (r != GPU_SUCCESS) return r;
    }

    return GPU_SUCCESS;
}

/* ============================================================================
 * Section 4: Sensor Slot Init (lazy)
 * ============================================================================ */

/* Helper: determine output mode and dimensions from sensor type */
static bool sensor_type_to_gpu_params(SensorType type, const Sensor* sensor,
                                       uint32_t* out_mode,
                                       uint32_t* out_width,
                                       uint32_t* out_height,
                                       uint32_t* out_fpp) {
    switch (type) {
        case SENSOR_TYPE_CAMERA_DEPTH: {
            CameraImpl* cam = (CameraImpl*)sensor->impl;
            if (cam == NULL) return false;
            *out_mode = OUTPUT_MODE_DEPTH;
            *out_width = cam->width;
            *out_height = cam->height;
            *out_fpp = 1;
            return true;
        }
        case SENSOR_TYPE_CAMERA_RGB: {
            CameraImpl* cam = (CameraImpl*)sensor->impl;
            if (cam == NULL) return false;
            *out_mode = OUTPUT_MODE_RGB;
            *out_width = cam->width;
            *out_height = cam->height;
            *out_fpp = 3;
            return true;
        }
        case SENSOR_TYPE_CAMERA_SEGMENTATION: {
            CameraImpl* cam = (CameraImpl*)sensor->impl;
            if (cam == NULL) return false;
            *out_mode = OUTPUT_MODE_MATERIAL;
            *out_width = cam->width;
            *out_height = cam->height;
            *out_fpp = 1;
            return true;
        }
        case SENSOR_TYPE_LIDAR_3D: {
            LiDAR3DImpl* impl = (LiDAR3DImpl*)sensor->impl;
            if (impl == NULL) return false;
            *out_mode = OUTPUT_MODE_DISTANCE;
            *out_width = impl->horizontal_rays;
            *out_height = impl->vertical_layers;
            *out_fpp = 1;
            return true;
        }
        case SENSOR_TYPE_LIDAR_2D: {
            LiDAR2DImpl* impl = (LiDAR2DImpl*)sensor->impl;
            if (impl == NULL) return false;
            *out_mode = OUTPUT_MODE_DISTANCE;
            *out_width = impl->num_rays;
            *out_height = 1;
            *out_fpp = 1;
            return true;
        }
        case SENSOR_TYPE_TOF: {
            *out_mode = OUTPUT_MODE_DISTANCE;
            *out_width = 1;
            *out_height = 1;
            *out_fpp = 1;
            return true;
        }
        default:
            return false;
    }
}

/* Helper: get ray directions as Vec3 array for any supported sensor type */
static const Vec3* sensor_get_ray_directions(const Sensor* sensor, uint32_t* count) {
    switch (sensor->type) {
        case SENSOR_TYPE_CAMERA_DEPTH:
        case SENSOR_TYPE_CAMERA_RGB:
        case SENSOR_TYPE_CAMERA_SEGMENTATION: {
            CameraImpl* cam = (CameraImpl*)sensor->impl;
            if (cam == NULL || cam->ray_directions == NULL) return NULL;
            *count = cam->total_pixels;
            return cam->ray_directions;
        }
        case SENSOR_TYPE_LIDAR_3D: {
            LiDAR3DImpl* impl = (LiDAR3DImpl*)sensor->impl;
            if (impl == NULL || impl->ray_directions == NULL) return NULL;
            *count = impl->total_rays;
            return impl->ray_directions;
        }
        case SENSOR_TYPE_LIDAR_2D: {
            LiDAR2DImpl* impl = (LiDAR2DImpl*)sensor->impl;
            if (impl == NULL || impl->ray_directions == NULL) return NULL;
            *count = impl->num_rays;
            return impl->ray_directions;
        }
        case SENSOR_TYPE_TOF: {
            ToFImpl* impl = (ToFImpl*)sensor->impl;
            if (impl == NULL) return NULL;
            *count = 1;
            return &impl->direction;
        }
        default:
            return NULL;
    }
}

GpuResult gpu_sensor_context_init_sensor(struct GpuSensorContext* ctx,
                                          const Sensor* sensor,
                                          uint32_t drone_count) {
    if (ctx == NULL || sensor == NULL) return GPU_ERROR_INVALID_ARG;
    if (ctx->slot_count >= GPU_MAX_SENSOR_SLOTS) return GPU_ERROR_NO_MEMORY;

    /* Check if already initialized for this sensor type */
    for (uint32_t i = 0; i < ctx->slot_count; i++) {
        if (ctx->slots[i].type == sensor->type) return GPU_SUCCESS;
    }

    /* Determine GPU params */
    uint32_t mode, width, height, fpp;
    if (!sensor_type_to_gpu_params(sensor->type, sensor, &mode, &width, &height, &fpp)) {
        return GPU_ERROR_INVALID_ARG;
    }

    GpuSensorSlot* slot = &ctx->slots[ctx->slot_count];
    memset(slot, 0, sizeof(GpuSensorSlot));
    slot->type = sensor->type;
    slot->output_mode = mode;
    slot->image_width = width;
    slot->image_height = height;
    slot->floats_per_pixel = fpp;

    /* Create kernel from default metallib */
    slot->kernel = gpu_kernel_create(ctx->device, "raymarch_unified");
    /* kernel may be NULL if metallib not found - allow graceful fallback */

    /* Upload ray directions */
    uint32_t ray_count = 0;
    const Vec3* rays = sensor_get_ray_directions(sensor, &ray_count);
    if (rays != NULL && ray_count > 0) {
        slot->ray_table = gpu_ray_table_create(ctx->device, rays, ray_count);
    }

    /* Create output buffer */
    uint32_t total_floats = drone_count * width * height * fpp;
    slot->output = gpu_sensor_output_create(ctx->device, total_floats);

    /* Create drone index buffer */
    slot->drone_indices = gpu_buffer_create(ctx->device,
        drone_count * sizeof(uint32_t), GPU_MEMORY_SHARED);

    slot->initialized = (slot->kernel != NULL &&
                         slot->ray_table.rays != NULL &&
                         slot->output.buffer != NULL &&
                         slot->drone_indices != NULL);

    ctx->slot_count++;
    return slot->initialized ? GPU_SUCCESS : GPU_ERROR_COMPILE;
}

/* ============================================================================
 * Section 5: Slot Accessors (used by gpu_sensor_dispatch.c)
 * ============================================================================ */

GpuSensorSlot* gpu_sensor_context_find_slot(struct GpuSensorContext* ctx,
                                             SensorType type) {
    if (ctx == NULL) return NULL;
    for (uint32_t i = 0; i < ctx->slot_count; i++) {
        if (ctx->slots[i].type == type && ctx->slots[i].initialized) {
            return &ctx->slots[i];
        }
    }
    return NULL;
}

GpuDevice* gpu_sensor_context_device(struct GpuSensorContext* ctx) {
    return ctx ? ctx->device : NULL;
}

GpuCommandQueue* gpu_sensor_context_queue(struct GpuSensorContext* ctx) {
    return ctx ? ctx->queue : NULL;
}

GpuSdfAtlas* gpu_sensor_context_atlas(struct GpuSensorContext* ctx) {
    return ctx ? &ctx->atlas : NULL;
}

GpuDronePoses* gpu_sensor_context_poses(struct GpuSensorContext* ctx) {
    return ctx ? &ctx->poses : NULL;
}

GpuKernel* gpu_slot_kernel(GpuSensorSlot* slot) {
    return slot ? slot->kernel : NULL;
}

GpuBuffer* gpu_slot_ray_buffer(GpuSensorSlot* slot) {
    return slot ? slot->ray_table.rays : NULL;
}

GpuBuffer* gpu_slot_output_buffer(GpuSensorSlot* slot) {
    return slot ? slot->output.buffer : NULL;
}

GpuBuffer* gpu_slot_drone_indices(GpuSensorSlot* slot) {
    return slot ? slot->drone_indices : NULL;
}

uint32_t gpu_slot_image_width(GpuSensorSlot* slot) {
    return slot ? slot->image_width : 0;
}

uint32_t gpu_slot_image_height(GpuSensorSlot* slot) {
    return slot ? slot->image_height : 0;
}

uint32_t gpu_slot_floats_per_pixel(GpuSensorSlot* slot) {
    return slot ? slot->floats_per_pixel : 0;
}

void gpu_slot_set_dispatch_info(GpuSensorSlot* slot, uint32_t sensor_idx,
                                 uint32_t drone_count) {
    if (slot == NULL) return;
    slot->dispatch_sensor_idx = sensor_idx;
    slot->dispatch_drone_count = drone_count;
}

uint32_t gpu_slot_dispatch_sensor_idx(GpuSensorSlot* slot) {
    return slot ? slot->dispatch_sensor_idx : 0;
}

uint32_t gpu_slot_dispatch_drone_count(GpuSensorSlot* slot) {
    return slot ? slot->dispatch_drone_count : 0;
}

bool gpu_slot_is_initialized(GpuSensorSlot* slot) {
    return slot ? slot->initialized : false;
}

uint32_t gpu_sensor_context_slot_count(struct GpuSensorContext* ctx) {
    return ctx ? ctx->slot_count : 0;
}

GpuSensorSlot* gpu_sensor_context_slot_at(struct GpuSensorContext* ctx, uint32_t i) {
    if (ctx == NULL || i >= ctx->slot_count) return NULL;
    return &ctx->slots[i];
}

#endif /* GPU_AVAILABLE */
