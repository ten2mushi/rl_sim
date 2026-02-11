/**
 * Sensor System Implementation
 *
 * Implements vtable-based polymorphic sensor framework with batch-by-type
 * processing for cache-efficient observation generation.
 */

#include "sensor_system.h"
#include "noise.h"
#include <string.h>

/* ============================================================================
 * Section 1: Sensor Type Names
 * ============================================================================ */

static const char* SENSOR_TYPE_NAMES[SENSOR_TYPE_COUNT] = {
    "IMU",
    "ToF",
    "LiDAR-2D",
    "LiDAR-3D",
    "Camera-RGB",
    "Camera-Depth",
    "Camera-Seg",
    "Position",
    "Velocity",
    "Neighbor"
};

const char* sensor_type_name(SensorType type) {
    if (type >= SENSOR_TYPE_COUNT) {
        return "Unknown";
    }
    return SENSOR_TYPE_NAMES[type];
}

/* ============================================================================
 * Section 2: Lifecycle Functions
 * ============================================================================ */

SensorSystem* sensor_system_create(Arena* arena, uint32_t max_drones,
                                   uint32_t max_sensors, size_t max_obs_dim) {
    if (arena == NULL || max_drones == 0 || max_sensors == 0 || max_obs_dim == 0) {
        return NULL;
    }

    /* Allocate system structure */
    SensorSystem* sys = arena_alloc_type(arena, SensorSystem);
    if (sys == NULL) {
        return NULL;
    }

    memset(sys, 0, sizeof(SensorSystem));
    sys->max_drones = max_drones;
    sys->max_sensors = max_sensors;
    sys->obs_dim = max_obs_dim;
    sys->dt = 0.02f;  /* Default 50Hz, overridden by engine */
    sys->persistent_arena = arena;

    /* Create scratch arena for per-frame allocations */
    /* Size: enough for batch outputs + temporary data */
    size_t scratch_size = max_drones * max_obs_dim * sizeof(float) + 1024 * 1024;
    sys->scratch_arena = arena_create(scratch_size);
    if (sys->scratch_arena == NULL) {
        return NULL;
    }

    /* Allocate sensor array */
    sys->sensors = arena_alloc_array(arena, Sensor, max_sensors);
    if (sys->sensors == NULL) {
        return NULL;
    }
    memset(sys->sensors, 0, sizeof(Sensor) * max_sensors);

    /* Allocate attachment arrays */
    size_t attach_count = (size_t)max_drones * MAX_SENSORS_PER_DRONE;
    sys->attachments = arena_alloc_array(arena, SensorAttachment, attach_count);
    if (sys->attachments == NULL) {
        return NULL;
    }
    memset(sys->attachments, 0, sizeof(SensorAttachment) * attach_count);

    sys->attachment_counts = arena_alloc_array(arena, uint32_t, max_drones);
    if (sys->attachment_counts == NULL) {
        return NULL;
    }
    memset(sys->attachment_counts, 0, sizeof(uint32_t) * max_drones);

    /* Allocate observation buffer (32-byte aligned for SIMD) */
    size_t obs_buffer_size = (size_t)max_drones * max_obs_dim * sizeof(float);
    sys->observation_buffer = arena_alloc_aligned(arena, obs_buffer_size, SENSOR_OBS_ALIGNMENT);
    if (sys->observation_buffer == NULL) {
        return NULL;
    }
    memset(sys->observation_buffer, 0, obs_buffer_size);

    /* Allocate drones-by-sensor lists */
    sys->drones_by_sensor = arena_alloc_array(arena, uint32_t*, max_sensors);
    if (sys->drones_by_sensor == NULL) {
        return NULL;
    }
    for (uint32_t i = 0; i < max_sensors; i++) {
        sys->drones_by_sensor[i] = arena_alloc_array(arena, uint32_t, max_drones);
        if (sys->drones_by_sensor[i] == NULL) {
            return NULL;
        }
    }

    sys->drones_per_sensor = arena_alloc_array(arena, uint32_t, max_sensors);
    if (sys->drones_per_sensor == NULL) {
        return NULL;
    }
    memset(sys->drones_per_sensor, 0, sizeof(uint32_t) * max_sensors);

    /* Initialize registry */
    sensor_registry_init(&sys->registry);

    return sys;
}

void sensor_system_destroy(SensorSystem* sys) {
    if (sys == NULL) {
        return;
    }

    /* Call destroy on all sensors */
    for (uint32_t i = 0; i < sys->sensor_count; i++) {
        Sensor* sensor = &sys->sensors[i];
        if (sensor->vtable != NULL && sensor->vtable->destroy != NULL) {
            sensor->vtable->destroy(sensor);
        }
    }

    /* Destroy scratch arena */
    if (sys->scratch_arena != NULL) {
        arena_destroy(sys->scratch_arena);
    }

    /* No other cleanup needed with arena allocation */
}

void sensor_system_reset(SensorSystem* sys) {
    if (sys == NULL) {
        return;
    }

    /* Clear observation buffer */
    size_t obs_buffer_size = (size_t)sys->max_drones * sys->obs_dim * sizeof(float);
    memset(sys->observation_buffer, 0, obs_buffer_size);

    /* Reset scratch arena */
    if (sys->scratch_arena != NULL) {
        arena_reset(sys->scratch_arena);
    }

    /* Reset per-sensor drone counts */
    memset(sys->drones_per_sensor, 0, sizeof(uint32_t) * sys->max_sensors);

    /* Call reset on all sensors for all drones */
    for (uint32_t s = 0; s < sys->sensor_count; s++) {
        Sensor* sensor = &sys->sensors[s];
        if (sensor->vtable != NULL && sensor->vtable->reset != NULL) {
            for (uint32_t d = 0; d < sys->max_drones; d++) {
                sensor->vtable->reset(sensor, d);
            }
        }
        /* Reset noise state for all drones */
        if (sensor->noise_state != NULL) {
            noise_state_reset_all(sensor->noise_state);
        }
    }
}

/* ============================================================================
 * Section 3: Registry Functions
 * ============================================================================ */

void sensor_registry_init(SensorRegistry* registry) {
    if (registry == NULL) {
        return;
    }

    memset(registry->vtables, 0, sizeof(registry->vtables));
    registry->initialized = true;

    /* Note: Actual vtables are registered by sensor_implementations module */
}

const SensorVTable* sensor_registry_get(const SensorRegistry* registry, SensorType type) {
    if (registry == NULL || !registry->initialized || type >= SENSOR_TYPE_COUNT) {
        return NULL;
    }
    return registry->vtables[type];
}

void sensor_registry_register(SensorRegistry* registry, SensorType type,
                              const SensorVTable* vtable) {
    if (registry == NULL || type >= SENSOR_TYPE_COUNT || vtable == NULL) {
        return;
    }
    registry->vtables[type] = vtable;
}

/* ============================================================================
 * Section 4: Sensor Management Functions
 * ============================================================================ */

uint32_t sensor_system_create_sensor(SensorSystem* sys, const SensorConfig* config) {
    if (sys == NULL || config == NULL) {
        return UINT32_MAX;
    }

    if (sys->sensor_count >= sys->max_sensors) {
        return UINT32_MAX;
    }

    /* Get vtable for this sensor type */
    const SensorVTable* vtable = sensor_registry_get(&sys->registry, config->type);
    if (vtable == NULL) {
        return UINT32_MAX;
    }

    /* Initialize the sensor */
    uint32_t idx = sys->sensor_count;
    Sensor* sensor = &sys->sensors[idx];

    sensor->vtable = vtable;
    sensor->sensor_id = idx;
    sensor->type = config->type;
    sensor->position_offset = config->position_offset;
    sensor->orientation_offset = config->orientation_offset;
    sensor->sample_rate = config->sample_rate;
    sensor->last_sample_time = 0.0f;
    sensor->noise_config = config->noise_config;
    sensor->noise_state = NULL;
    sensor->impl = NULL;

    /* Call type-specific initialization */
    if (vtable->init != NULL) {
        vtable->init(sensor, config, sys->persistent_arena);
    }

    /* Cache output size */
    if (vtable->get_output_size != NULL) {
        sensor->output_size = vtable->get_output_size(sensor);
    }

    /* Create noise state if noise is configured */
    if (sensor->noise_config.group_count > 0) {
        sensor->noise_state = noise_state_create(sys->persistent_arena,
                                                  &sensor->noise_config,
                                                  sys->max_drones, sensor->sensor_id);
    }

    sys->sensor_count++;
    return idx;
}

uint32_t sensor_system_attach(SensorSystem* sys, uint32_t drone_idx, uint32_t sensor_idx) {
    if (sys == NULL || drone_idx >= sys->max_drones || sensor_idx >= sys->sensor_count) {
        return UINT32_MAX;
    }

    uint32_t attach_count = sys->attachment_counts[drone_idx];
    if (attach_count >= MAX_SENSORS_PER_DRONE) {
        return UINT32_MAX;
    }

    /* Compute output offset based on existing attachments */
    uint32_t output_offset = 0;
    uint32_t base_idx = drone_idx * MAX_SENSORS_PER_DRONE;
    for (uint32_t i = 0; i < attach_count; i++) {
        output_offset += (uint32_t)sys->attachments[base_idx + i].output_size;
    }

    /* Check if offset + size exceeds obs_dim */
    Sensor* sensor = &sys->sensors[sensor_idx];
    if (output_offset + sensor->output_size > sys->obs_dim) {
        return UINT32_MAX;
    }

    /* Create attachment */
    SensorAttachment* attach = &sys->attachments[base_idx + attach_count];
    attach->sensor_idx = sensor_idx;
    attach->output_offset = output_offset;
    attach->output_size = sensor->output_size;

    sys->attachment_counts[drone_idx] = attach_count + 1;

    return output_offset;
}

void sensor_system_detach(SensorSystem* sys, uint32_t drone_idx, uint32_t attachment_idx) {
    if (sys == NULL || drone_idx >= sys->max_drones) {
        return;
    }

    uint32_t attach_count = sys->attachment_counts[drone_idx];
    if (attachment_idx >= attach_count) {
        return;
    }

    /* Shift remaining attachments down */
    uint32_t base_idx = drone_idx * MAX_SENSORS_PER_DRONE;

    /* Recalculate output offsets for shifted attachments */
    uint32_t new_offset = 0;
    if (attachment_idx > 0) {
        for (uint32_t i = 0; i < attachment_idx; i++) {
            new_offset += (uint32_t)sys->attachments[base_idx + i].output_size;
        }
    }

    for (uint32_t i = attachment_idx; i < attach_count - 1; i++) {
        sys->attachments[base_idx + i] = sys->attachments[base_idx + i + 1];
        sys->attachments[base_idx + i].output_offset = new_offset;
        new_offset += (uint32_t)sys->attachments[base_idx + i].output_size;
    }

    /* Clear last slot */
    memset(&sys->attachments[base_idx + attach_count - 1], 0, sizeof(SensorAttachment));

    sys->attachment_counts[drone_idx] = attach_count - 1;
}

Sensor* sensor_system_get_sensor(SensorSystem* sys, uint32_t sensor_idx) {
    if (sys == NULL || sensor_idx >= sys->sensor_count) {
        return NULL;
    }
    return &sys->sensors[sensor_idx];
}

uint32_t sensor_system_get_attachment_count(const SensorSystem* sys, uint32_t drone_idx) {
    if (sys == NULL || drone_idx >= sys->max_drones) {
        return 0;
    }
    return sys->attachment_counts[drone_idx];
}

/* ============================================================================
 * Section 5: Batch Processing Functions
 * ============================================================================ */

void sensor_system_sample_all(SensorSystem* sys, const DroneStateSOA* drones,
                              const struct WorldBrickMap* world,
                              const struct CollisionSystem* collision,
                              uint32_t drone_count) {
    if (sys == NULL || drones == NULL || drone_count == 0) {
        return;
    }

    /* Clamp drone count to max */
    if (drone_count > sys->max_drones) {
        drone_count = sys->max_drones;
    }

    /* Phase 1: Group drones by sensor type */
    for (uint32_t i = 0; i < sys->sensor_count; i++) {
        sys->drones_per_sensor[i] = 0;
    }

    for (uint32_t d = 0; d < drone_count; d++) {
        uint32_t attach_count = sys->attachment_counts[d];
        uint32_t base_idx = d * MAX_SENSORS_PER_DRONE;

        for (uint32_t a = 0; a < attach_count; a++) {
            uint32_t sensor_idx = sys->attachments[base_idx + a].sensor_idx;
            uint32_t list_idx = sys->drones_per_sensor[sensor_idx];
            sys->drones_by_sensor[sensor_idx][list_idx] = d;
            sys->drones_per_sensor[sensor_idx] = list_idx + 1;
        }
    }

    /* Phase 2: Process each sensor in batch (SINGLE vtable dispatch per type) */
    SensorContext ctx;
    ctx.drones = drones;
    ctx.world = world;
    ctx.collision = collision;
    ctx.scratch = sys->scratch_arena;

    for (uint32_t s = 0; s < sys->sensor_count; s++) {
        uint32_t count = sys->drones_per_sensor[s];
        if (count == 0) {
            continue;
        }

        Sensor* sensor = &sys->sensors[s];
        ctx.drone_indices = sys->drones_by_sensor[s];
        ctx.drone_count = count;

        /* Allocate batch output from scratch arena */
        size_t batch_size = count * sensor->output_size * sizeof(float);
        float* output = arena_alloc_aligned(sys->scratch_arena, batch_size, SENSOR_OBS_ALIGNMENT);
        if (output == NULL) {
            continue;  /* Skip this sensor if allocation fails */
        }

        /* ONE vtable call processes ALL drones with this sensor */
        if (sensor->vtable->batch_sample != NULL) {
            sensor->vtable->batch_sample(sensor, &ctx, output);
        }

        /* Apply composable noise pipeline to batch output */
        if (sensor->noise_state != NULL) {
            noise_apply(&sensor->noise_config, sensor->noise_state,
                        output, ctx.drone_indices, count,
                        (uint32_t)sensor->output_size, sys->dt);
        }

        /* Scatter to per-drone observation buffers */
        for (uint32_t i = 0; i < count; i++) {
            uint32_t drone_idx = ctx.drone_indices[i];
            uint32_t base_idx = drone_idx * MAX_SENSORS_PER_DRONE;
            uint32_t attach_count = sys->attachment_counts[drone_idx];

            /* Find the attachment for this sensor */
            for (uint32_t a = 0; a < attach_count; a++) {
                if (sys->attachments[base_idx + a].sensor_idx == s) {
                    float* dst = sys->observation_buffer +
                                 drone_idx * sys->obs_dim +
                                 sys->attachments[base_idx + a].output_offset;
                    memcpy(dst, output + i * sensor->output_size,
                           sensor->output_size * sizeof(float));
                    break;
                }
            }
        }
    }

    /* Reset scratch arena for next frame */
    arena_reset(sys->scratch_arena);
}

void sensor_system_sample_cpu_only(SensorSystem* sys, const DroneStateSOA* drones,
                                    const struct WorldBrickMap* world,
                                    const struct CollisionSystem* collision,
                                    uint32_t drone_count) {
    if (sys == NULL || drones == NULL || drone_count == 0) {
        return;
    }

    /* Clamp drone count to max */
    if (drone_count > sys->max_drones) {
        drone_count = sys->max_drones;
    }

    /* Phase 1: Group drones by sensor type (same as sample_all) */
    for (uint32_t i = 0; i < sys->sensor_count; i++) {
        sys->drones_per_sensor[i] = 0;
    }

    for (uint32_t d = 0; d < drone_count; d++) {
        uint32_t attach_count = sys->attachment_counts[d];
        uint32_t base_idx = d * MAX_SENSORS_PER_DRONE;

        for (uint32_t a = 0; a < attach_count; a++) {
            uint32_t sensor_idx = sys->attachments[base_idx + a].sensor_idx;
            uint32_t list_idx = sys->drones_per_sensor[sensor_idx];
            sys->drones_by_sensor[sensor_idx][list_idx] = d;
            sys->drones_per_sensor[sensor_idx] = list_idx + 1;
        }
    }

    /* Phase 2: Process only CPU sensors (skip GPU-accelerated types) */
    SensorContext ctx;
    ctx.drones = drones;
    ctx.world = world;
    ctx.collision = collision;
    ctx.scratch = sys->scratch_arena;
    ctx.gpu = NULL;

    for (uint32_t s = 0; s < sys->sensor_count; s++) {
        Sensor* sensor = &sys->sensors[s];

        /* Skip sensors that have GPU acceleration */
        if (sensor->vtable->batch_sample_gpu != NULL) {
            continue;
        }

        uint32_t count = sys->drones_per_sensor[s];
        if (count == 0) {
            continue;
        }

        ctx.drone_indices = sys->drones_by_sensor[s];
        ctx.drone_count = count;

        /* Allocate batch output from scratch arena */
        size_t batch_size = count * sensor->output_size * sizeof(float);
        float* output = arena_alloc_aligned(sys->scratch_arena, batch_size, SENSOR_OBS_ALIGNMENT);
        if (output == NULL) {
            continue;
        }

        /* ONE vtable call processes ALL drones with this sensor */
        if (sensor->vtable->batch_sample != NULL) {
            sensor->vtable->batch_sample(sensor, &ctx, output);
        }

        /* Apply composable noise pipeline to batch output */
        if (sensor->noise_state != NULL) {
            noise_apply(&sensor->noise_config, sensor->noise_state,
                        output, ctx.drone_indices, count,
                        (uint32_t)sensor->output_size, sys->dt);
        }

        /* Scatter to per-drone observation buffers */
        for (uint32_t i = 0; i < count; i++) {
            uint32_t drone_idx = ctx.drone_indices[i];
            uint32_t base_idx = drone_idx * MAX_SENSORS_PER_DRONE;
            uint32_t attach_count = sys->attachment_counts[drone_idx];

            for (uint32_t a = 0; a < attach_count; a++) {
                if (sys->attachments[base_idx + a].sensor_idx == s) {
                    float* dst = sys->observation_buffer +
                                 drone_idx * sys->obs_dim +
                                 sys->attachments[base_idx + a].output_offset;
                    memcpy(dst, output + i * sensor->output_size,
                           sensor->output_size * sizeof(float));
                    break;
                }
            }
        }
    }

    /* Reset scratch arena for next frame */
    arena_reset(sys->scratch_arena);
}

void sensor_system_sample_sensor(SensorSystem* sys, uint32_t sensor_idx,
                                 const DroneStateSOA* drones,
                                 const struct WorldBrickMap* world,
                                 const struct CollisionSystem* collision,
                                 uint32_t drone_count) {
    if (sys == NULL || drones == NULL || sensor_idx >= sys->sensor_count || drone_count == 0) {
        return;
    }

    if (drone_count > sys->max_drones) {
        drone_count = sys->max_drones;
    }

    /* Build drone list for this specific sensor */
    sys->drones_per_sensor[sensor_idx] = 0;

    for (uint32_t d = 0; d < drone_count; d++) {
        uint32_t attach_count = sys->attachment_counts[d];
        uint32_t base_idx = d * MAX_SENSORS_PER_DRONE;

        for (uint32_t a = 0; a < attach_count; a++) {
            if (sys->attachments[base_idx + a].sensor_idx == sensor_idx) {
                uint32_t list_idx = sys->drones_per_sensor[sensor_idx];
                sys->drones_by_sensor[sensor_idx][list_idx] = d;
                sys->drones_per_sensor[sensor_idx] = list_idx + 1;
                break;
            }
        }
    }

    uint32_t count = sys->drones_per_sensor[sensor_idx];
    if (count == 0) {
        return;
    }

    Sensor* sensor = &sys->sensors[sensor_idx];

    /* Create context */
    SensorContext ctx;
    ctx.drones = drones;
    ctx.drone_indices = sys->drones_by_sensor[sensor_idx];
    ctx.drone_count = count;
    ctx.world = world;
    ctx.collision = collision;
    ctx.scratch = sys->scratch_arena;

    /* Allocate batch output */
    size_t batch_size = count * sensor->output_size * sizeof(float);
    float* output = arena_alloc_aligned(sys->scratch_arena, batch_size, SENSOR_OBS_ALIGNMENT);
    if (output == NULL) {
        return;
    }

    /* Sample */
    if (sensor->vtable->batch_sample != NULL) {
        sensor->vtable->batch_sample(sensor, &ctx, output);
    }

    /* Scatter to observation buffers */
    for (uint32_t i = 0; i < count; i++) {
        uint32_t drone_idx = ctx.drone_indices[i];
        uint32_t base_idx = drone_idx * MAX_SENSORS_PER_DRONE;
        uint32_t attach_count = sys->attachment_counts[drone_idx];

        for (uint32_t a = 0; a < attach_count; a++) {
            if (sys->attachments[base_idx + a].sensor_idx == sensor_idx) {
                float* dst = sys->observation_buffer +
                             drone_idx * sys->obs_dim +
                             sys->attachments[base_idx + a].output_offset;
                memcpy(dst, output + i * sensor->output_size,
                       sensor->output_size * sizeof(float));
                break;
            }
        }
    }

    arena_reset(sys->scratch_arena);
}

/* ============================================================================
 * Section 6: Observation Access Functions
 * ============================================================================ */

float* sensor_system_get_observations(SensorSystem* sys) {
    if (sys == NULL) {
        return NULL;
    }
    return sys->observation_buffer;
}

const float* sensor_system_get_observations_const(const SensorSystem* sys) {
    if (sys == NULL) {
        return NULL;
    }
    return sys->observation_buffer;
}

size_t sensor_system_get_obs_dim(const SensorSystem* sys) {
    if (sys == NULL) {
        return 0;
    }
    return sys->obs_dim;
}

float* sensor_system_get_drone_obs(SensorSystem* sys, uint32_t drone_idx) {
    if (sys == NULL || drone_idx >= sys->max_drones) {
        return NULL;
    }
    return sys->observation_buffer + drone_idx * sys->obs_dim;
}

const float* sensor_system_get_drone_obs_const(const SensorSystem* sys, uint32_t drone_idx) {
    if (sys == NULL || drone_idx >= sys->max_drones) {
        return NULL;
    }
    return sys->observation_buffer + drone_idx * sys->obs_dim;
}

void sensor_system_set_external_buffer(SensorSystem* sys, float* buffer) {
    if (sys == NULL || buffer == NULL) return;
    sys->observation_buffer = buffer;
}

/* ============================================================================
 * Section 7: Configuration Helper Functions
 * ============================================================================ */

SensorConfig sensor_config_default(SensorType type) {
    SensorConfig config;
    memset(&config, 0, sizeof(SensorConfig));

    config.type = type;
    config.position_offset = VEC3_ZERO;
    config.orientation_offset = QUAT_IDENTITY;
    config.sample_rate = 0.0f;  /* Every step */

    switch (type) {
        case SENSOR_TYPE_IMU:
            config.imu._reserved = 0;
            break;

        case SENSOR_TYPE_TOF:
            config.tof.direction = VEC3(0.0f, 0.0f, -1.0f);  /* Down */
            config.tof.max_range = 10.0f;
            break;

        case SENSOR_TYPE_LIDAR_2D:
            config.lidar_2d.num_rays = 64;
            config.lidar_2d.fov = 3.14159f;  /* 180 degrees */
            config.lidar_2d.max_range = 20.0f;
            break;

        case SENSOR_TYPE_LIDAR_3D:
            config.lidar_3d.horizontal_rays = 64;
            config.lidar_3d.vertical_layers = 16;
            config.lidar_3d.horizontal_fov = 6.28318f;  /* 360 degrees */
            config.lidar_3d.vertical_fov = 0.52360f;   /* 30 degrees */
            config.lidar_3d.max_range = 50.0f;
            break;

        case SENSOR_TYPE_CAMERA_RGB:
        case SENSOR_TYPE_CAMERA_DEPTH:
        case SENSOR_TYPE_CAMERA_SEGMENTATION:
            config.camera.width = 64;
            config.camera.height = 64;
            config.camera.fov_horizontal = 1.5708f;  /* 90 degrees */
            config.camera.fov_vertical = 1.5708f;
            config.camera.near_clip = 0.1f;
            config.camera.far_clip = 100.0f;
            config.camera.num_classes = 16;
            break;

        case SENSOR_TYPE_POSITION:
        case SENSOR_TYPE_VELOCITY:
            /* No additional parameters */
            break;

        case SENSOR_TYPE_NEIGHBOR:
            config.neighbor.k = 5;
            config.neighbor.max_range = 20.0f;
            config.neighbor.include_self = false;
            break;

        default:
            break;
    }

    return config;
}

SensorConfig sensor_config_imu(void) {
    return sensor_config_default(SENSOR_TYPE_IMU);
}

SensorConfig sensor_config_tof(Vec3 direction, float max_range) {
    SensorConfig config = sensor_config_default(SENSOR_TYPE_TOF);
    config.tof.direction = vec3_normalize(direction);
    config.tof.max_range = max_range;
    return config;
}

SensorConfig sensor_config_lidar_2d(uint32_t num_rays, float fov, float max_range) {
    SensorConfig config = sensor_config_default(SENSOR_TYPE_LIDAR_2D);
    config.lidar_2d.num_rays = num_rays;
    config.lidar_2d.fov = fov;
    config.lidar_2d.max_range = max_range;
    return config;
}

SensorConfig sensor_config_lidar_3d(uint32_t horizontal_rays, uint32_t vertical_layers,
                                    float horizontal_fov, float vertical_fov,
                                    float max_range) {
    SensorConfig config = sensor_config_default(SENSOR_TYPE_LIDAR_3D);
    config.lidar_3d.horizontal_rays = horizontal_rays;
    config.lidar_3d.vertical_layers = vertical_layers;
    config.lidar_3d.horizontal_fov = horizontal_fov;
    config.lidar_3d.vertical_fov = vertical_fov;
    config.lidar_3d.max_range = max_range;
    return config;
}

SensorConfig sensor_config_camera(uint32_t width, uint32_t height,
                                  float fov, float max_range) {
    SensorConfig config = sensor_config_default(SENSOR_TYPE_CAMERA_RGB);
    config.camera.width = width;
    config.camera.height = height;
    config.camera.fov_horizontal = fov;
    /* Calculate vertical FOV based on aspect ratio */
    float aspect = (float)height / (float)width;
    config.camera.fov_vertical = 2.0f * atanf(aspect * tanf(fov * 0.5f));
    config.camera.far_clip = max_range;
    return config;
}

SensorConfig sensor_config_neighbor(uint32_t k, float max_range) {
    SensorConfig config = sensor_config_default(SENSOR_TYPE_NEIGHBOR);
    config.neighbor.k = k;
    config.neighbor.max_range = max_range;
    return config;
}

SensorConfig sensor_config_position(void) {
    return sensor_config_default(SENSOR_TYPE_POSITION);
}

SensorConfig sensor_config_velocity(void) {
    return sensor_config_default(SENSOR_TYPE_VELOCITY);
}

/* ============================================================================
 * Section 8: Utility Functions
 * ============================================================================ */

size_t sensor_system_memory_size(uint32_t max_drones, uint32_t max_sensors,
                                 size_t max_obs_dim) {
    size_t total = 0;

    /* SensorSystem struct */
    total += sizeof(SensorSystem);

    /* Sensor array */
    total += sizeof(Sensor) * max_sensors;

    /* Attachments */
    total += sizeof(SensorAttachment) * max_drones * MAX_SENSORS_PER_DRONE;

    /* Attachment counts */
    total += sizeof(uint32_t) * max_drones;

    /* Observation buffer (with alignment padding) */
    total += max_drones * max_obs_dim * sizeof(float) + SENSOR_OBS_ALIGNMENT;

    /* Drones-by-sensor lists */
    total += sizeof(uint32_t*) * max_sensors;
    total += sizeof(uint32_t) * max_sensors * max_drones;

    /* Drones-per-sensor counts */
    total += sizeof(uint32_t) * max_sensors;

    return total;
}

size_t sensor_system_compute_obs_dim(const SensorSystem* sys, uint32_t drone_idx) {
    if (sys == NULL || drone_idx >= sys->max_drones) {
        return 0;
    }

    size_t total = 0;
    uint32_t attach_count = sys->attachment_counts[drone_idx];
    uint32_t base_idx = drone_idx * MAX_SENSORS_PER_DRONE;

    for (uint32_t i = 0; i < attach_count; i++) {
        total += sys->attachments[base_idx + i].output_size;
    }

    return total;
}
