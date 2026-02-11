/**
 * LiDAR Sensor Implementations (2D and 3D)
 *
 * Multi-ray distance sensors for comprehensive environmental scanning.
 *
 * LiDAR 2D: N rays in a horizontal plane (e.g., for 2D navigation)
 * LiDAR 3D: N*M rays in a spherical grid (vertical layers x horizontal rays)
 *
 * Both use precomputed ray directions for cache-efficient batch processing.
 *
 * Performance Targets:
 * - LiDAR 2D (64 rays): <5ms for 1024 drones
 * - LiDAR 3D (16x64 rays): <20ms for 1024 drones
 */

#include "sensor_implementations.h"
#include "world_brick_map.h"
#include <string.h>
#include <math.h>

/* Constants */
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ============================================================================
 * Ray Direction Precomputation
 * ============================================================================ */

Vec3* precompute_lidar_2d_rays(Arena* arena, uint32_t num_rays, float fov) {
    if (arena == NULL || num_rays == 0) {
        return NULL;
    }

    /* Allocate with 32-byte alignment for SIMD */
    Vec3* rays = arena_alloc_aligned(arena, sizeof(Vec3) * num_rays, 32);
    if (rays == NULL) {
        return NULL;
    }

    /* Distribute rays evenly across FOV in XY plane */
    float half_fov = fov * 0.5f;
    float angle_step = (num_rays > 1) ? fov / (float)(num_rays - 1) : 0.0f;

    for (uint32_t i = 0; i < num_rays; i++) {
        float angle = -half_fov + angle_step * (float)i;
        rays[i] = VEC3(cosf(angle), sinf(angle), 0.0f);
    }

    return rays;
}

Vec3* precompute_lidar_3d_rays(Arena* arena, uint32_t horizontal_rays,
                               uint32_t vertical_layers,
                               float horizontal_fov, float vertical_fov) {
    if (arena == NULL || horizontal_rays == 0 || vertical_layers == 0) {
        return NULL;
    }

    uint32_t total_rays = horizontal_rays * vertical_layers;

    /* Allocate with 32-byte alignment */
    Vec3* rays = arena_alloc_aligned(arena, sizeof(Vec3) * total_rays, 32);
    if (rays == NULL) {
        return NULL;
    }

    float half_h_fov = horizontal_fov * 0.5f;
    float half_v_fov = vertical_fov * 0.5f;
    float h_step = (horizontal_rays > 1) ? horizontal_fov / (float)(horizontal_rays - 1) : 0.0f;
    float v_step = (vertical_layers > 1) ? vertical_fov / (float)(vertical_layers - 1) : 0.0f;

    for (uint32_t v = 0; v < vertical_layers; v++) {
        float elevation = -half_v_fov + v_step * (float)v;
        float cos_el = cosf(elevation);
        float sin_el = sinf(elevation);

        for (uint32_t h = 0; h < horizontal_rays; h++) {
            float azimuth = -half_h_fov + h_step * (float)h;
            uint32_t idx = v * horizontal_rays + h;

            rays[idx] = VEC3(
                cos_el * cosf(azimuth),
                cos_el * sinf(azimuth),
                sin_el
            );
        }
    }

    return rays;
}

/* ============================================================================
 * LiDAR 2D Implementation
 * ============================================================================ */

static void lidar_2d_init(Sensor* sensor, const SensorConfig* config, Arena* arena) {
    LiDAR2DImpl* impl = arena_alloc_type(arena, LiDAR2DImpl);
    if (impl == NULL) {
        sensor->impl = NULL;
        return;
    }

    impl->num_rays = config->lidar_2d.num_rays;
    impl->fov = config->lidar_2d.fov;
    impl->max_range = config->lidar_2d.max_range;

    /* Precompute ray directions */
    impl->ray_directions = precompute_lidar_2d_rays(arena, impl->num_rays, impl->fov);
    if (impl->ray_directions == NULL) {
        sensor->impl = NULL;
        return;
    }

    sensor->impl = impl;
}

static size_t lidar_2d_get_output_size(const Sensor* sensor) {
    LiDAR2DImpl* impl = (LiDAR2DImpl*)sensor->impl;
    if (impl == NULL) return 64;  /* Default */
    return impl->num_rays;
}

static const char* lidar_2d_get_output_dtype(const Sensor* sensor) {
    (void)sensor;
    return "float32";
}

static uint32_t lidar_2d_get_output_shape(const Sensor* sensor, uint32_t* shape) {
    LiDAR2DImpl* impl = (LiDAR2DImpl*)sensor->impl;
    shape[0] = impl ? impl->num_rays : 64;
    return 1;
}

static void lidar_2d_batch_sample(Sensor* sensor, const SensorContext* ctx, float* output_buffer) {
    LiDAR2DImpl* impl = (LiDAR2DImpl*)sensor->impl;
    if (impl == NULL || impl->ray_directions == NULL) {
        memset(output_buffer, 0, ctx->drone_count * 64 * sizeof(float));
        return;
    }

    const DroneStateSOA* drones = ctx->drones;
    const uint32_t* indices = ctx->drone_indices;
    uint32_t count = ctx->drone_count;
    const struct WorldBrickMap* world = ctx->world;
    uint32_t num_rays = impl->num_rays;

    for (uint32_t i = 0; i < count; i++) {
        uint32_t d = indices[i];
        float* out = &output_buffer[i * num_rays];

        /* Get drone position and orientation */
        Vec3 pos = VEC3(drones->pos_x[d], drones->pos_y[d], drones->pos_z[d]);
        Quat q = QUAT(drones->quat_w[d], drones->quat_x[d],
                      drones->quat_y[d], drones->quat_z[d]);

        /* Apply sensor mounting offset */
        Vec3 sensor_pos = vec3_add(pos, quat_rotate(q, sensor->position_offset));

        /* Process each ray */
        for (uint32_t r = 0; r < num_rays; r++) {
            /* Transform ray direction from body to world frame */
            Vec3 world_dir = quat_rotate(q, impl->ray_directions[r]);

            /* Raymarch */
            float distance = impl->max_range;
            if (world != NULL) {
                RayHit hit = world_raymarch(world, sensor_pos, world_dir, impl->max_range);
                if (hit.hit) {
                    distance = hit.distance;
                }
            }

            out[r] = distance;  /* Noise applied by pipeline */
        }
    }
}

static void lidar_2d_reset(Sensor* sensor, uint32_t drone_index) {
    (void)sensor;
    (void)drone_index;
}

static void lidar_2d_destroy(Sensor* sensor) {
    (void)sensor;
}

const SensorVTable SENSOR_VTABLE_LIDAR_2D = {
    .name = "LiDAR-2D",
    .type = SENSOR_TYPE_LIDAR_2D,
    .init = lidar_2d_init,
    .get_output_size = lidar_2d_get_output_size,
    .get_output_dtype = lidar_2d_get_output_dtype,
    .get_output_shape = lidar_2d_get_output_shape,
    .batch_sample = lidar_2d_batch_sample,
    .reset = lidar_2d_reset,
    .destroy = lidar_2d_destroy
};

/* ============================================================================
 * LiDAR 3D Implementation
 * ============================================================================ */

static void lidar_3d_init(Sensor* sensor, const SensorConfig* config, Arena* arena) {
    LiDAR3DImpl* impl = arena_alloc_type(arena, LiDAR3DImpl);
    if (impl == NULL) {
        sensor->impl = NULL;
        return;
    }

    impl->horizontal_rays = config->lidar_3d.horizontal_rays;
    impl->vertical_layers = config->lidar_3d.vertical_layers;
    impl->total_rays = impl->horizontal_rays * impl->vertical_layers;
    impl->horizontal_fov = config->lidar_3d.horizontal_fov;
    impl->vertical_fov = config->lidar_3d.vertical_fov;
    impl->max_range = config->lidar_3d.max_range;

    /* Precompute ray directions */
    impl->ray_directions = precompute_lidar_3d_rays(arena, impl->horizontal_rays,
                                                    impl->vertical_layers,
                                                    impl->horizontal_fov,
                                                    impl->vertical_fov);
    if (impl->ray_directions == NULL) {
        sensor->impl = NULL;
        return;
    }

    sensor->impl = impl;
}

static size_t lidar_3d_get_output_size(const Sensor* sensor) {
    LiDAR3DImpl* impl = (LiDAR3DImpl*)sensor->impl;
    if (impl == NULL) return 64 * 16;  /* Default */
    return impl->total_rays;
}

static const char* lidar_3d_get_output_dtype(const Sensor* sensor) {
    (void)sensor;
    return "float32";
}

static uint32_t lidar_3d_get_output_shape(const Sensor* sensor, uint32_t* shape) {
    LiDAR3DImpl* impl = (LiDAR3DImpl*)sensor->impl;
    if (impl == NULL) {
        shape[0] = 16;
        shape[1] = 64;
    } else {
        shape[0] = impl->vertical_layers;
        shape[1] = impl->horizontal_rays;
    }
    return 2;  /* 2D tensor [layers, rays] */
}

static void lidar_3d_batch_sample(Sensor* sensor, const SensorContext* ctx, float* output_buffer) {
    LiDAR3DImpl* impl = (LiDAR3DImpl*)sensor->impl;
    if (impl == NULL || impl->ray_directions == NULL) {
        memset(output_buffer, 0, ctx->drone_count * 64 * 16 * sizeof(float));
        return;
    }

    const DroneStateSOA* drones = ctx->drones;
    const uint32_t* indices = ctx->drone_indices;
    uint32_t count = ctx->drone_count;
    const struct WorldBrickMap* world = ctx->world;
    uint32_t total_rays = impl->total_rays;

    /* Fast path: no world - just fill with max_range */
    if (world == NULL) {
        for (uint32_t i = 0; i < count; i++) {
            float* out = &output_buffer[i * total_rays];
            for (uint32_t r = 0; r < total_rays; r++) {
                out[r] = impl->max_range;
            }
        }
        return;
    }

    for (uint32_t i = 0; i < count; i++) {
        uint32_t d = indices[i];
        float* out = &output_buffer[i * total_rays];

        /* Get drone position and orientation */
        Vec3 pos = VEC3(drones->pos_x[d], drones->pos_y[d], drones->pos_z[d]);
        Quat q = QUAT(drones->quat_w[d], drones->quat_x[d],
                      drones->quat_y[d], drones->quat_z[d]);

        /* Apply sensor mounting offset */
        Vec3 sensor_pos = vec3_add(pos, quat_rotate(q, sensor->position_offset));

        /* Process each ray */
        for (uint32_t r = 0; r < total_rays; r++) {
            /* Transform ray direction from body to world frame */
            Vec3 world_dir = quat_rotate(q, impl->ray_directions[r]);

            /* Raymarch */
            float distance = impl->max_range;
            if (world != NULL) {
                RayHit hit = world_raymarch(world, sensor_pos, world_dir, impl->max_range);
                if (hit.hit) {
                    distance = hit.distance;
                }
            }

            out[r] = distance;  /* Noise applied by pipeline */
        }
    }
}

static void lidar_3d_reset(Sensor* sensor, uint32_t drone_index) {
    (void)sensor;
    (void)drone_index;
}

static void lidar_3d_destroy(Sensor* sensor) {
    (void)sensor;
}

const SensorVTable SENSOR_VTABLE_LIDAR_3D = {
    .name = "LiDAR-3D",
    .type = SENSOR_TYPE_LIDAR_3D,
    .init = lidar_3d_init,
    .get_output_size = lidar_3d_get_output_size,
    .get_output_dtype = lidar_3d_get_output_dtype,
    .get_output_shape = lidar_3d_get_output_shape,
    .batch_sample = lidar_3d_batch_sample,
    .reset = lidar_3d_reset,
    .destroy = lidar_3d_destroy
};
