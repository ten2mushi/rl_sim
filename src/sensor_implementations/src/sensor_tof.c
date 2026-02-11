/**
 * ToF (Time-of-Flight) Sensor Implementation
 *
 * Single-ray distance sensor that measures distance in a specified direction.
 * Commonly used for altitude hold, obstacle detection, or proximity sensing.
 *
 * Output: 1 float (distance in meters, or max_range if no hit)
 *
 * Performance Target: <0.5ms for 1024 drones
 */

#include "sensor_implementations.h"
#include "world_brick_map.h"
#include <string.h>
#include <math.h>

/* ============================================================================
 * ToF Sensor Implementation
 * ============================================================================ */

static void tof_init(Sensor* sensor, const SensorConfig* config, Arena* arena) {
    /* Allocate implementation data */
    ToFImpl* impl = arena_alloc_type(arena, ToFImpl);
    if (impl == NULL) {
        sensor->impl = NULL;
        return;
    }

    /* Normalize direction vector */
    impl->direction = vec3_normalize(config->tof.direction);
    impl->max_range = config->tof.max_range;

    sensor->impl = impl;
}

static size_t tof_get_output_size(const Sensor* sensor) {
    (void)sensor;
    return 1;  /* Single distance value */
}

static const char* tof_get_output_dtype(const Sensor* sensor) {
    (void)sensor;
    return "float32";
}

static uint32_t tof_get_output_shape(const Sensor* sensor, uint32_t* shape) {
    (void)sensor;
    shape[0] = 1;
    return 1;  /* 1D tensor */
}

static void tof_batch_sample(Sensor* sensor, const SensorContext* ctx, float* output_buffer) {
    ToFImpl* impl = (ToFImpl*)sensor->impl;
    if (impl == NULL) {
        /* Fill with max range if not properly initialized */
        for (uint32_t i = 0; i < ctx->drone_count; i++) {
            output_buffer[i] = 100.0f;  /* Default max range */
        }
        return;
    }

    const DroneStateSOA* drones = ctx->drones;
    const uint32_t* indices = ctx->drone_indices;
    uint32_t count = ctx->drone_count;
    const struct WorldBrickMap* world = ctx->world;

    for (uint32_t i = 0; i < count; i++) {
        uint32_t d = indices[i];

        /* Get drone position */
        Vec3 pos = VEC3(
            drones->pos_x[d],
            drones->pos_y[d],
            drones->pos_z[d]
        );

        /* Get drone orientation */
        Quat q = QUAT(
            drones->quat_w[d],
            drones->quat_x[d],
            drones->quat_y[d],
            drones->quat_z[d]
        );

        /* Apply sensor mounting offset */
        Vec3 sensor_pos = vec3_add(pos, quat_rotate(q, sensor->position_offset));

        /* Transform ray direction from body to world frame */
        Vec3 world_dir = quat_rotate(q, impl->direction);

        /* Raymarch to find distance */
        float distance = impl->max_range;

        if (world != NULL) {
            RayHit hit = world_raymarch(world, sensor_pos, world_dir, impl->max_range);
            if (hit.hit) {
                distance = hit.distance;
            }
        }

        output_buffer[i] = distance;  /* Noise applied by pipeline */
    }
}

static void tof_reset(Sensor* sensor, uint32_t drone_index) {
    (void)sensor;
    (void)drone_index;
    /* No per-drone state to reset */
}

static void tof_destroy(Sensor* sensor) {
    (void)sensor;
    /* Arena-allocated, no cleanup needed */
}

const SensorVTable SENSOR_VTABLE_TOF = {
    .name = "ToF",
    .type = SENSOR_TYPE_TOF,
    .init = tof_init,
    .get_output_size = tof_get_output_size,
    .get_output_dtype = tof_get_output_dtype,
    .get_output_shape = tof_get_output_shape,
    .batch_sample = tof_batch_sample,
    .reset = tof_reset,
    .destroy = tof_destroy
};
