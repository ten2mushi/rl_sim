/**
 * Direct Sensor Implementations - Position and Velocity Oracle Sensors
 *
 * These are "oracle" sensors that provide direct access to ground truth state.
 * Useful for debugging, curriculum learning, or privileged critics.
 *
 * Position: 3 floats (x, y, z)
 * Velocity: 6 floats (vx, vy, vz, wx, wy, wz)
 */

#include "sensor_implementations.h"
#include <string.h>

/* ============================================================================
 * Position Sensor Implementation
 * ============================================================================ */

static void position_init(Sensor* sensor, const SensorConfig* config, Arena* arena) {
    (void)config;
    (void)arena;
    sensor->impl = NULL;  /* No additional data needed */
}

static size_t position_get_output_size(const Sensor* sensor) {
    (void)sensor;
    return 3;  /* x, y, z */
}

static const char* position_get_output_dtype(const Sensor* sensor) {
    (void)sensor;
    return "float32";
}

static uint32_t position_get_output_shape(const Sensor* sensor, uint32_t* shape) {
    (void)sensor;
    shape[0] = 3;
    return 1;  /* 1D tensor */
}

static void position_batch_sample(Sensor* sensor, const SensorContext* ctx, float* output_buffer) {
    (void)sensor;

    const DroneStateSOA* drones = ctx->drones;
    const uint32_t* indices = ctx->drone_indices;
    uint32_t count = ctx->drone_count;

    for (uint32_t i = 0; i < count; i++) {
        uint32_t d = indices[i];
        float* out = &output_buffer[i * 3];

        out[0] = drones->pos_x[d];
        out[1] = drones->pos_y[d];
        out[2] = drones->pos_z[d];
    }
}

static void position_reset(Sensor* sensor, uint32_t drone_index) {
    (void)sensor;
    (void)drone_index;
    /* No state to reset */
}

static void position_destroy(Sensor* sensor) {
    (void)sensor;
    /* No resources to free */
}

const SensorVTable SENSOR_VTABLE_POSITION = {
    .name = "Position",
    .type = SENSOR_TYPE_POSITION,
    .init = position_init,
    .get_output_size = position_get_output_size,
    .get_output_dtype = position_get_output_dtype,
    .get_output_shape = position_get_output_shape,
    .batch_sample = position_batch_sample,
    .reset = position_reset,
    .destroy = position_destroy
};

/* ============================================================================
 * Velocity Sensor Implementation
 * ============================================================================ */

static void velocity_init(Sensor* sensor, const SensorConfig* config, Arena* arena) {
    (void)config;
    (void)arena;
    sensor->impl = NULL;  /* No additional data needed */
}

static size_t velocity_get_output_size(const Sensor* sensor) {
    (void)sensor;
    return 6;  /* vx, vy, vz, wx, wy, wz */
}

static const char* velocity_get_output_dtype(const Sensor* sensor) {
    (void)sensor;
    return "float32";
}

static uint32_t velocity_get_output_shape(const Sensor* sensor, uint32_t* shape) {
    (void)sensor;
    shape[0] = 6;
    return 1;  /* 1D tensor */
}

static void velocity_batch_sample(Sensor* sensor, const SensorContext* ctx, float* output_buffer) {
    (void)sensor;

    const DroneStateSOA* drones = ctx->drones;
    const uint32_t* indices = ctx->drone_indices;
    uint32_t count = ctx->drone_count;

    for (uint32_t i = 0; i < count; i++) {
        uint32_t d = indices[i];
        float* out = &output_buffer[i * 6];

        /* Linear velocity (world frame) */
        out[0] = drones->vel_x[d];
        out[1] = drones->vel_y[d];
        out[2] = drones->vel_z[d];

        /* Angular velocity (body frame) */
        out[3] = drones->omega_x[d];
        out[4] = drones->omega_y[d];
        out[5] = drones->omega_z[d];
    }
}

static void velocity_reset(Sensor* sensor, uint32_t drone_index) {
    (void)sensor;
    (void)drone_index;
    /* No state to reset */
}

static void velocity_destroy(Sensor* sensor) {
    (void)sensor;
    /* No resources to free */
}

const SensorVTable SENSOR_VTABLE_VELOCITY = {
    .name = "Velocity",
    .type = SENSOR_TYPE_VELOCITY,
    .init = velocity_init,
    .get_output_size = velocity_get_output_size,
    .get_output_dtype = velocity_get_output_dtype,
    .get_output_shape = velocity_get_output_shape,
    .batch_sample = velocity_batch_sample,
    .reset = velocity_reset,
    .destroy = velocity_destroy
};
