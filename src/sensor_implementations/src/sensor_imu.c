/**
 * IMU Sensor Implementation
 *
 * 6-axis inertial measurement unit providing:
 * - Accelerometer: 3 floats (ax, ay, az) in body frame
 * - Gyroscope: 3 floats (gx, gy, gz) in body frame
 *
 * The accelerometer measures specific force (gravity + acceleration) in body frame.
 * When hovering at rest, accelerometer reads approximately (0, 0, +9.81) due to
 * gravity being measured as upward specific force in body frame.
 *
 * Performance Target: <0.1ms for 1024 drones
 */

#include "sensor_implementations.h"
#include <string.h>
#include <math.h>

/* ============================================================================
 * IMU Sensor Implementation
 * ============================================================================ */

static void imu_init(Sensor* sensor, const SensorConfig* config, Arena* arena) {
    (void)config;
    (void)arena;
    sensor->impl = NULL;
}

static size_t imu_get_output_size(const Sensor* sensor) {
    (void)sensor;
    return 6;  /* ax, ay, az, gx, gy, gz */
}

static const char* imu_get_output_dtype(const Sensor* sensor) {
    (void)sensor;
    return "float32";
}

static uint32_t imu_get_output_shape(const Sensor* sensor, uint32_t* shape) {
    (void)sensor;
    shape[0] = 6;
    return 1;  /* 1D tensor */
}

static void imu_batch_sample(Sensor* sensor, const SensorContext* ctx, float* output_buffer) {
    (void)sensor;

    const RigidBodyStateSOA* drones = ctx->agents;
    const uint32_t* indices = ctx->agent_indices;
    uint32_t count = ctx->agent_count;

    /* Gravity vector in world frame (Z-up convention) */
    Vec3 gravity_world = VEC3(0.0f, 0.0f, 9.81f);

    for (uint32_t i = 0; i < count; i++) {
        uint32_t d = indices[i];
        float* out = &output_buffer[i * 6];

        /* Get drone orientation quaternion */
        Quat q = QUAT(
            drones->quat_w[d],
            drones->quat_x[d],
            drones->quat_y[d],
            drones->quat_z[d]
        );

        /* Get angular velocity (already in body frame) */
        Vec3 omega = VEC3(
            drones->omega_x[d],
            drones->omega_y[d],
            drones->omega_z[d]
        );

        /* Accelerometer: specific force in body frame
         * When stationary, accelerometer measures -gravity (upward reaction force)
         * Transform gravity from world to body frame using inverse rotation */
        Quat q_inv = quat_conjugate(q);
        Vec3 accel_body = quat_rotate(q_inv, gravity_world);

        /* Output: accelerometer then gyroscope (noise applied by pipeline) */
        out[0] = accel_body.x;
        out[1] = accel_body.y;
        out[2] = accel_body.z;
        out[3] = omega.x;
        out[4] = omega.y;
        out[5] = omega.z;
    }
}

static void imu_reset(Sensor* sensor, uint32_t agent_index) {
    (void)sensor;
    (void)agent_index;
    /* No per-drone state to reset (noise handled by pipeline) */
}

static void imu_destroy(Sensor* sensor) {
    (void)sensor;
    /* Arena-allocated, no cleanup needed */
}

const SensorVTable SENSOR_VTABLE_IMU = {
    .name = "IMU",
    .type = SENSOR_TYPE_IMU,
    .init = imu_init,
    .get_output_size = imu_get_output_size,
    .get_output_dtype = imu_get_output_dtype,
    .get_output_shape = imu_get_output_shape,
    .batch_sample = imu_batch_sample,
    .reset = imu_reset,
    .destroy = imu_destroy
};
