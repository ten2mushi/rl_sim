/**
 * Physics Engine Module Implementation
 *
 * RK4-integrated quadcopter physics with SIMD-optimized batch processing.
 */

#include "../include/physics.h"
#include <math.h>

/* Small epsilon for numerical stability */
#define PHYSICS_EPSILON 1e-8f

/* ============================================================================
 * Section 1: Lifecycle Functions (Phase 1)
 * ============================================================================ */

PhysicsConfig physics_config_default(void) {
    PhysicsConfig config = {
        /* Timestep */
        .dt = 0.02f,              /* 50 Hz */
        .dt_variance = 0.0f,
        .substeps = 4,

        /* Physical constants */
        .gravity = 9.81f,
        .air_density = 1.225f,

        /* Feature flags */
        .enable_drag = true,
        .enable_ground_effect = true,
        .enable_motor_dynamics = true,
        .enable_gyroscopic = false,

        /* Ground effect parameters */
        .ground_effect_height = 0.5f,
        .ground_effect_coeff = 1.15f,

        /* Numerical stability */
        .max_linear_accel = 100.0f,
        .max_angular_accel = 200.0f
    };
    return config;
}

PhysicsSystem* physics_create(Arena* persistent_arena, Arena* scratch_arena,
                              const PhysicsConfig* config, uint32_t max_drones) {
    if (persistent_arena == NULL || scratch_arena == NULL || max_drones == 0) {
        return NULL;
    }

    /* Allocate the system struct */
    PhysicsSystem* physics = arena_alloc_type(persistent_arena, PhysicsSystem);
    if (physics == NULL) {
        return NULL;
    }

    /* Initialize configuration */
    if (config != NULL) {
        physics->config = *config;
    } else {
        physics->config = physics_config_default();
    }

    physics->scratch_arena = scratch_arena;
    physics->max_drones = max_drones;
    physics->step_count = 0;
    physics->total_integration_time = 0.0;
    physics->sdf_distances = NULL;

    /* Allocate RK4 scratch buffers (5 DroneStateSOA structures) */
    physics->k1 = drone_state_create(persistent_arena, max_drones);
    physics->k2 = drone_state_create(persistent_arena, max_drones);
    physics->k3 = drone_state_create(persistent_arena, max_drones);
    physics->k4 = drone_state_create(persistent_arena, max_drones);
    physics->temp_state = drone_state_create(persistent_arena, max_drones);

    if (physics->k1 == NULL || physics->k2 == NULL || physics->k3 == NULL ||
        physics->k4 == NULL || physics->temp_state == NULL) {
        return NULL;
    }

    /* Allocate force/torque buffers (6 float arrays, 32-byte aligned) */
    size_t aligned_size = align_up_size(max_drones * sizeof(float), 32);

    physics->forces_x = arena_alloc_aligned(persistent_arena, aligned_size, 32);
    physics->forces_y = arena_alloc_aligned(persistent_arena, aligned_size, 32);
    physics->forces_z = arena_alloc_aligned(persistent_arena, aligned_size, 32);
    physics->torques_x = arena_alloc_aligned(persistent_arena, aligned_size, 32);
    physics->torques_y = arena_alloc_aligned(persistent_arena, aligned_size, 32);
    physics->torques_z = arena_alloc_aligned(persistent_arena, aligned_size, 32);

    if (physics->forces_x == NULL || physics->forces_y == NULL ||
        physics->forces_z == NULL || physics->torques_x == NULL ||
        physics->torques_y == NULL || physics->torques_z == NULL) {
        return NULL;
    }

    return physics;
}

void physics_destroy(PhysicsSystem* physics) {
    if (physics == NULL) {
        return;
    }

    /* Arena memory can't be individually freed, just reset statistics */
    physics->step_count = 0;
    physics->total_integration_time = 0.0;

    /* Nullify pointers for safety */
    physics->k1 = NULL;
    physics->k2 = NULL;
    physics->k3 = NULL;
    physics->k4 = NULL;
    physics->temp_state = NULL;
    physics->forces_x = NULL;
    physics->forces_y = NULL;
    physics->forces_z = NULL;
    physics->torques_x = NULL;
    physics->torques_y = NULL;
    physics->torques_z = NULL;
}

size_t physics_memory_size(uint32_t max_drones) {
    if (max_drones == 0) {
        return 0;
    }

    size_t aligned_array = align_up_size(max_drones * sizeof(float), 32);

    /* PhysicsSystem struct */
    size_t total = sizeof(PhysicsSystem);

    /* 5 DroneStateSOA structures for RK4 */
    total += 5 * drone_state_memory_size(max_drones);

    /* 6 force/torque arrays */
    total += 6 * aligned_array;

    return total;
}

/* ============================================================================
 * Section 2: Numerical Stability Functions (Phase 2)
 * ============================================================================ */

void physics_normalize_quaternions(DroneStateSOA* states, uint32_t count) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");

    if (states == NULL || count == 0) {
        return;
    }

    SIMD_LOOP_START(count);

    /* SIMD loop */
    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_float w = simd_load_ps(&states->quat_w[i]);
        simd_float x = simd_load_ps(&states->quat_x[i]);
        simd_float y = simd_load_ps(&states->quat_y[i]);
        simd_float z = simd_load_ps(&states->quat_z[i]);

        /* ||q||^2 = w^2 + x^2 + y^2 + z^2 */
        simd_float mag_sq = simd_fmadd_ps(w, w,
                           simd_fmadd_ps(x, x,
                           simd_fmadd_ps(y, y,
                           simd_mul_ps(z, z))));

        /* 1/||q|| via rsqrt + Newton-Raphson refinement */
        simd_float inv_mag = simd_rsqrt_ps(mag_sq);
        simd_float half = simd_set1_ps(0.5f);
        simd_float three = simd_set1_ps(1.5f);
        inv_mag = simd_mul_ps(inv_mag,
                  simd_sub_ps(three,
                  simd_mul_ps(simd_mul_ps(half, mag_sq),
                  simd_mul_ps(inv_mag, inv_mag))));

        simd_store_ps(&states->quat_w[i], simd_mul_ps(w, inv_mag));
        simd_store_ps(&states->quat_x[i], simd_mul_ps(x, inv_mag));
        simd_store_ps(&states->quat_y[i], simd_mul_ps(y, inv_mag));
        simd_store_ps(&states->quat_z[i], simd_mul_ps(z, inv_mag));
    }

    /* Scalar remainder */
    SIMD_LOOP_REMAINDER(i, count) {
        float w = states->quat_w[i], x = states->quat_x[i];
        float y = states->quat_y[i], z = states->quat_z[i];
        float inv_mag = 1.0f / sqrtf(w*w + x*x + y*y + z*z + PHYSICS_EPSILON);
        states->quat_w[i] = w * inv_mag;
        states->quat_x[i] = x * inv_mag;
        states->quat_y[i] = y * inv_mag;
        states->quat_z[i] = z * inv_mag;
    }
}

void physics_clamp_velocities(DroneStateSOA* states, const DroneParamsSOA* params, uint32_t count) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    FOUNDATION_ASSERT(params != NULL, "params is NULL");

    if (states == NULL || params == NULL || count == 0) {
        return;
    }

    SIMD_LOOP_START(count);

    /* SIMD loop for linear velocity clamping */
    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_float vx = simd_load_ps(&states->vel_x[i]);
        simd_float vy = simd_load_ps(&states->vel_y[i]);
        simd_float vz = simd_load_ps(&states->vel_z[i]);
        simd_float max_vel = simd_load_ps(&params->max_vel[i]);

        /* Compute speed^2 */
        simd_float speed_sq = simd_fmadd_ps(vx, vx,
                             simd_fmadd_ps(vy, vy,
                             simd_mul_ps(vz, vz)));

        /* Compute scale factor: min(1, max_vel / speed) */
        simd_float max_vel_sq = simd_mul_ps(max_vel, max_vel);
        simd_float needs_clamp = simd_cmp_gt_ps(speed_sq, max_vel_sq);
        simd_float inv_speed = simd_rsqrt_ps(simd_add_ps(speed_sq, simd_set1_ps(PHYSICS_EPSILON)));
        simd_float scale = simd_mul_ps(max_vel, inv_speed);
        scale = simd_blendv_ps(simd_set1_ps(1.0f), scale, needs_clamp);

        simd_store_ps(&states->vel_x[i], simd_mul_ps(vx, scale));
        simd_store_ps(&states->vel_y[i], simd_mul_ps(vy, scale));
        simd_store_ps(&states->vel_z[i], simd_mul_ps(vz, scale));
    }

    /* SIMD loop for angular velocity clamping */
    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_float ox = simd_load_ps(&states->omega_x[i]);
        simd_float oy = simd_load_ps(&states->omega_y[i]);
        simd_float oz = simd_load_ps(&states->omega_z[i]);
        simd_float max_omega = simd_load_ps(&params->max_omega[i]);

        simd_float omega_sq = simd_fmadd_ps(ox, ox,
                             simd_fmadd_ps(oy, oy,
                             simd_mul_ps(oz, oz)));

        simd_float max_omega_sq = simd_mul_ps(max_omega, max_omega);
        simd_float needs_clamp = simd_cmp_gt_ps(omega_sq, max_omega_sq);
        simd_float inv_omega = simd_rsqrt_ps(simd_add_ps(omega_sq, simd_set1_ps(PHYSICS_EPSILON)));
        simd_float scale = simd_mul_ps(max_omega, inv_omega);
        scale = simd_blendv_ps(simd_set1_ps(1.0f), scale, needs_clamp);

        simd_store_ps(&states->omega_x[i], simd_mul_ps(ox, scale));
        simd_store_ps(&states->omega_y[i], simd_mul_ps(oy, scale));
        simd_store_ps(&states->omega_z[i], simd_mul_ps(oz, scale));
    }

    /* Scalar remainder */
    SIMD_LOOP_REMAINDER(i, count) {
        /* Linear velocity */
        float vx = states->vel_x[i], vy = states->vel_y[i], vz = states->vel_z[i];
        float speed_sq = vx*vx + vy*vy + vz*vz;
        float max_vel = params->max_vel[i];
        if (speed_sq > max_vel * max_vel) {
            float scale = max_vel / sqrtf(speed_sq + PHYSICS_EPSILON);
            states->vel_x[i] = vx * scale;
            states->vel_y[i] = vy * scale;
            states->vel_z[i] = vz * scale;
        }

        /* Angular velocity */
        float ox = states->omega_x[i], oy = states->omega_y[i], oz = states->omega_z[i];
        float omega_sq = ox*ox + oy*oy + oz*oz;
        float max_omega = params->max_omega[i];
        if (omega_sq > max_omega * max_omega) {
            float scale = max_omega / sqrtf(omega_sq + PHYSICS_EPSILON);
            states->omega_x[i] = ox * scale;
            states->omega_y[i] = oy * scale;
            states->omega_z[i] = oz * scale;
        }
    }
}

void physics_clamp_accelerations(float* accel_x, float* accel_y, float* accel_z,
                                 float max_accel, uint32_t count) {
    if (accel_x == NULL || accel_y == NULL || accel_z == NULL || count == 0) {
        return;
    }

    float max_accel_sq = max_accel * max_accel;

    SIMD_LOOP_START(count);

    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_float ax = simd_load_ps(&accel_x[i]);
        simd_float ay = simd_load_ps(&accel_y[i]);
        simd_float az = simd_load_ps(&accel_z[i]);

        simd_float accel_sq = simd_fmadd_ps(ax, ax,
                             simd_fmadd_ps(ay, ay,
                             simd_mul_ps(az, az)));

        simd_float max_sq = simd_set1_ps(max_accel_sq);
        simd_float needs_clamp = simd_cmp_gt_ps(accel_sq, max_sq);
        simd_float inv_accel = simd_rsqrt_ps(simd_add_ps(accel_sq, simd_set1_ps(PHYSICS_EPSILON)));
        simd_float scale = simd_mul_ps(simd_set1_ps(max_accel), inv_accel);
        scale = simd_blendv_ps(simd_set1_ps(1.0f), scale, needs_clamp);

        simd_store_ps(&accel_x[i], simd_mul_ps(ax, scale));
        simd_store_ps(&accel_y[i], simd_mul_ps(ay, scale));
        simd_store_ps(&accel_z[i], simd_mul_ps(az, scale));
    }

    SIMD_LOOP_REMAINDER(i, count) {
        float ax = accel_x[i], ay = accel_y[i], az = accel_z[i];
        float accel_sq = ax*ax + ay*ay + az*az;
        if (accel_sq > max_accel_sq) {
            float scale = max_accel / sqrtf(accel_sq + PHYSICS_EPSILON);
            accel_x[i] = ax * scale;
            accel_y[i] = ay * scale;
            accel_z[i] = az * scale;
        }
    }
}

uint32_t physics_sanitize_state(DroneStateSOA* states, uint32_t count) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");

    if (states == NULL || count == 0) {
        return 0;
    }

    uint32_t reset_count = 0;

    for (uint32_t i = 0; i < count; i++) {
        bool needs_reset = false;

        /* Check for NaN/Inf in position */
        if (!isfinite(states->pos_x[i]) || !isfinite(states->pos_y[i]) ||
            !isfinite(states->pos_z[i])) {
#ifndef NDEBUG
            FOUNDATION_ASSERT(0, "NaN/Inf detected in drone position after physics step");
#endif
            needs_reset = true;
        }

        /* Check for NaN/Inf in velocity */
        if (!isfinite(states->vel_x[i]) || !isfinite(states->vel_y[i]) ||
            !isfinite(states->vel_z[i])) {
            needs_reset = true;
        }

        /* Check for NaN/Inf in quaternion */
        if (!isfinite(states->quat_w[i]) || !isfinite(states->quat_x[i]) ||
            !isfinite(states->quat_y[i]) || !isfinite(states->quat_z[i])) {
            needs_reset = true;
        }

        /* Check for NaN/Inf in angular velocity */
        if (!isfinite(states->omega_x[i]) || !isfinite(states->omega_y[i]) ||
            !isfinite(states->omega_z[i])) {
            needs_reset = true;
        }

        /* Check for NaN/Inf in RPMs */
        if (!isfinite(states->rpm_0[i]) || !isfinite(states->rpm_1[i]) ||
            !isfinite(states->rpm_2[i]) || !isfinite(states->rpm_3[i])) {
            needs_reset = true;
        }

        if (needs_reset) {
            /* Reset to safe state */
            states->pos_x[i] = 0.0f;
            states->pos_y[i] = 0.0f;
            states->pos_z[i] = 0.0f;

            states->vel_x[i] = 0.0f;
            states->vel_y[i] = 0.0f;
            states->vel_z[i] = 0.0f;

            states->quat_w[i] = 1.0f;
            states->quat_x[i] = 0.0f;
            states->quat_y[i] = 0.0f;
            states->quat_z[i] = 0.0f;

            states->omega_x[i] = 0.0f;
            states->omega_y[i] = 0.0f;
            states->omega_z[i] = 0.0f;

            states->rpm_0[i] = 0.0f;
            states->rpm_1[i] = 0.0f;
            states->rpm_2[i] = 0.0f;
            states->rpm_3[i] = 0.0f;

            reset_count++;
        }
    }

    return reset_count;
}

/* ============================================================================
 * Section 3: Physics Primitives (Phase 3)
 * ============================================================================ */

void physics_motor_dynamics(const float* rpm_commands, float* actual_rpms,
                            const DroneParamsSOA* params, float dt, uint32_t count) {
    FOUNDATION_ASSERT(rpm_commands != NULL, "rpm_commands is NULL");
    FOUNDATION_ASSERT(actual_rpms != NULL, "actual_rpms is NULL");
    FOUNDATION_ASSERT(params != NULL, "params is NULL");

    if (rpm_commands == NULL || actual_rpms == NULL || params == NULL || count == 0) {
        return;
    }

    /* Process all 4 motors for each drone */
    /* Motor 0 */
    SIMD_LOOP_START(count);

    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_float tau = simd_load_ps(&params->motor_tau[i]);
        simd_float max_rpm = simd_load_ps(&params->max_rpm[i]);
        simd_float dt_v = simd_set1_ps(dt);
        simd_float one = simd_set1_ps(1.0f);
        simd_float zero = simd_setzero_ps();

        /* Compute alpha = min(dt / tau, 1.0) */
        simd_float alpha = simd_div_ps(dt_v, simd_add_ps(tau, simd_set1_ps(PHYSICS_EPSILON)));
        alpha = simd_min_ps(alpha, one);

        /* Motor 0 */
        simd_float cmd0 = simd_load_ps(&rpm_commands[i * 4]);
        simd_float curr0 = simd_load_ps(&actual_rpms[i * 4]);
        cmd0 = simd_min_ps(simd_max_ps(cmd0, zero), max_rpm);
        simd_float new0 = simd_fmadd_ps(simd_sub_ps(cmd0, curr0), alpha, curr0);
        simd_store_ps(&actual_rpms[i * 4], new0);
    }

    /* Scalar processing for all motors (simpler and correct) */
    for (uint32_t i = 0; i < count; i++) {
        float tau = params->motor_tau[i];
        float max_rpm = params->max_rpm[i];
        float alpha = dt / (tau + PHYSICS_EPSILON);
        if (alpha > 1.0f) alpha = 1.0f;

        for (int m = 0; m < 4; m++) {
            float cmd = rpm_commands[i * 4 + m];
            cmd = clampf(cmd, 0.0f, max_rpm);
            float curr = actual_rpms[i * 4 + m];
            actual_rpms[i * 4 + m] = curr + (cmd - curr) * alpha;
        }
    }
}

void physics_compute_forces_torques(const DroneStateSOA* states, const DroneParamsSOA* params,
                                    float* forces_x, float* forces_y, float* forces_z,
                                    float* torques_x, float* torques_y, float* torques_z,
                                    uint32_t count) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    FOUNDATION_ASSERT(params != NULL, "params is NULL");

    if (states == NULL || params == NULL || count == 0) {
        return;
    }

    SIMD_LOOP_START(count);

    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        /* Load motor RPMs */
        simd_float rpm0 = simd_load_ps(&states->rpm_0[i]);
        simd_float rpm1 = simd_load_ps(&states->rpm_1[i]);
        simd_float rpm2 = simd_load_ps(&states->rpm_2[i]);
        simd_float rpm3 = simd_load_ps(&states->rpm_3[i]);

        /* Load parameters */
        simd_float k_thrust = simd_load_ps(&params->k_thrust[i]);
        simd_float k_torque = simd_load_ps(&params->k_torque[i]);
        simd_float arm_len = simd_load_ps(&params->arm_length[i]);

        /* Compute thrust for each motor: T_i = k_thrust * rpm^2 */
        simd_float T0 = simd_mul_ps(k_thrust, simd_mul_ps(rpm0, rpm0));
        simd_float T1 = simd_mul_ps(k_thrust, simd_mul_ps(rpm1, rpm1));
        simd_float T2 = simd_mul_ps(k_thrust, simd_mul_ps(rpm2, rpm2));
        simd_float T3 = simd_mul_ps(k_thrust, simd_mul_ps(rpm3, rpm3));

        /* Total thrust in body frame (along body Z axis) */
        simd_float total_thrust = simd_add_ps(simd_add_ps(T0, T1), simd_add_ps(T2, T3));

        /* Load quaternion */
        simd_float qw = simd_load_ps(&states->quat_w[i]);
        simd_float qx = simd_load_ps(&states->quat_x[i]);
        simd_float qy = simd_load_ps(&states->quat_y[i]);
        simd_float qz = simd_load_ps(&states->quat_z[i]);

        /* Rotate thrust to world frame
         * For F_body = [0, 0, thrust], optimized rotation:
         * fx = 2 * (qx*qz + qw*qy) * thrust
         * fy = 2 * (qy*qz - qw*qx) * thrust
         * fz = (qw^2 - qx^2 - qy^2 + qz^2) * thrust
         */
        simd_float two = simd_set1_ps(2.0f);

        simd_float fx_world = simd_mul_ps(two,
                             simd_mul_ps(simd_add_ps(simd_mul_ps(qx, qz),
                                                     simd_mul_ps(qw, qy)),
                                        total_thrust));

        simd_float fy_world = simd_mul_ps(two,
                             simd_mul_ps(simd_sub_ps(simd_mul_ps(qy, qz),
                                                     simd_mul_ps(qw, qx)),
                                        total_thrust));

        simd_float qw2 = simd_mul_ps(qw, qw);
        simd_float qx2 = simd_mul_ps(qx, qx);
        simd_float qy2 = simd_mul_ps(qy, qy);
        simd_float qz2 = simd_mul_ps(qz, qz);

        simd_float fz_world = simd_mul_ps(
            simd_sub_ps(simd_add_ps(qw2, qz2), simd_add_ps(qx2, qy2)),
            total_thrust);

        simd_store_ps(&forces_x[i], fx_world);
        simd_store_ps(&forces_y[i], fy_world);
        simd_store_ps(&forces_z[i], fz_world);

        /* Compute torques (body frame, X-configuration)
         * Roll (tau_x): arm_length * (T1 + T3 - T0 - T2)
         * Pitch (tau_y): arm_length * (T0 + T1 - T2 - T3)
         * Yaw (tau_z): k_torque * (rpm0^2 + rpm1^2 - rpm2^2 - rpm3^2)
         */
        simd_float tau_x = simd_mul_ps(arm_len,
                          simd_sub_ps(simd_add_ps(T1, T3), simd_add_ps(T0, T2)));
        simd_float tau_y = simd_mul_ps(arm_len,
                          simd_sub_ps(simd_add_ps(T0, T1), simd_add_ps(T2, T3)));

        /* Yaw torque from motor reaction torques */
        simd_float rpm0_sq = simd_mul_ps(rpm0, rpm0);
        simd_float rpm1_sq = simd_mul_ps(rpm1, rpm1);
        simd_float rpm2_sq = simd_mul_ps(rpm2, rpm2);
        simd_float rpm3_sq = simd_mul_ps(rpm3, rpm3);
        simd_float tau_z = simd_mul_ps(k_torque,
                          simd_sub_ps(simd_add_ps(rpm0_sq, rpm1_sq),
                                      simd_add_ps(rpm2_sq, rpm3_sq)));

        simd_store_ps(&torques_x[i], tau_x);
        simd_store_ps(&torques_y[i], tau_y);
        simd_store_ps(&torques_z[i], tau_z);
    }

    /* Scalar remainder */
    SIMD_LOOP_REMAINDER(i, count) {
        float rpm0 = states->rpm_0[i], rpm1 = states->rpm_1[i];
        float rpm2 = states->rpm_2[i], rpm3 = states->rpm_3[i];
        float k_thrust = params->k_thrust[i];
        float k_torque = params->k_torque[i];
        float arm_len = params->arm_length[i];

        /* Thrusts */
        float T0 = k_thrust * rpm0 * rpm0;
        float T1 = k_thrust * rpm1 * rpm1;
        float T2 = k_thrust * rpm2 * rpm2;
        float T3 = k_thrust * rpm3 * rpm3;
        float total_thrust = T0 + T1 + T2 + T3;

        /* Rotate to world frame */
        float qw = states->quat_w[i], qx = states->quat_x[i];
        float qy = states->quat_y[i], qz = states->quat_z[i];

        forces_x[i] = 2.0f * (qx*qz + qw*qy) * total_thrust;
        forces_y[i] = 2.0f * (qy*qz - qw*qx) * total_thrust;
        forces_z[i] = (qw*qw - qx*qx - qy*qy + qz*qz) * total_thrust;

        /* Torques */
        torques_x[i] = arm_len * (T1 + T3 - T0 - T2);
        torques_y[i] = arm_len * (T0 + T1 - T2 - T3);
        torques_z[i] = k_torque * (rpm0*rpm0 + rpm1*rpm1 - rpm2*rpm2 - rpm3*rpm3);
    }
}

void physics_apply_drag(const DroneStateSOA* states, const DroneParamsSOA* params,
                        float* forces_x, float* forces_y, float* forces_z,
                        uint32_t count, float air_density) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    FOUNDATION_ASSERT(params != NULL, "params is NULL");

    if (states == NULL || params == NULL || count == 0) {
        return;
    }

    (void)air_density;  /* For now, using simplified linear drag model */

    SIMD_LOOP_START(count);

    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_float vx = simd_load_ps(&states->vel_x[i]);
        simd_float vy = simd_load_ps(&states->vel_y[i]);
        simd_float vz = simd_load_ps(&states->vel_z[i]);
        simd_float k_drag = simd_load_ps(&params->k_drag[i]);

        /* Compute speed */
        simd_float speed_sq = simd_fmadd_ps(vx, vx,
                             simd_fmadd_ps(vy, vy,
                             simd_mul_ps(vz, vz)));
        simd_float speed = simd_sqrt_ps(simd_add_ps(speed_sq, simd_set1_ps(PHYSICS_EPSILON)));

        /* F_drag = -k_drag * |v| * v (linear drag model) */
        simd_float drag_factor = simd_mul_ps(simd_sub_ps(simd_setzero_ps(), k_drag), speed);

        simd_float fx = simd_load_ps(&forces_x[i]);
        simd_float fy = simd_load_ps(&forces_y[i]);
        simd_float fz = simd_load_ps(&forces_z[i]);

        simd_store_ps(&forces_x[i], simd_fmadd_ps(drag_factor, vx, fx));
        simd_store_ps(&forces_y[i], simd_fmadd_ps(drag_factor, vy, fy));
        simd_store_ps(&forces_z[i], simd_fmadd_ps(drag_factor, vz, fz));
    }

    SIMD_LOOP_REMAINDER(i, count) {
        float vx = states->vel_x[i], vy = states->vel_y[i], vz = states->vel_z[i];
        float k_drag = params->k_drag[i];
        float speed = sqrtf(vx*vx + vy*vy + vz*vz + PHYSICS_EPSILON);
        float drag_factor = -k_drag * speed;

        forces_x[i] += drag_factor * vx;
        forces_y[i] += drag_factor * vy;
        forces_z[i] += drag_factor * vz;
    }
}

void physics_apply_ground_effect(const DroneStateSOA* states, const DroneParamsSOA* params,
                                 float* forces_z, const float* sdf_distances,
                                 uint32_t count,
                                 float ground_height, float effect_coeff) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");

    if (states == NULL || sdf_distances == NULL || count == 0) {
        return;
    }

    (void)params;  /* Currently unused */

    SIMD_LOOP_START(count);

    simd_float h_ref = simd_set1_ps(ground_height);
    simd_float k_max = simd_set1_ps(effect_coeff);
    simd_float one = simd_set1_ps(1.0f);
    simd_float zero = simd_setzero_ps();

    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_float sdf = simd_load_ps(&sdf_distances[i]);
        simd_float fz = simd_load_ps(&forces_z[i]);

        /* k_ge = 1 + (k_max - 1) * exp(-sdf / h_ref)
         * Apply when sdf > 0 (outside surface) AND sdf < ground_height (close) */
        simd_float ratio = simd_div_ps(simd_sub_ps(zero, sdf),
                                        simd_add_ps(h_ref, simd_set1_ps(PHYSICS_EPSILON)));

        /* Clamp ratio to prevent extreme values */
        ratio = simd_max_ps(ratio, simd_set1_ps(-10.0f));
        ratio = simd_min_ps(ratio, simd_set1_ps(0.0f));

        /* exp approximation: 1 + x + x^2/2 (good for small x) */
        simd_float exp_approx = simd_fmadd_ps(
            simd_mul_ps(ratio, ratio), simd_set1_ps(0.5f),
            simd_add_ps(one, ratio));

        simd_float k_ge = simd_fmadd_ps(simd_sub_ps(k_max, one), exp_approx, one);
        k_ge = simd_max_ps(k_ge, one);  /* Clamp to at least 1 */

        /* Only apply when close to surface (sdf < ground_height) AND outside (sdf > 0) */
        simd_float close_to_surface = simd_cmp_lt_ps(sdf, h_ref);
        simd_float outside_surface = simd_cmp_gt_ps(sdf, zero);
        simd_float apply_ge = simd_and_ps(close_to_surface, outside_surface);
        simd_float final_k = simd_blendv_ps(one, k_ge, apply_ge);

        simd_store_ps(&forces_z[i], simd_mul_ps(fz, final_k));
    }

    SIMD_LOOP_REMAINDER(i, count) {
        float sdf = sdf_distances[i];
        if (sdf > 0.0f && sdf < ground_height) {
            float ratio = -sdf / (ground_height + PHYSICS_EPSILON);
            float exp_val = expf(ratio);
            float k_ge = 1.0f + (effect_coeff - 1.0f) * exp_val;
            forces_z[i] *= k_ge;
        }
    }
}

/* ============================================================================
 * Section 4: Derivative Computation (Phase 4)
 * ============================================================================ */

void physics_compute_derivatives(const DroneStateSOA* states, const DroneParamsSOA* params,
                                 const float* actions, DroneStateSOA* derivatives,
                                 uint32_t count, const PhysicsConfig* config,
                                 const float* sdf_distances) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    FOUNDATION_ASSERT(params != NULL, "params is NULL");
    FOUNDATION_ASSERT(derivatives != NULL, "derivatives is NULL");
    FOUNDATION_ASSERT(config != NULL, "config is NULL");

    if (states == NULL || params == NULL || derivatives == NULL ||
        config == NULL || count == 0) {
        return;
    }

    (void)actions;  /* Actions affect RPMs which are in states */

    /* Position derivative = velocity */
    SIMD_LOOP_START(count);

    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_store_ps(&derivatives->pos_x[i], simd_load_ps(&states->vel_x[i]));
        simd_store_ps(&derivatives->pos_y[i], simd_load_ps(&states->vel_y[i]));
        simd_store_ps(&derivatives->pos_z[i], simd_load_ps(&states->vel_z[i]));
    }

    SIMD_LOOP_REMAINDER(i, count) {
        derivatives->pos_x[i] = states->vel_x[i];
        derivatives->pos_y[i] = states->vel_y[i];
        derivatives->pos_z[i] = states->vel_z[i];
    }

    /* Compute thrust forces and torques */
    /* Use derivatives arrays temporarily for force/torque storage */
    float* forces_x = derivatives->vel_x;
    float* forces_y = derivatives->vel_y;
    float* forces_z = derivatives->vel_z;

    physics_compute_forces_torques(states, params,
                                   forces_x, forces_y, forces_z,
                                   derivatives->omega_x, derivatives->omega_y, derivatives->omega_z,
                                   count);

    /* Apply drag if enabled */
    if (config->enable_drag) {
        physics_apply_drag(states, params, forces_x, forces_y, forces_z,
                          count, config->air_density);
    }

    /* Apply ground effect if enabled and SDF distances available */
    if (config->enable_ground_effect && sdf_distances != NULL) {
        physics_apply_ground_effect(states, params, forces_z, sdf_distances, count,
                                   config->ground_effect_height, config->ground_effect_coeff);
    }

    /* Add gravity and convert forces to accelerations */
    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_float fx = simd_load_ps(&forces_x[i]);
        simd_float fy = simd_load_ps(&forces_y[i]);
        simd_float fz = simd_load_ps(&forces_z[i]);
        simd_float mass = simd_load_ps(&params->mass[i]);
        simd_float gravity = simd_load_ps(&params->gravity[i]);
        simd_float inv_mass = simd_div_ps(simd_set1_ps(1.0f),
                             simd_add_ps(mass, simd_set1_ps(PHYSICS_EPSILON)));

        /* a = F/m, subtract gravity in Z */
        simd_store_ps(&derivatives->vel_x[i], simd_mul_ps(fx, inv_mass));
        simd_store_ps(&derivatives->vel_y[i], simd_mul_ps(fy, inv_mass));
        simd_store_ps(&derivatives->vel_z[i],
            simd_sub_ps(simd_mul_ps(fz, inv_mass), gravity));
    }

    SIMD_LOOP_REMAINDER(i, count) {
        float mass = params->mass[i];
        float gravity = params->gravity[i];
        float inv_mass = 1.0f / (mass + PHYSICS_EPSILON);

        derivatives->vel_x[i] = forces_x[i] * inv_mass;
        derivatives->vel_y[i] = forces_y[i] * inv_mass;
        derivatives->vel_z[i] = forces_z[i] * inv_mass - gravity;
    }

    /* Clamp accelerations for stability */
    physics_clamp_accelerations(derivatives->vel_x, derivatives->vel_y, derivatives->vel_z,
                               config->max_linear_accel, count);

    /* Compute angular acceleration: omega_dot = I^-1 * (tau - omega x (I * omega))
     * Euler equation for rigid body with diagonal inertia tensor
     */
    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_float tau_x = simd_load_ps(&derivatives->omega_x[i]);
        simd_float tau_y = simd_load_ps(&derivatives->omega_y[i]);
        simd_float tau_z = simd_load_ps(&derivatives->omega_z[i]);

        simd_float ox = simd_load_ps(&states->omega_x[i]);
        simd_float oy = simd_load_ps(&states->omega_y[i]);
        simd_float oz = simd_load_ps(&states->omega_z[i]);

        simd_float ixx = simd_load_ps(&params->ixx[i]);
        simd_float iyy = simd_load_ps(&params->iyy[i]);
        simd_float izz = simd_load_ps(&params->izz[i]);

        /* Load angular damping */
        simd_float k_ang_damp = simd_load_ps(&params->k_ang_damp[i]);

        /* Apply angular damping: tau -= k_ang_damp * omega */
        tau_x = simd_sub_ps(tau_x, simd_mul_ps(k_ang_damp, ox));
        tau_y = simd_sub_ps(tau_y, simd_mul_ps(k_ang_damp, oy));
        tau_z = simd_sub_ps(tau_z, simd_mul_ps(k_ang_damp, oz));

        /* Euler equations:
         * omega_dot_x = (tau_x - (izz - iyy) * oy * oz) / ixx
         * omega_dot_y = (tau_y - (ixx - izz) * oz * ox) / iyy
         * omega_dot_z = (tau_z - (iyy - ixx) * ox * oy) / izz
         */
        simd_float eps = simd_set1_ps(PHYSICS_EPSILON);

        simd_float omega_dot_x = simd_div_ps(
            simd_sub_ps(tau_x, simd_mul_ps(simd_sub_ps(izz, iyy), simd_mul_ps(oy, oz))),
            simd_add_ps(ixx, eps));

        simd_float omega_dot_y = simd_div_ps(
            simd_sub_ps(tau_y, simd_mul_ps(simd_sub_ps(ixx, izz), simd_mul_ps(oz, ox))),
            simd_add_ps(iyy, eps));

        simd_float omega_dot_z = simd_div_ps(
            simd_sub_ps(tau_z, simd_mul_ps(simd_sub_ps(iyy, ixx), simd_mul_ps(ox, oy))),
            simd_add_ps(izz, eps));

        simd_store_ps(&derivatives->omega_x[i], omega_dot_x);
        simd_store_ps(&derivatives->omega_y[i], omega_dot_y);
        simd_store_ps(&derivatives->omega_z[i], omega_dot_z);
    }

    SIMD_LOOP_REMAINDER(i, count) {
        float tau_x = derivatives->omega_x[i];
        float tau_y = derivatives->omega_y[i];
        float tau_z = derivatives->omega_z[i];

        float ox = states->omega_x[i];
        float oy = states->omega_y[i];
        float oz = states->omega_z[i];

        float ixx = params->ixx[i];
        float iyy = params->iyy[i];
        float izz = params->izz[i];
        float k_ang_damp = params->k_ang_damp[i];

        /* Apply damping */
        tau_x -= k_ang_damp * ox;
        tau_y -= k_ang_damp * oy;
        tau_z -= k_ang_damp * oz;

        derivatives->omega_x[i] = (tau_x - (izz - iyy) * oy * oz) / (ixx + PHYSICS_EPSILON);
        derivatives->omega_y[i] = (tau_y - (ixx - izz) * oz * ox) / (iyy + PHYSICS_EPSILON);
        derivatives->omega_z[i] = (tau_z - (iyy - ixx) * ox * oy) / (izz + PHYSICS_EPSILON);
    }

    /* Clamp angular accelerations */
    physics_clamp_accelerations(derivatives->omega_x, derivatives->omega_y, derivatives->omega_z,
                               config->max_angular_accel, count);

    /* Compute quaternion derivative: q_dot = 0.5 * q * [0, omega] */
    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_float qw = simd_load_ps(&states->quat_w[i]);
        simd_float qx = simd_load_ps(&states->quat_x[i]);
        simd_float qy = simd_load_ps(&states->quat_y[i]);
        simd_float qz = simd_load_ps(&states->quat_z[i]);

        simd_float ox = simd_load_ps(&states->omega_x[i]);
        simd_float oy = simd_load_ps(&states->omega_y[i]);
        simd_float oz = simd_load_ps(&states->omega_z[i]);

        simd_float half = simd_set1_ps(0.5f);

        /* q_dot = 0.5 * q * [0, omega]
         * qw_dot = 0.5 * (-qx*ox - qy*oy - qz*oz)
         * qx_dot = 0.5 * (qw*ox + qy*oz - qz*oy)
         * qy_dot = 0.5 * (qw*oy + qz*ox - qx*oz)
         * qz_dot = 0.5 * (qw*oz + qx*oy - qy*ox)
         */
        simd_float qw_dot = simd_mul_ps(half,
            simd_sub_ps(simd_sub_ps(simd_sub_ps(simd_setzero_ps(),
                simd_mul_ps(qx, ox)), simd_mul_ps(qy, oy)), simd_mul_ps(qz, oz)));

        simd_float qx_dot = simd_mul_ps(half,
            simd_add_ps(simd_sub_ps(simd_mul_ps(qw, ox), simd_mul_ps(qz, oy)),
                       simd_mul_ps(qy, oz)));

        simd_float qy_dot = simd_mul_ps(half,
            simd_add_ps(simd_sub_ps(simd_mul_ps(qw, oy), simd_mul_ps(qx, oz)),
                       simd_mul_ps(qz, ox)));

        simd_float qz_dot = simd_mul_ps(half,
            simd_add_ps(simd_sub_ps(simd_mul_ps(qw, oz), simd_mul_ps(qy, ox)),
                       simd_mul_ps(qx, oy)));

        simd_store_ps(&derivatives->quat_w[i], qw_dot);
        simd_store_ps(&derivatives->quat_x[i], qx_dot);
        simd_store_ps(&derivatives->quat_y[i], qy_dot);
        simd_store_ps(&derivatives->quat_z[i], qz_dot);
    }

    SIMD_LOOP_REMAINDER(i, count) {
        float qw = states->quat_w[i], qx = states->quat_x[i];
        float qy = states->quat_y[i], qz = states->quat_z[i];
        float ox = states->omega_x[i], oy = states->omega_y[i], oz = states->omega_z[i];

        derivatives->quat_w[i] = 0.5f * (-qx*ox - qy*oy - qz*oz);
        derivatives->quat_x[i] = 0.5f * (qw*ox + qy*oz - qz*oy);
        derivatives->quat_y[i] = 0.5f * (qw*oy + qz*ox - qx*oz);
        derivatives->quat_z[i] = 0.5f * (qw*oz + qx*oy - qy*ox);
    }

    /* Zero RPM derivatives (motor dynamics handled separately) */
    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_float zero = simd_setzero_ps();
        simd_store_ps(&derivatives->rpm_0[i], zero);
        simd_store_ps(&derivatives->rpm_1[i], zero);
        simd_store_ps(&derivatives->rpm_2[i], zero);
        simd_store_ps(&derivatives->rpm_3[i], zero);
    }

    SIMD_LOOP_REMAINDER(i, count) {
        derivatives->rpm_0[i] = 0.0f;
        derivatives->rpm_1[i] = 0.0f;
        derivatives->rpm_2[i] = 0.0f;
        derivatives->rpm_3[i] = 0.0f;
    }
}

/* ============================================================================
 * Section 5: RK4 Integration (Phase 5)
 * ============================================================================ */

void physics_rk4_substep(const DroneStateSOA* current, const DroneStateSOA* derivative,
                         DroneStateSOA* output, float dt_scale, uint32_t count) {
    FOUNDATION_ASSERT(current != NULL, "current is NULL");
    FOUNDATION_ASSERT(derivative != NULL, "derivative is NULL");
    FOUNDATION_ASSERT(output != NULL, "output is NULL");

    if (current == NULL || derivative == NULL || output == NULL || count == 0) {
        return;
    }

    SIMD_LOOP_START(count);
    simd_float dt = simd_set1_ps(dt_scale);

    /* Position: p_new = p + dp * dt */
    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_store_ps(&output->pos_x[i],
            simd_fmadd_ps(simd_load_ps(&derivative->pos_x[i]), dt,
                         simd_load_ps(&current->pos_x[i])));
        simd_store_ps(&output->pos_y[i],
            simd_fmadd_ps(simd_load_ps(&derivative->pos_y[i]), dt,
                         simd_load_ps(&current->pos_y[i])));
        simd_store_ps(&output->pos_z[i],
            simd_fmadd_ps(simd_load_ps(&derivative->pos_z[i]), dt,
                         simd_load_ps(&current->pos_z[i])));
    }

    /* Velocity: v_new = v + dv * dt */
    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_store_ps(&output->vel_x[i],
            simd_fmadd_ps(simd_load_ps(&derivative->vel_x[i]), dt,
                         simd_load_ps(&current->vel_x[i])));
        simd_store_ps(&output->vel_y[i],
            simd_fmadd_ps(simd_load_ps(&derivative->vel_y[i]), dt,
                         simd_load_ps(&current->vel_y[i])));
        simd_store_ps(&output->vel_z[i],
            simd_fmadd_ps(simd_load_ps(&derivative->vel_z[i]), dt,
                         simd_load_ps(&current->vel_z[i])));
    }

    /* Quaternion: q_new = q + dq * dt */
    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_store_ps(&output->quat_w[i],
            simd_fmadd_ps(simd_load_ps(&derivative->quat_w[i]), dt,
                         simd_load_ps(&current->quat_w[i])));
        simd_store_ps(&output->quat_x[i],
            simd_fmadd_ps(simd_load_ps(&derivative->quat_x[i]), dt,
                         simd_load_ps(&current->quat_x[i])));
        simd_store_ps(&output->quat_y[i],
            simd_fmadd_ps(simd_load_ps(&derivative->quat_y[i]), dt,
                         simd_load_ps(&current->quat_y[i])));
        simd_store_ps(&output->quat_z[i],
            simd_fmadd_ps(simd_load_ps(&derivative->quat_z[i]), dt,
                         simd_load_ps(&current->quat_z[i])));
    }

    /* Angular velocity: omega_new = omega + domega * dt */
    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_store_ps(&output->omega_x[i],
            simd_fmadd_ps(simd_load_ps(&derivative->omega_x[i]), dt,
                         simd_load_ps(&current->omega_x[i])));
        simd_store_ps(&output->omega_y[i],
            simd_fmadd_ps(simd_load_ps(&derivative->omega_y[i]), dt,
                         simd_load_ps(&current->omega_y[i])));
        simd_store_ps(&output->omega_z[i],
            simd_fmadd_ps(simd_load_ps(&derivative->omega_z[i]), dt,
                         simd_load_ps(&current->omega_z[i])));
    }

    /* RPMs: copy from current (motor dynamics handled separately) */
    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_store_ps(&output->rpm_0[i], simd_load_ps(&current->rpm_0[i]));
        simd_store_ps(&output->rpm_1[i], simd_load_ps(&current->rpm_1[i]));
        simd_store_ps(&output->rpm_2[i], simd_load_ps(&current->rpm_2[i]));
        simd_store_ps(&output->rpm_3[i], simd_load_ps(&current->rpm_3[i]));
    }

    /* Scalar remainder */
    SIMD_LOOP_REMAINDER(i, count) {
        output->pos_x[i] = current->pos_x[i] + derivative->pos_x[i] * dt_scale;
        output->pos_y[i] = current->pos_y[i] + derivative->pos_y[i] * dt_scale;
        output->pos_z[i] = current->pos_z[i] + derivative->pos_z[i] * dt_scale;

        output->vel_x[i] = current->vel_x[i] + derivative->vel_x[i] * dt_scale;
        output->vel_y[i] = current->vel_y[i] + derivative->vel_y[i] * dt_scale;
        output->vel_z[i] = current->vel_z[i] + derivative->vel_z[i] * dt_scale;

        output->quat_w[i] = current->quat_w[i] + derivative->quat_w[i] * dt_scale;
        output->quat_x[i] = current->quat_x[i] + derivative->quat_x[i] * dt_scale;
        output->quat_y[i] = current->quat_y[i] + derivative->quat_y[i] * dt_scale;
        output->quat_z[i] = current->quat_z[i] + derivative->quat_z[i] * dt_scale;

        output->omega_x[i] = current->omega_x[i] + derivative->omega_x[i] * dt_scale;
        output->omega_y[i] = current->omega_y[i] + derivative->omega_y[i] * dt_scale;
        output->omega_z[i] = current->omega_z[i] + derivative->omega_z[i] * dt_scale;

        output->rpm_0[i] = current->rpm_0[i];
        output->rpm_1[i] = current->rpm_1[i];
        output->rpm_2[i] = current->rpm_2[i];
        output->rpm_3[i] = current->rpm_3[i];
    }
}

void physics_rk4_combine(DroneStateSOA* states, const DroneStateSOA* k1,
                         const DroneStateSOA* k2, const DroneStateSOA* k3,
                         const DroneStateSOA* k4, float dt, uint32_t count) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    FOUNDATION_ASSERT(k1 != NULL, "k1 is NULL");
    FOUNDATION_ASSERT(k2 != NULL, "k2 is NULL");
    FOUNDATION_ASSERT(k3 != NULL, "k3 is NULL");
    FOUNDATION_ASSERT(k4 != NULL, "k4 is NULL");

    if (states == NULL || k1 == NULL || k2 == NULL || k3 == NULL || k4 == NULL || count == 0) {
        return;
    }

    /* Weighted combination: state += (k1 + 2*k2 + 2*k3 + k4) * dt/6 */
    float dt_over_6 = dt / 6.0f;

    SIMD_LOOP_START(count);
    simd_float dt6 = simd_set1_ps(dt_over_6);
    simd_float two = simd_set1_ps(2.0f);

    /* Helper macro for weighted combination */
    #define RK4_COMBINE_ARRAY(arr) \
        for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) { \
            simd_float s = simd_load_ps(&states->arr[i]); \
            simd_float d1 = simd_load_ps(&k1->arr[i]); \
            simd_float d2 = simd_load_ps(&k2->arr[i]); \
            simd_float d3 = simd_load_ps(&k3->arr[i]); \
            simd_float d4 = simd_load_ps(&k4->arr[i]); \
            /* sum = k1 + 2*k2 + 2*k3 + k4 */ \
            simd_float sum = simd_add_ps(d1, d4); \
            sum = simd_fmadd_ps(two, d2, sum); \
            sum = simd_fmadd_ps(two, d3, sum); \
            simd_store_ps(&states->arr[i], simd_fmadd_ps(sum, dt6, s)); \
        }

    RK4_COMBINE_ARRAY(pos_x);
    RK4_COMBINE_ARRAY(pos_y);
    RK4_COMBINE_ARRAY(pos_z);
    RK4_COMBINE_ARRAY(vel_x);
    RK4_COMBINE_ARRAY(vel_y);
    RK4_COMBINE_ARRAY(vel_z);
    RK4_COMBINE_ARRAY(quat_w);
    RK4_COMBINE_ARRAY(quat_x);
    RK4_COMBINE_ARRAY(quat_y);
    RK4_COMBINE_ARRAY(quat_z);
    RK4_COMBINE_ARRAY(omega_x);
    RK4_COMBINE_ARRAY(omega_y);
    RK4_COMBINE_ARRAY(omega_z);

    #undef RK4_COMBINE_ARRAY

    /* Scalar remainder */
    SIMD_LOOP_REMAINDER(i, count) {
        #define RK4_COMBINE_SCALAR(arr) \
            states->arr[i] += (k1->arr[i] + 2.0f * k2->arr[i] + \
                              2.0f * k3->arr[i] + k4->arr[i]) * dt_over_6

        RK4_COMBINE_SCALAR(pos_x);
        RK4_COMBINE_SCALAR(pos_y);
        RK4_COMBINE_SCALAR(pos_z);
        RK4_COMBINE_SCALAR(vel_x);
        RK4_COMBINE_SCALAR(vel_y);
        RK4_COMBINE_SCALAR(vel_z);
        RK4_COMBINE_SCALAR(quat_w);
        RK4_COMBINE_SCALAR(quat_x);
        RK4_COMBINE_SCALAR(quat_y);
        RK4_COMBINE_SCALAR(quat_z);
        RK4_COMBINE_SCALAR(omega_x);
        RK4_COMBINE_SCALAR(omega_y);
        RK4_COMBINE_SCALAR(omega_z);

        #undef RK4_COMBINE_SCALAR
    }
}

void physics_rk4_integrate(PhysicsSystem* physics, DroneStateSOA* states,
                           const DroneParamsSOA* params, const float* actions,
                           float dt, uint32_t count) {
    FOUNDATION_ASSERT(physics != NULL, "physics is NULL");
    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    FOUNDATION_ASSERT(params != NULL, "params is NULL");

    if (physics == NULL || states == NULL || params == NULL || count == 0) {
        return;
    }

    /* k1 = f(t, y) */
    physics_compute_derivatives(states, params, actions, physics->k1, count, &physics->config, physics->sdf_distances);

    /* k2 = f(t + dt/2, y + k1*dt/2) */
    physics_rk4_substep(states, physics->k1, physics->temp_state, dt * 0.5f, count);
    physics_normalize_quaternions(physics->temp_state, count);
    physics_compute_derivatives(physics->temp_state, params, actions, physics->k2, count, &physics->config, physics->sdf_distances);

    /* k3 = f(t + dt/2, y + k2*dt/2) */
    physics_rk4_substep(states, physics->k2, physics->temp_state, dt * 0.5f, count);
    physics_normalize_quaternions(physics->temp_state, count);
    physics_compute_derivatives(physics->temp_state, params, actions, physics->k3, count, &physics->config, physics->sdf_distances);

    /* k4 = f(t + dt, y + k3*dt) */
    physics_rk4_substep(states, physics->k3, physics->temp_state, dt, count);
    physics_normalize_quaternions(physics->temp_state, count);
    physics_compute_derivatives(physics->temp_state, params, actions, physics->k4, count, &physics->config, physics->sdf_distances);

    /* y_new = y + (k1 + 2*k2 + 2*k3 + k4) * dt/6 */
    physics_rk4_combine(states, physics->k1, physics->k2, physics->k3, physics->k4, dt, count);
}

/* ============================================================================
 * Section 6: Main Physics Step (Phase 6)
 * ============================================================================ */

void physics_step(PhysicsSystem* physics, DroneStateSOA* states,
                  const DroneParamsSOA* params, const float* actions, uint32_t count) {
    physics_step_dt(physics, states, params, actions, count, physics->config.dt);
}

void physics_step_dt(PhysicsSystem* physics, DroneStateSOA* states,
                     const DroneParamsSOA* params, const float* actions,
                     uint32_t count, float dt) {
    FOUNDATION_ASSERT(physics != NULL, "physics is NULL");
    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    FOUNDATION_ASSERT(params != NULL, "params is NULL");

    if (physics == NULL || states == NULL || params == NULL || count == 0) {
        return;
    }

    /* Calculate substep dt */
    uint32_t substeps = physics->config.substeps;
    if (substeps == 0) substeps = 1;
    float substep_dt = dt / (float)substeps;

    /* Convert actions to RPM commands (4 per drone) */
    /* Allocate temporary buffer for RPM commands */
    float* rpm_commands = physics->forces_x;  /* Reuse buffer temporarily */

    if (actions != NULL) {
        for (uint32_t i = 0; i < count; i++) {
            float max_rpm = params->max_rpm[i];
            rpm_commands[i * 4 + 0] = action_to_rpm(actions[i * 4 + 0], max_rpm);
            rpm_commands[i * 4 + 1] = action_to_rpm(actions[i * 4 + 1], max_rpm);
            rpm_commands[i * 4 + 2] = action_to_rpm(actions[i * 4 + 2], max_rpm);
            rpm_commands[i * 4 + 3] = action_to_rpm(actions[i * 4 + 3], max_rpm);
        }
    }

    /* Pack current RPMs into array for motor dynamics */
    float* actual_rpms = physics->forces_y;  /* Reuse buffer */
    for (uint32_t i = 0; i < count; i++) {
        actual_rpms[i * 4 + 0] = states->rpm_0[i];
        actual_rpms[i * 4 + 1] = states->rpm_1[i];
        actual_rpms[i * 4 + 2] = states->rpm_2[i];
        actual_rpms[i * 4 + 3] = states->rpm_3[i];
    }

    /* Run substeps */
    for (uint32_t s = 0; s < substeps; s++) {
        /* Apply motor dynamics if enabled */
        if (physics->config.enable_motor_dynamics && actions != NULL) {
            physics_motor_dynamics(rpm_commands, actual_rpms, params, substep_dt, count);

            /* Unpack RPMs back to state */
            for (uint32_t i = 0; i < count; i++) {
                states->rpm_0[i] = actual_rpms[i * 4 + 0];
                states->rpm_1[i] = actual_rpms[i * 4 + 1];
                states->rpm_2[i] = actual_rpms[i * 4 + 2];
                states->rpm_3[i] = actual_rpms[i * 4 + 3];
            }
        } else if (actions != NULL) {
            /* No motor dynamics - set RPMs directly */
            for (uint32_t i = 0; i < count; i++) {
                states->rpm_0[i] = rpm_commands[i * 4 + 0];
                states->rpm_1[i] = rpm_commands[i * 4 + 1];
                states->rpm_2[i] = rpm_commands[i * 4 + 2];
                states->rpm_3[i] = rpm_commands[i * 4 + 3];
            }
        }

        /* RK4 integration */
        physics_rk4_integrate(physics, states, params, actions, substep_dt, count);

        /* Normalize quaternions */
        physics_normalize_quaternions(states, count);

        /* Clamp velocities */
        physics_clamp_velocities(states, params, count);
    }

    /* Sanitize state (detect NaN/Inf) */
    physics_sanitize_state(states, count);

    /* Update statistics */
    physics->step_count++;
}
