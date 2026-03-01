/**
 * Physics Engine Module Implementation
 *
 * RK4-integrated rigid body physics with platform-agnostic VTable dispatch.
 * Forces/torques, actuator dynamics, and platform effects are dispatched
 * through the PlatformVTable set at physics_create() time.
 */

#include "../include/physics.h"
#include "platform.h"
#include <math.h>

/* Small epsilon for numerical stability */
#define PHYSICS_EPSILON 1e-8f

/* ============================================================================
 * Section 1: Lifecycle Functions
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
                              const PhysicsConfig* config, uint32_t max_agents,
                              const PlatformVTable* vtable) {
    if (persistent_arena == NULL || scratch_arena == NULL || max_agents == 0) {
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

    physics->vtable = vtable;
    physics->scratch_arena = scratch_arena;
    physics->max_agents = max_agents;
    physics->step_count = 0;
    physics->total_integration_time = 0.0f;
    physics->sdf_distances = NULL;

    /* Allocate RK4 scratch buffers (5 RigidBodyStateSOA structures) */
    physics->k1 = rigid_body_state_create(persistent_arena, max_agents);
    physics->k2 = rigid_body_state_create(persistent_arena, max_agents);
    physics->k3 = rigid_body_state_create(persistent_arena, max_agents);
    physics->k4 = rigid_body_state_create(persistent_arena, max_agents);
    physics->temp_state = rigid_body_state_create(persistent_arena, max_agents);

    if (physics->k1 == NULL || physics->k2 == NULL || physics->k3 == NULL ||
        physics->k4 == NULL || physics->temp_state == NULL) {
        return NULL;
    }

    /* Allocate force/torque buffers (6 float arrays, 32-byte aligned) */
    size_t aligned_size = align_up_size(max_agents * sizeof(float), 32);

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

    /* Arena memory can't be individually freed; zero entire struct for safety */
    memset(physics, 0, sizeof(PhysicsSystem));
}

size_t physics_memory_size(uint32_t max_agents) {
    if (max_agents == 0) {
        return 0;
    }

    size_t aligned_array = align_up_size(max_agents * sizeof(float), 32);

    /* PhysicsSystem struct */
    size_t total = sizeof(PhysicsSystem);

    /* 5 RigidBodyStateSOA structures for RK4 */
    total += 5 * rigid_body_state_memory_size(max_agents);

    /* 6 force/torque arrays */
    total += 6 * aligned_array;

    return total;
}

/* ============================================================================
 * Section 2: Numerical Stability Functions
 * ============================================================================ */

void physics_normalize_quaternions(PlatformStateSOA* states, uint32_t count) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");

    if (states == NULL || count == 0) {
        return;
    }

    RigidBodyStateSOA* rb = &states->rigid_body;

    SIMD_LOOP_START(count);

    /* SIMD loop */
    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_float w = simd_load_ps(&rb->quat_w[i]);
        simd_float x = simd_load_ps(&rb->quat_x[i]);
        simd_float y = simd_load_ps(&rb->quat_y[i]);
        simd_float z = simd_load_ps(&rb->quat_z[i]);

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

        simd_store_ps(&rb->quat_w[i], simd_mul_ps(w, inv_mag));
        simd_store_ps(&rb->quat_x[i], simd_mul_ps(x, inv_mag));
        simd_store_ps(&rb->quat_y[i], simd_mul_ps(y, inv_mag));
        simd_store_ps(&rb->quat_z[i], simd_mul_ps(z, inv_mag));
    }

    /* Scalar remainder */
    SIMD_LOOP_REMAINDER(i, count) {
        float w = rb->quat_w[i], x = rb->quat_x[i];
        float y = rb->quat_y[i], z = rb->quat_z[i];
        float inv_mag = 1.0f / sqrtf(w*w + x*x + y*y + z*z + PHYSICS_EPSILON);
        rb->quat_w[i] = w * inv_mag;
        rb->quat_x[i] = x * inv_mag;
        rb->quat_y[i] = y * inv_mag;
        rb->quat_z[i] = z * inv_mag;
    }
}

void physics_clamp_velocities(PlatformStateSOA* states, const PlatformParamsSOA* params, uint32_t count) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    FOUNDATION_ASSERT(params != NULL, "params is NULL");

    if (states == NULL || params == NULL || count == 0) {
        return;
    }

    RigidBodyStateSOA* rb = &states->rigid_body;
    const RigidBodyParamsSOA* rbp = &params->rigid_body;

    SIMD_LOOP_START(count);

    /* SIMD loop for linear velocity clamping */
    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_float vx = simd_load_ps(&rb->vel_x[i]);
        simd_float vy = simd_load_ps(&rb->vel_y[i]);
        simd_float vz = simd_load_ps(&rb->vel_z[i]);
        simd_float max_vel = simd_load_ps(&rbp->max_vel[i]);

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

        simd_store_ps(&rb->vel_x[i], simd_mul_ps(vx, scale));
        simd_store_ps(&rb->vel_y[i], simd_mul_ps(vy, scale));
        simd_store_ps(&rb->vel_z[i], simd_mul_ps(vz, scale));
    }

    /* SIMD loop for angular velocity clamping */
    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_float ox = simd_load_ps(&rb->omega_x[i]);
        simd_float oy = simd_load_ps(&rb->omega_y[i]);
        simd_float oz = simd_load_ps(&rb->omega_z[i]);
        simd_float max_omega = simd_load_ps(&rbp->max_omega[i]);

        simd_float omega_sq = simd_fmadd_ps(ox, ox,
                             simd_fmadd_ps(oy, oy,
                             simd_mul_ps(oz, oz)));

        simd_float max_omega_sq = simd_mul_ps(max_omega, max_omega);
        simd_float needs_clamp = simd_cmp_gt_ps(omega_sq, max_omega_sq);
        simd_float inv_omega = simd_rsqrt_ps(simd_add_ps(omega_sq, simd_set1_ps(PHYSICS_EPSILON)));
        simd_float scale = simd_mul_ps(max_omega, inv_omega);
        scale = simd_blendv_ps(simd_set1_ps(1.0f), scale, needs_clamp);

        simd_store_ps(&rb->omega_x[i], simd_mul_ps(ox, scale));
        simd_store_ps(&rb->omega_y[i], simd_mul_ps(oy, scale));
        simd_store_ps(&rb->omega_z[i], simd_mul_ps(oz, scale));
    }

    /* Scalar remainder */
    SIMD_LOOP_REMAINDER(i, count) {
        /* Linear velocity */
        float vx = rb->vel_x[i], vy = rb->vel_y[i], vz = rb->vel_z[i];
        float speed_sq = vx*vx + vy*vy + vz*vz;
        float max_vel = rbp->max_vel[i];
        if (speed_sq > max_vel * max_vel) {
            float scale = max_vel / sqrtf(speed_sq + PHYSICS_EPSILON);
            rb->vel_x[i] = vx * scale;
            rb->vel_y[i] = vy * scale;
            rb->vel_z[i] = vz * scale;
        }

        /* Angular velocity */
        float ox = rb->omega_x[i], oy = rb->omega_y[i], oz = rb->omega_z[i];
        float omega_sq = ox*ox + oy*oy + oz*oz;
        float max_omega = rbp->max_omega[i];
        if (omega_sq > max_omega * max_omega) {
            float scale = max_omega / sqrtf(omega_sq + PHYSICS_EPSILON);
            rb->omega_x[i] = ox * scale;
            rb->omega_y[i] = oy * scale;
            rb->omega_z[i] = oz * scale;
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

uint32_t physics_sanitize_state(PlatformStateSOA* states, uint32_t count) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");

    if (states == NULL || count == 0) {
        return 0;
    }

    RigidBodyStateSOA* rb = &states->rigid_body;
    uint32_t reset_count = 0;

    for (uint32_t i = 0; i < count; i++) {
        bool needs_reset = false;

        /* Check for NaN/Inf in position */
        if (!isfinite(rb->pos_x[i]) || !isfinite(rb->pos_y[i]) ||
            !isfinite(rb->pos_z[i])) {
#ifndef NDEBUG
            FOUNDATION_ASSERT(0, "NaN/Inf detected in agent position after physics step");
#endif
            needs_reset = true;
        }

        /* Check for NaN/Inf in velocity */
        if (!isfinite(rb->vel_x[i]) || !isfinite(rb->vel_y[i]) ||
            !isfinite(rb->vel_z[i])) {
            needs_reset = true;
        }

        /* Check for NaN/Inf in quaternion */
        if (!isfinite(rb->quat_w[i]) || !isfinite(rb->quat_x[i]) ||
            !isfinite(rb->quat_y[i]) || !isfinite(rb->quat_z[i])) {
            needs_reset = true;
        }

        /* Check for NaN/Inf in angular velocity */
        if (!isfinite(rb->omega_x[i]) || !isfinite(rb->omega_y[i]) ||
            !isfinite(rb->omega_z[i])) {
            needs_reset = true;
        }

        /* Check extensions for NaN/Inf */
        for (uint32_t e = 0; e < states->extension_count; e++) {
            if (states->extension[e] != NULL && !isfinite(states->extension[e][i])) {
                needs_reset = true;
                break;
            }
        }

        if (needs_reset) {
            /* Reset rigid body to safe state */
            rb->pos_x[i] = 0.0f;
            rb->pos_y[i] = 0.0f;
            rb->pos_z[i] = 0.0f;

            rb->vel_x[i] = 0.0f;
            rb->vel_y[i] = 0.0f;
            rb->vel_z[i] = 0.0f;

            rb->quat_w[i] = 1.0f;
            rb->quat_x[i] = 0.0f;
            rb->quat_y[i] = 0.0f;
            rb->quat_z[i] = 0.0f;

            rb->omega_x[i] = 0.0f;
            rb->omega_y[i] = 0.0f;
            rb->omega_z[i] = 0.0f;

            /* Zero extensions */
            for (uint32_t e = 0; e < states->extension_count; e++) {
                if (states->extension[e] != NULL) {
                    states->extension[e][i] = 0.0f;
                }
            }

            reset_count++;
        }
    }

    return reset_count;
}

/* ============================================================================
 * Section 3: Derivative Computation (generic via VTable)
 * ============================================================================ */

void physics_compute_derivatives(PhysicsSystem* physics,
                                 const PlatformStateSOA* states,
                                 const PlatformParamsSOA* params,
                                 RigidBodyStateSOA* derivatives,
                                 uint32_t count) {
    FOUNDATION_ASSERT(physics != NULL, "physics is NULL");
    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    FOUNDATION_ASSERT(params != NULL, "params is NULL");
    FOUNDATION_ASSERT(derivatives != NULL, "derivatives is NULL");

    if (physics == NULL || states == NULL || params == NULL ||
        derivatives == NULL || count == 0) {
        return;
    }

    const RigidBodyStateSOA* rb = &states->rigid_body;
    const RigidBodyParamsSOA* rbp = &params->rigid_body;
    const PhysicsConfig* config = &physics->config;

    /* Position derivative = velocity */
    SIMD_LOOP_START(count);

    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_store_ps(&derivatives->pos_x[i], simd_load_ps(&rb->vel_x[i]));
        simd_store_ps(&derivatives->pos_y[i], simd_load_ps(&rb->vel_y[i]));
        simd_store_ps(&derivatives->pos_z[i], simd_load_ps(&rb->vel_z[i]));
    }

    SIMD_LOOP_REMAINDER(i, count) {
        derivatives->pos_x[i] = rb->vel_x[i];
        derivatives->pos_y[i] = rb->vel_y[i];
        derivatives->pos_z[i] = rb->vel_z[i];
    }

    /* Compute platform-specific forces and torques via vtable */
    float* forces_x = physics->forces_x;
    float* forces_y = physics->forces_y;
    float* forces_z = physics->forces_z;
    float* torques_x = physics->torques_x;
    float* torques_y = physics->torques_y;
    float* torques_z = physics->torques_z;

    if (physics->vtable && physics->vtable->compute_forces_torques) {
        physics->vtable->compute_forces_torques(
            rb,
            (float* const*)states->extension, states->extension_count,
            (float* const*)params->extension, params->extension_count,
            rbp,
            forces_x, forces_y, forces_z,
            torques_x, torques_y, torques_z,
            count);
    } else {
        /* No platform forces - zero everything */
        memset(forces_x, 0, count * sizeof(float));
        memset(forces_y, 0, count * sizeof(float));
        memset(forces_z, 0, count * sizeof(float));
        memset(torques_x, 0, count * sizeof(float));
        memset(torques_y, 0, count * sizeof(float));
        memset(torques_z, 0, count * sizeof(float));
    }

    /* Add gravity and convert forces to accelerations */
    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_float fx = simd_load_ps(&forces_x[i]);
        simd_float fy = simd_load_ps(&forces_y[i]);
        simd_float fz = simd_load_ps(&forces_z[i]);
        simd_float mass = simd_load_ps(&rbp->mass[i]);
        simd_float gravity = simd_load_ps(&rbp->gravity[i]);
        simd_float inv_mass = simd_div_ps(simd_set1_ps(1.0f),
                             simd_add_ps(mass, simd_set1_ps(PHYSICS_EPSILON)));

        /* a = F/m, subtract gravity in Z */
        simd_store_ps(&derivatives->vel_x[i], simd_mul_ps(fx, inv_mass));
        simd_store_ps(&derivatives->vel_y[i], simd_mul_ps(fy, inv_mass));
        simd_store_ps(&derivatives->vel_z[i],
            simd_sub_ps(simd_mul_ps(fz, inv_mass), gravity));
    }

    SIMD_LOOP_REMAINDER(i, count) {
        float mass = rbp->mass[i];
        float gravity = rbp->gravity[i];
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
        simd_float tau_x = simd_load_ps(&torques_x[i]);
        simd_float tau_y = simd_load_ps(&torques_y[i]);
        simd_float tau_z = simd_load_ps(&torques_z[i]);

        simd_float ox = simd_load_ps(&rb->omega_x[i]);
        simd_float oy = simd_load_ps(&rb->omega_y[i]);
        simd_float oz = simd_load_ps(&rb->omega_z[i]);

        simd_float ixx = simd_load_ps(&rbp->ixx[i]);
        simd_float iyy = simd_load_ps(&rbp->iyy[i]);
        simd_float izz = simd_load_ps(&rbp->izz[i]);

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
        float tau_x = torques_x[i];
        float tau_y = torques_y[i];
        float tau_z = torques_z[i];

        float ox = rb->omega_x[i];
        float oy = rb->omega_y[i];
        float oz = rb->omega_z[i];

        float ixx = rbp->ixx[i];
        float iyy = rbp->iyy[i];
        float izz = rbp->izz[i];

        derivatives->omega_x[i] = (tau_x - (izz - iyy) * oy * oz) / (ixx + PHYSICS_EPSILON);
        derivatives->omega_y[i] = (tau_y - (ixx - izz) * oz * ox) / (iyy + PHYSICS_EPSILON);
        derivatives->omega_z[i] = (tau_z - (iyy - ixx) * ox * oy) / (izz + PHYSICS_EPSILON);
    }

    /* Clamp angular accelerations */
    physics_clamp_accelerations(derivatives->omega_x, derivatives->omega_y, derivatives->omega_z,
                               config->max_angular_accel, count);

    /* Compute quaternion derivative: q_dot = 0.5 * q * [0, omega] */
    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_float qw = simd_load_ps(&rb->quat_w[i]);
        simd_float qx = simd_load_ps(&rb->quat_x[i]);
        simd_float qy = simd_load_ps(&rb->quat_y[i]);
        simd_float qz = simd_load_ps(&rb->quat_z[i]);

        simd_float ox = simd_load_ps(&rb->omega_x[i]);
        simd_float oy = simd_load_ps(&rb->omega_y[i]);
        simd_float oz = simd_load_ps(&rb->omega_z[i]);

        simd_float half = simd_set1_ps(0.5f);

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
        float qw = rb->quat_w[i], qx = rb->quat_x[i];
        float qy = rb->quat_y[i], qz = rb->quat_z[i];
        float ox = rb->omega_x[i], oy = rb->omega_y[i], oz = rb->omega_z[i];

        derivatives->quat_w[i] = 0.5f * (-qx*ox - qy*oy - qz*oz);
        derivatives->quat_x[i] = 0.5f * (qw*ox + qy*oz - qz*oy);
        derivatives->quat_y[i] = 0.5f * (qw*oy + qz*ox - qx*oz);
        derivatives->quat_z[i] = 0.5f * (qw*oz + qx*oy - qy*ox);
    }
}

/* ============================================================================
 * Section 4: RK4 Integration (operates on RigidBodyStateSOA)
 * ============================================================================ */

void physics_rk4_substep(const RigidBodyStateSOA* current, const RigidBodyStateSOA* derivative,
                         RigidBodyStateSOA* output, float dt_scale, uint32_t count) {
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
    }
}

void physics_rk4_combine(RigidBodyStateSOA* states, const RigidBodyStateSOA* k1,
                         const RigidBodyStateSOA* k2, const RigidBodyStateSOA* k3,
                         const RigidBodyStateSOA* k4, float dt, uint32_t count) {
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

void physics_rk4_integrate(PhysicsSystem* physics, PlatformStateSOA* states,
                           const PlatformParamsSOA* params,
                           float dt, uint32_t count) {
    FOUNDATION_ASSERT(physics != NULL, "physics is NULL");
    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    FOUNDATION_ASSERT(params != NULL, "params is NULL");

    if (physics == NULL || states == NULL || params == NULL || count == 0) {
        return;
    }

    RigidBodyStateSOA* rb = &states->rigid_body;

    /* k1 = f(t, y) */
    physics_compute_derivatives(physics, states, params, physics->k1, count);

    /* k2 = f(t + dt/2, y + k1*dt/2) */
    physics_rk4_substep(rb, physics->k1, physics->temp_state, dt * 0.5f, count);
    /* Copy temp_state back to rb for derivative evaluation */
    {
        /* Temporarily swap rb pointers for k2 evaluation:
         * We need to compute derivatives at temp_state position, but with
         * the platform's extension arrays (RPMs etc.) unchanged.
         * Strategy: copy temp_state into rb, compute, then restore. */
        /* Actually, we create a temporary PlatformStateSOA on the stack
         * that points to temp_state's rb arrays but keeps the original extensions */
        PlatformStateSOA temp_plat;
        temp_plat.rigid_body = *physics->temp_state;
        temp_plat.extension = states->extension;
        temp_plat.extension_count = states->extension_count;
        physics_compute_derivatives(physics, &temp_plat, params, physics->k2, count);
    }

    /* k3 = f(t + dt/2, y + k2*dt/2) */
    physics_rk4_substep(rb, physics->k2, physics->temp_state, dt * 0.5f, count);
    {
        PlatformStateSOA temp_plat;
        temp_plat.rigid_body = *physics->temp_state;
        temp_plat.extension = states->extension;
        temp_plat.extension_count = states->extension_count;
        physics_compute_derivatives(physics, &temp_plat, params, physics->k3, count);
    }

    /* k4 = f(t + dt, y + k3*dt) */
    physics_rk4_substep(rb, physics->k3, physics->temp_state, dt, count);
    {
        PlatformStateSOA temp_plat;
        temp_plat.rigid_body = *physics->temp_state;
        temp_plat.extension = states->extension;
        temp_plat.extension_count = states->extension_count;
        physics_compute_derivatives(physics, &temp_plat, params, physics->k4, count);
    }

    /* y_new = y + (k1 + 2*k2 + 2*k3 + k4) * dt/6 */
    physics_rk4_combine(rb, physics->k1, physics->k2, physics->k3, physics->k4, dt, count);
}

/* ============================================================================
 * Section 5: Main Physics Step (generic via VTable)
 * ============================================================================ */

void physics_step(PhysicsSystem* physics, PlatformStateSOA* states,
                  const PlatformParamsSOA* params, const float* actions, uint32_t count) {
    physics_step_dt(physics, states, params, actions, count, physics->config.dt);
}

void physics_step_dt(PhysicsSystem* physics, PlatformStateSOA* states,
                     const PlatformParamsSOA* params, const float* actions,
                     uint32_t count, float dt) {
    FOUNDATION_ASSERT(physics != NULL, "physics is NULL");
    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    FOUNDATION_ASSERT(params != NULL, "params is NULL");

    if (physics == NULL || states == NULL || params == NULL || count == 0) {
        return;
    }

    const PlatformVTable* vtable = physics->vtable;

    /* Calculate substep dt */
    uint32_t substeps = physics->config.substeps;
    if (substeps == 0) substeps = 1;
    float substep_dt = dt / (float)substeps;

    /* Reset scratch arena -- all allocations below are local to this call */
    arena_reset(physics->scratch_arena);

    /* Allocate command buffer from scratch arena */
    float* commands = NULL;
    if (actions != NULL && vtable != NULL) {
        size_t cmd_size = align_up_size(count * vtable->action_dim * sizeof(float), 32);
        commands = (float*)arena_alloc_aligned(physics->scratch_arena, cmd_size, 32);
    }

    /* Run substeps */
    for (uint32_t s = 0; s < substeps; s++) {
        /* Map actions and apply actuator dynamics via vtable */
        if (vtable != NULL && actions != NULL && commands != NULL) {
            /* Map normalized actions to platform-specific commands */
            if (vtable->map_actions) {
                vtable->map_actions(actions, commands,
                                    (float* const*)params->extension, params->extension_count,
                                    count);
            }

            /* Apply actuator dynamics (e.g., motor lag) */
            if (vtable->actuator_dynamics && physics->config.enable_motor_dynamics) {
                vtable->actuator_dynamics(commands,
                                          states->extension, states->extension_count,
                                          (float* const*)params->extension, params->extension_count,
                                          substep_dt, count);
            } else if (vtable->actuator_dynamics) {
                /* Motor dynamics disabled: set RPMs directly from commands */
                uint32_t adim = vtable->action_dim;
                for (uint32_t i = 0; i < count; i++) {
                    for (uint32_t d = 0; d < adim && d < states->extension_count; d++) {
                        states->extension[d][i] = commands[i * adim + d];
                    }
                }
            }
        }

        /* RK4 integration of rigid body state */
        physics_rk4_integrate(physics, states, params, substep_dt, count);

        /* Normalize quaternions */
        physics_normalize_quaternions(states, count);

        /* Clamp velocities */
        physics_clamp_velocities(states, params, count);

        /* Apply platform-specific effects (drag, ground effect, etc.) */
        if (vtable != NULL && vtable->apply_platform_effects) {
            vtable->apply_platform_effects(
                &states->rigid_body,
                states->extension, states->extension_count,
                &params->rigid_body,
                (float* const*)params->extension, params->extension_count,
                physics->forces_x, physics->forces_y, physics->forces_z,
                physics->sdf_distances,
                &physics->config,
                count);
        }
    }

    /* Sanitize state (detect NaN/Inf) */
    physics_sanitize_state(states, count);

    /* Update statistics */
    physics->step_count++;
}
