/**
 * Quadcopter Platform Implementation
 *
 * Implements PlatformVTable for X-configuration quadcopter.
 * In Phase 2 (additive), these are thin adapters calling existing physics.c functions.
 * In Phase 4, the physics.c functions will be extracted here directly.
 */

#include "../include/platform_quadcopter.h"
#include "physics.h"
#include <string.h>

/* ============================================================================
 * Section 1: Hot Path - Action Mapping
 * ============================================================================ */

static void quad_map_actions(const float* actions, float* commands,
                             float* const* params_ext, uint32_t params_ext_count,
                             uint32_t count) {
    (void)params_ext_count;

    const float* max_rpm = params_ext[QUAD_PEXT_MAX_RPM];

    for (uint32_t i = 0; i < count; i++) {
        float mr = max_rpm[i];
        commands[i * 4 + 0] = action_to_rpm(actions[i * 4 + 0], mr);
        commands[i * 4 + 1] = action_to_rpm(actions[i * 4 + 1], mr);
        commands[i * 4 + 2] = action_to_rpm(actions[i * 4 + 2], mr);
        commands[i * 4 + 3] = action_to_rpm(actions[i * 4 + 3], mr);
    }
}

/* ============================================================================
 * Section 2: Hot Path - Actuator Dynamics
 * ============================================================================ */

static void quad_actuator_dynamics(const float* commands,
                                   float** state_ext, uint32_t state_ext_count,
                                   float* const* params_ext, uint32_t params_ext_count,
                                   float dt, uint32_t count) {
    (void)state_ext_count;
    (void)params_ext_count;

    /* Deinterleave commands from AoS [count*4] to SoA [4][count] */
    /* Using scratch stack buffers for the deinterleaved commands */
    /* For now, just iterate and apply first-order lag inline */

    float* rpm_0 = state_ext[QUAD_EXT_RPM_0];
    float* rpm_1 = state_ext[QUAD_EXT_RPM_1];
    float* rpm_2 = state_ext[QUAD_EXT_RPM_2];
    float* rpm_3 = state_ext[QUAD_EXT_RPM_3];

    const float* motor_tau = params_ext[QUAD_PEXT_MOTOR_TAU];
    const float* max_rpm = params_ext[QUAD_PEXT_MAX_RPM];

    for (uint32_t i = 0; i < count; i++) {
        float tau = motor_tau[i];
        float mr = max_rpm[i];
        float alpha = dt / (tau + 1e-8f);
        if (alpha > 1.0f) alpha = 1.0f;

        float c0 = clampf(commands[i * 4 + 0], 0.0f, mr);
        float c1 = clampf(commands[i * 4 + 1], 0.0f, mr);
        float c2 = clampf(commands[i * 4 + 2], 0.0f, mr);
        float c3 = clampf(commands[i * 4 + 3], 0.0f, mr);

        rpm_0[i] = rpm_0[i] + (c0 - rpm_0[i]) * alpha;
        rpm_1[i] = rpm_1[i] + (c1 - rpm_1[i]) * alpha;
        rpm_2[i] = rpm_2[i] + (c2 - rpm_2[i]) * alpha;
        rpm_3[i] = rpm_3[i] + (c3 - rpm_3[i]) * alpha;
    }
}

/* ============================================================================
 * Section 3: Hot Path - Forces and Torques
 * ============================================================================ */

static void quad_compute_forces_torques(const RigidBodyStateSOA* rb,
                                        float* const* state_ext, uint32_t state_ext_count,
                                        float* const* params_ext, uint32_t params_ext_count,
                                        const RigidBodyParamsSOA* rb_params,
                                        float* forces_x, float* forces_y, float* forces_z,
                                        float* torques_x, float* torques_y, float* torques_z,
                                        uint32_t count) {
    (void)state_ext_count;
    (void)params_ext_count;
    (void)rb_params;

    const float* rpm_0 = state_ext[QUAD_EXT_RPM_0];
    const float* rpm_1 = state_ext[QUAD_EXT_RPM_1];
    const float* rpm_2 = state_ext[QUAD_EXT_RPM_2];
    const float* rpm_3 = state_ext[QUAD_EXT_RPM_3];

    const float* k_thrust = params_ext[QUAD_PEXT_K_THRUST];
    const float* k_torque = params_ext[QUAD_PEXT_K_TORQUE];
    const float* arm_length = params_ext[QUAD_PEXT_ARM_LENGTH];

    for (uint32_t i = 0; i < count; i++) {
        float r0 = rpm_0[i], r1 = rpm_1[i], r2 = rpm_2[i], r3 = rpm_3[i];
        float kt = k_thrust[i];
        float kq = k_torque[i];
        float arm = arm_length[i];

        /* Thrust per motor: T = k_thrust * rpm^2 */
        float T0 = kt * r0 * r0;
        float T1 = kt * r1 * r1;
        float T2 = kt * r2 * r2;
        float T3 = kt * r3 * r3;
        float total_thrust = T0 + T1 + T2 + T3;

        /* Rotate thrust to world frame */
        float qw = rb->quat_w[i], qx = rb->quat_x[i];
        float qy = rb->quat_y[i], qz = rb->quat_z[i];

        forces_x[i] = 2.0f * (qx * qz + qw * qy) * total_thrust;
        forces_y[i] = 2.0f * (qy * qz - qw * qx) * total_thrust;
        forces_z[i] = (qw * qw - qx * qx - qy * qy + qz * qz) * total_thrust;

        /* Torques (body frame, X-configuration) */
        torques_x[i] = arm * (T1 + T3 - T0 - T2);
        torques_y[i] = arm * (T0 + T1 - T2 - T3);
        torques_z[i] = kq * (r0 * r0 + r1 * r1 - r2 * r2 - r3 * r3);
    }
}

/* ============================================================================
 * Section 4: Platform Effects (Drag + Ground Effect)
 * ============================================================================ */

static void quad_apply_platform_effects(RigidBodyStateSOA* rb,
                                        float** state_ext, uint32_t state_ext_count,
                                        const RigidBodyParamsSOA* rb_params,
                                        float* const* params_ext, uint32_t params_ext_count,
                                        float* forces_x, float* forces_y, float* forces_z,
                                        const float* sdf_distances,
                                        const struct PhysicsConfig* physics_config,
                                        uint32_t count) {
    (void)state_ext;
    (void)state_ext_count;
    (void)rb_params;
    (void)params_ext_count;

    const float* k_drag = params_ext[QUAD_PEXT_K_DRAG];
    const float* k_ang_damp = params_ext[QUAD_PEXT_K_ANG_DAMP];

    /* Apply drag: F_drag = -k_drag * |v| * v */
    if (physics_config->enable_drag) {
        for (uint32_t i = 0; i < count; i++) {
            float vx = rb->vel_x[i], vy = rb->vel_y[i], vz = rb->vel_z[i];
            float kd = k_drag[i];
            float speed = sqrtf(vx * vx + vy * vy + vz * vz + 1e-8f);
            float drag_factor = -kd * speed;

            forces_x[i] += drag_factor * vx;
            forces_y[i] += drag_factor * vy;
            forces_z[i] += drag_factor * vz;
        }
    }

    /* Apply angular damping: tau_damp = -k_ang_damp * omega (body frame) */
    if (physics_config->enable_drag) {
        for (uint32_t i = 0; i < count; i++) {
            float kad = k_ang_damp[i];
            rb->omega_x[i] *= (1.0f - kad);
            rb->omega_y[i] *= (1.0f - kad);
            rb->omega_z[i] *= (1.0f - kad);
        }
    }

    /* Apply ground effect using SDF distances */
    if (physics_config->enable_ground_effect && sdf_distances != NULL) {
        float gh = physics_config->ground_effect_height;
        float gc = physics_config->ground_effect_coeff;

        for (uint32_t i = 0; i < count; i++) {
            float sdf = sdf_distances[i];
            if (sdf > 0.0f && sdf < gh) {
                float ratio = -sdf / (gh + 1e-8f);
                float exp_val = expf(ratio);
                float k_ge = 1.0f + (gc - 1.0f) * exp_val;
                forces_z[i] *= k_ge;
            }
        }
    }
}

/* ============================================================================
 * Section 5: Lifecycle Functions
 * ============================================================================ */

static void quad_init_state(float** state_ext, uint32_t ext_count, uint32_t index) {
    (void)ext_count;
    state_ext[QUAD_EXT_RPM_0][index] = 0.0f;
    state_ext[QUAD_EXT_RPM_1][index] = 0.0f;
    state_ext[QUAD_EXT_RPM_2][index] = 0.0f;
    state_ext[QUAD_EXT_RPM_3][index] = 0.0f;
}

static void quad_reset_state(float** state_ext, uint32_t ext_count, uint32_t index) {
    quad_init_state(state_ext, ext_count, index);
}

static void quad_init_params(float** params_ext, uint32_t ext_count, uint32_t index) {
    (void)ext_count;
    params_ext[QUAD_PEXT_ARM_LENGTH][index] = 0.1f;
    params_ext[QUAD_PEXT_K_THRUST][index] = 3.16e-10f;
    params_ext[QUAD_PEXT_K_TORQUE][index] = 7.94e-12f;
    params_ext[QUAD_PEXT_MOTOR_TAU][index] = 0.02f;
    params_ext[QUAD_PEXT_MAX_RPM][index] = 2500.0f;
    params_ext[QUAD_PEXT_K_DRAG][index] = 0.1f;
    params_ext[QUAD_PEXT_K_ANG_DAMP][index] = 0.01f;
}

/* ============================================================================
 * Section 6: Configuration Functions
 * ============================================================================ */

static size_t quad_config_size(void) {
    return sizeof(QuadcopterConfig);
}

static void quad_config_set_defaults(void* platform_config) {
    QuadcopterConfig* cfg = (QuadcopterConfig*)platform_config;
    cfg->arm_length = 0.1f;
    cfg->k_thrust = 3.16e-10f;
    cfg->k_torque = 7.94e-12f;
    cfg->motor_tau = 0.02f;
    cfg->max_rpm = 2500.0f;
    cfg->k_drag = 0.1f;
    cfg->k_ang_damp = 0.01f;
}

static void quad_config_to_params(const void* platform_config,
                                  float** params_ext, uint32_t ext_count,
                                  RigidBodyParamsSOA* rb_params, uint32_t index) {
    (void)ext_count;
    (void)rb_params;
    const QuadcopterConfig* cfg = (const QuadcopterConfig*)platform_config;

    params_ext[QUAD_PEXT_ARM_LENGTH][index] = cfg->arm_length;
    params_ext[QUAD_PEXT_K_THRUST][index] = cfg->k_thrust;
    params_ext[QUAD_PEXT_K_TORQUE][index] = cfg->k_torque;
    params_ext[QUAD_PEXT_MOTOR_TAU][index] = cfg->motor_tau;
    params_ext[QUAD_PEXT_MAX_RPM][index] = cfg->max_rpm;
    params_ext[QUAD_PEXT_K_DRAG][index] = cfg->k_drag;
    params_ext[QUAD_PEXT_K_ANG_DAMP][index] = cfg->k_ang_damp;
}

/* ============================================================================
 * Section 7: Static VTable Instance
 * ============================================================================ */

const PlatformVTable PLATFORM_QUADCOPTER = {
    .name                    = "quadcopter",
    .action_dim              = 4,
    .state_extension_count   = QUAD_STATE_EXT_COUNT,
    .params_extension_count  = QUAD_PARAMS_EXT_COUNT,

    .map_actions             = quad_map_actions,
    .actuator_dynamics       = quad_actuator_dynamics,
    .compute_forces_torques  = quad_compute_forces_torques,
    .apply_platform_effects  = quad_apply_platform_effects,

    .init_state              = quad_init_state,
    .reset_state             = quad_reset_state,
    .init_params             = quad_init_params,

    .config_size             = quad_config_size,
    .config_set_defaults     = quad_config_set_defaults,
    .config_to_params        = quad_config_to_params,
};
