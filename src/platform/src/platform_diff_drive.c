/**
 * Diff-Drive Platform Implementation
 *
 * Implements PlatformVTable for a two-wheel differential drive ground robot.
 *
 * Physics model:
 * - Spring-based velocity tracking: high-stiffness springs map desired
 *   diff-drive kinematics to forces/torques on the rigid body.
 * - Ground plane constraint: z=0, yaw-only orientation.
 * - First-order lag on wheel velocities (tau = 0.05s).
 * - Lateral friction damping to prevent sliding.
 */

#include "../include/platform_diff_drive.h"
#include "physics.h"
#include <math.h>
#include <string.h>

/* Wheel actuator time constant (seconds) */
#define DD_WHEEL_TAU 0.05f

/* ============================================================================
 * Section 1: Hot Path - Action Mapping
 * ============================================================================ */

static void dd_map_actions(const float* actions, float* commands,
                           float* const* params_ext, uint32_t params_ext_count,
                           uint32_t count) {
    (void)params_ext_count;

    const float* max_wheel_vel = params_ext[DD_PEXT_MAX_WHEEL_VEL];

    for (uint32_t i = 0; i < count; i++) {
        float mwv = max_wheel_vel[i];
        /* Actions in [-1, 1] map to [-max_wheel_vel, +max_wheel_vel] */
        float al = clampf(actions[i * 2 + 0], -1.0f, 1.0f);
        float ar = clampf(actions[i * 2 + 1], -1.0f, 1.0f);
        commands[i * 2 + 0] = al * mwv;
        commands[i * 2 + 1] = ar * mwv;
    }
}

/* ============================================================================
 * Section 2: Hot Path - Actuator Dynamics
 * ============================================================================ */

static void dd_actuator_dynamics(const float* commands,
                                 float** state_ext, uint32_t state_ext_count,
                                 float* const* params_ext, uint32_t params_ext_count,
                                 float dt, uint32_t count) {
    (void)state_ext_count;
    (void)params_ext;
    (void)params_ext_count;

    float* wl = state_ext[DD_EXT_WHEEL_VEL_L];
    float* wr = state_ext[DD_EXT_WHEEL_VEL_R];

    float alpha = dt / (DD_WHEEL_TAU + 1e-8f);
    if (alpha > 1.0f) alpha = 1.0f;

    for (uint32_t i = 0; i < count; i++) {
        float target_l = commands[i * 2 + 0];
        float target_r = commands[i * 2 + 1];

        wl[i] = wl[i] + (target_l - wl[i]) * alpha;
        wr[i] = wr[i] + (target_r - wr[i]) * alpha;
    }
}

/* ============================================================================
 * Section 3: Hot Path - Forces and Torques
 * ============================================================================ */

static void dd_compute_forces_torques(const RigidBodyStateSOA* rb,
                                      float* const* state_ext, uint32_t state_ext_count,
                                      float* const* params_ext, uint32_t params_ext_count,
                                      const RigidBodyParamsSOA* rb_params,
                                      float* forces_x, float* forces_y, float* forces_z,
                                      float* torques_x, float* torques_y, float* torques_z,
                                      uint32_t count) {
    (void)state_ext_count;
    (void)params_ext_count;

    const float* wl = state_ext[DD_EXT_WHEEL_VEL_L];
    const float* wr = state_ext[DD_EXT_WHEEL_VEL_R];

    const float* wheel_radius = params_ext[DD_PEXT_WHEEL_RADIUS];
    const float* axle_length = params_ext[DD_PEXT_AXLE_LENGTH];

    const float k_spring = 50.0f;
    const float k_yaw = 50.0f;

    for (uint32_t i = 0; i < count; i++) {
        float R = wheel_radius[i];
        float L = axle_length[i];
        float mass = rb_params->mass[i];
        float izz = rb_params->izz[i];

        /* Diff-drive kinematics */
        float v_forward = (wl[i] + wr[i]) * R * 0.5f;
        float yaw_rate_target = (wr[i] - wl[i]) * R / (L + 1e-8f);

        /* Get body-frame forward direction from quaternion */
        float qw = rb->quat_w[i];
        float qx = rb->quat_x[i];
        float qy = rb->quat_y[i];
        float qz = rb->quat_z[i];

        /* Forward = rotate (1,0,0) by quaternion */
        float fx_body = 1.0f - 2.0f * (qy * qy + qz * qz);
        float fy_body = 2.0f * (qx * qy + qw * qz);
        /* fz_body not needed for ground robot */

        /* Desired world-frame velocity */
        float vx_desired = fx_body * v_forward;
        float vy_desired = fy_body * v_forward;

        /* Force = stiffness * (desired_velocity - current_velocity) * mass */
        forces_x[i] = k_spring * (vx_desired - rb->vel_x[i]) * mass;
        forces_y[i] = k_spring * (vy_desired - rb->vel_y[i]) * mass;
        forces_z[i] = 0.0f;  /* No vertical force */

        /* Yaw torque: high-stiffness spring to target yaw rate */
        torques_x[i] = 0.0f;
        torques_y[i] = 0.0f;
        torques_z[i] = k_yaw * (yaw_rate_target - rb->omega_z[i]) * izz;
    }
}

/* ============================================================================
 * Section 4: Platform Effects (Ground Constraint + Friction)
 * ============================================================================ */

static void dd_apply_platform_effects(RigidBodyStateSOA* rb,
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
    (void)params_ext;
    (void)params_ext_count;
    (void)forces_x;
    (void)forces_y;
    (void)forces_z;
    (void)sdf_distances;
    (void)physics_config;

    const float k_friction = 0.95f;

    for (uint32_t i = 0; i < count; i++) {
        /* Ground constraint: keep on z=0 plane */
        rb->pos_z[i] = 0.0f;
        rb->vel_z[i] = 0.0f;

        /* Zero roll/pitch: reconstruct yaw-only quaternion */
        float qw = rb->quat_w[i];
        float qx = rb->quat_x[i];
        float qy = rb->quat_y[i];
        float qz = rb->quat_z[i];

        float yaw = atan2f(2.0f * (qw * qz + qx * qy),
                           1.0f - 2.0f * (qy * qy + qz * qz));
        rb->quat_w[i] = cosf(yaw * 0.5f);
        rb->quat_x[i] = 0.0f;
        rb->quat_y[i] = 0.0f;
        rb->quat_z[i] = sinf(yaw * 0.5f);

        /* Zero roll/pitch angular velocity */
        rb->omega_x[i] = 0.0f;
        rb->omega_y[i] = 0.0f;

        /* Lateral velocity damping (Coulomb-like friction) */
        /* Recompute forward direction from cleaned-up quaternion */
        float cqw = rb->quat_w[i];
        float cqz = rb->quat_z[i];
        float fx_body = 1.0f - 2.0f * cqz * cqz;  /* qx=qy=0 simplification */
        float fy_body = 2.0f * cqw * cqz;

        /* Lateral axis = perpendicular to forward on ground plane */
        float lat_x = -fy_body;
        float lat_y = fx_body;

        float v_lat = rb->vel_x[i] * lat_x + rb->vel_y[i] * lat_y;
        rb->vel_x[i] -= v_lat * lat_x * k_friction;
        rb->vel_y[i] -= v_lat * lat_y * k_friction;
    }
}

/* ============================================================================
 * Section 5: Lifecycle Functions
 * ============================================================================ */

static void dd_init_state(float** state_ext, uint32_t ext_count, uint32_t index) {
    (void)ext_count;
    state_ext[DD_EXT_WHEEL_VEL_L][index] = 0.0f;
    state_ext[DD_EXT_WHEEL_VEL_R][index] = 0.0f;
}

static void dd_reset_state(float** state_ext, uint32_t ext_count, uint32_t index) {
    dd_init_state(state_ext, ext_count, index);
}

static void dd_init_params(float** params_ext, uint32_t ext_count, uint32_t index) {
    (void)ext_count;
    params_ext[DD_PEXT_WHEEL_RADIUS][index] = 0.033f;
    params_ext[DD_PEXT_AXLE_LENGTH][index] = 0.16f;
    params_ext[DD_PEXT_MAX_WHEEL_VEL][index] = 6.67f;
}

/* ============================================================================
 * Section 6: Configuration Functions
 * ============================================================================ */

static size_t dd_config_size(void) {
    return sizeof(DiffDriveConfig);
}

static void dd_config_set_defaults(void* platform_config) {
    DiffDriveConfig* cfg = (DiffDriveConfig*)platform_config;
    cfg->wheel_radius = 0.033f;
    cfg->axle_length = 0.16f;
    cfg->max_wheel_vel = 6.67f;
}

static void dd_config_to_params(const void* platform_config,
                                float** params_ext, uint32_t ext_count,
                                RigidBodyParamsSOA* rb_params, uint32_t index) {
    (void)ext_count;
    (void)rb_params;
    const DiffDriveConfig* cfg = (const DiffDriveConfig*)platform_config;

    params_ext[DD_PEXT_WHEEL_RADIUS][index] = cfg->wheel_radius;
    params_ext[DD_PEXT_AXLE_LENGTH][index] = cfg->axle_length;
    params_ext[DD_PEXT_MAX_WHEEL_VEL][index] = cfg->max_wheel_vel;
}

/* ============================================================================
 * Section 7: Static VTable Instance
 * ============================================================================ */

const PlatformVTable PLATFORM_DIFF_DRIVE = {
    .name                    = "diff_drive",
    .action_dim              = 2,
    .state_extension_count   = DD_STATE_EXT_COUNT,
    .params_extension_count  = DD_PARAMS_EXT_COUNT,

    .map_actions             = dd_map_actions,
    .actuator_dynamics       = dd_actuator_dynamics,
    .compute_forces_torques  = dd_compute_forces_torques,
    .apply_platform_effects  = dd_apply_platform_effects,

    .init_state              = dd_init_state,
    .reset_state             = dd_reset_state,
    .init_params             = dd_init_params,

    .config_size             = dd_config_size,
    .config_set_defaults     = dd_config_set_defaults,
    .config_to_params        = dd_config_to_params,
};
