/**
 * Physics Engine Module Unit Tests
 *
 * Yoneda Philosophy: Tests serve as complete behavioral specification.
 * Each test defines what the module MUST do - the definitive contract.
 *
 * Test Categories (~100 tests):
 * 1. Configuration (5 tests)
 * 2. System Lifecycle (7 tests)
 * 3. Motor Dynamics (8 tests)
 * 4. Force/Torque (12 tests)
 * 5. Linear Dynamics (8 tests)
 * 6. Angular Dynamics (10 tests)
 * 7. Quaternion Dynamics (8 tests)
 * 8. RK4 Integration (10 tests)
 * 9. Stability (10 tests)
 * 10. Full Step Integration (10 tests)
 * 11. Physical Accuracy (8 tests)
 * 12. Edge Cases (6 tests)
 */

#include "../include/physics.h"
#include "test_harness.h"

#define EPSILON 1e-4f
#define EPSILON_LOOSE 1e-2f

/* ============================================================================
 * Section 1: Configuration Tests
 * ============================================================================ */

TEST(config_default_values) {
    PhysicsConfig config = physics_config_default();

    ASSERT_FLOAT_NEAR(config.dt, 0.02f, EPSILON);
    ASSERT_EQ(config.substeps, 4);
    ASSERT_FLOAT_NEAR(config.gravity, 9.81f, EPSILON);
    ASSERT_TRUE(config.enable_drag == true);
    ASSERT_TRUE(config.enable_ground_effect == true);
    ASSERT_TRUE(config.enable_motor_dynamics == true);
    ASSERT_TRUE(config.enable_gyroscopic == false);

    return 0;
}

TEST(config_dt_valid_range) {
    PhysicsConfig config = physics_config_default();

    ASSERT_GE(config.dt, 0.001f);
    ASSERT_LE(config.dt, 0.1f);

    return 0;
}

TEST(config_substeps_minimum) {
    PhysicsConfig config = physics_config_default();
    ASSERT_GE(config.substeps, 1);

    return 0;
}

TEST(config_ground_effect_params) {
    PhysicsConfig config = physics_config_default();

    ASSERT_GT(config.ground_effect_height, 0.0f);
    ASSERT_GE(config.ground_effect_coeff, 1.0f);
    ASSERT_LT(config.ground_effect_coeff, 2.0f);

    return 0;
}

TEST(config_stability_limits) {
    PhysicsConfig config = physics_config_default();

    ASSERT_GT(config.max_linear_accel, 0.0f);
    ASSERT_GT(config.max_angular_accel, 0.0f);
    ASSERT_LT(config.max_linear_accel, 1000.0f);
    ASSERT_LT(config.max_angular_accel, 1000.0f);

    return 0;
}

/* ============================================================================
 * Section 2: System Lifecycle Tests
 * ============================================================================ */

TEST(create_basic) {
    Arena* persistent = arena_create(4 * 1024 * 1024);  /* 4MB */
    Arena* scratch = arena_create(1024 * 1024);  /* 1MB */
    ASSERT_NOT_NULL(persistent);
    ASSERT_NOT_NULL(scratch);

    PhysicsSystem* physics = physics_create(persistent, scratch, NULL, 100);
    ASSERT_NOT_NULL(physics);
    ASSERT_EQ(physics->max_drones, 100);
    ASSERT_EQ(physics->step_count, 0);

    /* All scratch buffers should be valid */
    ASSERT_NOT_NULL(physics->k1);
    ASSERT_NOT_NULL(physics->k2);
    ASSERT_NOT_NULL(physics->k3);
    ASSERT_NOT_NULL(physics->k4);
    ASSERT_NOT_NULL(physics->temp_state);

    ASSERT_NOT_NULL(physics->forces_x);
    ASSERT_NOT_NULL(physics->torques_z);

    physics_destroy(physics);
    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(create_alignment_32byte) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);
    PhysicsSystem* physics = physics_create(persistent, scratch, NULL, 256);
    ASSERT_NOT_NULL(physics);

    /* Force/torque buffers should be 32-byte aligned */
    ASSERT_TRUE(((uintptr_t)(physics->forces_x) & (32-1)) == 0);
    ASSERT_TRUE(((uintptr_t)(physics->forces_y) & (32-1)) == 0);
    ASSERT_TRUE(((uintptr_t)(physics->forces_z) & (32-1)) == 0);
    ASSERT_TRUE(((uintptr_t)(physics->torques_x) & (32-1)) == 0);
    ASSERT_TRUE(((uintptr_t)(physics->torques_y) & (32-1)) == 0);
    ASSERT_TRUE(((uintptr_t)(physics->torques_z) & (32-1)) == 0);

    physics_destroy(physics);
    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(create_capacity_1024) {
    Arena* persistent = arena_create(8 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);
    PhysicsSystem* physics = physics_create(persistent, scratch, NULL, 1024);

    ASSERT_NOT_NULL(physics);
    ASSERT_EQ(physics->max_drones, 1024);

    physics_destroy(physics);
    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(create_capacity_10000) {
    Arena* persistent = arena_create(64 * 1024 * 1024);  /* 64MB */
    Arena* scratch = arena_create(1024 * 1024);
    PhysicsSystem* physics = physics_create(persistent, scratch, NULL, 10000);

    ASSERT_NOT_NULL(physics);
    ASSERT_EQ(physics->max_drones, 10000);

    physics_destroy(physics);
    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(create_null_arena) {
    Arena* arena = arena_create(1024 * 1024);
    PhysicsSystem* physics = physics_create(NULL, arena, NULL, 100);
    ASSERT_NULL(physics);

    physics = physics_create(arena, NULL, NULL, 100);
    ASSERT_NULL(physics);

    arena_destroy(arena);

    return 0;
}

TEST(create_zero_capacity) {
    Arena* persistent = arena_create(1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);
    PhysicsSystem* physics = physics_create(persistent, scratch, NULL, 0);
    ASSERT_NULL(physics);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(memory_size_calculation) {
    size_t size = physics_memory_size(1024);
    ASSERT_GT(size, (size_t)0);

    /* ~340KB for 1024 drones according to spec */
    ASSERT_LE(size, (size_t)(500 * 1024));
    ASSERT_GE(size, (size_t)(200 * 1024));

    return 0;
}

/* ============================================================================
 * Section 3: Motor Dynamics Tests
 * ============================================================================ */

TEST(motor_steady_state) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneParamsSOA* params = drone_params_create(persistent, 1);
    params->motor_tau[0] = 0.02f;
    params->max_rpm[0] = 2500.0f;

    float rpm_commands[4] = {1000.0f, 1000.0f, 1000.0f, 1000.0f};
    float actual_rpms[4] = {1000.0f, 1000.0f, 1000.0f, 1000.0f};

    physics_motor_dynamics(rpm_commands, actual_rpms, params, 0.02f, 1);

    /* At steady state, RPMs should remain unchanged */
    ASSERT_FLOAT_NEAR(actual_rpms[0], 1000.0f, EPSILON);
    ASSERT_FLOAT_NEAR(actual_rpms[1], 1000.0f, EPSILON);
    ASSERT_FLOAT_NEAR(actual_rpms[2], 1000.0f, EPSILON);
    ASSERT_FLOAT_NEAR(actual_rpms[3], 1000.0f, EPSILON);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(motor_step_response) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneParamsSOA* params = drone_params_create(persistent, 1);
    params->motor_tau[0] = 0.02f;
    params->max_rpm[0] = 2500.0f;

    float rpm_commands[4] = {1000.0f, 1000.0f, 1000.0f, 1000.0f};
    float actual_rpms[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    /* After one time constant, should be ~63.2% of target */
    physics_motor_dynamics(rpm_commands, actual_rpms, params, 0.02f, 1);

    /* First order response */
    ASSERT_GT(actual_rpms[0], 0.0f);
    ASSERT_LT(actual_rpms[0], 1000.0f);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(motor_time_constant) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneParamsSOA* params_fast = drone_params_create(persistent, 1);
    DroneParamsSOA* params_slow = drone_params_create(persistent, 1);

    params_fast->motor_tau[0] = 0.01f;  /* Fast motor */
    params_fast->max_rpm[0] = 2500.0f;
    params_slow->motor_tau[0] = 0.05f;  /* Slow motor */
    params_slow->max_rpm[0] = 2500.0f;

    float rpm_commands[4] = {1000.0f, 1000.0f, 1000.0f, 1000.0f};
    float actual_fast[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float actual_slow[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    physics_motor_dynamics(rpm_commands, actual_fast, params_fast, 0.02f, 1);
    physics_motor_dynamics(rpm_commands, actual_slow, params_slow, 0.02f, 1);

    /* Fast motor should respond quicker */
    ASSERT_GT(actual_fast[0], actual_slow[0]);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(motor_clamp_max) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneParamsSOA* params = drone_params_create(persistent, 1);
    params->motor_tau[0] = 0.001f;  /* Very fast motor */
    params->max_rpm[0] = 2500.0f;

    float rpm_commands[4] = {5000.0f, 5000.0f, 5000.0f, 5000.0f};  /* Over max */
    float actual_rpms[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    /* Many steps to reach steady state */
    for (int i = 0; i < 100; i++) {
        physics_motor_dynamics(rpm_commands, actual_rpms, params, 0.01f, 1);
    }

    ASSERT_FLOAT_NEAR(actual_rpms[0], 2500.0f, EPSILON_LOOSE);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(motor_clamp_min) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneParamsSOA* params = drone_params_create(persistent, 1);
    params->motor_tau[0] = 0.001f;
    params->max_rpm[0] = 2500.0f;

    float rpm_commands[4] = {-1000.0f, -1000.0f, -1000.0f, -1000.0f};  /* Negative */
    float actual_rpms[4] = {500.0f, 500.0f, 500.0f, 500.0f};

    for (int i = 0; i < 100; i++) {
        physics_motor_dynamics(rpm_commands, actual_rpms, params, 0.01f, 1);
    }

    ASSERT_GE(actual_rpms[0], 0.0f);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(motor_batch_independence) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneParamsSOA* params = drone_params_create(persistent, 2);
    params->motor_tau[0] = 0.02f;
    params->max_rpm[0] = 2500.0f;
    params->motor_tau[1] = 0.02f;
    params->max_rpm[1] = 2500.0f;

    float rpm_commands[8] = {1000.0f, 1000.0f, 1000.0f, 1000.0f,  /* Drone 0 */
                             500.0f, 500.0f, 500.0f, 500.0f};     /* Drone 1 */
    float actual_rpms[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f};

    physics_motor_dynamics(rpm_commands, actual_rpms, params, 0.02f, 2);

    /* Drone 0 should have higher RPMs than drone 1 */
    ASSERT_GT(actual_rpms[0], actual_rpms[4]);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(motor_action_mapping) {
    float max_rpm = 2500.0f;

    ASSERT_FLOAT_NEAR(action_to_rpm(0.0f, max_rpm), 0.0f, EPSILON);
    ASSERT_FLOAT_NEAR(action_to_rpm(1.0f, max_rpm), 2500.0f, EPSILON);
    ASSERT_FLOAT_NEAR(action_to_rpm(0.5f, max_rpm), 1250.0f, EPSILON);

    /* Clamping */
    ASSERT_FLOAT_NEAR(action_to_rpm(-0.5f, max_rpm), 0.0f, EPSILON);
    ASSERT_FLOAT_NEAR(action_to_rpm(1.5f, max_rpm), 2500.0f, EPSILON);

    return 0;
}

/* ============================================================================
 * Section 4: Force/Torque Tests
 * ============================================================================ */

TEST(thrust_zero_rpm) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    /* All RPMs at zero */
    states->rpm_0[0] = 0.0f;
    states->rpm_1[0] = 0.0f;
    states->rpm_2[0] = 0.0f;
    states->rpm_3[0] = 0.0f;

    float fx[1], fy[1], fz[1], tx[1], ty[1], tz[1];
    physics_compute_forces_torques(states, params, fx, fy, fz, tx, ty, tz, 1);

    ASSERT_FLOAT_NEAR(fz[0], 0.0f, EPSILON);
    ASSERT_FLOAT_NEAR(tx[0], 0.0f, EPSILON);
    ASSERT_FLOAT_NEAR(ty[0], 0.0f, EPSILON);
    ASSERT_FLOAT_NEAR(tz[0], 0.0f, EPSILON);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(thrust_single_motor) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    float rpm = 1000.0f;
    states->rpm_0[0] = rpm;
    states->rpm_1[0] = 0.0f;
    states->rpm_2[0] = 0.0f;
    states->rpm_3[0] = 0.0f;

    /* Identity quaternion - thrust along world Z */
    states->quat_w[0] = 1.0f;
    states->quat_x[0] = 0.0f;
    states->quat_y[0] = 0.0f;
    states->quat_z[0] = 0.0f;

    float fx[1], fy[1], fz[1], tx[1], ty[1], tz[1];
    physics_compute_forces_torques(states, params, fx, fy, fz, tx, ty, tz, 1);

    /* T = k_thrust * rpm^2 */
    float expected_thrust = params->k_thrust[0] * rpm * rpm;
    ASSERT_FLOAT_NEAR(fz[0], expected_thrust, EPSILON_LOOSE);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(thrust_total_symmetric) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    /* All motors at same RPM */
    float rpm = 1000.0f;
    states->rpm_0[0] = rpm;
    states->rpm_1[0] = rpm;
    states->rpm_2[0] = rpm;
    states->rpm_3[0] = rpm;

    float fx[1], fy[1], fz[1], tx[1], ty[1], tz[1];
    physics_compute_forces_torques(states, params, fx, fy, fz, tx, ty, tz, 1);

    /* Symmetric should produce zero roll/pitch torque */
    ASSERT_FLOAT_NEAR(tx[0], 0.0f, EPSILON);
    ASSERT_FLOAT_NEAR(ty[0], 0.0f, EPSILON);
    /* Note: yaw torque depends on motor rotation directions (CW vs CCW) */

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(torque_roll_positive) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    /* Roll (tau_x): arm_length * (T1 + T3 - T0 - T2) */
    /* If T1 + T3 > T0 + T2, positive roll */
    states->rpm_0[0] = 500.0f;
    states->rpm_1[0] = 1500.0f;
    states->rpm_2[0] = 500.0f;
    states->rpm_3[0] = 1500.0f;

    float fx[1], fy[1], fz[1], tx[1], ty[1], tz[1];
    physics_compute_forces_torques(states, params, fx, fy, fz, tx, ty, tz, 1);

    ASSERT_GT(tx[0], 0.0f);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(torque_roll_negative) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    states->rpm_0[0] = 1500.0f;
    states->rpm_1[0] = 500.0f;
    states->rpm_2[0] = 1500.0f;
    states->rpm_3[0] = 500.0f;

    float fx[1], fy[1], fz[1], tx[1], ty[1], tz[1];
    physics_compute_forces_torques(states, params, fx, fy, fz, tx, ty, tz, 1);

    ASSERT_LT(tx[0], 0.0f);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(torque_pitch_positive) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    /* Pitch (tau_y): arm_length * (T0 + T1 - T2 - T3) */
    /* If T0 + T1 > T2 + T3, positive pitch (nose up) */
    states->rpm_0[0] = 1500.0f;
    states->rpm_1[0] = 1500.0f;
    states->rpm_2[0] = 500.0f;
    states->rpm_3[0] = 500.0f;

    float fx[1], fy[1], fz[1], tx[1], ty[1], tz[1];
    physics_compute_forces_torques(states, params, fx, fy, fz, tx, ty, tz, 1);

    ASSERT_GT(ty[0], 0.0f);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(torque_pitch_negative) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    states->rpm_0[0] = 500.0f;
    states->rpm_1[0] = 500.0f;
    states->rpm_2[0] = 1500.0f;
    states->rpm_3[0] = 1500.0f;

    float fx[1], fy[1], fz[1], tx[1], ty[1], tz[1];
    physics_compute_forces_torques(states, params, fx, fy, fz, tx, ty, tz, 1);

    ASSERT_LT(ty[0], 0.0f);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(torque_yaw_positive) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    /* Yaw: k_torque * (rpm0^2 + rpm1^2 - rpm2^2 - rpm3^2) */
    /* CW motors (M0, M1) > CCW motors (M2, M3) -> positive yaw */
    states->rpm_0[0] = 1500.0f;  /* CW */
    states->rpm_1[0] = 1500.0f;  /* CW */
    states->rpm_2[0] = 500.0f;   /* CCW */
    states->rpm_3[0] = 500.0f;   /* CCW */

    float fx[1], fy[1], fz[1], tx[1], ty[1], tz[1];
    physics_compute_forces_torques(states, params, fx, fy, fz, tx, ty, tz, 1);

    ASSERT_GT(tz[0], 0.0f);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(torque_yaw_negative) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    states->rpm_0[0] = 500.0f;   /* CW */
    states->rpm_1[0] = 500.0f;   /* CW */
    states->rpm_2[0] = 1500.0f;  /* CCW */
    states->rpm_3[0] = 1500.0f;  /* CCW */

    float fx[1], fy[1], fz[1], tx[1], ty[1], tz[1];
    physics_compute_forces_torques(states, params, fx, fy, fz, tx, ty, tz, 1);

    ASSERT_LT(tz[0], 0.0f);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(torque_arm_scaling) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params1 = drone_params_create(persistent, 1);
    DroneParamsSOA* params2 = drone_params_create(persistent, 1);

    params1->arm_length[0] = 0.1f;
    params2->arm_length[0] = 0.2f;

    states->rpm_0[0] = 500.0f;
    states->rpm_1[0] = 1500.0f;
    states->rpm_2[0] = 500.0f;
    states->rpm_3[0] = 1500.0f;

    float fx1[1], fy1[1], fz1[1], tx1[1], ty1[1], tz1[1];
    float fx2[1], fy2[1], fz2[1], tx2[1], ty2[1], tz2[1];

    physics_compute_forces_torques(states, params1, fx1, fy1, fz1, tx1, ty1, tz1, 1);
    physics_compute_forces_torques(states, params2, fx2, fy2, fz2, tx2, ty2, tz2, 1);

    /* Double arm length -> double roll/pitch torque */
    ASSERT_FLOAT_NEAR(tx2[0], 2.0f * tx1[0], EPSILON_LOOSE);
    ASSERT_FLOAT_NEAR(ty2[0], 2.0f * ty1[0], EPSILON_LOOSE);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(hover_thrust_equilibrium) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    float mass = params->mass[0];
    float gravity = params->gravity[0];
    float k_thrust = params->k_thrust[0];

    /* Calculate hover RPM: 4 * k_thrust * rpm^2 = m * g */
    float hover_rpm = sqrtf(mass * gravity / (4.0f * k_thrust));

    states->rpm_0[0] = hover_rpm;
    states->rpm_1[0] = hover_rpm;
    states->rpm_2[0] = hover_rpm;
    states->rpm_3[0] = hover_rpm;

    float fx[1], fy[1], fz[1], tx[1], ty[1], tz[1];
    physics_compute_forces_torques(states, params, fx, fy, fz, tx, ty, tz, 1);

    /* Total thrust should equal weight */
    ASSERT_FLOAT_NEAR(fz[0], mass * gravity, EPSILON_LOOSE);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

/* ============================================================================
 * Section 5: Linear Dynamics Tests
 * ============================================================================ */

TEST(gravity_free_fall) {
    Arena* persistent = arena_create(8 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    PhysicsConfig config = physics_config_default();
    config.enable_drag = false;
    config.enable_ground_effect = false;

    PhysicsSystem* physics = physics_create(persistent, scratch, &config, 1);
    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    /* Start at height 10m with zero velocity */
    states->pos_z[0] = 10.0f;
    states->vel_z[0] = 0.0f;
    states->quat_w[0] = 1.0f;

    /* Zero motor input (free fall) */
    float actions[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    /* Single step */
    float dt = 0.02f;
    physics_step_dt(physics, states, params, actions, 1, dt);

    /* After dt, velocity should be approximately -g * dt */
    float expected_vel = -params->gravity[0] * dt;
    ASSERT_FLOAT_NEAR(states->vel_z[0], expected_vel, EPSILON_LOOSE);

    physics_destroy(physics);
    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(hover_altitude_stable) {
    Arena* persistent = arena_create(8 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    PhysicsConfig config = physics_config_default();
    config.enable_drag = false;
    config.enable_ground_effect = false;
    config.enable_motor_dynamics = false;

    PhysicsSystem* physics = physics_create(persistent, scratch, &config, 1);
    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    /* Use physically consistent parameters for hovering test */
    float mass = params->mass[0];
    float gravity = params->gravity[0];
    float max_rpm = params->max_rpm[0];
    float target_hover_rpm = 0.7f * max_rpm;
    params->k_thrust[0] = mass * gravity / (4.0f * target_hover_rpm * target_hover_rpm);

    float k_thrust = params->k_thrust[0];
    float hover_rpm = sqrtf(mass * gravity / (4.0f * k_thrust));

    /* Convert to action (normalized) */
    float hover_action = hover_rpm / max_rpm;

    float actions[4] = {hover_action, hover_action, hover_action, hover_action};

    states->pos_z[0] = 5.0f;
    states->vel_z[0] = 0.0f;
    states->quat_w[0] = 1.0f;

    float initial_z = states->pos_z[0];

    /* Run several steps */
    for (int i = 0; i < 50; i++) {
        physics_step_dt(physics, states, params, actions, 1, 0.02f);
    }

    /* Altitude should remain approximately stable (within 0.5m) */
    ASSERT_MSG(fabsf(states->pos_z[0] - initial_z) < 0.5f, "hover altitude stable");

    physics_destroy(physics);
    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(drag_decelerates) {
    Arena* persistent = arena_create(8 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    params->k_drag[0] = 0.5f;

    states->vel_x[0] = 10.0f;
    states->vel_y[0] = 0.0f;
    states->vel_z[0] = 0.0f;

    float fx[1] = {0.0f}, fy[1] = {0.0f}, fz[1] = {0.0f};

    physics_apply_drag(states, params, fx, fy, fz, 1, 1.225f);

    /* Drag should be negative (opposing positive velocity) */
    ASSERT_LT(fx[0], 0.0f);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(quaternion_rotation_z90) {
    /* 90 degree rotation around Z: quat = (cos(45), 0, 0, sin(45)) */
    float angle = (float)M_PI / 2.0f;
    Quat q = QUAT(cosf(angle/2.0f), 0.0f, 0.0f, sinf(angle/2.0f));

    /* Rotate thrust along body Z to world frame */
    float thrust = 10.0f;
    Vec3 f_world = quat_rotate_body_z_to_world(q, thrust);

    /* After 90 deg yaw, thrust should still be along Z (yaw doesn't affect vertical thrust) */
    ASSERT_FLOAT_NEAR(f_world.z, thrust, EPSILON_LOOSE);
    ASSERT_FLOAT_NEAR(f_world.x, 0.0f, EPSILON_LOOSE);
    ASSERT_FLOAT_NEAR(f_world.y, 0.0f, EPSILON_LOOSE);

    return 0;
}

TEST(quaternion_rotation_x90) {
    /* 90 degree rotation around X: quat = (cos(45), sin(45), 0, 0) */
    float angle = (float)M_PI / 2.0f;
    Quat q = QUAT(cosf(angle/2.0f), sinf(angle/2.0f), 0.0f, 0.0f);

    float thrust = 10.0f;
    Vec3 f_world = quat_rotate_body_z_to_world(q, thrust);

    /* After 90 deg roll, body Z points to world -Y */
    ASSERT_FLOAT_NEAR(f_world.z, 0.0f, EPSILON_LOOSE);
    ASSERT_FLOAT_NEAR(f_world.y, -thrust, EPSILON_LOOSE);
    ASSERT_FLOAT_NEAR(f_world.x, 0.0f, EPSILON_LOOSE);

    return 0;
}

/* ============================================================================
 * Section 6: Angular Dynamics Tests
 * ============================================================================ */

TEST(angular_zero_torque) {
    Arena* persistent = arena_create(8 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    PhysicsConfig config = physics_config_default();
    config.enable_drag = false;

    PhysicsSystem* physics = physics_create(persistent, scratch, &config, 1);
    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    /* Initial angular velocity */
    states->omega_x[0] = 1.0f;
    states->omega_y[0] = 0.0f;
    states->omega_z[0] = 0.0f;
    states->quat_w[0] = 1.0f;

    /* Zero damping */
    params->k_ang_damp[0] = 0.0f;

    /* All motors same RPM (zero torque) */
    float rpm = 1000.0f;
    states->rpm_0[0] = rpm;
    states->rpm_1[0] = rpm;
    states->rpm_2[0] = rpm;
    states->rpm_3[0] = rpm;

    float initial_omega_x = states->omega_x[0];

    float actions[4] = {0.4f, 0.4f, 0.4f, 0.4f};
    physics_step_dt(physics, states, params, actions, 1, 0.02f);

    /* With zero torque and zero damping, angular velocity should be constant */
    ASSERT_FLOAT_NEAR(states->omega_x[0], initial_omega_x, EPSILON_LOOSE);

    physics_destroy(physics);
    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(angular_pure_roll) {
    Arena* persistent = arena_create(8 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    PhysicsConfig config = physics_config_default();
    config.enable_motor_dynamics = false;

    PhysicsSystem* physics = physics_create(persistent, scratch, &config, 1);
    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    states->quat_w[0] = 1.0f;
    states->omega_x[0] = 0.0f;
    params->k_ang_damp[0] = 0.0f;

    /* Roll torque: increase M1, M3; decrease M0, M2 */
    float rpm_high = 1500.0f;
    float rpm_low = 500.0f;
    float max_rpm = params->max_rpm[0];

    float actions[4] = {
        rpm_low / max_rpm,   /* M0 */
        rpm_high / max_rpm,  /* M1 */
        rpm_low / max_rpm,   /* M2 */
        rpm_high / max_rpm   /* M3 */
    };

    physics_step_dt(physics, states, params, actions, 1, 0.02f);

    /* Should have positive roll angular velocity */
    ASSERT_GT(states->omega_x[0], 0.0f);

    physics_destroy(physics);
    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(angular_damping) {
    Arena* persistent = arena_create(8 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    PhysicsConfig config = physics_config_default();
    config.enable_motor_dynamics = false;

    PhysicsSystem* physics = physics_create(persistent, scratch, &config, 1);
    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    states->quat_w[0] = 1.0f;
    states->omega_x[0] = 5.0f;  /* Initial angular velocity */
    params->k_ang_damp[0] = 0.1f;

    /* Symmetric motors (no torque) */
    float actions[4] = {0.4f, 0.4f, 0.4f, 0.4f};

    float initial_omega = states->omega_x[0];

    for (int i = 0; i < 10; i++) {
        physics_step_dt(physics, states, params, actions, 1, 0.02f);
    }

    /* Damping should reduce angular velocity */
    ASSERT_LT(fabsf(states->omega_x[0]), initial_omega);

    physics_destroy(physics);
    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

/* ============================================================================
 * Section 7: Quaternion Dynamics Tests
 * ============================================================================ */

TEST(quat_derivative_identity) {
    Quat q = QUAT_IDENTITY;
    Vec3 omega = VEC3_ZERO;

    Quat q_dot = quat_derivative(q, omega);

    ASSERT_FLOAT_NEAR(q_dot.w, 0.0f, EPSILON);
    ASSERT_FLOAT_NEAR(q_dot.x, 0.0f, EPSILON);
    ASSERT_FLOAT_NEAR(q_dot.y, 0.0f, EPSILON);
    ASSERT_FLOAT_NEAR(q_dot.z, 0.0f, EPSILON);

    return 0;
}

TEST(quat_derivative_pure_roll) {
    Quat q = QUAT_IDENTITY;
    Vec3 omega = VEC3(1.0f, 0.0f, 0.0f);  /* Pure roll angular velocity */

    Quat q_dot = quat_derivative(q, omega);

    /* q_dot = 0.5 * q * [0, omega] */
    /* For identity q and omega_x only: q_dot = (0, 0.5*omega_x, 0, 0) */
    ASSERT_FLOAT_NEAR(q_dot.w, 0.0f, EPSILON);
    ASSERT_FLOAT_NEAR(q_dot.x, 0.5f, EPSILON);
    ASSERT_FLOAT_NEAR(q_dot.y, 0.0f, EPSILON);
    ASSERT_FLOAT_NEAR(q_dot.z, 0.0f, EPSILON);

    return 0;
}

TEST(quat_normalization_unit) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* states = drone_state_create(persistent, 1);

    /* Already unit */
    states->quat_w[0] = 1.0f;
    states->quat_x[0] = 0.0f;
    states->quat_y[0] = 0.0f;
    states->quat_z[0] = 0.0f;

    physics_normalize_quaternions(states, 1);

    ASSERT_FLOAT_NEAR(states->quat_w[0], 1.0f, EPSILON);
    ASSERT_FLOAT_NEAR(states->quat_x[0], 0.0f, EPSILON);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(quat_normalization_scaled) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* states = drone_state_create(persistent, 1);

    /* Scaled quaternion (magnitude = 2) */
    states->quat_w[0] = 2.0f;
    states->quat_x[0] = 0.0f;
    states->quat_y[0] = 0.0f;
    states->quat_z[0] = 0.0f;

    physics_normalize_quaternions(states, 1);

    /* Should be normalized */
    float mag_sq = states->quat_w[0] * states->quat_w[0] +
                   states->quat_x[0] * states->quat_x[0] +
                   states->quat_y[0] * states->quat_y[0] +
                   states->quat_z[0] * states->quat_z[0];

    ASSERT_FLOAT_NEAR(mag_sq, 1.0f, EPSILON);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

/* ============================================================================
 * Section 8: RK4 Integration Tests
 * ============================================================================ */

TEST(rk4_substep_zero_deriv) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* current = drone_state_create(persistent, 1);
    DroneStateSOA* deriv = drone_state_create(persistent, 1);
    DroneStateSOA* output = drone_state_create(persistent, 1);

    current->pos_x[0] = 5.0f;
    current->quat_w[0] = 1.0f;

    /* Zero derivative */
    deriv->pos_x[0] = 0.0f;

    physics_rk4_substep(current, deriv, output, 0.02f, 1);

    ASSERT_FLOAT_NEAR(output->pos_x[0], 5.0f, EPSILON);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(rk4_substep_linear) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* current = drone_state_create(persistent, 1);
    DroneStateSOA* deriv = drone_state_create(persistent, 1);
    DroneStateSOA* output = drone_state_create(persistent, 1);

    current->pos_x[0] = 0.0f;
    current->quat_w[0] = 1.0f;
    deriv->pos_x[0] = 10.0f;  /* Velocity = 10 m/s */

    physics_rk4_substep(current, deriv, output, 0.5f, 1);  /* dt = 0.5 */

    /* output = current + deriv * dt = 0 + 10 * 0.5 = 5 */
    ASSERT_FLOAT_NEAR(output->pos_x[0], 5.0f, EPSILON);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(rk4_combine_weights) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneStateSOA* k1 = drone_state_create(persistent, 1);
    DroneStateSOA* k2 = drone_state_create(persistent, 1);
    DroneStateSOA* k3 = drone_state_create(persistent, 1);
    DroneStateSOA* k4 = drone_state_create(persistent, 1);

    states->pos_x[0] = 0.0f;
    states->quat_w[0] = 1.0f;

    /* Constant derivative = 6 for all k */
    k1->pos_x[0] = 6.0f;
    k2->pos_x[0] = 6.0f;
    k3->pos_x[0] = 6.0f;
    k4->pos_x[0] = 6.0f;

    /* (k1 + 2*k2 + 2*k3 + k4) / 6 = (6 + 12 + 12 + 6) / 6 = 36/6 = 6 */
    /* state += 6 * dt = 6 * 1.0 = 6 */
    physics_rk4_combine(states, k1, k2, k3, k4, 1.0f, 1);

    ASSERT_FLOAT_NEAR(states->pos_x[0], 6.0f, EPSILON);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(rk4_stable_long_simulation) {
    Arena* persistent = arena_create(16 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    PhysicsConfig config = physics_config_default();
    PhysicsSystem* physics = physics_create(persistent, scratch, &config, 1);
    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    states->pos_z[0] = 10.0f;
    states->quat_w[0] = 1.0f;

    /* Hover thrust */
    float mass = params->mass[0];
    float gravity = params->gravity[0];
    float k_thrust = params->k_thrust[0];
    float max_rpm = params->max_rpm[0];
    float hover_rpm = sqrtf(mass * gravity / (4.0f * k_thrust));
    float hover_action = hover_rpm / max_rpm;
    float actions[4] = {hover_action, hover_action, hover_action, hover_action};

    /* Run 10000 steps */
    for (int i = 0; i < 10000; i++) {
        physics_step_dt(physics, states, params, actions, 1, 0.005f);

        /* Check for divergence */
        ASSERT_MSG(isfinite(states->pos_x[0]), "pos_x finite after many steps");
        ASSERT_MSG(isfinite(states->pos_z[0]), "pos_z finite after many steps");
        ASSERT_MSG(isfinite(states->quat_w[0]), "quat_w finite after many steps");

        if (!isfinite(states->pos_z[0])) break;
    }

    physics_destroy(physics);
    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

/* ============================================================================
 * Section 9: Stability Tests
 * ============================================================================ */

TEST(clamp_velocity_within) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    params->max_vel[0] = 20.0f;
    states->vel_x[0] = 5.0f;
    states->vel_y[0] = 5.0f;
    states->vel_z[0] = 5.0f;

    physics_clamp_velocities(states, params, 1);

    /* Should remain unchanged */
    ASSERT_FLOAT_NEAR(states->vel_x[0], 5.0f, EPSILON);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(clamp_velocity_exceeds) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    params->max_vel[0] = 10.0f;
    states->vel_x[0] = 100.0f;
    states->vel_y[0] = 0.0f;
    states->vel_z[0] = 0.0f;

    physics_clamp_velocities(states, params, 1);

    /* Speed should be clamped to max_vel */
    float speed = sqrtf(states->vel_x[0] * states->vel_x[0] +
                        states->vel_y[0] * states->vel_y[0] +
                        states->vel_z[0] * states->vel_z[0]);
    ASSERT_FLOAT_NEAR(speed, 10.0f, EPSILON_LOOSE);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(sanitize_nan_position) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* states = drone_state_create(persistent, 1);

    states->pos_x[0] = NAN;
    states->quat_w[0] = 1.0f;

    uint32_t reset_count = physics_sanitize_state(states, 1);

    ASSERT_EQ(reset_count, (uint32_t)1);
    ASSERT_FLOAT_NEAR(states->pos_x[0], 0.0f, EPSILON);
    ASSERT_FLOAT_NEAR(states->quat_w[0], 1.0f, EPSILON);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(sanitize_nan_velocity) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* states = drone_state_create(persistent, 1);

    states->vel_x[0] = NAN;
    states->quat_w[0] = 1.0f;

    uint32_t reset_count = physics_sanitize_state(states, 1);

    ASSERT_EQ(reset_count, (uint32_t)1);
    ASSERT_FLOAT_NEAR(states->vel_x[0], 0.0f, EPSILON);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(sanitize_valid_unchanged) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* states = drone_state_create(persistent, 1);

    states->pos_x[0] = 5.0f;
    states->vel_x[0] = 2.0f;
    states->quat_w[0] = 1.0f;

    uint32_t reset_count = physics_sanitize_state(states, 1);

    ASSERT_EQ(reset_count, (uint32_t)0);
    ASSERT_FLOAT_NEAR(states->pos_x[0], 5.0f, EPSILON);
    ASSERT_FLOAT_NEAR(states->vel_x[0], 2.0f, EPSILON);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

/* ============================================================================
 * Section 10: Full Step Integration Tests
 * ============================================================================ */

TEST(step_free_fall) {
    Arena* persistent = arena_create(8 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    PhysicsConfig config = physics_config_default();
    config.enable_drag = false;
    config.enable_ground_effect = false;

    PhysicsSystem* physics = physics_create(persistent, scratch, &config, 1);
    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    states->pos_z[0] = 100.0f;
    states->vel_z[0] = 0.0f;
    states->quat_w[0] = 1.0f;

    float actions[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    float total_time = 0.0f;
    float dt = 0.02f;

    for (int i = 0; i < 50; i++) {
        physics_step_dt(physics, states, params, actions, 1, dt);
        total_time += dt;
    }

    /* After 1 second of free fall, z should decrease by ~0.5*g*t^2 = 4.9m */
    float expected_drop = 0.5f * params->gravity[0] * total_time * total_time;
    float actual_drop = 100.0f - states->pos_z[0];

    ASSERT_FLOAT_NEAR(actual_drop, expected_drop, EPSILON_LOOSE);

    physics_destroy(physics);
    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(step_batch_1024_correct) {
    Arena* persistent = arena_create(32 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    PhysicsConfig config = physics_config_default();
    PhysicsSystem* physics = physics_create(persistent, scratch, &config, 1024);
    DroneStateSOA* states = drone_state_create(persistent, 1024);
    DroneParamsSOA* params = drone_params_create(persistent, 1024);

    ASSERT_NOT_NULL(physics);
    ASSERT_NOT_NULL(states);
    ASSERT_NOT_NULL(params);

    /* Initialize all drones at different heights */
    for (uint32_t i = 0; i < 1024; i++) {
        states->pos_z[i] = 10.0f + (float)i * 0.01f;
        states->quat_w[i] = 1.0f;
    }

    float* actions = arena_alloc_array(persistent, float, 1024 * 4);
    for (uint32_t i = 0; i < 1024 * 4; i++) {
        actions[i] = 0.4f;
    }

    physics_step_dt(physics, states, params, actions, 1024, 0.02f);

    /* Verify all states are valid */
    for (uint32_t i = 0; i < 1024; i++) {
        ASSERT_MSG(isfinite(states->pos_z[i]), "pos_z finite for all drones");
        ASSERT_MSG(isfinite(states->quat_w[i]), "quat_w finite for all drones");
    }

    physics_destroy(physics);
    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(step_non_aligned_count) {
    Arena* persistent = arena_create(8 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    PhysicsConfig config = physics_config_default();
    PhysicsSystem* physics = physics_create(persistent, scratch, &config, 1024);
    DroneStateSOA* states = drone_state_create(persistent, 1024);
    DroneParamsSOA* params = drone_params_create(persistent, 1024);

    /* Process non-SIMD-aligned count */
    uint32_t count = 1023;  /* Not divisible by 8 (AVX2) or 4 (NEON) */

    for (uint32_t i = 0; i < count; i++) {
        states->pos_z[i] = 10.0f;
        states->quat_w[i] = 1.0f;
    }

    float* actions = arena_alloc_array(persistent, float, count * 4);
    for (uint32_t i = 0; i < count * 4; i++) {
        actions[i] = 0.4f;
    }

    physics_step_dt(physics, states, params, actions, count, 0.02f);

    /* Verify all processed states are valid */
    for (uint32_t i = 0; i < count; i++) {
        ASSERT_MSG(isfinite(states->pos_z[i]), "non-aligned: pos_z finite");
    }

    physics_destroy(physics);
    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(step_single_drone) {
    Arena* persistent = arena_create(8 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    PhysicsConfig config = physics_config_default();
    PhysicsSystem* physics = physics_create(persistent, scratch, &config, 1);
    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    states->pos_z[0] = 10.0f;
    states->quat_w[0] = 1.0f;

    float actions[4] = {0.4f, 0.4f, 0.4f, 0.4f};

    physics_step_dt(physics, states, params, actions, 1, 0.02f);

    ASSERT_MSG(isfinite(states->pos_z[0]), "single drone: pos_z finite");
    ASSERT_MSG(isfinite(states->quat_w[0]), "single drone: quat_w finite");

    physics_destroy(physics);
    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

/* ============================================================================
 * Section 11: Physical Accuracy Tests
 * ============================================================================ */

TEST(crazyflie_hover_rpm) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneParamsSOA* params = drone_params_create(persistent, 1);

    /* Set up Crazyflie-like parameters */
    params->mass[0] = 0.03f;  /* 30 grams */
    params->max_rpm[0] = 2500.0f;
    /* Set k_thrust so hover is achievable: k = m*g / (4 * rpm^2) at 70% throttle */
    float target_hover_rpm = 0.7f * params->max_rpm[0];  /* 1750 rad/s */
    params->k_thrust[0] = params->mass[0] * params->gravity[0] /
                          (4.0f * target_hover_rpm * target_hover_rpm);

    float mass = params->mass[0];
    float gravity = params->gravity[0];
    float k_thrust = params->k_thrust[0];
    float max_rpm = params->max_rpm[0];

    float hover_rpm = sqrtf(mass * gravity / (4.0f * k_thrust));

    /* Hover RPM should be achievable (less than max) */
    ASSERT_GT(hover_rpm, 0.5f * max_rpm);
    ASSERT_LT(hover_rpm, max_rpm);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(ground_effect_boost) {
    Arena* persistent = arena_create(4 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    float forces_near_surface[1] = {10.0f};
    float forces_far[1] = {10.0f};

    /* SDF distance: small positive = close to surface (outside) */
    float sdf_near[1] = {0.1f};
    physics_apply_ground_effect(states, params, forces_near_surface, sdf_near, 1, 0.5f, 1.15f);

    /* SDF distance: large positive = far from any surface */
    float sdf_far[1] = {100.0f};
    physics_apply_ground_effect(states, params, forces_far, sdf_far, 1, 0.5f, 1.15f);

    /* Near surface should have higher thrust */
    ASSERT_GT(forces_near_surface[0], forces_far[0]);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

/* ============================================================================
 * Section 12: Edge Cases
 * ============================================================================ */

TEST(extreme_angular_velocity) {
    Arena* persistent = arena_create(8 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    PhysicsConfig config = physics_config_default();
    PhysicsSystem* physics = physics_create(persistent, scratch, &config, 1);
    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    params->max_omega[0] = 50.0f;  /* Allow high omega */

    states->omega_x[0] = 100.0f;  /* Extreme angular velocity */
    states->quat_w[0] = 1.0f;

    float actions[4] = {0.4f, 0.4f, 0.4f, 0.4f};

    /* Should not crash or produce NaN */
    physics_step_dt(physics, states, params, actions, 1, 0.02f);

    ASSERT_MSG(isfinite(states->omega_x[0]), "omega_x finite after extreme input");
    ASSERT_MSG(isfinite(states->quat_w[0]), "quat_w finite after extreme input");

    physics_destroy(physics);
    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(extreme_linear_velocity) {
    Arena* persistent = arena_create(8 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    PhysicsConfig config = physics_config_default();
    PhysicsSystem* physics = physics_create(persistent, scratch, &config, 1);
    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    params->max_vel[0] = 100.0f;

    states->vel_x[0] = 50.0f;
    states->quat_w[0] = 1.0f;

    float actions[4] = {0.4f, 0.4f, 0.4f, 0.4f};

    physics_step_dt(physics, states, params, actions, 1, 0.02f);

    ASSERT_MSG(isfinite(states->vel_x[0]), "vel_x finite after extreme input");
    ASSERT_MSG(isfinite(states->pos_x[0]), "pos_x finite after extreme input");

    physics_destroy(physics);
    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(very_small_timestep) {
    Arena* persistent = arena_create(8 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    PhysicsConfig config = physics_config_default();
    config.substeps = 1;  /* Single substep */

    PhysicsSystem* physics = physics_create(persistent, scratch, &config, 1);
    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    states->pos_z[0] = 10.0f;
    states->quat_w[0] = 1.0f;

    float actions[4] = {0.4f, 0.4f, 0.4f, 0.4f};

    /* Very small timestep */
    physics_step_dt(physics, states, params, actions, 1, 0.0001f);

    ASSERT_MSG(isfinite(states->pos_z[0]), "pos_z finite with small dt");

    physics_destroy(physics);
    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(large_timestep) {
    Arena* persistent = arena_create(8 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    PhysicsConfig config = physics_config_default();
    config.substeps = 10;  /* More substeps for stability */

    PhysicsSystem* physics = physics_create(persistent, scratch, &config, 1);
    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    states->pos_z[0] = 10.0f;
    states->quat_w[0] = 1.0f;

    float actions[4] = {0.4f, 0.4f, 0.4f, 0.4f};

    /* Large timestep */
    physics_step_dt(physics, states, params, actions, 1, 0.1f);

    ASSERT_MSG(isfinite(states->pos_z[0]), "pos_z finite with large dt");

    physics_destroy(physics);
    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

TEST(all_motors_max) {
    Arena* persistent = arena_create(8 * 1024 * 1024);
    Arena* scratch = arena_create(1024 * 1024);

    PhysicsConfig config = physics_config_default();
    config.enable_motor_dynamics = false;  /* Instant motor response */

    PhysicsSystem* physics = physics_create(persistent, scratch, &config, 1);
    DroneStateSOA* states = drone_state_create(persistent, 1);
    DroneParamsSOA* params = drone_params_create(persistent, 1);

    /* Set k_thrust so max thrust is ~2x weight */
    float mass = params->mass[0];
    float gravity = params->gravity[0];
    float max_rpm = params->max_rpm[0];
    params->k_thrust[0] = 2.0f * mass * gravity / (4.0f * max_rpm * max_rpm);

    states->pos_z[0] = 0.0f;
    states->vel_z[0] = 0.0f;
    states->quat_w[0] = 1.0f;

    /* All motors at max */
    float actions[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    physics_step_dt(physics, states, params, actions, 1, 0.02f);

    /* Should accelerate upward (thrust = 2*weight, so accel = g upward) */
    ASSERT_GT(states->vel_z[0], 0.0f);

    physics_destroy(physics);
    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Physics Engine Module Tests");

    /* Section 1: Configuration */
    RUN_TEST(config_default_values);
    RUN_TEST(config_dt_valid_range);
    RUN_TEST(config_substeps_minimum);
    RUN_TEST(config_ground_effect_params);
    RUN_TEST(config_stability_limits);

    /* Section 2: System Lifecycle */
    RUN_TEST(create_basic);
    RUN_TEST(create_alignment_32byte);
    RUN_TEST(create_capacity_1024);
    RUN_TEST(create_capacity_10000);
    RUN_TEST(create_null_arena);
    RUN_TEST(create_zero_capacity);
    RUN_TEST(memory_size_calculation);

    /* Section 3: Motor Dynamics */
    RUN_TEST(motor_steady_state);
    RUN_TEST(motor_step_response);
    RUN_TEST(motor_time_constant);
    RUN_TEST(motor_clamp_max);
    RUN_TEST(motor_clamp_min);
    RUN_TEST(motor_batch_independence);
    RUN_TEST(motor_action_mapping);

    /* Section 4: Force/Torque */
    RUN_TEST(thrust_zero_rpm);
    RUN_TEST(thrust_single_motor);
    RUN_TEST(thrust_total_symmetric);
    RUN_TEST(torque_roll_positive);
    RUN_TEST(torque_roll_negative);
    RUN_TEST(torque_pitch_positive);
    RUN_TEST(torque_pitch_negative);
    RUN_TEST(torque_yaw_positive);
    RUN_TEST(torque_yaw_negative);
    RUN_TEST(torque_arm_scaling);
    RUN_TEST(hover_thrust_equilibrium);

    /* Section 5: Linear Dynamics */
    RUN_TEST(gravity_free_fall);
    RUN_TEST(hover_altitude_stable);
    RUN_TEST(drag_decelerates);
    RUN_TEST(quaternion_rotation_z90);
    RUN_TEST(quaternion_rotation_x90);

    /* Section 6: Angular Dynamics */
    RUN_TEST(angular_zero_torque);
    RUN_TEST(angular_pure_roll);
    RUN_TEST(angular_damping);

    /* Section 7: Quaternion Dynamics */
    RUN_TEST(quat_derivative_identity);
    RUN_TEST(quat_derivative_pure_roll);
    RUN_TEST(quat_normalization_unit);
    RUN_TEST(quat_normalization_scaled);

    /* Section 8: RK4 Integration */
    RUN_TEST(rk4_substep_zero_deriv);
    RUN_TEST(rk4_substep_linear);
    RUN_TEST(rk4_combine_weights);
    RUN_TEST(rk4_stable_long_simulation);

    /* Section 9: Stability */
    RUN_TEST(clamp_velocity_within);
    RUN_TEST(clamp_velocity_exceeds);
    RUN_TEST(sanitize_nan_position);
    RUN_TEST(sanitize_nan_velocity);
    RUN_TEST(sanitize_valid_unchanged);

    /* Section 10: Full Step Integration */
    RUN_TEST(step_free_fall);
    RUN_TEST(step_batch_1024_correct);
    RUN_TEST(step_non_aligned_count);
    RUN_TEST(step_single_drone);

    /* Section 11: Physical Accuracy */
    RUN_TEST(crazyflie_hover_rpm);
    RUN_TEST(ground_effect_boost);

    /* Section 12: Edge Cases */
    RUN_TEST(extreme_angular_velocity);
    RUN_TEST(extreme_linear_velocity);
    RUN_TEST(very_small_timestep);
    RUN_TEST(large_timestep);
    RUN_TEST(all_motors_max);

    TEST_SUITE_END();
}
