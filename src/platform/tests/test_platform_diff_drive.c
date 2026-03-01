/**
 * Diff-Drive Platform Tests
 *
 * Verifies:
 * 1. VTable is properly populated with correct dimensions
 * 2. Registry lookup finds diff_drive
 * 3. Config defaults match expected TurtleBot3 values
 * 4. Straight-line motion: equal wheel velocities -> forward, no yaw
 * 5. Spin-in-place: opposite wheel velocities -> pure rotation
 * 6. Circle turn: differential velocities -> curved path
 * 7. Ground constraint: z=0, no roll/pitch after platform effects
 * 8. Init/reset zeroes wheel velocities
 */

#include "../include/platform.h"
#include "../include/platform_diff_drive.h"
#include "drone_state.h"
#include "physics.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

#define TEST_CAPACITY 16
#define TEST_ARENA_SIZE (4 * 1024 * 1024)

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  [%d] %-55s ", tests_run, name); \
    fflush(stdout); \
} while(0)

#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); } while(0)

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { FAIL(msg); return; } \
} while(0)

#define ASSERT_FLOAT_EQ(a, b, eps, msg) do { \
    if (fabsf((a) - (b)) > (eps)) { \
        printf("FAIL: %s (got %.10e, expected %.10e, diff=%.10e)\n", \
               msg, (double)(a), (double)(b), (double)fabsf((a)-(b))); \
        return; \
    } \
} while(0)

/* ============================================================================
 * VTable Population Tests
 * ============================================================================ */

static void test_vtable_fields(void) {
    TEST("PLATFORM_DIFF_DRIVE vtable fields");

    const PlatformVTable* vt = &PLATFORM_DIFF_DRIVE;
    ASSERT_TRUE(vt->name != NULL, "name is NULL");
    ASSERT_TRUE(strcmp(vt->name, "diff_drive") == 0, "wrong name");
    ASSERT_TRUE(vt->action_dim == 2, "action_dim should be 2");
    ASSERT_TRUE(vt->state_extension_count == DD_STATE_EXT_COUNT, "state_ext_count should be 2");
    ASSERT_TRUE(vt->params_extension_count == DD_PARAMS_EXT_COUNT, "params_ext_count should be 3");

    ASSERT_TRUE(vt->map_actions != NULL, "map_actions is NULL");
    ASSERT_TRUE(vt->actuator_dynamics != NULL, "actuator_dynamics is NULL");
    ASSERT_TRUE(vt->compute_forces_torques != NULL, "compute_forces_torques is NULL");
    ASSERT_TRUE(vt->apply_platform_effects != NULL, "apply_platform_effects is NULL");
    ASSERT_TRUE(vt->init_state != NULL, "init_state is NULL");
    ASSERT_TRUE(vt->reset_state != NULL, "reset_state is NULL");
    ASSERT_TRUE(vt->init_params != NULL, "init_params is NULL");
    ASSERT_TRUE(vt->config_size != NULL, "config_size is NULL");
    ASSERT_TRUE(vt->config_set_defaults != NULL, "config_set_defaults is NULL");
    ASSERT_TRUE(vt->config_to_params != NULL, "config_to_params is NULL");

    PASS();
}

/* ============================================================================
 * Registry Tests
 * ============================================================================ */

static void test_registry_find(void) {
    TEST("platform_registry finds diff_drive");

    PlatformRegistry registry;
    platform_registry_init(&registry);

    ASSERT_TRUE(registry.count >= 2, "should have at least 2 platforms");

    const PlatformVTable* dd = platform_registry_find(&registry, "diff_drive");
    ASSERT_TRUE(dd != NULL, "diff_drive not found");
    ASSERT_TRUE(dd == &PLATFORM_DIFF_DRIVE, "should be PLATFORM_DIFF_DRIVE");

    /* Quadcopter should still be there */
    const PlatformVTable* quad = platform_registry_find(&registry, "quadcopter");
    ASSERT_TRUE(quad != NULL, "quadcopter not found");

    PASS();
}

/* ============================================================================
 * Config Defaults Tests
 * ============================================================================ */

static void test_config_defaults(void) {
    TEST("DiffDriveConfig defaults");

    DiffDriveConfig cfg;
    PLATFORM_DIFF_DRIVE.config_set_defaults(&cfg);

    ASSERT_FLOAT_EQ(cfg.wheel_radius, 0.033f, 1e-6f, "wheel_radius");
    ASSERT_FLOAT_EQ(cfg.axle_length, 0.16f, 1e-6f, "axle_length");
    ASSERT_FLOAT_EQ(cfg.max_wheel_vel, 6.67f, 1e-3f, "max_wheel_vel");

    ASSERT_TRUE(PLATFORM_DIFF_DRIVE.config_size() == sizeof(DiffDriveConfig),
                "config_size mismatch");

    PASS();
}

/* ============================================================================
 * Init/Reset State Tests
 * ============================================================================ */

static void test_init_reset_state(void) {
    TEST("dd init_state and reset_state zero wheel vels");

    Arena* arena = arena_create(TEST_ARENA_SIZE);
    ASSERT_TRUE(arena != NULL, "arena creation failed");

    PlatformStateSOA* pstate = platform_state_create(arena, TEST_CAPACITY, DD_STATE_EXT_COUNT);
    ASSERT_TRUE(pstate != NULL, "state creation failed");

    /* Set non-zero wheel velocities */
    pstate->extension[DD_EXT_WHEEL_VEL_L][3] = 5.0f;
    pstate->extension[DD_EXT_WHEEL_VEL_R][3] = -3.0f;

    /* Reset agent 3 */
    PLATFORM_DIFF_DRIVE.reset_state(pstate->extension, pstate->extension_count, 3);

    ASSERT_FLOAT_EQ(pstate->extension[DD_EXT_WHEEL_VEL_L][3], 0.0f, 1e-9f, "wl reset");
    ASSERT_FLOAT_EQ(pstate->extension[DD_EXT_WHEEL_VEL_R][3], 0.0f, 1e-9f, "wr reset");

    /* Init on a different agent */
    pstate->extension[DD_EXT_WHEEL_VEL_L][7] = 10.0f;
    PLATFORM_DIFF_DRIVE.init_state(pstate->extension, pstate->extension_count, 7);
    ASSERT_FLOAT_EQ(pstate->extension[DD_EXT_WHEEL_VEL_L][7], 0.0f, 1e-9f, "wl init");

    arena_destroy(arena);
    PASS();
}

/* ============================================================================
 * Action Mapping Tests
 * ============================================================================ */

static void test_map_actions(void) {
    TEST("dd_map_actions [-1,1] -> [-max_wheel_vel, +max_wheel_vel]");

    Arena* arena = arena_create(TEST_ARENA_SIZE);
    PlatformParamsSOA* pparams = platform_params_create(arena, TEST_CAPACITY, DD_PARAMS_EXT_COUNT);
    ASSERT_TRUE(pparams != NULL, "params creation failed");

    PLATFORM_DIFF_DRIVE.init_params(pparams->extension, pparams->extension_count, 0);

    float max_wv = pparams->extension[DD_PEXT_MAX_WHEEL_VEL][0];

    /* Test actions: full forward left, zero right */
    float actions[2] = {1.0f, 0.0f};
    float commands[2] = {0};

    PLATFORM_DIFF_DRIVE.map_actions(actions, commands,
                                     pparams->extension, pparams->extension_count, 1);

    ASSERT_FLOAT_EQ(commands[0], max_wv, 1e-3f, "action 1.0 -> max_wheel_vel");
    ASSERT_FLOAT_EQ(commands[1], 0.0f, 1e-6f, "action 0.0 -> 0");

    /* Test negative action (reverse) */
    float actions2[2] = {-1.0f, -0.5f};
    PLATFORM_DIFF_DRIVE.map_actions(actions2, commands,
                                     pparams->extension, pparams->extension_count, 1);

    ASSERT_FLOAT_EQ(commands[0], -max_wv, 1e-3f, "action -1.0 -> -max_wheel_vel");
    ASSERT_FLOAT_EQ(commands[1], -0.5f * max_wv, 1e-3f, "action -0.5 -> -half");

    /* Test clamping */
    float actions3[2] = {2.0f, -3.0f};
    PLATFORM_DIFF_DRIVE.map_actions(actions3, commands,
                                     pparams->extension, pparams->extension_count, 1);

    ASSERT_FLOAT_EQ(commands[0], max_wv, 1e-3f, "action 2.0 clamped to max");
    ASSERT_FLOAT_EQ(commands[1], -max_wv, 1e-3f, "action -3.0 clamped to -max");

    arena_destroy(arena);
    PASS();
}

/* ============================================================================
 * Ground Constraint Test
 * ============================================================================ */

static void test_ground_constraint(void) {
    TEST("dd_apply_platform_effects enforces z=0 ground");

    Arena* arena = arena_create(TEST_ARENA_SIZE);
    PlatformStateSOA* pstate = platform_state_create(arena, TEST_CAPACITY, DD_STATE_EXT_COUNT);
    PlatformParamsSOA* pparams = platform_params_create(arena, TEST_CAPACITY, DD_PARAMS_EXT_COUNT);
    ASSERT_TRUE(pstate != NULL && pparams != NULL, "creation failed");

    /* Initialize */
    for (uint32_t i = 0; i < 1; i++) {
        PLATFORM_DIFF_DRIVE.init_state(pstate->extension, pstate->extension_count, i);
        PLATFORM_DIFF_DRIVE.init_params(pparams->extension, pparams->extension_count, i);
        rigid_body_params_init(&pparams->rigid_body, i);
    }

    /* Set some non-ground state: z offset, tilted orientation, omega_x/y */
    pstate->rigid_body.pos_z[0] = 5.0f;
    pstate->rigid_body.vel_z[0] = -2.0f;
    pstate->rigid_body.omega_x[0] = 1.5f;
    pstate->rigid_body.omega_y[0] = -0.8f;

    /* Set a tilted quaternion (30-degree roll) */
    float roll = 0.5236f; /* ~30 degrees */
    pstate->rigid_body.quat_w[0] = cosf(roll / 2.0f);
    pstate->rigid_body.quat_x[0] = sinf(roll / 2.0f);
    pstate->rigid_body.quat_y[0] = 0.0f;
    pstate->rigid_body.quat_z[0] = 0.0f;

    /* Dummy force/torque buffers (not used by dd platform effects) */
    size_t bsz = TEST_CAPACITY * sizeof(float);
    float* fx = arena_alloc_aligned(arena, bsz, 32);
    float* fy = arena_alloc_aligned(arena, bsz, 32);
    float* fz = arena_alloc_aligned(arena, bsz, 32);
    memset(fx, 0, bsz);
    memset(fy, 0, bsz);
    memset(fz, 0, bsz);

    PhysicsConfig phys_cfg = physics_config_default();

    PLATFORM_DIFF_DRIVE.apply_platform_effects(
        &pstate->rigid_body,
        pstate->extension, pstate->extension_count,
        &pparams->rigid_body,
        (float* const*)pparams->extension, pparams->extension_count,
        fx, fy, fz,
        NULL, &phys_cfg, 1);

    /* Verify ground constraints */
    ASSERT_FLOAT_EQ(pstate->rigid_body.pos_z[0], 0.0f, 1e-9f, "pos_z should be 0");
    ASSERT_FLOAT_EQ(pstate->rigid_body.vel_z[0], 0.0f, 1e-9f, "vel_z should be 0");
    ASSERT_FLOAT_EQ(pstate->rigid_body.omega_x[0], 0.0f, 1e-9f, "omega_x should be 0");
    ASSERT_FLOAT_EQ(pstate->rigid_body.omega_y[0], 0.0f, 1e-9f, "omega_y should be 0");

    /* Verify yaw-only quaternion: qx=0, qy=0 */
    ASSERT_FLOAT_EQ(pstate->rigid_body.quat_x[0], 0.0f, 1e-6f, "quat_x should be 0");
    ASSERT_FLOAT_EQ(pstate->rigid_body.quat_y[0], 0.0f, 1e-6f, "quat_y should be 0");

    /* Quaternion should be normalized */
    float qn = pstate->rigid_body.quat_w[0] * pstate->rigid_body.quat_w[0] +
               pstate->rigid_body.quat_z[0] * pstate->rigid_body.quat_z[0];
    ASSERT_FLOAT_EQ(qn, 1.0f, 1e-5f, "quaternion should be unit length");

    arena_destroy(arena);
    PASS();
}

/* ============================================================================
 * Straight Line Motion Test
 * ============================================================================ */

static void test_straight_line(void) {
    TEST("equal wheel vels -> forward motion, no yaw change");

    Arena* arena = arena_create(TEST_ARENA_SIZE);
    Arena* scratch = arena_create(TEST_ARENA_SIZE);
    ASSERT_TRUE(arena != NULL && scratch != NULL, "arena creation failed");

    uint32_t count = 1;

    /* Create physics system with diff-drive vtable */
    PhysicsConfig phys_cfg = physics_config_default();
    phys_cfg.dt = 0.02f;
    phys_cfg.substeps = 4;
    phys_cfg.enable_drag = false;
    phys_cfg.enable_ground_effect = false;
    phys_cfg.enable_motor_dynamics = true;

    PhysicsSystem* physics = physics_create(arena, scratch, &phys_cfg, count, &PLATFORM_DIFF_DRIVE);
    ASSERT_TRUE(physics != NULL, "physics creation failed");

    /* Create state and params */
    PlatformStateSOA* states = platform_state_create(arena, count, DD_STATE_EXT_COUNT);
    PlatformParamsSOA* params = platform_params_create(arena, count, DD_PARAMS_EXT_COUNT);
    ASSERT_TRUE(states != NULL && params != NULL, "state/params creation failed");

    /* Initialize */
    rigid_body_params_init(&params->rigid_body, 0);
    params->rigid_body.mass[0] = 1.0f;
    params->rigid_body.ixx[0] = 0.001f;
    params->rigid_body.iyy[0] = 0.001f;
    params->rigid_body.izz[0] = 0.002f;
    params->rigid_body.gravity[0] = 0.0f; /* No gravity for ground robot */
    PLATFORM_DIFF_DRIVE.init_params(params->extension, params->extension_count, 0);
    PLATFORM_DIFF_DRIVE.init_state(states->extension, states->extension_count, 0);

    /* Identity quaternion: facing +X */
    states->rigid_body.quat_w[0] = 1.0f;
    states->rigid_body.quat_x[0] = 0.0f;
    states->rigid_body.quat_y[0] = 0.0f;
    states->rigid_body.quat_z[0] = 0.0f;
    states->rigid_body.pos_x[0] = 0.0f;
    states->rigid_body.pos_y[0] = 0.0f;
    states->rigid_body.pos_z[0] = 0.0f;

    /* Actions: both wheels forward at 50% */
    float actions[2] = {0.5f, 0.5f};

    /* Step physics for 100 iterations */
    for (int step = 0; step < 100; step++) {
        physics_step(physics, states, params, actions, count);
    }

    float pos_x = states->rigid_body.pos_x[0];
    float pos_y = states->rigid_body.pos_y[0];
    float pos_z = states->rigid_body.pos_z[0];

    /* Extract yaw from quaternion */
    float qw = states->rigid_body.quat_w[0];
    float qz = states->rigid_body.quat_z[0];
    float yaw = 2.0f * atan2f(qz, qw);

    /* Robot should have moved forward (in +X) */
    ASSERT_TRUE(pos_x > 0.01f, "pos_x should be positive (forward motion)");

    /* Y should be near zero (straight line) */
    ASSERT_FLOAT_EQ(pos_y, 0.0f, 0.05f, "pos_y should be ~0 for straight line");

    /* Z should be zero (ground constraint) */
    ASSERT_FLOAT_EQ(pos_z, 0.0f, 1e-6f, "pos_z should be 0");

    /* Yaw should be near zero (no turning) */
    ASSERT_FLOAT_EQ(yaw, 0.0f, 0.05f, "yaw should be ~0 for equal wheel vels");

    arena_destroy(scratch);
    arena_destroy(arena);
    PASS();
}

/* ============================================================================
 * Spin In Place Test
 * ============================================================================ */

static void test_spin_in_place(void) {
    TEST("opposite wheel vels -> pure rotation, ~no translation");

    Arena* arena = arena_create(TEST_ARENA_SIZE);
    Arena* scratch = arena_create(TEST_ARENA_SIZE);
    ASSERT_TRUE(arena != NULL && scratch != NULL, "arena creation failed");

    uint32_t count = 1;

    PhysicsConfig phys_cfg = physics_config_default();
    phys_cfg.dt = 0.02f;
    phys_cfg.substeps = 4;
    phys_cfg.enable_drag = false;
    phys_cfg.enable_ground_effect = false;
    phys_cfg.enable_motor_dynamics = true;

    PhysicsSystem* physics = physics_create(arena, scratch, &phys_cfg, count, &PLATFORM_DIFF_DRIVE);
    ASSERT_TRUE(physics != NULL, "physics creation failed");

    PlatformStateSOA* states = platform_state_create(arena, count, DD_STATE_EXT_COUNT);
    PlatformParamsSOA* params = platform_params_create(arena, count, DD_PARAMS_EXT_COUNT);
    ASSERT_TRUE(states != NULL && params != NULL, "state/params creation failed");

    rigid_body_params_init(&params->rigid_body, 0);
    params->rigid_body.mass[0] = 1.0f;
    params->rigid_body.ixx[0] = 0.001f;
    params->rigid_body.iyy[0] = 0.001f;
    params->rigid_body.izz[0] = 0.002f;
    params->rigid_body.gravity[0] = 0.0f;
    PLATFORM_DIFF_DRIVE.init_params(params->extension, params->extension_count, 0);
    PLATFORM_DIFF_DRIVE.init_state(states->extension, states->extension_count, 0);

    /* Identity quaternion */
    states->rigid_body.quat_w[0] = 1.0f;
    states->rigid_body.quat_x[0] = 0.0f;
    states->rigid_body.quat_y[0] = 0.0f;
    states->rigid_body.quat_z[0] = 0.0f;

    /* Opposite wheel velocities: left=-0.5, right=+0.5 -> spin CCW */
    float actions[2] = {-0.5f, 0.5f};

    for (int step = 0; step < 100; step++) {
        physics_step(physics, states, params, actions, count);
    }

    float pos_x = states->rigid_body.pos_x[0];
    float pos_y = states->rigid_body.pos_y[0];

    /* Position should stay near origin */
    float dist_from_origin = sqrtf(pos_x * pos_x + pos_y * pos_y);
    ASSERT_TRUE(dist_from_origin < 0.1f, "should stay near origin for spin-in-place");

    /* Yaw should have changed significantly */
    float qw = states->rigid_body.quat_w[0];
    float qz = states->rigid_body.quat_z[0];
    float yaw = 2.0f * atan2f(qz, qw);
    ASSERT_TRUE(fabsf(yaw) > 0.1f, "yaw should change for opposite wheel vels");

    arena_destroy(scratch);
    arena_destroy(arena);
    PASS();
}

/* ============================================================================
 * Circle Turn Test
 * ============================================================================ */

static void test_circle_turn(void) {
    TEST("differential wheel vels -> curved path, yaw changes");

    Arena* arena = arena_create(TEST_ARENA_SIZE);
    Arena* scratch = arena_create(TEST_ARENA_SIZE);
    ASSERT_TRUE(arena != NULL && scratch != NULL, "arena creation failed");

    uint32_t count = 1;

    PhysicsConfig phys_cfg = physics_config_default();
    phys_cfg.dt = 0.02f;
    phys_cfg.substeps = 4;
    phys_cfg.enable_drag = false;
    phys_cfg.enable_ground_effect = false;
    phys_cfg.enable_motor_dynamics = true;

    PhysicsSystem* physics = physics_create(arena, scratch, &phys_cfg, count, &PLATFORM_DIFF_DRIVE);
    ASSERT_TRUE(physics != NULL, "physics creation failed");

    PlatformStateSOA* states = platform_state_create(arena, count, DD_STATE_EXT_COUNT);
    PlatformParamsSOA* params = platform_params_create(arena, count, DD_PARAMS_EXT_COUNT);
    ASSERT_TRUE(states != NULL && params != NULL, "state/params creation failed");

    rigid_body_params_init(&params->rigid_body, 0);
    params->rigid_body.mass[0] = 1.0f;
    params->rigid_body.ixx[0] = 0.001f;
    params->rigid_body.iyy[0] = 0.001f;
    params->rigid_body.izz[0] = 0.002f;
    params->rigid_body.gravity[0] = 0.0f;
    PLATFORM_DIFF_DRIVE.init_params(params->extension, params->extension_count, 0);
    PLATFORM_DIFF_DRIVE.init_state(states->extension, states->extension_count, 0);

    states->rigid_body.quat_w[0] = 1.0f;
    states->rigid_body.quat_x[0] = 0.0f;
    states->rigid_body.quat_y[0] = 0.0f;
    states->rigid_body.quat_z[0] = 0.0f;

    /* Differential: left=0.3, right=0.7 -> curves left */
    float actions[2] = {0.3f, 0.7f};

    float prev_yaw = 0.0f;
    int yaw_increased = 0;

    for (int step = 0; step < 100; step++) {
        physics_step(physics, states, params, actions, count);

        float qw = states->rigid_body.quat_w[0];
        float qz = states->rigid_body.quat_z[0];
        float yaw = 2.0f * atan2f(qz, qw);

        if (step > 10 && yaw > prev_yaw + 0.001f) {
            yaw_increased++;
        }
        prev_yaw = yaw;
    }

    /* Should have translated (not pure rotation) */
    float pos_x = states->rigid_body.pos_x[0];
    float pos_y = states->rigid_body.pos_y[0];
    float dist = sqrtf(pos_x * pos_x + pos_y * pos_y);
    ASSERT_TRUE(dist > 0.01f, "should have translated during circle turn");

    /* Yaw should have changed consistently (right > left -> positive yaw = CCW) */
    ASSERT_TRUE(yaw_increased > 20, "yaw should increase consistently for circle turn");

    arena_destroy(scratch);
    arena_destroy(arena);
    PASS();
}

/* ============================================================================
 * Actuator Dynamics Test
 * ============================================================================ */

static void test_actuator_dynamics(void) {
    TEST("dd_actuator_dynamics first-order lag");

    Arena* arena = arena_create(TEST_ARENA_SIZE);
    PlatformStateSOA* pstate = platform_state_create(arena, TEST_CAPACITY, DD_STATE_EXT_COUNT);
    PlatformParamsSOA* pparams = platform_params_create(arena, TEST_CAPACITY, DD_PARAMS_EXT_COUNT);
    ASSERT_TRUE(pstate != NULL && pparams != NULL, "creation failed");

    PLATFORM_DIFF_DRIVE.init_state(pstate->extension, pstate->extension_count, 0);
    PLATFORM_DIFF_DRIVE.init_params(pparams->extension, pparams->extension_count, 0);

    /* Wheels at rest, command to full speed */
    float commands[2] = {5.0f, -3.0f};

    PLATFORM_DIFF_DRIVE.actuator_dynamics(
        commands,
        pstate->extension, pstate->extension_count,
        (float* const*)pparams->extension, pparams->extension_count,
        0.005f, 1);

    float wl = pstate->extension[DD_EXT_WHEEL_VEL_L][0];
    float wr = pstate->extension[DD_EXT_WHEEL_VEL_R][0];

    /* Wheel vels should move toward commands but not reach them */
    ASSERT_TRUE(wl > 0.0f, "wl should move toward positive command");
    ASSERT_TRUE(wl < 5.0f, "wl should not overshoot command");
    ASSERT_TRUE(wr < 0.0f, "wr should move toward negative command");
    ASSERT_TRUE(wr > -3.0f, "wr should not overshoot command");

    /* After many steps, should converge to commands */
    for (int i = 0; i < 500; i++) {
        PLATFORM_DIFF_DRIVE.actuator_dynamics(
            commands,
            pstate->extension, pstate->extension_count,
            (float* const*)pparams->extension, pparams->extension_count,
            0.005f, 1);
    }

    wl = pstate->extension[DD_EXT_WHEEL_VEL_L][0];
    wr = pstate->extension[DD_EXT_WHEEL_VEL_R][0];

    ASSERT_FLOAT_EQ(wl, 5.0f, 0.01f, "wl should converge to command");
    ASSERT_FLOAT_EQ(wr, -3.0f, 0.01f, "wr should converge to command");

    arena_destroy(arena);
    PASS();
}

/* ============================================================================
 * Config to Params Test
 * ============================================================================ */

static void test_config_to_params(void) {
    TEST("dd_config_to_params populates extension arrays");

    Arena* arena = arena_create(TEST_ARENA_SIZE);
    PlatformParamsSOA* pparams = platform_params_create(arena, TEST_CAPACITY, DD_PARAMS_EXT_COUNT);
    ASSERT_TRUE(pparams != NULL, "params creation failed");

    DiffDriveConfig cfg;
    cfg.wheel_radius = 0.05f;
    cfg.axle_length = 0.20f;
    cfg.max_wheel_vel = 10.0f;

    PLATFORM_DIFF_DRIVE.config_to_params(&cfg,
        pparams->extension, pparams->extension_count,
        &pparams->rigid_body, 0);

    ASSERT_FLOAT_EQ(pparams->extension[DD_PEXT_WHEEL_RADIUS][0], 0.05f, 1e-9f, "wheel_radius");
    ASSERT_FLOAT_EQ(pparams->extension[DD_PEXT_AXLE_LENGTH][0], 0.20f, 1e-9f, "axle_length");
    ASSERT_FLOAT_EQ(pparams->extension[DD_PEXT_MAX_WHEEL_VEL][0], 10.0f, 1e-9f, "max_wheel_vel");

    arena_destroy(arena);
    PASS();
}

/* ============================================================================
 * Forces/Torques Consistency Test
 * ============================================================================ */

static void test_forces_torques(void) {
    TEST("dd_compute_forces_torques produces expected forces");

    Arena* arena = arena_create(TEST_ARENA_SIZE);

    PlatformStateSOA* pstate = platform_state_create(arena, TEST_CAPACITY, DD_STATE_EXT_COUNT);
    PlatformParamsSOA* pparams = platform_params_create(arena, TEST_CAPACITY, DD_PARAMS_EXT_COUNT);
    ASSERT_TRUE(pstate != NULL && pparams != NULL, "creation failed");

    PLATFORM_DIFF_DRIVE.init_state(pstate->extension, pstate->extension_count, 0);
    PLATFORM_DIFF_DRIVE.init_params(pparams->extension, pparams->extension_count, 0);
    rigid_body_params_init(&pparams->rigid_body, 0);
    pparams->rigid_body.mass[0] = 1.0f;
    pparams->rigid_body.izz[0] = 0.002f;

    /* Identity quaternion, zero velocity, both wheels forward */
    pstate->rigid_body.quat_w[0] = 1.0f;
    pstate->rigid_body.quat_x[0] = 0.0f;
    pstate->rigid_body.quat_y[0] = 0.0f;
    pstate->rigid_body.quat_z[0] = 0.0f;
    pstate->rigid_body.vel_x[0] = 0.0f;
    pstate->rigid_body.vel_y[0] = 0.0f;
    pstate->rigid_body.vel_z[0] = 0.0f;
    pstate->rigid_body.omega_z[0] = 0.0f;

    /* Set wheel velocities for forward motion */
    float wvel = 3.0f; /* rad/s */
    pstate->extension[DD_EXT_WHEEL_VEL_L][0] = wvel;
    pstate->extension[DD_EXT_WHEEL_VEL_R][0] = wvel;

    size_t bsz = TEST_CAPACITY * sizeof(float);
    float* fx = arena_alloc_aligned(arena, bsz, 32);
    float* fy = arena_alloc_aligned(arena, bsz, 32);
    float* fz = arena_alloc_aligned(arena, bsz, 32);
    float* tx = arena_alloc_aligned(arena, bsz, 32);
    float* ty = arena_alloc_aligned(arena, bsz, 32);
    float* tz = arena_alloc_aligned(arena, bsz, 32);
    memset(fx, 0, bsz);
    memset(fy, 0, bsz);
    memset(fz, 0, bsz);
    memset(tx, 0, bsz);
    memset(ty, 0, bsz);
    memset(tz, 0, bsz);

    PLATFORM_DIFF_DRIVE.compute_forces_torques(
        &pstate->rigid_body,
        (float* const*)pstate->extension, pstate->extension_count,
        (float* const*)pparams->extension, pparams->extension_count,
        &pparams->rigid_body,
        fx, fy, fz, tx, ty, tz, 1);

    /* Equal wheel velocities, identity quat -> force in +X, no torque */
    float R = pparams->extension[DD_PEXT_WHEEL_RADIUS][0];
    float v_fwd = wvel * R; /* = 3.0 * 0.033 = 0.099 m/s */
    float k_spring = 50.0f;
    float expected_fx = k_spring * v_fwd * 1.0f; /* mass=1 */

    ASSERT_FLOAT_EQ(fx[0], expected_fx, 0.01f, "force_x for forward motion");
    ASSERT_FLOAT_EQ(fy[0], 0.0f, 0.001f, "force_y should be ~0 for straight");
    ASSERT_FLOAT_EQ(fz[0], 0.0f, 1e-9f, "force_z should be 0");
    ASSERT_FLOAT_EQ(tz[0], 0.0f, 0.001f, "torque_z should be ~0 for equal wheels");

    arena_destroy(arena);
    PASS();
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("=== Diff-Drive Platform Tests ===\n\n");

    printf("VTable:\n");
    test_vtable_fields();

    printf("\nRegistry:\n");
    test_registry_find();

    printf("\nConfig:\n");
    test_config_defaults();
    test_config_to_params();

    printf("\nLifecycle:\n");
    test_init_reset_state();

    printf("\nActuator Dynamics:\n");
    test_actuator_dynamics();

    printf("\nAction Mapping:\n");
    test_map_actions();

    printf("\nForces/Torques:\n");
    test_forces_torques();

    printf("\nGround Constraint:\n");
    test_ground_constraint();

    printf("\nIntegration (with physics engine):\n");
    test_straight_line();
    test_spin_in_place();
    test_circle_turn();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
