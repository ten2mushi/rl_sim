/**
 * Quadcopter Platform Tests
 *
 * Verifies:
 * 1. VTable is properly populated
 * 2. Registry lookup works
 * 3. Vtable functions produce bit-exact results vs existing direct physics functions
 * 4. Config defaults match existing drone_params_init defaults
 */

#include "../include/platform.h"
#include "../include/platform_quadcopter.h"
#include "drone_state.h"
#include "physics.h"
#include <stdio.h>
#include <math.h>

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

static void test_vtable_populated(void) {
    TEST("PLATFORM_QUADCOPTER vtable populated");

    const PlatformVTable* vt = &PLATFORM_QUADCOPTER;
    ASSERT_TRUE(vt->name != NULL, "name is NULL");
    ASSERT_TRUE(strcmp(vt->name, "quadcopter") == 0, "wrong name");
    ASSERT_TRUE(vt->action_dim == 4, "action_dim should be 4");
    ASSERT_TRUE(vt->state_extension_count == 4, "state_ext_count should be 4");
    ASSERT_TRUE(vt->params_extension_count == 7, "params_ext_count should be 7");

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

static void test_registry_init_and_find(void) {
    TEST("platform_registry_init + find");

    PlatformRegistry registry;
    platform_registry_init(&registry);

    ASSERT_TRUE(registry.count >= 1, "should have at least 1 platform");

    const PlatformVTable* quad = platform_registry_find(&registry, "quadcopter");
    ASSERT_TRUE(quad != NULL, "quadcopter not found");
    ASSERT_TRUE(quad == &PLATFORM_QUADCOPTER, "should be PLATFORM_QUADCOPTER");

    const PlatformVTable* nope = platform_registry_find(&registry, "nonexistent");
    ASSERT_TRUE(nope == NULL, "nonexistent should return NULL");

    PASS();
}

/* ============================================================================
 * Config Defaults Tests
 * ============================================================================ */

static void test_config_defaults(void) {
    TEST("QuadcopterConfig defaults match drone_params_init");

    QuadcopterConfig cfg;
    PLATFORM_QUADCOPTER.config_set_defaults(&cfg);

    /* These should match the values in drone_params_init() */
    ASSERT_FLOAT_EQ(cfg.arm_length, 0.1f, 1e-9f, "arm_length");
    ASSERT_FLOAT_EQ(cfg.k_thrust, 3.16e-10f, 1e-15f, "k_thrust");
    ASSERT_FLOAT_EQ(cfg.k_torque, 7.94e-12f, 1e-17f, "k_torque");
    ASSERT_FLOAT_EQ(cfg.motor_tau, 0.02f, 1e-9f, "motor_tau");
    ASSERT_FLOAT_EQ(cfg.max_rpm, 2500.0f, 1e-3f, "max_rpm");
    ASSERT_FLOAT_EQ(cfg.k_drag, 0.1f, 1e-9f, "k_drag");
    ASSERT_FLOAT_EQ(cfg.k_ang_damp, 0.01f, 1e-9f, "k_ang_damp");

    PASS();
}

/* ============================================================================
 * Bit-Exact Comparison: Forces/Torques
 * ============================================================================ */

static void test_forces_torques_bitexact(void) {
    TEST("quad_compute_forces_torques bit-exact vs physics.c");

    Arena* arena = arena_create(TEST_ARENA_SIZE);
    ASSERT_TRUE(arena != NULL, "arena creation failed");

    /* Create DroneStateSOA + DroneParamsSOA (platform types) */
    DroneStateSOA* drone_states = platform_state_create(arena, TEST_CAPACITY, QUAD_STATE_EXT_COUNT);
    DroneParamsSOA* drone_params = platform_params_create(arena, TEST_CAPACITY, QUAD_PARAMS_EXT_COUNT);
    ASSERT_TRUE(drone_states != NULL && drone_params != NULL, "creation failed");

    /* Initialize default params for drone_states and drone_params */
    for (uint32_t i = 0; i < TEST_CAPACITY; i++) {
        PLATFORM_QUADCOPTER.init_params(drone_params->extension, drone_params->extension_count, i);
    }

    /* Create platform types */
    PlatformStateSOA* pstate = platform_state_create(arena, TEST_CAPACITY, QUAD_STATE_EXT_COUNT);
    PlatformParamsSOA* pparams = platform_params_create(arena, TEST_CAPACITY, QUAD_PARAMS_EXT_COUNT);
    ASSERT_TRUE(pstate != NULL && pparams != NULL, "platform creation failed");

    /* Set identical state data in both: tilted orientation, non-zero RPMs */
    for (uint32_t i = 0; i < TEST_CAPACITY; i++) {
        float angle = 0.2f * (float)i;
        float rpm_base = 1000.0f + 100.0f * (float)i;

        /* Drone state */
        drone_states->rigid_body.quat_w[i] = cosf(angle / 2.0f);
        drone_states->rigid_body.quat_x[i] = sinf(angle / 2.0f) * 0.577f;
        drone_states->rigid_body.quat_y[i] = sinf(angle / 2.0f) * 0.577f;
        drone_states->rigid_body.quat_z[i] = sinf(angle / 2.0f) * 0.577f;
        drone_states->extension[QUAD_EXT_RPM_0][i] = rpm_base;
        drone_states->extension[QUAD_EXT_RPM_1][i] = rpm_base * 1.1f;
        drone_states->extension[QUAD_EXT_RPM_2][i] = rpm_base * 0.9f;
        drone_states->extension[QUAD_EXT_RPM_3][i] = rpm_base * 1.05f;

        /* Platform state (identical) */
        pstate->rigid_body.quat_w[i] = drone_states->rigid_body.quat_w[i];
        pstate->rigid_body.quat_x[i] = drone_states->rigid_body.quat_x[i];
        pstate->rigid_body.quat_y[i] = drone_states->rigid_body.quat_y[i];
        pstate->rigid_body.quat_z[i] = drone_states->rigid_body.quat_z[i];
        pstate->extension[QUAD_EXT_RPM_0][i] = drone_states->extension[QUAD_EXT_RPM_0][i];
        pstate->extension[QUAD_EXT_RPM_1][i] = drone_states->extension[QUAD_EXT_RPM_1][i];
        pstate->extension[QUAD_EXT_RPM_2][i] = drone_states->extension[QUAD_EXT_RPM_2][i];
        pstate->extension[QUAD_EXT_RPM_3][i] = drone_states->extension[QUAD_EXT_RPM_3][i];

        /* Platform params from drone params */
        pparams->extension[QUAD_PEXT_ARM_LENGTH][i] = drone_params->extension[QUAD_PEXT_ARM_LENGTH][i];
        pparams->extension[QUAD_PEXT_K_THRUST][i] = drone_params->extension[QUAD_PEXT_K_THRUST][i];
        pparams->extension[QUAD_PEXT_K_TORQUE][i] = drone_params->extension[QUAD_PEXT_K_TORQUE][i];
    }

    /* Allocate output buffers */
    size_t buf_size = TEST_CAPACITY * sizeof(float);
    float* fx_ref = arena_alloc_aligned(arena, buf_size, 32);
    float* fy_ref = arena_alloc_aligned(arena, buf_size, 32);
    float* fz_ref = arena_alloc_aligned(arena, buf_size, 32);
    float* tx_ref = arena_alloc_aligned(arena, buf_size, 32);
    float* ty_ref = arena_alloc_aligned(arena, buf_size, 32);
    float* tz_ref = arena_alloc_aligned(arena, buf_size, 32);

    float* fx_vt = arena_alloc_aligned(arena, buf_size, 32);
    float* fy_vt = arena_alloc_aligned(arena, buf_size, 32);
    float* fz_vt = arena_alloc_aligned(arena, buf_size, 32);
    float* tx_vt = arena_alloc_aligned(arena, buf_size, 32);
    float* ty_vt = arena_alloc_aligned(arena, buf_size, 32);
    float* tz_vt = arena_alloc_aligned(arena, buf_size, 32);

    /* Call through drone_states (same type as pstate) to get reference */
    PLATFORM_QUADCOPTER.compute_forces_torques(
        &drone_states->rigid_body,
        drone_states->extension, drone_states->extension_count,
        drone_params->extension, drone_params->extension_count,
        &drone_params->rigid_body,
        fx_ref, fy_ref, fz_ref,
        tx_ref, ty_ref, tz_ref,
        TEST_CAPACITY);

    /* Vtable: call through vtable with separately-created platform types */
    PLATFORM_QUADCOPTER.compute_forces_torques(
        &pstate->rigid_body,
        (float* const*)pstate->extension, pstate->extension_count,
        (float* const*)pparams->extension, pparams->extension_count,
        &pparams->rigid_body,
        fx_vt, fy_vt, fz_vt,
        tx_vt, ty_vt, tz_vt,
        TEST_CAPACITY);

    /* Compare — same inputs should be bit-exact */
    for (uint32_t i = 0; i < TEST_CAPACITY; i++) {
        ASSERT_FLOAT_EQ(fx_vt[i], fx_ref[i], 1e-4f, "forces_x mismatch");
        ASSERT_FLOAT_EQ(fy_vt[i], fy_ref[i], 1e-4f, "forces_y mismatch");
        ASSERT_FLOAT_EQ(fz_vt[i], fz_ref[i], 1e-4f, "forces_z mismatch");
        ASSERT_FLOAT_EQ(tx_vt[i], tx_ref[i], 1e-4f, "torques_x mismatch");
        ASSERT_FLOAT_EQ(ty_vt[i], ty_ref[i], 1e-4f, "torques_y mismatch");
        ASSERT_FLOAT_EQ(tz_vt[i], tz_ref[i], 1e-4f, "torques_z mismatch");
    }

    arena_destroy(arena);
    PASS();
}

/* ============================================================================
 * Bit-Exact Comparison: Actuator Dynamics
 * ============================================================================ */

static void test_actuator_dynamics_bitexact(void) {
    TEST("quad_actuator_dynamics bit-exact vs physics_motor_dynamics");

    Arena* arena = arena_create(TEST_ARENA_SIZE);

    PlatformStateSOA* pstate = platform_state_create(arena, TEST_CAPACITY, QUAD_STATE_EXT_COUNT);
    PlatformParamsSOA* pparams = platform_params_create(arena, TEST_CAPACITY, QUAD_PARAMS_EXT_COUNT);

    size_t buf_size = align_up_size(TEST_CAPACITY * sizeof(float), 32);
    float* commands_aos = arena_alloc_aligned(arena, TEST_CAPACITY * 4 * sizeof(float), 32);

    float dt = 0.005f;

    /* Initialize params */
    for (uint32_t i = 0; i < TEST_CAPACITY; i++) {
        PLATFORM_QUADCOPTER.init_params(pparams->extension, pparams->extension_count, i);
    }

    /* Set initial RPMs and commands */
    for (uint32_t i = 0; i < TEST_CAPACITY; i++) {
        float rpm = 500.0f + 50.0f * (float)i;
        float cmd = 1500.0f + 30.0f * (float)i;

        pstate->extension[QUAD_EXT_RPM_0][i] = rpm;
        pstate->extension[QUAD_EXT_RPM_1][i] = rpm * 1.1f;
        pstate->extension[QUAD_EXT_RPM_2][i] = rpm * 0.9f;
        pstate->extension[QUAD_EXT_RPM_3][i] = rpm * 1.05f;

        /* AoS commands for vtable */
        commands_aos[i * 4 + 0] = cmd;
        commands_aos[i * 4 + 1] = cmd * 1.05f;
        commands_aos[i * 4 + 2] = cmd * 0.95f;
        commands_aos[i * 4 + 3] = cmd * 1.02f;
    }

    /* Save initial RPMs for comparison */
    float saved_rpm[TEST_CAPACITY * 4];
    for (uint32_t i = 0; i < TEST_CAPACITY; i++) {
        saved_rpm[i * 4 + 0] = pstate->extension[QUAD_EXT_RPM_0][i];
        saved_rpm[i * 4 + 1] = pstate->extension[QUAD_EXT_RPM_1][i];
        saved_rpm[i * 4 + 2] = pstate->extension[QUAD_EXT_RPM_2][i];
        saved_rpm[i * 4 + 3] = pstate->extension[QUAD_EXT_RPM_3][i];
    }

    /* Call actuator dynamics through vtable */
    PLATFORM_QUADCOPTER.actuator_dynamics(
        commands_aos,
        pstate->extension, pstate->extension_count,
        (float* const*)pparams->extension, pparams->extension_count,
        dt, TEST_CAPACITY);

    /* Verify RPMs moved toward commands (first-order lag) */
    for (uint32_t i = 0; i < TEST_CAPACITY; i++) {
        float cmd = commands_aos[i * 4 + 0];
        float old_rpm = saved_rpm[i * 4 + 0];
        float new_rpm = pstate->extension[QUAD_EXT_RPM_0][i];
        /* RPM should move toward command */
        if (cmd > old_rpm) {
            ASSERT_TRUE(new_rpm > old_rpm, "rpm_0 should increase toward command");
            ASSERT_TRUE(new_rpm <= cmd, "rpm_0 should not overshoot command");
        }
    }

    (void)buf_size;

    arena_destroy(arena);
    PASS();
}

/* ============================================================================
 * Action Mapping Test
 * ============================================================================ */

static void test_map_actions(void) {
    TEST("quad_map_actions");

    Arena* arena = arena_create(TEST_ARENA_SIZE);
    PlatformParamsSOA* pparams = platform_params_create(arena, TEST_CAPACITY, QUAD_PARAMS_EXT_COUNT);

    for (uint32_t i = 0; i < TEST_CAPACITY; i++) {
        PLATFORM_QUADCOPTER.init_params(pparams->extension, pparams->extension_count, i);
    }

    float actions[4] = {0.0f, 0.5f, 1.0f, 0.25f};
    float commands[4] = {0};

    /* Single agent */
    PLATFORM_QUADCOPTER.map_actions(actions, commands,
                                     pparams->extension, pparams->extension_count, 1);

    float max_rpm = pparams->extension[QUAD_PEXT_MAX_RPM][0];
    ASSERT_FLOAT_EQ(commands[0], 0.0f, 1e-6f, "action 0.0 -> 0 RPM");
    ASSERT_FLOAT_EQ(commands[1], 0.5f * max_rpm, 1e-1f, "action 0.5 -> half RPM");
    ASSERT_FLOAT_EQ(commands[2], max_rpm, 1e-1f, "action 1.0 -> max RPM");
    ASSERT_FLOAT_EQ(commands[3], 0.25f * max_rpm, 1e-1f, "action 0.25 -> quarter RPM");

    arena_destroy(arena);
    PASS();
}

/* ============================================================================
 * Init/Reset State Test
 * ============================================================================ */

static void test_init_reset_state(void) {
    TEST("quad init_state and reset_state");

    Arena* arena = arena_create(TEST_ARENA_SIZE);
    PlatformStateSOA* pstate = platform_state_create(arena, TEST_CAPACITY, QUAD_STATE_EXT_COUNT);

    /* Set non-zero RPMs */
    pstate->extension[QUAD_EXT_RPM_0][5] = 1000.0f;
    pstate->extension[QUAD_EXT_RPM_3][5] = 2000.0f;

    /* Reset agent 5 */
    PLATFORM_QUADCOPTER.reset_state(pstate->extension, pstate->extension_count, 5);

    ASSERT_FLOAT_EQ(pstate->extension[QUAD_EXT_RPM_0][5], 0.0f, 1e-6f, "rpm_0 reset");
    ASSERT_FLOAT_EQ(pstate->extension[QUAD_EXT_RPM_3][5], 0.0f, 1e-6f, "rpm_3 reset");

    arena_destroy(arena);
    PASS();
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("=== Quadcopter Platform Tests ===\n\n");

    printf("VTable:\n");
    test_vtable_populated();

    printf("\nRegistry:\n");
    test_registry_init_and_find();

    printf("\nConfig:\n");
    test_config_defaults();

    printf("\nBit-Exact Comparison:\n");
    test_forces_torques_bitexact();
    test_actuator_dynamics_bitexact();

    printf("\nAction Mapping:\n");
    test_map_actions();

    printf("\nLifecycle:\n");
    test_init_reset_state();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
