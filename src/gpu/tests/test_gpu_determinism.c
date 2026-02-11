/**
 * GPU Determinism Tests
 *
 * Verifies that GPU sensor output is bit-exact deterministic across:
 * 1. Repeated runs with the same seed (reset + re-run)
 * 2. Multiple engine_reset calls
 * 3. Sustained multi-step trajectories
 *
 * The engine's RNG is re-seeded from config.seed on every engine_reset(),
 * so identical seeds must produce identical spawn positions, physics
 * trajectories, and therefore identical sensor observations.
 *
 * GPU sensors (camera depth, etc.) must produce the same raymarching
 * results given the same drone poses and world state.
 */

#include "environment_manager.h"
#include "gpu_hal.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "test_harness.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * Constants
 * ============================================================================ */

#define TEST_NUM_ENVS         8
#define TEST_DRONES_PER_ENV   8
#define TEST_TOTAL_DRONES     (TEST_NUM_ENVS * TEST_DRONES_PER_ENV)
#define TEST_NUM_STEPS        10
#define TEST_SEED             42

/* ============================================================================
 * Helper: Create a deterministic GPU test engine
 * ============================================================================ */

static BatchDroneEngine* create_deterministic_engine(uint64_t seed,
                                                      SensorConfig* sensors,
                                                      uint32_t num_sensors) {
    EngineConfig cfg = engine_config_default();
    cfg.num_envs = TEST_NUM_ENVS;
    cfg.drones_per_env = TEST_DRONES_PER_ENV;
    cfg.seed = seed;
    cfg.persistent_arena_size = 256 * 1024 * 1024;
    cfg.frame_arena_size = 64 * 1024 * 1024;
    cfg.sensor_configs = sensors;
    cfg.num_sensor_configs = num_sensors;

    /* Fixed physics parameters to avoid any non-determinism from defaults changing */
    cfg.timestep = 0.02f;
    cfg.physics_substeps = 4;
    cfg.gravity = 9.81f;

    char error[ENGINE_ERROR_MSG_SIZE];
    BatchDroneEngine* engine = engine_create(&cfg, error);
    if (engine == NULL) {
        printf("\n    engine_create failed: %s", error);
    }
    return engine;
}

/* Helper: Build the standard sensor config for determinism tests */
static uint32_t build_test_sensors(SensorConfig* sensors) {
    /* Sensor 0: IMU (CPU sensor) */
    sensors[0] = sensor_config_imu();

    /* Sensor 1: Camera depth 16x16 (GPU sensor) */
    sensors[1] = sensor_config_default(SENSOR_TYPE_CAMERA_DEPTH);
    sensors[1].camera.width = 16;
    sensors[1].camera.height = 16;
    sensors[1].camera.fov_horizontal = (float)(M_PI / 2.0);
    sensors[1].camera.fov_vertical = (float)(M_PI / 2.0);
    sensors[1].camera.far_clip = 50.0f;

    return 2;
}

/* Helper: Add deterministic world geometry */
static void add_test_geometry(WorldBrickMap* world) {
    /* Ground plane box */
    Vec3 ground_center = {0.0f, 0.0f, -2.0f};
    Vec3 ground_half = {20.0f, 20.0f, 1.0f};
    world_set_box(world, ground_center, ground_half, 1);

    /* Obstacle sphere */
    Vec3 sphere_center = {5.0f, 3.0f, 4.0f};
    world_set_sphere(world, sphere_center, 2.0f, 2);
}

/* Helper: Capture full observation buffer into a newly allocated snapshot */
static float* capture_observations(BatchDroneEngine* engine, size_t* out_size) {
    uint32_t obs_dim = engine_get_obs_dim(engine);
    uint32_t total = engine->config.total_drones;
    size_t buf_size = (size_t)total * obs_dim * sizeof(float);

    float* snapshot = (float*)malloc(buf_size);
    if (!snapshot) return NULL;

    float* obs = engine_get_observations(engine);
    memcpy(snapshot, obs, buf_size);

    if (out_size) *out_size = buf_size;
    return snapshot;
}

/* Helper: Run N steps with zero actions, return observation snapshot after last step */
static float* run_steps_and_capture(BatchDroneEngine* engine, int num_steps, size_t* out_size) {
    float* actions = engine_get_actions(engine);
    uint32_t total = engine->config.total_drones;

    /* Zero actions (hover) for deterministic behavior */
    memset(actions, 0, (size_t)total * ENGINE_ACTION_DIM * sizeof(float));

    for (int i = 0; i < num_steps; i++) {
        engine_step(engine);
    }

    return capture_observations(engine, out_size);
}

/* Helper: Check for NaN/Inf in float array */
static int count_nan_inf(const float* data, size_t count) {
    int bad = 0;
    for (size_t i = 0; i < count; i++) {
        if (isnan(data[i]) || isinf(data[i])) bad++;
    }
    return bad;
}

/* Helper: Find first difference between two buffers, returns byte offset or -1 */
static long find_first_diff(const void* a, const void* b, size_t size) {
    const uint8_t* pa = (const uint8_t*)a;
    const uint8_t* pb = (const uint8_t*)b;
    for (size_t i = 0; i < size; i++) {
        if (pa[i] != pb[i]) return (long)i;
    }
    return -1;
}

/* ============================================================================
 * Test 1: Basic determinism -- reset and re-run produces identical observations
 *
 * Steps:
 *   1. Create engine, reset, run N steps, snapshot observations
 *   2. Reset same engine (re-seeds RNG), run N steps, snapshot again
 *   3. Compare byte-for-byte
 * ============================================================================ */

TEST(reset_determinism_basic) {
    printf("\n    Checking basic reset determinism (seed=%d, %d drones, %d steps)...",
           TEST_SEED, TEST_TOTAL_DRONES, TEST_NUM_STEPS);

    SensorConfig sensors[2];
    uint32_t ns = build_test_sensors(sensors);

    BatchDroneEngine* engine = create_deterministic_engine(TEST_SEED, sensors, ns);
    ASSERT_MSG(engine != NULL, "engine creation");

    add_test_geometry(engine->world);

    /* --- Run 1 --- */
    printf("\n    Run 1: reset + %d steps...", TEST_NUM_STEPS);
    engine_reset(engine);

    size_t obs_size = 0;
    float* snapshot1 = run_steps_and_capture(engine, TEST_NUM_STEPS, &obs_size);
    ASSERT_MSG(snapshot1 != NULL, "snapshot1 allocation");
    ASSERT_MSG(obs_size > 0, "observation size must be > 0");

    /* Sanity check: no NaN/Inf */
    size_t num_floats = obs_size / sizeof(float);
    int bad = count_nan_inf(snapshot1, num_floats);
    ASSERT_MSG(bad == 0, "no NaN/Inf in run 1 observations");

    /* --- Run 2 (same engine, reset re-seeds RNG) --- */
    printf("\n    Run 2: reset + %d steps...", TEST_NUM_STEPS);
    engine_reset(engine);

    size_t obs_size2 = 0;
    float* snapshot2 = run_steps_and_capture(engine, TEST_NUM_STEPS, &obs_size2);
    ASSERT_MSG(snapshot2 != NULL, "snapshot2 allocation");
    ASSERT_MSG((int)obs_size == (int)obs_size2, "observation buffer sizes must match");

    /* --- Compare --- */
    printf("\n    Comparing %zu bytes of observations...", obs_size);
    int cmp = memcmp(snapshot1, snapshot2, obs_size);
    if (cmp != 0) {
        long diff_offset = find_first_diff(snapshot1, snapshot2, obs_size);
        long float_idx = diff_offset / (long)sizeof(float);
        printf("\n    MISMATCH at byte offset %ld (float index %ld): "
               "run1=%.8e, run2=%.8e",
               diff_offset, float_idx,
               snapshot1[float_idx], snapshot2[float_idx]);
    }
    ASSERT_MSG(cmp == 0, "observations must be bit-exact identical after reset");

    free(snapshot1);
    free(snapshot2);
    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 2: Per-step determinism -- observations match at every step, not just final
 *
 * Captures observations after every single step and compares.
 * ============================================================================ */

TEST(per_step_determinism) {
    printf("\n    Checking per-step determinism...");

    SensorConfig sensors[2];
    uint32_t ns = build_test_sensors(sensors);

    BatchDroneEngine* engine = create_deterministic_engine(TEST_SEED, sensors, ns);
    ASSERT_MSG(engine != NULL, "engine creation");

    add_test_geometry(engine->world);

    uint32_t obs_dim = 0;
    uint32_t total = 0;
    size_t step_obs_size = 0;

    /* --- Run 1: capture observations at each step --- */
    engine_reset(engine);

    obs_dim = engine_get_obs_dim(engine);
    total = engine->config.total_drones;
    step_obs_size = (size_t)total * obs_dim * sizeof(float);

    float* snapshots1 = (float*)malloc(step_obs_size * TEST_NUM_STEPS);
    ASSERT_MSG(snapshots1 != NULL, "snapshots1 allocation");

    float* actions = engine_get_actions(engine);
    memset(actions, 0, (size_t)total * ENGINE_ACTION_DIM * sizeof(float));

    for (int step = 0; step < TEST_NUM_STEPS; step++) {
        engine_step(engine);
        float* obs = engine_get_observations(engine);
        memcpy(snapshots1 + step * (size_t)total * obs_dim, obs, step_obs_size);
    }

    /* --- Run 2: capture observations at each step --- */
    engine_reset(engine);

    float* snapshots2 = (float*)malloc(step_obs_size * TEST_NUM_STEPS);
    ASSERT_MSG(snapshots2 != NULL, "snapshots2 allocation");

    actions = engine_get_actions(engine);
    memset(actions, 0, (size_t)total * ENGINE_ACTION_DIM * sizeof(float));

    for (int step = 0; step < TEST_NUM_STEPS; step++) {
        engine_step(engine);
        float* obs = engine_get_observations(engine);
        memcpy(snapshots2 + step * (size_t)total * obs_dim, obs, step_obs_size);
    }

    /* --- Compare step-by-step --- */
    int mismatches = 0;
    for (int step = 0; step < TEST_NUM_STEPS; step++) {
        float* s1 = snapshots1 + step * (size_t)total * obs_dim;
        float* s2 = snapshots2 + step * (size_t)total * obs_dim;
        if (memcmp(s1, s2, step_obs_size) != 0) {
            long diff_offset = find_first_diff(s1, s2, step_obs_size);
            long float_idx = diff_offset / (long)sizeof(float);
            printf("\n    Step %d MISMATCH at float index %ld: %.8e vs %.8e",
                   step + 1, float_idx, s1[float_idx], s2[float_idx]);
            mismatches++;
        }
    }

    printf("\n    %d/%d steps matched bit-exactly", TEST_NUM_STEPS - mismatches, TEST_NUM_STEPS);
    ASSERT_MSG(mismatches == 0, "all steps must match bit-exactly");

    free(snapshots1);
    free(snapshots2);
    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 3: Determinism across multiple engine_reset calls
 *
 * Resets the engine 5 times, each time running N steps. All 5 runs
 * should produce identical final observations.
 * ============================================================================ */

TEST(multi_reset_determinism) {
    printf("\n    Checking determinism across 5 consecutive resets...");

    SensorConfig sensors[2];
    uint32_t ns = build_test_sensors(sensors);

    BatchDroneEngine* engine = create_deterministic_engine(TEST_SEED, sensors, ns);
    ASSERT_MSG(engine != NULL, "engine creation");

    add_test_geometry(engine->world);

    int num_resets = 5;
    size_t obs_size = 0;

    /* Capture reference from first run */
    engine_reset(engine);
    float* reference = run_steps_and_capture(engine, TEST_NUM_STEPS, &obs_size);
    ASSERT_MSG(reference != NULL, "reference allocation");

    /* Run multiple more times and compare */
    int mismatches = 0;
    for (int r = 1; r < num_resets; r++) {
        engine_reset(engine);
        size_t sz = 0;
        float* snapshot = run_steps_and_capture(engine, TEST_NUM_STEPS, &sz);
        ASSERT_MSG(snapshot != NULL, "snapshot allocation");
        ASSERT_MSG((int)sz == (int)obs_size, "buffer sizes must match across resets");

        if (memcmp(reference, snapshot, obs_size) != 0) {
            long diff_offset = find_first_diff(reference, snapshot, obs_size);
            long float_idx = diff_offset / (long)sizeof(float);
            printf("\n    Reset %d MISMATCH at float index %ld: ref=%.8e, got=%.8e",
                   r + 1, float_idx, reference[float_idx], snapshot[float_idx]);
            mismatches++;
        }
        free(snapshot);
    }

    printf("\n    %d/%d resets produced identical results", num_resets - mismatches, num_resets);
    ASSERT_MSG(mismatches == 0, "all resets must produce identical observations");

    free(reference);
    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 4: Two separate engines with same seed produce identical output
 *
 * Creates two independent engines with the same config and seed.
 * Both should produce bit-exact identical observations.
 * ============================================================================ */

TEST(separate_engines_same_seed) {
    printf("\n    Checking determinism across two separate engine instances...");

    SensorConfig sensors[2];
    uint32_t ns = build_test_sensors(sensors);

    /* Engine A */
    BatchDroneEngine* engine_a = create_deterministic_engine(TEST_SEED, sensors, ns);
    ASSERT_MSG(engine_a != NULL, "engine_a creation");
    add_test_geometry(engine_a->world);

    /* Engine B */
    BatchDroneEngine* engine_b = create_deterministic_engine(TEST_SEED, sensors, ns);
    ASSERT_MSG(engine_b != NULL, "engine_b creation");
    add_test_geometry(engine_b->world);

    /* Run A */
    engine_reset(engine_a);
    size_t obs_size_a = 0;
    float* snapshot_a = run_steps_and_capture(engine_a, TEST_NUM_STEPS, &obs_size_a);
    ASSERT_MSG(snapshot_a != NULL, "snapshot_a allocation");

    /* Run B */
    engine_reset(engine_b);
    size_t obs_size_b = 0;
    float* snapshot_b = run_steps_and_capture(engine_b, TEST_NUM_STEPS, &obs_size_b);
    ASSERT_MSG(snapshot_b != NULL, "snapshot_b allocation");

    ASSERT_MSG((int)obs_size_a == (int)obs_size_b, "observation sizes must match");

    printf("\n    Comparing %zu bytes from two engines...", obs_size_a);
    int cmp = memcmp(snapshot_a, snapshot_b, obs_size_a);
    if (cmp != 0) {
        long diff_offset = find_first_diff(snapshot_a, snapshot_b, obs_size_a);
        long float_idx = diff_offset / (long)sizeof(float);
        printf("\n    MISMATCH at float index %ld: A=%.8e, B=%.8e",
               float_idx, snapshot_a[float_idx], snapshot_b[float_idx]);
    }
    ASSERT_MSG(cmp == 0, "separate engines with same seed must match");

    free(snapshot_a);
    free(snapshot_b);
    engine_destroy(engine_a);
    engine_destroy(engine_b);
    return 0;
}

/* ============================================================================
 * Test 5: Different seeds produce different observations
 *
 * Sanity check: if we change the seed, observations must differ.
 * (Catches trivially-constant outputs that would appear "deterministic".)
 * ============================================================================ */

TEST(different_seeds_differ) {
    printf("\n    Checking that different seeds produce different observations...");

    SensorConfig sensors[2];
    uint32_t ns = build_test_sensors(sensors);

    /* Engine with seed 42 */
    BatchDroneEngine* engine_a = create_deterministic_engine(42, sensors, ns);
    ASSERT_MSG(engine_a != NULL, "engine_a creation");
    add_test_geometry(engine_a->world);

    /* Engine with seed 99 */
    BatchDroneEngine* engine_b = create_deterministic_engine(99, sensors, ns);
    ASSERT_MSG(engine_b != NULL, "engine_b creation");
    add_test_geometry(engine_b->world);

    engine_reset(engine_a);
    size_t obs_size_a = 0;
    float* snapshot_a = run_steps_and_capture(engine_a, TEST_NUM_STEPS, &obs_size_a);
    ASSERT_MSG(snapshot_a != NULL, "snapshot_a allocation");

    engine_reset(engine_b);
    size_t obs_size_b = 0;
    float* snapshot_b = run_steps_and_capture(engine_b, TEST_NUM_STEPS, &obs_size_b);
    ASSERT_MSG(snapshot_b != NULL, "snapshot_b allocation");

    ASSERT_MSG((int)obs_size_a == (int)obs_size_b, "observation sizes must match");

    int cmp = memcmp(snapshot_a, snapshot_b, obs_size_a);
    ASSERT_MSG(cmp != 0, "different seeds must produce different observations");
    printf(" (confirmed different)");

    free(snapshot_a);
    free(snapshot_b);
    engine_destroy(engine_a);
    engine_destroy(engine_b);
    return 0;
}

/* ============================================================================
 * Test 6: GPU vs CPU fallback determinism consistency
 *
 * If GPU is available, the GPU path should produce the same results
 * on repeated runs. This test runs 3 resets and checks all produce
 * identical depth camera outputs (the GPU-accelerated sensor).
 * ============================================================================ */

TEST(gpu_depth_sensor_consistency) {
    printf("\n    Checking GPU depth sensor consistency across 3 resets...");

    SensorConfig sensors[1];
    sensors[0] = sensor_config_default(SENSOR_TYPE_CAMERA_DEPTH);
    sensors[0].camera.width = 32;
    sensors[0].camera.height = 32;
    sensors[0].camera.fov_horizontal = (float)(M_PI / 2.0);
    sensors[0].camera.fov_vertical = (float)(M_PI / 2.0);
    sensors[0].camera.far_clip = 50.0f;

    BatchDroneEngine* engine = create_deterministic_engine(TEST_SEED, sensors, 1);
    ASSERT_MSG(engine != NULL, "engine creation");

    add_test_geometry(engine->world);

    if (gpu_is_available() && engine->gpu_sensor_ctx != NULL) {
        printf(" (GPU active)");
    } else {
        printf(" (CPU fallback)");
    }

    /* Reference run */
    engine_reset(engine);
    size_t obs_size = 0;
    float* reference = run_steps_and_capture(engine, TEST_NUM_STEPS, &obs_size);
    ASSERT_MSG(reference != NULL, "reference allocation");

    /* Verify non-trivial output */
    size_t num_floats = obs_size / sizeof(float);
    int non_zero = 0;
    for (size_t i = 0; i < num_floats; i++) {
        if (reference[i] != 0.0f) non_zero++;
    }
    printf("\n    Depth observations: %d/%zu non-zero values", non_zero, num_floats);
    ASSERT_MSG(non_zero > 0, "depth observations should have non-zero values");

    /* Comparison runs */
    int mismatches = 0;
    for (int r = 0; r < 2; r++) {
        engine_reset(engine);
        size_t sz = 0;
        float* snapshot = run_steps_and_capture(engine, TEST_NUM_STEPS, &sz);
        ASSERT_MSG(snapshot != NULL, "snapshot allocation");

        if (memcmp(reference, snapshot, obs_size) != 0) {
            long diff_offset = find_first_diff(reference, snapshot, obs_size);
            long float_idx = diff_offset / (long)sizeof(float);
            printf("\n    Reset %d MISMATCH at float %ld: ref=%.8e, got=%.8e",
                   r + 2, float_idx, reference[float_idx], snapshot[float_idx]);
            mismatches++;
        }
        free(snapshot);
    }

    ASSERT_MSG(mismatches == 0, "all depth sensor runs must be identical");

    free(reference);
    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 7: Extended run determinism (50 steps)
 *
 * Longer runs can expose drift or accumulated non-determinism
 * in physics integration or GPU floating point.
 * ============================================================================ */

TEST(extended_run_determinism) {
    int extended_steps = 50;
    printf("\n    Checking determinism over %d steps...", extended_steps);

    SensorConfig sensors[2];
    uint32_t ns = build_test_sensors(sensors);

    BatchDroneEngine* engine = create_deterministic_engine(TEST_SEED, sensors, ns);
    ASSERT_MSG(engine != NULL, "engine creation");

    add_test_geometry(engine->world);

    /* Run 1 */
    engine_reset(engine);
    size_t obs_size = 0;
    float* snapshot1 = run_steps_and_capture(engine, extended_steps, &obs_size);
    ASSERT_MSG(snapshot1 != NULL, "snapshot1 allocation");

    /* Run 2 */
    engine_reset(engine);
    size_t obs_size2 = 0;
    float* snapshot2 = run_steps_and_capture(engine, extended_steps, &obs_size2);
    ASSERT_MSG(snapshot2 != NULL, "snapshot2 allocation");

    ASSERT_MSG((int)obs_size == (int)obs_size2, "observation sizes must match");

    int cmp = memcmp(snapshot1, snapshot2, obs_size);
    if (cmp != 0) {
        long diff_offset = find_first_diff(snapshot1, snapshot2, obs_size);
        long float_idx = diff_offset / (long)sizeof(float);
        printf("\n    MISMATCH at float index %ld after %d steps: %.8e vs %.8e",
               float_idx, extended_steps, snapshot1[float_idx], snapshot2[float_idx]);
    }
    ASSERT_MSG(cmp == 0, "extended run must be deterministic");

    /* Verify observations are not all-zero (meaningful simulation) */
    size_t num_floats = obs_size / sizeof(float);
    int bad = count_nan_inf(snapshot1, num_floats);
    ASSERT_MSG(bad == 0, "no NaN/Inf in extended observations");

    free(snapshot1);
    free(snapshot2);
    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 8: Mixed GPU+CPU sensor determinism
 *
 * Tests with a mix of GPU sensors (camera depth) and CPU sensors
 * (IMU, position, velocity) to ensure the async GPU pipeline
 * does not introduce non-determinism in observation assembly.
 * ============================================================================ */

TEST(mixed_sensor_determinism) {
    printf("\n    Checking mixed GPU+CPU sensor determinism...");

    SensorConfig sensors[4];
    /* CPU sensors */
    sensors[0] = sensor_config_imu();
    sensors[1] = sensor_config_position();
    sensors[2] = sensor_config_velocity();

    /* GPU sensor */
    sensors[3] = sensor_config_default(SENSOR_TYPE_CAMERA_DEPTH);
    sensors[3].camera.width = 16;
    sensors[3].camera.height = 16;
    sensors[3].camera.fov_horizontal = (float)(M_PI / 2.0);
    sensors[3].camera.fov_vertical = (float)(M_PI / 2.0);
    sensors[3].camera.far_clip = 50.0f;

    BatchDroneEngine* engine = create_deterministic_engine(TEST_SEED, sensors, 4);
    ASSERT_MSG(engine != NULL, "engine creation");

    add_test_geometry(engine->world);

    uint32_t obs_dim = engine_get_obs_dim(engine);
    printf("\n    obs_dim = %u (mixed CPU+GPU)", obs_dim);

    /* Run 1 */
    engine_reset(engine);
    size_t obs_size = 0;
    float* snapshot1 = run_steps_and_capture(engine, TEST_NUM_STEPS, &obs_size);
    ASSERT_MSG(snapshot1 != NULL, "snapshot1 allocation");

    /* Run 2 */
    engine_reset(engine);
    size_t obs_size2 = 0;
    float* snapshot2 = run_steps_and_capture(engine, TEST_NUM_STEPS, &obs_size2);
    ASSERT_MSG(snapshot2 != NULL, "snapshot2 allocation");

    ASSERT_MSG((int)obs_size == (int)obs_size2, "observation sizes must match");

    int cmp = memcmp(snapshot1, snapshot2, obs_size);
    if (cmp != 0) {
        long diff_offset = find_first_diff(snapshot1, snapshot2, obs_size);
        long float_idx = diff_offset / (long)sizeof(float);
        uint32_t drone_idx = (uint32_t)(float_idx / obs_dim);
        uint32_t obs_offset = (uint32_t)(float_idx % obs_dim);
        printf("\n    MISMATCH at drone %u, obs offset %u: %.8e vs %.8e",
               drone_idx, obs_offset, snapshot1[float_idx], snapshot2[float_idx]);
    }
    ASSERT_MSG(cmp == 0, "mixed sensor observations must be deterministic");

    free(snapshot1);
    free(snapshot2);
    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("  GPU available: %s\n", gpu_is_available() ? "YES" : "NO");
    if (gpu_is_available()) {
        GpuDevice* dev = gpu_device_create();
        if (dev) {
            printf("  Device: %s\n", gpu_device_name(dev));
            gpu_device_destroy(dev);
        }
    }
    printf("  Test config: %d drones (%dx%d), %d steps, seed=%d\n",
           TEST_TOTAL_DRONES, TEST_NUM_ENVS, TEST_DRONES_PER_ENV,
           TEST_NUM_STEPS, TEST_SEED);

    TEST_SUITE_BEGIN("GPU Determinism Tests");

    RUN_TEST(reset_determinism_basic);
    RUN_TEST(per_step_determinism);
    RUN_TEST(multi_reset_determinism);
    RUN_TEST(separate_engines_same_seed);
    RUN_TEST(different_seeds_differ);
    RUN_TEST(gpu_depth_sensor_consistency);
    RUN_TEST(extended_run_determinism);
    RUN_TEST(mixed_sensor_determinism);

    TEST_SUITE_END();
}
