/**
 * GPU Pipeline Integration Tests (Phase 4)
 *
 * Tests full engine_step() integration with GPU-accelerated sensors,
 * async dispatch, CPU/GPU overlap, and observation correctness.
 *
 * Creates engines with GPU-capable sensors (camera depth) plus CPU-only
 * sensors (IMU, position) and verifies the async pipeline works end-to-end.
 */

#include "environment_manager.h"
#include "gpu_hal.h"
#include "world_brick_map.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "test_harness.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * Helper: Create engine with GPU-capable sensor config
 * ============================================================================ */

static BatchEngine* create_gpu_test_engine(uint32_t num_envs,
                                                  uint32_t agents_per_env,
                                                  SensorConfig* sensors,
                                                  uint32_t num_sensors) {
    EngineConfig cfg = engine_config_default();
    cfg.num_envs = num_envs;
    cfg.agents_per_env = agents_per_env;
    cfg.persistent_arena_size = 256 * 1024 * 1024;
    cfg.frame_arena_size = 64 * 1024 * 1024;
    cfg.sensor_configs = sensors;
    cfg.num_sensor_configs = num_sensors;

    char error[ENGINE_ERROR_MSG_SIZE];
    BatchEngine* engine = engine_create(&cfg, error);
    if (engine == NULL) {
        printf("\n    engine_create failed: %s", error);
    }
    return engine;
}

/* Helper: add small obstacle geometry to the world */
static void add_test_geometry(WorldBrickMap* world) {
    Vec3 center = {5.0f, 0.0f, 5.0f};
    world_set_sphere(world, center, 2.0f, 1);

    Vec3 box_center = {0.0f, 0.0f, -1.0f};
    Vec3 half_size = {10.0f, 10.0f, 1.0f};
    world_set_box(world, box_center, half_size, 2);
}

/* Helper: check for NaN/Inf in float array */
static int count_nan_inf(const float* data, size_t count) {
    int bad = 0;
    for (size_t i = 0; i < count; i++) {
        if (isnan(data[i]) || isinf(data[i])) bad++;
    }
    return bad;
}

/* Helper: check if any value is non-zero */
static int has_nonzero(const float* data, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (data[i] != 0.0f) return 1;
    }
    return 0;
}

/* ============================================================================
 * Test 1: Engine creates with GPU context on Metal-capable hardware
 * ============================================================================ */

TEST(gpu_context_created) {
    SensorConfig sensors[3];
    sensors[0] = sensor_config_imu();
    sensors[1] = sensor_config_position();

    /* Camera depth 16x16 - small for fast testing */
    sensors[2] = sensor_config_default(SENSOR_TYPE_CAMERA_DEPTH);
    sensors[2].camera.width = 16;
    sensors[2].camera.height = 16;
    sensors[2].camera.fov_horizontal = (float)(M_PI / 2.0);
    sensors[2].camera.fov_vertical = (float)(M_PI / 2.0);
    sensors[2].camera.far_clip = 50.0f;

    BatchEngine* engine = create_gpu_test_engine(4, 4, sensors, 3);
    ASSERT_NOT_NULL(engine);

    if (gpu_is_available()) {
        ASSERT_NOT_NULL(engine->gpu_sensor_ctx);
        printf(" (GPU active)");
    } else {
        ASSERT_TRUE(engine->gpu_sensor_ctx == NULL);
        printf(" (GPU not available, CPU fallback)");
    }

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 2: Engine step works with GPU sensors (depth camera + IMU + position)
 * ============================================================================ */

TEST(step_with_gpu_sensors) {
    SensorConfig sensors[3];
    sensors[0] = sensor_config_imu();
    sensors[1] = sensor_config_position();
    sensors[2] = sensor_config_default(SENSOR_TYPE_CAMERA_DEPTH);
    sensors[2].camera.width = 16;
    sensors[2].camera.height = 16;
    sensors[2].camera.fov_horizontal = (float)(M_PI / 2.0);
    sensors[2].camera.fov_vertical = (float)(M_PI / 2.0);
    sensors[2].camera.far_clip = 50.0f;

    BatchEngine* engine = create_gpu_test_engine(4, 4, sensors, 3);
    ASSERT_NOT_NULL(engine);

    /* Add geometry so camera has something to see */
    add_test_geometry(engine->world);

    engine_reset(engine);

    /* Run 10 steps */
    for (int i = 0; i < 10; i++) {
        engine_step(engine);
    }

    ASSERT_EQ(engine->total_steps, 10);

    /* Observations should be populated */
    float* obs = engine_get_observations(engine);
    ASSERT_NOT_NULL(obs);

    uint32_t obs_dim = engine_get_obs_dim(engine);
    uint32_t total = engine->config.total_agents;
    ASSERT_GT(obs_dim, 0);
    ASSERT_TRUE(has_nonzero(obs, (size_t)total * obs_dim));

    /* No NaN/Inf */
    int bad = count_nan_inf(obs, (size_t)total * obs_dim);
    ASSERT_MSG(bad == 0, "Found NaN/Inf in observations");

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 3: GPU timing is recorded
 * ============================================================================ */

TEST(gpu_timing_recorded) {
    SensorConfig sensors[2];
    sensors[0] = sensor_config_imu();
    sensors[1] = sensor_config_default(SENSOR_TYPE_CAMERA_DEPTH);
    sensors[1].camera.width = 32;
    sensors[1].camera.height = 32;
    sensors[1].camera.fov_horizontal = (float)(M_PI / 2.0);
    sensors[1].camera.fov_vertical = (float)(M_PI / 2.0);
    sensors[1].camera.far_clip = 50.0f;

    BatchEngine* engine = create_gpu_test_engine(4, 4, sensors, 2);
    ASSERT_NOT_NULL(engine);
    add_test_geometry(engine->world);

    engine_reset(engine);
    engine_step(engine);

    /* Timing should be non-negative */
    ASSERT_TRUE(engine->physics_time_ms >= 0.0);
    ASSERT_TRUE(engine->collision_time_ms >= 0.0);
    ASSERT_TRUE(engine->sensor_time_ms >= 0.0);
    ASSERT_TRUE(engine->reward_time_ms >= 0.0);

    if (gpu_is_available()) {
        ASSERT_TRUE(engine->gpu_sensor_time_ms >= 0.0);
        printf(" (GPU: %.2fms)", engine->gpu_sensor_time_ms);
    }

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 4: Multiple GPU sensor types work together
 * ============================================================================ */

TEST(mixed_gpu_cpu_sensors) {
    SensorConfig sensors[5];
    /* CPU sensors */
    sensors[0] = sensor_config_imu();
    sensors[1] = sensor_config_position();
    sensors[2] = sensor_config_velocity();

    /* GPU sensors */
    sensors[3] = sensor_config_default(SENSOR_TYPE_CAMERA_DEPTH);
    sensors[3].camera.width = 16;
    sensors[3].camera.height = 16;
    sensors[3].camera.fov_horizontal = (float)(M_PI / 2.0);
    sensors[3].camera.fov_vertical = (float)(M_PI / 2.0);
    sensors[3].camera.far_clip = 50.0f;

    Vec3 tof_dir = {1.0f, 0.0f, 0.0f};
    sensors[4] = sensor_config_tof(tof_dir, 50.0f);

    BatchEngine* engine = create_gpu_test_engine(4, 4, sensors, 5);
    ASSERT_NOT_NULL(engine);
    add_test_geometry(engine->world);

    engine_reset(engine);

    for (int i = 0; i < 5; i++) {
        engine_step(engine);
    }

    ASSERT_EQ(engine->total_steps, 5);

    float* obs = engine_get_observations(engine);
    ASSERT_NOT_NULL(obs);
    uint32_t obs_dim = engine_get_obs_dim(engine);
    uint32_t total = engine->config.total_agents;

    /* Check for sanity: obs should have values and no NaN */
    ASSERT_TRUE(has_nonzero(obs, (size_t)total * obs_dim));
    int bad = count_nan_inf(obs, (size_t)total * obs_dim);
    ASSERT_MSG(bad == 0, "Found NaN/Inf in observations");

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 5: Sustained steps (memory stability)
 * ============================================================================ */

TEST(sustained_gpu_steps) {
    SensorConfig sensors[2];
    sensors[0] = sensor_config_position();
    sensors[1] = sensor_config_default(SENSOR_TYPE_CAMERA_DEPTH);
    sensors[1].camera.width = 16;
    sensors[1].camera.height = 16;
    sensors[1].camera.fov_horizontal = (float)(M_PI / 2.0);
    sensors[1].camera.fov_vertical = (float)(M_PI / 2.0);
    sensors[1].camera.far_clip = 50.0f;

    BatchEngine* engine = create_gpu_test_engine(4, 4, sensors, 2);
    ASSERT_NOT_NULL(engine);
    add_test_geometry(engine->world);

    engine_reset(engine);

    /* 100 steps - checks memory stability and no leaks in frame-level GPU ops */
    for (int i = 0; i < 100; i++) {
        engine_step(engine);
    }

    ASSERT_EQ(engine->total_steps, 100);

    /* Final observations still valid */
    float* obs = engine_get_observations(engine);
    ASSERT_NOT_NULL(obs);
    uint32_t obs_dim = engine_get_obs_dim(engine);
    uint32_t total = engine->config.total_agents;
    int bad = count_nan_inf(obs, (size_t)total * obs_dim);
    ASSERT_MSG(bad == 0, "Found NaN/Inf after 100 steps");

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 6: step_no_reset with GPU sensors
 * ============================================================================ */

TEST(step_no_reset_gpu) {
    SensorConfig sensors[2];
    sensors[0] = sensor_config_imu();
    sensors[1] = sensor_config_default(SENSOR_TYPE_CAMERA_DEPTH);
    sensors[1].camera.width = 8;
    sensors[1].camera.height = 8;
    sensors[1].camera.fov_horizontal = (float)(M_PI / 2.0);
    sensors[1].camera.fov_vertical = (float)(M_PI / 2.0);
    sensors[1].camera.far_clip = 50.0f;

    BatchEngine* engine = create_gpu_test_engine(4, 4, sensors, 2);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    for (int i = 0; i < 5; i++) {
        engine_step_no_reset(engine);
    }

    ASSERT_EQ(engine->total_steps, 5);

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 7: Larger drone count scaling
 * ============================================================================ */

TEST(scaling_64_drones) {
    SensorConfig sensors[2];
    sensors[0] = sensor_config_position();
    sensors[1] = sensor_config_default(SENSOR_TYPE_CAMERA_DEPTH);
    sensors[1].camera.width = 16;
    sensors[1].camera.height = 16;
    sensors[1].camera.fov_horizontal = (float)(M_PI / 2.0);
    sensors[1].camera.fov_vertical = (float)(M_PI / 2.0);
    sensors[1].camera.far_clip = 50.0f;

    /* 8 envs x 8 drones = 64 drones */
    BatchEngine* engine = create_gpu_test_engine(8, 8, sensors, 2);
    ASSERT_NOT_NULL(engine);
    /* No geometry - tests scaling without world obstacles */

    engine_reset(engine);

    double total_time = 0.0;
    int num_steps = 10;
    for (int i = 0; i < num_steps; i++) {
        double start = engine_get_time_ms();
        engine_step(engine);
        total_time += engine_get_time_ms() - start;
    }

    double avg_ms = total_time / num_steps;
    printf(" (64 drones, avg %.2fms/step)", avg_ms);

    /* Observations valid */
    float* obs = engine_get_observations(engine);
    uint32_t obs_dim = engine_get_obs_dim(engine);
    uint32_t total = engine->config.total_agents;
    int bad = count_nan_inf(obs, (size_t)total * obs_dim);
    ASSERT_MSG(bad == 0, "Found NaN/Inf");

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 8: Auto-reset works correctly with GPU sensors
 * ============================================================================ */

TEST(auto_reset_gpu) {
    SensorConfig sensors[2];
    sensors[0] = sensor_config_position();
    sensors[1] = sensor_config_default(SENSOR_TYPE_CAMERA_DEPTH);
    sensors[1].camera.width = 8;
    sensors[1].camera.height = 8;
    sensors[1].camera.fov_horizontal = (float)(M_PI / 2.0);
    sensors[1].camera.fov_vertical = (float)(M_PI / 2.0);
    sensors[1].camera.far_clip = 50.0f;

    BatchEngine* engine = create_gpu_test_engine(4, 4, sensors, 2);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Force a drone out of bounds to trigger termination + auto-reset */
    engine->states->rigid_body.pos_z[0] = engine->config.world_min.z - 20.0f;

    engine_step(engine);

    /* After auto-reset, drone should be back in bounds */
    ASSERT_TRUE(engine->states->rigid_body.pos_z[0] >= engine->config.world_min.z);

    /* Run more steps to verify stability after reset */
    for (int i = 0; i < 10; i++) {
        engine_step(engine);
    }

    float* obs = engine_get_observations(engine);
    uint32_t obs_dim = engine_get_obs_dim(engine);
    uint32_t total = engine->config.total_agents;
    int bad = count_nan_inf(obs, (size_t)total * obs_dim);
    ASSERT_MSG(bad == 0, "Found NaN/Inf after auto-reset");

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 9: CPU-only fallback (no GPU sensors) still works
 * ============================================================================ */

TEST(cpu_only_fallback) {
    /* Only CPU sensors - no GPU dispatch should happen */
    SensorConfig sensors[3];
    sensors[0] = sensor_config_imu();
    sensors[1] = sensor_config_position();
    sensors[2] = sensor_config_velocity();

    BatchEngine* engine = create_gpu_test_engine(4, 4, sensors, 3);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);
    for (int i = 0; i < 10; i++) {
        engine_step(engine);
    }

    ASSERT_EQ(engine->total_steps, 10);

    float* obs = engine_get_observations(engine);
    ASSERT_NOT_NULL(obs);
    uint32_t obs_dim = engine_get_obs_dim(engine);
    uint32_t total = engine->config.total_agents;
    ASSERT_TRUE(has_nonzero(obs, (size_t)total * obs_dim));

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

    TEST_SUITE_BEGIN("GPU Pipeline Integration Tests");

    RUN_TEST(gpu_context_created);
    RUN_TEST(step_with_gpu_sensors);
    RUN_TEST(gpu_timing_recorded);
    RUN_TEST(mixed_gpu_cpu_sensors);
    RUN_TEST(sustained_gpu_steps);
    RUN_TEST(step_no_reset_gpu);
    RUN_TEST(scaling_64_drones);
    RUN_TEST(auto_reset_gpu);
    RUN_TEST(cpu_only_fallback);

    TEST_SUITE_END();
}
