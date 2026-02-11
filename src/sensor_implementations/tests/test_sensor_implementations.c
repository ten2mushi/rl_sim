/**
 * Sensor Implementations Unit Tests
 *
 * Tests for all 10 sensor type implementations.
 *
 * Test Categories per Sensor:
 * - Output size and shape
 * - Correct data type
 * - Batch sampling behavior
 * - Noise application
 * - Edge cases and error handling
 */

#include "sensor_system.h"
#include "sensor_implementations.h"
#include "test_harness.h"

/* ============================================================================
 * Test Helpers
 * ============================================================================ */

static Arena* test_arena = NULL;
static SensorSystem* test_sys = NULL;
static DroneStateSOA* test_drones = NULL;

static int setup_test_env(uint32_t num_drones) {
    test_arena = arena_create(16 * 1024 * 1024);
    if (!test_arena) return -1;

    test_sys = sensor_system_create(test_arena, num_drones, 16, 256);
    if (!test_sys) return -1;

    sensor_implementations_register_all(&test_sys->registry);

    test_drones = drone_state_create(test_arena, num_drones);
    if (!test_drones) return -1;

    for (uint32_t i = 0; i < num_drones; i++) {
        drone_state_init(test_drones, i);
        /* Set some non-trivial state */
        test_drones->pos_x[i] = (float)i * 1.0f;
        test_drones->pos_y[i] = (float)i * 2.0f;
        test_drones->pos_z[i] = (float)i * 0.5f + 10.0f;
        test_drones->vel_x[i] = 0.1f;
        test_drones->vel_y[i] = 0.2f;
        test_drones->vel_z[i] = 0.3f;
        test_drones->omega_x[i] = 0.01f;
        test_drones->omega_y[i] = 0.02f;
        test_drones->omega_z[i] = 0.03f;
    }
    test_drones->count = num_drones;
    return 0;
}

static void teardown_test_env(void) {
    if (test_sys) sensor_system_destroy(test_sys);
    if (test_arena) arena_destroy(test_arena);
    test_sys = NULL;
    test_arena = NULL;
    test_drones = NULL;
}

/* ============================================================================
 * Position Sensor Tests
 * ============================================================================ */

TEST(position_output_size) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_position();
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    ASSERT_NE(idx, UINT32_MAX);

    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);
    ASSERT_EQ(sensor->output_size, 3);

    teardown_test_env();
    return 0;
}

TEST(position_reads_correct_values) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_position();
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);
    sensor_system_attach(test_sys, 2, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs0 = sensor_system_get_drone_obs_const(test_sys, 0);
    const float* obs2 = sensor_system_get_drone_obs_const(test_sys, 2);

    ASSERT_FLOAT_NEAR(obs0[0], 0.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs0[1], 0.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs0[2], 10.0f, 0.001f);

    ASSERT_FLOAT_NEAR(obs2[0], 2.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs2[1], 4.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs2[2], 11.0f, 0.001f);

    teardown_test_env();
    return 0;
}

/* ============================================================================
 * Velocity Sensor Tests
 * ============================================================================ */

TEST(velocity_output_size) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_velocity();
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    ASSERT_NE(idx, UINT32_MAX);

    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);
    ASSERT_EQ(sensor->output_size, 6);

    teardown_test_env();
    return 0;
}

TEST(velocity_reads_correct_values) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_velocity();
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 1, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 1);

    ASSERT_FLOAT_NEAR(obs[0], 0.1f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[1], 0.2f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[2], 0.3f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[3], 0.01f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[4], 0.02f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[5], 0.03f, 0.001f);

    teardown_test_env();
    return 0;
}

/* ============================================================================
 * IMU Sensor Tests
 * ============================================================================ */

TEST(imu_output_size) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_imu();  /* No noise */
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    ASSERT_NE(idx, UINT32_MAX);

    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);
    ASSERT_EQ(sensor->output_size, 6);

    teardown_test_env();
    return 0;
}

TEST(imu_gravity_at_hover) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_imu();  /* No noise */
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    /* Drone 0 has identity quaternion (hover), omega already set */
    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    /* At hover with identity quat, accel should be approximately (0, 0, 9.81) */
    ASSERT_FLOAT_NEAR(obs[0], 0.0f, 0.01f);
    ASSERT_FLOAT_NEAR(obs[1], 0.0f, 0.01f);
    ASSERT_FLOAT_NEAR(obs[2], 9.81f, 0.1f);

    /* Gyro should match omega */
    ASSERT_FLOAT_NEAR(obs[3], 0.01f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[4], 0.02f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[5], 0.03f, 0.001f);

    teardown_test_env();
    return 0;
}

TEST(imu_clean_signal_no_noise) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_imu();
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    /* Sample multiple times - without noise config, output should be deterministic */
    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);
    const float* obs1 = sensor_system_get_drone_obs_const(test_sys, 0);
    float val1 = obs1[0];

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);
    const float* obs2 = sensor_system_get_drone_obs_const(test_sys, 0);
    float val2 = obs2[0];

    /* Without noise pipeline, repeated samples produce identical values */
    ASSERT_FLOAT_EQ(val1, val2);

    teardown_test_env();
    return 0;
}

/* ============================================================================
 * ToF Sensor Tests
 * ============================================================================ */

TEST(tof_output_size) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_tof(VEC3(0, 0, -1), 10.0f);
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    ASSERT_NE(idx, UINT32_MAX);

    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);
    ASSERT_EQ(sensor->output_size, 1);

    teardown_test_env();
    return 0;
}

TEST(tof_no_world_returns_max) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_tof(VEC3(0, 0, -1), 50.0f);
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);
    ASSERT_FLOAT_NEAR(obs[0], 50.0f, 0.01f);

    teardown_test_env();
    return 0;
}

/* ============================================================================
 * LiDAR 2D Tests
 * ============================================================================ */

TEST(lidar_2d_output_size) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_lidar_2d(64, 3.14159f, 20.0f);
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    ASSERT_NE(idx, UINT32_MAX);

    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);
    ASSERT_EQ(sensor->output_size, 64);

    teardown_test_env();
    return 0;
}

TEST(lidar_2d_shape) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_lidar_2d(32, 3.14159f, 20.0f);
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);

    uint32_t shape[4];
    uint32_t ndim = sensor->vtable->get_output_shape(sensor, shape);

    ASSERT_EQ(ndim, 1);
    ASSERT_EQ(shape[0], 32);

    teardown_test_env();
    return 0;
}

/* ============================================================================
 * LiDAR 3D Tests
 * ============================================================================ */

TEST(lidar_3d_output_size) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_lidar_3d(64, 16, 6.28f, 0.5f, 50.0f);
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    ASSERT_NE(idx, UINT32_MAX);

    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);
    ASSERT_EQ(sensor->output_size, 64 * 16);

    teardown_test_env();
    return 0;
}

TEST(lidar_3d_shape) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_lidar_3d(64, 16, 6.28f, 0.5f, 50.0f);
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);

    uint32_t shape[4];
    uint32_t ndim = sensor->vtable->get_output_shape(sensor, shape);

    ASSERT_EQ(ndim, 2);
    ASSERT_EQ(shape[0], 16);
    ASSERT_EQ(shape[1], 64);

    teardown_test_env();
    return 0;
}

/* ============================================================================
 * Camera RGB Tests
 * ============================================================================ */

TEST(camera_rgb_output_size) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_camera(32, 32, 1.57f, 100.0f);
    config.type = SENSOR_TYPE_CAMERA_RGB;
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    ASSERT_NE(idx, UINT32_MAX);

    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);
    ASSERT_EQ(sensor->output_size, 32 * 32 * 3);

    teardown_test_env();
    return 0;
}

TEST(camera_rgb_shape) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_camera(64, 48, 1.57f, 100.0f);
    config.type = SENSOR_TYPE_CAMERA_RGB;
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);

    uint32_t shape[4];
    uint32_t ndim = sensor->vtable->get_output_shape(sensor, shape);

    ASSERT_EQ(ndim, 3);
    ASSERT_EQ(shape[0], 48);
    ASSERT_EQ(shape[1], 64);
    ASSERT_EQ(shape[2], 3);

    teardown_test_env();
    return 0;
}

TEST(camera_rgb_sky_on_miss) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_camera(8, 8, 1.57f, 100.0f);
    config.type = SENSOR_TYPE_CAMERA_RGB;
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    /* With no world, should return sky color (0.5, 0.7, 0.9) for all pixels */
    ASSERT_FLOAT_NEAR(obs[0], 0.5f, 0.01f);
    ASSERT_FLOAT_NEAR(obs[1], 0.7f, 0.01f);
    ASSERT_FLOAT_NEAR(obs[2], 0.9f, 0.01f);

    teardown_test_env();
    return 0;
}

/* ============================================================================
 * Camera Depth Tests
 * ============================================================================ */

TEST(camera_depth_output_size) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_camera(32, 32, 1.57f, 100.0f);
    config.type = SENSOR_TYPE_CAMERA_DEPTH;
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    ASSERT_NE(idx, UINT32_MAX);

    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);
    ASSERT_EQ(sensor->output_size, 32 * 32);

    teardown_test_env();
    return 0;
}

TEST(camera_depth_no_world_returns_max) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_camera(8, 8, 1.57f, 100.0f);
    config.type = SENSOR_TYPE_CAMERA_DEPTH;
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    /* With no world, depth should be 1.0 (max normalized depth) */
    ASSERT_FLOAT_NEAR(obs[0], 1.0f, 0.01f);

    teardown_test_env();
    return 0;
}

/* ============================================================================
 * Camera Segmentation Tests
 * ============================================================================ */

TEST(camera_seg_output_size) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_camera(32, 32, 1.57f, 100.0f);
    config.type = SENSOR_TYPE_CAMERA_SEGMENTATION;
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    ASSERT_NE(idx, UINT32_MAX);

    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);
    ASSERT_EQ(sensor->output_size, 32 * 32);

    teardown_test_env();
    return 0;
}

TEST(camera_seg_no_world_returns_zero) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_camera(8, 8, 1.57f, 100.0f);
    config.type = SENSOR_TYPE_CAMERA_SEGMENTATION;
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    /* With no world, material should be 0 (sky/air) */
    ASSERT_FLOAT_NEAR(obs[0], 0.0f, 0.01f);

    teardown_test_env();
    return 0;
}

/* ============================================================================
 * Neighbor Sensor Tests
 * ============================================================================ */

TEST(neighbor_output_size) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_neighbor(5, 20.0f);
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    ASSERT_NE(idx, UINT32_MAX);

    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);
    ASSERT_EQ(sensor->output_size, 5 * 4);

    teardown_test_env();
    return 0;
}

TEST(neighbor_shape) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_neighbor(3, 20.0f);
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);

    uint32_t shape[4];
    uint32_t ndim = sensor->vtable->get_output_shape(sensor, shape);

    ASSERT_EQ(ndim, 2);
    ASSERT_EQ(shape[0], 3);
    ASSERT_EQ(shape[1], 4);

    teardown_test_env();
    return 0;
}

TEST(neighbor_finds_nearest) {
    ASSERT_EQ(setup_test_env(16), 0);

    /* Position drones in a line */
    for (uint32_t i = 0; i < 16; i++) {
        test_drones->pos_x[i] = (float)i * 2.0f;  /* 0, 2, 4, 6, ... */
        test_drones->pos_y[i] = 0.0f;
        test_drones->pos_z[i] = 0.0f;
    }

    SensorConfig config = sensor_config_neighbor(3, 10.0f);
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);  /* Drone 0 at x=0 */

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 16);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    /* Nearest neighbors to drone 0 (at x=0) should be drones 1,2,3 (at x=2,4,6) */
    /* First neighbor (drone 1 at x=2): dx=2, dy=0, dz=0, dist=2 */
    ASSERT_FLOAT_NEAR(obs[0], 2.0f, 0.1f);
    ASSERT_FLOAT_NEAR(obs[3], 2.0f, 0.1f);

    teardown_test_env();
    return 0;
}

/* ============================================================================
 * Registration Tests
 * ============================================================================ */

TEST(register_all_sensors) {
    ASSERT_EQ(setup_test_env(4), 0);

    /* All sensor types should be registered */
    for (int t = 0; t < SENSOR_TYPE_COUNT; t++) {
        const SensorVTable* vtable = sensor_registry_get(&test_sys->registry, (SensorType)t);
        ASSERT_NOT_NULL(vtable);
    }

    teardown_test_env();
    return 0;
}

TEST(dtype_is_float32) {
    ASSERT_EQ(setup_test_env(4), 0);

    /* All sensors should return "float32" dtype */
    SensorType types[] = {
        SENSOR_TYPE_IMU, SENSOR_TYPE_TOF, SENSOR_TYPE_LIDAR_2D,
        SENSOR_TYPE_POSITION, SENSOR_TYPE_VELOCITY, SENSOR_TYPE_NEIGHBOR
    };

    for (size_t i = 0; i < sizeof(types)/sizeof(types[0]); i++) {
        SensorConfig config = sensor_config_default(types[i]);
        uint32_t idx = sensor_system_create_sensor(test_sys, &config);
        Sensor* sensor = sensor_system_get_sensor(test_sys, idx);

        const char* dtype = sensor->vtable->get_output_dtype(sensor);
        ASSERT_TRUE(strcmp(dtype, "float32") == 0);
    }

    teardown_test_env();
    return 0;
}

/* ============================================================================
 * Batch Processing Tests
 * ============================================================================ */

TEST(batch_processing_many_drones) {
    ASSERT_EQ(setup_test_env(256), 0);

    /* Create IMU sensor and attach to all drones */
    SensorConfig config = sensor_config_imu();
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);

    for (uint32_t d = 0; d < 256; d++) {
        sensor_system_attach(test_sys, d, idx);
    }

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 256);

    /* Verify all drones have valid observations */
    for (uint32_t d = 0; d < 256; d++) {
        const float* obs = sensor_system_get_drone_obs_const(test_sys, d);
        /* Should have gravity in Z */
        ASSERT_TRUE(obs[2] > 9.0f);
        ASSERT_TRUE(obs[2] < 10.0f);
    }

    teardown_test_env();
    return 0;
}

/* ============================================================================
 * Main Test Runner
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Sensor Implementations");

    /* Position Sensor Tests */
    RUN_TEST(position_output_size);
    RUN_TEST(position_reads_correct_values);

    /* Velocity Sensor Tests */
    RUN_TEST(velocity_output_size);
    RUN_TEST(velocity_reads_correct_values);

    /* IMU Sensor Tests */
    RUN_TEST(imu_output_size);
    RUN_TEST(imu_gravity_at_hover);
    RUN_TEST(imu_clean_signal_no_noise);

    /* ToF Sensor Tests */
    RUN_TEST(tof_output_size);
    RUN_TEST(tof_no_world_returns_max);

    /* LiDAR 2D Tests */
    RUN_TEST(lidar_2d_output_size);
    RUN_TEST(lidar_2d_shape);

    /* LiDAR 3D Tests */
    RUN_TEST(lidar_3d_output_size);
    RUN_TEST(lidar_3d_shape);

    /* Camera RGB Tests */
    RUN_TEST(camera_rgb_output_size);
    RUN_TEST(camera_rgb_shape);
    RUN_TEST(camera_rgb_sky_on_miss);

    /* Camera Depth Tests */
    RUN_TEST(camera_depth_output_size);
    RUN_TEST(camera_depth_no_world_returns_max);

    /* Camera Segmentation Tests */
    RUN_TEST(camera_seg_output_size);
    RUN_TEST(camera_seg_no_world_returns_zero);

    /* Neighbor Sensor Tests */
    RUN_TEST(neighbor_output_size);
    RUN_TEST(neighbor_shape);
    RUN_TEST(neighbor_finds_nearest);

    /* Registration Tests */
    RUN_TEST(register_all_sensors);
    RUN_TEST(dtype_is_float32);

    /* Batch Processing Tests */
    RUN_TEST(batch_processing_many_drones);

    TEST_SUITE_END();
}
