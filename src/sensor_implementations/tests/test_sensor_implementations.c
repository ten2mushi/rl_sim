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
#include "drone_state.h"
#include "platform_quadcopter.h"
#include "test_harness.h"

/* ============================================================================
 * Test Helpers
 * ============================================================================ */

static Arena* test_arena = NULL;
static SensorSystem* test_sys = NULL;
static PlatformStateSOA* test_drones = NULL;

static int setup_test_env(uint32_t num_agents) {
    test_arena = arena_create(16 * 1024 * 1024);
    if (!test_arena) return -1;

    test_sys = sensor_system_create(test_arena, num_agents, 16, 256);
    if (!test_sys) return -1;

    sensor_implementations_register_all(&test_sys->registry);

    test_drones = platform_state_create(test_arena, num_agents, QUAD_STATE_EXT_COUNT);
    if (!test_drones) return -1;

    for (uint32_t i = 0; i < num_agents; i++) {
        platform_state_init(test_drones, i);
        /* Set some non-trivial state */
        test_drones->rigid_body.pos_x[i] = (float)i * 1.0f;
        test_drones->rigid_body.pos_y[i] = (float)i * 2.0f;
        test_drones->rigid_body.pos_z[i] = (float)i * 0.5f + 10.0f;
        test_drones->rigid_body.vel_x[i] = 0.1f;
        test_drones->rigid_body.vel_y[i] = 0.2f;
        test_drones->rigid_body.vel_z[i] = 0.3f;
        test_drones->rigid_body.omega_x[i] = 0.01f;
        test_drones->rigid_body.omega_y[i] = 0.02f;
        test_drones->rigid_body.omega_z[i] = 0.03f;
    }
    test_drones->rigid_body.count = num_agents;
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
        test_drones->rigid_body.pos_x[i] = (float)i * 2.0f;  /* 0, 2, 4, 6, ... */
        test_drones->rigid_body.pos_y[i] = 0.0f;
        test_drones->rigid_body.pos_z[i] = 0.0f;
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
 * Ray Precomputation Tests
 * ============================================================================ */

TEST(lidar_2d_rays_are_unit_length) {
    Arena* arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    Vec3* rays = precompute_lidar_2d_rays(arena, 32, 3.14159f);
    ASSERT_NOT_NULL(rays);

    for (uint32_t i = 0; i < 32; i++) {
        float len = sqrtf(rays[i].x * rays[i].x + rays[i].y * rays[i].y + rays[i].z * rays[i].z);
        ASSERT_FLOAT_NEAR(len, 1.0f, 1e-5f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(lidar_2d_rays_lie_in_xy_plane) {
    Arena* arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    Vec3* rays = precompute_lidar_2d_rays(arena, 64, 3.14159f);
    ASSERT_NOT_NULL(rays);

    /* All 2D lidar rays should have z=0 (horizontal plane) */
    for (uint32_t i = 0; i < 64; i++) {
        ASSERT_FLOAT_NEAR(rays[i].z, 0.0f, 1e-6f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(lidar_2d_rays_span_fov_symmetrically) {
    Arena* arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    float fov = 3.14159f; /* 180 degrees */
    uint32_t num_rays = 33; /* Odd so there is a center ray */
    Vec3* rays = precompute_lidar_2d_rays(arena, num_rays, fov);
    ASSERT_NOT_NULL(rays);

    /* First ray should be at -half_fov, last at +half_fov */
    float half_fov = fov * 0.5f;
    float first_angle = atan2f(rays[0].y, rays[0].x);
    float last_angle = atan2f(rays[num_rays - 1].y, rays[num_rays - 1].x);

    ASSERT_FLOAT_NEAR(first_angle, -half_fov, 0.01f);
    ASSERT_FLOAT_NEAR(last_angle, half_fov, 0.01f);

    /* Center ray (index 16) should point along +X (angle = 0) */
    float center_angle = atan2f(rays[16].y, rays[16].x);
    ASSERT_FLOAT_NEAR(center_angle, 0.0f, 0.01f);

    arena_destroy(arena);
    return 0;
}

TEST(lidar_2d_rays_single_ray_points_forward) {
    Arena* arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Single ray: angle_step=0, angle = -half_fov + 0 = -half_fov
     * Actually with num_rays=1, angle_step = 0, angle = -half_fov for i=0 */
    Vec3* rays = precompute_lidar_2d_rays(arena, 1, 1.0f);
    ASSERT_NOT_NULL(rays);

    /* Single ray starts at -half_fov. With fov=1.0, angle = -0.5 */
    float expected_angle = -0.5f;
    ASSERT_FLOAT_NEAR(rays[0].x, cosf(expected_angle), 1e-5f);
    ASSERT_FLOAT_NEAR(rays[0].y, sinf(expected_angle), 1e-5f);

    arena_destroy(arena);
    return 0;
}

TEST(lidar_2d_rays_null_arena_returns_null) {
    Vec3* rays = precompute_lidar_2d_rays(NULL, 32, 3.14159f);
    ASSERT_NULL(rays);
    return 0;
}

TEST(lidar_2d_rays_zero_count_returns_null) {
    Arena* arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    Vec3* rays = precompute_lidar_2d_rays(arena, 0, 3.14159f);
    ASSERT_NULL(rays);

    arena_destroy(arena);
    return 0;
}

TEST(lidar_3d_rays_total_count) {
    Arena* arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    uint32_t h = 16;
    uint32_t v = 8;
    Vec3* rays = precompute_lidar_3d_rays(arena, h, v, 3.14159f, 0.5236f);
    ASSERT_NOT_NULL(rays);

    /* Verify all rays are unit length (total = h * v = 128) */
    for (uint32_t i = 0; i < h * v; i++) {
        float len = sqrtf(rays[i].x * rays[i].x + rays[i].y * rays[i].y + rays[i].z * rays[i].z);
        ASSERT_FLOAT_NEAR(len, 1.0f, 1e-4f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(lidar_3d_rays_vertical_spread) {
    Arena* arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 1 horizontal ray, 5 vertical layers, vertical_fov = 60 degrees */
    float v_fov = 1.0472f; /* ~60 degrees */
    Vec3* rays = precompute_lidar_3d_rays(arena, 1, 5, 3.14159f, v_fov);
    ASSERT_NOT_NULL(rays);

    /* With 1 horizontal ray, rays are at index 0,1,2,3,4
     * Elevation goes from -half_v_fov to +half_v_fov
     * Bottom ray (v=0) should have negative z, top ray (v=4) positive z */
    ASSERT_TRUE(rays[0].z < 0.0f); /* Bottom layer: negative elevation */
    ASSERT_FLOAT_NEAR(rays[2].z, 0.0f, 0.01f); /* Middle layer: ~zero elevation */
    ASSERT_TRUE(rays[4].z > 0.0f); /* Top layer: positive elevation */

    arena_destroy(arena);
    return 0;
}

TEST(lidar_3d_rays_null_inputs) {
    Arena* arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    ASSERT_NULL(precompute_lidar_3d_rays(NULL, 16, 8, 3.14f, 0.5f));
    ASSERT_NULL(precompute_lidar_3d_rays(arena, 0, 8, 3.14f, 0.5f));
    ASSERT_NULL(precompute_lidar_3d_rays(arena, 16, 0, 3.14f, 0.5f));

    arena_destroy(arena);
    return 0;
}

TEST(camera_rays_total_count_and_unit_length) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    uint32_t w = 16;
    uint32_t h = 12;
    float fov_h = 1.5708f; /* 90 degrees */
    float fov_v = 1.2f;
    Vec3* rays = precompute_camera_rays(arena, w, h, fov_h, fov_v);
    ASSERT_NOT_NULL(rays);

    /* Verify all w*h rays are unit length */
    for (uint32_t i = 0; i < w * h; i++) {
        float len = sqrtf(rays[i].x * rays[i].x + rays[i].y * rays[i].y + rays[i].z * rays[i].z);
        ASSERT_FLOAT_NEAR(len, 1.0f, 1e-4f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(camera_rays_center_points_forward) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Even width/height: center is between pixels, but the closest to center should
     * be nearly along +X. Use odd dimensions for exact center. */
    uint32_t w = 17;
    uint32_t h = 17;
    float fov_h = 1.5708f;
    float fov_v = 1.5708f;
    Vec3* rays = precompute_camera_rays(arena, w, h, fov_h, fov_v);
    ASSERT_NOT_NULL(rays);

    /* Center pixel at (8, 8): u and v should be ~0 */
    uint32_t center_idx = 8 * w + 8;
    Vec3 center = rays[center_idx];

    /* Camera looks along +X, so center ray should be mostly +X */
    ASSERT_TRUE(center.x > 0.9f);
    /* Y and Z should be near zero for center pixel */
    ASSERT_FLOAT_NEAR(center.y, 0.0f, 0.05f);
    ASSERT_FLOAT_NEAR(center.z, 0.0f, 0.05f);

    arena_destroy(arena);
    return 0;
}

TEST(camera_rays_null_inputs) {
    Arena* arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    ASSERT_NULL(precompute_camera_rays(NULL, 16, 16, 1.5f, 1.5f));
    ASSERT_NULL(precompute_camera_rays(arena, 0, 16, 1.5f, 1.5f));
    ASSERT_NULL(precompute_camera_rays(arena, 16, 0, 1.5f, 1.5f));

    arena_destroy(arena);
    return 0;
}

TEST(camera_rays_horizontal_symmetry) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Square image with symmetric FOV */
    uint32_t w = 16;
    uint32_t h = 16;
    float fov = 1.5708f;
    Vec3* rays = precompute_camera_rays(arena, w, h, fov, fov);
    ASSERT_NOT_NULL(rays);

    /* Row center: for each row, left and right pixels should be mirror images in Y.
     * pixel(row, 0) and pixel(row, 15) should have opposite Y components. */
    for (uint32_t row = 0; row < h; row++) {
        Vec3 left = rays[row * w + 0];
        Vec3 right = rays[row * w + (w - 1)];
        /* Y component should be approximately negated */
        ASSERT_FLOAT_NEAR(left.y, -right.y, 0.05f);
        /* X and Z should be the same (same forward and vertical component) */
        ASSERT_FLOAT_NEAR(left.x, right.x, 0.01f);
        ASSERT_FLOAT_NEAR(left.z, right.z, 0.01f);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * IMU Behavioral Tests
 * ============================================================================ */

TEST(imu_rotated_drone_changes_accel) {
    /* A drone pitched 90 degrees forward should measure gravity differently.
     * With a 90-degree pitch (rotation about Y by -pi/2), the drone's body +Z
     * points along world +X. Gravity (world +Z) projects onto body -X.
     * So accel should be approximately (-9.81, 0, 0) in body frame. */
    ASSERT_EQ(setup_test_env(4), 0);

    /* Set drone 0 orientation: 90 degrees pitch forward (rotate around Y axis) */
    Quat pitched = quat_from_axis_angle(VEC3(0, 1, 0), -1.5708f);
    test_drones->rigid_body.quat_w[0] = pitched.w;
    test_drones->rigid_body.quat_x[0] = pitched.x;
    test_drones->rigid_body.quat_y[0] = pitched.y;
    test_drones->rigid_body.quat_z[0] = pitched.z;
    test_drones->rigid_body.omega_x[0] = 0.0f;
    test_drones->rigid_body.omega_y[0] = 0.0f;
    test_drones->rigid_body.omega_z[0] = 0.0f;

    SensorConfig config = sensor_config_imu();
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    /* Gravity in body frame after 90-degree pitch:
     * q_inv rotates world gravity (0,0,9.81) into body frame.
     * The exact values depend on the quaternion convention, but the key test is
     * that Z-acceleration is NOT ~9.81 (it was at hover), and some other axis
     * has large magnitude. */
    float accel_mag = sqrtf(obs[0]*obs[0] + obs[1]*obs[1] + obs[2]*obs[2]);
    ASSERT_FLOAT_NEAR(accel_mag, 9.81f, 0.2f); /* Total magnitude preserved */

    /* Z-component should be near zero (gravity now along another body axis) */
    ASSERT_TRUE(fabsf(obs[2]) < 1.0f);

    /* Gyro should be zero */
    ASSERT_FLOAT_NEAR(obs[3], 0.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[4], 0.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[5], 0.0f, 0.001f);

    teardown_test_env();
    return 0;
}

TEST(imu_inverted_drone_negative_z_accel) {
    /* A drone flipped upside down (180 degrees about X) should see
     * accel Z = -9.81 (gravity now points in -Z body frame). */
    ASSERT_EQ(setup_test_env(4), 0);

    Quat inverted = quat_from_axis_angle(VEC3(1, 0, 0), 3.14159f);
    test_drones->rigid_body.quat_w[0] = inverted.w;
    test_drones->rigid_body.quat_x[0] = inverted.x;
    test_drones->rigid_body.quat_y[0] = inverted.y;
    test_drones->rigid_body.quat_z[0] = inverted.z;
    test_drones->rigid_body.omega_x[0] = 0.0f;
    test_drones->rigid_body.omega_y[0] = 0.0f;
    test_drones->rigid_body.omega_z[0] = 0.0f;

    SensorConfig config = sensor_config_imu();
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    /* Inverted: body Z is world -Z, so gravity (world +Z) = body -Z */
    ASSERT_FLOAT_NEAR(obs[0], 0.0f, 0.5f);
    ASSERT_FLOAT_NEAR(obs[1], 0.0f, 0.5f);
    ASSERT_FLOAT_NEAR(obs[2], -9.81f, 0.2f);

    teardown_test_env();
    return 0;
}

TEST(imu_gyro_reads_omega_directly) {
    ASSERT_EQ(setup_test_env(4), 0);

    /* Set specific angular velocity */
    test_drones->rigid_body.omega_x[0] = 1.5f;
    test_drones->rigid_body.omega_y[0] = -0.5f;
    test_drones->rigid_body.omega_z[0] = 2.0f;

    SensorConfig config = sensor_config_imu();
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    /* Gyro output should exactly match body-frame omega */
    ASSERT_FLOAT_NEAR(obs[3], 1.5f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[4], -0.5f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[5], 2.0f, 0.001f);

    teardown_test_env();
    return 0;
}

TEST(imu_zero_angular_velocity) {
    ASSERT_EQ(setup_test_env(4), 0);

    /* Explicitly set omega to zero */
    test_drones->rigid_body.omega_x[0] = 0.0f;
    test_drones->rigid_body.omega_y[0] = 0.0f;
    test_drones->rigid_body.omega_z[0] = 0.0f;

    SensorConfig config = sensor_config_imu();
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    ASSERT_FLOAT_EQ(obs[3], 0.0f);
    ASSERT_FLOAT_EQ(obs[4], 0.0f);
    ASSERT_FLOAT_EQ(obs[5], 0.0f);

    teardown_test_env();
    return 0;
}

/* ============================================================================
 * ToF Behavioral Tests
 * ============================================================================ */

TEST(tof_different_direction_vectors) {
    /* Two ToF sensors pointing in different directions should both
     * return max range when no world is present, but they should store
     * different direction vectors in their impl. */
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config_down = sensor_config_tof(VEC3(0, 0, -1), 10.0f);
    SensorConfig config_fwd = sensor_config_tof(VEC3(1, 0, 0), 25.0f);

    uint32_t idx_down = sensor_system_create_sensor(test_sys, &config_down);
    uint32_t idx_fwd = sensor_system_create_sensor(test_sys, &config_fwd);

    sensor_system_attach(test_sys, 0, idx_down);
    sensor_system_attach(test_sys, 0, idx_fwd);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    /* First sensor (down, max_range=10): obs[0] */
    ASSERT_FLOAT_NEAR(obs[0], 10.0f, 0.01f);

    /* Second sensor (forward, max_range=25): obs[1] */
    ASSERT_FLOAT_NEAR(obs[1], 25.0f, 0.01f);

    teardown_test_env();
    return 0;
}

TEST(tof_max_range_clamping) {
    /* ToF with very small max_range should return that small value on miss */
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_tof(VEC3(0, 0, -1), 0.5f);
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);
    ASSERT_FLOAT_NEAR(obs[0], 0.5f, 0.01f);

    teardown_test_env();
    return 0;
}

TEST(tof_unnormalized_direction_is_normalized) {
    /* Verify that a non-unit direction vector gets normalized internally */
    ASSERT_EQ(setup_test_env(4), 0);

    /* Provide non-unit direction (3,4,0) with length 5 */
    SensorConfig config = sensor_config_tof(VEC3(3.0f, 4.0f, 0.0f), 10.0f);
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    ASSERT_NE(idx, UINT32_MAX);

    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);
    ASSERT_NOT_NULL(sensor);
    ASSERT_NOT_NULL(sensor->impl);

    /* The impl direction should be normalized */
    ToFImpl* impl = (ToFImpl*)sensor->impl;
    float len = sqrtf(impl->direction.x * impl->direction.x +
                      impl->direction.y * impl->direction.y +
                      impl->direction.z * impl->direction.z);
    ASSERT_FLOAT_NEAR(len, 1.0f, 1e-5f);
    ASSERT_FLOAT_NEAR(impl->direction.x, 0.6f, 1e-4f);
    ASSERT_FLOAT_NEAR(impl->direction.y, 0.8f, 1e-4f);

    teardown_test_env();
    return 0;
}

/* ============================================================================
 * LiDAR Behavioral Tests
 * ============================================================================ */

TEST(lidar_2d_no_world_returns_max_range) {
    ASSERT_EQ(setup_test_env(4), 0);

    float max_range = 35.0f;
    SensorConfig config = sensor_config_lidar_2d(16, 3.14159f, max_range);
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    /* All rays should return max_range when no world is present */
    for (uint32_t i = 0; i < 16; i++) {
        ASSERT_FLOAT_NEAR(obs[i], max_range, 0.01f);
    }

    teardown_test_env();
    return 0;
}

TEST(lidar_3d_no_world_returns_max_range) {
    ASSERT_EQ(setup_test_env(4), 0);

    float max_range = 42.0f;
    SensorConfig config = sensor_config_lidar_3d(8, 4, 6.28f, 0.5f, max_range);
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    /* All 8*4=32 rays should return max_range */
    for (uint32_t i = 0; i < 32; i++) {
        ASSERT_FLOAT_NEAR(obs[i], max_range, 0.01f);
    }

    teardown_test_env();
    return 0;
}

TEST(lidar_2d_ray_count_matches_config) {
    ASSERT_EQ(setup_test_env(4), 0);

    uint32_t ray_counts[] = {1, 8, 16, 64, 128};
    for (size_t t = 0; t < sizeof(ray_counts)/sizeof(ray_counts[0]); t++) {
        /* Need fresh env for each iteration due to sensor accumulation */
        teardown_test_env();
        ASSERT_EQ(setup_test_env(4), 0);

        SensorConfig config = sensor_config_lidar_2d(ray_counts[t], 3.14159f, 20.0f);
        uint32_t idx = sensor_system_create_sensor(test_sys, &config);
        ASSERT_NE(idx, UINT32_MAX);

        Sensor* sensor = sensor_system_get_sensor(test_sys, idx);
        ASSERT_EQ(sensor->output_size, ray_counts[t]);

        uint32_t shape[4];
        uint32_t ndim = sensor->vtable->get_output_shape(sensor, shape);
        ASSERT_EQ(ndim, 1);
        ASSERT_EQ(shape[0], ray_counts[t]);
    }

    teardown_test_env();
    return 0;
}

TEST(lidar_3d_ray_count_matches_config) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_lidar_3d(32, 8, 6.28f, 0.5f, 50.0f);
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);

    ASSERT_EQ(sensor->output_size, 32 * 8);

    uint32_t shape[4];
    uint32_t ndim = sensor->vtable->get_output_shape(sensor, shape);
    ASSERT_EQ(ndim, 2);
    ASSERT_EQ(shape[0], 8);   /* vertical layers */
    ASSERT_EQ(shape[1], 32);  /* horizontal rays */

    teardown_test_env();
    return 0;
}

/* ============================================================================
 * Camera Behavioral Tests
 * ============================================================================ */

TEST(camera_depth_all_pixels_max_no_world) {
    /* Verify that every pixel in a depth camera returns 1.0 when no world present */
    ASSERT_EQ(setup_test_env(4), 0);

    uint32_t w = 16;
    uint32_t h = 12;
    SensorConfig config = sensor_config_camera(w, h, 1.57f, 100.0f);
    config.type = SENSOR_TYPE_CAMERA_DEPTH;
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    for (uint32_t p = 0; p < w * h; p++) {
        ASSERT_FLOAT_NEAR(obs[p], 1.0f, 0.01f);
    }

    teardown_test_env();
    return 0;
}

TEST(camera_depth_values_in_zero_one) {
    /* Without a world, depth is 1.0. Verify all values are in [0,1] range. */
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_camera(8, 8, 1.57f, 50.0f);
    config.type = SENSOR_TYPE_CAMERA_DEPTH;
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    for (uint32_t p = 0; p < 64; p++) {
        ASSERT_GE(obs[p], 0.0f);
        ASSERT_LE(obs[p], 1.0f);
    }

    teardown_test_env();
    return 0;
}

TEST(camera_seg_all_pixels_zero_no_world) {
    /* All segmentation pixels should be 0 (sky/air) when no world is present */
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_camera(16, 16, 1.57f, 100.0f);
    config.type = SENSOR_TYPE_CAMERA_SEGMENTATION;
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    for (uint32_t p = 0; p < 256; p++) {
        ASSERT_FLOAT_EQ(obs[p], 0.0f);
    }

    teardown_test_env();
    return 0;
}

TEST(camera_seg_values_are_integers) {
    /* Segmentation outputs should be integer-valued floats (material IDs) */
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_camera(8, 8, 1.57f, 50.0f);
    config.type = SENSOR_TYPE_CAMERA_SEGMENTATION;
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    for (uint32_t p = 0; p < 64; p++) {
        /* Each value should be an integer (floor == value) */
        ASSERT_FLOAT_EQ(obs[p], floorf(obs[p]));
        /* And non-negative */
        ASSERT_GE(obs[p], 0.0f);
    }

    teardown_test_env();
    return 0;
}

TEST(camera_rgb_all_pixels_sky_no_world) {
    /* Verify every pixel is sky color when no world is present */
    ASSERT_EQ(setup_test_env(4), 0);

    uint32_t w = 8;
    uint32_t h = 8;
    SensorConfig config = sensor_config_camera(w, h, 1.57f, 100.0f);
    config.type = SENSOR_TYPE_CAMERA_RGB;
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    /* Sky color: (0.5, 0.7, 0.9) for all 64 pixels */
    for (uint32_t p = 0; p < w * h; p++) {
        ASSERT_FLOAT_NEAR(obs[p * 3 + 0], 0.5f, 0.01f);
        ASSERT_FLOAT_NEAR(obs[p * 3 + 1], 0.7f, 0.01f);
        ASSERT_FLOAT_NEAR(obs[p * 3 + 2], 0.9f, 0.01f);
    }

    teardown_test_env();
    return 0;
}

TEST(camera_rgb_values_in_zero_one) {
    /* All RGB values should be in [0,1] range */
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_camera(8, 8, 1.57f, 100.0f);
    config.type = SENSOR_TYPE_CAMERA_RGB;
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    for (uint32_t i = 0; i < 8 * 8 * 3; i++) {
        ASSERT_GE(obs[i], 0.0f);
        ASSERT_LE(obs[i], 1.0f);
    }

    teardown_test_env();
    return 0;
}

TEST(camera_depth_shape_is_2d) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_camera(32, 24, 1.57f, 100.0f);
    config.type = SENSOR_TYPE_CAMERA_DEPTH;
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);

    uint32_t shape[4];
    uint32_t ndim = sensor->vtable->get_output_shape(sensor, shape);

    ASSERT_EQ(ndim, 2);
    ASSERT_EQ(shape[0], 24); /* height */
    ASSERT_EQ(shape[1], 32); /* width */

    teardown_test_env();
    return 0;
}

TEST(camera_seg_shape_is_2d) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_camera(16, 12, 1.57f, 100.0f);
    config.type = SENSOR_TYPE_CAMERA_SEGMENTATION;
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);

    uint32_t shape[4];
    uint32_t ndim = sensor->vtable->get_output_shape(sensor, shape);

    ASSERT_EQ(ndim, 2);
    ASSERT_EQ(shape[0], 12); /* height */
    ASSERT_EQ(shape[1], 16); /* width */

    teardown_test_env();
    return 0;
}

/* ============================================================================
 * Neighbor Sensor Edge Case Tests
 * ============================================================================ */

TEST(neighbor_no_neighbors_in_range) {
    /* Place all drones very far apart so none are within max_range */
    ASSERT_EQ(setup_test_env(4), 0);

    for (uint32_t i = 0; i < 4; i++) {
        test_drones->rigid_body.pos_x[i] = (float)i * 1000.0f; /* 1000m apart */
        test_drones->rigid_body.pos_y[i] = 0.0f;
        test_drones->rigid_body.pos_z[i] = 0.0f;
    }

    SensorConfig config = sensor_config_neighbor(3, 5.0f); /* 5m range */
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    /* All neighbor slots should have default values: dx=0, dy=0, dz=0, dist=max_range */
    for (uint32_t n = 0; n < 3; n++) {
        ASSERT_FLOAT_EQ(obs[n * 4 + 0], 0.0f);  /* dx */
        ASSERT_FLOAT_EQ(obs[n * 4 + 1], 0.0f);  /* dy */
        ASSERT_FLOAT_EQ(obs[n * 4 + 2], 0.0f);  /* dz */
        ASSERT_FLOAT_NEAR(obs[n * 4 + 3], 5.0f, 0.01f);  /* dist = max_range */
    }

    teardown_test_env();
    return 0;
}

TEST(neighbor_single_drone_no_neighbors) {
    /* With only 1 drone, there should be no neighbors found */
    ASSERT_EQ(setup_test_env(1), 0);

    test_drones->rigid_body.pos_x[0] = 0.0f;
    test_drones->rigid_body.pos_y[0] = 0.0f;
    test_drones->rigid_body.pos_z[0] = 0.0f;

    SensorConfig config = sensor_config_neighbor(3, 100.0f);
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 1);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    /* All slots: default (no neighbors) */
    for (uint32_t n = 0; n < 3; n++) {
        ASSERT_FLOAT_EQ(obs[n * 4 + 0], 0.0f);
        ASSERT_FLOAT_EQ(obs[n * 4 + 1], 0.0f);
        ASSERT_FLOAT_EQ(obs[n * 4 + 2], 0.0f);
        ASSERT_FLOAT_NEAR(obs[n * 4 + 3], 100.0f, 0.01f);
    }

    teardown_test_env();
    return 0;
}

TEST(neighbor_max_range_boundary) {
    /* Place one neighbor exactly at max_range and one just beyond */
    ASSERT_EQ(setup_test_env(4), 0);

    float max_range = 10.0f;
    /* Drone 0 at origin */
    test_drones->rigid_body.pos_x[0] = 0.0f;
    test_drones->rigid_body.pos_y[0] = 0.0f;
    test_drones->rigid_body.pos_z[0] = 0.0f;

    /* Drone 1 just inside max_range */
    test_drones->rigid_body.pos_x[1] = 9.9f;
    test_drones->rigid_body.pos_y[1] = 0.0f;
    test_drones->rigid_body.pos_z[1] = 0.0f;

    /* Drone 2 beyond max_range */
    test_drones->rigid_body.pos_x[2] = 10.1f;
    test_drones->rigid_body.pos_y[2] = 0.0f;
    test_drones->rigid_body.pos_z[2] = 0.0f;

    /* Drone 3 far away */
    test_drones->rigid_body.pos_x[3] = 100.0f;
    test_drones->rigid_body.pos_y[3] = 0.0f;
    test_drones->rigid_body.pos_z[3] = 0.0f;

    SensorConfig config = sensor_config_neighbor(3, max_range);
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    /* First neighbor should be drone 1 at distance ~9.9 */
    ASSERT_FLOAT_NEAR(obs[0], 9.9f, 0.1f);   /* dx */
    ASSERT_FLOAT_NEAR(obs[3], 9.9f, 0.1f);   /* distance */

    /* Second slot should be default (no neighbor found within range) */
    ASSERT_FLOAT_NEAR(obs[4 + 3], max_range, 0.1f);

    teardown_test_env();
    return 0;
}

TEST(neighbor_sorted_by_distance) {
    /* Verify neighbors are returned in order of increasing distance */
    ASSERT_EQ(setup_test_env(8), 0);

    /* Drone 0 at origin */
    test_drones->rigid_body.pos_x[0] = 0.0f;
    test_drones->rigid_body.pos_y[0] = 0.0f;
    test_drones->rigid_body.pos_z[0] = 0.0f;

    /* Place other drones at various distances */
    test_drones->rigid_body.pos_x[1] = 5.0f;  /* dist=5 */
    test_drones->rigid_body.pos_y[1] = 0.0f;
    test_drones->rigid_body.pos_z[1] = 0.0f;

    test_drones->rigid_body.pos_x[2] = 2.0f;  /* dist=2 */
    test_drones->rigid_body.pos_y[2] = 0.0f;
    test_drones->rigid_body.pos_z[2] = 0.0f;

    test_drones->rigid_body.pos_x[3] = 8.0f;  /* dist=8 */
    test_drones->rigid_body.pos_y[3] = 0.0f;
    test_drones->rigid_body.pos_z[3] = 0.0f;

    test_drones->rigid_body.pos_x[4] = 1.0f;  /* dist=1 */
    test_drones->rigid_body.pos_y[4] = 0.0f;
    test_drones->rigid_body.pos_z[4] = 0.0f;

    for (uint32_t i = 5; i < 8; i++) {
        test_drones->rigid_body.pos_x[i] = 100.0f;
        test_drones->rigid_body.pos_y[i] = 0.0f;
        test_drones->rigid_body.pos_z[i] = 0.0f;
    }

    SensorConfig config = sensor_config_neighbor(3, 20.0f);
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 8);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    /* Should be sorted: drone4(1m), drone2(2m), drone1(5m) */
    ASSERT_FLOAT_NEAR(obs[0 * 4 + 3], 1.0f, 0.1f);
    ASSERT_FLOAT_NEAR(obs[1 * 4 + 3], 2.0f, 0.1f);
    ASSERT_FLOAT_NEAR(obs[2 * 4 + 3], 5.0f, 0.1f);

    teardown_test_env();
    return 0;
}

TEST(neighbor_relative_position_correctness) {
    /* Verify that dx,dy,dz are correct relative positions */
    ASSERT_EQ(setup_test_env(4), 0);

    /* Drone 0 at (1, 2, 3) */
    test_drones->rigid_body.pos_x[0] = 1.0f;
    test_drones->rigid_body.pos_y[0] = 2.0f;
    test_drones->rigid_body.pos_z[0] = 3.0f;

    /* Drone 1 at (4, 6, 3) => delta = (3, 4, 0), dist = 5 */
    test_drones->rigid_body.pos_x[1] = 4.0f;
    test_drones->rigid_body.pos_y[1] = 6.0f;
    test_drones->rigid_body.pos_z[1] = 3.0f;

    /* Drone 2 and 3 far away */
    test_drones->rigid_body.pos_x[2] = 1000.0f;
    test_drones->rigid_body.pos_y[2] = 0.0f;
    test_drones->rigid_body.pos_z[2] = 0.0f;
    test_drones->rigid_body.pos_x[3] = 1000.0f;
    test_drones->rigid_body.pos_y[3] = 0.0f;
    test_drones->rigid_body.pos_z[3] = 0.0f;

    SensorConfig config = sensor_config_neighbor(2, 20.0f);
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    /* First neighbor: delta = (3, 4, 0), dist = 5 */
    ASSERT_FLOAT_NEAR(obs[0], 3.0f, 0.1f);   /* dx */
    ASSERT_FLOAT_NEAR(obs[1], 4.0f, 0.1f);   /* dy */
    ASSERT_FLOAT_NEAR(obs[2], 0.0f, 0.1f);   /* dz */
    ASSERT_FLOAT_NEAR(obs[3], 5.0f, 0.1f);   /* distance */

    teardown_test_env();
    return 0;
}

/* ============================================================================
 * Multi-Sensor Composition Tests
 * ============================================================================ */

TEST(multi_sensor_obs_dim_accumulates) {
    /* Attach multiple sensors to same drone, verify total obs_dim */
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig pos_cfg = sensor_config_position();       /* 3 floats */
    SensorConfig vel_cfg = sensor_config_velocity();       /* 6 floats */
    SensorConfig imu_cfg = sensor_config_imu();            /* 6 floats */

    uint32_t pos_idx = sensor_system_create_sensor(test_sys, &pos_cfg);
    uint32_t vel_idx = sensor_system_create_sensor(test_sys, &vel_cfg);
    uint32_t imu_idx = sensor_system_create_sensor(test_sys, &imu_cfg);

    sensor_system_attach(test_sys, 0, pos_idx);  /* offset 0, size 3 */
    sensor_system_attach(test_sys, 0, vel_idx);  /* offset 3, size 6 */
    sensor_system_attach(test_sys, 0, imu_idx);  /* offset 9, size 6 */

    /* Total obs_dim for drone 0 should be 3+6+6=15 */
    size_t obs_dim = sensor_system_compute_obs_dim(test_sys, 0);
    ASSERT_EQ(obs_dim, 15);

    /* Drone 1 has no sensors attached */
    size_t obs_dim_1 = sensor_system_compute_obs_dim(test_sys, 1);
    ASSERT_EQ(obs_dim_1, 0);

    teardown_test_env();
    return 0;
}

TEST(multi_sensor_data_non_overlapping) {
    /* Verify that multiple sensors write to non-overlapping regions
     * and each produces correct output */
    ASSERT_EQ(setup_test_env(4), 0);

    /* Setup drone 0 with known state */
    test_drones->rigid_body.pos_x[0] = 1.0f;
    test_drones->rigid_body.pos_y[0] = 2.0f;
    test_drones->rigid_body.pos_z[0] = 3.0f;
    test_drones->rigid_body.vel_x[0] = 4.0f;
    test_drones->rigid_body.vel_y[0] = 5.0f;
    test_drones->rigid_body.vel_z[0] = 6.0f;
    test_drones->rigid_body.omega_x[0] = 0.1f;
    test_drones->rigid_body.omega_y[0] = 0.2f;
    test_drones->rigid_body.omega_z[0] = 0.3f;

    SensorConfig pos_cfg = sensor_config_position();  /* 3 floats */
    SensorConfig vel_cfg = sensor_config_velocity();  /* 6 floats */

    uint32_t pos_idx = sensor_system_create_sensor(test_sys, &pos_cfg);
    uint32_t vel_idx = sensor_system_create_sensor(test_sys, &vel_cfg);

    uint32_t pos_offset = sensor_system_attach(test_sys, 0, pos_idx);
    uint32_t vel_offset = sensor_system_attach(test_sys, 0, vel_idx);

    ASSERT_EQ(pos_offset, 0);
    ASSERT_EQ(vel_offset, 3);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs = sensor_system_get_drone_obs_const(test_sys, 0);

    /* Position at offset 0 */
    ASSERT_FLOAT_NEAR(obs[0], 1.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[1], 2.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[2], 3.0f, 0.001f);

    /* Velocity at offset 3 */
    ASSERT_FLOAT_NEAR(obs[3], 4.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[4], 5.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[5], 6.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[6], 0.1f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[7], 0.2f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[8], 0.3f, 0.001f);

    teardown_test_env();
    return 0;
}

TEST(multi_sensor_different_drones_independent) {
    /* Verify that sensors on different drones produce independent observations */
    ASSERT_EQ(setup_test_env(4), 0);

    /* Give drones distinct positions */
    test_drones->rigid_body.pos_x[0] = 10.0f;
    test_drones->rigid_body.pos_y[0] = 20.0f;
    test_drones->rigid_body.pos_z[0] = 30.0f;

    test_drones->rigid_body.pos_x[1] = 100.0f;
    test_drones->rigid_body.pos_y[1] = 200.0f;
    test_drones->rigid_body.pos_z[1] = 300.0f;

    SensorConfig pos_cfg = sensor_config_position();
    uint32_t pos_idx = sensor_system_create_sensor(test_sys, &pos_cfg);

    sensor_system_attach(test_sys, 0, pos_idx);
    sensor_system_attach(test_sys, 1, pos_idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);

    const float* obs0 = sensor_system_get_drone_obs_const(test_sys, 0);
    const float* obs1 = sensor_system_get_drone_obs_const(test_sys, 1);

    /* Drone 0 */
    ASSERT_FLOAT_NEAR(obs0[0], 10.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs0[1], 20.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs0[2], 30.0f, 0.001f);

    /* Drone 1 */
    ASSERT_FLOAT_NEAR(obs1[0], 100.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs1[1], 200.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs1[2], 300.0f, 0.001f);

    teardown_test_env();
    return 0;
}

TEST(multi_sensor_max_attachment_count) {
    /* Verify we can attach up to MAX_SENSORS_PER_DRONE (8) sensors */
    ASSERT_EQ(setup_test_env(4), 0);

    /* Create 9 small sensors (position = 3 floats each, 9*3=27 < 256 max_obs_dim) */
    uint32_t sensor_ids[9];
    for (int i = 0; i < 9; i++) {
        SensorConfig cfg = sensor_config_position();
        sensor_ids[i] = sensor_system_create_sensor(test_sys, &cfg);
        ASSERT_NE(sensor_ids[i], UINT32_MAX);
    }

    /* First 8 attachments should succeed */
    for (int i = 0; i < 8; i++) {
        uint32_t offset = sensor_system_attach(test_sys, 0, sensor_ids[i]);
        ASSERT_NE(offset, UINT32_MAX);
    }

    /* 9th attachment should fail (MAX_SENSORS_PER_DRONE = 8) */
    uint32_t overflow_offset = sensor_system_attach(test_sys, 0, sensor_ids[8]);
    ASSERT_EQ(overflow_offset, UINT32_MAX);

    teardown_test_env();
    return 0;
}

/* ============================================================================
 * Determinism Tests
 * ============================================================================ */

TEST(position_deterministic_no_noise) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_position();
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    /* First sample */
    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);
    const float* obs1 = sensor_system_get_drone_obs_const(test_sys, 0);
    float v1_x = obs1[0];
    float v1_y = obs1[1];
    float v1_z = obs1[2];

    /* Second sample with same state */
    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);
    const float* obs2 = sensor_system_get_drone_obs_const(test_sys, 0);

    ASSERT_FLOAT_EQ(v1_x, obs2[0]);
    ASSERT_FLOAT_EQ(v1_y, obs2[1]);
    ASSERT_FLOAT_EQ(v1_z, obs2[2]);

    teardown_test_env();
    return 0;
}

TEST(velocity_deterministic_no_noise) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_velocity();
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);
    const float* obs1 = sensor_system_get_drone_obs_const(test_sys, 0);
    float vals1[6];
    for (int i = 0; i < 6; i++) vals1[i] = obs1[i];

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);
    const float* obs2 = sensor_system_get_drone_obs_const(test_sys, 0);

    for (int i = 0; i < 6; i++) {
        ASSERT_FLOAT_EQ(vals1[i], obs2[i]);
    }

    teardown_test_env();
    return 0;
}

TEST(tof_deterministic_no_noise) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_tof(VEC3(0, 0, -1), 20.0f);
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);
    const float* obs1 = sensor_system_get_drone_obs_const(test_sys, 0);
    float v1 = obs1[0];

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);
    const float* obs2 = sensor_system_get_drone_obs_const(test_sys, 0);

    ASSERT_FLOAT_EQ(v1, obs2[0]);

    teardown_test_env();
    return 0;
}

TEST(lidar_2d_deterministic_no_noise) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_lidar_2d(16, 3.14159f, 20.0f);
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);
    const float* obs1 = sensor_system_get_drone_obs_const(test_sys, 0);
    float vals1[16];
    for (int i = 0; i < 16; i++) vals1[i] = obs1[i];

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);
    const float* obs2 = sensor_system_get_drone_obs_const(test_sys, 0);

    for (int i = 0; i < 16; i++) {
        ASSERT_FLOAT_EQ(vals1[i], obs2[i]);
    }

    teardown_test_env();
    return 0;
}

TEST(camera_depth_deterministic_no_noise) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_camera(8, 8, 1.57f, 50.0f);
    config.type = SENSOR_TYPE_CAMERA_DEPTH;
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);
    const float* obs1 = sensor_system_get_drone_obs_const(test_sys, 0);
    float vals1[64];
    for (int i = 0; i < 64; i++) vals1[i] = obs1[i];

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 4);
    const float* obs2 = sensor_system_get_drone_obs_const(test_sys, 0);

    for (int i = 0; i < 64; i++) {
        ASSERT_FLOAT_EQ(vals1[i], obs2[i]);
    }

    teardown_test_env();
    return 0;
}

TEST(neighbor_deterministic_no_noise) {
    ASSERT_EQ(setup_test_env(8), 0);

    /* Place drones in a line */
    for (uint32_t i = 0; i < 8; i++) {
        test_drones->rigid_body.pos_x[i] = (float)i * 3.0f;
        test_drones->rigid_body.pos_y[i] = 0.0f;
        test_drones->rigid_body.pos_z[i] = 0.0f;
    }

    SensorConfig config = sensor_config_neighbor(3, 20.0f);
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    sensor_system_attach(test_sys, 0, idx);

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 8);
    const float* obs1 = sensor_system_get_drone_obs_const(test_sys, 0);
    float vals1[12];
    for (int i = 0; i < 12; i++) vals1[i] = obs1[i];

    sensor_system_sample_all(test_sys, test_drones, NULL, NULL, 8);
    const float* obs2 = sensor_system_get_drone_obs_const(test_sys, 0);

    for (int i = 0; i < 12; i++) {
        ASSERT_FLOAT_EQ(vals1[i], obs2[i]);
    }

    teardown_test_env();
    return 0;
}

/* ============================================================================
 * Config Helper Roundtrip Tests
 * ============================================================================ */

TEST(config_roundtrip_imu) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_imu();
    ASSERT_EQ(config.type, SENSOR_TYPE_IMU);

    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    ASSERT_NE(idx, UINT32_MAX);

    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);
    ASSERT_EQ(sensor->output_size, 6);
    ASSERT_EQ(sensor->type, SENSOR_TYPE_IMU);

    teardown_test_env();
    return 0;
}

TEST(config_roundtrip_tof) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_tof(VEC3(1, 0, 0), 15.0f);
    ASSERT_EQ(config.type, SENSOR_TYPE_TOF);
    ASSERT_FLOAT_NEAR(config.tof.max_range, 15.0f, 0.001f);

    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);
    ASSERT_EQ(sensor->output_size, 1);

    teardown_test_env();
    return 0;
}

TEST(config_roundtrip_lidar_2d) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_lidar_2d(128, 6.28f, 30.0f);
    ASSERT_EQ(config.type, SENSOR_TYPE_LIDAR_2D);
    ASSERT_EQ(config.lidar_2d.num_rays, 128);

    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);
    ASSERT_EQ(sensor->output_size, 128);

    teardown_test_env();
    return 0;
}

TEST(config_roundtrip_lidar_3d) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_lidar_3d(32, 8, 6.28f, 0.5f, 40.0f);
    ASSERT_EQ(config.type, SENSOR_TYPE_LIDAR_3D);
    ASSERT_EQ(config.lidar_3d.horizontal_rays, 32);
    ASSERT_EQ(config.lidar_3d.vertical_layers, 8);

    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);
    ASSERT_EQ(sensor->output_size, 32 * 8);

    teardown_test_env();
    return 0;
}

TEST(config_roundtrip_camera_rgb) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_camera(32, 24, 1.57f, 80.0f);
    config.type = SENSOR_TYPE_CAMERA_RGB;

    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);
    ASSERT_EQ(sensor->output_size, 32 * 24 * 3);

    teardown_test_env();
    return 0;
}

TEST(config_roundtrip_camera_depth) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_camera(32, 24, 1.57f, 80.0f);
    config.type = SENSOR_TYPE_CAMERA_DEPTH;

    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);
    ASSERT_EQ(sensor->output_size, 32 * 24);

    teardown_test_env();
    return 0;
}

TEST(config_roundtrip_camera_seg) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_camera(32, 24, 1.57f, 80.0f);
    config.type = SENSOR_TYPE_CAMERA_SEGMENTATION;

    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);
    ASSERT_EQ(sensor->output_size, 32 * 24);

    teardown_test_env();
    return 0;
}

TEST(config_roundtrip_position) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_position();
    ASSERT_EQ(config.type, SENSOR_TYPE_POSITION);

    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);
    ASSERT_EQ(sensor->output_size, 3);

    teardown_test_env();
    return 0;
}

TEST(config_roundtrip_velocity) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_velocity();
    ASSERT_EQ(config.type, SENSOR_TYPE_VELOCITY);

    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);
    ASSERT_EQ(sensor->output_size, 6);

    teardown_test_env();
    return 0;
}

TEST(config_roundtrip_neighbor) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_neighbor(7, 15.0f);
    ASSERT_EQ(config.type, SENSOR_TYPE_NEIGHBOR);
    ASSERT_EQ(config.neighbor.k, 7);

    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);
    ASSERT_EQ(sensor->output_size, 7 * 4);

    teardown_test_env();
    return 0;
}

TEST(config_default_sets_reasonable_values) {
    /* Verify that sensor_config_default provides sane defaults for all types */
    SensorConfig imu_cfg = sensor_config_default(SENSOR_TYPE_IMU);
    ASSERT_EQ(imu_cfg.type, SENSOR_TYPE_IMU);

    SensorConfig tof_cfg = sensor_config_default(SENSOR_TYPE_TOF);
    ASSERT_TRUE(tof_cfg.tof.max_range > 0.0f);

    SensorConfig l2d_cfg = sensor_config_default(SENSOR_TYPE_LIDAR_2D);
    ASSERT_TRUE(l2d_cfg.lidar_2d.num_rays > 0);
    ASSERT_TRUE(l2d_cfg.lidar_2d.fov > 0.0f);
    ASSERT_TRUE(l2d_cfg.lidar_2d.max_range > 0.0f);

    SensorConfig l3d_cfg = sensor_config_default(SENSOR_TYPE_LIDAR_3D);
    ASSERT_TRUE(l3d_cfg.lidar_3d.horizontal_rays > 0);
    ASSERT_TRUE(l3d_cfg.lidar_3d.vertical_layers > 0);
    ASSERT_TRUE(l3d_cfg.lidar_3d.max_range > 0.0f);

    SensorConfig cam_cfg = sensor_config_default(SENSOR_TYPE_CAMERA_RGB);
    ASSERT_TRUE(cam_cfg.camera.width > 0);
    ASSERT_TRUE(cam_cfg.camera.height > 0);
    ASSERT_TRUE(cam_cfg.camera.fov_horizontal > 0.0f);
    ASSERT_TRUE(cam_cfg.camera.far_clip > cam_cfg.camera.near_clip);

    SensorConfig nbr_cfg = sensor_config_default(SENSOR_TYPE_NEIGHBOR);
    ASSERT_TRUE(nbr_cfg.neighbor.k > 0);
    ASSERT_TRUE(nbr_cfg.neighbor.max_range > 0.0f);

    return 0;
}

/* ============================================================================
 * IMU Shape Test
 * ============================================================================ */

TEST(imu_shape) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_imu();
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);

    uint32_t shape[4];
    uint32_t ndim = sensor->vtable->get_output_shape(sensor, shape);

    ASSERT_EQ(ndim, 1);
    ASSERT_EQ(shape[0], 6);

    teardown_test_env();
    return 0;
}

/* ============================================================================
 * ToF Shape Test
 * ============================================================================ */

TEST(tof_shape) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_tof(VEC3(0, 0, -1), 10.0f);
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);

    uint32_t shape[4];
    uint32_t ndim = sensor->vtable->get_output_shape(sensor, shape);

    ASSERT_EQ(ndim, 1);
    ASSERT_EQ(shape[0], 1);

    teardown_test_env();
    return 0;
}

/* ============================================================================
 * Position / Velocity Shape Tests
 * ============================================================================ */

TEST(position_shape) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_position();
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);

    uint32_t shape[4];
    uint32_t ndim = sensor->vtable->get_output_shape(sensor, shape);

    ASSERT_EQ(ndim, 1);
    ASSERT_EQ(shape[0], 3);

    teardown_test_env();
    return 0;
}

TEST(velocity_shape) {
    ASSERT_EQ(setup_test_env(4), 0);

    SensorConfig config = sensor_config_velocity();
    uint32_t idx = sensor_system_create_sensor(test_sys, &config);
    Sensor* sensor = sensor_system_get_sensor(test_sys, idx);

    uint32_t shape[4];
    uint32_t ndim = sensor->vtable->get_output_shape(sensor, shape);

    ASSERT_EQ(ndim, 1);
    ASSERT_EQ(shape[0], 6);

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

    /* Ray Precomputation Tests */
    RUN_TEST(lidar_2d_rays_are_unit_length);
    RUN_TEST(lidar_2d_rays_lie_in_xy_plane);
    RUN_TEST(lidar_2d_rays_span_fov_symmetrically);
    RUN_TEST(lidar_2d_rays_single_ray_points_forward);
    RUN_TEST(lidar_2d_rays_null_arena_returns_null);
    RUN_TEST(lidar_2d_rays_zero_count_returns_null);
    RUN_TEST(lidar_3d_rays_total_count);
    RUN_TEST(lidar_3d_rays_vertical_spread);
    RUN_TEST(lidar_3d_rays_null_inputs);
    RUN_TEST(camera_rays_total_count_and_unit_length);
    RUN_TEST(camera_rays_center_points_forward);
    RUN_TEST(camera_rays_null_inputs);
    RUN_TEST(camera_rays_horizontal_symmetry);

    /* IMU Behavioral Tests */
    RUN_TEST(imu_rotated_drone_changes_accel);
    RUN_TEST(imu_inverted_drone_negative_z_accel);
    RUN_TEST(imu_gyro_reads_omega_directly);
    RUN_TEST(imu_zero_angular_velocity);
    RUN_TEST(imu_shape);

    /* ToF Behavioral Tests */
    RUN_TEST(tof_different_direction_vectors);
    RUN_TEST(tof_max_range_clamping);
    RUN_TEST(tof_unnormalized_direction_is_normalized);
    RUN_TEST(tof_shape);

    /* LiDAR Behavioral Tests */
    RUN_TEST(lidar_2d_no_world_returns_max_range);
    RUN_TEST(lidar_3d_no_world_returns_max_range);
    RUN_TEST(lidar_2d_ray_count_matches_config);
    RUN_TEST(lidar_3d_ray_count_matches_config);

    /* Camera Behavioral Tests */
    RUN_TEST(camera_depth_all_pixels_max_no_world);
    RUN_TEST(camera_depth_values_in_zero_one);
    RUN_TEST(camera_seg_all_pixels_zero_no_world);
    RUN_TEST(camera_seg_values_are_integers);
    RUN_TEST(camera_rgb_all_pixels_sky_no_world);
    RUN_TEST(camera_rgb_values_in_zero_one);
    RUN_TEST(camera_depth_shape_is_2d);
    RUN_TEST(camera_seg_shape_is_2d);

    /* Neighbor Edge Case Tests */
    RUN_TEST(neighbor_no_neighbors_in_range);
    RUN_TEST(neighbor_single_drone_no_neighbors);
    RUN_TEST(neighbor_max_range_boundary);
    RUN_TEST(neighbor_sorted_by_distance);
    RUN_TEST(neighbor_relative_position_correctness);

    /* Multi-Sensor Composition Tests */
    RUN_TEST(multi_sensor_obs_dim_accumulates);
    RUN_TEST(multi_sensor_data_non_overlapping);
    RUN_TEST(multi_sensor_different_drones_independent);
    RUN_TEST(multi_sensor_max_attachment_count);

    /* Determinism Tests */
    RUN_TEST(position_deterministic_no_noise);
    RUN_TEST(velocity_deterministic_no_noise);
    RUN_TEST(tof_deterministic_no_noise);
    RUN_TEST(lidar_2d_deterministic_no_noise);
    RUN_TEST(camera_depth_deterministic_no_noise);
    RUN_TEST(neighbor_deterministic_no_noise);

    /* Config Helper Roundtrip Tests */
    RUN_TEST(config_roundtrip_imu);
    RUN_TEST(config_roundtrip_tof);
    RUN_TEST(config_roundtrip_lidar_2d);
    RUN_TEST(config_roundtrip_lidar_3d);
    RUN_TEST(config_roundtrip_camera_rgb);
    RUN_TEST(config_roundtrip_camera_depth);
    RUN_TEST(config_roundtrip_camera_seg);
    RUN_TEST(config_roundtrip_position);
    RUN_TEST(config_roundtrip_velocity);
    RUN_TEST(config_roundtrip_neighbor);
    RUN_TEST(config_default_sets_reasonable_values);

    /* Shape Tests */
    RUN_TEST(position_shape);
    RUN_TEST(velocity_shape);

    TEST_SUITE_END();
}
