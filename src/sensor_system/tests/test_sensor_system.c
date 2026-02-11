/**
 * Sensor System Unit Tests
 *
 * Tests for the core sensor system functionality following Yoneda philosophy:
 * tests as behavioral specifications.
 *
 * Test Categories:
 * - Allocation Tests: system creation, alignment, capacity
 * - Registry Tests: vtable registration and lookup
 * - Sensor Management: creation, attachment, detachment
 * - Batch Processing: grouping, dispatch, scatter
 * - Observation Buffer: access, dimensions, offsets
 * - Reset/Lifecycle: cleanup, state reset
 */

#include "sensor_system.h"
#include "sensor_implementations.h"
#include "test_harness.h"

/* ============================================================================
 * Mock VTable for Testing
 * ============================================================================ */

static uint32_t mock_batch_sample_calls = 0;
static uint32_t mock_last_drone_count = 0;

static void mock_init(Sensor* sensor, const SensorConfig* config, Arena* arena) {
    (void)config; (void)arena;
    sensor->impl = NULL;
}

static size_t mock_get_output_size(const Sensor* sensor) {
    (void)sensor;
    return 4;  /* 4 floats per output */
}

static const char* mock_get_output_dtype(const Sensor* sensor) {
    (void)sensor;
    return "float32";
}

static uint32_t mock_get_output_shape(const Sensor* sensor, uint32_t* shape) {
    (void)sensor;
    shape[0] = 4;
    return 1;
}

static void mock_batch_sample(Sensor* sensor, const SensorContext* ctx, float* output_buffer) {
    (void)sensor;
    mock_batch_sample_calls++;
    mock_last_drone_count = ctx->drone_count;

    /* Fill output with known pattern: drone_idx * 10 + channel */
    for (uint32_t i = 0; i < ctx->drone_count; i++) {
        uint32_t d = ctx->drone_indices[i];
        for (uint32_t c = 0; c < 4; c++) {
            output_buffer[i * 4 + c] = (float)(d * 10 + c);
        }
    }
}

static void mock_reset(Sensor* sensor, uint32_t drone_index) {
    (void)sensor; (void)drone_index;
}

static void mock_destroy(Sensor* sensor) {
    (void)sensor;
}

static const SensorVTable MOCK_VTABLE = {
    .name = "Mock",
    .type = SENSOR_TYPE_IMU,  /* Reuse IMU type for testing */
    .init = mock_init,
    .get_output_size = mock_get_output_size,
    .get_output_dtype = mock_get_output_dtype,
    .get_output_shape = mock_get_output_shape,
    .batch_sample = mock_batch_sample,
    .reset = mock_reset,
    .destroy = mock_destroy
};

/* ============================================================================
 * Allocation Tests
 * ============================================================================ */

TEST(system_create_basic) {
    Arena* arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    SensorSystem* sys = sensor_system_create(arena, 64, 8, 32);
    ASSERT_NOT_NULL(sys);
    ASSERT_EQ(sys->max_drones, 64);
    ASSERT_EQ(sys->max_sensors, 8);
    ASSERT_EQ(sys->obs_dim, 32);
    ASSERT_EQ(sys->sensor_count, 0);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(system_create_null_arena) {
    SensorSystem* sys = sensor_system_create(NULL, 64, 8, 32);
    ASSERT_NULL(sys);
    return 0;
}

TEST(system_create_zero_params) {
    Arena* arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    SensorSystem* sys = sensor_system_create(arena, 0, 8, 32);
    ASSERT_NULL(sys);

    sys = sensor_system_create(arena, 64, 0, 32);
    ASSERT_NULL(sys);

    sys = sensor_system_create(arena, 64, 8, 0);
    ASSERT_NULL(sys);

    arena_destroy(arena);
    return 0;
}

TEST(observation_buffer_alignment) {
    Arena* arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    SensorSystem* sys = sensor_system_create(arena, 64, 8, 32);
    ASSERT_NOT_NULL(sys);

    float* obs = sensor_system_get_observations(sys);
    ASSERT_NOT_NULL(obs);

    /* Check 32-byte alignment */
    uintptr_t addr = (uintptr_t)obs;
    ASSERT_EQ(addr % 32, 0);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Registry Tests
 * ============================================================================ */

TEST(registry_init) {
    SensorRegistry registry;
    memset(&registry, 0, sizeof(registry));

    sensor_registry_init(&registry);
    ASSERT_TRUE(registry.initialized);

    /* All vtables should be NULL after init (before registration) */
    for (int i = 0; i < SENSOR_TYPE_COUNT; i++) {
        ASSERT_NULL(registry.vtables[i]);
    }
    return 0;
}

TEST(registry_register_and_get) {
    SensorRegistry registry;
    sensor_registry_init(&registry);

    sensor_registry_register(&registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);

    const SensorVTable* vtable = sensor_registry_get(&registry, SENSOR_TYPE_IMU);
    ASSERT_NOT_NULL(vtable);
    ASSERT_EQ(vtable, &MOCK_VTABLE);
    return 0;
}

TEST(registry_get_unregistered) {
    SensorRegistry registry;
    sensor_registry_init(&registry);

    const SensorVTable* vtable = sensor_registry_get(&registry, SENSOR_TYPE_TOF);
    ASSERT_NULL(vtable);
    return 0;
}

TEST(registry_get_invalid_type) {
    SensorRegistry registry;
    sensor_registry_init(&registry);

    const SensorVTable* vtable = sensor_registry_get(&registry, SENSOR_TYPE_COUNT);
    ASSERT_NULL(vtable);

    vtable = sensor_registry_get(&registry, (SensorType)100);
    ASSERT_NULL(vtable);
    return 0;
}

/* ============================================================================
 * Sensor Management Tests
 * ============================================================================ */

TEST(sensor_create_basic) {
    Arena* arena = arena_create(1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 64, 8, 32);
    ASSERT_NOT_NULL(sys);

    /* Register mock vtable */
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);
    uint32_t idx = sensor_system_create_sensor(sys, &config);

    ASSERT_NE(idx, UINT32_MAX);
    ASSERT_EQ(idx, 0);
    ASSERT_EQ(sys->sensor_count, 1);

    Sensor* sensor = sensor_system_get_sensor(sys, idx);
    ASSERT_NOT_NULL(sensor);
    ASSERT_EQ(sensor->vtable, &MOCK_VTABLE);
    ASSERT_EQ(sensor->type, SENSOR_TYPE_IMU);
    ASSERT_EQ(sensor->output_size, 4);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(sensor_create_multiple) {
    Arena* arena = arena_create(1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 64, 8, 32);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);

    uint32_t idx0 = sensor_system_create_sensor(sys, &config);
    uint32_t idx1 = sensor_system_create_sensor(sys, &config);
    uint32_t idx2 = sensor_system_create_sensor(sys, &config);

    ASSERT_EQ(idx0, 0);
    ASSERT_EQ(idx1, 1);
    ASSERT_EQ(idx2, 2);
    ASSERT_EQ(sys->sensor_count, 3);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(sensor_create_max_capacity) {
    Arena* arena = arena_create(1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 64, 4, 32);  /* Max 4 sensors */
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);

    for (int i = 0; i < 4; i++) {
        uint32_t idx = sensor_system_create_sensor(sys, &config);
        ASSERT_NE(idx, UINT32_MAX);
    }

    uint32_t idx = sensor_system_create_sensor(sys, &config);
    ASSERT_EQ(idx, UINT32_MAX);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(sensor_get_invalid) {
    Arena* arena = arena_create(1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 64, 8, 32);

    Sensor* sensor = sensor_system_get_sensor(sys, 0);
    ASSERT_NULL(sensor);

    sensor = sensor_system_get_sensor(sys, 100);
    ASSERT_NULL(sensor);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Attachment Tests
 * ============================================================================ */

TEST(attach_single) {
    Arena* arena = arena_create(1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 64, 8, 32);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);
    uint32_t sensor_idx = sensor_system_create_sensor(sys, &config);

    uint32_t offset = sensor_system_attach(sys, 0, sensor_idx);
    ASSERT_NE(offset, UINT32_MAX);
    ASSERT_EQ(offset, 0);
    ASSERT_EQ(sensor_system_get_attachment_count(sys, 0), 1);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(attach_multiple_per_drone) {
    Arena* arena = arena_create(1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 64, 8, 64);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);
    uint32_t s0 = sensor_system_create_sensor(sys, &config);
    uint32_t s1 = sensor_system_create_sensor(sys, &config);

    uint32_t off0 = sensor_system_attach(sys, 0, s0);
    uint32_t off1 = sensor_system_attach(sys, 0, s1);

    ASSERT_EQ(off0, 0);
    ASSERT_EQ(off1, 4);
    ASSERT_EQ(sensor_system_get_attachment_count(sys, 0), 2);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(attach_shared_sensor) {
    Arena* arena = arena_create(1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 64, 8, 32);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);
    uint32_t sensor_idx = sensor_system_create_sensor(sys, &config);

    /* Attach same sensor to multiple drones */
    uint32_t off0 = sensor_system_attach(sys, 0, sensor_idx);
    uint32_t off1 = sensor_system_attach(sys, 1, sensor_idx);
    uint32_t off2 = sensor_system_attach(sys, 2, sensor_idx);

    ASSERT_EQ(off0, 0);
    ASSERT_EQ(off1, 0);
    ASSERT_EQ(off2, 0);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(attach_max_per_drone) {
    Arena* arena = arena_create(1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 64, 16, 256);  /* Large obs_dim */
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);

    /* Create MAX_SENSORS_PER_DRONE + 1 sensors */
    for (int i = 0; i < MAX_SENSORS_PER_DRONE + 1; i++) {
        sensor_system_create_sensor(sys, &config);
    }

    /* Attach MAX_SENSORS_PER_DRONE sensors - should all succeed */
    for (int i = 0; i < MAX_SENSORS_PER_DRONE; i++) {
        uint32_t offset = sensor_system_attach(sys, 0, (uint32_t)i);
        ASSERT_NE(offset, UINT32_MAX);
    }

    /* One more should fail */
    uint32_t offset = sensor_system_attach(sys, 0, MAX_SENSORS_PER_DRONE);
    ASSERT_EQ(offset, UINT32_MAX);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(detach_basic) {
    Arena* arena = arena_create(1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 64, 8, 64);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);
    uint32_t s0 = sensor_system_create_sensor(sys, &config);
    uint32_t s1 = sensor_system_create_sensor(sys, &config);

    sensor_system_attach(sys, 0, s0);
    sensor_system_attach(sys, 0, s1);
    ASSERT_EQ(sensor_system_get_attachment_count(sys, 0), 2);

    sensor_system_detach(sys, 0, 0);
    ASSERT_EQ(sensor_system_get_attachment_count(sys, 0), 1);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Batch Processing Tests
 * ============================================================================ */

TEST(batch_sample_single_call_per_sensor) {
    Arena* arena = arena_create(1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 64, 8, 32);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);
    uint32_t sensor_idx = sensor_system_create_sensor(sys, &config);

    /* Attach to 10 drones */
    for (uint32_t i = 0; i < 10; i++) {
        sensor_system_attach(sys, i, sensor_idx);
    }

    /* Create mock drone state */
    DroneStateSOA* drones = drone_state_create(arena, 64);
    for (uint32_t i = 0; i < 10; i++) {
        drone_state_init(drones, i);
    }
    drones->count = 10;

    mock_batch_sample_calls = 0;
    mock_last_drone_count = 0;

    sensor_system_sample_all(sys, drones, NULL, NULL, 10);

    ASSERT_EQ(mock_batch_sample_calls, 1);
    ASSERT_EQ(mock_last_drone_count, 10);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(batch_sample_writes_correct_output) {
    Arena* arena = arena_create(1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 64, 8, 32);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);
    uint32_t sensor_idx = sensor_system_create_sensor(sys, &config);

    /* Attach to drones 0, 5, 10 */
    sensor_system_attach(sys, 0, sensor_idx);
    sensor_system_attach(sys, 5, sensor_idx);
    sensor_system_attach(sys, 10, sensor_idx);

    DroneStateSOA* drones = drone_state_create(arena, 64);
    for (uint32_t i = 0; i < 11; i++) {
        drone_state_init(drones, i);
    }
    drones->count = 11;

    sensor_system_sample_all(sys, drones, NULL, NULL, 11);

    /* Check observations were scattered correctly */
    const float* obs0 = sensor_system_get_drone_obs_const(sys, 0);
    const float* obs5 = sensor_system_get_drone_obs_const(sys, 5);
    const float* obs10 = sensor_system_get_drone_obs_const(sys, 10);

    /* Mock outputs: drone_idx * 10 + channel */
    ASSERT_FLOAT_NEAR(obs0[0], 0.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs5[0], 50.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs10[0], 100.0f, 0.001f);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Observation Buffer Tests
 * ============================================================================ */

TEST(get_observations_returns_buffer) {
    Arena* arena = arena_create(1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 64, 8, 32);

    float* obs = sensor_system_get_observations(sys);
    ASSERT_NOT_NULL(obs);

    const float* obs_const = sensor_system_get_observations_const(sys);
    ASSERT_EQ(obs, obs_const);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(get_obs_dim) {
    Arena* arena = arena_create(1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 64, 8, 128);

    size_t dim = sensor_system_get_obs_dim(sys);
    ASSERT_EQ(dim, 128);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(get_drone_obs_offset) {
    Arena* arena = arena_create(1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 64, 8, 32);

    float* base = sensor_system_get_observations(sys);
    float* drone0 = sensor_system_get_drone_obs(sys, 0);
    float* drone5 = sensor_system_get_drone_obs(sys, 5);

    ASSERT_EQ(drone0, base);
    ASSERT_EQ(drone5, base + 5 * 32);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Reset Tests
 * ============================================================================ */

TEST(reset_clears_buffer) {
    Arena* arena = arena_create(1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 64, 8, 32);

    /* Write some data */
    float* obs = sensor_system_get_observations(sys);
    for (size_t i = 0; i < 64 * 32; i++) {
        obs[i] = 1.0f;
    }

    sensor_system_reset(sys);

    /* Check buffer is zeroed */
    for (size_t i = 0; i < 64 * 32; i++) {
        ASSERT_FLOAT_NEAR(obs[i], 0.0f, 0.001f);
    }

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Configuration Helper Tests
 * ============================================================================ */

TEST(config_default_imu) {
    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);
    ASSERT_EQ(config.type, SENSOR_TYPE_IMU);
    /* Noise is now configured via NoiseConfig, not per-sensor params */
    ASSERT_EQ(config.noise_config.group_count, 0u);
    return 0;
}

TEST(config_helper_lidar_2d) {
    SensorConfig config = sensor_config_lidar_2d(128, 3.14159f, 50.0f);
    ASSERT_EQ(config.type, SENSOR_TYPE_LIDAR_2D);
    ASSERT_EQ(config.lidar_2d.num_rays, 128);
    ASSERT_FLOAT_NEAR(config.lidar_2d.fov, 3.14159f, 0.001f);
    ASSERT_FLOAT_NEAR(config.lidar_2d.max_range, 50.0f, 0.001f);
    return 0;
}

TEST(config_helper_camera) {
    SensorConfig config = sensor_config_camera(128, 128, 1.5708f, 100.0f);
    ASSERT_EQ(config.type, SENSOR_TYPE_CAMERA_RGB);
    ASSERT_EQ(config.camera.width, 128);
    ASSERT_EQ(config.camera.height, 128);
    ASSERT_FLOAT_NEAR(config.camera.fov_horizontal, 1.5708f, 0.001f);
    ASSERT_FLOAT_NEAR(config.camera.far_clip, 100.0f, 0.001f);
    return 0;
}

/* ============================================================================
 * Utility Tests
 * ============================================================================ */

TEST(sensor_type_name) {
    ASSERT_STR_EQ(sensor_type_name(SENSOR_TYPE_IMU), "IMU");
    ASSERT_STR_EQ(sensor_type_name(SENSOR_TYPE_TOF), "ToF");
    ASSERT_STR_EQ(sensor_type_name(SENSOR_TYPE_LIDAR_2D), "LiDAR-2D");
    ASSERT_STR_EQ(sensor_type_name((SensorType)100), "Unknown");
    return 0;
}

TEST(memory_size_calculation) {
    size_t size = sensor_system_memory_size(1024, 64, 128);
    ASSERT_GT(size, (size_t)0);
    /* Should be roughly 873 KB according to spec */
    ASSERT_GT(size, (size_t)(800 * 1024));
    ASSERT_LT(size, (size_t)(1024 * 1024));
    return 0;
}

/* ============================================================================
 * Main Test Runner
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Sensor System");

    /* Allocation Tests */
    RUN_TEST(system_create_basic);
    RUN_TEST(system_create_null_arena);
    RUN_TEST(system_create_zero_params);
    RUN_TEST(observation_buffer_alignment);

    /* Registry Tests */
    RUN_TEST(registry_init);
    RUN_TEST(registry_register_and_get);
    RUN_TEST(registry_get_unregistered);
    RUN_TEST(registry_get_invalid_type);

    /* Sensor Management Tests */
    RUN_TEST(sensor_create_basic);
    RUN_TEST(sensor_create_multiple);
    RUN_TEST(sensor_create_max_capacity);
    RUN_TEST(sensor_get_invalid);

    /* Attachment Tests */
    RUN_TEST(attach_single);
    RUN_TEST(attach_multiple_per_drone);
    RUN_TEST(attach_shared_sensor);
    RUN_TEST(attach_max_per_drone);
    RUN_TEST(detach_basic);

    /* Batch Processing Tests */
    RUN_TEST(batch_sample_single_call_per_sensor);
    RUN_TEST(batch_sample_writes_correct_output);

    /* Observation Buffer Tests */
    RUN_TEST(get_observations_returns_buffer);
    RUN_TEST(get_obs_dim);
    RUN_TEST(get_drone_obs_offset);

    /* Reset Tests */
    RUN_TEST(reset_clears_buffer);

    /* Configuration Helper Tests */
    RUN_TEST(config_default_imu);
    RUN_TEST(config_helper_lidar_2d);
    RUN_TEST(config_helper_camera);

    /* Utility Tests */
    RUN_TEST(sensor_type_name);
    RUN_TEST(memory_size_calculation);

    TEST_SUITE_END();
}
