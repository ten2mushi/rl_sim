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
#include "drone_state.h"
#include "platform_quadcopter.h"
#include "test_harness.h"

/* ============================================================================
 * Mock VTable for Testing
 * ============================================================================ */

static uint32_t mock_batch_sample_calls = 0;
static uint32_t mock_last_agent_count = 0;

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
    mock_last_agent_count = ctx->agent_count;

    /* Fill output with known pattern: agent_idx * 10 + channel */
    for (uint32_t i = 0; i < ctx->agent_count; i++) {
        uint32_t d = ctx->agent_indices[i];
        for (uint32_t c = 0; c < 4; c++) {
            output_buffer[i * 4 + c] = (float)(d * 10 + c);
        }
    }
}

static void mock_reset(Sensor* sensor, uint32_t agent_index) {
    (void)sensor; (void)agent_index;
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
    ASSERT_EQ(sys->max_agents, 64);
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
    PlatformStateSOA* drones = platform_state_create(arena, 64, QUAD_STATE_EXT_COUNT);
    for (uint32_t i = 0; i < 10; i++) {
        platform_state_init(drones, i);
    }
    drones->rigid_body.count = 10;

    mock_batch_sample_calls = 0;
    mock_last_agent_count = 0;

    sensor_system_sample_all(sys, drones, NULL, NULL, 10);

    ASSERT_EQ(mock_batch_sample_calls, 1);
    ASSERT_EQ(mock_last_agent_count, 10);

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

    PlatformStateSOA* drones = platform_state_create(arena, 64, QUAD_STATE_EXT_COUNT);
    for (uint32_t i = 0; i < 11; i++) {
        platform_state_init(drones, i);
    }
    drones->rigid_body.count = 11;

    sensor_system_sample_all(sys, drones, NULL, NULL, 11);

    /* Check observations were scattered correctly */
    const float* obs0 = sensor_system_get_drone_obs_const(sys, 0);
    const float* obs5 = sensor_system_get_drone_obs_const(sys, 5);
    const float* obs10 = sensor_system_get_drone_obs_const(sys, 10);

    /* Mock outputs: agent_idx * 10 + channel */
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
 * Additional Mock VTables for Extended Tests
 * ============================================================================ */

/* Track reset calls per drone index */
static uint32_t mock_reset_calls = 0;
static uint32_t mock_reset_last_drone = UINT32_MAX;

static void mock_reset_tracking(Sensor* sensor, uint32_t agent_index) {
    (void)sensor;
    mock_reset_calls++;
    mock_reset_last_drone = agent_index;
}

/* Track destroy calls */
static uint32_t mock_destroy_calls = 0;

static void mock_destroy_tracking(Sensor* sensor) {
    (void)sensor;
    mock_destroy_calls++;
}

/* A second mock with output_size=3, used for multi-sensor offset tests */
static size_t mock_get_output_size_3(const Sensor* sensor) {
    (void)sensor;
    return 3;
}

static void mock_batch_sample_3(Sensor* sensor, const SensorContext* ctx, float* output_buffer) {
    (void)sensor;
    for (uint32_t i = 0; i < ctx->agent_count; i++) {
        uint32_t d = ctx->agent_indices[i];
        for (uint32_t c = 0; c < 3; c++) {
            output_buffer[i * 3 + c] = (float)(d * 100 + c);
        }
    }
}

static uint32_t mock_get_output_shape_3(const Sensor* sensor, uint32_t* shape) {
    (void)sensor;
    shape[0] = 3;
    return 1;
}

static const SensorVTable MOCK_VTABLE_3 = {
    .name = "Mock3",
    .type = SENSOR_TYPE_TOF,  /* Reuse ToF type for a second mock */
    .init = mock_init,
    .get_output_size = mock_get_output_size_3,
    .get_output_dtype = mock_get_output_dtype,
    .get_output_shape = mock_get_output_shape_3,
    .batch_sample = mock_batch_sample_3,
    .batch_sample_gpu = NULL,
    .reset = mock_reset,
    .destroy = mock_destroy
};

/* A mock vtable that claims GPU support (batch_sample_gpu != NULL) */
static uint32_t mock_gpu_sample_calls = 0;

static int32_t mock_batch_sample_gpu(Sensor* sensor, const SensorContext* ctx, float* output_buffer) {
    (void)sensor; (void)ctx; (void)output_buffer;
    mock_gpu_sample_calls++;
    return 0;
}

static uint32_t mock_cpu_only_batch_calls = 0;

static void mock_batch_sample_cpu_tracking(Sensor* sensor, const SensorContext* ctx, float* output_buffer) {
    (void)sensor;
    mock_cpu_only_batch_calls++;
    for (uint32_t i = 0; i < ctx->agent_count; i++) {
        uint32_t d = ctx->agent_indices[i];
        for (uint32_t c = 0; c < 4; c++) {
            output_buffer[i * 4 + c] = (float)(d * 10 + c);
        }
    }
}

static const SensorVTable MOCK_VTABLE_GPU = {
    .name = "MockGPU",
    .type = SENSOR_TYPE_CAMERA_RGB,
    .init = mock_init,
    .get_output_size = mock_get_output_size,
    .get_output_dtype = mock_get_output_dtype,
    .get_output_shape = mock_get_output_shape,
    .batch_sample = mock_batch_sample_cpu_tracking,
    .batch_sample_gpu = mock_batch_sample_gpu,  /* Non-NULL => GPU-capable */
    .reset = mock_reset,
    .destroy = mock_destroy
};

/* A CPU-only vtable with tracking (batch_sample_gpu == NULL) */
static const SensorVTable MOCK_VTABLE_CPU_ONLY = {
    .name = "MockCPU",
    .type = SENSOR_TYPE_POSITION,
    .init = mock_init,
    .get_output_size = mock_get_output_size,
    .get_output_dtype = mock_get_output_dtype,
    .get_output_shape = mock_get_output_shape,
    .batch_sample = mock_batch_sample_cpu_tracking,
    .batch_sample_gpu = NULL,  /* NULL => CPU only */
    .reset = mock_reset,
    .destroy = mock_destroy
};

/* A vtable with tracking reset and destroy */
static const SensorVTable MOCK_VTABLE_LIFECYCLE = {
    .name = "MockLifecycle",
    .type = SENSOR_TYPE_VELOCITY,
    .init = mock_init,
    .get_output_size = mock_get_output_size,
    .get_output_dtype = mock_get_output_dtype,
    .get_output_shape = mock_get_output_shape,
    .batch_sample = mock_batch_sample,
    .batch_sample_gpu = NULL,
    .reset = mock_reset_tracking,
    .destroy = mock_destroy_tracking
};

/* ============================================================================
 * External Buffer Tests
 * ============================================================================ */

TEST(external_buffer_redirects_observation_storage) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 16, 8, 32);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);
    uint32_t sensor_idx = sensor_system_create_sensor(sys, &config);

    sensor_system_attach(sys, 0, sensor_idx);
    sensor_system_attach(sys, 3, sensor_idx);

    /* Create an external buffer large enough for 16 drones * 32 obs_dim */
    float external_buf[16 * 32];
    memset(external_buf, 0, sizeof(external_buf));

    /* Save original internal buffer pointer for comparison */
    float* original_buf = sensor_system_get_observations(sys);
    ASSERT_NOT_NULL(original_buf);

    /* Redirect to external buffer */
    sensor_system_set_external_buffer(sys, external_buf);

    /* Verify the pointer changed */
    float* current_buf = sensor_system_get_observations(sys);
    ASSERT_EQ(current_buf, external_buf);
    ASSERT_NE(current_buf, original_buf);

    /* Sample and verify data lands in external buffer */
    PlatformStateSOA* drones = platform_state_create(arena, 16, QUAD_STATE_EXT_COUNT);
    for (uint32_t i = 0; i < 4; i++) {
        platform_state_init(drones, i);
    }
    drones->rigid_body.count = 4;

    sensor_system_sample_all(sys, drones, NULL, NULL, 4);

    /* Drone 0: obs at external_buf[0..31], sensor at offset 0, pattern d*10+c */
    ASSERT_FLOAT_NEAR(external_buf[0 * 32 + 0], 0.0f, 0.001f);  /* 0*10+0 */
    ASSERT_FLOAT_NEAR(external_buf[0 * 32 + 1], 1.0f, 0.001f);  /* 0*10+1 */
    ASSERT_FLOAT_NEAR(external_buf[0 * 32 + 2], 2.0f, 0.001f);  /* 0*10+2 */
    ASSERT_FLOAT_NEAR(external_buf[0 * 32 + 3], 3.0f, 0.001f);  /* 0*10+3 */

    /* Drone 3: obs at external_buf[3*32..], sensor at offset 0, pattern d*10+c */
    ASSERT_FLOAT_NEAR(external_buf[3 * 32 + 0], 30.0f, 0.001f);  /* 3*10+0 */
    ASSERT_FLOAT_NEAR(external_buf[3 * 32 + 1], 31.0f, 0.001f);  /* 3*10+1 */
    ASSERT_FLOAT_NEAR(external_buf[3 * 32 + 2], 32.0f, 0.001f);  /* 3*10+2 */
    ASSERT_FLOAT_NEAR(external_buf[3 * 32 + 3], 33.0f, 0.001f);  /* 3*10+3 */

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(external_buffer_set_null_is_noop) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 16, 8, 32);

    float* original_buf = sensor_system_get_observations(sys);

    /* Passing NULL buffer should not change the pointer */
    sensor_system_set_external_buffer(sys, NULL);
    float* after = sensor_system_get_observations(sys);
    ASSERT_EQ(after, original_buf);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(external_buffer_does_not_change_obs_dim) {
    /*
     * set_external_buffer only changes the pointer, NOT obs_dim.
     * This test documents that obs_dim must be updated separately if needed
     * (as noted in CLAUDE.md re: sensor_system obs_dim vs engine obs_dim).
     */
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 16, 8, 64);

    ASSERT_EQ(sensor_system_get_obs_dim(sys), 64);

    float external_buf[16 * 32];
    sensor_system_set_external_buffer(sys, external_buf);

    /* obs_dim stays at 64 even though external buffer may be sized differently */
    ASSERT_EQ(sensor_system_get_obs_dim(sys), 64);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(external_buffer_get_drone_obs_uses_external) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 8, 4, 16);

    float external_buf[8 * 16];
    memset(external_buf, 0, sizeof(external_buf));

    /* Put a sentinel value in the external buffer */
    external_buf[3 * 16 + 5] = 42.0f;

    sensor_system_set_external_buffer(sys, external_buf);

    /* get_drone_obs should point into external buffer */
    const float* drone3_obs = sensor_system_get_drone_obs_const(sys, 3);
    ASSERT_FLOAT_NEAR(drone3_obs[5], 42.0f, 0.001f);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * sample_cpu_only Tests
 * ============================================================================ */

TEST(sample_cpu_only_skips_gpu_sensors) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 16, 8, 64);

    /* Register GPU-capable and CPU-only vtables */
    sensor_registry_register(&sys->registry, SENSOR_TYPE_CAMERA_RGB, &MOCK_VTABLE_GPU);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_POSITION, &MOCK_VTABLE_CPU_ONLY);

    /* Create one GPU sensor and one CPU sensor */
    SensorConfig gpu_config = sensor_config_default(SENSOR_TYPE_CAMERA_RGB);
    SensorConfig cpu_config = sensor_config_default(SENSOR_TYPE_POSITION);

    uint32_t gpu_sensor = sensor_system_create_sensor(sys, &gpu_config);
    uint32_t cpu_sensor = sensor_system_create_sensor(sys, &cpu_config);

    /* Attach both to drone 0 */
    sensor_system_attach(sys, 0, gpu_sensor);
    sensor_system_attach(sys, 0, cpu_sensor);

    PlatformStateSOA* drones = platform_state_create(arena, 16, QUAD_STATE_EXT_COUNT);
    platform_state_init(drones, 0);
    drones->rigid_body.count = 1;

    mock_cpu_only_batch_calls = 0;

    sensor_system_sample_cpu_only(sys, drones, NULL, NULL, 1);

    /* CPU sensor should have been sampled exactly once (it has 1 drone).
     * GPU sensor should NOT be sampled.
     * mock_cpu_only_batch_calls is shared by both vtables' batch_sample,
     * but only the CPU sensor should have called it.
     */
    ASSERT_EQ(mock_cpu_only_batch_calls, 1);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(sample_cpu_only_processes_all_cpu_sensors) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 16, 8, 64);

    sensor_registry_register(&sys->registry, SENSOR_TYPE_POSITION, &MOCK_VTABLE_CPU_ONLY);

    /* Create 2 CPU-only sensors and attach both to 3 drones */
    SensorConfig cpu_config = sensor_config_default(SENSOR_TYPE_POSITION);
    uint32_t s0 = sensor_system_create_sensor(sys, &cpu_config);
    uint32_t s1 = sensor_system_create_sensor(sys, &cpu_config);

    for (uint32_t d = 0; d < 3; d++) {
        sensor_system_attach(sys, d, s0);
        sensor_system_attach(sys, d, s1);
    }

    PlatformStateSOA* drones = platform_state_create(arena, 16, QUAD_STATE_EXT_COUNT);
    for (uint32_t i = 0; i < 3; i++) {
        platform_state_init(drones, i);
    }
    drones->rigid_body.count = 3;

    mock_cpu_only_batch_calls = 0;

    sensor_system_sample_cpu_only(sys, drones, NULL, NULL, 3);

    /* Two sensors, each called once for its batch of 3 drones */
    ASSERT_EQ(mock_cpu_only_batch_calls, 2);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * sample_sensor Tests
 * ============================================================================ */

TEST(sample_sensor_only_samples_specified_sensor) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 16, 8, 64);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_TOF, &MOCK_VTABLE_3);

    SensorConfig config_a = sensor_config_default(SENSOR_TYPE_IMU);
    SensorConfig config_b = sensor_config_default(SENSOR_TYPE_TOF);
    uint32_t sa = sensor_system_create_sensor(sys, &config_a);
    uint32_t sb = sensor_system_create_sensor(sys, &config_b);

    /* Attach sensor A (output_size=4) and sensor B (output_size=3) to drone 0 */
    sensor_system_attach(sys, 0, sa);
    sensor_system_attach(sys, 0, sb);

    PlatformStateSOA* drones = platform_state_create(arena, 16, QUAD_STATE_EXT_COUNT);
    platform_state_init(drones, 0);
    drones->rigid_body.count = 1;

    /* Clear obs buffer first */
    sensor_system_reset(sys);

    /* Sample only sensor B (idx=1) */
    sensor_system_sample_sensor(sys, sb, drones, NULL, NULL, 1);

    const float* obs = sensor_system_get_drone_obs_const(sys, 0);

    /* Sensor A (offset 0..3) should be untouched (zeroed from reset) */
    ASSERT_FLOAT_NEAR(obs[0], 0.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[1], 0.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[2], 0.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[3], 0.0f, 0.001f);

    /* Sensor B (offset 4..6) should have pattern d*100+c for drone 0 */
    ASSERT_FLOAT_NEAR(obs[4], 0.0f, 0.001f);   /* 0*100+0 */
    ASSERT_FLOAT_NEAR(obs[5], 1.0f, 0.001f);   /* 0*100+1 */
    ASSERT_FLOAT_NEAR(obs[6], 2.0f, 0.001f);   /* 0*100+2 */

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(sample_sensor_with_invalid_index_is_noop) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 16, 8, 32);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);
    sensor_system_create_sensor(sys, &config);

    PlatformStateSOA* drones = platform_state_create(arena, 16, QUAD_STATE_EXT_COUNT);
    platform_state_init(drones, 0);
    drones->rigid_body.count = 1;

    mock_batch_sample_calls = 0;

    /* Invalid sensor index -- should be a no-op */
    sensor_system_sample_sensor(sys, 999, drones, NULL, NULL, 1);
    ASSERT_EQ(mock_batch_sample_calls, 0);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(sample_sensor_no_attached_drones_is_noop) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 16, 8, 32);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);
    uint32_t sensor_idx = sensor_system_create_sensor(sys, &config);
    /* Sensor created but NOT attached to any drone */

    PlatformStateSOA* drones = platform_state_create(arena, 16, QUAD_STATE_EXT_COUNT);
    platform_state_init(drones, 0);
    drones->rigid_body.count = 1;

    mock_batch_sample_calls = 0;

    sensor_system_sample_sensor(sys, sensor_idx, drones, NULL, NULL, 1);
    ASSERT_EQ(mock_batch_sample_calls, 0);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * compute_obs_dim Tests
 * ============================================================================ */

TEST(compute_obs_dim_no_attachments_returns_zero) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 16, 8, 64);

    size_t dim = sensor_system_compute_obs_dim(sys, 0);
    ASSERT_EQ(dim, 0);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(compute_obs_dim_single_sensor) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 16, 8, 64);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);
    uint32_t s = sensor_system_create_sensor(sys, &config);
    sensor_system_attach(sys, 0, s);

    /* MOCK_VTABLE output_size = 4 */
    size_t dim = sensor_system_compute_obs_dim(sys, 0);
    ASSERT_EQ(dim, 4);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(compute_obs_dim_multiple_sensors_sums_correctly) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 16, 8, 64);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_TOF, &MOCK_VTABLE_3);

    SensorConfig config_a = sensor_config_default(SENSOR_TYPE_IMU);
    SensorConfig config_b = sensor_config_default(SENSOR_TYPE_TOF);
    uint32_t sa = sensor_system_create_sensor(sys, &config_a);  /* output_size=4 */
    uint32_t sb = sensor_system_create_sensor(sys, &config_b);  /* output_size=3 */

    sensor_system_attach(sys, 2, sa);
    sensor_system_attach(sys, 2, sb);

    /* 4 + 3 = 7 */
    size_t dim = sensor_system_compute_obs_dim(sys, 2);
    ASSERT_EQ(dim, 7);

    /* Drone 0 has no attachments */
    ASSERT_EQ(sensor_system_compute_obs_dim(sys, 0), 0);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(compute_obs_dim_invalid_drone_returns_zero) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 16, 8, 64);

    ASSERT_EQ(sensor_system_compute_obs_dim(sys, 999), 0);
    ASSERT_EQ(sensor_system_compute_obs_dim(NULL, 0), 0);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Scatter Correctness Tests
 * ============================================================================ */

TEST(scatter_noncontiguous_agent_indices) {
    /*
     * Attach a sensor to drones 1, 5, 13 (non-contiguous).
     * Verify each drone's observation slot gets the correct pattern.
     */
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 16, 8, 32);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);
    uint32_t sensor_idx = sensor_system_create_sensor(sys, &config);

    sensor_system_attach(sys, 1, sensor_idx);
    sensor_system_attach(sys, 5, sensor_idx);
    sensor_system_attach(sys, 13, sensor_idx);

    PlatformStateSOA* drones = platform_state_create(arena, 16, QUAD_STATE_EXT_COUNT);
    for (uint32_t i = 0; i < 14; i++) {
        platform_state_init(drones, i);
    }
    drones->rigid_body.count = 14;

    sensor_system_reset(sys);
    sensor_system_sample_all(sys, drones, NULL, NULL, 14);

    /* Drone 1 pattern: 1*10+c => 10,11,12,13 */
    const float* obs1 = sensor_system_get_drone_obs_const(sys, 1);
    ASSERT_FLOAT_NEAR(obs1[0], 10.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs1[1], 11.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs1[2], 12.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs1[3], 13.0f, 0.001f);

    /* Drone 5 pattern: 5*10+c => 50,51,52,53 */
    const float* obs5 = sensor_system_get_drone_obs_const(sys, 5);
    ASSERT_FLOAT_NEAR(obs5[0], 50.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs5[1], 51.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs5[2], 52.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs5[3], 53.0f, 0.001f);

    /* Drone 13 pattern: 13*10+c => 130,131,132,133 */
    const float* obs13 = sensor_system_get_drone_obs_const(sys, 13);
    ASSERT_FLOAT_NEAR(obs13[0], 130.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs13[1], 131.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs13[2], 132.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs13[3], 133.0f, 0.001f);

    /* Drone 0 should remain zeroed (no sensor attached) */
    const float* obs0 = sensor_system_get_drone_obs_const(sys, 0);
    ASSERT_FLOAT_NEAR(obs0[0], 0.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs0[1], 0.0f, 0.001f);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(scatter_multiple_sensors_correct_offsets) {
    /*
     * Drone 0 has two sensors attached:
     *   Sensor A (IMU mock, output_size=4) at offset 0
     *   Sensor B (ToF mock, output_size=3) at offset 4
     * Verify each sensor's output lands at the correct offset.
     */
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 16, 8, 64);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_TOF, &MOCK_VTABLE_3);

    SensorConfig config_a = sensor_config_default(SENSOR_TYPE_IMU);
    SensorConfig config_b = sensor_config_default(SENSOR_TYPE_TOF);
    uint32_t sa = sensor_system_create_sensor(sys, &config_a);  /* output_size=4 */
    uint32_t sb = sensor_system_create_sensor(sys, &config_b);  /* output_size=3 */

    uint32_t off_a = sensor_system_attach(sys, 0, sa);
    uint32_t off_b = sensor_system_attach(sys, 0, sb);

    ASSERT_EQ(off_a, 0);
    ASSERT_EQ(off_b, 4);

    PlatformStateSOA* drones = platform_state_create(arena, 16, QUAD_STATE_EXT_COUNT);
    platform_state_init(drones, 0);
    drones->rigid_body.count = 1;

    sensor_system_reset(sys);
    sensor_system_sample_all(sys, drones, NULL, NULL, 1);

    const float* obs = sensor_system_get_drone_obs_const(sys, 0);

    /* Sensor A pattern (d=0): 0*10+c = 0,1,2,3 at offsets [0..3] */
    ASSERT_FLOAT_NEAR(obs[0], 0.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[1], 1.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[2], 2.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[3], 3.0f, 0.001f);

    /* Sensor B pattern (d=0): 0*100+c = 0,1,2 at offsets [4..6] */
    ASSERT_FLOAT_NEAR(obs[4], 0.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[5], 1.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs[6], 2.0f, 0.001f);

    /* Beyond both sensors should be zero */
    ASSERT_FLOAT_NEAR(obs[7], 0.0f, 0.001f);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(scatter_multiple_sensors_noncontiguous_drones) {
    /*
     * Drones 2 and 7 each have two sensors (A=4 floats, B=3 floats).
     * Verify scatter writes correct values at correct obs offsets for each.
     */
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 16, 8, 64);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_TOF, &MOCK_VTABLE_3);

    SensorConfig config_a = sensor_config_default(SENSOR_TYPE_IMU);
    SensorConfig config_b = sensor_config_default(SENSOR_TYPE_TOF);
    uint32_t sa = sensor_system_create_sensor(sys, &config_a);
    uint32_t sb = sensor_system_create_sensor(sys, &config_b);

    sensor_system_attach(sys, 2, sa);
    sensor_system_attach(sys, 2, sb);
    sensor_system_attach(sys, 7, sa);
    sensor_system_attach(sys, 7, sb);

    PlatformStateSOA* drones = platform_state_create(arena, 16, QUAD_STATE_EXT_COUNT);
    for (uint32_t i = 0; i < 8; i++) {
        platform_state_init(drones, i);
    }
    drones->rigid_body.count = 8;

    sensor_system_reset(sys);
    sensor_system_sample_all(sys, drones, NULL, NULL, 8);

    /* Drone 2: sensor A => d=2: 20,21,22,23 at [0..3], sensor B => d=2: 200,201,202 at [4..6] */
    const float* obs2 = sensor_system_get_drone_obs_const(sys, 2);
    ASSERT_FLOAT_NEAR(obs2[0], 20.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs2[1], 21.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs2[2], 22.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs2[3], 23.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs2[4], 200.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs2[5], 201.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs2[6], 202.0f, 0.001f);

    /* Drone 7: sensor A => d=7: 70,71,72,73 at [0..3], sensor B => d=7: 700,701,702 at [4..6] */
    const float* obs7 = sensor_system_get_drone_obs_const(sys, 7);
    ASSERT_FLOAT_NEAR(obs7[0], 70.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs7[1], 71.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs7[2], 72.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs7[3], 73.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs7[4], 700.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs7[5], 701.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs7[6], 702.0f, 0.001f);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Detach Edge Cases Tests
 * ============================================================================ */

TEST(detach_last_sensor_leaves_zero_attachments) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 16, 8, 32);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);
    uint32_t s = sensor_system_create_sensor(sys, &config);
    sensor_system_attach(sys, 0, s);
    ASSERT_EQ(sensor_system_get_attachment_count(sys, 0), 1);

    sensor_system_detach(sys, 0, 0);
    ASSERT_EQ(sensor_system_get_attachment_count(sys, 0), 0);

    /* After detach, sampling should produce no output for this drone */
    PlatformStateSOA* drones = platform_state_create(arena, 16, QUAD_STATE_EXT_COUNT);
    platform_state_init(drones, 0);
    drones->rigid_body.count = 1;

    sensor_system_reset(sys);
    mock_batch_sample_calls = 0;
    sensor_system_sample_all(sys, drones, NULL, NULL, 1);

    /* No sensor should have been called (no drones grouped) */
    ASSERT_EQ(mock_batch_sample_calls, 0);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(detach_from_empty_drone_is_noop) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 16, 8, 32);

    /* Drone 0 has no attachments */
    ASSERT_EQ(sensor_system_get_attachment_count(sys, 0), 0);

    /* Detaching from an empty list should not crash */
    sensor_system_detach(sys, 0, 0);
    ASSERT_EQ(sensor_system_get_attachment_count(sys, 0), 0);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(detach_invalid_attachment_index_is_noop) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 16, 8, 32);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);
    uint32_t s = sensor_system_create_sensor(sys, &config);
    sensor_system_attach(sys, 0, s);
    ASSERT_EQ(sensor_system_get_attachment_count(sys, 0), 1);

    /* Attachment index 1 is out of range (only have 1 attachment at index 0) */
    sensor_system_detach(sys, 0, 1);
    ASSERT_EQ(sensor_system_get_attachment_count(sys, 0), 1);

    /* Very large index */
    sensor_system_detach(sys, 0, 999);
    ASSERT_EQ(sensor_system_get_attachment_count(sys, 0), 1);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(detach_invalid_agent_index_is_noop) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 16, 8, 32);

    /* Out of range drone index */
    sensor_system_detach(sys, 999, 0);
    /* Should not crash -- the function checks agent_idx >= max_agents */

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(detach_middle_recompacts_offsets) {
    /*
     * Attach three sensors (A=4, B=3, A=4) to drone 0.
     * Detach the middle (attachment index 1, sensor B at offset 4).
     * The third attachment should shift down and offsets recompute.
     */
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 16, 8, 64);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_TOF, &MOCK_VTABLE_3);

    SensorConfig config_a = sensor_config_default(SENSOR_TYPE_IMU);
    SensorConfig config_b = sensor_config_default(SENSOR_TYPE_TOF);
    uint32_t sa = sensor_system_create_sensor(sys, &config_a);
    uint32_t sb = sensor_system_create_sensor(sys, &config_b);

    sensor_system_attach(sys, 0, sa);   /* idx=0, offset=0, size=4 */
    sensor_system_attach(sys, 0, sb);   /* idx=1, offset=4, size=3 */
    sensor_system_attach(sys, 0, sa);   /* idx=2, offset=7, size=4 */
    ASSERT_EQ(sensor_system_get_attachment_count(sys, 0), 3);

    /* Detach the middle sensor (attachment index 1 = sensor B) */
    sensor_system_detach(sys, 0, 1);
    ASSERT_EQ(sensor_system_get_attachment_count(sys, 0), 2);

    /* After detach, the remaining attachments should be:
     *   idx=0: sensor A at offset 0, size 4
     *   idx=1: sensor A at offset 4 (recompacted from 7)
     */
    size_t dim = sensor_system_compute_obs_dim(sys, 0);
    ASSERT_EQ(dim, 8);  /* 4 + 4 */

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Reset Semantics Tests
 * ============================================================================ */

TEST(reset_zeroes_buffer_and_calls_sensor_reset) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 4, 8, 16);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_VELOCITY, &MOCK_VTABLE_LIFECYCLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_VELOCITY);
    sensor_system_create_sensor(sys, &config);

    /* Write some data to the observation buffer */
    float* obs = sensor_system_get_observations(sys);
    for (size_t i = 0; i < 4 * 16; i++) {
        obs[i] = 99.0f;
    }

    mock_reset_calls = 0;
    mock_reset_last_drone = UINT32_MAX;

    sensor_system_reset(sys);

    /* Buffer should be zeroed */
    for (size_t i = 0; i < 4 * 16; i++) {
        ASSERT_FLOAT_NEAR(obs[i], 0.0f, 0.001f);
    }

    /* Reset should have been called for each drone in the system.
     * With 1 sensor and max_agents=4, expect 4 reset calls. */
    ASSERT_EQ(mock_reset_calls, 4);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(reset_with_multiple_sensors_calls_all_resets) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 2, 8, 32);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_VELOCITY, &MOCK_VTABLE_LIFECYCLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_VELOCITY);
    sensor_system_create_sensor(sys, &config);
    sensor_system_create_sensor(sys, &config);

    mock_reset_calls = 0;

    sensor_system_reset(sys);

    /* 2 sensors * 2 max_agents = 4 reset calls */
    ASSERT_EQ(mock_reset_calls, 4);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(reset_with_external_buffer_zeroes_external) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 4, 4, 8);

    float external_buf[4 * 8];
    for (int i = 0; i < 4 * 8; i++) {
        external_buf[i] = 77.0f;
    }

    sensor_system_set_external_buffer(sys, external_buf);
    sensor_system_reset(sys);

    /* The reset should zero the external buffer since that is now observation_buffer */
    for (int i = 0; i < 4 * 8; i++) {
        ASSERT_FLOAT_NEAR(external_buf[i], 0.0f, 0.001f);
    }

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(reset_null_system_is_noop) {
    /* Should not crash */
    sensor_system_reset(NULL);
    return 0;
}

/* ============================================================================
 * Destroy Semantics Tests
 * ============================================================================ */

TEST(destroy_calls_sensor_destroy_callbacks) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 8, 8, 16);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_VELOCITY, &MOCK_VTABLE_LIFECYCLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_VELOCITY);
    sensor_system_create_sensor(sys, &config);
    sensor_system_create_sensor(sys, &config);
    sensor_system_create_sensor(sys, &config);

    mock_destroy_calls = 0;

    sensor_system_destroy(sys);

    ASSERT_EQ(mock_destroy_calls, 3);

    arena_destroy(arena);
    return 0;
}

TEST(destroy_null_system_is_noop) {
    /* Should not crash */
    sensor_system_destroy(NULL);
    return 0;
}

/* ============================================================================
 * Memory Size Tests
 * ============================================================================ */

TEST(memory_size_scales_with_drones) {
    size_t size_small = sensor_system_memory_size(64, 8, 32);
    size_t size_large = sensor_system_memory_size(1024, 8, 32);

    ASSERT_GT(size_large, size_small);

    /* The observation buffer scales as drones * obs_dim * 4 bytes.
     * 1024*32*4 = 131072, 64*32*4 = 8192. Diff = 122880.
     * Attachments also scale. The large should be substantially bigger. */
    ASSERT_GT(size_large - size_small, (size_t)100000);

    return 0;
}

TEST(memory_size_scales_with_obs_dim) {
    size_t size_a = sensor_system_memory_size(256, 8, 32);
    size_t size_b = sensor_system_memory_size(256, 8, 128);

    ASSERT_GT(size_b, size_a);

    /* obs buffer diff: 256*(128-32)*4 = 98304 */
    ASSERT_GT(size_b - size_a, (size_t)90000);

    return 0;
}

TEST(memory_size_scales_with_sensors) {
    size_t size_a = sensor_system_memory_size(256, 4, 64);
    size_t size_b = sensor_system_memory_size(256, 64, 64);

    ASSERT_GT(size_b, size_a);

    /* drones_by_sensor dominates: 64*256*4 - 4*256*4 = 60*256*4 = 61440 */
    ASSERT_GT(size_b - size_a, (size_t)50000);

    return 0;
}

TEST(memory_size_matches_formula) {
    /* Verify the formula from sensor_system_memory_size against hand-computation */
    uint32_t drones = 128;
    uint32_t sensors = 16;
    size_t obs_dim = 64;

    size_t expected = 0;
    expected += sizeof(SensorSystem);
    expected += sizeof(Sensor) * sensors;
    expected += sizeof(SensorAttachment) * drones * MAX_SENSORS_PER_DRONE;
    expected += sizeof(uint32_t) * drones;
    expected += drones * obs_dim * sizeof(float) + SENSOR_OBS_ALIGNMENT;
    expected += sizeof(uint32_t*) * sensors;
    expected += sizeof(uint32_t) * sensors * drones;
    expected += sizeof(uint32_t) * sensors;

    size_t actual = sensor_system_memory_size(drones, sensors, obs_dim);
    ASSERT_EQ(actual, expected);

    return 0;
}

/* ============================================================================
 * Additional Edge Case Tests
 * ============================================================================ */

TEST(sample_all_with_zero_drones_is_noop) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 16, 8, 32);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);
    uint32_t s = sensor_system_create_sensor(sys, &config);
    sensor_system_attach(sys, 0, s);

    PlatformStateSOA* drones = platform_state_create(arena, 16, QUAD_STATE_EXT_COUNT);
    drones->rigid_body.count = 0;

    mock_batch_sample_calls = 0;
    sensor_system_sample_all(sys, drones, NULL, NULL, 0);
    ASSERT_EQ(mock_batch_sample_calls, 0);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(sample_all_clamps_agent_count_to_max) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 4, 8, 32);  /* max_agents = 4 */
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);
    uint32_t s = sensor_system_create_sensor(sys, &config);

    /* Attach to drones 0..3 (all of max capacity) */
    for (uint32_t d = 0; d < 4; d++) {
        sensor_system_attach(sys, d, s);
    }

    PlatformStateSOA* drones = platform_state_create(arena, 16, QUAD_STATE_EXT_COUNT);
    for (uint32_t i = 0; i < 4; i++) {
        platform_state_init(drones, i);
    }
    drones->rigid_body.count = 4;

    mock_batch_sample_calls = 0;
    mock_last_agent_count = 0;

    /* Pass agent_count=100, should be clamped to max_agents=4 */
    sensor_system_sample_all(sys, drones, NULL, NULL, 100);
    ASSERT_EQ(mock_batch_sample_calls, 1);
    ASSERT_EQ(mock_last_agent_count, 4);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(attach_exceeds_obs_dim_fails) {
    /*
     * If attaching a sensor would push the total output past obs_dim,
     * the attach should fail (return UINT32_MAX).
     */
    Arena* arena = arena_create(2 * 1024 * 1024);
    /* obs_dim=6, mock output_size=4 */
    SensorSystem* sys = sensor_system_create(arena, 8, 8, 6);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);
    uint32_t s0 = sensor_system_create_sensor(sys, &config);
    uint32_t s1 = sensor_system_create_sensor(sys, &config);

    /* First attach: offset 0, size 4. 0+4 <= 6, OK */
    uint32_t off0 = sensor_system_attach(sys, 0, s0);
    ASSERT_EQ(off0, 0);

    /* Second attach: offset 4, size 4. 4+4=8 > 6, FAIL */
    uint32_t off1 = sensor_system_attach(sys, 0, s1);
    ASSERT_EQ(off1, UINT32_MAX);
    ASSERT_EQ(sensor_system_get_attachment_count(sys, 0), 1);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(create_sensor_unregistered_type_fails) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 8, 8, 32);
    /* Do NOT register any vtable for SENSOR_TYPE_LIDAR_3D */

    SensorConfig config = sensor_config_default(SENSOR_TYPE_LIDAR_3D);
    uint32_t idx = sensor_system_create_sensor(sys, &config);
    ASSERT_EQ(idx, UINT32_MAX);
    ASSERT_EQ(sys->sensor_count, 0);

    sensor_system_destroy(sys);
    arena_destroy(arena);
    return 0;
}

TEST(shared_sensor_scatter_writes_independent_drone_obs) {
    /*
     * One sensor shared by drones 0 and 3.
     * Verify that each drone's observation slot gets its own correct data
     * (the mock pattern uses agent_idx, so drone 0 and drone 3 should differ).
     */
    Arena* arena = arena_create(2 * 1024 * 1024);
    SensorSystem* sys = sensor_system_create(arena, 8, 8, 32);
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &MOCK_VTABLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);
    uint32_t s = sensor_system_create_sensor(sys, &config);

    sensor_system_attach(sys, 0, s);
    sensor_system_attach(sys, 3, s);

    PlatformStateSOA* drones = platform_state_create(arena, 8, QUAD_STATE_EXT_COUNT);
    for (uint32_t i = 0; i < 4; i++) {
        platform_state_init(drones, i);
    }
    drones->rigid_body.count = 4;

    sensor_system_reset(sys);
    sensor_system_sample_all(sys, drones, NULL, NULL, 4);

    const float* obs0 = sensor_system_get_drone_obs_const(sys, 0);
    const float* obs3 = sensor_system_get_drone_obs_const(sys, 3);

    /* Drone 0: 0,1,2,3 */
    ASSERT_FLOAT_NEAR(obs0[0], 0.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs0[3], 3.0f, 0.001f);

    /* Drone 3: 30,31,32,33 */
    ASSERT_FLOAT_NEAR(obs3[0], 30.0f, 0.001f);
    ASSERT_FLOAT_NEAR(obs3[3], 33.0f, 0.001f);

    /* They must differ */
    ASSERT_TRUE(obs0[0] != obs3[0]);

    sensor_system_destroy(sys);
    arena_destroy(arena);
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

    /* External Buffer Tests */
    RUN_TEST(external_buffer_redirects_observation_storage);
    RUN_TEST(external_buffer_set_null_is_noop);
    RUN_TEST(external_buffer_does_not_change_obs_dim);
    RUN_TEST(external_buffer_get_drone_obs_uses_external);

    /* sample_cpu_only Tests */
    RUN_TEST(sample_cpu_only_skips_gpu_sensors);
    RUN_TEST(sample_cpu_only_processes_all_cpu_sensors);

    /* sample_sensor Tests */
    RUN_TEST(sample_sensor_only_samples_specified_sensor);
    RUN_TEST(sample_sensor_with_invalid_index_is_noop);
    RUN_TEST(sample_sensor_no_attached_drones_is_noop);

    /* compute_obs_dim Tests */
    RUN_TEST(compute_obs_dim_no_attachments_returns_zero);
    RUN_TEST(compute_obs_dim_single_sensor);
    RUN_TEST(compute_obs_dim_multiple_sensors_sums_correctly);
    RUN_TEST(compute_obs_dim_invalid_drone_returns_zero);

    /* Scatter Correctness Tests */
    RUN_TEST(scatter_noncontiguous_agent_indices);
    RUN_TEST(scatter_multiple_sensors_correct_offsets);
    RUN_TEST(scatter_multiple_sensors_noncontiguous_drones);

    /* Detach Edge Cases Tests */
    RUN_TEST(detach_last_sensor_leaves_zero_attachments);
    RUN_TEST(detach_from_empty_drone_is_noop);
    RUN_TEST(detach_invalid_attachment_index_is_noop);
    RUN_TEST(detach_invalid_agent_index_is_noop);
    RUN_TEST(detach_middle_recompacts_offsets);

    /* Reset Semantics Tests */
    RUN_TEST(reset_zeroes_buffer_and_calls_sensor_reset);
    RUN_TEST(reset_with_multiple_sensors_calls_all_resets);
    RUN_TEST(reset_with_external_buffer_zeroes_external);
    RUN_TEST(reset_null_system_is_noop);

    /* Destroy Semantics Tests */
    RUN_TEST(destroy_calls_sensor_destroy_callbacks);
    RUN_TEST(destroy_null_system_is_noop);

    /* Memory Size Tests */
    RUN_TEST(memory_size_scales_with_drones);
    RUN_TEST(memory_size_scales_with_obs_dim);
    RUN_TEST(memory_size_scales_with_sensors);
    RUN_TEST(memory_size_matches_formula);

    /* Additional Edge Case Tests */
    RUN_TEST(sample_all_with_zero_drones_is_noop);
    RUN_TEST(sample_all_clamps_agent_count_to_max);
    RUN_TEST(attach_exceeds_obs_dim_fails);
    RUN_TEST(create_sensor_unregistered_type_fails);
    RUN_TEST(shared_sensor_scatter_writes_independent_drone_obs);

    TEST_SUITE_END();
}
