/**
 * Configuration Module Tests
 *
 * Comprehensive tests covering:
 * - Default value initialization
 * - TOML parsing (valid and invalid)
 * - Validation (schema and semantic)
 * - Conversion to/from DroneParamsSOA
 * - Serialization (save/load roundtrip, hash, compare, clone)
 */

#include "test_harness.h"
#include "configuration.h"
#include "drone_state.h"
#include <stdlib.h>

/* ============================================================================
 * Default Value Tests
 * ============================================================================ */

TEST(default_drone_is_crazyflie) {
    Config cfg;
    config_set_defaults(&cfg);

    /* Crazyflie 2.0 parameters */
    ASSERT_STR_EQ(cfg.drone.name, "crazyflie2");
    ASSERT_FLOAT_EQ(cfg.drone.mass, 0.027f);
    ASSERT_FLOAT_EQ(cfg.drone.arm_length, 0.046f);
    ASSERT_FLOAT_NEAR(cfg.drone.ixx, 1.4e-5f, 1e-8f);
    ASSERT_FLOAT_NEAR(cfg.drone.iyy, 1.4e-5f, 1e-8f);
    ASSERT_FLOAT_NEAR(cfg.drone.izz, 2.17e-5f, 1e-8f);
    ASSERT_FLOAT_NEAR(cfg.drone.k_thrust, 2.88e-8f, 1e-11f);
    ASSERT_FLOAT_NEAR(cfg.drone.k_torque, 7.24e-10f, 1e-13f);
    ASSERT_FLOAT_EQ(cfg.drone.motor_tau, 0.02f);
    ASSERT_FLOAT_EQ(cfg.drone.max_rpm, 21702.0f);

    return 0;
}

TEST(default_environment_values) {
    Config cfg;
    config_set_defaults(&cfg);

    ASSERT_EQ(cfg.environment.num_envs, 64);
    ASSERT_EQ(cfg.environment.drones_per_env, 16);
    ASSERT_FLOAT_EQ(cfg.environment.world_size[0], 20.0f);
    ASSERT_FLOAT_EQ(cfg.environment.world_size[1], 20.0f);
    ASSERT_FLOAT_EQ(cfg.environment.world_size[2], 10.0f);
    ASSERT_FLOAT_EQ(cfg.environment.voxel_size, 0.1f);
    ASSERT_EQ(cfg.environment.max_episode_steps, 1000);
    ASSERT_TRUE(cfg.environment.auto_reset);
    ASSERT_EQ(cfg.environment.seed, 42);

    return 0;
}

TEST(default_physics_values) {
    Config cfg;
    config_set_defaults(&cfg);

    ASSERT_FLOAT_EQ(cfg.physics.timestep, 0.02f);
    ASSERT_EQ(cfg.physics.substeps, 4);
    ASSERT_FLOAT_EQ(cfg.physics.gravity, 9.81f);
    ASSERT_STR_EQ(cfg.physics.integrator, "rk4");
    ASSERT_TRUE(cfg.physics.normalize_quaternions);
    ASSERT_TRUE(cfg.physics.enable_ground_effect);

    return 0;
}

TEST(default_reward_values) {
    Config cfg;
    config_set_defaults(&cfg);

    ASSERT_STR_EQ(cfg.reward.task, "hover");
    ASSERT_FLOAT_EQ(cfg.reward.distance_scale, 1.0f);
    ASSERT_FLOAT_EQ(cfg.reward.collision_penalty, 10.0f);
    ASSERT_FLOAT_EQ(cfg.reward.alive_bonus, 0.01f);

    return 0;
}

TEST(default_training_values) {
    Config cfg;
    config_set_defaults(&cfg);

    ASSERT_STR_EQ(cfg.training.algorithm, "ppo");
    ASSERT_FLOAT_NEAR(cfg.training.learning_rate, 3.0e-4f, 1e-7f);
    ASSERT_FLOAT_EQ(cfg.training.gamma, 0.99f);
    ASSERT_FLOAT_EQ(cfg.training.gae_lambda, 0.95f);
    ASSERT_FLOAT_EQ(cfg.training.clip_range, 0.2f);
    ASSERT_EQ(cfg.training.batch_size, 2048);

    return 0;
}

TEST(default_no_sensors) {
    Config cfg;
    config_set_defaults(&cfg);

    ASSERT_TRUE(cfg.sensors == NULL);
    ASSERT_EQ(cfg.num_sensors, 0);

    return 0;
}

/* ============================================================================
 * Parsing Tests
 * ============================================================================ */

TEST(parse_empty_string) {
    /* Empty string should return defaults */
    Config cfg;
    char error_msg[256] = {0};

    int result = config_load_string("", &cfg, error_msg);

    ASSERT_TRUE(result == 0);
    ASSERT_FLOAT_EQ(cfg.drone.mass, 0.027f);  /* Crazyflie default */

    config_free(&cfg);
    return 0;
}

TEST(parse_minimal_valid) {
    /* Minimal valid config with one section */
    const char* toml = "[drone]\nmass = 1.0\n";
    Config cfg;
    char error_msg[256] = {0};

    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_TRUE(result == 0);
    ASSERT_FLOAT_EQ(cfg.drone.mass, 1.0f);
    ASSERT_FLOAT_NEAR(cfg.drone.ixx, 1.4e-5f, 1e-8f);  /* Default for unspecified */

    config_free(&cfg);
    return 0;
}

TEST(parse_all_drone_fields) {
    /* Verify all drone fields parse correctly */
    const char* toml =
        "[drone]\n"
        "name = \"test_drone\"\n"
        "mass = 0.5\n"
        "arm_length = 0.1\n"
        "ixx = 1e-4\n"
        "iyy = 1e-4\n"
        "izz = 2e-4\n"
        "collision_radius = 0.12\n"
        "k_thrust = 3e-8\n"
        "k_torque = 8e-10\n"
        "motor_tau = 0.03\n"
        "max_rpm = 3000.0\n"
        "k_drag = 0.1\n"
        "k_drag_angular = 0.01\n"
        "max_velocity = 15.0\n"
        "max_angular_velocity = 40.0\n"
        "max_tilt_angle = 0.8\n"
        "color = [1.0, 0.5, 0.0]\n"
        "scale = 2.0\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_TRUE(result == 0);
    ASSERT_STR_EQ(cfg.drone.name, "test_drone");
    ASSERT_FLOAT_EQ(cfg.drone.mass, 0.5f);
    ASSERT_FLOAT_EQ(cfg.drone.arm_length, 0.1f);
    ASSERT_FLOAT_NEAR(cfg.drone.ixx, 1e-4f, 1e-8f);
    ASSERT_FLOAT_NEAR(cfg.drone.iyy, 1e-4f, 1e-8f);
    ASSERT_FLOAT_NEAR(cfg.drone.izz, 2e-4f, 1e-8f);
    ASSERT_FLOAT_EQ(cfg.drone.collision_radius, 0.12f);
    ASSERT_FLOAT_NEAR(cfg.drone.k_thrust, 3e-8f, 1e-11f);
    ASSERT_FLOAT_NEAR(cfg.drone.k_torque, 8e-10f, 1e-13f);
    ASSERT_FLOAT_EQ(cfg.drone.motor_tau, 0.03f);
    ASSERT_FLOAT_EQ(cfg.drone.max_rpm, 3000.0f);
    ASSERT_FLOAT_EQ(cfg.drone.k_drag, 0.1f);
    ASSERT_FLOAT_EQ(cfg.drone.k_drag_angular, 0.01f);
    ASSERT_FLOAT_EQ(cfg.drone.max_velocity, 15.0f);
    ASSERT_FLOAT_EQ(cfg.drone.max_angular_velocity, 40.0f);
    ASSERT_FLOAT_EQ(cfg.drone.max_tilt_angle, 0.8f);
    ASSERT_FLOAT_EQ(cfg.drone.color[0], 1.0f);
    ASSERT_FLOAT_EQ(cfg.drone.color[1], 0.5f);
    ASSERT_FLOAT_EQ(cfg.drone.color[2], 0.0f);
    ASSERT_FLOAT_EQ(cfg.drone.scale, 2.0f);

    config_free(&cfg);
    return 0;
}

TEST(parse_sensor_array) {
    /* [[sensors]] array parsing */
    const char* toml =
        "[[sensors]]\n"
        "type = \"imu\"\n"
        "name = \"main_imu\"\n"
        "sample_rate = 500.0\n"
        "\n"
        "[[sensors]]\n"
        "type = \"camera_depth\"\n"
        "name = \"front_camera\"\n"
        "sample_rate = 30.0\n"
        "width = 128\n"
        "height = 128\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_TRUE(result == 0);
    ASSERT_EQ(cfg.num_sensors, 2);
    ASSERT_STR_EQ(cfg.sensors[0].type, "imu");
    ASSERT_STR_EQ(cfg.sensors[0].name, "main_imu");
    ASSERT_FLOAT_EQ(cfg.sensors[0].sample_rate, 500.0f);
    ASSERT_STR_EQ(cfg.sensors[1].type, "camera_depth");
    ASSERT_EQ(cfg.sensors[1].width, 128);

    config_free(&cfg);
    return 0;
}

TEST(parse_noise_groups) {
    /* [[sensors.noise_groups]] parsing with stages */
    const char* toml =
        "[[sensors]]\n"
        "type = \"imu\"\n"
        "name = \"noisy_imu\"\n"
        "\n"
        "[[sensors.noise_groups]]\n"
        "channels = [0, 3]\n"
        "\n"
        "[[sensors.noise_groups.stages]]\n"
        "type = \"constant_bias\"\n"
        "values = [0.02, -0.01, 0.03]\n"
        "\n"
        "[[sensors.noise_groups.stages]]\n"
        "type = \"white_gaussian\"\n"
        "stddev = 0.008\n"
        "\n"
        "[[sensors.noise_groups]]\n"
        "channels = [3, 3]\n"
        "\n"
        "[[sensors.noise_groups.stages]]\n"
        "type = \"bias_drift\"\n"
        "tau = 100.0\n"
        "sigma = 0.001\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_TRUE(result == 0);
    ASSERT_EQ(cfg.num_sensors, 1);
    ASSERT_STR_EQ(cfg.sensors[0].type, "imu");

    /* First noise group: channels [0,3], two stages */
    ASSERT_EQ(cfg.sensors[0].num_noise_groups, 2);
    ASSERT_EQ(cfg.sensors[0].noise_groups[0].channels[0], 0);
    ASSERT_EQ(cfg.sensors[0].noise_groups[0].channels[1], 3);
    ASSERT_EQ(cfg.sensors[0].noise_groups[0].num_stages, 2);
    ASSERT_STR_EQ(cfg.sensors[0].noise_groups[0].stages[0].type, "constant_bias");
    ASSERT_FLOAT_EQ(cfg.sensors[0].noise_groups[0].stages[0].values[0], 0.02f);
    ASSERT_FLOAT_EQ(cfg.sensors[0].noise_groups[0].stages[0].values[1], -0.01f);
    ASSERT_FLOAT_EQ(cfg.sensors[0].noise_groups[0].stages[0].values[2], 0.03f);
    ASSERT_EQ(cfg.sensors[0].noise_groups[0].stages[0].value_count, 3);
    ASSERT_STR_EQ(cfg.sensors[0].noise_groups[0].stages[1].type, "white_gaussian");
    ASSERT_FLOAT_EQ(cfg.sensors[0].noise_groups[0].stages[1].stddev, 0.008f);

    /* Second noise group: channels [3,3], one stage */
    ASSERT_EQ(cfg.sensors[0].noise_groups[1].channels[0], 3);
    ASSERT_EQ(cfg.sensors[0].noise_groups[1].channels[1], 3);
    ASSERT_EQ(cfg.sensors[0].noise_groups[1].num_stages, 1);
    ASSERT_STR_EQ(cfg.sensors[0].noise_groups[1].stages[0].type, "bias_drift");
    ASSERT_FLOAT_EQ(cfg.sensors[0].noise_groups[1].stages[0].tau, 100.0f);
    ASSERT_FLOAT_EQ(cfg.sensors[0].noise_groups[1].stages[0].sigma, 0.001f);

    config_free(&cfg);
    return 0;
}

TEST(parse_scientific_notation) {
    const char* toml = "[drone]\nixx = 1.4e-5\nk_thrust = 2.88E-8\n";
    Config cfg;
    char error_msg[256] = {0};

    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_TRUE(result == 0);
    ASSERT_FLOAT_NEAR(cfg.drone.ixx, 1.4e-5f, 1e-9f);
    ASSERT_FLOAT_NEAR(cfg.drone.k_thrust, 2.88e-8f, 1e-12f);

    config_free(&cfg);
    return 0;
}

TEST(parse_integer_as_float) {
    /* TOML integers should convert to floats */
    const char* toml = "[drone]\nmass = 1\n";
    Config cfg;
    char error_msg[256] = {0};

    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_TRUE(result == 0);
    ASSERT_FLOAT_EQ(cfg.drone.mass, 1.0f);

    config_free(&cfg);
    return 0;
}

/* ============================================================================
 * Validation Tests
 * ============================================================================ */

TEST(validate_valid_config) {
    Config cfg;
    config_set_defaults(&cfg);
    ConfigError errors[CONFIG_MAX_ERRORS];

    int num_errors = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);

    ASSERT_EQ((uint32_t)num_errors, 0);
    return 0;
}

TEST(validate_mass_zero) {
    Config cfg;
    config_set_defaults(&cfg);
    cfg.drone.mass = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];

    int num_errors = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);

    ASSERT_TRUE(num_errors > 0);
    ASSERT_TRUE(strstr(errors[0].field, "mass") != NULL);
    return 0;
}

TEST(validate_mass_negative) {
    Config cfg;
    config_set_defaults(&cfg);
    cfg.drone.mass = -1.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];

    int num_errors = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);

    ASSERT_TRUE(num_errors > 0);
    ASSERT_TRUE(strstr(errors[0].field, "mass") != NULL);
    return 0;
}

TEST(validate_timestep_zero) {
    Config cfg;
    config_set_defaults(&cfg);
    cfg.physics.timestep = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];

    int num_errors = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);

    ASSERT_TRUE(num_errors > 0);
    ASSERT_TRUE(strstr(errors[0].field, "timestep") != NULL);
    return 0;
}

TEST(validate_num_envs_zero) {
    Config cfg;
    config_set_defaults(&cfg);
    cfg.environment.num_envs = 0;
    ConfigError errors[CONFIG_MAX_ERRORS];

    int num_errors = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);

    ASSERT_TRUE(num_errors > 0);
    return 0;
}

TEST(validate_integrator_invalid) {
    Config cfg;
    config_set_defaults(&cfg);
    strcpy(cfg.physics.integrator, "invalid_method");
    ConfigError errors[CONFIG_MAX_ERRORS];

    int num_errors = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);

    ASSERT_TRUE(num_errors > 0);
    ASSERT_TRUE(strstr(errors[0].field, "integrator") != NULL);
    return 0;
}

TEST(validate_spawn_height_inverted) {
    Config cfg;
    config_set_defaults(&cfg);
    cfg.environment.spawn_height_min = 10.0f;
    cfg.environment.spawn_height_max = 5.0f;  /* min > max */
    ConfigError errors[CONFIG_MAX_ERRORS];

    int num_errors = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);

    ASSERT_TRUE(num_errors > 0);
    ASSERT_TRUE(strstr(errors[0].field, "spawn_height") != NULL);
    return 0;
}

TEST(validate_sensor_type_unknown) {
    Config cfg;
    config_set_defaults(&cfg);
    cfg.sensors = (SensorConfigEntry*)malloc(sizeof(SensorConfigEntry));
    cfg.num_sensors = 1;
    memset(cfg.sensors, 0, sizeof(SensorConfigEntry));
    strcpy(cfg.sensors[0].type, "unknown_sensor");
    cfg.sensors[0].orientation[0] = 1.0f;  /* Valid quaternion */
    ConfigError errors[CONFIG_MAX_ERRORS];

    int num_errors = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);

    ASSERT_TRUE(num_errors > 0);
    ASSERT_TRUE(strstr(errors[0].message, "Unknown sensor") != NULL);

    config_free(&cfg);
    return 0;
}

TEST(validate_multiple_errors) {
    Config cfg;
    config_set_defaults(&cfg);
    cfg.drone.mass = 0.0f;
    cfg.drone.ixx = 0.0f;
    cfg.physics.timestep = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];

    int num_errors = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);

    ASSERT_TRUE(num_errors >= 3);  /* At least 3 errors */
    return 0;
}

/* ============================================================================
 * Conversion Tests
 * ============================================================================ */

TEST(config_to_params_single) {
    Config cfg;
    config_set_defaults(&cfg);
    Arena* arena = arena_create(65536);
    DroneParamsSOA* params = drone_params_create(arena, 1);

    drone_config_to_params(&cfg.drone, &cfg.physics, params, 0, 1);

    /* Verify field mapping (note name differences) */
    ASSERT_FLOAT_EQ(params->mass[0], cfg.drone.mass);
    ASSERT_FLOAT_NEAR(params->ixx[0], cfg.drone.ixx, 1e-9f);
    ASSERT_FLOAT_NEAR(params->iyy[0], cfg.drone.iyy, 1e-9f);
    ASSERT_FLOAT_NEAR(params->izz[0], cfg.drone.izz, 1e-9f);
    ASSERT_FLOAT_EQ(params->arm_length[0], cfg.drone.arm_length);
    ASSERT_FLOAT_EQ(params->collision_radius[0], cfg.drone.collision_radius);
    ASSERT_FLOAT_NEAR(params->k_thrust[0], cfg.drone.k_thrust, 1e-12f);
    ASSERT_FLOAT_NEAR(params->k_torque[0], cfg.drone.k_torque, 1e-14f);
    ASSERT_FLOAT_EQ(params->k_drag[0], cfg.drone.k_drag);
    ASSERT_FLOAT_EQ(params->k_ang_damp[0], cfg.drone.k_drag_angular);  /* Name differs */
    ASSERT_FLOAT_EQ(params->motor_tau[0], cfg.drone.motor_tau);
    ASSERT_FLOAT_EQ(params->max_rpm[0], cfg.drone.max_rpm);
    ASSERT_FLOAT_EQ(params->max_vel[0], cfg.drone.max_velocity);       /* Name differs */
    ASSERT_FLOAT_EQ(params->max_omega[0], cfg.drone.max_angular_velocity); /* Name differs */
    ASSERT_FLOAT_EQ(params->gravity[0], cfg.physics.gravity);          /* From ConfigPhysics */

    arena_destroy(arena);
    return 0;
}

TEST(config_to_params_broadcast) {
    /* Broadcast single config to 1024 drones */
    Config cfg;
    config_set_defaults(&cfg);
    Arena* arena = arena_create(1024 * 1024);
    DroneParamsSOA* params = drone_params_create(arena, 1024);

    drone_config_to_params(&cfg.drone, &cfg.physics, params, 0, 1024);

    /* Verify all have same values */
    for (uint32_t i = 0; i < 1024; i++) {
        ASSERT_FLOAT_EQ(params->mass[i], cfg.drone.mass);
        ASSERT_FLOAT_EQ(params->gravity[i], cfg.physics.gravity);
    }

    arena_destroy(arena);
    return 0;
}

TEST(params_to_config_roundtrip) {
    Config original;
    config_set_defaults(&original);
    Arena* arena = arena_create(65536);
    DroneParamsSOA* params = drone_params_create(arena, 1);

    drone_config_to_params(&original.drone, &original.physics, params, 0, 1);
    DroneConfig extracted = drone_params_to_config(params, 0);

    /* Note: gravity not extracted (belongs to ConfigPhysics) */
    ASSERT_FLOAT_EQ(extracted.mass, original.drone.mass);
    ASSERT_FLOAT_NEAR(extracted.ixx, original.drone.ixx, 1e-9f);
    ASSERT_FLOAT_EQ(extracted.arm_length, original.drone.arm_length);
    ASSERT_FLOAT_EQ(extracted.k_drag_angular, original.drone.k_drag_angular);
    ASSERT_FLOAT_EQ(extracted.max_velocity, original.drone.max_velocity);
    ASSERT_FLOAT_EQ(extracted.max_angular_velocity, original.drone.max_angular_velocity);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Serialization Tests
 * ============================================================================ */

TEST(hash_same_config) {
    Config cfg1, cfg2;
    config_set_defaults(&cfg1);
    config_set_defaults(&cfg2);

    uint64_t h1 = config_hash(&cfg1);
    uint64_t h2 = config_hash(&cfg2);

    ASSERT_TRUE(h1 == h2);
    return 0;
}

TEST(hash_different_config) {
    Config cfg1, cfg2;
    config_set_defaults(&cfg1);
    config_set_defaults(&cfg2);
    cfg2.drone.mass = 1.0f;

    uint64_t h1 = config_hash(&cfg1);
    uint64_t h2 = config_hash(&cfg2);

    ASSERT_TRUE(h1 != h2);
    return 0;
}

TEST(config_compare_identical) {
    Config cfg1, cfg2;
    config_set_defaults(&cfg1);
    config_set_defaults(&cfg2);

    int result = config_compare(&cfg1, &cfg2);

    ASSERT_TRUE(result == 0);  /* Identical */
    return 0;
}

TEST(config_compare_different) {
    Config cfg1, cfg2;
    config_set_defaults(&cfg1);
    config_set_defaults(&cfg2);
    cfg2.drone.mass = 1.0f;

    int result = config_compare(&cfg1, &cfg2);

    ASSERT_TRUE(result != 0);  /* Different */
    return 0;
}

TEST(config_clone) {
    Config src;
    config_set_defaults(&src);
    src.drone.mass = 0.5f;
    src.sensors = (SensorConfigEntry*)malloc(sizeof(SensorConfigEntry));
    src.num_sensors = 1;
    memset(src.sensors, 0, sizeof(SensorConfigEntry));
    strcpy(src.sensors[0].type, "imu");

    Arena* arena = arena_create(4096);
    Config dst;
    config_clone(&src, &dst, arena);

    ASSERT_FLOAT_EQ(dst.drone.mass, 0.5f);
    ASSERT_EQ(dst.num_sensors, 1);
    ASSERT_STR_EQ(dst.sensors[0].type, "imu");
    ASSERT_TRUE(dst.sensors != src.sensors);  /* Deep copy */

    config_free(&src);
    arena_destroy(arena);
    return 0;
}

TEST(config_to_json) {
    Config cfg;
    config_set_defaults(&cfg);
    cfg.config_hash = config_hash(&cfg);

    char buffer[1024];
    int result = config_to_json(&cfg, buffer, sizeof(buffer));

    ASSERT_TRUE(result == 0);
    ASSERT_TRUE(strstr(buffer, "\"drone\"") != NULL);
    ASSERT_TRUE(strstr(buffer, "crazyflie2") != NULL);
    ASSERT_TRUE(strstr(buffer, "\"num_envs\": 64") != NULL);
    return 0;
}

/* ============================================================================
 * Main Test Runner
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Configuration Module");

    /* Default Value Tests */
    RUN_TEST(default_drone_is_crazyflie);
    RUN_TEST(default_environment_values);
    RUN_TEST(default_physics_values);
    RUN_TEST(default_reward_values);
    RUN_TEST(default_training_values);
    RUN_TEST(default_no_sensors);

    /* Parsing Tests */
    RUN_TEST(parse_empty_string);
    RUN_TEST(parse_minimal_valid);
    RUN_TEST(parse_all_drone_fields);
    RUN_TEST(parse_sensor_array);
    RUN_TEST(parse_noise_groups);
    RUN_TEST(parse_scientific_notation);
    RUN_TEST(parse_integer_as_float);

    /* Validation Tests */
    RUN_TEST(validate_valid_config);
    RUN_TEST(validate_mass_zero);
    RUN_TEST(validate_mass_negative);
    RUN_TEST(validate_timestep_zero);
    RUN_TEST(validate_num_envs_zero);
    RUN_TEST(validate_integrator_invalid);
    RUN_TEST(validate_spawn_height_inverted);
    RUN_TEST(validate_sensor_type_unknown);
    RUN_TEST(validate_multiple_errors);

    /* Conversion Tests */
    RUN_TEST(config_to_params_single);
    RUN_TEST(config_to_params_broadcast);
    RUN_TEST(params_to_config_roundtrip);

    /* Serialization Tests */
    RUN_TEST(hash_same_config);
    RUN_TEST(hash_different_config);
    RUN_TEST(config_compare_identical);
    RUN_TEST(config_compare_different);
    RUN_TEST(config_clone);
    RUN_TEST(config_to_json);

    TEST_SUITE_END();
}
