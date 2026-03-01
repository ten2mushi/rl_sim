/**
 * Configuration Module Tests
 *
 * Comprehensive tests covering:
 * - Default value initialization (exhaustive, every field)
 * - Sensor defaults for all 10 sensor types
 * - Sensor type registry (is_valid_sensor_type)
 * - TOML parsing (valid, malformed, NULL params, all sections, noise stages)
 * - Validation (schema and semantic, every field boundary)
 * - Conversion to/from PlatformParamsSOA (offset, count, capacity, NULL safety)
 * - Serialization (save/load roundtrip, hash sensitivity, compare, clone, JSON)
 * - config_memory_size, config_free safety
 * - Round-trip consistency and idempotency
 * - Struct size verification
 * - File I/O error paths
 */

#include "test_harness.h"
#include "configuration.h"
#include "drone_state.h"
#include "platform_quadcopter.h"
#include <stdlib.h>

/* ============================================================================
 * Default Value Tests
 * ============================================================================ */

TEST(default_drone_is_crazyflie) {
    Config cfg;
    config_set_defaults(&cfg);

    /* Crazyflie 2.0 parameters */
    ASSERT_STR_EQ(cfg.platform.name, "crazyflie2");
    ASSERT_FLOAT_EQ(cfg.platform.mass, 0.027f);
    ASSERT_FLOAT_NEAR(cfg.platform.ixx, 1.4e-5f, 1e-8f);
    ASSERT_FLOAT_NEAR(cfg.platform.iyy, 1.4e-5f, 1e-8f);
    ASSERT_FLOAT_NEAR(cfg.platform.izz, 2.17e-5f, 1e-8f);

    /* Quadcopter-specific defaults via platform_specific */
    ASSERT_NOT_NULL(cfg.platform.platform_specific);
    const QuadcopterConfig* q = (const QuadcopterConfig*)cfg.platform.platform_specific;
    ASSERT_FLOAT_EQ(q->arm_length, 0.046f);
    ASSERT_FLOAT_NEAR(q->k_thrust, 2.88e-8f, 1e-11f);
    ASSERT_FLOAT_NEAR(q->k_torque, 7.24e-10f, 1e-13f);
    ASSERT_FLOAT_EQ(q->motor_tau, 0.02f);
    ASSERT_FLOAT_EQ(q->max_rpm, 21702.0f);

    return 0;
}

TEST(default_environment_values) {
    Config cfg;
    config_set_defaults(&cfg);

    ASSERT_EQ(cfg.environment.num_envs, 64);
    ASSERT_EQ(cfg.environment.agents_per_env, 16);
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
 * Extended Default Value Tests - Exhaustive field verification
 * ============================================================================ */

TEST(default_drone_all_fields_exhaustive) {
    PlatformConfig pc;
    platform_config_set_defaults(&pc);

    ASSERT_STR_EQ(pc.name, "crazyflie2");
    ASSERT_TRUE(pc.model_path[0] == '\0');
    ASSERT_FLOAT_EQ(pc.collision_radius, 0.056f);
    ASSERT_FLOAT_EQ(pc.max_velocity, 10.0f);
    ASSERT_FLOAT_EQ(pc.max_angular_velocity, 20.0f);
    ASSERT_FLOAT_EQ(pc.max_tilt_angle, 1.0f);
    ASSERT_FLOAT_EQ(pc.color[0], 0.2f);
    ASSERT_FLOAT_EQ(pc.color[1], 0.6f);
    ASSERT_FLOAT_EQ(pc.color[2], 1.0f);
    ASSERT_FLOAT_EQ(pc.scale, 1.0f);

    /* Quadcopter-specific defaults via platform_specific */
    ASSERT_NOT_NULL(pc.platform_specific);
    const QuadcopterConfig* qc = (const QuadcopterConfig*)pc.platform_specific;
    ASSERT_FLOAT_EQ(qc->k_drag, 0.0f);
    ASSERT_FLOAT_EQ(qc->k_ang_damp, 0.0f);

    free(pc.platform_specific);
    return 0;
}

TEST(default_environment_all_fields_exhaustive) {
    EnvironmentConfig ec;
    environment_config_set_defaults(&ec);

    ASSERT_FLOAT_EQ(ec.world_origin[0], 0.0f);
    ASSERT_FLOAT_EQ(ec.world_origin[1], 0.0f);
    ASSERT_FLOAT_EQ(ec.world_origin[2], 5.0f);
    ASSERT_EQ(ec.max_bricks, 8192u);
    ASSERT_FLOAT_EQ(ec.spawn_radius, 5.0f);
    ASSERT_FLOAT_EQ(ec.spawn_height_min, 2.0f);
    ASSERT_FLOAT_EQ(ec.spawn_height_max, 8.0f);
    ASSERT_FLOAT_EQ(ec.min_separation, 1.0f);
    ASSERT_STR_EQ(ec.world_type, "obstacles");
    ASSERT_EQ(ec.num_obstacles, 20u);

    return 0;
}

TEST(default_physics_all_fields_exhaustive) {
    ConfigPhysics pc;
    physics_config_set_defaults(&pc);

    ASSERT_FLOAT_EQ(pc.velocity_clamp, 20.0f);
    ASSERT_FLOAT_EQ(pc.angular_velocity_clamp, 30.0f);
    ASSERT_FLOAT_EQ(pc.ground_effect_height, 0.5f);
    ASSERT_FLOAT_EQ(pc.ground_effect_strength, 1.5f);
    ASSERT_FLOAT_EQ(pc.dt_variance, 0.0f);
    ASSERT_FLOAT_EQ(pc.mass_variance, 0.0f);
    ASSERT_FLOAT_EQ(pc.thrust_variance, 0.0f);

    return 0;
}

TEST(default_reward_all_fields_exhaustive) {
    RewardConfigData rc;
    reward_config_data_set_defaults(&rc);

    ASSERT_STR_EQ(rc.task, "hover");
    ASSERT_FLOAT_EQ(rc.distance_scale, 1.0f);
    ASSERT_FLOAT_EQ(rc.distance_exp, 1.0f);
    ASSERT_FLOAT_EQ(rc.reach_bonus, 10.0f);
    ASSERT_FLOAT_EQ(rc.reach_radius, 0.5f);
    ASSERT_FLOAT_EQ(rc.velocity_match_scale, 0.0f);
    ASSERT_FLOAT_EQ(rc.uprightness_scale, 0.1f);
    ASSERT_FLOAT_EQ(rc.energy_scale, 0.001f);
    ASSERT_FLOAT_NEAR(rc.jerk_scale, 0.0001f, 1e-8f);
    ASSERT_FLOAT_EQ(rc.collision_penalty, 10.0f);
    ASSERT_FLOAT_EQ(rc.world_collision_penalty, 5.0f);
    ASSERT_FLOAT_EQ(rc.drone_collision_penalty, 2.0f);
    ASSERT_FLOAT_EQ(rc.alive_bonus, 0.01f);
    ASSERT_FLOAT_EQ(rc.success_bonus, 100.0f);
    ASSERT_FLOAT_EQ(rc.reward_min, -10.0f);
    ASSERT_FLOAT_EQ(rc.reward_max, 10.0f);

    return 0;
}

TEST(default_training_all_fields_exhaustive) {
    TrainingConfig tc;
    training_config_set_defaults(&tc);

    ASSERT_STR_EQ(tc.algorithm, "ppo");
    ASSERT_FLOAT_NEAR(tc.learning_rate, 3.0e-4f, 1e-7f);
    ASSERT_FLOAT_EQ(tc.gamma, 0.99f);
    ASSERT_FLOAT_EQ(tc.gae_lambda, 0.95f);
    ASSERT_FLOAT_EQ(tc.clip_range, 0.2f);
    ASSERT_FLOAT_EQ(tc.entropy_coef, 0.01f);
    ASSERT_FLOAT_EQ(tc.value_coef, 0.5f);
    ASSERT_FLOAT_EQ(tc.max_grad_norm, 0.5f);
    ASSERT_EQ(tc.batch_size, 2048u);
    ASSERT_EQ(tc.num_epochs, 10u);
    ASSERT_EQ(tc.rollout_length, 128u);
    ASSERT_EQ(tc.log_interval, 10u);
    ASSERT_EQ(tc.save_interval, 100u);
    ASSERT_STR_EQ(tc.checkpoint_dir, "checkpoints");

    return 0;
}

/* ============================================================================
 * Sensor Default Tests
 * ============================================================================ */

TEST(default_sensor_imu) {
    SensorConfigEntry s = sensor_config_entry_default("imu");
    ASSERT_STR_EQ(s.type, "imu");
    ASSERT_STR_EQ(s.name, "sensor");
    ASSERT_FLOAT_EQ(s.orientation[0], 1.0f);
    ASSERT_FLOAT_EQ(s.orientation[1], 0.0f);
    ASSERT_FLOAT_EQ(s.orientation[2], 0.0f);
    ASSERT_FLOAT_EQ(s.orientation[3], 0.0f);
    ASSERT_FLOAT_EQ(s.position[0], 0.0f);
    ASSERT_FLOAT_EQ(s.sample_rate, 0.0f);
    ASSERT_EQ(s.num_noise_groups, 0u);
    return 0;
}

TEST(default_sensor_tof) {
    SensorConfigEntry s = sensor_config_entry_default("tof");
    ASSERT_STR_EQ(s.type, "tof");
    ASSERT_FLOAT_EQ(s.max_range, 4.0f);
    ASSERT_EQ(s.num_rays, 1u);
    ASSERT_FLOAT_EQ(s.orientation[0], 1.0f);
    return 0;
}

TEST(default_sensor_lidar_2d) {
    SensorConfigEntry s = sensor_config_entry_default("lidar_2d");
    ASSERT_STR_EQ(s.type, "lidar_2d");
    ASSERT_FLOAT_EQ(s.max_range, 10.0f);
    ASSERT_EQ(s.num_rays, 64u);
    ASSERT_FLOAT_NEAR(s.fov, 6.28318f, 1e-3f);
    return 0;
}

TEST(default_sensor_lidar_3d) {
    SensorConfigEntry s = sensor_config_entry_default("lidar_3d");
    ASSERT_STR_EQ(s.type, "lidar_3d");
    ASSERT_FLOAT_EQ(s.max_range, 20.0f);
    ASSERT_EQ(s.num_rays, 64u);
    ASSERT_FLOAT_NEAR(s.fov, 6.28318f, 1e-3f);
    ASSERT_FLOAT_NEAR(s.fov_vertical, 0.524f, 1e-3f);
    ASSERT_EQ(s.vertical_layers, 16u);
    return 0;
}

TEST(default_sensor_camera_rgb) {
    SensorConfigEntry s = sensor_config_entry_default("camera_rgb");
    ASSERT_STR_EQ(s.type, "camera_rgb");
    ASSERT_EQ(s.width, 84u);
    ASSERT_EQ(s.height, 84u);
    ASSERT_FLOAT_NEAR(s.fov, 1.57f, 1e-2f);
    ASSERT_FLOAT_NEAR(s.fov_vertical, 1.18f, 1e-2f);
    ASSERT_FLOAT_EQ(s.near_clip, 0.1f);
    ASSERT_FLOAT_EQ(s.far_clip, 100.0f);
    ASSERT_EQ(s.num_classes, 10u);
    return 0;
}

TEST(default_sensor_camera_depth) {
    SensorConfigEntry s = sensor_config_entry_default("camera_depth");
    ASSERT_STR_EQ(s.type, "camera_depth");
    ASSERT_EQ(s.width, 84u);
    ASSERT_EQ(s.height, 84u);
    ASSERT_FLOAT_EQ(s.near_clip, 0.1f);
    ASSERT_FLOAT_EQ(s.far_clip, 100.0f);
    return 0;
}

TEST(default_sensor_camera_segmentation) {
    SensorConfigEntry s = sensor_config_entry_default("camera_segmentation");
    ASSERT_STR_EQ(s.type, "camera_segmentation");
    ASSERT_EQ(s.width, 84u);
    ASSERT_EQ(s.height, 84u);
    ASSERT_EQ(s.num_classes, 10u);
    return 0;
}

TEST(default_sensor_position) {
    SensorConfigEntry s = sensor_config_entry_default("position");
    ASSERT_STR_EQ(s.type, "position");
    ASSERT_FLOAT_EQ(s.orientation[0], 1.0f);
    ASSERT_EQ(s.num_rays, 0u);
    ASSERT_EQ(s.width, 0u);
    return 0;
}

TEST(default_sensor_velocity) {
    SensorConfigEntry s = sensor_config_entry_default("velocity");
    ASSERT_STR_EQ(s.type, "velocity");
    ASSERT_FLOAT_EQ(s.orientation[0], 1.0f);
    return 0;
}

TEST(default_sensor_neighbor) {
    SensorConfigEntry s = sensor_config_entry_default("neighbor");
    ASSERT_STR_EQ(s.type, "neighbor");
    ASSERT_EQ(s.k_neighbors, 5u);
    ASSERT_FLOAT_EQ(s.max_range, 10.0f);
    return 0;
}

TEST(default_sensor_unknown_type_gets_zero_fields) {
    SensorConfigEntry s = sensor_config_entry_default("nonexistent");
    ASSERT_STR_EQ(s.type, "nonexistent");
    ASSERT_STR_EQ(s.name, "sensor");
    ASSERT_FLOAT_EQ(s.orientation[0], 1.0f);
    ASSERT_EQ(s.num_rays, 0u);
    ASSERT_EQ(s.width, 0u);
    ASSERT_FLOAT_EQ(s.max_range, 0.0f);
    ASSERT_EQ(s.k_neighbors, 0u);
    return 0;
}

/* ============================================================================
 * is_valid_sensor_type Tests
 * ============================================================================ */

TEST(sensor_type_all_valid_types_recognized) {
    ASSERT_TRUE(is_valid_sensor_type("imu"));
    ASSERT_TRUE(is_valid_sensor_type("tof"));
    ASSERT_TRUE(is_valid_sensor_type("lidar_2d"));
    ASSERT_TRUE(is_valid_sensor_type("lidar_3d"));
    ASSERT_TRUE(is_valid_sensor_type("camera_rgb"));
    ASSERT_TRUE(is_valid_sensor_type("camera_depth"));
    ASSERT_TRUE(is_valid_sensor_type("camera_segmentation"));
    ASSERT_TRUE(is_valid_sensor_type("position"));
    ASSERT_TRUE(is_valid_sensor_type("velocity"));
    ASSERT_TRUE(is_valid_sensor_type("neighbor"));
    return 0;
}

TEST(sensor_type_rejects_invalid_strings) {
    ASSERT_FALSE(is_valid_sensor_type(""));
    ASSERT_FALSE(is_valid_sensor_type("IMU"));
    ASSERT_FALSE(is_valid_sensor_type("camera"));
    ASSERT_FALSE(is_valid_sensor_type("lidar"));
    ASSERT_FALSE(is_valid_sensor_type("gps"));
    return 0;
}

/* ============================================================================
 * Parsing Tests
 * ============================================================================ */

TEST(parse_empty_string) {
    Config cfg;
    char error_msg[256] = {0};

    int result = config_load_string("", &cfg, error_msg);

    ASSERT_TRUE(result == 0);
    ASSERT_FLOAT_EQ(cfg.platform.mass, 0.027f);

    config_free(&cfg);
    return 0;
}

TEST(parse_minimal_valid) {
    const char* toml = "[drone]\nmass = 1.0\n";
    Config cfg;
    char error_msg[256] = {0};

    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_TRUE(result == 0);
    ASSERT_FLOAT_EQ(cfg.platform.mass, 1.0f);
    ASSERT_FLOAT_NEAR(cfg.platform.ixx, 1.4e-5f, 1e-8f);

    config_free(&cfg);
    return 0;
}

TEST(parse_all_drone_fields) {
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
    ASSERT_STR_EQ(cfg.platform.name, "test_drone");
    ASSERT_FLOAT_EQ(cfg.platform.mass, 0.5f);
    ASSERT_FLOAT_NEAR(cfg.platform.ixx, 1e-4f, 1e-8f);
    ASSERT_FLOAT_NEAR(cfg.platform.iyy, 1e-4f, 1e-8f);
    ASSERT_FLOAT_NEAR(cfg.platform.izz, 2e-4f, 1e-8f);
    ASSERT_FLOAT_EQ(cfg.platform.collision_radius, 0.12f);
    ASSERT_FLOAT_EQ(cfg.platform.max_velocity, 15.0f);
    ASSERT_FLOAT_EQ(cfg.platform.max_angular_velocity, 40.0f);
    ASSERT_FLOAT_EQ(cfg.platform.max_tilt_angle, 0.8f);
    ASSERT_FLOAT_EQ(cfg.platform.color[0], 1.0f);
    ASSERT_FLOAT_EQ(cfg.platform.color[1], 0.5f);
    ASSERT_FLOAT_EQ(cfg.platform.color[2], 0.0f);
    ASSERT_FLOAT_EQ(cfg.platform.scale, 2.0f);

    /* Quadcopter-specific fields via platform_specific */
    ASSERT_NOT_NULL(cfg.platform.platform_specific);
    const QuadcopterConfig* q = (const QuadcopterConfig*)cfg.platform.platform_specific;
    ASSERT_FLOAT_EQ(q->arm_length, 0.1f);
    ASSERT_FLOAT_NEAR(q->k_thrust, 3e-8f, 1e-11f);
    ASSERT_FLOAT_NEAR(q->k_torque, 8e-10f, 1e-13f);
    ASSERT_FLOAT_EQ(q->motor_tau, 0.03f);
    ASSERT_FLOAT_EQ(q->max_rpm, 3000.0f);
    ASSERT_FLOAT_EQ(q->k_drag, 0.1f);
    ASSERT_FLOAT_EQ(q->k_ang_damp, 0.01f);

    config_free(&cfg);
    return 0;
}

TEST(parse_sensor_array) {
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
    ASSERT_FLOAT_NEAR(cfg.platform.ixx, 1.4e-5f, 1e-9f);

    /* k_thrust is quad-specific, accessed via platform_specific */
    ASSERT_NOT_NULL(cfg.platform.platform_specific);
    const QuadcopterConfig* q = (const QuadcopterConfig*)cfg.platform.platform_specific;
    ASSERT_FLOAT_NEAR(q->k_thrust, 2.88e-8f, 1e-12f);

    config_free(&cfg);
    return 0;
}

TEST(parse_integer_as_float) {
    const char* toml = "[drone]\nmass = 1\n";
    Config cfg;
    char error_msg[256] = {0};

    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_TRUE(result == 0);
    ASSERT_FLOAT_EQ(cfg.platform.mass, 1.0f);

    config_free(&cfg);
    return 0;
}

/* ============================================================================
 * Extended Parsing Tests
 * ============================================================================ */

TEST(parse_malformed_toml_returns_error) {
    const char* toml = "[drone\nmass = 1.0\n";
    Config cfg;
    char error_msg[256] = {0};

    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_TRUE(result == -2);
    ASSERT_TRUE(strlen(error_msg) > 0);
    ASSERT_TRUE(strstr(error_msg, "TOML parse error") != NULL);
    return 0;
}

TEST(parse_null_string_returns_error) {
    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(NULL, &cfg, error_msg);
    ASSERT_TRUE(result == -1);
    return 0;
}

TEST(parse_null_config_returns_error) {
    char error_msg[256] = {0};
    int result = config_load_string("[drone]\n", NULL, error_msg);
    ASSERT_TRUE(result == -1);
    return 0;
}

TEST(parse_null_error_msg_returns_error) {
    Config cfg;
    int result = config_load_string("[drone]\n", &cfg, NULL);
    ASSERT_TRUE(result == -1);
    return 0;
}

TEST(parse_all_environment_fields) {
    const char* toml =
        "[environment]\n"
        "num_envs = 128\n"
        "agents_per_env = 32\n"
        "world_size = [40.0, 40.0, 20.0]\n"
        "world_origin = [1.0, 2.0, 3.0]\n"
        "voxel_size = 0.2\n"
        "max_bricks = 16384\n"
        "spawn_radius = 10.0\n"
        "spawn_height_min = 3.0\n"
        "spawn_height_max = 15.0\n"
        "min_separation = 2.0\n"
        "max_episode_steps = 2000\n"
        "auto_reset = false\n"
        "world_type = \"maze\"\n"
        "num_obstacles = 50\n"
        "seed = 123\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_TRUE(result == 0);
    ASSERT_EQ(cfg.environment.num_envs, 128u);
    ASSERT_EQ(cfg.environment.agents_per_env, 32u);
    ASSERT_FLOAT_EQ(cfg.environment.world_size[0], 40.0f);
    ASSERT_FLOAT_EQ(cfg.environment.world_size[1], 40.0f);
    ASSERT_FLOAT_EQ(cfg.environment.world_size[2], 20.0f);
    ASSERT_FLOAT_EQ(cfg.environment.world_origin[0], 1.0f);
    ASSERT_FLOAT_EQ(cfg.environment.world_origin[1], 2.0f);
    ASSERT_FLOAT_EQ(cfg.environment.world_origin[2], 3.0f);
    ASSERT_FLOAT_EQ(cfg.environment.voxel_size, 0.2f);
    ASSERT_EQ(cfg.environment.max_bricks, 16384u);
    ASSERT_FLOAT_EQ(cfg.environment.spawn_radius, 10.0f);
    ASSERT_FLOAT_EQ(cfg.environment.spawn_height_min, 3.0f);
    ASSERT_FLOAT_EQ(cfg.environment.spawn_height_max, 15.0f);
    ASSERT_FLOAT_EQ(cfg.environment.min_separation, 2.0f);
    ASSERT_EQ(cfg.environment.max_episode_steps, 2000u);
    ASSERT_FALSE(cfg.environment.auto_reset);
    ASSERT_STR_EQ(cfg.environment.world_type, "maze");
    ASSERT_EQ(cfg.environment.num_obstacles, 50u);
    ASSERT_EQ(cfg.environment.seed, 123u);

    config_free(&cfg);
    return 0;
}

TEST(parse_all_physics_fields) {
    const char* toml =
        "[physics]\n"
        "timestep = 0.01\n"
        "substeps = 8\n"
        "gravity = 3.72\n"
        "integrator = \"euler\"\n"
        "velocity_clamp = 50.0\n"
        "angular_velocity_clamp = 60.0\n"
        "normalize_quaternions = false\n"
        "enable_ground_effect = false\n"
        "ground_effect_height = 1.0\n"
        "ground_effect_strength = 2.0\n"
        "dt_variance = 0.001\n"
        "mass_variance = 0.01\n"
        "thrust_variance = 0.02\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_TRUE(result == 0);
    ASSERT_FLOAT_EQ(cfg.physics.timestep, 0.01f);
    ASSERT_EQ(cfg.physics.substeps, 8u);
    ASSERT_FLOAT_EQ(cfg.physics.gravity, 3.72f);
    ASSERT_STR_EQ(cfg.physics.integrator, "euler");
    ASSERT_FLOAT_EQ(cfg.physics.velocity_clamp, 50.0f);
    ASSERT_FLOAT_EQ(cfg.physics.angular_velocity_clamp, 60.0f);
    ASSERT_FALSE(cfg.physics.normalize_quaternions);
    ASSERT_FALSE(cfg.physics.enable_ground_effect);
    ASSERT_FLOAT_EQ(cfg.physics.ground_effect_height, 1.0f);
    ASSERT_FLOAT_EQ(cfg.physics.ground_effect_strength, 2.0f);
    ASSERT_FLOAT_EQ(cfg.physics.dt_variance, 0.001f);
    ASSERT_FLOAT_EQ(cfg.physics.mass_variance, 0.01f);
    ASSERT_FLOAT_EQ(cfg.physics.thrust_variance, 0.02f);

    config_free(&cfg);
    return 0;
}

TEST(parse_all_reward_fields) {
    const char* toml =
        "[reward]\n"
        "task = \"waypoint\"\n"
        "distance_scale = 2.0\n"
        "distance_exp = 2.0\n"
        "reach_bonus = 20.0\n"
        "reach_radius = 1.0\n"
        "velocity_match_scale = 0.5\n"
        "uprightness_scale = 0.2\n"
        "energy_scale = 0.01\n"
        "jerk_scale = 0.001\n"
        "collision_penalty = 20.0\n"
        "world_collision_penalty = 15.0\n"
        "drone_collision_penalty = 5.0\n"
        "alive_bonus = 0.1\n"
        "success_bonus = 200.0\n"
        "reward_min = -50.0\n"
        "reward_max = 50.0\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_TRUE(result == 0);
    ASSERT_STR_EQ(cfg.reward.task, "waypoint");
    ASSERT_FLOAT_EQ(cfg.reward.distance_scale, 2.0f);
    ASSERT_FLOAT_EQ(cfg.reward.distance_exp, 2.0f);
    ASSERT_FLOAT_EQ(cfg.reward.reach_bonus, 20.0f);
    ASSERT_FLOAT_EQ(cfg.reward.reach_radius, 1.0f);
    ASSERT_FLOAT_EQ(cfg.reward.velocity_match_scale, 0.5f);
    ASSERT_FLOAT_EQ(cfg.reward.uprightness_scale, 0.2f);
    ASSERT_FLOAT_EQ(cfg.reward.energy_scale, 0.01f);
    ASSERT_FLOAT_EQ(cfg.reward.jerk_scale, 0.001f);
    ASSERT_FLOAT_EQ(cfg.reward.collision_penalty, 20.0f);
    ASSERT_FLOAT_EQ(cfg.reward.world_collision_penalty, 15.0f);
    ASSERT_FLOAT_EQ(cfg.reward.drone_collision_penalty, 5.0f);
    ASSERT_FLOAT_EQ(cfg.reward.alive_bonus, 0.1f);
    ASSERT_FLOAT_EQ(cfg.reward.success_bonus, 200.0f);
    ASSERT_FLOAT_EQ(cfg.reward.reward_min, -50.0f);
    ASSERT_FLOAT_EQ(cfg.reward.reward_max, 50.0f);

    config_free(&cfg);
    return 0;
}

TEST(parse_all_training_fields) {
    const char* toml =
        "[training]\n"
        "algorithm = \"sac\"\n"
        "learning_rate = 1e-3\n"
        "gamma = 0.995\n"
        "gae_lambda = 0.97\n"
        "clip_range = 0.3\n"
        "entropy_coef = 0.02\n"
        "value_coef = 1.0\n"
        "max_grad_norm = 1.0\n"
        "batch_size = 4096\n"
        "num_epochs = 20\n"
        "rollout_length = 256\n"
        "log_interval = 5\n"
        "save_interval = 50\n"
        "checkpoint_dir = \"/tmp/ckpts\"\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_TRUE(result == 0);
    ASSERT_STR_EQ(cfg.training.algorithm, "sac");
    ASSERT_FLOAT_NEAR(cfg.training.learning_rate, 1e-3f, 1e-7f);
    ASSERT_FLOAT_EQ(cfg.training.gamma, 0.995f);
    ASSERT_FLOAT_EQ(cfg.training.gae_lambda, 0.97f);
    ASSERT_FLOAT_EQ(cfg.training.clip_range, 0.3f);
    ASSERT_FLOAT_EQ(cfg.training.entropy_coef, 0.02f);
    ASSERT_FLOAT_EQ(cfg.training.value_coef, 1.0f);
    ASSERT_FLOAT_EQ(cfg.training.max_grad_norm, 1.0f);
    ASSERT_EQ(cfg.training.batch_size, 4096u);
    ASSERT_EQ(cfg.training.num_epochs, 20u);
    ASSERT_EQ(cfg.training.rollout_length, 256u);
    ASSERT_EQ(cfg.training.log_interval, 5u);
    ASSERT_EQ(cfg.training.save_interval, 50u);
    ASSERT_STR_EQ(cfg.training.checkpoint_dir, "/tmp/ckpts");

    config_free(&cfg);
    return 0;
}

TEST(parse_multi_section_config) {
    const char* toml =
        "[drone]\nmass = 0.5\n"
        "[physics]\ngravity = 3.72\n"
        "[environment]\nseed = 999\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_TRUE(result == 0);
    ASSERT_FLOAT_EQ(cfg.platform.mass, 0.5f);
    ASSERT_FLOAT_EQ(cfg.physics.gravity, 3.72f);
    ASSERT_EQ(cfg.environment.seed, 999u);
    /* Unspecified fields retain defaults */
    const QuadcopterConfig* q = (const QuadcopterConfig*)cfg.platform.platform_specific;
    ASSERT_FLOAT_EQ(q->arm_length, 0.046f);
    ASSERT_STR_EQ(cfg.physics.integrator, "rk4");
    ASSERT_EQ(cfg.environment.num_envs, 64u);

    config_free(&cfg);
    return 0;
}

TEST(parse_sensor_inherits_type_defaults) {
    const char* toml =
        "[[sensors]]\ntype = \"lidar_2d\"\nname = \"my_lidar\"\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_TRUE(result == 0);
    ASSERT_EQ(cfg.num_sensors, 1u);
    ASSERT_FLOAT_EQ(cfg.sensors[0].max_range, 10.0f);
    ASSERT_EQ(cfg.sensors[0].num_rays, 64u);
    ASSERT_FLOAT_NEAR(cfg.sensors[0].fov, 6.28318f, 1e-3f);

    config_free(&cfg);
    return 0;
}

TEST(parse_sensor_overrides_type_defaults) {
    const char* toml =
        "[[sensors]]\ntype = \"lidar_2d\"\nname = \"custom\"\n"
        "max_range = 50.0\nnum_rays = 256\nfov = 3.14\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_TRUE(result == 0);
    ASSERT_FLOAT_EQ(cfg.sensors[0].max_range, 50.0f);
    ASSERT_EQ(cfg.sensors[0].num_rays, 256u);
    ASSERT_FLOAT_NEAR(cfg.sensors[0].fov, 3.14f, 1e-2f);

    config_free(&cfg);
    return 0;
}

TEST(parse_validation_failure_resets_to_defaults) {
    const char* toml = "[drone]\nmass = -1.0\n";
    Config cfg;
    char error_msg[256] = {0};

    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_TRUE(result == -4);
    ASSERT_TRUE(strlen(error_msg) > 0);
    ASSERT_TRUE(strstr(error_msg, "Validation error") != NULL);
    ASSERT_FLOAT_EQ(cfg.platform.mass, 0.027f);
    return 0;
}

TEST(parse_boolean_values) {
    const char* toml =
        "[physics]\nnormalize_quaternions = false\nenable_ground_effect = false\n"
        "[environment]\nauto_reset = false\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_TRUE(result == 0);
    ASSERT_FALSE(cfg.physics.normalize_quaternions);
    ASSERT_FALSE(cfg.physics.enable_ground_effect);
    ASSERT_FALSE(cfg.environment.auto_reset);

    config_free(&cfg);
    return 0;
}

TEST(parse_sets_config_hash_nonzero) {
    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string("", &cfg, error_msg);

    ASSERT_TRUE(result == 0);
    ASSERT_TRUE(cfg.config_hash != 0);

    config_free(&cfg);
    return 0;
}

TEST(parse_noise_stage_all_types) {
    const char* toml =
        "[[sensors]]\ntype = \"imu\"\n\n"
        "[[sensors.noise_groups]]\nchannels = [0, 6]\n\n"
        "[[sensors.noise_groups.stages]]\ntype = \"scale_factor\"\nerror = 0.05\n\n"
        "[[sensors.noise_groups.stages]]\ntype = \"distance_dependent\"\ncoeff = 0.1\npower = 2.0\n\n"
        "[[sensors.noise_groups.stages]]\ntype = \"quantization\"\nstep = 0.01\n\n"
        "[[sensors.noise_groups.stages]]\ntype = \"dropout\"\nprobability = 0.05\nreplacement = -1.0\n\n"
        "[[sensors.noise_groups.stages]]\ntype = \"saturation\"\nmin_val = -10.0\nmax_val = 10.0\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_TRUE(result == 0);
    ASSERT_EQ(cfg.sensors[0].noise_groups[0].num_stages, 5u);

    ASSERT_STR_EQ(cfg.sensors[0].noise_groups[0].stages[0].type, "scale_factor");
    ASSERT_FLOAT_EQ(cfg.sensors[0].noise_groups[0].stages[0].error, 0.05f);

    ASSERT_STR_EQ(cfg.sensors[0].noise_groups[0].stages[1].type, "distance_dependent");
    ASSERT_FLOAT_EQ(cfg.sensors[0].noise_groups[0].stages[1].coeff, 0.1f);
    ASSERT_FLOAT_EQ(cfg.sensors[0].noise_groups[0].stages[1].power, 2.0f);

    ASSERT_STR_EQ(cfg.sensors[0].noise_groups[0].stages[2].type, "quantization");
    ASSERT_FLOAT_EQ(cfg.sensors[0].noise_groups[0].stages[2].step, 0.01f);

    ASSERT_STR_EQ(cfg.sensors[0].noise_groups[0].stages[3].type, "dropout");
    ASSERT_FLOAT_EQ(cfg.sensors[0].noise_groups[0].stages[3].probability, 0.05f);
    ASSERT_FLOAT_EQ(cfg.sensors[0].noise_groups[0].stages[3].replacement, -1.0f);

    ASSERT_STR_EQ(cfg.sensors[0].noise_groups[0].stages[4].type, "saturation");
    ASSERT_FLOAT_EQ(cfg.sensors[0].noise_groups[0].stages[4].min_val, -10.0f);
    ASSERT_FLOAT_EQ(cfg.sensors[0].noise_groups[0].stages[4].max_val, 10.0f);

    config_free(&cfg);
    return 0;
}

TEST(parse_noise_power_defaults_to_one) {
    const char* toml =
        "[[sensors]]\ntype = \"imu\"\n\n"
        "[[sensors.noise_groups]]\nchannels = [0, 1]\n\n"
        "[[sensors.noise_groups.stages]]\ntype = \"distance_dependent\"\ncoeff = 0.5\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_TRUE(result == 0);
    ASSERT_FLOAT_EQ(cfg.sensors[0].noise_groups[0].stages[0].power, 1.0f);

    config_free(&cfg);
    return 0;
}

TEST(parse_sensor_position_and_orientation) {
    const char* toml =
        "[[sensors]]\ntype = \"imu\"\n"
        "position = [0.01, -0.02, 0.03]\n"
        "orientation = [0.707, 0.0, 0.707, 0.0]\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_TRUE(result == 0);
    ASSERT_FLOAT_EQ(cfg.sensors[0].position[0], 0.01f);
    ASSERT_FLOAT_EQ(cfg.sensors[0].position[1], -0.02f);
    ASSERT_FLOAT_EQ(cfg.sensors[0].position[2], 0.03f);
    ASSERT_FLOAT_NEAR(cfg.sensors[0].orientation[0], 0.707f, 1e-3f);
    ASSERT_FLOAT_NEAR(cfg.sensors[0].orientation[2], 0.707f, 1e-3f);

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
    cfg.platform.mass = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int num_errors = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(num_errors > 0);
    ASSERT_TRUE(strstr(errors[0].field, "mass") != NULL);
    return 0;
}

TEST(validate_mass_negative) {
    Config cfg;
    config_set_defaults(&cfg);
    cfg.platform.mass = -1.0f;
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
    cfg.environment.spawn_height_max = 5.0f;
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
    cfg.sensors[0].orientation[0] = 1.0f;
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
    cfg.platform.mass = 0.0f;
    cfg.platform.ixx = 0.0f;
    cfg.physics.timestep = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int num_errors = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(num_errors >= 3);
    return 0;
}

/* ============================================================================
 * Extended Drone Validation
 * ============================================================================ */

TEST(validate_drone_arm_length_zero) {
    PlatformConfig pc;
    platform_config_set_defaults(&pc);
    QuadcopterConfig* qc = (QuadcopterConfig*)pc.platform_specific;
    qc->arm_length = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "arm_length") != NULL);
    free(pc.platform_specific);
    return 0;
}

TEST(validate_drone_inertia_iyy_zero) {
    PlatformConfig pc;
    platform_config_set_defaults(&pc);
    pc.iyy = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "iyy") != NULL);
    free(pc.platform_specific);
    return 0;
}

TEST(validate_drone_inertia_izz_zero) {
    PlatformConfig pc;
    platform_config_set_defaults(&pc);
    pc.izz = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "izz") != NULL);
    free(pc.platform_specific);
    return 0;
}

TEST(validate_drone_k_thrust_zero) {
    PlatformConfig pc;
    platform_config_set_defaults(&pc);
    QuadcopterConfig* qc = (QuadcopterConfig*)pc.platform_specific;
    qc->k_thrust = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "k_thrust") != NULL);
    free(pc.platform_specific);
    return 0;
}

TEST(validate_drone_motor_tau_zero) {
    PlatformConfig pc;
    platform_config_set_defaults(&pc);
    QuadcopterConfig* qc = (QuadcopterConfig*)pc.platform_specific;
    qc->motor_tau = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "motor_tau") != NULL);
    free(pc.platform_specific);
    return 0;
}

TEST(validate_drone_max_rpm_zero) {
    PlatformConfig pc;
    platform_config_set_defaults(&pc);
    QuadcopterConfig* qc = (QuadcopterConfig*)pc.platform_specific;
    qc->max_rpm = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "max_rpm") != NULL);
    free(pc.platform_specific);
    return 0;
}

TEST(validate_drone_collision_radius_zero) {
    PlatformConfig pc;
    platform_config_set_defaults(&pc);
    pc.collision_radius = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "collision_radius") != NULL);
    free(pc.platform_specific);
    return 0;
}

TEST(validate_drone_k_drag_negative) {
    PlatformConfig pc;
    platform_config_set_defaults(&pc);
    QuadcopterConfig* qc = (QuadcopterConfig*)pc.platform_specific;
    qc->k_drag = -0.01f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "k_drag") != NULL);
    free(pc.platform_specific);
    return 0;
}

TEST(validate_drone_k_drag_zero_is_valid) {
    PlatformConfig pc;
    platform_config_set_defaults(&pc);
    QuadcopterConfig* qc = (QuadcopterConfig*)pc.platform_specific;
    qc->k_drag = 0.0f;
    qc->k_ang_damp = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_EQ(n, 0);
    free(pc.platform_specific);
    return 0;
}

TEST(validate_drone_max_velocity_zero) {
    PlatformConfig pc;
    platform_config_set_defaults(&pc);
    pc.max_velocity = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "max_velocity") != NULL);
    free(pc.platform_specific);
    return 0;
}

TEST(validate_drone_color_out_of_range) {
    PlatformConfig pc;
    platform_config_set_defaults(&pc);
    pc.color[0] = 1.5f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "color") != NULL);
    free(pc.platform_specific);
    return 0;
}

TEST(validate_drone_color_negative) {
    PlatformConfig pc;
    platform_config_set_defaults(&pc);
    pc.color[1] = -0.1f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "color") != NULL);
    free(pc.platform_specific);
    return 0;
}

TEST(validate_drone_color_boundary_valid) {
    PlatformConfig pc;
    platform_config_set_defaults(&pc);
    pc.color[0] = 0.0f;
    pc.color[1] = 0.5f;
    pc.color[2] = 1.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_EQ(n, 0);
    free(pc.platform_specific);
    return 0;
}

TEST(validate_drone_scale_zero) {
    PlatformConfig pc;
    platform_config_set_defaults(&pc);
    pc.scale = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "scale") != NULL);
    free(pc.platform_specific);
    return 0;
}

TEST(validate_drone_defaults_pass) {
    PlatformConfig pc;
    platform_config_set_defaults(&pc);
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_EQ(n, 0);
    free(pc.platform_specific);
    return 0;
}

/* ============================================================================
 * Extended Environment Validation
 * ============================================================================ */

TEST(validate_environment_world_size_zero) {
    EnvironmentConfig ec;
    environment_config_set_defaults(&ec);
    ec.world_size[0] = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_environment(&ec, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "world_size") != NULL);
    return 0;
}

TEST(validate_environment_voxel_size_zero) {
    EnvironmentConfig ec;
    environment_config_set_defaults(&ec);
    ec.voxel_size = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_environment(&ec, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "voxel_size") != NULL);
    return 0;
}

TEST(validate_environment_spawn_radius_negative) {
    EnvironmentConfig ec;
    environment_config_set_defaults(&ec);
    ec.spawn_radius = -1.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_environment(&ec, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "spawn_radius") != NULL);
    return 0;
}

TEST(validate_environment_min_separation_negative) {
    EnvironmentConfig ec;
    environment_config_set_defaults(&ec);
    ec.min_separation = -0.5f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_environment(&ec, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "min_separation") != NULL);
    return 0;
}

TEST(validate_environment_max_episode_steps_zero) {
    EnvironmentConfig ec;
    environment_config_set_defaults(&ec);
    ec.max_episode_steps = 0;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_environment(&ec, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "max_episode_steps") != NULL);
    return 0;
}

TEST(validate_environment_world_type_unknown) {
    EnvironmentConfig ec;
    environment_config_set_defaults(&ec);
    strcpy(ec.world_type, "battlefield");
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_environment(&ec, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "world_type") != NULL);
    return 0;
}

TEST(validate_environment_all_valid_world_types) {
    const char* valid_types[] = {"empty", "obstacles", "maze", "race", "custom"};
    for (int i = 0; i < 5; i++) {
        EnvironmentConfig ec;
        environment_config_set_defaults(&ec);
        strncpy(ec.world_type, valid_types[i], sizeof(ec.world_type) - 1);
        ConfigError errors[CONFIG_MAX_ERRORS];
        int n = config_validate_environment(&ec, errors, CONFIG_MAX_ERRORS);
        for (int e = 0; e < n; e++) {
            ASSERT_TRUE(strstr(errors[e].field, "world_type") == NULL);
        }
    }
    return 0;
}

TEST(validate_environment_spawn_height_equal) {
    EnvironmentConfig ec;
    environment_config_set_defaults(&ec);
    ec.spawn_height_min = 5.0f;
    ec.spawn_height_max = 5.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_environment(&ec, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "spawn_height") != NULL);
    return 0;
}

TEST(validate_environment_defaults_pass) {
    EnvironmentConfig ec;
    environment_config_set_defaults(&ec);
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_environment(&ec, errors, CONFIG_MAX_ERRORS);
    ASSERT_EQ(n, 0);
    return 0;
}

/* ============================================================================
 * Extended Physics Validation
 * ============================================================================ */

TEST(validate_physics_timestep_negative) {
    ConfigPhysics pc;
    physics_config_set_defaults(&pc);
    pc.timestep = -0.01f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "timestep") != NULL);
    return 0;
}

TEST(validate_physics_timestep_too_large) {
    ConfigPhysics pc;
    physics_config_set_defaults(&pc);
    pc.timestep = 0.2f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "timestep") != NULL);
    ASSERT_TRUE(strstr(errors[0].message, "stability") != NULL);
    return 0;
}

TEST(validate_physics_timestep_boundary_valid) {
    ConfigPhysics pc;
    physics_config_set_defaults(&pc);
    pc.timestep = 0.1f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&pc, errors, CONFIG_MAX_ERRORS);
    for (int i = 0; i < n; i++) {
        ASSERT_TRUE(strstr(errors[i].field, "timestep") == NULL);
    }
    return 0;
}

TEST(validate_physics_euler_valid) {
    ConfigPhysics pc;
    physics_config_set_defaults(&pc);
    strcpy(pc.integrator, "euler");
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&pc, errors, CONFIG_MAX_ERRORS);
    for (int i = 0; i < n; i++) {
        ASSERT_TRUE(strstr(errors[i].field, "integrator") == NULL);
    }
    return 0;
}

TEST(validate_physics_substeps_zero) {
    ConfigPhysics pc;
    physics_config_set_defaults(&pc);
    pc.substeps = 0;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "substeps") != NULL);
    return 0;
}

TEST(validate_physics_gravity_negative) {
    ConfigPhysics pc;
    physics_config_set_defaults(&pc);
    pc.gravity = -9.81f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "gravity") != NULL);
    return 0;
}

TEST(validate_physics_gravity_zero_valid) {
    ConfigPhysics pc;
    physics_config_set_defaults(&pc);
    pc.gravity = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&pc, errors, CONFIG_MAX_ERRORS);
    for (int i = 0; i < n; i++) {
        ASSERT_TRUE(strstr(errors[i].field, "gravity") == NULL);
    }
    return 0;
}

TEST(validate_physics_velocity_clamp_zero) {
    ConfigPhysics pc;
    physics_config_set_defaults(&pc);
    pc.velocity_clamp = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "velocity_clamp") != NULL);
    return 0;
}

TEST(validate_physics_ground_effect_height_zero_when_enabled) {
    ConfigPhysics pc;
    physics_config_set_defaults(&pc);
    pc.enable_ground_effect = true;
    pc.ground_effect_height = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "ground_effect_height") != NULL);
    return 0;
}

TEST(validate_physics_ground_effect_strength_below_one) {
    ConfigPhysics pc;
    physics_config_set_defaults(&pc);
    pc.enable_ground_effect = true;
    pc.ground_effect_strength = 0.5f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "ground_effect_strength") != NULL);
    return 0;
}

TEST(validate_physics_ground_effect_not_checked_when_disabled) {
    ConfigPhysics pc;
    physics_config_set_defaults(&pc);
    pc.enable_ground_effect = false;
    pc.ground_effect_height = 0.0f;
    pc.ground_effect_strength = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&pc, errors, CONFIG_MAX_ERRORS);
    for (int i = 0; i < n; i++) {
        ASSERT_TRUE(strstr(errors[i].field, "ground_effect") == NULL);
    }
    return 0;
}

TEST(validate_physics_dt_variance_negative) {
    ConfigPhysics pc;
    physics_config_set_defaults(&pc);
    pc.dt_variance = -0.001f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "dt_variance") != NULL);
    return 0;
}

TEST(validate_physics_mass_variance_negative) {
    ConfigPhysics pc;
    physics_config_set_defaults(&pc);
    pc.mass_variance = -0.01f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "mass_variance") != NULL);
    return 0;
}

TEST(validate_physics_defaults_pass) {
    ConfigPhysics pc;
    physics_config_set_defaults(&pc);
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&pc, errors, CONFIG_MAX_ERRORS);
    ASSERT_EQ(n, 0);
    return 0;
}

/* ============================================================================
 * Extended Sensor Validation
 * ============================================================================ */

TEST(validate_sensor_quaternion_not_normalized) {
    SensorConfigEntry s = sensor_config_entry_default("imu");
    s.orientation[0] = 0.5f;
    s.orientation[1] = 0.0f;
    s.orientation[2] = 0.0f;
    s.orientation[3] = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_sensors(&s, 1, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "orientation") != NULL);
    return 0;
}

TEST(validate_sensor_quaternion_zero) {
    SensorConfigEntry s = sensor_config_entry_default("imu");
    s.orientation[0] = 0.0f;
    s.orientation[1] = 0.0f;
    s.orientation[2] = 0.0f;
    s.orientation[3] = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_sensors(&s, 1, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "orientation") != NULL);
    return 0;
}

TEST(validate_sensor_sample_rate_negative) {
    SensorConfigEntry s = sensor_config_entry_default("imu");
    s.sample_rate = -10.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_sensors(&s, 1, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "sample_rate") != NULL);
    return 0;
}

TEST(validate_sensor_tof_max_range_zero) {
    SensorConfigEntry s = sensor_config_entry_default("tof");
    s.max_range = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_sensors(&s, 1, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "max_range") != NULL);
    return 0;
}

TEST(validate_sensor_lidar_2d_zero_rays) {
    SensorConfigEntry s = sensor_config_entry_default("lidar_2d");
    s.num_rays = 0;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_sensors(&s, 1, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "num_rays") != NULL);
    return 0;
}

TEST(validate_sensor_lidar_2d_zero_fov) {
    SensorConfigEntry s = sensor_config_entry_default("lidar_2d");
    s.fov = 0.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_sensors(&s, 1, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "fov") != NULL);
    return 0;
}

TEST(validate_sensor_camera_zero_dimensions) {
    SensorConfigEntry s = sensor_config_entry_default("camera_rgb");
    s.width = 0;
    s.height = 0;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_sensors(&s, 1, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    return 0;
}

TEST(validate_sensor_camera_near_geq_far) {
    SensorConfigEntry s = sensor_config_entry_default("camera_depth");
    s.near_clip = 100.0f;
    s.far_clip = 10.0f;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_sensors(&s, 1, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    return 0;
}

TEST(validate_sensor_neighbor_k_zero) {
    SensorConfigEntry s = sensor_config_entry_default("neighbor");
    s.k_neighbors = 0;
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_sensors(&s, 1, errors, CONFIG_MAX_ERRORS);
    ASSERT_TRUE(n > 0);
    ASSERT_TRUE(strstr(errors[0].field, "k_neighbors") != NULL);
    return 0;
}

TEST(validate_sensor_defaults_all_pass) {
    const char* types[] = {
        "imu", "tof", "lidar_2d", "lidar_3d",
        "camera_rgb", "camera_depth", "camera_segmentation",
        "position", "velocity", "neighbor"
    };
    for (size_t i = 0; i < sizeof(types)/sizeof(types[0]); i++) {
        SensorConfigEntry s = sensor_config_entry_default(types[i]);
        ConfigError errors[CONFIG_MAX_ERRORS];
        int n = config_validate_sensors(&s, 1, errors, CONFIG_MAX_ERRORS);
        ASSERT_MSG(n == 0, types[i]);
    }
    return 0;
}

TEST(validate_sensors_null_array_zero_count) {
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_sensors(NULL, 0, errors, CONFIG_MAX_ERRORS);
    ASSERT_EQ(n, 0);
    return 0;
}

TEST(validate_respects_max_errors_limit) {
    Config cfg;
    config_set_defaults(&cfg);
    cfg.platform.mass = 0.0f;
    cfg.platform.ixx = 0.0f;
    cfg.platform.iyy = 0.0f;
    cfg.platform.izz = 0.0f;
    QuadcopterConfig* qc = (QuadcopterConfig*)cfg.platform.platform_specific;
    qc->arm_length = 0.0f;
    qc->k_thrust = 0.0f;
    qc->motor_tau = 0.0f;
    qc->max_rpm = 0.0f;
    ConfigError errors[2];
    int n = config_validate(&cfg, errors, 2);
    ASSERT_TRUE(n <= 2);
    ASSERT_TRUE(n > 0);
    config_free(&cfg);
    return 0;
}

/* ============================================================================
 * Conversion Tests
 * ============================================================================ */

TEST(config_to_params_single) {
    Config cfg;
    config_set_defaults(&cfg);
    Arena* arena = arena_create(65536);
    PlatformParamsSOA* params = platform_params_create(arena, 1, QUAD_PARAMS_EXT_COUNT);

    platform_config_to_params(&cfg.platform, &cfg.physics, params, 0, 1);

    const QuadcopterConfig* qc = (const QuadcopterConfig*)cfg.platform.platform_specific;
    ASSERT_FLOAT_EQ(params->rigid_body.mass[0], cfg.platform.mass);
    ASSERT_FLOAT_NEAR(params->rigid_body.ixx[0], cfg.platform.ixx, 1e-9f);
    ASSERT_FLOAT_NEAR(params->rigid_body.iyy[0], cfg.platform.iyy, 1e-9f);
    ASSERT_FLOAT_NEAR(params->rigid_body.izz[0], cfg.platform.izz, 1e-9f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_ARM_LENGTH][0], qc->arm_length);
    ASSERT_FLOAT_EQ(params->rigid_body.collision_radius[0], cfg.platform.collision_radius);
    ASSERT_FLOAT_NEAR(params->extension[QUAD_PEXT_K_THRUST][0], qc->k_thrust, 1e-12f);
    ASSERT_FLOAT_NEAR(params->extension[QUAD_PEXT_K_TORQUE][0], qc->k_torque, 1e-14f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_K_DRAG][0], qc->k_drag);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_K_ANG_DAMP][0], qc->k_ang_damp);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_MOTOR_TAU][0], qc->motor_tau);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_MAX_RPM][0], qc->max_rpm);
    ASSERT_FLOAT_EQ(params->rigid_body.max_vel[0], cfg.platform.max_velocity);
    ASSERT_FLOAT_EQ(params->rigid_body.max_omega[0], cfg.platform.max_angular_velocity);
    ASSERT_FLOAT_EQ(params->rigid_body.gravity[0], cfg.physics.gravity);

    config_free(&cfg);
    arena_destroy(arena);
    return 0;
}

TEST(config_to_params_broadcast) {
    Config cfg;
    config_set_defaults(&cfg);
    Arena* arena = arena_create(1024 * 1024);
    PlatformParamsSOA* params = platform_params_create(arena, 1024, QUAD_PARAMS_EXT_COUNT);

    platform_config_to_params(&cfg.platform, &cfg.physics, params, 0, 1024);

    for (uint32_t i = 0; i < 1024; i++) {
        ASSERT_FLOAT_EQ(params->rigid_body.mass[i], cfg.platform.mass);
        ASSERT_FLOAT_EQ(params->rigid_body.gravity[i], cfg.physics.gravity);
    }

    arena_destroy(arena);
    return 0;
}

TEST(params_to_config_roundtrip) {
    Config original;
    config_set_defaults(&original);
    Arena* arena = arena_create(65536);
    PlatformParamsSOA* params = platform_params_create(arena, 1, QUAD_PARAMS_EXT_COUNT);

    platform_config_to_params(&original.platform, &original.physics, params, 0, 1);
    PlatformConfig extracted = platform_params_to_config(params, 0);

    const QuadcopterConfig* oq = (const QuadcopterConfig*)original.platform.platform_specific;
    const QuadcopterConfig* eq = (const QuadcopterConfig*)extracted.platform_specific;
    ASSERT_FLOAT_EQ(extracted.mass, original.platform.mass);
    ASSERT_FLOAT_NEAR(extracted.ixx, original.platform.ixx, 1e-9f);
    ASSERT_FLOAT_EQ(eq->arm_length, oq->arm_length);
    ASSERT_FLOAT_EQ(eq->k_ang_damp, oq->k_ang_damp);
    ASSERT_FLOAT_EQ(extracted.max_velocity, original.platform.max_velocity);
    ASSERT_FLOAT_EQ(extracted.max_angular_velocity, original.platform.max_angular_velocity);

    free(extracted.platform_specific);
    config_free(&original);
    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Extended Conversion Tests
 * ============================================================================ */

TEST(config_to_params_with_offset) {
    Config cfg;
    config_set_defaults(&cfg);
    Arena* arena = arena_create(65536);
    PlatformParamsSOA* params = platform_params_create(arena, 10, QUAD_PARAMS_EXT_COUNT);

    PlatformConfig custom;
    platform_config_set_defaults(&custom);
    custom.mass = 1.0f;
    ConfigPhysics phys;
    physics_config_set_defaults(&phys);

    platform_config_to_params(&custom, &phys, params, 0, 5);
    platform_config_to_params(&cfg.platform, &cfg.physics, params, 5, 5);

    ASSERT_FLOAT_EQ(params->rigid_body.mass[0], 1.0f);
    ASSERT_FLOAT_EQ(params->rigid_body.mass[4], 1.0f);
    ASSERT_FLOAT_EQ(params->rigid_body.mass[5], 0.027f);
    ASSERT_FLOAT_EQ(params->rigid_body.mass[9], 0.027f);

    arena_destroy(arena);
    return 0;
}

TEST(config_to_params_updates_count) {
    Config cfg;
    config_set_defaults(&cfg);
    Arena* arena = arena_create(65536);
    PlatformParamsSOA* params = platform_params_create(arena, 100, QUAD_PARAMS_EXT_COUNT);

    ASSERT_EQ(params->rigid_body.count, 0u);
    platform_config_to_params(&cfg.platform, &cfg.physics, params, 0, 50);
    ASSERT_EQ(params->rigid_body.count, 50u);
    platform_config_to_params(&cfg.platform, &cfg.physics, params, 70, 10);
    ASSERT_EQ(params->rigid_body.count, 80u);
    /* Writing within existing range does not shrink count */
    platform_config_to_params(&cfg.platform, &cfg.physics, params, 0, 5);
    ASSERT_EQ(params->rigid_body.count, 80u);

    arena_destroy(arena);
    return 0;
}

TEST(config_to_params_capacity_overflow_guard) {
    Config cfg;
    config_set_defaults(&cfg);
    Arena* arena = arena_create(65536);
    PlatformParamsSOA* params = platform_params_create(arena, 10, QUAD_PARAMS_EXT_COUNT);

    platform_config_to_params(&cfg.platform, &cfg.physics, params, 8, 5);
    ASSERT_TRUE(params->rigid_body.count <= params->rigid_body.capacity);

    arena_destroy(arena);
    return 0;
}

TEST(config_to_params_null_safety) {
    Config cfg;
    config_set_defaults(&cfg);
    /* Should not crash with NULL params */
    platform_config_to_params(&cfg.platform, &cfg.physics, NULL, 0, 1);
    platform_config_to_params(NULL, &cfg.physics, NULL, 0, 1);
    platform_config_to_params(&cfg.platform, NULL, NULL, 0, 1);
    return 0;
}

TEST(params_to_config_out_of_bounds_returns_defaults) {
    Arena* arena = arena_create(65536);
    PlatformParamsSOA* params = platform_params_create(arena, 5, QUAD_PARAMS_EXT_COUNT);
    params->rigid_body.count = 3;
    PlatformConfig cfg = platform_params_to_config(params, 10);
    ASSERT_FLOAT_EQ(cfg.mass, 0.027f);
    arena_destroy(arena);
    return 0;
}

TEST(params_to_config_null_returns_defaults) {
    PlatformConfig cfg = platform_params_to_config(NULL, 0);
    ASSERT_FLOAT_EQ(cfg.mass, 0.027f);
    return 0;
}

TEST(params_to_config_extracted_name) {
    Arena* arena = arena_create(65536);
    PlatformParamsSOA* params = platform_params_create(arena, 1, QUAD_PARAMS_EXT_COUNT);
    Config cfg;
    config_set_defaults(&cfg);
    platform_config_to_params(&cfg.platform, &cfg.physics, params, 0, 1);
    PlatformConfig extracted = platform_params_to_config(params, 0);
    ASSERT_STR_EQ(extracted.name, "extracted");
    ASSERT_FLOAT_EQ(extracted.max_tilt_angle, 1.0f);
    ASSERT_FLOAT_EQ(extracted.scale, 1.0f);
    ASSERT_FLOAT_EQ(extracted.color[0], 0.2f);
    arena_destroy(arena);
    return 0;
}

TEST(config_init_platform_params_convenience) {
    Config cfg;
    config_set_defaults(&cfg);
    Arena* arena = arena_create(65536);
    PlatformParamsSOA* params = platform_params_create(arena, 16, QUAD_PARAMS_EXT_COUNT);
    config_init_platform_params(&cfg, params, 16);
    for (uint32_t i = 0; i < 16; i++) {
        ASSERT_FLOAT_EQ(params->rigid_body.mass[i], 0.027f);
        ASSERT_FLOAT_EQ(params->rigid_body.gravity[i], 9.81f);
    }
    ASSERT_EQ(params->rigid_body.count, 16u);
    arena_destroy(arena);
    return 0;
}

TEST(config_init_platform_params_null_safety) {
    Config cfg;
    config_set_defaults(&cfg);
    config_init_platform_params(NULL, NULL, 0);
    config_init_platform_params(&cfg, NULL, 10);
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
    cfg2.platform.mass = 1.0f;
    uint64_t h1 = config_hash(&cfg1);
    uint64_t h2 = config_hash(&cfg2);
    ASSERT_TRUE(h1 != h2);
    config_free(&cfg1);
    config_free(&cfg2);
    return 0;
}

TEST(config_compare_identical) {
    Config cfg1, cfg2;
    config_set_defaults(&cfg1);
    config_set_defaults(&cfg2);
    int result = config_compare(&cfg1, &cfg2);
    ASSERT_TRUE(result == 0);
    config_free(&cfg1);
    config_free(&cfg2);
    return 0;
}

TEST(config_compare_different) {
    Config cfg1, cfg2;
    config_set_defaults(&cfg1);
    config_set_defaults(&cfg2);
    cfg2.platform.mass = 1.0f;
    int result = config_compare(&cfg1, &cfg2);
    ASSERT_TRUE(result != 0);
    config_free(&cfg1);
    config_free(&cfg2);
    return 0;
}

TEST(config_clone_test) {
    Config src;
    config_set_defaults(&src);
    src.platform.mass = 0.5f;
    src.sensors = (SensorConfigEntry*)malloc(sizeof(SensorConfigEntry));
    src.num_sensors = 1;
    memset(src.sensors, 0, sizeof(SensorConfigEntry));
    strcpy(src.sensors[0].type, "imu");

    Arena* arena = arena_create(4096);
    Config dst;
    config_clone(&src, &dst, arena);

    ASSERT_FLOAT_EQ(dst.platform.mass, 0.5f);
    ASSERT_EQ(dst.num_sensors, 1);
    ASSERT_STR_EQ(dst.sensors[0].type, "imu");
    ASSERT_TRUE(dst.sensors != src.sensors);

    config_free(&src);
    arena_destroy(arena);
    return 0;
}

TEST(config_to_json_test) {
    Config cfg;
    config_set_defaults(&cfg);
    cfg.config_hash = config_hash(&cfg);

    char buffer[1024];
    int result = config_to_json(&cfg, buffer, sizeof(buffer));

    ASSERT_TRUE(result == 0);
    ASSERT_TRUE(strstr(buffer, "\"platform\"") != NULL);
    ASSERT_TRUE(strstr(buffer, "crazyflie2") != NULL);
    ASSERT_TRUE(strstr(buffer, "\"num_envs\": 64") != NULL);
    config_free(&cfg);
    return 0;
}

/* ============================================================================
 * Extended Serialization Tests
 * ============================================================================ */

TEST(hash_null_returns_zero) {
    uint64_t h = config_hash(NULL);
    ASSERT_TRUE(h == 0);
    return 0;
}

TEST(hash_nonzero_for_defaults) {
    Config cfg;
    config_set_defaults(&cfg);
    uint64_t h = config_hash(&cfg);
    ASSERT_TRUE(h != 0);
    return 0;
}

TEST(hash_sensitive_to_physics_change) {
    Config cfg1, cfg2;
    config_set_defaults(&cfg1);
    config_set_defaults(&cfg2);
    cfg2.physics.gravity = 3.72f;
    ASSERT_TRUE(config_hash(&cfg1) != config_hash(&cfg2));
    return 0;
}

TEST(hash_sensitive_to_environment_change) {
    Config cfg1, cfg2;
    config_set_defaults(&cfg1);
    config_set_defaults(&cfg2);
    cfg2.environment.seed = 99;
    ASSERT_TRUE(config_hash(&cfg1) != config_hash(&cfg2));
    return 0;
}

TEST(hash_sensitive_to_reward_change) {
    Config cfg1, cfg2;
    config_set_defaults(&cfg1);
    config_set_defaults(&cfg2);
    cfg2.reward.alive_bonus = 1.0f;
    ASSERT_TRUE(config_hash(&cfg1) != config_hash(&cfg2));
    return 0;
}

TEST(hash_sensitive_to_training_change) {
    Config cfg1, cfg2;
    config_set_defaults(&cfg1);
    config_set_defaults(&cfg2);
    cfg2.training.batch_size = 512;
    ASSERT_TRUE(config_hash(&cfg1) != config_hash(&cfg2));
    return 0;
}

TEST(hash_deterministic_across_calls) {
    Config cfg;
    config_set_defaults(&cfg);
    uint64_t h1 = config_hash(&cfg);
    uint64_t h2 = config_hash(&cfg);
    uint64_t h3 = config_hash(&cfg);
    ASSERT_TRUE(h1 == h2);
    ASSERT_TRUE(h2 == h3);
    return 0;
}

TEST(config_compare_null_returns_nonzero) {
    Config cfg;
    config_set_defaults(&cfg);
    ASSERT_TRUE(config_compare(NULL, &cfg) != 0);
    ASSERT_TRUE(config_compare(&cfg, NULL) != 0);
    ASSERT_TRUE(config_compare(NULL, NULL) != 0);
    return 0;
}

TEST(config_compare_sensor_count_mismatch) {
    Config cfg1, cfg2;
    config_set_defaults(&cfg1);
    config_set_defaults(&cfg2);
    cfg1.sensors = (SensorConfigEntry*)malloc(sizeof(SensorConfigEntry));
    cfg1.num_sensors = 1;
    memset(cfg1.sensors, 0, sizeof(SensorConfigEntry));
    int result = config_compare(&cfg1, &cfg2);
    ASSERT_TRUE(result != 0);
    config_free(&cfg1);
    return 0;
}

TEST(config_compare_sensor_content_mismatch) {
    Config cfg1, cfg2;
    config_set_defaults(&cfg1);
    config_set_defaults(&cfg2);

    cfg1.sensors = (SensorConfigEntry*)malloc(sizeof(SensorConfigEntry));
    cfg1.num_sensors = 1;
    cfg1.sensors[0] = sensor_config_entry_default("imu");

    cfg2.sensors = (SensorConfigEntry*)malloc(sizeof(SensorConfigEntry));
    cfg2.num_sensors = 1;
    cfg2.sensors[0] = sensor_config_entry_default("tof");

    ASSERT_TRUE(config_compare(&cfg1, &cfg2) != 0);
    config_free(&cfg1);
    config_free(&cfg2);
    return 0;
}

TEST(config_compare_sensors_identical) {
    Config cfg1, cfg2;
    config_set_defaults(&cfg1);
    config_set_defaults(&cfg2);

    cfg1.sensors = (SensorConfigEntry*)malloc(sizeof(SensorConfigEntry));
    cfg1.num_sensors = 1;
    cfg1.sensors[0] = sensor_config_entry_default("imu");

    cfg2.sensors = (SensorConfigEntry*)malloc(sizeof(SensorConfigEntry));
    cfg2.num_sensors = 1;
    cfg2.sensors[0] = sensor_config_entry_default("imu");

    ASSERT_TRUE(config_compare(&cfg1, &cfg2) == 0);
    config_free(&cfg1);
    config_free(&cfg2);
    return 0;
}

TEST(config_clone_without_sensors) {
    Config src;
    config_set_defaults(&src);
    src.platform.mass = 1.0f;
    Arena* arena = arena_create(4096);
    Config dst;
    config_clone(&src, &dst, arena);
    ASSERT_FLOAT_EQ(dst.platform.mass, 1.0f);
    ASSERT_NULL(dst.sensors);
    ASSERT_EQ(dst.num_sensors, 0u);
    ASSERT_TRUE(dst.config_path[0] == '\0');
    config_free(&src);
    arena_destroy(arena);
    return 0;
}

TEST(config_clone_null_arena_uses_malloc) {
    Config src;
    config_set_defaults(&src);
    src.sensors = (SensorConfigEntry*)malloc(sizeof(SensorConfigEntry));
    src.num_sensors = 1;
    src.sensors[0] = sensor_config_entry_default("tof");

    Config dst;
    config_clone(&src, &dst, NULL);

    ASSERT_EQ(dst.num_sensors, 1u);
    ASSERT_NOT_NULL(dst.sensors);
    ASSERT_STR_EQ(dst.sensors[0].type, "tof");
    ASSERT_TRUE(dst.sensors != src.sensors);

    config_free(&src);
    free(dst.sensors);
    return 0;
}

TEST(config_clone_recomputes_hash) {
    Config src;
    config_set_defaults(&src);
    src.config_hash = 12345;
    Arena* arena = arena_create(4096);
    Config dst;
    config_clone(&src, &dst, arena);
    uint64_t expected = config_hash(&dst);
    ASSERT_TRUE(dst.config_hash == expected);
    arena_destroy(arena);
    return 0;
}

TEST(config_to_json_buffer_too_small) {
    Config cfg;
    config_set_defaults(&cfg);
    char buffer[16];
    int result = config_to_json(&cfg, buffer, sizeof(buffer));
    ASSERT_TRUE(result == -1);
    return 0;
}

TEST(config_to_json_null_safety) {
    Config cfg;
    config_set_defaults(&cfg);
    char buffer[1024];
    ASSERT_TRUE(config_to_json(NULL, buffer, sizeof(buffer)) == -1);
    ASSERT_TRUE(config_to_json(&cfg, NULL, sizeof(buffer)) == -1);
    ASSERT_TRUE(config_to_json(&cfg, buffer, 0) == -1);
    return 0;
}

TEST(config_to_json_contains_all_sections) {
    Config cfg;
    config_set_defaults(&cfg);
    cfg.config_hash = config_hash(&cfg);
    char buffer[2048];
    int result = config_to_json(&cfg, buffer, sizeof(buffer));
    ASSERT_TRUE(result == 0);
    ASSERT_TRUE(strstr(buffer, "\"platform\"") != NULL);
    ASSERT_TRUE(strstr(buffer, "\"environment\"") != NULL);
    ASSERT_TRUE(strstr(buffer, "\"physics\"") != NULL);
    ASSERT_TRUE(strstr(buffer, "\"reward\"") != NULL);
    ASSERT_TRUE(strstr(buffer, "\"training\"") != NULL);
    ASSERT_TRUE(strstr(buffer, "\"num_sensors\"") != NULL);
    ASSERT_TRUE(strstr(buffer, "\"config_hash\"") != NULL);
    config_free(&cfg);
    return 0;
}

/* ============================================================================
 * config_memory_size Tests
 * ============================================================================ */

TEST(config_memory_size_zero_sensors) {
    size_t sz = config_memory_size(0);
    ASSERT_EQ(sz, sizeof(Config));
    return 0;
}

TEST(config_memory_size_with_sensors) {
    size_t sz = config_memory_size(5);
    size_t expected = sizeof(Config) + 5 * sizeof(SensorConfigEntry);
    ASSERT_EQ(sz, expected);
    return 0;
}

TEST(config_memory_size_max_sensors) {
    size_t sz = config_memory_size(CONFIG_MAX_SENSORS);
    size_t expected = sizeof(Config) + CONFIG_MAX_SENSORS * sizeof(SensorConfigEntry);
    ASSERT_EQ(sz, expected);
    return 0;
}

/* ============================================================================
 * config_free Tests
 * ============================================================================ */

TEST(config_free_null_safe) {
    config_free(NULL);
    return 0;
}

TEST(config_free_no_sensors) {
    Config cfg;
    config_set_defaults(&cfg);
    config_free(&cfg);
    ASSERT_NULL(cfg.sensors);
    ASSERT_EQ(cfg.num_sensors, 0u);
    return 0;
}

TEST(config_free_clears_pointer) {
    Config cfg;
    config_set_defaults(&cfg);
    cfg.sensors = (SensorConfigEntry*)malloc(sizeof(SensorConfigEntry));
    cfg.num_sensors = 1;
    config_free(&cfg);
    ASSERT_NULL(cfg.sensors);
    ASSERT_EQ(cfg.num_sensors, 0u);
    return 0;
}

TEST(config_free_double_free_safe) {
    Config cfg;
    config_set_defaults(&cfg);
    cfg.sensors = (SensorConfigEntry*)malloc(sizeof(SensorConfigEntry));
    cfg.num_sensors = 1;
    config_free(&cfg);
    config_free(&cfg);
    ASSERT_NULL(cfg.sensors);
    return 0;
}

/* ============================================================================
 * Round-trip and Consistency Tests
 * ============================================================================ */

TEST(defaults_validate_roundtrip) {
    Config cfg;
    config_set_defaults(&cfg);
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);
    ASSERT_EQ(n, 0);
    return 0;
}

TEST(parse_validate_roundtrip) {
    const char* toml =
        "[drone]\nmass = 0.5\narm_length = 0.1\n"
        "[physics]\nintegrator = \"euler\"\n"
        "[environment]\nworld_type = \"empty\"\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);
    ASSERT_TRUE(result == 0);

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);
    ASSERT_EQ(n, 0);

    config_free(&cfg);
    return 0;
}

TEST(parse_hash_consistency) {
    const char* toml = "[drone]\nmass = 0.5\n";
    Config cfg1, cfg2;
    char err1[256] = {0};
    char err2[256] = {0};
    config_load_string(toml, &cfg1, err1);
    config_load_string(toml, &cfg2, err2);
    ASSERT_TRUE(cfg1.config_hash == cfg2.config_hash);
    ASSERT_TRUE(config_compare(&cfg1, &cfg2) == 0);
    config_free(&cfg1);
    config_free(&cfg2);
    return 0;
}

TEST(config_set_defaults_idempotent) {
    Config cfg1, cfg2;
    config_set_defaults(&cfg1);
    config_set_defaults(&cfg1);
    config_set_defaults(&cfg2);
    ASSERT_TRUE(config_compare(&cfg1, &cfg2) == 0);
    return 0;
}

TEST(subsection_defaults_match_full_defaults) {
    Config full;
    config_set_defaults(&full);

    /* PlatformConfig: can't memcmp due to platform_specific pointer; compare key fields */
    PlatformConfig plat;
    platform_config_set_defaults(&plat);
    ASSERT_STR_EQ(plat.name, full.platform.name);
    ASSERT_FLOAT_EQ(plat.mass, full.platform.mass);
    ASSERT_FLOAT_EQ(plat.ixx, full.platform.ixx);
    ASSERT_FLOAT_EQ(plat.collision_radius, full.platform.collision_radius);
    ASSERT_FLOAT_EQ(plat.max_velocity, full.platform.max_velocity);
    ASSERT_EQ(plat.platform_specific_size, full.platform.platform_specific_size);
    if (plat.platform_specific_size > 0) {
        ASSERT_TRUE(memcmp(plat.platform_specific, full.platform.platform_specific,
                           plat.platform_specific_size) == 0);
    }
    free(plat.platform_specific);

    EnvironmentConfig ec;
    environment_config_set_defaults(&ec);
    ASSERT_TRUE(memcmp(&ec, &full.environment, sizeof(EnvironmentConfig)) == 0);

    ConfigPhysics phys;
    physics_config_set_defaults(&phys);
    ASSERT_TRUE(memcmp(&phys, &full.physics, sizeof(ConfigPhysics)) == 0);

    RewardConfigData rc;
    reward_config_data_set_defaults(&rc);
    ASSERT_TRUE(memcmp(&rc, &full.reward, sizeof(RewardConfigData)) == 0);

    TrainingConfig tc;
    training_config_set_defaults(&tc);
    ASSERT_TRUE(memcmp(&tc, &full.training, sizeof(TrainingConfig)) == 0);

    config_free(&full);
    return 0;
}

/* ============================================================================
 * Struct Size Verification
 * ============================================================================ */

TEST(struct_sizes_within_limits) {
    ASSERT_TRUE(sizeof(PlatformConfig) < 512);
    ASSERT_TRUE(sizeof(EnvironmentConfig) < 256);
    ASSERT_TRUE(sizeof(ConfigPhysics) < 128);
    ASSERT_TRUE(sizeof(SensorConfigEntry) < 4096);
    ASSERT_TRUE(sizeof(Config) < 2048);
    return 0;
}

/* ============================================================================
 * File I/O Error Tests
 * ============================================================================ */

TEST(config_load_nonexistent_file) {
    Config cfg;
    char error_msg[256] = {0};
    int result = config_load("/nonexistent/path/config.toml", &cfg, error_msg);
    ASSERT_TRUE(result == -1);
    ASSERT_TRUE(strlen(error_msg) > 0);
    ASSERT_TRUE(strstr(error_msg, "Cannot open") != NULL);
    return 0;
}

TEST(config_load_null_path) {
    Config cfg;
    char error_msg[256] = {0};
    int result = config_load(NULL, &cfg, error_msg);
    ASSERT_TRUE(result == -1);
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

    /* Extended Default Value Tests */
    RUN_TEST(default_drone_all_fields_exhaustive);
    RUN_TEST(default_environment_all_fields_exhaustive);
    RUN_TEST(default_physics_all_fields_exhaustive);
    RUN_TEST(default_reward_all_fields_exhaustive);
    RUN_TEST(default_training_all_fields_exhaustive);

    /* Sensor Default Tests */
    RUN_TEST(default_sensor_imu);
    RUN_TEST(default_sensor_tof);
    RUN_TEST(default_sensor_lidar_2d);
    RUN_TEST(default_sensor_lidar_3d);
    RUN_TEST(default_sensor_camera_rgb);
    RUN_TEST(default_sensor_camera_depth);
    RUN_TEST(default_sensor_camera_segmentation);
    RUN_TEST(default_sensor_position);
    RUN_TEST(default_sensor_velocity);
    RUN_TEST(default_sensor_neighbor);
    RUN_TEST(default_sensor_unknown_type_gets_zero_fields);

    /* Sensor Type Registry Tests */
    RUN_TEST(sensor_type_all_valid_types_recognized);
    RUN_TEST(sensor_type_rejects_invalid_strings);

    /* Parsing Tests */
    RUN_TEST(parse_empty_string);
    RUN_TEST(parse_minimal_valid);
    RUN_TEST(parse_all_drone_fields);
    RUN_TEST(parse_sensor_array);
    RUN_TEST(parse_noise_groups);
    RUN_TEST(parse_scientific_notation);
    RUN_TEST(parse_integer_as_float);

    /* Extended Parsing Tests */
    RUN_TEST(parse_malformed_toml_returns_error);
    RUN_TEST(parse_null_string_returns_error);
    RUN_TEST(parse_null_config_returns_error);
    RUN_TEST(parse_null_error_msg_returns_error);
    RUN_TEST(parse_all_environment_fields);
    RUN_TEST(parse_all_physics_fields);
    RUN_TEST(parse_all_reward_fields);
    RUN_TEST(parse_all_training_fields);
    RUN_TEST(parse_multi_section_config);
    RUN_TEST(parse_sensor_inherits_type_defaults);
    RUN_TEST(parse_sensor_overrides_type_defaults);
    RUN_TEST(parse_validation_failure_resets_to_defaults);
    RUN_TEST(parse_boolean_values);
    RUN_TEST(parse_sets_config_hash_nonzero);
    RUN_TEST(parse_noise_stage_all_types);
    RUN_TEST(parse_noise_power_defaults_to_one);
    RUN_TEST(parse_sensor_position_and_orientation);

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

    /* Extended Drone Validation */
    RUN_TEST(validate_drone_arm_length_zero);
    RUN_TEST(validate_drone_inertia_iyy_zero);
    RUN_TEST(validate_drone_inertia_izz_zero);
    RUN_TEST(validate_drone_k_thrust_zero);
    RUN_TEST(validate_drone_motor_tau_zero);
    RUN_TEST(validate_drone_max_rpm_zero);
    RUN_TEST(validate_drone_collision_radius_zero);
    RUN_TEST(validate_drone_k_drag_negative);
    RUN_TEST(validate_drone_k_drag_zero_is_valid);
    RUN_TEST(validate_drone_max_velocity_zero);
    RUN_TEST(validate_drone_color_out_of_range);
    RUN_TEST(validate_drone_color_negative);
    RUN_TEST(validate_drone_color_boundary_valid);
    RUN_TEST(validate_drone_scale_zero);
    RUN_TEST(validate_drone_defaults_pass);

    /* Extended Environment Validation */
    RUN_TEST(validate_environment_world_size_zero);
    RUN_TEST(validate_environment_voxel_size_zero);
    RUN_TEST(validate_environment_spawn_radius_negative);
    RUN_TEST(validate_environment_min_separation_negative);
    RUN_TEST(validate_environment_max_episode_steps_zero);
    RUN_TEST(validate_environment_world_type_unknown);
    RUN_TEST(validate_environment_all_valid_world_types);
    RUN_TEST(validate_environment_spawn_height_equal);
    RUN_TEST(validate_environment_defaults_pass);

    /* Extended Physics Validation */
    RUN_TEST(validate_physics_timestep_negative);
    RUN_TEST(validate_physics_timestep_too_large);
    RUN_TEST(validate_physics_timestep_boundary_valid);
    RUN_TEST(validate_physics_euler_valid);
    RUN_TEST(validate_physics_substeps_zero);
    RUN_TEST(validate_physics_gravity_negative);
    RUN_TEST(validate_physics_gravity_zero_valid);
    RUN_TEST(validate_physics_velocity_clamp_zero);
    RUN_TEST(validate_physics_ground_effect_height_zero_when_enabled);
    RUN_TEST(validate_physics_ground_effect_strength_below_one);
    RUN_TEST(validate_physics_ground_effect_not_checked_when_disabled);
    RUN_TEST(validate_physics_dt_variance_negative);
    RUN_TEST(validate_physics_mass_variance_negative);
    RUN_TEST(validate_physics_defaults_pass);

    /* Extended Sensor Validation */
    RUN_TEST(validate_sensor_quaternion_not_normalized);
    RUN_TEST(validate_sensor_quaternion_zero);
    RUN_TEST(validate_sensor_sample_rate_negative);
    RUN_TEST(validate_sensor_tof_max_range_zero);
    RUN_TEST(validate_sensor_lidar_2d_zero_rays);
    RUN_TEST(validate_sensor_lidar_2d_zero_fov);
    RUN_TEST(validate_sensor_camera_zero_dimensions);
    RUN_TEST(validate_sensor_camera_near_geq_far);
    RUN_TEST(validate_sensor_neighbor_k_zero);
    RUN_TEST(validate_sensor_defaults_all_pass);
    RUN_TEST(validate_sensors_null_array_zero_count);
    RUN_TEST(validate_respects_max_errors_limit);

    /* Conversion Tests */
    RUN_TEST(config_to_params_single);
    RUN_TEST(config_to_params_broadcast);
    RUN_TEST(params_to_config_roundtrip);

    /* Extended Conversion Tests */
    RUN_TEST(config_to_params_with_offset);
    RUN_TEST(config_to_params_updates_count);
    RUN_TEST(config_to_params_capacity_overflow_guard);
    RUN_TEST(config_to_params_null_safety);
    RUN_TEST(params_to_config_out_of_bounds_returns_defaults);
    RUN_TEST(params_to_config_null_returns_defaults);
    RUN_TEST(params_to_config_extracted_name);
    RUN_TEST(config_init_platform_params_convenience);
    RUN_TEST(config_init_platform_params_null_safety);

    /* Serialization Tests */
    RUN_TEST(hash_same_config);
    RUN_TEST(hash_different_config);
    RUN_TEST(config_compare_identical);
    RUN_TEST(config_compare_different);
    RUN_TEST(config_clone_test);
    RUN_TEST(config_to_json_test);

    /* Extended Serialization Tests */
    RUN_TEST(hash_null_returns_zero);
    RUN_TEST(hash_nonzero_for_defaults);
    RUN_TEST(hash_sensitive_to_physics_change);
    RUN_TEST(hash_sensitive_to_environment_change);
    RUN_TEST(hash_sensitive_to_reward_change);
    RUN_TEST(hash_sensitive_to_training_change);
    RUN_TEST(hash_deterministic_across_calls);
    RUN_TEST(config_compare_null_returns_nonzero);
    RUN_TEST(config_compare_sensor_count_mismatch);
    RUN_TEST(config_compare_sensor_content_mismatch);
    RUN_TEST(config_compare_sensors_identical);
    RUN_TEST(config_clone_without_sensors);
    RUN_TEST(config_clone_null_arena_uses_malloc);
    RUN_TEST(config_clone_recomputes_hash);
    RUN_TEST(config_to_json_buffer_too_small);
    RUN_TEST(config_to_json_null_safety);
    RUN_TEST(config_to_json_contains_all_sections);

    /* Memory Size Tests */
    RUN_TEST(config_memory_size_zero_sensors);
    RUN_TEST(config_memory_size_with_sensors);
    RUN_TEST(config_memory_size_max_sensors);

    /* config_free Tests */
    RUN_TEST(config_free_null_safe);
    RUN_TEST(config_free_no_sensors);
    RUN_TEST(config_free_clears_pointer);
    RUN_TEST(config_free_double_free_safe);

    /* Round-trip and Consistency Tests */
    RUN_TEST(defaults_validate_roundtrip);
    RUN_TEST(parse_validate_roundtrip);
    RUN_TEST(parse_hash_consistency);
    RUN_TEST(config_set_defaults_idempotent);
    RUN_TEST(subsection_defaults_match_full_defaults);

    /* Struct Size Verification */
    RUN_TEST(struct_sizes_within_limits);

    /* File I/O Error Tests */
    RUN_TEST(config_load_nonexistent_file);
    RUN_TEST(config_load_null_path);

    TEST_SUITE_END();
}
