/**
 * Deep Configuration Validation Tests (Yoneda Philosophy)
 *
 * These tests serve as the authoritative behavioral specification for the
 * configuration module. Following the Yoneda lemma's principle, we
 * characterize the module completely by exploring every possible interaction:
 * every input category, edge case, boundary condition, and error path.
 *
 * Organization:
 *   1. Three-phase validation pipeline end-to-end
 *   2. Sensor array configs at limits
 *   3. config_load_string edge cases
 *   4. Semantic validation (conflicting physics, negative mass, etc.)
 *   5. config_hash determinism and variation
 *   6. platform_config_to_params conversion completeness
 *   7. Config defaults produce valid config
 *   8. Field boundary values
 *   9. Validation error reporting fidelity
 *  10. Cross-section semantic consistency
 */

#include "test_harness.h"
#include "configuration.h"
#include "drone_state.h"
#include "platform_quadcopter.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>

/* ============================================================================
 * 1. Three-Phase Validation Pipeline End-to-End
 *
 * The pipeline is: parse TOML -> validate schema -> validate semantics.
 * config_load_string runs all three in sequence: it calls config_set_defaults,
 * then toml_parse, then config_validate. We verify the full chain.
 * ============================================================================ */

TEST(pipeline_valid_full_config_passes_all_phases) {
    /* A fully-specified TOML config should parse, validate schema, and
     * validate semantics without errors. */
    const char* toml =
        "[drone]\n"
        "name = \"test\"\n"
        "mass = 0.5\n"
        "arm_length = 0.1\n"
        "ixx = 1e-4\n"
        "iyy = 1e-4\n"
        "izz = 2e-4\n"
        "k_thrust = 3e-8\n"
        "k_torque = 8e-10\n"
        "motor_tau = 0.03\n"
        "max_rpm = 3000.0\n"
        "collision_radius = 0.12\n"
        "max_velocity = 15.0\n"
        "max_angular_velocity = 40.0\n"
        "scale = 2.0\n"
        "color = [0.5, 0.5, 0.5]\n"
        "\n"
        "[environment]\n"
        "num_envs = 8\n"
        "agents_per_env = 4\n"
        "world_size = [10.0, 10.0, 5.0]\n"
        "voxel_size = 0.2\n"
        "spawn_radius = 3.0\n"
        "spawn_height_min = 1.0\n"
        "spawn_height_max = 4.0\n"
        "max_episode_steps = 500\n"
        "world_type = \"empty\"\n"
        "\n"
        "[physics]\n"
        "timestep = 0.01\n"
        "substeps = 2\n"
        "gravity = 9.81\n"
        "integrator = \"euler\"\n"
        "velocity_clamp = 15.0\n"
        "angular_velocity_clamp = 25.0\n"
        "\n"
        "[[sensors]]\n"
        "type = \"imu\"\n"
        "name = \"main_imu\"\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_EQ(result, 0);
    /* Verify parse phase worked (values overridden from defaults) */
    ASSERT_STR_EQ(cfg.platform.name, "test");
    ASSERT_FLOAT_EQ(cfg.platform.mass, 0.5f);
    ASSERT_EQ(cfg.environment.num_envs, (uint32_t)8);
    ASSERT_STR_EQ(cfg.physics.integrator, "euler");
    ASSERT_EQ(cfg.num_sensors, (uint32_t)1);

    /* Verify validation phase passed (hash was computed, meaning no errors) */
    ASSERT_TRUE(cfg.config_hash != 0);

    /* Verify the error message is empty (no errors) */
    ASSERT_EQ(error_msg[0], '\0');

    config_free(&cfg);
    return 0;
}

TEST(pipeline_parse_failure_returns_error) {
    /* Malformed TOML should fail at parse phase (return -2) */
    const char* toml = "[drone\nmass = bad_value\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    /* Parse error returns -2 */
    ASSERT_EQ(result, -2);
    ASSERT_TRUE(strlen(error_msg) > 0);
    ASSERT_TRUE(strstr(error_msg, "TOML parse error") != NULL);

    config_free(&cfg);
    return 0;
}

TEST(pipeline_semantic_failure_returns_error) {
    /* Valid TOML syntax but semantically invalid (zero mass) should fail
     * at validation phase (return -4) */
    const char* toml = "[drone]\nmass = 0.0\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    /* Validation error returns -4 */
    ASSERT_EQ(result, -4);
    ASSERT_TRUE(strlen(error_msg) > 0);
    ASSERT_TRUE(strstr(error_msg, "mass") != NULL);

    /* After validation failure, config should be reset to defaults */
    ASSERT_FLOAT_EQ(cfg.platform.mass, 0.027f);

    config_free(&cfg);
    return 0;
}

TEST(pipeline_validation_resets_to_defaults_on_failure) {
    /* When validation fails, config_load_string calls config_free + config_set_defaults.
     * The returned config should be valid defaults, not the invalid parsed values. */
    const char* toml = "[drone]\nmass = -5.0\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_EQ(result, -4);

    /* Verify config was reset to defaults after validation failure */
    ConfigError errors[CONFIG_MAX_ERRORS];
    int num_errors = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);
    ASSERT_EQ(num_errors, 0);

    config_free(&cfg);
    return 0;
}

/* ============================================================================
 * 2. Sensor Array Configs at Limits
 *
 * CONFIG_MAX_SENSORS = 32. Test 0, 1, 32, and 33+ sensors.
 * ============================================================================ */

TEST(sensor_count_zero) {
    /* Config with no sensors section should have 0 sensors */
    const char* toml = "[drone]\nmass = 0.5\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_EQ(result, 0);
    ASSERT_EQ(cfg.num_sensors, (uint32_t)0);
    ASSERT_NULL(cfg.sensors);

    config_free(&cfg);
    return 0;
}

TEST(sensor_count_one) {
    const char* toml =
        "[[sensors]]\n"
        "type = \"imu\"\n"
        "name = \"sole_sensor\"\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_EQ(result, 0);
    ASSERT_EQ(cfg.num_sensors, (uint32_t)1);
    ASSERT_NOT_NULL(cfg.sensors);
    ASSERT_STR_EQ(cfg.sensors[0].name, "sole_sensor");

    config_free(&cfg);
    return 0;
}

TEST(sensor_count_at_max) {
    /* Build a TOML string with exactly CONFIG_MAX_SENSORS (32) sensors */
    char toml[8192];
    int offset = 0;
    for (uint32_t i = 0; i < CONFIG_MAX_SENSORS; i++) {
        offset += snprintf(toml + offset, sizeof(toml) - (size_t)offset,
                           "[[sensors]]\ntype = \"imu\"\nname = \"imu_%u\"\n\n",
                           i);
    }

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_EQ(result, 0);
    ASSERT_EQ(cfg.num_sensors, CONFIG_MAX_SENSORS);

    /* Verify first and last sensor names */
    ASSERT_STR_EQ(cfg.sensors[0].name, "imu_0");
    char expected_last[64];
    snprintf(expected_last, sizeof(expected_last), "imu_%u", CONFIG_MAX_SENSORS - 1);
    ASSERT_STR_EQ(cfg.sensors[CONFIG_MAX_SENSORS - 1].name, expected_last);

    config_free(&cfg);
    return 0;
}

TEST(sensor_count_over_max_is_clamped) {
    /* Build a TOML string with CONFIG_MAX_SENSORS + 1 sensors.
     * The parser should clamp to CONFIG_MAX_SENSORS. */
    char toml[8192];
    int offset = 0;
    for (uint32_t i = 0; i < CONFIG_MAX_SENSORS + 1; i++) {
        offset += snprintf(toml + offset, sizeof(toml) - (size_t)offset,
                           "[[sensors]]\ntype = \"imu\"\nname = \"imu_%u\"\n\n",
                           i);
    }

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_EQ(result, 0);
    /* Parser clamps to CONFIG_MAX_SENSORS */
    ASSERT_EQ(cfg.num_sensors, CONFIG_MAX_SENSORS);

    config_free(&cfg);
    return 0;
}

TEST(sensor_validation_zero_count_no_errors) {
    /* Validating zero sensors should produce no errors */
    ConfigError errors[CONFIG_MAX_ERRORS];
    int num_errors = config_validate_sensors(NULL, 0, errors, CONFIG_MAX_ERRORS);
    ASSERT_EQ(num_errors, 0);

    return 0;
}

/* ============================================================================
 * 3. config_load_string Edge Cases
 * ============================================================================ */

TEST(load_string_null_toml) {
    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(NULL, &cfg, error_msg);

    ASSERT_EQ(result, -1);

    return 0;
}

TEST(load_string_null_config) {
    char error_msg[256] = {0};
    int result = config_load_string("", NULL, error_msg);

    ASSERT_EQ(result, -1);

    return 0;
}

TEST(load_string_null_error_msg) {
    Config cfg;
    memset(&cfg, 0, sizeof(cfg));
    int result = config_load_string("", &cfg, NULL);

    ASSERT_EQ(result, -1);

    config_free(&cfg);
    return 0;
}

TEST(load_string_empty_returns_defaults) {
    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string("", &cfg, error_msg);

    ASSERT_EQ(result, 0);
    /* Empty string => defaults are used */
    ASSERT_FLOAT_EQ(cfg.platform.mass, 0.027f);
    ASSERT_EQ(cfg.environment.num_envs, (uint32_t)64);
    ASSERT_FLOAT_EQ(cfg.physics.timestep, 0.02f);
    ASSERT_STR_EQ(cfg.reward.task, "hover");
    ASSERT_STR_EQ(cfg.training.algorithm, "ppo");

    /* Hash should still be computed */
    ASSERT_TRUE(cfg.config_hash != 0);

    config_free(&cfg);
    return 0;
}

TEST(load_string_whitespace_only) {
    /* Whitespace-only string is valid TOML (empty document) but tomlc99
     * may or may not parse it. Let's verify behavior. */
    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string("   \n\n  \t  \n", &cfg, error_msg);

    /* Should succeed (valid empty TOML) with defaults */
    ASSERT_EQ(result, 0);
    ASSERT_FLOAT_EQ(cfg.platform.mass, 0.027f);

    config_free(&cfg);
    return 0;
}

TEST(load_string_comment_only) {
    const char* toml = "# This is a comment\n# Another comment\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_EQ(result, 0);
    /* All defaults */
    ASSERT_FLOAT_EQ(cfg.platform.mass, 0.027f);

    config_free(&cfg);
    return 0;
}

TEST(load_string_unknown_section_ignored) {
    /* Unknown TOML sections should be silently ignored, not cause errors.
     * Only recognized sections are extracted. */
    const char* toml =
        "[unknown_section]\n"
        "foo = 42\n"
        "bar = \"baz\"\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_EQ(result, 0);
    /* Defaults remain */
    ASSERT_FLOAT_EQ(cfg.platform.mass, 0.027f);

    config_free(&cfg);
    return 0;
}

TEST(load_string_partial_override) {
    /* Only override a few fields; rest should keep defaults */
    const char* toml =
        "[drone]\n"
        "mass = 1.0\n"
        "[environment]\n"
        "num_envs = 128\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_EQ(result, 0);
    /* Overridden fields */
    ASSERT_FLOAT_EQ(cfg.platform.mass, 1.0f);
    ASSERT_EQ(cfg.environment.num_envs, (uint32_t)128);
    /* Non-overridden fields keep defaults */
    const QuadcopterConfig* q = (const QuadcopterConfig*)cfg.platform.platform_specific;
    ASSERT_FLOAT_EQ(q->arm_length, 0.046f);
    ASSERT_FLOAT_NEAR(cfg.platform.ixx, 1.4e-5f, 1e-8f);
    ASSERT_EQ(cfg.physics.substeps, (uint32_t)4);
    ASSERT_STR_EQ(cfg.physics.integrator, "rk4");

    config_free(&cfg);
    return 0;
}

/* ============================================================================
 * 4. Semantic Validation: Conflicting/Invalid Physics Parameters
 *
 * We test each validation rule in config_validate_physics,
 * config_validate_platform, config_validate_environment, and
 * config_validate_sensors independently to verify they detect
 * the exact conditions they should.
 * ============================================================================ */

TEST(validate_physics_timestep_zero) {
    ConfigPhysics physics;
    physics_config_set_defaults(&physics);
    physics.timestep = 0.0f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&physics, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    ASSERT_STR_EQ(errors[0].field, "physics.timestep");
    ASSERT_TRUE(strstr(errors[0].message, "positive") != NULL);

    return 0;
}

TEST(validate_physics_timestep_negative) {
    ConfigPhysics physics;
    physics_config_set_defaults(&physics);
    physics.timestep = -0.01f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&physics, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    ASSERT_STR_EQ(errors[0].field, "physics.timestep");

    return 0;
}

TEST(validate_physics_timestep_too_large) {
    /* Timestep > 0.1s is rejected for stability */
    ConfigPhysics physics;
    physics_config_set_defaults(&physics);
    physics.timestep = 0.2f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&physics, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    ASSERT_STR_EQ(errors[0].field, "physics.timestep");
    ASSERT_TRUE(strstr(errors[0].message, "stability") != NULL);

    return 0;
}

TEST(validate_physics_timestep_boundary_at_0_1) {
    /* Exactly 0.1 should be valid (check is > 0.1, not >=) */
    ConfigPhysics physics;
    physics_config_set_defaults(&physics);
    physics.timestep = 0.1f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&physics, errors, CONFIG_MAX_ERRORS);

    /* All physics errors should not include timestep */
    for (int i = 0; i < n; i++) {
        ASSERT_TRUE(strcmp(errors[i].field, "physics.timestep") != 0);
    }

    return 0;
}

TEST(validate_physics_substeps_zero) {
    /* substeps < 1 is invalid. Since substeps is uint32_t, substeps=0 is the
     * only way to violate substeps < 1. */
    ConfigPhysics physics;
    physics_config_set_defaults(&physics);
    physics.substeps = 0;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&physics, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    /* Find the substeps error */
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "substeps") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_physics_negative_gravity) {
    ConfigPhysics physics;
    physics_config_set_defaults(&physics);
    physics.gravity = -9.81f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&physics, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "gravity") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_physics_zero_gravity_is_valid) {
    /* Zero gravity (microgravity) should be accepted: check is gravity < 0 */
    ConfigPhysics physics;
    physics_config_set_defaults(&physics);
    physics.gravity = 0.0f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&physics, errors, CONFIG_MAX_ERRORS);

    /* No gravity-related errors */
    for (int i = 0; i < n; i++) {
        ASSERT_TRUE(strstr(errors[i].field, "gravity") == NULL);
    }

    return 0;
}

TEST(validate_physics_invalid_integrator) {
    ConfigPhysics physics;
    physics_config_set_defaults(&physics);
    strncpy(physics.integrator, "verlet", sizeof(physics.integrator) - 1);

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&physics, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "integrator") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_physics_euler_integrator_valid) {
    ConfigPhysics physics;
    physics_config_set_defaults(&physics);
    strncpy(physics.integrator, "euler", sizeof(physics.integrator) - 1);

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&physics, errors, CONFIG_MAX_ERRORS);

    for (int i = 0; i < n; i++) {
        ASSERT_TRUE(strstr(errors[i].field, "integrator") == NULL);
    }

    return 0;
}

TEST(validate_physics_rk4_integrator_valid) {
    ConfigPhysics physics;
    physics_config_set_defaults(&physics);
    strncpy(physics.integrator, "rk4", sizeof(physics.integrator) - 1);

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&physics, errors, CONFIG_MAX_ERRORS);

    for (int i = 0; i < n; i++) {
        ASSERT_TRUE(strstr(errors[i].field, "integrator") == NULL);
    }

    return 0;
}

TEST(validate_physics_negative_velocity_clamp) {
    ConfigPhysics physics;
    physics_config_set_defaults(&physics);
    physics.velocity_clamp = -1.0f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&physics, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "velocity_clamp") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_physics_negative_angular_velocity_clamp) {
    ConfigPhysics physics;
    physics_config_set_defaults(&physics);
    physics.angular_velocity_clamp = 0.0f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&physics, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "angular_velocity_clamp") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_physics_ground_effect_bad_height) {
    /* When ground effect is enabled, height must be positive */
    ConfigPhysics physics;
    physics_config_set_defaults(&physics);
    physics.enable_ground_effect = true;
    physics.ground_effect_height = 0.0f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&physics, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "ground_effect_height") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_physics_ground_effect_strength_below_1) {
    /* Ground effect strength must be >= 1.0 */
    ConfigPhysics physics;
    physics_config_set_defaults(&physics);
    physics.enable_ground_effect = true;
    physics.ground_effect_strength = 0.9f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&physics, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "ground_effect_strength") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_physics_ground_effect_disabled_skips_checks) {
    /* When ground effect is disabled, bad height/strength should NOT cause errors */
    ConfigPhysics physics;
    physics_config_set_defaults(&physics);
    physics.enable_ground_effect = false;
    physics.ground_effect_height = -100.0f;
    physics.ground_effect_strength = -5.0f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&physics, errors, CONFIG_MAX_ERRORS);

    /* No ground-effect errors when disabled */
    for (int i = 0; i < n; i++) {
        ASSERT_TRUE(strstr(errors[i].field, "ground_effect") == NULL);
    }

    return 0;
}

TEST(validate_physics_negative_variance) {
    ConfigPhysics physics;
    physics_config_set_defaults(&physics);
    physics.dt_variance = -0.001f;
    physics.mass_variance = -0.01f;
    physics.thrust_variance = -0.1f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&physics, errors, CONFIG_MAX_ERRORS);

    ASSERT_GE(n, 3);

    return 0;
}

TEST(validate_drone_negative_mass) {
    PlatformConfig drone;
    platform_config_set_defaults(&drone);
    drone.mass = -0.5f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&drone, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    ASSERT_STR_EQ(errors[0].field, "platform.mass");

    return 0;
}

TEST(validate_drone_zero_inertia_tensor) {
    /* Each diagonal element of the inertia tensor must be positive */
    PlatformConfig drone;
    platform_config_set_defaults(&drone);
    drone.ixx = 0.0f;
    drone.iyy = 0.0f;
    drone.izz = 0.0f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&drone, errors, CONFIG_MAX_ERRORS);

    /* Should have at least 3 errors (one per axis) */
    ASSERT_GE(n, 3);

    return 0;
}

TEST(validate_drone_negative_k_thrust) {
    PlatformConfig drone;
    platform_config_set_defaults(&drone);
    QuadcopterConfig* qc = (QuadcopterConfig*)drone.platform_specific;
    qc->k_thrust = -1e-8f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&drone, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "k_thrust") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    free(drone.platform_specific);
    return 0;
}

TEST(validate_drone_negative_motor_tau) {
    PlatformConfig drone;
    platform_config_set_defaults(&drone);
    QuadcopterConfig* qc = (QuadcopterConfig*)drone.platform_specific;
    qc->motor_tau = 0.0f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&drone, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "motor_tau") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    free(drone.platform_specific);
    return 0;
}

TEST(validate_drone_negative_max_rpm) {
    PlatformConfig drone;
    platform_config_set_defaults(&drone);
    QuadcopterConfig* qc = (QuadcopterConfig*)drone.platform_specific;
    qc->max_rpm = -100.0f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&drone, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "max_rpm") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    free(drone.platform_specific);
    return 0;
}

TEST(validate_drone_negative_drag_detected) {
    /* k_drag must be non-negative (0 is ok, negative is not) */
    PlatformConfig drone;
    platform_config_set_defaults(&drone);
    QuadcopterConfig* qc = (QuadcopterConfig*)drone.platform_specific;
    qc->k_drag = -0.001f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&drone, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "k_drag") != NULL &&
            strstr(errors[i].field, "angular") == NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    free(drone.platform_specific);
    return 0;
}

TEST(validate_drone_zero_drag_is_valid) {
    /* k_drag = 0 is perfectly valid (no drag) */
    PlatformConfig drone;
    platform_config_set_defaults(&drone);
    QuadcopterConfig* qc = (QuadcopterConfig*)drone.platform_specific;
    qc->k_drag = 0.0f;
    qc->k_ang_damp = 0.0f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&drone, errors, CONFIG_MAX_ERRORS);

    /* No drag-related errors */
    for (int i = 0; i < n; i++) {
        ASSERT_TRUE(strstr(errors[i].field, "k_drag") == NULL);
    }

    free(drone.platform_specific);
    return 0;
}

TEST(validate_drone_color_out_of_range) {
    PlatformConfig drone;
    platform_config_set_defaults(&drone);
    drone.color[0] = 1.5f;  /* > 1.0 */

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&drone, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "color") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_drone_color_negative) {
    PlatformConfig drone;
    platform_config_set_defaults(&drone);
    drone.color[2] = -0.1f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&drone, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "color") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_drone_zero_scale) {
    PlatformConfig drone;
    platform_config_set_defaults(&drone);
    drone.scale = 0.0f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&drone, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "scale") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_environment_zero_num_envs) {
    EnvironmentConfig env;
    environment_config_set_defaults(&env);
    env.num_envs = 0;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_environment(&env, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    ASSERT_TRUE(strstr(errors[0].field, "num_envs") != NULL);

    return 0;
}

TEST(validate_environment_negative_world_size) {
    EnvironmentConfig env;
    environment_config_set_defaults(&env);
    env.world_size[1] = -5.0f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_environment(&env, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "world_size") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_environment_zero_voxel_size) {
    EnvironmentConfig env;
    environment_config_set_defaults(&env);
    env.voxel_size = 0.0f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_environment(&env, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "voxel_size") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_environment_negative_spawn_radius) {
    EnvironmentConfig env;
    environment_config_set_defaults(&env);
    env.spawn_radius = -1.0f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_environment(&env, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);

    return 0;
}

TEST(validate_environment_spawn_height_equal) {
    /* spawn_height_min == spawn_height_max should fail (must be strict <) */
    EnvironmentConfig env;
    environment_config_set_defaults(&env);
    env.spawn_height_min = 5.0f;
    env.spawn_height_max = 5.0f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_environment(&env, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "spawn_height") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_environment_negative_min_separation) {
    EnvironmentConfig env;
    environment_config_set_defaults(&env);
    env.min_separation = -0.5f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_environment(&env, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "min_separation") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_environment_zero_max_episode_steps) {
    EnvironmentConfig env;
    environment_config_set_defaults(&env);
    env.max_episode_steps = 0;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_environment(&env, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "max_episode_steps") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_environment_unknown_world_type) {
    EnvironmentConfig env;
    environment_config_set_defaults(&env);
    strncpy(env.world_type, "forest", sizeof(env.world_type) - 1);

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_environment(&env, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "world_type") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_environment_all_valid_world_types) {
    /* Each of these must be accepted */
    const char* valid_types[] = {"empty", "obstacles", "maze", "race", "custom"};
    for (int t = 0; t < 5; t++) {
        EnvironmentConfig env;
        environment_config_set_defaults(&env);
        strncpy(env.world_type, valid_types[t], sizeof(env.world_type) - 1);

        ConfigError errors[CONFIG_MAX_ERRORS];
        int n = config_validate_environment(&env, errors, CONFIG_MAX_ERRORS);

        /* No world_type errors */
        for (int i = 0; i < n; i++) {
            ASSERT_MSG(strstr(errors[i].field, "world_type") == NULL,
                       "valid world_type was rejected");
        }
    }

    return 0;
}

/* ============================================================================
 * 4b. Sensor Validation Deep
 * ============================================================================ */

TEST(validate_sensor_all_valid_types) {
    /* Every recognized sensor type should pass type validation */
    const char* valid_types[] = {
        "imu", "tof", "lidar_2d", "lidar_3d",
        "camera_rgb", "camera_depth", "camera_segmentation",
        "position", "velocity", "neighbor"
    };
    int num_types = 10;

    for (int t = 0; t < num_types; t++) {
        SensorConfigEntry sensor = sensor_config_entry_default(valid_types[t]);

        ConfigError errors[CONFIG_MAX_ERRORS];
        int n = config_validate_sensors(&sensor, 1, errors, CONFIG_MAX_ERRORS);

        /* No "Unknown sensor type" error */
        for (int i = 0; i < n; i++) {
            ASSERT_MSG(strstr(errors[i].message, "Unknown sensor") == NULL,
                       "valid sensor type was rejected");
        }
    }

    return 0;
}

TEST(validate_sensor_unnormalized_quaternion) {
    SensorConfigEntry sensor = sensor_config_entry_default("imu");
    /* Set quaternion that is far from normalized */
    sensor.orientation[0] = 2.0f;
    sensor.orientation[1] = 0.0f;
    sensor.orientation[2] = 0.0f;
    sensor.orientation[3] = 0.0f;
    /* |q|^2 = 4.0, |q| = 2.0, tolerance is 0.01 around 1.0 */

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_sensors(&sensor, 1, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].message, "normalized") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_sensor_zero_quaternion) {
    /* All-zero quaternion: |q|^2 = 0, far from 1.0 */
    SensorConfigEntry sensor = sensor_config_entry_default("imu");
    sensor.orientation[0] = 0.0f;
    sensor.orientation[1] = 0.0f;
    sensor.orientation[2] = 0.0f;
    sensor.orientation[3] = 0.0f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_sensors(&sensor, 1, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].message, "normalized") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_sensor_nearly_normalized_quaternion_ok) {
    /* Quaternion with |q|^2 within 0.01 of 1.0 should pass */
    SensorConfigEntry sensor = sensor_config_entry_default("imu");
    /* |q|^2 = 0.7^2 + 0.7^2 + 0.1^2 + 0.1^2 = 0.49 + 0.49 + 0.01 + 0.01 = 1.00 */
    sensor.orientation[0] = 0.7f;
    sensor.orientation[1] = 0.7f;
    sensor.orientation[2] = 0.1f;
    sensor.orientation[3] = 0.1f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_sensors(&sensor, 1, errors, CONFIG_MAX_ERRORS);

    for (int i = 0; i < n; i++) {
        ASSERT_TRUE(strstr(errors[i].message, "normalized") == NULL);
    }

    return 0;
}

TEST(validate_sensor_negative_sample_rate) {
    SensorConfigEntry sensor = sensor_config_entry_default("imu");
    sensor.sample_rate = -10.0f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_sensors(&sensor, 1, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "sample_rate") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_sensor_tof_zero_range) {
    SensorConfigEntry sensor = sensor_config_entry_default("tof");
    sensor.max_range = 0.0f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_sensors(&sensor, 1, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "max_range") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_sensor_lidar_zero_rays) {
    SensorConfigEntry sensor = sensor_config_entry_default("lidar_2d");
    sensor.num_rays = 0;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_sensors(&sensor, 1, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "num_rays") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_sensor_lidar_zero_fov) {
    SensorConfigEntry sensor = sensor_config_entry_default("lidar_2d");
    sensor.fov = 0.0f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_sensors(&sensor, 1, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "fov") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_sensor_camera_zero_dimensions) {
    SensorConfigEntry sensor = sensor_config_entry_default("camera_rgb");
    sensor.width = 0;
    sensor.height = 0;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_sensors(&sensor, 1, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "width/height") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_sensor_camera_near_ge_far) {
    SensorConfigEntry sensor = sensor_config_entry_default("camera_depth");
    sensor.near_clip = 100.0f;
    sensor.far_clip = 10.0f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_sensors(&sensor, 1, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "near_clip/far_clip") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_sensor_camera_near_equal_far) {
    SensorConfigEntry sensor = sensor_config_entry_default("camera_rgb");
    sensor.near_clip = 50.0f;
    sensor.far_clip = 50.0f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_sensors(&sensor, 1, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "near_clip/far_clip") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(validate_sensor_neighbor_zero_k) {
    SensorConfigEntry sensor = sensor_config_entry_default("neighbor");
    sensor.k_neighbors = 0;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_sensors(&sensor, 1, errors, CONFIG_MAX_ERRORS);

    ASSERT_GT(n, 0);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "k_neighbors") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

/* ============================================================================
 * 5. config_hash Determinism and Variation
 *
 * FNV-1a properties: same input -> same output (determinism),
 * different input -> different output (with high probability).
 * ============================================================================ */

TEST(hash_deterministic_same_config) {
    Config cfg;
    config_set_defaults(&cfg);

    uint64_t h1 = config_hash(&cfg);
    uint64_t h2 = config_hash(&cfg);
    uint64_t h3 = config_hash(&cfg);

    ASSERT_TRUE(h1 == h2);
    ASSERT_TRUE(h2 == h3);

    return 0;
}

TEST(hash_deterministic_two_default_configs) {
    Config a, b;
    config_set_defaults(&a);
    config_set_defaults(&b);

    ASSERT_TRUE(config_hash(&a) == config_hash(&b));

    return 0;
}

TEST(hash_varies_with_single_float_change) {
    Config a, b;
    config_set_defaults(&a);
    config_set_defaults(&b);
    b.platform.mass = 0.028f;  /* Tiny change from 0.027 */

    ASSERT_TRUE(config_hash(&a) != config_hash(&b));

    config_free(&a);
    config_free(&b);
    return 0;
}

TEST(hash_varies_with_string_change) {
    Config a, b;
    config_set_defaults(&a);
    config_set_defaults(&b);
    strncpy(b.platform.name, "custom_drone", CONFIG_NAME_MAX - 1);

    ASSERT_TRUE(config_hash(&a) != config_hash(&b));

    return 0;
}

TEST(hash_varies_with_uint_change) {
    Config a, b;
    config_set_defaults(&a);
    config_set_defaults(&b);
    b.environment.num_envs = 65;

    ASSERT_TRUE(config_hash(&a) != config_hash(&b));

    return 0;
}

TEST(hash_varies_with_bool_change) {
    Config a, b;
    config_set_defaults(&a);
    config_set_defaults(&b);
    b.physics.normalize_quaternions = false;

    ASSERT_TRUE(config_hash(&a) != config_hash(&b));

    return 0;
}

TEST(hash_varies_across_sections) {
    /* Changing physics vs changing platform should produce different hashes */
    Config base, mod_platform, mod_physics;
    config_set_defaults(&base);
    config_set_defaults(&mod_platform);
    config_set_defaults(&mod_physics);
    mod_platform.platform.mass = 1.0f;
    mod_physics.physics.timestep = 0.01f;

    uint64_t h_base = config_hash(&base);
    uint64_t h_platform = config_hash(&mod_platform);
    uint64_t h_physics = config_hash(&mod_physics);

    ASSERT_TRUE(h_base != h_platform);
    ASSERT_TRUE(h_base != h_physics);
    ASSERT_TRUE(h_platform != h_physics);

    config_free(&base);
    config_free(&mod_platform);
    config_free(&mod_physics);
    return 0;
}

TEST(hash_null_config_returns_zero) {
    uint64_t h = config_hash(NULL);
    ASSERT_EQ(h, (uint64_t)0);

    return 0;
}

TEST(hash_includes_sensors) {
    Config a, b;
    config_set_defaults(&a);
    config_set_defaults(&b);

    /* b has one sensor, a has none */
    b.sensors = (SensorConfigEntry*)malloc(sizeof(SensorConfigEntry));
    b.num_sensors = 1;
    b.sensors[0] = sensor_config_entry_default("imu");

    ASSERT_TRUE(config_hash(&a) != config_hash(&b));

    config_free(&b);
    return 0;
}

TEST(hash_sensor_order_matters) {
    Config a, b;
    config_set_defaults(&a);
    config_set_defaults(&b);

    a.sensors = (SensorConfigEntry*)malloc(2 * sizeof(SensorConfigEntry));
    a.num_sensors = 2;
    a.sensors[0] = sensor_config_entry_default("imu");
    a.sensors[1] = sensor_config_entry_default("tof");

    b.sensors = (SensorConfigEntry*)malloc(2 * sizeof(SensorConfigEntry));
    b.num_sensors = 2;
    b.sensors[0] = sensor_config_entry_default("tof");
    b.sensors[1] = sensor_config_entry_default("imu");

    /* Different sensor order should produce different hash */
    ASSERT_TRUE(config_hash(&a) != config_hash(&b));

    config_free(&a);
    config_free(&b);
    return 0;
}

/* ============================================================================
 * 6. platform_config_to_params Conversion Completeness
 *
 * Verify every field is mapped correctly, including the name mappings:
 *   k_drag_angular -> k_ang_damp
 *   max_velocity -> max_vel
 *   max_angular_velocity -> max_omega
 *   physics.gravity -> gravity
 * ============================================================================ */

TEST(conversion_all_fields_mapped) {
    /* Use non-default values for every field to detect any missing mapping */
    PlatformConfig drone;
    platform_config_set_defaults(&drone);
    drone.mass = 1.234f;
    drone.ixx = 5.678e-4f;
    drone.iyy = 6.789e-4f;
    drone.izz = 7.890e-4f;
    drone.collision_radius = 0.234f;
    drone.max_velocity = 25.0f;
    drone.max_angular_velocity = 50.0f;
    QuadcopterConfig* qc = (QuadcopterConfig*)drone.platform_specific;
    qc->arm_length = 0.123f;
    qc->k_thrust = 4.567e-7f;
    qc->k_torque = 5.678e-9f;
    qc->k_drag = 0.345f;
    qc->k_ang_damp = 0.456f;
    qc->motor_tau = 0.078f;
    qc->max_rpm = 9876.0f;

    ConfigPhysics physics;
    physics_config_set_defaults(&physics);
    physics.gravity = 3.72f;  /* Mars gravity */

    Arena* arena = arena_create(65536);
    PlatformParamsSOA* params = platform_params_create(arena, 1, QUAD_PARAMS_EXT_COUNT);
    ASSERT_NOT_NULL(params);

    platform_config_to_params(&drone, &physics, params, 0, 1);

    /* Verify every field with exact match */
    ASSERT_FLOAT_EQ(params->rigid_body.mass[0], 1.234f);
    ASSERT_FLOAT_NEAR(params->rigid_body.ixx[0], 5.678e-4f, 1e-9f);
    ASSERT_FLOAT_NEAR(params->rigid_body.iyy[0], 6.789e-4f, 1e-9f);
    ASSERT_FLOAT_NEAR(params->rigid_body.izz[0], 7.890e-4f, 1e-9f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_ARM_LENGTH][0], 0.123f);
    ASSERT_FLOAT_EQ(params->rigid_body.collision_radius[0], 0.234f);
    ASSERT_FLOAT_NEAR(params->extension[QUAD_PEXT_K_THRUST][0], 4.567e-7f, 1e-12f);
    ASSERT_FLOAT_NEAR(params->extension[QUAD_PEXT_K_TORQUE][0], 5.678e-9f, 1e-14f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_K_DRAG][0], 0.345f);
    /* Name mapping: k_drag_angular -> k_ang_damp */
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_K_ANG_DAMP][0], 0.456f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_MOTOR_TAU][0], 0.078f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_MAX_RPM][0], 9876.0f);
    /* Name mapping: max_velocity -> max_vel */
    ASSERT_FLOAT_EQ(params->rigid_body.max_vel[0], 25.0f);
    /* Name mapping: max_angular_velocity -> max_omega */
    ASSERT_FLOAT_EQ(params->rigid_body.max_omega[0], 50.0f);
    /* gravity comes from ConfigPhysics, not PlatformConfig */
    ASSERT_FLOAT_EQ(params->rigid_body.gravity[0], 3.72f);

    free(drone.platform_specific);
    arena_destroy(arena);
    return 0;
}

TEST(conversion_broadcast_to_range) {
    /* Broadcasting to indices [2, 5) should only affect indices 2, 3, 4 */
    PlatformConfig drone;
    platform_config_set_defaults(&drone);
    drone.mass = 2.5f;

    ConfigPhysics physics;
    physics_config_set_defaults(&physics);

    Arena* arena = arena_create(65536);
    PlatformParamsSOA* params = platform_params_create(arena, 8, QUAD_PARAMS_EXT_COUNT);
    ASSERT_NOT_NULL(params);

    /* First fill all 8 with defaults */
    platform_config_set_defaults(&drone);
    platform_config_to_params(&drone, &physics, params, 0, 8);

    /* Now override indices 2-4 with different mass */
    drone.mass = 2.5f;
    platform_config_to_params(&drone, &physics, params, 2, 3);

    /* Indices 0, 1 should have default mass (0.027) */
    ASSERT_FLOAT_EQ(params->rigid_body.mass[0], 0.027f);
    ASSERT_FLOAT_EQ(params->rigid_body.mass[1], 0.027f);
    /* Indices 2, 3, 4 should have 2.5 */
    ASSERT_FLOAT_EQ(params->rigid_body.mass[2], 2.5f);
    ASSERT_FLOAT_EQ(params->rigid_body.mass[3], 2.5f);
    ASSERT_FLOAT_EQ(params->rigid_body.mass[4], 2.5f);
    /* Indices 5, 6, 7 should have default mass */
    ASSERT_FLOAT_EQ(params->rigid_body.mass[5], 0.027f);
    ASSERT_FLOAT_EQ(params->rigid_body.mass[6], 0.027f);
    ASSERT_FLOAT_EQ(params->rigid_body.mass[7], 0.027f);

    arena_destroy(arena);
    return 0;
}

TEST(conversion_null_params_is_noop) {
    PlatformConfig drone;
    platform_config_set_defaults(&drone);
    ConfigPhysics physics;
    physics_config_set_defaults(&physics);

    /* Should not crash */
    platform_config_to_params(&drone, &physics, NULL, 0, 1);
    platform_config_to_params(NULL, &physics, NULL, 0, 1);
    platform_config_to_params(&drone, NULL, NULL, 0, 1);

    return 0;
}

TEST(conversion_exceeds_capacity_is_noop) {
    PlatformConfig drone;
    platform_config_set_defaults(&drone);
    ConfigPhysics physics;
    physics_config_set_defaults(&physics);

    Arena* arena = arena_create(65536);
    PlatformParamsSOA* params = platform_params_create(arena, 4, QUAD_PARAMS_EXT_COUNT);
    ASSERT_NOT_NULL(params);

    /* Try to write beyond capacity: start_index + count > capacity */
    platform_config_to_params(&drone, &physics, params, 3, 5);

    /* count should NOT have been updated since operation was rejected */
    ASSERT_EQ(params->rigid_body.count, (uint32_t)0);

    arena_destroy(arena);
    return 0;
}

TEST(conversion_updates_count) {
    PlatformConfig drone;
    platform_config_set_defaults(&drone);
    ConfigPhysics physics;
    physics_config_set_defaults(&physics);

    Arena* arena = arena_create(65536);
    PlatformParamsSOA* params = platform_params_create(arena, 16, QUAD_PARAMS_EXT_COUNT);
    ASSERT_NOT_NULL(params);

    ASSERT_EQ(params->rigid_body.count, (uint32_t)0);

    platform_config_to_params(&drone, &physics, params, 0, 8);
    ASSERT_EQ(params->rigid_body.count, (uint32_t)8);

    /* Writing to higher indices should extend count */
    platform_config_to_params(&drone, &physics, params, 12, 4);
    ASSERT_EQ(params->rigid_body.count, (uint32_t)16);

    /* Writing to lower indices should not reduce count */
    platform_config_to_params(&drone, &physics, params, 0, 2);
    ASSERT_EQ(params->rigid_body.count, (uint32_t)16);

    arena_destroy(arena);
    return 0;
}

TEST(conversion_roundtrip_preserves_values) {
    /* config -> params -> config should preserve all mapped fields */
    PlatformConfig original;
    platform_config_set_defaults(&original);
    original.mass = 0.5f;
    original.ixx = 3e-4f;
    original.iyy = 4e-4f;
    original.izz = 5e-4f;
    original.collision_radius = 0.2f;
    original.max_velocity = 20.0f;
    original.max_angular_velocity = 35.0f;
    QuadcopterConfig* oq = (QuadcopterConfig*)original.platform_specific;
    oq->arm_length = 0.15f;
    oq->k_thrust = 5e-7f;
    oq->k_torque = 1e-8f;
    oq->k_drag = 0.05f;
    oq->k_ang_damp = 0.02f;
    oq->motor_tau = 0.05f;
    oq->max_rpm = 15000.0f;

    ConfigPhysics physics;
    physics_config_set_defaults(&physics);

    Arena* arena = arena_create(65536);
    PlatformParamsSOA* params = platform_params_create(arena, 1, QUAD_PARAMS_EXT_COUNT);
    ASSERT_NOT_NULL(params);

    platform_config_to_params(&original, &physics, params, 0, 1);
    PlatformConfig extracted = platform_params_to_config(params, 0);

    /* All mapped fields should match */
    const QuadcopterConfig* eq = (const QuadcopterConfig*)extracted.platform_specific;
    ASSERT_FLOAT_EQ(extracted.mass, original.mass);
    ASSERT_FLOAT_NEAR(extracted.ixx, original.ixx, 1e-9f);
    ASSERT_FLOAT_NEAR(extracted.iyy, original.iyy, 1e-9f);
    ASSERT_FLOAT_NEAR(extracted.izz, original.izz, 1e-9f);
    ASSERT_FLOAT_EQ(eq->arm_length, oq->arm_length);
    ASSERT_FLOAT_EQ(extracted.collision_radius, original.collision_radius);
    ASSERT_FLOAT_NEAR(eq->k_thrust, oq->k_thrust, 1e-12f);
    ASSERT_FLOAT_NEAR(eq->k_torque, oq->k_torque, 1e-14f);
    ASSERT_FLOAT_EQ(eq->k_drag, oq->k_drag);
    ASSERT_FLOAT_EQ(eq->k_ang_damp, oq->k_ang_damp);
    ASSERT_FLOAT_EQ(eq->motor_tau, oq->motor_tau);
    ASSERT_FLOAT_EQ(eq->max_rpm, oq->max_rpm);
    ASSERT_FLOAT_EQ(extracted.max_velocity, original.max_velocity);
    ASSERT_FLOAT_EQ(extracted.max_angular_velocity, original.max_angular_velocity);

    /* Name should be "extracted", not the original */
    ASSERT_STR_EQ(extracted.name, "extracted");

    /* Fields not stored in SoA get defaults */
    ASSERT_FLOAT_EQ(extracted.max_tilt_angle, 1.0f);
    ASSERT_FLOAT_EQ(extracted.scale, 1.0f);
    ASSERT_FLOAT_EQ(extracted.color[0], 0.2f);
    ASSERT_FLOAT_EQ(extracted.color[1], 0.6f);
    ASSERT_FLOAT_EQ(extracted.color[2], 1.0f);

    free(original.platform_specific);
    free(extracted.platform_specific);
    arena_destroy(arena);
    return 0;
}

TEST(conversion_params_to_config_invalid_index_returns_defaults) {
    Arena* arena = arena_create(65536);
    PlatformParamsSOA* params = platform_params_create(arena, 4, QUAD_PARAMS_EXT_COUNT);
    ASSERT_NOT_NULL(params);

    /* count = 0, so index 0 is out of range */
    PlatformConfig cfg = platform_params_to_config(params, 0);

    /* Should return defaults (Crazyflie 2.0) */
    ASSERT_FLOAT_EQ(cfg.mass, 0.027f);
    ASSERT_STR_EQ(cfg.name, "crazyflie2");

    arena_destroy(arena);
    return 0;
}

TEST(conversion_params_to_config_null_returns_defaults) {
    PlatformConfig cfg = platform_params_to_config(NULL, 0);

    ASSERT_FLOAT_EQ(cfg.mass, 0.027f);
    ASSERT_STR_EQ(cfg.name, "crazyflie2");

    return 0;
}

TEST(conversion_config_init_platform_params_convenience) {
    Config cfg;
    config_set_defaults(&cfg);
    cfg.platform.mass = 0.75f;
    cfg.physics.gravity = 1.62f;  /* Moon gravity */

    Arena* arena = arena_create(65536);
    PlatformParamsSOA* params = platform_params_create(arena, 4, QUAD_PARAMS_EXT_COUNT);
    ASSERT_NOT_NULL(params);

    config_init_platform_params(&cfg, params, 4);

    /* Verify all drones got the values */
    for (uint32_t i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(params->rigid_body.mass[i], 0.75f);
        ASSERT_FLOAT_EQ(params->rigid_body.gravity[i], 1.62f);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * 7. Config Defaults Produce a Fully Valid Config
 * ============================================================================ */

TEST(defaults_pass_full_validation) {
    Config cfg;
    config_set_defaults(&cfg);

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);

    ASSERT_EQ(n, 0);

    return 0;
}

TEST(defaults_drone_passes_validation) {
    PlatformConfig drone;
    platform_config_set_defaults(&drone);

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&drone, errors, CONFIG_MAX_ERRORS);

    ASSERT_EQ(n, 0);

    return 0;
}

TEST(defaults_environment_passes_validation) {
    EnvironmentConfig env;
    environment_config_set_defaults(&env);

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_environment(&env, errors, CONFIG_MAX_ERRORS);

    ASSERT_EQ(n, 0);

    return 0;
}

TEST(defaults_physics_passes_validation) {
    ConfigPhysics physics;
    physics_config_set_defaults(&physics);

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(&physics, errors, CONFIG_MAX_ERRORS);

    ASSERT_EQ(n, 0);

    return 0;
}

TEST(defaults_no_sensors_is_valid) {
    Config cfg;
    config_set_defaults(&cfg);

    ASSERT_NULL(cfg.sensors);
    ASSERT_EQ(cfg.num_sensors, (uint32_t)0);

    /* Validation with no sensors should produce no sensor errors */
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);
    ASSERT_EQ(n, 0);

    return 0;
}

TEST(defaults_sensor_entries_valid_for_all_types) {
    /* Each sensor type's default should pass validation */
    const char* types[] = {
        "imu", "tof", "lidar_2d", "lidar_3d",
        "camera_rgb", "camera_depth", "camera_segmentation",
        "position", "velocity", "neighbor"
    };

    for (int t = 0; t < 10; t++) {
        SensorConfigEntry sensor = sensor_config_entry_default(types[t]);

        ConfigError errors[CONFIG_MAX_ERRORS];
        int n = config_validate_sensors(&sensor, 1, errors, CONFIG_MAX_ERRORS);

        ASSERT_MSG(n == 0, "Default sensor entry failed validation");
    }

    return 0;
}

TEST(defaults_sensor_entry_imu_has_identity_quaternion) {
    SensorConfigEntry sensor = sensor_config_entry_default("imu");

    ASSERT_FLOAT_EQ(sensor.orientation[0], 1.0f);
    ASSERT_FLOAT_EQ(sensor.orientation[1], 0.0f);
    ASSERT_FLOAT_EQ(sensor.orientation[2], 0.0f);
    ASSERT_FLOAT_EQ(sensor.orientation[3], 0.0f);

    return 0;
}

TEST(defaults_sensor_entry_camera_has_valid_clip_planes) {
    SensorConfigEntry sensor = sensor_config_entry_default("camera_rgb");

    ASSERT_TRUE(sensor.near_clip < sensor.far_clip);
    ASSERT_TRUE(sensor.width > 0);
    ASSERT_TRUE(sensor.height > 0);

    return 0;
}

/* ============================================================================
 * 8. Field Boundary Values
 * ============================================================================ */

TEST(boundary_very_small_positive_mass) {
    /* Extremely small but positive mass should be valid */
    Config cfg;
    config_set_defaults(&cfg);
    cfg.platform.mass = 1e-10f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);

    /* No mass-related errors */
    for (int i = 0; i < n; i++) {
        ASSERT_TRUE(strstr(errors[i].field, "mass") == NULL);
    }

    return 0;
}

TEST(boundary_very_large_mass) {
    /* Very large mass should be valid (no upper bound check) */
    Config cfg;
    config_set_defaults(&cfg);
    cfg.platform.mass = 1e6f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);

    for (int i = 0; i < n; i++) {
        ASSERT_TRUE(strstr(errors[i].field, "mass") == NULL);
    }

    return 0;
}

TEST(boundary_float_max_values) {
    /* FLT_MAX should not crash validation */
    Config cfg;
    config_set_defaults(&cfg);
    cfg.platform.mass = FLT_MAX;
    cfg.platform.max_velocity = FLT_MAX;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);

    /* Should not crash -- some fields may not like FLT_MAX but no segfault */
    (void)n;

    return 0;
}

TEST(boundary_very_small_timestep) {
    /* Very small but positive timestep should be valid */
    Config cfg;
    config_set_defaults(&cfg);
    cfg.physics.timestep = 1e-8f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);

    for (int i = 0; i < n; i++) {
        ASSERT_TRUE(strstr(errors[i].field, "timestep") == NULL);
    }

    return 0;
}

TEST(boundary_timestep_just_over_limit) {
    /* 0.1000001 should be rejected (> 0.1) */
    Config cfg;
    config_set_defaults(&cfg);
    cfg.physics.timestep = 0.1001f;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);

    int found = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "timestep") != NULL &&
            strstr(errors[i].message, "stability") != NULL) {
            found = 1;
            break;
        }
    }
    ASSERT_TRUE(found);

    return 0;
}

TEST(boundary_max_uint32_agent_count_in_broadcast) {
    /* Test that the capacity check prevents overflow.
     * We can't allocate UINT32_MAX drones, but we can check that
     * start_index + count overflow is handled. */
    PlatformConfig drone;
    platform_config_set_defaults(&drone);
    ConfigPhysics physics;
    physics_config_set_defaults(&physics);

    Arena* arena = arena_create(65536);
    PlatformParamsSOA* params = platform_params_create(arena, 4, QUAD_PARAMS_EXT_COUNT);
    ASSERT_NOT_NULL(params);

    /* UINT32_MAX as count with start_index=0 should exceed capacity=4 */
    platform_config_to_params(&drone, &physics, params, 0, UINT32_MAX);
    /* Should be a no-op since 0 + UINT32_MAX > 4 */
    ASSERT_EQ(params->rigid_body.count, (uint32_t)0);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * 9. Validation Error Reporting Fidelity
 * ============================================================================ */

TEST(validation_null_config_returns_zero) {
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate(NULL, errors, CONFIG_MAX_ERRORS);
    ASSERT_EQ(n, 0);

    return 0;
}

TEST(validation_null_errors_returns_zero) {
    Config cfg;
    config_set_defaults(&cfg);
    cfg.platform.mass = -1.0f;

    int n = config_validate(&cfg, NULL, CONFIG_MAX_ERRORS);
    ASSERT_EQ(n, 0);

    return 0;
}

TEST(validation_max_errors_respected) {
    /* Create many errors but limit to 2 */
    Config cfg;
    config_set_defaults(&cfg);
    cfg.platform.mass = 0.0f;
    cfg.platform.ixx = 0.0f;
    cfg.platform.iyy = 0.0f;
    cfg.platform.izz = 0.0f;
    cfg.platform.collision_radius = 0.0f;
    cfg.platform.max_velocity = 0.0f;
    cfg.platform.max_angular_velocity = 0.0f;
    cfg.platform.scale = 0.0f;
    QuadcopterConfig* qc = (QuadcopterConfig*)cfg.platform.platform_specific;
    qc->arm_length = 0.0f;
    qc->k_thrust = 0.0f;
    qc->motor_tau = 0.0f;
    qc->max_rpm = 0.0f;

    ConfigError errors[2];
    int n = config_validate(&cfg, errors, 2);

    /* Should report exactly 2 errors even though there are 12+ violations */
    ASSERT_EQ(n, 2);

    config_free(&cfg);
    return 0;
}

TEST(validation_errors_have_valid_field_names) {
    Config cfg;
    config_set_defaults(&cfg);
    cfg.platform.mass = -1.0f;
    cfg.physics.timestep = -1.0f;
    cfg.environment.num_envs = 0;

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);

    ASSERT_GE(n, 3);

    /* Every error should have a non-empty field name */
    for (int i = 0; i < n; i++) {
        ASSERT_TRUE(strlen(errors[i].field) > 0);
        ASSERT_TRUE(strlen(errors[i].message) > 0);
        /* Line number should be -1 (unknown for programmatic validation) */
        ASSERT_EQ(errors[i].line_number, -1);
    }

    return 0;
}

TEST(validation_error_accumulation_across_sections) {
    /* Errors from platform, physics, environment should all be reported */
    Config cfg;
    config_set_defaults(&cfg);
    cfg.platform.mass = 0.0f;          /* platform error */
    cfg.physics.timestep = 0.0f;    /* physics error */
    cfg.environment.num_envs = 0;   /* environment error */

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate(&cfg, errors, CONFIG_MAX_ERRORS);

    /* Should have errors from at least 3 different sections */
    int found_platform = 0, found_physics = 0, found_env = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(errors[i].field, "platform.") != NULL) found_platform = 1;
        if (strstr(errors[i].field, "physics.") != NULL) found_physics = 1;
        if (strstr(errors[i].field, "environment.") != NULL) found_env = 1;
    }
    ASSERT_TRUE(found_platform);
    ASSERT_TRUE(found_physics);
    ASSERT_TRUE(found_env);

    config_free(&cfg);
    return 0;
}

/* ============================================================================
 * 10. Cross-Section Semantic Consistency and Edge Cases
 * ============================================================================ */

TEST(config_compare_identical_configs) {
    Config a, b;
    config_set_defaults(&a);
    config_set_defaults(&b);

    ASSERT_EQ(config_compare(&a, &b), 0);

    return 0;
}

TEST(config_compare_detects_float_difference) {
    Config a, b;
    config_set_defaults(&a);
    config_set_defaults(&b);
    b.platform.mass = 0.028f;

    ASSERT_NE(config_compare(&a, &b), 0);

    config_free(&a);
    config_free(&b);
    return 0;
}

TEST(config_compare_null_returns_nonzero) {
    Config cfg;
    config_set_defaults(&cfg);

    ASSERT_NE(config_compare(NULL, &cfg), 0);
    ASSERT_NE(config_compare(&cfg, NULL), 0);
    ASSERT_NE(config_compare(NULL, NULL), 0);

    return 0;
}

TEST(config_clone_deep_copy_independence) {
    Config src;
    config_set_defaults(&src);
    src.sensors = (SensorConfigEntry*)malloc(2 * sizeof(SensorConfigEntry));
    src.num_sensors = 2;
    src.sensors[0] = sensor_config_entry_default("imu");
    src.sensors[1] = sensor_config_entry_default("tof");

    Arena* arena = arena_create(16384);
    Config dst;
    config_clone(&src, &dst, arena);

    /* Modify source -- should not affect destination */
    src.platform.mass = 999.0f;
    strcpy(src.sensors[0].name, "modified");

    ASSERT_FLOAT_EQ(dst.platform.mass, 0.027f);
    ASSERT_STR_EQ(dst.sensors[0].name, "sensor");

    /* Pointers must be different (deep copy) */
    ASSERT_TRUE(dst.sensors != src.sensors);

    config_free(&src);
    /* dst.sensors was arena-allocated, no free needed */
    arena_destroy(arena);
    return 0;
}

TEST(config_memory_size_calculation) {
    size_t base = config_memory_size(0);
    size_t with_1 = config_memory_size(1);
    size_t with_32 = config_memory_size(32);

    ASSERT_EQ(base, sizeof(Config));
    ASSERT_EQ(with_1, sizeof(Config) + sizeof(SensorConfigEntry));
    ASSERT_EQ(with_32, sizeof(Config) + 32 * sizeof(SensorConfigEntry));

    /* Monotonically increasing */
    ASSERT_TRUE(base < with_1);
    ASSERT_TRUE(with_1 < with_32);

    return 0;
}

TEST(config_free_null_safe) {
    /* Should not crash */
    config_free(NULL);

    Config cfg;
    config_set_defaults(&cfg);
    config_free(&cfg);
    /* Double free should also be safe (sensors was set to NULL) */
    config_free(&cfg);

    return 0;
}

TEST(config_free_clears_sensors) {
    Config cfg;
    config_set_defaults(&cfg);
    cfg.sensors = (SensorConfigEntry*)malloc(sizeof(SensorConfigEntry));
    cfg.num_sensors = 1;

    config_free(&cfg);

    ASSERT_NULL(cfg.sensors);
    ASSERT_EQ(cfg.num_sensors, (uint32_t)0);

    return 0;
}

TEST(parse_all_environment_fields) {
    const char* toml =
        "[environment]\n"
        "num_envs = 32\n"
        "agents_per_env = 8\n"
        "world_size = [30.0, 30.0, 15.0]\n"
        "world_origin = [1.0, 2.0, 3.0]\n"
        "voxel_size = 0.05\n"
        "max_bricks = 16384\n"
        "spawn_radius = 10.0\n"
        "spawn_height_min = 3.0\n"
        "spawn_height_max = 12.0\n"
        "min_separation = 2.0\n"
        "max_episode_steps = 2000\n"
        "auto_reset = false\n"
        "world_type = \"maze\"\n"
        "num_obstacles = 50\n"
        "seed = 12345\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_EQ(result, 0);
    ASSERT_EQ(cfg.environment.num_envs, (uint32_t)32);
    ASSERT_EQ(cfg.environment.agents_per_env, (uint32_t)8);
    ASSERT_FLOAT_EQ(cfg.environment.world_size[0], 30.0f);
    ASSERT_FLOAT_EQ(cfg.environment.world_size[1], 30.0f);
    ASSERT_FLOAT_EQ(cfg.environment.world_size[2], 15.0f);
    ASSERT_FLOAT_EQ(cfg.environment.world_origin[0], 1.0f);
    ASSERT_FLOAT_EQ(cfg.environment.world_origin[1], 2.0f);
    ASSERT_FLOAT_EQ(cfg.environment.world_origin[2], 3.0f);
    ASSERT_FLOAT_EQ(cfg.environment.voxel_size, 0.05f);
    ASSERT_EQ(cfg.environment.max_bricks, (uint32_t)16384);
    ASSERT_FLOAT_EQ(cfg.environment.spawn_radius, 10.0f);
    ASSERT_FLOAT_EQ(cfg.environment.spawn_height_min, 3.0f);
    ASSERT_FLOAT_EQ(cfg.environment.spawn_height_max, 12.0f);
    ASSERT_FLOAT_EQ(cfg.environment.min_separation, 2.0f);
    ASSERT_EQ(cfg.environment.max_episode_steps, (uint32_t)2000);
    ASSERT_FALSE(cfg.environment.auto_reset);
    ASSERT_STR_EQ(cfg.environment.world_type, "maze");
    ASSERT_EQ(cfg.environment.num_obstacles, (uint32_t)50);
    ASSERT_EQ(cfg.environment.seed, (uint32_t)12345);

    config_free(&cfg);
    return 0;
}

TEST(parse_all_physics_fields) {
    const char* toml =
        "[physics]\n"
        "timestep = 0.005\n"
        "substeps = 8\n"
        "gravity = 1.62\n"
        "integrator = \"euler\"\n"
        "velocity_clamp = 50.0\n"
        "angular_velocity_clamp = 100.0\n"
        "normalize_quaternions = false\n"
        "enable_ground_effect = false\n"
        "ground_effect_height = 1.0\n"
        "ground_effect_strength = 2.0\n"
        "dt_variance = 0.001\n"
        "mass_variance = 0.01\n"
        "thrust_variance = 0.05\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_EQ(result, 0);
    ASSERT_FLOAT_EQ(cfg.physics.timestep, 0.005f);
    ASSERT_EQ(cfg.physics.substeps, (uint32_t)8);
    ASSERT_FLOAT_EQ(cfg.physics.gravity, 1.62f);
    ASSERT_STR_EQ(cfg.physics.integrator, "euler");
    ASSERT_FLOAT_EQ(cfg.physics.velocity_clamp, 50.0f);
    ASSERT_FLOAT_EQ(cfg.physics.angular_velocity_clamp, 100.0f);
    ASSERT_FALSE(cfg.physics.normalize_quaternions);
    ASSERT_FALSE(cfg.physics.enable_ground_effect);
    ASSERT_FLOAT_EQ(cfg.physics.ground_effect_height, 1.0f);
    ASSERT_FLOAT_EQ(cfg.physics.ground_effect_strength, 2.0f);
    ASSERT_FLOAT_EQ(cfg.physics.dt_variance, 0.001f);
    ASSERT_FLOAT_EQ(cfg.physics.mass_variance, 0.01f);
    ASSERT_FLOAT_EQ(cfg.physics.thrust_variance, 0.05f);

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
        "world_collision_penalty = 10.0\n"
        "drone_collision_penalty = 5.0\n"
        "alive_bonus = 0.1\n"
        "success_bonus = 200.0\n"
        "reward_min = -20.0\n"
        "reward_max = 20.0\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_EQ(result, 0);
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
    ASSERT_FLOAT_EQ(cfg.reward.world_collision_penalty, 10.0f);
    ASSERT_FLOAT_EQ(cfg.reward.drone_collision_penalty, 5.0f);
    ASSERT_FLOAT_EQ(cfg.reward.alive_bonus, 0.1f);
    ASSERT_FLOAT_EQ(cfg.reward.success_bonus, 200.0f);
    ASSERT_FLOAT_EQ(cfg.reward.reward_min, -20.0f);
    ASSERT_FLOAT_EQ(cfg.reward.reward_max, 20.0f);

    config_free(&cfg);
    return 0;
}

TEST(parse_all_training_fields) {
    const char* toml =
        "[training]\n"
        "algorithm = \"sac\"\n"
        "learning_rate = 1e-3\n"
        "gamma = 0.999\n"
        "gae_lambda = 0.98\n"
        "clip_range = 0.1\n"
        "entropy_coef = 0.005\n"
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

    ASSERT_EQ(result, 0);
    ASSERT_STR_EQ(cfg.training.algorithm, "sac");
    ASSERT_FLOAT_NEAR(cfg.training.learning_rate, 1e-3f, 1e-7f);
    ASSERT_FLOAT_EQ(cfg.training.gamma, 0.999f);
    ASSERT_FLOAT_EQ(cfg.training.gae_lambda, 0.98f);
    ASSERT_FLOAT_EQ(cfg.training.clip_range, 0.1f);
    ASSERT_FLOAT_EQ(cfg.training.entropy_coef, 0.005f);
    ASSERT_FLOAT_EQ(cfg.training.value_coef, 1.0f);
    ASSERT_FLOAT_EQ(cfg.training.max_grad_norm, 1.0f);
    ASSERT_EQ(cfg.training.batch_size, (uint32_t)4096);
    ASSERT_EQ(cfg.training.num_epochs, (uint32_t)20);
    ASSERT_EQ(cfg.training.rollout_length, (uint32_t)256);
    ASSERT_EQ(cfg.training.log_interval, (uint32_t)5);
    ASSERT_EQ(cfg.training.save_interval, (uint32_t)50);
    ASSERT_STR_EQ(cfg.training.checkpoint_dir, "/tmp/ckpts");

    config_free(&cfg);
    return 0;
}

TEST(parse_sensor_type_specific_defaults) {
    /* When parsing a sensor, type-specific defaults should be set */
    const char* toml =
        "[[sensors]]\n"
        "type = \"lidar_3d\"\n"
        "name = \"top_lidar\"\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_EQ(result, 0);
    ASSERT_EQ(cfg.num_sensors, (uint32_t)1);
    ASSERT_STR_EQ(cfg.sensors[0].type, "lidar_3d");
    /* Type-specific defaults for lidar_3d */
    ASSERT_FLOAT_EQ(cfg.sensors[0].max_range, 20.0f);
    ASSERT_EQ(cfg.sensors[0].num_rays, (uint32_t)64);
    ASSERT_FLOAT_NEAR(cfg.sensors[0].fov, 6.28318f, 0.001f);
    ASSERT_FLOAT_NEAR(cfg.sensors[0].fov_vertical, 0.524f, 0.001f);
    ASSERT_EQ(cfg.sensors[0].vertical_layers, (uint32_t)16);

    config_free(&cfg);
    return 0;
}

TEST(parse_sensor_overrides_type_defaults) {
    /* Explicit values should override type-specific defaults */
    const char* toml =
        "[[sensors]]\n"
        "type = \"lidar_2d\"\n"
        "max_range = 50.0\n"
        "num_rays = 128\n"
        "fov = 3.14159\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_EQ(result, 0);
    ASSERT_FLOAT_EQ(cfg.sensors[0].max_range, 50.0f);
    ASSERT_EQ(cfg.sensors[0].num_rays, (uint32_t)128);
    ASSERT_FLOAT_NEAR(cfg.sensors[0].fov, 3.14159f, 0.001f);

    config_free(&cfg);
    return 0;
}

TEST(json_export_small_buffer_fails) {
    Config cfg;
    config_set_defaults(&cfg);

    char buffer[10];
    int result = config_to_json(&cfg, buffer, sizeof(buffer));

    ASSERT_EQ(result, -1);

    return 0;
}

TEST(json_export_null_args) {
    Config cfg;
    config_set_defaults(&cfg);
    char buffer[1024];

    ASSERT_EQ(config_to_json(NULL, buffer, sizeof(buffer)), -1);
    ASSERT_EQ(config_to_json(&cfg, NULL, sizeof(buffer)), -1);
    ASSERT_EQ(config_to_json(&cfg, buffer, 0), -1);

    return 0;
}

TEST(hash_computed_on_successful_load) {
    const char* toml =
        "[drone]\n"
        "mass = 0.5\n";

    Config cfg;
    char error_msg[256] = {0};
    int result = config_load_string(toml, &cfg, error_msg);

    ASSERT_EQ(result, 0);
    ASSERT_TRUE(cfg.config_hash != 0);

    /* Hash should match manual computation */
    uint64_t expected = config_hash(&cfg);
    ASSERT_TRUE(cfg.config_hash == expected);

    config_free(&cfg);
    return 0;
}

TEST(validate_drone_null_returns_zero) {
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(NULL, errors, CONFIG_MAX_ERRORS);
    ASSERT_EQ(n, 0);

    return 0;
}

TEST(validate_environment_null_returns_zero) {
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_environment(NULL, errors, CONFIG_MAX_ERRORS);
    ASSERT_EQ(n, 0);

    return 0;
}

TEST(validate_physics_null_returns_zero) {
    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_physics(NULL, errors, CONFIG_MAX_ERRORS);
    ASSERT_EQ(n, 0);

    return 0;
}

TEST(validate_sensors_null_errors_returns_zero) {
    SensorConfigEntry sensor = sensor_config_entry_default("imu");
    int n = config_validate_sensors(&sensor, 1, NULL, CONFIG_MAX_ERRORS);
    ASSERT_EQ(n, 0);

    return 0;
}

TEST(validate_drone_many_simultaneous_errors) {
    /* Zero out every positive-required field to maximize error count */
    PlatformConfig drone;
    memset(&drone, 0, sizeof(drone));

    /* Set up zeroed quadcopter config so quad-specific validation runs */
    QuadcopterConfig quad;
    memset(&quad, 0, sizeof(quad));
    strncpy(drone.type, "quadcopter", CONFIG_NAME_MAX - 1);
    drone.platform_specific = &quad;
    drone.platform_specific_size = sizeof(QuadcopterConfig);

    ConfigError errors[CONFIG_MAX_ERRORS];
    int n = config_validate_platform(&drone, errors, CONFIG_MAX_ERRORS);

    /* mass, arm_length, ixx, iyy, izz, k_thrust, motor_tau, max_rpm,
     * collision_radius, max_velocity, max_angular_velocity, scale = 12 fields
     * color is [0,0,0] which is valid. k_drag=0 is valid. k_ang_damp=0 is valid. */
    ASSERT_GE(n, 12);

    /* Don't free quad -- it's stack-allocated */
    return 0;
}

/* ============================================================================
 * Main Test Runner
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Config Validation Deep");

    /* 1. Three-phase pipeline */
    RUN_TEST(pipeline_valid_full_config_passes_all_phases);
    RUN_TEST(pipeline_parse_failure_returns_error);
    RUN_TEST(pipeline_semantic_failure_returns_error);
    RUN_TEST(pipeline_validation_resets_to_defaults_on_failure);

    /* 2. Sensor array limits */
    RUN_TEST(sensor_count_zero);
    RUN_TEST(sensor_count_one);
    RUN_TEST(sensor_count_at_max);
    RUN_TEST(sensor_count_over_max_is_clamped);
    RUN_TEST(sensor_validation_zero_count_no_errors);

    /* 3. config_load_string edge cases */
    RUN_TEST(load_string_null_toml);
    RUN_TEST(load_string_null_config);
    RUN_TEST(load_string_null_error_msg);
    RUN_TEST(load_string_empty_returns_defaults);
    RUN_TEST(load_string_whitespace_only);
    RUN_TEST(load_string_comment_only);
    RUN_TEST(load_string_unknown_section_ignored);
    RUN_TEST(load_string_partial_override);

    /* 4. Semantic validation: physics */
    RUN_TEST(validate_physics_timestep_zero);
    RUN_TEST(validate_physics_timestep_negative);
    RUN_TEST(validate_physics_timestep_too_large);
    RUN_TEST(validate_physics_timestep_boundary_at_0_1);
    RUN_TEST(validate_physics_substeps_zero);
    RUN_TEST(validate_physics_negative_gravity);
    RUN_TEST(validate_physics_zero_gravity_is_valid);
    RUN_TEST(validate_physics_invalid_integrator);
    RUN_TEST(validate_physics_euler_integrator_valid);
    RUN_TEST(validate_physics_rk4_integrator_valid);
    RUN_TEST(validate_physics_negative_velocity_clamp);
    RUN_TEST(validate_physics_negative_angular_velocity_clamp);
    RUN_TEST(validate_physics_ground_effect_bad_height);
    RUN_TEST(validate_physics_ground_effect_strength_below_1);
    RUN_TEST(validate_physics_ground_effect_disabled_skips_checks);
    RUN_TEST(validate_physics_negative_variance);

    /* 4. Semantic validation: drone */
    RUN_TEST(validate_drone_negative_mass);
    RUN_TEST(validate_drone_zero_inertia_tensor);
    RUN_TEST(validate_drone_negative_k_thrust);
    RUN_TEST(validate_drone_negative_motor_tau);
    RUN_TEST(validate_drone_negative_max_rpm);
    RUN_TEST(validate_drone_negative_drag_detected);
    RUN_TEST(validate_drone_zero_drag_is_valid);
    RUN_TEST(validate_drone_color_out_of_range);
    RUN_TEST(validate_drone_color_negative);
    RUN_TEST(validate_drone_zero_scale);

    /* 4. Semantic validation: environment */
    RUN_TEST(validate_environment_zero_num_envs);
    RUN_TEST(validate_environment_negative_world_size);
    RUN_TEST(validate_environment_zero_voxel_size);
    RUN_TEST(validate_environment_negative_spawn_radius);
    RUN_TEST(validate_environment_spawn_height_equal);
    RUN_TEST(validate_environment_negative_min_separation);
    RUN_TEST(validate_environment_zero_max_episode_steps);
    RUN_TEST(validate_environment_unknown_world_type);
    RUN_TEST(validate_environment_all_valid_world_types);

    /* 4b. Sensor validation deep */
    RUN_TEST(validate_sensor_all_valid_types);
    RUN_TEST(validate_sensor_unnormalized_quaternion);
    RUN_TEST(validate_sensor_zero_quaternion);
    RUN_TEST(validate_sensor_nearly_normalized_quaternion_ok);
    RUN_TEST(validate_sensor_negative_sample_rate);
    RUN_TEST(validate_sensor_tof_zero_range);
    RUN_TEST(validate_sensor_lidar_zero_rays);
    RUN_TEST(validate_sensor_lidar_zero_fov);
    RUN_TEST(validate_sensor_camera_zero_dimensions);
    RUN_TEST(validate_sensor_camera_near_ge_far);
    RUN_TEST(validate_sensor_camera_near_equal_far);
    RUN_TEST(validate_sensor_neighbor_zero_k);

    /* 5. Hash determinism and variation */
    RUN_TEST(hash_deterministic_same_config);
    RUN_TEST(hash_deterministic_two_default_configs);
    RUN_TEST(hash_varies_with_single_float_change);
    RUN_TEST(hash_varies_with_string_change);
    RUN_TEST(hash_varies_with_uint_change);
    RUN_TEST(hash_varies_with_bool_change);
    RUN_TEST(hash_varies_across_sections);
    RUN_TEST(hash_null_config_returns_zero);
    RUN_TEST(hash_includes_sensors);
    RUN_TEST(hash_sensor_order_matters);

    /* 6. Conversion completeness */
    RUN_TEST(conversion_all_fields_mapped);
    RUN_TEST(conversion_broadcast_to_range);
    RUN_TEST(conversion_null_params_is_noop);
    RUN_TEST(conversion_exceeds_capacity_is_noop);
    RUN_TEST(conversion_updates_count);
    RUN_TEST(conversion_roundtrip_preserves_values);
    RUN_TEST(conversion_params_to_config_invalid_index_returns_defaults);
    RUN_TEST(conversion_params_to_config_null_returns_defaults);
    RUN_TEST(conversion_config_init_platform_params_convenience);

    /* 7. Defaults produce valid config */
    RUN_TEST(defaults_pass_full_validation);
    RUN_TEST(defaults_drone_passes_validation);
    RUN_TEST(defaults_environment_passes_validation);
    RUN_TEST(defaults_physics_passes_validation);
    RUN_TEST(defaults_no_sensors_is_valid);
    RUN_TEST(defaults_sensor_entries_valid_for_all_types);
    RUN_TEST(defaults_sensor_entry_imu_has_identity_quaternion);
    RUN_TEST(defaults_sensor_entry_camera_has_valid_clip_planes);

    /* 8. Boundary values */
    RUN_TEST(boundary_very_small_positive_mass);
    RUN_TEST(boundary_very_large_mass);
    RUN_TEST(boundary_float_max_values);
    RUN_TEST(boundary_very_small_timestep);
    RUN_TEST(boundary_timestep_just_over_limit);
    RUN_TEST(boundary_max_uint32_agent_count_in_broadcast);

    /* 9. Error reporting fidelity */
    RUN_TEST(validation_null_config_returns_zero);
    RUN_TEST(validation_null_errors_returns_zero);
    RUN_TEST(validation_max_errors_respected);
    RUN_TEST(validation_errors_have_valid_field_names);
    RUN_TEST(validation_error_accumulation_across_sections);

    /* 10. Cross-section and utilities */
    RUN_TEST(config_compare_identical_configs);
    RUN_TEST(config_compare_detects_float_difference);
    RUN_TEST(config_compare_null_returns_nonzero);
    RUN_TEST(config_clone_deep_copy_independence);
    RUN_TEST(config_memory_size_calculation);
    RUN_TEST(config_free_null_safe);
    RUN_TEST(config_free_clears_sensors);

    /* Parse completeness for every section */
    RUN_TEST(parse_all_environment_fields);
    RUN_TEST(parse_all_physics_fields);
    RUN_TEST(parse_all_reward_fields);
    RUN_TEST(parse_all_training_fields);
    RUN_TEST(parse_sensor_type_specific_defaults);
    RUN_TEST(parse_sensor_overrides_type_defaults);

    /* Serialization edge cases */
    RUN_TEST(json_export_small_buffer_fails);
    RUN_TEST(json_export_null_args);
    RUN_TEST(hash_computed_on_successful_load);

    /* Null-safety for individual validators */
    RUN_TEST(validate_drone_null_returns_zero);
    RUN_TEST(validate_environment_null_returns_zero);
    RUN_TEST(validate_physics_null_returns_zero);
    RUN_TEST(validate_sensors_null_errors_returns_zero);
    RUN_TEST(validate_drone_many_simultaneous_errors);

    TEST_SUITE_END();
}
