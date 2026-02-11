/**
 * Engine Configuration Tests
 *
 * Tests for configuration validation and default values.
 */

#include "environment_manager.h"
#include <stdio.h>
#include <string.h>
#include "test_harness.h"

/* ============================================================================
 * Test Cases
 * ============================================================================ */

/* 1.1 Default config produces valid engine */
TEST(config_default_is_valid) {
    EngineConfig cfg = engine_config_default();
    char error[ENGINE_ERROR_MSG_SIZE];
    ASSERT_EQ(engine_config_validate(&cfg, error), 0);
    return 0;
}

/* 1.2 Zero environments rejected */
TEST(config_zero_envs_rejected) {
    EngineConfig cfg = engine_config_default();
    cfg.num_envs = 0;
    char error[ENGINE_ERROR_MSG_SIZE];
    ASSERT_NE(engine_config_validate(&cfg, error), 0);
    return 0;
}

/* 1.3 Zero drones per env rejected */
TEST(config_zero_drones_rejected) {
    EngineConfig cfg = engine_config_default();
    cfg.drones_per_env = 0;
    char error[ENGINE_ERROR_MSG_SIZE];
    ASSERT_NE(engine_config_validate(&cfg, error), 0);
    return 0;
}

/* 1.4 Invalid world bounds rejected (min >= max) */
TEST(config_invalid_world_bounds_rejected) {
    EngineConfig cfg = engine_config_default();
    cfg.world_min.x = 10.0f;
    cfg.world_max.x = 5.0f;  /* min > max */
    char error[ENGINE_ERROR_MSG_SIZE];
    ASSERT_NE(engine_config_validate(&cfg, error), 0);
    return 0;
}

/* 1.5 Zero timestep rejected */
TEST(config_zero_timestep_rejected) {
    EngineConfig cfg = engine_config_default();
    cfg.timestep = 0.0f;
    char error[ENGINE_ERROR_MSG_SIZE];
    ASSERT_NE(engine_config_validate(&cfg, error), 0);
    return 0;
}

/* 1.6 Negative gravity rejected */
TEST(config_negative_gravity_rejected) {
    EngineConfig cfg = engine_config_default();
    cfg.gravity = -9.81f;
    char error[ENGINE_ERROR_MSG_SIZE];
    ASSERT_NE(engine_config_validate(&cfg, error), 0);
    return 0;
}

/* 1.7 Zero max_episode_steps rejected */
TEST(config_zero_max_steps_rejected) {
    EngineConfig cfg = engine_config_default();
    cfg.max_episode_steps = 0;
    char error[ENGINE_ERROR_MSG_SIZE];
    ASSERT_NE(engine_config_validate(&cfg, error), 0);
    return 0;
}

/* 1.8 Arena size too small rejected */
TEST(config_small_arena_rejected) {
    EngineConfig cfg = engine_config_default();
    cfg.persistent_arena_size = 1024;  /* Too small */
    char error[ENGINE_ERROR_MSG_SIZE];
    ASSERT_NE(engine_config_validate(&cfg, error), 0);
    return 0;
}

/* 1.9 Default values are sensible */
TEST(config_default_values) {
    EngineConfig cfg = engine_config_default();

    ASSERT_EQ(cfg.num_envs, ENGINE_DEFAULT_NUM_ENVS);
    ASSERT_EQ(cfg.drones_per_env, ENGINE_DEFAULT_DRONES_PER_ENV);
    ASSERT_FLOAT_NEAR(cfg.timestep, ENGINE_DEFAULT_TIMESTEP, 0.001f);
    ASSERT_EQ(cfg.physics_substeps, ENGINE_DEFAULT_PHYSICS_SUBSTEPS);
    ASSERT_FLOAT_NEAR(cfg.gravity, ENGINE_DEFAULT_GRAVITY, 0.01f);
    ASSERT_EQ(cfg.max_episode_steps, ENGINE_DEFAULT_MAX_EPISODE_STEPS);

    return 0;
}

/* 1.10 Domain randomization bounds validated */
TEST(config_domain_randomization_bounds) {
    EngineConfig cfg = engine_config_default();

    cfg.domain_randomization = -0.5f;
    char error[ENGINE_ERROR_MSG_SIZE];
    ASSERT_NE(engine_config_validate(&cfg, error), 0);

    cfg.domain_randomization = 1.5f;
    ASSERT_NE(engine_config_validate(&cfg, error), 0);

    cfg.domain_randomization = 0.5f;
    ASSERT_EQ(engine_config_validate(&cfg, error), 0);

    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Engine Configuration Tests");

    RUN_TEST(config_default_is_valid);
    RUN_TEST(config_zero_envs_rejected);
    RUN_TEST(config_zero_drones_rejected);
    RUN_TEST(config_invalid_world_bounds_rejected);
    RUN_TEST(config_zero_timestep_rejected);
    RUN_TEST(config_negative_gravity_rejected);
    RUN_TEST(config_zero_max_steps_rejected);
    RUN_TEST(config_small_arena_rejected);
    RUN_TEST(config_default_values);
    RUN_TEST(config_domain_randomization_bounds);

    TEST_SUITE_END();
}
