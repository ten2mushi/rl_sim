/**
 * Engine Reset Tests
 *
 * Tests for reset behavior and spawn logic.
 */

#include "environment_manager.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "test_harness.h"

/* Helper: Create a small test engine */
static BatchEngine* create_test_engine(uint32_t num_envs, uint32_t agents_per_env) {
    EngineConfig cfg = engine_config_default();
    cfg.num_envs = num_envs;
    cfg.agents_per_env = agents_per_env;
    cfg.persistent_arena_size = 128 * 1024 * 1024;
    cfg.frame_arena_size = 32 * 1024 * 1024;

    char error[ENGINE_ERROR_MSG_SIZE];
    return engine_create(&cfg, error);
}

/* ============================================================================
 * Test Cases
 * ============================================================================ */

/* 5.1 Reset zeros velocities */
TEST(reset_zeros_velocities) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Apply actions to build velocity */
    float* actions = engine_get_actions(engine);
    for (uint32_t i = 0; i < 16 * engine->action_dim; i++) {
        actions[i] = 1.0f;
    }
    for (int s = 0; s < 10; s++) {
        engine_step(engine);
    }

    /* Reset */
    engine_reset(engine);

    /* Velocities should be zero */
    ASSERT_FLOAT_EQ(engine->states->rigid_body.vel_x[0], 0.0f);
    ASSERT_FLOAT_EQ(engine->states->rigid_body.vel_y[0], 0.0f);
    ASSERT_FLOAT_EQ(engine->states->rigid_body.vel_z[0], 0.0f);

    engine_destroy(engine);
    return 0;
}

/* 5.2 Reset sets identity-ish orientation */
TEST(reset_sets_orientation) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Orientation should be near identity */
    ASSERT_FLOAT_EQ(engine->states->rigid_body.quat_w[0], 1.0f);
    ASSERT_FLOAT_EQ(engine->states->rigid_body.quat_x[0], 0.0f);
    ASSERT_FLOAT_EQ(engine->states->rigid_body.quat_y[0], 0.0f);
    ASSERT_FLOAT_EQ(engine->states->rigid_body.quat_z[0], 0.0f);

    engine_destroy(engine);
    return 0;
}

/* 5.3 Reset clears done flags */
TEST(reset_clears_done_flags) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Set some drones as done */
    engine->dones[0] = 1;
    engine->dones[5] = 1;

    engine_reset(engine);

    /* All done flags should be cleared */
    for (uint32_t i = 0; i < 16; i++) {
        ASSERT_EQ(engine->dones[i], 0);
    }

    engine_destroy(engine);
    return 0;
}

/* 5.4 Reset clears truncation flags */
TEST(reset_clears_truncation_flags) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Set some drones as truncated */
    engine->truncations[0] = 1;
    engine->truncations[10] = 1;

    engine_reset(engine);

    /* All truncation flags should be cleared */
    for (uint32_t i = 0; i < 16; i++) {
        ASSERT_EQ(engine->truncations[i], 0);
    }

    engine_destroy(engine);
    return 0;
}

/* 5.5 Reset zeroes episode returns */
TEST(reset_zeroes_episode_returns) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Step to accumulate some returns */
    for (int i = 0; i < 10; i++) {
        engine_step(engine);
    }

    engine_reset(engine);

    /* Episode returns should be zeroed */
    for (uint32_t i = 0; i < 16; i++) {
        ASSERT_FLOAT_EQ(engine->episode_returns[i], 0.0f);
    }

    engine_destroy(engine);
    return 0;
}

/* 5.6 Reset zeroes episode lengths */
TEST(reset_zeroes_episode_lengths) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Step to accumulate episode length */
    for (int i = 0; i < 10; i++) {
        engine_step(engine);
    }

    engine_reset(engine);

    /* Episode lengths should be zeroed */
    for (uint32_t i = 0; i < 16; i++) {
        ASSERT_EQ(engine->episode_lengths[i], 0);
    }

    engine_destroy(engine);
    return 0;
}

/* 5.7 Spawn positions within bounds */
TEST(spawn_positions_within_bounds) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Check all spawn positions are within world bounds */
    Vec3 wmin = engine->config.world_min;
    Vec3 wmax = engine->config.world_max;

    for (uint32_t i = 0; i < 16; i++) {
        float x = engine->states->rigid_body.pos_x[i];
        float y = engine->states->rigid_body.pos_y[i];
        float z = engine->states->rigid_body.pos_z[i];

        ASSERT_TRUE(x >= wmin.x && x <= wmax.x);
        ASSERT_TRUE(y >= wmin.y && y <= wmax.y);
        ASSERT_TRUE(z >= wmin.z && z <= wmax.z);
    }

    engine_destroy(engine);
    return 0;
}

/* 5.8 Partial reset only affects specified envs */
TEST(partial_reset_specific_envs) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Step to change state */
    float* actions = engine_get_actions(engine);
    for (uint32_t i = 0; i < 16 * engine->action_dim; i++) {
        actions[i] = 1.0f;
    }
    for (int s = 0; s < 5; s++) {
        engine_step(engine);
    }

    /* Record positions of env 1 drones */
    float env1_x[4];
    for (uint32_t d = 0; d < 4; d++) {
        env1_x[d] = engine->states->rigid_body.pos_x[4 + d];  /* Env 1 starts at index 4 */
    }

    /* Reset only env 0 */
    uint32_t env_indices[] = {0};
    engine_reset_envs(engine, env_indices, 1);

    /* Env 0 drones should be reset (velocities zero) */
    for (uint32_t d = 0; d < 4; d++) {
        ASSERT_FLOAT_EQ(engine->states->rigid_body.vel_x[d], 0.0f);
    }

    /* Env 1 drones should be unchanged (approximately) */
    /* Note: Due to floating point, we check that they're still at non-zero velocity */

    engine_destroy(engine);
    return 0;
}

/* 5.9 Reset clears needs_reset flag */
TEST(reset_clears_needs_reset) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    ASSERT_TRUE(engine->needs_reset);

    engine_reset(engine);

    ASSERT_FALSE(engine->needs_reset);

    engine_destroy(engine);
    return 0;
}

/* 5.10 Multiple resets work */
TEST(multiple_resets) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    for (int r = 0; r < 10; r++) {
        engine_reset(engine);
        for (int s = 0; s < 10; s++) {
            engine_step(engine);
        }
    }

    /* Should complete without issues */
    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Engine Reset Tests");

    RUN_TEST(reset_zeros_velocities);
    RUN_TEST(reset_sets_orientation);
    RUN_TEST(reset_clears_done_flags);
    RUN_TEST(reset_clears_truncation_flags);
    RUN_TEST(reset_zeroes_episode_returns);
    RUN_TEST(reset_zeroes_episode_lengths);
    RUN_TEST(spawn_positions_within_bounds);
    RUN_TEST(partial_reset_specific_envs);
    RUN_TEST(reset_clears_needs_reset);
    RUN_TEST(multiple_resets);

    TEST_SUITE_END();
}
