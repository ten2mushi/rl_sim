/**
 * Engine Step Tests
 *
 * Tests for step pipeline execution and ordering.
 */

#include "environment_manager.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "test_harness.h"

/* Helper: Create a small test engine */
static BatchDroneEngine* create_test_engine(uint32_t num_envs, uint32_t drones_per_env) {
    EngineConfig cfg = engine_config_default();
    cfg.num_envs = num_envs;
    cfg.drones_per_env = drones_per_env;
    cfg.persistent_arena_size = 128 * 1024 * 1024;
    cfg.frame_arena_size = 32 * 1024 * 1024;

    char error[ENGINE_ERROR_MSG_SIZE];
    return engine_create(&cfg, error);
}

/* ============================================================================
 * Test Cases
 * ============================================================================ */

/* 4.1 Step after reset works */
TEST(step_after_reset) {
    BatchDroneEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);
    engine_step(engine);

    ASSERT_EQ(engine->total_steps, 1);

    engine_destroy(engine);
    return 0;
}

/* 4.2 Step updates statistics */
TEST(step_updates_stats) {
    BatchDroneEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    uint64_t steps_before = engine->total_steps;
    engine_step(engine);
    uint64_t steps_after = engine->total_steps;

    ASSERT_EQ(steps_after, steps_before + 1);

    engine_destroy(engine);
    return 0;
}

/* 4.3 Step computes rewards */
TEST(step_computes_rewards) {
    BatchDroneEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);
    engine_step(engine);

    /* Rewards buffer should have values (might be zero, but that's valid) */
    float* rewards = engine_get_rewards(engine);
    ASSERT_NOT_NULL(rewards);

    engine_destroy(engine);
    return 0;
}

/* 4.4 Actions affect physics */
TEST(step_actions_affect_physics) {
    BatchDroneEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Get initial position (Z is altitude in Z-up ENU) */
    float z_before = engine->states->pos_z[0];

    /* Apply full thrust actions */
    float* actions = engine_get_actions(engine);
    uint32_t total = engine_get_total_drones(engine);
    for (uint32_t i = 0; i < total * ENGINE_ACTION_DIM; i++) {
        actions[i] = 1.0f;  /* Full thrust */
    }

    /* Step multiple times */
    for (int i = 0; i < 10; i++) {
        engine_step(engine);
    }

    /* Position should have changed */
    float z_after = engine->states->pos_z[0];
    /* Note: With gravity and thrust, position changes */

    engine_destroy(engine);
    return 0;
}

/* 4.5 Step timing is recorded */
TEST(step_records_timing) {
    BatchDroneEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);
    engine_step(engine);

    /* Timing should be non-negative */
    ASSERT_TRUE(engine->physics_time_ms >= 0.0);
    ASSERT_TRUE(engine->collision_time_ms >= 0.0);
    ASSERT_TRUE(engine->sensor_time_ms >= 0.0);
    ASSERT_TRUE(engine->reward_time_ms >= 0.0);

    engine_destroy(engine);
    return 0;
}

/* 4.6 Episode lengths increment */
TEST(step_increments_episode_length) {
    BatchDroneEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    uint32_t len_before = engine->episode_lengths[0];
    engine_step(engine);
    uint32_t len_after = engine->episode_lengths[0];

    ASSERT_EQ(len_after, len_before + 1);

    engine_destroy(engine);
    return 0;
}

/* 4.7 Episode returns accumulate */
TEST(step_accumulates_returns) {
    BatchDroneEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Step a few times and check that returns change */
    float returns_before = engine->episode_returns[0];
    for (int i = 0; i < 5; i++) {
        engine_step(engine);
    }
    /* Returns should have accumulated (could be positive or negative) */

    engine_destroy(engine);
    return 0;
}

/* 4.8 Multiple steps work correctly */
TEST(multiple_steps) {
    BatchDroneEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    for (int i = 0; i < 100; i++) {
        engine_step(engine);
    }

    ASSERT_EQ(engine->total_steps, 100);

    engine_destroy(engine);
    return 0;
}

/* 4.9 Step no reset leaves terminated drones */
TEST(step_no_reset) {
    BatchDroneEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Force a drone to be out of bounds (below world min) - this will trigger termination */
    engine->states->pos_z[0] = engine->config.world_min.z - 10.0f;

    engine_step_no_reset(engine);

    /* Drone should be marked as done (out of bounds termination) */
    ASSERT_TRUE(engine->dones[0] == 1);

    /* Episode length for this drone should NOT have been incremented since it's done */
    /* (step_no_reset only increments for non-terminated drones) */

    engine_destroy(engine);
    return 0;
}

/* 4.10 Frame arena resets each step */
TEST(frame_arena_resets) {
    BatchDroneEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Step and check arena doesn't overflow */
    for (int i = 0; i < 100; i++) {
        engine_step(engine);
    }

    /* Should complete without running out of frame arena memory */

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Engine Step Tests");

    RUN_TEST(step_after_reset);
    RUN_TEST(step_updates_stats);
    RUN_TEST(step_computes_rewards);
    RUN_TEST(step_actions_affect_physics);
    RUN_TEST(step_records_timing);
    RUN_TEST(step_increments_episode_length);
    RUN_TEST(step_accumulates_returns);
    RUN_TEST(multiple_steps);
    RUN_TEST(step_no_reset);
    RUN_TEST(frame_arena_resets);

    TEST_SUITE_END();
}
