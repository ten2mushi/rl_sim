/**
 * Engine Buffers Tests
 *
 * Tests for buffer integrity and zero-copy access.
 */

#include "environment_manager.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
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

/* 3.1 Observation buffer has correct size */
TEST(obs_buffer_size) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    float* obs = engine_get_observations(engine);
    ASSERT_NOT_NULL(obs);

    /* Write to last element should not crash */
    uint32_t total = engine_get_total_agents(engine);
    uint32_t dim = engine_get_obs_dim(engine);
    obs[total * dim - 1] = 1.0f;

    engine_destroy(engine);
    return 0;
}

/* 3.2 Action buffer has correct size */
TEST(action_buffer_size) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    float* actions = engine_get_actions(engine);
    ASSERT_NOT_NULL(actions);

    uint32_t total = engine_get_total_agents(engine);
    uint32_t dim = engine_get_action_dim(engine);
    actions[total * dim - 1] = 1.0f;

    engine_destroy(engine);
    return 0;
}

/* 3.3 Reward buffer has correct size */
TEST(reward_buffer_size) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    float* rewards = engine_get_rewards(engine);
    ASSERT_NOT_NULL(rewards);

    uint32_t total = engine_get_total_agents(engine);
    rewards[total - 1] = 1.0f;

    engine_destroy(engine);
    return 0;
}

/* 3.4 Done buffer has correct size */
TEST(done_buffer_size) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    uint8_t* dones = engine_get_dones(engine);
    ASSERT_NOT_NULL(dones);

    uint32_t total = engine_get_total_agents(engine);
    dones[total - 1] = 1;

    engine_destroy(engine);
    return 0;
}

/* 3.5 Truncation buffer has correct size */
TEST(truncation_buffer_size) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    uint8_t* truncations = engine_get_truncations(engine);
    ASSERT_NOT_NULL(truncations);

    uint32_t total = engine_get_total_agents(engine);
    truncations[total - 1] = 1;

    engine_destroy(engine);
    return 0;
}

/* 3.6 Buffers are 32-byte aligned */
TEST(buffers_are_aligned) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    float* obs = engine_get_observations(engine);
    float* actions = engine_get_actions(engine);
    float* rewards = engine_get_rewards(engine);

    ASSERT_TRUE(((uintptr_t)obs & 31) == 0);
    ASSERT_TRUE(((uintptr_t)actions & 31) == 0);
    ASSERT_TRUE(((uintptr_t)rewards & 31) == 0);

    engine_destroy(engine);
    return 0;
}

/* 3.7 Buffers are writable between steps */
TEST(buffers_writable_between_steps) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    float* actions = engine_get_actions(engine);
    uint32_t total = engine_get_total_agents(engine);

    /* Write actions before step */
    for (uint32_t i = 0; i < total * engine->action_dim; i++) {
        actions[i] = 0.5f;
    }

    engine_step(engine);

    /* Write again */
    for (uint32_t i = 0; i < total * engine->action_dim; i++) {
        actions[i] = 0.7f;
    }

    engine_step(engine);

    engine_destroy(engine);
    return 0;
}

/* 3.8 Buffers persist across multiple steps */
TEST(buffers_persist_across_steps) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    float* obs1 = engine_get_observations(engine);
    float* actions1 = engine_get_actions(engine);

    engine_step(engine);

    float* obs2 = engine_get_observations(engine);
    float* actions2 = engine_get_actions(engine);

    /* Pointers should be the same (zero-copy) */
    ASSERT_TRUE(obs1 == obs2);
    ASSERT_TRUE(actions1 == actions2);

    engine_destroy(engine);
    return 0;
}

/* 3.9 Buffer dimensions match config */
TEST(buffer_dimensions_match_config) {
    BatchEngine* engine = create_test_engine(8, 16);
    ASSERT_NOT_NULL(engine);

    ASSERT_EQ(engine_get_num_envs(engine), 8);
    ASSERT_EQ(engine_get_agents_per_env(engine), 16);
    ASSERT_EQ(engine_get_total_agents(engine), 128);
    ASSERT_EQ(engine_get_action_dim(engine), engine->action_dim);

    engine_destroy(engine);
    return 0;
}

/* 3.10 Zero-copy: buffer pointers unchanged after step */
TEST(zero_copy_buffer_pointers) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    float* obs_before = engine_get_observations(engine);
    float* actions_before = engine_get_actions(engine);
    float* rewards_before = engine_get_rewards(engine);
    uint8_t* dones_before = engine_get_dones(engine);

    engine_reset(engine);

    for (int i = 0; i < 10; i++) {
        engine_step(engine);
    }

    float* obs_after = engine_get_observations(engine);
    float* actions_after = engine_get_actions(engine);
    float* rewards_after = engine_get_rewards(engine);
    uint8_t* dones_after = engine_get_dones(engine);

    /* Pointers should be unchanged */
    ASSERT_TRUE(obs_before == obs_after);
    ASSERT_TRUE(actions_before == actions_after);
    ASSERT_TRUE(rewards_before == rewards_after);
    ASSERT_TRUE(dones_before == dones_after);

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Engine Buffers Tests");

    RUN_TEST(obs_buffer_size);
    RUN_TEST(action_buffer_size);
    RUN_TEST(reward_buffer_size);
    RUN_TEST(done_buffer_size);
    RUN_TEST(truncation_buffer_size);
    RUN_TEST(buffers_are_aligned);
    RUN_TEST(buffers_writable_between_steps);
    RUN_TEST(buffers_persist_across_steps);
    RUN_TEST(buffer_dimensions_match_config);
    RUN_TEST(zero_copy_buffer_pointers);

    TEST_SUITE_END();
}
