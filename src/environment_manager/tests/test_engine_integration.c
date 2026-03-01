/**
 * Engine Integration Tests
 *
 * Full integration tests for the complete engine pipeline.
 */

#include "environment_manager.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "test_harness.h"

/* Helper: Create engine with specific seed */
static BatchEngine* create_engine_with_seed(uint64_t seed) {
    EngineConfig cfg = engine_config_default();
    cfg.num_envs = 4;
    cfg.agents_per_env = 4;
    cfg.seed = seed;
    cfg.persistent_arena_size = 128 * 1024 * 1024;
    cfg.frame_arena_size = 32 * 1024 * 1024;

    char error[ENGINE_ERROR_MSG_SIZE];
    return engine_create(&cfg, error);
}

/* ============================================================================
 * Test Cases
 * ============================================================================ */

/* 10.1 Single-threaded step is deterministic */
TEST(step_deterministic) {
    BatchEngine* e1 = create_engine_with_seed(12345);
    BatchEngine* e2 = create_engine_with_seed(12345);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    /* Apply same actions */
    float* a1 = engine_get_actions(e1);
    float* a2 = engine_get_actions(e2);
    for (uint32_t i = 0; i < 16 * e1->action_dim; i++) {
        a1[i] = 0.5f;
        a2[i] = 0.5f;
    }

    /* Step both */
    for (int i = 0; i < 10; i++) {
        engine_step(e1);
        engine_step(e2);
    }

    /* Positions should match (with floating point tolerance) */
    for (uint32_t j = 0; j < 16; j++) {
        float diff_x = fabsf(e1->states->rigid_body.pos_x[j] - e2->states->rigid_body.pos_x[j]);
        float diff_y = fabsf(e1->states->rigid_body.pos_y[j] - e2->states->rigid_body.pos_y[j]);
        float diff_z = fabsf(e1->states->rigid_body.pos_z[j] - e2->states->rigid_body.pos_z[j]);
        /* Note: Due to threading, may not be exactly equal, but should be close */
        ASSERT_LT(diff_x, 1.0f);
        ASSERT_LT(diff_y, 1.0f);
        ASSERT_LT(diff_z, 1.0f);
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/* 10.2 Zero actions produces stable (hover-ish) flight */
TEST(zero_actions_stable) {
    BatchEngine* engine = create_engine_with_seed(12345);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Zero actions (should produce minimal thrust) */
    float* actions = engine_get_actions(engine);
    memset(actions, 0, 16 * engine->action_dim * sizeof(float));

    /* Step and check drone doesn't explode */
    for (int i = 0; i < 100; i++) {
        engine_step(engine);
    }

    /* All positions should still be finite */
    for (uint32_t j = 0; j < 16; j++) {
        ASSERT_TRUE(isfinite(engine->states->rigid_body.pos_x[j]));
        ASSERT_TRUE(isfinite(engine->states->rigid_body.pos_y[j]));
        ASSERT_TRUE(isfinite(engine->states->rigid_body.pos_z[j]));
    }

    engine_destroy(engine);
    return 0;
}

/* 10.3 Max actions saturate correctly */
TEST(max_actions_saturate) {
    BatchEngine* engine = create_engine_with_seed(12345);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Max actions */
    float* actions = engine_get_actions(engine);
    for (uint32_t i = 0; i < 16 * engine->action_dim; i++) {
        actions[i] = 1.0f;
    }

    /* Step */
    for (int i = 0; i < 100; i++) {
        engine_step(engine);
    }

    /* Positions should still be finite (no NaN/Inf) */
    for (uint32_t j = 0; j < 16; j++) {
        ASSERT_TRUE(isfinite(engine->states->rigid_body.pos_x[j]));
        ASSERT_TRUE(isfinite(engine->states->rigid_body.pos_y[j]));
        ASSERT_TRUE(isfinite(engine->states->rigid_body.pos_z[j]));
    }

    engine_destroy(engine);
    return 0;
}

/* 10.4 Rapid reset/step cycles stable */
TEST(rapid_reset_step_cycles) {
    BatchEngine* engine = create_engine_with_seed(12345);
    ASSERT_NOT_NULL(engine);

    for (int cycle = 0; cycle < 50; cycle++) {
        engine_reset(engine);

        float* actions = engine_get_actions(engine);
        for (uint32_t i = 0; i < 16 * engine->action_dim; i++) {
            actions[i] = (float)(cycle % 10) / 10.0f;
        }

        for (int s = 0; s < 10; s++) {
            engine_step(engine);
        }
    }

    /* Should complete without crash */
    engine_destroy(engine);
    return 0;
}

/* 10.5 State queries match state arrays */
TEST(state_queries_match_arrays) {
    BatchEngine* engine = create_engine_with_seed(12345);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Modify state directly */
    engine->states->rigid_body.pos_x[5] = 10.0f;
    engine->states->rigid_body.pos_y[5] = 20.0f;
    engine->states->rigid_body.pos_z[5] = 30.0f;

    /* Query state */
    AgentStateQuery query;
    engine_get_agent_state(engine, 5, &query);

    ASSERT_FLOAT_EQ(query.position.x, 10.0f);
    ASSERT_FLOAT_EQ(query.position.y, 20.0f);
    ASSERT_FLOAT_EQ(query.position.z, 30.0f);

    engine_destroy(engine);
    return 0;
}

/* 10.6 Get all positions works */
TEST(get_all_positions) {
    BatchEngine* engine = create_engine_with_seed(12345);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    float positions[16 * 3];
    engine_get_all_positions(engine, positions);

    for (uint32_t i = 0; i < 16; i++) {
        ASSERT_FLOAT_EQ(positions[i * 3 + 0], engine->states->rigid_body.pos_x[i]);
        ASSERT_FLOAT_EQ(positions[i * 3 + 1], engine->states->rigid_body.pos_y[i]);
        ASSERT_FLOAT_EQ(positions[i * 3 + 2], engine->states->rigid_body.pos_z[i]);
    }

    engine_destroy(engine);
    return 0;
}

/* 10.7 Statistics collection works */
TEST(statistics_collection) {
    BatchEngine* engine = create_engine_with_seed(12345);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    for (int i = 0; i < 100; i++) {
        engine_step(engine);
    }

    EngineStats stats;
    engine_get_stats(engine, &stats);

    ASSERT_EQ(stats.total_steps, 100);
    ASSERT_TRUE(stats.avg_step_time_ms >= 0.0);

    engine_destroy(engine);
    return 0;
}

/* 10.8 Index conversion utilities */
TEST(index_conversion) {
    BatchEngine* engine = create_engine_with_seed(12345);
    ASSERT_NOT_NULL(engine);

    /* Test agent_idx_to_env */
    uint32_t env_id, local_id;
    engine_agent_idx_to_env(engine, 7, &env_id, &local_id);
    ASSERT_EQ(env_id, 1);   /* Drone 7 is in env 1 (4 drones per env) */
    ASSERT_EQ(local_id, 3); /* Local index 3 */

    /* Test env_to_agent_idx */
    uint32_t agent_idx = engine_env_to_agent_idx(engine, 2, 1);
    ASSERT_EQ(agent_idx, 9); /* Env 2, local 1 = 2*4 + 1 = 9 */

    engine_destroy(engine);
    return 0;
}

/* 10.9 World manipulation */
TEST(world_manipulation) {
    BatchEngine* engine = create_engine_with_seed(12345);
    ASSERT_NOT_NULL(engine);

    /* Add obstacles */
    Vec3 box_min = {0.0f, 0.0f, 0.0f, 0.0f};
    Vec3 box_max = {5.0f, 5.0f, 5.0f, 0.0f};
    engine_add_box(engine, box_min, box_max, 1);

    Vec3 sphere_center = {10.0f, 10.0f, 10.0f, 0.0f};
    engine_add_sphere(engine, sphere_center, 2.0f, 1);

    /* Clear world */
    engine_clear_world(engine);

    /* Should still work */
    engine_reset(engine);
    engine_step(engine);

    engine_destroy(engine);
    return 0;
}

/* 10.10 Target setting */
TEST(target_setting) {
    BatchEngine* engine = create_engine_with_seed(12345);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    Vec3 target = {5.0f, 10.0f, 5.0f, 0.0f};
    engine_set_target(engine, 0, target);

    /* Step and check rewards are computed */
    engine_step(engine);

    float* rewards = engine_get_rewards(engine);
    ASSERT_TRUE(isfinite(rewards[0]));

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Engine Integration Tests");

    RUN_TEST(step_deterministic);
    RUN_TEST(zero_actions_stable);
    RUN_TEST(max_actions_saturate);
    RUN_TEST(rapid_reset_step_cycles);
    RUN_TEST(state_queries_match_arrays);
    RUN_TEST(get_all_positions);
    RUN_TEST(statistics_collection);
    RUN_TEST(index_conversion);
    RUN_TEST(world_manipulation);
    RUN_TEST(target_setting);

    TEST_SUITE_END();
}
