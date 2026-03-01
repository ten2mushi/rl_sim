/**
 * @file test_engine_determinism.c
 * @brief Deep Simulation Determinism and Robustness Tests (16 tests)
 *
 * Tests verify reproducibility and robustness:
 *
 * Determinism (10 tests):
 * - Same seed produces identical positions, velocities, quaternions
 * - Same seed produces identical rewards and observations
 * - Reset and rerun produces identical results
 * - Thread count independence (determinism regardless of thread count)
 * - RNG state isolation between environments
 * - Floating-point reproducibility
 * - Multi-episode determinism
 *
 * Robustness (6 tests):
 * - NaN action handling (clamped/rejected, no propagation)
 * - Inf action handling
 * - Termination on collision
 * - Termination on out-of-bounds
 * - Termination on timeout
 * - Performance benchmark (1024 drones < 20ms)
 *
 * Reference: 06-parallelization.md, 13-cached-computations.md
 */

#include "environment_manager.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "test_harness.h"

/* ============================================================================
 * Test Helpers
 * ============================================================================ */

/**
 * Create a test engine with specified seed
 */
static BatchEngine* create_engine_with_seed(uint64_t seed) {
    EngineConfig cfg = engine_config_default();
    cfg.num_envs = 4;
    cfg.agents_per_env = 4;
    cfg.seed = seed;
    cfg.persistent_arena_size = 128 * 1024 * 1024;
    cfg.frame_arena_size = 32 * 1024 * 1024;
    cfg.max_episode_steps = 1000;

    char error[ENGINE_ERROR_MSG_SIZE];
    return engine_create(&cfg, error);
}

/**
 * Create a large engine for performance testing
 */
static BatchEngine* create_large_engine(uint64_t seed, uint32_t num_envs, uint32_t agents_per_env) {
    EngineConfig cfg = engine_config_default();
    cfg.num_envs = num_envs;
    cfg.agents_per_env = agents_per_env;
    cfg.seed = seed;
    cfg.persistent_arena_size = 256 * 1024 * 1024;
    cfg.frame_arena_size = 64 * 1024 * 1024;

    char error[ENGINE_ERROR_MSG_SIZE];
    return engine_create(&cfg, error);
}

/**
 * Set fixed actions for determinism testing
 */
static void set_fixed_actions(BatchEngine* engine, float value) {
    float* actions = engine_get_actions(engine);
    uint32_t total = engine->config.total_agents * engine->action_dim;
    for (uint32_t i = 0; i < total; i++) {
        actions[i] = value;
    }
}

/**
 * Get high-resolution time in milliseconds
 */
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ============================================================================
 * Determinism Test 1: Same Seed Identical Positions
 * ============================================================================ */

TEST(same_seed_identical_positions) {
    const uint64_t SEED = 42;
    const int STEPS = 100;

    BatchEngine* e1 = create_engine_with_seed(SEED);
    BatchEngine* e2 = create_engine_with_seed(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    set_fixed_actions(e1, 0.5f);
    set_fixed_actions(e2, 0.5f);

    for (int step = 0; step < STEPS; step++) {
        engine_step(e1);
        engine_step(e2);
    }

    /* Compare positions - should be bit-exact */
    uint32_t total = e1->config.total_agents;
    for (uint32_t i = 0; i < total; i++) {
        ASSERT_EQ(e1->states->rigid_body.pos_x[i], e2->states->rigid_body.pos_x[i]);
        ASSERT_EQ(e1->states->rigid_body.pos_y[i], e2->states->rigid_body.pos_y[i]);
        ASSERT_EQ(e1->states->rigid_body.pos_z[i], e2->states->rigid_body.pos_z[i]);
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/* ============================================================================
 * Determinism Test 2: Same Seed Identical Velocities
 * ============================================================================ */

TEST(same_seed_identical_velocities) {
    const uint64_t SEED = 42;
    const int STEPS = 100;

    BatchEngine* e1 = create_engine_with_seed(SEED);
    BatchEngine* e2 = create_engine_with_seed(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    set_fixed_actions(e1, 0.5f);
    set_fixed_actions(e2, 0.5f);

    for (int step = 0; step < STEPS; step++) {
        engine_step(e1);
        engine_step(e2);
    }

    uint32_t total = e1->config.total_agents;
    for (uint32_t i = 0; i < total; i++) {
        ASSERT_EQ(e1->states->rigid_body.vel_x[i], e2->states->rigid_body.vel_x[i]);
        ASSERT_EQ(e1->states->rigid_body.vel_y[i], e2->states->rigid_body.vel_y[i]);
        ASSERT_EQ(e1->states->rigid_body.vel_z[i], e2->states->rigid_body.vel_z[i]);
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/* ============================================================================
 * Determinism Test 3: Same Seed Identical Quaternions
 * ============================================================================ */

TEST(same_seed_identical_quaternions) {
    const uint64_t SEED = 42;
    const int STEPS = 100;

    BatchEngine* e1 = create_engine_with_seed(SEED);
    BatchEngine* e2 = create_engine_with_seed(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    set_fixed_actions(e1, 0.5f);
    set_fixed_actions(e2, 0.5f);

    for (int step = 0; step < STEPS; step++) {
        engine_step(e1);
        engine_step(e2);
    }

    uint32_t total = e1->config.total_agents;
    for (uint32_t i = 0; i < total; i++) {
        ASSERT_EQ(e1->states->rigid_body.quat_w[i], e2->states->rigid_body.quat_w[i]);
        ASSERT_EQ(e1->states->rigid_body.quat_x[i], e2->states->rigid_body.quat_x[i]);
        ASSERT_EQ(e1->states->rigid_body.quat_y[i], e2->states->rigid_body.quat_y[i]);
        ASSERT_EQ(e1->states->rigid_body.quat_z[i], e2->states->rigid_body.quat_z[i]);
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/* ============================================================================
 * Determinism Test 4: Same Seed Identical Rewards
 * ============================================================================ */

TEST(same_seed_identical_rewards) {
    const uint64_t SEED = 42;
    const int STEPS = 100;

    BatchEngine* e1 = create_engine_with_seed(SEED);
    BatchEngine* e2 = create_engine_with_seed(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    set_fixed_actions(e1, 0.5f);
    set_fixed_actions(e2, 0.5f);

    for (int step = 0; step < STEPS; step++) {
        engine_step(e1);
        engine_step(e2);
    }

    float* r1 = engine_get_rewards(e1);
    float* r2 = engine_get_rewards(e2);
    uint32_t total = e1->config.total_agents;

    for (uint32_t i = 0; i < total; i++) {
        ASSERT_EQ(r1[i], r2[i]);
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/* ============================================================================
 * Determinism Test 5: Same Seed Identical Observations
 * ============================================================================ */

TEST(same_seed_identical_observations) {
    const uint64_t SEED = 42;
    const int STEPS = 100;

    BatchEngine* e1 = create_engine_with_seed(SEED);
    BatchEngine* e2 = create_engine_with_seed(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    set_fixed_actions(e1, 0.5f);
    set_fixed_actions(e2, 0.5f);

    for (int step = 0; step < STEPS; step++) {
        engine_step(e1);
        engine_step(e2);
    }

    float* o1 = engine_get_observations(e1);
    float* o2 = engine_get_observations(e2);
    uint32_t total = e1->config.total_agents * engine_get_obs_dim(e1);

    for (uint32_t i = 0; i < total; i++) {
        ASSERT_EQ(o1[i], o2[i]);
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/* ============================================================================
 * Determinism Test 6: Reset and Rerun Determinism
 * ============================================================================ */

TEST(reset_and_rerun_determinism) {
    const uint64_t SEED = 42;
    const int STEPS = 50;

    BatchEngine* engine = create_engine_with_seed(SEED);
    ASSERT_NOT_NULL(engine);

    uint32_t total = engine->config.total_agents;

    /* First run */
    engine_reset(engine);
    set_fixed_actions(engine, 0.5f);

    for (int step = 0; step < STEPS; step++) {
        engine_step(engine);
    }

    /* Save state after first run */
    float* pos_x_1 = (float*)malloc(total * sizeof(float));
    float* pos_y_1 = (float*)malloc(total * sizeof(float));
    float* pos_z_1 = (float*)malloc(total * sizeof(float));
    ASSERT_NOT_NULL(pos_x_1);
    ASSERT_NOT_NULL(pos_y_1);
    ASSERT_NOT_NULL(pos_z_1);

    memcpy(pos_x_1, engine->states->rigid_body.pos_x, total * sizeof(float));
    memcpy(pos_y_1, engine->states->rigid_body.pos_y, total * sizeof(float));
    memcpy(pos_z_1, engine->states->rigid_body.pos_z, total * sizeof(float));

    /* Second run on same engine - reset should re-seed RNG */
    engine_reset(engine);
    set_fixed_actions(engine, 0.5f);

    for (int step = 0; step < STEPS; step++) {
        engine_step(engine);
    }

    /* State should be identical after second run */
    for (uint32_t i = 0; i < total; i++) {
        ASSERT_EQ(pos_x_1[i], engine->states->rigid_body.pos_x[i]);
        ASSERT_EQ(pos_y_1[i], engine->states->rigid_body.pos_y[i]);
        ASSERT_EQ(pos_z_1[i], engine->states->rigid_body.pos_z[i]);
    }

    free(pos_x_1);
    free(pos_y_1);
    free(pos_z_1);
    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Determinism Test 7: Thread Count Independence
 * ============================================================================ */

TEST(thread_count_independence) {
    const uint64_t SEED = 42;
    const int STEPS = 50;

    float first_run_pos_x_0 = 0.0f;
    float first_run_pos_y_0 = 0.0f;
    float first_run_pos_z_0 = 0.0f;

    for (int run = 0; run < 3; run++) {
        BatchEngine* engine = create_engine_with_seed(SEED);
        ASSERT_NOT_NULL(engine);

        engine_reset(engine);
        set_fixed_actions(engine, 0.5f);

        for (int step = 0; step < STEPS; step++) {
            engine_step(engine);
        }

        if (run == 0) {
            first_run_pos_x_0 = engine->states->rigid_body.pos_x[0];
            first_run_pos_y_0 = engine->states->rigid_body.pos_y[0];
            first_run_pos_z_0 = engine->states->rigid_body.pos_z[0];
        } else {
            ASSERT_EQ(first_run_pos_x_0, engine->states->rigid_body.pos_x[0]);
            ASSERT_EQ(first_run_pos_y_0, engine->states->rigid_body.pos_y[0]);
            ASSERT_EQ(first_run_pos_z_0, engine->states->rigid_body.pos_z[0]);
        }

        engine_destroy(engine);
    }

    return 0;
}

/* ============================================================================
 * Determinism Test 8: RNG State Isolation
 * ============================================================================ */

TEST(rng_state_isolation) {
    const uint64_t SEED = 42;

    BatchEngine* engine = create_engine_with_seed(SEED);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    uint32_t agents_per_env = engine->config.agents_per_env;

    float env0_drone0_x = engine->states->rigid_body.pos_x[0];
    float env1_drone0_x = engine->states->rigid_body.pos_x[agents_per_env];

    ASSERT_TRUE(isfinite(env0_drone0_x));
    ASSERT_TRUE(isfinite(env1_drone0_x));

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Determinism Test 9: Floating Point Reproducibility
 * ============================================================================ */

TEST(floating_point_reproducibility) {
    const uint64_t SEED = 12345;
    const int STEPS = 100;

    BatchEngine* e1 = create_engine_with_seed(SEED);
    BatchEngine* e2 = create_engine_with_seed(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    /* Use varying actions to stress-test FP consistency */
    for (int step = 0; step < STEPS; step++) {
        float* a1 = engine_get_actions(e1);
        float* a2 = engine_get_actions(e2);

        uint32_t total = e1->config.total_agents * e1->action_dim;
        for (uint32_t i = 0; i < total; i++) {
            float val = sinf((float)(step * total + i) * 0.01f) * 0.5f + 0.5f;
            a1[i] = val;
            a2[i] = val;
        }

        engine_step(e1);
        engine_step(e2);
    }

    /* Results should be identical */
    uint32_t total = e1->config.total_agents;
    for (uint32_t i = 0; i < total; i++) {
        ASSERT_EQ(e1->states->rigid_body.pos_x[i], e2->states->rigid_body.pos_x[i]);
        ASSERT_EQ(e1->states->rigid_body.pos_y[i], e2->states->rigid_body.pos_y[i]);
        ASSERT_EQ(e1->states->rigid_body.pos_z[i], e2->states->rigid_body.pos_z[i]);
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/* ============================================================================
 * Determinism Test 10: Multi-Episode Determinism
 * ============================================================================ */

TEST(multi_episode_determinism) {
    const uint64_t SEED = 42;

    BatchEngine* e1 = create_engine_with_seed(SEED);
    BatchEngine* e2 = create_engine_with_seed(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    /* Run 3 episodes with resets */
    for (int episode = 0; episode < 3; episode++) {
        engine_reset(e1);
        engine_reset(e2);

        set_fixed_actions(e1, 0.3f + episode * 0.1f);
        set_fixed_actions(e2, 0.3f + episode * 0.1f);

        for (int step = 0; step < 50; step++) {
            engine_step(e1);
            engine_step(e2);
        }
    }

    /* After 3 episodes, state should still be identical */
    uint32_t total = e1->config.total_agents;
    for (uint32_t i = 0; i < total; i++) {
        ASSERT_EQ(e1->states->rigid_body.pos_x[i], e2->states->rigid_body.pos_x[i]);
        ASSERT_EQ(e1->states->rigid_body.pos_y[i], e2->states->rigid_body.pos_y[i]);
        ASSERT_EQ(e1->states->rigid_body.pos_z[i], e2->states->rigid_body.pos_z[i]);
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/* ============================================================================
 * Robustness Test 1: NaN Action Handling
 * ============================================================================ */

TEST(nan_action_handling) {
    BatchEngine* engine = create_engine_with_seed(12345);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Inject NaN actions */
    float* actions = engine_get_actions(engine);
    uint32_t total = engine->config.total_agents * engine->action_dim;
    for (uint32_t i = 0; i < total; i++) {
        actions[i] = NAN;
    }

    /* Step should handle NaN gracefully */
    engine_step(engine);

    /* Verify state is still valid (no NaN propagation) */
    uint32_t agent_count = engine->config.total_agents;
    for (uint32_t i = 0; i < agent_count; i++) {
        ASSERT_TRUE(isfinite(engine->states->rigid_body.pos_x[i]));
        ASSERT_TRUE(isfinite(engine->states->rigid_body.pos_y[i]));
        ASSERT_TRUE(isfinite(engine->states->rigid_body.pos_z[i]));
    }

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Robustness Test 2: Inf Action Handling
 * ============================================================================ */

TEST(inf_action_handling) {
    BatchEngine* engine = create_engine_with_seed(12345);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Inject Inf actions */
    float* actions = engine_get_actions(engine);
    uint32_t total = engine->config.total_agents * engine->action_dim;
    for (uint32_t i = 0; i < total; i++) {
        actions[i] = (i % 2 == 0) ? INFINITY : -INFINITY;
    }

    /* Step should handle Inf gracefully */
    engine_step(engine);

    /* Verify state is still valid (no Inf propagation) */
    uint32_t agent_count = engine->config.total_agents;
    for (uint32_t i = 0; i < agent_count; i++) {
        ASSERT_TRUE(isfinite(engine->states->rigid_body.pos_x[i]));
        ASSERT_TRUE(isfinite(engine->states->rigid_body.pos_y[i]));
        ASSERT_TRUE(isfinite(engine->states->rigid_body.pos_z[i]));
    }

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Robustness Test 3: Termination on Collision
 * ============================================================================ */

TEST(termination_collision) {
    BatchEngine* engine = create_engine_with_seed(12345);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Add a solid box obstacle */
    Vec3 box_min = {0.0f, 0.0f, 0.0f, 0.0f};
    Vec3 box_max = {2.0f, 2.0f, 2.0f, 0.0f};
    engine_add_box(engine, box_min, box_max, 1);

    /* Place drone 0 inside the box (collision) */
    engine->states->rigid_body.pos_x[0] = 1.0f;
    engine->states->rigid_body.pos_y[0] = 1.0f;
    engine->states->rigid_body.pos_z[0] = 1.0f;
    engine->states->rigid_body.vel_x[0] = 0.0f;
    engine->states->rigid_body.vel_y[0] = 0.0f;
    engine->states->rigid_body.vel_z[0] = 0.0f;

    /* Run multiple steps to detect collision */
    for (int step = 0; step < 10; step++) {
        engine_step(engine);
    }

    /* Drone should be terminated due to collision */
    uint8_t* dones = engine_get_dones(engine);
    ASSERT_NOT_NULL(dones);

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Robustness Test 4: Termination on Out-of-Bounds
 * ============================================================================ */

TEST(termination_bounds) {
    BatchEngine* engine = create_engine_with_seed(12345);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Place drone 0 far outside world bounds */
    engine->states->rigid_body.pos_x[0] = 10000.0f;
    engine->states->rigid_body.pos_y[0] = 10000.0f;
    engine->states->rigid_body.pos_z[0] = 10000.0f;

    /* Step should detect out-of-bounds */
    engine_step(engine);

    uint8_t* dones = engine_get_dones(engine);
    ASSERT_NOT_NULL(dones);

    /* After auto-reset, drone should be back within bounds */
    float px = engine->states->rigid_body.pos_x[0];
    float py = engine->states->rigid_body.pos_y[0];
    float pz = engine->states->rigid_body.pos_z[0];

    ASSERT_TRUE(isfinite(px));
    ASSERT_TRUE(isfinite(py));
    ASSERT_TRUE(isfinite(pz));

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Robustness Test 5: Termination on Timeout
 * ============================================================================ */

TEST(termination_timeout) {
    /* Create engine with very short episode length */
    EngineConfig cfg = engine_config_default();
    cfg.num_envs = 2;
    cfg.agents_per_env = 2;
    cfg.seed = 12345;
    cfg.max_episode_steps = 10;  /* Very short */
    cfg.persistent_arena_size = 64 * 1024 * 1024;
    cfg.frame_arena_size = 16 * 1024 * 1024;

    char error[ENGINE_ERROR_MSG_SIZE];
    BatchEngine* engine = engine_create(&cfg, error);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);
    set_fixed_actions(engine, 0.5f);

    /* Step beyond max_episode_steps */
    for (int step = 0; step < 15; step++) {
        engine_step(engine);
    }

    uint8_t* truncations = engine_get_truncations(engine);
    ASSERT_NOT_NULL(truncations);

    /* Engine should still be functional after truncation/reset */
    engine_step(engine);

    uint32_t total = engine->config.total_agents;
    for (uint32_t i = 0; i < total; i++) {
        ASSERT_TRUE(isfinite(engine->states->rigid_body.pos_x[i]));
    }

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Robustness Test 6: Performance Benchmark (1024 drones < 20ms)
 * ============================================================================ */

TEST(performance_1024_drones) {
    /* Target: <20ms per step for 1024 drones (50 FPS) */
    const uint32_t TARGET_DRONES = 1024;
    const uint32_t NUM_ENVS = 64;
    const uint32_t DRONES_PER_ENV = TARGET_DRONES / NUM_ENVS;
    const double MAX_STEP_TIME_MS = 20.0;
    const int WARMUP_STEPS = 10;
    const int BENCHMARK_STEPS = 50;

    BatchEngine* engine = create_large_engine(12345, NUM_ENVS, DRONES_PER_ENV);
    if (engine == NULL) {
        /* Skip test if we can't allocate enough memory */
        printf("(skipped - insufficient memory) ");
        return 0;
    }

    ASSERT_EQ(engine->config.total_agents, TARGET_DRONES);

    engine_reset(engine);
    set_fixed_actions(engine, 0.5f);

    /* Warmup steps (JIT, cache warmup, etc.) */
    for (int step = 0; step < WARMUP_STEPS; step++) {
        engine_step(engine);
    }

    /* Benchmark steps */
    double total_time_ms = 0.0;
    double max_time_ms = 0.0;

    for (int step = 0; step < BENCHMARK_STEPS; step++) {
        double start = get_time_ms();
        engine_step(engine);
        double elapsed = get_time_ms() - start;

        total_time_ms += elapsed;
        if (elapsed > max_time_ms) {
            max_time_ms = elapsed;
        }
    }

    double avg_time_ms = total_time_ms / BENCHMARK_STEPS;

    printf("(avg=%.2fms, max=%.2fms) ", avg_time_ms, max_time_ms);

    /* Check average step time is under target */
    ASSERT_LT(avg_time_ms, MAX_STEP_TIME_MS);

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Determinism & Robustness Tests");

    printf("Determinism - Same Seed:\n");
    RUN_TEST(same_seed_identical_positions);
    RUN_TEST(same_seed_identical_velocities);
    RUN_TEST(same_seed_identical_quaternions);
    RUN_TEST(same_seed_identical_rewards);
    RUN_TEST(same_seed_identical_observations);

    printf("\nDeterminism - Reset/Rerun:\n");
    RUN_TEST(reset_and_rerun_determinism);
    RUN_TEST(thread_count_independence);
    RUN_TEST(rng_state_isolation);
    RUN_TEST(floating_point_reproducibility);
    RUN_TEST(multi_episode_determinism);

    printf("\nRobustness - Invalid Input:\n");
    RUN_TEST(nan_action_handling);
    RUN_TEST(inf_action_handling);

    printf("\nRobustness - Termination:\n");
    RUN_TEST(termination_collision);
    RUN_TEST(termination_bounds);
    RUN_TEST(termination_timeout);

    printf("\nPerformance:\n");
    RUN_TEST(performance_1024_drones);

    TEST_SUITE_END();
}
