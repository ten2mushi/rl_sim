/**
 * Engine Lifecycle Tests -- Comprehensive Behavioral Specification
 *
 * Tests for engine creation, destruction, validation, and pointer stability
 * across resets and steps. These tests operate at the BatchEngine level
 * (not PufferEnv) and serve as the authoritative specification for engine
 * lifecycle behavior.
 *
 * KEY BUG DOCUMENTED:
 *   sensor_system.obs_dim (max_obs_dim) != engine->obs_dim (actual_obs_dim).
 *   When sensor_system_set_external_buffer() redirects the sensor obs buffer
 *   to engine->observations, any sensor_system function that uses obs_dim for
 *   stride or memset size will overflow the engine's observations buffer.
 *   This corrupts everything allocated after it in the persistent arena.
 */

#include "environment_manager.h"
#include <stdio.h>
#include <string.h>
#include "test_harness.h"

/* ============================================================================
 * Helpers
 * ============================================================================ */

/** Create a small test engine with given parameters */
static BatchEngine* create_test_engine(uint32_t num_envs, uint32_t agents_per_env) {
    EngineConfig cfg = engine_config_default();
    cfg.num_envs = num_envs;
    cfg.agents_per_env = agents_per_env;
    cfg.persistent_arena_size = 128 * 1024 * 1024;  /* 128 MB */
    cfg.frame_arena_size = 32 * 1024 * 1024;        /* 32 MB */

    char error[ENGINE_ERROR_MSG_SIZE];
    return engine_create(&cfg, error);
}

/** Create a test engine with explicit seed */
static BatchEngine* create_test_engine_seeded(uint32_t num_envs, uint32_t agents_per_env,
                                                     uint64_t seed) {
    EngineConfig cfg = engine_config_default();
    cfg.num_envs = num_envs;
    cfg.agents_per_env = agents_per_env;
    cfg.seed = seed;
    cfg.persistent_arena_size = 128 * 1024 * 1024;
    cfg.frame_arena_size = 32 * 1024 * 1024;

    char error[ENGINE_ERROR_MSG_SIZE];
    return engine_create(&cfg, error);
}

/* ============================================================================
 * Section 1: Creation Tests
 * ============================================================================ */

/** engine_create with valid config succeeds */
TEST(engine_create_succeeds) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);
    ASSERT_TRUE(engine->initialized);
    engine_destroy(engine);
    return 0;
}

/** engine_create with invalid config (num_envs=0) returns NULL */
TEST(engine_create_invalid_returns_null) {
    EngineConfig cfg = engine_config_default();
    cfg.num_envs = 0;  /* Invalid */

    char error[ENGINE_ERROR_MSG_SIZE];
    BatchEngine* engine = engine_create(&cfg, error);
    ASSERT_NULL(engine);
    return 0;
}

/** engine_create with invalid config (agents_per_env=0) returns NULL */
TEST(engine_create_zero_drones_returns_null) {
    EngineConfig cfg = engine_config_default();
    cfg.agents_per_env = 0;

    char error[ENGINE_ERROR_MSG_SIZE];
    BatchEngine* engine = engine_create(&cfg, error);
    ASSERT_NULL(engine);
    return 0;
}

/** engine_create with NULL config returns NULL */
TEST(engine_create_null_config_returns_null) {
    char error[ENGINE_ERROR_MSG_SIZE];
    BatchEngine* engine = engine_create(NULL, error);
    ASSERT_NULL(engine);
    return 0;
}

/** engine_create fills in error message on failure */
TEST(engine_create_populates_error_on_failure) {
    EngineConfig cfg = engine_config_default();
    cfg.num_envs = 0;

    char error[ENGINE_ERROR_MSG_SIZE] = {0};
    engine_create(&cfg, error);
    ASSERT_TRUE(strlen(error) > 0);
    return 0;
}

/** engine_create clears error message on success */
TEST(engine_create_clears_error_on_success) {
    EngineConfig cfg = engine_config_default();
    cfg.num_envs = 1;
    cfg.agents_per_env = 4;
    cfg.persistent_arena_size = 128 * 1024 * 1024;
    cfg.frame_arena_size = 32 * 1024 * 1024;

    char error[ENGINE_ERROR_MSG_SIZE] = "stale error";
    BatchEngine* engine = engine_create(&cfg, error);
    ASSERT_NOT_NULL(engine);
    ASSERT_EQ(error[0], '\0');  /* Error should be cleared */
    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Section 2: Subsystem Allocation Tests
 * ============================================================================ */

/** engine_create allocates all subsystems */
TEST(engine_create_allocates_subsystems) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    ASSERT_NOT_NULL(engine->states);
    ASSERT_NOT_NULL(engine->params);
    ASSERT_NOT_NULL(engine->world);
    ASSERT_NOT_NULL(engine->physics);
    ASSERT_NOT_NULL(engine->collision);
    ASSERT_NOT_NULL(engine->sensors);
    ASSERT_NOT_NULL(engine->rewards);
    ASSERT_NOT_NULL(engine->thread_pool);
    ASSERT_NOT_NULL(engine->scheduler);

    engine_destroy(engine);
    return 0;
}

/** engine_create correctly computes total_agents */
TEST(engine_computes_total_agents) {
    BatchEngine* engine = create_test_engine(8, 16);
    ASSERT_NOT_NULL(engine);
    ASSERT_EQ(engine->config.total_agents, 8u * 16u);
    engine_destroy(engine);
    return 0;
}

/** engine_create initializes env_ids correctly */
TEST(engine_initializes_env_ids) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    for (uint32_t env = 0; env < 4; env++) {
        for (uint32_t d = 0; d < 4; d++) {
            uint32_t idx = env * 4 + d;
            ASSERT_EQ(engine->env_ids[idx], env);
        }
    }

    engine_destroy(engine);
    return 0;
}

/** engine_create allocates all buffers */
TEST(engine_allocates_buffers) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    ASSERT_NOT_NULL(engine->observations);
    ASSERT_NOT_NULL(engine->actions);
    ASSERT_NOT_NULL(engine->rewards_buffer);
    ASSERT_NOT_NULL(engine->dones);
    ASSERT_NOT_NULL(engine->truncations);
    ASSERT_NOT_NULL(engine->term_success);
    ASSERT_NOT_NULL(engine->term_collision);
    ASSERT_NOT_NULL(engine->term_out_of_bounds);
    ASSERT_NOT_NULL(engine->term_timeout);

    engine_destroy(engine);
    return 0;
}

/** engine_create allocates episode tracking arrays */
TEST(engine_allocates_episode_tracking) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);

    ASSERT_NOT_NULL(engine->episode_returns);
    ASSERT_NOT_NULL(engine->episode_lengths);
    ASSERT_NOT_NULL(engine->env_ids);

    engine_destroy(engine);
    return 0;
}

/** engine_create sets correct obs_dim for default sensors (IMU+Position+Velocity=15) */
TEST(engine_obs_dim_correct_for_defaults) {
    BatchEngine* engine = create_test_engine(1, 4);
    ASSERT_NOT_NULL(engine);
    /* IMU(6) + Position(3) + Velocity(6) = 15 */
    ASSERT_EQ(engine->obs_dim, 15u);
    engine_destroy(engine);
    return 0;
}

/** engine_create sets action_dim to 4 (quadcopter) */
TEST(engine_action_dim_is_4) {
    BatchEngine* engine = create_test_engine(1, 4);
    ASSERT_NOT_NULL(engine);
    ASSERT_EQ(engine->action_dim, 4u);
    ASSERT_EQ(engine->action_dim, 4u);
    engine_destroy(engine);
    return 0;
}

/** engine_create initializes state flags correctly */
TEST(engine_state_flags_after_create) {
    BatchEngine* engine = create_test_engine(1, 4);
    ASSERT_NOT_NULL(engine);

    ASSERT_TRUE(engine->initialized);
    ASSERT_TRUE(engine->needs_reset);  /* Must reset before stepping */
    ASSERT_EQ(engine->total_steps, 0u);
    ASSERT_EQ(engine->total_episodes, 0u);

    engine_destroy(engine);
    return 0;
}

/** engine_create allocates both arenas */
TEST(engine_arenas_allocated) {
    BatchEngine* engine = create_test_engine(1, 4);
    ASSERT_NOT_NULL(engine);
    ASSERT_NOT_NULL(engine->persistent_arena);
    ASSERT_NOT_NULL(engine->frame_arena);
    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Section 3: Destruction Tests
 * ============================================================================ */

/** engine_destroy handles NULL gracefully */
TEST(engine_destroy_null_safe) {
    engine_destroy(NULL);  /* Should not crash */
    return 0;
}

/** Sequential create/destroy cycles don't leak (valgrind/asan test) */
TEST(sequential_create_destroy_no_leak) {
    for (int i = 0; i < 8; i++) {
        BatchEngine* engine = create_test_engine_seeded(1, 4, 42 + (uint64_t)i);
        ASSERT_NOT_NULL(engine);
        engine_destroy(engine);
    }
    return 0;
}

/** Multiple engines can coexist and be destroyed in any order */
TEST(multi_engine_destroy_any_order) {
    const int N = 4;
    BatchEngine* engines[4];

    for (int i = 0; i < N; i++) {
        engines[i] = create_test_engine_seeded(1, 4, 42 + (uint64_t)i);
        ASSERT_NOT_NULL(engines[i]);
    }

    /* Destroy in reverse order */
    for (int i = N - 1; i >= 0; i--) {
        engine_destroy(engines[i]);
    }
    return 0;
}

/* ============================================================================
 * Section 4: Validation Tests
 * ============================================================================ */

/** engine_is_valid returns true for valid engine */
TEST(engine_is_valid_true) {
    BatchEngine* engine = create_test_engine(4, 4);
    ASSERT_NOT_NULL(engine);
    ASSERT_TRUE(engine_is_valid(engine));
    engine_destroy(engine);
    return 0;
}

/** engine_is_valid returns false for NULL */
TEST(engine_is_valid_false_for_null) {
    ASSERT_FALSE(engine_is_valid(NULL));
    return 0;
}

/* ============================================================================
 * Section 5: obs_dim Consistency Tests (BUG detection)
 *
 * These tests document the fundamental mismatch between the sensor system's
 * obs_dim and the engine's obs_dim that causes the buffer overflow.
 * ============================================================================ */

/** Sensor system obs_dim should match engine obs_dim when external buffer is set.
 *
 * BUG DETECTED: The sensor system is created with max_obs_dim=256, but the
 * engine's observations buffer is sized for actual_obs_dim=15. After
 * sensor_system_set_external_buffer(), the sensor system writes using its
 * own obs_dim (256) as the stride, overflowing the 15-stride buffer.
 *
 * Expected: sensors->obs_dim == engine->obs_dim
 * Actual: sensors->obs_dim=256, engine->obs_dim=15 */
TEST(sensor_obs_dim_consistency) {
    BatchEngine* engine = create_test_engine(1, 4);
    ASSERT_NOT_NULL(engine);

    uint32_t engine_dim = engine->obs_dim;
    size_t sensor_dim = engine->sensors->obs_dim;

    printf("\n    [DIAG] engine->obs_dim = %u\n", engine_dim);
    printf("    [DIAG] sensors->obs_dim = %zu\n", sensor_dim);

    /* BUG: This will fail. sensor_dim is max_obs_dim (256), not actual (15). */
    ASSERT_MSG(sensor_dim == (size_t)engine_dim,
               "BUG: sensor obs_dim mismatch causes buffer overflow on reset/sample");

    engine_destroy(engine);
    return 0;
}

/** The observations buffer must be large enough for sensor_system operations.
 *
 * BUG DETECTED: obs buffer = total_agents * engine->obs_dim * sizeof(float),
 * but sensor_system uses total_agents * sensors->obs_dim * sizeof(float).
 * With default config: 4 * 15 * 4 = 240 bytes vs 4 * 256 * 4 = 4096 bytes. */
TEST(obs_buffer_size_vs_sensor_requirement) {
    BatchEngine* engine = create_test_engine(1, 4);
    ASSERT_NOT_NULL(engine);

    uint32_t td = engine->config.total_agents;
    size_t buf_alloc = ((size_t)td * engine->obs_dim * sizeof(float) + 31) & ~(size_t)31;
    size_t sensor_needs = (size_t)engine->sensors->max_agents *
                           engine->sensors->obs_dim * sizeof(float);

    printf("\n    [DIAG] obs buffer allocated: %zu bytes\n", buf_alloc);
    printf("    [DIAG] sensor system needs: %zu bytes\n", sensor_needs);

    /* BUG: This will fail. */
    ASSERT_MSG(buf_alloc >= sensor_needs,
               "BUG: obs buffer too small for sensor_system -- overflow on reset/sample");

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Section 6: Pointer Stability Across Resets
 *
 * All engine subsystem pointers should remain stable (unchanged) across
 * engine_reset() calls. The persistent arena is never reset.
 * ============================================================================ */

/** All subsystem pointers remain valid after engine_reset */
TEST(engine_pointers_stable_after_reset) {
    BatchEngine* engine = create_test_engine(1, 4);
    ASSERT_NOT_NULL(engine);

    /* Save all pointers before reset */
    PlatformStateSOA* states = engine->states;
    PlatformParamsSOA* params = engine->params;
    WorldBrickMap* world = engine->world;
    PhysicsSystem* physics = engine->physics;
    CollisionSystem* collision = engine->collision;
    SensorSystem* sensors = engine->sensors;
    RewardSystem* rewards = engine->rewards;
    float* observations = engine->observations;
    float* actions = engine->actions;
    float* rewards_buf = engine->rewards_buffer;
    uint8_t* dones = engine->dones;
    uint8_t* truncations = engine->truncations;
    Arena* persistent = engine->persistent_arena;
    Arena* frame = engine->frame_arena;

    engine_reset(engine);

    /* Verify all pointers unchanged */
    ASSERT_EQ(engine->states, states);
    ASSERT_EQ(engine->params, params);
    ASSERT_EQ(engine->world, world);
    ASSERT_EQ(engine->physics, physics);
    ASSERT_EQ(engine->collision, collision);
    ASSERT_EQ(engine->sensors, sensors);
    ASSERT_EQ(engine->rewards, rewards);
    ASSERT_EQ(engine->observations, observations);
    ASSERT_EQ(engine->actions, actions);
    ASSERT_EQ(engine->rewards_buffer, rewards_buf);
    ASSERT_EQ(engine->dones, dones);
    ASSERT_EQ(engine->truncations, truncations);
    ASSERT_EQ(engine->persistent_arena, persistent);
    ASSERT_EQ(engine->frame_arena, frame);

    /* Still valid */
    ASSERT_TRUE(engine->initialized);
    ASSERT_FALSE(engine->needs_reset);

    engine_destroy(engine);
    return 0;
}

/** Pointers stable after multiple resets */
TEST(engine_pointers_stable_after_many_resets) {
    BatchEngine* engine = create_test_engine(1, 4);
    ASSERT_NOT_NULL(engine);

    float* obs = engine->observations;
    uint8_t* dones = engine->dones;

    for (int i = 0; i < 10; i++) {
        engine_reset(engine);
        ASSERT_EQ(engine->observations, obs);
        ASSERT_EQ(engine->dones, dones);
        ASSERT_TRUE(engine->initialized);
    }

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Section 7: Pointer Stability Across Steps
 * ============================================================================ */

/** All subsystem pointers remain valid after engine_step */
TEST(engine_pointers_stable_after_step) {
    BatchEngine* engine = create_test_engine(1, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    float* observations = engine->observations;
    float* actions = engine->actions;
    float* rewards_buf = engine->rewards_buffer;
    uint8_t* dones = engine->dones;
    PlatformStateSOA* states = engine->states;
    SensorSystem* sensors = engine->sensors;

    engine_step(engine);

    ASSERT_EQ(engine->observations, observations);
    ASSERT_EQ(engine->actions, actions);
    ASSERT_EQ(engine->rewards_buffer, rewards_buf);
    ASSERT_EQ(engine->dones, dones);
    ASSERT_EQ(engine->states, states);
    ASSERT_EQ(engine->sensors, sensors);
    ASSERT_TRUE(engine->initialized);

    engine_destroy(engine);
    return 0;
}

/** Pointers stable across many steps */
TEST(engine_pointers_stable_after_many_steps) {
    BatchEngine* engine = create_test_engine(1, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    float* obs = engine->observations;
    for (int i = 0; i < 100; i++) {
        engine_step(engine);
        ASSERT_EQ(engine->observations, obs);
    }

    engine_destroy(engine);
    return 0;
}

/** engine_step increments total_steps */
TEST(engine_step_increments_total_steps) {
    BatchEngine* engine = create_test_engine(1, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);
    ASSERT_EQ(engine->total_steps, 0u);

    engine_step(engine);
    ASSERT_EQ(engine->total_steps, 1u);

    engine_step(engine);
    ASSERT_EQ(engine->total_steps, 2u);

    engine_destroy(engine);
    return 0;
}

/** engine_step clears needs_reset after reset */
TEST(engine_needs_reset_cleared) {
    BatchEngine* engine = create_test_engine(1, 4);
    ASSERT_NOT_NULL(engine);

    ASSERT_TRUE(engine->needs_reset);
    engine_reset(engine);
    ASSERT_FALSE(engine->needs_reset);

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Section 8: Frame Arena Behavior
 * ============================================================================ */

/** Frame arena is reset at start of each step (used grows, then resets) */
TEST(frame_arena_reset_each_step) {
    BatchEngine* engine = create_test_engine(1, 4);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* After reset, frame arena should be relatively empty */
    size_t used_before = engine->frame_arena->used;

    engine_step(engine);

    /* After step, frame arena was reset at beginning of step, then used for
     * scratch allocations. The exact used amount depends on implementation.
     * Key: it should not grow unboundedly. */
    size_t used_after = engine->frame_arena->used;

    /* Run another step -- frame arena used should not accumulate */
    engine_step(engine);
    size_t used_after2 = engine->frame_arena->used;

    /* Frame arena used after step should be roughly the same each time
     * (it gets reset at the start of each step, then refilled) */
    printf("\n    [DIAG] frame_arena.used: before=%zu, after_step1=%zu, after_step2=%zu\n",
           used_before, used_after, used_after2);

    /* Both steps should use similar amounts (not accumulating) */
    ASSERT_TRUE(used_after2 < engine->frame_arena->capacity);

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Section 9: Buffer Alignment Tests
 * ============================================================================ */

/** Observations buffer is 32-byte aligned */
TEST(observations_32byte_aligned) {
    BatchEngine* engine = create_test_engine(1, 4);
    ASSERT_NOT_NULL(engine);
    ASSERT_TRUE(((uintptr_t)engine->observations & 31) == 0);
    engine_destroy(engine);
    return 0;
}

/** Actions buffer is 32-byte aligned */
TEST(actions_32byte_aligned) {
    BatchEngine* engine = create_test_engine(1, 4);
    ASSERT_NOT_NULL(engine);
    ASSERT_TRUE(((uintptr_t)engine->actions & 31) == 0);
    engine_destroy(engine);
    return 0;
}

/** Rewards buffer is 32-byte aligned */
TEST(rewards_32byte_aligned) {
    BatchEngine* engine = create_test_engine(1, 4);
    ASSERT_NOT_NULL(engine);
    ASSERT_TRUE(((uintptr_t)engine->rewards_buffer & 31) == 0);
    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Section 10: Buffer Initialization Tests
 * ============================================================================ */

/** All buffers are zeroed after creation */
TEST(buffers_zeroed_after_create) {
    BatchEngine* engine = create_test_engine(1, 4);
    ASSERT_NOT_NULL(engine);

    uint32_t td = engine->config.total_agents;

    /* Observations should be zeroed */
    for (uint32_t i = 0; i < td * engine->obs_dim; i++) {
        ASSERT_FLOAT_EQ(engine->observations[i], 0.0f);
    }

    /* Actions should be zeroed */
    for (uint32_t i = 0; i < td * engine->action_dim; i++) {
        ASSERT_FLOAT_EQ(engine->actions[i], 0.0f);
    }

    /* Rewards should be zeroed */
    for (uint32_t i = 0; i < td; i++) {
        ASSERT_FLOAT_EQ(engine->rewards_buffer[i], 0.0f);
    }

    /* Dones should be zeroed */
    for (uint32_t i = 0; i < td; i++) {
        ASSERT_EQ(engine->dones[i], 0u);
    }

    /* Truncations should be zeroed */
    for (uint32_t i = 0; i < td; i++) {
        ASSERT_EQ(engine->truncations[i], 0u);
    }

    engine_destroy(engine);
    return 0;
}

/** Episode tracking is zeroed after creation */
TEST(episode_tracking_zeroed) {
    BatchEngine* engine = create_test_engine(1, 4);
    ASSERT_NOT_NULL(engine);

    uint32_t td = engine->config.total_agents;
    for (uint32_t i = 0; i < td; i++) {
        ASSERT_FLOAT_EQ(engine->episode_returns[i], 0.0f);
        ASSERT_EQ(engine->episode_lengths[i], 0u);
    }

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Section 11: Memory Size Helpers
 * ============================================================================ */

/** engine_memory_size returns non-zero for valid config */
TEST(memory_size_nonzero) {
    EngineConfig cfg = engine_config_default();
    size_t size = engine_memory_size(&cfg);
    ASSERT_TRUE(size > 0);
    return 0;
}

/** engine_memory_size returns 0 for NULL */
TEST(memory_size_null) {
    ASSERT_EQ(engine_memory_size(NULL), 0u);
    return 0;
}

/** engine_observation_buffer_size is 32-byte aligned */
TEST(obs_buffer_size_aligned) {
    size_t size = engine_observation_buffer_size(4, 15);
    ASSERT_TRUE((size & 31) == 0);
    return 0;
}

/** engine_action_buffer_size is 32-byte aligned */
TEST(act_buffer_size_aligned) {
    size_t size = engine_action_buffer_size(4, 4);
    ASSERT_TRUE((size & 31) == 0);
    return 0;
}

/* ============================================================================
 * Section 12: Various Config Sizes
 * ============================================================================ */

/** Engine creation works for various sizes of num_envs and agents_per_env */
TEST(various_config_sizes) {
    uint32_t configs[][2] = {
        {1, 1},
        {1, 4},
        {2, 2},
        {4, 4},
        {1, 16},
    };

    for (int i = 0; i < 5; i++) {
        BatchEngine* engine = create_test_engine(configs[i][0], configs[i][1]);
        ASSERT_NOT_NULL(engine);
        ASSERT_EQ(engine->config.total_agents, configs[i][0] * configs[i][1]);
        ASSERT_EQ(engine->obs_dim, 15u);  /* Default sensors always give 15 */
        engine_destroy(engine);
    }
    return 0;
}

/* ============================================================================
 * Section 13: Sensor System External Buffer Safety
 *
 * These tests verify that the sensor system's external buffer operations
 * don't overflow when the external buffer is smaller than the sensor
 * system's internal obs_dim would require.
 * ============================================================================ */

/** sensor_system_set_external_buffer should update obs_dim or validate size.
 *
 * BUG DETECTED: sensor_system_set_external_buffer() only sets the pointer
 * without checking or adjusting obs_dim. The caller (engine_lifecycle.c)
 * sets a buffer sized for engine->obs_dim, but the sensor system continues
 * to use its original max_obs_dim for all stride calculations.
 *
 * The fix should either:
 * (a) Have engine_lifecycle set sensor_system.obs_dim = engine->obs_dim
 *     after calling sensor_system_set_external_buffer, OR
 * (b) Allocate the observations buffer with max_obs_dim instead of actual, OR
 * (c) Have sensor_system use a separate stride parameter for external buffers.
 *
 * Option (a) is simplest: add sensor_system.obs_dim = engine->obs_dim after
 * the set_external_buffer call in engine_lifecycle.c ~line 576. */
TEST(external_buffer_stride_matches_obs_dim) {
    BatchEngine* engine = create_test_engine(1, 4);
    ASSERT_NOT_NULL(engine);

    /* The external buffer pointer should be the engine's observations */
    ASSERT_EQ(engine->sensors->observation_buffer, engine->observations);

    /* The sensor system obs_dim must match the buffer stride */
    size_t sensor_stride = engine->sensors->obs_dim;
    uint32_t engine_stride = engine->obs_dim;

    printf("\n    [DIAG] sensor stride=%zu, engine stride=%u\n",
           sensor_stride, engine_stride);

    /* BUG: sensor_stride (256) != engine_stride (15) */
    ASSERT_MSG(sensor_stride == (size_t)engine_stride,
               "BUG: sensor system uses wrong stride for external buffer -- "
               "will overflow on reset/sample. Fix: set sensors->obs_dim = engine->obs_dim "
               "after sensor_system_set_external_buffer()");

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Engine Lifecycle Tests");

    /* Creation */
    RUN_TEST(engine_create_succeeds);
    RUN_TEST(engine_create_invalid_returns_null);
    RUN_TEST(engine_create_zero_drones_returns_null);
    RUN_TEST(engine_create_null_config_returns_null);
    RUN_TEST(engine_create_populates_error_on_failure);
    RUN_TEST(engine_create_clears_error_on_success);

    /* Subsystem allocation */
    RUN_TEST(engine_create_allocates_subsystems);
    RUN_TEST(engine_computes_total_agents);
    RUN_TEST(engine_initializes_env_ids);
    RUN_TEST(engine_allocates_buffers);
    RUN_TEST(engine_allocates_episode_tracking);
    RUN_TEST(engine_obs_dim_correct_for_defaults);
    RUN_TEST(engine_action_dim_is_4);
    RUN_TEST(engine_state_flags_after_create);
    RUN_TEST(engine_arenas_allocated);

    /* Destruction */
    RUN_TEST(engine_destroy_null_safe);
    RUN_TEST(sequential_create_destroy_no_leak);
    RUN_TEST(multi_engine_destroy_any_order);

    /* Validation */
    RUN_TEST(engine_is_valid_true);
    RUN_TEST(engine_is_valid_false_for_null);

    /* obs_dim consistency (BUG detection) */
    RUN_TEST(sensor_obs_dim_consistency);
    RUN_TEST(obs_buffer_size_vs_sensor_requirement);

    /* Pointer stability */
    RUN_TEST(engine_pointers_stable_after_reset);
    RUN_TEST(engine_pointers_stable_after_many_resets);
    RUN_TEST(engine_pointers_stable_after_step);
    RUN_TEST(engine_pointers_stable_after_many_steps);
    RUN_TEST(engine_step_increments_total_steps);
    RUN_TEST(engine_needs_reset_cleared);

    /* Frame arena */
    RUN_TEST(frame_arena_reset_each_step);

    /* Buffer alignment */
    RUN_TEST(observations_32byte_aligned);
    RUN_TEST(actions_32byte_aligned);
    RUN_TEST(rewards_32byte_aligned);

    /* Buffer initialization */
    RUN_TEST(buffers_zeroed_after_create);
    RUN_TEST(episode_tracking_zeroed);

    /* Memory size helpers */
    RUN_TEST(memory_size_nonzero);
    RUN_TEST(memory_size_null);
    RUN_TEST(obs_buffer_size_aligned);
    RUN_TEST(act_buffer_size_aligned);

    /* Various sizes */
    RUN_TEST(various_config_sizes);

    /* External buffer safety (BUG detection) */
    RUN_TEST(external_buffer_stride_matches_obs_dim);

    TEST_SUITE_END();
}
