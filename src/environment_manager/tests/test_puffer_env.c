/**
 * Test: PufferEnv lifecycle -- exhaustive behavioral specification.
 *
 * Tests the PufferEnv wrapper's complete lifecycle: creation, field
 * verification, reset, step, close, and edge cases. These tests serve
 * as executable documentation of the PufferEnv contract.
 *
 * KEY BUG DOCUMENTED HERE:
 *   sensor_system's obs_dim (max_obs_dim=256) != engine's obs_dim (actual=15).
 *   When sensor_system_set_external_buffer() points the sensor system at
 *   engine->observations, the sensor system still uses its own obs_dim for
 *   strides and memset sizes. This causes a buffer overflow during
 *   sensor_system_reset() and sensor_system_sample_all() that corrupts
 *   everything after the observations buffer in the persistent arena,
 *   including the PufferEnv struct (which is allocated last).
 */

#include "environment_manager.h"
#include "test_harness.h"
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Helpers
 * ============================================================================ */

/** Create a small PufferEnv for testing */
static PufferEnv* create_test_env(uint32_t num_envs, uint32_t agents_per_env,
                                   uint64_t seed, char* error) {
    EngineConfig config = engine_config_default();
    config.num_envs = num_envs;
    config.agents_per_env = agents_per_env;
    config.total_agents = num_envs * agents_per_env;
    config.seed = seed;
    return puffer_env_create_from_config(&config, error);
}

/* ============================================================================
 * Section 1: Creation Tests
 * ============================================================================ */

/** puffer_env_create_from_config returns non-NULL for valid config */
TEST(create_valid_config_succeeds) {
    char error[ENGINE_ERROR_MSG_SIZE] = {0};
    PufferEnv* env = create_test_env(1, 4, 42, error);
    ASSERT_NOT_NULL(env);
    puffer_env_close(env);
    return 0;
}

/** puffer_env_create_from_config returns NULL for NULL config */
TEST(create_null_config_returns_null) {
    char error[ENGINE_ERROR_MSG_SIZE] = {0};
    PufferEnv* env = puffer_env_create_from_config(NULL, error);
    ASSERT_NULL(env);
    return 0;
}

/** puffer_env_create_from_config returns NULL for invalid config (num_envs=0) */
TEST(create_invalid_config_returns_null) {
    EngineConfig config = engine_config_default();
    config.num_envs = 0;  /* Invalid */
    char error[ENGINE_ERROR_MSG_SIZE] = {0};
    PufferEnv* env = puffer_env_create_from_config(&config, error);
    ASSERT_NULL(env);
    return 0;
}

/** puffer_env_create returns NULL for nonexistent config path */
TEST(create_bad_path_returns_null) {
    PufferEnv* env = puffer_env_create("/nonexistent/path/to/config.toml");
    ASSERT_NULL(env);
    return 0;
}

/** puffer_env_create with NULL path uses defaults */
TEST(create_null_path_uses_defaults) {
    PufferEnv* env = puffer_env_create(NULL);
    ASSERT_NOT_NULL(env);
    ASSERT_NOT_NULL(env->engine);
    /* Default: 64 envs, 16 agents_per_env */
    ASSERT_EQ(env->num_envs, 64);
    ASSERT_EQ(env->num_agents, 16);
    puffer_env_close(env);
    return 0;
}

/* ============================================================================
 * Section 2: Field Verification After Creation
 * ============================================================================ */

/** All PufferEnv fields are correctly set after creation */
TEST(fields_correct_after_create) {
    char error[ENGINE_ERROR_MSG_SIZE] = {0};
    PufferEnv* env = create_test_env(2, 8, 42, error);
    ASSERT_NOT_NULL(env);

    /* Engine pointer is set */
    ASSERT_NOT_NULL(env->engine);
    ASSERT_TRUE(env->engine->initialized);

    /* Dimensions match config */
    ASSERT_EQ(env->num_envs, 2);
    ASSERT_EQ(env->num_agents, 8);
    ASSERT_EQ(env->action_size, 4);  /* 4 for quadcopter */
    ASSERT_TRUE(env->obs_size > 0);

    /* Buffer aliases are set */
    ASSERT_NOT_NULL(env->observations);
    ASSERT_NOT_NULL(env->actions);
    ASSERT_NOT_NULL(env->rewards);
    ASSERT_NOT_NULL(env->terminals);
    ASSERT_NOT_NULL(env->truncations);

    /* Metadata */
    ASSERT_NOT_NULL(env->name);
    ASSERT_NOT_NULL(env->version);
    ASSERT_STR_EQ(env->name, "DroneSwarm");
    ASSERT_STR_EQ(env->version, "1.0.0");

    puffer_env_close(env);
    return 0;
}

/** Buffer aliases point to engine's internal buffers (zero-copy) */
TEST(buffer_aliasing_is_zero_copy) {
    char error[ENGINE_ERROR_MSG_SIZE] = {0};
    PufferEnv* env = create_test_env(1, 4, 42, error);
    ASSERT_NOT_NULL(env);

    /* Verify aliasing: PufferEnv buffers should point to same memory as engine */
    ASSERT_TRUE(env->observations == env->engine->observations);
    ASSERT_TRUE(env->actions == env->engine->actions);
    ASSERT_TRUE(env->rewards == env->engine->rewards_buffer);
    ASSERT_TRUE(env->terminals == env->engine->dones);
    ASSERT_TRUE(env->truncations == env->engine->truncations);

    puffer_env_close(env);
    return 0;
}

/** obs_size matches engine->obs_dim (the actual observation dimension) */
TEST(obs_size_matches_engine_obs_dim) {
    char error[ENGINE_ERROR_MSG_SIZE] = {0};
    PufferEnv* env = create_test_env(1, 4, 42, error);
    ASSERT_NOT_NULL(env);

    /* obs_size should equal the actual sensor output, not max_obs_dim */
    ASSERT_EQ((uint32_t)env->obs_size, env->engine->obs_dim);
    /* Default sensors: IMU(6) + Position(3) + Velocity(6) = 15 */
    ASSERT_EQ(env->obs_size, 15);

    puffer_env_close(env);
    return 0;
}

/* ============================================================================
 * Section 3: Observation Space and Action Space Queries
 * ============================================================================ */

/** get_observation_space returns correct shape */
TEST(observation_space_shape) {
    char error[ENGINE_ERROR_MSG_SIZE] = {0};
    PufferEnv* env = create_test_env(2, 4, 42, error);
    ASSERT_NOT_NULL(env);

    int shape[4] = {0};
    int ndim = 0;
    puffer_env_get_observation_space(env, shape, &ndim);

    ASSERT_EQ(ndim, 2);
    ASSERT_EQ(shape[0], 2 * 4);   /* num_envs * num_agents = total agents */
    ASSERT_EQ(shape[1], 15);       /* obs_size */

    puffer_env_close(env);
    return 0;
}

/** get_action_space returns correct shape */
TEST(action_space_shape) {
    char error[ENGINE_ERROR_MSG_SIZE] = {0};
    PufferEnv* env = create_test_env(2, 4, 42, error);
    ASSERT_NOT_NULL(env);

    int shape[4] = {0};
    int ndim = 0;
    puffer_env_get_action_space(env, shape, &ndim);

    ASSERT_EQ(ndim, 2);
    ASSERT_EQ(shape[0], 2 * 4);   /* total agents */
    ASSERT_EQ(shape[1], 4);        /* action_size (4 motors) */

    puffer_env_close(env);
    return 0;
}

/** Space queries are safe with NULL args */
TEST(space_queries_null_safe) {
    int shape[4] = {0};
    int ndim = 0;
    /* All of these should not crash */
    puffer_env_get_observation_space(NULL, shape, &ndim);
    puffer_env_get_action_space(NULL, shape, &ndim);

    char error[ENGINE_ERROR_MSG_SIZE] = {0};
    PufferEnv* env = create_test_env(1, 4, 42, error);
    ASSERT_NOT_NULL(env);
    puffer_env_get_observation_space(env, NULL, &ndim);
    puffer_env_get_observation_space(env, shape, NULL);
    puffer_env_close(env);
    return 0;
}

/* ============================================================================
 * Section 4: obs_dim Consistency Check
 *
 * BUG DETECTED: The sensor system's obs_dim (max_obs_dim=256) differs from
 * the engine's obs_dim (actual_obs_dim=15). The external buffer is sized
 * for the engine's obs_dim, but the sensor system uses its own obs_dim
 * for memset and scatter strides. This is the root cause of the segfault.
 * ============================================================================ */

/** sensor obs_dim must equal engine obs_dim when external buffer is set.
 *
 * BUG DETECTED: sensor system's obs_dim is 256 (max_obs_dim passed at creation),
 * but the external buffer (engine->observations) is sized for engine->obs_dim=15.
 * sensor_system_reset() does memset(obs_buffer, 0, max_agents * 256 * 4) = 4096 bytes,
 * but the buffer is only 256 bytes (4 drones * 15 floats * 4 bytes, rounded to 32).
 *
 * Expected: sensor_system.obs_dim == engine->obs_dim
 * Actual: sensor_system.obs_dim=256, engine->obs_dim=15 */
TEST(sensor_obs_dim_matches_engine_obs_dim) {
    char error[ENGINE_ERROR_MSG_SIZE] = {0};
    PufferEnv* env = create_test_env(1, 4, 42, error);
    ASSERT_NOT_NULL(env);

    uint32_t engine_dim = env->engine->obs_dim;
    size_t sensor_dim = env->engine->sensors->obs_dim;

    /* This assertion documents the bug. It WILL fail until fixed. */
    ASSERT_MSG(sensor_dim == engine_dim,
               "BUG: sensor obs_dim != engine obs_dim. "
               "sensor_system_reset and sample_all will overflow the external buffer.");

    puffer_env_close(env);
    return 0;
}

/** Observation buffer is large enough for sensor system's max operations.
 *
 * BUG DETECTED: The allocated observations buffer is too small for
 * sensor_system operations. The buffer must be at least
 * max_agents * sensor_obs_dim * sizeof(float) bytes.
 *
 * Expected: buffer size >= max_agents * sensor_obs_dim * sizeof(float)
 * Actual: buffer size = max_agents * engine_obs_dim * sizeof(float) (much smaller) */
TEST(obs_buffer_sufficient_for_sensor_system) {
    char error[ENGINE_ERROR_MSG_SIZE] = {0};
    PufferEnv* env = create_test_env(1, 4, 42, error);
    ASSERT_NOT_NULL(env);

    /* Calculate what sensor_system needs vs what engine allocated */
    uint32_t total_agents = env->engine->config.total_agents;
    size_t sensor_stride = env->engine->sensors->obs_dim;
    size_t needed = total_agents * sensor_stride * sizeof(float);

    size_t engine_obs_dim = env->engine->obs_dim;
    size_t allocated = ((total_agents * engine_obs_dim * sizeof(float)) + 31) & ~(size_t)31;

    printf("\n    [DIAG] Sensor system needs %zu bytes, engine allocated %zu bytes\n",
           needed, allocated);

    /* BUG: This WILL fail. The engine allocates far less than the sensor system needs. */
    ASSERT_MSG(allocated >= needed,
               "BUG: observations buffer too small for sensor_system operations");

    puffer_env_close(env);
    return 0;
}

/* ============================================================================
 * Section 5: Reset Tests
 * ============================================================================ */

/** puffer_env_reset with NULL is safe */
TEST(reset_null_safe) {
    puffer_env_reset(NULL);  /* Should not crash */
    return 0;
}

/** Engine pointer survives a single reset.
 *
 * BUG DETECTED: Due to the obs_dim mismatch, sensor_system_reset()
 * overflows the observations buffer and corrupts subsequent arena
 * allocations including the PufferEnv.
 *
 * Expected: env->engine unchanged after reset
 * Actual: env->engine becomes NULL (zeroed by overflow memset) */
TEST(engine_pointer_survives_single_reset) {
    char error[ENGINE_ERROR_MSG_SIZE] = {0};
    PufferEnv* env = create_test_env(1, 4, 42, error);
    ASSERT_NOT_NULL(env);
    ASSERT_NOT_NULL(env->engine);

    BatchEngine* saved = env->engine;

    puffer_env_reset(env);

    /* BUG: This fails because sensor_system_reset overflows the obs buffer */
    ASSERT_MSG(env->engine == saved,
               "BUG: engine pointer corrupted by puffer_env_reset");
    ASSERT_MSG(env->engine != NULL,
               "BUG: engine pointer is NULL after reset");

    puffer_env_close(env);
    return 0;
}

/** Engine pointer survives multiple resets */
TEST(engine_pointer_survives_multiple_resets) {
    char error[ENGINE_ERROR_MSG_SIZE] = {0};
    PufferEnv* env = create_test_env(1, 4, 42, error);
    ASSERT_NOT_NULL(env);

    BatchEngine* saved = env->engine;

    for (int i = 0; i < 10; i++) {
        puffer_env_reset(env);
        ASSERT_MSG(env->engine == saved,
                   "BUG: engine pointer changed across resets");
    }

    puffer_env_close(env);
    return 0;
}

/* ============================================================================
 * Section 6: Step Tests
 * ============================================================================ */

/** puffer_env_step with NULL is safe */
TEST(step_null_safe) {
    puffer_env_step(NULL);  /* Should not crash */
    return 0;
}

/** Engine pointer survives a single step */
TEST(engine_pointer_survives_single_step) {
    char error[ENGINE_ERROR_MSG_SIZE] = {0};
    PufferEnv* env = create_test_env(1, 4, 42, error);
    ASSERT_NOT_NULL(env);

    BatchEngine* saved = env->engine;
    puffer_env_reset(env);

    /* If reset already corrupted env->engine, we cannot proceed */
    if (env->engine == NULL) {
        printf("\n    [BUG] engine pointer NULL after reset, cannot test step\n");
        /* Cannot close safely if engine is NULL -- env memory is corrupt */
        return -__LINE__;
    }

    puffer_env_step(env);
    ASSERT_MSG(env->engine == saved,
               "BUG: engine pointer corrupted by puffer_env_step");

    puffer_env_close(env);
    return 0;
}

/** Engine pointer survives many steps */
TEST(engine_pointer_survives_many_steps) {
    char error[ENGINE_ERROR_MSG_SIZE] = {0};
    PufferEnv* env = create_test_env(1, 4, 42, error);
    ASSERT_NOT_NULL(env);

    BatchEngine* saved = env->engine;
    puffer_env_reset(env);

    if (env->engine == NULL) {
        printf("\n    [BUG] engine pointer NULL after reset\n");
        return -__LINE__;
    }

    for (int step = 0; step < 100; step++) {
        puffer_env_step(env);
        if (env->engine != saved) {
            printf("\n    [BUG] engine pointer changed at step %d\n", step);
            ASSERT_MSG(0, "BUG: engine pointer corrupted during stepping");
        }
    }

    puffer_env_close(env);
    return 0;
}

/** Observations buffer gets populated after reset */
TEST(observations_populated_after_reset) {
    char error[ENGINE_ERROR_MSG_SIZE] = {0};
    PufferEnv* env = create_test_env(1, 4, 42, error);
    ASSERT_NOT_NULL(env);

    puffer_env_reset(env);

    /* After reset, sensor_system_sample_all is called which populates obs.
     * Due to the obs_dim mismatch bug, this write may overflow.
     * But IF it doesn't crash, check that some observations are non-zero. */
    if (env->engine == NULL) {
        printf("\n    [BUG] engine pointer NULL, cannot verify observations\n");
        return -__LINE__;
    }

    /* At least some observation should be non-zero (position sensor) */
    int has_nonzero = 0;
    uint32_t total = env->engine->config.total_agents * env->engine->obs_dim;
    for (uint32_t i = 0; i < total; i++) {
        if (env->engine->observations[i] != 0.0f) {
            has_nonzero = 1;
            break;
        }
    }
    ASSERT_TRUE(has_nonzero);

    puffer_env_close(env);
    return 0;
}

/* ============================================================================
 * Section 7: Close Tests
 * ============================================================================ */

/** puffer_env_close with NULL is safe */
TEST(close_null_safe) {
    puffer_env_close(NULL);  /* Should not crash */
    return 0;
}

/** puffer_env_close releases all resources */
TEST(close_releases_resources) {
    char error[ENGINE_ERROR_MSG_SIZE] = {0};
    PufferEnv* env = create_test_env(1, 4, 42, error);
    ASSERT_NOT_NULL(env);

    /* After close, the env pointer is dangling (arena-freed).
     * We just verify it doesn't crash. */
    puffer_env_close(env);
    return 0;
}

/** Sequential create-close cycles don't leak */
TEST(sequential_create_close_no_leak) {
    for (int i = 0; i < 10; i++) {
        char error[ENGINE_ERROR_MSG_SIZE] = {0};
        PufferEnv* env = create_test_env(1, 4, 42 + (uint64_t)i, error);
        ASSERT_NOT_NULL(env);
        puffer_env_close(env);
    }
    return 0;
}

/* ============================================================================
 * Section 8: Render (placeholder)
 * ============================================================================ */

/** puffer_env_render with NULL is safe */
TEST(render_null_safe) {
    puffer_env_render(NULL, "human");  /* Should not crash */
    return 0;
}

/** puffer_env_render with valid env does not crash */
TEST(render_does_not_crash) {
    char error[ENGINE_ERROR_MSG_SIZE] = {0};
    PufferEnv* env = create_test_env(1, 4, 42, error);
    ASSERT_NOT_NULL(env);
    puffer_env_render(env, "human");
    puffer_env_render(env, "rgb_array");
    puffer_env_render(env, NULL);
    puffer_env_close(env);
    return 0;
}

/* ============================================================================
 * Section 9: Multi-Environment Isolation
 * ============================================================================ */

/** Multiple PufferEnvs have independent engines */
TEST(multi_env_independent_engines) {
    const int N = 4;
    PufferEnv* envs[4] = {0};

    for (int i = 0; i < N; i++) {
        char error[ENGINE_ERROR_MSG_SIZE] = {0};
        envs[i] = create_test_env(1, 4, 42 + (uint64_t)i, error);
        ASSERT_NOT_NULL(envs[i]);
        ASSERT_NOT_NULL(envs[i]->engine);
    }

    /* Each engine should be distinct */
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            ASSERT_TRUE(envs[i]->engine != envs[j]->engine);
            ASSERT_TRUE(envs[i]->observations != envs[j]->observations);
            ASSERT_TRUE(envs[i]->actions != envs[j]->actions);
        }
    }

    for (int i = 0; i < N; i++) {
        puffer_env_close(envs[i]);
    }
    return 0;
}

/** PufferEnv's engine pointer is at arena tail -- a reset on one env
 * should not affect another env's PufferEnv.
 *
 * Each PufferEnv lives in its own engine's persistent arena, so reset
 * on env[0] should not affect env[1]. But if the obs_dim mismatch bug
 * causes an overflow WITHIN env[0]'s own arena, only env[0]'s PufferEnv
 * is corrupted (env[1] has a separate arena). */
TEST(reset_one_does_not_corrupt_another) {
    PufferEnv* envs[2] = {0};
    BatchEngine* saved[2] = {0};

    for (int i = 0; i < 2; i++) {
        char error[ENGINE_ERROR_MSG_SIZE] = {0};
        envs[i] = create_test_env(1, 4, 42 + (uint64_t)i, error);
        ASSERT_NOT_NULL(envs[i]);
        saved[i] = envs[i]->engine;
    }

    /* Reset only env[0] */
    puffer_env_reset(envs[0]);

    /* env[1] should be completely unaffected (separate arena) */
    ASSERT_MSG(envs[1]->engine == saved[1],
               "BUG: reset of env[0] corrupted env[1]'s engine pointer");

    /* env[0] may be corrupted by the obs_dim mismatch bug */
    if (envs[0]->engine != saved[0]) {
        printf("\n    [BUG] env[0]->engine corrupted by its own reset "
               "(obs_dim mismatch overflow)\n");
    }

    for (int i = 0; i < 2; i++) {
        /* Use engine_destroy directly if PufferEnv is corrupted */
        if (envs[i]->engine != NULL) {
            puffer_env_close(envs[i]);
        } else {
            engine_destroy(saved[i]);
        }
    }
    return 0;
}

/* ============================================================================
 * Section 10: Arena Memory Layout Verification
 * ============================================================================ */

/** PufferEnv is allocated after all engine buffers in persistent arena */
TEST(puffer_env_after_engine_buffers) {
    char error[ENGINE_ERROR_MSG_SIZE] = {0};
    PufferEnv* env = create_test_env(1, 4, 42, error);
    ASSERT_NOT_NULL(env);

    /* PufferEnv should be at a higher address than engine buffers
     * (it's allocated last from the persistent arena bump allocator) */
    uintptr_t env_addr = (uintptr_t)env;
    uintptr_t obs_addr = (uintptr_t)env->engine->observations;
    uintptr_t act_addr = (uintptr_t)env->engine->actions;
    uintptr_t rew_addr = (uintptr_t)env->engine->rewards_buffer;
    uintptr_t done_addr = (uintptr_t)env->engine->dones;

    printf("\n    [DIAG] Arena layout:\n");
    printf("    [DIAG]   observations = %p\n", (void*)obs_addr);
    printf("    [DIAG]   actions      = %p\n", (void*)act_addr);
    printf("    [DIAG]   rewards      = %p\n", (void*)rew_addr);
    printf("    [DIAG]   dones        = %p\n", (void*)done_addr);
    printf("    [DIAG]   PufferEnv    = %p\n", (void*)env_addr);
    printf("    [DIAG]   gap obs->PufferEnv = %td bytes\n",
           (ptrdiff_t)(env_addr - obs_addr));

    /* PufferEnv should be AFTER all these buffers */
    ASSERT_TRUE(env_addr > obs_addr);
    ASSERT_TRUE(env_addr > act_addr);
    ASSERT_TRUE(env_addr > rew_addr);
    ASSERT_TRUE(env_addr > done_addr);

    puffer_env_close(env);
    return 0;
}

/** Verify that sensor_system_reset's memset would overflow into PufferEnv.
 *
 * This test computes the exact overflow distance and verifies that
 * the PufferEnv memory falls within the overflow region. */
TEST(overflow_reaches_puffer_env) {
    char error[ENGINE_ERROR_MSG_SIZE] = {0};
    PufferEnv* env = create_test_env(1, 4, 42, error);
    ASSERT_NOT_NULL(env);

    uintptr_t obs_start = (uintptr_t)env->engine->observations;
    uintptr_t env_start = (uintptr_t)env;

    size_t sensor_memset_size = (size_t)env->engine->sensors->max_agents *
                                 env->engine->sensors->obs_dim * sizeof(float);

    uintptr_t memset_end = obs_start + sensor_memset_size;

    printf("\n    [DIAG] obs buffer start  = %p\n", (void*)obs_start);
    printf("    [DIAG] PufferEnv start   = %p\n", (void*)env_start);
    printf("    [DIAG] sensor memset end = %p\n", (void*)memset_end);
    printf("    [DIAG] memset extends %td bytes past obs buffer start\n",
           (ptrdiff_t)sensor_memset_size);

    if (memset_end > env_start) {
        printf("    [BUG] sensor_system_reset memset reaches %td bytes INTO PufferEnv!\n",
               (ptrdiff_t)(memset_end - env_start));
        /* This documents the overflow. The memset zeroes the PufferEnv memory. */
    }

    /* If the sensor obs_dim equals engine obs_dim, memset should NOT reach PufferEnv */
    size_t correct_memset = (size_t)env->engine->config.total_agents *
                             env->engine->obs_dim * sizeof(float);
    uintptr_t correct_end = obs_start + correct_memset;

    ASSERT_MSG(correct_end <= env_start,
               "Even with correct obs_dim, memset reaches PufferEnv -- "
               "layout issue");

    /* But with wrong obs_dim, it DOES reach PufferEnv */
    if (env->engine->sensors->obs_dim != env->engine->obs_dim) {
        ASSERT_MSG(memset_end > env_start,
                   "Expected overflow to reach PufferEnv with mismatched obs_dim");
    }

    puffer_env_close(env);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("PufferEnv Lifecycle Tests");

    /* Creation */
    RUN_TEST(create_valid_config_succeeds);
    RUN_TEST(create_null_config_returns_null);
    RUN_TEST(create_invalid_config_returns_null);
    RUN_TEST(create_bad_path_returns_null);
    RUN_TEST(create_null_path_uses_defaults);

    /* Field verification */
    RUN_TEST(fields_correct_after_create);
    RUN_TEST(buffer_aliasing_is_zero_copy);
    RUN_TEST(obs_size_matches_engine_obs_dim);

    /* Space queries */
    RUN_TEST(observation_space_shape);
    RUN_TEST(action_space_shape);
    RUN_TEST(space_queries_null_safe);

    /* obs_dim consistency (BUG detection) */
    RUN_TEST(sensor_obs_dim_matches_engine_obs_dim);
    RUN_TEST(obs_buffer_sufficient_for_sensor_system);

    /* Reset */
    RUN_TEST(reset_null_safe);
    RUN_TEST(engine_pointer_survives_single_reset);
    RUN_TEST(engine_pointer_survives_multiple_resets);

    /* Step */
    RUN_TEST(step_null_safe);
    RUN_TEST(engine_pointer_survives_single_step);
    RUN_TEST(engine_pointer_survives_many_steps);
    RUN_TEST(observations_populated_after_reset);

    /* Close */
    RUN_TEST(close_null_safe);
    RUN_TEST(close_releases_resources);
    RUN_TEST(sequential_create_close_no_leak);

    /* Render */
    RUN_TEST(render_null_safe);
    RUN_TEST(render_does_not_crash);

    /* Multi-environment isolation */
    RUN_TEST(multi_env_independent_engines);
    RUN_TEST(reset_one_does_not_corrupt_another);

    /* Arena layout verification */
    RUN_TEST(puffer_env_after_engine_buffers);
    RUN_TEST(overflow_reaches_puffer_env);

    TEST_SUITE_END();
}
