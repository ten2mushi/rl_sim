/**
 * Test: multiple engine instances with repeated steps.
 * Mimics what the Python binding does: create N engines (one per env),
 * step them, then close. Catches heap buffer overflow and use-after-free.
 *
 * BUG DETECTED: sensor_system_reset() and sensor_system_sample_all() use
 * sys->obs_dim (= max_obs_dim, typically 256) to size memset and compute
 * scatter offsets, but the external observations buffer (engine->observations)
 * is sized for engine->obs_dim (= actual obs_dim, typically 15).
 * This causes a massive buffer overflow (e.g., 3840 bytes past end) that
 * clobbers everything allocated after observations in the persistent arena,
 * including the PufferEnv struct's engine pointer.
 *
 * Root cause: engine_lifecycle.c allocates obs buffer with actual_obs_dim,
 * but sensor_system uses max_obs_dim for strides and memset sizes.
 * The mismatch is introduced when sensor_system_set_external_buffer() is
 * called -- the external buffer is too small for the sensor system's obs_dim.
 */
#include "environment_manager.h"
#include "test_harness.h"
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Helper: Create a small config for testing
 * ============================================================================ */

static EngineConfig small_config(int agents_per_env, uint64_t seed) {
    EngineConfig config = engine_config_default();
    config.num_envs = 1;
    config.agents_per_env = (uint32_t)agents_per_env;
    config.total_agents = (uint32_t)agents_per_env;
    config.seed = seed;
    return config;
}

/* ============================================================================
 * Test Cases
 * ============================================================================ */

TEST(single_engine_create) {
    EngineConfig config = small_config(4, 42);

    char error[ENGINE_ERROR_MSG_SIZE];
    error[0] = '\0';
    BatchEngine* engine = engine_create(&config, error);
    ASSERT_NOT_NULL(engine);
    ASSERT_TRUE(engine->obs_dim == 15);
    engine_destroy(engine);
    return 0;
}

TEST(sequential_create_destroy) {
    for (int i = 0; i < 4; i++) {
        EngineConfig config = small_config(4, 42 + i);

        char error[ENGINE_ERROR_MSG_SIZE];
        error[0] = '\0';
        BatchEngine* engine = engine_create(&config, error);
        ASSERT_NOT_NULL(engine);
        engine_destroy(engine);
    }
    return 0;
}

TEST(multi_engine_create_close) {
    const int NUM_ENGINES = 4;
    BatchEngine* engines[4] = {0};

    for (int i = 0; i < NUM_ENGINES; i++) {
        EngineConfig config = small_config(4, 42 + i);

        char error[ENGINE_ERROR_MSG_SIZE];
        error[0] = '\0';
        engines[i] = engine_create(&config, error);
        ASSERT_NOT_NULL(engines[i]);
    }

    for (int i = 0; i < NUM_ENGINES; i++) {
        engine_destroy(engines[i]);
    }
    return 0;
}

TEST(multi_engine_step_close) {
    const int NUM_ENGINES = 4;
    BatchEngine* engines[4] = {0};

    for (int i = 0; i < NUM_ENGINES; i++) {
        EngineConfig config = small_config(4, 42 + i);

        char error[ENGINE_ERROR_MSG_SIZE];
        error[0] = '\0';
        engines[i] = engine_create(&config, error);
        ASSERT_NOT_NULL(engines[i]);
    }

    for (int i = 0; i < NUM_ENGINES; i++) {
        engine_reset(engines[i]);
    }

    for (int step = 0; step < 500; step++) {
        for (int i = 0; i < NUM_ENGINES; i++) {
            engine_step(engines[i]);
        }
    }

    for (int i = 0; i < NUM_ENGINES; i++) {
        engine_destroy(engines[i]);
    }
    return 0;
}

TEST(puffer_env_create_close) {
    const int NUM_ENVS = 4;
    PufferEnv* envs[4] = {0};

    for (int i = 0; i < NUM_ENVS; i++) {
        EngineConfig config = small_config(4, 42 + i);

        char error[ENGINE_ERROR_MSG_SIZE];
        error[0] = '\0';
        envs[i] = puffer_env_create_from_config(&config, error);
        ASSERT_NOT_NULL(envs[i]);
    }

    for (int i = 0; i < NUM_ENVS; i++) {
        puffer_env_reset(envs[i]);
    }
    for (int step = 0; step < 500; step++) {
        for (int i = 0; i < NUM_ENVS; i++) {
            puffer_env_step(envs[i]);
        }
    }
    for (int i = 0; i < NUM_ENVS; i++) {
        puffer_env_close(envs[i]);
    }
    return 0;
}

/* ============================================================================
 * Diagnostic: obs_dim mismatch detection
 *
 * BUG DETECTED: engine->obs_dim (actual, 15) != sensors->obs_dim (max, 256).
 * sensor_system_reset() memsets max_agents * sensors->obs_dim * sizeof(float)
 * bytes through the external buffer, but the buffer is only sized for
 * max_agents * engine->obs_dim * sizeof(float). This overflows by
 * max_agents * (sensors->obs_dim - engine->obs_dim) * sizeof(float) bytes.
 *
 * Expected: engine->obs_dim == sensors->obs_dim (or at least the memset
 * and scatter should use engine->obs_dim, not sensors->obs_dim)
 * Actual: engine->obs_dim=15, sensors->obs_dim=256 -- 16x mismatch
 * ============================================================================ */
TEST(obs_dim_mismatch_detection) {
    EngineConfig config = small_config(4, 42);

    char error[ENGINE_ERROR_MSG_SIZE];
    BatchEngine* engine = engine_create(&config, error);
    ASSERT_NOT_NULL(engine);

    uint32_t engine_obs_dim = engine->obs_dim;
    size_t sensor_obs_dim = engine->sensors->obs_dim;

    printf("\n    [DIAG] engine->obs_dim = %u\n", engine_obs_dim);
    printf("    [DIAG] sensors->obs_dim = %zu\n", sensor_obs_dim);
    printf("    [DIAG] total_agents = %u\n", engine->config.total_agents);

    size_t engine_buffer_bytes = ((size_t)engine->config.total_agents *
                                   engine_obs_dim * sizeof(float) + 31) & ~(size_t)31;
    size_t sensor_memset_bytes = (size_t)engine->sensors->max_agents *
                                  sensor_obs_dim * sizeof(float);

    printf("    [DIAG] engine obs buffer = %zu bytes\n", engine_buffer_bytes);
    printf("    [DIAG] sensor memset = %zu bytes\n", sensor_memset_bytes);

    if (sensor_memset_bytes > engine_buffer_bytes) {
        printf("    [BUG] OVERFLOW: sensor_system_reset() will write %zu bytes "
               "past end of obs buffer!\n",
               sensor_memset_bytes - engine_buffer_bytes);
    }

    /* BUG DETECTED: This assertion documents the mismatch.
     * When external buffer is set, sensor system's obs_dim MUST match
     * the actual buffer stride, otherwise memset and scatter overflow.
     *
     * Expected: sensor_obs_dim == engine_obs_dim
     * Actual:   sensor_obs_dim (256) >> engine_obs_dim (15) */
    ASSERT_MSG(sensor_obs_dim == engine_obs_dim,
               "BUG: sensor obs_dim != engine obs_dim -- "
               "sensor_system_reset memset will overflow external buffer");

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Diagnostic: engine pointer stability after reset
 *
 * This test isolates whether engine_reset corrupts the PufferEnv.
 * We save the engine pointer before reset, call reset, then verify.
 * If the obs_dim mismatch bug is present, this WILL fail because
 * sensor_system_reset overwrites the PufferEnv memory.
 * ============================================================================ */
TEST(engine_pointer_stable_after_reset) {
    EngineConfig config = small_config(4, 42);

    char error[ENGINE_ERROR_MSG_SIZE];
    PufferEnv* env = puffer_env_create_from_config(&config, error);
    ASSERT_NOT_NULL(env);
    ASSERT_NOT_NULL(env->engine);

    /* Save engine pointer and sentinel values */
    BatchEngine* engine_before = env->engine;
    const char* name_before = env->name;

    printf("\n    [DIAG] env=%p, env->engine=%p before reset\n",
           (void*)env, (void*)env->engine);

    /* Reset triggers sensor_system_reset which memsets obs buffer.
     * If buffer is too small (obs_dim mismatch bug), this clobbers PufferEnv. */
    puffer_env_reset(env);

    printf("    [DIAG] env->engine=%p after reset\n", (void*)env->engine);

    /* BUG DETECTED: env->engine becomes NULL after reset due to buffer overflow
     * in sensor_system_reset(). The memset writes max_agents * max_obs_dim * 4
     * bytes through the external buffer, overflowing into subsequent arena
     * allocations including the PufferEnv struct.
     *
     * Expected: env->engine == engine_before (pointer unchanged)
     * Actual: env->engine == NULL (zeroed by overflow) */
    ASSERT_MSG(env->engine == engine_before,
               "BUG: engine pointer corrupted after reset -- "
               "sensor_system_reset buffer overflow");
    ASSERT_MSG(env->name == name_before,
               "BUG: PufferEnv metadata corrupted after reset");

    puffer_env_close(env);
    return 0;
}

/* ============================================================================
 * Diagnostic: pinpoint which step causes corruption in shared buffer pattern
 * ============================================================================ */
TEST(shared_buffer_pattern_diagnostic) {
    const int NUM_ENVS = 4;
    const int DRONES_PER_ENV = 4;
    const int TOTAL_DRONES = NUM_ENVS * DRONES_PER_ENV;
    const int ACT_DIM = 4;
    const int STEPS = 10;

    PufferEnv* envs[4] = {0};
    for (int i = 0; i < NUM_ENVS; i++) {
        EngineConfig config = small_config(DRONES_PER_ENV, 42 + i);

        char error[ENGINE_ERROR_MSG_SIZE];
        error[0] = '\0';
        envs[i] = puffer_env_create_from_config(&config, error);
        ASSERT_NOT_NULL(envs[i]);
        ASSERT_NOT_NULL(envs[i]->engine);
    }

    /* Save engine pointers before any reset/step */
    BatchEngine* saved_engines[4];
    for (int i = 0; i < NUM_ENVS; i++) {
        saved_engines[i] = envs[i]->engine;
        printf("\n    [DIAG] envs[%d]->engine = %p", i, (void*)saved_engines[i]);
    }
    printf("\n");

    /* Check engine pointers after each reset */
    for (int i = 0; i < NUM_ENVS; i++) {
        puffer_env_reset(envs[i]);
        /* Check ALL env engine pointers after each reset */
        for (int j = 0; j < NUM_ENVS; j++) {
            if (envs[j]->engine != saved_engines[j]) {
                printf("    [BUG] After reset(envs[%d]): envs[%d]->engine "
                       "changed from %p to %p\n",
                       i, j, (void*)saved_engines[j], (void*)envs[j]->engine);
                ASSERT_MSG(0, "BUG: engine pointer corrupted by puffer_env_reset");
            }
        }
    }

    /* Get obs_dim for memcpy sizing */
    uint32_t obs_dim = envs[0]->engine->obs_dim;

    /* Allocate shared buffers */
    float* all_act = (float*)calloc((size_t)TOTAL_DRONES * ACT_DIM, sizeof(float));
    ASSERT_NOT_NULL(all_act);

    for (int step = 0; step < STEPS; step++) {
        for (int i = 0; i < NUM_ENVS; i++) {
            /* Verify engine pointer before each access */
            if (envs[i]->engine == NULL) {
                printf("    [BUG] step=%d, i=%d: envs[%d]->engine is NULL!\n",
                       step, i, i);
                free(all_act);
                ASSERT_MSG(0, "BUG: engine pointer is NULL during step loop");
            }

            BatchEngine* engine = envs[i]->engine;
            int start = i * DRONES_PER_ENV;

            memcpy(engine->actions, &all_act[start * ACT_DIM],
                   (size_t)DRONES_PER_ENV * ACT_DIM * sizeof(float));

            puffer_env_step(envs[i]);

            /* Check all engine pointers after each step */
            for (int j = 0; j < NUM_ENVS; j++) {
                if (envs[j]->engine != saved_engines[j]) {
                    printf("    [BUG] After step(%d) env[%d]: envs[%d]->engine "
                           "changed from %p to %p\n",
                           step, i, j,
                           (void*)saved_engines[j], (void*)envs[j]->engine);
                    free(all_act);
                    ASSERT_MSG(0, "BUG: engine pointer corrupted by puffer_env_step");
                }
            }
        }
    }

    for (int i = 0; i < NUM_ENVS; i++) {
        puffer_env_close(envs[i]);
    }
    free(all_act);
    return 0;
}

/* ============================================================================
 * Original shared_buffer_pattern (kept for regression once bug is fixed)
 *
 * NOTE: This test WILL segfault due to the obs_dim mismatch bug documented
 * above. The first puffer_env_reset() call triggers sensor_system_reset()
 * which overflows the observations buffer and zeros the PufferEnv.engine
 * pointer. Do not run this test until the bug is fixed.
 * ============================================================================ */
TEST(shared_buffer_pattern) {
    /* Mimics the binding: shared obs/act/rew/done buffers */
    const int NUM_ENVS = 4;
    const int DRONES_PER_ENV = 4;
    const int TOTAL_DRONES = NUM_ENVS * DRONES_PER_ENV;
    const int ACT_DIM = 4;
    const int STEPS = 100;

    PufferEnv* envs[4] = {0};
    for (int i = 0; i < NUM_ENVS; i++) {
        EngineConfig config = small_config(DRONES_PER_ENV, 42 + i);

        char error[ENGINE_ERROR_MSG_SIZE];
        error[0] = '\0';
        envs[i] = puffer_env_create_from_config(&config, error);
        ASSERT_NOT_NULL(envs[i]);
    }

    /* Get actual obs_dim from engine */
    uint32_t obs_dim = envs[0]->engine->obs_dim;
    ASSERT_TRUE(obs_dim > 0);

    /* Allocate shared buffers (like numpy arrays) */
    float* all_obs = (float*)calloc((size_t)TOTAL_DRONES * obs_dim, sizeof(float));
    float* all_act = (float*)calloc((size_t)TOTAL_DRONES * ACT_DIM, sizeof(float));
    float* all_rew = (float*)calloc(TOTAL_DRONES, sizeof(float));
    uint8_t* all_done = (uint8_t*)calloc(TOTAL_DRONES, sizeof(uint8_t));
    ASSERT_NOT_NULL(all_obs);

    for (int i = 0; i < NUM_ENVS; i++) {
        puffer_env_reset(envs[i]);
    }

    for (int step = 0; step < STEPS; step++) {
        for (int i = 0; i < NUM_ENVS; i++) {
            int start = i * DRONES_PER_ENV;
            BatchEngine* engine = envs[i]->engine;

            memcpy(engine->actions, &all_act[start * ACT_DIM],
                   (size_t)DRONES_PER_ENV * ACT_DIM * sizeof(float));

            puffer_env_step(envs[i]);

            memcpy(&all_obs[start * obs_dim], engine->observations,
                   (size_t)DRONES_PER_ENV * obs_dim * sizeof(float));
            memcpy(&all_rew[start], engine->rewards_buffer,
                   (size_t)DRONES_PER_ENV * sizeof(float));
            memcpy(&all_done[start], engine->dones,
                   DRONES_PER_ENV * sizeof(uint8_t));
        }
    }

    for (int i = 0; i < NUM_ENVS; i++) {
        puffer_env_close(envs[i]);
    }

    free(all_obs);
    free(all_act);
    free(all_rew);
    free(all_done);
    return 0;
}

/* ============================================================================
 * Main -- run diagnostic tests first, then regression tests
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Multi-Engine Tests");

    /* Diagnostic tests that expose the bug */
    RUN_TEST(obs_dim_mismatch_detection);
    RUN_TEST(engine_pointer_stable_after_reset);
    RUN_TEST(shared_buffer_pattern_diagnostic);

    /* These tests work at the raw engine level (no PufferEnv arena issue) */
    RUN_TEST(single_engine_create);
    RUN_TEST(sequential_create_destroy);
    RUN_TEST(multi_engine_create_close);
    RUN_TEST(multi_engine_step_close);

    /* These tests use PufferEnv and will fail due to the obs_dim mismatch bug */
    RUN_TEST(puffer_env_create_close);
    RUN_TEST(shared_buffer_pattern);

    TEST_SUITE_END();
}
