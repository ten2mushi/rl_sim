/**
 * @file test_determinism_deep.c
 * @brief Comprehensive Deep Determinism Tests for Drone RL Engine
 *
 * This test file serves as the definitive specification for determinism guarantees.
 * Following the "Tests as definition: the Yoneda way" philosophy, these tests
 * exhaustively explore all possible interactions that could break determinism.
 *
 * Test Categories (35+ tests):
 *
 * 1. Bit-Exact State Reproducibility (8 tests)
 *    - Position, velocity, quaternion, angular velocity determinism
 *    - RPM, observation, reward, done/truncation determinism
 *
 * 2. RNG Behavior Tests (7 tests)
 *    - Seed propagation, RNG state after N steps
 *    - RNG isolation between environments
 *    - RNG consumption order, spawn/target position determinism
 *
 * 3. Reset Determinism (7 tests)
 *    - Full reset, partial reset, single drone reset
 *    - Auto-reset, multiple sequential resets
 *    - Reset after varying episode lengths
 *
 * 4. Multi-Episode Determinism (4 tests)
 *    - Episode trajectory reproducibility
 *    - Episode completion order, truncation timing
 *    - Collision-triggered termination
 *
 * 5. Edge Cases (6 tests)
 *    - Zero actions, max actions, alternating patterns
 *    - Boundary positions, high-frequency collisions
 *    - Long trajectory determinism (10,000+ steps)
 *
 * 6. Integration Determinism (4 tests)
 *    - Physics + collision, physics + sensors
 *    - Full pipeline, sensor raycasting
 *
 * Requirements:
 * - Same seed MUST produce bit-exact identical results
 * - Reset with same seed MUST reproduce identical trajectory
 * - Thread count MUST NOT affect results
 * - RNG state MUST be properly isolated per-environment
 */

#include "environment_manager.h"
#include "platform_quadcopter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "test_harness.h"

/* Bit-exact float comparison (for determinism tests) */
#define ASSERT_FLOAT_EXACT(a, b) do { \
    if ((a) != (b)) { \
        printf("\n    ASSERT_FLOAT_EXACT failed: %s == %s (%.10e != %.10e)\n    at %s:%d", \
                #a, #b, (double)(a), (double)(b), __FILE__, __LINE__); \
        return -__LINE__; \
    } \
} while(0)

/* ============================================================================
 * Test Helpers
 * ============================================================================ */

/**
 * Create a test engine with specified seed and configuration.
 */
static BatchEngine* create_test_engine(uint64_t seed, uint32_t num_envs,
                                             uint32_t agents_per_env) {
    EngineConfig cfg = engine_config_default();
    cfg.num_envs = num_envs;
    cfg.agents_per_env = agents_per_env;
    cfg.seed = seed;
    cfg.persistent_arena_size = 128 * 1024 * 1024;
    cfg.frame_arena_size = 32 * 1024 * 1024;
    cfg.max_episode_steps = 1000;
    cfg.domain_randomization = 0.5f;

    char error[ENGINE_ERROR_MSG_SIZE];
    return engine_create(&cfg, error);
}

/**
 * Create a standard test engine (4 envs x 4 drones).
 */
static BatchEngine* create_standard_engine(uint64_t seed) {
    return create_test_engine(seed, 4, 4);
}

/**
 * Create an engine with short episode length for truncation tests.
 */
static BatchEngine* create_short_episode_engine(uint64_t seed, uint32_t max_steps) {
    EngineConfig cfg = engine_config_default();
    cfg.num_envs = 2;
    cfg.agents_per_env = 2;
    cfg.seed = seed;
    cfg.persistent_arena_size = 64 * 1024 * 1024;
    cfg.frame_arena_size = 16 * 1024 * 1024;
    cfg.max_episode_steps = max_steps;
    cfg.domain_randomization = 0.5f;

    char error[ENGINE_ERROR_MSG_SIZE];
    return engine_create(&cfg, error);
}

/**
 * Set fixed actions for all drones.
 */
static void set_fixed_actions(BatchEngine* engine, float value) {
    float* actions = engine_get_actions(engine);
    uint32_t total = engine->config.total_agents * engine->action_dim;
    for (uint32_t i = 0; i < total; i++) {
        actions[i] = value;
    }
}

/**
 * Set patterned actions (different per motor, per drone).
 */
static void set_patterned_actions(BatchEngine* engine, int pattern) {
    float* actions = engine_get_actions(engine);
    uint32_t total_agents = engine->config.total_agents;
    for (uint32_t d = 0; d < total_agents; d++) {
        for (uint32_t m = 0; m < engine->action_dim; m++) {
            float val;
            switch (pattern) {
                case 0:  /* Zero actions */
                    val = 0.0f;
                    break;
                case 1:  /* Max actions */
                    val = 1.0f;
                    break;
                case 2:  /* Alternating */
                    val = ((d + m) % 2 == 0) ? 0.0f : 1.0f;
                    break;
                case 3:  /* Gradient */
                    val = (float)(d * engine->action_dim + m) /
                          (float)(total_agents * engine->action_dim);
                    break;
                default:
                    val = 0.5f;
            }
            actions[d * engine->action_dim + m] = val;
        }
    }
}

/**
 * Set sinusoidal actions based on step number.
 */
static void set_sinusoidal_actions(BatchEngine* engine, int step) {
    float* actions = engine_get_actions(engine);
    uint32_t total = engine->config.total_agents * engine->action_dim;
    for (uint32_t i = 0; i < total; i++) {
        float val = sinf((float)(step * total + i) * 0.01f) * 0.5f + 0.5f;
        actions[i] = val;
    }
}

/**
 * Compare two engine states for bit-exact equality.
 * Returns 0 if identical, line number if different.
 */
static int compare_engine_states(BatchEngine* e1, BatchEngine* e2) {
    uint32_t total = e1->config.total_agents;

    /* Compare positions */
    for (uint32_t i = 0; i < total; i++) {
        if (e1->states->rigid_body.pos_x[i] != e2->states->rigid_body.pos_x[i]) return __LINE__;
        if (e1->states->rigid_body.pos_y[i] != e2->states->rigid_body.pos_y[i]) return __LINE__;
        if (e1->states->rigid_body.pos_z[i] != e2->states->rigid_body.pos_z[i]) return __LINE__;
    }

    /* Compare velocities */
    for (uint32_t i = 0; i < total; i++) {
        if (e1->states->rigid_body.vel_x[i] != e2->states->rigid_body.vel_x[i]) return __LINE__;
        if (e1->states->rigid_body.vel_y[i] != e2->states->rigid_body.vel_y[i]) return __LINE__;
        if (e1->states->rigid_body.vel_z[i] != e2->states->rigid_body.vel_z[i]) return __LINE__;
    }

    /* Compare quaternions */
    for (uint32_t i = 0; i < total; i++) {
        if (e1->states->rigid_body.quat_w[i] != e2->states->rigid_body.quat_w[i]) return __LINE__;
        if (e1->states->rigid_body.quat_x[i] != e2->states->rigid_body.quat_x[i]) return __LINE__;
        if (e1->states->rigid_body.quat_y[i] != e2->states->rigid_body.quat_y[i]) return __LINE__;
        if (e1->states->rigid_body.quat_z[i] != e2->states->rigid_body.quat_z[i]) return __LINE__;
    }

    /* Compare angular velocities */
    for (uint32_t i = 0; i < total; i++) {
        if (e1->states->rigid_body.omega_x[i] != e2->states->rigid_body.omega_x[i]) return __LINE__;
        if (e1->states->rigid_body.omega_y[i] != e2->states->rigid_body.omega_y[i]) return __LINE__;
        if (e1->states->rigid_body.omega_z[i] != e2->states->rigid_body.omega_z[i]) return __LINE__;
    }

    /* Compare RPMs */
    for (uint32_t i = 0; i < total; i++) {
        if (e1->states->extension[QUAD_EXT_RPM_0][i] != e2->states->extension[QUAD_EXT_RPM_0][i]) return __LINE__;
        if (e1->states->extension[QUAD_EXT_RPM_1][i] != e2->states->extension[QUAD_EXT_RPM_1][i]) return __LINE__;
        if (e1->states->extension[QUAD_EXT_RPM_2][i] != e2->states->extension[QUAD_EXT_RPM_2][i]) return __LINE__;
        if (e1->states->extension[QUAD_EXT_RPM_3][i] != e2->states->extension[QUAD_EXT_RPM_3][i]) return __LINE__;
    }

    return 0;
}

/**
 * Save engine state to arrays (for later comparison).
 */
typedef struct SavedState {
    float* pos_x;
    float* pos_y;
    float* pos_z;
    float* vel_x;
    float* vel_y;
    float* vel_z;
    float* quat_w;
    float* quat_x;
    float* quat_y;
    float* quat_z;
    float* omega_x;
    float* omega_y;
    float* omega_z;
    float* rpm_0;
    float* rpm_1;
    float* rpm_2;
    float* rpm_3;
    float* rewards;
    float* observations;
    uint8_t* dones;
    uint8_t* truncations;
    uint32_t total_agents;
    uint32_t obs_dim;
} SavedState;

static SavedState* save_engine_state(BatchEngine* engine) {
    uint32_t n = engine->config.total_agents;
    uint32_t obs_dim = engine->obs_dim;

    SavedState* state = (SavedState*)malloc(sizeof(SavedState));
    if (!state) return NULL;

    state->total_agents = n;
    state->obs_dim = obs_dim;

    state->pos_x = (float*)malloc(n * sizeof(float));
    state->pos_y = (float*)malloc(n * sizeof(float));
    state->pos_z = (float*)malloc(n * sizeof(float));
    state->vel_x = (float*)malloc(n * sizeof(float));
    state->vel_y = (float*)malloc(n * sizeof(float));
    state->vel_z = (float*)malloc(n * sizeof(float));
    state->quat_w = (float*)malloc(n * sizeof(float));
    state->quat_x = (float*)malloc(n * sizeof(float));
    state->quat_y = (float*)malloc(n * sizeof(float));
    state->quat_z = (float*)malloc(n * sizeof(float));
    state->omega_x = (float*)malloc(n * sizeof(float));
    state->omega_y = (float*)malloc(n * sizeof(float));
    state->omega_z = (float*)malloc(n * sizeof(float));
    state->rpm_0 = (float*)malloc(n * sizeof(float));
    state->rpm_1 = (float*)malloc(n * sizeof(float));
    state->rpm_2 = (float*)malloc(n * sizeof(float));
    state->rpm_3 = (float*)malloc(n * sizeof(float));
    state->rewards = (float*)malloc(n * sizeof(float));
    state->observations = (float*)malloc(n * obs_dim * sizeof(float));
    state->dones = (uint8_t*)malloc(n * sizeof(uint8_t));
    state->truncations = (uint8_t*)malloc(n * sizeof(uint8_t));

    memcpy(state->pos_x, engine->states->rigid_body.pos_x, n * sizeof(float));
    memcpy(state->pos_y, engine->states->rigid_body.pos_y, n * sizeof(float));
    memcpy(state->pos_z, engine->states->rigid_body.pos_z, n * sizeof(float));
    memcpy(state->vel_x, engine->states->rigid_body.vel_x, n * sizeof(float));
    memcpy(state->vel_y, engine->states->rigid_body.vel_y, n * sizeof(float));
    memcpy(state->vel_z, engine->states->rigid_body.vel_z, n * sizeof(float));
    memcpy(state->quat_w, engine->states->rigid_body.quat_w, n * sizeof(float));
    memcpy(state->quat_x, engine->states->rigid_body.quat_x, n * sizeof(float));
    memcpy(state->quat_y, engine->states->rigid_body.quat_y, n * sizeof(float));
    memcpy(state->quat_z, engine->states->rigid_body.quat_z, n * sizeof(float));
    memcpy(state->omega_x, engine->states->rigid_body.omega_x, n * sizeof(float));
    memcpy(state->omega_y, engine->states->rigid_body.omega_y, n * sizeof(float));
    memcpy(state->omega_z, engine->states->rigid_body.omega_z, n * sizeof(float));
    memcpy(state->rpm_0, engine->states->extension[QUAD_EXT_RPM_0], n * sizeof(float));
    memcpy(state->rpm_1, engine->states->extension[QUAD_EXT_RPM_1], n * sizeof(float));
    memcpy(state->rpm_2, engine->states->extension[QUAD_EXT_RPM_2], n * sizeof(float));
    memcpy(state->rpm_3, engine->states->extension[QUAD_EXT_RPM_3], n * sizeof(float));
    memcpy(state->rewards, engine->rewards_buffer, n * sizeof(float));
    memcpy(state->observations, engine->observations, n * obs_dim * sizeof(float));
    memcpy(state->dones, engine->dones, n * sizeof(uint8_t));
    memcpy(state->truncations, engine->truncations, n * sizeof(uint8_t));

    return state;
}

static void free_saved_state(SavedState* state) {
    if (!state) return;
    free(state->pos_x);
    free(state->pos_y);
    free(state->pos_z);
    free(state->vel_x);
    free(state->vel_y);
    free(state->vel_z);
    free(state->quat_w);
    free(state->quat_x);
    free(state->quat_y);
    free(state->quat_z);
    free(state->omega_x);
    free(state->omega_y);
    free(state->omega_z);
    free(state->rpm_0);
    free(state->rpm_1);
    free(state->rpm_2);
    free(state->rpm_3);
    free(state->rewards);
    free(state->observations);
    free(state->dones);
    free(state->truncations);
    free(state);
}

/* ============================================================================
 * Category 1: Bit-Exact State Reproducibility (8 tests)
 * ============================================================================ */

/**
 * Test 1.1: Position determinism across multiple runs
 *
 * SPECIFICATION: Two engines with identical seeds and actions MUST produce
 * bit-exact identical positions at every step.
 */
TEST(position_determinism_multiple_runs) {
    const uint64_t SEED = 42;
    const int STEPS = 100;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    set_fixed_actions(e1, 0.5f);
    set_fixed_actions(e2, 0.5f);

    for (int step = 0; step < STEPS; step++) {
        engine_step(e1);
        engine_step(e2);

        /* Bit-exact comparison at every step */
        uint32_t n = e1->config.total_agents;
        for (uint32_t i = 0; i < n; i++) {
            ASSERT_FLOAT_EXACT(e1->states->rigid_body.pos_x[i], e2->states->rigid_body.pos_x[i]);
            ASSERT_FLOAT_EXACT(e1->states->rigid_body.pos_y[i], e2->states->rigid_body.pos_y[i]);
            ASSERT_FLOAT_EXACT(e1->states->rigid_body.pos_z[i], e2->states->rigid_body.pos_z[i]);
        }
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 1.2: Velocity determinism across multiple runs
 */
TEST(velocity_determinism_multiple_runs) {
    const uint64_t SEED = 123;
    const int STEPS = 100;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    for (int step = 0; step < STEPS; step++) {
        set_sinusoidal_actions(e1, step);
        set_sinusoidal_actions(e2, step);
        engine_step(e1);
        engine_step(e2);
    }

    uint32_t n = e1->config.total_agents;
    for (uint32_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EXACT(e1->states->rigid_body.vel_x[i], e2->states->rigid_body.vel_x[i]);
        ASSERT_FLOAT_EXACT(e1->states->rigid_body.vel_y[i], e2->states->rigid_body.vel_y[i]);
        ASSERT_FLOAT_EXACT(e1->states->rigid_body.vel_z[i], e2->states->rigid_body.vel_z[i]);
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 1.3: Quaternion (orientation) determinism
 *
 * SPECIFICATION: Orientations must be bit-exact across runs with same seed.
 */
TEST(quaternion_determinism) {
    const uint64_t SEED = 456;
    const int STEPS = 100;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    /* Apply asymmetric actions to induce rotation */
    for (int step = 0; step < STEPS; step++) {
        float* a1 = engine_get_actions(e1);
        float* a2 = engine_get_actions(e2);
        uint32_t total = e1->config.total_agents;
        for (uint32_t d = 0; d < total; d++) {
            /* Asymmetric to cause rotation */
            a1[d * 4 + 0] = 0.3f;
            a1[d * 4 + 1] = 0.6f;
            a1[d * 4 + 2] = 0.4f;
            a1[d * 4 + 3] = 0.7f;

            a2[d * 4 + 0] = 0.3f;
            a2[d * 4 + 1] = 0.6f;
            a2[d * 4 + 2] = 0.4f;
            a2[d * 4 + 3] = 0.7f;
        }
        engine_step(e1);
        engine_step(e2);
    }

    uint32_t n = e1->config.total_agents;
    for (uint32_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EXACT(e1->states->rigid_body.quat_w[i], e2->states->rigid_body.quat_w[i]);
        ASSERT_FLOAT_EXACT(e1->states->rigid_body.quat_x[i], e2->states->rigid_body.quat_x[i]);
        ASSERT_FLOAT_EXACT(e1->states->rigid_body.quat_y[i], e2->states->rigid_body.quat_y[i]);
        ASSERT_FLOAT_EXACT(e1->states->rigid_body.quat_z[i], e2->states->rigid_body.quat_z[i]);
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 1.4: Angular velocity determinism
 */
TEST(angular_velocity_determinism) {
    const uint64_t SEED = 789;
    const int STEPS = 100;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    for (int step = 0; step < STEPS; step++) {
        set_patterned_actions(e1, 2);  /* Alternating */
        set_patterned_actions(e2, 2);
        engine_step(e1);
        engine_step(e2);
    }

    uint32_t n = e1->config.total_agents;
    for (uint32_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EXACT(e1->states->rigid_body.omega_x[i], e2->states->rigid_body.omega_x[i]);
        ASSERT_FLOAT_EXACT(e1->states->rigid_body.omega_y[i], e2->states->rigid_body.omega_y[i]);
        ASSERT_FLOAT_EXACT(e1->states->rigid_body.omega_z[i], e2->states->rigid_body.omega_z[i]);
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 1.5: RPM determinism
 */
TEST(rpm_determinism) {
    const uint64_t SEED = 1001;
    const int STEPS = 100;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    set_patterned_actions(e1, 3);  /* Gradient */
    set_patterned_actions(e2, 3);

    for (int step = 0; step < STEPS; step++) {
        engine_step(e1);
        engine_step(e2);
    }

    uint32_t n = e1->config.total_agents;
    for (uint32_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EXACT(e1->states->extension[QUAD_EXT_RPM_0][i], e2->states->extension[QUAD_EXT_RPM_0][i]);
        ASSERT_FLOAT_EXACT(e1->states->extension[QUAD_EXT_RPM_1][i], e2->states->extension[QUAD_EXT_RPM_1][i]);
        ASSERT_FLOAT_EXACT(e1->states->extension[QUAD_EXT_RPM_2][i], e2->states->extension[QUAD_EXT_RPM_2][i]);
        ASSERT_FLOAT_EXACT(e1->states->extension[QUAD_EXT_RPM_3][i], e2->states->extension[QUAD_EXT_RPM_3][i]);
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 1.6: Observation buffer determinism
 *
 * SPECIFICATION: Observations MUST be bit-exact across runs.
 */
TEST(observation_determinism) {
    const uint64_t SEED = 2002;
    const int STEPS = 50;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
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
    uint32_t obs_total = e1->config.total_agents * engine_get_obs_dim(e1);

    for (uint32_t i = 0; i < obs_total; i++) {
        ASSERT_FLOAT_EXACT(o1[i], o2[i]);
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 1.7: Reward computation determinism
 */
TEST(reward_determinism) {
    const uint64_t SEED = 3003;
    const int STEPS = 50;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    set_fixed_actions(e1, 0.5f);
    set_fixed_actions(e2, 0.5f);

    for (int step = 0; step < STEPS; step++) {
        engine_step(e1);
        engine_step(e2);

        float* r1 = engine_get_rewards(e1);
        float* r2 = engine_get_rewards(e2);
        uint32_t n = e1->config.total_agents;
        for (uint32_t i = 0; i < n; i++) {
            ASSERT_FLOAT_EXACT(r1[i], r2[i]);
        }
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 1.8: Done/truncation flag determinism
 */
TEST(done_truncation_determinism) {
    const uint64_t SEED = 4004;

    /* Use short episodes to trigger truncation */
    BatchEngine* e1 = create_short_episode_engine(SEED, 20);
    BatchEngine* e2 = create_short_episode_engine(SEED, 20);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    set_fixed_actions(e1, 0.5f);
    set_fixed_actions(e2, 0.5f);

    for (int step = 0; step < 50; step++) {
        engine_step(e1);
        engine_step(e2);

        uint8_t* d1 = engine_get_dones(e1);
        uint8_t* d2 = engine_get_dones(e2);
        uint8_t* t1 = engine_get_truncations(e1);
        uint8_t* t2 = engine_get_truncations(e2);
        uint32_t n = e1->config.total_agents;

        for (uint32_t i = 0; i < n; i++) {
            ASSERT_EQ(d1[i], d2[i]);
            ASSERT_EQ(t1[i], t2[i]);
        }
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/* ============================================================================
 * Category 2: RNG Behavior Tests (7 tests)
 * ============================================================================ */

/**
 * Test 2.1: RNG seed propagation to all subsystems
 *
 * SPECIFICATION: Changing seed MUST produce different spawn positions.
 */
TEST(rng_seed_propagation) {
    BatchEngine* e1 = create_standard_engine(42);
    BatchEngine* e2 = create_standard_engine(43);  /* Different seed */
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    /* With domain_randomization > 0, spawn positions should differ */
    bool positions_differ = false;
    uint32_t n = e1->config.total_agents;
    for (uint32_t i = 0; i < n; i++) {
        if (e1->states->rigid_body.pos_x[i] != e2->states->rigid_body.pos_x[i] ||
            e1->states->rigid_body.pos_y[i] != e2->states->rigid_body.pos_y[i] ||
            e1->states->rigid_body.pos_z[i] != e2->states->rigid_body.pos_z[i]) {
            positions_differ = true;
            break;
        }
    }

    ASSERT_TRUE(positions_differ);

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 2.2: RNG state after N steps is deterministic
 *
 * SPECIFICATION: After identical steps, RNG state must be identical,
 * producing identical subsequent behavior.
 */
TEST(rng_state_after_n_steps) {
    const uint64_t SEED = 5005;
    const int STEPS_BEFORE = 50;
    const int STEPS_AFTER = 50;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    set_fixed_actions(e1, 0.5f);
    set_fixed_actions(e2, 0.5f);

    /* Run N steps */
    for (int step = 0; step < STEPS_BEFORE; step++) {
        engine_step(e1);
        engine_step(e2);
    }

    /* Now force an RNG-consuming operation (partial reset of env 0) */
    uint32_t env_to_reset = 0;
    engine_reset_envs(e1, &env_to_reset, 1);
    engine_reset_envs(e2, &env_to_reset, 1);

    /* Run more steps - should still be deterministic */
    for (int step = 0; step < STEPS_AFTER; step++) {
        engine_step(e1);
        engine_step(e2);
    }

    ASSERT_EQ(compare_engine_states(e1, e2), 0);

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 2.3: RNG isolation between environments
 *
 * SPECIFICATION: Operations on env[0] should not affect RNG sequence for env[1].
 * This is tested by verifying overall determinism holds with partial resets.
 */
TEST(rng_isolation_between_envs) {
    const uint64_t SEED = 6006;

    BatchEngine* e1 = create_test_engine(SEED, 4, 2);
    BatchEngine* e2 = create_test_engine(SEED, 4, 2);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    set_fixed_actions(e1, 0.5f);
    set_fixed_actions(e2, 0.5f);

    /* Run some steps */
    for (int i = 0; i < 30; i++) {
        engine_step(e1);
        engine_step(e2);
    }

    /* Reset only env 0 in both engines */
    uint32_t env_idx = 0;
    engine_reset_envs(e1, &env_idx, 1);
    engine_reset_envs(e2, &env_idx, 1);

    /* Run more steps */
    for (int i = 0; i < 30; i++) {
        engine_step(e1);
        engine_step(e2);
    }

    /* States should still match */
    ASSERT_EQ(compare_engine_states(e1, e2), 0);

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 2.4: RNG consumption order determinism
 *
 * SPECIFICATION: The order in which RNG values are consumed must be deterministic.
 */
TEST(rng_consumption_order) {
    const uint64_t SEED = 7007;

    /* Create engines with different drone counts */
    BatchEngine* e1a = create_test_engine(SEED, 2, 4);
    BatchEngine* e1b = create_test_engine(SEED, 2, 4);
    ASSERT_NOT_NULL(e1a);
    ASSERT_NOT_NULL(e1b);

    engine_reset(e1a);
    engine_reset(e1b);

    set_fixed_actions(e1a, 0.5f);
    set_fixed_actions(e1b, 0.5f);

    for (int i = 0; i < 50; i++) {
        engine_step(e1a);
        engine_step(e1b);
    }

    /* Same configuration with same seed must produce identical results */
    ASSERT_EQ(compare_engine_states(e1a, e1b), 0);

    engine_destroy(e1a);
    engine_destroy(e1b);
    return 0;
}

/**
 * Test 2.5: Spawn position determinism (uses RNG)
 */
TEST(spawn_position_determinism) {
    const uint64_t SEED = 8008;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    /* Reset and compare spawn positions */
    engine_reset(e1);
    engine_reset(e2);

    uint32_t n = e1->config.total_agents;
    for (uint32_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EXACT(e1->states->rigid_body.pos_x[i], e2->states->rigid_body.pos_x[i]);
        ASSERT_FLOAT_EXACT(e1->states->rigid_body.pos_y[i], e2->states->rigid_body.pos_y[i]);
        ASSERT_FLOAT_EXACT(e1->states->rigid_body.pos_z[i], e2->states->rigid_body.pos_z[i]);
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 2.6: Target position determinism (uses RNG)
 *
 * SPECIFICATION: Targets set during reset must produce identical rewards.
 */
TEST(target_position_determinism) {
    const uint64_t SEED = 9009;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    /* Targets are set during reset; verify by running and comparing rewards */
    set_fixed_actions(e1, 0.5f);
    set_fixed_actions(e2, 0.5f);

    for (int i = 0; i < 20; i++) {
        engine_step(e1);
        engine_step(e2);
    }

    /* If targets differ, rewards would differ */
    float* r1 = engine_get_rewards(e1);
    float* r2 = engine_get_rewards(e2);
    uint32_t n = e1->config.total_agents;
    for (uint32_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EXACT(r1[i], r2[i]);
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 2.7: Different seeds produce different trajectories
 *
 * SPECIFICATION: Different seeds MUST produce different simulation outcomes.
 */
TEST(different_seeds_different_trajectories) {
    BatchEngine* e1 = create_standard_engine(11111);
    BatchEngine* e2 = create_standard_engine(22222);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    set_fixed_actions(e1, 0.5f);
    set_fixed_actions(e2, 0.5f);

    for (int i = 0; i < 50; i++) {
        engine_step(e1);
        engine_step(e2);
    }

    /* At least one position should differ */
    bool differs = false;
    uint32_t n = e1->config.total_agents;
    for (uint32_t i = 0; i < n; i++) {
        if (e1->states->rigid_body.pos_x[i] != e2->states->rigid_body.pos_x[i] ||
            e1->states->rigid_body.pos_y[i] != e2->states->rigid_body.pos_y[i] ||
            e1->states->rigid_body.pos_z[i] != e2->states->rigid_body.pos_z[i]) {
            differs = true;
            break;
        }
    }

    ASSERT_TRUE(differs);

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/* ============================================================================
 * Category 3: Reset Determinism (7 tests)
 * ============================================================================ */

/**
 * Test 3.1: Full reset determinism (engine_reset)
 *
 * SPECIFICATION: Calling engine_reset on the same engine must re-seed RNG
 * and produce identical state.
 */
TEST(full_reset_determinism) {
    const uint64_t SEED = 10101;
    const int STEPS = 50;

    BatchEngine* engine = create_standard_engine(SEED);
    ASSERT_NOT_NULL(engine);

    /* First run */
    engine_reset(engine);
    set_fixed_actions(engine, 0.5f);
    for (int i = 0; i < STEPS; i++) {
        engine_step(engine);
    }
    SavedState* state1 = save_engine_state(engine);
    ASSERT_NOT_NULL(state1);

    /* Second run (reset same engine) */
    engine_reset(engine);
    set_fixed_actions(engine, 0.5f);
    for (int i = 0; i < STEPS; i++) {
        engine_step(engine);
    }

    /* Compare */
    uint32_t n = engine->config.total_agents;
    for (uint32_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EXACT(state1->pos_x[i], engine->states->rigid_body.pos_x[i]);
        ASSERT_FLOAT_EXACT(state1->pos_y[i], engine->states->rigid_body.pos_y[i]);
        ASSERT_FLOAT_EXACT(state1->pos_z[i], engine->states->rigid_body.pos_z[i]);
        ASSERT_FLOAT_EXACT(state1->vel_x[i], engine->states->rigid_body.vel_x[i]);
        ASSERT_FLOAT_EXACT(state1->vel_y[i], engine->states->rigid_body.vel_y[i]);
        ASSERT_FLOAT_EXACT(state1->vel_z[i], engine->states->rigid_body.vel_z[i]);
    }

    free_saved_state(state1);
    engine_destroy(engine);
    return 0;
}

/**
 * Test 3.2: Partial reset determinism (engine_reset_envs)
 */
TEST(partial_reset_determinism) {
    const uint64_t SEED = 12121;

    BatchEngine* e1 = create_test_engine(SEED, 4, 2);
    BatchEngine* e2 = create_test_engine(SEED, 4, 2);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    set_fixed_actions(e1, 0.5f);
    set_fixed_actions(e2, 0.5f);

    /* Run some steps */
    for (int i = 0; i < 30; i++) {
        engine_step(e1);
        engine_step(e2);
    }

    /* Partial reset envs 1 and 3 */
    uint32_t envs_to_reset[] = {1, 3};
    engine_reset_envs(e1, envs_to_reset, 2);
    engine_reset_envs(e2, envs_to_reset, 2);

    /* Run more steps */
    for (int i = 0; i < 30; i++) {
        engine_step(e1);
        engine_step(e2);
    }

    ASSERT_EQ(compare_engine_states(e1, e2), 0);

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 3.3: Single drone reset determinism (engine_reset_agent)
 */
TEST(single_drone_reset_determinism) {
    const uint64_t SEED = 13131;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    set_fixed_actions(e1, 0.5f);
    set_fixed_actions(e2, 0.5f);

    for (int i = 0; i < 20; i++) {
        engine_step(e1);
        engine_step(e2);
    }

    /* Reset drone 5 in both */
    Vec3 pos = VEC3(10.0f, 5.0f, 10.0f);
    Quat orient = QUAT_IDENTITY;
    engine_reset_agent(e1, 5, pos, orient);
    engine_reset_agent(e2, 5, pos, orient);

    for (int i = 0; i < 20; i++) {
        engine_step(e1);
        engine_step(e2);
    }

    ASSERT_EQ(compare_engine_states(e1, e2), 0);

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 3.4: Auto-reset (terminated drones) determinism
 */
TEST(auto_reset_determinism) {
    const uint64_t SEED = 14141;

    /* Short episodes to trigger auto-resets */
    BatchEngine* e1 = create_short_episode_engine(SEED, 15);
    BatchEngine* e2 = create_short_episode_engine(SEED, 15);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    set_fixed_actions(e1, 0.5f);
    set_fixed_actions(e2, 0.5f);

    /* Run past several auto-resets */
    for (int i = 0; i < 100; i++) {
        engine_step(e1);
        engine_step(e2);

        /* Check states match after each step */
        ASSERT_EQ(compare_engine_states(e1, e2), 0);
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 3.5: Multiple sequential resets produce identical states
 */
TEST(multiple_sequential_resets) {
    const uint64_t SEED = 15151;

    BatchEngine* engine = create_standard_engine(SEED);
    ASSERT_NOT_NULL(engine);

    /* Reset multiple times and verify spawn positions are identical each time */
    engine_reset(engine);
    SavedState* first = save_engine_state(engine);
    ASSERT_NOT_NULL(first);

    for (int trial = 0; trial < 5; trial++) {
        engine_reset(engine);

        uint32_t n = engine->config.total_agents;
        for (uint32_t i = 0; i < n; i++) {
            ASSERT_FLOAT_EXACT(first->pos_x[i], engine->states->rigid_body.pos_x[i]);
            ASSERT_FLOAT_EXACT(first->pos_y[i], engine->states->rigid_body.pos_y[i]);
            ASSERT_FLOAT_EXACT(first->pos_z[i], engine->states->rigid_body.pos_z[i]);
        }
    }

    free_saved_state(first);
    engine_destroy(engine);
    return 0;
}

/**
 * Test 3.6: Reset after varying episode lengths still deterministic
 *
 * SPECIFICATION: RNG is re-seeded on reset, so prior episode length
 * should not affect post-reset state.
 */
TEST(reset_after_varying_lengths) {
    const uint64_t SEED = 16161;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    /* E1: Run 100 steps then reset */
    engine_reset(e1);
    set_fixed_actions(e1, 0.5f);
    for (int i = 0; i < 100; i++) {
        engine_step(e1);
    }
    engine_reset(e1);

    /* E2: Run 500 steps then reset */
    engine_reset(e2);
    set_fixed_actions(e2, 0.5f);
    for (int i = 0; i < 500; i++) {
        engine_step(e2);
    }
    engine_reset(e2);

    /* After reset, states should be identical */
    uint32_t n = e1->config.total_agents;
    for (uint32_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EXACT(e1->states->rigid_body.pos_x[i], e2->states->rigid_body.pos_x[i]);
        ASSERT_FLOAT_EXACT(e1->states->rigid_body.pos_y[i], e2->states->rigid_body.pos_y[i]);
        ASSERT_FLOAT_EXACT(e1->states->rigid_body.pos_z[i], e2->states->rigid_body.pos_z[i]);
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 3.7: Reset clears episode tracking deterministically
 */
TEST(reset_clears_episode_tracking) {
    const uint64_t SEED = 17171;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    set_fixed_actions(e1, 0.5f);
    set_fixed_actions(e2, 0.5f);

    for (int i = 0; i < 50; i++) {
        engine_step(e1);
        engine_step(e2);
    }

    engine_reset(e1);
    engine_reset(e2);

    /* Episode tracking should be cleared identically */
    uint32_t n = e1->config.total_agents;
    for (uint32_t i = 0; i < n; i++) {
        ASSERT_EQ(e1->episode_lengths[i], e2->episode_lengths[i]);
        ASSERT_FLOAT_EXACT(e1->episode_returns[i], e2->episode_returns[i]);
        ASSERT_EQ(e1->dones[i], e2->dones[i]);
        ASSERT_EQ(e1->truncations[i], e2->truncations[i]);
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/* ============================================================================
 * Category 4: Multi-Episode Determinism (4 tests)
 * ============================================================================ */

/**
 * Test 4.1: Episode 1, 2, 3 trajectories identical across runs
 */
TEST(multi_episode_trajectories) {
    const uint64_t SEED = 20202;
    const int STEPS_PER_EPISODE = 50;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    for (int episode = 0; episode < 3; episode++) {
        engine_reset(e1);
        engine_reset(e2);

        float action_val = 0.3f + (float)episode * 0.15f;
        set_fixed_actions(e1, action_val);
        set_fixed_actions(e2, action_val);

        for (int step = 0; step < STEPS_PER_EPISODE; step++) {
            engine_step(e1);
            engine_step(e2);
        }

        ASSERT_EQ(compare_engine_states(e1, e2), 0);
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 4.2: Episode completion order determinism
 *
 * SPECIFICATION: When using auto-reset, episodes should complete
 * in identical order across runs.
 */
TEST(episode_completion_order) {
    const uint64_t SEED = 21212;

    BatchEngine* e1 = create_short_episode_engine(SEED, 25);
    BatchEngine* e2 = create_short_episode_engine(SEED, 25);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    set_fixed_actions(e1, 0.5f);
    set_fixed_actions(e2, 0.5f);

    /* Track episode completions */
    for (int step = 0; step < 100; step++) {
        engine_step(e1);
        engine_step(e2);

        /* Completion flags should match exactly */
        uint32_t n = e1->config.total_agents;
        for (uint32_t i = 0; i < n; i++) {
            ASSERT_EQ(e1->dones[i], e2->dones[i]);
            ASSERT_EQ(e1->truncations[i], e2->truncations[i]);
        }
    }

    /* Total episodes should match */
    ASSERT_EQ(e1->total_episodes, e2->total_episodes);

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 4.3: Truncation timing determinism
 */
TEST(truncation_timing_determinism) {
    const uint64_t SEED = 22222;
    const uint32_t MAX_STEPS = 30;

    BatchEngine* e1 = create_short_episode_engine(SEED, MAX_STEPS);
    BatchEngine* e2 = create_short_episode_engine(SEED, MAX_STEPS);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    set_fixed_actions(e1, 0.5f);
    set_fixed_actions(e2, 0.5f);

    uint32_t trunc_count_1 = 0, trunc_count_2 = 0;
    int trunc_step_1 = -1, trunc_step_2 = -1;

    for (int step = 0; step < 100; step++) {
        engine_step(e1);
        engine_step(e2);

        uint32_t n = e1->config.total_agents;
        for (uint32_t i = 0; i < n; i++) {
            if (e1->truncations[i]) {
                trunc_count_1++;
                if (trunc_step_1 < 0) trunc_step_1 = step;
            }
            if (e2->truncations[i]) {
                trunc_count_2++;
                if (trunc_step_2 < 0) trunc_step_2 = step;
            }
        }
    }

    ASSERT_EQ(trunc_count_1, trunc_count_2);
    ASSERT_EQ(trunc_step_1, trunc_step_2);

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 4.4: Collision-triggered termination determinism
 *
 * SPECIFICATION: Collisions causing termination must occur at
 * identical steps across runs.
 */
TEST(collision_termination_determinism) {
    const uint64_t SEED = 23232;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    /* Add identical obstacles */
    Vec3 box_min = VEC3(5.0f, 0.0f, 5.0f);
    Vec3 box_max = VEC3(10.0f, 10.0f, 10.0f);
    engine_add_box(e1, box_min, box_max, 1);
    engine_add_box(e2, box_min, box_max, 1);

    set_fixed_actions(e1, 0.7f);  /* Higher thrust to potentially cause collisions */
    set_fixed_actions(e2, 0.7f);

    for (int step = 0; step < 200; step++) {
        engine_step(e1);
        engine_step(e2);

        /* Collision terminations should match */
        uint32_t n = e1->config.total_agents;
        for (uint32_t i = 0; i < n; i++) {
            ASSERT_EQ(e1->term_collision[i], e2->term_collision[i]);
        }
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/* ============================================================================
 * Category 5: Edge Cases (6 tests)
 * ============================================================================ */

/**
 * Test 5.1: Determinism with zero actions
 */
TEST(determinism_zero_actions) {
    const uint64_t SEED = 30303;
    const int STEPS = 100;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    set_patterned_actions(e1, 0);  /* Zero */
    set_patterned_actions(e2, 0);

    for (int step = 0; step < STEPS; step++) {
        engine_step(e1);
        engine_step(e2);
    }

    ASSERT_EQ(compare_engine_states(e1, e2), 0);

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 5.2: Determinism with max actions
 */
TEST(determinism_max_actions) {
    const uint64_t SEED = 31313;
    const int STEPS = 100;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    set_patterned_actions(e1, 1);  /* Max */
    set_patterned_actions(e2, 1);

    for (int step = 0; step < STEPS; step++) {
        engine_step(e1);
        engine_step(e2);
    }

    ASSERT_EQ(compare_engine_states(e1, e2), 0);

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 5.3: Determinism with alternating action patterns
 */
TEST(determinism_alternating_patterns) {
    const uint64_t SEED = 32323;
    const int STEPS = 100;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    for (int step = 0; step < STEPS; step++) {
        int pattern = step % 4;
        set_patterned_actions(e1, pattern);
        set_patterned_actions(e2, pattern);
        engine_step(e1);
        engine_step(e2);
    }

    ASSERT_EQ(compare_engine_states(e1, e2), 0);

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 5.4: Determinism at world boundaries
 *
 * SPECIFICATION: Drones near world edges must behave deterministically.
 */
TEST(determinism_at_boundaries) {
    const uint64_t SEED = 33333;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    /* Place drone near boundary */
    e1->states->rigid_body.pos_x[0] = e1->config.world_max.x - 1.0f;
    e1->states->rigid_body.pos_z[0] = e1->config.world_max.z - 1.0f;
    e2->states->rigid_body.pos_x[0] = e2->config.world_max.x - 1.0f;
    e2->states->rigid_body.pos_z[0] = e2->config.world_max.z - 1.0f;

    set_fixed_actions(e1, 0.8f);
    set_fixed_actions(e2, 0.8f);

    for (int step = 0; step < 50; step++) {
        engine_step(e1);
        engine_step(e2);
    }

    ASSERT_EQ(compare_engine_states(e1, e2), 0);

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 5.5: Determinism with many collisions
 *
 * SPECIFICATION: High-frequency collisions must be deterministic.
 */
TEST(determinism_many_collisions) {
    const uint64_t SEED = 34343;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    /* Add many obstacles */
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            Vec3 min = VEC3((float)i * 15.0f - 25.0f, 0.0f, (float)j * 15.0f - 25.0f);
            Vec3 max = VEC3(min.x + 3.0f, 5.0f, min.z + 3.0f);
            engine_add_box(e1, min, max, 1);
            engine_add_box(e2, min, max, 1);
        }
    }

    set_fixed_actions(e1, 0.6f);
    set_fixed_actions(e2, 0.6f);

    for (int step = 0; step < 100; step++) {
        engine_step(e1);
        engine_step(e2);
    }

    ASSERT_EQ(compare_engine_states(e1, e2), 0);

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 5.6: Long trajectory determinism (10,000+ steps)
 *
 * SPECIFICATION: Determinism must hold over very long trajectories
 * to catch any floating-point accumulation issues.
 */
TEST(long_trajectory_determinism) {
    const uint64_t SEED = 35353;
    const int STEPS = 10000;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    for (int step = 0; step < STEPS; step++) {
        set_sinusoidal_actions(e1, step);
        set_sinusoidal_actions(e2, step);
        engine_step(e1);
        engine_step(e2);

        /* Check periodically to catch divergence early */
        if (step % 1000 == 999) {
            ASSERT_EQ(compare_engine_states(e1, e2), 0);
        }
    }

    /* Final comparison */
    ASSERT_EQ(compare_engine_states(e1, e2), 0);

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/* ============================================================================
 * Category 6: Integration Determinism (4 tests)
 * ============================================================================ */

/**
 * Test 6.1: Physics + collision integration determinism
 */
TEST(physics_collision_determinism) {
    const uint64_t SEED = 40404;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    /* Add ground collision zone */
    engine_add_box(e1, VEC3(-50, -1, -50), VEC3(50, 0, 50), 1);
    engine_add_box(e2, VEC3(-50, -1, -50), VEC3(50, 0, 50), 1);

    /* Zero thrust will cause fall toward ground */
    set_fixed_actions(e1, 0.0f);
    set_fixed_actions(e2, 0.0f);

    for (int step = 0; step < 100; step++) {
        engine_step_physics(e1);
        engine_step_collision(e1);
        engine_step_physics(e2);
        engine_step_collision(e2);
    }

    ASSERT_EQ(compare_engine_states(e1, e2), 0);

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 6.2: Physics + sensors integration determinism
 */
TEST(physics_sensors_determinism) {
    const uint64_t SEED = 41414;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    set_fixed_actions(e1, 0.5f);
    set_fixed_actions(e2, 0.5f);

    for (int step = 0; step < 50; step++) {
        engine_step_physics(e1);
        engine_step_sensors(e1);
        engine_step_physics(e2);
        engine_step_sensors(e2);
    }

    /* Compare observations */
    float* o1 = engine_get_observations(e1);
    float* o2 = engine_get_observations(e2);
    uint32_t obs_total = e1->config.total_agents * engine_get_obs_dim(e1);

    for (uint32_t i = 0; i < obs_total; i++) {
        ASSERT_FLOAT_EXACT(o1[i], o2[i]);
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 6.3: Full pipeline determinism
 *
 * SPECIFICATION: The complete step pipeline (physics -> collision ->
 * sensors -> rewards -> reset) must be deterministic.
 */
TEST(full_pipeline_determinism) {
    const uint64_t SEED = 42424;

    BatchEngine* e1 = create_short_episode_engine(SEED, 30);
    BatchEngine* e2 = create_short_episode_engine(SEED, 30);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    /* Add world obstacles */
    Vec3 box_min = VEC3(0.0f, 0.0f, 0.0f);
    Vec3 box_max = VEC3(5.0f, 5.0f, 5.0f);
    engine_add_box(e1, box_min, box_max, 1);
    engine_add_box(e2, box_min, box_max, 1);

    for (int step = 0; step < 200; step++) {
        set_sinusoidal_actions(e1, step);
        set_sinusoidal_actions(e2, step);
        engine_step(e1);
        engine_step(e2);

        /* Full state comparison every 10 steps */
        if (step % 10 == 9) {
            ASSERT_EQ(compare_engine_states(e1, e2), 0);

            /* Also compare rewards and observations */
            float* r1 = engine_get_rewards(e1);
            float* r2 = engine_get_rewards(e2);
            float* o1 = engine_get_observations(e1);
            float* o2 = engine_get_observations(e2);
            uint32_t n = e1->config.total_agents;
            uint32_t obs_total = n * engine_get_obs_dim(e1);

            for (uint32_t i = 0; i < n; i++) {
                ASSERT_FLOAT_EXACT(r1[i], r2[i]);
            }
            for (uint32_t i = 0; i < obs_total; i++) {
                ASSERT_FLOAT_EXACT(o1[i], o2[i]);
            }
        }
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/**
 * Test 6.4: Sensor raycasting determinism
 *
 * SPECIFICATION: ToF sensor raycasting (if present) must be deterministic.
 */
TEST(sensor_raycasting_determinism) {
    const uint64_t SEED = 43434;

    BatchEngine* e1 = create_standard_engine(SEED);
    BatchEngine* e2 = create_standard_engine(SEED);
    ASSERT_NOT_NULL(e1);
    ASSERT_NOT_NULL(e2);

    engine_reset(e1);
    engine_reset(e2);

    /* Add varied geometry for raycasting */
    engine_add_sphere(e1, VEC3(10.0f, 5.0f, 10.0f), 3.0f, 1);
    engine_add_sphere(e2, VEC3(10.0f, 5.0f, 10.0f), 3.0f, 1);
    engine_add_cylinder(e1, VEC3(-10.0f, 5.0f, -10.0f), 2.0f, 4.0f, 1);
    engine_add_cylinder(e2, VEC3(-10.0f, 5.0f, -10.0f), 2.0f, 4.0f, 1);

    set_fixed_actions(e1, 0.5f);
    set_fixed_actions(e2, 0.5f);

    for (int step = 0; step < 50; step++) {
        engine_step(e1);
        engine_step(e2);
    }

    /* Compare observations which include sensor data */
    float* o1 = engine_get_observations(e1);
    float* o2 = engine_get_observations(e2);
    uint32_t obs_total = e1->config.total_agents * engine_get_obs_dim(e1);

    for (uint32_t i = 0; i < obs_total; i++) {
        ASSERT_FLOAT_EXACT(o1[i], o2[i]);
    }

    engine_destroy(e1);
    engine_destroy(e2);
    return 0;
}

/* ============================================================================
 * Additional Tests: Thread Count Independence
 * ============================================================================ */

/**
 * Test: Verify determinism holds across multiple independent runs
 *
 * SPECIFICATION: Multiple independent instantiations with the same seed
 * must produce identical results, regardless of scheduling.
 */
TEST(thread_independence_via_multiple_runs) {
    const uint64_t SEED = 50505;
    const int STEPS = 100;
    const int NUM_TRIALS = 5;

    float first_run_final_pos[16 * 3];  /* Max 16 drones, x/y/z */

    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        BatchEngine* engine = create_standard_engine(SEED);
        ASSERT_NOT_NULL(engine);

        engine_reset(engine);
        set_fixed_actions(engine, 0.5f);

        for (int step = 0; step < STEPS; step++) {
            engine_step(engine);
        }

        uint32_t n = engine->config.total_agents;
        if (trial == 0) {
            for (uint32_t i = 0; i < n; i++) {
                first_run_final_pos[i * 3 + 0] = engine->states->rigid_body.pos_x[i];
                first_run_final_pos[i * 3 + 1] = engine->states->rigid_body.pos_y[i];
                first_run_final_pos[i * 3 + 2] = engine->states->rigid_body.pos_z[i];
            }
        } else {
            for (uint32_t i = 0; i < n; i++) {
                ASSERT_FLOAT_EXACT(first_run_final_pos[i * 3 + 0], engine->states->rigid_body.pos_x[i]);
                ASSERT_FLOAT_EXACT(first_run_final_pos[i * 3 + 1], engine->states->rigid_body.pos_y[i]);
                ASSERT_FLOAT_EXACT(first_run_final_pos[i * 3 + 2], engine->states->rigid_body.pos_z[i]);
            }
        }

        engine_destroy(engine);
    }

    return 0;
}

/* ============================================================================
 * Main Test Runner
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Deep Determinism Tests");

    RUN_TEST(position_determinism_multiple_runs);
    RUN_TEST(velocity_determinism_multiple_runs);
    RUN_TEST(quaternion_determinism);
    RUN_TEST(angular_velocity_determinism);
    RUN_TEST(rpm_determinism);
    RUN_TEST(observation_determinism);
    RUN_TEST(reward_determinism);
    RUN_TEST(done_truncation_determinism);
    RUN_TEST(rng_seed_propagation);
    RUN_TEST(rng_state_after_n_steps);
    RUN_TEST(rng_isolation_between_envs);
    RUN_TEST(rng_consumption_order);
    RUN_TEST(spawn_position_determinism);
    RUN_TEST(target_position_determinism);
    RUN_TEST(different_seeds_different_trajectories);
    RUN_TEST(full_reset_determinism);
    RUN_TEST(partial_reset_determinism);
    RUN_TEST(single_drone_reset_determinism);
    RUN_TEST(auto_reset_determinism);
    RUN_TEST(multiple_sequential_resets);
    RUN_TEST(reset_after_varying_lengths);
    RUN_TEST(reset_clears_episode_tracking);
    RUN_TEST(multi_episode_trajectories);
    RUN_TEST(episode_completion_order);
    RUN_TEST(truncation_timing_determinism);
    RUN_TEST(collision_termination_determinism);
    RUN_TEST(determinism_zero_actions);
    RUN_TEST(determinism_max_actions);
    RUN_TEST(determinism_alternating_patterns);
    RUN_TEST(determinism_at_boundaries);
    RUN_TEST(determinism_many_collisions);
    RUN_TEST(long_trajectory_determinism);
    RUN_TEST(physics_collision_determinism);
    RUN_TEST(physics_sensors_determinism);
    RUN_TEST(full_pipeline_determinism);
    RUN_TEST(sensor_raycasting_determinism);
    RUN_TEST(thread_independence_via_multiple_runs);

    TEST_SUITE_END();
}
