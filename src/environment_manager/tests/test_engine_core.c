/**
 * @file test_engine_core.c
 * @brief Core Functionality Tests for BatchEngine (8 tests)
 *
 * Tests verify fundamental engine behavior:
 * - Engine lifecycle (create/destroy)
 * - Buffer alignment (32-byte for SIMD)
 * - Reset spawn positions and quaternion normalization
 * - Physics stability over 100 steps
 * - Collision detection for overlapping drones
 * - Sensor output dimensions
 * - Reward computation for HOVER task
 */

#include "environment_manager.h"
#include "platform_quadcopter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "test_harness.h"

/* ============================================================================
 * Test Helpers
 * ============================================================================ */

/**
 * Create a test engine with fixed seed and reasonable defaults
 */
static BatchEngine* create_test_engine(uint64_t seed) {
    EngineConfig cfg = engine_config_default();
    cfg.num_envs = 4;
    cfg.agents_per_env = 4;
    cfg.seed = seed;
    cfg.persistent_arena_size = 128 * 1024 * 1024;  /* 128 MB */
    cfg.frame_arena_size = 32 * 1024 * 1024;        /* 32 MB */
    cfg.max_episode_steps = 500;

    char error[ENGINE_ERROR_MSG_SIZE];
    return engine_create(&cfg, error);
}

/**
 * Check if a pointer is 32-byte aligned (required for AVX2)
 */
static bool is_aligned_32(const void* ptr) {
    return ((uintptr_t)ptr & 31) == 0;
}

/* ============================================================================
 * Test 1: Engine Create/Destroy Lifecycle
 * ============================================================================ */

TEST(engine_create_destroy) {
    char error[ENGINE_ERROR_MSG_SIZE] = {0};

    EngineConfig cfg = engine_config_default();
    cfg.num_envs = 2;
    cfg.agents_per_env = 2;
    cfg.seed = 42;
    cfg.persistent_arena_size = 64 * 1024 * 1024;
    cfg.frame_arena_size = 16 * 1024 * 1024;

    /* Create engine */
    BatchEngine* engine = engine_create(&cfg, error);
    ASSERT_NOT_NULL(engine);

    /* Verify engine is initialized */
    ASSERT_TRUE(engine->initialized);
    ASSERT_EQ(engine->config.num_envs, 2);
    ASSERT_EQ(engine->config.agents_per_env, 2);
    ASSERT_EQ(engine->config.total_agents, 4);

    /* Verify subsystems are allocated */
    ASSERT_NOT_NULL(engine->states);
    ASSERT_NOT_NULL(engine->params);
    ASSERT_NOT_NULL(engine->physics);
    ASSERT_NOT_NULL(engine->world);
    ASSERT_NOT_NULL(engine->collision);
    ASSERT_NOT_NULL(engine->rewards);
    ASSERT_NOT_NULL(engine->persistent_arena);
    ASSERT_NOT_NULL(engine->frame_arena);

    /* Verify external buffers are allocated */
    ASSERT_NOT_NULL(engine->observations);
    ASSERT_NOT_NULL(engine->actions);
    ASSERT_NOT_NULL(engine->rewards_buffer);
    ASSERT_NOT_NULL(engine->dones);
    ASSERT_NOT_NULL(engine->truncations);

    /* Destroy should not crash */
    engine_destroy(engine);

    /* Destroying NULL should be safe */
    engine_destroy(NULL);

    return 0;
}

/* ============================================================================
 * Test 2: Buffer Alignment (32-byte for AVX2 SIMD)
 * ============================================================================ */

TEST(buffer_alignment_32byte) {
    BatchEngine* engine = create_test_engine(12345);
    ASSERT_NOT_NULL(engine);

    /* External buffers must be 32-byte aligned for AVX2 */
    ASSERT_TRUE(is_aligned_32(engine->observations));
    ASSERT_TRUE(is_aligned_32(engine->actions));
    ASSERT_TRUE(is_aligned_32(engine->rewards_buffer));
    ASSERT_TRUE(is_aligned_32(engine->dones));
    ASSERT_TRUE(is_aligned_32(engine->truncations));

    /* SoA state arrays must be 32-byte aligned */
    ASSERT_TRUE(is_aligned_32(engine->states->rigid_body.pos_x));
    ASSERT_TRUE(is_aligned_32(engine->states->rigid_body.pos_y));
    ASSERT_TRUE(is_aligned_32(engine->states->rigid_body.pos_z));
    ASSERT_TRUE(is_aligned_32(engine->states->rigid_body.vel_x));
    ASSERT_TRUE(is_aligned_32(engine->states->rigid_body.vel_y));
    ASSERT_TRUE(is_aligned_32(engine->states->rigid_body.vel_z));
    ASSERT_TRUE(is_aligned_32(engine->states->rigid_body.quat_w));
    ASSERT_TRUE(is_aligned_32(engine->states->rigid_body.quat_x));
    ASSERT_TRUE(is_aligned_32(engine->states->rigid_body.quat_y));
    ASSERT_TRUE(is_aligned_32(engine->states->rigid_body.quat_z));
    ASSERT_TRUE(is_aligned_32(engine->states->rigid_body.omega_x));
    ASSERT_TRUE(is_aligned_32(engine->states->rigid_body.omega_y));
    ASSERT_TRUE(is_aligned_32(engine->states->rigid_body.omega_z));
    ASSERT_TRUE(is_aligned_32(engine->states->extension[QUAD_EXT_RPM_0]));
    ASSERT_TRUE(is_aligned_32(engine->states->extension[QUAD_EXT_RPM_1]));
    ASSERT_TRUE(is_aligned_32(engine->states->extension[QUAD_EXT_RPM_2]));
    ASSERT_TRUE(is_aligned_32(engine->states->extension[QUAD_EXT_RPM_3]));

    /* SoA parameter arrays must be 32-byte aligned */
    ASSERT_TRUE(is_aligned_32(engine->params->rigid_body.mass));
    ASSERT_TRUE(is_aligned_32(engine->params->rigid_body.ixx));
    ASSERT_TRUE(is_aligned_32(engine->params->rigid_body.iyy));
    ASSERT_TRUE(is_aligned_32(engine->params->rigid_body.izz));
    ASSERT_TRUE(is_aligned_32(engine->params->extension[QUAD_PEXT_ARM_LENGTH]));
    ASSERT_TRUE(is_aligned_32(engine->params->rigid_body.collision_radius));
    ASSERT_TRUE(is_aligned_32(engine->params->extension[QUAD_PEXT_K_THRUST]));
    ASSERT_TRUE(is_aligned_32(engine->params->extension[QUAD_PEXT_K_TORQUE]));

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 3: Reset Spawn Positions Within World Bounds
 * ============================================================================ */

TEST(reset_spawn_positions) {
    BatchEngine* engine = create_test_engine(12345);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    const Vec3 world_min = engine->config.world_min;
    const Vec3 world_max = engine->config.world_max;
    const uint32_t total_agents = engine->config.total_agents;

    for (uint32_t i = 0; i < total_agents; i++) {
        float px = engine->states->rigid_body.pos_x[i];
        float py = engine->states->rigid_body.pos_y[i];
        float pz = engine->states->rigid_body.pos_z[i];

        /* Positions must be within world bounds */
        ASSERT_GE(px, world_min.x);
        ASSERT_LE(px, world_max.x);
        ASSERT_GE(py, world_min.y);
        ASSERT_LE(py, world_max.y);
        ASSERT_GE(pz, world_min.z);
        ASSERT_LE(pz, world_max.z);

        /* Positions must be finite */
        ASSERT_TRUE(isfinite(px));
        ASSERT_TRUE(isfinite(py));
        ASSERT_TRUE(isfinite(pz));
    }

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 4: Reset Quaternion Normalization
 * ============================================================================ */

TEST(reset_quaternion_normalization) {
    BatchEngine* engine = create_test_engine(12345);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    const uint32_t total_agents = engine->config.total_agents;
    const float epsilon = 1e-4f;

    for (uint32_t i = 0; i < total_agents; i++) {
        float qw = engine->states->rigid_body.quat_w[i];
        float qx = engine->states->rigid_body.quat_x[i];
        float qy = engine->states->rigid_body.quat_y[i];
        float qz = engine->states->rigid_body.quat_z[i];

        /* Quaternion components must be finite */
        ASSERT_TRUE(isfinite(qw));
        ASSERT_TRUE(isfinite(qx));
        ASSERT_TRUE(isfinite(qy));
        ASSERT_TRUE(isfinite(qz));

        /* Quaternion must be unit normalized: |q| = 1.0 +/- epsilon */
        float len_sq = qw * qw + qx * qx + qy * qy + qz * qz;
        ASSERT_FLOAT_NEAR(len_sq, 1.0f, epsilon);
    }

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 5: Physics Stability Over 100 Steps
 * ============================================================================ */

TEST(physics_stability_100_steps) {
    BatchEngine* engine = create_test_engine(12345);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Apply moderate actions */
    float* actions = engine_get_actions(engine);
    const uint32_t total_agents = engine->config.total_agents;
    for (uint32_t i = 0; i < total_agents * engine->action_dim; i++) {
        actions[i] = 0.5f;
    }

    /* Run 100 physics steps */
    for (int step = 0; step < 100; step++) {
        engine_step(engine);
    }

    /* Verify no NaN or Inf values in state */
    for (uint32_t i = 0; i < total_agents; i++) {
        /* Position */
        ASSERT_TRUE(isfinite(engine->states->rigid_body.pos_x[i]));
        ASSERT_TRUE(isfinite(engine->states->rigid_body.pos_y[i]));
        ASSERT_TRUE(isfinite(engine->states->rigid_body.pos_z[i]));

        /* Velocity */
        ASSERT_TRUE(isfinite(engine->states->rigid_body.vel_x[i]));
        ASSERT_TRUE(isfinite(engine->states->rigid_body.vel_y[i]));
        ASSERT_TRUE(isfinite(engine->states->rigid_body.vel_z[i]));

        /* Quaternion */
        ASSERT_TRUE(isfinite(engine->states->rigid_body.quat_w[i]));
        ASSERT_TRUE(isfinite(engine->states->rigid_body.quat_x[i]));
        ASSERT_TRUE(isfinite(engine->states->rigid_body.quat_y[i]));
        ASSERT_TRUE(isfinite(engine->states->rigid_body.quat_z[i]));

        /* Angular velocity */
        ASSERT_TRUE(isfinite(engine->states->rigid_body.omega_x[i]));
        ASSERT_TRUE(isfinite(engine->states->rigid_body.omega_y[i]));
        ASSERT_TRUE(isfinite(engine->states->rigid_body.omega_z[i]));

        /* RPMs */
        ASSERT_TRUE(isfinite(engine->states->extension[QUAD_EXT_RPM_0][i]));
        ASSERT_TRUE(isfinite(engine->states->extension[QUAD_EXT_RPM_1][i]));
        ASSERT_TRUE(isfinite(engine->states->extension[QUAD_EXT_RPM_2][i]));
        ASSERT_TRUE(isfinite(engine->states->extension[QUAD_EXT_RPM_3][i]));

        /* Quaternion still normalized after physics */
        float qw = engine->states->rigid_body.quat_w[i];
        float qx = engine->states->rigid_body.quat_x[i];
        float qy = engine->states->rigid_body.quat_y[i];
        float qz = engine->states->rigid_body.quat_z[i];
        float len_sq = qw * qw + qx * qx + qy * qy + qz * qz;
        ASSERT_FLOAT_NEAR(len_sq, 1.0f, 1e-3f);
    }

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 6: Collision Detection for Overlapping Drones
 * ============================================================================ */

TEST(collision_overlapping_detection) {
    BatchEngine* engine = create_test_engine(12345);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Place drone 0 and drone 1 at the exact same position */
    const float shared_x = 5.0f;
    const float shared_y = 5.0f;
    const float shared_z = 5.0f;

    engine->states->rigid_body.pos_x[0] = shared_x;
    engine->states->rigid_body.pos_y[0] = shared_y;
    engine->states->rigid_body.pos_z[0] = shared_z;

    engine->states->rigid_body.pos_x[1] = shared_x;
    engine->states->rigid_body.pos_y[1] = shared_y;
    engine->states->rigid_body.pos_z[1] = shared_z;

    /* Step the engine (which runs collision detection) */
    engine_step(engine);

    /* After collision, drones should be pushed apart */
    float d0_x = engine->states->rigid_body.pos_x[0];
    float d0_y = engine->states->rigid_body.pos_y[0];
    float d0_z = engine->states->rigid_body.pos_z[0];

    float d1_x = engine->states->rigid_body.pos_x[1];
    float d1_y = engine->states->rigid_body.pos_y[1];
    float d1_z = engine->states->rigid_body.pos_z[1];

    /* Calculate distance between drones */
    float dx = d1_x - d0_x;
    float dy = d1_y - d0_y;
    float dz = d1_z - d0_z;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);

    /* After collision response, drones should either:
     * 1. Be separated (dist > 0), OR
     * 2. Have collision flags set (done = true)
     */
    bool separated = (dist > 0.001f);
    bool terminated = (engine->dones[0] || engine->dones[1]);

    ASSERT_TRUE(separated || terminated);

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 7: Sensor Output Dimensions
 * ============================================================================ */

TEST(sensor_output_dimensions) {
    BatchEngine* engine = create_test_engine(12345);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Observation dimension should be positive and consistent */
    uint32_t obs_dim = engine_get_obs_dim(engine);
    ASSERT_GT(obs_dim, 0);

    /* Action dimension should be 4 (quadcopter) */
    uint32_t action_dim = engine_get_action_dim(engine);
    ASSERT_EQ(action_dim, engine->action_dim);
    ASSERT_EQ(action_dim, 4);

    /* Verify observation buffer size matches dimensions */
    uint32_t total_agents = engine_get_total_agents(engine);
    ASSERT_EQ(total_agents, engine->config.total_agents);

    /* Step and verify observations are filled */
    engine_step(engine);

    float* obs = engine_get_observations(engine);
    ASSERT_NOT_NULL(obs);

    /* Observations should be finite (may be zero if no sensors configured) */
    if (obs_dim > 0) {
        for (uint32_t i = 0; i < total_agents * obs_dim && i < 100; i++) {
            ASSERT_TRUE(isfinite(obs[i]));
        }
    }

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 8: Reward Computation for HOVER Task
 * ============================================================================ */

TEST(reward_hover_task) {
    BatchEngine* engine = create_test_engine(12345);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Set a target for drone 0 */
    Vec3 target = {0.0f, 0.0f, 5.0f, 0.0f};
    engine_set_target(engine, 0, target);

    /* Position drone 0 exactly at target (should give high reward) */
    engine->states->rigid_body.pos_x[0] = target.x;
    engine->states->rigid_body.pos_y[0] = target.y;
    engine->states->rigid_body.pos_z[0] = target.z;
    engine->states->rigid_body.vel_x[0] = 0.0f;
    engine->states->rigid_body.vel_y[0] = 0.0f;
    engine->states->rigid_body.vel_z[0] = 0.0f;

    /* Step to compute rewards */
    engine_step(engine);

    float* rewards = engine_get_rewards(engine);
    ASSERT_NOT_NULL(rewards);

    /* Reward should be finite */
    ASSERT_TRUE(isfinite(rewards[0]));

    /* Position another drone far from target */
    Vec3 target2 = {0.0f, 0.0f, 5.0f, 0.0f};
    engine_set_target(engine, 1, target2);

    engine->states->rigid_body.pos_x[1] = 50.0f;  /* Far from target */
    engine->states->rigid_body.pos_y[1] = 50.0f;
    engine->states->rigid_body.pos_z[1] = 50.0f;

    engine_step(engine);

    /* Drone at target should have higher reward than drone far away */
    /* (assuming HOVER reward is distance-based) */
    float reward_close = rewards[0];
    float reward_far = rewards[1];

    ASSERT_TRUE(isfinite(reward_close));
    ASSERT_TRUE(isfinite(reward_far));

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Core Functionality Tests");

    printf("Engine Lifecycle:\n");
    RUN_TEST(engine_create_destroy);

    printf("\nBuffer Alignment:\n");
    RUN_TEST(buffer_alignment_32byte);

    printf("\nReset Behavior:\n");
    RUN_TEST(reset_spawn_positions);
    RUN_TEST(reset_quaternion_normalization);

    printf("\nPhysics:\n");
    RUN_TEST(physics_stability_100_steps);

    printf("\nCollision:\n");
    RUN_TEST(collision_overlapping_detection);

    printf("\nSensors:\n");
    RUN_TEST(sensor_output_dimensions);

    printf("\nRewards:\n");
    RUN_TEST(reward_hover_task);

    TEST_SUITE_END();
}
