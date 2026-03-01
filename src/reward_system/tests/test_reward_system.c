/**
 * Reward System Module Unit Tests
 *
 * Yoneda Philosophy: Tests serve as complete behavioral specification.
 * Each test defines what the module MUST do - the definitive contract.
 */

#include "../include/reward_system.h"
#include "test_harness.h"

#define EPSILON 1e-4f

/* ============================================================================
 * Section 1: System Allocation Tests
 * ============================================================================ */

TEST(allocation_basic) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    RewardSystem* sys = reward_create(arena, NULL, 100, 0);
    ASSERT_NOT_NULL(sys);
    ASSERT_EQ(sys->max_agents, 100);
    ASSERT_NOT_NULL(sys->targets);
    ASSERT_NOT_NULL(sys->termination);

    arena_destroy(arena);
    return 0;
}

TEST(allocation_null_arena) {
    RewardSystem* sys = reward_create(NULL, NULL, 100, 0);
    ASSERT_NULL(sys);
    return 0;
}

TEST(allocation_zero_drones) {
    Arena* arena = arena_create(1024 * 1024);
    RewardSystem* sys = reward_create(arena, NULL, 0, 0);
    ASSERT_NULL(sys);

    arena_destroy(arena);
    return 0;
}

TEST(allocation_large_capacity) {
    Arena* arena = arena_create(16 * 1024 * 1024);
    RewardSystem* sys = reward_create(arena, NULL, 1024, 10);
    ASSERT_NOT_NULL(sys);
    ASSERT_NOT_NULL(sys->gates);
    ASSERT_EQ(sys->max_gates, 10);

    arena_destroy(arena);
    return 0;
}

TEST(allocation_memory_size) {
    size_t size = reward_memory_size(1024, 10);
    ASSERT_TRUE(size > 0);
    ASSERT_TRUE(size <= 128 * 1024);

    size_t size_no_gates = reward_memory_size(1024, 0);
    ASSERT_LT(size_no_gates, size);

    return 0;
}

/* ============================================================================
 * Section 2: System Reset Tests
 * ============================================================================ */

TEST(reset_single) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RewardSystem* sys = reward_create(arena, NULL, 100, 0);

    sys->episode_return[50] = 123.0f;
    sys->episode_length[50] = 500;
    sys->termination->done[50] = 1;

    reward_reset(sys, 50);

    ASSERT_FLOAT_NEAR(sys->episode_return[50], 0.0f, EPSILON);
    ASSERT_EQ(sys->episode_length[50], 0);
    ASSERT_EQ(sys->termination->done[50], 0);

    arena_destroy(arena);
    return 0;
}

TEST(reset_batch) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RewardSystem* sys = reward_create(arena, NULL, 100, 0);

    uint32_t indices[] = {10, 25, 50, 75};
    for (int i = 0; i < 4; i++) {
        sys->episode_return[indices[i]] = 100.0f;
        sys->episode_length[indices[i]] = 200;
    }

    reward_reset_batch(sys, indices, 4);

    for (int i = 0; i < 4; i++) {
        ASSERT_FLOAT_NEAR(sys->episode_return[indices[i]], 0.0f, EPSILON);
        ASSERT_EQ(sys->episode_length[indices[i]], 0);
    }

    arena_destroy(arena);
    return 0;
}

TEST(reset_gate_progress) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    RewardConfig cfg = reward_config_default(TASK_RACE);
    RewardSystem* sys = reward_create(arena, &cfg, 100, 10);

    Vec3 centers[3] = {VEC3(10, 0, 5), VEC3(20, 0, 5), VEC3(30, 0, 5)};
    Vec3 normals[3] = {VEC3(1, 0, 0), VEC3(1, 0, 0), VEC3(1, 0, 0)};
    float radii[3] = {2.0f, 2.0f, 2.0f};
    reward_set_gates(sys, centers, normals, radii, 3);

    sys->gates->current_gate[42] = 2;
    sys->gates_passed[42] = 2;

    reward_reset_gates(sys, 42);

    ASSERT_EQ(sys->gates->current_gate[42], 0);
    ASSERT_EQ(sys->gates_passed[42], 0);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 3: Target Management Tests
 * ============================================================================ */

TEST(target_set_single) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RewardSystem* sys = reward_create(arena, NULL, 100, 0);

    Vec3 pos = VEC3(10.0f, 20.0f, 30.0f);
    Vec3 vel = VEC3(1.0f, 2.0f, 3.0f);
    reward_set_target(sys, 42, pos, vel, 1.5f);

    ASSERT_FLOAT_NEAR(sys->targets->target_x[42], 10.0f, EPSILON);
    ASSERT_FLOAT_NEAR(sys->targets->target_y[42], 20.0f, EPSILON);
    ASSERT_FLOAT_NEAR(sys->targets->target_z[42], 30.0f, EPSILON);
    ASSERT_FLOAT_NEAR(sys->targets->target_vx[42], 1.0f, EPSILON);
    ASSERT_FLOAT_NEAR(sys->targets->target_radius[42], 1.5f, EPSILON);

    arena_destroy(arena);
    return 0;
}

TEST(target_set_random) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RewardSystem* sys = reward_create(arena, NULL, 100, 0);

    PCG32 rng;
    pcg32_seed(&rng, 12345);

    Vec3 bounds_min = VEC3(-10, -10, 0);
    Vec3 bounds_max = VEC3(10, 10, 20);
    reward_set_targets_random(sys, 50, bounds_min, bounds_max, &rng);

    for (uint32_t i = 0; i < 50; i++) {
        ASSERT_TRUE(sys->targets->target_x[i] >= -10.0f &&
                    sys->targets->target_x[i] <= 10.0f);
        ASSERT_TRUE(sys->targets->target_z[i] >= 0.0f &&
                    sys->targets->target_z[i] <= 20.0f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(target_update) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RewardSystem* sys = reward_create(arena, NULL, 10, 0);

    reward_set_target(sys, 0, VEC3(0, 0, 0), VEC3(1, 2, 3), 0.5f);
    reward_update_targets(sys, 0.1f, 10);

    ASSERT_FLOAT_NEAR(sys->targets->target_x[0], 0.1f, EPSILON);
    ASSERT_FLOAT_NEAR(sys->targets->target_y[0], 0.2f, EPSILON);
    ASSERT_FLOAT_NEAR(sys->targets->target_z[0], 0.3f, EPSILON);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 4: Gate Management Tests
 * ============================================================================ */

TEST(gate_set) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    RewardConfig cfg = reward_config_default(TASK_RACE);
    RewardSystem* sys = reward_create(arena, &cfg, 100, 10);

    Vec3 centers[5] = {
        VEC3(10, 0, 5), VEC3(20, 5, 5), VEC3(30, 0, 5),
        VEC3(40, -5, 5), VEC3(50, 0, 5)
    };
    Vec3 normals[5] = {
        VEC3(1, 0, 0), VEC3(1, 0, 0), VEC3(1, 0, 0),
        VEC3(1, 0, 0), VEC3(1, 0, 0)
    };
    float radii[5] = {2.0f, 2.5f, 2.0f, 2.5f, 3.0f};

    reward_set_gates(sys, centers, normals, radii, 5);

    ASSERT_EQ(sys->gates->num_gates, 5);
    ASSERT_FLOAT_NEAR(sys->gates->center_x[0], 10.0f, EPSILON);
    ASSERT_FLOAT_NEAR(sys->gates->radius[4], 3.0f, EPSILON);

    arena_destroy(arena);
    return 0;
}

TEST(gate_crossing_detection) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    RewardConfig cfg = reward_config_default(TASK_RACE);
    RewardSystem* sys = reward_create(arena, &cfg, 10, 5);

    Vec3 centers[1] = {VEC3(10, 0, 5)};
    Vec3 normals[1] = {VEC3(1, 0, 0)};
    float radii[1] = {3.0f};
    reward_set_gates(sys, centers, normals, radii, 1);

    states->pos_x[0] = 11.0f;
    states->pos_y[0] = 0.0f;
    states->pos_z[0] = 5.0f;

    Vec3 prev_pos = VEC3(9.0f, 0.0f, 5.0f);
    bool crossed = reward_check_gate_crossing(sys, states, 0, 0, prev_pos);
    ASSERT_TRUE(crossed);

    prev_pos = VEC3(11.0f, 0.0f, 5.0f);
    crossed = reward_check_gate_crossing(sys, states, 0, 0, prev_pos);
    ASSERT_FALSE(crossed);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 5: Distance Computation Tests
 * ============================================================================ */

TEST(distance_basic) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    RewardSystem* sys = reward_create(arena, NULL, 10, 0);

    states->pos_x[0] = 3.0f;
    states->pos_y[0] = 0.0f;
    states->pos_z[0] = 4.0f;
    reward_set_target(sys, 0, VEC3(0, 0, 0), VEC3_ZERO, 0.5f);

    float dist = reward_distance_to_target(sys, states, 0);
    ASSERT_FLOAT_NEAR(dist, 5.0f, EPSILON);

    arena_destroy(arena);
    return 0;
}

TEST(distance_zero) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    RewardSystem* sys = reward_create(arena, NULL, 10, 0);

    states->pos_x[0] = 5.0f;
    states->pos_y[0] = 5.0f;
    states->pos_z[0] = 5.0f;
    reward_set_target(sys, 0, VEC3(5, 5, 5), VEC3_ZERO, 0.5f);

    float dist = reward_distance_to_target(sys, states, 0);
    ASSERT_FLOAT_NEAR(dist, 0.0f, EPSILON);

    arena_destroy(arena);
    return 0;
}

TEST(reached_target) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    RewardSystem* sys = reward_create(arena, NULL, 10, 0);

    reward_set_target(sys, 0, VEC3(0, 0, 0), VEC3_ZERO, 1.0f);

    states->pos_x[0] = 0.5f;
    states->pos_y[0] = 0.0f;
    states->pos_z[0] = 0.0f;
    ASSERT_TRUE(reward_reached_target(sys, states, 0));

    states->pos_x[0] = 2.0f;
    ASSERT_FALSE(reward_reached_target(sys, states, 0));

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 6: Hover Reward Tests
 * ============================================================================ */

TEST(hover_alive_bonus) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);
    RewardConfig cfg = reward_config_default(TASK_HOVER);
    cfg.distance_scale = 0.0f;
    cfg.uprightness_scale = 0.0f;
    cfg.delta_distance_scale = 0.0f;
    cfg.reach_bonus = 0.0f;
    cfg.energy_scale = 0.0f;
    cfg.jerk_scale = 0.0f;
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    /* Put drone far from target so reach bonus doesn't apply */
    states->pos_x[0] = 100.0f;
    reward_set_target(sys, 0, VEC3(0, 0, 0), VEC3_ZERO, 0.5f);

    float rewards[10];
    reward_compute_hover(sys, states, NULL, NULL, NULL, rewards, 1);

    ASSERT_FLOAT_NEAR(rewards[0], cfg.alive_bonus, 0.01f);

    arena_destroy(arena);
    return 0;
}

TEST(hover_distance_penalty) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_HOVER);
    cfg.alive_bonus = 0.0f;
    cfg.uprightness_scale = 0.0f;
    cfg.distance_scale = 1.0f;
    cfg.delta_distance_scale = 0.0f;
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    reward_set_target(sys, 0, VEC3(0, 0, 0), VEC3_ZERO, 0.5f);
    states->pos_x[0] = 5.0f;

    float rewards[10];
    reward_compute_hover(sys, states, NULL, NULL, NULL, rewards, 1);

    ASSERT_LT(rewards[0], 0.0f);
    ASSERT_FLOAT_NEAR(rewards[0], -5.0f, 0.1f);

    arena_destroy(arena);
    return 0;
}

TEST(hover_reach_bonus) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_HOVER);
    cfg.reach_radius = 1.0f;
    cfg.reach_bonus = 10.0f;
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    reward_set_target(sys, 0, VEC3(0, 0, 0), VEC3_ZERO, 1.0f);
    states->pos_x[0] = 0.3f;

    float rewards[10];
    reward_compute_hover(sys, states, NULL, NULL, NULL, rewards, 1);

    ASSERT_GT(rewards[0], cfg.alive_bonus);

    arena_destroy(arena);
    return 0;
}

TEST(hover_uprightness) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_HOVER);
    cfg.distance_scale = 0.0f;
    cfg.alive_bonus = 0.0f;
    cfg.uprightness_scale = 1.0f;
    cfg.delta_distance_scale = 0.0f;
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    float rewards[10];
    reward_compute_hover(sys, states, NULL, NULL, NULL, rewards, 1);
    float upright_reward = rewards[0];

    states->quat_w[0] = 0.7071f;
    states->quat_x[0] = 0.7071f;
    reward_compute_hover(sys, states, NULL, NULL, NULL, rewards, 1);
    float tilted_reward = rewards[0];

    ASSERT_GT(upright_reward, tilted_reward);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 7: Race Reward Tests
 * ============================================================================ */

TEST(race_gate_bonus) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_RACE);
    cfg.gate_pass_bonus = 20.0f;
    RewardSystem* sys = reward_create(arena, &cfg, 10, 5);

    Vec3 centers[1] = {VEC3(5, 0, 5)};
    Vec3 normals[1] = {VEC3(1, 0, 0)};
    float radii[1] = {3.0f};
    reward_set_gates(sys, centers, normals, radii, 1);

    /* Position drone past the gate with high velocity (to estimate prev_pos before gate) */
    states->pos_x[0] = 6.0f;
    states->pos_z[0] = 5.0f;
    states->vel_x[0] = 200.0f;  /* High velocity so prev_pos estimation puts us before gate */
    sys->prev_distance[0] = 1.0f;  /* Indicates we have valid prev state */

    float rewards[10];
    reward_compute_race(sys, states, NULL, NULL, NULL, rewards, 1);

    ASSERT_EQ(sys->gates_passed[0], 1);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 8: Collision Penalty Tests
 * ============================================================================ */

TEST(collision_world_penalty) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_HOVER);
    cfg.collision_penalty = 10.0f;
    cfg.world_collision_penalty = 5.0f;
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    uint8_t world_flags[10] = {0};
    world_flags[0] = 1;

    CollisionResults collisions = {
        .pairs = NULL,
        .pair_count = 0,
        .world_flags = world_flags,
        .penetration = NULL,
        .normals = NULL
    };

    float rewards_no_collision[10], rewards_collision[10];

    reward_reset(sys, 0);
    reward_compute_hover(sys, states, NULL, NULL, NULL, rewards_no_collision, 1);

    reward_reset(sys, 0);
    reward_compute_hover(sys, states, NULL, NULL, &collisions, rewards_collision, 1);

    float penalty = rewards_no_collision[0] - rewards_collision[0];
    ASSERT_FLOAT_NEAR(penalty, 15.0f, 0.1f);

    arena_destroy(arena);
    return 0;
}

TEST(collision_drone_penalty) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_HOVER);
    cfg.collision_penalty = 10.0f;
    cfg.drone_collision_penalty = 5.0f;
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    uint32_t pairs[2] = {0, 1};
    CollisionResults collisions = {
        .pairs = pairs,
        .pair_count = 1,
        .world_flags = NULL,
        .penetration = NULL,
        .normals = NULL
    };

    float rewards[10];
    reward_compute_hover(sys, states, NULL, NULL, &collisions, rewards, 2);

    ASSERT_LT(rewards[0], sys->config.alive_bonus);
    ASSERT_LT(rewards[1], sys->config.alive_bonus);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 9: Reward Clipping Tests
 * ============================================================================ */

TEST(reward_clipping_min) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_HOVER);
    cfg.distance_scale = 1000.0f;
    cfg.reward_min = -10.0f;
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    states->pos_x[0] = 100.0f;
    reward_set_target(sys, 0, VEC3(0, 0, 0), VEC3_ZERO, 0.5f);

    float rewards[10];
    reward_compute_hover(sys, states, NULL, NULL, NULL, rewards, 1);

    ASSERT_GE(rewards[0], -10.0f);

    arena_destroy(arena);
    return 0;
}

TEST(reward_clipping_max) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_HOVER);
    cfg.reach_bonus = 1000.0f;
    cfg.reward_max = 50.0f;
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    reward_set_target(sys, 0, VEC3(0, 0, 0), VEC3_ZERO, 10.0f);

    float rewards[10];
    reward_compute_hover(sys, states, NULL, NULL, NULL, rewards, 1);

    ASSERT_LE(rewards[0], 50.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 10: Termination Condition Tests
 * ============================================================================ */

TEST(termination_collision) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);
    RewardSystem* sys = reward_create(arena, NULL, 10, 0);

    uint8_t world_flags[10] = {0};
    world_flags[0] = 1;
    CollisionResults collisions = {
        .pairs = NULL, .pair_count = 0,
        .world_flags = world_flags
    };

    TerminationFlags flags;
    uint8_t done[10], truncated[10], success[10], collision[10], oob[10], timeout[10];
    flags.done = done; flags.truncated = truncated; flags.success = success;
    flags.collision = collision; flags.out_of_bounds = oob; flags.timeout = timeout;
    flags.capacity = 10;

    reward_compute_terminations(sys, states, &collisions,
                                VEC3(-100, -100, -100), VEC3(100, 100, 100),
                                1000, &flags, 10);

    ASSERT_EQ(flags.done[0], 1);
    ASSERT_EQ(flags.collision[0], 1);
    ASSERT_EQ(flags.done[1], 0);

    arena_destroy(arena);
    return 0;
}

TEST(termination_out_of_bounds) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);
    RewardSystem* sys = reward_create(arena, NULL, 10, 0);

    states->pos_x[0] = 150.0f;

    TerminationFlags flags;
    uint8_t done[10], truncated[10], success[10], collision[10], oob[10], timeout[10];
    flags.done = done; flags.truncated = truncated; flags.success = success;
    flags.collision = collision; flags.out_of_bounds = oob; flags.timeout = timeout;

    reward_compute_terminations(sys, states, NULL,
                                VEC3(-100, -100, -100), VEC3(100, 100, 100),
                                1000, &flags, 10);

    ASSERT_EQ(flags.done[0], 1);
    ASSERT_EQ(flags.out_of_bounds[0], 1);

    arena_destroy(arena);
    return 0;
}

TEST(termination_timeout) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);
    RewardSystem* sys = reward_create(arena, NULL, 10, 0);

    sys->episode_length[0] = 1000;

    TerminationFlags flags;
    uint8_t done[10], truncated[10], success[10], collision[10], oob[10], timeout[10];
    flags.done = done; flags.truncated = truncated; flags.success = success;
    flags.collision = collision; flags.out_of_bounds = oob; flags.timeout = timeout;

    reward_compute_terminations(sys, states, NULL,
                                VEC3(-100, -100, -100), VEC3(100, 100, 100),
                                1000, &flags, 10);

    ASSERT_EQ(flags.done[0], 1);
    ASSERT_EQ(flags.timeout[0], 1);
    ASSERT_EQ(flags.truncated[0], 1);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 11: Episode Statistics Tests
 * ============================================================================ */

TEST(episode_stats) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);
    RewardSystem* sys = reward_create(arena, NULL, 10, 0);

    float rewards[10];
    for (int step = 0; step < 10; step++) {
        reward_compute_hover(sys, states, NULL, NULL, NULL, rewards, 1);
    }

    EpisodeStats stats = reward_get_episode_stats(sys, 0);

    ASSERT_EQ(stats.episode_length, 10);
    ASSERT_TRUE(stats.episode_return != 0.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 12: Task Type Tests
 * ============================================================================ */

TEST(task_type_names) {
    ASSERT_TRUE(strcmp(task_type_name(TASK_HOVER), "HOVER") == 0);
    ASSERT_TRUE(strcmp(task_type_name(TASK_RACE), "RACE") == 0);
    ASSERT_TRUE(strcmp(task_type_name(TASK_TRACK), "TRACK") == 0);
    ASSERT_TRUE(strcmp(task_type_name(TASK_LAND), "LAND") == 0);
    ASSERT_TRUE(strcmp(task_type_name(TASK_FORMATION), "FORMATION") == 0);
    return 0;
}

TEST(config_defaults) {
    RewardConfig hover_cfg = reward_config_default(TASK_HOVER);
    ASSERT_EQ(hover_cfg.task_type, TASK_HOVER);
    ASSERT_GT(hover_cfg.alive_bonus, 0.0f);

    RewardConfig race_cfg = reward_config_default(TASK_RACE);
    ASSERT_GT(race_cfg.gate_pass_bonus, 0.0f);

    return 0;
}

/* ============================================================================
 * Section 13: Alignment Tests
 * ============================================================================ */

TEST(alignment) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    RewardSystem* sys = reward_create(arena, NULL, 256, 10);

    ASSERT_TRUE(((uintptr_t)(sys->targets->target_x) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(sys->targets->target_y) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(sys->targets->target_z) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(sys->prev_distance) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(sys->episode_return) & (32 - 1)) == 0);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 14: Edge Case Tests
 * ============================================================================ */

TEST(edge_zero_count) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    RewardSystem* sys = reward_create(arena, NULL, 10, 0);

    float rewards[10];
    reward_compute(sys, states, NULL, NULL, NULL, rewards, 0);

    arena_destroy(arena);
    return 0;
}

TEST(edge_boundary_indices) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 100);
    rigid_body_state_zero(states);
    RewardSystem* sys = reward_create(arena, NULL, 100, 0);

    reward_set_target(sys, 0, VEC3(1, 2, 3), VEC3_ZERO, 0.5f);
    reward_set_target(sys, 99, VEC3(4, 5, 6), VEC3_ZERO, 0.5f);

    ASSERT_FLOAT_NEAR(sys->targets->target_x[0], 1.0f, EPSILON);
    ASSERT_FLOAT_NEAR(sys->targets->target_x[99], 4.0f, EPSILON);

    float rewards[100];
    reward_compute_hover(sys, states, NULL, NULL, NULL, rewards, 100);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 15: Track Task Tests
 * ============================================================================ */

TEST(track_velocity_matching) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_TRACK);
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    reward_set_target(sys, 0, VEC3(0, 0, 5), VEC3(1, 0, 0), 1.0f);
    states->pos_z[0] = 5.0f;
    states->vel_x[0] = 1.0f;

    float rewards_match[10];
    reward_compute_track(sys, states, NULL, NULL, NULL, rewards_match, 1);

    reward_reset(sys, 0);
    states->vel_x[0] = 5.0f;
    float rewards_mismatch[10];
    reward_compute_track(sys, states, NULL, NULL, NULL, rewards_mismatch, 1);

    ASSERT_GT(rewards_match[0], rewards_mismatch[0]);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 16: Land Task Tests
 * ============================================================================ */

TEST(land_soft_landing) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_LAND);
    cfg.reach_radius = 1.0f;
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    reward_set_target(sys, 0, VEC3(0, 0, 0), VEC3_ZERO, 1.0f);

    states->pos_z[0] = 0.5f;
    states->vel_z[0] = -0.1f;
    float rewards_slow[10];
    reward_compute_land(sys, states, NULL, NULL, NULL, rewards_slow, 1);

    reward_reset(sys, 0);
    states->vel_z[0] = -5.0f;
    float rewards_fast[10];
    reward_compute_land(sys, states, NULL, NULL, NULL, rewards_fast, 1);

    ASSERT_GT(rewards_slow[0], rewards_fast[0]);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 17: Formation Task Tests
 * ============================================================================ */

TEST(formation_relative_position) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_FORMATION);
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    states->pos_x[0] = 0.0f;
    reward_set_target(sys, 1, VEC3(2, 0, 0), VEC3_ZERO, 0.5f);

    states->pos_x[1] = 2.0f;
    float rewards_correct[10];
    reward_compute_formation(sys, states, NULL, NULL, NULL, rewards_correct, 2);

    reward_reset(sys, 1);
    states->pos_x[1] = 5.0f;
    float rewards_wrong[10];
    reward_compute_formation(sys, states, NULL, NULL, NULL, rewards_wrong, 2);

    ASSERT_GT(rewards_correct[1], rewards_wrong[1]);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 18: AI-108 Regression Tests — SIMD Boundary Consistency
 * Verify hover reward is identical for all drones regardless of count
 * (was broken by incomplete SIMD path on AVX2).
 * ============================================================================ */

/* Helper: set up identical states for N drones, compute hover, verify all equal */
static int hover_consistency_for_count(uint32_t count) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);
    uint32_t cap = count > 0 ? count : 1;
    RigidBodyStateSOA* states = rigid_body_state_create(arena, cap);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_HOVER);
    RewardSystem* sys = reward_create(arena, &cfg, cap, 0);

    /* Set identical state for all drones: 5m from target, upright */
    for (uint32_t i = 0; i < count; i++) {
        states->pos_x[i] = 5.0f;
        states->pos_y[i] = 0.0f;
        states->pos_z[i] = 5.0f;
        states->quat_w[i] = 1.0f;
        states->quat_x[i] = 0.0f;
        states->quat_y[i] = 0.0f;
        states->quat_z[i] = 0.0f;
        reward_set_target(sys, i, VEC3(0, 0, 5), VEC3_ZERO, 0.5f);
    }

    float* rewards = (float*)arena_alloc(arena, count * sizeof(float));
    if (count > 0) {
        reward_compute_hover(sys, states, NULL, NULL, NULL, rewards, count);

        /* All drones should get identical reward */
        for (uint32_t i = 1; i < count; i++) {
            ASSERT_FLOAT_NEAR(rewards[i], rewards[0], 1e-5f);
        }
    }

    arena_destroy(arena);
    return 0;
}

TEST(hover_consistency_count_1)   { return hover_consistency_for_count(1); }
TEST(hover_consistency_count_7)   { return hover_consistency_for_count(7); }
TEST(hover_consistency_count_8)   { return hover_consistency_for_count(8); }
TEST(hover_consistency_count_9)   { return hover_consistency_for_count(9); }
TEST(hover_consistency_count_15)  { return hover_consistency_for_count(15); }
TEST(hover_consistency_count_16)  { return hover_consistency_for_count(16); }
TEST(hover_consistency_count_17)  { return hover_consistency_for_count(17); }
TEST(hover_consistency_count_100) { return hover_consistency_for_count(100); }

TEST(hover_energy_penalty_applied) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_HOVER);
    cfg.energy_scale = 1.0f;
    cfg.distance_scale = 0.0f;
    cfg.uprightness_scale = 0.0f;
    cfg.delta_distance_scale = 0.0f;
    cfg.reach_bonus = 0.0f;
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    /* Put drone far from target so reach doesn't trigger */
    states->pos_x[0] = 100.0f;
    reward_set_target(sys, 0, VEC3(0, 0, 0), VEC3_ZERO, 0.5f);

    /* Actions with high values → high energy */
    float actions[40] = {0};
    actions[0] = 1.0f; actions[1] = 1.0f; actions[2] = 1.0f; actions[3] = 1.0f;

    float rewards_no_action[10], rewards_with_action[10];
    reward_compute_hover(sys, states, NULL, NULL, NULL, rewards_no_action, 1);
    reward_reset(sys, 0);
    reward_compute_hover(sys, states, NULL, actions, NULL, rewards_with_action, 1);

    ASSERT_LT(rewards_with_action[0], rewards_no_action[0]);

    arena_destroy(arena);
    return 0;
}

TEST(hover_jerk_penalty_applied) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_HOVER);
    cfg.jerk_scale = 1.0f;
    cfg.energy_scale = 0.0f;
    cfg.distance_scale = 0.0f;
    cfg.uprightness_scale = 0.0f;
    cfg.delta_distance_scale = 0.0f;
    cfg.reach_bonus = 0.0f;
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    states->pos_x[0] = 100.0f;
    reward_set_target(sys, 0, VEC3(0, 0, 0), VEC3_ZERO, 0.5f);

    /* Step 1: set prev_actions by computing with some actions */
    float actions1[40] = {0};
    actions1[0] = 0.0f; actions1[1] = 0.0f; actions1[2] = 0.0f; actions1[3] = 0.0f;
    float rewards_step1[10];
    reward_compute(sys, states, NULL, actions1, NULL, rewards_step1, 1);

    /* Step 2: change actions drastically → jerk penalty should apply */
    float actions2[40] = {0};
    actions2[0] = 1.0f; actions2[1] = 1.0f; actions2[2] = 1.0f; actions2[3] = 1.0f;
    float rewards_step2[10];
    reward_compute(sys, states, NULL, actions2, NULL, rewards_step2, 1);

    /* Step 3: same actions as step 2 → no jerk */
    float rewards_step3[10];
    reward_compute(sys, states, NULL, actions2, NULL, rewards_step3, 1);

    /* Step 2 should have more penalty (jerk) than step 3 (no jerk) */
    ASSERT_LT(rewards_step2[0], rewards_step3[0]);

    arena_destroy(arena);
    return 0;
}

TEST(hover_distance_exp_nonlinear) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_HOVER);
    cfg.distance_scale = 1.0f;
    cfg.distance_exp = 2.0f;  /* Quadratic distance */
    cfg.alive_bonus = 0.0f;
    cfg.uprightness_scale = 0.0f;
    cfg.delta_distance_scale = 0.0f;
    cfg.reach_bonus = 0.0f;
    cfg.energy_scale = 0.0f;
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    reward_set_target(sys, 0, VEC3(0, 0, 0), VEC3_ZERO, 0.5f);
    states->pos_x[0] = 3.0f;  /* dist = 3.0 */

    float rewards[10];
    reward_compute_hover(sys, states, NULL, NULL, NULL, rewards, 1);

    /* With exp=2: penalty = 1.0 * 3^2 = 9.0, so reward = -9.0 (clipped) */
    ASSERT_FLOAT_NEAR(rewards[0], -9.0f, 0.1f);

    arena_destroy(arena);
    return 0;
}

TEST(hover_episode_return_all_drones) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    uint32_t count = 16;
    RigidBodyStateSOA* states = rigid_body_state_create(arena, count);
    rigid_body_state_zero(states);
    RewardConfig cfg = reward_config_default(TASK_HOVER);
    RewardSystem* sys = reward_create(arena, &cfg, count, 0);

    for (uint32_t i = 0; i < count; i++) {
        reward_set_target(sys, i, VEC3(0, 0, 5), VEC3_ZERO, 0.5f);
        states->pos_z[i] = 5.0f;
    }

    float* rewards = (float*)arena_alloc(arena, count * sizeof(float));

    /* 5 steps */
    for (int step = 0; step < 5; step++) {
        reward_compute_hover(sys, states, NULL, NULL, NULL, rewards, count);
    }

    /* All drones should have episode_return accumulated and episode_length = 5 */
    for (uint32_t i = 0; i < count; i++) {
        ASSERT_EQ(sys->episode_length[i], 5);
        ASSERT_TRUE(sys->episode_return[i] != 0.0f);
        /* All identical states → identical returns */
        ASSERT_FLOAT_NEAR(sys->episode_return[i], sys->episode_return[0], 1e-4f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(hover_best_distance_all_drones) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    uint32_t count = 16;
    RigidBodyStateSOA* states = rigid_body_state_create(arena, count);
    rigid_body_state_zero(states);
    RewardConfig cfg = reward_config_default(TASK_HOVER);
    RewardSystem* sys = reward_create(arena, &cfg, count, 0);

    for (uint32_t i = 0; i < count; i++) {
        reward_set_target(sys, i, VEC3(0, 0, 5), VEC3_ZERO, 0.5f);
        states->pos_x[i] = 10.0f;
        states->pos_z[i] = 5.0f;
    }

    float* rewards = (float*)arena_alloc(arena, count * sizeof(float));
    reward_compute_hover(sys, states, NULL, NULL, NULL, rewards, count);

    /* Move drones closer */
    for (uint32_t i = 0; i < count; i++) {
        states->pos_x[i] = 3.0f;
    }
    reward_compute_hover(sys, states, NULL, NULL, NULL, rewards, count);

    /* best_distance should be updated for ALL drones */
    for (uint32_t i = 0; i < count; i++) {
        ASSERT_FLOAT_NEAR(sys->best_distance[i], 3.0f, 0.1f);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 19: AI-106 Tests — compute_collision_penalty Helper
 * ============================================================================ */

TEST(collision_helper_no_collisions) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_HOVER);
    cfg.collision_penalty = 10.0f;
    cfg.world_collision_penalty = 5.0f;
    cfg.drone_collision_penalty = 3.0f;
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    float rewards_no_collision[10];
    reward_compute_hover(sys, states, NULL, NULL, NULL, rewards_no_collision, 1);

    /* No collisions → reward = alive_bonus + uprightness (no penalties) */
    ASSERT_GE(rewards_no_collision[0], cfg.alive_bonus - 0.1f);

    arena_destroy(arena);
    return 0;
}

TEST(collision_helper_world_only) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_HOVER);
    cfg.collision_penalty = 10.0f;
    cfg.world_collision_penalty = 5.0f;
    cfg.drone_collision_penalty = 3.0f;
    cfg.distance_scale = 0.0f;
    cfg.uprightness_scale = 0.0f;
    cfg.delta_distance_scale = 0.0f;
    cfg.reach_bonus = 0.0f;
    cfg.energy_scale = 0.0f;
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    states->pos_x[0] = 100.0f;
    reward_set_target(sys, 0, VEC3(0, 0, 0), VEC3_ZERO, 0.5f);

    uint8_t world_flags[10] = {0};
    world_flags[0] = 1;
    CollisionResults collisions = {
        .pairs = NULL, .pair_count = 0,
        .world_flags = world_flags
    };

    float rewards_base[10], rewards_coll[10];
    reward_compute_hover(sys, states, NULL, NULL, NULL, rewards_base, 1);
    reward_reset(sys, 0);
    reward_compute_hover(sys, states, NULL, NULL, &collisions, rewards_coll, 1);

    float penalty = rewards_base[0] - rewards_coll[0];
    /* collision_penalty + world_collision_penalty = 10 + 5 = 15 */
    ASSERT_FLOAT_NEAR(penalty, 15.0f, 0.1f);

    arena_destroy(arena);
    return 0;
}

TEST(collision_helper_drone_only) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_HOVER);
    cfg.collision_penalty = 10.0f;
    cfg.drone_collision_penalty = 3.0f;
    cfg.world_collision_penalty = 5.0f;
    cfg.distance_scale = 0.0f;
    cfg.uprightness_scale = 0.0f;
    cfg.delta_distance_scale = 0.0f;
    cfg.reach_bonus = 0.0f;
    cfg.energy_scale = 0.0f;
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    for (int i = 0; i < 2; i++) {
        states->pos_x[i] = 100.0f;
        reward_set_target(sys, (uint32_t)i, VEC3(0, 0, 0), VEC3_ZERO, 0.5f);
    }

    uint32_t pairs[2] = {0, 1};
    CollisionResults collisions = {
        .pairs = pairs, .pair_count = 1,
        .world_flags = NULL
    };

    float rewards_base[10], rewards_coll[10];
    reward_compute_hover(sys, states, NULL, NULL, NULL, rewards_base, 2);
    reward_reset(sys, 0);
    reward_reset(sys, 1);
    reward_compute_hover(sys, states, NULL, NULL, &collisions, rewards_coll, 2);

    float penalty = rewards_base[0] - rewards_coll[0];
    /* collision_penalty + drone_collision_penalty * 1.0 = 10 + 3 = 13 */
    ASSERT_FLOAT_NEAR(penalty, 13.0f, 0.1f);

    arena_destroy(arena);
    return 0;
}

TEST(collision_helper_both) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_HOVER);
    cfg.collision_penalty = 10.0f;
    cfg.world_collision_penalty = 5.0f;
    cfg.drone_collision_penalty = 3.0f;
    cfg.distance_scale = 0.0f;
    cfg.uprightness_scale = 0.0f;
    cfg.delta_distance_scale = 0.0f;
    cfg.reach_bonus = 0.0f;
    cfg.energy_scale = 0.0f;
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    for (int i = 0; i < 2; i++) {
        states->pos_x[i] = 100.0f;
        reward_set_target(sys, (uint32_t)i, VEC3(0, 0, 0), VEC3_ZERO, 0.5f);
    }

    uint8_t world_flags[10] = {0};
    world_flags[0] = 1;
    uint32_t pairs[2] = {0, 1};
    CollisionResults collisions = {
        .pairs = pairs, .pair_count = 1,
        .world_flags = world_flags
    };

    float rewards_base[10], rewards_coll[10];
    reward_compute_hover(sys, states, NULL, NULL, NULL, rewards_base, 1);
    reward_reset(sys, 0);
    reward_compute_hover(sys, states, NULL, NULL, &collisions, rewards_coll, 1);

    float penalty = rewards_base[0] - rewards_coll[0];
    /* world: 10+5=15, drone: 10+3=13, total: 28 */
    ASSERT_FLOAT_NEAR(penalty, 28.0f, 0.1f);

    arena_destroy(arena);
    return 0;
}

TEST(collision_helper_formation_multiplier) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_FORMATION);
    cfg.collision_penalty = 10.0f;
    cfg.drone_collision_penalty = 3.0f;
    cfg.world_collision_penalty = 0.0f;
    cfg.distance_scale = 0.0f;
    cfg.uprightness_scale = 0.0f;
    cfg.formation_position_scale = 0.0f;
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    /* Set leader target */
    reward_set_target(sys, 0, VEC3(0, 0, 0), VEC3_ZERO, 0.5f);
    reward_set_target(sys, 1, VEC3(2, 0, 0), VEC3_ZERO, 0.5f);
    states->pos_x[1] = 2.0f;

    uint32_t pairs[2] = {0, 1};
    CollisionResults collisions = {
        .pairs = pairs, .pair_count = 1,
        .world_flags = NULL
    };

    float rewards_base[10], rewards_coll[10];
    reward_compute_formation(sys, states, NULL, NULL, NULL, rewards_base, 2);
    reward_reset(sys, 0);
    reward_reset(sys, 1);
    reward_compute_formation(sys, states, NULL, NULL, &collisions, rewards_coll, 2);

    float penalty = rewards_base[0] - rewards_coll[0];
    /* collision_penalty + drone_collision_penalty * 2.0 = 10 + 6 = 16 */
    ASSERT_FLOAT_NEAR(penalty, 16.0f, 0.1f);

    arena_destroy(arena);
    return 0;
}

TEST(collision_helper_null_collisions) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_HOVER);
    cfg.collision_penalty = 100.0f;
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    float rewards[10];
    /* NULL collisions should not crash and should not apply penalty */
    reward_compute_hover(sys, states, NULL, NULL, NULL, rewards, 1);
    float baseline = rewards[0];

    reward_reset(sys, 0);
    CollisionResults empty = { .pairs = NULL, .pair_count = 0, .world_flags = NULL };
    reward_compute_hover(sys, states, NULL, NULL, &empty, rewards, 1);

    ASSERT_FLOAT_NEAR(rewards[0], baseline, 0.01f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 20: Full Task Composition Tests
 * ============================================================================ */

TEST(hover_full_composition) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_HOVER);
    cfg.alive_bonus = 1.0f;
    cfg.distance_scale = 0.5f;
    cfg.distance_exp = 1.0f;
    cfg.uprightness_scale = 0.2f;
    cfg.energy_scale = 0.0f;
    cfg.delta_distance_scale = 0.0f;
    cfg.reach_bonus = 0.0f;
    cfg.collision_penalty = 0.0f;
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    reward_set_target(sys, 0, VEC3(0, 0, 5), VEC3_ZERO, 0.5f);
    states->pos_x[0] = 3.0f;
    states->pos_y[0] = 4.0f;
    states->pos_z[0] = 5.0f;
    /* Distance = 5.0, upright (quat_w=1 → up_z=1.0) */

    float rewards[10];
    reward_compute_hover(sys, states, NULL, NULL, NULL, rewards, 1);

    /* Expected: alive(1.0) - dist_scale*dist(0.5*5=2.5) + upright(0.2*1.0=0.2) = -1.3 */
    ASSERT_FLOAT_NEAR(rewards[0], -1.3f, 0.1f);

    arena_destroy(arena);
    return 0;
}

TEST(track_full_composition) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_TRACK);
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    reward_set_target(sys, 0, VEC3(10, 0, 5), VEC3(1, 0, 0), 1.0f);
    states->pos_x[0] = 10.0f;
    states->pos_z[0] = 5.0f;
    states->vel_x[0] = 1.0f;  /* Match target velocity */

    float rewards[10];
    reward_compute_track(sys, states, NULL, NULL, NULL, rewards, 1);

    /* At target, matching velocity → high reward */
    ASSERT_GT(rewards[0], 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(land_full_composition) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_LAND);
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    reward_set_target(sys, 0, VEC3(0, 0, 0), VEC3_ZERO, 1.0f);
    /* Drone near target, low velocity, upright */
    states->pos_z[0] = 0.3f;
    states->vel_z[0] = -0.1f;

    float rewards[10];
    reward_compute_land(sys, states, NULL, NULL, NULL, rewards, 1);

    /* Close to target, slow descent, upright → should get success bonus */
    ASSERT_GT(rewards[0], cfg.alive_bonus);

    arena_destroy(arena);
    return 0;
}

TEST(formation_full_composition) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_FORMATION);
    RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

    /* Leader at origin, follower at relative (5,0,0) */
    states->pos_x[0] = 0.0f;
    reward_set_target(sys, 0, VEC3(0, 0, 0), VEC3_ZERO, 0.5f);
    reward_set_target(sys, 1, VEC3(5, 0, 0), VEC3_ZERO, 0.5f);

    /* Follower in correct position */
    states->pos_x[1] = 5.0f;
    float rewards_correct[10];
    reward_compute_formation(sys, states, NULL, NULL, NULL, rewards_correct, 2);

    /* Follower in wrong position */
    reward_reset(sys, 0);
    reward_reset(sys, 1);
    states->pos_x[1] = 10.0f;
    float rewards_wrong[10];
    reward_compute_formation(sys, states, NULL, NULL, NULL, rewards_wrong, 2);

    /* Correct position should yield higher reward */
    ASSERT_GT(rewards_correct[1], rewards_wrong[1]);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 21: Collision Penalty Across All Task Types
 * ============================================================================ */

TEST(collision_penalty_all_tasks) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, 10);
    rigid_body_state_zero(states);

    uint8_t world_flags[10] = {0};
    world_flags[0] = 1;
    CollisionResults collisions = {
        .pairs = NULL, .pair_count = 0,
        .world_flags = world_flags
    };

    /* Test that collision penalty applies to track, land, and formation too */
    TaskType tasks[] = { TASK_TRACK, TASK_LAND, TASK_FORMATION };
    for (int t = 0; t < 3; t++) {
        RewardConfig cfg = reward_config_default(tasks[t]);
        cfg.collision_penalty = 10.0f;
        cfg.world_collision_penalty = 5.0f;
        RewardSystem* sys = reward_create(arena, &cfg, 10, 0);

        reward_set_target(sys, 0, VEC3(0, 0, 5), VEC3_ZERO, 0.5f);
        states->pos_z[0] = 5.0f;

        float rewards_no_coll[10], rewards_with_coll[10];

        if (tasks[t] == TASK_TRACK) {
            reward_compute_track(sys, states, NULL, NULL, NULL, rewards_no_coll, 1);
            reward_reset(sys, 0);
            reward_compute_track(sys, states, NULL, NULL, &collisions, rewards_with_coll, 1);
        } else if (tasks[t] == TASK_LAND) {
            reward_compute_land(sys, states, NULL, NULL, NULL, rewards_no_coll, 1);
            reward_reset(sys, 0);
            reward_compute_land(sys, states, NULL, NULL, &collisions, rewards_with_coll, 1);
        } else {
            reward_compute_formation(sys, states, NULL, NULL, NULL, rewards_no_coll, 1);
            reward_reset(sys, 0);
            reward_compute_formation(sys, states, NULL, NULL, &collisions, rewards_with_coll, 1);
        }

        ASSERT_GT(rewards_no_coll[0], rewards_with_coll[0]);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Reward System Module Tests");

    /* Allocation tests */
    RUN_TEST(allocation_basic);
    RUN_TEST(allocation_null_arena);
    RUN_TEST(allocation_zero_drones);
    RUN_TEST(allocation_large_capacity);
    RUN_TEST(allocation_memory_size);

    /* Reset tests */
    RUN_TEST(reset_single);
    RUN_TEST(reset_batch);
    RUN_TEST(reset_gate_progress);

    /* Target management tests */
    RUN_TEST(target_set_single);
    RUN_TEST(target_set_random);
    RUN_TEST(target_update);

    /* Gate management tests */
    RUN_TEST(gate_set);
    RUN_TEST(gate_crossing_detection);

    /* Distance tests */
    RUN_TEST(distance_basic);
    RUN_TEST(distance_zero);
    RUN_TEST(reached_target);

    /* Hover reward tests */
    RUN_TEST(hover_alive_bonus);
    RUN_TEST(hover_distance_penalty);
    RUN_TEST(hover_reach_bonus);
    RUN_TEST(hover_uprightness);

    /* Race reward tests */
    RUN_TEST(race_gate_bonus);

    /* Collision penalty tests */
    RUN_TEST(collision_world_penalty);
    RUN_TEST(collision_drone_penalty);

    /* Reward clipping tests */
    RUN_TEST(reward_clipping_min);
    RUN_TEST(reward_clipping_max);

    /* Termination tests */
    RUN_TEST(termination_collision);
    RUN_TEST(termination_out_of_bounds);
    RUN_TEST(termination_timeout);

    /* Episode statistics tests */
    RUN_TEST(episode_stats);

    /* Task type tests */
    RUN_TEST(task_type_names);
    RUN_TEST(config_defaults);

    /* Alignment tests */
    RUN_TEST(alignment);

    /* Edge case tests */
    RUN_TEST(edge_zero_count);
    RUN_TEST(edge_boundary_indices);

    /* Track task tests */
    RUN_TEST(track_velocity_matching);

    /* Land task tests */
    RUN_TEST(land_soft_landing);

    /* Formation task tests */
    RUN_TEST(formation_relative_position);

    /* AI-108 regression: SIMD boundary consistency */
    RUN_TEST(hover_consistency_count_1);
    RUN_TEST(hover_consistency_count_7);
    RUN_TEST(hover_consistency_count_8);
    RUN_TEST(hover_consistency_count_9);
    RUN_TEST(hover_consistency_count_15);
    RUN_TEST(hover_consistency_count_16);
    RUN_TEST(hover_consistency_count_17);
    RUN_TEST(hover_consistency_count_100);
    RUN_TEST(hover_energy_penalty_applied);
    RUN_TEST(hover_jerk_penalty_applied);
    RUN_TEST(hover_distance_exp_nonlinear);
    RUN_TEST(hover_episode_return_all_drones);
    RUN_TEST(hover_best_distance_all_drones);

    /* AI-106: collision penalty helper */
    RUN_TEST(collision_helper_no_collisions);
    RUN_TEST(collision_helper_world_only);
    RUN_TEST(collision_helper_drone_only);
    RUN_TEST(collision_helper_both);
    RUN_TEST(collision_helper_formation_multiplier);
    RUN_TEST(collision_helper_null_collisions);

    /* Full task composition */
    RUN_TEST(hover_full_composition);
    RUN_TEST(track_full_composition);
    RUN_TEST(land_full_composition);
    RUN_TEST(formation_full_composition);
    RUN_TEST(collision_penalty_all_tasks);

    TEST_SUITE_END();
}
