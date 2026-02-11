/**
 * Reward System Module - Implementation
 *
 * SIMD-optimized reward computation for drone RL tasks.
 */

#include "../include/reward_system.h"
#include <string.h>
#include <float.h>

/* ============================================================================
 * Section 1: Task Type String Conversion
 * ============================================================================ */

static const char* TASK_TYPE_NAMES[TASK_TYPE_COUNT] = {
    "HOVER",
    "RACE",
    "TRACK",
    "LAND",
    "FORMATION",
    "EXPLORE",
    "CUSTOM"
};

const char* task_type_name(TaskType type) {
    if (type < 0 || type >= TASK_TYPE_COUNT) {
        return "UNKNOWN";
    }
    return TASK_TYPE_NAMES[type];
}

/* ============================================================================
 * Section 2: Default Configurations
 * ============================================================================ */

RewardConfig reward_config_default(TaskType task) {
    RewardConfig config = {0};

    /* Common defaults */
    config.task_type = task;
    config.distance_scale = 1.0f;
    config.distance_exp = 1.0f;
    config.reach_bonus = 10.0f;
    config.reach_radius = 0.5f;
    config.velocity_match_scale = 0.1f;
    config.max_velocity_penalty = 0.5f;
    config.uprightness_scale = 0.5f;
    config.heading_scale = 0.2f;
    config.energy_scale = 0.01f;
    config.jerk_scale = 0.05f;
    config.collision_penalty = 10.0f;
    config.world_collision_penalty = 5.0f;
    config.drone_collision_penalty = 5.0f;
    config.alive_bonus = 0.1f;
    config.success_bonus = 100.0f;
    config.reward_min = -100.0f;
    config.reward_max = 100.0f;
    config.gate_pass_bonus = 10.0f;
    config.landing_velocity_scale = 5.0f;
    config.formation_position_scale = 1.0f;
    config.exploration_coverage_scale = 0.1f;
    config.progress_scale = 0.5f;
    config.delta_distance_scale = 1.0f;

    /* Task-specific overrides */
    switch (task) {
        case TASK_HOVER:
            config.uprightness_scale = 1.0f;
            config.distance_scale = 2.0f;
            break;

        case TASK_RACE:
            config.gate_pass_bonus = 20.0f;
            config.progress_scale = 1.0f;
            config.heading_scale = 0.5f;
            break;

        case TASK_TRACK:
            config.velocity_match_scale = 0.5f;
            config.distance_scale = 1.5f;
            break;

        case TASK_LAND:
            config.landing_velocity_scale = 10.0f;
            config.uprightness_scale = 2.0f;
            break;

        case TASK_FORMATION:
            config.formation_position_scale = 2.0f;
            config.drone_collision_penalty = 10.0f;
            break;

        case TASK_EXPLORE:
            config.exploration_coverage_scale = 0.5f;
            config.alive_bonus = 0.2f;
            break;

        case TASK_CUSTOM:
        default:
            break;
    }

    return config;
}

/* ============================================================================
 * Section 3: Memory Size Calculation
 * ============================================================================ */

size_t reward_memory_size(uint32_t max_drones, uint32_t max_gates) {
    if (max_drones == 0) return 0;

    size_t size = 0;

    /* RewardSystem struct */
    size += sizeof(RewardSystem);

    /* TargetSOA: 7 aligned float arrays */
    size_t aligned_float_array = align_up_size(max_drones * sizeof(float), 32);
    size += sizeof(TargetSOA);
    size += 7 * aligned_float_array;

    /* Previous state tracking */
    size += aligned_float_array;           /* prev_distance */
    size += align_up_size(max_drones * 4 * sizeof(float), 32);  /* prev_actions */

    /* Episode tracking */
    size += aligned_float_array;           /* episode_return */
    size += align_up_size(max_drones * sizeof(uint32_t), 32);   /* episode_length */
    size += align_up_size(max_drones * sizeof(uint32_t), 32);   /* gates_passed */
    size += aligned_float_array;           /* best_distance */

    /* TerminationFlags: 6 uint8_t arrays */
    size += sizeof(TerminationFlags);
    size_t aligned_u8_array = align_up_size(max_drones * sizeof(uint8_t), 32);
    size += 6 * aligned_u8_array;

    /* GateSOA (if racing) */
    if (max_gates > 0) {
        size += sizeof(GateSOA);
        size_t gate_float_array = align_up_size(max_gates * sizeof(float), 32);
        size += 7 * gate_float_array;  /* center_xyz, normal_xyz, radius */
        size += align_up_size(max_drones * max_gates * sizeof(uint8_t), 32);  /* passed */
        size += align_up_size(max_drones * sizeof(uint32_t), 32);  /* current_gate */
    }

    return size;
}

/* ============================================================================
 * Section 4: Lifecycle Functions
 * ============================================================================ */

RewardSystem* reward_create(Arena* arena, const RewardConfig* config,
                            uint32_t max_drones, uint32_t max_gates) {
    if (!arena || max_drones == 0) return NULL;

    /* Allocate main structure */
    RewardSystem* sys = arena_alloc_type(arena, RewardSystem);
    if (!sys) return NULL;

    memset(sys, 0, sizeof(RewardSystem));
    sys->arena = arena;
    sys->max_drones = max_drones;
    sys->max_gates = max_gates;

    /* Set configuration */
    if (config) {
        sys->config = *config;
    } else {
        sys->config = reward_config_default(TASK_HOVER);
    }

    /* Allocate TargetSOA */
    sys->targets = arena_alloc_type(arena, TargetSOA);
    if (!sys->targets) return NULL;
    memset(sys->targets, 0, sizeof(TargetSOA));
    sys->targets->capacity = max_drones;

    size_t aligned_size = align_up_size(max_drones * sizeof(float), 32);
    sys->targets->target_x = arena_alloc_aligned(arena, aligned_size, 32);
    sys->targets->target_y = arena_alloc_aligned(arena, aligned_size, 32);
    sys->targets->target_z = arena_alloc_aligned(arena, aligned_size, 32);
    sys->targets->target_vx = arena_alloc_aligned(arena, aligned_size, 32);
    sys->targets->target_vy = arena_alloc_aligned(arena, aligned_size, 32);
    sys->targets->target_vz = arena_alloc_aligned(arena, aligned_size, 32);
    sys->targets->target_radius = arena_alloc_aligned(arena, aligned_size, 32);

    if (!sys->targets->target_x || !sys->targets->target_y ||
        !sys->targets->target_z || !sys->targets->target_radius) {
        return NULL;
    }

    /* Initialize targets to defaults */
    memset(sys->targets->target_x, 0, aligned_size);
    memset(sys->targets->target_y, 0, aligned_size);
    memset(sys->targets->target_z, 0, aligned_size);
    memset(sys->targets->target_vx, 0, aligned_size);
    memset(sys->targets->target_vy, 0, aligned_size);
    memset(sys->targets->target_vz, 0, aligned_size);
    for (uint32_t i = 0; i < max_drones; i++) {
        sys->targets->target_radius[i] = 0.5f;  /* Default radius */
    }

    /* Allocate previous state tracking */
    sys->prev_distance = arena_alloc_aligned(arena, aligned_size, 32);
    sys->prev_actions = arena_alloc_aligned(arena,
        align_up_size(max_drones * 4 * sizeof(float), 32), 32);
    if (!sys->prev_distance || !sys->prev_actions) return NULL;

    /* Initialize to max float to indicate "no previous" */
    for (uint32_t i = 0; i < max_drones; i++) {
        sys->prev_distance[i] = FLT_MAX;
    }
    memset(sys->prev_actions, 0, max_drones * 4 * sizeof(float));

    /* Allocate episode tracking */
    sys->episode_return = arena_alloc_aligned(arena, aligned_size, 32);
    sys->episode_length = arena_alloc_aligned(arena,
        align_up_size(max_drones * sizeof(uint32_t), 32), 32);
    sys->gates_passed = arena_alloc_aligned(arena,
        align_up_size(max_drones * sizeof(uint32_t), 32), 32);
    sys->best_distance = arena_alloc_aligned(arena, aligned_size, 32);

    if (!sys->episode_return || !sys->episode_length ||
        !sys->gates_passed || !sys->best_distance) {
        return NULL;
    }

    memset(sys->episode_return, 0, aligned_size);
    memset(sys->episode_length, 0, max_drones * sizeof(uint32_t));
    memset(sys->gates_passed, 0, max_drones * sizeof(uint32_t));
    for (uint32_t i = 0; i < max_drones; i++) {
        sys->best_distance[i] = FLT_MAX;
    }

    /* Allocate termination flags */
    sys->termination = arena_alloc_type(arena, TerminationFlags);
    if (!sys->termination) return NULL;
    memset(sys->termination, 0, sizeof(TerminationFlags));
    sys->termination->capacity = max_drones;

    size_t u8_aligned = align_up_size(max_drones * sizeof(uint8_t), 32);
    sys->termination->done = arena_alloc_aligned(arena, u8_aligned, 32);
    sys->termination->truncated = arena_alloc_aligned(arena, u8_aligned, 32);
    sys->termination->success = arena_alloc_aligned(arena, u8_aligned, 32);
    sys->termination->collision = arena_alloc_aligned(arena, u8_aligned, 32);
    sys->termination->out_of_bounds = arena_alloc_aligned(arena, u8_aligned, 32);
    sys->termination->timeout = arena_alloc_aligned(arena, u8_aligned, 32);

    if (!sys->termination->done || !sys->termination->truncated ||
        !sys->termination->success || !sys->termination->collision ||
        !sys->termination->out_of_bounds || !sys->termination->timeout) {
        return NULL;
    }

    memset(sys->termination->done, 0, u8_aligned);
    memset(sys->termination->truncated, 0, u8_aligned);
    memset(sys->termination->success, 0, u8_aligned);
    memset(sys->termination->collision, 0, u8_aligned);
    memset(sys->termination->out_of_bounds, 0, u8_aligned);
    memset(sys->termination->timeout, 0, u8_aligned);

    /* Allocate GateSOA if racing */
    if (max_gates > 0) {
        sys->gates = arena_alloc_type(arena, GateSOA);
        if (!sys->gates) return NULL;
        memset(sys->gates, 0, sizeof(GateSOA));
        sys->gates->num_gates = 0;
        sys->gates->max_drones = max_drones;

        size_t gate_aligned = align_up_size(max_gates * sizeof(float), 32);
        sys->gates->center_x = arena_alloc_aligned(arena, gate_aligned, 32);
        sys->gates->center_y = arena_alloc_aligned(arena, gate_aligned, 32);
        sys->gates->center_z = arena_alloc_aligned(arena, gate_aligned, 32);
        sys->gates->normal_x = arena_alloc_aligned(arena, gate_aligned, 32);
        sys->gates->normal_y = arena_alloc_aligned(arena, gate_aligned, 32);
        sys->gates->normal_z = arena_alloc_aligned(arena, gate_aligned, 32);
        sys->gates->radius = arena_alloc_aligned(arena, gate_aligned, 32);

        size_t passed_size = align_up_size(max_drones * max_gates * sizeof(uint8_t), 32);
        sys->gates->passed = arena_alloc_aligned(arena, passed_size, 32);
        sys->gates->current_gate = arena_alloc_aligned(arena,
            align_up_size(max_drones * sizeof(uint32_t), 32), 32);

        if (!sys->gates->center_x || !sys->gates->center_y ||
            !sys->gates->center_z || !sys->gates->passed ||
            !sys->gates->current_gate) {
            return NULL;
        }

        memset(sys->gates->center_x, 0, gate_aligned);
        memset(sys->gates->center_y, 0, gate_aligned);
        memset(sys->gates->center_z, 0, gate_aligned);
        memset(sys->gates->normal_x, 0, gate_aligned);
        memset(sys->gates->normal_y, 0, gate_aligned);
        memset(sys->gates->normal_z, 0, gate_aligned);
        memset(sys->gates->radius, 0, gate_aligned);
        memset(sys->gates->passed, 0, passed_size);
        memset(sys->gates->current_gate, 0, max_drones * sizeof(uint32_t));
    }

    return sys;
}

void reward_destroy(RewardSystem* sys) {
    /* No-op for arena allocation */
    (void)sys;
}

void reward_reset(RewardSystem* sys, uint32_t drone_idx) {
    if (!sys || drone_idx >= sys->max_drones) return;

    /* Reset episode tracking */
    sys->episode_return[drone_idx] = 0.0f;
    sys->episode_length[drone_idx] = 0;
    sys->gates_passed[drone_idx] = 0;
    sys->best_distance[drone_idx] = FLT_MAX;

    /* Reset previous state */
    sys->prev_distance[drone_idx] = FLT_MAX;
    for (int i = 0; i < 4; i++) {
        sys->prev_actions[drone_idx * 4 + i] = 0.0f;
    }

    /* Reset termination flags */
    sys->termination->done[drone_idx] = 0;
    sys->termination->truncated[drone_idx] = 0;
    sys->termination->success[drone_idx] = 0;
    sys->termination->collision[drone_idx] = 0;
    sys->termination->out_of_bounds[drone_idx] = 0;
    sys->termination->timeout[drone_idx] = 0;

    /* Reset gate progress */
    if (sys->gates && sys->gates->num_gates > 0) {
        for (uint32_t g = 0; g < sys->gates->num_gates; g++) {
            sys->gates->passed[drone_idx * sys->max_gates + g] = 0;
        }
        sys->gates->current_gate[drone_idx] = 0;
    }
}

void reward_reset_batch(RewardSystem* sys, const uint32_t* indices, uint32_t count) {
    if (!sys || !indices) return;

    for (uint32_t i = 0; i < count; i++) {
        reward_reset(sys, indices[i]);
    }
}

/* ============================================================================
 * Section 5: Target Management Functions
 * ============================================================================ */

void reward_set_target(RewardSystem* sys, uint32_t drone_idx,
                       Vec3 position, Vec3 velocity, float radius) {
    if (!sys || !sys->targets || drone_idx >= sys->max_drones) return;

    sys->targets->target_x[drone_idx] = position.x;
    sys->targets->target_y[drone_idx] = position.y;
    sys->targets->target_z[drone_idx] = position.z;
    sys->targets->target_vx[drone_idx] = velocity.x;
    sys->targets->target_vy[drone_idx] = velocity.y;
    sys->targets->target_vz[drone_idx] = velocity.z;
    sys->targets->target_radius[drone_idx] = radius;
}

void reward_set_targets_random(RewardSystem* sys, uint32_t count,
                               Vec3 bounds_min, Vec3 bounds_max, PCG32* rng) {
    if (!sys || !sys->targets || !rng) return;

    uint32_t n = count < sys->max_drones ? count : sys->max_drones;
    for (uint32_t i = 0; i < n; i++) {
        Vec3 pos = pcg32_vec3_range(rng, bounds_min, bounds_max);
        sys->targets->target_x[i] = pos.x;
        sys->targets->target_y[i] = pos.y;
        sys->targets->target_z[i] = pos.z;
        sys->targets->target_vx[i] = 0.0f;
        sys->targets->target_vy[i] = 0.0f;
        sys->targets->target_vz[i] = 0.0f;
    }
}

void reward_update_targets(RewardSystem* sys, float dt) {
    if (!sys || !sys->targets) return;

    /* Update moving targets (simple linear motion) */
    for (uint32_t i = 0; i < sys->max_drones; i++) {
        sys->targets->target_x[i] += sys->targets->target_vx[i] * dt;
        sys->targets->target_y[i] += sys->targets->target_vy[i] * dt;
        sys->targets->target_z[i] += sys->targets->target_vz[i] * dt;
    }
}

void reward_set_gates(RewardSystem* sys, const Vec3* centers,
                      const Vec3* normals, const float* radii, uint32_t num_gates) {
    if (!sys || !sys->gates || !centers || !normals || !radii) return;
    if (num_gates > sys->max_gates) num_gates = sys->max_gates;

    sys->gates->num_gates = num_gates;

    for (uint32_t g = 0; g < num_gates; g++) {
        sys->gates->center_x[g] = centers[g].x;
        sys->gates->center_y[g] = centers[g].y;
        sys->gates->center_z[g] = centers[g].z;
        sys->gates->normal_x[g] = normals[g].x;
        sys->gates->normal_y[g] = normals[g].y;
        sys->gates->normal_z[g] = normals[g].z;
        sys->gates->radius[g] = radii[g];
    }

    /* Reset all drones' gate progress */
    for (uint32_t d = 0; d < sys->max_drones; d++) {
        for (uint32_t g = 0; g < num_gates; g++) {
            sys->gates->passed[d * sys->max_gates + g] = 0;
        }
        sys->gates->current_gate[d] = 0;
    }
}

void reward_reset_gates(RewardSystem* sys, uint32_t drone_idx) {
    if (!sys || !sys->gates || drone_idx >= sys->max_drones) return;

    for (uint32_t g = 0; g < sys->gates->num_gates; g++) {
        sys->gates->passed[drone_idx * sys->max_gates + g] = 0;
    }
    sys->gates->current_gate[drone_idx] = 0;
    sys->gates_passed[drone_idx] = 0;
}

/* ============================================================================
 * Section 6: Utility Functions
 * ============================================================================ */

float reward_distance_to_target(const RewardSystem* sys,
                                const DroneStateSOA* states,
                                uint32_t drone_idx) {
    if (!sys || !states || drone_idx >= sys->max_drones) return FLT_MAX;

    float dx = states->pos_x[drone_idx] - sys->targets->target_x[drone_idx];
    float dy = states->pos_y[drone_idx] - sys->targets->target_y[drone_idx];
    float dz = states->pos_z[drone_idx] - sys->targets->target_z[drone_idx];

    return sqrtf(dx * dx + dy * dy + dz * dz);
}

bool reward_reached_target(const RewardSystem* sys,
                           const DroneStateSOA* states,
                           uint32_t drone_idx) {
    if (!sys || !states || drone_idx >= sys->max_drones) return false;

    float dist = reward_distance_to_target(sys, states, drone_idx);
    return dist <= sys->targets->target_radius[drone_idx];
}

bool reward_check_gate_crossing(const RewardSystem* sys,
                                const DroneStateSOA* states,
                                uint32_t drone_idx, uint32_t gate_idx,
                                Vec3 prev_pos) {
    if (!sys || !sys->gates || !states) return false;
    if (drone_idx >= sys->max_drones || gate_idx >= sys->gates->num_gates) return false;

    /* Get current position */
    Vec3 curr_pos = VEC3(
        states->pos_x[drone_idx],
        states->pos_y[drone_idx],
        states->pos_z[drone_idx]
    );

    /* Get gate center and normal */
    Vec3 gate_center = VEC3(
        sys->gates->center_x[gate_idx],
        sys->gates->center_y[gate_idx],
        sys->gates->center_z[gate_idx]
    );
    Vec3 gate_normal = VEC3(
        sys->gates->normal_x[gate_idx],
        sys->gates->normal_y[gate_idx],
        sys->gates->normal_z[gate_idx]
    );
    float gate_radius = sys->gates->radius[gate_idx];

    /* Line-plane intersection test */
    Vec3 dir = vec3_sub(curr_pos, prev_pos);
    float denom = vec3_dot(gate_normal, dir);

    /* If denom is near zero, line is parallel to plane */
    if (fabsf(denom) < 1e-6f) return false;

    Vec3 to_plane = vec3_sub(gate_center, prev_pos);
    float t = vec3_dot(gate_normal, to_plane) / denom;

    /* Check if intersection is between prev and curr positions */
    if (t < 0.0f || t > 1.0f) return false;

    /* Compute intersection point */
    Vec3 intersection = vec3_add(prev_pos, vec3_scale(dir, t));

    /* Check if intersection is within gate radius */
    float dist_to_center = vec3_distance(intersection, gate_center);
    if (dist_to_center > gate_radius) return false;

    /* Check crossing direction (should be positive normal direction) */
    if (denom < 0.0f) return false;  /* Wrong direction */

    return true;
}

EpisodeStats reward_get_episode_stats(const RewardSystem* sys, uint32_t drone_idx) {
    EpisodeStats stats = {0};
    if (!sys || drone_idx >= sys->max_drones) return stats;

    stats.episode_return = sys->episode_return[drone_idx];
    stats.best_distance = sys->best_distance[drone_idx];
    stats.episode_length = sys->episode_length[drone_idx];
    stats.gates_passed = sys->gates_passed[drone_idx];
    stats.success = sys->termination->success[drone_idx] != 0;

    return stats;
}

bool reward_is_done(const RewardSystem* sys, uint32_t drone_idx) {
    if (!sys || drone_idx >= sys->max_drones) return false;
    return sys->termination->done[drone_idx] != 0;
}

bool reward_is_success(const RewardSystem* sys, uint32_t drone_idx) {
    if (!sys || drone_idx >= sys->max_drones) return false;
    return sys->termination->success[drone_idx] != 0;
}

/* ============================================================================
 * Section 7: SIMD Helper Functions
 * ============================================================================ */

/* Compute uprightness from quaternion (z-component of up vector in world frame) */
FOUNDATION_INLINE float compute_uprightness(float qw, float qx, float qy, float qz) {
    /* For unit quaternion, world-frame up (body z) is:
     * up_z = 1 - 2*(qx*qx + qy*qy) */
    return 1.0f - 2.0f * (qx * qx + qy * qy);
}

/* Compute energy penalty from motor commands */
FOUNDATION_INLINE float compute_energy_penalty(const float* actions, uint32_t idx) {
    float energy = 0.0f;
    for (int i = 0; i < 4; i++) {
        float a = actions[idx * 4 + i];
        energy += a * a;
    }
    return energy;
}

/* Compute jerk penalty (action smoothness) */
FOUNDATION_INLINE float compute_jerk_penalty(const float* actions, const float* prev_actions,
                                             uint32_t idx) {
    float jerk = 0.0f;
    for (int i = 0; i < 4; i++) {
        float da = actions[idx * 4 + i] - prev_actions[idx * 4 + i];
        jerk += da * da;
    }
    return jerk;
}

/* ============================================================================
 * Section 8: Reward Computation - Hover Task
 * ============================================================================ */

void reward_compute_hover(RewardSystem* sys, const DroneStateSOA* states,
                          const DroneParamsSOA* params, const float* actions,
                          const CollisionResults* collisions,
                          float* rewards, uint32_t count) {
    if (!sys || !states || !rewards) return;

    const RewardConfig* cfg = &sys->config;

    SIMD_LOOP_START(count);

#if defined(FOUNDATION_SIMD_AVX2)
    /* AVX2 SIMD path: process 8 drones at a time */
    simd_float distance_scale = simd_set1_ps(cfg->distance_scale);
    simd_float alive_bonus = simd_set1_ps(cfg->alive_bonus);
    simd_float uprightness_scale = simd_set1_ps(cfg->uprightness_scale);
    simd_float energy_scale = simd_set1_ps(cfg->energy_scale);
    simd_float jerk_scale = simd_set1_ps(cfg->jerk_scale);
    simd_float reward_min = simd_set1_ps(cfg->reward_min);
    simd_float reward_max = simd_set1_ps(cfg->reward_max);
    simd_float reach_radius = simd_set1_ps(cfg->reach_radius);
    simd_float reach_bonus = simd_set1_ps(cfg->reach_bonus);
    simd_float delta_scale = simd_set1_ps(cfg->delta_distance_scale);
    simd_float two = simd_set1_ps(2.0f);
    simd_float one = simd_set1_ps(1.0f);

    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        /* Load positions */
        simd_float px = simd_load_ps(&states->pos_x[i]);
        simd_float py = simd_load_ps(&states->pos_y[i]);
        simd_float pz = simd_load_ps(&states->pos_z[i]);

        /* Load targets */
        simd_float tx = simd_load_ps(&sys->targets->target_x[i]);
        simd_float ty = simd_load_ps(&sys->targets->target_y[i]);
        simd_float tz = simd_load_ps(&sys->targets->target_z[i]);

        /* Compute distance */
        simd_float dx = simd_sub_ps(tx, px);
        simd_float dy = simd_sub_ps(ty, py);
        simd_float dz = simd_sub_ps(tz, pz);
        simd_float dist_sq = simd_fmadd_ps(dx, dx, simd_fmadd_ps(dy, dy, simd_mul_ps(dz, dz)));
        simd_float dist = simd_sqrt_ps(dist_sq);

        /* Distance reward (negative) */
        simd_float reward = simd_sub_ps(simd_setzero_ps(), simd_mul_ps(distance_scale, dist));

        /* Add alive bonus */
        reward = simd_add_ps(reward, alive_bonus);

        /* Uprightness reward */
        simd_float qw = simd_load_ps(&states->quat_w[i]);
        simd_float qx = simd_load_ps(&states->quat_x[i]);
        simd_float qy = simd_load_ps(&states->quat_y[i]);
        simd_float qz = simd_load_ps(&states->quat_z[i]);
        simd_float qxqx = simd_mul_ps(qx, qx);
        simd_float qyqy = simd_mul_ps(qy, qy);
        simd_float up_z = simd_sub_ps(one, simd_mul_ps(two, simd_add_ps(qxqx, qyqy)));
        reward = simd_fmadd_ps(uprightness_scale, up_z, reward);

        /* Reach bonus */
        simd_float reached_mask = simd_cmp_le_ps(dist, reach_radius);
        simd_float bonus = simd_and_ps(reached_mask, reach_bonus);
        reward = simd_add_ps(reward, bonus);

        /* Load previous distance for delta reward */
        simd_float prev_dist = simd_load_ps(&sys->prev_distance[i]);
        simd_float delta = simd_sub_ps(prev_dist, dist);
        simd_float delta_reward = simd_mul_ps(delta_scale, delta);
        /* Only apply if prev_dist was valid (not FLT_MAX) */
        simd_float flt_max_vec = simd_set1_ps(FLT_MAX * 0.5f);
        simd_float valid_mask = simd_cmp_lt_ps(prev_dist, flt_max_vec);
        delta_reward = simd_and_ps(valid_mask, delta_reward);
        reward = simd_add_ps(reward, delta_reward);

        /* Clip reward */
        reward = simd_max_ps(reward, reward_min);
        reward = simd_min_ps(reward, reward_max);

        /* Store reward */
        simd_store_ps(&rewards[i], reward);

        /* Update previous distance */
        simd_store_ps(&sys->prev_distance[i], dist);
    }
#endif

    /* Scalar path for remainder and non-AVX2 platforms */
    SIMD_LOOP_REMAINDER(i, count) {
        float reward = cfg->alive_bonus;

        /* Distance penalty */
        float dist = reward_distance_to_target(sys, states, i);
        reward -= cfg->distance_scale * powf(dist, cfg->distance_exp);

        /* Delta distance reward */
        if (sys->prev_distance[i] < FLT_MAX * 0.5f) {
            float delta = sys->prev_distance[i] - dist;
            reward += cfg->delta_distance_scale * delta;
        }

        /* Uprightness reward */
        float up_z = compute_uprightness(
            states->quat_w[i], states->quat_x[i],
            states->quat_y[i], states->quat_z[i]);
        reward += cfg->uprightness_scale * up_z;

        /* Energy penalty */
        if (actions) {
            reward -= cfg->energy_scale * compute_energy_penalty(actions, i);

            /* Jerk penalty */
            if (sys->prev_actions) {
                reward -= cfg->jerk_scale * compute_jerk_penalty(actions, sys->prev_actions, i);
            }
        }

        /* Reach bonus */
        if (dist <= cfg->reach_radius) {
            reward += cfg->reach_bonus;
        }

        /* Collision penalties */
        if (collisions) {
            if (collisions->world_flags && collisions->world_flags[i]) {
                reward -= cfg->collision_penalty + cfg->world_collision_penalty;
            }
            /* Check drone-drone collisions */
            for (uint32_t p = 0; p < collisions->pair_count; p++) {
                if (collisions->pairs[p * 2] == i || collisions->pairs[p * 2 + 1] == i) {
                    reward -= cfg->collision_penalty + cfg->drone_collision_penalty;
                    break;
                }
            }
        }

        /* Update tracking */
        if (dist < sys->best_distance[i]) {
            sys->best_distance[i] = dist;
        }
        sys->prev_distance[i] = dist;

        /* Clip and store */
        reward = clampf(reward, cfg->reward_min, cfg->reward_max);
        rewards[i] = reward;

        /* Accumulate episode return */
        sys->episode_return[i] += reward;
        sys->episode_length[i]++;
    }

    /* Update previous actions */
    if (actions && sys->prev_actions) {
        memcpy(sys->prev_actions, actions, count * 4 * sizeof(float));
    }
}

/* ============================================================================
 * Section 9: Reward Computation - Race Task
 * ============================================================================ */

void reward_compute_race(RewardSystem* sys, const DroneStateSOA* states,
                         const DroneParamsSOA* params, const float* actions,
                         const CollisionResults* collisions,
                         float* rewards, uint32_t count) {
    if (!sys || !states || !rewards || !sys->gates) return;

    const RewardConfig* cfg = &sys->config;

    for (uint32_t i = 0; i < count; i++) {
        float reward = cfg->alive_bonus;

        Vec3 curr_pos = VEC3(
            states->pos_x[i], states->pos_y[i], states->pos_z[i]);

        /* Check gate crossings */
        uint32_t current_gate = sys->gates->current_gate[i];
        if (current_gate < sys->gates->num_gates) {
            /* Get previous position from prev_distance encoding or compute from velocity */
            Vec3 prev_pos = curr_pos;
            if (sys->prev_distance[i] < FLT_MAX * 0.5f) {
                /* Estimate previous position (simple approximation) */
                float dt = 0.01f;  /* Assume ~100Hz */
                prev_pos.x = curr_pos.x - states->vel_x[i] * dt;
                prev_pos.y = curr_pos.y - states->vel_y[i] * dt;
                prev_pos.z = curr_pos.z - states->vel_z[i] * dt;
            }

            if (reward_check_gate_crossing(sys, states, i, current_gate, prev_pos)) {
                /* Gate passed! */
                sys->gates->passed[i * sys->max_gates + current_gate] = 1;
                sys->gates->current_gate[i] = current_gate + 1;
                sys->gates_passed[i]++;
                reward += cfg->gate_pass_bonus;
            }
        }

        /* Progress toward next gate */
        if (current_gate < sys->gates->num_gates) {
            Vec3 gate_center = VEC3(
                sys->gates->center_x[current_gate],
                sys->gates->center_y[current_gate],
                sys->gates->center_z[current_gate]);
            float dist = vec3_distance(curr_pos, gate_center);

            /* Distance improvement reward */
            if (sys->prev_distance[i] < FLT_MAX * 0.5f) {
                float delta = sys->prev_distance[i] - dist;
                reward += cfg->progress_scale * delta;
            }
            sys->prev_distance[i] = dist;

            /* Heading alignment with gate normal */
            if (fabsf(states->vel_x[i]) + fabsf(states->vel_y[i]) +
                fabsf(states->vel_z[i]) > 0.1f) {
                Vec3 vel = VEC3(states->vel_x[i], states->vel_y[i], states->vel_z[i]);
                vel = vec3_normalize(vel);
                Vec3 gate_normal = VEC3(
                    sys->gates->normal_x[current_gate],
                    sys->gates->normal_y[current_gate],
                    sys->gates->normal_z[current_gate]);
                float alignment = vec3_dot(vel, gate_normal);
                reward += cfg->heading_scale * alignment;
            }
        }

        /* Success bonus for completing all gates */
        if (sys->gates->current_gate[i] >= sys->gates->num_gates &&
            !sys->termination->success[i]) {
            reward += cfg->success_bonus;
            sys->termination->success[i] = 1;
        }

        /* Energy penalty */
        if (actions) {
            reward -= cfg->energy_scale * compute_energy_penalty(actions, i);
            if (sys->prev_actions) {
                reward -= cfg->jerk_scale * compute_jerk_penalty(actions, sys->prev_actions, i);
            }
        }

        /* Collision penalties */
        if (collisions) {
            if (collisions->world_flags && collisions->world_flags[i]) {
                reward -= cfg->collision_penalty + cfg->world_collision_penalty;
            }
            for (uint32_t p = 0; p < collisions->pair_count; p++) {
                if (collisions->pairs[p * 2] == i || collisions->pairs[p * 2 + 1] == i) {
                    reward -= cfg->collision_penalty + cfg->drone_collision_penalty;
                    break;
                }
            }
        }

        /* Clip and store */
        reward = clampf(reward, cfg->reward_min, cfg->reward_max);
        rewards[i] = reward;

        sys->episode_return[i] += reward;
        sys->episode_length[i]++;
    }

    /* Update previous actions */
    if (actions && sys->prev_actions) {
        memcpy(sys->prev_actions, actions, count * 4 * sizeof(float));
    }
}

/* ============================================================================
 * Section 10: Reward Computation - Track Task
 * ============================================================================ */

void reward_compute_track(RewardSystem* sys, const DroneStateSOA* states,
                          const DroneParamsSOA* params, const float* actions,
                          const CollisionResults* collisions,
                          float* rewards, uint32_t count) {
    if (!sys || !states || !rewards) return;

    const RewardConfig* cfg = &sys->config;

    for (uint32_t i = 0; i < count; i++) {
        float reward = cfg->alive_bonus;

        /* Distance penalty */
        float dist = reward_distance_to_target(sys, states, i);
        reward -= cfg->distance_scale * dist;

        /* Velocity matching */
        float dvx = states->vel_x[i] - sys->targets->target_vx[i];
        float dvy = states->vel_y[i] - sys->targets->target_vy[i];
        float dvz = states->vel_z[i] - sys->targets->target_vz[i];
        float vel_error = sqrtf(dvx * dvx + dvy * dvy + dvz * dvz);
        reward -= cfg->velocity_match_scale * vel_error;

        /* Delta distance */
        if (sys->prev_distance[i] < FLT_MAX * 0.5f) {
            float delta = sys->prev_distance[i] - dist;
            reward += cfg->delta_distance_scale * delta;
        }
        sys->prev_distance[i] = dist;

        /* Uprightness */
        float up_z = compute_uprightness(
            states->quat_w[i], states->quat_x[i],
            states->quat_y[i], states->quat_z[i]);
        reward += cfg->uprightness_scale * up_z;

        /* Energy penalty */
        if (actions) {
            reward -= cfg->energy_scale * compute_energy_penalty(actions, i);
        }

        /* Collision penalties */
        if (collisions && collisions->world_flags && collisions->world_flags[i]) {
            reward -= cfg->collision_penalty + cfg->world_collision_penalty;
        }

        /* Clip and store */
        reward = clampf(reward, cfg->reward_min, cfg->reward_max);
        rewards[i] = reward;

        sys->episode_return[i] += reward;
        sys->episode_length[i]++;
    }

    if (actions && sys->prev_actions) {
        memcpy(sys->prev_actions, actions, count * 4 * sizeof(float));
    }
}

/* ============================================================================
 * Section 11: Reward Computation - Land Task
 * ============================================================================ */

void reward_compute_land(RewardSystem* sys, const DroneStateSOA* states,
                         const DroneParamsSOA* params, const float* actions,
                         const CollisionResults* collisions,
                         float* rewards, uint32_t count) {
    if (!sys || !states || !rewards) return;

    const RewardConfig* cfg = &sys->config;

    for (uint32_t i = 0; i < count; i++) {
        float reward = cfg->alive_bonus;

        /* Distance to landing target */
        float dist = reward_distance_to_target(sys, states, i);
        reward -= cfg->distance_scale * dist;

        /* Landing velocity penalty (soft landing desired) */
        float vel_sq = states->vel_x[i] * states->vel_x[i] +
                       states->vel_y[i] * states->vel_y[i] +
                       states->vel_z[i] * states->vel_z[i];
        float vel = sqrtf(vel_sq);

        /* At target: reward low velocity */
        if (dist < cfg->reach_radius) {
            float vel_penalty = cfg->landing_velocity_scale * vel;
            reward -= vel_penalty;

            /* Bonus for successful soft landing */
            if (vel < 0.5f) {  /* Less than 0.5 m/s */
                reward += cfg->success_bonus;
                sys->termination->success[i] = 1;
            }
        }

        /* Strong uprightness requirement for landing */
        float up_z = compute_uprightness(
            states->quat_w[i], states->quat_x[i],
            states->quat_y[i], states->quat_z[i]);
        reward += cfg->uprightness_scale * 2.0f * up_z;  /* Double weight for landing */

        /* Penalize tilting during descent */
        if (up_z < 0.9f) {
            reward -= 1.0f * (0.9f - up_z);
        }

        /* Delta distance */
        if (sys->prev_distance[i] < FLT_MAX * 0.5f) {
            float delta = sys->prev_distance[i] - dist;
            reward += cfg->delta_distance_scale * delta;
        }
        sys->prev_distance[i] = dist;

        /* Clip and store */
        reward = clampf(reward, cfg->reward_min, cfg->reward_max);
        rewards[i] = reward;

        sys->episode_return[i] += reward;
        sys->episode_length[i]++;
    }

    if (actions && sys->prev_actions) {
        memcpy(sys->prev_actions, actions, count * 4 * sizeof(float));
    }
}

/* ============================================================================
 * Section 12: Reward Computation - Formation Task
 * ============================================================================ */

void reward_compute_formation(RewardSystem* sys, const DroneStateSOA* states,
                              const DroneParamsSOA* params, const float* actions,
                              const CollisionResults* collisions,
                              float* rewards, uint32_t count) {
    if (!sys || !states || !rewards || count == 0) return;

    const RewardConfig* cfg = &sys->config;

    /* For formation, targets specify relative positions from a leader (drone 0) */
    float leader_x = states->pos_x[0];
    float leader_y = states->pos_y[0];
    float leader_z = states->pos_z[0];

    for (uint32_t i = 0; i < count; i++) {
        float reward = cfg->alive_bonus;

        if (i == 0) {
            /* Leader: follow its own target */
            float dist = reward_distance_to_target(sys, states, i);
            reward -= cfg->distance_scale * dist;
        } else {
            /* Followers: maintain relative position to leader */
            float target_x = leader_x + sys->targets->target_x[i];
            float target_y = leader_y + sys->targets->target_y[i];
            float target_z = leader_z + sys->targets->target_z[i];

            float dx = states->pos_x[i] - target_x;
            float dy = states->pos_y[i] - target_y;
            float dz = states->pos_z[i] - target_z;
            float dist = sqrtf(dx * dx + dy * dy + dz * dz);

            reward -= cfg->formation_position_scale * dist;

            /* Update tracking */
            if (dist < sys->best_distance[i]) {
                sys->best_distance[i] = dist;
            }
        }

        /* Uprightness for all drones */
        float up_z = compute_uprightness(
            states->quat_w[i], states->quat_x[i],
            states->quat_y[i], states->quat_z[i]);
        reward += cfg->uprightness_scale * up_z;

        /* Collision penalties - extra penalty for formation */
        if (collisions) {
            if (collisions->world_flags && collisions->world_flags[i]) {
                reward -= cfg->collision_penalty + cfg->world_collision_penalty;
            }
            for (uint32_t p = 0; p < collisions->pair_count; p++) {
                if (collisions->pairs[p * 2] == i || collisions->pairs[p * 2 + 1] == i) {
                    /* Extra penalty in formation - should maintain separation */
                    reward -= cfg->collision_penalty + cfg->drone_collision_penalty * 2.0f;
                    break;
                }
            }
        }

        /* Clip and store */
        reward = clampf(reward, cfg->reward_min, cfg->reward_max);
        rewards[i] = reward;

        sys->episode_return[i] += reward;
        sys->episode_length[i]++;
    }

    if (actions && sys->prev_actions) {
        memcpy(sys->prev_actions, actions, count * 4 * sizeof(float));
    }
}

/* ============================================================================
 * Section 13: Main Reward Dispatch Function
 * ============================================================================ */

void reward_compute(RewardSystem* sys, const DroneStateSOA* states,
                    const DroneParamsSOA* params, const float* actions,
                    const CollisionResults* collisions,
                    float* rewards, uint32_t count) {
    if (!sys || !states || !rewards || count == 0) return;

    switch (sys->config.task_type) {
        case TASK_HOVER:
            reward_compute_hover(sys, states, params, actions, collisions, rewards, count);
            break;

        case TASK_RACE:
            reward_compute_race(sys, states, params, actions, collisions, rewards, count);
            break;

        case TASK_TRACK:
            reward_compute_track(sys, states, params, actions, collisions, rewards, count);
            break;

        case TASK_LAND:
            reward_compute_land(sys, states, params, actions, collisions, rewards, count);
            break;

        case TASK_FORMATION:
            reward_compute_formation(sys, states, params, actions, collisions, rewards, count);
            break;

        case TASK_EXPLORE:
        case TASK_CUSTOM:
        default:
            /* Default to hover-like behavior */
            reward_compute_hover(sys, states, params, actions, collisions, rewards, count);
            break;
    }
}

/* ============================================================================
 * Section 14: Termination Computation
 * ============================================================================ */

void reward_compute_terminations(RewardSystem* sys, const DroneStateSOA* states,
                                 const CollisionResults* collisions,
                                 Vec3 bounds_min, Vec3 bounds_max,
                                 uint32_t max_steps, TerminationFlags* flags,
                                 uint32_t count) {
    if (!sys || !states || !flags) return;

    for (uint32_t i = 0; i < count; i++) {
        uint8_t done = 0;
        uint8_t collision_flag = 0;
        uint8_t oob = 0;
        uint8_t timeout = 0;
        uint8_t truncated = 0;

        /* Check collisions */
        if (collisions) {
            /* World collision */
            if (collisions->world_flags && collisions->world_flags[i]) {
                collision_flag = 1;
                done = 1;
            }

            /* Drone-drone collision (terminates both drones) */
            for (uint32_t p = 0; p < collisions->pair_count; p++) {
                if (collisions->pairs[p * 2] == i || collisions->pairs[p * 2 + 1] == i) {
                    collision_flag = 1;
                    done = 1;
                    break;
                }
            }
        }

        /* Check out of bounds */
        float px = states->pos_x[i];
        float py = states->pos_y[i];
        float pz = states->pos_z[i];

        if (px < bounds_min.x || px > bounds_max.x ||
            py < bounds_min.y || py > bounds_max.y ||
            pz < bounds_min.z || pz > bounds_max.z) {
            oob = 1;
            done = 1;
        }

        /* Check timeout/truncation */
        if (sys->episode_length[i] >= max_steps) {
            timeout = 1;
            truncated = 1;
            done = 1;
        }

        /* Check success (task-specific, already set during reward computation) */
        if (sys->termination->success[i]) {
            done = 1;
        }

        /* Store flags */
        flags->done[i] = done;
        flags->collision[i] = collision_flag;
        flags->out_of_bounds[i] = oob;
        flags->timeout[i] = timeout;
        flags->truncated[i] = truncated;
        flags->success[i] = sys->termination->success[i];

        /* Also update internal termination state */
        sys->termination->done[i] = done;
        sys->termination->collision[i] = collision_flag;
        sys->termination->out_of_bounds[i] = oob;
        sys->termination->timeout[i] = timeout;
        sys->termination->truncated[i] = truncated;
    }
}
