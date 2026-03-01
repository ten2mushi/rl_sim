/**
 * Drone State Module Implementation
 *
 * SoA data structures with SIMD-optimized batch operations.
 * Platform-specific create/zero/reset/copy operations are in rigid_body_state.c.
 * This file provides episode management, AoS accessors, validation, and printing.
 */

#include "../include/drone_state.h"

/* ============================================================================
 * Section 1: Lifecycle Functions
 * ============================================================================ */

AgentEpisodeData* agent_episode_create(Arena* arena, uint32_t capacity) {
    if (arena == NULL || capacity == 0) {
        return NULL;
    }

    AgentEpisodeData* episodes = arena_alloc_array(arena, AgentEpisodeData, capacity);
    if (episodes == NULL) {
        return NULL;
    }

    /* Initialize all episodes */
    for (uint32_t i = 0; i < capacity; i++) {
        agent_episode_init(episodes, i, 0, i);
    }

    return episodes;
}

/* ============================================================================
 * Section 2: Initialization Functions
 * ============================================================================ */

void platform_state_init(PlatformStateSOA* states, uint32_t index) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    FOUNDATION_ASSERT(index < states->rigid_body.capacity, "index out of bounds");

    if (states == NULL || index >= states->rigid_body.capacity) {
        return;
    }

    /* Position at origin */
    states->rigid_body.pos_x[index] = 0.0f;
    states->rigid_body.pos_y[index] = 0.0f;
    states->rigid_body.pos_z[index] = 0.0f;

    /* Zero velocity */
    states->rigid_body.vel_x[index] = 0.0f;
    states->rigid_body.vel_y[index] = 0.0f;
    states->rigid_body.vel_z[index] = 0.0f;

    /* Identity quaternion */
    states->rigid_body.quat_w[index] = 1.0f;
    states->rigid_body.quat_x[index] = 0.0f;
    states->rigid_body.quat_y[index] = 0.0f;
    states->rigid_body.quat_z[index] = 0.0f;

    /* Zero angular velocity */
    states->rigid_body.omega_x[index] = 0.0f;
    states->rigid_body.omega_y[index] = 0.0f;
    states->rigid_body.omega_z[index] = 0.0f;

    /* Zero all extension arrays (platform-specific state like RPMs) */
    for (uint32_t e = 0; e < states->extension_count; e++) {
        if (states->extension[e]) {
            states->extension[e][index] = 0.0f;
        }
    }
}

void platform_params_init(PlatformParamsSOA* params, uint32_t index) {
    FOUNDATION_ASSERT(params != NULL, "params is NULL");
    FOUNDATION_ASSERT(index < params->rigid_body.capacity, "index out of bounds");

    if (params == NULL || index >= params->rigid_body.capacity) {
        return;
    }

    /* Default: ~0.5kg small robot */
    params->rigid_body.mass[index] = 0.5f;

    /* Approximate inertia (diagonal matrix) */
    params->rigid_body.ixx[index] = 0.0025f;
    params->rigid_body.iyy[index] = 0.0025f;
    params->rigid_body.izz[index] = 0.0045f;

    /* Geometry */
    params->rigid_body.collision_radius[index] = 0.15f;

    /* Limits */
    params->rigid_body.max_vel[index] = 20.0f;       /* m/s */
    params->rigid_body.max_omega[index] = 10.0f;      /* rad/s */

    /* Environment */
    params->rigid_body.gravity[index] = 9.81f;
}

void agent_episode_init(AgentEpisodeData* episodes, uint32_t index,
                        uint32_t env_id, uint32_t agent_id) {
    FOUNDATION_ASSERT(episodes != NULL, "episodes is NULL");

    if (episodes == NULL) {
        return;
    }

    episodes[index] = (AgentEpisodeData){
        .episode_return = 0.0f,
        .best_episode_return = -1e30f,
        .episode_length = 0,
        .total_episodes = 0,
        .env_id = env_id,
        .agent_id = agent_id,
        .done = 0,
        .truncated = 0,
    };
}

/* ============================================================================
 * Section 3: Single-Agent Accessors
 * ============================================================================ */

PlatformStateAoS platform_state_get(const PlatformStateSOA* states, uint32_t index) {
    PlatformStateAoS result = {0};

    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    FOUNDATION_ASSERT(index < states->rigid_body.capacity, "index out of bounds");

    if (states == NULL || index >= states->rigid_body.capacity) {
        /* Return zeroed state with identity quaternion */
        result.orientation = QUAT_IDENTITY;
        return result;
    }

    result.position = VEC3(states->rigid_body.pos_x[index],
                           states->rigid_body.pos_y[index],
                           states->rigid_body.pos_z[index]);

    result.velocity = VEC3(states->rigid_body.vel_x[index],
                           states->rigid_body.vel_y[index],
                           states->rigid_body.vel_z[index]);

    result.orientation = QUAT(states->rigid_body.quat_w[index],
                              states->rigid_body.quat_x[index],
                              states->rigid_body.quat_y[index],
                              states->rigid_body.quat_z[index]);

    result.omega = VEC3(states->rigid_body.omega_x[index],
                        states->rigid_body.omega_y[index],
                        states->rigid_body.omega_z[index]);

    return result;
}

void platform_state_set(PlatformStateSOA* states, uint32_t index, const PlatformStateAoS* state) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    FOUNDATION_ASSERT(state != NULL, "state is NULL");
    FOUNDATION_ASSERT(index < states->rigid_body.capacity, "index out of bounds");

    if (states == NULL || state == NULL || index >= states->rigid_body.capacity) {
        return;
    }

    states->rigid_body.pos_x[index] = state->position.x;
    states->rigid_body.pos_y[index] = state->position.y;
    states->rigid_body.pos_z[index] = state->position.z;

    states->rigid_body.vel_x[index] = state->velocity.x;
    states->rigid_body.vel_y[index] = state->velocity.y;
    states->rigid_body.vel_z[index] = state->velocity.z;

    states->rigid_body.quat_w[index] = state->orientation.w;
    states->rigid_body.quat_x[index] = state->orientation.x;
    states->rigid_body.quat_y[index] = state->orientation.y;
    states->rigid_body.quat_z[index] = state->orientation.z;

    states->rigid_body.omega_x[index] = state->omega.x;
    states->rigid_body.omega_y[index] = state->omega.y;
    states->rigid_body.omega_z[index] = state->omega.z;
}

PlatformParamsAoS platform_params_get(const PlatformParamsSOA* params, uint32_t index) {
    PlatformParamsAoS result = {0};

    FOUNDATION_ASSERT(params != NULL, "params is NULL");
    FOUNDATION_ASSERT(index < params->rigid_body.capacity, "index out of bounds");

    if (params == NULL || index >= params->rigid_body.capacity) {
        return result;
    }

    result.mass = params->rigid_body.mass[index];
    result.ixx = params->rigid_body.ixx[index];
    result.iyy = params->rigid_body.iyy[index];
    result.izz = params->rigid_body.izz[index];

    result.collision_radius = params->rigid_body.collision_radius[index];

    result.max_vel = params->rigid_body.max_vel[index];
    result.max_omega = params->rigid_body.max_omega[index];

    result.gravity = params->rigid_body.gravity[index];

    return result;
}

void platform_params_set(PlatformParamsSOA* params, uint32_t index, const PlatformParamsAoS* param) {
    FOUNDATION_ASSERT(params != NULL, "params is NULL");
    FOUNDATION_ASSERT(param != NULL, "param is NULL");
    FOUNDATION_ASSERT(index < params->rigid_body.capacity, "index out of bounds");

    if (params == NULL || param == NULL || index >= params->rigid_body.capacity) {
        return;
    }

    params->rigid_body.mass[index] = param->mass;
    params->rigid_body.ixx[index] = param->ixx;
    params->rigid_body.iyy[index] = param->iyy;
    params->rigid_body.izz[index] = param->izz;

    params->rigid_body.collision_radius[index] = param->collision_radius;

    params->rigid_body.max_vel[index] = param->max_vel;
    params->rigid_body.max_omega[index] = param->max_omega;

    params->rigid_body.gravity[index] = param->gravity;
}

/* ============================================================================
 * Section 4: Utility Functions
 * ============================================================================ */

bool platform_state_validate(const PlatformStateSOA* states, uint32_t index) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    FOUNDATION_ASSERT(index < states->rigid_body.capacity, "index out of bounds");

    if (states == NULL || index >= states->rigid_body.capacity) {
        return false;
    }

    const RigidBodyStateSOA* rb = &states->rigid_body;

    /* Check for NaN or Inf in position */
    if (isnan(rb->pos_x[index]) || isnan(rb->pos_y[index]) ||
        isnan(rb->pos_z[index]) ||
        isinf(rb->pos_x[index]) || isinf(rb->pos_y[index]) ||
        isinf(rb->pos_z[index])) {
        return false;
    }

    /* Check for NaN or Inf in velocity */
    if (isnan(rb->vel_x[index]) || isnan(rb->vel_y[index]) ||
        isnan(rb->vel_z[index]) ||
        isinf(rb->vel_x[index]) || isinf(rb->vel_y[index]) ||
        isinf(rb->vel_z[index])) {
        return false;
    }

    /* Check for NaN or Inf in quaternion */
    if (isnan(rb->quat_w[index]) || isnan(rb->quat_x[index]) ||
        isnan(rb->quat_y[index]) || isnan(rb->quat_z[index]) ||
        isinf(rb->quat_w[index]) || isinf(rb->quat_x[index]) ||
        isinf(rb->quat_y[index]) || isinf(rb->quat_z[index])) {
        return false;
    }

    /* Check for NaN or Inf in angular velocity */
    if (isnan(rb->omega_x[index]) || isnan(rb->omega_y[index]) ||
        isnan(rb->omega_z[index]) ||
        isinf(rb->omega_x[index]) || isinf(rb->omega_y[index]) ||
        isinf(rb->omega_z[index])) {
        return false;
    }

    /* Check quaternion unit norm: |q|^2 ~ 1.0 (tolerance 1e-4) */
    float mag_sq = rb->quat_w[index] * rb->quat_w[index]
                 + rb->quat_x[index] * rb->quat_x[index]
                 + rb->quat_y[index] * rb->quat_y[index]
                 + rb->quat_z[index] * rb->quat_z[index];

    if (fabsf(mag_sq - 1.0f) > 1e-4f) {
        return false;
    }

    /* Check for NaN or Inf in extension arrays (e.g., RPMs) */
    for (uint32_t e = 0; e < states->extension_count; e++) {
        if (states->extension[e]) {
            if (isnan(states->extension[e][index]) ||
                isinf(states->extension[e][index])) {
                return false;
            }
        }
    }

    return true;
}

void platform_state_print(const PlatformStateSOA* states, uint32_t index) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    FOUNDATION_ASSERT(index < states->rigid_body.capacity, "index out of bounds");

    if (states == NULL || index >= states->rigid_body.capacity) {
        printf("AgentState[%u]: <invalid>\n", index);
        return;
    }

    const RigidBodyStateSOA* rb = &states->rigid_body;

    printf("AgentState[%u]:\n", index);
    printf("  pos: (%.4f, %.4f, %.4f)\n",
           rb->pos_x[index], rb->pos_y[index], rb->pos_z[index]);
    printf("  vel: (%.4f, %.4f, %.4f)\n",
           rb->vel_x[index], rb->vel_y[index], rb->vel_z[index]);
    printf("  quat: (w=%.4f, x=%.4f, y=%.4f, z=%.4f)\n",
           rb->quat_w[index], rb->quat_x[index],
           rb->quat_y[index], rb->quat_z[index]);
    printf("  omega: (%.4f, %.4f, %.4f)\n",
           rb->omega_x[index], rb->omega_y[index], rb->omega_z[index]);
}

void platform_params_print(const PlatformParamsSOA* params, uint32_t index) {
    FOUNDATION_ASSERT(params != NULL, "params is NULL");
    FOUNDATION_ASSERT(index < params->rigid_body.capacity, "index out of bounds");

    if (params == NULL || index >= params->rigid_body.capacity) {
        printf("AgentParams[%u]: <invalid>\n", index);
        return;
    }

    const RigidBodyParamsSOA* rb = &params->rigid_body;

    printf("AgentParams[%u]:\n", index);
    printf("  mass: %.4f kg\n", rb->mass[index]);
    printf("  inertia: (%.6f, %.6f, %.6f) kg*m^2\n",
           rb->ixx[index], rb->iyy[index], rb->izz[index]);
    printf("  collision_radius: %.4f m\n", rb->collision_radius[index]);
    printf("  max_vel: %.1f m/s, max_omega: %.1f rad/s\n",
           rb->max_vel[index], rb->max_omega[index]);
    printf("  gravity: %.4f m/s^2\n", rb->gravity[index]);
}
