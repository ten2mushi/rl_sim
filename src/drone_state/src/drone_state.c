/**
 * Drone State Module Implementation
 *
 * SoA data structures with SIMD-optimized batch operations.
 */

#include "../include/drone_state.h"

/* ============================================================================
 * Section 1: Lifecycle Functions
 * ============================================================================ */

DroneStateSOA* drone_state_create(Arena* arena, uint32_t capacity) {
    if (arena == NULL || capacity == 0) {
        return NULL;
    }

    /* Allocate the struct itself */
    DroneStateSOA* states = arena_alloc_type(arena, DroneStateSOA);
    if (states == NULL) {
        return NULL;
    }

    /* Calculate aligned size for each array (32-byte alignment for AVX2) */
    size_t aligned_size = align_up_size(capacity * sizeof(float), 32);

    /* Allocate all 17 float arrays with 32-byte alignment */
    states->pos_x = arena_alloc_aligned(arena, aligned_size, 32);
    states->pos_y = arena_alloc_aligned(arena, aligned_size, 32);
    states->pos_z = arena_alloc_aligned(arena, aligned_size, 32);

    states->vel_x = arena_alloc_aligned(arena, aligned_size, 32);
    states->vel_y = arena_alloc_aligned(arena, aligned_size, 32);
    states->vel_z = arena_alloc_aligned(arena, aligned_size, 32);

    states->quat_w = arena_alloc_aligned(arena, aligned_size, 32);
    states->quat_x = arena_alloc_aligned(arena, aligned_size, 32);
    states->quat_y = arena_alloc_aligned(arena, aligned_size, 32);
    states->quat_z = arena_alloc_aligned(arena, aligned_size, 32);

    states->omega_x = arena_alloc_aligned(arena, aligned_size, 32);
    states->omega_y = arena_alloc_aligned(arena, aligned_size, 32);
    states->omega_z = arena_alloc_aligned(arena, aligned_size, 32);

    states->rpm_0 = arena_alloc_aligned(arena, aligned_size, 32);
    states->rpm_1 = arena_alloc_aligned(arena, aligned_size, 32);
    states->rpm_2 = arena_alloc_aligned(arena, aligned_size, 32);
    states->rpm_3 = arena_alloc_aligned(arena, aligned_size, 32);

    /* Verify all allocations succeeded */
    if (states->pos_x == NULL || states->pos_y == NULL || states->pos_z == NULL ||
        states->vel_x == NULL || states->vel_y == NULL || states->vel_z == NULL ||
        states->quat_w == NULL || states->quat_x == NULL ||
        states->quat_y == NULL || states->quat_z == NULL ||
        states->omega_x == NULL || states->omega_y == NULL || states->omega_z == NULL ||
        states->rpm_0 == NULL || states->rpm_1 == NULL ||
        states->rpm_2 == NULL || states->rpm_3 == NULL) {
        /* Arena allocation failed - can't free individual allocs from arena */
        return NULL;
    }

    states->capacity = capacity;
    states->count = 0;

    /* Initialize to zero with identity quaternion */
    drone_state_zero(states);

    return states;
}

DroneParamsSOA* drone_params_create(Arena* arena, uint32_t capacity) {
    if (arena == NULL || capacity == 0) {
        return NULL;
    }

    DroneParamsSOA* params = arena_alloc_type(arena, DroneParamsSOA);
    if (params == NULL) {
        return NULL;
    }

    size_t aligned_size = align_up_size(capacity * sizeof(float), 32);

    /* Allocate all 15 float arrays */
    params->mass = arena_alloc_aligned(arena, aligned_size, 32);
    params->ixx = arena_alloc_aligned(arena, aligned_size, 32);
    params->iyy = arena_alloc_aligned(arena, aligned_size, 32);
    params->izz = arena_alloc_aligned(arena, aligned_size, 32);

    params->arm_length = arena_alloc_aligned(arena, aligned_size, 32);
    params->collision_radius = arena_alloc_aligned(arena, aligned_size, 32);

    params->k_thrust = arena_alloc_aligned(arena, aligned_size, 32);
    params->k_torque = arena_alloc_aligned(arena, aligned_size, 32);

    params->k_drag = arena_alloc_aligned(arena, aligned_size, 32);
    params->k_ang_damp = arena_alloc_aligned(arena, aligned_size, 32);

    params->motor_tau = arena_alloc_aligned(arena, aligned_size, 32);
    params->max_rpm = arena_alloc_aligned(arena, aligned_size, 32);

    params->max_vel = arena_alloc_aligned(arena, aligned_size, 32);
    params->max_omega = arena_alloc_aligned(arena, aligned_size, 32);

    params->gravity = arena_alloc_aligned(arena, aligned_size, 32);

    /* Verify allocations */
    if (params->mass == NULL || params->ixx == NULL ||
        params->iyy == NULL || params->izz == NULL ||
        params->arm_length == NULL || params->collision_radius == NULL ||
        params->k_thrust == NULL || params->k_torque == NULL ||
        params->k_drag == NULL || params->k_ang_damp == NULL ||
        params->motor_tau == NULL || params->max_rpm == NULL ||
        params->max_vel == NULL || params->max_omega == NULL ||
        params->gravity == NULL) {
        return NULL;
    }

    params->capacity = capacity;
    params->count = 0;

    /* Initialize all parameters to sensible defaults */
    for (uint32_t i = 0; i < capacity; i++) {
        drone_params_init(params, i);
    }

    return params;
}

DroneEpisodeData* drone_episode_create(Arena* arena, uint32_t capacity) {
    if (arena == NULL || capacity == 0) {
        return NULL;
    }

    DroneEpisodeData* episodes = arena_alloc_array(arena, DroneEpisodeData, capacity);
    if (episodes == NULL) {
        return NULL;
    }

    /* Initialize all episodes */
    for (uint32_t i = 0; i < capacity; i++) {
        drone_episode_init(episodes, i, 0, i);
    }

    return episodes;
}

/* ============================================================================
 * Section 2: Initialization Functions
 * ============================================================================ */

void drone_state_init(DroneStateSOA* states, uint32_t index) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    FOUNDATION_ASSERT(index < states->capacity, "index out of bounds");

    if (states == NULL || index >= states->capacity) {
        return;
    }

    /* Position at origin */
    states->pos_x[index] = 0.0f;
    states->pos_y[index] = 0.0f;
    states->pos_z[index] = 0.0f;

    /* Zero velocity */
    states->vel_x[index] = 0.0f;
    states->vel_y[index] = 0.0f;
    states->vel_z[index] = 0.0f;

    /* Identity quaternion */
    states->quat_w[index] = 1.0f;
    states->quat_x[index] = 0.0f;
    states->quat_y[index] = 0.0f;
    states->quat_z[index] = 0.0f;

    /* Zero angular velocity */
    states->omega_x[index] = 0.0f;
    states->omega_y[index] = 0.0f;
    states->omega_z[index] = 0.0f;

    /* Zero RPMs */
    states->rpm_0[index] = 0.0f;
    states->rpm_1[index] = 0.0f;
    states->rpm_2[index] = 0.0f;
    states->rpm_3[index] = 0.0f;
}

void drone_params_init(DroneParamsSOA* params, uint32_t index) {
    FOUNDATION_ASSERT(params != NULL, "params is NULL");
    FOUNDATION_ASSERT(index < params->capacity, "index out of bounds");

    if (params == NULL || index >= params->capacity) {
        return;
    }

    /* Default: ~0.5kg small quadcopter */
    params->mass[index] = 0.5f;

    /* Approximate inertia for small quad (diagonal matrix) */
    params->ixx[index] = 0.0025f;
    params->iyy[index] = 0.0025f;
    params->izz[index] = 0.0045f;

    /* Geometry */
    params->arm_length[index] = 0.1f;
    params->collision_radius[index] = 0.15f;

    /* Thrust/torque coefficients (typical values) */
    params->k_thrust[index] = 3.16e-10f;  /* N/(rad/s)² */
    params->k_torque[index] = 7.94e-12f;  /* N·m/(rad/s)² */

    /* Damping */
    params->k_drag[index] = 0.1f;
    params->k_ang_damp[index] = 0.01f;

    /* Motor dynamics */
    params->motor_tau[index] = 0.02f;     /* 20ms time constant */
    params->max_rpm[index] = 2500.0f;     /* ~24000 RPM in rad/s */

    /* Limits */
    params->max_vel[index] = 20.0f;       /* m/s */
    params->max_omega[index] = 10.0f;     /* rad/s */

    /* Environment */
    params->gravity[index] = 9.81f;
}

void drone_episode_init(DroneEpisodeData* episodes, uint32_t index,
                        uint32_t env_id, uint32_t drone_id) {
    FOUNDATION_ASSERT(episodes != NULL, "episodes is NULL");

    if (episodes == NULL) {
        return;
    }

    episodes[index].episode_return = 0.0f;
    episodes[index].best_episode_return = -1e30f;  /* Very negative initial */
    episodes[index].episode_length = 0;
    episodes[index].total_episodes = 0;
    episodes[index].env_id = env_id;
    episodes[index].drone_id = drone_id;
    episodes[index].done = 0;
    episodes[index].truncated = 0;
    episodes[index]._pad[0] = 0;
    episodes[index]._pad[1] = 0;
}

/* ============================================================================
 * Section 3: Batch Operations (SIMD-optimized)
 * ============================================================================ */

void drone_state_zero(DroneStateSOA* states) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");

    if (states == NULL) {
        return;
    }

    uint32_t count = states->capacity;

#if defined(FOUNDATION_SIMD_AVX2)
    /* AVX2: Process 8 floats at a time */
    simd_float zero = simd_setzero_ps();
    simd_float one = simd_set1_ps(1.0f);

    uint32_t i = 0;
    uint32_t simd_end = count & ~7u;  /* Round down to multiple of 8 */

    for (; i < simd_end; i += 8) {
        /* Zero position */
        simd_store_ps(&states->pos_x[i], zero);
        simd_store_ps(&states->pos_y[i], zero);
        simd_store_ps(&states->pos_z[i], zero);

        /* Zero velocity */
        simd_store_ps(&states->vel_x[i], zero);
        simd_store_ps(&states->vel_y[i], zero);
        simd_store_ps(&states->vel_z[i], zero);

        /* Identity quaternion: w=1, x=y=z=0 */
        simd_store_ps(&states->quat_w[i], one);
        simd_store_ps(&states->quat_x[i], zero);
        simd_store_ps(&states->quat_y[i], zero);
        simd_store_ps(&states->quat_z[i], zero);

        /* Zero angular velocity */
        simd_store_ps(&states->omega_x[i], zero);
        simd_store_ps(&states->omega_y[i], zero);
        simd_store_ps(&states->omega_z[i], zero);

        /* Zero RPMs */
        simd_store_ps(&states->rpm_0[i], zero);
        simd_store_ps(&states->rpm_1[i], zero);
        simd_store_ps(&states->rpm_2[i], zero);
        simd_store_ps(&states->rpm_3[i], zero);
    }

    /* Scalar remainder */
    for (; i < count; i++) {
        drone_state_init(states, i);
    }

#elif defined(FOUNDATION_SIMD_NEON)
    /* NEON: Process 4 floats at a time */
    simd_float zero = simd_setzero_ps();
    simd_float one = simd_set1_ps(1.0f);

    uint32_t i = 0;
    uint32_t simd_end = count & ~3u;  /* Round down to multiple of 4 */

    for (; i < simd_end; i += 4) {
        simd_store_ps(&states->pos_x[i], zero);
        simd_store_ps(&states->pos_y[i], zero);
        simd_store_ps(&states->pos_z[i], zero);

        simd_store_ps(&states->vel_x[i], zero);
        simd_store_ps(&states->vel_y[i], zero);
        simd_store_ps(&states->vel_z[i], zero);

        simd_store_ps(&states->quat_w[i], one);
        simd_store_ps(&states->quat_x[i], zero);
        simd_store_ps(&states->quat_y[i], zero);
        simd_store_ps(&states->quat_z[i], zero);

        simd_store_ps(&states->omega_x[i], zero);
        simd_store_ps(&states->omega_y[i], zero);
        simd_store_ps(&states->omega_z[i], zero);

        simd_store_ps(&states->rpm_0[i], zero);
        simd_store_ps(&states->rpm_1[i], zero);
        simd_store_ps(&states->rpm_2[i], zero);
        simd_store_ps(&states->rpm_3[i], zero);
    }

    for (; i < count; i++) {
        drone_state_init(states, i);
    }

#else
    /* Scalar fallback */
    for (uint32_t i = 0; i < count; i++) {
        drone_state_init(states, i);
    }
#endif
}

void drone_state_reset_batch(DroneStateSOA* states,
                             const uint32_t* indices,
                             const Vec3* positions,
                             const Quat* orientations,
                             uint32_t count) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    FOUNDATION_ASSERT(indices != NULL, "indices is NULL");
    FOUNDATION_ASSERT(positions != NULL, "positions is NULL");
    FOUNDATION_ASSERT(orientations != NULL, "orientations is NULL");

    if (states == NULL || indices == NULL ||
        positions == NULL || orientations == NULL) {
        return;
    }

    /* Scatter pattern: SIMD gather/scatter not beneficial for scattered indices */
    for (uint32_t i = 0; i < count; i++) {
        uint32_t idx = indices[i];
        FOUNDATION_ASSERT(idx < states->capacity, "index out of bounds");

        if (idx >= states->capacity) {
            continue;
        }

        /* Set position */
        states->pos_x[idx] = positions[i].x;
        states->pos_y[idx] = positions[i].y;
        states->pos_z[idx] = positions[i].z;

        /* Zero velocity */
        states->vel_x[idx] = 0.0f;
        states->vel_y[idx] = 0.0f;
        states->vel_z[idx] = 0.0f;

        /* Set orientation */
        states->quat_w[idx] = orientations[i].w;
        states->quat_x[idx] = orientations[i].x;
        states->quat_y[idx] = orientations[i].y;
        states->quat_z[idx] = orientations[i].z;

        /* Zero angular velocity */
        states->omega_x[idx] = 0.0f;
        states->omega_y[idx] = 0.0f;
        states->omega_z[idx] = 0.0f;

        /* Zero RPMs */
        states->rpm_0[idx] = 0.0f;
        states->rpm_1[idx] = 0.0f;
        states->rpm_2[idx] = 0.0f;
        states->rpm_3[idx] = 0.0f;
    }
}

void drone_state_copy(DroneStateSOA* dst, const DroneStateSOA* src,
                      uint32_t dst_offset, uint32_t src_offset, uint32_t count) {
    FOUNDATION_ASSERT(dst != NULL, "dst is NULL");
    FOUNDATION_ASSERT(src != NULL, "src is NULL");
    FOUNDATION_ASSERT(dst_offset + count <= dst->capacity, "dst overflow");
    FOUNDATION_ASSERT(src_offset + count <= src->capacity, "src overflow");

    if (dst == NULL || src == NULL) {
        return;
    }

    if (dst_offset + count > dst->capacity || src_offset + count > src->capacity) {
        return;
    }

    size_t bytes = count * sizeof(float);

    /* Copy all 17 arrays (memcpy is compiler-optimized) */
    memcpy(&dst->pos_x[dst_offset], &src->pos_x[src_offset], bytes);
    memcpy(&dst->pos_y[dst_offset], &src->pos_y[src_offset], bytes);
    memcpy(&dst->pos_z[dst_offset], &src->pos_z[src_offset], bytes);

    memcpy(&dst->vel_x[dst_offset], &src->vel_x[src_offset], bytes);
    memcpy(&dst->vel_y[dst_offset], &src->vel_y[src_offset], bytes);
    memcpy(&dst->vel_z[dst_offset], &src->vel_z[src_offset], bytes);

    memcpy(&dst->quat_w[dst_offset], &src->quat_w[src_offset], bytes);
    memcpy(&dst->quat_x[dst_offset], &src->quat_x[src_offset], bytes);
    memcpy(&dst->quat_y[dst_offset], &src->quat_y[src_offset], bytes);
    memcpy(&dst->quat_z[dst_offset], &src->quat_z[src_offset], bytes);

    memcpy(&dst->omega_x[dst_offset], &src->omega_x[src_offset], bytes);
    memcpy(&dst->omega_y[dst_offset], &src->omega_y[src_offset], bytes);
    memcpy(&dst->omega_z[dst_offset], &src->omega_z[src_offset], bytes);

    memcpy(&dst->rpm_0[dst_offset], &src->rpm_0[src_offset], bytes);
    memcpy(&dst->rpm_1[dst_offset], &src->rpm_1[src_offset], bytes);
    memcpy(&dst->rpm_2[dst_offset], &src->rpm_2[src_offset], bytes);
    memcpy(&dst->rpm_3[dst_offset], &src->rpm_3[src_offset], bytes);
}

/* ============================================================================
 * Section 4: Single-Drone Accessors
 * ============================================================================ */

DroneStateAoS drone_state_get(const DroneStateSOA* states, uint32_t index) {
    DroneStateAoS result = {0};

    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    FOUNDATION_ASSERT(index < states->capacity, "index out of bounds");

    if (states == NULL || index >= states->capacity) {
        /* Return zeroed state with identity quaternion */
        result.orientation = QUAT_IDENTITY;
        return result;
    }

    result.position = VEC3(states->pos_x[index],
                           states->pos_y[index],
                           states->pos_z[index]);

    result.velocity = VEC3(states->vel_x[index],
                           states->vel_y[index],
                           states->vel_z[index]);

    result.orientation = QUAT(states->quat_w[index],
                              states->quat_x[index],
                              states->quat_y[index],
                              states->quat_z[index]);

    result.omega = VEC3(states->omega_x[index],
                        states->omega_y[index],
                        states->omega_z[index]);

    result.rpm[0] = states->rpm_0[index];
    result.rpm[1] = states->rpm_1[index];
    result.rpm[2] = states->rpm_2[index];
    result.rpm[3] = states->rpm_3[index];

    return result;
}

void drone_state_set(DroneStateSOA* states, uint32_t index, const DroneStateAoS* state) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    FOUNDATION_ASSERT(state != NULL, "state is NULL");
    FOUNDATION_ASSERT(index < states->capacity, "index out of bounds");

    if (states == NULL || state == NULL || index >= states->capacity) {
        return;
    }

    states->pos_x[index] = state->position.x;
    states->pos_y[index] = state->position.y;
    states->pos_z[index] = state->position.z;

    states->vel_x[index] = state->velocity.x;
    states->vel_y[index] = state->velocity.y;
    states->vel_z[index] = state->velocity.z;

    states->quat_w[index] = state->orientation.w;
    states->quat_x[index] = state->orientation.x;
    states->quat_y[index] = state->orientation.y;
    states->quat_z[index] = state->orientation.z;

    states->omega_x[index] = state->omega.x;
    states->omega_y[index] = state->omega.y;
    states->omega_z[index] = state->omega.z;

    states->rpm_0[index] = state->rpm[0];
    states->rpm_1[index] = state->rpm[1];
    states->rpm_2[index] = state->rpm[2];
    states->rpm_3[index] = state->rpm[3];
}

DroneParamsAoS drone_params_get(const DroneParamsSOA* params, uint32_t index) {
    DroneParamsAoS result = {0};

    FOUNDATION_ASSERT(params != NULL, "params is NULL");
    FOUNDATION_ASSERT(index < params->capacity, "index out of bounds");

    if (params == NULL || index >= params->capacity) {
        return result;
    }

    result.mass = params->mass[index];
    result.ixx = params->ixx[index];
    result.iyy = params->iyy[index];
    result.izz = params->izz[index];

    result.arm_length = params->arm_length[index];
    result.collision_radius = params->collision_radius[index];

    result.k_thrust = params->k_thrust[index];
    result.k_torque = params->k_torque[index];

    result.k_drag = params->k_drag[index];
    result.k_ang_damp = params->k_ang_damp[index];

    result.motor_tau = params->motor_tau[index];
    result.max_rpm = params->max_rpm[index];

    result.max_vel = params->max_vel[index];
    result.max_omega = params->max_omega[index];

    result.gravity = params->gravity[index];

    return result;
}

void drone_params_set(DroneParamsSOA* params, uint32_t index, const DroneParamsAoS* param) {
    FOUNDATION_ASSERT(params != NULL, "params is NULL");
    FOUNDATION_ASSERT(param != NULL, "param is NULL");
    FOUNDATION_ASSERT(index < params->capacity, "index out of bounds");

    if (params == NULL || param == NULL || index >= params->capacity) {
        return;
    }

    params->mass[index] = param->mass;
    params->ixx[index] = param->ixx;
    params->iyy[index] = param->iyy;
    params->izz[index] = param->izz;

    params->arm_length[index] = param->arm_length;
    params->collision_radius[index] = param->collision_radius;

    params->k_thrust[index] = param->k_thrust;
    params->k_torque[index] = param->k_torque;

    params->k_drag[index] = param->k_drag;
    params->k_ang_damp[index] = param->k_ang_damp;

    params->motor_tau[index] = param->motor_tau;
    params->max_rpm[index] = param->max_rpm;

    params->max_vel[index] = param->max_vel;
    params->max_omega[index] = param->max_omega;

    params->gravity[index] = param->gravity;
}

/* ============================================================================
 * Section 5: Utility Functions
 * ============================================================================ */

size_t drone_state_memory_size(uint32_t capacity) {
    if (capacity == 0) {
        return 0;
    }

    /* Each array is 32-byte aligned */
    size_t aligned_array = align_up_size(capacity * sizeof(float), 32);

    /* Struct + 17 arrays */
    return sizeof(DroneStateSOA) + 17 * aligned_array;
}

size_t drone_params_memory_size(uint32_t capacity) {
    if (capacity == 0) {
        return 0;
    }

    size_t aligned_array = align_up_size(capacity * sizeof(float), 32);

    /* Struct + 15 arrays */
    return sizeof(DroneParamsSOA) + 15 * aligned_array;
}

bool drone_state_validate(const DroneStateSOA* states, uint32_t index) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    FOUNDATION_ASSERT(index < states->capacity, "index out of bounds");

    if (states == NULL || index >= states->capacity) {
        return false;
    }

    /* Check for NaN in all 17 fields */
    if (isnan(states->pos_x[index]) || isnan(states->pos_y[index]) ||
        isnan(states->pos_z[index])) {
        return false;
    }

    if (isnan(states->vel_x[index]) || isnan(states->vel_y[index]) ||
        isnan(states->vel_z[index])) {
        return false;
    }

    if (isnan(states->quat_w[index]) || isnan(states->quat_x[index]) ||
        isnan(states->quat_y[index]) || isnan(states->quat_z[index])) {
        return false;
    }

    if (isnan(states->omega_x[index]) || isnan(states->omega_y[index]) ||
        isnan(states->omega_z[index])) {
        return false;
    }

    if (isnan(states->rpm_0[index]) || isnan(states->rpm_1[index]) ||
        isnan(states->rpm_2[index]) || isnan(states->rpm_3[index])) {
        return false;
    }

    /* Check quaternion unit norm: |q|² ≈ 1.0 (tolerance 1e-4) */
    float mag_sq = states->quat_w[index] * states->quat_w[index]
                 + states->quat_x[index] * states->quat_x[index]
                 + states->quat_y[index] * states->quat_y[index]
                 + states->quat_z[index] * states->quat_z[index];

    if (fabsf(mag_sq - 1.0f) > 1e-4f) {
        return false;
    }

    /* Check non-negative RPMs */
    if (states->rpm_0[index] < 0.0f || states->rpm_1[index] < 0.0f ||
        states->rpm_2[index] < 0.0f || states->rpm_3[index] < 0.0f) {
        return false;
    }

    return true;
}

void drone_state_print(const DroneStateSOA* states, uint32_t index) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    FOUNDATION_ASSERT(index < states->capacity, "index out of bounds");

    if (states == NULL || index >= states->capacity) {
        printf("DroneState[%u]: <invalid>\n", index);
        return;
    }

    printf("DroneState[%u]:\n", index);
    printf("  pos: (%.4f, %.4f, %.4f)\n",
           states->pos_x[index], states->pos_y[index], states->pos_z[index]);
    printf("  vel: (%.4f, %.4f, %.4f)\n",
           states->vel_x[index], states->vel_y[index], states->vel_z[index]);
    printf("  quat: (w=%.4f, x=%.4f, y=%.4f, z=%.4f)\n",
           states->quat_w[index], states->quat_x[index],
           states->quat_y[index], states->quat_z[index]);
    printf("  omega: (%.4f, %.4f, %.4f)\n",
           states->omega_x[index], states->omega_y[index], states->omega_z[index]);
    printf("  rpms: [%.1f, %.1f, %.1f, %.1f]\n",
           states->rpm_0[index], states->rpm_1[index],
           states->rpm_2[index], states->rpm_3[index]);
}

void drone_params_print(const DroneParamsSOA* params, uint32_t index) {
    FOUNDATION_ASSERT(params != NULL, "params is NULL");
    FOUNDATION_ASSERT(index < params->capacity, "index out of bounds");

    if (params == NULL || index >= params->capacity) {
        printf("DroneParams[%u]: <invalid>\n", index);
        return;
    }

    printf("DroneParams[%u]:\n", index);
    printf("  mass: %.4f kg\n", params->mass[index]);
    printf("  inertia: (%.6f, %.6f, %.6f) kg·m²\n",
           params->ixx[index], params->iyy[index], params->izz[index]);
    printf("  arm_length: %.4f m, collision_radius: %.4f m\n",
           params->arm_length[index], params->collision_radius[index]);
    printf("  k_thrust: %.3e, k_torque: %.3e\n",
           params->k_thrust[index], params->k_torque[index]);
    printf("  k_drag: %.4f, k_ang_damp: %.4f\n",
           params->k_drag[index], params->k_ang_damp[index]);
    printf("  motor_tau: %.4f s, max_rpm: %.1f rad/s\n",
           params->motor_tau[index], params->max_rpm[index]);
    printf("  max_vel: %.1f m/s, max_omega: %.1f rad/s\n",
           params->max_vel[index], params->max_omega[index]);
    printf("  gravity: %.4f m/s²\n", params->gravity[index]);
}
