/**
 * Rigid Body State Module Implementation
 *
 * SoA data structures with batch operations for the platform abstraction layer.
 */

#include "../include/rigid_body_state.h"

/* ============================================================================
 * Section 1: RigidBodyStateSOA Lifecycle
 * ============================================================================ */

RigidBodyStateSOA* rigid_body_state_create(Arena* arena, uint32_t capacity) {
    if (arena == NULL || capacity == 0) {
        return NULL;
    }

    RigidBodyStateSOA* states = arena_alloc_type(arena, RigidBodyStateSOA);
    if (states == NULL) {
        return NULL;
    }

    size_t aligned_size = align_up_size(capacity * sizeof(float), 32);

    /* Allocate all 13 float arrays with 32-byte alignment */
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

    if (states->omega_z == NULL) {
        return NULL;
    }

    states->capacity = capacity;
    states->count = 0;

    rigid_body_state_zero(states);

    return states;
}

void rigid_body_state_zero(RigidBodyStateSOA* states) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    if (states == NULL) return;

    uint32_t count = states->capacity;
    size_t bytes = count * sizeof(float);

    memset(states->pos_x, 0, bytes);
    memset(states->pos_y, 0, bytes);
    memset(states->pos_z, 0, bytes);
    memset(states->vel_x, 0, bytes);
    memset(states->vel_y, 0, bytes);
    memset(states->vel_z, 0, bytes);
    memset(states->quat_x, 0, bytes);
    memset(states->quat_y, 0, bytes);
    memset(states->quat_z, 0, bytes);
    memset(states->omega_x, 0, bytes);
    memset(states->omega_y, 0, bytes);
    memset(states->omega_z, 0, bytes);

    for (uint32_t i = 0; i < count; i++) {
        states->quat_w[i] = 1.0f;
    }
}

void rigid_body_state_reset_batch(RigidBodyStateSOA* states,
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

    for (uint32_t i = 0; i < count; i++) {
        uint32_t idx = indices[i];
        FOUNDATION_ASSERT(idx < states->capacity, "index out of bounds");

        if (idx >= states->capacity) {
            continue;
        }

        states->pos_x[idx] = positions[i].x;
        states->pos_y[idx] = positions[i].y;
        states->pos_z[idx] = positions[i].z;

        states->vel_x[idx] = 0.0f;
        states->vel_y[idx] = 0.0f;
        states->vel_z[idx] = 0.0f;

        states->quat_w[idx] = orientations[i].w;
        states->quat_x[idx] = orientations[i].x;
        states->quat_y[idx] = orientations[i].y;
        states->quat_z[idx] = orientations[i].z;

        states->omega_x[idx] = 0.0f;
        states->omega_y[idx] = 0.0f;
        states->omega_z[idx] = 0.0f;
    }
}

void rigid_body_state_copy(RigidBodyStateSOA* dst, const RigidBodyStateSOA* src,
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
}

size_t rigid_body_state_memory_size(uint32_t capacity) {
    if (capacity == 0) {
        return 0;
    }
    size_t aligned_array = align_up_size(capacity * sizeof(float), 32);
    return sizeof(RigidBodyStateSOA) + RIGID_BODY_STATE_ARRAY_COUNT * aligned_array;
}

/* ============================================================================
 * Section 2: RigidBodyParamsSOA Lifecycle
 * ============================================================================ */

RigidBodyParamsSOA* rigid_body_params_create(Arena* arena, uint32_t capacity) {
    if (arena == NULL || capacity == 0) {
        return NULL;
    }

    RigidBodyParamsSOA* params = arena_alloc_type(arena, RigidBodyParamsSOA);
    if (params == NULL) {
        return NULL;
    }

    size_t aligned_size = align_up_size(capacity * sizeof(float), 32);

    params->mass = arena_alloc_aligned(arena, aligned_size, 32);
    params->ixx = arena_alloc_aligned(arena, aligned_size, 32);
    params->iyy = arena_alloc_aligned(arena, aligned_size, 32);
    params->izz = arena_alloc_aligned(arena, aligned_size, 32);
    params->collision_radius = arena_alloc_aligned(arena, aligned_size, 32);
    params->max_vel = arena_alloc_aligned(arena, aligned_size, 32);
    params->max_omega = arena_alloc_aligned(arena, aligned_size, 32);
    params->gravity = arena_alloc_aligned(arena, aligned_size, 32);

    if (params->gravity == NULL) {
        return NULL;
    }

    params->capacity = capacity;
    params->count = 0;

    for (uint32_t i = 0; i < capacity; i++) {
        rigid_body_params_init(params, i);
    }

    return params;
}

void rigid_body_params_init(RigidBodyParamsSOA* params, uint32_t index) {
    FOUNDATION_ASSERT(params != NULL, "params is NULL");
    FOUNDATION_ASSERT(index < params->capacity, "index out of bounds");

    if (params == NULL || index >= params->capacity) {
        return;
    }

    params->mass[index] = 0.5f;
    params->ixx[index] = 0.0025f;
    params->iyy[index] = 0.0025f;
    params->izz[index] = 0.0045f;
    params->collision_radius[index] = 0.15f;
    params->max_vel[index] = 20.0f;
    params->max_omega[index] = 10.0f;
    params->gravity[index] = 9.81f;
}

size_t rigid_body_params_memory_size(uint32_t capacity) {
    if (capacity == 0) {
        return 0;
    }
    size_t aligned_array = align_up_size(capacity * sizeof(float), 32);
    return sizeof(RigidBodyParamsSOA) + RIGID_BODY_PARAMS_ARRAY_COUNT * aligned_array;
}

/* ============================================================================
 * Section 3: PlatformStateSOA Lifecycle
 * ============================================================================ */

PlatformStateSOA* platform_state_create(Arena* arena, uint32_t capacity,
                                        uint32_t extension_count) {
    if (arena == NULL || capacity == 0) {
        return NULL;
    }

    PlatformStateSOA* pstate = arena_alloc_type(arena, PlatformStateSOA);
    if (pstate == NULL) {
        return NULL;
    }

    /* Allocate rigid body core inline (not a separate allocation) */
    size_t aligned_size = align_up_size(capacity * sizeof(float), 32);

    RigidBodyStateSOA* rb = &pstate->rigid_body;

    rb->pos_x = arena_alloc_aligned(arena, aligned_size, 32);
    rb->pos_y = arena_alloc_aligned(arena, aligned_size, 32);
    rb->pos_z = arena_alloc_aligned(arena, aligned_size, 32);

    rb->vel_x = arena_alloc_aligned(arena, aligned_size, 32);
    rb->vel_y = arena_alloc_aligned(arena, aligned_size, 32);
    rb->vel_z = arena_alloc_aligned(arena, aligned_size, 32);

    rb->quat_w = arena_alloc_aligned(arena, aligned_size, 32);
    rb->quat_x = arena_alloc_aligned(arena, aligned_size, 32);
    rb->quat_y = arena_alloc_aligned(arena, aligned_size, 32);
    rb->quat_z = arena_alloc_aligned(arena, aligned_size, 32);

    rb->omega_x = arena_alloc_aligned(arena, aligned_size, 32);
    rb->omega_y = arena_alloc_aligned(arena, aligned_size, 32);
    rb->omega_z = arena_alloc_aligned(arena, aligned_size, 32);

    if (rb->omega_z == NULL) {
        return NULL;
    }

    rb->capacity = capacity;
    rb->count = 0;

    /* Allocate extension pointer array + data arrays */
    pstate->extension_count = extension_count;
    if (extension_count > 0) {
        pstate->extension = arena_alloc_array(arena, float*, extension_count);
        if (pstate->extension == NULL) {
            return NULL;
        }
        for (uint32_t i = 0; i < extension_count; i++) {
            pstate->extension[i] = arena_alloc_aligned(arena, aligned_size, 32);
            if (pstate->extension[i] == NULL) {
                return NULL;
            }
        }
    } else {
        pstate->extension = NULL;
    }

    platform_state_zero(pstate);

    return pstate;
}

void platform_state_zero(PlatformStateSOA* states) {
    FOUNDATION_ASSERT(states != NULL, "states is NULL");
    if (states == NULL) return;

    rigid_body_state_zero(&states->rigid_body);

    /* Zero extension arrays */
    uint32_t cap = states->rigid_body.capacity;
    size_t bytes = cap * sizeof(float);
    for (uint32_t i = 0; i < states->extension_count; i++) {
        memset(states->extension[i], 0, bytes);
    }
}

size_t platform_state_memory_size(uint32_t capacity, uint32_t extension_count) {
    if (capacity == 0) {
        return 0;
    }
    size_t aligned_array = align_up_size(capacity * sizeof(float), 32);
    size_t total = sizeof(PlatformStateSOA);
    total += RIGID_BODY_STATE_ARRAY_COUNT * aligned_array;
    if (extension_count > 0) {
        total += extension_count * sizeof(float*);
        total += extension_count * aligned_array;
    }
    return total;
}

void platform_state_copy(PlatformStateSOA* dst, const PlatformStateSOA* src,
                          uint32_t dst_offset, uint32_t src_offset, uint32_t count) {
    FOUNDATION_ASSERT(dst != NULL && src != NULL, "NULL state pointer");
    if (dst == NULL || src == NULL || count == 0) return;

    /* Copy rigid body arrays */
    rigid_body_state_copy(&dst->rigid_body, &src->rigid_body,
                          dst_offset, src_offset, count);

    /* Copy extension arrays */
    uint32_t ext_count = dst->extension_count < src->extension_count
                       ? dst->extension_count : src->extension_count;
    size_t bytes = count * sizeof(float);
    for (uint32_t e = 0; e < ext_count; e++) {
        memcpy(&dst->extension[e][dst_offset],
               &src->extension[e][src_offset], bytes);
    }
}

/* ============================================================================
 * Section 4: PlatformParamsSOA Lifecycle
 * ============================================================================ */

PlatformParamsSOA* platform_params_create(Arena* arena, uint32_t capacity,
                                          uint32_t extension_count) {
    if (arena == NULL || capacity == 0) {
        return NULL;
    }

    PlatformParamsSOA* pparams = arena_alloc_type(arena, PlatformParamsSOA);
    if (pparams == NULL) {
        return NULL;
    }

    size_t aligned_size = align_up_size(capacity * sizeof(float), 32);

    RigidBodyParamsSOA* rb = &pparams->rigid_body;

    rb->mass = arena_alloc_aligned(arena, aligned_size, 32);
    rb->ixx = arena_alloc_aligned(arena, aligned_size, 32);
    rb->iyy = arena_alloc_aligned(arena, aligned_size, 32);
    rb->izz = arena_alloc_aligned(arena, aligned_size, 32);
    rb->collision_radius = arena_alloc_aligned(arena, aligned_size, 32);
    rb->max_vel = arena_alloc_aligned(arena, aligned_size, 32);
    rb->max_omega = arena_alloc_aligned(arena, aligned_size, 32);
    rb->gravity = arena_alloc_aligned(arena, aligned_size, 32);

    if (rb->gravity == NULL) {
        return NULL;
    }

    rb->capacity = capacity;
    rb->count = 0;

    for (uint32_t i = 0; i < capacity; i++) {
        rigid_body_params_init(rb, i);
    }

    /* Allocate extension pointer array + data arrays */
    pparams->extension_count = extension_count;
    if (extension_count > 0) {
        pparams->extension = arena_alloc_array(arena, float*, extension_count);
        if (pparams->extension == NULL) {
            return NULL;
        }
        for (uint32_t i = 0; i < extension_count; i++) {
            pparams->extension[i] = arena_alloc_aligned(arena, aligned_size, 32);
            if (pparams->extension[i] == NULL) {
                return NULL;
            }
            memset(pparams->extension[i], 0, aligned_size);
        }
    } else {
        pparams->extension = NULL;
    }

    return pparams;
}

size_t platform_params_memory_size(uint32_t capacity, uint32_t extension_count) {
    if (capacity == 0) {
        return 0;
    }
    size_t aligned_array = align_up_size(capacity * sizeof(float), 32);
    size_t total = sizeof(PlatformParamsSOA);
    total += RIGID_BODY_PARAMS_ARRAY_COUNT * aligned_array;
    if (extension_count > 0) {
        total += extension_count * sizeof(float*);
        total += extension_count * aligned_array;
    }
    return total;
}
