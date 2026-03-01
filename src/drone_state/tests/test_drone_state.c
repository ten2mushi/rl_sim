/**
 * Drone State Module Unit Tests
 *
 * Yoneda Philosophy: Tests serve as complete behavioral specification.
 * Each test defines what the module MUST do - the definitive contract.
 */

#include "../include/drone_state.h"
#include "platform_quadcopter.h"
#include "test_harness.h"

/* ============================================================================
 * Section 1: Allocation Tests
 * ============================================================================ */

TEST(allocation_basic) {
    Arena* arena = arena_create(1024 * 1024);  /* 1MB */
    ASSERT_NOT_NULL(arena);

    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);
    ASSERT_EQ(states->rigid_body.capacity, 100);
    ASSERT_EQ(states->rigid_body.count, 0);

    /* All array pointers should be valid */
    ASSERT_NOT_NULL(states->rigid_body.pos_x);
    ASSERT_NOT_NULL(states->rigid_body.pos_y);
    ASSERT_NOT_NULL(states->rigid_body.pos_z);
    ASSERT_NOT_NULL(states->rigid_body.quat_w);
    ASSERT_NOT_NULL(states->extension[QUAD_EXT_RPM_3]);

    arena_destroy(arena);
    return 0;
}

TEST(allocation_alignment_32byte) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 256, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    /* All float arrays must be 32-byte aligned for AVX2 */
    ASSERT_TRUE(((uintptr_t)(states->rigid_body.pos_x) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->rigid_body.pos_y) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->rigid_body.pos_z) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->rigid_body.vel_x) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->rigid_body.vel_y) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->rigid_body.vel_z) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->rigid_body.quat_w) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->rigid_body.quat_x) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->rigid_body.quat_y) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->rigid_body.quat_z) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->rigid_body.omega_x) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->rigid_body.omega_y) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->rigid_body.omega_z) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->extension[QUAD_EXT_RPM_0]) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->extension[QUAD_EXT_RPM_1]) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->extension[QUAD_EXT_RPM_2]) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->extension[QUAD_EXT_RPM_3]) & (32 - 1)) == 0);

    arena_destroy(arena);
    return 0;
}

TEST(allocation_capacity_1) {
    Arena* arena = arena_create(64 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 1, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);
    ASSERT_EQ(states->rigid_body.capacity, 1);

    /* Should still be 32-byte aligned */
    ASSERT_TRUE(((uintptr_t)(states->rigid_body.pos_x) & (32 - 1)) == 0);

    arena_destroy(arena);
    return 0;
}

TEST(allocation_capacity_1024) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 1024, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);
    ASSERT_EQ(states->rigid_body.capacity, 1024);

    /* Verify memory fits expectations (~68KB for state) */
    size_t expected_size = platform_state_memory_size(1024, QUAD_STATE_EXT_COUNT);
    ASSERT_TRUE(expected_size <= 80 * 1024);

    arena_destroy(arena);
    return 0;
}

TEST(allocation_capacity_10000) {
    Arena* arena = arena_create(10 * 1024 * 1024);  /* 10MB */
    PlatformStateSOA* states = platform_state_create(arena, 10000, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);
    ASSERT_EQ(states->rigid_body.capacity, 10000);

    arena_destroy(arena);
    return 0;
}

TEST(allocation_zero_capacity) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 0, QUAD_STATE_EXT_COUNT);
    ASSERT_NULL(states);

    arena_destroy(arena);
    return 0;
}

TEST(allocation_arena_overflow) {
    /* Tiny arena that can't fit allocation */
    Arena* arena = arena_create(64);  /* Too small for even 1 drone */
    PlatformStateSOA* states = platform_state_create(arena, 1000, QUAD_STATE_EXT_COUNT);
    ASSERT_NULL(states);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 2: Zero Operation Tests
 * ============================================================================ */

TEST(zero_all_positions) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    /* Manually set non-zero values */
    for (uint32_t i = 0; i < states->rigid_body.capacity; i++) {
        states->rigid_body.pos_x[i] = 1.0f + (float)i;
        states->rigid_body.pos_y[i] = 2.0f + (float)i;
        states->rigid_body.pos_z[i] = 3.0f + (float)i;
    }

    /* Zero should reset */
    platform_state_zero(states);

    for (uint32_t i = 0; i < states->rigid_body.capacity; i++) {
        ASSERT_FLOAT_EQ(states->rigid_body.pos_x[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.pos_y[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.pos_z[i], 0.0f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(zero_all_velocities) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    for (uint32_t i = 0; i < states->rigid_body.capacity; i++) {
        states->rigid_body.vel_x[i] = 5.0f;
        states->rigid_body.vel_y[i] = 6.0f;
        states->rigid_body.vel_z[i] = 7.0f;
    }

    platform_state_zero(states);

    for (uint32_t i = 0; i < states->rigid_body.capacity; i++) {
        ASSERT_FLOAT_EQ(states->rigid_body.vel_x[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.vel_y[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.vel_z[i], 0.0f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(zero_identity_quaternion) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    /* Set non-identity quaternions */
    for (uint32_t i = 0; i < states->rigid_body.capacity; i++) {
        states->rigid_body.quat_w[i] = 0.5f;
        states->rigid_body.quat_x[i] = 0.5f;
        states->rigid_body.quat_y[i] = 0.5f;
        states->rigid_body.quat_z[i] = 0.5f;
    }

    platform_state_zero(states);

    /* Identity quaternion: w=1, x=y=z=0 */
    for (uint32_t i = 0; i < states->rigid_body.capacity; i++) {
        ASSERT_FLOAT_EQ(states->rigid_body.quat_w[i], 1.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.quat_x[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.quat_y[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.quat_z[i], 0.0f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(zero_all_rpms) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    for (uint32_t i = 0; i < states->rigid_body.capacity; i++) {
        states->extension[QUAD_EXT_RPM_0][i] = 1000.0f;
        states->extension[QUAD_EXT_RPM_1][i] = 1100.0f;
        states->extension[QUAD_EXT_RPM_2][i] = 1200.0f;
        states->extension[QUAD_EXT_RPM_3][i] = 1300.0f;
    }

    platform_state_zero(states);

    for (uint32_t i = 0; i < states->rigid_body.capacity; i++) {
        ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_0][i], 0.0f);
        ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_1][i], 0.0f);
        ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_2][i], 0.0f);
        ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_3][i], 0.0f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(zero_all_angular_velocity) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    for (uint32_t i = 0; i < states->rigid_body.capacity; i++) {
        states->rigid_body.omega_x[i] = 1.5f;
        states->rigid_body.omega_y[i] = 2.5f;
        states->rigid_body.omega_z[i] = 3.5f;
    }

    platform_state_zero(states);

    for (uint32_t i = 0; i < states->rigid_body.capacity; i++) {
        ASSERT_FLOAT_EQ(states->rigid_body.omega_x[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.omega_y[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.omega_z[i], 0.0f);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 3: Accessor Round-Trip Tests
 * ============================================================================ */

TEST(accessor_set_get_position) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    PlatformStateAoS state = {0};
    state.position = VEC3(1.5f, 2.5f, 3.5f);
    state.orientation = QUAT_IDENTITY;

    platform_state_set(states, 42, &state);
    PlatformStateAoS result = platform_state_get(states, 42);

    ASSERT_FLOAT_EQ(result.position.x, 1.5f);
    ASSERT_FLOAT_EQ(result.position.y, 2.5f);
    ASSERT_FLOAT_EQ(result.position.z, 3.5f);

    arena_destroy(arena);
    return 0;
}

TEST(accessor_set_get_velocity) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    PlatformStateAoS state = {0};
    state.velocity = VEC3(-1.0f, 2.0f, -3.0f);
    state.orientation = QUAT_IDENTITY;

    platform_state_set(states, 25, &state);
    PlatformStateAoS result = platform_state_get(states, 25);

    ASSERT_FLOAT_EQ(result.velocity.x, -1.0f);
    ASSERT_FLOAT_EQ(result.velocity.y, 2.0f);
    ASSERT_FLOAT_EQ(result.velocity.z, -3.0f);

    arena_destroy(arena);
    return 0;
}

TEST(accessor_set_get_orientation) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    /* 90 degree rotation around Z axis (approximately) */
    PlatformStateAoS state = {0};
    state.orientation = QUAT(0.7071f, 0.0f, 0.0f, 0.7071f);

    platform_state_set(states, 10, &state);
    PlatformStateAoS result = platform_state_get(states, 10);

    ASSERT_FLOAT_EQ(result.orientation.w, 0.7071f);
    ASSERT_FLOAT_EQ(result.orientation.x, 0.0f);
    ASSERT_FLOAT_EQ(result.orientation.y, 0.0f);
    ASSERT_FLOAT_EQ(result.orientation.z, 0.7071f);

    arena_destroy(arena);
    return 0;
}

TEST(accessor_set_get_omega) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    PlatformStateAoS state = {0};
    state.omega = VEC3(0.1f, 0.2f, 0.3f);
    state.orientation = QUAT_IDENTITY;

    platform_state_set(states, 5, &state);
    PlatformStateAoS result = platform_state_get(states, 5);

    ASSERT_FLOAT_EQ(result.omega.x, 0.1f);
    ASSERT_FLOAT_EQ(result.omega.y, 0.2f);
    ASSERT_FLOAT_EQ(result.omega.z, 0.3f);

    arena_destroy(arena);
    return 0;
}

TEST(accessor_set_get_rpms) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    PlatformStateAoS state = {0};
    states->extension[QUAD_EXT_RPM_0][77] = 1000.0f;
    states->extension[QUAD_EXT_RPM_1][77] = 1100.0f;
    states->extension[QUAD_EXT_RPM_2][77] = 1200.0f;
    states->extension[QUAD_EXT_RPM_3][77] = 1300.0f;
    state.orientation = QUAT_IDENTITY;

    platform_state_set(states, 77, &state);
    PlatformStateAoS result = platform_state_get(states, 77);

    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_0][77], 1000.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_1][77], 1100.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_2][77], 1200.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_3][77], 1300.0f);

    arena_destroy(arena);
    return 0;
}

TEST(accessor_all_fields_roundtrip) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    /* Set RPMs at a different index to verify set/get at index 33 doesn't affect it */
    states->extension[QUAD_EXT_RPM_0][77] = 100.0f;
    states->extension[QUAD_EXT_RPM_1][77] = 200.0f;
    states->extension[QUAD_EXT_RPM_2][77] = 300.0f;
    states->extension[QUAD_EXT_RPM_3][77] = 400.0f;

    PlatformStateAoS original = {
        .position = VEC3(1.0f, 2.0f, 3.0f),
        .velocity = VEC3(4.0f, 5.0f, 6.0f),
        .orientation = QUAT(0.5f, 0.5f, 0.5f, 0.5f),
        .omega = VEC3(0.1f, 0.2f, 0.3f),
    };

    platform_state_set(states, 33, &original);
    PlatformStateAoS result = platform_state_get(states, 33);

    ASSERT_FLOAT_EQ(result.position.x, original.position.x);
    ASSERT_FLOAT_EQ(result.position.y, original.position.y);
    ASSERT_FLOAT_EQ(result.position.z, original.position.z);
    ASSERT_FLOAT_EQ(result.velocity.x, original.velocity.x);
    ASSERT_FLOAT_EQ(result.velocity.y, original.velocity.y);
    ASSERT_FLOAT_EQ(result.velocity.z, original.velocity.z);
    ASSERT_FLOAT_EQ(result.orientation.w, original.orientation.w);
    ASSERT_FLOAT_EQ(result.orientation.x, original.orientation.x);
    ASSERT_FLOAT_EQ(result.orientation.y, original.orientation.y);
    ASSERT_FLOAT_EQ(result.orientation.z, original.orientation.z);
    ASSERT_FLOAT_EQ(result.omega.x, original.omega.x);
    ASSERT_FLOAT_EQ(result.omega.y, original.omega.y);
    ASSERT_FLOAT_EQ(result.omega.z, original.omega.z);
    /* Verify RPMs at index 77 are untouched by set/get at index 33 */
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_0][77], 100.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_1][77], 200.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_2][77], 300.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_3][77], 400.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 4: Batch Reset Tests
 * ============================================================================ */

TEST(reset_batch_single_index) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    /* Set some values */
    for (uint32_t i = 0; i < states->rigid_body.capacity; i++) {
        states->rigid_body.pos_x[i] = (float)i;
        states->rigid_body.vel_x[i] = (float)i * 2.0f;
    }

    /* Reset single index */
    uint32_t indices[1] = {50};
    Vec3 positions[1] = {VEC3(10.0f, 20.0f, 30.0f)};
    Quat orientations[1] = {QUAT_IDENTITY};

    rigid_body_state_reset_batch(&states->rigid_body, indices, positions, orientations, 1);

    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[50], 10.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_y[50], 20.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_z[50], 30.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.vel_x[50], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.quat_w[50], 1.0f);

    /* Adjacent indices should be unchanged */
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[49], 49.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[51], 51.0f);

    arena_destroy(arena);
    return 0;
}

TEST(reset_batch_multiple_indices) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    uint32_t indices[3] = {0, 50, 99};
    Vec3 positions[3] = {
        VEC3(1.0f, 1.0f, 1.0f),
        VEC3(2.0f, 2.0f, 2.0f),
        VEC3(3.0f, 3.0f, 3.0f)
    };
    Quat orientations[3] = {QUAT_IDENTITY, QUAT_IDENTITY, QUAT_IDENTITY};

    rigid_body_state_reset_batch(&states->rigid_body, indices, positions, orientations, 3);

    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[0], 1.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[50], 2.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[99], 3.0f);

    arena_destroy(arena);
    return 0;
}

TEST(reset_batch_verify_untouched) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    /* Set recognizable pattern */
    for (uint32_t i = 0; i < states->rigid_body.capacity; i++) {
        states->rigid_body.pos_x[i] = (float)(i + 1000);
    }

    uint32_t indices[2] = {25, 75};
    Vec3 positions[2] = {VEC3(0.0f, 0.0f, 0.0f), VEC3(0.0f, 0.0f, 0.0f)};
    Quat orientations[2] = {QUAT_IDENTITY, QUAT_IDENTITY};

    rigid_body_state_reset_batch(&states->rigid_body, indices, positions, orientations, 2);

    /* Verify untouched */
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[0], 1000.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[24], 1024.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[26], 1026.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[99], 1099.0f);

    arena_destroy(arena);
    return 0;
}

TEST(reset_batch_scattered_pattern) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 1000, QUAD_STATE_EXT_COUNT);

    /* Scattered indices simulating random done environments */
    uint32_t indices[10] = {7, 103, 256, 512, 513, 600, 777, 888, 901, 999};
    Vec3 positions[10];
    Quat orientations[10];

    for (int i = 0; i < 10; i++) {
        positions[i] = VEC3((float)i, (float)i, (float)i);
        orientations[i] = QUAT_IDENTITY;
    }

    rigid_body_state_reset_batch(&states->rigid_body, indices, positions, orientations, 10);

    for (int i = 0; i < 10; i++) {
        ASSERT_FLOAT_EQ(states->rigid_body.pos_x[indices[i]], (float)i);
        ASSERT_FLOAT_EQ(states->rigid_body.vel_x[indices[i]], 0.0f);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 5: Copy Tests
 * ============================================================================ */

TEST(copy_full_range) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    PlatformStateSOA* src = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);
    PlatformStateSOA* dst = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    /* Set source values */
    for (uint32_t i = 0; i < src->rigid_body.capacity; i++) {
        src->rigid_body.pos_x[i] = (float)i;
        src->rigid_body.pos_y[i] = (float)i * 2;
        src->rigid_body.quat_w[i] = 1.0f;
    }

    rigid_body_state_copy(&dst->rigid_body, &src->rigid_body, 0, 0, 100);

    for (uint32_t i = 0; i < 100; i++) {
        ASSERT_FLOAT_EQ(dst->rigid_body.pos_x[i], (float)i);
        ASSERT_FLOAT_EQ(dst->rigid_body.pos_y[i], (float)i * 2);
    }

    arena_destroy(arena);
    return 0;
}

TEST(copy_partial_range) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    PlatformStateSOA* src = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);
    PlatformStateSOA* dst = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    /* Set source values */
    for (uint32_t i = 0; i < src->rigid_body.capacity; i++) {
        src->rigid_body.pos_x[i] = (float)(i + 500);
    }

    /* Copy subset: src[10..20] -> dst[50..60] */
    rigid_body_state_copy(&dst->rigid_body, &src->rigid_body, 50, 10, 10);

    for (uint32_t i = 0; i < 10; i++) {
        ASSERT_FLOAT_EQ(dst->rigid_body.pos_x[50 + i], (float)(510 + i));
    }

    /* Verify areas outside copy range are still zero */
    ASSERT_FLOAT_EQ(dst->rigid_body.pos_x[0], 0.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.pos_x[49], 0.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.pos_x[60], 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(copy_verify_independence) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    PlatformStateSOA* src = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);
    PlatformStateSOA* dst = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    for (uint32_t i = 0; i < src->rigid_body.capacity; i++) {
        src->rigid_body.pos_x[i] = 42.0f;
    }

    rigid_body_state_copy(&dst->rigid_body, &src->rigid_body, 0, 0, 100);

    /* Modify source after copy */
    for (uint32_t i = 0; i < src->rigid_body.capacity; i++) {
        src->rigid_body.pos_x[i] = 999.0f;
    }

    /* Destination should be unaffected */
    for (uint32_t i = 0; i < 100; i++) {
        ASSERT_FLOAT_EQ(dst->rigid_body.pos_x[i], 42.0f);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 6: Validation Tests
 * ============================================================================ */

TEST(validation_valid_state) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    /* Fresh state should be valid */
    ASSERT_TRUE(platform_state_validate(states, 0));
    ASSERT_TRUE(platform_state_validate(states, 50));
    ASSERT_TRUE(platform_state_validate(states, 99));

    arena_destroy(arena);
    return 0;
}

TEST(validation_nan_position) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    states->rigid_body.pos_x[0] = NAN;
    ASSERT_FALSE(platform_state_validate(states, 0));

    platform_state_init(states, 0);
    states->rigid_body.pos_y[0] = NAN;
    ASSERT_FALSE(platform_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

TEST(validation_nan_velocity) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    states->rigid_body.vel_x[0] = NAN;
    ASSERT_FALSE(platform_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

TEST(validation_nan_quaternion) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    states->rigid_body.quat_w[0] = NAN;
    ASSERT_FALSE(platform_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

TEST(validation_unnormalized_quat) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    /* Unnormalized quaternion: |q|^2 = 4, not 1 */
    states->rigid_body.quat_w[0] = 1.0f;
    states->rigid_body.quat_x[0] = 1.0f;
    states->rigid_body.quat_y[0] = 1.0f;
    states->rigid_body.quat_z[0] = 1.0f;

    ASSERT_FALSE(platform_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

TEST(validation_negative_rpm) {
    /* Negative extension values are finite floats, so generic validate accepts them.
     * Platform-specific constraints (e.g., non-negative RPMs) are not enforced here. */
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    platform_state_init(states, 0);
    states->extension[QUAD_EXT_RPM_0][0] = -100.0f;
    ASSERT_TRUE(platform_state_validate(states, 0));

    states->extension[QUAD_EXT_RPM_2][0] = -1.0f;
    ASSERT_TRUE(platform_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 7: Boundary Tests
 * ============================================================================ */

TEST(boundary_index_zero) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    PlatformStateAoS state = {
        .position = VEC3(1.0f, 2.0f, 3.0f),
        .orientation = QUAT_IDENTITY
    };

    platform_state_set(states, 0, &state);
    PlatformStateAoS result = platform_state_get(states, 0);

    ASSERT_FLOAT_EQ(result.position.x, 1.0f);

    arena_destroy(arena);
    return 0;
}

TEST(boundary_index_max) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    PlatformStateAoS state = {
        .position = VEC3(99.0f, 99.0f, 99.0f),
        .orientation = QUAT_IDENTITY
    };

    platform_state_set(states, 99, &state);
    PlatformStateAoS result = platform_state_get(states, 99);

    ASSERT_FLOAT_EQ(result.position.x, 99.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 8: Memory Size Tests
 * ============================================================================ */

TEST(memory_size_state) {
    size_t size = platform_state_memory_size(1024, QUAD_STATE_EXT_COUNT);
    ASSERT_TRUE(size > 0);
    ASSERT_TRUE(size <= 80 * 1024);
    ASSERT_TRUE(size >= 68 * 1024);

    size_t size_1 = platform_state_memory_size(1, QUAD_STATE_EXT_COUNT);
    ASSERT_TRUE(size_1 > 0);

    return 0;
}

TEST(memory_size_params) {
    size_t size = platform_params_memory_size(1024, QUAD_PARAMS_EXT_COUNT);
    ASSERT_TRUE(size > 0);
    ASSERT_TRUE(size <= 70 * 1024);

    return 0;
}

/* ============================================================================
 * Section 9: Episode Data Tests
 * ============================================================================ */

TEST(episode_create) {
    Arena* arena = arena_create(1024 * 1024);
    AgentEpisodeData* episodes = agent_episode_create(arena, 100);
    ASSERT_NOT_NULL(episodes);

    arena_destroy(arena);
    return 0;
}

TEST(episode_init) {
    Arena* arena = arena_create(1024 * 1024);
    AgentEpisodeData* episodes = agent_episode_create(arena, 100);

    agent_episode_init(episodes, 42, 10, 5);

    ASSERT_FLOAT_EQ(episodes[42].episode_return, 0.0f);
    ASSERT_EQ(episodes[42].episode_length, 0);
    ASSERT_EQ(episodes[42].env_id, 10);
    ASSERT_EQ(episodes[42].agent_id, 5);
    ASSERT_EQ(episodes[42].done, 0);
    ASSERT_EQ(episodes[42].truncated, 0);

    arena_destroy(arena);
    return 0;
}

TEST(episode_size) {
    ASSERT_EQ((int)sizeof(AgentEpisodeData), 28);
    return 0;
}

/* ============================================================================
 * Section 10: Params Tests
 * ============================================================================ */

TEST(params_create) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformParamsSOA* params = platform_params_create(arena, 100, QUAD_PARAMS_EXT_COUNT);
    ASSERT_NOT_NULL(params);
    ASSERT_EQ(params->rigid_body.capacity, 100);

    /* Check default values */
    ASSERT_FLOAT_EQ(params->rigid_body.mass[0], 0.5f);
    ASSERT_FLOAT_EQ(params->rigid_body.gravity[0], 9.81f);

    arena_destroy(arena);
    return 0;
}

TEST(params_alignment) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformParamsSOA* params = platform_params_create(arena, 256, QUAD_PARAMS_EXT_COUNT);
    ASSERT_NOT_NULL(params);

    ASSERT_TRUE(((uintptr_t)(params->rigid_body.mass) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(params->rigid_body.ixx) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(params->rigid_body.gravity) & (32 - 1)) == 0);

    arena_destroy(arena);
    return 0;
}

TEST(params_accessor_roundtrip) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformParamsSOA* params = platform_params_create(arena, 100, QUAD_PARAMS_EXT_COUNT);

    PlatformParamsAoS original = {
        .mass = 1.5f,
        .ixx = 0.01f,
        .iyy = 0.02f,
        .izz = 0.03f,
        .collision_radius = 0.25f,
        .max_vel = 25.0f,
        .max_omega = 15.0f,
        .gravity = 10.0f
    };

    platform_params_set(params, 50, &original);
    PlatformParamsAoS result = platform_params_get(params, 50);

    ASSERT_FLOAT_EQ(result.mass, 1.5f);
    ASSERT_FLOAT_EQ(result.ixx, 0.01f);
    ASSERT_FLOAT_EQ(result.gravity, 10.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 11: platform_state_init Default Value Verification
 * ============================================================================ */

TEST(init_default_values_all_fields) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    states->rigid_body.pos_x[10] = 999.0f;
    states->rigid_body.vel_y[10] = -42.0f;
    states->rigid_body.quat_w[10] = 0.5f;
    states->rigid_body.quat_z[10] = 0.5f;
    states->rigid_body.omega_x[10] = 7.77f;
    states->extension[QUAD_EXT_RPM_0][10] = 5000.0f;

    platform_state_init(states, 10);

    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[10], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_y[10], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_z[10], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.vel_x[10], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.vel_y[10], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.vel_z[10], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.quat_w[10], 1.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.quat_x[10], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.quat_y[10], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.quat_z[10], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.omega_x[10], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.omega_y[10], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.omega_z[10], 0.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_0][10], 0.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_1][10], 0.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_2][10], 0.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_3][10], 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(init_overwrites_previous_values) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 50, QUAD_STATE_EXT_COUNT);

    uint32_t idx = 25;
    states->rigid_body.pos_x[idx] = 100.0f;
    states->rigid_body.pos_y[idx] = 200.0f;
    states->rigid_body.pos_z[idx] = 300.0f;
    states->rigid_body.vel_x[idx] = 10.0f;
    states->rigid_body.vel_y[idx] = 20.0f;
    states->rigid_body.vel_z[idx] = 30.0f;
    states->rigid_body.quat_w[idx] = 0.5f;
    states->rigid_body.quat_x[idx] = 0.5f;
    states->rigid_body.quat_y[idx] = 0.5f;
    states->rigid_body.quat_z[idx] = 0.5f;
    states->rigid_body.omega_x[idx] = 1.1f;
    states->rigid_body.omega_y[idx] = 2.2f;
    states->rigid_body.omega_z[idx] = 3.3f;
    states->extension[QUAD_EXT_RPM_0][idx] = 1000.0f;
    states->extension[QUAD_EXT_RPM_1][idx] = 1100.0f;
    states->extension[QUAD_EXT_RPM_2][idx] = 1200.0f;
    states->extension[QUAD_EXT_RPM_3][idx] = 1300.0f;

    platform_state_init(states, idx);

    PlatformStateAoS result = platform_state_get(states, idx);
    ASSERT_FLOAT_EQ(result.position.x, 0.0f);
    ASSERT_FLOAT_EQ(result.position.y, 0.0f);
    ASSERT_FLOAT_EQ(result.position.z, 0.0f);
    ASSERT_FLOAT_EQ(result.velocity.x, 0.0f);
    ASSERT_FLOAT_EQ(result.velocity.y, 0.0f);
    ASSERT_FLOAT_EQ(result.velocity.z, 0.0f);
    ASSERT_FLOAT_EQ(result.orientation.w, 1.0f);
    ASSERT_FLOAT_EQ(result.orientation.x, 0.0f);
    ASSERT_FLOAT_EQ(result.orientation.y, 0.0f);
    ASSERT_FLOAT_EQ(result.orientation.z, 0.0f);
    ASSERT_FLOAT_EQ(result.omega.x, 0.0f);
    ASSERT_FLOAT_EQ(result.omega.y, 0.0f);
    ASSERT_FLOAT_EQ(result.omega.z, 0.0f);
    /* Verify RPMs were zeroed at the init'd index */
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_0][idx], 0.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_1][idx], 0.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_2][idx], 0.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_3][idx], 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(init_boundary_indices) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 64, QUAD_STATE_EXT_COUNT);

    states->rigid_body.pos_x[0] = 42.0f;
    states->rigid_body.vel_z[0] = -7.0f;
    states->extension[QUAD_EXT_RPM_3][0] = 999.0f;
    states->rigid_body.pos_x[63] = 42.0f;
    states->rigid_body.vel_z[63] = -7.0f;
    states->extension[QUAD_EXT_RPM_3][63] = 999.0f;

    platform_state_init(states, 0);
    platform_state_init(states, 63);

    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[0], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.vel_z[0], 0.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_3][0], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.quat_w[0], 1.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[63], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.vel_z[63], 0.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_3][63], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.quat_w[63], 1.0f);

    arena_destroy(arena);
    return 0;
}

TEST(init_does_not_affect_adjacent_indices) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    float sentinel = 777.0f;
    states->rigid_body.pos_x[4] = sentinel;
    states->rigid_body.pos_x[5] = sentinel;
    states->rigid_body.pos_x[6] = sentinel;
    states->rigid_body.vel_y[4] = sentinel;
    states->rigid_body.vel_y[5] = sentinel;
    states->rigid_body.vel_y[6] = sentinel;
    states->extension[QUAD_EXT_RPM_2][4] = sentinel;
    states->extension[QUAD_EXT_RPM_2][5] = sentinel;
    states->extension[QUAD_EXT_RPM_2][6] = sentinel;

    platform_state_init(states, 5);

    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[5], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.vel_y[5], 0.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_2][5], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[4], sentinel);
    ASSERT_FLOAT_EQ(states->rigid_body.vel_y[4], sentinel);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_2][4], sentinel);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[6], sentinel);
    ASSERT_FLOAT_EQ(states->rigid_body.vel_y[6], sentinel);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_2][6], sentinel);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 12: platform_params_init_flat Default Value Verification
 * ============================================================================ */

TEST(params_init_all_15_defaults) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformParamsSOA* params = platform_params_create(arena, 100, QUAD_PARAMS_EXT_COUNT);

    /* Rigid body defaults are set by platform_params_create() */
    PlatformParamsAoS p = platform_params_get(params, 0);

    ASSERT_FLOAT_EQ(p.mass, 0.5f);
    ASSERT_FLOAT_EQ(p.ixx, 0.0025f);
    ASSERT_FLOAT_EQ(p.iyy, 0.0025f);
    ASSERT_FLOAT_EQ(p.izz, 0.0045f);
    ASSERT_FLOAT_EQ(p.collision_radius, 0.15f);
    ASSERT_FLOAT_EQ(p.max_vel, 20.0f);
    ASSERT_FLOAT_EQ(p.max_omega, 10.0f);
    ASSERT_FLOAT_EQ(p.gravity, 9.81f);

    /* Quad-specific defaults require vtable init_params call */
    PLATFORM_QUADCOPTER.init_params(params->extension, params->extension_count, 0);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_ARM_LENGTH][0], 0.1f);
    ASSERT_FLOAT_NEAR(params->extension[QUAD_PEXT_K_THRUST][0], 3.16e-10f, 1e-12f);
    ASSERT_FLOAT_NEAR(params->extension[QUAD_PEXT_K_TORQUE][0], 7.94e-12f, 1e-14f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_K_DRAG][0], 0.1f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_K_ANG_DAMP][0], 0.01f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_MOTOR_TAU][0], 0.02f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_MAX_RPM][0], 2500.0f);

    arena_destroy(arena);
    return 0;
}

TEST(params_init_multiple_indices_independent) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformParamsSOA* params = platform_params_create(arena, 100, QUAD_PARAMS_EXT_COUNT);

    PlatformParamsAoS custom = {
        .mass = 2.0f, .ixx = 0.1f, .iyy = 0.2f, .izz = 0.3f,
        .collision_radius = 0.6f,
        .max_vel = 50.0f, .max_omega = 30.0f,
        .gravity = 3.71f
    };
    platform_params_set(params, 10, &custom);
    platform_params_init(params, 10);

    PlatformParamsAoS p10 = platform_params_get(params, 10);
    ASSERT_FLOAT_EQ(p10.mass, 0.5f);
    ASSERT_FLOAT_EQ(p10.gravity, 9.81f);

    PlatformParamsAoS p11 = platform_params_get(params, 11);
    ASSERT_FLOAT_EQ(p11.mass, 0.5f);
    ASSERT_FLOAT_EQ(p11.gravity, 9.81f);

    arena_destroy(arena);
    return 0;
}

TEST(params_null_arena_returns_null) {
    PlatformParamsSOA* params = platform_params_create(NULL, 100, QUAD_PARAMS_EXT_COUNT);
    ASSERT_NULL(params);
    return 0;
}

/* ============================================================================
 * Section 13: Validation Extended Edge Cases
 * ============================================================================ */

TEST(validation_nan_in_omega) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 10, QUAD_STATE_EXT_COUNT);

    states->rigid_body.omega_x[0] = NAN;
    ASSERT_FALSE(platform_state_validate(states, 0));

    platform_state_init(states, 0);
    states->rigid_body.omega_y[0] = NAN;
    ASSERT_FALSE(platform_state_validate(states, 0));

    platform_state_init(states, 0);
    states->rigid_body.omega_z[0] = NAN;
    ASSERT_FALSE(platform_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

TEST(validation_nan_in_rpm) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 10, QUAD_STATE_EXT_COUNT);

    states->extension[QUAD_EXT_RPM_0][0] = NAN;
    ASSERT_FALSE(platform_state_validate(states, 0));

    platform_state_init(states, 0);
    states->extension[QUAD_EXT_RPM_1][0] = NAN;
    ASSERT_FALSE(platform_state_validate(states, 0));

    platform_state_init(states, 0);
    states->extension[QUAD_EXT_RPM_2][0] = NAN;
    ASSERT_FALSE(platform_state_validate(states, 0));

    platform_state_init(states, 0);
    states->extension[QUAD_EXT_RPM_3][0] = NAN;
    ASSERT_FALSE(platform_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

TEST(validation_zero_quaternion) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 10, QUAD_STATE_EXT_COUNT);

    states->rigid_body.quat_w[0] = 0.0f;
    states->rigid_body.quat_x[0] = 0.0f;
    states->rigid_body.quat_y[0] = 0.0f;
    states->rigid_body.quat_z[0] = 0.0f;

    ASSERT_FALSE(platform_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

TEST(validation_after_init_always_passes) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    for (uint32_t i = 0; i < 100; i++) {
        states->rigid_body.quat_w[i] = 0.0f;
        states->extension[QUAD_EXT_RPM_0][i] = -1.0f;
        platform_state_init(states, i);
        ASSERT_TRUE(platform_state_validate(states, i));
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 14: Copy Edge Cases
 * ============================================================================ */

TEST(copy_count_zero_is_noop) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    PlatformStateSOA* src = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);
    PlatformStateSOA* dst = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    dst->rigid_body.pos_x[0] = 123.0f;
    rigid_body_state_copy(&dst->rigid_body, &src->rigid_body, 0, 0, 0);
    ASSERT_FLOAT_EQ(dst->rigid_body.pos_x[0], 123.0f);

    arena_destroy(arena);
    return 0;
}

TEST(copy_self_non_overlapping) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    for (uint32_t i = 0; i < 10; i++) {
        states->rigid_body.pos_x[i] = (float)(i + 1) * 11.0f;
        states->rigid_body.vel_y[i] = (float)(i + 1) * 22.0f;
        states->rigid_body.quat_w[i] = 1.0f;
        states->extension[QUAD_EXT_RPM_3][i] = (float)(i + 1) * 33.0f;
    }

    platform_state_copy(states, states, 50, 0, 10);

    for (uint32_t i = 0; i < 10; i++) {
        ASSERT_FLOAT_EQ(states->rigid_body.pos_x[50 + i], (float)(i + 1) * 11.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.vel_y[50 + i], (float)(i + 1) * 22.0f);
        ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_3][50 + i], (float)(i + 1) * 33.0f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(copy_preserves_all_17_arrays) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    PlatformStateSOA* src = platform_state_create(arena, 10, QUAD_STATE_EXT_COUNT);
    PlatformStateSOA* dst = platform_state_create(arena, 10, QUAD_STATE_EXT_COUNT);

    uint32_t idx = 3;
    src->rigid_body.pos_x[idx]  = 1.0f;
    src->rigid_body.pos_y[idx]  = 2.0f;
    src->rigid_body.pos_z[idx]  = 3.0f;
    src->rigid_body.vel_x[idx]  = 4.0f;
    src->rigid_body.vel_y[idx]  = 5.0f;
    src->rigid_body.vel_z[idx]  = 6.0f;
    src->rigid_body.quat_w[idx] = 7.0f;
    src->rigid_body.quat_x[idx] = 8.0f;
    src->rigid_body.quat_y[idx] = 9.0f;
    src->rigid_body.quat_z[idx] = 10.0f;
    src->rigid_body.omega_x[idx] = 11.0f;
    src->rigid_body.omega_y[idx] = 12.0f;
    src->rigid_body.omega_z[idx] = 13.0f;
    src->extension[QUAD_EXT_RPM_0][idx]  = 14.0f;
    src->extension[QUAD_EXT_RPM_1][idx]  = 15.0f;
    src->extension[QUAD_EXT_RPM_2][idx]  = 16.0f;
    src->extension[QUAD_EXT_RPM_3][idx]  = 17.0f;

    platform_state_copy(dst, src, 3, 3, 1);

    ASSERT_FLOAT_EQ(dst->rigid_body.pos_x[idx],  1.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.pos_y[idx],  2.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.pos_z[idx],  3.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.vel_x[idx],  4.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.vel_y[idx],  5.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.vel_z[idx],  6.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.quat_w[idx], 7.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.quat_x[idx], 8.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.quat_y[idx], 9.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.quat_z[idx], 10.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.omega_x[idx], 11.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.omega_y[idx], 12.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.omega_z[idx], 13.0f);
    ASSERT_FLOAT_EQ(dst->extension[QUAD_EXT_RPM_0][idx],  14.0f);
    ASSERT_FLOAT_EQ(dst->extension[QUAD_EXT_RPM_1][idx],  15.0f);
    ASSERT_FLOAT_EQ(dst->extension[QUAD_EXT_RPM_2][idx],  16.0f);
    ASSERT_FLOAT_EQ(dst->extension[QUAD_EXT_RPM_3][idx],  17.0f);

    arena_destroy(arena);
    return 0;
}

TEST(copy_overflow_is_safe_noop) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    PlatformStateSOA* src = platform_state_create(arena, 10, QUAD_STATE_EXT_COUNT);
    PlatformStateSOA* dst = platform_state_create(arena, 10, QUAD_STATE_EXT_COUNT);

    dst->rigid_body.pos_x[0] = 777.0f;
    rigid_body_state_copy(&dst->rigid_body, &src->rigid_body, 0, 5, 10);
    ASSERT_FLOAT_EQ(dst->rigid_body.pos_x[0], 777.0f);

    dst->rigid_body.pos_x[0] = 888.0f;
    rigid_body_state_copy(&dst->rigid_body, &src->rigid_body, 5, 0, 10);
    ASSERT_FLOAT_EQ(dst->rigid_body.pos_x[0], 888.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 15: Batch Reset Edge Cases
 * ============================================================================ */

TEST(reset_batch_count_zero_is_noop) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    states->rigid_body.pos_x[0] = 42.0f;
    uint32_t indices[1] = {0};
    Vec3 positions[1] = {VEC3(999.0f, 999.0f, 999.0f)};
    Quat orientations[1] = {QUAT_IDENTITY};

    rigid_body_state_reset_batch(&states->rigid_body, indices, positions, orientations, 0);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[0], 42.0f);

    arena_destroy(arena);
    return 0;
}

TEST(reset_batch_duplicate_index_last_wins) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    uint32_t indices[3] = {5, 5, 5};
    Vec3 positions[3] = {
        VEC3(1.0f, 1.0f, 1.0f),
        VEC3(2.0f, 2.0f, 2.0f),
        VEC3(3.0f, 3.0f, 3.0f)
    };
    Quat orientations[3] = {QUAT_IDENTITY, QUAT_IDENTITY, QUAT_IDENTITY};

    rigid_body_state_reset_batch(&states->rigid_body, indices, positions, orientations, 3);

    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[5], 3.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_y[5], 3.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_z[5], 3.0f);

    arena_destroy(arena);
    return 0;
}

TEST(reset_batch_non_identity_orientation) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 100, QUAD_STATE_EXT_COUNT);

    float cos_half = 0.9238795f;
    float sin_half = 0.3826834f;

    uint32_t indices[1] = {7};
    Vec3 positions[1] = {VEC3(10.0f, 20.0f, 30.0f)};
    Quat orientations[1] = {QUAT(cos_half, 0.0f, 0.0f, sin_half)};

    rigid_body_state_reset_batch(&states->rigid_body, indices, positions, orientations, 1);

    ASSERT_FLOAT_NEAR(states->rigid_body.quat_w[7], cos_half, 1e-6f);
    ASSERT_FLOAT_NEAR(states->rigid_body.quat_x[7], 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR(states->rigid_body.quat_y[7], 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR(states->rigid_body.quat_z[7], sin_half, 1e-6f);
    ASSERT_FLOAT_EQ(states->rigid_body.vel_x[7], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.vel_y[7], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.vel_z[7], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.omega_x[7], 0.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_0][7], 0.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 16: AgentEpisodeData Field Tests
 * ============================================================================ */

TEST(episode_accumulation) {
    Arena* arena = arena_create(1024 * 1024);
    AgentEpisodeData* episodes = agent_episode_create(arena, 10);

    episodes[3].episode_return += 1.5f;
    episodes[3].episode_length += 1;
    episodes[3].episode_return += 2.0f;
    episodes[3].episode_length += 1;
    episodes[3].episode_return += -0.5f;
    episodes[3].episode_length += 1;

    ASSERT_FLOAT_EQ(episodes[3].episode_return, 3.0f);
    ASSERT_EQ(episodes[3].episode_length, 3);

    arena_destroy(arena);
    return 0;
}

TEST(episode_best_return_tracking) {
    Arena* arena = arena_create(1024 * 1024);
    AgentEpisodeData* episodes = agent_episode_create(arena, 10);

    ASSERT_FLOAT_NEAR(episodes[0].best_episode_return, -1e30f, 1e25f);

    float ep1_return = 5.0f;
    if (ep1_return > episodes[0].best_episode_return) {
        episodes[0].best_episode_return = ep1_return;
    }
    ASSERT_FLOAT_EQ(episodes[0].best_episode_return, 5.0f);

    agent_episode_init(episodes, 0, 0, 0);
    ASSERT_FLOAT_NEAR(episodes[0].best_episode_return, -1e30f, 1e25f);

    arena_destroy(arena);
    return 0;
}

TEST(episode_total_episodes_counter) {
    Arena* arena = arena_create(1024 * 1024);
    AgentEpisodeData* episodes = agent_episode_create(arena, 10);

    ASSERT_EQ(episodes[5].total_episodes, 0);
    episodes[5].total_episodes++;
    ASSERT_EQ(episodes[5].total_episodes, 1);
    episodes[5].total_episodes++;
    ASSERT_EQ(episodes[5].total_episodes, 2);

    agent_episode_init(episodes, 5, 0, 5);
    ASSERT_EQ(episodes[5].total_episodes, 0);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 17: Params Accessor Extended
 * ============================================================================ */

TEST(params_set_get_boundary_indices) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformParamsSOA* params = platform_params_create(arena, 50, QUAD_PARAMS_EXT_COUNT);

    PlatformParamsAoS custom = {
        .mass = 3.0f, .ixx = 0.05f, .iyy = 0.06f, .izz = 0.07f,
        .collision_radius = 0.35f,
        .max_vel = 30.0f, .max_omega = 20.0f,
        .gravity = 1.62f
    };

    platform_params_set(params, 0, &custom);
    PlatformParamsAoS r0 = platform_params_get(params, 0);
    ASSERT_FLOAT_EQ(r0.mass, 3.0f);
    ASSERT_FLOAT_EQ(r0.gravity, 1.62f);

    custom.mass = 7.0f;
    custom.gravity = 24.79f;
    platform_params_set(params, 49, &custom);
    PlatformParamsAoS r49 = platform_params_get(params, 49);
    ASSERT_FLOAT_EQ(r49.mass, 7.0f);
    ASSERT_FLOAT_EQ(r49.gravity, 24.79f);

    PlatformParamsAoS r25 = platform_params_get(params, 25);
    ASSERT_FLOAT_EQ(r25.mass, 0.5f);
    ASSERT_FLOAT_EQ(r25.gravity, 9.81f);

    arena_destroy(arena);
    return 0;
}

TEST(params_zero_capacity_returns_null) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformParamsSOA* params = platform_params_create(arena, 0, QUAD_PARAMS_EXT_COUNT);
    ASSERT_NULL(params);
    arena_destroy(arena);
    return 0;
}

TEST(params_all_15_fields_roundtrip) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformParamsSOA* params = platform_params_create(arena, 10, QUAD_PARAMS_EXT_COUNT);

    PlatformParamsAoS custom = {
        .mass = 1.234f, .ixx = 0.011f, .iyy = 0.022f, .izz = 0.033f,
        .collision_radius = 0.345f,
        .max_vel = 27.5f, .max_omega = 17.3f,
        .gravity = 3.71f
    };

    platform_params_set(params, 7, &custom);

    /* Set quad-specific extensions directly */
    params->extension[QUAD_PEXT_ARM_LENGTH][7] = 0.234f;
    params->extension[QUAD_PEXT_K_THRUST][7] = 4.56e-10f;
    params->extension[QUAD_PEXT_K_TORQUE][7] = 7.89e-12f;
    params->extension[QUAD_PEXT_K_DRAG][7] = 0.234f;
    params->extension[QUAD_PEXT_K_ANG_DAMP][7] = 0.0234f;
    params->extension[QUAD_PEXT_MOTOR_TAU][7] = 0.0345f;
    params->extension[QUAD_PEXT_MAX_RPM][7] = 3456.0f;

    PlatformParamsAoS result = platform_params_get(params, 7);

    /* Rigid body fields */
    ASSERT_FLOAT_EQ(result.mass, 1.234f);
    ASSERT_FLOAT_EQ(result.ixx, 0.011f);
    ASSERT_FLOAT_EQ(result.iyy, 0.022f);
    ASSERT_FLOAT_EQ(result.izz, 0.033f);
    ASSERT_FLOAT_EQ(result.collision_radius, 0.345f);
    ASSERT_FLOAT_EQ(result.max_vel, 27.5f);
    ASSERT_FLOAT_EQ(result.max_omega, 17.3f);
    ASSERT_FLOAT_EQ(result.gravity, 3.71f);

    /* Quad-specific fields via extensions */
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_ARM_LENGTH][7], 0.234f);
    ASSERT_FLOAT_NEAR(params->extension[QUAD_PEXT_K_THRUST][7], 4.56e-10f, 1e-12f);
    ASSERT_FLOAT_NEAR(params->extension[QUAD_PEXT_K_TORQUE][7], 7.89e-12f, 1e-14f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_K_DRAG][7], 0.234f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_K_ANG_DAMP][7], 0.0234f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_MOTOR_TAU][7], 0.0345f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_MAX_RPM][7], 3456.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 18: Memory Size Consistency
 * ============================================================================ */

TEST(memory_size_grows_linearly) {
    size_t size_1000 = platform_state_memory_size(1000, QUAD_STATE_EXT_COUNT);
    size_t size_2000 = platform_state_memory_size(2000, QUAD_STATE_EXT_COUNT);

    ASSERT_TRUE(size_1000 > 0);
    ASSERT_TRUE(size_2000 > size_1000);

    double ratio = (double)size_2000 / (double)size_1000;
    ASSERT_TRUE(ratio > 1.9);
    ASSERT_TRUE(ratio < 2.1);

    size_t psize_1000 = platform_params_memory_size(1000, QUAD_PARAMS_EXT_COUNT);
    size_t psize_2000 = platform_params_memory_size(2000, QUAD_PARAMS_EXT_COUNT);
    double pratio = (double)psize_2000 / (double)psize_1000;
    ASSERT_TRUE(pratio > 1.9);
    ASSERT_TRUE(pratio < 2.1);

    return 0;
}

TEST(memory_size_fits_actual_allocation) {
    uint32_t cap = 512;
    size_t needed = platform_state_memory_size(cap, QUAD_STATE_EXT_COUNT);

    Arena* arena = arena_create(needed + 4096);
    ASSERT_NOT_NULL(arena);

    PlatformStateSOA* states = platform_state_create(arena, cap, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);
    ASSERT_EQ(states->rigid_body.capacity, cap);

    states->rigid_body.pos_x[cap - 1] = 42.0f;
    states->extension[QUAD_EXT_RPM_3][cap - 1] = 99.0f;
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[cap - 1], 42.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_3][cap - 1], 99.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Drone State Module Tests");

    /* Allocation tests */
    RUN_TEST(allocation_basic);
    RUN_TEST(allocation_alignment_32byte);
    RUN_TEST(allocation_capacity_1);
    RUN_TEST(allocation_capacity_1024);
    RUN_TEST(allocation_capacity_10000);
    RUN_TEST(allocation_zero_capacity);
    RUN_TEST(allocation_arena_overflow);

    /* Zero operation tests */
    RUN_TEST(zero_all_positions);
    RUN_TEST(zero_all_velocities);
    RUN_TEST(zero_identity_quaternion);
    RUN_TEST(zero_all_rpms);
    RUN_TEST(zero_all_angular_velocity);

    /* Accessor tests */
    RUN_TEST(accessor_set_get_position);
    RUN_TEST(accessor_set_get_velocity);
    RUN_TEST(accessor_set_get_orientation);
    RUN_TEST(accessor_set_get_omega);
    RUN_TEST(accessor_set_get_rpms);
    RUN_TEST(accessor_all_fields_roundtrip);

    /* Batch reset tests */
    RUN_TEST(reset_batch_single_index);
    RUN_TEST(reset_batch_multiple_indices);
    RUN_TEST(reset_batch_verify_untouched);
    RUN_TEST(reset_batch_scattered_pattern);

    /* Copy tests */
    RUN_TEST(copy_full_range);
    RUN_TEST(copy_partial_range);
    RUN_TEST(copy_verify_independence);

    /* Validation tests */
    RUN_TEST(validation_valid_state);
    RUN_TEST(validation_nan_position);
    RUN_TEST(validation_nan_velocity);
    RUN_TEST(validation_nan_quaternion);
    RUN_TEST(validation_unnormalized_quat);
    RUN_TEST(validation_negative_rpm);

    /* Boundary tests */
    RUN_TEST(boundary_index_zero);
    RUN_TEST(boundary_index_max);

    /* Memory size tests */
    RUN_TEST(memory_size_state);
    RUN_TEST(memory_size_params);

    /* Episode tests */
    RUN_TEST(episode_create);
    RUN_TEST(episode_init);
    RUN_TEST(episode_size);

    /* Params tests */
    RUN_TEST(params_create);
    RUN_TEST(params_alignment);
    RUN_TEST(params_accessor_roundtrip);

    /* platform_state_init default value verification */
    RUN_TEST(init_default_values_all_fields);
    RUN_TEST(init_overwrites_previous_values);
    RUN_TEST(init_boundary_indices);
    RUN_TEST(init_does_not_affect_adjacent_indices);

    /* platform_params_init_flat default value verification */
    RUN_TEST(params_init_all_15_defaults);
    RUN_TEST(params_init_multiple_indices_independent);
    RUN_TEST(params_null_arena_returns_null);

    /* Validation extended edge cases */
    RUN_TEST(validation_nan_in_omega);
    RUN_TEST(validation_nan_in_rpm);
    RUN_TEST(validation_zero_quaternion);
    RUN_TEST(validation_after_init_always_passes);

    /* Copy edge cases */
    RUN_TEST(copy_count_zero_is_noop);
    RUN_TEST(copy_self_non_overlapping);
    RUN_TEST(copy_preserves_all_17_arrays);
    RUN_TEST(copy_overflow_is_safe_noop);

    /* Batch reset edge cases */
    RUN_TEST(reset_batch_count_zero_is_noop);
    RUN_TEST(reset_batch_duplicate_index_last_wins);
    RUN_TEST(reset_batch_non_identity_orientation);

    /* Episode data field tests */
    RUN_TEST(episode_accumulation);
    RUN_TEST(episode_best_return_tracking);
    RUN_TEST(episode_total_episodes_counter);

    /* Params accessor extended */
    RUN_TEST(params_set_get_boundary_indices);
    RUN_TEST(params_zero_capacity_returns_null);
    RUN_TEST(params_all_15_fields_roundtrip);

    /* Memory size consistency */
    RUN_TEST(memory_size_grows_linearly);
    RUN_TEST(memory_size_fits_actual_allocation);

    TEST_SUITE_END();
}
