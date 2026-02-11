/**
 * Drone State Module Unit Tests
 *
 * Yoneda Philosophy: Tests serve as complete behavioral specification.
 * Each test defines what the module MUST do - the definitive contract.
 */

#include "../include/drone_state.h"
#include "test_harness.h"

/* ============================================================================
 * Section 1: Allocation Tests
 * ============================================================================ */

TEST(allocation_basic) {
    Arena* arena = arena_create(1024 * 1024);  /* 1MB */
    ASSERT_NOT_NULL(arena);

    DroneStateSOA* states = drone_state_create(arena, 100);
    ASSERT_NOT_NULL(states);
    ASSERT_EQ(states->capacity, 100);
    ASSERT_EQ(states->count, 0);

    /* All array pointers should be valid */
    ASSERT_NOT_NULL(states->pos_x);
    ASSERT_NOT_NULL(states->pos_y);
    ASSERT_NOT_NULL(states->pos_z);
    ASSERT_NOT_NULL(states->quat_w);
    ASSERT_NOT_NULL(states->rpm_3);

    arena_destroy(arena);
    return 0;
}

TEST(allocation_alignment_32byte) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 256);
    ASSERT_NOT_NULL(states);

    /* All float arrays must be 32-byte aligned for AVX2 */
    ASSERT_TRUE(((uintptr_t)(states->pos_x) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->pos_y) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->pos_z) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->vel_x) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->vel_y) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->vel_z) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->quat_w) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->quat_x) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->quat_y) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->quat_z) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->omega_x) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->omega_y) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->omega_z) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->rpm_0) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->rpm_1) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->rpm_2) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(states->rpm_3) & (32 - 1)) == 0);

    arena_destroy(arena);
    return 0;
}

TEST(allocation_capacity_1) {
    Arena* arena = arena_create(64 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 1);
    ASSERT_NOT_NULL(states);
    ASSERT_EQ(states->capacity, 1);

    /* Should still be 32-byte aligned */
    ASSERT_TRUE(((uintptr_t)(states->pos_x) & (32 - 1)) == 0);

    arena_destroy(arena);
    return 0;
}

TEST(allocation_capacity_1024) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 1024);
    ASSERT_NOT_NULL(states);
    ASSERT_EQ(states->capacity, 1024);

    /* Verify memory fits expectations (~68KB for state) */
    size_t expected_size = drone_state_memory_size(1024);
    ASSERT_TRUE(expected_size <= 80 * 1024);

    arena_destroy(arena);
    return 0;
}

TEST(allocation_capacity_10000) {
    Arena* arena = arena_create(10 * 1024 * 1024);  /* 10MB */
    DroneStateSOA* states = drone_state_create(arena, 10000);
    ASSERT_NOT_NULL(states);
    ASSERT_EQ(states->capacity, 10000);

    arena_destroy(arena);
    return 0;
}

TEST(allocation_zero_capacity) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 0);
    ASSERT_NULL(states);

    arena_destroy(arena);
    return 0;
}

TEST(allocation_arena_overflow) {
    /* Tiny arena that can't fit allocation */
    Arena* arena = arena_create(64);  /* Too small for even 1 drone */
    DroneStateSOA* states = drone_state_create(arena, 1000);
    ASSERT_NULL(states);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 2: Zero Operation Tests
 * ============================================================================ */

TEST(zero_all_positions) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 100);

    /* Manually set non-zero values */
    for (uint32_t i = 0; i < states->capacity; i++) {
        states->pos_x[i] = 1.0f + (float)i;
        states->pos_y[i] = 2.0f + (float)i;
        states->pos_z[i] = 3.0f + (float)i;
    }

    /* Zero should reset */
    drone_state_zero(states);

    for (uint32_t i = 0; i < states->capacity; i++) {
        ASSERT_FLOAT_EQ(states->pos_x[i], 0.0f);
        ASSERT_FLOAT_EQ(states->pos_y[i], 0.0f);
        ASSERT_FLOAT_EQ(states->pos_z[i], 0.0f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(zero_all_velocities) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 100);

    for (uint32_t i = 0; i < states->capacity; i++) {
        states->vel_x[i] = 5.0f;
        states->vel_y[i] = 6.0f;
        states->vel_z[i] = 7.0f;
    }

    drone_state_zero(states);

    for (uint32_t i = 0; i < states->capacity; i++) {
        ASSERT_FLOAT_EQ(states->vel_x[i], 0.0f);
        ASSERT_FLOAT_EQ(states->vel_y[i], 0.0f);
        ASSERT_FLOAT_EQ(states->vel_z[i], 0.0f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(zero_identity_quaternion) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 100);

    /* Set non-identity quaternions */
    for (uint32_t i = 0; i < states->capacity; i++) {
        states->quat_w[i] = 0.5f;
        states->quat_x[i] = 0.5f;
        states->quat_y[i] = 0.5f;
        states->quat_z[i] = 0.5f;
    }

    drone_state_zero(states);

    /* Identity quaternion: w=1, x=y=z=0 */
    for (uint32_t i = 0; i < states->capacity; i++) {
        ASSERT_FLOAT_EQ(states->quat_w[i], 1.0f);
        ASSERT_FLOAT_EQ(states->quat_x[i], 0.0f);
        ASSERT_FLOAT_EQ(states->quat_y[i], 0.0f);
        ASSERT_FLOAT_EQ(states->quat_z[i], 0.0f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(zero_all_rpms) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 100);

    for (uint32_t i = 0; i < states->capacity; i++) {
        states->rpm_0[i] = 1000.0f;
        states->rpm_1[i] = 1100.0f;
        states->rpm_2[i] = 1200.0f;
        states->rpm_3[i] = 1300.0f;
    }

    drone_state_zero(states);

    for (uint32_t i = 0; i < states->capacity; i++) {
        ASSERT_FLOAT_EQ(states->rpm_0[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rpm_1[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rpm_2[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rpm_3[i], 0.0f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(zero_all_angular_velocity) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 100);

    for (uint32_t i = 0; i < states->capacity; i++) {
        states->omega_x[i] = 1.5f;
        states->omega_y[i] = 2.5f;
        states->omega_z[i] = 3.5f;
    }

    drone_state_zero(states);

    for (uint32_t i = 0; i < states->capacity; i++) {
        ASSERT_FLOAT_EQ(states->omega_x[i], 0.0f);
        ASSERT_FLOAT_EQ(states->omega_y[i], 0.0f);
        ASSERT_FLOAT_EQ(states->omega_z[i], 0.0f);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 3: Accessor Round-Trip Tests
 * ============================================================================ */

TEST(accessor_set_get_position) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 100);

    DroneStateAoS state = {0};
    state.position = VEC3(1.5f, 2.5f, 3.5f);
    state.orientation = QUAT_IDENTITY;

    drone_state_set(states, 42, &state);
    DroneStateAoS result = drone_state_get(states, 42);

    ASSERT_FLOAT_EQ(result.position.x, 1.5f);
    ASSERT_FLOAT_EQ(result.position.y, 2.5f);
    ASSERT_FLOAT_EQ(result.position.z, 3.5f);

    arena_destroy(arena);
    return 0;
}

TEST(accessor_set_get_velocity) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 100);

    DroneStateAoS state = {0};
    state.velocity = VEC3(-1.0f, 2.0f, -3.0f);
    state.orientation = QUAT_IDENTITY;

    drone_state_set(states, 25, &state);
    DroneStateAoS result = drone_state_get(states, 25);

    ASSERT_FLOAT_EQ(result.velocity.x, -1.0f);
    ASSERT_FLOAT_EQ(result.velocity.y, 2.0f);
    ASSERT_FLOAT_EQ(result.velocity.z, -3.0f);

    arena_destroy(arena);
    return 0;
}

TEST(accessor_set_get_orientation) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 100);

    /* 90 degree rotation around Z axis (approximately) */
    DroneStateAoS state = {0};
    state.orientation = QUAT(0.7071f, 0.0f, 0.0f, 0.7071f);

    drone_state_set(states, 10, &state);
    DroneStateAoS result = drone_state_get(states, 10);

    ASSERT_FLOAT_EQ(result.orientation.w, 0.7071f);
    ASSERT_FLOAT_EQ(result.orientation.x, 0.0f);
    ASSERT_FLOAT_EQ(result.orientation.y, 0.0f);
    ASSERT_FLOAT_EQ(result.orientation.z, 0.7071f);

    arena_destroy(arena);
    return 0;
}

TEST(accessor_set_get_omega) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 100);

    DroneStateAoS state = {0};
    state.omega = VEC3(0.1f, 0.2f, 0.3f);
    state.orientation = QUAT_IDENTITY;

    drone_state_set(states, 5, &state);
    DroneStateAoS result = drone_state_get(states, 5);

    ASSERT_FLOAT_EQ(result.omega.x, 0.1f);
    ASSERT_FLOAT_EQ(result.omega.y, 0.2f);
    ASSERT_FLOAT_EQ(result.omega.z, 0.3f);

    arena_destroy(arena);
    return 0;
}

TEST(accessor_set_get_rpms) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 100);

    DroneStateAoS state = {0};
    state.rpm[0] = 1000.0f;
    state.rpm[1] = 1100.0f;
    state.rpm[2] = 1200.0f;
    state.rpm[3] = 1300.0f;
    state.orientation = QUAT_IDENTITY;

    drone_state_set(states, 77, &state);
    DroneStateAoS result = drone_state_get(states, 77);

    ASSERT_FLOAT_EQ(result.rpm[0], 1000.0f);
    ASSERT_FLOAT_EQ(result.rpm[1], 1100.0f);
    ASSERT_FLOAT_EQ(result.rpm[2], 1200.0f);
    ASSERT_FLOAT_EQ(result.rpm[3], 1300.0f);

    arena_destroy(arena);
    return 0;
}

TEST(accessor_all_fields_roundtrip) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 100);

    DroneStateAoS original = {
        .position = VEC3(1.0f, 2.0f, 3.0f),
        .velocity = VEC3(4.0f, 5.0f, 6.0f),
        .orientation = QUAT(0.5f, 0.5f, 0.5f, 0.5f),
        .omega = VEC3(0.1f, 0.2f, 0.3f),
        .rpm = {100.0f, 200.0f, 300.0f, 400.0f}
    };

    drone_state_set(states, 33, &original);
    DroneStateAoS result = drone_state_get(states, 33);

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
    ASSERT_FLOAT_EQ(result.rpm[0], original.rpm[0]);
    ASSERT_FLOAT_EQ(result.rpm[1], original.rpm[1]);
    ASSERT_FLOAT_EQ(result.rpm[2], original.rpm[2]);
    ASSERT_FLOAT_EQ(result.rpm[3], original.rpm[3]);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 4: Batch Reset Tests
 * ============================================================================ */

TEST(reset_batch_single_index) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 100);

    /* Set some values */
    for (uint32_t i = 0; i < states->capacity; i++) {
        states->pos_x[i] = (float)i;
        states->vel_x[i] = (float)i * 2.0f;
    }

    /* Reset single index */
    uint32_t indices[1] = {50};
    Vec3 positions[1] = {VEC3(10.0f, 20.0f, 30.0f)};
    Quat orientations[1] = {QUAT_IDENTITY};

    drone_state_reset_batch(states, indices, positions, orientations, 1);

    ASSERT_FLOAT_EQ(states->pos_x[50], 10.0f);
    ASSERT_FLOAT_EQ(states->pos_y[50], 20.0f);
    ASSERT_FLOAT_EQ(states->pos_z[50], 30.0f);
    ASSERT_FLOAT_EQ(states->vel_x[50], 0.0f);
    ASSERT_FLOAT_EQ(states->quat_w[50], 1.0f);

    /* Adjacent indices should be unchanged */
    ASSERT_FLOAT_EQ(states->pos_x[49], 49.0f);
    ASSERT_FLOAT_EQ(states->pos_x[51], 51.0f);

    arena_destroy(arena);
    return 0;
}

TEST(reset_batch_multiple_indices) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 100);

    uint32_t indices[3] = {0, 50, 99};
    Vec3 positions[3] = {
        VEC3(1.0f, 1.0f, 1.0f),
        VEC3(2.0f, 2.0f, 2.0f),
        VEC3(3.0f, 3.0f, 3.0f)
    };
    Quat orientations[3] = {QUAT_IDENTITY, QUAT_IDENTITY, QUAT_IDENTITY};

    drone_state_reset_batch(states, indices, positions, orientations, 3);

    ASSERT_FLOAT_EQ(states->pos_x[0], 1.0f);
    ASSERT_FLOAT_EQ(states->pos_x[50], 2.0f);
    ASSERT_FLOAT_EQ(states->pos_x[99], 3.0f);

    arena_destroy(arena);
    return 0;
}

TEST(reset_batch_verify_untouched) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 100);

    /* Set recognizable pattern */
    for (uint32_t i = 0; i < states->capacity; i++) {
        states->pos_x[i] = (float)(i + 1000);
    }

    uint32_t indices[2] = {25, 75};
    Vec3 positions[2] = {VEC3(0.0f, 0.0f, 0.0f), VEC3(0.0f, 0.0f, 0.0f)};
    Quat orientations[2] = {QUAT_IDENTITY, QUAT_IDENTITY};

    drone_state_reset_batch(states, indices, positions, orientations, 2);

    /* Verify untouched */
    ASSERT_FLOAT_EQ(states->pos_x[0], 1000.0f);
    ASSERT_FLOAT_EQ(states->pos_x[24], 1024.0f);
    ASSERT_FLOAT_EQ(states->pos_x[26], 1026.0f);
    ASSERT_FLOAT_EQ(states->pos_x[99], 1099.0f);

    arena_destroy(arena);
    return 0;
}

TEST(reset_batch_scattered_pattern) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 1000);

    /* Scattered indices simulating random done environments */
    uint32_t indices[10] = {7, 103, 256, 512, 513, 600, 777, 888, 901, 999};
    Vec3 positions[10];
    Quat orientations[10];

    for (int i = 0; i < 10; i++) {
        positions[i] = VEC3((float)i, (float)i, (float)i);
        orientations[i] = QUAT_IDENTITY;
    }

    drone_state_reset_batch(states, indices, positions, orientations, 10);

    for (int i = 0; i < 10; i++) {
        ASSERT_FLOAT_EQ(states->pos_x[indices[i]], (float)i);
        ASSERT_FLOAT_EQ(states->vel_x[indices[i]], 0.0f);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 5: Copy Tests
 * ============================================================================ */

TEST(copy_full_range) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    DroneStateSOA* src = drone_state_create(arena, 100);
    DroneStateSOA* dst = drone_state_create(arena, 100);

    /* Set source values */
    for (uint32_t i = 0; i < src->capacity; i++) {
        src->pos_x[i] = (float)i;
        src->pos_y[i] = (float)i * 2;
        src->quat_w[i] = 1.0f;
    }

    drone_state_copy(dst, src, 0, 0, 100);

    for (uint32_t i = 0; i < 100; i++) {
        ASSERT_FLOAT_EQ(dst->pos_x[i], (float)i);
        ASSERT_FLOAT_EQ(dst->pos_y[i], (float)i * 2);
    }

    arena_destroy(arena);
    return 0;
}

TEST(copy_partial_range) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    DroneStateSOA* src = drone_state_create(arena, 100);
    DroneStateSOA* dst = drone_state_create(arena, 100);

    /* Set source values */
    for (uint32_t i = 0; i < src->capacity; i++) {
        src->pos_x[i] = (float)(i + 500);
    }

    /* Copy subset: src[10..20] -> dst[50..60] */
    drone_state_copy(dst, src, 50, 10, 10);

    for (uint32_t i = 0; i < 10; i++) {
        ASSERT_FLOAT_EQ(dst->pos_x[50 + i], (float)(510 + i));
    }

    /* Verify areas outside copy range are still zero */
    ASSERT_FLOAT_EQ(dst->pos_x[0], 0.0f);
    ASSERT_FLOAT_EQ(dst->pos_x[49], 0.0f);
    ASSERT_FLOAT_EQ(dst->pos_x[60], 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(copy_verify_independence) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    DroneStateSOA* src = drone_state_create(arena, 100);
    DroneStateSOA* dst = drone_state_create(arena, 100);

    for (uint32_t i = 0; i < src->capacity; i++) {
        src->pos_x[i] = 42.0f;
    }

    drone_state_copy(dst, src, 0, 0, 100);

    /* Modify source after copy */
    for (uint32_t i = 0; i < src->capacity; i++) {
        src->pos_x[i] = 999.0f;
    }

    /* Destination should be unaffected */
    for (uint32_t i = 0; i < 100; i++) {
        ASSERT_FLOAT_EQ(dst->pos_x[i], 42.0f);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 6: Validation Tests
 * ============================================================================ */

TEST(validation_valid_state) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 100);

    /* Fresh state should be valid */
    ASSERT_TRUE(drone_state_validate(states, 0));
    ASSERT_TRUE(drone_state_validate(states, 50));
    ASSERT_TRUE(drone_state_validate(states, 99));

    arena_destroy(arena);
    return 0;
}

TEST(validation_nan_position) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 100);

    states->pos_x[0] = NAN;
    ASSERT_FALSE(drone_state_validate(states, 0));

    drone_state_init(states, 0);
    states->pos_y[0] = NAN;
    ASSERT_FALSE(drone_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

TEST(validation_nan_velocity) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 100);

    states->vel_x[0] = NAN;
    ASSERT_FALSE(drone_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

TEST(validation_nan_quaternion) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 100);

    states->quat_w[0] = NAN;
    ASSERT_FALSE(drone_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

TEST(validation_unnormalized_quat) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 100);

    /* Unnormalized quaternion: |q|^2 = 4, not 1 */
    states->quat_w[0] = 1.0f;
    states->quat_x[0] = 1.0f;
    states->quat_y[0] = 1.0f;
    states->quat_z[0] = 1.0f;

    ASSERT_FALSE(drone_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

TEST(validation_negative_rpm) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 100);

    states->rpm_0[0] = -100.0f;
    ASSERT_FALSE(drone_state_validate(states, 0));

    drone_state_init(states, 0);
    states->rpm_2[0] = -1.0f;
    ASSERT_FALSE(drone_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 7: Boundary Tests
 * ============================================================================ */

TEST(boundary_index_zero) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 100);

    DroneStateAoS state = {
        .position = VEC3(1.0f, 2.0f, 3.0f),
        .orientation = QUAT_IDENTITY
    };

    drone_state_set(states, 0, &state);
    DroneStateAoS result = drone_state_get(states, 0);

    ASSERT_FLOAT_EQ(result.position.x, 1.0f);

    arena_destroy(arena);
    return 0;
}

TEST(boundary_index_max) {
    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, 100);

    DroneStateAoS state = {
        .position = VEC3(99.0f, 99.0f, 99.0f),
        .orientation = QUAT_IDENTITY
    };

    drone_state_set(states, 99, &state);
    DroneStateAoS result = drone_state_get(states, 99);

    ASSERT_FLOAT_EQ(result.position.x, 99.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 8: Memory Size Tests
 * ============================================================================ */

TEST(memory_size_state) {
    size_t size = drone_state_memory_size(1024);
    ASSERT_TRUE(size > 0);
    ASSERT_TRUE(size <= 80 * 1024);
    ASSERT_TRUE(size >= 68 * 1024);

    size_t size_1 = drone_state_memory_size(1);
    ASSERT_TRUE(size_1 > 0);

    return 0;
}

TEST(memory_size_params) {
    size_t size = drone_params_memory_size(1024);
    ASSERT_TRUE(size > 0);
    ASSERT_TRUE(size <= 70 * 1024);

    return 0;
}

/* ============================================================================
 * Section 9: Episode Data Tests
 * ============================================================================ */

TEST(episode_create) {
    Arena* arena = arena_create(1024 * 1024);
    DroneEpisodeData* episodes = drone_episode_create(arena, 100);
    ASSERT_NOT_NULL(episodes);

    arena_destroy(arena);
    return 0;
}

TEST(episode_init) {
    Arena* arena = arena_create(1024 * 1024);
    DroneEpisodeData* episodes = drone_episode_create(arena, 100);

    drone_episode_init(episodes, 42, 10, 5);

    ASSERT_FLOAT_EQ(episodes[42].episode_return, 0.0f);
    ASSERT_EQ(episodes[42].episode_length, 0);
    ASSERT_EQ(episodes[42].env_id, 10);
    ASSERT_EQ(episodes[42].drone_id, 5);
    ASSERT_EQ(episodes[42].done, 0);
    ASSERT_EQ(episodes[42].truncated, 0);

    arena_destroy(arena);
    return 0;
}

TEST(episode_size) {
    ASSERT_EQ((int)sizeof(DroneEpisodeData), 28);
    return 0;
}

/* ============================================================================
 * Section 10: Params Tests
 * ============================================================================ */

TEST(params_create) {
    Arena* arena = arena_create(1024 * 1024);
    DroneParamsSOA* params = drone_params_create(arena, 100);
    ASSERT_NOT_NULL(params);
    ASSERT_EQ(params->capacity, 100);

    /* Check default values */
    ASSERT_FLOAT_EQ(params->mass[0], 0.5f);
    ASSERT_FLOAT_EQ(params->gravity[0], 9.81f);

    arena_destroy(arena);
    return 0;
}

TEST(params_alignment) {
    Arena* arena = arena_create(1024 * 1024);
    DroneParamsSOA* params = drone_params_create(arena, 256);
    ASSERT_NOT_NULL(params);

    ASSERT_TRUE(((uintptr_t)(params->mass) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(params->ixx) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(params->gravity) & (32 - 1)) == 0);

    arena_destroy(arena);
    return 0;
}

TEST(params_accessor_roundtrip) {
    Arena* arena = arena_create(1024 * 1024);
    DroneParamsSOA* params = drone_params_create(arena, 100);

    DroneParamsAoS original = {
        .mass = 1.5f,
        .ixx = 0.01f,
        .iyy = 0.02f,
        .izz = 0.03f,
        .arm_length = 0.2f,
        .collision_radius = 0.25f,
        .k_thrust = 1e-9f,
        .k_torque = 1e-11f,
        .k_drag = 0.2f,
        .k_ang_damp = 0.02f,
        .motor_tau = 0.03f,
        .max_rpm = 3000.0f,
        .max_vel = 25.0f,
        .max_omega = 15.0f,
        .gravity = 10.0f
    };

    drone_params_set(params, 50, &original);
    DroneParamsAoS result = drone_params_get(params, 50);

    ASSERT_FLOAT_EQ(result.mass, 1.5f);
    ASSERT_FLOAT_EQ(result.ixx, 0.01f);
    ASSERT_FLOAT_EQ(result.gravity, 10.0f);

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

    TEST_SUITE_END();
}
