/**
 * Drone State Module Deep Tests -- Scale, Boundary, Copy Semantics
 *
 * Yoneda Philosophy: These tests define the complete behavioral contract of the
 * drone_state module so thoroughly that any implementation passing all tests must
 * be functionally equivalent to the original. We explore every possible interaction,
 * input, and edge case to characterize the module completely.
 *
 * Coverage categories:
 *   1. Scale tests: capacity=0, 1, 4096 -- allocation, init, operations at extremes
 *   2. Boundary tests: first/last valid index, first invalid index
 *   3. Copy semantics: offsets, count=0, all-field fidelity, independence
 *   4. Reset semantics: count=0 no-op, count=capacity full reset, custom orientations
 *   5. Validation: NaN, Inf, zero quaternion, near-unit quaternion, negative RPMs
 *   6. Alignment: all 17+15 array pointers verified 32-byte aligned
 *   7. Zero+validate interaction: zeroed state must pass validation
 *   8. Init correctness: every field checked against documented defaults
 *   9. PlatformParamsSOA: create, init, accessor round-trip at multiple capacities
 *  10. Episode data: init, field correctness, done/truncated transitions
 *  11. Memory size: monotonicity, zero capacity, consistency with actual allocation
 *
 * Build note: This test is compiled WITHOUT FOUNDATION_DEBUG=1 because several
 * tests exercise out-of-bounds / NULL paths that rely on graceful early-return
 * fallback behavior. With FOUNDATION_DEBUG, FOUNDATION_ASSERT calls abort()
 * before the fallback code is reached, crashing the test runner.
 */

#include "../include/drone_state.h"
#include "platform_quadcopter.h"
#include "test_harness.h"

/* ============================================================================
 * Section 1: Create at Extreme Capacities
 * ============================================================================ */

/**
 * capacity=0 must return NULL. The implementation explicitly checks for this.
 * Also verify that platform_state_memory_size(0, QUAD_STATE_EXT_COUNT) returns 0 (not a garbage value).
 */
TEST(create_capacity_zero_returns_null) {
    Arena* arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    PlatformStateSOA* states = platform_state_create(arena, 0, QUAD_STATE_EXT_COUNT);
    ASSERT_NULL(states);

    PlatformParamsSOA* params = platform_params_create(arena, 0, QUAD_PARAMS_EXT_COUNT);
    ASSERT_NULL(params);

    AgentEpisodeData* episodes = agent_episode_create(arena, 0);
    ASSERT_NULL(episodes);

    arena_destroy(arena);
    return 0;
}

TEST(memory_size_zero_capacity_returns_zero) {
    size_t state_size = platform_state_memory_size(0, QUAD_STATE_EXT_COUNT);
    ASSERT_EQ(state_size, (size_t)0);

    size_t params_size = platform_params_memory_size(0, QUAD_PARAMS_EXT_COUNT);
    ASSERT_EQ(params_size, (size_t)0);

    return 0;
}

/**
 * capacity=1 is the minimal valid allocation.
 * All operations (init, copy, reset, validate, get/set) must work correctly.
 */
TEST(create_capacity_one_all_operations) {
    Arena* arena = arena_create(64 * 1024);
    ASSERT_NOT_NULL(arena);

    PlatformStateSOA* states = platform_state_create(arena, 1, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);
    ASSERT_EQ(states->rigid_body.capacity, (uint32_t)1);
    ASSERT_EQ(states->rigid_body.count, (uint32_t)0);

    /* platform_state_create_flat calls platform_state_zero_flat internally.
     * After creation, index 0 should have identity quaternion and zeros elsewhere. */
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[0], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.quat_w[0], 1.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.quat_x[0], 0.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_0][0], 0.0f);

    /* Validate should pass on freshly created state */
    ASSERT_TRUE(platform_state_validate(states, 0));

    /* Init should work at index 0 */
    platform_state_init(states, 0);
    ASSERT_TRUE(platform_state_validate(states, 0));

    /* Set/get round-trip at index 0 */
    PlatformStateAoS aos = {
        .position = VEC3(1.0f, 2.0f, 3.0f),
        .velocity = VEC3(4.0f, 5.0f, 6.0f),
        .orientation = QUAT(0.5f, 0.5f, 0.5f, 0.5f),
        .omega = VEC3(0.1f, 0.2f, 0.3f),
    };
    platform_state_set(states, 0, &aos);
    PlatformStateAoS result = platform_state_get(states, 0);
    ASSERT_FLOAT_EQ(result.position.x, 1.0f);
    ASSERT_FLOAT_EQ(result.position.y, 2.0f);
    ASSERT_FLOAT_EQ(result.position.z, 3.0f);
    ASSERT_FLOAT_EQ(result.velocity.x, 4.0f);
    ASSERT_FLOAT_EQ(result.velocity.y, 5.0f);
    ASSERT_FLOAT_EQ(result.velocity.z, 6.0f);
    ASSERT_FLOAT_EQ(result.orientation.w, 0.5f);
    ASSERT_FLOAT_EQ(result.orientation.x, 0.5f);
    ASSERT_FLOAT_EQ(result.orientation.y, 0.5f);
    ASSERT_FLOAT_EQ(result.orientation.z, 0.5f);
    ASSERT_FLOAT_EQ(result.omega.x, 0.1f);
    ASSERT_FLOAT_EQ(result.omega.y, 0.2f);
    ASSERT_FLOAT_EQ(result.omega.z, 0.3f);

    /* Verify extensions at index 0 are unaffected by platform_state_set (rigid body only) */
    states->extension[QUAD_EXT_RPM_0][0] = 100.0f;
    states->extension[QUAD_EXT_RPM_1][0] = 200.0f;
    states->extension[QUAD_EXT_RPM_2][0] = 300.0f;
    states->extension[QUAD_EXT_RPM_3][0] = 400.0f;
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_0][0], 100.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_1][0], 200.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_2][0], 300.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_3][0], 400.0f);

    /* Reset batch at index 0 */
    uint32_t idx = 0;
    Vec3 pos = VEC3(7.0f, 8.0f, 9.0f);
    Quat orient = QUAT_IDENTITY;
    rigid_body_state_reset_batch(&states->rigid_body, &idx, &pos, &orient, 1);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[0], 7.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.vel_x[0], 0.0f);
    /* Note: rigid_body_state_reset_batch does not touch extensions */
    ASSERT_TRUE(platform_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

/**
 * capacity=1 copy: src and dst both capacity=1. Copy the single element.
 */
TEST(copy_capacity_one) {
    Arena* arena = arena_create(64 * 1024);
    PlatformStateSOA* src = platform_state_create(arena, 1, QUAD_STATE_EXT_COUNT);
    PlatformStateSOA* dst = platform_state_create(arena, 1, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(src);
    ASSERT_NOT_NULL(dst);

    src->rigid_body.pos_x[0] = 42.0f;
    src->rigid_body.pos_y[0] = 43.0f;
    src->rigid_body.pos_z[0] = 44.0f;
    src->rigid_body.vel_x[0] = 1.0f;
    src->rigid_body.vel_y[0] = 2.0f;
    src->rigid_body.vel_z[0] = 3.0f;
    src->rigid_body.quat_w[0] = 1.0f;
    src->rigid_body.quat_x[0] = 0.0f;
    src->rigid_body.quat_y[0] = 0.0f;
    src->rigid_body.quat_z[0] = 0.0f;
    src->rigid_body.omega_x[0] = 0.5f;
    src->rigid_body.omega_y[0] = 0.6f;
    src->rigid_body.omega_z[0] = 0.7f;
    src->extension[QUAD_EXT_RPM_0][0] = 100.0f;
    src->extension[QUAD_EXT_RPM_1][0] = 200.0f;
    src->extension[QUAD_EXT_RPM_2][0] = 300.0f;
    src->extension[QUAD_EXT_RPM_3][0] = 400.0f;

    platform_state_copy(dst, src, 0, 0, 1);

    ASSERT_FLOAT_EQ(dst->rigid_body.pos_x[0], 42.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.pos_y[0], 43.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.pos_z[0], 44.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.vel_x[0], 1.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.vel_y[0], 2.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.vel_z[0], 3.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.quat_w[0], 1.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.quat_x[0], 0.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.quat_y[0], 0.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.quat_z[0], 0.0f);
    ASSERT_FLOAT_EQ(dst->rigid_body.omega_x[0], 0.5f);
    ASSERT_FLOAT_EQ(dst->rigid_body.omega_y[0], 0.6f);
    ASSERT_FLOAT_EQ(dst->rigid_body.omega_z[0], 0.7f);
    ASSERT_FLOAT_EQ(dst->extension[QUAD_EXT_RPM_0][0], 100.0f);
    ASSERT_FLOAT_EQ(dst->extension[QUAD_EXT_RPM_1][0], 200.0f);
    ASSERT_FLOAT_EQ(dst->extension[QUAD_EXT_RPM_2][0], 300.0f);
    ASSERT_FLOAT_EQ(dst->extension[QUAD_EXT_RPM_3][0], 400.0f);

    arena_destroy(arena);
    return 0;
}

/**
 * capacity=4096 stress test. Verify all indices are addressable and that
 * init, set/get, zero, and validate work across the full range.
 */
TEST(create_capacity_4096_stress) {
    Arena* arena = arena_create(4 * 1024 * 1024); /* 4MB for 4096 drones */
    ASSERT_NOT_NULL(arena);

    PlatformStateSOA* states = platform_state_create(arena, 4096, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);
    ASSERT_EQ(states->rigid_body.capacity, (uint32_t)4096);

    /* Write a unique value to every slot in every array */
    for (uint32_t i = 0; i < 4096; i++) {
        float v = (float)i;
        states->rigid_body.pos_x[i] = v;
        states->rigid_body.pos_y[i] = v + 0.1f;
        states->rigid_body.pos_z[i] = v + 0.2f;
        states->rigid_body.vel_x[i] = v + 0.3f;
        states->rigid_body.vel_y[i] = v + 0.4f;
        states->rigid_body.vel_z[i] = v + 0.5f;
        states->rigid_body.quat_w[i] = 1.0f; /* keep unit for validation */
        states->rigid_body.quat_x[i] = 0.0f;
        states->rigid_body.quat_y[i] = 0.0f;
        states->rigid_body.quat_z[i] = 0.0f;
        states->rigid_body.omega_x[i] = v + 0.6f;
        states->rigid_body.omega_y[i] = v + 0.7f;
        states->rigid_body.omega_z[i] = v + 0.8f;
        states->extension[QUAD_EXT_RPM_0][i] = v;
        states->extension[QUAD_EXT_RPM_1][i] = v + 1.0f;
        states->extension[QUAD_EXT_RPM_2][i] = v + 2.0f;
        states->extension[QUAD_EXT_RPM_3][i] = v + 3.0f;
    }

    /* Read back and verify every slot */
    for (uint32_t i = 0; i < 4096; i++) {
        float v = (float)i;
        ASSERT_FLOAT_EQ(states->rigid_body.pos_x[i], v);
        ASSERT_FLOAT_EQ(states->rigid_body.pos_y[i], v + 0.1f);
        ASSERT_FLOAT_EQ(states->rigid_body.pos_z[i], v + 0.2f);
        ASSERT_FLOAT_EQ(states->rigid_body.vel_x[i], v + 0.3f);
        ASSERT_FLOAT_EQ(states->rigid_body.vel_y[i], v + 0.4f);
        ASSERT_FLOAT_EQ(states->rigid_body.vel_z[i], v + 0.5f);
        ASSERT_FLOAT_EQ(states->rigid_body.omega_x[i], v + 0.6f);
        ASSERT_FLOAT_EQ(states->rigid_body.omega_y[i], v + 0.7f);
        ASSERT_FLOAT_EQ(states->rigid_body.omega_z[i], v + 0.8f);
        ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_0][i], v);
        ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_1][i], v + 1.0f);
        ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_2][i], v + 2.0f);
        ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_3][i], v + 3.0f);
    }

    /* Validate first, middle, last */
    ASSERT_TRUE(platform_state_validate(states, 0));
    ASSERT_TRUE(platform_state_validate(states, 2048));
    ASSERT_TRUE(platform_state_validate(states, 4095));

    /* Zero all and verify */
    platform_state_zero(states);
    for (uint32_t i = 0; i < 4096; i++) {
        ASSERT_FLOAT_EQ(states->rigid_body.pos_x[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.quat_w[i], 1.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.quat_x[i], 0.0f);
        ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_0][i], 0.0f);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 2: Boundary Index Access
 * ============================================================================ */

/**
 * Last valid index (capacity-1). All operations must succeed.
 */
TEST(boundary_last_valid_index_operations) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 64, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    uint32_t last = states->rigid_body.capacity - 1; /* 63 */

    /* Init at last index */
    platform_state_init(states, last);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[last], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.quat_w[last], 1.0f);

    /* Set/get at last index */
    PlatformStateAoS aos = {
        .position = VEC3(99.0f, 98.0f, 97.0f),
        .velocity = VEC3(1.1f, 2.2f, 3.3f),
        .orientation = QUAT_IDENTITY,
        .omega = VEC3(0.5f, 0.6f, 0.7f),
    };
    platform_state_set(states, last, &aos);
    PlatformStateAoS result = platform_state_get(states, last);
    ASSERT_FLOAT_EQ(result.position.x, 99.0f);
    ASSERT_FLOAT_EQ(result.position.y, 98.0f);
    ASSERT_FLOAT_EQ(result.position.z, 97.0f);
    ASSERT_FLOAT_EQ(result.velocity.x, 1.1f);
    ASSERT_FLOAT_EQ(result.velocity.y, 2.2f);
    ASSERT_FLOAT_EQ(result.velocity.z, 3.3f);
    ASSERT_FLOAT_EQ(result.omega.x, 0.5f);
    ASSERT_FLOAT_EQ(result.omega.y, 0.6f);
    ASSERT_FLOAT_EQ(result.omega.z, 0.7f);

    /* Verify extension set/get at last valid index */
    states->extension[QUAD_EXT_RPM_0][last] = 500.0f;
    states->extension[QUAD_EXT_RPM_1][last] = 600.0f;
    states->extension[QUAD_EXT_RPM_2][last] = 700.0f;
    states->extension[QUAD_EXT_RPM_3][last] = 800.0f;
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_0][last], 500.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_1][last], 600.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_2][last], 700.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_3][last], 800.0f);

    /* Validate at last index */
    ASSERT_TRUE(platform_state_validate(states, last));

    /* Reset batch at last index */
    uint32_t idx = last;
    Vec3 pos = VEC3(0.0f, 0.0f, 0.0f);
    Quat orient = QUAT_IDENTITY;
    rigid_body_state_reset_batch(&states->rigid_body, &idx, &pos, &orient, 1);
    PLATFORM_QUADCOPTER.reset_state(states->extension, states->extension_count, last);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[last], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.vel_x[last], 0.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_0][last], 0.0f);

    arena_destroy(arena);
    return 0;
}

/**
 * First invalid index (index == capacity). The implementation checks
 * index >= capacity and returns early / returns false. Verify this for
 * validate, init, set, and get.
 */
TEST(boundary_first_invalid_index_validate) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 64, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    /* validate at index == capacity should return false (bounds check) */
    ASSERT_FALSE(platform_state_validate(states, 64));

    arena_destroy(arena);
    return 0;
}

TEST(boundary_first_invalid_index_get_returns_identity) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 64, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    /* get at out-of-bounds returns zeroed state with identity quaternion */
    PlatformStateAoS result = platform_state_get(states, 64);
    ASSERT_FLOAT_EQ(result.position.x, 0.0f);
    ASSERT_FLOAT_EQ(result.position.y, 0.0f);
    ASSERT_FLOAT_EQ(result.position.z, 0.0f);
    ASSERT_FLOAT_EQ(result.orientation.w, 1.0f);
    ASSERT_FLOAT_EQ(result.orientation.x, 0.0f);
    ASSERT_FLOAT_EQ(result.orientation.y, 0.0f);
    ASSERT_FLOAT_EQ(result.orientation.z, 0.0f);

    arena_destroy(arena);
    return 0;
}

/**
 * reset_batch with an out-of-bounds index should skip that index (continue).
 * Verify that in-bounds indices in the same batch are still processed.
 */
TEST(boundary_reset_batch_oob_index_skipped) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 64, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    /* Set recognizable values at index 0 and 63 */
    states->rigid_body.pos_x[0] = 111.0f;
    states->rigid_body.pos_x[63] = 222.0f;

    /* Batch with one out-of-bounds index (64) and two valid ones */
    uint32_t indices[3] = {0, 64, 63};
    Vec3 positions[3] = {
        VEC3(1.0f, 1.0f, 1.0f),
        VEC3(2.0f, 2.0f, 2.0f),
        VEC3(3.0f, 3.0f, 3.0f)
    };
    Quat orientations[3] = {QUAT_IDENTITY, QUAT_IDENTITY, QUAT_IDENTITY};

    rigid_body_state_reset_batch(&states->rigid_body, indices, positions, orientations, 3);

    /* Index 0 should have been reset */
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[0], 1.0f);
    /* Index 63 should have been reset */
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[63], 3.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 3: Copy Semantics -- All 17 Arrays, Offsets, count=0
 * ============================================================================ */

/**
 * Copy with count=0 should be a no-op: destination must remain unchanged.
 */
TEST(copy_count_zero_is_noop) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* src = platform_state_create(arena, 16, QUAD_STATE_EXT_COUNT);
    PlatformStateSOA* dst = platform_state_create(arena, 16, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(src);
    ASSERT_NOT_NULL(dst);

    /* Set a recognizable pattern in dst */
    dst->rigid_body.pos_x[0] = 999.0f;
    dst->rigid_body.quat_w[0] = 1.0f;

    /* Set different values in src */
    src->rigid_body.pos_x[0] = 111.0f;

    /* Copy 0 elements */
    rigid_body_state_copy(&dst->rigid_body, &src->rigid_body, 0, 0, 0);

    /* dst should be unchanged */
    ASSERT_FLOAT_EQ(dst->rigid_body.pos_x[0], 999.0f);

    arena_destroy(arena);
    return 0;
}

/**
 * Copy all 17 arrays with non-zero offsets. Verify every field is copied
 * correctly and that elements outside the copy range are untouched.
 */
TEST(copy_all_17_arrays_with_offsets) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    PlatformStateSOA* src = platform_state_create(arena, 32, QUAD_STATE_EXT_COUNT);
    PlatformStateSOA* dst = platform_state_create(arena, 32, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(src);
    ASSERT_NOT_NULL(dst);

    /* Fill source with unique per-index, per-array values */
    for (uint32_t i = 0; i < 32; i++) {
        float base = (float)(i * 100);
        src->rigid_body.pos_x[i] = base + 1.0f;
        src->rigid_body.pos_y[i] = base + 2.0f;
        src->rigid_body.pos_z[i] = base + 3.0f;
        src->rigid_body.vel_x[i] = base + 4.0f;
        src->rigid_body.vel_y[i] = base + 5.0f;
        src->rigid_body.vel_z[i] = base + 6.0f;
        src->rigid_body.quat_w[i] = base + 7.0f;
        src->rigid_body.quat_x[i] = base + 8.0f;
        src->rigid_body.quat_y[i] = base + 9.0f;
        src->rigid_body.quat_z[i] = base + 10.0f;
        src->rigid_body.omega_x[i] = base + 11.0f;
        src->rigid_body.omega_y[i] = base + 12.0f;
        src->rigid_body.omega_z[i] = base + 13.0f;
        src->extension[QUAD_EXT_RPM_0][i] = base + 14.0f;
        src->extension[QUAD_EXT_RPM_1][i] = base + 15.0f;
        src->extension[QUAD_EXT_RPM_2][i] = base + 16.0f;
        src->extension[QUAD_EXT_RPM_3][i] = base + 17.0f;
    }

    /* Copy src[4..12) to dst[20..28) -- 8 elements, non-zero offsets */
    platform_state_copy(dst, src, 20, 4, 8);

    /* Verify copied region: dst[20+j] should equal src[4+j] for j in [0,8) */
    for (uint32_t j = 0; j < 8; j++) {
        float base = (float)((4 + j) * 100);
        ASSERT_FLOAT_EQ(dst->rigid_body.pos_x[20 + j], base + 1.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.pos_y[20 + j], base + 2.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.pos_z[20 + j], base + 3.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.vel_x[20 + j], base + 4.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.vel_y[20 + j], base + 5.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.vel_z[20 + j], base + 6.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.quat_w[20 + j], base + 7.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.quat_x[20 + j], base + 8.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.quat_y[20 + j], base + 9.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.quat_z[20 + j], base + 10.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.omega_x[20 + j], base + 11.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.omega_y[20 + j], base + 12.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.omega_z[20 + j], base + 13.0f);
        ASSERT_FLOAT_EQ(dst->extension[QUAD_EXT_RPM_0][20 + j], base + 14.0f);
        ASSERT_FLOAT_EQ(dst->extension[QUAD_EXT_RPM_1][20 + j], base + 15.0f);
        ASSERT_FLOAT_EQ(dst->extension[QUAD_EXT_RPM_2][20 + j], base + 16.0f);
        ASSERT_FLOAT_EQ(dst->extension[QUAD_EXT_RPM_3][20 + j], base + 17.0f);
    }

    /* Verify element just before copied region is still at default (0.0) */
    ASSERT_FLOAT_EQ(dst->rigid_body.pos_x[19], 0.0f);
    /* Verify element just after copied region is still at default */
    ASSERT_FLOAT_EQ(dst->rigid_body.pos_x[28], 0.0f);

    arena_destroy(arena);
    return 0;
}

/**
 * Copy produces a deep (value) copy, not a shallow (pointer) copy.
 * Mutating source after copy must not affect destination.
 */
TEST(copy_deep_independence_all_arrays) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    PlatformStateSOA* src = platform_state_create(arena, 8, QUAD_STATE_EXT_COUNT);
    PlatformStateSOA* dst = platform_state_create(arena, 8, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(src);
    ASSERT_NOT_NULL(dst);

    for (uint32_t i = 0; i < 8; i++) {
        src->rigid_body.pos_x[i] = 1.0f;
        src->rigid_body.pos_y[i] = 2.0f;
        src->rigid_body.pos_z[i] = 3.0f;
        src->rigid_body.vel_x[i] = 4.0f;
        src->rigid_body.vel_y[i] = 5.0f;
        src->rigid_body.vel_z[i] = 6.0f;
        src->rigid_body.quat_w[i] = 1.0f;
        src->rigid_body.quat_x[i] = 0.0f;
        src->rigid_body.quat_y[i] = 0.0f;
        src->rigid_body.quat_z[i] = 0.0f;
        src->rigid_body.omega_x[i] = 7.0f;
        src->rigid_body.omega_y[i] = 8.0f;
        src->rigid_body.omega_z[i] = 9.0f;
        src->extension[QUAD_EXT_RPM_0][i] = 10.0f;
        src->extension[QUAD_EXT_RPM_1][i] = 11.0f;
        src->extension[QUAD_EXT_RPM_2][i] = 12.0f;
        src->extension[QUAD_EXT_RPM_3][i] = 13.0f;
    }

    platform_state_copy(dst, src, 0, 0, 8);

    /* Mutate every array in source */
    for (uint32_t i = 0; i < 8; i++) {
        src->rigid_body.pos_x[i] = -1.0f;
        src->rigid_body.pos_y[i] = -2.0f;
        src->rigid_body.pos_z[i] = -3.0f;
        src->rigid_body.vel_x[i] = -4.0f;
        src->rigid_body.vel_y[i] = -5.0f;
        src->rigid_body.vel_z[i] = -6.0f;
        src->rigid_body.quat_w[i] = 0.0f;
        src->rigid_body.quat_x[i] = 1.0f;
        src->rigid_body.quat_y[i] = 0.0f;
        src->rigid_body.quat_z[i] = 0.0f;
        src->rigid_body.omega_x[i] = -7.0f;
        src->rigid_body.omega_y[i] = -8.0f;
        src->rigid_body.omega_z[i] = -9.0f;
        src->extension[QUAD_EXT_RPM_0][i] = -10.0f;
        src->extension[QUAD_EXT_RPM_1][i] = -11.0f;
        src->extension[QUAD_EXT_RPM_2][i] = -12.0f;
        src->extension[QUAD_EXT_RPM_3][i] = -13.0f;
    }

    /* Destination must be unchanged */
    for (uint32_t i = 0; i < 8; i++) {
        ASSERT_FLOAT_EQ(dst->rigid_body.pos_x[i], 1.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.pos_y[i], 2.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.pos_z[i], 3.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.vel_x[i], 4.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.vel_y[i], 5.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.vel_z[i], 6.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.quat_w[i], 1.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.quat_x[i], 0.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.quat_y[i], 0.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.quat_z[i], 0.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.omega_x[i], 7.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.omega_y[i], 8.0f);
        ASSERT_FLOAT_EQ(dst->rigid_body.omega_z[i], 9.0f);
        ASSERT_FLOAT_EQ(dst->extension[QUAD_EXT_RPM_0][i], 10.0f);
        ASSERT_FLOAT_EQ(dst->extension[QUAD_EXT_RPM_1][i], 11.0f);
        ASSERT_FLOAT_EQ(dst->extension[QUAD_EXT_RPM_2][i], 12.0f);
        ASSERT_FLOAT_EQ(dst->extension[QUAD_EXT_RPM_3][i], 13.0f);
    }

    arena_destroy(arena);
    return 0;
}

/**
 * Copy overflow protection: dst_offset + count > dst->rigid_body.capacity should be
 * silently rejected (no copy performed).
 */
TEST(copy_dst_overflow_rejected) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* src = platform_state_create(arena, 16, QUAD_STATE_EXT_COUNT);
    PlatformStateSOA* dst = platform_state_create(arena, 16, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(src);
    ASSERT_NOT_NULL(dst);

    src->rigid_body.pos_x[0] = 77.0f;
    dst->rigid_body.pos_x[15] = 999.0f;

    /* Try to copy 2 elements starting at dst offset 15 (15+2=17 > 16) */
    rigid_body_state_copy(&dst->rigid_body, &src->rigid_body, 15, 0, 2);

    /* dst should be unchanged because the copy was rejected */
    ASSERT_FLOAT_EQ(dst->rigid_body.pos_x[15], 999.0f);

    arena_destroy(arena);
    return 0;
}

/**
 * Copy src overflow protection: src_offset + count > src->rigid_body.capacity.
 */
TEST(copy_src_overflow_rejected) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* src = platform_state_create(arena, 16, QUAD_STATE_EXT_COUNT);
    PlatformStateSOA* dst = platform_state_create(arena, 16, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(src);
    ASSERT_NOT_NULL(dst);

    dst->rigid_body.pos_x[0] = 888.0f;

    /* src only has 16 elements, try copying from offset 15 with count 2 */
    rigid_body_state_copy(&dst->rigid_body, &src->rigid_body, 0, 15, 2);

    /* dst should be unchanged */
    ASSERT_FLOAT_EQ(dst->rigid_body.pos_x[0], 888.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 4: Reset Batch -- count=0, count=capacity, custom orientations
 * ============================================================================ */

/**
 * reset_batch with count=0 is a no-op: existing state must be preserved.
 */
TEST(reset_batch_count_zero_is_noop) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 16, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    /* Set a recognizable pattern */
    states->rigid_body.pos_x[0] = 42.0f;
    states->rigid_body.vel_x[5] = 77.0f;
    states->extension[QUAD_EXT_RPM_2][15] = 123.0f;

    /* Reset with count=0 -- all pointers can be non-NULL */
    uint32_t dummy_idx = 0;
    Vec3 dummy_pos = VEC3(0.0f, 0.0f, 0.0f);
    Quat dummy_orient = QUAT_IDENTITY;
    rigid_body_state_reset_batch(&states->rigid_body, &dummy_idx, &dummy_pos, &dummy_orient, 0);

    /* Verify nothing changed */
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[0], 42.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.vel_x[5], 77.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_2][15], 123.0f);

    arena_destroy(arena);
    return 0;
}

/**
 * reset_batch with count=capacity: reset every drone.
 * After full reset, all velocities, omegas, and RPMs must be zero.
 * Positions and orientations must match the provided arrays.
 */
TEST(reset_batch_full_capacity) {
    Arena* arena = arena_create(1024 * 1024);
    uint32_t cap = 128;
    PlatformStateSOA* states = platform_state_create(arena, cap, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    /* Dirty up all the state first */
    for (uint32_t i = 0; i < cap; i++) {
        states->rigid_body.vel_x[i] = 100.0f;
        states->rigid_body.vel_y[i] = 200.0f;
        states->rigid_body.vel_z[i] = 300.0f;
        states->rigid_body.omega_x[i] = 5.0f;
        states->rigid_body.omega_y[i] = 6.0f;
        states->rigid_body.omega_z[i] = 7.0f;
        states->extension[QUAD_EXT_RPM_0][i] = 1000.0f;
        states->extension[QUAD_EXT_RPM_1][i] = 1100.0f;
        states->extension[QUAD_EXT_RPM_2][i] = 1200.0f;
        states->extension[QUAD_EXT_RPM_3][i] = 1300.0f;
    }

    /* Build index array and position/orientation arrays */
    uint32_t* indices = arena_alloc_array(arena, uint32_t, cap);
    Vec3* positions = arena_alloc_array(arena, Vec3, cap);
    Quat* orientations = arena_alloc_array(arena, Quat, cap);
    ASSERT_NOT_NULL(indices);
    ASSERT_NOT_NULL(positions);
    ASSERT_NOT_NULL(orientations);

    for (uint32_t i = 0; i < cap; i++) {
        indices[i] = i;
        positions[i] = VEC3((float)i, (float)(i + 1), (float)(i + 2));
        orientations[i] = QUAT_IDENTITY;
    }

    rigid_body_state_reset_batch(&states->rigid_body, indices, positions, orientations, cap);

    /* Reset extensions via vtable (rigid_body_state_reset_batch only handles rigid body) */
    for (uint32_t i = 0; i < cap; i++) {
        PLATFORM_QUADCOPTER.reset_state(states->extension, states->extension_count, i);
    }

    for (uint32_t i = 0; i < cap; i++) {
        ASSERT_FLOAT_EQ(states->rigid_body.pos_x[i], (float)i);
        ASSERT_FLOAT_EQ(states->rigid_body.pos_y[i], (float)(i + 1));
        ASSERT_FLOAT_EQ(states->rigid_body.pos_z[i], (float)(i + 2));
        ASSERT_FLOAT_EQ(states->rigid_body.vel_x[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.vel_y[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.vel_z[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.quat_w[i], 1.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.quat_x[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.quat_y[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.quat_z[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.omega_x[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.omega_y[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.omega_z[i], 0.0f);
        ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_0][i], 0.0f);
        ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_1][i], 0.0f);
        ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_2][i], 0.0f);
        ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_3][i], 0.0f);
    }

    arena_destroy(arena);
    return 0;
}

/**
 * reset_batch with non-identity custom orientations. Verify the quaternion
 * is stored exactly as provided (no normalization applied by reset_batch).
 */
TEST(reset_batch_custom_orientation) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 4, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    uint32_t indices[2] = {1, 3};
    Vec3 positions[2] = {
        VEC3(10.0f, 20.0f, 30.0f),
        VEC3(40.0f, 50.0f, 60.0f)
    };
    /* 90-degree rotation around Z axis: w=cos(45)=0.7071, z=sin(45)=0.7071 */
    Quat orientations[2] = {
        QUAT(0.7071f, 0.0f, 0.0f, 0.7071f),
        QUAT(0.0f, 1.0f, 0.0f, 0.0f)  /* 180-degree around X */
    };

    rigid_body_state_reset_batch(&states->rigid_body, indices, positions, orientations, 2);

    ASSERT_FLOAT_NEAR(states->rigid_body.quat_w[1], 0.7071f, 1e-4f);
    ASSERT_FLOAT_NEAR(states->rigid_body.quat_z[1], 0.7071f, 1e-4f);
    ASSERT_FLOAT_EQ(states->rigid_body.quat_x[1], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.quat_y[1], 0.0f);

    ASSERT_FLOAT_EQ(states->rigid_body.quat_w[3], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.quat_x[3], 1.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.quat_y[3], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.quat_z[3], 0.0f);

    /* Verify untouched indices still have identity from creation */
    ASSERT_FLOAT_EQ(states->rigid_body.quat_w[0], 1.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.quat_x[0], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.quat_w[2], 1.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.quat_x[2], 0.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 5: Validation -- NaN, Inf, Zero Quaternion, Near-Unit Quaternion
 * ============================================================================ */

/**
 * NaN in omega fields must be detected by validation.
 * (Existing tests cover pos/vel/quat NaN; this covers omega and rpm NaN.)
 */
TEST(validate_nan_in_omega) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 4, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    /* NaN in omega_x */
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

TEST(validate_nan_in_rpm) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 4, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

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

/**
 * Inf in position, velocity, and angular velocity.
 *
 * The validation function checks for NaN using isnan(). Per the C standard,
 * isinf values are NOT NaN, so isnan(INFINITY) == false. The validation
 * function does NOT check for Inf -- only NaN and quaternion norm.
 *
 * This test documents the actual behavior: Inf in position/velocity/omega
 * must be rejected by validation because Inf in physical state indicates
 * divergence (e.g., unbounded force integration).
 */
TEST(validate_inf_in_position_rejected) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 4, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    states->rigid_body.pos_x[0] = INFINITY;
    ASSERT_FALSE(platform_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

TEST(validate_inf_in_velocity_rejected) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 4, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    states->rigid_body.vel_y[0] = -INFINITY;
    ASSERT_FALSE(platform_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

TEST(validate_inf_in_omega_rejected) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 4, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    states->rigid_body.omega_z[0] = INFINITY;
    ASSERT_FALSE(platform_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

/**
 * Inf in quaternion: the quaternion norm check (|q|^2 approx 1.0) will fail
 * because INFINITY * INFINITY = INFINITY, and |Inf - 1.0| > 1e-4.
 * But NaN propagation could be subtle: Inf + finite = Inf, fabsf(Inf - 1) = Inf.
 * Verify the validator correctly rejects this.
 */
TEST(validate_inf_in_quaternion_rejected) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 4, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    states->rigid_body.quat_w[0] = INFINITY;
    /* |q|^2 = Inf + 0 + 0 + 0 = Inf. fabsf(Inf - 1.0f) = Inf > 1e-4. */
    ASSERT_FALSE(platform_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

/**
 * Zero quaternion (0,0,0,0): |q|^2 = 0, which deviates from 1.0 by 1.0.
 * Validation must reject it (1.0 > 1e-4 tolerance).
 */
TEST(validate_zero_quaternion_rejected) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 4, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    states->rigid_body.quat_w[0] = 0.0f;
    states->rigid_body.quat_x[0] = 0.0f;
    states->rigid_body.quat_y[0] = 0.0f;
    states->rigid_body.quat_z[0] = 0.0f;

    ASSERT_FALSE(platform_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

/**
 * Near-unit quaternion that is just barely within tolerance (1e-4).
 * |q|^2 = 1.0 + eps where eps < 1e-4 should pass.
 */
TEST(validate_near_unit_quaternion_within_tolerance) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 4, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    /* quat = (w, 0, 0, 0) where w^2 = 1.00005. w = sqrt(1.00005) ~ 1.000025 */
    states->rigid_body.quat_w[0] = 1.000025f;
    states->rigid_body.quat_x[0] = 0.0f;
    states->rigid_body.quat_y[0] = 0.0f;
    states->rigid_body.quat_z[0] = 0.0f;
    /* |q|^2 = (1.000025)^2 = ~1.00005 => |1.00005 - 1.0| = 5e-5 < 1e-4 */
    ASSERT_TRUE(platform_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

/**
 * Near-unit quaternion that is just barely outside tolerance.
 * |q|^2 = 1.0 + eps where eps > 1e-4 should fail.
 */
TEST(validate_near_unit_quaternion_outside_tolerance) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 4, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    /* quat = (w, 0, 0, 0) where w^2 = 1.0002. w = sqrt(1.0002) ~ 1.0001 */
    states->rigid_body.quat_w[0] = 1.0001f;
    states->rigid_body.quat_x[0] = 0.0f;
    states->rigid_body.quat_y[0] = 0.0f;
    states->rigid_body.quat_z[0] = 0.0f;
    /* |q|^2 = (1.0001)^2 = ~1.00020001 => |1.00020001 - 1.0| = 2e-4 > 1e-4 */
    ASSERT_FALSE(platform_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

/**
 * Negative RPM values are finite floats. The generic validator only checks
 * for NaN/Inf in extensions. Platform-specific constraints (e.g., non-negative
 * RPMs) are not enforced by the generic validator.
 * However, -INFINITY is caught by the isinf() check.
 */
TEST(validate_negative_rpm_each_motor) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 4, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    /* Negative finite RPMs are accepted by generic validator */
    platform_state_init(states, 0);
    states->extension[QUAD_EXT_RPM_0][0] = -0.001f;
    ASSERT_TRUE(platform_state_validate(states, 0));

    platform_state_init(states, 0);
    states->extension[QUAD_EXT_RPM_1][0] = -1e-6f;
    ASSERT_TRUE(platform_state_validate(states, 0));

    platform_state_init(states, 0);
    states->extension[QUAD_EXT_RPM_2][0] = -1000.0f;
    ASSERT_TRUE(platform_state_validate(states, 0));

    /* -Inf IS caught by generic isinf() check */
    platform_state_init(states, 0);
    states->extension[QUAD_EXT_RPM_3][0] = -INFINITY;
    ASSERT_FALSE(platform_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

/**
 * Positive Inf RPM: isinf check now catches this.
 */
TEST(validate_positive_inf_rpm_rejected) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 4, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    states->extension[QUAD_EXT_RPM_0][0] = INFINITY;
    ASSERT_FALSE(platform_state_validate(states, 0));

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 6: SoA Alignment -- All Arrays 32-Byte Aligned
 * ============================================================================ */

/**
 * Verify all 17 PlatformStateSOA float arrays are 32-byte aligned, at multiple
 * capacities including non-power-of-two to stress the alignment logic.
 */
TEST(alignment_state_all_17_arrays_capacity_7) {
    /* capacity=7 is non-power-of-two -- alignment must still hold */
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 7, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    ASSERT_TRUE(((uintptr_t)states->rigid_body.pos_x % 32) == 0);
    ASSERT_TRUE(((uintptr_t)states->rigid_body.pos_y % 32) == 0);
    ASSERT_TRUE(((uintptr_t)states->rigid_body.pos_z % 32) == 0);
    ASSERT_TRUE(((uintptr_t)states->rigid_body.vel_x % 32) == 0);
    ASSERT_TRUE(((uintptr_t)states->rigid_body.vel_y % 32) == 0);
    ASSERT_TRUE(((uintptr_t)states->rigid_body.vel_z % 32) == 0);
    ASSERT_TRUE(((uintptr_t)states->rigid_body.quat_w % 32) == 0);
    ASSERT_TRUE(((uintptr_t)states->rigid_body.quat_x % 32) == 0);
    ASSERT_TRUE(((uintptr_t)states->rigid_body.quat_y % 32) == 0);
    ASSERT_TRUE(((uintptr_t)states->rigid_body.quat_z % 32) == 0);
    ASSERT_TRUE(((uintptr_t)states->rigid_body.omega_x % 32) == 0);
    ASSERT_TRUE(((uintptr_t)states->rigid_body.omega_y % 32) == 0);
    ASSERT_TRUE(((uintptr_t)states->rigid_body.omega_z % 32) == 0);
    ASSERT_TRUE(((uintptr_t)states->extension[QUAD_EXT_RPM_0] % 32) == 0);
    ASSERT_TRUE(((uintptr_t)states->extension[QUAD_EXT_RPM_1] % 32) == 0);
    ASSERT_TRUE(((uintptr_t)states->extension[QUAD_EXT_RPM_2] % 32) == 0);
    ASSERT_TRUE(((uintptr_t)states->extension[QUAD_EXT_RPM_3] % 32) == 0);

    arena_destroy(arena);
    return 0;
}

/**
 * Verify all 15 PlatformParamsSOA float arrays are 32-byte aligned.
 */
TEST(alignment_params_all_15_arrays) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformParamsSOA* params = platform_params_create(arena, 13, QUAD_PARAMS_EXT_COUNT); /* odd capacity */
    ASSERT_NOT_NULL(params);

    ASSERT_TRUE(((uintptr_t)params->rigid_body.mass % 32) == 0);
    ASSERT_TRUE(((uintptr_t)params->rigid_body.ixx % 32) == 0);
    ASSERT_TRUE(((uintptr_t)params->rigid_body.iyy % 32) == 0);
    ASSERT_TRUE(((uintptr_t)params->rigid_body.izz % 32) == 0);
    ASSERT_TRUE(((uintptr_t)params->extension[QUAD_PEXT_ARM_LENGTH] % 32) == 0);
    ASSERT_TRUE(((uintptr_t)params->rigid_body.collision_radius % 32) == 0);
    ASSERT_TRUE(((uintptr_t)params->extension[QUAD_PEXT_K_THRUST] % 32) == 0);
    ASSERT_TRUE(((uintptr_t)params->extension[QUAD_PEXT_K_TORQUE] % 32) == 0);
    ASSERT_TRUE(((uintptr_t)params->extension[QUAD_PEXT_K_DRAG] % 32) == 0);
    ASSERT_TRUE(((uintptr_t)params->extension[QUAD_PEXT_K_ANG_DAMP] % 32) == 0);
    ASSERT_TRUE(((uintptr_t)params->extension[QUAD_PEXT_MOTOR_TAU] % 32) == 0);
    ASSERT_TRUE(((uintptr_t)params->extension[QUAD_PEXT_MAX_RPM] % 32) == 0);
    ASSERT_TRUE(((uintptr_t)params->rigid_body.max_vel % 32) == 0);
    ASSERT_TRUE(((uintptr_t)params->rigid_body.max_omega % 32) == 0);
    ASSERT_TRUE(((uintptr_t)params->rigid_body.gravity % 32) == 0);

    arena_destroy(arena);
    return 0;
}

/**
 * Array pointers must be distinct (no aliasing). Each of the 17 state arrays
 * should point to a different memory region.
 */
TEST(alignment_state_arrays_no_aliasing) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 16, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    /* Collect all 17 pointers */
    float* ptrs[17] = {
        states->rigid_body.pos_x, states->rigid_body.pos_y, states->rigid_body.pos_z,
        states->rigid_body.vel_x, states->rigid_body.vel_y, states->rigid_body.vel_z,
        states->rigid_body.quat_w, states->rigid_body.quat_x, states->rigid_body.quat_y, states->rigid_body.quat_z,
        states->rigid_body.omega_x, states->rigid_body.omega_y, states->rigid_body.omega_z,
        states->extension[QUAD_EXT_RPM_0], states->extension[QUAD_EXT_RPM_1], states->extension[QUAD_EXT_RPM_2], states->extension[QUAD_EXT_RPM_3]
    };

    /* Verify all pointers are distinct */
    for (int i = 0; i < 17; i++) {
        for (int j = i + 1; j < 17; j++) {
            ASSERT_NE(ptrs[i], ptrs[j]);
        }
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 7: platform_state_zero_flat + validate Interaction
 * ============================================================================ */

/**
 * platform_state_zero_flat sets quat_w=1 and everything else to 0.
 * Resulting state should pass validation (identity quaternion is unit, RPMs=0 >= 0).
 */
TEST(zero_then_validate_passes) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 64, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    /* Dirty the state with invalid data */
    for (uint32_t i = 0; i < 64; i++) {
        states->rigid_body.quat_w[i] = 5.0f;
        states->rigid_body.quat_x[i] = 5.0f;
        states->extension[QUAD_EXT_RPM_0][i] = -10.0f;
    }

    platform_state_zero(states);

    /* Every index should now pass validation */
    for (uint32_t i = 0; i < 64; i++) {
        ASSERT_MSG(platform_state_validate(states, i),
                   "Zeroed state must pass validation at every index");
    }

    arena_destroy(arena);
    return 0;
}

/**
 * After platform_state_zero_flat, verify that quat_w is exactly 1.0 (not a close
 * approximation). The SIMD path writes 1.0f via simd_set1_ps(1.0f).
 */
TEST(zero_quaternion_w_is_exact_one) {
    Arena* arena = arena_create(1024 * 1024);
    /* Use a capacity that exercises SIMD remainder handling:
     * NEON=4-wide, AVX2=8-wide. capacity=11 has remainder in both. */
    PlatformStateSOA* states = platform_state_create(arena, 11, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    for (uint32_t i = 0; i < 11; i++) {
        /* Exact equality, not approximate -- 1.0f is representable in IEEE754 */
        ASSERT_TRUE(states->rigid_body.quat_w[i] == 1.0f);
        ASSERT_TRUE(states->rigid_body.quat_x[i] == 0.0f);
        ASSERT_TRUE(states->rigid_body.quat_y[i] == 0.0f);
        ASSERT_TRUE(states->rigid_body.quat_z[i] == 0.0f);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 8: platform_state_init Correctness
 * ============================================================================ */

/**
 * Verify every single field set by platform_state_init matches the documented
 * defaults: position=origin, velocity=0, orientation=identity, omega=0, rpms=0.
 */
TEST(init_all_fields_exact_defaults) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 4, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    /* Dirty the state first to ensure init actually writes */
    states->rigid_body.pos_x[2] = 99.0f;
    states->rigid_body.pos_y[2] = 98.0f;
    states->rigid_body.pos_z[2] = 97.0f;
    states->rigid_body.vel_x[2] = 96.0f;
    states->rigid_body.vel_y[2] = 95.0f;
    states->rigid_body.vel_z[2] = 94.0f;
    states->rigid_body.quat_w[2] = 0.5f;
    states->rigid_body.quat_x[2] = 0.5f;
    states->rigid_body.quat_y[2] = 0.5f;
    states->rigid_body.quat_z[2] = 0.5f;
    states->rigid_body.omega_x[2] = 93.0f;
    states->rigid_body.omega_y[2] = 92.0f;
    states->rigid_body.omega_z[2] = 91.0f;
    states->extension[QUAD_EXT_RPM_0][2] = 90.0f;
    states->extension[QUAD_EXT_RPM_1][2] = 89.0f;
    states->extension[QUAD_EXT_RPM_2][2] = 88.0f;
    states->extension[QUAD_EXT_RPM_3][2] = 87.0f;

    platform_state_init(states, 2);

    /* Check all 17 fields at index 2 */
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[2], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_y[2], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_z[2], 0.0f);

    ASSERT_FLOAT_EQ(states->rigid_body.vel_x[2], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.vel_y[2], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.vel_z[2], 0.0f);

    ASSERT_FLOAT_EQ(states->rigid_body.quat_w[2], 1.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.quat_x[2], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.quat_y[2], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.quat_z[2], 0.0f);

    ASSERT_FLOAT_EQ(states->rigid_body.omega_x[2], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.omega_y[2], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.omega_z[2], 0.0f);

    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_0][2], 0.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_1][2], 0.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_2][2], 0.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_3][2], 0.0f);

    arena_destroy(arena);
    return 0;
}

/**
 * platform_state_init must NOT affect adjacent indices.
 */
TEST(init_does_not_affect_neighbors) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 4, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    /* Set unique recognizable values at indices 0, 1, 2, 3 */
    for (uint32_t i = 0; i < 4; i++) {
        states->rigid_body.pos_x[i] = (float)(i + 100);
        states->rigid_body.vel_x[i] = (float)(i + 200);
        states->rigid_body.omega_x[i] = (float)(i + 300);
        states->extension[QUAD_EXT_RPM_0][i] = (float)(i + 400);
    }

    /* Init only index 2 */
    platform_state_init(states, 2);

    /* Index 2 should be reset */
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[2], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.vel_x[2], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.omega_x[2], 0.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_0][2], 0.0f);

    /* Neighbors must be unchanged */
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[0], 100.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[1], 101.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[3], 103.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.vel_x[0], 200.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.vel_x[1], 201.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.vel_x[3], 203.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.omega_x[0], 300.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.omega_x[1], 301.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.omega_x[3], 303.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_0][0], 400.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_0][1], 401.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_0][3], 403.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 9: PlatformParamsSOA -- Create, Init, Accessors
 * ============================================================================ */

/**
 * PlatformParamsSOA created with capacity=1: verify all default values.
 */
TEST(params_capacity_one_defaults) {
    Arena* arena = arena_create(64 * 1024);
    PlatformParamsSOA* params = platform_params_create(arena, 1, QUAD_PARAMS_EXT_COUNT);
    ASSERT_NOT_NULL(params);
    ASSERT_EQ(params->rigid_body.capacity, (uint32_t)1);
    ASSERT_EQ(params->rigid_body.count, (uint32_t)0);

    /* platform_params_create inits rigid body defaults; quad extensions need vtable */
    PLATFORM_QUADCOPTER.init_params(params->extension, params->extension_count, 0);

    /* Verify all 15 default values at index 0 */
    ASSERT_FLOAT_EQ(params->rigid_body.mass[0], 0.5f);
    ASSERT_FLOAT_EQ(params->rigid_body.ixx[0], 0.0025f);
    ASSERT_FLOAT_EQ(params->rigid_body.iyy[0], 0.0025f);
    ASSERT_FLOAT_EQ(params->rigid_body.izz[0], 0.0045f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_ARM_LENGTH][0], 0.1f);
    ASSERT_FLOAT_EQ(params->rigid_body.collision_radius[0], 0.15f);
    ASSERT_FLOAT_NEAR(params->extension[QUAD_PEXT_K_THRUST][0], 3.16e-10f, 1e-12f);
    ASSERT_FLOAT_NEAR(params->extension[QUAD_PEXT_K_TORQUE][0], 7.94e-12f, 1e-14f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_K_DRAG][0], 0.1f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_K_ANG_DAMP][0], 0.01f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_MOTOR_TAU][0], 0.02f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_MAX_RPM][0], 2500.0f);
    ASSERT_FLOAT_EQ(params->rigid_body.max_vel[0], 20.0f);
    ASSERT_FLOAT_EQ(params->rigid_body.max_omega[0], 10.0f);
    ASSERT_FLOAT_EQ(params->rigid_body.gravity[0], 9.81f);

    arena_destroy(arena);
    return 0;
}

/**
 * PlatformParamsSOA at capacity=1024: all 1024 entries have defaults.
 */
TEST(params_capacity_1024_all_defaults) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    PlatformParamsSOA* params = platform_params_create(arena, 1024, QUAD_PARAMS_EXT_COUNT);
    ASSERT_NOT_NULL(params);
    ASSERT_EQ(params->rigid_body.capacity, (uint32_t)1024);

    /* Init quad-specific extension defaults for all indices */
    for (uint32_t i = 0; i < 1024; i++) {
        PLATFORM_QUADCOPTER.init_params(params->extension, params->extension_count, i);
    }

    /* Spot-check first, middle, last */
    for (uint32_t i = 0; i < 1024; i += 511) {
        ASSERT_FLOAT_EQ(params->rigid_body.mass[i], 0.5f);
        ASSERT_FLOAT_EQ(params->rigid_body.gravity[i], 9.81f);
        ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_ARM_LENGTH][i], 0.1f);
        ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_MAX_RPM][i], 2500.0f);
    }
    /* Also check the very last element */
    ASSERT_FLOAT_EQ(params->rigid_body.mass[1023], 0.5f);
    ASSERT_FLOAT_EQ(params->rigid_body.gravity[1023], 9.81f);

    arena_destroy(arena);
    return 0;
}

/**
 * Full round-trip for all 15 PlatformParamsSOA fields through set/get accessors.
 * Uses deliberately non-default values to verify every field is copied.
 */
TEST(params_accessor_full_roundtrip_all_15_fields) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformParamsSOA* params = platform_params_create(arena, 8, QUAD_PARAMS_EXT_COUNT);
    ASSERT_NOT_NULL(params);

    PlatformParamsAoS custom = {
        .mass = 2.5f,
        .ixx = 0.05f,
        .iyy = 0.06f,
        .izz = 0.07f,
        .collision_radius = 0.35f,
        .max_vel = 50.0f,
        .max_omega = 30.0f,
        .gravity = 1.62f  /* Moon gravity */
    };

    platform_params_set(params, 3, &custom);

    /* Set quad-specific extensions directly */
    params->extension[QUAD_PEXT_ARM_LENGTH][3] = 0.3f;
    params->extension[QUAD_PEXT_K_THRUST][3] = 5.0e-9f;
    params->extension[QUAD_PEXT_K_TORQUE][3] = 2.0e-11f;
    params->extension[QUAD_PEXT_K_DRAG][3] = 0.5f;
    params->extension[QUAD_PEXT_K_ANG_DAMP][3] = 0.05f;
    params->extension[QUAD_PEXT_MOTOR_TAU][3] = 0.05f;
    params->extension[QUAD_PEXT_MAX_RPM][3] = 5000.0f;

    PlatformParamsAoS result = platform_params_get(params, 3);

    /* Rigid body fields via AoS accessor */
    ASSERT_FLOAT_EQ(result.mass, 2.5f);
    ASSERT_FLOAT_EQ(result.ixx, 0.05f);
    ASSERT_FLOAT_EQ(result.iyy, 0.06f);
    ASSERT_FLOAT_EQ(result.izz, 0.07f);
    ASSERT_FLOAT_EQ(result.collision_radius, 0.35f);
    ASSERT_FLOAT_EQ(result.max_vel, 50.0f);
    ASSERT_FLOAT_EQ(result.max_omega, 30.0f);
    ASSERT_FLOAT_EQ(result.gravity, 1.62f);

    /* Quad-specific fields via extensions */
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_ARM_LENGTH][3], 0.3f);
    ASSERT_FLOAT_NEAR(params->extension[QUAD_PEXT_K_THRUST][3], 5.0e-9f, 1e-11f);
    ASSERT_FLOAT_NEAR(params->extension[QUAD_PEXT_K_TORQUE][3], 2.0e-11f, 1e-13f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_K_DRAG][3], 0.5f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_K_ANG_DAMP][3], 0.05f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_MOTOR_TAU][3], 0.05f);
    ASSERT_FLOAT_EQ(params->extension[QUAD_PEXT_MAX_RPM][3], 5000.0f);

    /* Verify adjacent index is unchanged (still has defaults) */
    PlatformParamsAoS neighbor = platform_params_get(params, 2);
    ASSERT_FLOAT_EQ(neighbor.mass, 0.5f);
    ASSERT_FLOAT_EQ(neighbor.gravity, 9.81f);

    arena_destroy(arena);
    return 0;
}

/**
 * platform_params_get at out-of-bounds index returns zeroed struct.
 */
TEST(params_get_oob_returns_zeroed) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformParamsSOA* params = platform_params_create(arena, 4, QUAD_PARAMS_EXT_COUNT);
    ASSERT_NOT_NULL(params);

    PlatformParamsAoS result = platform_params_get(params, 4); /* out of bounds */
    ASSERT_FLOAT_EQ(result.mass, 0.0f);
    ASSERT_FLOAT_EQ(result.gravity, 0.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 10: Episode Data
 * ============================================================================ */

/**
 * Episode data init: verify all fields including best_episode_return default.
 */
TEST(episode_init_all_fields) {
    Arena* arena = arena_create(1024 * 1024);
    AgentEpisodeData* episodes = agent_episode_create(arena, 8);
    ASSERT_NOT_NULL(episodes);

    agent_episode_init(episodes, 3, 42, 7);

    ASSERT_FLOAT_EQ(episodes[3].episode_return, 0.0f);
    /* best_episode_return is initialized to -1e30f (very negative) */
    ASSERT_TRUE(episodes[3].best_episode_return < -1e29f);
    ASSERT_EQ(episodes[3].episode_length, (uint32_t)0);
    ASSERT_EQ(episodes[3].total_episodes, (uint32_t)0);
    ASSERT_EQ(episodes[3].env_id, (uint32_t)42);
    ASSERT_EQ(episodes[3].agent_id, (uint32_t)7);
    ASSERT_EQ(episodes[3].done, (uint8_t)0);
    ASSERT_EQ(episodes[3].truncated, (uint8_t)0);
    ASSERT_EQ(episodes[3]._pad[0], (uint8_t)0);
    ASSERT_EQ(episodes[3]._pad[1], (uint8_t)0);

    arena_destroy(arena);
    return 0;
}

/**
 * Episode data: verify that agent_episode_create initializes ALL entries,
 * not just the first one.
 */
TEST(episode_create_initializes_all) {
    Arena* arena = arena_create(1024 * 1024);
    uint32_t cap = 64;
    AgentEpisodeData* episodes = agent_episode_create(arena, cap);
    ASSERT_NOT_NULL(episodes);

    for (uint32_t i = 0; i < cap; i++) {
        ASSERT_FLOAT_EQ(episodes[i].episode_return, 0.0f);
        ASSERT_TRUE(episodes[i].best_episode_return < -1e29f);
        ASSERT_EQ(episodes[i].episode_length, (uint32_t)0);
        ASSERT_EQ(episodes[i].total_episodes, (uint32_t)0);
        /* agent_episode_create calls agent_episode_init(episodes, i, 0, i)
         * so env_id=0 and agent_id=i for all entries */
        ASSERT_EQ(episodes[i].env_id, (uint32_t)0);
        ASSERT_EQ(episodes[i].agent_id, i);
        ASSERT_EQ(episodes[i].done, (uint8_t)0);
        ASSERT_EQ(episodes[i].truncated, (uint8_t)0);
    }

    arena_destroy(arena);
    return 0;
}

/**
 * Episode data: done and truncated flag manipulation.
 * These are plain uint8_t fields -- verify write/read semantics.
 */
TEST(episode_done_truncated_transitions) {
    Arena* arena = arena_create(1024 * 1024);
    AgentEpisodeData* episodes = agent_episode_create(arena, 4);
    ASSERT_NOT_NULL(episodes);

    /* Initially not done, not truncated */
    ASSERT_EQ(episodes[0].done, (uint8_t)0);
    ASSERT_EQ(episodes[0].truncated, (uint8_t)0);

    /* Simulate episode termination */
    episodes[0].done = 1;
    ASSERT_EQ(episodes[0].done, (uint8_t)1);
    ASSERT_EQ(episodes[0].truncated, (uint8_t)0);

    /* Simulate truncation (time limit) */
    episodes[1].truncated = 1;
    ASSERT_EQ(episodes[1].done, (uint8_t)0);
    ASSERT_EQ(episodes[1].truncated, (uint8_t)1);

    /* Both done and truncated (edge case) */
    episodes[2].done = 1;
    episodes[2].truncated = 1;
    ASSERT_EQ(episodes[2].done, (uint8_t)1);
    ASSERT_EQ(episodes[2].truncated, (uint8_t)1);

    /* Re-init should reset both flags */
    agent_episode_init(episodes, 2, 0, 2);
    ASSERT_EQ(episodes[2].done, (uint8_t)0);
    ASSERT_EQ(episodes[2].truncated, (uint8_t)0);

    arena_destroy(arena);
    return 0;
}

/**
 * Episode data: episode_return and episode_length can be accumulated.
 * Verify basic accumulation + best return tracking pattern.
 */
TEST(episode_accumulation_pattern) {
    Arena* arena = arena_create(1024 * 1024);
    AgentEpisodeData* episodes = agent_episode_create(arena, 1);
    ASSERT_NOT_NULL(episodes);

    /* Simulate 100 steps of reward accumulation */
    for (uint32_t step = 0; step < 100; step++) {
        episodes[0].episode_return += 1.5f;
        episodes[0].episode_length++;
    }

    ASSERT_FLOAT_NEAR(episodes[0].episode_return, 150.0f, 0.01f);
    ASSERT_EQ(episodes[0].episode_length, (uint32_t)100);

    /* Track best return */
    if (episodes[0].episode_return > episodes[0].best_episode_return) {
        episodes[0].best_episode_return = episodes[0].episode_return;
    }
    ASSERT_FLOAT_NEAR(episodes[0].best_episode_return, 150.0f, 0.01f);

    /* Increment total episodes */
    episodes[0].total_episodes++;
    ASSERT_EQ(episodes[0].total_episodes, (uint32_t)1);

    /* Re-init for new episode preserves nothing (full reset) */
    agent_episode_init(episodes, 0, 0, 0);
    ASSERT_FLOAT_EQ(episodes[0].episode_return, 0.0f);
    ASSERT_EQ(episodes[0].episode_length, (uint32_t)0);
    /* best_episode_return is also reset by init (to -1e30f) */
    ASSERT_TRUE(episodes[0].best_episode_return < -1e29f);
    /* total_episodes is also reset */
    ASSERT_EQ(episodes[0].total_episodes, (uint32_t)0);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 11: Memory Size -- Monotonicity, Consistency
 * ============================================================================ */

/**
 * Memory size must be monotonically increasing with capacity.
 */
TEST(memory_size_monotonic_increasing) {
    size_t prev_state = 0;
    size_t prev_params = 0;

    uint32_t capacities[] = {1, 2, 4, 7, 8, 15, 16, 31, 32, 63, 64, 128, 256, 512, 1024, 4096};
    int n = (int)(sizeof(capacities) / sizeof(capacities[0]));

    for (int i = 0; i < n; i++) {
        size_t state_sz = platform_state_memory_size(capacities[i], QUAD_STATE_EXT_COUNT);
        size_t params_sz = platform_params_memory_size(capacities[i], QUAD_PARAMS_EXT_COUNT);

        /* Non-decreasing due to 32-byte alignment (small capacities round up identically) */
        ASSERT_GE(state_sz, prev_state);
        ASSERT_GE(params_sz, prev_params);

        prev_state = state_sz;
        prev_params = params_sz;
    }

    /* But overall it must grow: first < last */
    size_t first_state = platform_state_memory_size(capacities[0], QUAD_STATE_EXT_COUNT);
    size_t last_state = platform_state_memory_size(capacities[n - 1], QUAD_STATE_EXT_COUNT);
    ASSERT_GT(last_state, first_state);

    return 0;
}

/**
 * Memory size at capacity=1 should be:
 *   sizeof(PlatformStateSOA) + 13 rigid body arrays * 32
 *   + ext_count * sizeof(float*)  (extension pointer array)
 *   + ext_count * 32  (extension data arrays)
 * Each aligned array for 1 float (4 bytes) aligned to 32 bytes = 32 bytes.
 */
TEST(memory_size_capacity_one_exact) {
    size_t aligned = 32;  /* align_up(1 * sizeof(float), 32) */
    size_t expected = sizeof(PlatformStateSOA)
                    + RIGID_BODY_STATE_ARRAY_COUNT * aligned
                    + QUAD_STATE_EXT_COUNT * sizeof(float*)
                    + QUAD_STATE_EXT_COUNT * aligned;
    size_t actual = platform_state_memory_size(1, QUAD_STATE_EXT_COUNT);
    ASSERT_EQ(actual, expected);

    size_t expected_params = sizeof(PlatformParamsSOA)
                           + RIGID_BODY_PARAMS_ARRAY_COUNT * aligned
                           + QUAD_PARAMS_EXT_COUNT * sizeof(float*)
                           + QUAD_PARAMS_EXT_COUNT * aligned;
    size_t actual_params = platform_params_memory_size(1, QUAD_PARAMS_EXT_COUNT);
    ASSERT_EQ(actual_params, expected_params);

    return 0;
}

/**
 * Memory size at capacity=8: each array is 8*4=32 bytes, already 32-byte aligned.
 */
TEST(memory_size_capacity_eight) {
    size_t aligned = 32;  /* align_up(8 * sizeof(float), 32) = 32 */
    size_t expected = sizeof(PlatformStateSOA)
                    + RIGID_BODY_STATE_ARRAY_COUNT * aligned
                    + QUAD_STATE_EXT_COUNT * sizeof(float*)
                    + QUAD_STATE_EXT_COUNT * aligned;
    size_t actual = platform_state_memory_size(8, QUAD_STATE_EXT_COUNT);
    ASSERT_EQ(actual, expected);

    return 0;
}

/**
 * Memory size at capacity=9: 9*4=36 bytes, aligned up to 64 (next multiple of 32).
 */
TEST(memory_size_capacity_nine) {
    size_t aligned = 64;  /* align_up(9 * sizeof(float), 32) = 64 */
    size_t expected = sizeof(PlatformStateSOA)
                    + RIGID_BODY_STATE_ARRAY_COUNT * aligned
                    + QUAD_STATE_EXT_COUNT * sizeof(float*)
                    + QUAD_STATE_EXT_COUNT * aligned;
    size_t actual = platform_state_memory_size(9, QUAD_STATE_EXT_COUNT);
    ASSERT_EQ(actual, expected);

    return 0;
}

/* ============================================================================
 * Section 12: AoS Accessor Edge Cases
 * ============================================================================ */

/**
 * platform_state_set then platform_state_get should produce exact bit-for-bit
 * equality for all fields, including negative values and small magnitudes.
 */
TEST(accessor_roundtrip_negative_and_small_values) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 4, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    PlatformStateAoS aos = {
        .position = VEC3(-1000.0f, -0.001f, 1e-7f),
        .velocity = VEC3(-50.0f, 0.0f, 1e-10f),
        .orientation = QUAT(0.5f, -0.5f, 0.5f, -0.5f), /* valid unit quat */
        .omega = VEC3(-3.14f, 0.0f, 3.14f),
    };

    platform_state_set(states, 1, &aos);
    PlatformStateAoS result = platform_state_get(states, 1);

    /* Exact float equality for values that are exactly representable */
    ASSERT_FLOAT_EQ(result.position.x, -1000.0f);
    ASSERT_FLOAT_EQ(result.position.y, -0.001f);
    ASSERT_FLOAT_NEAR(result.position.z, 1e-7f, 1e-12f);
    ASSERT_FLOAT_EQ(result.velocity.x, -50.0f);
    ASSERT_FLOAT_EQ(result.velocity.y, 0.0f);
    ASSERT_FLOAT_NEAR(result.velocity.z, 1e-10f, 1e-15f);

    /* Quaternion components including negative */
    ASSERT_FLOAT_EQ(result.orientation.w, 0.5f);
    ASSERT_FLOAT_EQ(result.orientation.x, -0.5f);
    ASSERT_FLOAT_EQ(result.orientation.y, 0.5f);
    ASSERT_FLOAT_EQ(result.orientation.z, -0.5f);

    ASSERT_FLOAT_EQ(result.omega.x, -3.14f);
    ASSERT_FLOAT_EQ(result.omega.y, 0.0f);
    ASSERT_FLOAT_EQ(result.omega.z, 3.14f);

    /* Verify extension set/get at index 1 (same as rigid body set/get index) */
    states->extension[QUAD_EXT_RPM_0][1] = 0.0f;
    states->extension[QUAD_EXT_RPM_1][1] = 0.001f;
    states->extension[QUAD_EXT_RPM_2][1] = 999.9f;
    states->extension[QUAD_EXT_RPM_3][1] = 2500.0f;
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_0][1], 0.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_1][1], 0.001f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_2][1], 999.9f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_3][1], 2500.0f);

    arena_destroy(arena);
    return 0;
}

/**
 * Multiple set operations at the same index: last write wins.
 */
TEST(accessor_last_write_wins) {
    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 4, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    PlatformStateAoS first = {
        .position = VEC3(1.0f, 1.0f, 1.0f),
        .orientation = QUAT_IDENTITY,
    };
    PlatformStateAoS second = {
        .position = VEC3(2.0f, 2.0f, 2.0f),
        .orientation = QUAT_IDENTITY,
    };

    platform_state_set(states, 0, &first);
    platform_state_set(states, 0, &second);

    PlatformStateAoS result = platform_state_get(states, 0);
    ASSERT_FLOAT_EQ(result.position.x, 2.0f);
    ASSERT_FLOAT_EQ(result.position.y, 2.0f);
    ASSERT_FLOAT_EQ(result.position.z, 2.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 13: Null Pointer Safety
 * ============================================================================ */

/**
 * All functions should handle NULL arena gracefully (return NULL).
 */
TEST(null_arena_returns_null) {
    PlatformStateSOA* states = platform_state_create(NULL, 10, QUAD_STATE_EXT_COUNT);
    ASSERT_NULL(states);

    PlatformParamsSOA* params = platform_params_create(NULL, 10, QUAD_PARAMS_EXT_COUNT);
    ASSERT_NULL(params);

    AgentEpisodeData* episodes = agent_episode_create(NULL, 10);
    ASSERT_NULL(episodes);

    return 0;
}

/**
 * platform_state_validate with NULL states returns false.
 */
TEST(validate_null_states_returns_false) {
    ASSERT_FALSE(platform_state_validate(NULL, 0));
    return 0;
}

/* ============================================================================
 * Section 14: platform_state_zero_flat on SIMD-remainder Capacities
 *
 * The SIMD path processes in chunks of 8 (AVX2) or 4 (NEON), then handles
 * the remainder with scalar code. Non-multiple capacities exercise this.
 * ============================================================================ */

TEST(zero_simd_remainder_capacity_1) {
    Arena* arena = arena_create(64 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 1, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    /* Dirty and re-zero */
    states->rigid_body.pos_x[0] = 999.0f;
    states->rigid_body.quat_w[0] = 0.0f;
    states->extension[QUAD_EXT_RPM_3][0] = 555.0f;

    platform_state_zero(states);

    ASSERT_FLOAT_EQ(states->rigid_body.pos_x[0], 0.0f);
    ASSERT_FLOAT_EQ(states->rigid_body.quat_w[0], 1.0f);
    ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_3][0], 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(zero_simd_remainder_capacity_3) {
    Arena* arena = arena_create(64 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, 3, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    for (uint32_t i = 0; i < 3; i++) {
        states->rigid_body.vel_z[i] = 123.0f;
        states->rigid_body.quat_w[i] = 0.0f;
    }

    platform_state_zero(states);

    for (uint32_t i = 0; i < 3; i++) {
        ASSERT_FLOAT_EQ(states->rigid_body.vel_z[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.quat_w[i], 1.0f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(zero_simd_remainder_capacity_5) {
    Arena* arena = arena_create(64 * 1024);
    /* 5 = 4 + 1 (NEON-aligned + 1 remainder) or 0 + 5 (AVX2 all remainder) */
    PlatformStateSOA* states = platform_state_create(arena, 5, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    for (uint32_t i = 0; i < 5; i++) {
        states->rigid_body.omega_y[i] = 77.0f;
        states->rigid_body.quat_w[i] = 0.0f;
        states->extension[QUAD_EXT_RPM_2][i] = 88.0f;
    }

    platform_state_zero(states);

    for (uint32_t i = 0; i < 5; i++) {
        ASSERT_FLOAT_EQ(states->rigid_body.omega_y[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.quat_w[i], 1.0f);
        ASSERT_FLOAT_EQ(states->extension[QUAD_EXT_RPM_2][i], 0.0f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(zero_simd_remainder_capacity_9) {
    Arena* arena = arena_create(64 * 1024);
    /* 9 = 8 + 1 (AVX2-aligned + 1 remainder) or 8 + 1 (NEON 2 chunks + 1) */
    PlatformStateSOA* states = platform_state_create(arena, 9, QUAD_STATE_EXT_COUNT);
    ASSERT_NOT_NULL(states);

    for (uint32_t i = 0; i < 9; i++) {
        states->rigid_body.pos_x[i] = (float)i + 100.0f;
        states->rigid_body.quat_w[i] = 0.0f;
    }

    platform_state_zero(states);

    for (uint32_t i = 0; i < 9; i++) {
        ASSERT_FLOAT_EQ(states->rigid_body.pos_x[i], 0.0f);
        ASSERT_FLOAT_EQ(states->rigid_body.quat_w[i], 1.0f);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Drone State Deep Tests (Scale, Boundary, Copy Semantics)");

    /* Section 1: Create at extreme capacities */
    RUN_TEST(create_capacity_zero_returns_null);
    RUN_TEST(memory_size_zero_capacity_returns_zero);
    RUN_TEST(create_capacity_one_all_operations);
    RUN_TEST(copy_capacity_one);
    RUN_TEST(create_capacity_4096_stress);

    /* Section 2: Boundary index access */
    RUN_TEST(boundary_last_valid_index_operations);
    RUN_TEST(boundary_first_invalid_index_validate);
    RUN_TEST(boundary_first_invalid_index_get_returns_identity);
    RUN_TEST(boundary_reset_batch_oob_index_skipped);

    /* Section 3: Copy semantics */
    RUN_TEST(copy_count_zero_is_noop);
    RUN_TEST(copy_all_17_arrays_with_offsets);
    RUN_TEST(copy_deep_independence_all_arrays);
    RUN_TEST(copy_dst_overflow_rejected);
    RUN_TEST(copy_src_overflow_rejected);

    /* Section 4: Reset batch semantics */
    RUN_TEST(reset_batch_count_zero_is_noop);
    RUN_TEST(reset_batch_full_capacity);
    RUN_TEST(reset_batch_custom_orientation);

    /* Section 5: Validation -- NaN, Inf, quaternion edge cases */
    RUN_TEST(validate_nan_in_omega);
    RUN_TEST(validate_nan_in_rpm);
    RUN_TEST(validate_inf_in_position_rejected);
    RUN_TEST(validate_inf_in_velocity_rejected);
    RUN_TEST(validate_inf_in_omega_rejected);
    RUN_TEST(validate_inf_in_quaternion_rejected);
    RUN_TEST(validate_zero_quaternion_rejected);
    RUN_TEST(validate_near_unit_quaternion_within_tolerance);
    RUN_TEST(validate_near_unit_quaternion_outside_tolerance);
    RUN_TEST(validate_negative_rpm_each_motor);
    RUN_TEST(validate_positive_inf_rpm_rejected);

    /* Section 6: SoA alignment */
    RUN_TEST(alignment_state_all_17_arrays_capacity_7);
    RUN_TEST(alignment_params_all_15_arrays);
    RUN_TEST(alignment_state_arrays_no_aliasing);

    /* Section 7: Zero + validate interaction */
    RUN_TEST(zero_then_validate_passes);
    RUN_TEST(zero_quaternion_w_is_exact_one);

    /* Section 8: Init correctness */
    RUN_TEST(init_all_fields_exact_defaults);
    RUN_TEST(init_does_not_affect_neighbors);

    /* Section 9: PlatformParamsSOA */
    RUN_TEST(params_capacity_one_defaults);
    RUN_TEST(params_capacity_1024_all_defaults);
    RUN_TEST(params_accessor_full_roundtrip_all_15_fields);
    RUN_TEST(params_get_oob_returns_zeroed);

    /* Section 10: Episode data */
    RUN_TEST(episode_init_all_fields);
    RUN_TEST(episode_create_initializes_all);
    RUN_TEST(episode_done_truncated_transitions);
    RUN_TEST(episode_accumulation_pattern);

    /* Section 11: Memory size */
    RUN_TEST(memory_size_monotonic_increasing);
    RUN_TEST(memory_size_capacity_one_exact);
    RUN_TEST(memory_size_capacity_eight);
    RUN_TEST(memory_size_capacity_nine);

    /* Section 12: AoS accessor edge cases */
    RUN_TEST(accessor_roundtrip_negative_and_small_values);
    RUN_TEST(accessor_last_write_wins);

    /* Section 13: Null pointer safety */
    RUN_TEST(null_arena_returns_null);
    RUN_TEST(validate_null_states_returns_false);

    /* Section 14: SIMD remainder handling */
    RUN_TEST(zero_simd_remainder_capacity_1);
    RUN_TEST(zero_simd_remainder_capacity_3);
    RUN_TEST(zero_simd_remainder_capacity_5);
    RUN_TEST(zero_simd_remainder_capacity_9);

    TEST_SUITE_END();
}
