/**
 * Rigid Body State Module Tests
 *
 * Unit tests for RigidBodyStateSOA, RigidBodyParamsSOA,
 * PlatformStateSOA, and PlatformParamsSOA.
 */

#include "../include/rigid_body_state.h"
#include <stdio.h>
#include <math.h>

#define TEST_CAPACITY 64
#define TEST_ARENA_SIZE (1024 * 1024)  /* 1 MB */

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  [%d] %-50s ", tests_run, name); \
    fflush(stdout); \
} while(0)

#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); } while(0)

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { FAIL(msg); return; } \
} while(0)

#define ASSERT_FLOAT_EQ(a, b, eps, msg) do { \
    if (fabsf((a) - (b)) > (eps)) { \
        printf("FAIL: %s (got %f, expected %f)\n", msg, (double)(a), (double)(b)); \
        return; \
    } \
} while(0)

/* ============================================================================
 * RigidBodyStateSOA Tests
 * ============================================================================ */

static void test_rb_state_create(void) {
    TEST("rigid_body_state_create");

    Arena* arena = arena_create(TEST_ARENA_SIZE);
    ASSERT_TRUE(arena != NULL, "arena creation failed");

    RigidBodyStateSOA* states = rigid_body_state_create(arena, TEST_CAPACITY);
    ASSERT_TRUE(states != NULL, "state creation failed");
    ASSERT_TRUE(states->capacity == TEST_CAPACITY, "wrong capacity");
    ASSERT_TRUE(states->count == 0, "count should be 0");
    ASSERT_TRUE(states->pos_x != NULL, "pos_x is NULL");
    ASSERT_TRUE(states->omega_z != NULL, "omega_z is NULL");

    /* Verify identity quaternion */
    ASSERT_FLOAT_EQ(states->quat_w[0], 1.0f, 1e-6f, "quat_w[0] should be 1");
    ASSERT_FLOAT_EQ(states->quat_x[0], 0.0f, 1e-6f, "quat_x[0] should be 0");

    /* Verify zero velocity */
    ASSERT_FLOAT_EQ(states->vel_x[0], 0.0f, 1e-6f, "vel_x[0] should be 0");

    arena_destroy(arena);
    PASS();
}

static void test_rb_state_create_null(void) {
    TEST("rigid_body_state_create NULL safety");

    ASSERT_TRUE(rigid_body_state_create(NULL, TEST_CAPACITY) == NULL, "NULL arena should fail");

    Arena* arena = arena_create(TEST_ARENA_SIZE);
    ASSERT_TRUE(rigid_body_state_create(arena, 0) == NULL, "zero capacity should fail");
    arena_destroy(arena);
    PASS();
}

static void test_rb_state_zero(void) {
    TEST("rigid_body_state_zero");

    Arena* arena = arena_create(TEST_ARENA_SIZE);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, TEST_CAPACITY);
    ASSERT_TRUE(states != NULL, "state creation failed");

    /* Dirty the data */
    states->pos_x[0] = 42.0f;
    states->vel_z[0] = -99.0f;
    states->quat_w[0] = 0.5f;

    rigid_body_state_zero(states);

    ASSERT_FLOAT_EQ(states->pos_x[0], 0.0f, 1e-6f, "pos_x should be 0");
    ASSERT_FLOAT_EQ(states->vel_z[0], 0.0f, 1e-6f, "vel_z should be 0");
    ASSERT_FLOAT_EQ(states->quat_w[0], 1.0f, 1e-6f, "quat_w should be 1");
    ASSERT_FLOAT_EQ(states->quat_x[0], 0.0f, 1e-6f, "quat_x should be 0");

    arena_destroy(arena);
    PASS();
}

static void test_rb_state_reset_batch(void) {
    TEST("rigid_body_state_reset_batch");

    Arena* arena = arena_create(TEST_ARENA_SIZE);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, TEST_CAPACITY);
    ASSERT_TRUE(states != NULL, "state creation failed");

    /* Set some velocity */
    states->vel_x[5] = 10.0f;
    states->vel_y[5] = 20.0f;

    uint32_t indices[] = {5, 10};
    Vec3 positions[] = {VEC3(1.0f, 2.0f, 3.0f), VEC3(4.0f, 5.0f, 6.0f)};
    Quat orientations[] = {QUAT_IDENTITY, QUAT_IDENTITY};

    rigid_body_state_reset_batch(states, indices, positions, orientations, 2);

    ASSERT_FLOAT_EQ(states->pos_x[5], 1.0f, 1e-6f, "pos_x[5]");
    ASSERT_FLOAT_EQ(states->pos_y[5], 2.0f, 1e-6f, "pos_y[5]");
    ASSERT_FLOAT_EQ(states->pos_z[5], 3.0f, 1e-6f, "pos_z[5]");
    ASSERT_FLOAT_EQ(states->vel_x[5], 0.0f, 1e-6f, "vel_x[5] should be zeroed");
    ASSERT_FLOAT_EQ(states->vel_y[5], 0.0f, 1e-6f, "vel_y[5] should be zeroed");

    ASSERT_FLOAT_EQ(states->pos_x[10], 4.0f, 1e-6f, "pos_x[10]");

    arena_destroy(arena);
    PASS();
}

static void test_rb_state_copy(void) {
    TEST("rigid_body_state_copy");

    Arena* arena = arena_create(TEST_ARENA_SIZE);
    RigidBodyStateSOA* src = rigid_body_state_create(arena, TEST_CAPACITY);
    RigidBodyStateSOA* dst = rigid_body_state_create(arena, TEST_CAPACITY);
    ASSERT_TRUE(src != NULL && dst != NULL, "creation failed");

    src->pos_x[0] = 1.0f;
    src->pos_y[0] = 2.0f;
    src->pos_z[0] = 3.0f;
    src->quat_w[0] = 0.707f;
    src->quat_z[0] = 0.707f;

    rigid_body_state_copy(dst, src, 5, 0, 1);

    ASSERT_FLOAT_EQ(dst->pos_x[5], 1.0f, 1e-6f, "copied pos_x");
    ASSERT_FLOAT_EQ(dst->pos_y[5], 2.0f, 1e-6f, "copied pos_y");
    ASSERT_FLOAT_EQ(dst->quat_w[5], 0.707f, 1e-3f, "copied quat_w");

    arena_destroy(arena);
    PASS();
}

static void test_rb_state_memory_size(void) {
    TEST("rigid_body_state_memory_size");

    ASSERT_TRUE(rigid_body_state_memory_size(0) == 0, "zero capacity should return 0");

    size_t size = rigid_body_state_memory_size(TEST_CAPACITY);
    ASSERT_TRUE(size > 0, "size should be positive");

    /* 13 arrays * aligned(64*4, 32) + struct */
    size_t aligned_array = align_up_size(TEST_CAPACITY * sizeof(float), 32);
    size_t expected = sizeof(RigidBodyStateSOA) + 13 * aligned_array;
    ASSERT_TRUE(size == expected, "size mismatch");

    arena_destroy(NULL); /* no-op, just for balance */
    PASS();
}

/* ============================================================================
 * RigidBodyParamsSOA Tests
 * ============================================================================ */

static void test_rb_params_create(void) {
    TEST("rigid_body_params_create");

    Arena* arena = arena_create(TEST_ARENA_SIZE);
    RigidBodyParamsSOA* params = rigid_body_params_create(arena, TEST_CAPACITY);
    ASSERT_TRUE(params != NULL, "params creation failed");
    ASSERT_TRUE(params->capacity == TEST_CAPACITY, "wrong capacity");
    ASSERT_TRUE(params->mass != NULL, "mass is NULL");

    /* Verify defaults */
    ASSERT_FLOAT_EQ(params->mass[0], 0.5f, 1e-6f, "default mass");
    ASSERT_FLOAT_EQ(params->gravity[0], 9.81f, 1e-2f, "default gravity");
    ASSERT_FLOAT_EQ(params->collision_radius[0], 0.15f, 1e-6f, "default collision_radius");
    ASSERT_FLOAT_EQ(params->max_vel[0], 20.0f, 1e-6f, "default max_vel");

    arena_destroy(arena);
    PASS();
}

static void test_rb_params_memory_size(void) {
    TEST("rigid_body_params_memory_size");

    ASSERT_TRUE(rigid_body_params_memory_size(0) == 0, "zero capacity should return 0");

    size_t size = rigid_body_params_memory_size(TEST_CAPACITY);
    size_t aligned_array = align_up_size(TEST_CAPACITY * sizeof(float), 32);
    size_t expected = sizeof(RigidBodyParamsSOA) + 8 * aligned_array;
    ASSERT_TRUE(size == expected, "size mismatch");

    PASS();
}

/* ============================================================================
 * PlatformStateSOA Tests
 * ============================================================================ */

static void test_platform_state_create_no_ext(void) {
    TEST("platform_state_create (no extensions)");

    Arena* arena = arena_create(TEST_ARENA_SIZE);
    PlatformStateSOA* pstate = platform_state_create(arena, TEST_CAPACITY, 0);
    ASSERT_TRUE(pstate != NULL, "creation failed");
    ASSERT_TRUE(pstate->extension == NULL, "extension should be NULL");
    ASSERT_TRUE(pstate->extension_count == 0, "extension_count should be 0");
    ASSERT_TRUE(pstate->rigid_body.capacity == TEST_CAPACITY, "wrong capacity");
    ASSERT_FLOAT_EQ(pstate->rigid_body.quat_w[0], 1.0f, 1e-6f, "identity quat");

    arena_destroy(arena);
    PASS();
}

static void test_platform_state_create_with_ext(void) {
    TEST("platform_state_create (4 extensions)");

    Arena* arena = arena_create(TEST_ARENA_SIZE);
    PlatformStateSOA* pstate = platform_state_create(arena, TEST_CAPACITY, 4);
    ASSERT_TRUE(pstate != NULL, "creation failed");
    ASSERT_TRUE(pstate->extension != NULL, "extension should not be NULL");
    ASSERT_TRUE(pstate->extension_count == 4, "extension_count should be 4");

    /* Extensions should be zero-initialized */
    for (uint32_t ext = 0; ext < 4; ext++) {
        ASSERT_TRUE(pstate->extension[ext] != NULL, "extension array is NULL");
        ASSERT_FLOAT_EQ(pstate->extension[ext][0], 0.0f, 1e-6f, "extension should be 0");
    }

    /* Write to extensions */
    pstate->extension[0][0] = 100.0f;
    pstate->extension[3][TEST_CAPACITY - 1] = 999.0f;
    ASSERT_FLOAT_EQ(pstate->extension[0][0], 100.0f, 1e-6f, "ext write");
    ASSERT_FLOAT_EQ(pstate->extension[3][TEST_CAPACITY - 1], 999.0f, 1e-6f, "ext write end");

    arena_destroy(arena);
    PASS();
}

static void test_platform_state_zero(void) {
    TEST("platform_state_zero");

    Arena* arena = arena_create(TEST_ARENA_SIZE);
    PlatformStateSOA* pstate = platform_state_create(arena, TEST_CAPACITY, 4);
    ASSERT_TRUE(pstate != NULL, "creation failed");

    /* Dirty data */
    pstate->rigid_body.pos_x[0] = 42.0f;
    pstate->extension[0][0] = 123.0f;
    pstate->extension[3][5] = 456.0f;

    platform_state_zero(pstate);

    ASSERT_FLOAT_EQ(pstate->rigid_body.pos_x[0], 0.0f, 1e-6f, "pos_x zeroed");
    ASSERT_FLOAT_EQ(pstate->rigid_body.quat_w[0], 1.0f, 1e-6f, "identity quat");
    ASSERT_FLOAT_EQ(pstate->extension[0][0], 0.0f, 1e-6f, "ext[0] zeroed");
    ASSERT_FLOAT_EQ(pstate->extension[3][5], 0.0f, 1e-6f, "ext[3] zeroed");

    arena_destroy(arena);
    PASS();
}

static void test_platform_state_memory_size(void) {
    TEST("platform_state_memory_size");

    size_t size0 = platform_state_memory_size(TEST_CAPACITY, 0);
    size_t size4 = platform_state_memory_size(TEST_CAPACITY, 4);

    ASSERT_TRUE(size0 > 0, "should be positive");
    ASSERT_TRUE(size4 > size0, "4 extensions should be larger");

    size_t aligned_array = align_up_size(TEST_CAPACITY * sizeof(float), 32);
    size_t ext_overhead = 4 * sizeof(float*) + 4 * aligned_array;
    ASSERT_TRUE(size4 - size0 == ext_overhead, "extension overhead mismatch");

    PASS();
}

/* ============================================================================
 * PlatformParamsSOA Tests
 * ============================================================================ */

static void test_platform_params_create(void) {
    TEST("platform_params_create (7 extensions)");

    Arena* arena = arena_create(TEST_ARENA_SIZE);
    PlatformParamsSOA* pparams = platform_params_create(arena, TEST_CAPACITY, 7);
    ASSERT_TRUE(pparams != NULL, "creation failed");
    ASSERT_TRUE(pparams->extension_count == 7, "extension_count should be 7");
    ASSERT_TRUE(pparams->extension != NULL, "extension should not be NULL");

    /* Rigid body defaults */
    ASSERT_FLOAT_EQ(pparams->rigid_body.mass[0], 0.5f, 1e-6f, "default mass");
    ASSERT_FLOAT_EQ(pparams->rigid_body.gravity[0], 9.81f, 1e-2f, "default gravity");

    /* Extensions zero-initialized */
    for (uint32_t ext = 0; ext < 7; ext++) {
        ASSERT_TRUE(pparams->extension[ext] != NULL, "ext array NULL");
        ASSERT_FLOAT_EQ(pparams->extension[ext][0], 0.0f, 1e-6f, "ext should be 0");
    }

    arena_destroy(arena);
    PASS();
}

static void test_platform_params_memory_size(void) {
    TEST("platform_params_memory_size");

    size_t size0 = platform_params_memory_size(TEST_CAPACITY, 0);
    size_t size7 = platform_params_memory_size(TEST_CAPACITY, 7);

    ASSERT_TRUE(size0 > 0, "should be positive");
    ASSERT_TRUE(size7 > size0, "7 extensions should be larger");

    PASS();
}

/* ============================================================================
 * Alignment Verification
 * ============================================================================ */

static void test_alignment(void) {
    TEST("32-byte alignment of all arrays");

    Arena* arena = arena_create(TEST_ARENA_SIZE);
    RigidBodyStateSOA* states = rigid_body_state_create(arena, TEST_CAPACITY);
    ASSERT_TRUE(states != NULL, "creation failed");

    ASSERT_TRUE(((uintptr_t)states->pos_x % 32) == 0, "pos_x alignment");
    ASSERT_TRUE(((uintptr_t)states->pos_y % 32) == 0, "pos_y alignment");
    ASSERT_TRUE(((uintptr_t)states->pos_z % 32) == 0, "pos_z alignment");
    ASSERT_TRUE(((uintptr_t)states->vel_x % 32) == 0, "vel_x alignment");
    ASSERT_TRUE(((uintptr_t)states->quat_w % 32) == 0, "quat_w alignment");
    ASSERT_TRUE(((uintptr_t)states->omega_z % 32) == 0, "omega_z alignment");

    PlatformStateSOA* pstate = platform_state_create(arena, TEST_CAPACITY, 4);
    ASSERT_TRUE(pstate != NULL, "platform creation failed");

    for (uint32_t i = 0; i < 4; i++) {
        ASSERT_TRUE(((uintptr_t)pstate->extension[i] % 32) == 0, "extension alignment");
    }

    arena_destroy(arena);
    PASS();
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("=== Rigid Body State Module Tests ===\n\n");

    printf("RigidBodyStateSOA:\n");
    test_rb_state_create();
    test_rb_state_create_null();
    test_rb_state_zero();
    test_rb_state_reset_batch();
    test_rb_state_copy();
    test_rb_state_memory_size();

    printf("\nRigidBodyParamsSOA:\n");
    test_rb_params_create();
    test_rb_params_memory_size();

    printf("\nPlatformStateSOA:\n");
    test_platform_state_create_no_ext();
    test_platform_state_create_with_ext();
    test_platform_state_zero();
    test_platform_state_memory_size();

    printf("\nPlatformParamsSOA:\n");
    test_platform_params_create();
    test_platform_params_memory_size();

    printf("\nAlignment:\n");
    test_alignment();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
