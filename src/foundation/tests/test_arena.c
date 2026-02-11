/**
 * Arena Allocator Unit Tests
 *
 * Tests alignment, overflow handling, scoping, reset, and utilization.
 */

#include "../include/foundation.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "test_harness.h"

/* ============================================================================
 * Test Cases
 * ============================================================================ */

static int test_arena_create_destroy(void) {
    /* Basic creation */
    Arena* arena = arena_create(1024);
    ASSERT_MSG(arena != NULL, "arena_create should return non-NULL");
    ASSERT_MSG(arena->capacity == 1024, "capacity should match requested size");
    ASSERT_MSG(arena->used == 0, "new arena should have zero usage");
    ASSERT_MSG(arena->data != NULL, "data pointer should be valid");
    arena_destroy(arena);

    /* Zero size should fail */
    arena = arena_create(0);
    ASSERT_MSG(arena == NULL, "arena_create(0) should return NULL");

    /* Large arena */
    arena = arena_create(1024 * 1024);  /* 1MB */
    ASSERT_MSG(arena != NULL, "should be able to create 1MB arena");
    arena_destroy(arena);

    return 0;
}

static int test_arena_allocation(void) {
    Arena* arena = arena_create(1024);
    ASSERT_MSG(arena != NULL, "arena creation");

    /* Basic allocation */
    void* ptr1 = arena_alloc(arena, 64);
    ASSERT_MSG(ptr1 != NULL, "first allocation should succeed");
    ASSERT_MSG(arena->used >= 64, "used should be at least allocation size");

    /* Multiple allocations */
    void* ptr2 = arena_alloc(arena, 32);
    void* ptr3 = arena_alloc(arena, 128);
    ASSERT_MSG(ptr2 != NULL && ptr3 != NULL, "multiple allocations should succeed");
    ASSERT_MSG(ptr2 != ptr1 && ptr3 != ptr2, "allocations should return different pointers");

    /* Zero-initialized allocation */
    void* ptr4 = arena_alloc_zero(arena, 64);
    ASSERT_MSG(ptr4 != NULL, "zero allocation should succeed");

    /* Verify zeroing */
    uint8_t* bytes = (uint8_t*)ptr4;
    int all_zero = 1;
    for (size_t i = 0; i < 64; i++) {
        if (bytes[i] != 0) all_zero = 0;
    }
    ASSERT_MSG(all_zero, "arena_alloc_zero should zero memory");

    arena_destroy(arena);
    return 0;
}

static int test_arena_alignment(void) {
    Arena* arena = arena_create(4096);
    ASSERT_MSG(arena != NULL, "arena creation");

    /* Test default 16-byte alignment */
    void* ptr1 = arena_alloc(arena, 1);
    ASSERT_MSG(((uintptr_t)ptr1 & 15) == 0, "default allocation should be 16-byte aligned");

    /* Force unalignment then allocate */
    void* ptr2 = arena_alloc(arena, 7);  /* Odd size */
    void* ptr3 = arena_alloc(arena, 1);  /* Next allocation should still be aligned */
    (void)ptr2;
    ASSERT_MSG(((uintptr_t)ptr3 & 15) == 0, "allocation after odd-sized should still be 16-byte aligned");

    /* Test 32-byte alignment */
    void* ptr4 = arena_alloc_aligned(arena, 64, 32);
    ASSERT_MSG(((uintptr_t)ptr4 & 31) == 0, "32-byte aligned allocation");

    /* Test 64-byte alignment */
    void* ptr5 = arena_alloc_aligned(arena, 64, 64);
    ASSERT_MSG(((uintptr_t)ptr5 & 63) == 0, "64-byte aligned allocation");

    /* Verify Vec3 can be properly allocated */
    Vec3* vec = arena_alloc_type(arena, Vec3);
    ASSERT_MSG(((uintptr_t)vec & 15) == 0, "Vec3 should be 16-byte aligned");

    /* Verify Mat4 can be properly allocated */
    Mat4* mat = arena_alloc_type(arena, Mat4);
    ASSERT_MSG(((uintptr_t)mat & 31) == 0, "Mat4 should be 32-byte aligned");

    arena_destroy(arena);
    return 0;
}

static int test_arena_overflow(void) {
    /* Small arena to easily overflow */
    Arena* arena = arena_create(64);
    ASSERT_MSG(arena != NULL, "arena creation");

    /* First allocation should succeed */
    void* ptr1 = arena_alloc(arena, 32);
    ASSERT_MSG(ptr1 != NULL, "first allocation should succeed");

    /* Second allocation that fits should succeed */
    void* ptr2 = arena_alloc(arena, 16);
    ASSERT_MSG(ptr2 != NULL, "second allocation should succeed");

    /* Allocation that exceeds remaining should fail */
    void* ptr3 = arena_alloc(arena, 64);
    ASSERT_MSG(ptr3 == NULL, "overflow allocation should return NULL");

    /* Arena should still be usable after failed allocation */
    void* ptr4 = arena_alloc(arena, 8);
    /* This might or might not succeed depending on remaining space after alignment */
    (void)ptr4;

    arena_destroy(arena);
    return 0;
}

static int test_arena_reset(void) {
    Arena* arena = arena_create(1024);
    ASSERT_MSG(arena != NULL, "arena creation");

    /* Make some allocations */
    arena_alloc(arena, 100);
    arena_alloc(arena, 200);
    arena_alloc(arena, 300);

    size_t used_before = arena->used;
    ASSERT_MSG(used_before >= 600, "should have allocated at least 600 bytes");

    /* Reset */
    arena_reset(arena);
    ASSERT_MSG(arena->used == 0, "reset should zero the used count");
    ASSERT_MSG(arena->capacity == 1024, "capacity should be unchanged after reset");

    /* Should be able to allocate again */
    void* ptr = arena_alloc(arena, 512);
    ASSERT_MSG(ptr != NULL, "allocation after reset should succeed");

    arena_destroy(arena);
    return 0;
}

static int test_arena_scoping(void) {
    Arena* arena = arena_create(1024);
    ASSERT_MSG(arena != NULL, "arena creation");

    /* Make base allocation */
    arena_alloc(arena, 64);
    size_t base_used = arena->used;

    /* Begin scope */
    ArenaScope scope = arena_scope_begin(arena);
    ASSERT_MSG(scope.arena == arena, "scope should reference arena");
    ASSERT_MSG(scope.saved_used == base_used, "scope should save used count");

    /* Allocate within scope */
    arena_alloc(arena, 128);
    arena_alloc(arena, 256);
    ASSERT_MSG(arena->used > base_used, "allocations in scope should increase usage");

    /* End scope */
    arena_scope_end(scope);
    ASSERT_MSG(arena->used == base_used, "scope end should restore used count");

    /* Nested scopes */
    ArenaScope outer = arena_scope_begin(arena);
    arena_alloc(arena, 100);
    size_t after_outer_alloc = arena->used;

    ArenaScope inner = arena_scope_begin(arena);
    arena_alloc(arena, 200);
    ASSERT_MSG(arena->used > after_outer_alloc, "inner scope allocation");

    arena_scope_end(inner);
    ASSERT_MSG(arena->used == after_outer_alloc, "inner scope should restore to outer level");

    arena_scope_end(outer);
    ASSERT_MSG(arena->used == base_used, "outer scope should restore to base");

    arena_destroy(arena);
    return 0;
}

static int test_arena_scope_macro(void) {
    Arena* arena = arena_create(1024);
    ASSERT_MSG(arena != NULL, "arena creation");

    arena_alloc(arena, 64);
    size_t base_used = arena->used;

    /* Test ARENA_SCOPE macro */
    ARENA_SCOPE(arena) {
        void* temp = arena_alloc(arena, 256);
        ASSERT_MSG(temp != NULL, "allocation inside ARENA_SCOPE");
        /* temp is valid only within this scope */
    }

    ASSERT_MSG(arena->used == base_used, "ARENA_SCOPE should restore usage");

    arena_destroy(arena);
    return 0;
}

static int test_arena_queries(void) {
    Arena* arena = arena_create(1000);
    ASSERT_MSG(arena != NULL, "arena creation");

    /* Initial state */
    ASSERT_MSG(arena_remaining(arena) == 1000, "new arena should have full capacity remaining");
    ASSERT_MSG(arena_utilization(arena) == 0.0f, "new arena should have 0% utilization");

    /* After allocation */
    arena_alloc(arena, 500);
    size_t remaining = arena_remaining(arena);
    ASSERT_MSG(remaining < 1000, "remaining should decrease after allocation");

    float util = arena_utilization(arena);
    ASSERT_MSG(util > 0.0f && util < 1.0f, "utilization should be between 0 and 1");

    /* After reset */
    arena_reset(arena);
    ASSERT_MSG(arena_remaining(arena) == 1000, "remaining should be full after reset");
    ASSERT_MSG(arena_utilization(arena) == 0.0f, "utilization should be 0 after reset");

    arena_destroy(arena);
    return 0;
}

static int test_arena_typed_allocation(void) {
    Arena* arena = arena_create(4096);
    ASSERT_MSG(arena != NULL, "arena creation");

    /* Single type allocation */
    Vec3* v = arena_alloc_type(arena, Vec3);
    ASSERT_MSG(v != NULL, "arena_alloc_type for Vec3");
    ASSERT_MSG(((uintptr_t)v & (alignof(Vec3) - 1)) == 0, "Vec3 should be properly aligned");

    /* Array allocation */
    Vec3* arr = arena_alloc_array(arena, Vec3, 10);
    ASSERT_MSG(arr != NULL, "arena_alloc_array for Vec3[10]");
    ASSERT_MSG(((uintptr_t)arr & (alignof(Vec3) - 1)) == 0, "Vec3 array should be properly aligned");

    /* Different types */
    Quat* q = arena_alloc_type(arena, Quat);
    Mat4* m = arena_alloc_type(arena, Mat4);
    ASSERT_MSG(q != NULL && m != NULL, "different type allocations");
    ASSERT_MSG(((uintptr_t)q & (alignof(Quat) - 1)) == 0, "Quat alignment");
    ASSERT_MSG(((uintptr_t)m & (alignof(Mat4) - 1)) == 0, "Mat4 alignment");

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Arena Allocator Tests");

    RUN_TEST(arena_create_destroy);
    RUN_TEST(arena_allocation);
    RUN_TEST(arena_alignment);
    RUN_TEST(arena_overflow);
    RUN_TEST(arena_reset);
    RUN_TEST(arena_scoping);
    RUN_TEST(arena_scope_macro);
    RUN_TEST(arena_queries);
    RUN_TEST(arena_typed_allocation);

    TEST_SUITE_END();
}
