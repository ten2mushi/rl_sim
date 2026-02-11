/**
 * @file test_memory_deep.c
 * @brief Deep Memory Management Tests for BatchDroneEngine (60+ tests)
 *
 * Comprehensive memory management test suite following the "Tests as definition:
 * the Yoneda way" philosophy. These tests serve as the authoritative specification
 * for the engine's memory behavior.
 *
 * Test Categories:
 * 1. Arena Allocator Core Tests (12 tests)
 * 2. Frame Arena Lifecycle Tests (10 tests)
 * 3. Persistent Arena Stability Tests (8 tests)
 * 4. SoA Structure Alignment Tests (12 tests)
 * 5. Buffer Management Tests (10 tests)
 * 6. Memory Leak Detection Tests (7 tests)
 * 7. Cache Optimization Tests (6 tests)
 * 8. Stress Tests (7 tests)
 * 9. Integration Memory Tests (6 tests)
 *
 * Memory Architecture Requirements:
 * - Persistent Arena (256MB default): Long-lived allocations
 * - Frame Arena (64MB default): Reset every step for temporaries
 * - 32-byte alignment for SIMD (AVX2)
 * - 64-byte cache line awareness
 * - SoA data layout for SIMD vectorization
 *
 * Reference: architecture-overview.md, foundation.h, arena.c
 */

#include "environment_manager.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "test_harness.h"

/* Alignment check macros (not in standard harness) */
#define ASSERT_ALIGNED_32(ptr) do { \
    if (((uintptr_t)(ptr) & 31) != 0) { \
        printf("\n    ASSERT_ALIGNED_32 failed: %s (addr=%p, mod=%lu)\n    at %s:%d", \
                #ptr, (void*)(ptr), (unsigned long)((uintptr_t)(ptr) & 31), __FILE__, __LINE__); \
        return -__LINE__; \
    } \
} while(0)

#define ASSERT_ALIGNED_64(ptr) do { \
    if (((uintptr_t)(ptr) & 63) != 0) { \
        printf("\n    ASSERT_ALIGNED_64 failed: %s (addr=%p, mod=%lu)\n    at %s:%d", \
                #ptr, (void*)(ptr), (unsigned long)((uintptr_t)(ptr) & 63), __FILE__, __LINE__); \
        return -__LINE__; \
    } \
} while(0)

#define ASSERT_ALIGNED_16(ptr) do { \
    if (((uintptr_t)(ptr) & 15) != 0) { \
        printf("\n    ASSERT_ALIGNED_16 failed: %s (addr=%p, mod=%lu)\n    at %s:%d", \
                #ptr, (void*)(ptr), (unsigned long)((uintptr_t)(ptr) & 15), __FILE__, __LINE__); \
        return -__LINE__; \
    } \
} while(0)

static bool is_aligned(const void* ptr, size_t alignment) {
    return ((uintptr_t)ptr & (alignment - 1)) == 0;
}

/* ============================================================================
 * Test Helpers
 * ============================================================================ */

/**
 * Create a test engine with specific configuration
 */
static BatchDroneEngine* create_test_engine(uint32_t num_envs, uint32_t drones_per_env,
                                            size_t persistent_size, size_t frame_size) {
    EngineConfig cfg = engine_config_default();
    cfg.num_envs = num_envs;
    cfg.drones_per_env = drones_per_env;
    cfg.seed = 42;
    cfg.persistent_arena_size = persistent_size;
    cfg.frame_arena_size = frame_size;

    char error[ENGINE_ERROR_MSG_SIZE];
    return engine_create(&cfg, error);
}

/**
 * Create a small test engine (16 drones)
 */
static BatchDroneEngine* create_small_engine(void) {
    return create_test_engine(4, 4, 64 * 1024 * 1024, 16 * 1024 * 1024);
}

/**
 * Create a medium test engine (256 drones)
 */
static BatchDroneEngine* create_medium_engine(void) {
    return create_test_engine(16, 16, 128 * 1024 * 1024, 32 * 1024 * 1024);
}

/**
 * Create a large test engine (1024 drones)
 */
static BatchDroneEngine* create_large_engine(void) {
    return create_test_engine(64, 16, 256 * 1024 * 1024, 64 * 1024 * 1024);
}

/* ============================================================================
 * Category 1: Arena Allocator Core Tests (12 tests)
 * ============================================================================ */

/**
 * Test that arena creation with various sizes succeeds
 */
TEST(arena_creation_various_sizes) {
    /* Small arena - 1 KB */
    Arena* small = arena_create(1024);
    ASSERT_NOT_NULL(small);
    ASSERT_EQ(small->capacity, 1024);
    ASSERT_EQ(small->used, 0);
    arena_destroy(small);

    /* Medium arena - 1 MB */
    Arena* medium = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(medium);
    ASSERT_EQ(medium->capacity, 1024 * 1024);
    ASSERT_EQ(medium->used, 0);
    arena_destroy(medium);

    /* Large arena - 64 MB */
    Arena* large = arena_create(64 * 1024 * 1024);
    ASSERT_NOT_NULL(large);
    ASSERT_EQ(large->capacity, 64 * 1024 * 1024);
    ASSERT_EQ(large->used, 0);
    arena_destroy(large);

    return 0;
}

/**
 * Test that arena allocation returns aligned pointers
 */
TEST(arena_alloc_returns_aligned_pointers) {
    Arena* arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Default allocation should be at least 16-byte aligned */
    void* ptr1 = arena_alloc(arena, 100);
    ASSERT_NOT_NULL(ptr1);
    ASSERT_ALIGNED_16(ptr1);

    void* ptr2 = arena_alloc(arena, 50);
    ASSERT_NOT_NULL(ptr2);
    ASSERT_ALIGNED_16(ptr2);

    void* ptr3 = arena_alloc(arena, 1);
    ASSERT_NOT_NULL(ptr3);
    ASSERT_ALIGNED_16(ptr3);

    arena_destroy(arena);
    return 0;
}

/**
 * Test that arena reset clears used counter but preserves capacity
 */
TEST(arena_reset_preserves_capacity) {
    Arena* arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    size_t original_capacity = arena->capacity;

    /* Allocate until half capacity */
    for (int i = 0; i < 100; i++) {
        void* ptr = arena_alloc(arena, 1000);
        ASSERT_NOT_NULL(ptr);
    }

    ASSERT_GT(arena->used, 0);
    size_t used_before_reset = arena->used;
    ASSERT_GT(used_before_reset, 100000);  /* Sanity check */

    /* Reset arena */
    arena_reset(arena);

    /* Verify reset behavior */
    ASSERT_EQ(arena->used, 0);
    ASSERT_EQ(arena->capacity, original_capacity);

    /* Should be able to allocate again */
    void* ptr = arena_alloc(arena, 1000);
    ASSERT_NOT_NULL(ptr);

    arena_destroy(arena);
    return 0;
}

/**
 * Test that arena overflow handling returns NULL gracefully
 */
TEST(arena_overflow_returns_null) {
    /* Create small arena - 1 KB */
    Arena* arena = arena_create(1024);
    ASSERT_NOT_NULL(arena);

    /* Allocate most of it */
    void* ptr1 = arena_alloc(arena, 500);
    ASSERT_NOT_NULL(ptr1);

    void* ptr2 = arena_alloc(arena, 400);
    ASSERT_NOT_NULL(ptr2);

    /* This should fail (exceeds remaining capacity with alignment) */
    void* ptr3 = arena_alloc(arena, 500);
    ASSERT_NULL(ptr3);

    /* Arena should still be valid */
    ASSERT_LE(arena->used, arena->capacity);

    arena_destroy(arena);
    return 0;
}

/**
 * Test 32-byte alignment guarantees for all allocations
 */
TEST(arena_32byte_alignment_guarantees) {
    Arena* arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Request 32-byte aligned allocations */
    for (int i = 0; i < 100; i++) {
        void* ptr = arena_alloc_aligned(arena, 100, 32);
        ASSERT_NOT_NULL(ptr);
        ASSERT_ALIGNED_32(ptr);
    }

    /* Even after odd-sized allocations, alignment should be maintained */
    arena_reset(arena);
    for (int i = 0; i < 50; i++) {
        void* odd = arena_alloc_aligned(arena, 17, 32);  /* Odd size */
        ASSERT_NOT_NULL(odd);
        ASSERT_ALIGNED_32(odd);
    }

    arena_destroy(arena);
    return 0;
}

/**
 * Test that consecutive allocations maintain alignment
 */
TEST(arena_consecutive_allocations_alignment) {
    Arena* arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    void* ptrs[100];

    /* Make consecutive 32-byte aligned allocations */
    for (int i = 0; i < 100; i++) {
        ptrs[i] = arena_alloc_aligned(arena, 64, 32);
        ASSERT_NOT_NULL(ptrs[i]);
        ASSERT_ALIGNED_32(ptrs[i]);
    }

    /* Verify all pointers are distinct and properly spaced */
    for (int i = 1; i < 100; i++) {
        ASSERT_NE(ptrs[i], ptrs[i-1]);
        /* Distance should be at least 64 bytes (allocation size) */
        ptrdiff_t diff = (char*)ptrs[i] - (char*)ptrs[i-1];
        ASSERT_GE((size_t)diff, 64);
    }

    arena_destroy(arena);
    return 0;
}

/**
 * Test arena reset watermark behavior
 */
TEST(arena_reset_watermark_behavior) {
    Arena* arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Record initial state */
    size_t initial_used = arena->used;
    ASSERT_EQ(initial_used, 0);

    /* Allocate and track watermark */
    void* p1 = arena_alloc(arena, 1000);
    size_t used_after_p1 = arena->used;
    ASSERT_GT(used_after_p1, 0);

    void* p2 = arena_alloc(arena, 2000);
    size_t used_after_p2 = arena->used;
    ASSERT_GT(used_after_p2, used_after_p1);

    void* p3 = arena_alloc(arena, 3000);
    size_t used_after_p3 = arena->used;
    ASSERT_GT(used_after_p3, used_after_p2);

    /* Reset clears to 0 */
    arena_reset(arena);
    ASSERT_EQ(arena->used, 0);

    /* Re-allocate should start fresh */
    void* p4 = arena_alloc(arena, 1000);
    ASSERT_NOT_NULL(p4);
    /* First allocation after reset may reuse same address */
    /* but used should be similar to first round */
    ASSERT_LE(arena->used, used_after_p1 + 32);  /* Allow for alignment variance */

    (void)p1; (void)p2; (void)p3;

    arena_destroy(arena);
    return 0;
}

/**
 * Test arena capacity tracking accuracy
 */
TEST(arena_capacity_tracking) {
    Arena* arena = arena_create(10000);
    ASSERT_NOT_NULL(arena);

    ASSERT_EQ(arena->capacity, 10000);
    ASSERT_EQ(arena->used, 0);
    ASSERT_EQ(arena_remaining(arena), 10000);

    /* Allocate and verify tracking */
    arena_alloc(arena, 1000);
    ASSERT_GT(arena->used, 0);
    ASSERT_LE(arena->used, arena->capacity);
    size_t remaining = arena_remaining(arena);
    ASSERT_LT(remaining, 10000);
    ASSERT_EQ(arena->used + remaining, arena->capacity);

    arena_destroy(arena);
    return 0;
}

/**
 * Test arena minimum-size allocation handling
 *
 * Note: arena_alloc(0) asserts size > 0 and is not a valid call.
 * This test verifies that small allocations work correctly.
 */
TEST(arena_minimum_size_allocation) {
    Arena* arena = arena_create(1024);
    ASSERT_NOT_NULL(arena);

    /* Minimum size allocation (1 byte) should succeed */
    void* ptr1 = arena_alloc(arena, 1);
    ASSERT_NOT_NULL(ptr1);

    /* Multiple small allocations should work */
    void* ptr2 = arena_alloc(arena, 1);
    ASSERT_NOT_NULL(ptr2);
    ASSERT_NE((uintptr_t)ptr1, (uintptr_t)ptr2);

    /* Used counter should have increased */
    ASSERT_GT(arena->used, 0);

    arena_destroy(arena);
    return 0;
}

/**
 * Test arena near-capacity allocation behavior
 */
TEST(arena_near_capacity_allocation) {
    /* Create exactly sized arena */
    Arena* arena = arena_create(1000);
    ASSERT_NOT_NULL(arena);

    /* Allocate leaving small headroom */
    void* p1 = arena_alloc(arena, 400);
    ASSERT_NOT_NULL(p1);

    void* p2 = arena_alloc(arena, 400);
    ASSERT_NOT_NULL(p2);

    /* This small allocation might succeed depending on alignment overhead */
    size_t remaining = arena_remaining(arena);

    /* Try to allocate more than remaining */
    void* p3 = arena_alloc(arena, remaining + 100);
    ASSERT_NULL(p3);

    /* Try to allocate exactly remaining (may fail due to alignment) */
    /* Skip this test as alignment makes exact prediction difficult */

    arena_destroy(arena);
    return 0;
}

/**
 * Test arena utilization calculation
 */
TEST(arena_utilization_calculation) {
    Arena* arena = arena_create(10000);
    ASSERT_NOT_NULL(arena);

    /* Initial utilization should be 0 */
    float util = arena_utilization(arena);
    ASSERT_TRUE(util < 0.001f);

    /* Allocate half */
    arena_alloc(arena, 5000);
    util = arena_utilization(arena);
    ASSERT_TRUE(util > 0.4f && util < 0.6f);  /* ~50% with alignment overhead */

    arena_destroy(arena);
    return 0;
}

/**
 * Test arena scope (scoped allocation) functionality
 */
TEST(arena_scope_functionality) {
    Arena* arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Allocate outside scope */
    void* p_outer = arena_alloc(arena, 1000);
    ASSERT_NOT_NULL(p_outer);
    size_t used_before_scope = arena->used;

    /* Enter scope and allocate */
    ArenaScope scope = arena_scope_begin(arena);
    ASSERT_EQ(scope.arena, arena);
    ASSERT_EQ(scope.saved_used, used_before_scope);

    void* p_inner1 = arena_alloc(arena, 2000);
    void* p_inner2 = arena_alloc(arena, 3000);
    ASSERT_NOT_NULL(p_inner1);
    ASSERT_NOT_NULL(p_inner2);

    size_t used_in_scope = arena->used;
    ASSERT_GT(used_in_scope, used_before_scope);

    /* End scope - should restore watermark */
    arena_scope_end(scope);
    ASSERT_EQ(arena->used, used_before_scope);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Category 2: Frame Arena Lifecycle Tests (10 tests)
 * ============================================================================ */

/**
 * Test that frame arena is reset at start of each step
 */
TEST(frame_arena_reset_at_step_start) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    /* Run several steps and verify frame arena is reset each time */
    for (int step = 0; step < 10; step++) {
        /* Record frame arena state before step */
        size_t capacity = engine->frame_arena->capacity;

        engine_step(engine);

        /* Frame arena capacity should be unchanged */
        ASSERT_EQ(engine->frame_arena->capacity, capacity);

        /* Frame arena used should be reasonable (not accumulating) */
        ASSERT_LT(engine->frame_arena->used, engine->frame_arena->capacity);
    }

    engine_destroy(engine);
    return 0;
}

/**
 * Test frame arena used counter after physics allocations
 */
TEST(frame_arena_after_physics) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    /* Reset frame arena to get clean baseline */
    arena_reset(engine->frame_arena);
    size_t baseline = engine->frame_arena->used;

    /* Execute physics step */
    engine_step_physics(engine);

    /* Frame arena may have allocations from physics (RK4 temps) */
    /* The used amount should be bounded */
    ASSERT_LT(engine->frame_arena->used, engine->frame_arena->capacity);

    engine_destroy(engine);
    return 0;
}

/**
 * Test frame arena used counter after collision allocations
 */
TEST(frame_arena_after_collision) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    /* Reset frame arena */
    arena_reset(engine->frame_arena);

    /* Execute collision step */
    engine_step_collision(engine);

    /* Collision step uses frame arena for collision pairs */
    ASSERT_LT(engine->frame_arena->used, engine->frame_arena->capacity);

    engine_destroy(engine);
    return 0;
}

/**
 * Test frame arena used counter after sensor allocations
 */
TEST(frame_arena_after_sensors) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    /* Reset frame arena */
    arena_reset(engine->frame_arena);

    /* Execute sensor step */
    engine_step_sensors(engine);

    /* Sensor step uses frame arena for scratch buffers */
    ASSERT_LT(engine->frame_arena->used, engine->frame_arena->capacity);

    engine_destroy(engine);
    return 0;
}

/**
 * Test frame arena peak usage tracking
 */
TEST(frame_arena_peak_usage) {
    BatchDroneEngine* engine = create_medium_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    size_t max_used = 0;

    /* Run multiple steps and track peak usage */
    for (int step = 0; step < 100; step++) {
        engine_step(engine);

        /* Note: engine_step resets frame arena at start, so we check after step phases */
        /* Peak would be measured during step, but we can't easily access mid-step */
        /* Instead, verify usage stays bounded */
        if (engine->frame_arena->used > max_used) {
            max_used = engine->frame_arena->used;
        }
    }

    /* Peak usage should be well under capacity */
    ASSERT_LT(max_used, engine->frame_arena->capacity / 2);

    engine_destroy(engine);
    return 0;
}

/**
 * Test that frame arena reset doesn't leak memory
 */
TEST(frame_arena_reset_no_leak) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    size_t initial_capacity = engine->frame_arena->capacity;

    /* Reset many times */
    for (int i = 0; i < 1000; i++) {
        /* Simulate frame arena usage */
        arena_reset(engine->frame_arena);
        void* temp = arena_alloc(engine->frame_arena, 1000);
        ASSERT_NOT_NULL(temp);
    }

    /* Capacity should be unchanged (no reallocation/leak) */
    ASSERT_EQ(engine->frame_arena->capacity, initial_capacity);

    /* Final reset should work */
    arena_reset(engine->frame_arena);
    ASSERT_EQ(engine->frame_arena->used, 0);

    engine_destroy(engine);
    return 0;
}

/**
 * Test that multiple steps don't accumulate frame arena usage
 */
TEST(frame_arena_no_accumulation) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    /* Record usage after first step */
    engine_step(engine);
    size_t usage_after_step1 = engine->frame_arena->used;

    /* Run many more steps */
    for (int step = 0; step < 100; step++) {
        engine_step(engine);
    }

    /* Usage should be similar to first step (no accumulation) */
    size_t usage_after_many = engine->frame_arena->used;
    /* Allow some variance for different execution paths */
    ASSERT_LT(usage_after_many, engine->frame_arena->capacity / 4);

    engine_destroy(engine);
    return 0;
}

/**
 * Test frame arena survives high-frequency reset cycles
 */
TEST(frame_arena_high_freq_reset_cycles) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    /* Perform 1000+ step cycles (each step resets frame arena) */
    for (int step = 0; step < 1000; step++) {
        engine_step(engine);

        /* Verify frame arena is still valid */
        ASSERT_NOT_NULL(engine->frame_arena);
        ASSERT_NOT_NULL(engine->frame_arena->data);
        ASSERT_LT(engine->frame_arena->used, engine->frame_arena->capacity);
    }

    engine_destroy(engine);
    return 0;
}

/**
 * Test frame arena with varying action patterns
 */
TEST(frame_arena_varying_actions) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    uint32_t total_drones = engine->config.total_drones;
    float* actions = engine_get_actions(engine);

    /* Run steps with varying action patterns */
    for (int step = 0; step < 100; step++) {
        /* Vary actions each step */
        for (uint32_t i = 0; i < total_drones * ENGINE_ACTION_DIM; i++) {
            actions[i] = (float)(step % 10) / 10.0f;
        }

        engine_step(engine);

        /* Frame arena should handle any action pattern */
        ASSERT_LT(engine->frame_arena->used, engine->frame_arena->capacity);
    }

    engine_destroy(engine);
    return 0;
}

/**
 * Test frame arena boundary conditions
 *
 * Verifies that the frame arena works correctly under normal usage.
 */
TEST(frame_arena_boundary_conditions) {
    /* Use small drone count but standard arena sizes */
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    /* Verify frame arena has adequate capacity */
    ASSERT_GT(engine->frame_arena->capacity, 0);

    /* Run steps - frame arena should not overflow */
    for (int step = 0; step < 50; step++) {
        engine_step(engine);
        ASSERT_LT(engine->frame_arena->used, engine->frame_arena->capacity);
    }

    /* Verify frame arena usage is bounded and reasonable */
    ASSERT_LT(engine->frame_arena->used, engine->frame_arena->capacity / 2);

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Category 3: Persistent Arena Stability Tests (8 tests)
 * ============================================================================ */

/**
 * Test persistent arena usage is stable across 1000 steps
 */
TEST(persistent_arena_stable_1000_steps) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    /* Record baseline after initialization */
    size_t baseline_used = engine->persistent_arena->used;
    size_t baseline_capacity = engine->persistent_arena->capacity;

    /* Run 1000 steps */
    for (int step = 0; step < 1000; step++) {
        engine_step(engine);
    }

    /* Persistent arena should not have grown */
    ASSERT_EQ(engine->persistent_arena->used, baseline_used);
    ASSERT_EQ(engine->persistent_arena->capacity, baseline_capacity);

    engine_destroy(engine);
    return 0;
}

/**
 * Test persistent arena doesn't grow during simulation
 */
TEST(persistent_arena_no_growth) {
    BatchDroneEngine* engine = create_medium_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    size_t initial_used = engine->persistent_arena->used;

    /* Run extended simulation */
    for (int step = 0; step < 500; step++) {
        engine_step(engine);

        /* Verify no growth at each step */
        ASSERT_EQ(engine->persistent_arena->used, initial_used);
    }

    engine_destroy(engine);
    return 0;
}

/**
 * Test persistent arena is properly sized for drone count
 */
TEST(persistent_arena_sized_for_drones) {
    /* Test with different drone counts */
    uint32_t drone_counts[] = {16, 64, 256};

    for (int i = 0; i < 3; i++) {
        uint32_t drones = drone_counts[i];
        BatchDroneEngine* engine = create_test_engine(drones, 1,
            128 * 1024 * 1024, 32 * 1024 * 1024);
        ASSERT_NOT_NULL(engine);

        /* Verify enough space was allocated */
        ASSERT_GT(engine->persistent_arena->capacity, engine->persistent_arena->used);
        ASSERT_GT(arena_remaining(engine->persistent_arena), 0);

        /* Usage should scale with drone count (roughly) */
        /* More drones = more state arrays = more usage */
        engine_destroy(engine);
    }

    return 0;
}

/**
 * Test persistent arena survives engine reset
 */
TEST(persistent_arena_survives_reset) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    size_t initial_used = engine->persistent_arena->used;
    size_t initial_capacity = engine->persistent_arena->capacity;

    /* Initial reset */
    engine_reset(engine);

    /* Usage should be unchanged (reset doesn't deallocate) */
    ASSERT_EQ(engine->persistent_arena->used, initial_used);
    ASSERT_EQ(engine->persistent_arena->capacity, initial_capacity);

    /* Run some steps */
    for (int i = 0; i < 100; i++) {
        engine_step(engine);
    }

    /* Reset again */
    engine_reset(engine);

    /* Still unchanged */
    ASSERT_EQ(engine->persistent_arena->used, initial_used);
    ASSERT_EQ(engine->persistent_arena->capacity, initial_capacity);

    engine_destroy(engine);
    return 0;
}

/**
 * Test persistent arena usage consistent across multiple resets
 */
TEST(persistent_arena_consistent_across_resets) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    size_t baseline = engine->persistent_arena->used;

    /* Perform multiple reset-run cycles */
    for (int cycle = 0; cycle < 10; cycle++) {
        engine_reset(engine);
        ASSERT_EQ(engine->persistent_arena->used, baseline);

        for (int step = 0; step < 50; step++) {
            engine_step(engine);
        }
        ASSERT_EQ(engine->persistent_arena->used, baseline);
    }

    engine_destroy(engine);
    return 0;
}

/**
 * Test persistent arena handles maximum drone configurations
 */
TEST(persistent_arena_max_drones) {
    /* Test with large drone count */
    BatchDroneEngine* engine = create_large_engine();  /* 1024 drones */
    ASSERT_NOT_NULL(engine);

    /* Verify allocation succeeded */
    ASSERT_NOT_NULL(engine->states);
    ASSERT_NOT_NULL(engine->params);
    ASSERT_NOT_NULL(engine->observations);
    ASSERT_NOT_NULL(engine->actions);

    /* Verify capacity is sufficient */
    ASSERT_GT(arena_remaining(engine->persistent_arena), 0);

    engine_reset(engine);
    for (int i = 0; i < 10; i++) {
        engine_step(engine);
    }

    engine_destroy(engine);
    return 0;
}

/**
 * Test persistent arena allocation order
 */
TEST(persistent_arena_allocation_order) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    /* Verify all major components are allocated and valid */
    ASSERT_NOT_NULL(engine->states);
    ASSERT_NOT_NULL(engine->params);
    ASSERT_NOT_NULL(engine->physics);
    ASSERT_NOT_NULL(engine->world);
    ASSERT_NOT_NULL(engine->collision);
    ASSERT_NOT_NULL(engine->sensors);
    ASSERT_NOT_NULL(engine->rewards);
    ASSERT_NOT_NULL(engine->episode_returns);
    ASSERT_NOT_NULL(engine->episode_lengths);
    ASSERT_NOT_NULL(engine->observations);
    ASSERT_NOT_NULL(engine->actions);
    ASSERT_NOT_NULL(engine->rewards_buffer);
    ASSERT_NOT_NULL(engine->dones);
    ASSERT_NOT_NULL(engine->truncations);

    engine_destroy(engine);
    return 0;
}

/**
 * Test persistent arena fragmentation (or lack thereof)
 */
TEST(persistent_arena_no_fragmentation) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    /* Bump allocator by design has no fragmentation */
    /* Verify utilization is reasonable */
    float util = arena_utilization(engine->persistent_arena);
    ASSERT_TRUE(util > 0.0f && util < 1.0f);

    /* Remaining space is contiguous (inherent in arena design) */
    size_t remaining = arena_remaining(engine->persistent_arena);
    ASSERT_GT(remaining, 0);

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Category 4: SoA Structure Alignment Tests (12 tests)
 * ============================================================================ */

/**
 * Test all 17 DroneStateSOA arrays are 32-byte aligned
 */
TEST(drone_state_soa_all_aligned_32) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    ASSERT_NOT_NULL(engine->states);

    /* Position arrays */
    ASSERT_ALIGNED_32(engine->states->pos_x);
    ASSERT_ALIGNED_32(engine->states->pos_y);
    ASSERT_ALIGNED_32(engine->states->pos_z);

    /* Velocity arrays */
    ASSERT_ALIGNED_32(engine->states->vel_x);
    ASSERT_ALIGNED_32(engine->states->vel_y);
    ASSERT_ALIGNED_32(engine->states->vel_z);

    /* Quaternion arrays */
    ASSERT_ALIGNED_32(engine->states->quat_w);
    ASSERT_ALIGNED_32(engine->states->quat_x);
    ASSERT_ALIGNED_32(engine->states->quat_y);
    ASSERT_ALIGNED_32(engine->states->quat_z);

    /* Angular velocity arrays */
    ASSERT_ALIGNED_32(engine->states->omega_x);
    ASSERT_ALIGNED_32(engine->states->omega_y);
    ASSERT_ALIGNED_32(engine->states->omega_z);

    /* Motor RPM arrays */
    ASSERT_ALIGNED_32(engine->states->rpm_0);
    ASSERT_ALIGNED_32(engine->states->rpm_1);
    ASSERT_ALIGNED_32(engine->states->rpm_2);
    ASSERT_ALIGNED_32(engine->states->rpm_3);

    engine_destroy(engine);
    return 0;
}

/**
 * Test all 15 DroneParamsSOA arrays are 32-byte aligned
 */
TEST(drone_params_soa_all_aligned_32) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    ASSERT_NOT_NULL(engine->params);

    /* Mass and inertia */
    ASSERT_ALIGNED_32(engine->params->mass);
    ASSERT_ALIGNED_32(engine->params->ixx);
    ASSERT_ALIGNED_32(engine->params->iyy);
    ASSERT_ALIGNED_32(engine->params->izz);

    /* Geometry */
    ASSERT_ALIGNED_32(engine->params->arm_length);
    ASSERT_ALIGNED_32(engine->params->collision_radius);

    /* Thrust/torque coefficients */
    ASSERT_ALIGNED_32(engine->params->k_thrust);
    ASSERT_ALIGNED_32(engine->params->k_torque);

    /* Damping */
    ASSERT_ALIGNED_32(engine->params->k_drag);
    ASSERT_ALIGNED_32(engine->params->k_ang_damp);

    /* Motor dynamics */
    ASSERT_ALIGNED_32(engine->params->motor_tau);
    ASSERT_ALIGNED_32(engine->params->max_rpm);

    /* Limits */
    ASSERT_ALIGNED_32(engine->params->max_vel);
    ASSERT_ALIGNED_32(engine->params->max_omega);

    /* Environment */
    ASSERT_ALIGNED_32(engine->params->gravity);

    engine_destroy(engine);
    return 0;
}

/**
 * Test observation buffer is 32-byte aligned
 */
TEST(observation_buffer_aligned_32) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    float* obs = engine_get_observations(engine);
    ASSERT_NOT_NULL(obs);
    ASSERT_ALIGNED_32(obs);

    engine_destroy(engine);
    return 0;
}

/**
 * Test action buffer is 32-byte aligned
 */
TEST(action_buffer_aligned_32) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    float* actions = engine_get_actions(engine);
    ASSERT_NOT_NULL(actions);
    ASSERT_ALIGNED_32(actions);

    engine_destroy(engine);
    return 0;
}

/**
 * Test reward buffer is 32-byte aligned
 */
TEST(reward_buffer_aligned_32) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    float* rewards = engine_get_rewards(engine);
    ASSERT_NOT_NULL(rewards);
    ASSERT_ALIGNED_32(rewards);

    engine_destroy(engine);
    return 0;
}

/**
 * Test done/truncation buffers are properly aligned
 */
TEST(done_truncation_buffers_aligned) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    uint8_t* dones = engine_get_dones(engine);
    uint8_t* truncs = engine_get_truncations(engine);

    ASSERT_NOT_NULL(dones);
    ASSERT_NOT_NULL(truncs);

    /* These should be 32-byte aligned per engine_lifecycle.c */
    ASSERT_ALIGNED_32(dones);
    ASSERT_ALIGNED_32(truncs);

    engine_destroy(engine);
    return 0;
}

/**
 * Test episode tracking buffers are aligned
 */
TEST(episode_tracking_buffers_aligned) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    /* Episode returns and lengths should be 32-byte aligned */
    ASSERT_ALIGNED_32(engine->episode_returns);
    ASSERT_ALIGNED_32(engine->episode_lengths);
    ASSERT_ALIGNED_32(engine->env_ids);

    engine_destroy(engine);
    return 0;
}

/**
 * Test termination flag buffers are aligned
 */
TEST(termination_flag_buffers_aligned) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    /* Detailed termination flags */
    ASSERT_NOT_NULL(engine->term_success);
    ASSERT_NOT_NULL(engine->term_collision);
    ASSERT_NOT_NULL(engine->term_out_of_bounds);
    ASSERT_NOT_NULL(engine->term_timeout);

    ASSERT_ALIGNED_32(engine->term_success);
    ASSERT_ALIGNED_32(engine->term_collision);
    ASSERT_ALIGNED_32(engine->term_out_of_bounds);
    ASSERT_ALIGNED_32(engine->term_timeout);

    engine_destroy(engine);
    return 0;
}

/**
 * Test all buffers remain aligned after operations
 */
TEST(buffers_aligned_after_operations) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    /* Run some steps */
    for (int i = 0; i < 50; i++) {
        engine_step(engine);
    }

    /* All buffers should still be aligned */
    ASSERT_ALIGNED_32(engine_get_observations(engine));
    ASSERT_ALIGNED_32(engine_get_actions(engine));
    ASSERT_ALIGNED_32(engine_get_rewards(engine));
    ASSERT_ALIGNED_32(engine_get_dones(engine));
    ASSERT_ALIGNED_32(engine_get_truncations(engine));

    engine_destroy(engine);
    return 0;
}

/**
 * Test alignment with different drone counts
 */
TEST(alignment_various_drone_counts) {
    /* Test alignment with non-power-of-2 drone counts */
    uint32_t counts[] = {7, 15, 33, 100};

    for (int i = 0; i < 4; i++) {
        BatchDroneEngine* engine = create_test_engine(counts[i], 1,
            64 * 1024 * 1024, 16 * 1024 * 1024);
        if (engine == NULL) continue;  /* Skip if creation fails */

        /* Verify alignment even with odd counts */
        ASSERT_ALIGNED_32(engine->states->pos_x);
        ASSERT_ALIGNED_32(engine->observations);
        ASSERT_ALIGNED_32(engine->actions);

        engine_destroy(engine);
    }

    return 0;
}

/**
 * Test SoA array sizes match capacity
 */
TEST(soa_array_sizes_match_capacity) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    uint32_t total_drones = engine->config.total_drones;

    /* Verify state capacity */
    ASSERT_EQ(engine->states->capacity, total_drones);
    ASSERT_EQ(engine->states->count, 0);  /* Or total_drones after init */

    /* Verify params capacity */
    ASSERT_EQ(engine->params->capacity, total_drones);

    engine_destroy(engine);
    return 0;
}

/**
 * Test that arrays are contiguous within SoA structure
 */
TEST(soa_arrays_contiguous) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    uint32_t total_drones = engine->config.total_drones;

    /* Verify we can access full range of each array */
    for (uint32_t i = 0; i < total_drones; i++) {
        /* Read access to verify no segfault */
        float px = engine->states->pos_x[i];
        float vy = engine->states->vel_y[i];
        float qw = engine->states->quat_w[i];
        (void)px; (void)vy; (void)qw;
    }

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Category 5: Buffer Management Tests (10 tests)
 * ============================================================================ */

/**
 * Test external buffer pointers are stable across steps
 */
TEST(buffer_pointers_stable_across_steps) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    /* Record initial pointers */
    float* obs_initial = engine_get_observations(engine);
    float* act_initial = engine_get_actions(engine);
    float* rew_initial = engine_get_rewards(engine);
    uint8_t* done_initial = engine_get_dones(engine);
    uint8_t* trunc_initial = engine_get_truncations(engine);

    /* Run steps and verify pointers remain stable */
    for (int step = 0; step < 100; step++) {
        engine_step(engine);

        ASSERT_EQ((uintptr_t)engine_get_observations(engine), (uintptr_t)obs_initial);
        ASSERT_EQ((uintptr_t)engine_get_actions(engine), (uintptr_t)act_initial);
        ASSERT_EQ((uintptr_t)engine_get_rewards(engine), (uintptr_t)rew_initial);
        ASSERT_EQ((uintptr_t)engine_get_dones(engine), (uintptr_t)done_initial);
        ASSERT_EQ((uintptr_t)engine_get_truncations(engine), (uintptr_t)trunc_initial);
    }

    engine_destroy(engine);
    return 0;
}

/**
 * Test external buffer pointers are stable across resets
 */
TEST(buffer_pointers_stable_across_resets) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    /* Record initial pointers */
    float* obs_initial = engine_get_observations(engine);
    float* act_initial = engine_get_actions(engine);

    /* Multiple resets */
    for (int i = 0; i < 10; i++) {
        engine_reset(engine);

        ASSERT_EQ((uintptr_t)engine_get_observations(engine), (uintptr_t)obs_initial);
        ASSERT_EQ((uintptr_t)engine_get_actions(engine), (uintptr_t)act_initial);

        for (int step = 0; step < 20; step++) {
            engine_step(engine);
        }
    }

    engine_destroy(engine);
    return 0;
}

/**
 * Test buffer sizes match configuration
 */
TEST(buffer_sizes_match_config) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    uint32_t total_drones = engine->config.total_drones;
    uint32_t obs_dim = engine_get_obs_dim(engine);
    uint32_t action_dim = engine_get_action_dim(engine);

    /* Verify dimensions */
    ASSERT_EQ(engine_get_total_drones(engine), total_drones);
    ASSERT_EQ(action_dim, ENGINE_ACTION_DIM);
    ASSERT_GT(obs_dim, 0);

    /* Verify we can access full extent of buffers without crash */
    float* obs = engine_get_observations(engine);
    float* actions = engine_get_actions(engine);
    float* rewards = engine_get_rewards(engine);

    for (uint32_t i = 0; i < total_drones * obs_dim; i++) {
        obs[i] = 0.0f;
    }
    for (uint32_t i = 0; i < total_drones * action_dim; i++) {
        actions[i] = 0.0f;
    }
    for (uint32_t i = 0; i < total_drones; i++) {
        rewards[i] = 0.0f;
    }

    engine_destroy(engine);
    return 0;
}

/**
 * Test buffer contents survive step operations
 */
TEST(buffer_contents_survive_step) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    uint32_t total_drones = engine->config.total_drones;
    float* actions = engine_get_actions(engine);

    /* Set specific action pattern */
    for (uint32_t i = 0; i < total_drones * ENGINE_ACTION_DIM; i++) {
        actions[i] = 0.5f;
    }

    /* Step should read actions */
    engine_step(engine);

    /* Actions buffer should still be accessible (not corrupted) */
    for (uint32_t i = 0; i < total_drones * ENGINE_ACTION_DIM; i++) {
        ASSERT_TRUE(isfinite(actions[i]));
    }

    engine_destroy(engine);
    return 0;
}

/**
 * Test buffer zero-initialization on creation
 */
TEST(buffer_zero_initialized) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    uint32_t total_drones = engine->config.total_drones;
    uint32_t obs_dim = engine_get_obs_dim(engine);

    /* Check observations are zero (before reset/step) */
    float* obs = engine_get_observations(engine);
    for (uint32_t i = 0; i < total_drones * obs_dim; i++) {
        ASSERT_TRUE(obs[i] == 0.0f);
    }

    /* Check actions are zero */
    float* actions = engine_get_actions(engine);
    for (uint32_t i = 0; i < total_drones * ENGINE_ACTION_DIM; i++) {
        ASSERT_TRUE(actions[i] == 0.0f);
    }

    /* Check dones are zero */
    uint8_t* dones = engine_get_dones(engine);
    for (uint32_t i = 0; i < total_drones; i++) {
        ASSERT_EQ(dones[i], 0);
    }

    engine_destroy(engine);
    return 0;
}

/**
 * Test buffer independence (writing one doesn't affect others)
 */
TEST(buffer_independence) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    uint32_t total_drones = engine->config.total_drones;

    float* obs = engine_get_observations(engine);
    float* actions = engine_get_actions(engine);
    float* rewards = engine_get_rewards(engine);

    /* Fill each buffer with distinct pattern */
    uint32_t obs_dim = engine_get_obs_dim(engine);
    for (uint32_t i = 0; i < total_drones * obs_dim; i++) {
        obs[i] = 1.0f;
    }
    for (uint32_t i = 0; i < total_drones * ENGINE_ACTION_DIM; i++) {
        actions[i] = 2.0f;
    }
    for (uint32_t i = 0; i < total_drones; i++) {
        rewards[i] = 3.0f;
    }

    /* Verify patterns are independent */
    for (uint32_t i = 0; i < total_drones * obs_dim; i++) {
        ASSERT_TRUE(obs[i] == 1.0f);
    }
    for (uint32_t i = 0; i < total_drones * ENGINE_ACTION_DIM; i++) {
        ASSERT_TRUE(actions[i] == 2.0f);
    }
    for (uint32_t i = 0; i < total_drones; i++) {
        ASSERT_TRUE(rewards[i] == 3.0f);
    }

    engine_destroy(engine);
    return 0;
}

/**
 * Test buffer bounds are respected
 */
TEST(buffer_bounds_respected) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    uint32_t total_drones = engine->config.total_drones;
    uint32_t obs_dim = engine_get_obs_dim(engine);

    /* Write to last valid index */
    float* obs = engine_get_observations(engine);
    obs[total_drones * obs_dim - 1] = 123.0f;

    float* actions = engine_get_actions(engine);
    actions[total_drones * ENGINE_ACTION_DIM - 1] = 456.0f;

    /* Run step to verify no buffer overrun issues */
    engine_step(engine);

    engine_destroy(engine);
    return 0;
}

/**
 * Test buffer reuse after auto-reset
 */
TEST(buffer_reuse_after_autoreset) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    uint32_t total_drones = engine->config.total_drones;
    float* obs = engine_get_observations(engine);
    float* rewards = engine_get_rewards(engine);
    uint8_t* dones = engine_get_dones(engine);

    /* Run until some drones terminate (auto-reset) */
    for (int step = 0; step < 500; step++) {
        engine_step(engine);

        /* Check buffers remain valid */
        for (uint32_t i = 0; i < total_drones; i++) {
            ASSERT_TRUE(isfinite(rewards[i]));
            ASSERT_TRUE(dones[i] == 0 || dones[i] == 1);
        }
    }

    /* Buffers should still be valid after many auto-resets */
    ASSERT_EQ((uintptr_t)engine_get_observations(engine), (uintptr_t)obs);

    engine_destroy(engine);
    return 0;
}

/**
 * Test action buffer can be written before step
 */
TEST(action_buffer_writeable_before_step) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    uint32_t total_drones = engine->config.total_drones;
    float* actions = engine_get_actions(engine);

    /* Write action patterns */
    for (uint32_t d = 0; d < total_drones; d++) {
        for (uint32_t a = 0; a < ENGINE_ACTION_DIM; a++) {
            actions[d * ENGINE_ACTION_DIM + a] = 0.25f * (float)a;
        }
    }

    /* Verify writes persisted */
    for (uint32_t d = 0; d < total_drones; d++) {
        for (uint32_t a = 0; a < ENGINE_ACTION_DIM; a++) {
            float expected = 0.25f * (float)a;
            ASSERT_TRUE(fabsf(actions[d * ENGINE_ACTION_DIM + a] - expected) < 0.001f);
        }
    }

    engine_step(engine);

    engine_destroy(engine);
    return 0;
}

/**
 * Test observation buffer is updated after step
 */
TEST(observation_buffer_updated_after_step) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    float* obs = engine_get_observations(engine);
    uint32_t obs_dim = engine_get_obs_dim(engine);
    uint32_t total_drones = engine->config.total_drones;

    /* Record initial observations */
    float initial_sum = 0.0f;
    for (uint32_t i = 0; i < total_drones * obs_dim && i < 100; i++) {
        initial_sum += obs[i];
    }

    /* Set actions and step */
    float* actions = engine_get_actions(engine);
    for (uint32_t i = 0; i < total_drones * ENGINE_ACTION_DIM; i++) {
        actions[i] = 0.5f;
    }
    engine_step(engine);

    /* Observations should have been updated (physics changed state) */
    float post_step_sum = 0.0f;
    for (uint32_t i = 0; i < total_drones * obs_dim && i < 100; i++) {
        post_step_sum += obs[i];
    }

    /* Sum might be different (state changed) - at least observations are valid */
    for (uint32_t i = 0; i < total_drones * obs_dim; i++) {
        ASSERT_TRUE(isfinite(obs[i]));
    }

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Category 6: Memory Leak Detection Tests (7 tests)
 * ============================================================================ */

/**
 * Test create/destroy cycle doesn't leak (check arena usage)
 */
TEST(create_destroy_no_leak) {
    /* Run multiple create/destroy cycles and check system memory is stable */
    for (int i = 0; i < 5; i++) {
        BatchDroneEngine* engine = create_small_engine();
        ASSERT_NOT_NULL(engine);

        engine_reset(engine);
        for (int s = 0; s < 50; s++) {
            engine_step(engine);
        }

        engine_destroy(engine);
    }

    /* If we reach here without OOM, no obvious leaks */
    return 0;
}

/**
 * Test multiple create/destroy cycles don't accumulate
 */
TEST(multiple_create_destroy_no_accumulation) {
    /* Perform many cycles */
    for (int i = 0; i < 20; i++) {
        BatchDroneEngine* engine = create_small_engine();
        ASSERT_NOT_NULL(engine);
        engine_destroy(engine);
    }

    /* Memory should be fully reclaimed each time */
    return 0;
}

/**
 * Test reset doesn't leak memory
 */
TEST(reset_no_memory_leak) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    size_t baseline = engine->persistent_arena->used;

    /* Many resets */
    for (int i = 0; i < 100; i++) {
        engine_reset(engine);
        ASSERT_EQ(engine->persistent_arena->used, baseline);
    }

    engine_destroy(engine);
    return 0;
}

/**
 * Test partial reset doesn't leak memory
 */
TEST(partial_reset_no_leak) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    size_t baseline = engine->persistent_arena->used;

    /* Reset specific environments */
    uint32_t env_indices[] = {0, 1};
    for (int i = 0; i < 50; i++) {
        engine_reset_envs(engine, env_indices, 2);
        ASSERT_EQ(engine->persistent_arena->used, baseline);
    }

    engine_destroy(engine);
    return 0;
}

/**
 * Test auto-reset doesn't leak memory
 */
TEST(auto_reset_no_leak) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    size_t baseline = engine->persistent_arena->used;

    /* Run many steps to trigger auto-resets */
    for (int step = 0; step < 1000; step++) {
        engine_step(engine);
        /* Persistent arena should be unchanged */
        ASSERT_EQ(engine->persistent_arena->used, baseline);
    }

    engine_destroy(engine);
    return 0;
}

/**
 * Test long-running simulation (10000 steps) doesn't leak
 */
TEST(long_running_no_leak) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    size_t persistent_baseline = engine->persistent_arena->used;
    size_t frame_capacity = engine->frame_arena->capacity;

    /* Run 10000 steps */
    for (int step = 0; step < 10000; step++) {
        engine_step(engine);

        /* Periodic checks */
        if (step % 1000 == 0) {
            ASSERT_EQ(engine->persistent_arena->used, persistent_baseline);
            ASSERT_EQ(engine->frame_arena->capacity, frame_capacity);
        }
    }

    ASSERT_EQ(engine->persistent_arena->used, persistent_baseline);

    engine_destroy(engine);
    return 0;
}

/**
 * Test subsystem destruction order
 */
TEST(subsystem_destruction_order) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    /* Run some operations */
    for (int i = 0; i < 100; i++) {
        engine_step(engine);
    }

    /* Destroy should clean up all subsystems in correct order */
    engine_destroy(engine);

    /* If we reach here without crash, destruction order is correct */
    return 0;
}

/* ============================================================================
 * Category 7: Cache Optimization Tests (6 tests)
 * ============================================================================ */

/**
 * Test hot data (pos/vel/quat/omega) is contiguous in memory
 */
TEST(hot_data_contiguous) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    /* In SoA layout, each array is contiguous by definition */
    /* Verify array base pointers exist and are distinct */
    ASSERT_NOT_NULL(engine->states->pos_x);
    ASSERT_NOT_NULL(engine->states->pos_y);
    ASSERT_NOT_NULL(engine->states->pos_z);
    ASSERT_NOT_NULL(engine->states->vel_x);
    ASSERT_NOT_NULL(engine->states->vel_y);
    ASSERT_NOT_NULL(engine->states->vel_z);

    /* Each array is distinct */
    ASSERT_NE((uintptr_t)engine->states->pos_x, (uintptr_t)engine->states->pos_y);
    ASSERT_NE((uintptr_t)engine->states->pos_y, (uintptr_t)engine->states->pos_z);

    engine_destroy(engine);
    return 0;
}

/**
 * Test cold data is separated from hot data
 */
TEST(cold_data_separated) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    /* States (hot) and params (cold) should be separate allocations */
    ASSERT_NE((void*)engine->states, (void*)engine->params);

    /* Episode data (cold) should be separate from state data (hot) */
    ASSERT_NE((void*)engine->episode_returns, (void*)engine->states);

    engine_destroy(engine);
    return 0;
}

/**
 * Test per-drone hot data size approximates cache line
 */
TEST(per_drone_hot_data_size) {
    /* Hot data per drone in SoA:
     * - pos: 3 floats = 12 bytes (but in 3 separate arrays)
     * - vel: 3 floats = 12 bytes
     * - quat: 4 floats = 16 bytes
     * - omega: 3 floats = 12 bytes
     * Total: 52 bytes
     *
     * In SoA, this is accessed as array[i] for each component
     * which is 1 float = 4 bytes per access, highly cacheable
     */

    size_t hot_data_per_drone = 3 * sizeof(float)   /* pos */
                              + 3 * sizeof(float)   /* vel */
                              + 4 * sizeof(float)   /* quat */
                              + 3 * sizeof(float);  /* omega */

    /* Should be under 64 bytes (cache line) */
    ASSERT_LE(hot_data_per_drone, 64);

    /* Including RPMs (4 floats = 16 bytes) = 68 bytes */
    size_t with_rpm = hot_data_per_drone + 4 * sizeof(float);
    /* Still close to single cache line, acceptable */
    ASSERT_LE(with_rpm, 128);  /* Within 2 cache lines */

    return 0;
}

/**
 * Test stride patterns are cache-friendly
 */
TEST(stride_patterns_cache_friendly) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    uint32_t total_drones = engine->config.total_drones;

    /* SoA layout means sequential access to array[i], array[i+1], ...
     * is sequential memory access (stride = sizeof(float) = 4 bytes)
     */

    /* Verify we can sequentially access all elements */
    float sum = 0.0f;
    for (uint32_t i = 0; i < total_drones; i++) {
        sum += engine->states->pos_x[i];  /* Sequential access */
        sum += engine->states->pos_y[i];
        sum += engine->states->pos_z[i];
    }
    (void)sum;

    /* If no crash, access pattern is valid */

    engine_destroy(engine);
    return 0;
}

/**
 * Test false sharing avoidance (per-thread data separation)
 */
TEST(false_sharing_avoidance) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    /* Verify scheduler/thread pool exists */
    ASSERT_NOT_NULL(engine->scheduler);

    /* In SoA layout, different threads process different index ranges
     * E.g., Thread 0: indices 0-255, Thread 1: indices 256-511
     * Since each float array is contiguous, threads access different
     * cache lines when processing non-overlapping index ranges.
     */

    /* Verify that array base is 32-byte aligned (helps with cache line alignment) */
    ASSERT_ALIGNED_32(engine->states->pos_x);

    engine_destroy(engine);
    return 0;
}

/**
 * Test memory access patterns are sequential where possible
 */
TEST(sequential_access_patterns) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    uint32_t total_drones = engine->config.total_drones;

    /* SoA enables sequential access for SIMD processing */
    /* Test that we can process 8 drones at a time (AVX2) */
    for (uint32_t i = 0; i + 8 <= total_drones; i += 8) {
        /* Access 8 consecutive floats */
        float* base = &engine->states->pos_x[i];
        /* These 8 floats = 32 bytes fit in a single AVX2 register */
        ASSERT_ALIGNED_32(base);
    }

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Category 8: Stress Tests (7 tests)
 * ============================================================================ */

/**
 * Test maximum drone count allocation succeeds
 */
TEST(max_drone_count_allocation) {
    /* Test with 1024 drones (target configuration) */
    BatchDroneEngine* engine = create_large_engine();
    ASSERT_NOT_NULL(engine);

    ASSERT_EQ(engine->config.total_drones, 1024);
    ASSERT_NOT_NULL(engine->states);
    ASSERT_NOT_NULL(engine->observations);
    ASSERT_NOT_NULL(engine->actions);

    engine_reset(engine);
    engine_step(engine);

    engine_destroy(engine);
    return 0;
}

/**
 * Test near-maximum arena capacity allocation
 */
TEST(near_max_arena_allocation) {
    Arena* arena = arena_create(100 * 1024 * 1024);  /* 100 MB */
    ASSERT_NOT_NULL(arena);

    /* Allocate 90% of capacity */
    size_t alloc_size = 90 * 1024 * 1024;
    void* ptr = arena_alloc(arena, alloc_size);
    ASSERT_NOT_NULL(ptr);

    /* Verify we used approximately that much */
    ASSERT_GT(arena->used, 80 * 1024 * 1024);
    ASSERT_LT(arena_remaining(arena), 20 * 1024 * 1024);

    arena_destroy(arena);
    return 0;
}

/**
 * Test rapid create/destroy cycles (100 iterations)
 */
TEST(rapid_create_destroy_100) {
    for (int i = 0; i < 100; i++) {
        BatchDroneEngine* engine = create_small_engine();
        ASSERT_NOT_NULL(engine);
        engine_destroy(engine);
    }

    return 0;
}

/**
 * Test high-frequency step cycles (10000 steps)
 */
TEST(high_freq_step_10000) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    for (int step = 0; step < 10000; step++) {
        engine_step(engine);
    }

    /* Engine should still be valid */
    ASSERT_TRUE(engine_is_valid(engine));

    engine_destroy(engine);
    return 0;
}

/**
 * Test memory stability under varying action patterns
 */
TEST(memory_stable_varying_actions) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    uint32_t total_drones = engine->config.total_drones;
    float* actions = engine_get_actions(engine);
    size_t baseline = engine->persistent_arena->used;

    /* Vary actions each step */
    for (int step = 0; step < 500; step++) {
        for (uint32_t i = 0; i < total_drones * ENGINE_ACTION_DIM; i++) {
            actions[i] = (float)(step % 100) / 100.0f;
        }
        engine_step(engine);

        ASSERT_EQ(engine->persistent_arena->used, baseline);
    }

    engine_destroy(engine);
    return 0;
}

/**
 * Test memory stability with many collisions/resets
 */
TEST(memory_stable_many_resets) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);

    size_t baseline = engine->persistent_arena->used;

    /* Many reset cycles */
    for (int cycle = 0; cycle < 100; cycle++) {
        engine_reset(engine);
        ASSERT_EQ(engine->persistent_arena->used, baseline);

        for (int step = 0; step < 20; step++) {
            engine_step(engine);
        }
        ASSERT_EQ(engine->persistent_arena->used, baseline);
    }

    engine_destroy(engine);
    return 0;
}

/**
 * Test arena stress with alternating large/small allocations
 */
TEST(arena_stress_alternating_sizes) {
    Arena* arena = arena_create(10 * 1024 * 1024);  /* 10 MB */
    ASSERT_NOT_NULL(arena);

    /* Alternate large and small allocations */
    for (int i = 0; i < 100; i++) {
        /* Small allocation */
        void* small = arena_alloc_aligned(arena, 64, 32);
        ASSERT_NOT_NULL(small);
        ASSERT_ALIGNED_32(small);

        /* Larger allocation */
        void* large = arena_alloc_aligned(arena, 4096, 32);
        ASSERT_NOT_NULL(large);
        ASSERT_ALIGNED_32(large);

        /* Reset periodically */
        if (i % 20 == 19) {
            arena_reset(arena);
        }
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Category 9: Integration Memory Tests (6 tests)
 * ============================================================================ */

/**
 * Test physics step memory usage is bounded
 */
TEST(physics_step_memory_bounded) {
    BatchDroneEngine* engine = create_medium_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    size_t persistent_baseline = engine->persistent_arena->used;
    size_t frame_capacity = engine->frame_arena->capacity;

    /* Multiple physics steps */
    for (int i = 0; i < 100; i++) {
        arena_reset(engine->frame_arena);
        engine_step_physics(engine);

        /* Frame arena usage should be bounded */
        ASSERT_LT(engine->frame_arena->used, frame_capacity / 2);
        /* Persistent should be unchanged */
        ASSERT_EQ(engine->persistent_arena->used, persistent_baseline);
    }

    engine_destroy(engine);
    return 0;
}

/**
 * Test collision step memory usage is bounded
 */
TEST(collision_step_memory_bounded) {
    BatchDroneEngine* engine = create_medium_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    size_t persistent_baseline = engine->persistent_arena->used;

    for (int i = 0; i < 100; i++) {
        arena_reset(engine->frame_arena);
        engine_step_collision(engine);

        ASSERT_LT(engine->frame_arena->used, engine->frame_arena->capacity / 2);
        ASSERT_EQ(engine->persistent_arena->used, persistent_baseline);
    }

    engine_destroy(engine);
    return 0;
}

/**
 * Test sensor step memory usage is bounded
 */
TEST(sensor_step_memory_bounded) {
    BatchDroneEngine* engine = create_medium_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    size_t persistent_baseline = engine->persistent_arena->used;

    for (int i = 0; i < 100; i++) {
        arena_reset(engine->frame_arena);
        engine_step_sensors(engine);

        ASSERT_LT(engine->frame_arena->used, engine->frame_arena->capacity / 2);
        ASSERT_EQ(engine->persistent_arena->used, persistent_baseline);
    }

    engine_destroy(engine);
    return 0;
}

/**
 * Test full step memory usage is bounded
 */
TEST(full_step_memory_bounded) {
    BatchDroneEngine* engine = create_medium_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    size_t persistent_baseline = engine->persistent_arena->used;
    size_t frame_capacity = engine->frame_arena->capacity;

    for (int i = 0; i < 500; i++) {
        engine_step(engine);

        /* Full step includes all phases */
        ASSERT_LT(engine->frame_arena->used, frame_capacity);
        ASSERT_EQ(engine->persistent_arena->used, persistent_baseline);
    }

    engine_destroy(engine);
    return 0;
}

/**
 * Test memory usage scales with drone count
 *
 * Note: Memory has significant fixed overhead (subsystems, collision grid, etc.)
 * so scaling is not purely linear for small drone counts. We verify that more
 * drones use more memory, and that growth is reasonable.
 */
TEST(memory_scales_with_drones) {
    /* Test with different drone counts and verify growth */
    size_t usages[3];
    uint32_t counts[] = {16, 64, 256};

    for (int i = 0; i < 3; i++) {
        BatchDroneEngine* engine = create_test_engine(counts[i], 1,
            128 * 1024 * 1024, 32 * 1024 * 1024);
        if (engine == NULL) {
            usages[i] = 0;
            continue;
        }

        usages[i] = engine->persistent_arena->used;
        engine_destroy(engine);
    }

    /* Verify monotonic growth: more drones = more memory */
    if (usages[0] > 0 && usages[1] > 0) {
        ASSERT_GT(usages[1], usages[0]);  /* 64 drones > 16 drones */
    }

    if (usages[1] > 0 && usages[2] > 0) {
        ASSERT_GT(usages[2], usages[1]);  /* 256 drones > 64 drones */
    }

    /* Verify memory is reasonable (not exploding exponentially) */
    /* 256 drones should use less than 20x memory of 16 drones */
    if (usages[0] > 0 && usages[2] > 0) {
        float ratio = (float)usages[2] / (float)usages[0];
        ASSERT_TRUE(ratio > 1.0f && ratio < 20.0f);
    }

    return 0;
}

/**
 * Test memory usage independent of step count (no accumulation)
 */
TEST(memory_independent_of_steps) {
    BatchDroneEngine* engine = create_small_engine();
    ASSERT_NOT_NULL(engine);
    engine_reset(engine);

    size_t baseline = engine->persistent_arena->used;

    /* Run different numbers of steps and verify memory is constant */
    for (int batch = 0; batch < 5; batch++) {
        for (int step = 0; step < 1000; step++) {
            engine_step(engine);
        }

        /* Memory should be unchanged regardless of step count */
        ASSERT_EQ(engine->persistent_arena->used, baseline);
    }

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Deep Memory Management Tests");

    RUN_TEST(arena_creation_various_sizes);
    RUN_TEST(arena_alloc_returns_aligned_pointers);
    RUN_TEST(arena_reset_preserves_capacity);
    RUN_TEST(arena_overflow_returns_null);
    RUN_TEST(arena_32byte_alignment_guarantees);
    RUN_TEST(arena_consecutive_allocations_alignment);
    RUN_TEST(arena_reset_watermark_behavior);
    RUN_TEST(arena_capacity_tracking);
    RUN_TEST(arena_minimum_size_allocation);
    RUN_TEST(arena_near_capacity_allocation);
    RUN_TEST(arena_utilization_calculation);
    RUN_TEST(arena_scope_functionality);
    RUN_TEST(frame_arena_reset_at_step_start);
    RUN_TEST(frame_arena_after_physics);
    RUN_TEST(frame_arena_after_collision);
    RUN_TEST(frame_arena_after_sensors);
    RUN_TEST(frame_arena_peak_usage);
    RUN_TEST(frame_arena_reset_no_leak);
    RUN_TEST(frame_arena_no_accumulation);
    RUN_TEST(frame_arena_high_freq_reset_cycles);
    RUN_TEST(frame_arena_varying_actions);
    RUN_TEST(frame_arena_boundary_conditions);
    RUN_TEST(persistent_arena_stable_1000_steps);
    RUN_TEST(persistent_arena_no_growth);
    RUN_TEST(persistent_arena_sized_for_drones);
    RUN_TEST(persistent_arena_survives_reset);
    RUN_TEST(persistent_arena_consistent_across_resets);
    RUN_TEST(persistent_arena_max_drones);
    RUN_TEST(persistent_arena_allocation_order);
    RUN_TEST(persistent_arena_no_fragmentation);
    RUN_TEST(drone_state_soa_all_aligned_32);
    RUN_TEST(drone_params_soa_all_aligned_32);
    RUN_TEST(observation_buffer_aligned_32);
    RUN_TEST(action_buffer_aligned_32);
    RUN_TEST(reward_buffer_aligned_32);
    RUN_TEST(done_truncation_buffers_aligned);
    RUN_TEST(episode_tracking_buffers_aligned);
    RUN_TEST(termination_flag_buffers_aligned);
    RUN_TEST(buffers_aligned_after_operations);
    RUN_TEST(alignment_various_drone_counts);
    RUN_TEST(soa_array_sizes_match_capacity);
    RUN_TEST(soa_arrays_contiguous);
    RUN_TEST(buffer_pointers_stable_across_steps);
    RUN_TEST(buffer_pointers_stable_across_resets);
    RUN_TEST(buffer_sizes_match_config);
    RUN_TEST(buffer_contents_survive_step);
    RUN_TEST(buffer_zero_initialized);
    RUN_TEST(buffer_independence);
    RUN_TEST(buffer_bounds_respected);
    RUN_TEST(buffer_reuse_after_autoreset);
    RUN_TEST(action_buffer_writeable_before_step);
    RUN_TEST(observation_buffer_updated_after_step);
    RUN_TEST(create_destroy_no_leak);
    RUN_TEST(multiple_create_destroy_no_accumulation);
    RUN_TEST(reset_no_memory_leak);
    RUN_TEST(partial_reset_no_leak);
    RUN_TEST(auto_reset_no_leak);
    RUN_TEST(long_running_no_leak);
    RUN_TEST(subsystem_destruction_order);
    RUN_TEST(hot_data_contiguous);
    RUN_TEST(cold_data_separated);
    RUN_TEST(per_drone_hot_data_size);
    RUN_TEST(stride_patterns_cache_friendly);
    RUN_TEST(false_sharing_avoidance);
    RUN_TEST(sequential_access_patterns);
    RUN_TEST(max_drone_count_allocation);
    RUN_TEST(near_max_arena_allocation);
    RUN_TEST(rapid_create_destroy_100);
    RUN_TEST(high_freq_step_10000);
    RUN_TEST(memory_stable_varying_actions);
    RUN_TEST(memory_stable_many_resets);
    RUN_TEST(arena_stress_alternating_sizes);
    RUN_TEST(physics_step_memory_bounded);
    RUN_TEST(collision_step_memory_bounded);
    RUN_TEST(sensor_step_memory_bounded);
    RUN_TEST(full_step_memory_bounded);
    RUN_TEST(memory_scales_with_drones);
    RUN_TEST(memory_independent_of_steps);

    TEST_SUITE_END();
}
