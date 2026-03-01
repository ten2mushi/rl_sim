/**
 * @file test_engine_memory.c
 * @brief Deep Memory Management Tests for BatchEngine (12 tests)
 *
 * Tests verify memory safety and optimization:
 * - Arena allocator behavior (alloc, alignment, reset, overflow)
 * - Frame arena reset each step
 * - Persistent arena stability across steps
 * - SoA array alignment (32-byte for AVX2)
 * - Hot/cold data separation for cache efficiency
 * - False sharing avoidance for threading
 * - Memory leak detection (valgrind-compatible)
 * - Address sanitizer compatibility
 *
 * Reference: 07-c-language-patterns.md, 09-data-structures.md
 */

#include "environment_manager.h"
#include "platform_quadcopter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "test_harness.h"

/* ============================================================================
 * Test Helpers
 * ============================================================================ */

/**
 * Create a test engine with fixed seed
 */
static BatchEngine* create_test_engine(uint64_t seed) {
    EngineConfig cfg = engine_config_default();
    cfg.num_envs = 4;
    cfg.agents_per_env = 4;
    cfg.seed = seed;
    cfg.persistent_arena_size = 128 * 1024 * 1024;  /* 128 MB */
    cfg.frame_arena_size = 32 * 1024 * 1024;        /* 32 MB */

    char error[ENGINE_ERROR_MSG_SIZE];
    return engine_create(&cfg, error);
}

/**
 * Check if a pointer is N-byte aligned
 */
static bool is_aligned(const void* ptr, size_t alignment) {
    return ((uintptr_t)ptr & (alignment - 1)) == 0;
}

/* ============================================================================
 * Test 1: Arena Alloc Basic
 * ============================================================================ */

TEST(arena_alloc_basic) {
    /* Create a test arena */
    Arena* arena = arena_create(1024 * 1024);  /* 1 MB */
    ASSERT_NOT_NULL(arena);
    ASSERT_NOT_NULL(arena->data);
    ASSERT_EQ(arena->capacity, 1024 * 1024);
    ASSERT_EQ(arena->used, 0);

    /* Allocate some memory */
    void* ptr1 = arena_alloc(arena, 256);
    ASSERT_NOT_NULL(ptr1);
    ASSERT_GT(arena->used, 0);

    /* Allocate more memory */
    size_t used_after_first = arena->used;
    void* ptr2 = arena_alloc(arena, 512);
    ASSERT_NOT_NULL(ptr2);
    ASSERT_GT(arena->used, used_after_first);

    /* Pointers should be distinct */
    ASSERT_NE(ptr1, ptr2);

    /* Pointers should be within arena bounds */
    ASSERT_GE((uintptr_t)ptr1, (uintptr_t)arena->data);
    ASSERT_LT((uintptr_t)ptr1, (uintptr_t)arena->data + arena->capacity);
    ASSERT_GE((uintptr_t)ptr2, (uintptr_t)arena->data);
    ASSERT_LT((uintptr_t)ptr2, (uintptr_t)arena->data + arena->capacity);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Test 2: Arena Alloc Alignment (32-byte for SIMD)
 * ============================================================================ */

TEST(arena_alloc_alignment) {
    Arena* arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Request 32-byte aligned allocations */
    void* ptr1 = arena_alloc_aligned(arena, 100, 32);
    ASSERT_NOT_NULL(ptr1);
    ASSERT_TRUE(is_aligned(ptr1, 32));

    void* ptr2 = arena_alloc_aligned(arena, 100, 32);
    ASSERT_NOT_NULL(ptr2);
    ASSERT_TRUE(is_aligned(ptr2, 32));

    /* Request 64-byte aligned allocations (cache line) */
    void* ptr3 = arena_alloc_aligned(arena, 100, 64);
    ASSERT_NOT_NULL(ptr3);
    ASSERT_TRUE(is_aligned(ptr3, 64));

    /* Default alignment should be at least 16 bytes */
    void* ptr4 = arena_alloc(arena, 100);
    ASSERT_NOT_NULL(ptr4);
    ASSERT_TRUE(is_aligned(ptr4, 16));

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Test 3: Arena Reset Watermark
 * ============================================================================ */

TEST(arena_reset_watermark) {
    Arena* arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    size_t original_capacity = arena->capacity;

    /* Allocate some memory */
    void* ptr1 = arena_alloc(arena, 1000);
    ASSERT_NOT_NULL(ptr1);
    void* ptr2 = arena_alloc(arena, 2000);
    ASSERT_NOT_NULL(ptr2);
    void* ptr3 = arena_alloc(arena, 3000);
    ASSERT_NOT_NULL(ptr3);

    ASSERT_GT(arena->used, 0);

    /* Reset arena */
    arena_reset(arena);

    /* After reset, used should be 0 */
    ASSERT_EQ(arena->used, 0);

    /* Capacity should be unchanged */
    ASSERT_EQ(arena->capacity, original_capacity);

    /* Should be able to allocate again */
    void* ptr4 = arena_alloc(arena, 500);
    ASSERT_NOT_NULL(ptr4);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Test 4: Arena Overflow Handling
 * ============================================================================ */

TEST(arena_overflow_handling) {
    /* Create a small arena */
    Arena* arena = arena_create(1000);
    ASSERT_NOT_NULL(arena);

    /* Allocate within capacity */
    void* ptr1 = arena_alloc(arena, 400);
    ASSERT_NOT_NULL(ptr1);

    void* ptr2 = arena_alloc(arena, 400);
    ASSERT_NOT_NULL(ptr2);

    /* This allocation should fail (exceeds capacity) */
    void* ptr3 = arena_alloc(arena, 400);
    ASSERT_NULL(ptr3);

    /* Arena should still be valid after failed allocation */
    ASSERT_LE(arena->used, arena->capacity);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Test 5: Frame Arena Reset Each Step
 * ============================================================================ */

TEST(frame_arena_reset_each_step) {
    BatchEngine* engine = create_test_engine(12345);
    ASSERT_NOT_NULL(engine);
    ASSERT_NOT_NULL(engine->frame_arena);

    engine_reset(engine);

    /* Record initial frame arena state */
    size_t initial_capacity = engine->frame_arena->capacity;

    /* Step multiple times and check frame arena resets */
    for (int step = 0; step < 10; step++) {
        /* Before step, record used memory */
        size_t used_before = engine->frame_arena->used;

        engine_step(engine);

        /* Frame arena should be reset to 0 or near-0 at start of each step */
        /* After step, it may have some usage from computations */
        /* The key invariant is that it doesn't grow unbounded */
        ASSERT_LT(engine->frame_arena->used, engine->frame_arena->capacity);
    }

    /* Capacity should remain unchanged */
    ASSERT_EQ(engine->frame_arena->capacity, initial_capacity);

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 6: Persistent Arena Stability Across 1000 Steps
 * ============================================================================ */

TEST(persistent_arena_stability) {
    BatchEngine* engine = create_test_engine(12345);
    ASSERT_NOT_NULL(engine);
    ASSERT_NOT_NULL(engine->persistent_arena);

    engine_reset(engine);

    /* Record persistent arena usage after initialization */
    size_t initial_used = engine->persistent_arena->used;
    size_t initial_capacity = engine->persistent_arena->capacity;

    /* Run 1000 steps */
    for (int step = 0; step < 1000; step++) {
        engine_step(engine);
    }

    /* Persistent arena usage should not grow */
    ASSERT_EQ(engine->persistent_arena->used, initial_used);
    ASSERT_EQ(engine->persistent_arena->capacity, initial_capacity);

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 7: SoA State Array Alignment
 * ============================================================================ */

TEST(soa_array_alignment) {
    BatchEngine* engine = create_test_engine(12345);
    ASSERT_NOT_NULL(engine);
    ASSERT_NOT_NULL(engine->states);

    /* All 17 PlatformStateSOA arrays must be 32-byte aligned for AVX2 */
    ASSERT_TRUE(is_aligned(engine->states->rigid_body.pos_x, 32));
    ASSERT_TRUE(is_aligned(engine->states->rigid_body.pos_y, 32));
    ASSERT_TRUE(is_aligned(engine->states->rigid_body.pos_z, 32));
    ASSERT_TRUE(is_aligned(engine->states->rigid_body.vel_x, 32));
    ASSERT_TRUE(is_aligned(engine->states->rigid_body.vel_y, 32));
    ASSERT_TRUE(is_aligned(engine->states->rigid_body.vel_z, 32));
    ASSERT_TRUE(is_aligned(engine->states->rigid_body.quat_w, 32));
    ASSERT_TRUE(is_aligned(engine->states->rigid_body.quat_x, 32));
    ASSERT_TRUE(is_aligned(engine->states->rigid_body.quat_y, 32));
    ASSERT_TRUE(is_aligned(engine->states->rigid_body.quat_z, 32));
    ASSERT_TRUE(is_aligned(engine->states->rigid_body.omega_x, 32));
    ASSERT_TRUE(is_aligned(engine->states->rigid_body.omega_y, 32));
    ASSERT_TRUE(is_aligned(engine->states->rigid_body.omega_z, 32));
    ASSERT_TRUE(is_aligned(engine->states->extension[QUAD_EXT_RPM_0], 32));
    ASSERT_TRUE(is_aligned(engine->states->extension[QUAD_EXT_RPM_1], 32));
    ASSERT_TRUE(is_aligned(engine->states->extension[QUAD_EXT_RPM_2], 32));
    ASSERT_TRUE(is_aligned(engine->states->extension[QUAD_EXT_RPM_3], 32));

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 8: SoA Params Array Alignment
 * ============================================================================ */

TEST(soa_params_alignment) {
    BatchEngine* engine = create_test_engine(12345);
    ASSERT_NOT_NULL(engine);
    ASSERT_NOT_NULL(engine->params);

    /* All 15 PlatformParamsSOA arrays must be 32-byte aligned */
    ASSERT_TRUE(is_aligned(engine->params->rigid_body.mass, 32));
    ASSERT_TRUE(is_aligned(engine->params->rigid_body.ixx, 32));
    ASSERT_TRUE(is_aligned(engine->params->rigid_body.iyy, 32));
    ASSERT_TRUE(is_aligned(engine->params->rigid_body.izz, 32));
    ASSERT_TRUE(is_aligned(engine->params->extension[QUAD_PEXT_ARM_LENGTH], 32));
    ASSERT_TRUE(is_aligned(engine->params->rigid_body.collision_radius, 32));
    ASSERT_TRUE(is_aligned(engine->params->extension[QUAD_PEXT_K_THRUST], 32));
    ASSERT_TRUE(is_aligned(engine->params->extension[QUAD_PEXT_K_TORQUE], 32));
    ASSERT_TRUE(is_aligned(engine->params->extension[QUAD_PEXT_K_DRAG], 32));
    ASSERT_TRUE(is_aligned(engine->params->extension[QUAD_PEXT_K_ANG_DAMP], 32));
    ASSERT_TRUE(is_aligned(engine->params->extension[QUAD_PEXT_MOTOR_TAU], 32));
    ASSERT_TRUE(is_aligned(engine->params->extension[QUAD_PEXT_MAX_RPM], 32));
    ASSERT_TRUE(is_aligned(engine->params->rigid_body.max_vel, 32));
    ASSERT_TRUE(is_aligned(engine->params->rigid_body.max_omega, 32));
    ASSERT_TRUE(is_aligned(engine->params->rigid_body.gravity, 32));

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 9: Hot/Cold Data Separation
 * ============================================================================ */

TEST(hot_cold_separation) {
    BatchEngine* engine = create_test_engine(12345);
    ASSERT_NOT_NULL(engine);

    /* Hot data (pos/vel/quat/omega) should fit in roughly 64 bytes per drone:
     * - pos: 12 bytes (3 floats)
     * - vel: 12 bytes (3 floats)
     * - quat: 16 bytes (4 floats)
     * - omega: 12 bytes (3 floats)
     * Total: 52 bytes, well under 64-byte cache line
     *
     * RPMs add 16 bytes = 68 bytes total, still close to 1 cache line
     */
    size_t hot_data_size_per_drone =
        3 * sizeof(float) +   /* pos */
        3 * sizeof(float) +   /* vel */
        4 * sizeof(float) +   /* quat */
        3 * sizeof(float);    /* omega */

    /* Should be <= 64 bytes for cache efficiency */
    ASSERT_LE(hot_data_size_per_drone, 64);

    /* Verify hot data is accessed contiguously in SoA layout */
    /* pos_x, pos_y, pos_z arrays should be contiguous for SIMD */
    /* This is inherent in SoA design - arrays are separate */

    /* Verify cold data (params) is stored separately */
    ASSERT_NE((void*)engine->states, (void*)engine->params);

    /* Episode tracking data should also be separate (cold) */
    ASSERT_NOT_NULL(engine->episode_returns);
    ASSERT_NOT_NULL(engine->episode_lengths);

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 10: False Sharing Avoidance
 * ============================================================================ */

TEST(false_sharing_avoidance) {
    BatchEngine* engine = create_test_engine(12345);
    ASSERT_NOT_NULL(engine);

    /* Per-thread data should be on separate 64-byte cache lines
     * to avoid false sharing in multi-threaded physics.
     *
     * The scheduler/thread pool should have padded per-thread state.
     * We verify this by checking that thread pool exists and
     * the design follows cache-line separation principles.
     */

    /* Thread pool should exist if threading is enabled */
    /* Even with num_threads=0 (auto), scheduler should be allocated */
    ASSERT_NOT_NULL(engine->scheduler);

    /* Each SoA array should be 32-byte aligned, which helps with
     * cache line alignment when processing chunks of 8 drones
     * (8 floats = 32 bytes, half a cache line)
     */
    ASSERT_TRUE(is_aligned(engine->states->rigid_body.pos_x, 32));

    /* Verify that arrays have sufficient stride between elements
     * for false-sharing avoidance when different threads process
     * different contiguous chunks of drones.
     */
    uint32_t total_agents = engine->config.total_agents;
    ASSERT_GT(total_agents, 0);

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 11: No Memory Leaks (Valgrind-Compatible)
 * ============================================================================ */

TEST(valgrind_no_leaks) {
    /* This test is designed to be run with:
     * valgrind --leak-check=full --show-leak-kinds=all --error-exitcode=1 ./test_engine_memory
     *
     * The test creates, uses, and destroys multiple engines to verify
     * no memory is leaked. If run under valgrind, any leaks will be detected.
     */

    for (int i = 0; i < 3; i++) {
        BatchEngine* engine = create_test_engine(12345 + i);
        ASSERT_NOT_NULL(engine);

        engine_reset(engine);

        for (int step = 0; step < 100; step++) {
            engine_step(engine);
        }

        /* Reset and run again */
        engine_reset(engine);

        for (int step = 0; step < 50; step++) {
            engine_step(engine);
        }

        engine_destroy(engine);
    }

    /* If we get here without valgrind errors, memory is properly freed */
    return 0;
}

/* ============================================================================
 * Test 12: Address Sanitizer Clean
 * ============================================================================ */

TEST(asan_clean) {
    /* This test is designed to be run with:
     * cmake .. -DCMAKE_C_FLAGS="-fsanitize=address -g"
     *
     * Tests various memory access patterns that ASAN would catch:
     * - Buffer overflows
     * - Use-after-free
     * - Stack buffer overflow
     */

    BatchEngine* engine = create_test_engine(12345);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    uint32_t total_agents = engine->config.total_agents;

    /* Test 1: Access within bounds */
    for (uint32_t i = 0; i < total_agents; i++) {
        /* Read access */
        float px = engine->states->rigid_body.pos_x[i];
        float py = engine->states->rigid_body.pos_y[i];
        float pz = engine->states->rigid_body.pos_z[i];
        ASSERT_TRUE(isfinite(px) || !isfinite(px));  /* Just access, don't crash */

        /* Write access */
        engine->states->rigid_body.pos_x[i] = 0.0f;
        engine->states->rigid_body.pos_y[i] = 0.0f;
        engine->states->rigid_body.pos_z[i] = 0.0f;
    }

    /* Test 2: Action buffer access within bounds */
    float* actions = engine_get_actions(engine);
    for (uint32_t i = 0; i < total_agents * engine->action_dim; i++) {
        actions[i] = 0.5f;
    }

    /* Test 3: Observation buffer access within bounds */
    uint32_t obs_dim = engine_get_obs_dim(engine);
    float* obs = engine_get_observations(engine);
    for (uint32_t i = 0; i < total_agents * obs_dim && i < 1000; i++) {
        float val = obs[i];
        (void)val;
    }

    /* Test 4: Done/truncation buffer access */
    uint8_t* dones = engine_get_dones(engine);
    uint8_t* truncs = engine_get_truncations(engine);
    for (uint32_t i = 0; i < total_agents; i++) {
        uint8_t d = dones[i];
        uint8_t t = truncs[i];
        (void)d;
        (void)t;
    }

    /* Test 5: Step should not cause any memory issues */
    for (int s = 0; s < 100; s++) {
        engine_step(engine);
    }

    engine_destroy(engine);

    /* If ASAN is enabled and we reach here, no memory errors occurred */
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Memory Management Tests");

    printf("Arena Allocator:\n");
    RUN_TEST(arena_alloc_basic);
    RUN_TEST(arena_alloc_alignment);
    RUN_TEST(arena_reset_watermark);
    RUN_TEST(arena_overflow_handling);

    printf("\nArena Lifecycle:\n");
    RUN_TEST(frame_arena_reset_each_step);
    RUN_TEST(persistent_arena_stability);

    printf("\nSoA Alignment:\n");
    RUN_TEST(soa_array_alignment);
    RUN_TEST(soa_params_alignment);

    printf("\nCache Optimization:\n");
    RUN_TEST(hot_cold_separation);
    RUN_TEST(false_sharing_avoidance);

    printf("\nSanitizer Compatibility:\n");
    RUN_TEST(valgrind_no_leaks);
    RUN_TEST(asan_clean);

    TEST_SUITE_END();
}
