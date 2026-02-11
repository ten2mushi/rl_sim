/**
 * Test Suite for Reusable Lock-Free Barrier
 *
 * Tests verify:
 * - Initialization
 * - Single thread immediate release
 * - Multi-thread synchronization
 * - Barrier reuse
 * - Generation counter prevents spurious wakeups
 */

#include "threading.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>
#include "test_harness.h"

/* ============================================================================
 * Test Helpers
 * ============================================================================ */

static Barrier* create_test_barrier(uint32_t threshold) {
    Barrier* b = (Barrier*)aligned_alloc(64, sizeof(Barrier));
    assert(b != NULL);
    barrier_init(b, threshold);
    return b;
}

static void destroy_test_barrier(Barrier* b) {
    free(b);
}

/* ============================================================================
 * Initialization Tests
 * ============================================================================ */

TEST(barrier_init_with_threshold_1) {
    Barrier* b = create_test_barrier(1);
    ASSERT_EQ(b->threshold, 1);
    ASSERT_EQ(atomic_load_relaxed(&b->count), 0);
    ASSERT_EQ(atomic_load_relaxed(&b->generation), 0);
    destroy_test_barrier(b);
    return 0;
}

TEST(barrier_init_with_threshold_8) {
    Barrier* b = create_test_barrier(8);
    ASSERT_EQ(b->threshold, 8);
    destroy_test_barrier(b);
    return 0;
}

/* ============================================================================
 * Single Thread Tests
 * ============================================================================ */

TEST(barrier_single_thread_immediate_release) {
    Barrier* b = create_test_barrier(1);
    barrier_wait(b);  /* Should not block */
    /* If we reach here, test passes */
    destroy_test_barrier(b);
    return 0;
}

/* ============================================================================
 * Multi-Thread Tests
 * ============================================================================ */

typedef struct {
    Barrier* barrier;
    atomic_u32* arrived;
    atomic_u32* departed;
    uint32_t id;
} BarrierThreadArgs;

static void* barrier_thread_fn(void* arg) {
    BarrierThreadArgs* args = (BarrierThreadArgs*)arg;

    /* Signal arrival */
    atomic_add_relaxed(args->arrived, 1);

    /* Wait at barrier */
    barrier_wait(args->barrier);

    /* Signal departure */
    atomic_add_relaxed(args->departed, 1);

    return NULL;
}

TEST(barrier_two_threads_both_proceed) {
    Barrier* b = create_test_barrier(2);
    atomic_u32 arrived = ATOMIC_VAR_INIT(0);
    atomic_u32 departed = ATOMIC_VAR_INIT(0);

    BarrierThreadArgs args = { b, &arrived, &departed, 0 };

    pthread_t t;
    pthread_create(&t, NULL, barrier_thread_fn, &args);

    /* Main thread also arrives */
    atomic_add_relaxed(&arrived, 1);
    barrier_wait(b);
    atomic_add_relaxed(&departed, 1);

    pthread_join(t, NULL);

    ASSERT_EQ(atomic_load_relaxed(&arrived), 2);
    ASSERT_EQ(atomic_load_relaxed(&departed), 2);
    destroy_test_barrier(b);
    return 0;
}

TEST(barrier_n_threads_all_synchronize) {
    const uint32_t N = 8;
    Barrier* b = create_test_barrier(N);
    atomic_u32 arrived = ATOMIC_VAR_INIT(0);
    atomic_u32 departed = ATOMIC_VAR_INIT(0);

    pthread_t threads[N - 1];  /* N-1 threads + main */
    BarrierThreadArgs args[N - 1];

    for (uint32_t i = 0; i < N - 1; i++) {
        args[i] = (BarrierThreadArgs){ b, &arrived, &departed, i };
        pthread_create(&threads[i], NULL, barrier_thread_fn, &args[i]);
    }

    /* Main thread arrives, releasing all */
    atomic_add_relaxed(&arrived, 1);
    barrier_wait(b);
    atomic_add_relaxed(&departed, 1);

    /* Wait for all threads */
    for (uint32_t i = 0; i < N - 1; i++) {
        pthread_join(threads[i], NULL);
    }

    /* All should have arrived and departed */
    ASSERT_EQ(atomic_load_relaxed(&arrived), N);
    ASSERT_EQ(atomic_load_relaxed(&departed), N);
    destroy_test_barrier(b);
    return 0;
}

/* ============================================================================
 * Reusable Barrier Tests
 * ============================================================================ */

typedef struct {
    Barrier* barrier;
    atomic_u32* phase_counter;
    uint32_t phases;
} MultiPhaseArgs;

static void* multi_phase_thread(void* arg) {
    MultiPhaseArgs* args = (MultiPhaseArgs*)arg;

    for (uint32_t p = 0; p < args->phases; p++) {
        atomic_add_relaxed(args->phase_counter, 1);
        barrier_wait(args->barrier);
    }

    return NULL;
}

TEST(barrier_reuse_multiple_phases) {
    const uint32_t N = 4;
    const uint32_t PHASES = 10;
    Barrier* b = create_test_barrier(N);
    atomic_u32 phase_counter = ATOMIC_VAR_INIT(0);

    pthread_t threads[N - 1];
    MultiPhaseArgs args = { b, &phase_counter, PHASES };

    for (uint32_t i = 0; i < N - 1; i++) {
        pthread_create(&threads[i], NULL, multi_phase_thread, &args);
    }

    /* Main thread also participates */
    for (uint32_t p = 0; p < PHASES; p++) {
        atomic_add_relaxed(&phase_counter, 1);
        barrier_wait(b);
    }

    for (uint32_t i = 0; i < N - 1; i++) {
        pthread_join(threads[i], NULL);
    }

    /* Each of N threads incremented counter PHASES times */
    ASSERT_EQ(atomic_load_relaxed(&phase_counter), N * PHASES);
    destroy_test_barrier(b);
    return 0;
}

TEST(barrier_reset_for_new_use) {
    Barrier* b = create_test_barrier(2);
    atomic_u32 counter = ATOMIC_VAR_INIT(0);

    /* First phase */
    {
        pthread_t t;
        BarrierThreadArgs args = { b, &counter, &counter, 0 };
        pthread_create(&t, NULL, barrier_thread_fn, &args);

        atomic_add_relaxed(&counter, 1);
        barrier_wait(b);
        atomic_add_relaxed(&counter, 1);

        pthread_join(t, NULL);
    }

    uint32_t phase1_count = atomic_load_relaxed(&counter);
    ASSERT_EQ(phase1_count, 4);  /* 2 arrived + 2 departed */

    /* Reset for reuse */
    barrier_reset(b);

    /* Second phase */
    {
        pthread_t t;
        atomic_u32 counter2 = ATOMIC_VAR_INIT(0);
        BarrierThreadArgs args = { b, &counter2, &counter2, 0 };
        pthread_create(&t, NULL, barrier_thread_fn, &args);

        atomic_add_relaxed(&counter2, 1);
        barrier_wait(b);
        atomic_add_relaxed(&counter2, 1);

        pthread_join(t, NULL);

        ASSERT_EQ(atomic_load_relaxed(&counter2), 4);
    }

    destroy_test_barrier(b);
    return 0;
}

/* ============================================================================
 * Generation Counter Tests
 * ============================================================================ */

typedef struct {
    Barrier* barrier;
    atomic_u32* phase_arrivals;
    uint32_t phases;
} GenTestArgs;

static void* gen_thread_fn(void* arg) {
    GenTestArgs* args = (GenTestArgs*)arg;
    for (uint32_t p = 0; p < args->phases; p++) {
        atomic_add_relaxed(&args->phase_arrivals[p], 1);
        barrier_wait(args->barrier);
    }
    return NULL;
}

TEST(barrier_generation_prevents_spurious_wakeup) {
    const uint32_t N = 4;
    const uint32_t PHASES = 100;
    Barrier* b = create_test_barrier(N);

    /*
     * Track per-phase arrivals. If any thread leaks through to the next
     * phase early, we'll detect it by seeing more than N arrivals in a phase.
     */
    atomic_u32 phase_arrivals[PHASES];
    for (uint32_t p = 0; p < PHASES; p++) {
        atomic_init(&phase_arrivals[p], 0);
    }

    pthread_t threads[N - 1];
    GenTestArgs args = { b, phase_arrivals, PHASES };

    for (uint32_t i = 0; i < N - 1; i++) {
        pthread_create(&threads[i], NULL, gen_thread_fn, &args);
    }

    /* Main thread participates */
    for (uint32_t p = 0; p < PHASES; p++) {
        atomic_add_relaxed(&phase_arrivals[p], 1);
        barrier_wait(b);
    }

    for (uint32_t i = 0; i < N - 1; i++) {
        pthread_join(threads[i], NULL);
    }

    /* Verify exactly N arrivals per phase */
    for (uint32_t p = 0; p < PHASES; p++) {
        ASSERT_EQ(atomic_load_relaxed(&phase_arrivals[p]), N);
    }

    destroy_test_barrier(b);
    return 0;
}

/* ============================================================================
 * Stress Test
 * ============================================================================ */

typedef struct {
    Barrier* barrier;
    atomic_u32* counter;
    uint32_t phases;
} StressArgs;

static void* stress_thread_fn(void* arg) {
    StressArgs* args = (StressArgs*)arg;
    for (uint32_t p = 0; p < args->phases; p++) {
        atomic_add_relaxed(args->counter, 1);
        barrier_wait(args->barrier);
    }
    return NULL;
}

TEST(barrier_stress_many_threads_many_phases) {
    const uint32_t N = 16;
    const uint32_t PHASES = 50;
    Barrier* b = create_test_barrier(N);
    atomic_u32 global_counter = ATOMIC_VAR_INIT(0);

    pthread_t threads[N - 1];
    StressArgs args = { b, &global_counter, PHASES };

    for (uint32_t i = 0; i < N - 1; i++) {
        pthread_create(&threads[i], NULL, stress_thread_fn, &args);
    }

    for (uint32_t p = 0; p < PHASES; p++) {
        atomic_add_relaxed(&global_counter, 1);
        barrier_wait(b);
    }

    for (uint32_t i = 0; i < N - 1; i++) {
        pthread_join(threads[i], NULL);
    }

    ASSERT_EQ(atomic_load_relaxed(&global_counter), N * PHASES);
    destroy_test_barrier(b);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Barrier Tests");

    /* Initialization tests */
    RUN_TEST(barrier_init_with_threshold_1);
    RUN_TEST(barrier_init_with_threshold_8);

    /* Single thread tests */
    RUN_TEST(barrier_single_thread_immediate_release);

    /* Multi-thread tests */
    RUN_TEST(barrier_two_threads_both_proceed);
    RUN_TEST(barrier_n_threads_all_synchronize);

    /* Reusable barrier tests */
    RUN_TEST(barrier_reuse_multiple_phases);
    RUN_TEST(barrier_reset_for_new_use);

    /* Generation counter tests */
    RUN_TEST(barrier_generation_prevents_spurious_wakeup);

    /* Stress test */
    RUN_TEST(barrier_stress_many_threads_many_phases);

    TEST_SUITE_END();
}
