/**
 * Test Suite for Hybrid Scheduler
 *
 * Tests verify:
 * - Creation and configuration
 * - Static scheduling (physics-style)
 * - Work-stealing scheduling (sensor-style)
 * - Adaptive strategy selection
 * - Parallel for loop
 * - Correctness under various strategies
 */

#include "threading.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "test_harness.h"

/* ============================================================================
 * Test Work Functions
 * ============================================================================ */

static void mark_range_fn(void* data, uint32_t start, uint32_t end, uint32_t thread_id) {
    (void)thread_id;
    uint32_t* arr = (uint32_t*)data;
    for (uint32_t i = start; i < end; i++) {
        arr[i] = 1;
    }
}

static void sum_indices_fn(void* data, uint32_t start, uint32_t end, uint32_t thread_id) {
    (void)thread_id;
    atomic_u64* sum = (atomic_u64*)data;
    uint64_t local = 0;
    for (uint32_t i = start; i < end; i++) {
        local += i;
    }
    atomic_add_relaxed(sum, local);
}

static void count_invocations_fn(void* data, uint32_t start, uint32_t end, uint32_t thread_id) {
    (void)start;
    (void)end;
    (void)thread_id;
    atomic_u32* count = (atomic_u32*)data;
    atomic_add_relaxed(count, 1);
}

static void record_range_size_fn(void* data, uint32_t start, uint32_t end, uint32_t thread_id) {
    uint32_t* sizes = (uint32_t*)data;
    sizes[thread_id] = end - start;
}

/* ============================================================================
 * Test Helpers
 * ============================================================================ */

static ThreadPool* create_test_pool(uint32_t num_threads) {
    ThreadPoolConfig config = { .num_threads = num_threads };
    ThreadPool* pool = threadpool_create(&config);
    assert(pool != NULL);
    return pool;
}

/* ============================================================================
 * Creation Tests
 * ============================================================================ */

TEST(scheduler_create_with_pool) {
    ThreadPool* pool = create_test_pool(4);
    Scheduler* sched = scheduler_create(pool);
    ASSERT_NOT_NULL(sched);
    ASSERT_EQ(sched->pool, pool);
    scheduler_destroy(sched);
    threadpool_destroy(pool);
    return 0;
}

TEST(scheduler_create_null_pool_fails) {
    Scheduler* sched = scheduler_create(NULL);
    ASSERT_TRUE(sched == NULL);
    return 0;
}

TEST(scheduler_create_sets_defaults) {
    ThreadPool* pool = create_test_pool(4);
    Scheduler* sched = scheduler_create(pool);
    ASSERT_NOT_NULL(sched);
    ASSERT_EQ(sched->default_strategy, SCHEDULE_ADAPTIVE);
    ASSERT_GT(sched->steal_threshold, 0);
    ASSERT_GT(sched->min_chunk_size, 0);
    scheduler_destroy(sched);
    threadpool_destroy(pool);
    return 0;
}

TEST(scheduler_configure_thresholds) {
    ThreadPool* pool = create_test_pool(4);
    Scheduler* sched = scheduler_create(pool);

    scheduler_configure(sched, 1000, 128);
    ASSERT_EQ(sched->steal_threshold, 1000);
    ASSERT_EQ(sched->min_chunk_size, 128);

    scheduler_destroy(sched);
    threadpool_destroy(pool);
    return 0;
}

/* ============================================================================
 * scheduler_execute Tests
 * ============================================================================ */

TEST(scheduler_execute_processes_all_items) {
    ThreadPool* pool = create_test_pool(4);
    Scheduler* sched = scheduler_create(pool);

    const uint32_t N = 10000;
    uint32_t* data = (uint32_t*)calloc(N, sizeof(uint32_t));
    ASSERT_NOT_NULL(data);

    scheduler_execute(sched, mark_range_fn, data, N, SCHEDULE_WORK_STEALING);

    for (uint32_t i = 0; i < N; i++) {
        ASSERT_EQ(data[i], 1);
    }

    free(data);
    scheduler_destroy(sched);
    threadpool_destroy(pool);
    return 0;
}

TEST(scheduler_execute_empty_range) {
    ThreadPool* pool = create_test_pool(4);
    Scheduler* sched = scheduler_create(pool);

    /* Should not crash on empty range */
    scheduler_execute(sched, mark_range_fn, NULL, 0, SCHEDULE_WORK_STEALING);

    scheduler_destroy(sched);
    threadpool_destroy(pool);
    return 0;
}

TEST(scheduler_execute_single_item) {
    ThreadPool* pool = create_test_pool(4);
    Scheduler* sched = scheduler_create(pool);

    uint32_t data = 0;
    scheduler_execute(sched, mark_range_fn, &data, 1, SCHEDULE_WORK_STEALING);

    ASSERT_EQ(data, 1);

    scheduler_destroy(sched);
    threadpool_destroy(pool);
    return 0;
}

/* ============================================================================
 * Static Scheduling Tests (scheduler_physics)
 * ============================================================================ */

TEST(scheduler_physics_processes_all) {
    ThreadPool* pool = create_test_pool(4);
    Scheduler* sched = scheduler_create(pool);

    const uint32_t N = 1000;
    uint32_t* data = (uint32_t*)calloc(N, sizeof(uint32_t));
    ASSERT_NOT_NULL(data);

    scheduler_physics(sched, mark_range_fn, data, N);

    for (uint32_t i = 0; i < N; i++) {
        ASSERT_EQ(data[i], 1);
    }

    free(data);
    scheduler_destroy(sched);
    threadpool_destroy(pool);
    return 0;
}

TEST(scheduler_physics_creates_n_tasks) {
    ThreadPool* pool = create_test_pool(4);
    Scheduler* sched = scheduler_create(pool);

    atomic_u32 task_count = ATOMIC_VAR_INIT(0);

    scheduler_physics(sched, count_invocations_fn, &task_count, 1000);

    /* Static creates exactly num_threads tasks (or fewer if work < threads) */
    ASSERT_LE(atomic_load_relaxed(&task_count), 4);
    ASSERT_GT(atomic_load_relaxed(&task_count), 0);

    scheduler_destroy(sched);
    threadpool_destroy(pool);
    return 0;
}

TEST(scheduler_physics_balanced_partitioning) {
    ThreadPool* pool = create_test_pool(4);
    Scheduler* sched = scheduler_create(pool);

    uint32_t range_sizes[4] = {0};

    scheduler_physics(sched, record_range_size_fn, range_sizes, 1000);

    /* Each partition should be roughly 250 items (1000/4) */
    for (uint32_t i = 0; i < 4; i++) {
        if (range_sizes[i] > 0) {
            ASSERT_GE(range_sizes[i], 200);
            ASSERT_LE(range_sizes[i], 300);
        }
    }

    scheduler_destroy(sched);
    threadpool_destroy(pool);
    return 0;
}

/* ============================================================================
 * Work-Stealing Scheduling Tests
 * ============================================================================ */

TEST(scheduler_work_stealing_creates_many_tasks) {
    ThreadPool* pool = create_test_pool(4);
    Scheduler* sched = scheduler_create(pool);
    sched->min_chunk_size = 64;

    atomic_u32 task_count = ATOMIC_VAR_INIT(0);

    scheduler_execute(sched, count_invocations_fn, &task_count, 1000, SCHEDULE_WORK_STEALING);

    /* Work-stealing creates 1000/64 = ~16 tasks */
    uint32_t count = atomic_load_relaxed(&task_count);
    ASSERT_GT(count, 4);  /* More than static would create */
    ASSERT_LE(count, 20);

    scheduler_destroy(sched);
    threadpool_destroy(pool);
    return 0;
}

TEST(scheduler_work_stealing_sum_correctness) {
    ThreadPool* pool = create_test_pool(8);
    Scheduler* sched = scheduler_create(pool);

    atomic_u64 sum = ATOMIC_VAR_INIT(0);
    const uint32_t N = 100000;

    scheduler_execute(sched, sum_indices_fn, &sum, N, SCHEDULE_WORK_STEALING);

    uint64_t expected = (uint64_t)N * (N - 1) / 2;
    ASSERT_EQ(atomic_load_relaxed(&sum), expected);

    scheduler_destroy(sched);
    threadpool_destroy(pool);
    return 0;
}

/* ============================================================================
 * Adaptive Scheduling Tests
 * ============================================================================ */

TEST(scheduler_adaptive_uses_static_for_small_work) {
    ThreadPool* pool = create_test_pool(4);
    Scheduler* sched = scheduler_create(pool);
    sched->steal_threshold = 500;  /* Work < 500 uses static */
    sched->min_chunk_size = 10;

    atomic_u32 task_count = ATOMIC_VAR_INIT(0);

    /* 100 items < 500 threshold, should use static partitioning */
    scheduler_execute(sched, count_invocations_fn, &task_count, 100, SCHEDULE_ADAPTIVE);

    /* Static creates at most num_threads tasks */
    ASSERT_LE(atomic_load_relaxed(&task_count), 4);

    scheduler_destroy(sched);
    threadpool_destroy(pool);
    return 0;
}

TEST(scheduler_adaptive_uses_stealing_for_large_work) {
    ThreadPool* pool = create_test_pool(4);
    Scheduler* sched = scheduler_create(pool);
    sched->steal_threshold = 500;
    sched->min_chunk_size = 64;

    atomic_u32 task_count = ATOMIC_VAR_INIT(0);

    /* 1000 items > 500 threshold, should use work-stealing */
    scheduler_execute(sched, count_invocations_fn, &task_count, 1000, SCHEDULE_ADAPTIVE);

    /* Work-stealing creates many tasks */
    ASSERT_GT(atomic_load_relaxed(&task_count), 4);

    scheduler_destroy(sched);
    threadpool_destroy(pool);
    return 0;
}

/* ============================================================================
 * Parallel For Tests
 * ============================================================================ */

TEST(scheduler_parallel_for_processes_range) {
    ThreadPool* pool = create_test_pool(4);
    Scheduler* sched = scheduler_create(pool);

    const uint32_t N = 10000;
    uint32_t* data = (uint32_t*)calloc(N, sizeof(uint32_t));
    ASSERT_NOT_NULL(data);

    scheduler_parallel_for(sched, mark_range_fn, data, 0, N, SCHEDULE_WORK_STEALING);

    for (uint32_t i = 0; i < N; i++) {
        ASSERT_EQ(data[i], 1);
    }

    free(data);
    scheduler_destroy(sched);
    threadpool_destroy(pool);
    return 0;
}

TEST(scheduler_parallel_for_offset_range) {
    ThreadPool* pool = create_test_pool(4);
    Scheduler* sched = scheduler_create(pool);

    const uint32_t TOTAL = 10000;
    const uint32_t START = 2000;
    const uint32_t END = 8000;
    uint32_t* data = (uint32_t*)calloc(TOTAL, sizeof(uint32_t));
    ASSERT_NOT_NULL(data);

    scheduler_parallel_for(sched, mark_range_fn, data, START, END, SCHEDULE_WORK_STEALING);

    /* Only START..END should be marked */
    for (uint32_t i = 0; i < TOTAL; i++) {
        if (i >= START && i < END) {
            ASSERT_EQ(data[i], 1);
        } else {
            ASSERT_EQ(data[i], 0);
        }
    }

    free(data);
    scheduler_destroy(sched);
    threadpool_destroy(pool);
    return 0;
}

TEST(scheduler_parallel_for_empty_range) {
    ThreadPool* pool = create_test_pool(4);
    Scheduler* sched = scheduler_create(pool);

    /* Should not crash on empty range */
    scheduler_parallel_for(sched, mark_range_fn, NULL, 0, 0, SCHEDULE_WORK_STEALING);
    scheduler_parallel_for(sched, mark_range_fn, NULL, 100, 100, SCHEDULE_WORK_STEALING);
    scheduler_parallel_for(sched, mark_range_fn, NULL, 100, 50, SCHEDULE_WORK_STEALING);

    scheduler_destroy(sched);
    threadpool_destroy(pool);
    return 0;
}

/* ============================================================================
 * Sensor Scheduling Tests
 * ============================================================================ */

TEST(scheduler_sensors_processes_all_types) {
    ThreadPool* pool = create_test_pool(4);
    Scheduler* sched = scheduler_create(pool);

    /* Simulate 3 sensor types with different work amounts */
    uint32_t work_sizes[] = { 100, 200, 150 };
    uint32_t total = 100 + 200 + 150;
    uint32_t* data = (uint32_t*)calloc(total, sizeof(uint32_t));
    ASSERT_NOT_NULL(data);

    scheduler_sensors(sched, mark_range_fn, data, work_sizes, 3);

    /* All should be marked */
    for (uint32_t i = 0; i < total; i++) {
        ASSERT_EQ(data[i], 1);
    }

    free(data);
    scheduler_destroy(sched);
    threadpool_destroy(pool);
    return 0;
}

/* ============================================================================
 * Correctness Under High Contention
 * ============================================================================ */

TEST(scheduler_high_contention_no_items_lost) {
    ThreadPool* pool = create_test_pool(8);
    Scheduler* sched = scheduler_create(pool);

    atomic_u64 sum = ATOMIC_VAR_INIT(0);
    const uint32_t N = 100000;

    scheduler_execute(sched, sum_indices_fn, &sum, N, SCHEDULE_WORK_STEALING);

    uint64_t expected = (uint64_t)N * (N - 1) / 2;
    ASSERT_EQ(atomic_load_relaxed(&sum), expected);

    scheduler_destroy(sched);
    threadpool_destroy(pool);
    return 0;
}

TEST(scheduler_stress_mixed_strategies) {
    ThreadPool* pool = create_test_pool(4);
    Scheduler* sched = scheduler_create(pool);

    for (uint32_t round = 0; round < 10; round++) {
        atomic_u64 sum = ATOMIC_VAR_INIT(0);
        const uint32_t N = 1000;

        /* Alternate strategies */
        ScheduleStrategy strategy = (round % 3 == 0) ? SCHEDULE_STATIC :
                                    (round % 3 == 1) ? SCHEDULE_WORK_STEALING :
                                                       SCHEDULE_ADAPTIVE;

        scheduler_execute(sched, sum_indices_fn, &sum, N, strategy);

        uint64_t expected = (uint64_t)N * (N - 1) / 2;
        ASSERT_EQ(atomic_load_relaxed(&sum), expected);
    }

    scheduler_destroy(sched);
    threadpool_destroy(pool);
    return 0;
}

TEST(scheduler_stress_large_workload) {
    ThreadPool* pool = create_test_pool(8);
    Scheduler* sched = scheduler_create(pool);

    const uint32_t N = 1000000;
    atomic_u64 sum = ATOMIC_VAR_INIT(0);

    scheduler_execute(sched, sum_indices_fn, &sum, N, SCHEDULE_WORK_STEALING);

    uint64_t expected = (uint64_t)N * (N - 1) / 2;
    ASSERT_EQ(atomic_load_relaxed(&sum), expected);

    scheduler_destroy(sched);
    threadpool_destroy(pool);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Scheduler Tests");

    /* Creation tests */
    RUN_TEST(scheduler_create_with_pool);
    RUN_TEST(scheduler_create_null_pool_fails);
    RUN_TEST(scheduler_create_sets_defaults);
    RUN_TEST(scheduler_configure_thresholds);

    /* scheduler_execute tests */
    RUN_TEST(scheduler_execute_processes_all_items);
    RUN_TEST(scheduler_execute_empty_range);
    RUN_TEST(scheduler_execute_single_item);

    /* Static scheduling tests */
    RUN_TEST(scheduler_physics_processes_all);
    RUN_TEST(scheduler_physics_creates_n_tasks);
    RUN_TEST(scheduler_physics_balanced_partitioning);

    /* Work-stealing tests */
    RUN_TEST(scheduler_work_stealing_creates_many_tasks);
    RUN_TEST(scheduler_work_stealing_sum_correctness);

    /* Adaptive tests */
    RUN_TEST(scheduler_adaptive_uses_static_for_small_work);
    RUN_TEST(scheduler_adaptive_uses_stealing_for_large_work);

    /* Parallel for tests */
    RUN_TEST(scheduler_parallel_for_processes_range);
    RUN_TEST(scheduler_parallel_for_offset_range);
    RUN_TEST(scheduler_parallel_for_empty_range);

    /* Sensor scheduling tests */
    RUN_TEST(scheduler_sensors_processes_all_types);

    /* Correctness tests */
    RUN_TEST(scheduler_high_contention_no_items_lost);
    RUN_TEST(scheduler_stress_mixed_strategies);
    RUN_TEST(scheduler_stress_large_workload);

    TEST_SUITE_END();
}
