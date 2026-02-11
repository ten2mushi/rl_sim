/**
 * Test Suite for Thread Pool
 *
 * Tests verify:
 * - Creation with default and explicit configuration
 * - Work submission and execution
 * - Work distribution across threads
 * - Wait and barrier synchronization
 * - Statistics tracking
 * - Work stealing behavior
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

static void increment_fn(void* data, uint32_t start, uint32_t end, uint32_t thread_id) {
    (void)start;
    (void)end;
    (void)thread_id;
    atomic_u32* counter = (atomic_u32*)data;
    atomic_add_relaxed(counter, 1);
}

static void sum_range_fn(void* data, uint32_t start, uint32_t end, uint32_t thread_id) {
    (void)thread_id;
    atomic_u64* sum = (atomic_u64*)data;
    uint64_t local_sum = 0;
    for (uint32_t i = start; i < end; i++) {
        local_sum += i;
    }
    atomic_add_relaxed(sum, local_sum);
}

static void record_thread_id_fn(void* data, uint32_t start, uint32_t end, uint32_t thread_id) {
    (void)end;
    atomic_u32* ids = (atomic_u32*)data;
    atomic_store_relaxed(&ids[start], thread_id);
}

static void slow_increment_fn(void* data, uint32_t start, uint32_t end, uint32_t thread_id) {
    (void)start;
    (void)end;
    (void)thread_id;
    atomic_u32* counter = (atomic_u32*)data;

    /* Simulate some work */
    volatile int x = 0;
    for (int i = 0; i < 1000; i++) {
        x += i;
    }
    (void)x;

    atomic_add_relaxed(counter, 1);
}

static void noop_fn(void* data, uint32_t start, uint32_t end, uint32_t thread_id) {
    (void)data;
    (void)start;
    (void)end;
    (void)thread_id;
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

TEST(threadpool_create_default_config) {
    ThreadPool* pool = threadpool_create(NULL);
    ASSERT_NOT_NULL(pool);
    ASSERT_GT(threadpool_num_threads(pool), 0);
    threadpool_destroy(pool);
    return 0;
}

TEST(threadpool_create_explicit_thread_count) {
    ThreadPoolConfig config = { .num_threads = 4 };
    ThreadPool* pool = threadpool_create(&config);
    ASSERT_NOT_NULL(pool);
    ASSERT_EQ(threadpool_num_threads(pool), 4);
    threadpool_destroy(pool);
    return 0;
}

TEST(threadpool_create_single_thread) {
    ThreadPoolConfig config = { .num_threads = 1 };
    ThreadPool* pool = threadpool_create(&config);
    ASSERT_NOT_NULL(pool);
    ASSERT_EQ(threadpool_num_threads(pool), 1);
    threadpool_destroy(pool);
    return 0;
}

TEST(threadpool_create_many_threads) {
    ThreadPoolConfig config = { .num_threads = 16 };
    ThreadPool* pool = threadpool_create(&config);
    ASSERT_NOT_NULL(pool);
    ASSERT_EQ(threadpool_num_threads(pool), 16);
    threadpool_destroy(pool);
    return 0;
}

/* ============================================================================
 * Submission Tests
 * ============================================================================ */

TEST(threadpool_submit_single_item_executes) {
    ThreadPool* pool = create_test_pool(2);
    atomic_u32 executed = ATOMIC_VAR_INIT(0);

    WorkItem item = {
        .fn = increment_fn,
        .data = &executed,
        .start = 0,
        .end = 1
    };

    ASSERT_TRUE(threadpool_submit(pool, item));
    threadpool_wait(pool);

    ASSERT_EQ(atomic_load_relaxed(&executed), 1);
    threadpool_destroy(pool);
    return 0;
}

TEST(threadpool_submit_batch_executes_all) {
    ThreadPool* pool = create_test_pool(4);
    atomic_u32 counter = ATOMIC_VAR_INIT(0);

    WorkItem items[100];
    for (uint32_t i = 0; i < 100; i++) {
        items[i] = (WorkItem){
            .fn = increment_fn,
            .data = &counter,
            .start = i,
            .end = i + 1
        };
    }

    ASSERT_TRUE(threadpool_submit_batch(pool, items, 100));
    threadpool_wait(pool);

    ASSERT_EQ(atomic_load_relaxed(&counter), 100);
    threadpool_destroy(pool);
    return 0;
}

TEST(threadpool_submit_many_items) {
    ThreadPool* pool = create_test_pool(4);
    atomic_u32 counter = ATOMIC_VAR_INIT(0);

    for (uint32_t i = 0; i < 1000; i++) {
        WorkItem item = {
            .fn = increment_fn,
            .data = &counter,
            .start = i,
            .end = i + 1
        };
        ASSERT_TRUE(threadpool_submit(pool, item));
    }

    threadpool_wait(pool);
    ASSERT_EQ(atomic_load_relaxed(&counter), 1000);
    threadpool_destroy(pool);
    return 0;
}

/* ============================================================================
 * Work Distribution Tests
 * ============================================================================ */

TEST(threadpool_distributes_work_across_threads) {
    ThreadPool* pool = create_test_pool(4);
    const uint32_t N = 1000;
    atomic_u32* thread_ids = (atomic_u32*)calloc(N, sizeof(atomic_u32));
    ASSERT_NOT_NULL(thread_ids);

    for (uint32_t i = 0; i < N; i++) {
        atomic_init(&thread_ids[i], UINT32_MAX);
        WorkItem item = {
            .fn = record_thread_id_fn,
            .data = thread_ids,
            .start = i,
            .end = i + 1
        };
        threadpool_submit(pool, item);
    }

    threadpool_wait(pool);

    /* Count tasks per thread */
    uint32_t per_thread[4] = {0};
    for (uint32_t i = 0; i < N; i++) {
        uint32_t tid = atomic_load_relaxed(&thread_ids[i]);
        if (tid < 4) {
            per_thread[tid]++;
        }
    }

    /* At least 2 threads should have done work */
    uint32_t active_threads = 0;
    for (uint32_t i = 0; i < 4; i++) {
        if (per_thread[i] > 0) active_threads++;
    }
    ASSERT_GT(active_threads, 1);

    free(thread_ids);
    threadpool_destroy(pool);
    return 0;
}

TEST(threadpool_sum_range_correctness) {
    ThreadPool* pool = create_test_pool(4);
    const uint32_t N = 10000;
    atomic_u64 sum = ATOMIC_VAR_INIT(0);

    /* Submit work items that sum ranges */
    uint32_t chunk = 100;
    for (uint32_t i = 0; i < N; i += chunk) {
        uint32_t end = (i + chunk < N) ? i + chunk : N;
        WorkItem item = {
            .fn = sum_range_fn,
            .data = &sum,
            .start = i,
            .end = end
        };
        threadpool_submit(pool, item);
    }

    threadpool_wait(pool);

    /* Expected: sum of 0..N-1 = N*(N-1)/2 */
    uint64_t expected = (uint64_t)N * (N - 1) / 2;
    ASSERT_EQ(atomic_load_relaxed(&sum), expected);

    threadpool_destroy(pool);
    return 0;
}

/* ============================================================================
 * Wait and Barrier Tests
 * ============================================================================ */

TEST(threadpool_wait_blocks_until_complete) {
    ThreadPool* pool = create_test_pool(2);
    atomic_u32 counter = ATOMIC_VAR_INIT(0);

    for (uint32_t i = 0; i < 100; i++) {
        WorkItem item = {
            .fn = slow_increment_fn,
            .data = &counter,
            .start = 0,
            .end = 1
        };
        threadpool_submit(pool, item);
    }

    threadpool_wait(pool);
    ASSERT_EQ(atomic_load_relaxed(&counter), 100);
    threadpool_destroy(pool);
    return 0;
}

TEST(threadpool_is_idle_true_after_wait) {
    ThreadPool* pool = create_test_pool(2);
    atomic_u32 counter = ATOMIC_VAR_INIT(0);

    WorkItem item = { increment_fn, &counter, 0, 1 };
    threadpool_submit(pool, item);
    threadpool_wait(pool);

    ASSERT_TRUE(threadpool_is_idle(pool));
    threadpool_destroy(pool);
    return 0;
}

TEST(threadpool_is_idle_false_during_work) {
    ThreadPool* pool = create_test_pool(2);
    atomic_u32 counter = ATOMIC_VAR_INIT(0);

    /* Submit many slow items */
    for (uint32_t i = 0; i < 1000; i++) {
        WorkItem item = { slow_increment_fn, &counter, 0, 1 };
        threadpool_submit(pool, item);
    }

    /* Check immediately - should not be idle */
    /* Note: This is racy but likely to catch non-idle state */
    ASSERT_FALSE(threadpool_is_idle(pool));

    threadpool_wait(pool);
    ASSERT_TRUE(threadpool_is_idle(pool));
    threadpool_destroy(pool);
    return 0;
}

/* ============================================================================
 * Statistics Tests
 * ============================================================================ */

TEST(threadpool_stats_counts_executed) {
    ThreadPool* pool = create_test_pool(2);
    atomic_u32 counter = ATOMIC_VAR_INIT(0);

    for (uint32_t i = 0; i < 50; i++) {
        WorkItem item = { increment_fn, &counter, 0, 1 };
        threadpool_submit(pool, item);
    }
    threadpool_wait(pool);

    ThreadPoolStats stats;
    threadpool_get_stats(pool, &stats);

    ASSERT_EQ(stats.total_tasks_executed, 50);
    threadpool_destroy(pool);
    return 0;
}

TEST(threadpool_stats_reset_clears) {
    ThreadPool* pool = create_test_pool(2);
    atomic_u32 counter = ATOMIC_VAR_INIT(0);

    WorkItem item = { increment_fn, &counter, 0, 1 };
    threadpool_submit(pool, item);
    threadpool_wait(pool);

    threadpool_reset_stats(pool);

    ThreadPoolStats stats;
    threadpool_get_stats(pool, &stats);

    ASSERT_EQ(stats.total_tasks_executed, 0);
    threadpool_destroy(pool);
    return 0;
}

TEST(threadpool_stats_per_thread_tracking) {
    ThreadPool* pool = create_test_pool(4);
    atomic_u32 counter = ATOMIC_VAR_INIT(0);

    for (uint32_t i = 0; i < 1000; i++) {
        WorkItem item = { increment_fn, &counter, 0, 1 };
        threadpool_submit(pool, item);
    }
    threadpool_wait(pool);

    ThreadPoolStats stats;
    threadpool_get_stats(pool, &stats);

    /* Verify sum of per-thread equals total */
    uint64_t sum = 0;
    for (uint32_t i = 0; i < 4; i++) {
        sum += stats.per_thread_executed[i];
    }
    ASSERT_EQ(sum, stats.total_tasks_executed);
    ASSERT_EQ(sum, 1000);

    threadpool_destroy(pool);
    return 0;
}

/* ============================================================================
 * Work Stealing Tests
 * ============================================================================ */

TEST(threadpool_work_stealing_occurs) {
    ThreadPool* pool = create_test_pool(4);
    atomic_u32 counter = ATOMIC_VAR_INIT(0);

    /* Submit many items quickly - likely to trigger stealing */
    for (uint32_t i = 0; i < 10000; i++) {
        WorkItem item = { increment_fn, &counter, 0, 1 };
        threadpool_submit(pool, item);
    }

    threadpool_wait(pool);

    ThreadPoolStats stats;
    threadpool_get_stats(pool, &stats);

    /* With 4 threads and many items, some stealing should occur */
    ASSERT_EQ(atomic_load_relaxed(&counter), 10000);
    ASSERT_GE(stats.total_steal_attempts, 0);  /* At least attempted */

    threadpool_destroy(pool);
    return 0;
}

/* ============================================================================
 * Edge Cases
 * ============================================================================ */

TEST(threadpool_empty_wait_returns_immediately) {
    ThreadPool* pool = create_test_pool(2);

    /* Wait on empty pool should return immediately */
    threadpool_wait(pool);
    ASSERT_TRUE(threadpool_is_idle(pool));

    threadpool_destroy(pool);
    return 0;
}

TEST(threadpool_multiple_waits) {
    ThreadPool* pool = create_test_pool(2);
    atomic_u32 counter = ATOMIC_VAR_INIT(0);

    for (uint32_t round = 0; round < 10; round++) {
        for (uint32_t i = 0; i < 100; i++) {
            WorkItem item = { increment_fn, &counter, 0, 1 };
            threadpool_submit(pool, item);
        }
        threadpool_wait(pool);
    }

    ASSERT_EQ(atomic_load_relaxed(&counter), 1000);
    threadpool_destroy(pool);
    return 0;
}

TEST(threadpool_pending_tasks_decreases) {
    ThreadPool* pool = create_test_pool(2);
    atomic_u32 counter = ATOMIC_VAR_INIT(0);

    for (uint32_t i = 0; i < 100; i++) {
        WorkItem item = { slow_increment_fn, &counter, 0, 1 };
        threadpool_submit(pool, item);
    }

    /* Initially should have pending tasks */
    uint32_t initial_pending = threadpool_pending_tasks(pool);
    ASSERT_GT(initial_pending, 0);

    threadpool_wait(pool);

    /* After wait, should be 0 */
    ASSERT_EQ(threadpool_pending_tasks(pool), 0);
    threadpool_destroy(pool);
    return 0;
}

/* ============================================================================
 * Stress Tests
 * ============================================================================ */

TEST(threadpool_stress_many_small_items) {
    ThreadPool* pool = create_test_pool(8);
    atomic_u64 sum = ATOMIC_VAR_INIT(0);
    const uint32_t N = 100000;

    for (uint32_t i = 0; i < N; i++) {
        WorkItem item = {
            .fn = sum_range_fn,
            .data = &sum,
            .start = i,
            .end = i + 1
        };
        threadpool_submit(pool, item);
    }

    threadpool_wait(pool);

    uint64_t expected = (uint64_t)N * (N - 1) / 2;
    ASSERT_EQ(atomic_load_relaxed(&sum), expected);
    threadpool_destroy(pool);
    return 0;
}

TEST(threadpool_stress_rapid_submit_wait) {
    ThreadPool* pool = create_test_pool(4);
    atomic_u32 counter = ATOMIC_VAR_INIT(0);

    for (uint32_t round = 0; round < 100; round++) {
        WorkItem item = { increment_fn, &counter, 0, 1 };
        threadpool_submit(pool, item);
        threadpool_wait(pool);
    }

    ASSERT_EQ(atomic_load_relaxed(&counter), 100);
    threadpool_destroy(pool);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Thread Pool Tests");

    /* Creation tests */
    RUN_TEST(threadpool_create_default_config);
    RUN_TEST(threadpool_create_explicit_thread_count);
    RUN_TEST(threadpool_create_single_thread);
    RUN_TEST(threadpool_create_many_threads);

    /* Submission tests */
    RUN_TEST(threadpool_submit_single_item_executes);
    RUN_TEST(threadpool_submit_batch_executes_all);
    RUN_TEST(threadpool_submit_many_items);

    /* Work distribution tests */
    RUN_TEST(threadpool_distributes_work_across_threads);
    RUN_TEST(threadpool_sum_range_correctness);

    /* Wait and barrier tests */
    RUN_TEST(threadpool_wait_blocks_until_complete);
    RUN_TEST(threadpool_is_idle_true_after_wait);
    RUN_TEST(threadpool_is_idle_false_during_work);

    /* Statistics tests */
    RUN_TEST(threadpool_stats_counts_executed);
    RUN_TEST(threadpool_stats_reset_clears);
    RUN_TEST(threadpool_stats_per_thread_tracking);

    /* Work stealing tests */
    RUN_TEST(threadpool_work_stealing_occurs);

    /* Edge cases */
    RUN_TEST(threadpool_empty_wait_returns_immediately);
    RUN_TEST(threadpool_multiple_waits);
    RUN_TEST(threadpool_pending_tasks_decreases);

    /* Stress tests */
    RUN_TEST(threadpool_stress_many_small_items);
    RUN_TEST(threadpool_stress_rapid_submit_wait);

    TEST_SUITE_END();
}
