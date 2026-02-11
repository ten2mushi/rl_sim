/**
 * Test Suite for SPSC Queue (Single-Producer Single-Consumer)
 *
 * Tests verify:
 * - Creation and destruction
 * - FIFO ordering
 * - Capacity limits
 * - Producer-consumer correctness
 * - Wrap-around handling
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

static SPSCQueue* create_test_spsc(uint32_t capacity) {
    SPSCQueue* q = spsc_create(capacity);
    assert(q != NULL);
    return q;
}

/* ============================================================================
 * Creation Tests
 * ============================================================================ */

TEST(spsc_create_returns_non_null) {
    SPSCQueue* q = spsc_create(64);
    ASSERT_NOT_NULL(q);
    spsc_destroy(q);
    return 0;
}

TEST(spsc_create_initializes_empty) {
    SPSCQueue* q = create_test_spsc(64);
    ASSERT_TRUE(spsc_empty(q));
    ASSERT_EQ(spsc_size(q), 0);
    spsc_destroy(q);
    return 0;
}

TEST(spsc_create_rounds_up_capacity) {
    /* Non-power-of-2 capacity should be rounded up */
    SPSCQueue* q = spsc_create(100);
    ASSERT_NOT_NULL(q);
    /* Capacity should be 128 (next power of 2 after 100) */
    ASSERT_EQ(q->capacity, 128);
    spsc_destroy(q);
    return 0;
}

/* ============================================================================
 * Single-Thread Tests
 * ============================================================================ */

TEST(spsc_push_pop_single_item) {
    SPSCQueue* q = create_test_spsc(64);
    void* item = (void*)0x12345678;

    ASSERT_TRUE(spsc_push(q, item));
    ASSERT_EQ(spsc_size(q), 1);

    void* out;
    ASSERT_TRUE(spsc_pop(q, &out));
    ASSERT_EQ(out, item);
    ASSERT_TRUE(spsc_empty(q));

    spsc_destroy(q);
    return 0;
}

TEST(spsc_fifo_ordering) {
    SPSCQueue* q = create_test_spsc(64);

    /* Push 0, 1, 2, ..., 49 */
    for (uintptr_t i = 0; i < 50; i++) {
        ASSERT_TRUE(spsc_push(q, (void*)i));
    }

    /* Pop should return in same order (FIFO) */
    for (uintptr_t i = 0; i < 50; i++) {
        void* out;
        ASSERT_TRUE(spsc_pop(q, &out));
        ASSERT_EQ((uintptr_t)out, i);
    }

    spsc_destroy(q);
    return 0;
}

TEST(spsc_push_to_capacity) {
    SPSCQueue* q = create_test_spsc(64);

    for (uint32_t i = 0; i < 64; i++) {
        ASSERT_TRUE(spsc_push(q, (void*)(uintptr_t)i));
    }

    ASSERT_EQ(spsc_size(q), 64);
    /* Next push should fail */
    ASSERT_FALSE(spsc_push(q, (void*)999));

    spsc_destroy(q);
    return 0;
}

TEST(spsc_pop_from_empty) {
    SPSCQueue* q = create_test_spsc(64);
    void* out;
    ASSERT_FALSE(spsc_pop(q, &out));
    spsc_destroy(q);
    return 0;
}

TEST(spsc_wrap_around_correct) {
    SPSCQueue* q = create_test_spsc(64);
    const uint32_t ITERATIONS = 64 * 10;

    for (uint32_t i = 0; i < ITERATIONS; i++) {
        ASSERT_TRUE(spsc_push(q, (void*)(uintptr_t)i));

        void* out;
        ASSERT_TRUE(spsc_pop(q, &out));
        ASSERT_EQ((uintptr_t)out, i);
    }

    ASSERT_TRUE(spsc_empty(q));
    spsc_destroy(q);
    return 0;
}

/* ============================================================================
 * Multi-Thread Tests
 * ============================================================================ */

typedef struct {
    SPSCQueue* queue;
    atomic_u64* sum;
    atomic_flag_t* producer_done;
    uint32_t count;
} ConsumerArgs;

static void* spsc_consumer_thread(void* arg) {
    ConsumerArgs* args = (ConsumerArgs*)arg;
    void* item;

    while (!atomic_load_acquire(args->producer_done) || !spsc_empty(args->queue)) {
        if (spsc_pop(args->queue, &item)) {
            atomic_add_relaxed(args->sum, (uintptr_t)item);
        } else {
            threading_yield();
        }
    }

    /* Drain any remaining */
    while (spsc_pop(args->queue, &item)) {
        atomic_add_relaxed(args->sum, (uintptr_t)item);
    }

    return NULL;
}

TEST(spsc_producer_consumer_no_loss) {
    SPSCQueue* q = create_test_spsc(512);
    const uint32_t N = 100000;
    atomic_u64 consumer_sum = ATOMIC_VAR_INIT(0);
    atomic_flag_t producer_done = ATOMIC_VAR_INIT(false);

    ConsumerArgs args = { q, &consumer_sum, &producer_done, N };
    pthread_t consumer;
    pthread_create(&consumer, NULL, spsc_consumer_thread, &args);

    /* Producer: push all items */
    uint64_t producer_sum = 0;
    for (uint32_t i = 0; i < N; i++) {
        while (!spsc_push(q, (void*)(uintptr_t)i)) {
            threading_yield();
        }
        producer_sum += i;
    }

    atomic_store_release(&producer_done, true);
    pthread_join(consumer, NULL);

    ASSERT_EQ(atomic_load_relaxed(&consumer_sum), producer_sum);
    spsc_destroy(q);
    return 0;
}

typedef struct {
    SPSCQueue* queue;
    atomic_u64* count;
    atomic_flag_t* done;
} CountArgs;

static void* count_consumer_thread(void* arg) {
    CountArgs* a = (CountArgs*)arg;
    void* item;
    while (!atomic_load_acquire(a->done) || !spsc_empty(a->queue)) {
        if (spsc_pop(a->queue, &item)) {
            atomic_add_relaxed(a->count, 1);
        }
    }
    while (spsc_pop(a->queue, &item)) {
        atomic_add_relaxed(a->count, 1);
    }
    return NULL;
}

TEST(spsc_high_throughput) {
    SPSCQueue* q = create_test_spsc(1024);
    const uint32_t N = 1000000;
    atomic_u64 consumer_count = ATOMIC_VAR_INIT(0);
    atomic_flag_t producer_done = ATOMIC_VAR_INIT(false);

    CountArgs args = { q, &consumer_count, &producer_done };
    pthread_t consumer;
    pthread_create(&consumer, NULL, count_consumer_thread, &args);

    for (uint32_t i = 0; i < N; i++) {
        while (!spsc_push(q, (void*)(uintptr_t)i)) {
            /* Busy wait - queue full */
        }
    }

    atomic_store_release(&producer_done, true);
    pthread_join(consumer, NULL);

    ASSERT_EQ(atomic_load_relaxed(&consumer_count), N);
    spsc_destroy(q);
    return 0;
}

/* ============================================================================
 * Edge Cases
 * ============================================================================ */

TEST(spsc_alternating_push_pop) {
    SPSCQueue* q = create_test_spsc(64);

    for (uint32_t round = 0; round < 1000; round++) {
        /* Push some items */
        uint32_t push_count = (round % 10) + 1;
        for (uint32_t i = 0; i < push_count; i++) {
            ASSERT_TRUE(spsc_push(q, (void*)(uintptr_t)(round * 100 + i)));
        }

        /* Pop all */
        void* out;
        uint32_t pop_count = 0;
        while (spsc_pop(q, &out)) {
            pop_count++;
        }
        ASSERT_EQ(pop_count, push_count);
    }

    spsc_destroy(q);
    return 0;
}

TEST(spsc_null_items_allowed) {
    SPSCQueue* q = create_test_spsc(64);

    /* NULL is a valid item */
    ASSERT_TRUE(spsc_push(q, NULL));
    ASSERT_TRUE(spsc_push(q, (void*)1));
    ASSERT_TRUE(spsc_push(q, NULL));

    void* out;
    ASSERT_TRUE(spsc_pop(q, &out));
    ASSERT_EQ(out, NULL);
    ASSERT_TRUE(spsc_pop(q, &out));
    ASSERT_EQ(out, (void*)1);
    ASSERT_TRUE(spsc_pop(q, &out));
    ASSERT_EQ(out, NULL);

    spsc_destroy(q);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("SPSC Queue Tests");

    /* Creation tests */
    RUN_TEST(spsc_create_returns_non_null);
    RUN_TEST(spsc_create_initializes_empty);
    RUN_TEST(spsc_create_rounds_up_capacity);

    /* Single-thread tests */
    RUN_TEST(spsc_push_pop_single_item);
    RUN_TEST(spsc_fifo_ordering);
    RUN_TEST(spsc_push_to_capacity);
    RUN_TEST(spsc_pop_from_empty);
    RUN_TEST(spsc_wrap_around_correct);

    /* Multi-thread tests */
    RUN_TEST(spsc_producer_consumer_no_loss);
    RUN_TEST(spsc_high_throughput);

    /* Edge cases */
    RUN_TEST(spsc_alternating_push_pop);
    RUN_TEST(spsc_null_items_allowed);

    TEST_SUITE_END();
}
