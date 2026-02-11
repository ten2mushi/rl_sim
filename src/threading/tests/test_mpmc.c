/**
 * Test Suite for MPMC Queue (Multi-Producer Multi-Consumer)
 *
 * Tests verify:
 * - Creation and destruction
 * - FIFO ordering (single thread)
 * - Multiple producers no loss
 * - Multiple consumers no duplicates
 * - Concurrent producer/consumer correctness
 * - High contention stress test
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

static MPMCQueue* create_test_mpmc(uint32_t capacity) {
    MPMCQueue* q = mpmc_create(capacity);
    assert(q != NULL);
    return q;
}

/* ============================================================================
 * Creation Tests
 * ============================================================================ */

TEST(mpmc_create_returns_non_null) {
    MPMCQueue* q = mpmc_create(64);
    ASSERT_NOT_NULL(q);
    mpmc_destroy(q);
    return 0;
}

TEST(mpmc_create_initializes_empty) {
    MPMCQueue* q = create_test_mpmc(64);
    ASSERT_TRUE(mpmc_empty(q));
    ASSERT_EQ(mpmc_size(q), 0);
    mpmc_destroy(q);
    return 0;
}

/* ============================================================================
 * Single-Thread Tests
 * ============================================================================ */

TEST(mpmc_push_pop_single_item) {
    MPMCQueue* q = create_test_mpmc(64);
    void* item = (void*)0xDEADBEEF;

    ASSERT_TRUE(mpmc_push(q, item));

    void* out;
    ASSERT_TRUE(mpmc_pop(q, &out));
    ASSERT_EQ(out, item);

    mpmc_destroy(q);
    return 0;
}

TEST(mpmc_fifo_ordering_single_thread) {
    MPMCQueue* q = create_test_mpmc(64);

    for (uintptr_t i = 0; i < 50; i++) {
        ASSERT_TRUE(mpmc_push(q, (void*)i));
    }

    for (uintptr_t i = 0; i < 50; i++) {
        void* out;
        ASSERT_TRUE(mpmc_pop(q, &out));
        ASSERT_EQ((uintptr_t)out, i);
    }

    mpmc_destroy(q);
    return 0;
}

TEST(mpmc_push_to_capacity) {
    MPMCQueue* q = create_test_mpmc(64);

    for (uint32_t i = 0; i < 64; i++) {
        ASSERT_TRUE(mpmc_push(q, (void*)(uintptr_t)i));
    }

    ASSERT_EQ(mpmc_size(q), 64);
    /* Queue full */
    ASSERT_FALSE(mpmc_push(q, (void*)999));

    mpmc_destroy(q);
    return 0;
}

TEST(mpmc_pop_from_empty) {
    MPMCQueue* q = create_test_mpmc(64);
    void* out;
    ASSERT_FALSE(mpmc_pop(q, &out));
    mpmc_destroy(q);
    return 0;
}

/* ============================================================================
 * Multi-Producer Tests
 * ============================================================================ */

typedef struct {
    MPMCQueue* queue;
    uint32_t start;
    uint32_t count;
    atomic_u64* sum;
} ProducerArgs;

static void* mpmc_producer_thread(void* arg) {
    ProducerArgs* args = (ProducerArgs*)arg;

    for (uint32_t i = 0; i < args->count; i++) {
        uintptr_t val = args->start + i;
        while (!mpmc_push(args->queue, (void*)val)) {
            threading_yield();
        }
        atomic_add_relaxed(args->sum, val);
    }

    return NULL;
}

TEST(mpmc_multiple_producers_no_loss) {
    const uint32_t N_PRODUCERS = 4;
    const uint32_t ITEMS_PER = 10000;
    /* Queue must be large enough to hold all items (no consumers during push phase) */
    MPMCQueue* q = create_test_mpmc(N_PRODUCERS * ITEMS_PER);

    atomic_u64 produced_sum = ATOMIC_VAR_INIT(0);
    pthread_t producers[N_PRODUCERS];
    ProducerArgs args[N_PRODUCERS];

    for (uint32_t p = 0; p < N_PRODUCERS; p++) {
        args[p] = (ProducerArgs){ q, p * ITEMS_PER, ITEMS_PER, &produced_sum };
        pthread_create(&producers[p], NULL, mpmc_producer_thread, &args[p]);
    }

    for (uint32_t p = 0; p < N_PRODUCERS; p++) {
        pthread_join(producers[p], NULL);
    }

    /* Verify all items in queue */
    ASSERT_EQ(mpmc_size(q), N_PRODUCERS * ITEMS_PER);

    /* Pop all and verify sum */
    uint64_t consumed_sum = 0;
    void* item;
    while (mpmc_pop(q, &item)) {
        consumed_sum += (uintptr_t)item;
    }

    ASSERT_EQ(consumed_sum, atomic_load_relaxed(&produced_sum));
    mpmc_destroy(q);
    return 0;
}

/* ============================================================================
 * Multi-Consumer Tests
 * ============================================================================ */

typedef struct {
    MPMCQueue* queue;
    atomic_u64* sum;
    atomic_u32* remaining;
} ConsumerArgs;

static void* mpmc_consumer_thread(void* arg) {
    ConsumerArgs* args = (ConsumerArgs*)arg;
    void* item;

    while (atomic_load_relaxed(args->remaining) > 0 || !mpmc_empty(args->queue)) {
        if (mpmc_pop(args->queue, &item)) {
            atomic_add_relaxed(args->sum, (uintptr_t)item);
            atomic_sub_relaxed(args->remaining, 1);
        } else {
            threading_yield();
        }
    }

    return NULL;
}

TEST(mpmc_multiple_consumers_no_duplicates) {
    const uint32_t N = 10000;
    /* Queue must hold all N items for pre-fill */
    MPMCQueue* q = create_test_mpmc(N);

    /* Pre-fill queue */
    uint64_t expected_sum = 0;
    for (uint32_t i = 0; i < N; i++) {
        ASSERT_TRUE(mpmc_push(q, (void*)(uintptr_t)i));
        expected_sum += i;
    }

    const uint32_t N_CONSUMERS = 4;
    atomic_u64 consumed_sum = ATOMIC_VAR_INIT(0);
    atomic_u32 remaining = ATOMIC_VAR_INIT(N);
    pthread_t consumers[N_CONSUMERS];
    ConsumerArgs args = { q, &consumed_sum, &remaining };

    for (uint32_t c = 0; c < N_CONSUMERS; c++) {
        pthread_create(&consumers[c], NULL, mpmc_consumer_thread, &args);
    }

    for (uint32_t c = 0; c < N_CONSUMERS; c++) {
        pthread_join(consumers[c], NULL);
    }

    ASSERT_EQ(atomic_load_relaxed(&consumed_sum), expected_sum);
    ASSERT_TRUE(mpmc_empty(q));
    mpmc_destroy(q);
    return 0;
}

/* ============================================================================
 * Multi-Producer Multi-Consumer Tests
 * ============================================================================ */

typedef struct {
    MPMCQueue* queue;
    atomic_u64* produced_sum;
    atomic_u32* producers_done;
    uint32_t start;
    uint32_t count;
} MPMCProducerArgs;

typedef struct {
    MPMCQueue* queue;
    atomic_u64* consumed_sum;
    atomic_u32* producers_done;
    uint32_t num_producers;
} MPMCConsumerArgs;

static void* mpmc_sum_producer(void* arg) {
    MPMCProducerArgs* args = (MPMCProducerArgs*)arg;

    for (uint32_t i = 0; i < args->count; i++) {
        uintptr_t val = args->start + i;
        while (!mpmc_push(args->queue, (void*)val)) {
            threading_yield();
        }
        atomic_add_relaxed(args->produced_sum, val);
    }

    atomic_add_relaxed(args->producers_done, 1);
    return NULL;
}

static void* mpmc_sum_consumer(void* arg) {
    MPMCConsumerArgs* args = (MPMCConsumerArgs*)arg;
    void* item;

    while (atomic_load_relaxed(args->producers_done) < args->num_producers ||
           !mpmc_empty(args->queue)) {
        if (mpmc_pop(args->queue, &item)) {
            atomic_add_relaxed(args->consumed_sum, (uintptr_t)item);
        } else {
            threading_yield();
        }
    }

    /* Drain remaining */
    while (mpmc_pop(args->queue, &item)) {
        atomic_add_relaxed(args->consumed_sum, (uintptr_t)item);
    }

    return NULL;
}

TEST(mpmc_concurrent_producers_consumers) {
    MPMCQueue* q = create_test_mpmc(2048);
    const uint32_t N_PRODUCERS = 4;
    const uint32_t N_CONSUMERS = 4;
    const uint32_t ITEMS_PER = 10000;

    atomic_u64 produced_sum = ATOMIC_VAR_INIT(0);
    atomic_u64 consumed_sum = ATOMIC_VAR_INIT(0);
    atomic_u32 producers_done = ATOMIC_VAR_INIT(0);

    pthread_t producers[N_PRODUCERS];
    pthread_t consumers[N_CONSUMERS];
    MPMCProducerArgs prod_args[N_PRODUCERS];
    MPMCConsumerArgs cons_args = { q, &consumed_sum, &producers_done, N_PRODUCERS };

    /* Start consumers first */
    for (uint32_t c = 0; c < N_CONSUMERS; c++) {
        pthread_create(&consumers[c], NULL, mpmc_sum_consumer, &cons_args);
    }

    /* Start producers */
    for (uint32_t p = 0; p < N_PRODUCERS; p++) {
        prod_args[p] = (MPMCProducerArgs){
            q, &produced_sum, &producers_done, p * ITEMS_PER, ITEMS_PER
        };
        pthread_create(&producers[p], NULL, mpmc_sum_producer, &prod_args[p]);
    }

    /* Wait for producers */
    for (uint32_t p = 0; p < N_PRODUCERS; p++) {
        pthread_join(producers[p], NULL);
    }

    /* Wait for consumers */
    for (uint32_t c = 0; c < N_CONSUMERS; c++) {
        pthread_join(consumers[c], NULL);
    }

    ASSERT_EQ(atomic_load_relaxed(&consumed_sum), atomic_load_relaxed(&produced_sum));
    mpmc_destroy(q);
    return 0;
}

/* ============================================================================
 * Stress Tests
 * ============================================================================ */

typedef struct {
    MPMCQueue* queue;
    atomic_u64* total_pushed;
    atomic_u64* total_popped;
    uint32_t ops;
} PushPopArgs;

static void* mpmc_push_pop_thread(void* arg) {
    PushPopArgs* args = (PushPopArgs*)arg;
    PCG32 rng;
    pcg32_seed(&rng, (uintptr_t)pthread_self());

    for (uint32_t i = 0; i < args->ops; i++) {
        uintptr_t val = pcg32_bounded(&rng, 10000);

        if (pcg32_bounded(&rng, 2) == 0) {
            /* Push */
            if (mpmc_push(args->queue, (void*)val)) {
                atomic_add_relaxed(args->total_pushed, val);
            }
        } else {
            /* Pop */
            void* item;
            if (mpmc_pop(args->queue, &item)) {
                atomic_add_relaxed(args->total_popped, (uintptr_t)item);
            }
        }
    }

    return NULL;
}

TEST(mpmc_stress_high_contention) {
    MPMCQueue* q = create_test_mpmc(256);  /* Small queue = high contention */
    const uint32_t N_THREADS = 8;
    const uint32_t OPS_PER = 50000;

    atomic_u64 total_pushed = ATOMIC_VAR_INIT(0);
    atomic_u64 total_popped = ATOMIC_VAR_INIT(0);

    pthread_t threads[N_THREADS];
    PushPopArgs args = { q, &total_pushed, &total_popped, OPS_PER };

    for (uint32_t t = 0; t < N_THREADS; t++) {
        pthread_create(&threads[t], NULL, mpmc_push_pop_thread, &args);
    }

    for (uint32_t t = 0; t < N_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

    /* Drain remaining items */
    void* item;
    while (mpmc_pop(q, &item)) {
        atomic_add_relaxed(&total_popped, (uintptr_t)item);
    }

    ASSERT_EQ(atomic_load_relaxed(&total_pushed), atomic_load_relaxed(&total_popped));
    mpmc_destroy(q);
    return 0;
}

TEST(mpmc_rapid_fill_drain) {
    MPMCQueue* q = create_test_mpmc(1024);
    const uint32_t ITERATIONS = 100;

    for (uint32_t iter = 0; iter < ITERATIONS; iter++) {
        /* Fill */
        for (uint32_t i = 0; i < 1024; i++) {
            ASSERT_TRUE(mpmc_push(q, (void*)(uintptr_t)i));
        }
        ASSERT_EQ(mpmc_size(q), 1024);

        /* Drain */
        void* item;
        for (uint32_t i = 0; i < 1024; i++) {
            ASSERT_TRUE(mpmc_pop(q, &item));
        }
        ASSERT_TRUE(mpmc_empty(q));
    }

    mpmc_destroy(q);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("MPMC Queue Tests");

    /* Creation tests */
    RUN_TEST(mpmc_create_returns_non_null);
    RUN_TEST(mpmc_create_initializes_empty);

    /* Single-thread tests */
    RUN_TEST(mpmc_push_pop_single_item);
    RUN_TEST(mpmc_fifo_ordering_single_thread);
    RUN_TEST(mpmc_push_to_capacity);
    RUN_TEST(mpmc_pop_from_empty);

    /* Multi-producer tests */
    RUN_TEST(mpmc_multiple_producers_no_loss);

    /* Multi-consumer tests */
    RUN_TEST(mpmc_multiple_consumers_no_duplicates);

    /* Multi-producer multi-consumer tests */
    RUN_TEST(mpmc_concurrent_producers_consumers);

    /* Stress tests */
    RUN_TEST(mpmc_stress_high_contention);
    RUN_TEST(mpmc_rapid_fill_drain);

    TEST_SUITE_END();
}
