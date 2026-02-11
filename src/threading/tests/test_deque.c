/**
 * Test Suite for Chase-Lev Work-Stealing Deque
 *
 * Tests verify:
 * - Initialization
 * - Owner push/pop operations (LIFO ordering)
 * - Thief steal operations (FIFO ordering)
 * - Concurrent owner and thief operations
 * - Multiple thief contention
 * - Boundary conditions (empty, full)
 */

#include "threading.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include "test_harness.h"

/* ============================================================================
 * Test Helpers
 * ============================================================================ */

static void dummy_fn(void* data, uint32_t start, uint32_t end, uint32_t thread_id) {
    (void)data;
    (void)start;
    (void)end;
    (void)thread_id;
}

static WorkItem make_test_item(uint32_t id, uint32_t start, uint32_t end) {
    return (WorkItem){
        .fn = dummy_fn,
        .data = (void*)(uintptr_t)id,
        .start = start,
        .end = end
    };
}

static WorkDeque* create_test_deque(void) {
    WorkDeque* dq = (WorkDeque*)aligned_alloc(64, sizeof(WorkDeque));
    assert(dq != NULL);
    deque_init(dq);
    return dq;
}

static void destroy_test_deque(WorkDeque* dq) {
    free(dq);
}

/* ============================================================================
 * Initialization Tests
 * ============================================================================ */

TEST(deque_init_initializes_empty) {
    WorkDeque* dq = create_test_deque();
    ASSERT_TRUE(deque_empty(dq));
    ASSERT_EQ(deque_size(dq), 0);
    destroy_test_deque(dq);
    return 0;
}

TEST(deque_init_sets_top_and_bottom_to_zero) {
    WorkDeque* dq = create_test_deque();
    ASSERT_EQ(atomic_load_relaxed(&dq->top), 0);
    ASSERT_EQ(atomic_load_relaxed(&dq->bottom), 0);
    destroy_test_deque(dq);
    return 0;
}

/* ============================================================================
 * Owner Push Tests
 * ============================================================================ */

TEST(deque_push_single_item_succeeds) {
    WorkDeque* dq = create_test_deque();
    WorkItem item = make_test_item(42, 0, 100);
    ASSERT_TRUE(deque_push(dq, item));
    ASSERT_EQ(deque_size(dq), 1);
    destroy_test_deque(dq);
    return 0;
}

TEST(deque_push_multiple_items) {
    WorkDeque* dq = create_test_deque();
    for (uint32_t i = 0; i < 100; i++) {
        ASSERT_TRUE(deque_push(dq, make_test_item(i, i, i + 1)));
    }
    ASSERT_EQ(deque_size(dq), 100);
    destroy_test_deque(dq);
    return 0;
}

TEST(deque_push_to_capacity_succeeds) {
    WorkDeque* dq = create_test_deque();
    for (uint32_t i = 0; i < THREADING_DEQUE_CAPACITY; i++) {
        ASSERT_TRUE(deque_push(dq, make_test_item(i, 0, 0)));
    }
    ASSERT_EQ(deque_size(dq), THREADING_DEQUE_CAPACITY);
    destroy_test_deque(dq);
    return 0;
}

TEST(deque_push_beyond_capacity_fails) {
    WorkDeque* dq = create_test_deque();
    for (uint32_t i = 0; i < THREADING_DEQUE_CAPACITY; i++) {
        ASSERT_TRUE(deque_push(dq, make_test_item(i, 0, 0)));
    }
    ASSERT_FALSE(deque_push(dq, make_test_item(999, 0, 0)));
    destroy_test_deque(dq);
    return 0;
}

/* ============================================================================
 * Owner Pop Tests (LIFO)
 * ============================================================================ */

TEST(deque_pop_returns_last_pushed) {
    WorkDeque* dq = create_test_deque();
    deque_push(dq, make_test_item(1, 0, 10));
    deque_push(dq, make_test_item(2, 10, 20));
    deque_push(dq, make_test_item(3, 20, 30));

    WorkItem out;
    ASSERT_TRUE(deque_pop(dq, &out));
    ASSERT_EQ(out.start, 20);  /* Last item first (LIFO) */
    ASSERT_TRUE(deque_pop(dq, &out));
    ASSERT_EQ(out.start, 10);
    ASSERT_TRUE(deque_pop(dq, &out));
    ASSERT_EQ(out.start, 0);

    destroy_test_deque(dq);
    return 0;
}

TEST(deque_pop_from_empty_returns_false) {
    WorkDeque* dq = create_test_deque();
    WorkItem out;
    ASSERT_FALSE(deque_pop(dq, &out));
    destroy_test_deque(dq);
    return 0;
}

TEST(deque_pop_all_items_leaves_empty) {
    WorkDeque* dq = create_test_deque();
    for (uint32_t i = 0; i < 10; i++) {
        ASSERT_TRUE(deque_push(dq, make_test_item(i, i, i + 1)));
    }
    WorkItem out;
    for (uint32_t i = 0; i < 10; i++) {
        ASSERT_TRUE(deque_pop(dq, &out));
    }
    ASSERT_TRUE(deque_empty(dq));
    ASSERT_FALSE(deque_pop(dq, &out));
    destroy_test_deque(dq);
    return 0;
}

/* ============================================================================
 * Steal Tests (FIFO)
 * ============================================================================ */

TEST(deque_steal_returns_first_pushed) {
    WorkDeque* dq = create_test_deque();
    deque_push(dq, make_test_item(1, 0, 10));
    deque_push(dq, make_test_item(2, 10, 20));
    deque_push(dq, make_test_item(3, 20, 30));

    WorkItem out;
    ASSERT_TRUE(deque_steal(dq, &out));
    ASSERT_EQ(out.start, 0);  /* First item (FIFO for thieves) */

    destroy_test_deque(dq);
    return 0;
}

TEST(deque_steal_from_empty_returns_false) {
    WorkDeque* dq = create_test_deque();
    WorkItem out;
    ASSERT_FALSE(deque_steal(dq, &out));
    destroy_test_deque(dq);
    return 0;
}

TEST(deque_steal_all_items) {
    WorkDeque* dq = create_test_deque();
    for (uint32_t i = 0; i < 100; i++) {
        ASSERT_TRUE(deque_push(dq, make_test_item(i, i, i + 1)));
    }

    WorkItem out;
    for (uint32_t i = 0; i < 100; i++) {
        ASSERT_TRUE(deque_steal(dq, &out));
        ASSERT_EQ(out.start, i);  /* FIFO order */
    }
    ASSERT_TRUE(deque_empty(dq));
    destroy_test_deque(dq);
    return 0;
}

/* ============================================================================
 * Mixed Push/Pop/Steal Tests
 * ============================================================================ */

TEST(deque_push_pop_preserves_all_items) {
    WorkDeque* dq = create_test_deque();
    uint64_t sum_pushed = 0, sum_popped = 0;

    for (uint32_t i = 0; i < 1000; i++) {
        ASSERT_TRUE(deque_push(dq, make_test_item(i, i, i + 1)));
        sum_pushed += i;
    }

    WorkItem out;
    while (deque_pop(dq, &out)) {
        sum_popped += out.start;
    }

    ASSERT_EQ(sum_pushed, sum_popped);
    destroy_test_deque(dq);
    return 0;
}

TEST(deque_interleaved_push_pop) {
    WorkDeque* dq = create_test_deque();
    uint64_t sum_pushed = 0, sum_popped = 0;

    for (uint32_t round = 0; round < 100; round++) {
        /* Push 10 items */
        for (uint32_t i = 0; i < 10; i++) {
            uint32_t val = round * 10 + i;
            ASSERT_TRUE(deque_push(dq, make_test_item(val, val, val + 1)));
            sum_pushed += val;
        }
        /* Pop 5 items */
        for (uint32_t i = 0; i < 5; i++) {
            WorkItem out;
            ASSERT_TRUE(deque_pop(dq, &out));
            sum_popped += out.start;
        }
    }

    /* Drain remaining */
    WorkItem out;
    while (deque_pop(dq, &out)) {
        sum_popped += out.start;
    }

    ASSERT_EQ(sum_pushed, sum_popped);
    destroy_test_deque(dq);
    return 0;
}

/* ============================================================================
 * Concurrent Tests
 * ============================================================================ */

typedef struct {
    WorkDeque* dq;
    atomic_u64* items_stolen;
    atomic_u32* done;
} ThiefArgs;

static void* thief_thread(void* arg) {
    ThiefArgs* args = (ThiefArgs*)arg;
    WorkItem out;

    while (!atomic_load_acquire(args->done)) {
        if (deque_steal(args->dq, &out)) {
            atomic_add_relaxed(args->items_stolen, 1);
        } else {
            threading_yield();
        }
    }

    /* Drain any remaining */
    while (deque_steal(args->dq, &out)) {
        atomic_add_relaxed(args->items_stolen, 1);
    }

    return NULL;
}

TEST(deque_concurrent_owner_and_thief_no_duplicates) {
    WorkDeque* dq = create_test_deque();
    atomic_u64 owner_processed = ATOMIC_VAR_INIT(0);
    atomic_u64 items_stolen = ATOMIC_VAR_INIT(0);
    atomic_u32 done = ATOMIC_VAR_INIT(0);

    ThiefArgs args = { dq, &items_stolen, &done };
    pthread_t thief;
    pthread_create(&thief, NULL, thief_thread, &args);

    const uint32_t NUM_ITEMS = 10000;
    PCG32 rng;
    pcg32_seed(&rng, 12345);

    for (uint32_t i = 0; i < NUM_ITEMS; i++) {
        while (!deque_push(dq, make_test_item(i, i, i + 1))) {
            threading_yield();
        }

        /* Randomly pop some items */
        if (pcg32_bounded(&rng, 2) == 0) {
            WorkItem out;
            if (deque_pop(dq, &out)) {
                atomic_add_relaxed(&owner_processed, 1);
            }
        }
    }

    /* Drain owner's items */
    WorkItem out;
    while (deque_pop(dq, &out)) {
        atomic_add_relaxed(&owner_processed, 1);
    }

    /* Signal thief to stop */
    atomic_store_release(&done, 1);
    pthread_join(thief, NULL);

    /* Verify all items processed exactly once */
    uint64_t total = atomic_load_relaxed(&owner_processed) +
                     atomic_load_relaxed(&items_stolen);
    ASSERT_EQ(total, NUM_ITEMS);

    destroy_test_deque(dq);
    return 0;
}

typedef struct {
    WorkDeque* dq;
    atomic_u64* sum;
} StealArgs;

static void* stealing_thread(void* arg) {
    StealArgs* args = (StealArgs*)arg;
    WorkItem out;

    while (deque_steal(args->dq, &out)) {
        atomic_add_relaxed(args->sum, out.start);
    }

    return NULL;
}

TEST(deque_multiple_thieves_no_duplicates) {
    WorkDeque* dq = create_test_deque();
    const uint32_t NUM_ITEMS = THREADING_DEQUE_CAPACITY - 1;
    const uint32_t NUM_THIEVES = 4;

    /* Pre-fill deque */
    uint64_t expected_sum = 0;
    for (uint32_t i = 0; i < NUM_ITEMS; i++) {
        ASSERT_TRUE(deque_push(dq, make_test_item(i, i, i + 1)));
        expected_sum += i;
    }

    atomic_u64 sums[NUM_THIEVES];
    pthread_t thieves[NUM_THIEVES];
    StealArgs args[NUM_THIEVES];

    for (uint32_t t = 0; t < NUM_THIEVES; t++) {
        atomic_init(&sums[t], 0);
        args[t] = (StealArgs){ dq, &sums[t] };
        pthread_create(&thieves[t], NULL, stealing_thread, &args[t]);
    }

    for (uint32_t t = 0; t < NUM_THIEVES; t++) {
        pthread_join(thieves[t], NULL);
    }

    /* Verify sum (no duplicates, no losses) */
    uint64_t total_sum = 0;
    for (uint32_t t = 0; t < NUM_THIEVES; t++) {
        total_sum += atomic_load_relaxed(&sums[t]);
    }
    ASSERT_EQ(total_sum, expected_sum);

    destroy_test_deque(dq);
    return 0;
}

/* ============================================================================
 * Stress Test
 * ============================================================================ */

typedef struct {
    WorkDeque* dq;
    atomic_u64* sum;
    atomic_u32* done;
} LoopingThiefArgs;

static void* looping_thief_thread(void* arg) {
    LoopingThiefArgs* la = (LoopingThiefArgs*)arg;
    WorkItem out;
    while (!atomic_load_acquire(la->done)) {
        if (deque_steal(la->dq, &out)) {
            atomic_add_relaxed(la->sum, out.start);
        } else {
            threading_pause();
        }
    }
    /* Drain remaining */
    while (deque_steal(la->dq, &out)) {
        atomic_add_relaxed(la->sum, out.start);
    }
    return NULL;
}

TEST(deque_stress_test) {
    WorkDeque* dq = create_test_deque();
    const uint32_t ITERATIONS = 100000;

    atomic_u64 sum_pushed = ATOMIC_VAR_INIT(0);
    atomic_u64 sum_popped = ATOMIC_VAR_INIT(0);
    atomic_u32 done = ATOMIC_VAR_INIT(0);

    /* Start thief threads */
    const uint32_t NUM_THIEVES = 3;
    pthread_t thieves[NUM_THIEVES];
    LoopingThiefArgs lt_args[NUM_THIEVES];

    for (uint32_t t = 0; t < NUM_THIEVES; t++) {
        lt_args[t] = (LoopingThiefArgs){ dq, &sum_popped, &done };
        pthread_create(&thieves[t], NULL, looping_thief_thread, &lt_args[t]);
    }

    /* Owner pushes and pops */
    PCG32 rng;
    pcg32_seed(&rng, 54321);

    for (uint32_t i = 0; i < ITERATIONS; i++) {
        if (deque_push(dq, make_test_item(i, i, i + 1))) {
            atomic_add_relaxed(&sum_pushed, i);
        }

        /* Randomly pop */
        if (pcg32_bounded(&rng, 3) == 0) {
            WorkItem out;
            if (deque_pop(dq, &out)) {
                atomic_add_relaxed(&sum_popped, out.start);
            }
        }
    }

    /* Drain owner's queue */
    WorkItem out;
    while (deque_pop(dq, &out)) {
        atomic_add_relaxed(&sum_popped, out.start);
    }

    /* Signal thieves and join */
    atomic_store_release(&done, 1);
    for (uint32_t t = 0; t < NUM_THIEVES; t++) {
        pthread_join(thieves[t], NULL);
    }

    /* Verify all items processed */
    ASSERT_EQ(atomic_load_relaxed(&sum_pushed), atomic_load_relaxed(&sum_popped));

    destroy_test_deque(dq);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Deque Tests");

    /* Initialization tests */
    RUN_TEST(deque_init_initializes_empty);
    RUN_TEST(deque_init_sets_top_and_bottom_to_zero);

    /* Push tests */
    RUN_TEST(deque_push_single_item_succeeds);
    RUN_TEST(deque_push_multiple_items);
    RUN_TEST(deque_push_to_capacity_succeeds);
    RUN_TEST(deque_push_beyond_capacity_fails);

    /* Pop tests */
    RUN_TEST(deque_pop_returns_last_pushed);
    RUN_TEST(deque_pop_from_empty_returns_false);
    RUN_TEST(deque_pop_all_items_leaves_empty);

    /* Steal tests */
    RUN_TEST(deque_steal_returns_first_pushed);
    RUN_TEST(deque_steal_from_empty_returns_false);
    RUN_TEST(deque_steal_all_items);

    /* Mixed tests */
    RUN_TEST(deque_push_pop_preserves_all_items);
    RUN_TEST(deque_interleaved_push_pop);

    /* Concurrent tests */
    RUN_TEST(deque_concurrent_owner_and_thief_no_duplicates);
    RUN_TEST(deque_multiple_thieves_no_duplicates);

    /* Stress test */
    RUN_TEST(deque_stress_test);

    TEST_SUITE_END();
}
