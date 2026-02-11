/**
 * Reusable Lock-Free Barrier Implementation
 *
 * A spinning barrier that uses a generation counter to prevent spurious wakeups.
 * All threads block until threshold threads have arrived, then all proceed together.
 *
 * The generation counter allows the barrier to be reused: threads spin-waiting
 * on the current generation will not leak through to the next barrier phase.
 *
 * Algorithm:
 * 1. Each arriving thread atomically increments count
 * 2. If count < threshold, spin on generation change
 * 3. Last thread (count == threshold) increments generation, resets count
 * 4. All waiting threads see generation change and proceed
 */

#include "threading.h"

/* ============================================================================
 * Initialization
 * ============================================================================ */

void barrier_init(Barrier* b, uint32_t threshold) {
    atomic_store_relaxed(&b->count, 0);
    atomic_store_relaxed(&b->generation, 0);
    b->threshold = threshold;
}

/* ============================================================================
 * Barrier Wait
 * ============================================================================ */

void barrier_wait(Barrier* b) {
    /* Capture current generation before incrementing count */
    uint32_t gen = atomic_load_acquire(&b->generation);

    /*
     * Atomically increment count and get previous value.
     * Uses acquire-release to synchronize with other threads.
     */
    uint32_t prev_count = atomic_fetch_add_explicit(
        &b->count,
        1,
        memory_order_acq_rel
    );

    if (prev_count + 1 == b->threshold) {
        /*
         * We are the last thread to arrive.
         * Reset count for next use and increment generation to release waiters.
         *
         * The order is important:
         * 1. Reset count first (with relaxed - no one is reading it yet)
         * 2. Release fence ensures count reset is visible
         * 3. Increment generation (releases all waiters)
         */
        atomic_store_relaxed(&b->count, 0);
        atomic_fence_release();
        atomic_store_release(&b->generation, gen + 1);
    } else {
        /*
         * Not the last thread - spin wait for generation change.
         * Uses exponential backoff to reduce contention.
         */
        uint32_t spin_count = 0;
        const uint32_t MAX_SPINS = 1000;

        while (atomic_load_acquire(&b->generation) == gen) {
            if (spin_count < MAX_SPINS) {
                /* Spin with pause hint */
                threading_pause();
                spin_count++;
            } else {
                /* Back off with yield after many spins */
                threading_yield();
            }
        }
    }

    /* Acquire fence ensures we see all writes from threads before the barrier */
    atomic_fence_acquire();
}

/* ============================================================================
 * Barrier Reset
 * ============================================================================ */

void barrier_reset(Barrier* b) {
    /*
     * Reset is not thread-safe - should only be called when no threads
     * are waiting at the barrier.
     */
    atomic_store_relaxed(&b->count, 0);
    /* Note: we don't reset generation - it continues to increment */
}
