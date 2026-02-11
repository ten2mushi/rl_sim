/**
 * Chase-Lev Work-Stealing Deque Implementation
 *
 * Based on: "Dynamic Circular Work-Stealing Deque" - Chase & Lev, SPAA 2005
 *
 * The deque supports:
 * - LIFO push/pop by owner thread (single-threaded, no contention)
 * - FIFO steal by thief threads (thread-safe, may fail under contention)
 *
 * Memory ordering:
 * - Owner push: relaxed store to item, release fence, relaxed store to bottom
 * - Owner pop: relaxed load/store to bottom, seq_cst fence, relaxed load top, CAS on last item
 * - Thief steal: acquire load top, seq_cst fence, acquire load bottom, relaxed load item, CAS
 */

#include "threading.h"

/* ============================================================================
 * Initialization
 * ============================================================================ */

void deque_init(WorkDeque* dq) {
    atomic_store_relaxed(&dq->top, 0);
    atomic_store_relaxed(&dq->bottom, 0);
    dq->mask = THREADING_DEQUE_MASK;
    dq->_pad = 0;

    /* Zero-initialize items array */
    memset(dq->items, 0, sizeof(dq->items));
}

/* ============================================================================
 * Owner Operations (Single-Threaded)
 * ============================================================================ */

bool deque_push(WorkDeque* dq, WorkItem item) {
    uint64_t b = atomic_load_relaxed(&dq->bottom);
    uint64_t t = atomic_load_relaxed(&dq->top);

    /* Check if deque is full */
    if (FOUNDATION_UNLIKELY(b - t >= THREADING_DEQUE_CAPACITY)) {
        return false;
    }

    /* Store item at bottom */
    dq->items[b & dq->mask] = item;

    /* Release fence ensures item write is visible before bottom increment */
    atomic_fence_release();

    /* Increment bottom (makes item visible to stealers) */
    atomic_store_relaxed(&dq->bottom, b + 1);

    return true;
}

bool deque_pop(WorkDeque* dq, WorkItem* out) {
    uint64_t b = atomic_load_relaxed(&dq->bottom);

    /* Empty check (fast path) */
    if (b == 0) {
        return false;
    }

    /* Decrement bottom speculatively */
    b--;
    atomic_store_relaxed(&dq->bottom, b);

    /*
     * Full memory barrier (seq_cst fence)
     *
     * This is critical: it ensures that our write to bottom (b-1) is visible
     * to all stealers before we read top. Without this, a stealer could:
     * 1. Read old bottom (b)
     * 2. Determine deque is not empty (t < b)
     * 3. Steal item at index t
     * Meanwhile, we could:
     * 1. Read old top (t)
     * 2. Determine t < b-1, so take item without CAS
     * 3. Both take the same item!
     */
    atomic_fence_seq_cst();

    uint64_t t = atomic_load_relaxed(&dq->top);

    if (t <= b) {
        /* Deque is not empty, take item */
        *out = dq->items[b & dq->mask];

        if (t == b) {
            /*
             * This is the last item - need CAS to prevent race with stealer.
             * If a stealer is trying to take this same item, exactly one of us
             * will succeed the CAS.
             */
            uint64_t expected = t;
            if (!atomic_cas_strong(&dq->top, &expected, t + 1)) {
                /* Lost race to stealer - item was stolen */
                atomic_store_relaxed(&dq->bottom, b + 1);
                return false;
            }
            /* Won race - restore bottom and return item */
            atomic_store_relaxed(&dq->bottom, b + 1);
        }
        return true;
    } else {
        /* Deque was empty (stealer took last item between our read of bottom and top) */
        atomic_store_relaxed(&dq->bottom, b + 1);
        return false;
    }
}

/* ============================================================================
 * Thief Operations (Thread-Safe)
 * ============================================================================ */

bool deque_steal(WorkDeque* dq, WorkItem* out) {
    /*
     * Acquire load of top - establishes synchronization with any
     * previous successful stealer's CAS on top
     */
    uint64_t t = atomic_load_acquire(&dq->top);

    /*
     * Full memory barrier (seq_cst fence)
     *
     * This ensures we see the owner's most recent write to bottom.
     * Pairs with the seq_cst fence in pop().
     */
    atomic_fence_seq_cst();

    /* Acquire load of bottom - synchronizes with owner's release store */
    uint64_t b = atomic_load_acquire(&dq->bottom);

    /* Check if deque is empty */
    if (t >= b) {
        return false;
    }

    /* Load item (relaxed - we haven't claimed it yet) */
    *out = dq->items[t & dq->mask];

    /*
     * CAS to claim the item - only one stealer (or owner pop) can succeed.
     * Uses acq_rel ordering:
     * - Acquire: see all writes before our successful CAS
     * - Release: our read of the item is visible before we increment top
     */
    uint64_t expected = t;
    if (!atomic_cas_strong(&dq->top, &expected, t + 1)) {
        /* Lost race to another stealer or owner pop */
        return false;
    }

    return true;
}

/* ============================================================================
 * Query Operations (Relaxed)
 * ============================================================================ */

uint32_t deque_size(const WorkDeque* dq) {
    uint64_t b = atomic_load_relaxed(&dq->bottom);
    uint64_t t = atomic_load_relaxed(&dq->top);

    int64_t size = (int64_t)(b - t);
    return (size > 0) ? (uint32_t)size : 0;
}

bool deque_empty(const WorkDeque* dq) {
    uint64_t b = atomic_load_relaxed(&dq->bottom);
    uint64_t t = atomic_load_relaxed(&dq->top);

    return t >= b;
}
