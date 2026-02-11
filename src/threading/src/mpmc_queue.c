/**
 * MPMC Queue (Multi-Producer Multi-Consumer)
 *
 * A lock-free bounded queue for multiple producers and consumers.
 * Based on Dmitry Vyukov's bounded MPMC queue using sequence numbers.
 *
 * Each slot has a sequence number that indicates its state:
 * - sequence == position: slot is ready to be written (empty)
 * - sequence == position + 1: slot contains data (full)
 *
 * Algorithm (push):
 * 1. Load head (enqueue position)
 * 2. Calculate slot index and load slot sequence
 * 3. If sequence == head, slot is ready - try CAS to claim head
 * 4. If CAS succeeds, write data, store sequence = head + 1
 * 5. If sequence < head, queue is full
 *
 * Algorithm (pop):
 * 1. Load tail (dequeue position)
 * 2. Calculate slot index and load slot sequence
 * 3. If sequence == tail + 1, slot has data - try CAS to claim tail
 * 4. If CAS succeeds, read data, store sequence = tail + capacity
 * 5. If sequence < tail + 1, queue is empty
 */

#include "threading.h"

/* ============================================================================
 * Creation and Destruction
 * ============================================================================ */

MPMCQueue* mpmc_create(uint32_t capacity) {
    /* Validate capacity is power of 2 */
    if (!is_power_of_two(capacity)) {
        capacity = next_power_of_two(capacity);
    }

    /* Allocate queue structure */
    MPMCQueue* q = (MPMCQueue*)aligned_alloc(THREADING_CACHE_LINE, sizeof(MPMCQueue));
    if (!q) {
        return NULL;
    }

    /* Allocate slot buffer */
    q->buffer = (MPMCSlot*)aligned_alloc(THREADING_CACHE_LINE, capacity * sizeof(MPMCSlot));
    if (!q->buffer) {
        free(q);
        return NULL;
    }

    q->capacity = capacity;
    q->mask = capacity - 1;

    /* Initialize head and tail */
    atomic_store_relaxed(&q->head, 0);
    atomic_store_relaxed(&q->tail, 0);

    /* Initialize slot sequences: sequence[i] = i (all slots ready for write) */
    for (uint32_t i = 0; i < capacity; i++) {
        atomic_store_relaxed(&q->buffer[i].sequence, i);
        q->buffer[i].data = NULL;
    }

    return q;
}

void mpmc_destroy(MPMCQueue* q) {
    if (q) {
        if (q->buffer) {
            free(q->buffer);
        }
        free(q);
    }
}

/* ============================================================================
 * Push (Enqueue)
 * ============================================================================ */

bool mpmc_push(MPMCQueue* q, void* item) {
    uint64_t head;
    MPMCSlot* slot;
    uint64_t seq;

    /* Spin trying to claim a slot */
    for (;;) {
        head = atomic_load_relaxed(&q->head);
        slot = &q->buffer[head & q->mask];
        seq = atomic_load_acquire(&slot->sequence);

        int64_t diff = (int64_t)seq - (int64_t)head;

        if (diff == 0) {
            /* Slot is ready for writing - try to claim head position */
            uint64_t expected = head;
            if (atomic_cas_weak(&q->head, &expected, head + 1)) {
                /* Successfully claimed this slot */
                break;
            }
            /* CAS failed - another producer claimed it, retry */
        } else if (diff < 0) {
            /* Slot not yet consumed - queue is full */
            return false;
        }
        /* else: diff > 0 means head advanced, reload and retry */

        /* Yield to reduce contention */
        threading_pause();
    }

    /* We own this slot - write data and update sequence */
    slot->data = item;

    /* Release store: makes data visible, marks slot as full */
    atomic_store_release(&slot->sequence, head + 1);

    return true;
}

/* ============================================================================
 * Pop (Dequeue)
 * ============================================================================ */

bool mpmc_pop(MPMCQueue* q, void** out) {
    uint64_t tail;
    MPMCSlot* slot;
    uint64_t seq;

    /* Spin trying to claim a slot */
    for (;;) {
        tail = atomic_load_relaxed(&q->tail);
        slot = &q->buffer[tail & q->mask];
        seq = atomic_load_acquire(&slot->sequence);

        int64_t diff = (int64_t)seq - (int64_t)(tail + 1);

        if (diff == 0) {
            /* Slot contains data - try to claim tail position */
            uint64_t expected = tail;
            if (atomic_cas_weak(&q->tail, &expected, tail + 1)) {
                /* Successfully claimed this slot */
                break;
            }
            /* CAS failed - another consumer claimed it, retry */
        } else if (diff < 0) {
            /* Slot not yet written - queue is empty */
            return false;
        }
        /* else: diff > 0 means tail advanced, reload and retry */

        /* Yield to reduce contention */
        threading_pause();
    }

    /* We own this slot - read data */
    *out = slot->data;

    /*
     * Release store: marks slot as empty (ready for producer).
     * Set sequence to tail + capacity so it will match when head
     * wraps around to this slot.
     */
    atomic_store_release(&slot->sequence, tail + q->capacity);

    return true;
}

/* ============================================================================
 * Query Operations
 * ============================================================================ */

bool mpmc_empty(const MPMCQueue* q) {
    uint64_t tail = atomic_load_relaxed(&q->tail);
    uint64_t head = atomic_load_relaxed(&q->head);

    return tail >= head;
}

uint32_t mpmc_size(const MPMCQueue* q) {
    uint64_t head = atomic_load_relaxed(&q->head);
    uint64_t tail = atomic_load_relaxed(&q->tail);

    int64_t size = (int64_t)(head - tail);
    return (size > 0) ? (uint32_t)size : 0;
}
