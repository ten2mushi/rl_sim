/**
 * SPSC Queue (Single-Producer Single-Consumer)
 *
 * A lock-free bounded queue for exactly one producer and one consumer thread.
 * Uses simple acquire-release semantics since there's no contention within
 * each role.
 *
 * Memory ordering:
 * - Producer: store item (relaxed), release store to head
 * - Consumer: acquire load from head, load item (relaxed), release store to tail
 *
 * The head/tail are cache-line aligned to prevent false sharing between
 * producer and consumer threads.
 */

#include "threading.h"

/* ============================================================================
 * Creation and Destruction
 * ============================================================================ */

SPSCQueue* spsc_create(uint32_t capacity) {
    /* Validate capacity is power of 2 */
    if (!is_power_of_two(capacity)) {
        capacity = next_power_of_two(capacity);
    }

    /* Allocate queue structure and buffer together */
    size_t total_size = sizeof(SPSCQueue) + (capacity * sizeof(void*));

    SPSCQueue* q = (SPSCQueue*)aligned_alloc(THREADING_CACHE_LINE, total_size);
    if (!q) {
        return NULL;
    }

    /* Buffer immediately follows the queue struct */
    q->buffer = (void**)((char*)q + sizeof(SPSCQueue));
    q->capacity = capacity;
    q->mask = capacity - 1;

    /* Initialize head and tail */
    atomic_store_relaxed(&q->head, 0);
    atomic_store_relaxed(&q->tail, 0);

    /* Zero-initialize buffer */
    memset(q->buffer, 0, capacity * sizeof(void*));

    return q;
}

void spsc_destroy(SPSCQueue* q) {
    if (q) {
        free(q);
    }
}

/* ============================================================================
 * Producer Operations
 * ============================================================================ */

bool spsc_push(SPSCQueue* q, void* item) {
    uint64_t head = atomic_load_relaxed(&q->head);
    uint64_t tail = atomic_load_acquire(&q->tail);

    /* Check if queue is full */
    if (head - tail >= q->capacity) {
        return false;
    }

    /* Store item */
    q->buffer[head & q->mask] = item;

    /* Release store to head - makes item visible to consumer */
    atomic_store_release(&q->head, head + 1);

    return true;
}

/* ============================================================================
 * Consumer Operations
 * ============================================================================ */

bool spsc_pop(SPSCQueue* q, void** out) {
    uint64_t tail = atomic_load_relaxed(&q->tail);
    uint64_t head = atomic_load_acquire(&q->head);

    /* Check if queue is empty */
    if (tail >= head) {
        return false;
    }

    /* Load item */
    *out = q->buffer[tail & q->mask];

    /* Release store to tail - allows producer to reuse slot */
    atomic_store_release(&q->tail, tail + 1);

    return true;
}

/* ============================================================================
 * Query Operations
 * ============================================================================ */

bool spsc_empty(const SPSCQueue* q) {
    uint64_t tail = atomic_load_relaxed(&q->tail);
    uint64_t head = atomic_load_relaxed(&q->head);

    return tail >= head;
}

uint32_t spsc_size(const SPSCQueue* q) {
    uint64_t head = atomic_load_relaxed(&q->head);
    uint64_t tail = atomic_load_relaxed(&q->tail);

    int64_t size = (int64_t)(head - tail);
    return (size > 0) ? (uint32_t)size : 0;
}
