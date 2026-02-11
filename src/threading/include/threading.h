/**
 * Threading Module - Work-stealing thread pool, synchronization primitives, and hybrid scheduling
 *
 * This module provides high-performance threading primitives for the RL drone engine:
 * - Chase-Lev work-stealing deque for dynamic load balancing
 * - Reusable lock-free barrier for phase synchronization
 * - SPSC/MPMC bounded queues for inter-thread communication
 * - Thread pool with work-stealing and static scheduling modes
 * - Hybrid scheduler for physics (static) and sensors (work-stealing)
 *
 * Dependencies: Foundation (atomics, arena), pthreads
 */

#ifndef THREADING_H
#define THREADING_H

#include "foundation.h"
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Section 1: Configuration Constants
 * ============================================================================ */

#define THREADING_MAX_THREADS    64
#define THREADING_DEQUE_CAPACITY 1024
#define THREADING_SPSC_CAPACITY  512
#define THREADING_MPMC_CAPACITY  2048
#define THREADING_DEFAULT_GRAIN  64
#define THREADING_CACHE_LINE     64

/* Mask for deque index wrapping (capacity must be power of 2) */
#define THREADING_DEQUE_MASK     (THREADING_DEQUE_CAPACITY - 1)

/* ============================================================================
 * Section 2: Forward Declarations
 * ============================================================================ */

typedef struct WorkDeque    WorkDeque;
typedef struct Barrier      Barrier;
typedef struct SPSCQueue    SPSCQueue;
typedef struct MPMCQueue    MPMCQueue;
typedef struct ThreadPool   ThreadPool;
typedef struct Scheduler    Scheduler;

/* ============================================================================
 * Section 3: Work Item
 * ============================================================================ */

/**
 * Work function signature
 * @param data      User data passed to the function
 * @param start     Start index of the work range (inclusive)
 * @param end       End index of the work range (exclusive)
 * @param thread_id Thread ID executing this work (for thread-local scratch buffers)
 */
typedef void (*WorkFunction)(void* data, uint32_t start, uint32_t end, uint32_t thread_id);

/**
 * Work item representing a unit of parallel work
 * 24 bytes total
 */
typedef struct WorkItem {
    WorkFunction fn;      /* 8 bytes - function pointer */
    void*        data;    /* 8 bytes - user data */
    uint32_t     start;   /* 4 bytes - batch start index */
    uint32_t     end;     /* 4 bytes - batch end index (exclusive) */
} WorkItem;

/* ============================================================================
 * Section 4: Chase-Lev Work-Stealing Deque
 * ============================================================================
 *
 * Lock-free deque supporting:
 * - LIFO push/pop by owner thread (better cache locality)
 * - FIFO steal by thief threads (work stealing)
 *
 * Based on: "Dynamic Circular Work-Stealing Deque" - Chase & Lev, SPAA 2005
 */

/**
 * Work-stealing deque (~24KB per thread)
 * Owner thread: push/pop from bottom (LIFO)
 * Thief threads: steal from top (FIFO)
 */
struct WorkDeque {
    /* Cache-line aligned to prevent false sharing */
    alignas(THREADING_CACHE_LINE) atomic_u64 top;      /* 8 bytes - thieves read/CAS */
    alignas(THREADING_CACHE_LINE) atomic_u64 bottom;   /* 8 bytes - owner read/write */
    WorkItem items[THREADING_DEQUE_CAPACITY];          /* 24,576 bytes */
    uint32_t mask;                                     /* 4 bytes - (DEQUE_CAPACITY - 1) */
    uint32_t _pad;                                     /* 4 bytes - alignment padding */
};

/**
 * Initialize a work-stealing deque
 * @param dq Pointer to deque (caller provides memory)
 */
void deque_init(WorkDeque* dq);

/**
 * Push a work item to the bottom of the deque (owner only)
 * @param dq   The deque
 * @param item Work item to push
 * @return true if push succeeded, false if deque is full
 */
bool deque_push(WorkDeque* dq, WorkItem item);

/**
 * Pop a work item from the bottom of the deque (owner only)
 * @param dq  The deque
 * @param out Output pointer for the popped item
 * @return true if pop succeeded, false if deque is empty
 */
bool deque_pop(WorkDeque* dq, WorkItem* out);

/**
 * Steal a work item from the top of the deque (thread-safe)
 * May fail under contention; caller should retry later
 * @param dq  The deque
 * @param out Output pointer for the stolen item
 * @return true if steal succeeded, false if empty or contended
 */
bool deque_steal(WorkDeque* dq, WorkItem* out);

/**
 * Query the approximate size of the deque (relaxed read)
 * @param dq The deque
 * @return Approximate number of items in the deque
 */
uint32_t deque_size(const WorkDeque* dq);

/**
 * Check if the deque is approximately empty (relaxed read)
 * @param dq The deque
 * @return true if deque appears empty
 */
bool deque_empty(const WorkDeque* dq);

/* ============================================================================
 * Section 5: Reusable Barrier (Lock-Free)
 * ============================================================================
 *
 * Spinning barrier with generation counter to prevent spurious wakeups.
 * All threads block until threshold threads have arrived, then all proceed.
 * Reusable: generation counter allows multiple barrier phases.
 */

/**
 * Reusable lock-free barrier
 * 12 bytes + cache-line padding
 */
struct Barrier {
    alignas(THREADING_CACHE_LINE) atomic_u32 count;      /* Current waiting count */
    atomic_u32 generation;                               /* Prevents spurious wakeups */
    uint32_t   threshold;                                /* Threads to wait for */
};

/**
 * Initialize a barrier
 * @param b         Pointer to barrier (caller provides memory)
 * @param threshold Number of threads that must arrive before release
 */
void barrier_init(Barrier* b, uint32_t threshold);

/**
 * Wait at the barrier until all threads arrive
 * @param b The barrier
 */
void barrier_wait(Barrier* b);

/**
 * Reset barrier for reuse (not thread-safe, call when no threads are waiting)
 * @param b The barrier
 */
void barrier_reset(Barrier* b);

/* ============================================================================
 * Section 6: SPSC Queue (Single-Producer Single-Consumer)
 * ============================================================================
 *
 * Lock-free bounded queue for exactly one producer and one consumer thread.
 * Higher throughput than MPMC when applicable.
 */

/**
 * Single-producer single-consumer bounded queue
 */
struct SPSCQueue {
    void**   buffer;      /* Item buffer */
    uint32_t capacity;    /* Buffer size (power of 2) */
    uint32_t mask;        /* capacity - 1 */
    alignas(THREADING_CACHE_LINE) atomic_u64 head;  /* Write position (producer) */
    alignas(THREADING_CACHE_LINE) atomic_u64 tail;  /* Read position (consumer) */
};

/**
 * Create an SPSC queue
 * @param capacity Maximum number of items (must be power of 2)
 * @return Pointer to new queue, or NULL on failure
 */
SPSCQueue* spsc_create(uint32_t capacity);

/**
 * Destroy an SPSC queue
 * @param q The queue to destroy
 */
void spsc_destroy(SPSCQueue* q);

/**
 * Push an item to the queue (producer only)
 * @param q    The queue
 * @param item Item to push
 * @return true if push succeeded, false if queue is full
 */
bool spsc_push(SPSCQueue* q, void* item);

/**
 * Pop an item from the queue (consumer only)
 * @param q   The queue
 * @param out Output pointer for the popped item
 * @return true if pop succeeded, false if queue is empty
 */
bool spsc_pop(SPSCQueue* q, void** out);

/**
 * Check if queue is empty (relaxed read)
 * @param q The queue
 * @return true if queue appears empty
 */
bool spsc_empty(const SPSCQueue* q);

/**
 * Get approximate queue size (relaxed read)
 * @param q The queue
 * @return Approximate number of items in queue
 */
uint32_t spsc_size(const SPSCQueue* q);

/* ============================================================================
 * Section 7: MPMC Queue (Multi-Producer Multi-Consumer)
 * ============================================================================
 *
 * Lock-free bounded queue for multiple producers and consumers.
 * Uses sequence numbers for wait-free operation.
 */

/**
 * Slot in the MPMC queue with sequence number
 */
typedef struct MPMCSlot {
    atomic_u64 sequence;
    void*      data;
} MPMCSlot;

/**
 * Multi-producer multi-consumer bounded queue
 */
struct MPMCQueue {
    MPMCSlot* buffer;     /* Slot buffer */
    uint32_t  capacity;   /* Buffer size (power of 2) */
    uint32_t  mask;       /* capacity - 1 */
    alignas(THREADING_CACHE_LINE) atomic_u64 head;  /* Enqueue position */
    alignas(THREADING_CACHE_LINE) atomic_u64 tail;  /* Dequeue position */
};

/**
 * Create an MPMC queue
 * @param capacity Maximum number of items (must be power of 2)
 * @return Pointer to new queue, or NULL on failure
 */
MPMCQueue* mpmc_create(uint32_t capacity);

/**
 * Destroy an MPMC queue
 * @param q The queue to destroy
 */
void mpmc_destroy(MPMCQueue* q);

/**
 * Push an item to the queue (thread-safe)
 * @param q    The queue
 * @param item Item to push
 * @return true if push succeeded, false if queue is full
 */
bool mpmc_push(MPMCQueue* q, void* item);

/**
 * Pop an item from the queue (thread-safe)
 * @param q   The queue
 * @param out Output pointer for the popped item
 * @return true if pop succeeded, false if queue is empty
 */
bool mpmc_pop(MPMCQueue* q, void** out);

/**
 * Check if queue is empty (relaxed read)
 * @param q The queue
 * @return true if queue appears empty
 */
bool mpmc_empty(const MPMCQueue* q);

/**
 * Get approximate queue size (relaxed read)
 * @param q The queue
 * @return Approximate number of items in queue
 */
uint32_t mpmc_size(const MPMCQueue* q);

/* ============================================================================
 * Section 8: Thread Pool
 * ============================================================================
 *
 * Work-stealing thread pool with:
 * - Per-thread local deques for pushed work
 * - Shared queue for initial work distribution
 * - Work stealing for load balancing
 */

/**
 * Thread pool statistics
 */
typedef struct ThreadPoolStats {
    uint64_t total_tasks_executed;
    uint64_t total_tasks_stolen;
    uint64_t total_steal_attempts;
    uint64_t per_thread_executed[THREADING_MAX_THREADS];
    uint64_t per_thread_stolen[THREADING_MAX_THREADS];
} ThreadPoolStats;

/**
 * Thread pool configuration
 */
typedef struct ThreadPoolConfig {
    uint32_t num_threads;     /* 0 = auto-detect CPU count */
    uint32_t queue_capacity;  /* 0 = default (4096) */
} ThreadPoolConfig;

/**
 * Thread pool with work-stealing
 */
struct ThreadPool {
    /* Thread handles */
    pthread_t threads[THREADING_MAX_THREADS];
    uint32_t  thread_count;

    /* Per-thread work deques (for work-stealing) */
    WorkDeque deques[THREADING_MAX_THREADS];

    /* Per-thread RNG for random victim selection */
    PCG32 rngs[THREADING_MAX_THREADS];

    /* Shared work queue (for initial distribution) */
    WorkItem* shared_queue;
    atomic_u32* queue_sequence;  /* Per-slot sequence numbers for synchronization */
    alignas(THREADING_CACHE_LINE) atomic_u32 queue_head;
    alignas(THREADING_CACHE_LINE) atomic_u32 queue_tail;
    uint32_t queue_capacity;
    uint32_t queue_mask;

    /* Synchronization */
    alignas(THREADING_CACHE_LINE) atomic_u32 active_tasks;
    atomic_u32 shutdown;
    atomic_u32 waiting_workers;

    /* Condition variable for idle workers */
    pthread_mutex_t mutex;
    pthread_cond_t  work_available;
    pthread_cond_t  work_done;

    /* Statistics */
    alignas(THREADING_CACHE_LINE) atomic_u64 stats_executed[THREADING_MAX_THREADS];
    atomic_u64 stats_stolen[THREADING_MAX_THREADS];
    atomic_u64 stats_steal_attempts;

    /* Thread-local data via pthread_key */
    pthread_key_t thread_id_key;
};

/**
 * Create a thread pool
 * @param config Configuration (NULL for defaults)
 * @return Pointer to new thread pool, or NULL on failure
 */
ThreadPool* threadpool_create(const ThreadPoolConfig* config);

/**
 * Destroy a thread pool (waits for all work to complete)
 * @param pool The thread pool to destroy
 */
void threadpool_destroy(ThreadPool* pool);

/**
 * Submit a single work item
 * @param pool The thread pool
 * @param item Work item to submit
 * @return true if submission succeeded, false if queue is full
 */
bool threadpool_submit(ThreadPool* pool, WorkItem item);

/**
 * Submit a batch of work items
 * @param pool  The thread pool
 * @param items Array of work items
 * @param count Number of items in array
 * @return true if all submissions succeeded
 */
bool threadpool_submit_batch(ThreadPool* pool, const WorkItem* items, uint32_t count);

/**
 * Wait for all pending work to complete
 * @param pool The thread pool
 */
void threadpool_wait(ThreadPool* pool);

/**
 * Barrier synchronization across all workers
 * @param pool The thread pool
 */
void threadpool_barrier(ThreadPool* pool);

/**
 * Get number of worker threads
 * @param pool The thread pool
 * @return Number of worker threads
 */
uint32_t threadpool_num_threads(const ThreadPool* pool);

/**
 * Get number of pending tasks
 * @param pool The thread pool
 * @return Approximate number of pending tasks
 */
uint32_t threadpool_pending_tasks(const ThreadPool* pool);

/**
 * Check if pool is idle (no pending work)
 * @param pool The thread pool
 * @return true if pool is idle
 */
bool threadpool_is_idle(const ThreadPool* pool);

/**
 * Get pool statistics
 * @param pool  The thread pool
 * @param stats Output for statistics
 */
void threadpool_get_stats(const ThreadPool* pool, ThreadPoolStats* stats);

/**
 * Reset pool statistics
 * @param pool The thread pool
 */
void threadpool_reset_stats(ThreadPool* pool);

/* ============================================================================
 * Section 9: Scheduler
 * ============================================================================
 *
 * High-level scheduler providing:
 * - Static partitioning for uniform work (physics)
 * - Work-stealing for variable work (sensors)
 * - Adaptive mode selection based on work size
 */

/**
 * Scheduling strategy
 */
typedef enum ScheduleStrategy {
    SCHEDULE_STATIC        = 0,  /* Fixed partitioning (physics) */
    SCHEDULE_WORK_STEALING = 1,  /* Dynamic load balancing (sensors) */
    SCHEDULE_ADAPTIVE      = 2   /* Runtime selection based on work size */
} ScheduleStrategy;

/**
 * Scheduler with hybrid scheduling support
 */
struct Scheduler {
    ThreadPool*      pool;
    ScheduleStrategy default_strategy;

    /* Adaptive scheduling thresholds */
    uint32_t steal_threshold;   /* Use stealing if work > threshold */
    uint32_t min_chunk_size;    /* Minimum work per task */
};

/**
 * Create a scheduler
 * @param pool Thread pool to use
 * @return Pointer to new scheduler, or NULL on failure
 */
Scheduler* scheduler_create(ThreadPool* pool);

/**
 * Destroy a scheduler
 * @param sched The scheduler to destroy
 */
void scheduler_destroy(Scheduler* sched);

/**
 * Configure scheduler thresholds
 * @param sched           The scheduler
 * @param steal_threshold Use work-stealing if work_count exceeds this
 * @param min_chunk_size  Minimum items per work chunk
 */
void scheduler_configure(Scheduler* sched, uint32_t steal_threshold, uint32_t min_chunk_size);

/**
 * Schedule physics work with static partitioning
 * Each thread processes drone_count/num_threads drones
 *
 * @param sched       The scheduler
 * @param physics_fn  Physics function to execute
 * @param physics_data Data to pass to physics function
 * @param drone_count Number of drones to process
 */
void scheduler_physics(
    Scheduler* sched,
    WorkFunction physics_fn,
    void* physics_data,
    uint32_t drone_count
);

/**
 * Schedule sensor work with work-stealing
 * Work is divided into fine-grained chunks for load balancing
 *
 * @param sched      The scheduler
 * @param sensor_fn  Sensor function to execute
 * @param sensor_data Data to pass to sensor function
 * @param work_sizes Array of work sizes per sensor type
 * @param num_types  Number of sensor types
 */
void scheduler_sensors(
    Scheduler* sched,
    WorkFunction sensor_fn,
    void* sensor_data,
    const uint32_t* work_sizes,
    uint32_t num_types
);

/**
 * Execute generic parallel work
 *
 * @param sched      The scheduler
 * @param fn         Work function
 * @param data       User data
 * @param work_count Total work items
 * @param strategy   Scheduling strategy (ADAPTIVE selects at runtime)
 */
void scheduler_execute(
    Scheduler* sched,
    WorkFunction fn,
    void* data,
    uint32_t work_count,
    ScheduleStrategy strategy
);

/**
 * Parallel for loop with automatic chunking
 *
 * @param sched    The scheduler
 * @param fn       Work function
 * @param data     User data
 * @param start    Start index (inclusive)
 * @param end      End index (exclusive)
 * @param strategy Scheduling strategy
 */
void scheduler_parallel_for(
    Scheduler* sched,
    WorkFunction fn,
    void* data,
    uint32_t start,
    uint32_t end,
    ScheduleStrategy strategy
);

/* ============================================================================
 * Section 10: Utility Functions
 * ============================================================================ */

/**
 * Get hardware thread count
 * @return Number of hardware threads (cores * hyperthreads)
 */
uint32_t threading_hardware_concurrency(void);

/**
 * Set thread affinity (pin to core)
 * @param thread  Thread handle
 * @param core_id Core to pin to
 */
void threading_set_affinity(pthread_t thread, uint32_t core_id);

/**
 * Set thread priority
 * @param thread   Thread handle
 * @param priority Priority level (0 = normal, positive = higher)
 */
void threading_set_priority(pthread_t thread, int priority);

/**
 * Yield current thread
 */
void threading_yield(void);

/**
 * Spin-wait with exponential backoff
 * @param iterations Base number of spin iterations
 */
void threading_spin_wait(uint32_t iterations);

/**
 * Pause hint for spin loops (reduces CPU power consumption)
 */
FOUNDATION_INLINE void threading_pause(void) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__) || defined(_M_ARM64)
    __asm__ volatile("yield" ::: "memory");
#else
    /* No-op for other architectures */
    atomic_thread_fence(memory_order_seq_cst);
#endif
}

/* ============================================================================
 * Section 11: Memory Size Helpers
 * ============================================================================ */

/**
 * Calculate memory required for thread pool
 * @param num_threads Number of worker threads
 * @param queue_capacity Shared queue capacity
 * @return Total bytes required
 */
FOUNDATION_INLINE size_t threading_pool_memory_size(uint32_t num_threads, uint32_t queue_capacity) {
    (void)num_threads;
    return sizeof(ThreadPool) + (queue_capacity * sizeof(WorkItem));
}

/**
 * Calculate memory required for SPSC queue
 * @param capacity Queue capacity
 * @return Total bytes required
 */
FOUNDATION_INLINE size_t threading_spsc_memory_size(uint32_t capacity) {
    return sizeof(SPSCQueue) + (capacity * sizeof(void*));
}

/**
 * Calculate memory required for MPMC queue
 * @param capacity Queue capacity
 * @return Total bytes required
 */
FOUNDATION_INLINE size_t threading_mpmc_memory_size(uint32_t capacity) {
    return sizeof(MPMCQueue) + (capacity * sizeof(MPMCSlot));
}

/* ============================================================================
 * Section 12: Type Size Verification
 * ============================================================================ */

FOUNDATION_STATIC_ASSERT(sizeof(WorkItem) == 24, "WorkItem must be 24 bytes");
FOUNDATION_STATIC_ASSERT(THREADING_DEQUE_CAPACITY > 0 &&
    (THREADING_DEQUE_CAPACITY & (THREADING_DEQUE_CAPACITY - 1)) == 0,
    "THREADING_DEQUE_CAPACITY must be power of 2");

#ifdef __cplusplus
}
#endif

#endif /* THREADING_H */
