/**
 * Thread Pool Implementation with Work-Stealing
 *
 * The thread pool manages a set of worker threads that process work items.
 * Work distribution strategy:
 * 1. Submitted work goes to the shared queue
 * 2. Workers pull from shared queue first
 * 3. If shared queue is empty, workers steal from other threads' local deques
 *
 * Each worker maintains a local deque for work that spills from the shared queue
 * or for work submitted by the worker itself (nested parallelism).
 */

#include "threading.h"
#include <unistd.h>
#include <sched.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

/* Default configuration */
#define DEFAULT_QUEUE_CAPACITY 4096

/* ============================================================================
 * Thread Context (for passing ID to workers)
 * ============================================================================ */

typedef struct {
    ThreadPool* pool;
    uint32_t    thread_id;
} WorkerContext;

/* Verify WorkerContext fits in the 16-byte slots allocated in ThreadPool._worker_ctx */
FOUNDATION_STATIC_ASSERT(sizeof(WorkerContext) <= 16, "WorkerContext must fit in 16 bytes");

/* ============================================================================
 * Forward Declarations
 * ============================================================================ */

static void* worker_thread(void* arg);
static bool try_get_work(ThreadPool* pool, uint32_t thread_id, WorkItem* out);
static bool try_steal_work(ThreadPool* pool, uint32_t thread_id, WorkItem* out);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

uint32_t threading_hardware_concurrency(void) {
#ifdef __APPLE__
    int count;
    size_t size = sizeof(count);
    if (sysctlbyname("hw.logicalcpu", &count, &size, NULL, 0) == 0) {
        return (uint32_t)count;
    }
    return 4; /* Fallback */
#else
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    return (nprocs > 0) ? (uint32_t)nprocs : 4;
#endif
}

void threading_set_affinity(pthread_t thread, uint32_t core_id) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
#else
    /* macOS doesn't support pthread_setaffinity_np */
    (void)thread;
    (void)core_id;
#endif
}

void threading_set_priority(pthread_t thread, int priority) {
    struct sched_param param;
    param.sched_priority = priority;
    pthread_setschedparam(thread, SCHED_OTHER, &param);
}

void threading_yield(void) {
    sched_yield();
}

void threading_spin_wait(uint32_t iterations) {
    for (uint32_t i = 0; i < iterations; i++) {
        threading_pause();
    }
}

/* ============================================================================
 * Thread Pool Creation
 * ============================================================================ */

ThreadPool* threadpool_create(const ThreadPoolConfig* config) {
    uint32_t num_threads = (config && config->num_threads > 0)
        ? config->num_threads
        : threading_hardware_concurrency();

    if (num_threads > THREADING_MAX_THREADS) {
        num_threads = THREADING_MAX_THREADS;
    }

    uint32_t queue_capacity = (config && config->queue_capacity > 0)
        ? config->queue_capacity
        : DEFAULT_QUEUE_CAPACITY;

    /* Ensure power of 2 */
    if (!is_power_of_two(queue_capacity)) {
        queue_capacity = next_power_of_two(queue_capacity);
    }

    /* Allocate pool */
    ThreadPool* pool = (ThreadPool*)aligned_alloc(THREADING_CACHE_LINE, sizeof(ThreadPool));
    if (!pool) {
        return NULL;
    }
    memset(pool, 0, sizeof(ThreadPool));

    /* Allocate shared queue */
    pool->shared_queue = (WorkItem*)aligned_alloc(
        THREADING_CACHE_LINE,
        queue_capacity * sizeof(WorkItem)
    );
    if (!pool->shared_queue) {
        free(pool);
        return NULL;
    }
    memset(pool->shared_queue, 0, queue_capacity * sizeof(WorkItem));

    /* Allocate sequence numbers for shared queue synchronization */
    pool->queue_sequence = (atomic_u32*)aligned_alloc(
        THREADING_CACHE_LINE,
        queue_capacity * sizeof(atomic_u32)
    );
    if (!pool->queue_sequence) {
        free(pool->shared_queue);
        free(pool);
        return NULL;
    }
    /* Initialize: sequence[i] = i means slot i is ready for writing */
    for (uint32_t i = 0; i < queue_capacity; i++) {
        atomic_store_relaxed(&pool->queue_sequence[i], i);
    }

    pool->queue_capacity = queue_capacity;
    pool->queue_mask = queue_capacity - 1;
    pool->thread_count = num_threads;

    /* Initialize atomics */
    atomic_store_relaxed(&pool->queue_head, 0);
    atomic_store_relaxed(&pool->queue_tail, 0);
    atomic_store_relaxed(&pool->active_tasks, 0);
    atomic_store_relaxed(&pool->shutdown, 0);
    atomic_store_relaxed(&pool->waiting_workers, 0);

    /* Initialize per-thread deques and RNGs */
    for (uint32_t i = 0; i < num_threads; i++) {
        deque_init(&pool->deques[i]);
        pcg32_seed_dual(&pool->rngs[i], i * 12345ULL + 67890ULL, i + 1);
        atomic_store_relaxed(&pool->stats_executed[i], 0);
        atomic_store_relaxed(&pool->stats_stolen[i], 0);
    }
    atomic_store_relaxed(&pool->stats_steal_attempts, 0);

    /* Initialize synchronization primitives */
    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->work_available, NULL);
    pthread_cond_init(&pool->work_done, NULL);

    /* Create thread-local key for thread ID */
    pthread_key_create(&pool->thread_id_key, NULL);

    /* Start worker threads */
    WorkerContext* contexts = (WorkerContext*)(void*)pool->_worker_ctx;
    for (uint32_t i = 0; i < num_threads; i++) {
        /* Set up context for this thread */
        contexts[i].pool = pool;
        contexts[i].thread_id = i;

        if (pthread_create(&pool->threads[i], NULL, worker_thread, &contexts[i]) != 0) {
            /* Failed to create thread - cleanup and fail */
            atomic_store_relaxed(&pool->shutdown, 1);
            pthread_cond_broadcast(&pool->work_available);

            for (uint32_t j = 0; j < i; j++) {
                pthread_join(pool->threads[j], NULL);
            }

            pthread_mutex_destroy(&pool->mutex);
            pthread_cond_destroy(&pool->work_available);
            pthread_cond_destroy(&pool->work_done);
            pthread_key_delete(pool->thread_id_key);
            free(pool->shared_queue);
            free(pool);
            return NULL;
        }
    }

    return pool;
}

/* ============================================================================
 * Thread Pool Destruction
 * ============================================================================ */

void threadpool_destroy(ThreadPool* pool) {
    if (!pool) {
        return;
    }

    /* Wait for pending work */
    threadpool_wait(pool);

    /* Signal shutdown */
    atomic_store_release(&pool->shutdown, 1);

    /* Wake all workers */
    pthread_mutex_lock(&pool->mutex);
    pthread_cond_broadcast(&pool->work_available);
    pthread_mutex_unlock(&pool->mutex);

    /* Join all threads */
    for (uint32_t i = 0; i < pool->thread_count; i++) {
        pthread_join(pool->threads[i], NULL);
    }

    /* Cleanup */
    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->work_available);
    pthread_cond_destroy(&pool->work_done);
    pthread_key_delete(pool->thread_id_key);
    free(pool->queue_sequence);
    free(pool->shared_queue);
    free(pool);
}

/* ============================================================================
 * Work Submission
 * ============================================================================ */

bool threadpool_submit(ThreadPool* pool, WorkItem item) {
    /*
     * Use sequence number pattern (like MPMC queue):
     * - sequence == position: slot is ready to write
     * - sequence == position + 1: slot contains data
     *
     * 1. Load tail position
     * 2. Load slot sequence
     * 3. If sequence == tail, slot is ready - try CAS to claim
     * 4. Write item
     * 5. Store sequence = tail + 1 to mark slot as containing data
     */

    uint32_t tail;
    uint32_t slot_idx;
    uint32_t seq;

    for (;;) {
        tail = atomic_load_relaxed(&pool->queue_tail);
        slot_idx = tail & pool->queue_mask;
        seq = atomic_load_acquire(&pool->queue_sequence[slot_idx]);

        int32_t diff = (int32_t)seq - (int32_t)tail;

        if (diff == 0) {
            /* Slot is ready for writing - try to claim tail position */
            uint32_t expected = tail;
            if (atomic_cas_weak(&pool->queue_tail, &expected, tail + 1)) {
                break;  /* Successfully claimed this slot */
            }
            /* CAS failed - another producer claimed it, retry */
        } else if (diff < 0) {
            /* Slot not yet consumed - queue is full */
            return false;
        }
        /* else: diff > 0 means tail advanced, reload and retry */
        threading_pause();
    }

    /* Write item to our claimed slot */
    pool->shared_queue[slot_idx] = item;

    /* Increment active tasks */
    atomic_fetch_add_explicit(&pool->active_tasks, 1, memory_order_release);

    /* Mark slot as containing data (sequence = tail + 1) */
    atomic_store_release(&pool->queue_sequence[slot_idx], tail + 1);

    /* Wake a worker if any are sleeping */
    uint32_t waiting = atomic_load_relaxed(&pool->waiting_workers);
    if (waiting > 0) {
        pthread_mutex_lock(&pool->mutex);
        pthread_cond_signal(&pool->work_available);
        pthread_mutex_unlock(&pool->mutex);
    }

    return true;
}

bool threadpool_submit_batch(ThreadPool* pool, const WorkItem* items, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        if (!threadpool_submit(pool, items[i])) {
            return false;
        }
    }
    return true;
}

/* ============================================================================
 * Synchronization
 * ============================================================================ */

void threadpool_wait(ThreadPool* pool) {
    /* Spin-wait for active tasks to reach 0 */
    while (atomic_load_acquire(&pool->active_tasks) > 0) {
        pthread_mutex_lock(&pool->mutex);
        if (atomic_load_acquire(&pool->active_tasks) > 0) {
            /* Wait with timeout to avoid deadlock */
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 1000000; /* 1ms timeout */
            if (ts.tv_nsec >= 1000000000) {
                ts.tv_sec++;
                ts.tv_nsec -= 1000000000;
            }
            pthread_cond_timedwait(&pool->work_done, &pool->mutex, &ts);
        }
        pthread_mutex_unlock(&pool->mutex);
    }
}

/* ============================================================================
 * Query Functions
 * ============================================================================ */

uint32_t threadpool_num_threads(const ThreadPool* pool) {
    return pool->thread_count;
}

uint32_t threadpool_pending_tasks(const ThreadPool* pool) {
    return atomic_load_relaxed(&pool->active_tasks);
}

bool threadpool_is_idle(const ThreadPool* pool) {
    return atomic_load_relaxed(&pool->active_tasks) == 0;
}

/* ============================================================================
 * Statistics
 * ============================================================================ */

void threadpool_get_stats(const ThreadPool* pool, ThreadPoolStats* stats) {
    memset(stats, 0, sizeof(ThreadPoolStats));

    for (uint32_t i = 0; i < pool->thread_count; i++) {
        stats->per_thread_executed[i] = atomic_load_relaxed(&pool->stats_executed[i]);
        stats->per_thread_stolen[i] = atomic_load_relaxed(&pool->stats_stolen[i]);
        stats->total_tasks_executed += stats->per_thread_executed[i];
        stats->total_tasks_stolen += stats->per_thread_stolen[i];
    }
    stats->total_steal_attempts = atomic_load_relaxed(&pool->stats_steal_attempts);
}

void threadpool_reset_stats(ThreadPool* pool) {
    for (uint32_t i = 0; i < pool->thread_count; i++) {
        atomic_store_relaxed(&pool->stats_executed[i], 0);
        atomic_store_relaxed(&pool->stats_stolen[i], 0);
    }
    atomic_store_relaxed(&pool->stats_steal_attempts, 0);
}

/* ============================================================================
 * Worker Thread
 * ============================================================================ */

static void* worker_thread(void* arg) {
    WorkerContext* ctx = (WorkerContext*)arg;
    ThreadPool* pool = ctx->pool;
    uint32_t my_id = ctx->thread_id;

    /* Store thread ID in TLS */
    pthread_setspecific(pool->thread_id_key, (void*)(uintptr_t)(my_id + 1));

    uint32_t spin_count = 0;
    const uint32_t MAX_SPINS = 64;

    while (!atomic_load_acquire(&pool->shutdown)) {
        WorkItem item;
        bool found = false;

        /* 1. Try to get work from shared queue or local deque */
        if (try_get_work(pool, my_id, &item)) {
            found = true;
        }
        /* 2. Try stealing from other threads */
        else if (try_steal_work(pool, my_id, &item)) {
            found = true;
            atomic_add_relaxed(&pool->stats_stolen[my_id], 1);
        }

        if (found) {
            spin_count = 0;

            /* Execute work */
            item.fn(item.data, item.start, item.end, my_id);

            /* Update stats */
            atomic_add_relaxed(&pool->stats_executed[my_id], 1);

            /* Decrement active tasks */
            uint32_t remaining = atomic_fetch_sub_explicit(
                &pool->active_tasks, 1, memory_order_acq_rel) - 1;

            /* Signal completion if all work is done */
            if (remaining == 0) {
                pthread_mutex_lock(&pool->mutex);
                pthread_cond_broadcast(&pool->work_done);
                pthread_mutex_unlock(&pool->mutex);
            }
        } else {
            /* No work available */
            spin_count++;

            if (spin_count < MAX_SPINS) {
                /* Spin with pause */
                threading_pause();
            } else {
                /* Go to sleep */
                pthread_mutex_lock(&pool->mutex);
                atomic_add_relaxed(&pool->waiting_workers, 1);

                /* Double-check for work and shutdown before sleeping */
                if (atomic_load_relaxed(&pool->active_tasks) == 0 &&
                    !atomic_load_relaxed(&pool->shutdown)) {
                    /* Wait for work or shutdown signal */
                    struct timespec ts;
                    clock_gettime(CLOCK_REALTIME, &ts);
                    ts.tv_nsec += 10000000; /* 10ms timeout */
                    if (ts.tv_nsec >= 1000000000) {
                        ts.tv_sec++;
                        ts.tv_nsec -= 1000000000;
                    }
                    pthread_cond_timedwait(&pool->work_available, &pool->mutex, &ts);
                }

                atomic_sub_relaxed(&pool->waiting_workers, 1);
                pthread_mutex_unlock(&pool->mutex);
                spin_count = 0;
            }
        }
    }

    return NULL;
}

/* ============================================================================
 * Work Acquisition
 * ============================================================================ */

static bool try_get_work(ThreadPool* pool, uint32_t thread_id, WorkItem* out) {
    /* 1. Try local deque first (LIFO - better locality) */
    if (deque_pop(&pool->deques[thread_id], out)) {
        return true;
    }

    /* 2. Try shared queue using sequence numbers */
    uint32_t head;
    uint32_t slot_idx;
    uint32_t seq;

    for (;;) {
        head = atomic_load_relaxed(&pool->queue_head);
        slot_idx = head & pool->queue_mask;
        seq = atomic_load_acquire(&pool->queue_sequence[slot_idx]);

        int32_t diff = (int32_t)seq - (int32_t)(head + 1);

        if (diff == 0) {
            /* Slot contains data - try to claim head position */
            uint32_t expected = head;
            if (atomic_cas_weak(&pool->queue_head, &expected, head + 1)) {
                /* Successfully claimed this slot */
                *out = pool->shared_queue[slot_idx];

                /* Mark slot as empty: sequence = head + capacity */
                atomic_store_release(&pool->queue_sequence[slot_idx], head + pool->queue_capacity);
                return true;
            }
            /* CAS failed - another consumer claimed it, retry */
        } else if (diff < 0) {
            /* Slot not yet written - queue is empty */
            return false;
        }
        /* else: diff > 0 means head advanced, reload and retry */
        threading_pause();
    }
}

static bool try_steal_work(ThreadPool* pool, uint32_t thread_id, WorkItem* out) {
    uint32_t num_threads = pool->thread_count;

    if (num_threads <= 1) {
        return false;
    }

    atomic_add_relaxed(&pool->stats_steal_attempts, 1);

    /* Try to steal from a random victim */
    PCG32* rng = &pool->rngs[thread_id];
    uint32_t start = pcg32_bounded(rng, num_threads);

    for (uint32_t i = 0; i < num_threads; i++) {
        uint32_t victim = (start + i) % num_threads;
        if (victim != thread_id) {
            if (deque_steal(&pool->deques[victim], out)) {
                return true;
            }
        }
    }

    return false;
}
