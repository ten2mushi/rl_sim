/**
 * Hybrid Scheduler Implementation
 *
 * Provides different scheduling strategies:
 *
 * SCHEDULE_STATIC:
 *   Fixed partitioning where each thread processes work_count/num_threads items.
 *   Best for uniform work (physics) where load is predictable.
 *
 * SCHEDULE_WORK_STEALING:
 *   Work is divided into fine-grained chunks that can be stolen between threads.
 *   Best for variable work (sensors) where some items take longer than others.
 *
 * SCHEDULE_ADAPTIVE:
 *   Automatically selects strategy based on work count:
 *   - Small work (<= steal_threshold): uses static (lower overhead)
 *   - Large work (> steal_threshold): uses work-stealing (better load balance)
 */

#include "threading.h"

/* Default thresholds */
#define DEFAULT_STEAL_THRESHOLD 256
#define DEFAULT_MIN_CHUNK_SIZE  64

/* ============================================================================
 * Scheduler Creation and Destruction
 * ============================================================================ */

Scheduler* scheduler_create(ThreadPool* pool) {
    if (!pool) {
        return NULL;
    }

    Scheduler* sched = (Scheduler*)malloc(sizeof(Scheduler));
    if (!sched) {
        return NULL;
    }

    sched->pool = pool;
    sched->default_strategy = SCHEDULE_ADAPTIVE;
    sched->steal_threshold = DEFAULT_STEAL_THRESHOLD;
    sched->min_chunk_size = DEFAULT_MIN_CHUNK_SIZE;

    return sched;
}

void scheduler_destroy(Scheduler* sched) {
    if (sched) {
        free(sched);
    }
}

void scheduler_configure(Scheduler* sched, uint32_t steal_threshold, uint32_t min_chunk_size) {
    sched->steal_threshold = steal_threshold;
    sched->min_chunk_size = (min_chunk_size > 0) ? min_chunk_size : DEFAULT_MIN_CHUNK_SIZE;
}

/* ============================================================================
 * Static Scheduling (Physics)
 * ============================================================================ */

void scheduler_physics(
    Scheduler* sched,
    WorkFunction physics_fn,
    void* physics_data,
    uint32_t agent_count
) {
    scheduler_execute(sched, physics_fn, physics_data, agent_count, SCHEDULE_STATIC);
}

/* ============================================================================
 * Work-Stealing Scheduling (Sensors)
 * ============================================================================ */

void scheduler_sensors(
    Scheduler* sched,
    WorkFunction sensor_fn,
    void* sensor_data,
    const uint32_t* work_sizes,
    uint32_t num_types
) {
    if (num_types == 0) {
        return;
    }

    ThreadPool* pool = sched->pool;
    uint32_t chunk_size = sched->min_chunk_size;

    /*
     * For sensor work, we create fine-grained tasks for each sensor type
     * to allow work-stealing between threads.
     */
    uint32_t offset = 0;

    for (uint32_t type = 0; type < num_types; type++) {
        uint32_t work = work_sizes[type];
        uint32_t type_start = offset;

        /* Divide this sensor type's work into chunks */
        for (uint32_t i = 0; i < work; i += chunk_size) {
            uint32_t start = type_start + i;
            uint32_t end = start + chunk_size;
            if (end > type_start + work) {
                end = type_start + work;
            }

            WorkItem item = {
                .fn = sensor_fn,
                .data = sensor_data,
                .start = start,
                .end = end
            };

            threadpool_submit(pool, item);
        }

        offset += work;
    }

    threadpool_wait(pool);
}

/* ============================================================================
 * Generic Execution
 * ============================================================================ */

void scheduler_execute(
    Scheduler* sched,
    WorkFunction fn,
    void* data,
    uint32_t work_count,
    ScheduleStrategy strategy
) {
    scheduler_parallel_for(sched, fn, data, 0, work_count, strategy);
}

/* ============================================================================
 * Parallel For
 * ============================================================================ */

void scheduler_parallel_for(
    Scheduler* sched,
    WorkFunction fn,
    void* data,
    uint32_t start,
    uint32_t end,
    ScheduleStrategy strategy
) {
    if (end <= start) {
        return;
    }

    uint32_t work_count = end - start;

    /* Resolve adaptive strategy */
    if (strategy == SCHEDULE_ADAPTIVE) {
        if (work_count <= sched->steal_threshold) {
            strategy = SCHEDULE_STATIC;
        } else {
            strategy = SCHEDULE_WORK_STEALING;
        }
    }

    ThreadPool* pool = sched->pool;

    switch (strategy) {
        case SCHEDULE_STATIC: {
            /* Static partitioning: one task per thread */
            uint32_t num_threads = pool->thread_count;
            uint32_t per_thread = (work_count + num_threads - 1) / num_threads;

            for (uint32_t t = 0; t < num_threads; t++) {
                uint32_t task_start = start + t * per_thread;
                if (task_start >= end) {
                    break;
                }

                uint32_t task_end = task_start + per_thread;
                if (task_end > end) {
                    task_end = end;
                }

                WorkItem item = {
                    .fn = fn,
                    .data = data,
                    .start = task_start,
                    .end = task_end
                };

                threadpool_submit(pool, item);
            }
            break;
        }

        case SCHEDULE_WORK_STEALING: {
            /* Fine-grained chunking for work-stealing */
            uint32_t chunk_size = sched->min_chunk_size;

            for (uint32_t i = start; i < end; i += chunk_size) {
                uint32_t task_end = i + chunk_size;
                if (task_end > end) {
                    task_end = end;
                }

                WorkItem item = {
                    .fn = fn,
                    .data = data,
                    .start = i,
                    .end = task_end
                };

                threadpool_submit(pool, item);
            }
            break;
        }

        default:
            /* SCHEDULE_ADAPTIVE resolved above; unknown strategies are no-ops */
            break;
    }

    threadpool_wait(pool);
}
