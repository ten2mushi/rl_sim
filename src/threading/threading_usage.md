# Threading Module Usage Guide

High-performance threading primitives for the RL drone engine. Provides work-stealing thread pool, synchronization primitives, and hybrid scheduling.

## Dependencies

```c
#include "threading.h"  // Includes foundation.h automatically
```

Link with: `-lthreading -lfoundation -lpthread`

---

## Quick Start

```c
// Create thread pool (auto-detects CPU count)
ThreadPool* pool = threadpool_create(NULL);

// Define work function
void my_work(void* data, uint32_t start, uint32_t end, uint32_t thread_id) {
    float* arr = (float*)data;
    for (uint32_t i = start; i < end; i++) {
        arr[i] *= 2.0f;
    }
}

// Submit work
WorkItem item = { .fn = my_work, .data = array, .start = 0, .end = 1000 };
threadpool_submit(pool, item);

// Wait for completion
threadpool_wait(pool);

// Cleanup
threadpool_destroy(pool);
```

---

## Configuration Constants

```c
THREADING_MAX_THREADS     64      // Maximum worker threads
THREADING_DEQUE_CAPACITY  1024    // Work items per thread deque
THREADING_CACHE_LINE      64      // Cache line size for alignment
```

---

## API Reference

### Work Function Signature

```c
typedef void (*WorkFunction)(void* data, uint32_t start, uint32_t end, uint32_t thread_id);
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `void*` | User data pointer |
| `start` | `uint32_t` | Start index (inclusive) |
| `end` | `uint32_t` | End index (exclusive) |
| `thread_id` | `uint32_t` | Executing thread ID (0 to num_threads-1) |

### WorkItem Structure

```c
typedef struct WorkItem {
    WorkFunction fn;      // Function to execute
    void*        data;    // User data
    uint32_t     start;   // Range start
    uint32_t     end;     // Range end
} WorkItem;
```

---

## Chase-Lev Work-Stealing Deque

Lock-free deque for work-stealing. Owner pushes/pops from bottom (LIFO), thieves steal from top (FIFO).

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `deque_init` | `WorkDeque* dq` | `void` | Initialize deque (caller allocates memory) |
| `deque_push` | `WorkDeque* dq, WorkItem item` | `bool` | Push item (owner only). Returns false if full |
| `deque_pop` | `WorkDeque* dq, WorkItem* out` | `bool` | Pop item (owner only). Returns false if empty |
| `deque_steal` | `WorkDeque* dq, WorkItem* out` | `bool` | Steal item (thread-safe). Returns false if empty/contended |
| `deque_size` | `const WorkDeque* dq` | `uint32_t` | Approximate item count |
| `deque_empty` | `const WorkDeque* dq` | `bool` | Check if approximately empty |

**Example:**
```c
WorkDeque* dq = aligned_alloc(64, sizeof(WorkDeque));
deque_init(dq);

// Owner thread
deque_push(dq, item);
WorkItem out;
if (deque_pop(dq, &out)) { /* process out */ }

// Thief thread
if (deque_steal(dq, &out)) { /* process stolen item */ }

free(dq);
```

---

## Reusable Barrier

Lock-free spinning barrier with generation counter to prevent spurious wakeups.

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `barrier_init` | `Barrier* b, uint32_t threshold` | `void` | Initialize barrier for `threshold` threads |
| `barrier_wait` | `Barrier* b` | `void` | Block until all threads arrive |
| `barrier_reset` | `Barrier* b` | `void` | Reset for reuse (call when no threads waiting) |

**Example:**
```c
Barrier* b = aligned_alloc(64, sizeof(Barrier));
barrier_init(b, 4);  // 4 threads must arrive

// In each thread:
barrier_wait(b);  // Blocks until 4 threads call barrier_wait

free(b);
```

---

## SPSC Queue (Single-Producer Single-Consumer)

Lock-free bounded queue for exactly one producer and one consumer thread.

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `spsc_create` | `uint32_t capacity` | `SPSCQueue*` | Create queue (capacity rounded to power of 2) |
| `spsc_destroy` | `SPSCQueue* q` | `void` | Destroy queue |
| `spsc_push` | `SPSCQueue* q, void* item` | `bool` | Push item (producer only). Returns false if full |
| `spsc_pop` | `SPSCQueue* q, void** out` | `bool` | Pop item (consumer only). Returns false if empty |
| `spsc_empty` | `const SPSCQueue* q` | `bool` | Check if approximately empty |
| `spsc_size` | `const SPSCQueue* q` | `uint32_t` | Approximate item count |

**Example:**
```c
SPSCQueue* q = spsc_create(1024);

// Producer thread
spsc_push(q, my_data);

// Consumer thread
void* data;
if (spsc_pop(q, &data)) { /* process data */ }

spsc_destroy(q);
```

---

## MPMC Queue (Multi-Producer Multi-Consumer)

Lock-free bounded queue for multiple producers and consumers. Uses sequence numbers for synchronization.

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `mpmc_create` | `uint32_t capacity` | `MPMCQueue*` | Create queue (capacity rounded to power of 2) |
| `mpmc_destroy` | `MPMCQueue* q` | `void` | Destroy queue |
| `mpmc_push` | `MPMCQueue* q, void* item` | `bool` | Push item (thread-safe). Returns false if full |
| `mpmc_pop` | `MPMCQueue* q, void** out` | `bool` | Pop item (thread-safe). Returns false if empty |
| `mpmc_empty` | `const MPMCQueue* q` | `bool` | Check if approximately empty |
| `mpmc_size` | `const MPMCQueue* q` | `uint32_t` | Approximate item count |

**Example:**
```c
MPMCQueue* q = mpmc_create(4096);

// Any producer thread
while (!mpmc_push(q, data)) { threading_yield(); }

// Any consumer thread
void* data;
if (mpmc_pop(q, &data)) { /* process data */ }

mpmc_destroy(q);
```

---

## Thread Pool

Work-stealing thread pool with shared queue and per-thread local deques.

### Configuration

```c
typedef struct ThreadPoolConfig {
    uint32_t num_threads;     // 0 = auto-detect CPU count
    uint32_t queue_capacity;  // 0 = default (4096)
} ThreadPoolConfig;
```

### Statistics

```c
typedef struct ThreadPoolStats {
    uint64_t total_tasks_executed;
    uint64_t total_tasks_stolen;
    uint64_t total_steal_attempts;
    uint64_t per_thread_executed[THREADING_MAX_THREADS];
    uint64_t per_thread_stolen[THREADING_MAX_THREADS];
} ThreadPoolStats;
```

### Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `threadpool_create` | `const ThreadPoolConfig* config` | `ThreadPool*` | Create pool (NULL config for defaults) |
| `threadpool_destroy` | `ThreadPool* pool` | `void` | Wait for work and destroy |
| `threadpool_submit` | `ThreadPool* pool, WorkItem item` | `bool` | Submit single item. Returns false if queue full |
| `threadpool_submit_batch` | `ThreadPool* pool, const WorkItem* items, uint32_t count` | `bool` | Submit multiple items |
| `threadpool_wait` | `ThreadPool* pool` | `void` | Block until all pending work completes |
| `threadpool_num_threads` | `const ThreadPool* pool` | `uint32_t` | Get worker thread count |
| `threadpool_pending_tasks` | `const ThreadPool* pool` | `uint32_t` | Get approximate pending task count |
| `threadpool_is_idle` | `const ThreadPool* pool` | `bool` | Check if no pending work |
| `threadpool_get_stats` | `const ThreadPool* pool, ThreadPoolStats* stats` | `void` | Get execution statistics |
| `threadpool_reset_stats` | `ThreadPool* pool` | `void` | Reset statistics to zero |

**Example:**
```c
ThreadPoolConfig config = { .num_threads = 8 };
ThreadPool* pool = threadpool_create(&config);

// Submit work items
for (int i = 0; i < 100; i++) {
    WorkItem item = { process_fn, data, i * 100, (i + 1) * 100 };
    threadpool_submit(pool, item);
}

threadpool_wait(pool);

ThreadPoolStats stats;
threadpool_get_stats(pool, &stats);
printf("Executed: %llu, Stolen: %llu\n",
       stats.total_tasks_executed, stats.total_tasks_stolen);

threadpool_destroy(pool);
```

---

## Scheduler

High-level scheduler providing static partitioning, work-stealing, and adaptive scheduling.

### Scheduling Strategies

```c
typedef enum ScheduleStrategy {
    SCHEDULE_STATIC        = 0,  // Fixed partitioning (best for uniform work)
    SCHEDULE_WORK_STEALING = 1,  // Dynamic load balancing (best for variable work)
    SCHEDULE_ADAPTIVE      = 2   // Auto-select based on work size
} ScheduleStrategy;
```

### Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `scheduler_create` | `ThreadPool* pool` | `Scheduler*` | Create scheduler using given pool |
| `scheduler_destroy` | `Scheduler* sched` | `void` | Destroy scheduler (does not destroy pool) |
| `scheduler_configure` | `Scheduler* sched, uint32_t steal_threshold, uint32_t min_chunk_size` | `void` | Configure adaptive thresholds |
| `scheduler_physics` | `Scheduler* sched, WorkFunction fn, void* data, uint32_t drone_count` | `void` | Static partitioning for physics |
| `scheduler_sensors` | `Scheduler* sched, WorkFunction fn, void* data, const uint32_t* work_sizes, uint32_t num_types` | `void` | Work-stealing for sensors |
| `scheduler_execute` | `Scheduler* sched, WorkFunction fn, void* data, uint32_t work_count, ScheduleStrategy strategy` | `void` | Execute with specified strategy |
| `scheduler_parallel_for` | `Scheduler* sched, WorkFunction fn, void* data, uint32_t start, uint32_t end, ScheduleStrategy strategy` | `void` | Parallel for loop over range |

**Example - Physics (Static Partitioning):**
```c
void integrate_physics(void* data, uint32_t start, uint32_t end, uint32_t tid) {
    PhysicsState* state = (PhysicsState*)data;
    for (uint32_t i = start; i < end; i++) {
        // Integrate drone i
    }
}

ThreadPool* pool = threadpool_create(NULL);
Scheduler* sched = scheduler_create(pool);

scheduler_physics(sched, integrate_physics, physics_state, num_drones);

scheduler_destroy(sched);
threadpool_destroy(pool);
```

**Example - Sensors (Work-Stealing):**
```c
void compute_sensors(void* data, uint32_t start, uint32_t end, uint32_t tid) {
    SensorData* sensors = (SensorData*)data;
    for (uint32_t i = start; i < end; i++) {
        // Compute sensor i
    }
}

uint32_t work_sizes[] = { 100, 200, 150 };  // Per sensor type
scheduler_sensors(sched, compute_sensors, sensor_data, work_sizes, 3);
```

**Example - Generic Parallel For:**
```c
scheduler_parallel_for(sched, my_fn, data, 0, 10000, SCHEDULE_ADAPTIVE);
```

---

## Utility Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `threading_hardware_concurrency` | `void` | `uint32_t` | Get hardware thread count |
| `threading_set_affinity` | `pthread_t thread, uint32_t core_id` | `void` | Pin thread to CPU core (Linux only) |
| `threading_set_priority` | `pthread_t thread, int priority` | `void` | Set thread scheduling priority |
| `threading_yield` | `void` | `void` | Yield current thread |
| `threading_spin_wait` | `uint32_t iterations` | `void` | Spin-wait with pause hints |
| `threading_pause` | `void` | `void` | CPU pause hint for spin loops |

---

## Memory Size Helpers

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `threading_pool_memory_size` | `uint32_t num_threads, uint32_t queue_capacity` | `size_t` | Bytes needed for thread pool |
| `threading_spsc_memory_size` | `uint32_t capacity` | `size_t` | Bytes needed for SPSC queue |
| `threading_mpmc_memory_size` | `uint32_t capacity` | `size_t` | Bytes needed for MPMC queue |

---

## Best Practices

1. **Use `SCHEDULE_STATIC` for uniform work** (physics integration)
2. **Use `SCHEDULE_WORK_STEALING` for variable work** (sensor computation with different costs)
3. **Use `SCHEDULE_ADAPTIVE`** when unsure - it selects based on work count
4. **Avoid very small work items** - overhead dominates for items < 1000 ops
5. **Use `thread_id` parameter** for thread-local scratch buffers to avoid false sharing
6. **Batch submissions** when possible using `threadpool_submit_batch`

---

## Thread Safety Summary

| Component | Thread Safety |
|-----------|---------------|
| `WorkDeque` | Owner: push/pop. Any thread: steal |
| `Barrier` | All threads can call `barrier_wait` |
| `SPSCQueue` | One producer, one consumer only |
| `MPMCQueue` | Any number of producers/consumers |
| `ThreadPool` | All functions are thread-safe |
| `Scheduler` | All functions are thread-safe |
