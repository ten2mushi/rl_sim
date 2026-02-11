/**
 * Threading Module Benchmarks
 *
 * Measures performance of:
 * - Deque push/pop/steal latency
 * - SPSC/MPMC queue throughput
 * - Thread pool submission and throughput
 * - Scheduler execution with different strategies
 * - Scaling efficiency across thread counts
 */

#include "threading.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/* ============================================================================
 * Timing Infrastructure
 * ============================================================================ */

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

#define BENCH_WARMUP_ITERATIONS 1000
#define BENCH_ITERATIONS 100000

typedef struct {
    const char* name;
    uint64_t total_ns;
    uint64_t iterations;
    double ns_per_op;
    double ops_per_sec;
} BenchResult;

static void print_result(const BenchResult* r) {
    printf("  %-40s %8.1f ns/op  %10.0f ops/sec\n",
           r->name, r->ns_per_op, r->ops_per_sec);
}

/* ============================================================================
 * Work Functions
 * ============================================================================ */

static void noop_fn(void* data, uint32_t start, uint32_t end, uint32_t thread_id) {
    (void)data;
    (void)start;
    (void)end;
    (void)thread_id;
}

static void increment_fn(void* data, uint32_t start, uint32_t end, uint32_t thread_id) {
    (void)start;
    (void)end;
    (void)thread_id;
    atomic_u32* counter = (atomic_u32*)data;
    atomic_add_relaxed(counter, 1);
}

static void sum_range_fn(void* data, uint32_t start, uint32_t end, uint32_t thread_id) {
    (void)thread_id;
    atomic_u64* sum = (atomic_u64*)data;
    uint64_t local = 0;
    for (uint32_t i = start; i < end; i++) {
        local += i;
    }
    atomic_add_relaxed(sum, local);
}

/* ============================================================================
 * Deque Benchmarks
 * ============================================================================ */

static BenchResult bench_deque_push_pop(void) {
    WorkDeque* dq = (WorkDeque*)aligned_alloc(64, sizeof(WorkDeque));
    deque_init(dq);
    WorkItem item = { noop_fn, NULL, 0, 1 };
    WorkItem out;

    /* Warmup */
    for (uint32_t i = 0; i < BENCH_WARMUP_ITERATIONS; i++) {
        deque_push(dq, item);
        deque_pop(dq, &out);
    }

    uint64_t start = get_time_ns();
    for (uint32_t i = 0; i < BENCH_ITERATIONS; i++) {
        deque_push(dq, item);
        deque_pop(dq, &out);
    }
    uint64_t elapsed = get_time_ns() - start;

    free(dq);

    BenchResult r = {
        .name = "deque_push_pop",
        .total_ns = elapsed,
        .iterations = BENCH_ITERATIONS,
        .ns_per_op = (double)elapsed / BENCH_ITERATIONS,
        .ops_per_sec = (double)BENCH_ITERATIONS * 1e9 / elapsed
    };
    return r;
}

static BenchResult bench_deque_steal(void) {
    WorkDeque* dq = (WorkDeque*)aligned_alloc(64, sizeof(WorkDeque));
    deque_init(dq);
    WorkItem item = { noop_fn, NULL, 0, 1 };
    WorkItem out;

    /* Pre-fill */
    for (uint32_t i = 0; i < THREADING_DEQUE_CAPACITY / 2; i++) {
        deque_push(dq, item);
    }

    uint64_t start = get_time_ns();
    uint32_t stolen = 0;
    while (deque_steal(dq, &out)) {
        stolen++;
    }
    uint64_t elapsed = get_time_ns() - start;

    free(dq);

    BenchResult r = {
        .name = "deque_steal",
        .total_ns = elapsed,
        .iterations = stolen,
        .ns_per_op = (stolen > 0) ? (double)elapsed / stolen : 0,
        .ops_per_sec = (stolen > 0) ? (double)stolen * 1e9 / elapsed : 0
    };
    return r;
}

/* ============================================================================
 * SPSC Queue Benchmarks
 * ============================================================================ */

typedef struct {
    SPSCQueue* queue;
    uint32_t count;
    atomic_flag_t* done;
} SPSCBenchArgs;

static void* spsc_consumer_bench(void* arg) {
    SPSCBenchArgs* args = (SPSCBenchArgs*)arg;
    void* item;
    uint32_t consumed = 0;

    while (consumed < args->count) {
        if (spsc_pop(args->queue, &item)) {
            consumed++;
        }
    }

    return NULL;
}

static BenchResult bench_spsc_throughput(void) {
    SPSCQueue* q = spsc_create(4096);
    const uint32_t N = 1000000;
    atomic_flag_t done = ATOMIC_VAR_INIT(false);

    SPSCBenchArgs args = { q, N, &done };
    pthread_t consumer;
    pthread_create(&consumer, NULL, spsc_consumer_bench, &args);

    uint64_t start = get_time_ns();
    for (uint32_t i = 0; i < N; i++) {
        while (!spsc_push(q, (void*)(uintptr_t)i)) {
            /* busy wait */
        }
    }

    pthread_join(consumer, NULL);
    uint64_t elapsed = get_time_ns() - start;

    spsc_destroy(q);

    BenchResult r = {
        .name = "spsc_throughput",
        .total_ns = elapsed,
        .iterations = N,
        .ns_per_op = (double)elapsed / N,
        .ops_per_sec = (double)N * 1e9 / elapsed
    };
    return r;
}

/* ============================================================================
 * MPMC Queue Benchmarks
 * ============================================================================ */

typedef struct {
    MPMCQueue* queue;
    atomic_u32* produced;
    atomic_u32* consumed;
    uint32_t ops_per_thread;
} MPMCBenchArgs;

static void* mpmc_producer_bench(void* arg) {
    MPMCBenchArgs* args = (MPMCBenchArgs*)arg;
    for (uint32_t i = 0; i < args->ops_per_thread; i++) {
        while (!mpmc_push(args->queue, (void*)(uintptr_t)i)) {
            threading_pause();
        }
        atomic_add_relaxed(args->produced, 1);
    }
    return NULL;
}

static void* mpmc_consumer_bench(void* arg) {
    MPMCBenchArgs* args = (MPMCBenchArgs*)arg;
    void* item;
    while (atomic_load_relaxed(args->consumed) < args->ops_per_thread * 4) {
        if (mpmc_pop(args->queue, &item)) {
            atomic_add_relaxed(args->consumed, 1);
        }
    }
    return NULL;
}

static BenchResult bench_mpmc_throughput(void) {
    MPMCQueue* q = mpmc_create(4096);
    const uint32_t N_THREADS = 4;
    const uint32_t OPS_PER = 100000;

    atomic_u32 produced = ATOMIC_VAR_INIT(0);
    atomic_u32 consumed = ATOMIC_VAR_INIT(0);
    MPMCBenchArgs args = { q, &produced, &consumed, OPS_PER };

    pthread_t producers[N_THREADS];
    pthread_t consumers[N_THREADS];

    uint64_t start = get_time_ns();

    for (uint32_t i = 0; i < N_THREADS; i++) {
        pthread_create(&producers[i], NULL, mpmc_producer_bench, &args);
        pthread_create(&consumers[i], NULL, mpmc_consumer_bench, &args);
    }

    for (uint32_t i = 0; i < N_THREADS; i++) {
        pthread_join(producers[i], NULL);
        pthread_join(consumers[i], NULL);
    }

    uint64_t elapsed = get_time_ns() - start;

    mpmc_destroy(q);

    uint32_t total_ops = N_THREADS * OPS_PER;
    BenchResult r = {
        .name = "mpmc_throughput (4p4c)",
        .total_ns = elapsed,
        .iterations = total_ops,
        .ns_per_op = (double)elapsed / total_ops,
        .ops_per_sec = (double)total_ops * 1e9 / elapsed
    };
    return r;
}

/* ============================================================================
 * Thread Pool Benchmarks
 * ============================================================================ */

static BenchResult bench_threadpool_submit(void) {
    ThreadPoolConfig config = { .num_threads = 4 };
    ThreadPool* pool = threadpool_create(&config);
    atomic_u32 counter = ATOMIC_VAR_INIT(0);

    /* Warmup */
    for (uint32_t i = 0; i < BENCH_WARMUP_ITERATIONS; i++) {
        WorkItem item = { noop_fn, &counter, 0, 1 };
        threadpool_submit(pool, item);
    }
    threadpool_wait(pool);

    uint64_t start = get_time_ns();
    for (uint32_t i = 0; i < BENCH_ITERATIONS; i++) {
        WorkItem item = { noop_fn, &counter, 0, 1 };
        threadpool_submit(pool, item);
    }
    uint64_t submit_time = get_time_ns() - start;

    threadpool_wait(pool);
    threadpool_destroy(pool);

    BenchResult r = {
        .name = "threadpool_submit",
        .total_ns = submit_time,
        .iterations = BENCH_ITERATIONS,
        .ns_per_op = (double)submit_time / BENCH_ITERATIONS,
        .ops_per_sec = (double)BENCH_ITERATIONS * 1e9 / submit_time
    };
    return r;
}

static BenchResult bench_threadpool_throughput(uint32_t num_threads) {
    ThreadPoolConfig config = { .num_threads = num_threads };
    ThreadPool* pool = threadpool_create(&config);
    atomic_u32 counter = ATOMIC_VAR_INIT(0);
    const uint32_t N = 100000;

    uint64_t start = get_time_ns();
    for (uint32_t i = 0; i < N; i++) {
        WorkItem item = { increment_fn, &counter, 0, 1 };
        threadpool_submit(pool, item);
    }
    threadpool_wait(pool);
    uint64_t elapsed = get_time_ns() - start;

    threadpool_destroy(pool);

    static char name[64];
    snprintf(name, sizeof(name), "threadpool_throughput (%u threads)", num_threads);

    BenchResult r = {
        .name = name,
        .total_ns = elapsed,
        .iterations = N,
        .ns_per_op = (double)elapsed / N,
        .ops_per_sec = (double)N * 1e9 / elapsed
    };
    return r;
}

/* ============================================================================
 * Scheduler Benchmarks
 * ============================================================================ */

static BenchResult bench_scheduler_static(uint32_t work_count) {
    ThreadPoolConfig config = { .num_threads = 4 };
    ThreadPool* pool = threadpool_create(&config);
    Scheduler* sched = scheduler_create(pool);
    atomic_u64 sum = ATOMIC_VAR_INIT(0);

    /* Warmup */
    for (uint32_t i = 0; i < 10; i++) {
        atomic_store_relaxed(&sum, 0);
        scheduler_execute(sched, sum_range_fn, &sum, work_count, SCHEDULE_STATIC);
    }

    const uint32_t ITERATIONS = 100;
    uint64_t start = get_time_ns();
    for (uint32_t i = 0; i < ITERATIONS; i++) {
        atomic_store_relaxed(&sum, 0);
        scheduler_execute(sched, sum_range_fn, &sum, work_count, SCHEDULE_STATIC);
    }
    uint64_t elapsed = get_time_ns() - start;

    scheduler_destroy(sched);
    threadpool_destroy(pool);

    static char name[64];
    snprintf(name, sizeof(name), "scheduler_static (%u items)", work_count);

    BenchResult r = {
        .name = name,
        .total_ns = elapsed,
        .iterations = ITERATIONS,
        .ns_per_op = (double)elapsed / ITERATIONS,
        .ops_per_sec = (double)ITERATIONS * 1e9 / elapsed
    };
    return r;
}

static BenchResult bench_scheduler_stealing(uint32_t work_count) {
    ThreadPoolConfig config = { .num_threads = 4 };
    ThreadPool* pool = threadpool_create(&config);
    Scheduler* sched = scheduler_create(pool);
    atomic_u64 sum = ATOMIC_VAR_INIT(0);

    /* Warmup */
    for (uint32_t i = 0; i < 10; i++) {
        atomic_store_relaxed(&sum, 0);
        scheduler_execute(sched, sum_range_fn, &sum, work_count, SCHEDULE_WORK_STEALING);
    }

    const uint32_t ITERATIONS = 100;
    uint64_t start = get_time_ns();
    for (uint32_t i = 0; i < ITERATIONS; i++) {
        atomic_store_relaxed(&sum, 0);
        scheduler_execute(sched, sum_range_fn, &sum, work_count, SCHEDULE_WORK_STEALING);
    }
    uint64_t elapsed = get_time_ns() - start;

    scheduler_destroy(sched);
    threadpool_destroy(pool);

    static char name[64];
    snprintf(name, sizeof(name), "scheduler_stealing (%u items)", work_count);

    BenchResult r = {
        .name = name,
        .total_ns = elapsed,
        .iterations = ITERATIONS,
        .ns_per_op = (double)elapsed / ITERATIONS,
        .ops_per_sec = (double)ITERATIONS * 1e9 / elapsed
    };
    return r;
}

/* ============================================================================
 * Scaling Benchmark
 * ============================================================================ */

static void bench_scaling(void) {
    printf("\nScaling Efficiency:\n");
    printf("  %-10s %12s %12s %12s\n", "Threads", "Time (ms)", "Speedup", "Efficiency");
    printf("  %s\n", "----------------------------------------------------");

    const uint32_t WORK = 1000000;
    double baseline_time = 0;

    for (uint32_t threads = 1; threads <= 8; threads++) {
        ThreadPoolConfig config = { .num_threads = threads };
        ThreadPool* pool = threadpool_create(&config);
        Scheduler* sched = scheduler_create(pool);
        atomic_u64 sum = ATOMIC_VAR_INIT(0);

        /* Warmup */
        scheduler_execute(sched, sum_range_fn, &sum, WORK, SCHEDULE_WORK_STEALING);

        /* Measure */
        const uint32_t ITERATIONS = 10;
        uint64_t start = get_time_ns();
        for (uint32_t i = 0; i < ITERATIONS; i++) {
            atomic_store_relaxed(&sum, 0);
            scheduler_execute(sched, sum_range_fn, &sum, WORK, SCHEDULE_WORK_STEALING);
        }
        uint64_t elapsed = get_time_ns() - start;
        double avg_ms = (double)elapsed / ITERATIONS / 1e6;

        if (threads == 1) {
            baseline_time = avg_ms;
        }

        double speedup = baseline_time / avg_ms;
        double efficiency = speedup / threads * 100;

        printf("  %-10u %12.3f %12.2fx %11.1f%%\n",
               threads, avg_ms, speedup, efficiency);

        scheduler_destroy(sched);
        threadpool_destroy(pool);
    }
}

/* ============================================================================
 * Barrier Benchmark
 * ============================================================================ */

typedef struct {
    Barrier* barrier;
    uint32_t iterations;
} BarrierBenchArgs;

static void* barrier_bench_thread(void* arg) {
    BarrierBenchArgs* args = (BarrierBenchArgs*)arg;
    for (uint32_t i = 0; i < args->iterations; i++) {
        barrier_wait(args->barrier);
    }
    return NULL;
}

static BenchResult bench_barrier(uint32_t num_threads) {
    Barrier* b = (Barrier*)aligned_alloc(64, sizeof(Barrier));
    barrier_init(b, num_threads);

    const uint32_t ITERATIONS = 10000;
    BarrierBenchArgs args = { b, ITERATIONS };

    pthread_t threads[16];
    for (uint32_t i = 0; i < num_threads - 1; i++) {
        pthread_create(&threads[i], NULL, barrier_bench_thread, &args);
    }

    uint64_t start = get_time_ns();
    for (uint32_t i = 0; i < ITERATIONS; i++) {
        barrier_wait(b);
    }
    uint64_t elapsed = get_time_ns() - start;

    for (uint32_t i = 0; i < num_threads - 1; i++) {
        pthread_join(threads[i], NULL);
    }

    free(b);

    static char name[64];
    snprintf(name, sizeof(name), "barrier_wait (%u threads)", num_threads);

    BenchResult r = {
        .name = name,
        .total_ns = elapsed,
        .iterations = ITERATIONS,
        .ns_per_op = (double)elapsed / ITERATIONS,
        .ops_per_sec = (double)ITERATIONS * 1e9 / elapsed
    };
    return r;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("Threading Module Benchmarks\n");
    printf("===========================\n\n");

    /* Deque benchmarks */
    printf("Deque:\n");
    print_result(&(BenchResult){0}); /* Header spacing */
    BenchResult r;

    r = bench_deque_push_pop();
    print_result(&r);

    r = bench_deque_steal();
    print_result(&r);

    /* Queue benchmarks */
    printf("\nQueues:\n");

    r = bench_spsc_throughput();
    print_result(&r);

    r = bench_mpmc_throughput();
    print_result(&r);

    /* Barrier benchmarks */
    printf("\nBarrier:\n");

    r = bench_barrier(2);
    print_result(&r);

    r = bench_barrier(4);
    print_result(&r);

    r = bench_barrier(8);
    print_result(&r);

    /* Thread pool benchmarks */
    printf("\nThread Pool:\n");

    r = bench_threadpool_submit();
    print_result(&r);

    r = bench_threadpool_throughput(1);
    print_result(&r);

    r = bench_threadpool_throughput(4);
    print_result(&r);

    r = bench_threadpool_throughput(8);
    print_result(&r);

    /* Scheduler benchmarks */
    printf("\nScheduler:\n");

    r = bench_scheduler_static(1000);
    print_result(&r);

    r = bench_scheduler_static(10000);
    print_result(&r);

    r = bench_scheduler_stealing(1000);
    print_result(&r);

    r = bench_scheduler_stealing(10000);
    print_result(&r);

    /* Scaling benchmark */
    bench_scaling();

    printf("\n");
    return 0;
}
