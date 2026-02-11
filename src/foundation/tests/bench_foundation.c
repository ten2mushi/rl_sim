/**
 * Foundation Module Performance Benchmarks
 *
 * Measures performance of critical operations:
 * - Arena allocation (<10 cycles target)
 * - Arena reset (<5 cycles target)
 * - Quat normalize (<20 cycles target)
 * - PCG32 random (<15 cycles target)
 *
 * Uses high-resolution timing to measure cycle counts.
 */

#include "../include/foundation.h"
#include <stdio.h>
#include <time.h>

/* Number of iterations for benchmarks */
#define WARMUP_ITERATIONS 10000
#define BENCH_ITERATIONS  1000000

/* Prevent compiler from optimizing away results */
static volatile uint64_t sink = 0;
static volatile float fsink = 0.0f;
static volatile void* psink = NULL;

/* ============================================================================
 * Timing Utilities
 * ============================================================================ */

#if defined(__APPLE__)
#include <mach/mach_time.h>

static double get_time_ns(void) {
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) {
        mach_timebase_info(&timebase);
    }
    uint64_t time = mach_absolute_time();
    return (double)time * timebase.numer / timebase.denom;
}

#elif defined(__linux__)
static double get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

#else
static double get_time_ns(void) {
    return (double)clock() / CLOCKS_PER_SEC * 1e9;
}
#endif

/* Estimate CPU frequency for cycle calculation */
static double estimate_cpu_freq_ghz(void) {
    /* Assume modern CPU frequency range */
    /* This is a rough estimate - real measurements would use RDTSC */
    return 3.0;  /* 3 GHz assumption */
}

/* Convert nanoseconds to cycles at given frequency */
static double ns_to_cycles(double ns, double freq_ghz) {
    return ns * freq_ghz;
}

/* ============================================================================
 * Benchmark Functions
 * ============================================================================ */

static void bench_arena_alloc(void) {
    printf("\n=== Arena Allocation Benchmark ===\n");
    printf("Target: <10 cycles per allocation\n\n");

    /* Create a large arena */
    Arena* arena = arena_create(1024 * 1024 * 100);  /* 100MB */
    if (!arena) {
        printf("ERROR: Failed to create arena\n");
        return;
    }

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        psink = arena_alloc(arena, 64);
    }
    arena_reset(arena);

    /* Benchmark */
    double start = get_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        psink = arena_alloc(arena, 64);
    }
    double end = get_time_ns();

    double total_ns = end - start;
    double per_op_ns = total_ns / BENCH_ITERATIONS;
    double freq_ghz = estimate_cpu_freq_ghz();
    double cycles = ns_to_cycles(per_op_ns, freq_ghz);

    printf("  Total time: %.2f ms\n", total_ns / 1e6);
    printf("  Per allocation: %.2f ns\n", per_op_ns);
    printf("  Estimated cycles: %.1f cycles @ %.1f GHz\n", cycles, freq_ghz);
    printf("  Throughput: %.2f M allocs/sec\n", 1e9 / per_op_ns / 1e6);
    printf("  Status: %s\n", cycles < 10 ? "PASS" : "NEEDS OPTIMIZATION");

    arena_destroy(arena);
}

static void bench_arena_reset(void) {
    printf("\n=== Arena Reset Benchmark ===\n");
    printf("Target: <5 cycles per reset\n\n");

    Arena* arena = arena_create(1024 * 1024);
    if (!arena) {
        printf("ERROR: Failed to create arena\n");
        return;
    }

    /* Fill arena somewhat before each reset */
    arena_alloc(arena, 1024);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        arena_reset(arena);
        arena->used = 1024;  /* Simulate usage */
    }

    /* Benchmark */
    double start = get_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        arena_reset(arena);
        arena->used = 1024;  /* Simulate usage */
    }
    double end = get_time_ns();

    double total_ns = end - start;
    double per_op_ns = total_ns / BENCH_ITERATIONS;
    double freq_ghz = estimate_cpu_freq_ghz();
    double cycles = ns_to_cycles(per_op_ns, freq_ghz);

    printf("  Total time: %.2f ms\n", total_ns / 1e6);
    printf("  Per reset: %.2f ns\n", per_op_ns);
    printf("  Estimated cycles: %.1f cycles @ %.1f GHz\n", cycles, freq_ghz);
    printf("  Status: %s\n", cycles < 5 ? "PASS" : "NEEDS OPTIMIZATION");

    arena_destroy(arena);
}

static void bench_quat_normalize(void) {
    printf("\n=== Quaternion Normalize Benchmark ===\n");
    printf("Target: <20 cycles per normalize\n\n");

    Quat q = QUAT(1.0f, 2.0f, 3.0f, 4.0f);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        Quat n = quat_normalize(q);
        fsink = n.w;
    }

    /* Benchmark */
    double start = get_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        Quat n = quat_normalize(q);
        fsink = n.w;
    }
    double end = get_time_ns();

    double total_ns = end - start;
    double per_op_ns = total_ns / BENCH_ITERATIONS;
    double freq_ghz = estimate_cpu_freq_ghz();
    double cycles = ns_to_cycles(per_op_ns, freq_ghz);

    printf("  Total time: %.2f ms\n", total_ns / 1e6);
    printf("  Per normalize: %.2f ns\n", per_op_ns);
    printf("  Estimated cycles: %.1f cycles @ %.1f GHz\n", cycles, freq_ghz);
    printf("  Throughput: %.2f M ops/sec\n", 1e9 / per_op_ns / 1e6);
    printf("  Status: %s\n", cycles < 20 ? "PASS" : "NEEDS OPTIMIZATION");
}

static void bench_quat_rotate(void) {
    printf("\n=== Quaternion Rotate Benchmark ===\n");
    printf("Target: <30 cycles per rotation\n\n");

    Quat q = quat_normalize(QUAT(1.0f, 2.0f, 3.0f, 4.0f));
    Vec3 v = VEC3(1.0f, 2.0f, 3.0f);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        Vec3 r = quat_rotate(q, v);
        fsink = r.x;
    }

    /* Benchmark */
    double start = get_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        Vec3 r = quat_rotate(q, v);
        fsink = r.x;
    }
    double end = get_time_ns();

    double total_ns = end - start;
    double per_op_ns = total_ns / BENCH_ITERATIONS;
    double freq_ghz = estimate_cpu_freq_ghz();
    double cycles = ns_to_cycles(per_op_ns, freq_ghz);

    printf("  Total time: %.2f ms\n", total_ns / 1e6);
    printf("  Per rotation: %.2f ns\n", per_op_ns);
    printf("  Estimated cycles: %.1f cycles @ %.1f GHz\n", cycles, freq_ghz);
    printf("  Throughput: %.2f M ops/sec\n", 1e9 / per_op_ns / 1e6);
    printf("  Status: %s\n", cycles < 30 ? "PASS" : "NEEDS OPTIMIZATION");
}

static void bench_pcg32_random(void) {
    printf("\n=== PCG32 Random Benchmark ===\n");
    printf("Target: <15 cycles per random\n\n");

    PCG32 rng;
    pcg32_seed(&rng, 12345);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        sink = pcg32_random(&rng);
    }

    /* Benchmark */
    double start = get_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        sink = pcg32_random(&rng);
    }
    double end = get_time_ns();

    double total_ns = end - start;
    double per_op_ns = total_ns / BENCH_ITERATIONS;
    double freq_ghz = estimate_cpu_freq_ghz();
    double cycles = ns_to_cycles(per_op_ns, freq_ghz);

    printf("  Total time: %.2f ms\n", total_ns / 1e6);
    printf("  Per random: %.2f ns\n", per_op_ns);
    printf("  Estimated cycles: %.1f cycles @ %.1f GHz\n", cycles, freq_ghz);
    printf("  Throughput: %.2f M randoms/sec\n", 1e9 / per_op_ns / 1e6);
    printf("  Status: %s\n", cycles < 15 ? "PASS" : "NEEDS OPTIMIZATION");
}

static void bench_vec3_operations(void) {
    printf("\n=== Vec3 Operations Benchmark ===\n");
    printf("Target: Native speed (inlined)\n\n");

    Vec3 a = VEC3(1.0f, 2.0f, 3.0f);
    Vec3 b = VEC3(4.0f, 5.0f, 6.0f);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        Vec3 r = vec3_add(vec3_cross(a, b), vec3_scale(a, 2.0f));
        fsink = r.x;
    }

    /* Benchmark compound operation */
    double start = get_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        Vec3 r = vec3_add(vec3_cross(a, b), vec3_scale(a, 2.0f));
        fsink = r.x;
    }
    double end = get_time_ns();

    double total_ns = end - start;
    double per_op_ns = total_ns / BENCH_ITERATIONS;
    double freq_ghz = estimate_cpu_freq_ghz();
    double cycles = ns_to_cycles(per_op_ns, freq_ghz);

    printf("  Operation: add(cross(a,b), scale(a,2))\n");
    printf("  Total time: %.2f ms\n", total_ns / 1e6);
    printf("  Per operation: %.2f ns\n", per_op_ns);
    printf("  Estimated cycles: %.1f cycles @ %.1f GHz\n", cycles, freq_ghz);
    printf("  Throughput: %.2f M ops/sec\n", 1e9 / per_op_ns / 1e6);
}

static void bench_vec3_normalize(void) {
    printf("\n=== Vec3 Normalize Benchmark ===\n");
    printf("Target: Fast (sqrt-bound)\n\n");

    Vec3 v = VEC3(3.0f, 4.0f, 5.0f);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        Vec3 n = vec3_normalize(v);
        fsink = n.x;
    }

    /* Benchmark */
    double start = get_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        Vec3 n = vec3_normalize(v);
        fsink = n.x;
    }
    double end = get_time_ns();

    double total_ns = end - start;
    double per_op_ns = total_ns / BENCH_ITERATIONS;
    double freq_ghz = estimate_cpu_freq_ghz();
    double cycles = ns_to_cycles(per_op_ns, freq_ghz);

    printf("  Total time: %.2f ms\n", total_ns / 1e6);
    printf("  Per normalize: %.2f ns\n", per_op_ns);
    printf("  Estimated cycles: %.1f cycles @ %.1f GHz\n", cycles, freq_ghz);
    printf("  Throughput: %.2f M ops/sec\n", 1e9 / per_op_ns / 1e6);
}

static void bench_simd_array_sum(void) {
    printf("\n=== SIMD Array Sum Benchmark ===\n");
    printf("Comparing scalar vs SIMD for array summation\n\n");

    const uint32_t count = 10000;
    alignas(32) float data[10000];

    /* Initialize data */
    for (uint32_t i = 0; i < count; i++) {
        data[i] = 1.0f;
    }

    /* Scalar sum */
    double scalar_start = get_time_ns();
    for (int iter = 0; iter < 10000; iter++) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < count; i++) {
            sum += data[i];
        }
        fsink = sum;
    }
    double scalar_end = get_time_ns();
    double scalar_ns = (scalar_end - scalar_start) / 10000;

    /* SIMD sum */
    double simd_start = get_time_ns();
    for (int iter = 0; iter < 10000; iter++) {
        float sum = 0.0f;
#if FOUNDATION_SIMD_WIDTH >= 4
        simd_float acc = simd_setzero_ps();
        SIMD_LOOP_START(count);
        for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
            simd_float v = simd_load_ps(&data[i]);
            acc = simd_add_ps(acc, v);
        }
        alignas(32) float acc_arr[8];
        simd_store_ps(acc_arr, acc);
        for (int i = 0; i < FOUNDATION_SIMD_WIDTH; i++) {
            sum += acc_arr[i];
        }
        SIMD_LOOP_REMAINDER(i, count) {
            sum += data[i];
        }
#else
        for (uint32_t i = 0; i < count; i++) {
            sum += data[i];
        }
#endif
        fsink = sum;
    }
    double simd_end = get_time_ns();
    double simd_ns = (simd_end - simd_start) / 10000;

    printf("  Array size: %u floats\n", count);
    printf("  Scalar time: %.2f ns\n", scalar_ns);
    printf("  SIMD time: %.2f ns\n", simd_ns);
    printf("  Speedup: %.2fx\n", scalar_ns / simd_ns);
    printf("  SIMD width: %d\n", FOUNDATION_SIMD_WIDTH);
}

static void bench_memory_throughput(void) {
    printf("\n=== Memory Throughput Benchmark ===\n");
    printf("Testing arena allocation throughput\n\n");

    /* Large arena for throughput test */
    Arena* arena = arena_create(1024 * 1024 * 256);  /* 256MB */
    if (!arena) {
        printf("ERROR: Failed to create arena\n");
        return;
    }

    /* Allocate until full, measuring throughput */
    size_t alloc_size = 64;
    size_t total_allocated = 0;

    double start = get_time_ns();
    while (arena_remaining(arena) >= alloc_size) {
        psink = arena_alloc(arena, alloc_size);
        total_allocated += alloc_size;
    }
    double end = get_time_ns();

    double total_ns = end - start;
    double throughput_gb_s = (total_allocated / 1e9) / (total_ns / 1e9);

    printf("  Total allocated: %.2f MB\n", total_allocated / 1e6);
    printf("  Total time: %.2f ms\n", total_ns / 1e6);
    printf("  Throughput: %.2f GB/s\n", throughput_gb_s);

    arena_destroy(arena);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("=======================================================\n");
    printf("       Foundation Module Performance Benchmarks\n");
    printf("=======================================================\n");
    printf("\nSIMD Platform: ");
#if defined(FOUNDATION_SIMD_AVX2)
    printf("AVX2 (width=%d)\n", FOUNDATION_SIMD_WIDTH);
#elif defined(FOUNDATION_SIMD_NEON)
    printf("ARM NEON (width=%d)\n", FOUNDATION_SIMD_WIDTH);
#else
    printf("Scalar fallback (width=%d)\n", FOUNDATION_SIMD_WIDTH);
#endif

    printf("\nBenchmark iterations: %d\n", BENCH_ITERATIONS);

    /* Run benchmarks */
    bench_arena_alloc();
    bench_arena_reset();
    bench_quat_normalize();
    bench_quat_rotate();
    bench_pcg32_random();
    bench_vec3_operations();
    bench_vec3_normalize();
    bench_simd_array_sum();
    bench_memory_throughput();

    printf("\n=======================================================\n");
    printf("                  Benchmark Complete\n");
    printf("=======================================================\n");

    return 0;
}
