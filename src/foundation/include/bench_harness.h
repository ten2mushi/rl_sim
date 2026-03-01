/**
 * Benchmark Harness
 *
 * Unified timing infrastructure for all benchmark files.
 * Provides high-resolution timing, CPU frequency estimation,
 * and compiler optimization barriers (sinks).
 *
 * Usage:
 *   #include "bench_harness.h"
 *
 *   double start = bench_time_ns();
 *   // ... work ...
 *   double elapsed = bench_time_ns() - start;
 *   double per_op = elapsed / iterations;
 *   double cycles = bench_ns_to_cycles(per_op);
 */

#ifndef BENCH_HARNESS_H
#define BENCH_HARNESS_H

#include <stdint.h>
#include <stdio.h>

/* ============================================================================
 * High-Resolution Timing
 * ============================================================================ */

#if defined(__APPLE__)
#include <mach/mach_time.h>

static inline double bench_time_ns(void) {
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) {
        mach_timebase_info(&timebase);
    }
    uint64_t time = mach_absolute_time();
    return (double)time * timebase.numer / timebase.denom;
}

#elif defined(__linux__)
#include <time.h>

static inline double bench_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

#else
#include <time.h>

static inline double bench_time_ns(void) {
    return (double)clock() / CLOCKS_PER_SEC * 1e9;
}

#endif

/* ============================================================================
 * CPU Frequency Estimation
 * ============================================================================ */

/** Assumed CPU frequency for cycle estimation. */
#define BENCH_CPU_FREQ_GHZ 3.0

/** Convert nanoseconds to estimated cycles. */
static inline double bench_ns_to_cycles(double ns) {
    return ns * BENCH_CPU_FREQ_GHZ;
}

/* ============================================================================
 * Optimization Barriers (Sinks)
 *
 * Assign benchmark results to these to prevent the compiler from
 * eliminating the computation as dead code.
 * ============================================================================ */

static volatile void*    bench_ptr_sink;
static volatile float    bench_float_sink;
static volatile uint64_t bench_u64_sink;

/* ============================================================================
 * Convenience Macro
 *
 * For simple benchmarks that just need ns/op and estimated cycles.
 * Complex benchmarks with custom setup/reporting should use
 * bench_time_ns() directly.
 * ============================================================================ */

#define BENCH_RUN(name, warmup_n, iter_n, body) do { \
    for (int _w = 0; _w < (warmup_n); _w++) { body; } \
    double _start = bench_time_ns(); \
    for (int _i = 0; _i < (iter_n); _i++) { body; } \
    double _elapsed = bench_time_ns() - _start; \
    double _per_op = _elapsed / (iter_n); \
    double _cycles = bench_ns_to_cycles(_per_op); \
    printf("  %-30s %7.2f ns  ~%.1f cycles\n", name, _per_op, _cycles); \
} while(0)

#endif /* BENCH_HARNESS_H */
