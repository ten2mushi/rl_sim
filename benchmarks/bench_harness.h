/**
 * Benchmark Harness - Shared Infrastructure for RL Engine Benchmarks
 *
 * Header-only library providing:
 * - High-resolution timing (CLOCK_MONOTONIC)
 * - Statistics collection (avg/min/max/stddev/p50/p99)
 * - CLI argument parsing (--drones, --iterations, --warmup, --seed, --verbose)
 * - Fixed-width columnar reporting (machine-parseable)
 * - Arena memory reporting
 * - Scaling tests across drone counts
 * - Performance degradation detection
 */

#ifndef BENCH_HARNESS_H
#define BENCH_HARNESS_H

#include "foundation.h"
#include "drone_state.h"
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Section 1: High-Resolution Timing
 * ============================================================================ */

static inline double bench_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

/* ============================================================================
 * Section 2: Statistics Structure
 * ============================================================================ */

typedef struct BenchStats {
    const char* name;
    uint32_t    drone_count;
    uint32_t    iterations;
    double      avg_ms;
    double      min_ms;
    double      max_ms;
    double      stddev_ms;
    double      p50_ms;
    double      p99_ms;
    double      target_ms;
    bool        passed;
    size_t      persistent_bytes;
    size_t      frame_bytes;
    double      items_per_sec;
} BenchStats;

/* ============================================================================
 * Section 3: Internal Helpers (qsort comparator)
 * ============================================================================ */

static int bench_cmp_double_(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

/* ============================================================================
 * Section 4: Measurement Runner
 * ============================================================================ */

typedef void (*BenchFn)(void* ctx);

static inline BenchStats bench_measure(const char* name, BenchFn fn, void* ctx,
                                        uint32_t warmup, uint32_t iterations,
                                        double target_ms) {
    BenchStats stats = {0};
    stats.name = name;
    stats.iterations = iterations;
    stats.target_ms = target_ms;
    stats.min_ms = 1e9;

    /* Warmup */
    for (uint32_t i = 0; i < warmup; i++) {
        fn(ctx);
    }

    /* Allocate per-iteration timings for percentiles */
    double* times = (double*)malloc(sizeof(double) * iterations);
    if (!times) {
        fprintf(stderr, "bench_measure: failed to allocate timing array\n");
        return stats;
    }

    double total = 0.0;
    for (uint32_t i = 0; i < iterations; i++) {
        double t0 = bench_time_ms();
        fn(ctx);
        double t1 = bench_time_ms();
        double elapsed = t1 - t0;
        times[i] = elapsed;
        total += elapsed;
        if (elapsed < stats.min_ms) stats.min_ms = elapsed;
        if (elapsed > stats.max_ms) stats.max_ms = elapsed;
    }

    stats.avg_ms = total / iterations;

    /* Stddev */
    double variance = 0.0;
    for (uint32_t i = 0; i < iterations; i++) {
        double diff = times[i] - stats.avg_ms;
        variance += diff * diff;
    }
    stats.stddev_ms = sqrt(variance / iterations);

    /* Sort for percentiles */
    qsort(times, iterations, sizeof(double), bench_cmp_double_);
    stats.p50_ms = times[iterations / 2];
    stats.p99_ms = times[(uint32_t)((iterations - 1) * 0.99)];

    stats.passed = stats.avg_ms <= target_ms;

    free(times);
    return stats;
}

/* ============================================================================
 * Section 5: Reporting
 * ============================================================================ */

#define BENCH_HEADER_FMT "%-36s %7s %7s %9s %9s %9s %8s %9s %9s %8s %6s\n"
#define BENCH_ROW_FMT    "%-36s %7u %7u %9.3f %9.3f %9.3f %8.3f %9.3f %9.3f %8.1f %6s\n"

static inline void bench_print_header(void) {
    printf(BENCH_HEADER_FMT,
           "NAME", "DRONES", "ITERS", "AVG_MS", "MIN_MS", "MAX_MS",
           "STDDEV", "P50_MS", "P99_MS", "TARGET", "RESULT");
    printf("-------------------------------------------------------------------"
           "-------------------------------------------------------------------\n");
}

static inline void bench_print_row(const BenchStats* s) {
    printf(BENCH_ROW_FMT,
           s->name, s->drone_count, s->iterations,
           s->avg_ms, s->min_ms, s->max_ms, s->stddev_ms,
           s->p50_ms, s->p99_ms, s->target_ms,
           s->passed ? "PASS" : "FAIL");
}

static inline void bench_print_separator(void) {
    printf("-------------------------------------------------------------------"
           "-------------------------------------------------------------------\n");
}

static inline void bench_print_summary(const BenchStats* results, uint32_t count) {
    uint32_t passed = 0, failed = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (results[i].passed) passed++;
        else failed++;
    }
    printf("\nSummary: %u passed, %u failed out of %u benchmarks\n",
           passed, failed, count);
}

/* ============================================================================
 * Section 6: CLI Parsing
 * ============================================================================ */

typedef struct BenchCLI {
    uint32_t drone_counts[8];
    uint32_t num_drone_counts;
    uint32_t iterations;
    uint32_t warmup;
    bool     verbose;
    uint64_t seed;
} BenchCLI;

static inline BenchCLI bench_default_cli(void) {
    BenchCLI cli = {0};
    cli.drone_counts[0] = 256;
    cli.drone_counts[1] = 512;
    cli.drone_counts[2] = 1024;
    cli.drone_counts[3] = 2048;
    cli.drone_counts[4] = 4096;
    cli.num_drone_counts = 5;
    cli.iterations = 200;
    cli.warmup = 20;
    cli.verbose = false;
    cli.seed = 42;
    return cli;
}

static inline BenchCLI bench_parse_cli(int argc, char** argv) {
    BenchCLI cli = bench_default_cli();

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--drones") == 0 && i + 1 < argc) {
            i++;
            /* Parse comma-separated drone counts */
            cli.num_drone_counts = 0;
            char buf[256];
            strncpy(buf, argv[i], sizeof(buf) - 1);
            buf[sizeof(buf) - 1] = '\0';
            char* tok = strtok(buf, ",");
            while (tok && cli.num_drone_counts < 8) {
                cli.drone_counts[cli.num_drone_counts++] = (uint32_t)atoi(tok);
                tok = strtok(NULL, ",");
            }
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            cli.iterations = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            cli.warmup = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cli.seed = (uint64_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "--verbose") == 0) {
            cli.verbose = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --drones N,N,N    Comma-separated drone counts (default: 256,512,1024,2048,4096)\n");
            printf("  --iterations N    Number of benchmark iterations (default: 200)\n");
            printf("  --warmup N        Number of warmup iterations (default: 20)\n");
            printf("  --seed N          Random seed (default: 42)\n");
            printf("  --verbose         Verbose output\n");
            printf("  --help            Show this help\n");
            exit(0);
        }
    }
    return cli;
}

/* ============================================================================
 * Section 7: Arena Memory Reporting
 * ============================================================================ */

static inline void bench_report_arena(const char* name, const Arena* arena) {
    if (!arena) return;
    printf("  Arena %-20s: %8zu / %8zu bytes  (%.1f%% utilization)\n",
           name, arena->used, arena->capacity,
           arena_utilization(arena) * 100.0f);
}

/* ============================================================================
 * Section 8: Scaling Test Helper
 * ============================================================================ */

typedef BenchStats (*BenchScaleFn)(uint32_t drone_count, uint32_t iterations,
                                    uint32_t warmup, uint64_t seed);

static inline void bench_scaling_test(const char* name, BenchScaleFn fn,
                                       const BenchCLI* cli) {
    printf("\n=== Scaling Test: %s ===\n", name);
    bench_print_header();
    BenchStats prev = {0};
    for (uint32_t i = 0; i < cli->num_drone_counts; i++) {
        BenchStats s = fn(cli->drone_counts[i], cli->iterations, cli->warmup, cli->seed);
        s.drone_count = cli->drone_counts[i];
        bench_print_row(&s);

        /* Check for super-linear scaling */
        if (i > 0 && prev.avg_ms > 0.0001) {
            double count_ratio = (double)cli->drone_counts[i] / cli->drone_counts[i - 1];
            double time_ratio = s.avg_ms / prev.avg_ms;
            if (time_ratio > count_ratio * 1.5) {
                printf("  WARNING: Super-linear scaling detected (%.1fx drones -> %.1fx time)\n",
                       count_ratio, time_ratio);
            }
        }
        prev = s;
    }
}

/* ============================================================================
 * Section 9: Performance Degradation Detection
 * ============================================================================ */

static inline bool bench_check_degradation(const char* name, BenchFn fn, void* ctx,
                                            uint32_t total_iters, uint32_t block_size,
                                            double max_drift_pct) {
    uint32_t num_blocks = total_iters / block_size;
    if (num_blocks < 2) return true;

    double* block_avgs = (double*)malloc(sizeof(double) * num_blocks);
    if (!block_avgs) return true;

    /* Warmup */
    for (uint32_t i = 0; i < block_size; i++) {
        fn(ctx);
    }

    /* Run blocks */
    for (uint32_t b = 0; b < num_blocks; b++) {
        double total = 0.0;
        for (uint32_t i = 0; i < block_size; i++) {
            double t0 = bench_time_ms();
            fn(ctx);
            double t1 = bench_time_ms();
            total += (t1 - t0);
        }
        block_avgs[b] = total / block_size;
    }

    /* Check drift: compare last block to first block */
    double first = block_avgs[0];
    double last = block_avgs[num_blocks - 1];
    double drift_pct = 0.0;
    if (first > 0.0001) {
        drift_pct = ((last - first) / first) * 100.0;
    }

    bool ok = drift_pct <= max_drift_pct;
    printf("  Degradation [%s]: first_block=%.3f ms, last_block=%.3f ms, "
           "drift=%.1f%% (max=%.1f%%) [%s]\n",
           name, first, last, drift_pct, max_drift_pct, ok ? "OK" : "DRIFT");

    free(block_avgs);
    return ok;
}

/* ============================================================================
 * Section 10: Drone State Initialization Helper
 * ============================================================================ */

static inline void bench_init_drones_grid(DroneStateSOA* drones, uint32_t count,
                                           float z_height, uint64_t seed) {
    PCG32 rng;
    pcg32_seed(&rng, seed);
    uint32_t grid_w = (uint32_t)ceilf(sqrtf((float)count));
    for (uint32_t d = 0; d < count; d++) {
        drone_state_init(drones, d);
        drones->pos_x[d] = (float)(d % grid_w) * 2.0f;
        drones->pos_y[d] = (float)(d / grid_w) * 2.0f;
        drones->pos_z[d] = z_height + pcg32_range(&rng, -0.5f, 0.5f);
    }
    drones->count = count;
}

static inline void bench_init_drones_scattered(DroneStateSOA* drones, uint32_t count,
                                                float range, uint64_t seed) {
    PCG32 rng;
    pcg32_seed(&rng, seed);
    for (uint32_t d = 0; d < count; d++) {
        drone_state_init(drones, d);
        drones->pos_x[d] = pcg32_range(&rng, -range, range);
        drones->pos_y[d] = pcg32_range(&rng, -range, range);
        drones->pos_z[d] = pcg32_range(&rng, 1.0f, range * 0.2f);
    }
    drones->count = count;
}

static inline void bench_init_drones_clustered(DroneStateSOA* drones, uint32_t count,
                                                Vec3 center, float cluster_size,
                                                uint64_t seed) {
    PCG32 rng;
    pcg32_seed(&rng, seed);
    float half = cluster_size * 0.5f;
    for (uint32_t d = 0; d < count; d++) {
        drone_state_init(drones, d);
        drones->pos_x[d] = center.x + pcg32_range(&rng, -half, half);
        drones->pos_y[d] = center.y + pcg32_range(&rng, -half, half);
        drones->pos_z[d] = center.z + pcg32_range(&rng, -half, half);
    }
    drones->count = count;
}

#ifdef __cplusplus
}
#endif

#endif /* BENCH_HARNESS_H */
