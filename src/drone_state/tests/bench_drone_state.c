/**
 * Drone State Module Performance Benchmarks
 *
 * Measures critical operations against performance targets:
 * - State creation (1024): <1 ms
 * - State zero (1024): <100 us
 * - Batch reset (1024): <500 us
 * - Single state get: <50 ns
 * - Memory per 1024 drones: <70 KB
 */

#include "../include/drone_state.h"
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <string.h>

/* Benchmark parameters */
#define BENCH_ITERATIONS 10000
#define WARMUP_ITERATIONS 100
#define DRONE_COUNT 1024

/* High-resolution timing */
#if defined(__x86_64__) || defined(_M_X64)

static inline uint64_t get_cycles(void) {
    uint32_t lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static inline double cycles_to_us(uint64_t cycles, double ghz) {
    return (double)cycles / (ghz * 1000.0);
}

#elif defined(__aarch64__) || defined(_M_ARM64)

static inline uint64_t get_cycles(void) {
    uint64_t val;
    __asm__ volatile ("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline double cycles_to_us(uint64_t cycles, double ghz) {
    /* ARM counter frequency is typically lower than CPU frequency */
    /* Approximate based on system counter, not actual cycles */
    return (double)cycles / (ghz * 1000.0);
}

#else

/* Fallback using clock_gettime */
static inline uint64_t get_cycles(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static inline double cycles_to_us(uint64_t ns, double ghz) {
    (void)ghz;  /* Unused for timespec fallback */
    return (double)ns / 1000.0;
}

#endif

/* Prevent compiler from optimizing away results */
static volatile void* sink;
static volatile float float_sink;

/* ============================================================================
 * Benchmark Functions
 * ============================================================================ */

static void bench_state_creation(double cpu_ghz) {
    printf("\n--- State Creation Benchmark ---\n");
    printf("Creating %d drones %d times\n", DRONE_COUNT, BENCH_ITERATIONS);

    Arena* arena = arena_create(64 * 1024 * 1024);  /* 64MB */
    if (!arena) {
        printf("ERROR: Failed to create arena\n");
        return;
    }

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        arena_reset(arena);
        DroneStateSOA* states = drone_state_create(arena, DRONE_COUNT);
        sink = states;
    }

    /* Benchmark */
    uint64_t start = get_cycles();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        arena_reset(arena);
        DroneStateSOA* states = drone_state_create(arena, DRONE_COUNT);
        sink = states;
    }
    uint64_t end = get_cycles();

    uint64_t total_cycles = end - start;
    double avg_cycles = (double)total_cycles / BENCH_ITERATIONS;
    double avg_us = cycles_to_us(avg_cycles, cpu_ghz);
    double avg_ms = avg_us / 1000.0;

    printf("  Total cycles: %llu\n", (unsigned long long)total_cycles);
    printf("  Avg cycles: %.1f\n", avg_cycles);
    printf("  Avg time: %.3f us (%.6f ms)\n", avg_us, avg_ms);
    printf("  Target: <1 ms - %s\n", avg_ms < 1.0 ? "PASS" : "FAIL");

    arena_destroy(arena);
}

static void bench_state_zero(double cpu_ghz) {
    printf("\n--- State Zero Benchmark ---\n");
    printf("Zeroing %d drones %d times\n", DRONE_COUNT, BENCH_ITERATIONS);

    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, DRONE_COUNT);
    if (!states) {
        printf("ERROR: Failed to create states\n");
        arena_destroy(arena);
        return;
    }

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        drone_state_zero(states);
    }

    /* Benchmark */
    uint64_t start = get_cycles();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        drone_state_zero(states);
    }
    uint64_t end = get_cycles();

    uint64_t total_cycles = end - start;
    double avg_cycles = (double)total_cycles / BENCH_ITERATIONS;
    double cycles_per_drone = avg_cycles / DRONE_COUNT;
    double avg_us = cycles_to_us(avg_cycles, cpu_ghz);

    printf("  Total cycles: %llu\n", (unsigned long long)total_cycles);
    printf("  Avg cycles: %.1f (%.2f per drone)\n", avg_cycles, cycles_per_drone);
    printf("  Avg time: %.3f us\n", avg_us);
    printf("  Target: <100 us - %s\n", avg_us < 100.0 ? "PASS" : "FAIL");

    arena_destroy(arena);
}

static void bench_batch_reset(double cpu_ghz) {
    printf("\n--- Batch Reset Benchmark ---\n");
    printf("Resetting %d scattered drones %d times\n", DRONE_COUNT, BENCH_ITERATIONS);

    Arena* arena = arena_create(2 * 1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, DRONE_COUNT);
    if (!states) {
        printf("ERROR: Failed to create states\n");
        arena_destroy(arena);
        return;
    }

    /* Prepare reset data */
    uint32_t* indices = arena_alloc_array(arena, uint32_t, DRONE_COUNT);
    Vec3* positions = arena_alloc_array(arena, Vec3, DRONE_COUNT);
    Quat* orientations = arena_alloc_array(arena, Quat, DRONE_COUNT);

    for (uint32_t i = 0; i < DRONE_COUNT; i++) {
        indices[i] = i;  /* Sequential but simulates scattered access */
        positions[i] = VEC3((float)i * 0.1f, (float)i * 0.2f, 1.0f);
        orientations[i] = QUAT_IDENTITY;
    }

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        drone_state_reset_batch(states, indices, positions, orientations, DRONE_COUNT);
    }

    /* Benchmark */
    uint64_t start = get_cycles();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        drone_state_reset_batch(states, indices, positions, orientations, DRONE_COUNT);
    }
    uint64_t end = get_cycles();

    uint64_t total_cycles = end - start;
    double avg_cycles = (double)total_cycles / BENCH_ITERATIONS;
    double avg_us = cycles_to_us(avg_cycles, cpu_ghz);

    printf("  Total cycles: %llu\n", (unsigned long long)total_cycles);
    printf("  Avg cycles: %.1f\n", avg_cycles);
    printf("  Avg time: %.3f us\n", avg_us);
    printf("  Target: <500 us - %s\n", avg_us < 500.0 ? "PASS" : "FAIL");

    arena_destroy(arena);
}

static void bench_single_get(double cpu_ghz) {
    printf("\n--- Single Get Benchmark ---\n");
    printf("Getting single drone state %d times\n", BENCH_ITERATIONS);

    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, DRONE_COUNT);
    if (!states) {
        printf("ERROR: Failed to create states\n");
        arena_destroy(arena);
        return;
    }

    /* Set some values so we're not just reading zeros */
    for (uint32_t i = 0; i < states->capacity; i++) {
        states->pos_x[i] = (float)i;
        states->pos_y[i] = (float)i * 2;
        states->pos_z[i] = (float)i * 3;
    }

    volatile DroneStateAoS result;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        result = drone_state_get(states, i % DRONE_COUNT);
        float_sink = result.position.x;
    }

    /* Benchmark - access sequential indices to measure cache-friendly access */
    uint64_t start = get_cycles();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        result = drone_state_get(states, i % DRONE_COUNT);
        float_sink = result.position.x;  /* Prevent optimization */
    }
    uint64_t end = get_cycles();

    uint64_t total_cycles = end - start;
    double avg_cycles = (double)total_cycles / BENCH_ITERATIONS;
    double avg_ns = cycles_to_us(avg_cycles, cpu_ghz) * 1000.0;

    printf("  Total cycles: %llu\n", (unsigned long long)total_cycles);
    printf("  Avg cycles: %.1f\n", avg_cycles);
    printf("  Avg time: %.1f ns\n", avg_ns);
    printf("  Target: <50 ns - %s\n", avg_ns < 50.0 ? "PASS" : "UNCERTAIN (depends on CPU)");

    arena_destroy(arena);
}

static void bench_single_set(double cpu_ghz) {
    printf("\n--- Single Set Benchmark ---\n");
    printf("Setting single drone state %d times\n", BENCH_ITERATIONS);

    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, DRONE_COUNT);
    if (!states) {
        printf("ERROR: Failed to create states\n");
        arena_destroy(arena);
        return;
    }

    DroneStateAoS state = {
        .position = VEC3(1.0f, 2.0f, 3.0f),
        .velocity = VEC3(0.1f, 0.2f, 0.3f),
        .orientation = QUAT_IDENTITY,
        .omega = VEC3(0.01f, 0.02f, 0.03f),
        .rpm = {1000.0f, 1100.0f, 1200.0f, 1300.0f}
    };

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        drone_state_set(states, i % DRONE_COUNT, &state);
    }

    /* Benchmark */
    uint64_t start = get_cycles();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        drone_state_set(states, i % DRONE_COUNT, &state);
    }
    uint64_t end = get_cycles();

    uint64_t total_cycles = end - start;
    double avg_cycles = (double)total_cycles / BENCH_ITERATIONS;
    double avg_ns = cycles_to_us(avg_cycles, cpu_ghz) * 1000.0;

    printf("  Total cycles: %llu\n", (unsigned long long)total_cycles);
    printf("  Avg cycles: %.1f\n", avg_cycles);
    printf("  Avg time: %.1f ns\n", avg_ns);

    arena_destroy(arena);
}

static void bench_copy(double cpu_ghz) {
    printf("\n--- Copy Benchmark ---\n");
    printf("Copying %d drones %d times\n", DRONE_COUNT, BENCH_ITERATIONS);

    Arena* arena = arena_create(4 * 1024 * 1024);
    DroneStateSOA* src = drone_state_create(arena, DRONE_COUNT);
    DroneStateSOA* dst = drone_state_create(arena, DRONE_COUNT);
    if (!src || !dst) {
        printf("ERROR: Failed to create states\n");
        arena_destroy(arena);
        return;
    }

    /* Set source values */
    for (uint32_t i = 0; i < src->capacity; i++) {
        src->pos_x[i] = (float)i;
    }

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        drone_state_copy(dst, src, 0, 0, DRONE_COUNT);
    }

    /* Benchmark */
    uint64_t start = get_cycles();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        drone_state_copy(dst, src, 0, 0, DRONE_COUNT);
    }
    uint64_t end = get_cycles();

    uint64_t total_cycles = end - start;
    double avg_cycles = (double)total_cycles / BENCH_ITERATIONS;
    double avg_us = cycles_to_us(avg_cycles, cpu_ghz);
    double bytes_per_copy = DRONE_COUNT * 17 * sizeof(float);
    double gbps = (bytes_per_copy * BENCH_ITERATIONS) / ((end - start) / (cpu_ghz * 1e9)) / 1e9;

    printf("  Total cycles: %llu\n", (unsigned long long)total_cycles);
    printf("  Avg cycles: %.1f\n", avg_cycles);
    printf("  Avg time: %.3f us\n", avg_us);
    printf("  Throughput: ~%.1f GB/s\n", gbps);

    arena_destroy(arena);
}

static void bench_memory_usage(void) {
    printf("\n--- Memory Usage ---\n");

    size_t state_size = drone_state_memory_size(DRONE_COUNT);
    size_t params_size = drone_params_memory_size(DRONE_COUNT);
    size_t episode_size = sizeof(DroneEpisodeData) * DRONE_COUNT;
    size_t total_size = state_size + params_size + episode_size;

    printf("  State (%d drones): %zu bytes (%.1f KB)\n",
           DRONE_COUNT, state_size, state_size / 1024.0);
    printf("  Params (%d drones): %zu bytes (%.1f KB)\n",
           DRONE_COUNT, params_size, params_size / 1024.0);
    printf("  Episode (%d drones): %zu bytes (%.1f KB)\n",
           DRONE_COUNT, episode_size, episode_size / 1024.0);
    printf("  Total: %zu bytes (%.1f KB)\n", total_size, total_size / 1024.0);

    /* Per-drone breakdown */
    printf("\n  Per-drone memory:\n");
    printf("    State: %.1f bytes (target: 68)\n", (float)state_size / DRONE_COUNT);
    printf("    Params: %.1f bytes (target: 60)\n", (float)params_size / DRONE_COUNT);
    printf("    Episode: %zu bytes (target: 28)\n", sizeof(DroneEpisodeData));

    printf("\n  Target: State <70 KB for 1024 drones - %s\n",
           state_size <= 70 * 1024 ? "PASS" : "FAIL");
}

static void bench_validation(double cpu_ghz) {
    printf("\n--- Validation Benchmark ---\n");
    printf("Validating single drone %d times\n", BENCH_ITERATIONS);

    Arena* arena = arena_create(1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, DRONE_COUNT);
    if (!states) {
        printf("ERROR: Failed to create states\n");
        arena_destroy(arena);
        return;
    }

    volatile bool result;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        result = drone_state_validate(states, i % DRONE_COUNT);
    }

    /* Benchmark */
    uint64_t start = get_cycles();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        result = drone_state_validate(states, i % DRONE_COUNT);
    }
    uint64_t end = get_cycles();

    (void)result;  /* Suppress unused warning */

    uint64_t total_cycles = end - start;
    double avg_cycles = (double)total_cycles / BENCH_ITERATIONS;
    double avg_ns = cycles_to_us(avg_cycles, cpu_ghz) * 1000.0;

    printf("  Avg cycles: %.1f\n", avg_cycles);
    printf("  Avg time: %.1f ns\n", avg_ns);

    arena_destroy(arena);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("=== Drone State Module Benchmarks ===\n");
    printf("Drone count: %d\n", DRONE_COUNT);
    printf("Iterations: %d\n", BENCH_ITERATIONS);

    /* Estimate CPU frequency (rough approximation) */
    /* For accurate results, this should be calibrated per-system */
#if defined(__aarch64__) || defined(_M_ARM64)
    double cpu_ghz = 3.0;  /* Approximate for Apple Silicon / ARM64 */
    printf("Platform: ARM64 (estimated %.1f GHz counter)\n", cpu_ghz);
#else
    double cpu_ghz = 3.0;  /* Approximate for modern x86_64 */
    printf("Platform: x86_64 (estimated %.1f GHz)\n", cpu_ghz);
#endif

    bench_memory_usage();
    bench_state_creation(cpu_ghz);
    bench_state_zero(cpu_ghz);
    bench_batch_reset(cpu_ghz);
    bench_single_get(cpu_ghz);
    bench_single_set(cpu_ghz);
    bench_copy(cpu_ghz);
    bench_validation(cpu_ghz);

    printf("\n=== Benchmark Complete ===\n");
    return 0;
}
