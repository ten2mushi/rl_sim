/**
 * Physics Engine Module Benchmarks
 *
 * Performance benchmarks for the physics engine targeting:
 * - Full RK4 step: <5ms for 1024 drones
 * - Single derivative computation: <1ms
 * - Total physics (4 substeps): <20ms per frame
 *
 * SIMD utilization targets:
 * - Position/velocity integration: >90%
 * - Quaternion operations: >70%
 * - Force/torque computation: >80%
 */

#include "../include/physics.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

/* ============================================================================
 * Timing Utilities
 * ============================================================================ */

/* High-resolution timer */
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

/* Benchmark result structure */
typedef struct {
    const char* name;
    uint32_t drone_count;
    uint32_t iterations;
    double total_ms;
    double mean_ms;
    double min_ms;
    double max_ms;
    double throughput;  /* operations per second */
} BenchResult;

static void print_bench_result(const BenchResult* result) {
    printf("%-35s | %6u drones | %5u iters | mean: %7.3f ms | min: %7.3f ms | max: %7.3f ms | %.1f ops/s\n",
           result->name, result->drone_count, result->iterations,
           result->mean_ms, result->min_ms, result->max_ms, result->throughput);
}

/* ============================================================================
 * Benchmark: Full Physics Step
 * ============================================================================ */

static BenchResult bench_full_step(uint32_t drone_count, uint32_t iterations) {
    BenchResult result = {
        .name = "physics_step (full)",
        .drone_count = drone_count,
        .iterations = iterations,
        .min_ms = 1e9,
        .max_ms = 0.0
    };

    /* Allocate memory */
    size_t arena_size = physics_memory_size(drone_count) +
                        drone_state_memory_size(drone_count) +
                        drone_params_memory_size(drone_count) +
                        drone_count * 4 * sizeof(float) +
                        16 * 1024 * 1024;  /* Extra buffer */

    Arena* persistent = arena_create(arena_size);
    Arena* scratch = arena_create(4 * 1024 * 1024);

    if (persistent == NULL || scratch == NULL) {
        printf("Failed to allocate memory for benchmark\n");
        result.total_ms = -1;
        return result;
    }

    PhysicsConfig config = physics_config_default();
    PhysicsSystem* physics = physics_create(persistent, scratch, &config, drone_count);
    DroneStateSOA* states = drone_state_create(persistent, drone_count);
    DroneParamsSOA* params = drone_params_create(persistent, drone_count);
    float* actions = arena_alloc_array(persistent, float, drone_count * 4);

    if (physics == NULL || states == NULL || params == NULL || actions == NULL) {
        printf("Failed to allocate physics structures\n");
        arena_destroy(scratch);
        arena_destroy(persistent);
        result.total_ms = -1;
        return result;
    }

    /* Initialize */
    for (uint32_t i = 0; i < drone_count; i++) {
        states->pos_z[i] = 10.0f;
        states->quat_w[i] = 1.0f;
        for (int m = 0; m < 4; m++) {
            actions[i * 4 + m] = 0.4f;
        }
    }

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        physics_step(physics, states, params, actions, drone_count);
    }

    /* Benchmark */
    double total = 0.0;
    for (uint32_t i = 0; i < iterations; i++) {
        double start = get_time_ms();
        physics_step(physics, states, params, actions, drone_count);
        double elapsed = get_time_ms() - start;

        total += elapsed;
        if (elapsed < result.min_ms) result.min_ms = elapsed;
        if (elapsed > result.max_ms) result.max_ms = elapsed;
    }

    result.total_ms = total;
    result.mean_ms = total / iterations;
    result.throughput = 1000.0 / result.mean_ms;

    /* Cleanup */
    physics_destroy(physics);
    arena_destroy(scratch);
    arena_destroy(persistent);

    return result;
}

/* ============================================================================
 * Benchmark: Single Derivative Computation
 * ============================================================================ */

static BenchResult bench_derivative_compute(uint32_t drone_count, uint32_t iterations) {
    BenchResult result = {
        .name = "physics_compute_derivatives",
        .drone_count = drone_count,
        .iterations = iterations,
        .min_ms = 1e9,
        .max_ms = 0.0
    };

    size_t arena_size = 2 * drone_state_memory_size(drone_count) +
                        drone_params_memory_size(drone_count) +
                        drone_count * 4 * sizeof(float) +
                        16 * 1024 * 1024;

    Arena* arena = arena_create(arena_size);
    if (arena == NULL) {
        result.total_ms = -1;
        return result;
    }

    DroneStateSOA* states = drone_state_create(arena, drone_count);
    DroneStateSOA* derivatives = drone_state_create(arena, drone_count);
    DroneParamsSOA* params = drone_params_create(arena, drone_count);
    float* actions = arena_alloc_array(arena, float, drone_count * 4);
    PhysicsConfig config = physics_config_default();

    if (states == NULL || derivatives == NULL || params == NULL) {
        arena_destroy(arena);
        result.total_ms = -1;
        return result;
    }

    /* Initialize */
    for (uint32_t i = 0; i < drone_count; i++) {
        states->pos_z[i] = 10.0f;
        states->quat_w[i] = 1.0f;
        states->rpm_0[i] = 1500.0f;
        states->rpm_1[i] = 1500.0f;
        states->rpm_2[i] = 1500.0f;
        states->rpm_3[i] = 1500.0f;
    }

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        physics_compute_derivatives(states, params, actions, derivatives, drone_count, &config, NULL);
    }

    /* Benchmark */
    double total = 0.0;
    for (uint32_t i = 0; i < iterations; i++) {
        double start = get_time_ms();
        physics_compute_derivatives(states, params, actions, derivatives, drone_count, &config, NULL);
        double elapsed = get_time_ms() - start;

        total += elapsed;
        if (elapsed < result.min_ms) result.min_ms = elapsed;
        if (elapsed > result.max_ms) result.max_ms = elapsed;
    }

    result.total_ms = total;
    result.mean_ms = total / iterations;
    result.throughput = 1000.0 / result.mean_ms;

    arena_destroy(arena);
    return result;
}

/* ============================================================================
 * Benchmark: Quaternion Normalization
 * ============================================================================ */

static BenchResult bench_quaternion_normalize(uint32_t drone_count, uint32_t iterations) {
    BenchResult result = {
        .name = "physics_normalize_quaternions",
        .drone_count = drone_count,
        .iterations = iterations,
        .min_ms = 1e9,
        .max_ms = 0.0
    };

    Arena* arena = arena_create(drone_state_memory_size(drone_count) + 1024 * 1024);
    if (arena == NULL) {
        result.total_ms = -1;
        return result;
    }

    DroneStateSOA* states = drone_state_create(arena, drone_count);
    if (states == NULL) {
        arena_destroy(arena);
        result.total_ms = -1;
        return result;
    }

    /* Initialize with slightly unnormalized quaternions */
    for (uint32_t i = 0; i < drone_count; i++) {
        states->quat_w[i] = 1.001f;
        states->quat_x[i] = 0.001f;
        states->quat_y[i] = 0.001f;
        states->quat_z[i] = 0.001f;
    }

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        physics_normalize_quaternions(states, drone_count);
    }

    /* Benchmark */
    double total = 0.0;
    for (uint32_t i = 0; i < iterations; i++) {
        /* Reset quaternions */
        for (uint32_t j = 0; j < drone_count; j++) {
            states->quat_w[j] = 1.001f;
        }

        double start = get_time_ms();
        physics_normalize_quaternions(states, drone_count);
        double elapsed = get_time_ms() - start;

        total += elapsed;
        if (elapsed < result.min_ms) result.min_ms = elapsed;
        if (elapsed > result.max_ms) result.max_ms = elapsed;
    }

    result.total_ms = total;
    result.mean_ms = total / iterations;
    result.throughput = 1000.0 / result.mean_ms;

    arena_destroy(arena);
    return result;
}

/* ============================================================================
 * Benchmark: RK4 Combine
 * ============================================================================ */

static BenchResult bench_rk4_combine(uint32_t drone_count, uint32_t iterations) {
    BenchResult result = {
        .name = "physics_rk4_combine",
        .drone_count = drone_count,
        .iterations = iterations,
        .min_ms = 1e9,
        .max_ms = 0.0
    };

    Arena* arena = arena_create(6 * drone_state_memory_size(drone_count) + 1024 * 1024);
    if (arena == NULL) {
        result.total_ms = -1;
        return result;
    }

    DroneStateSOA* states = drone_state_create(arena, drone_count);
    DroneStateSOA* k1 = drone_state_create(arena, drone_count);
    DroneStateSOA* k2 = drone_state_create(arena, drone_count);
    DroneStateSOA* k3 = drone_state_create(arena, drone_count);
    DroneStateSOA* k4 = drone_state_create(arena, drone_count);

    if (states == NULL || k1 == NULL || k2 == NULL || k3 == NULL || k4 == NULL) {
        arena_destroy(arena);
        result.total_ms = -1;
        return result;
    }

    /* Initialize */
    for (uint32_t i = 0; i < drone_count; i++) {
        states->quat_w[i] = 1.0f;
        k1->pos_x[i] = 1.0f; k2->pos_x[i] = 1.0f;
        k3->pos_x[i] = 1.0f; k4->pos_x[i] = 1.0f;
    }

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        physics_rk4_combine(states, k1, k2, k3, k4, 0.02f, drone_count);
    }

    /* Benchmark */
    double total = 0.0;
    for (uint32_t i = 0; i < iterations; i++) {
        double start = get_time_ms();
        physics_rk4_combine(states, k1, k2, k3, k4, 0.02f, drone_count);
        double elapsed = get_time_ms() - start;

        total += elapsed;
        if (elapsed < result.min_ms) result.min_ms = elapsed;
        if (elapsed > result.max_ms) result.max_ms = elapsed;
    }

    result.total_ms = total;
    result.mean_ms = total / iterations;
    result.throughput = 1000.0 / result.mean_ms;

    arena_destroy(arena);
    return result;
}

/* ============================================================================
 * Benchmark: Force/Torque Computation
 * ============================================================================ */

static BenchResult bench_forces_torques(uint32_t drone_count, uint32_t iterations) {
    BenchResult result = {
        .name = "physics_compute_forces_torques",
        .drone_count = drone_count,
        .iterations = iterations,
        .min_ms = 1e9,
        .max_ms = 0.0
    };

    size_t aligned_array = (drone_count * sizeof(float) + 31) & ~31;
    size_t arena_size = drone_state_memory_size(drone_count) +
                        drone_params_memory_size(drone_count) +
                        6 * aligned_array + 1024 * 1024;

    Arena* arena = arena_create(arena_size);
    if (arena == NULL) {
        result.total_ms = -1;
        return result;
    }

    DroneStateSOA* states = drone_state_create(arena, drone_count);
    DroneParamsSOA* params = drone_params_create(arena, drone_count);
    float* fx = arena_alloc_aligned(arena, aligned_array, 32);
    float* fy = arena_alloc_aligned(arena, aligned_array, 32);
    float* fz = arena_alloc_aligned(arena, aligned_array, 32);
    float* tx = arena_alloc_aligned(arena, aligned_array, 32);
    float* ty = arena_alloc_aligned(arena, aligned_array, 32);
    float* tz = arena_alloc_aligned(arena, aligned_array, 32);

    if (states == NULL || params == NULL || fx == NULL) {
        arena_destroy(arena);
        result.total_ms = -1;
        return result;
    }

    /* Initialize */
    for (uint32_t i = 0; i < drone_count; i++) {
        states->quat_w[i] = 1.0f;
        states->rpm_0[i] = 1500.0f;
        states->rpm_1[i] = 1500.0f;
        states->rpm_2[i] = 1500.0f;
        states->rpm_3[i] = 1500.0f;
    }

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        physics_compute_forces_torques(states, params, fx, fy, fz, tx, ty, tz, drone_count);
    }

    /* Benchmark */
    double total = 0.0;
    for (uint32_t i = 0; i < iterations; i++) {
        double start = get_time_ms();
        physics_compute_forces_torques(states, params, fx, fy, fz, tx, ty, tz, drone_count);
        double elapsed = get_time_ms() - start;

        total += elapsed;
        if (elapsed < result.min_ms) result.min_ms = elapsed;
        if (elapsed > result.max_ms) result.max_ms = elapsed;
    }

    result.total_ms = total;
    result.mean_ms = total / iterations;
    result.throughput = 1000.0 / result.mean_ms;

    arena_destroy(arena);
    return result;
}

/* ============================================================================
 * Benchmark: Scaling Test
 * ============================================================================ */

static void bench_scaling(void) {
    printf("\n=== Scaling Benchmark ===\n");
    printf("Testing physics_step at different drone counts:\n\n");

    uint32_t counts[] = {256, 512, 1024, 2048, 4096, 8192};
    int num_counts = sizeof(counts) / sizeof(counts[0]);

    for (int i = 0; i < num_counts; i++) {
        BenchResult result = bench_full_step(counts[i], 100);
        if (result.total_ms > 0) {
            print_bench_result(&result);
        }
    }
}

/* ============================================================================
 * Benchmark: Memory Throughput
 * ============================================================================ */

static void bench_memory_throughput(void) {
    printf("\n=== Memory Throughput Estimate ===\n");

    uint32_t drone_count = 1024;

    /* Estimate bytes accessed per physics step:
     * - DroneStateSOA: 17 arrays × 4 bytes × drone_count = 68KB
     * - DroneParamsSOA: 15 arrays × 4 bytes × drone_count = 60KB
     * - RK4 requires 4 derivative evaluations
     * - Each derivative touches most arrays multiple times
     *
     * Conservative estimate: ~500KB per physics step for 1024 drones
     */

    BenchResult result = bench_full_step(drone_count, 1000);

    if (result.total_ms > 0) {
        /* Estimate memory accessed per step */
        size_t state_bytes = drone_state_memory_size(drone_count);
        size_t params_bytes = drone_params_memory_size(drone_count);
        size_t bytes_per_step = (state_bytes + params_bytes) * 4;  /* 4 RK4 stages */

        double gb_per_sec = (bytes_per_step / 1e9) * result.throughput;

        printf("Estimated memory accessed per step: %.2f KB\n", bytes_per_step / 1024.0);
        printf("Mean step time: %.3f ms\n", result.mean_ms);
        printf("Estimated memory throughput: %.2f GB/s\n", gb_per_sec);
    }
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("=== Physics Engine Benchmarks ===\n");
    printf("SIMD width: %d floats\n", FOUNDATION_SIMD_WIDTH);
    printf("\n");

    /* Performance targets from spec */
    printf("Performance Targets:\n");
    printf("  - Full RK4 step (1024 drones): <5 ms\n");
    printf("  - Single derivative: <1 ms\n");
    printf("  - Quaternion normalize: <0.2 ms\n");
    printf("  - Total physics (4 substeps): <20 ms\n");
    printf("\n");

    /* Core benchmarks at 1024 drones */
    printf("=== Core Benchmarks (1024 drones, 1000 iterations) ===\n\n");

    BenchResult results[5];
    results[0] = bench_full_step(1024, 1000);
    results[1] = bench_derivative_compute(1024, 1000);
    results[2] = bench_quaternion_normalize(1024, 1000);
    results[3] = bench_rk4_combine(1024, 1000);
    results[4] = bench_forces_torques(1024, 1000);

    for (int i = 0; i < 5; i++) {
        if (results[i].total_ms > 0) {
            print_bench_result(&results[i]);
        }
    }

    /* Performance validation */
    printf("\n=== Performance Validation ===\n");
    if (results[0].mean_ms < 5.0) {
        printf("PASS: Full physics step (%.3f ms) < 5 ms target\n", results[0].mean_ms);
    } else {
        printf("FAIL: Full physics step (%.3f ms) > 5 ms target\n", results[0].mean_ms);
    }

    if (results[1].mean_ms < 1.0) {
        printf("PASS: Derivative computation (%.3f ms) < 1 ms target\n", results[1].mean_ms);
    } else {
        printf("FAIL: Derivative computation (%.3f ms) > 1 ms target\n", results[1].mean_ms);
    }

    if (results[2].mean_ms < 0.2) {
        printf("PASS: Quaternion normalize (%.3f ms) < 0.2 ms target\n", results[2].mean_ms);
    } else {
        printf("FAIL: Quaternion normalize (%.3f ms) > 0.2 ms target\n", results[2].mean_ms);
    }

    /* Scaling test */
    bench_scaling();

    /* Memory throughput */
    bench_memory_throughput();

    printf("\n=== Benchmark Complete ===\n");

    return 0;
}
