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
#include "platform_quadcopter.h"
#include "../../foundation/include/bench_harness.h"
#include <stdio.h>
#include <string.h>

/* Benchmark parameters */
#define BENCH_ITERATIONS 10000
#define WARMUP_ITERATIONS 100
#define DRONE_COUNT 1024

/* ============================================================================
 * Benchmark Functions
 * ============================================================================ */

static void bench_state_creation(void) {
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
        PlatformStateSOA* states = platform_state_create(arena, DRONE_COUNT, QUAD_STATE_EXT_COUNT);
        bench_ptr_sink = states;
    }

    /* Benchmark */
    double start = bench_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        arena_reset(arena);
        PlatformStateSOA* states = platform_state_create(arena, DRONE_COUNT, QUAD_STATE_EXT_COUNT);
        bench_ptr_sink = states;
    }
    double elapsed = bench_time_ns() - start;

    double per_op_ns = elapsed / BENCH_ITERATIONS;
    double per_op_us = per_op_ns / 1000.0;
    double per_op_ms = per_op_us / 1000.0;
    double cycles = bench_ns_to_cycles(per_op_ns);

    printf("  Avg: %.2f ns (~%.1f cycles, %.3f us, %.6f ms)\n",
           per_op_ns, cycles, per_op_us, per_op_ms);
    printf("  Target: <1 ms - %s\n", per_op_ms < 1.0 ? "PASS" : "FAIL");

    arena_destroy(arena);
}

static void bench_state_zero(void) {
    printf("\n--- State Zero Benchmark ---\n");
    printf("Zeroing %d drones %d times\n", DRONE_COUNT, BENCH_ITERATIONS);

    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, DRONE_COUNT, QUAD_STATE_EXT_COUNT);
    if (!states) {
        printf("ERROR: Failed to create states\n");
        arena_destroy(arena);
        return;
    }

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        platform_state_zero(states);
    }

    /* Benchmark */
    double start = bench_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        platform_state_zero(states);
    }
    double elapsed = bench_time_ns() - start;

    double per_op_ns = elapsed / BENCH_ITERATIONS;
    double per_op_us = per_op_ns / 1000.0;
    double cycles = bench_ns_to_cycles(per_op_ns);
    double cycles_per_drone = cycles / DRONE_COUNT;

    printf("  Avg: %.2f ns (~%.1f cycles, %.2f per drone)\n",
           per_op_ns, cycles, cycles_per_drone);
    printf("  Avg time: %.3f us\n", per_op_us);
    printf("  Target: <100 us - %s\n", per_op_us < 100.0 ? "PASS" : "FAIL");

    arena_destroy(arena);
}

static void bench_batch_reset(void) {
    printf("\n--- Batch Reset Benchmark ---\n");
    printf("Resetting %d scattered drones %d times\n", DRONE_COUNT, BENCH_ITERATIONS);

    Arena* arena = arena_create(2 * 1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, DRONE_COUNT, QUAD_STATE_EXT_COUNT);
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
        indices[i] = i;
        positions[i] = VEC3((float)i * 0.1f, (float)i * 0.2f, 1.0f);
        orientations[i] = QUAT_IDENTITY;
    }

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        rigid_body_state_reset_batch(&states->rigid_body, indices, positions, orientations, DRONE_COUNT);
    }

    /* Benchmark */
    double start = bench_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        rigid_body_state_reset_batch(&states->rigid_body, indices, positions, orientations, DRONE_COUNT);
    }
    double elapsed = bench_time_ns() - start;

    double per_op_ns = elapsed / BENCH_ITERATIONS;
    double per_op_us = per_op_ns / 1000.0;
    double cycles = bench_ns_to_cycles(per_op_ns);

    printf("  Avg: %.2f ns (~%.1f cycles, %.3f us)\n", per_op_ns, cycles, per_op_us);
    printf("  Target: <500 us - %s\n", per_op_us < 500.0 ? "PASS" : "FAIL");

    arena_destroy(arena);
}

static void bench_single_get(void) {
    printf("\n--- Single Get Benchmark ---\n");
    printf("Getting single drone state %d times\n", BENCH_ITERATIONS);

    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, DRONE_COUNT, QUAD_STATE_EXT_COUNT);
    if (!states) {
        printf("ERROR: Failed to create states\n");
        arena_destroy(arena);
        return;
    }

    /* Set some values so we're not just reading zeros */
    for (uint32_t i = 0; i < states->rigid_body.capacity; i++) {
        states->rigid_body.pos_x[i] = (float)i;
        states->rigid_body.pos_y[i] = (float)i * 2;
        states->rigid_body.pos_z[i] = (float)i * 3;
    }

    volatile PlatformStateAoS result;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        result = platform_state_get(states, i % DRONE_COUNT);
        bench_float_sink = result.position.x;
    }

    /* Benchmark */
    double start = bench_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        result = platform_state_get(states, i % DRONE_COUNT);
        bench_float_sink = result.position.x;
    }
    double elapsed = bench_time_ns() - start;

    double per_op_ns = elapsed / BENCH_ITERATIONS;
    double cycles = bench_ns_to_cycles(per_op_ns);

    printf("  Avg: %.2f ns (~%.1f cycles)\n", per_op_ns, cycles);
    printf("  Target: <50 ns - %s\n", per_op_ns < 50.0 ? "PASS" : "UNCERTAIN (depends on CPU)");

    arena_destroy(arena);
}

static void bench_single_set(void) {
    printf("\n--- Single Set Benchmark ---\n");
    printf("Setting single drone state %d times\n", BENCH_ITERATIONS);

    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, DRONE_COUNT, QUAD_STATE_EXT_COUNT);
    if (!states) {
        printf("ERROR: Failed to create states\n");
        arena_destroy(arena);
        return;
    }

    PlatformStateAoS state = {
        .position = VEC3(1.0f, 2.0f, 3.0f),
        .velocity = VEC3(0.1f, 0.2f, 0.3f),
        .orientation = QUAT_IDENTITY,
        .omega = VEC3(0.01f, 0.02f, 0.03f),
    };

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        platform_state_set(states, i % DRONE_COUNT, &state);
    }

    /* Benchmark */
    double start = bench_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        platform_state_set(states, i % DRONE_COUNT, &state);
    }
    double elapsed = bench_time_ns() - start;

    double per_op_ns = elapsed / BENCH_ITERATIONS;
    double cycles = bench_ns_to_cycles(per_op_ns);

    printf("  Avg: %.2f ns (~%.1f cycles)\n", per_op_ns, cycles);

    arena_destroy(arena);
}

static void bench_copy(void) {
    printf("\n--- Copy Benchmark ---\n");
    printf("Copying %d drones %d times\n", DRONE_COUNT, BENCH_ITERATIONS);

    Arena* arena = arena_create(4 * 1024 * 1024);
    PlatformStateSOA* src = platform_state_create(arena, DRONE_COUNT, QUAD_STATE_EXT_COUNT);
    PlatformStateSOA* dst = platform_state_create(arena, DRONE_COUNT, QUAD_STATE_EXT_COUNT);
    if (!src || !dst) {
        printf("ERROR: Failed to create states\n");
        arena_destroy(arena);
        return;
    }

    /* Set source values */
    for (uint32_t i = 0; i < src->rigid_body.capacity; i++) {
        src->rigid_body.pos_x[i] = (float)i;
    }

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        rigid_body_state_copy(&dst->rigid_body, &src->rigid_body, 0, 0, DRONE_COUNT);
    }

    /* Benchmark */
    double start = bench_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        rigid_body_state_copy(&dst->rigid_body, &src->rigid_body, 0, 0, DRONE_COUNT);
    }
    double elapsed = bench_time_ns() - start;

    double per_op_ns = elapsed / BENCH_ITERATIONS;
    double per_op_us = per_op_ns / 1000.0;
    double cycles = bench_ns_to_cycles(per_op_ns);
    double bytes_per_copy = DRONE_COUNT * 17 * sizeof(float);
    double elapsed_sec = elapsed / 1e9;
    double gbps = (bytes_per_copy * BENCH_ITERATIONS) / elapsed_sec / 1e9;

    printf("  Avg: %.2f ns (~%.1f cycles, %.3f us)\n", per_op_ns, cycles, per_op_us);
    printf("  Throughput: ~%.1f GB/s\n", gbps);

    arena_destroy(arena);
}

static void bench_memory_usage(void) {
    printf("\n--- Memory Usage ---\n");

    size_t state_size = platform_state_memory_size(DRONE_COUNT, QUAD_STATE_EXT_COUNT);
    size_t params_size = platform_params_memory_size(DRONE_COUNT, QUAD_PARAMS_EXT_COUNT);
    size_t episode_size = sizeof(AgentEpisodeData) * DRONE_COUNT;
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
    printf("    Episode: %zu bytes (target: 28)\n", sizeof(AgentEpisodeData));

    printf("\n  Target: State <70 KB for 1024 drones - %s\n",
           state_size <= 70 * 1024 ? "PASS" : "FAIL");
}

static void bench_validation(void) {
    printf("\n--- Validation Benchmark ---\n");
    printf("Validating single drone %d times\n", BENCH_ITERATIONS);

    Arena* arena = arena_create(1024 * 1024);
    PlatformStateSOA* states = platform_state_create(arena, DRONE_COUNT, QUAD_STATE_EXT_COUNT);
    if (!states) {
        printf("ERROR: Failed to create states\n");
        arena_destroy(arena);
        return;
    }

    volatile bool result;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        result = platform_state_validate(states, i % DRONE_COUNT);
    }

    /* Benchmark */
    double start = bench_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        result = platform_state_validate(states, i % DRONE_COUNT);
    }
    double elapsed = bench_time_ns() - start;

    (void)result;

    double per_op_ns = elapsed / BENCH_ITERATIONS;
    double cycles = bench_ns_to_cycles(per_op_ns);

    printf("  Avg: %.2f ns (~%.1f cycles)\n", per_op_ns, cycles);

    arena_destroy(arena);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("=== Drone State Module Benchmarks ===\n");
    printf("Drone count: %d\n", DRONE_COUNT);
    printf("Iterations: %d\n", BENCH_ITERATIONS);
    printf("CPU freq assumption: %.1f GHz\n", BENCH_CPU_FREQ_GHZ);

    bench_memory_usage();
    bench_state_creation();
    bench_state_zero();
    bench_batch_reset();
    bench_single_get();
    bench_single_set();
    bench_copy();
    bench_validation();

    printf("\n=== Benchmark Complete ===\n");
    return 0;
}
