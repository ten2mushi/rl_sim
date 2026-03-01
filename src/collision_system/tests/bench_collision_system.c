/**
 * Collision System Module Performance Benchmarks
 *
 * Performance targets (1024 drones):
 * - Spatial hash clear: <1us
 * - Spatial hash build: <100us
 * - Drone-drone detection (sparse): <300us
 * - Drone-drone detection (clustered): <500us
 * - World collision batch: <200us
 * - Collision response (world): <100us
 * - Collision response (drone): <50us
 * - Total collision frame: <1ms
 * - KNN single (k=8): <10us
 * - KNN batch (1024 x k=8): <10ms
 */

#include "../include/collision_system.h"
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* Benchmark parameters */
#define BENCH_ITERATIONS 1000
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
    /* ARM counter frequency may differ from CPU frequency */
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
    (void)ghz;
    return (double)ns / 1000.0;
}

#endif

/* Prevent compiler from optimizing away results */
static volatile void* sink;
static volatile uint32_t uint_sink;

/* Estimate CPU frequency */
static double estimate_cpu_ghz(void) {
#if defined(__aarch64__) || defined(_M_ARM64)
    /* ARM: Use system counter frequency estimation */
    uint64_t start = get_cycles();
    struct timespec ts = {0, 100000000};  /* 100ms */
    nanosleep(&ts, NULL);
    uint64_t end = get_cycles();
    uint64_t elapsed = end - start;
    return (double)elapsed / 100000.0 / 1000.0;  /* cycles per us -> GHz */
#else
    /* x86: Measure TSC over known time */
    uint64_t start = get_cycles();
    struct timespec ts = {0, 100000000};  /* 100ms */
    nanosleep(&ts, NULL);
    uint64_t end = get_cycles();
    return (double)(end - start) / 100000000.0;  /* cycles per ns -> GHz */
#endif
}

/* ============================================================================
 * Test Data Setup
 * ============================================================================ */

static RigidBodyStateSOA* create_bench_states(Arena* arena, uint32_t count) {
    RigidBodyStateSOA* states = rigid_body_state_create(arena, count);
    if (states == NULL) return NULL;

    for (uint32_t i = 0; i < count; i++) {
        states->quat_w[i] = 1.0f;
    }
    states->count = count;

    return states;
}

static RigidBodyParamsSOA* create_bench_params(Arena* arena, uint32_t count) {
    RigidBodyParamsSOA* params = rigid_body_params_create(arena, count);
    if (params == NULL) return NULL;

    for (uint32_t i = 0; i < count; i++) {
        rigid_body_params_init(params, i);
    }
    params->count = count;

    return params;
}

static void setup_scattered_positions(RigidBodyStateSOA* states, uint32_t count) {
    /* Drones spread across a 100m x 100m x 10m volume */
    for (uint32_t i = 0; i < count; i++) {
        states->pos_x[i] = (float)(i % 32) * 3.125f;      /* 0-100m */
        states->pos_y[i] = (float)((i / 32) % 32) * 3.125f; /* 0-100m */
        states->pos_z[i] = (float)(i / 1024) * 10.0f;     /* 0-10m */
    }
}

static void setup_clustered_positions(RigidBodyStateSOA* states, uint32_t count) {
    /* Drones clustered in a small area - worst case for collisions */
    for (uint32_t i = 0; i < count; i++) {
        states->pos_x[i] = (float)(i % 32) * 0.05f;       /* Dense cluster */
        states->pos_y[i] = (float)((i / 32) % 32) * 0.05f;
        states->pos_z[i] = (float)(i / 1024) * 0.05f;
    }
}

static void setup_moderate_collision_positions(RigidBodyStateSOA* states, uint32_t count,
                                                float collision_radius) {
    /* Setup where ~10% of drones have collisions */
    uint32_t collision_count = count / 10;

    for (uint32_t i = 0; i < count; i++) {
        if (i < collision_count) {
            /* Create collision pairs */
            uint32_t pair_id = i / 2;
            float base_x = (float)(pair_id % 10) * 5.0f;
            float base_y = (float)((pair_id / 10) % 10) * 5.0f;
            float base_z = (float)(pair_id / 100) * 5.0f;

            if (i % 2 == 0) {
                states->pos_x[i] = base_x;
                states->pos_y[i] = base_y;
                states->pos_z[i] = base_z;
            } else {
                states->pos_x[i] = base_x + collision_radius * 1.5f;
                states->pos_y[i] = base_y;
                states->pos_z[i] = base_z;
            }
        } else {
            /* Non-colliding drones */
            states->pos_x[i] = 100.0f + (float)(i % 32) * 5.0f;
            states->pos_y[i] = (float)((i / 32) % 32) * 5.0f;
            states->pos_z[i] = (float)(i / 1024) * 5.0f;
        }
    }
}

/* ============================================================================
 * Benchmark Functions
 * ============================================================================ */

static void bench_hash_clear(double cpu_ghz) {
    printf("\n--- Spatial Hash Clear Benchmark ---\n");
    printf("Clearing hash table %d times\n", BENCH_ITERATIONS);

    Arena* arena = arena_create(8 * 1024 * 1024);
    SpatialHashGrid* grid = spatial_hash_create(arena, DRONE_COUNT, 1.0f);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        spatial_hash_clear(grid);
    }

    /* Benchmark */
    uint64_t start = get_cycles();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        spatial_hash_clear(grid);
    }
    uint64_t end = get_cycles();

    double avg_cycles = (double)(end - start) / BENCH_ITERATIONS;
    double avg_us = cycles_to_us(avg_cycles, cpu_ghz);

    printf("  Avg cycles: %.1f\n", avg_cycles);
    printf("  Avg time: %.3f us\n", avg_us);
    printf("  Target: <1 us - %s\n", avg_us < 1.0 ? "PASS" : "FAIL");

    arena_destroy(arena);
}

static void bench_hash_build(double cpu_ghz) {
    printf("\n--- Spatial Hash Build Benchmark ---\n");
    printf("Building hash for %d drones %d times\n", DRONE_COUNT, BENCH_ITERATIONS);

    Arena* arena = arena_create(16 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, DRONE_COUNT, 0.1f, 1.0f);
    RigidBodyStateSOA* states = create_bench_states(arena, DRONE_COUNT);
    setup_scattered_positions(states, DRONE_COUNT);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        collision_build_spatial_hash(sys, states, DRONE_COUNT);
    }

    /* Benchmark */
    uint64_t start = get_cycles();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        collision_build_spatial_hash(sys, states, DRONE_COUNT);
    }
    uint64_t end = get_cycles();

    double avg_cycles = (double)(end - start) / BENCH_ITERATIONS;
    double avg_us = cycles_to_us(avg_cycles, cpu_ghz);

    printf("  Avg cycles: %.1f\n", avg_cycles);
    printf("  Avg time: %.3f us\n", avg_us);
    printf("  Target: <100 us - %s\n", avg_us < 100.0 ? "PASS" : "FAIL");

    arena_destroy(arena);
}

static void bench_detect_dd_sparse(double cpu_ghz) {
    printf("\n--- Drone-Drone Detection (Sparse) Benchmark ---\n");
    printf("Detecting collisions for %d scattered drones %d times\n",
           DRONE_COUNT, BENCH_ITERATIONS);

    Arena* arena = arena_create(16 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, DRONE_COUNT, 0.1f, 1.0f);
    RigidBodyStateSOA* states = create_bench_states(arena, DRONE_COUNT);
    setup_scattered_positions(states, DRONE_COUNT);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        collision_build_spatial_hash(sys, states, DRONE_COUNT);
        collision_detect_drone_drone(sys, states, DRONE_COUNT);
    }

    /* Benchmark detection only (hash already built) */
    collision_build_spatial_hash(sys, states, DRONE_COUNT);

    uint64_t start = get_cycles();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        sys->pair_count = 0;  /* Reset without full hash rebuild */
        collision_detect_drone_drone(sys, states, DRONE_COUNT);
    }
    uint64_t end = get_cycles();

    double avg_cycles = (double)(end - start) / BENCH_ITERATIONS;
    double avg_us = cycles_to_us(avg_cycles, cpu_ghz);

    printf("  Collisions found: %u\n", sys->pair_count);
    printf("  Avg cycles: %.1f\n", avg_cycles);
    printf("  Avg time: %.3f us\n", avg_us);
    printf("  Target: <300 us - %s\n", avg_us < 300.0 ? "PASS" : "FAIL");

    arena_destroy(arena);
}

static void bench_detect_dd_clustered(double cpu_ghz) {
    printf("\n--- Drone-Drone Detection (Clustered) Benchmark ---\n");
    printf("Detecting collisions for %d clustered drones %d times\n",
           DRONE_COUNT, BENCH_ITERATIONS);

    Arena* arena = arena_create(16 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, DRONE_COUNT, 0.1f, 1.0f);
    RigidBodyStateSOA* states = create_bench_states(arena, DRONE_COUNT);
    setup_clustered_positions(states, DRONE_COUNT);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        collision_build_spatial_hash(sys, states, DRONE_COUNT);
        collision_detect_drone_drone(sys, states, DRONE_COUNT);
    }

    /* Benchmark */
    collision_build_spatial_hash(sys, states, DRONE_COUNT);

    uint64_t start = get_cycles();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        sys->pair_count = 0;
        collision_detect_drone_drone(sys, states, DRONE_COUNT);
    }
    uint64_t end = get_cycles();

    double avg_cycles = (double)(end - start) / BENCH_ITERATIONS;
    double avg_us = cycles_to_us(avg_cycles, cpu_ghz);

    printf("  Collisions found: %u (capped at %u)\n", sys->pair_count, sys->max_pairs);
    printf("  Avg cycles: %.1f\n", avg_cycles);
    printf("  Avg time: %.3f us\n", avg_us);
    printf("  Target: <500 us - %s\n", avg_us < 500.0 ? "PASS" : "FAIL");

    arena_destroy(arena);
}

static void bench_response_world(double cpu_ghz) {
    printf("\n--- World Collision Response Benchmark ---\n");
    printf("Applying world response for %d drones %d times\n",
           DRONE_COUNT, BENCH_ITERATIONS);

    Arena* arena = arena_create(16 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, DRONE_COUNT, 0.1f, 1.0f);
    RigidBodyStateSOA* states = create_bench_states(arena, DRONE_COUNT);

    /* Setup ~10% of drones colliding with world */
    for (uint32_t i = 0; i < DRONE_COUNT; i++) {
        states->pos_x[i] = (float)i;
        states->pos_y[i] = 0.0f;
        states->pos_z[i] = 0.0f;
        states->vel_x[i] = 0.0f;
        states->vel_y[i] = 0.0f;
        states->vel_z[i] = -5.0f;

        if (i < DRONE_COUNT / 10) {
            sys->drone_world_collision[i] = 1;
            sys->penetration_depth[i] = -0.05f;
            sys->collision_normals[i] = VEC3(0.0f, 0.0f, 1.0f);
        } else {
            sys->drone_world_collision[i] = 0;
        }
    }

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        collision_apply_world_response(sys, states, 0.5f, 1.0f, DRONE_COUNT);
    }

    /* Benchmark */
    uint64_t start = get_cycles();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        collision_apply_world_response(sys, states, 0.5f, 1.0f, DRONE_COUNT);
    }
    uint64_t end = get_cycles();

    double avg_cycles = (double)(end - start) / BENCH_ITERATIONS;
    double avg_us = cycles_to_us(avg_cycles, cpu_ghz);

    printf("  Colliding drones: %u\n", DRONE_COUNT / 10);
    printf("  Avg cycles: %.1f\n", avg_cycles);
    printf("  Avg time: %.3f us\n", avg_us);
    printf("  Target: <100 us - %s\n", avg_us < 100.0 ? "PASS" : "FAIL");

    arena_destroy(arena);
}

static void bench_response_dd(double cpu_ghz) {
    printf("\n--- Drone-Drone Collision Response Benchmark ---\n");
    printf("Applying drone response %d times\n", BENCH_ITERATIONS);

    Arena* arena = arena_create(16 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, DRONE_COUNT, 0.1f, 1.0f);
    RigidBodyStateSOA* states = create_bench_states(arena, DRONE_COUNT);
    RigidBodyParamsSOA* params = create_bench_params(arena, DRONE_COUNT);

    /* Setup moderate collision scenario */
    setup_moderate_collision_positions(states, DRONE_COUNT, 0.1f);

    /* Detect to get realistic collision pairs */
    collision_detect_all(sys, states, NULL, DRONE_COUNT);
    uint32_t collision_count = sys->pair_count;

    printf("  Collision pairs: %u\n", collision_count);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        collision_apply_drone_response(sys, states, params, 0.5f, DRONE_COUNT);
    }

    /* Benchmark */
    uint64_t start = get_cycles();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        collision_apply_drone_response(sys, states, params, 0.5f, DRONE_COUNT);
    }
    uint64_t end = get_cycles();

    double avg_cycles = (double)(end - start) / BENCH_ITERATIONS;
    double avg_us = cycles_to_us(avg_cycles, cpu_ghz);

    printf("  Avg cycles: %.1f\n", avg_cycles);
    printf("  Avg time: %.3f us\n", avg_us);
    printf("  Target: <50 us - %s\n", avg_us < 50.0 ? "PASS" : "FAIL");

    arena_destroy(arena);
}

static void bench_total_frame(double cpu_ghz) {
    printf("\n--- Total Collision Frame Benchmark ---\n");
    printf("Full collision pipeline for %d drones %d times\n",
           DRONE_COUNT, BENCH_ITERATIONS);

    Arena* arena = arena_create(32 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, DRONE_COUNT, 0.1f, 1.0f);
    RigidBodyStateSOA* states = create_bench_states(arena, DRONE_COUNT);
    RigidBodyParamsSOA* params = create_bench_params(arena, DRONE_COUNT);

    /* Realistic scattered positions with some collisions */
    setup_moderate_collision_positions(states, DRONE_COUNT, 0.1f);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        collision_detect_all(sys, states, NULL, DRONE_COUNT);
        collision_apply_response(sys, states, params, 0.5f, 1.0f, DRONE_COUNT);
    }

    /* Benchmark full frame */
    uint64_t start = get_cycles();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        collision_detect_all(sys, states, NULL, DRONE_COUNT);
        collision_apply_response(sys, states, params, 0.5f, 1.0f, DRONE_COUNT);
    }
    uint64_t end = get_cycles();

    double avg_cycles = (double)(end - start) / BENCH_ITERATIONS;
    double avg_us = cycles_to_us(avg_cycles, cpu_ghz);
    double avg_ms = avg_us / 1000.0;

    printf("  Collision pairs: %u\n", sys->pair_count);
    printf("  Avg cycles: %.1f\n", avg_cycles);
    printf("  Avg time: %.3f us (%.6f ms)\n", avg_us, avg_ms);
    printf("  Target: <1 ms - %s\n", avg_ms < 1.0 ? "PASS" : "FAIL");

    arena_destroy(arena);
}

static void bench_knn_single(double cpu_ghz) {
    printf("\n--- K-Nearest Neighbor (Single Query) Benchmark ---\n");
    printf("Finding k=8 neighbors %d times\n", BENCH_ITERATIONS);

    Arena* arena = arena_create(16 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, DRONE_COUNT, 0.1f, 1.0f);
    RigidBodyStateSOA* states = create_bench_states(arena, DRONE_COUNT);
    setup_scattered_positions(states, DRONE_COUNT);

    collision_build_spatial_hash(sys, states, DRONE_COUNT);

    uint32_t indices[8];
    float distances[8];
    uint32_t count;
    Vec3 query_pos = VEC3(50.0f, 50.0f, 5.0f);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        collision_find_k_nearest(sys, states, query_pos, 8, indices, distances, &count);
    }

    /* Benchmark */
    uint64_t start = get_cycles();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        collision_find_k_nearest(sys, states, query_pos, 8, indices, distances, &count);
        uint_sink = count;
    }
    uint64_t end = get_cycles();

    double avg_cycles = (double)(end - start) / BENCH_ITERATIONS;
    double avg_us = cycles_to_us(avg_cycles, cpu_ghz);

    printf("  Neighbors found: %u\n", count);
    printf("  Avg cycles: %.1f\n", avg_cycles);
    printf("  Avg time: %.3f us\n", avg_us);
    printf("  Target: <10 us - %s\n", avg_us < 10.0 ? "PASS" : "FAIL");

    arena_destroy(arena);
}

static void bench_knn_batch(double cpu_ghz) {
    printf("\n--- K-Nearest Neighbor (Batch) Benchmark ---\n");
    printf("Finding k=8 neighbors for %d drones %d times\n", DRONE_COUNT, 100);

    Arena* arena = arena_create(32 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, DRONE_COUNT, 0.1f, 1.0f);
    RigidBodyStateSOA* states = create_bench_states(arena, DRONE_COUNT);
    setup_scattered_positions(states, DRONE_COUNT);

    collision_build_spatial_hash(sys, states, DRONE_COUNT);

    uint32_t* batch_indices = arena_alloc_array(arena, uint32_t, DRONE_COUNT * 8);
    float* batch_distances = arena_alloc_array(arena, float, DRONE_COUNT * 8);

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        collision_find_k_nearest_batch(sys, states, DRONE_COUNT, 8,
                                       batch_indices, batch_distances);
    }

    /* Benchmark (fewer iterations due to cost) */
    uint64_t start = get_cycles();
    for (int i = 0; i < 100; i++) {
        collision_find_k_nearest_batch(sys, states, DRONE_COUNT, 8,
                                       batch_indices, batch_distances);
    }
    uint64_t end = get_cycles();

    double avg_cycles = (double)(end - start) / 100;
    double avg_us = cycles_to_us(avg_cycles, cpu_ghz);
    double avg_ms = avg_us / 1000.0;

    printf("  Avg cycles: %.1f\n", avg_cycles);
    printf("  Avg time: %.3f us (%.3f ms)\n", avg_us, avg_ms);
    printf("  Target: <10 ms - %s\n", avg_ms < 10.0 ? "PASS" : "FAIL");

    arena_destroy(arena);
}

static void bench_memory_usage(void) {
    printf("\n--- Memory Usage Analysis ---\n");

    size_t hash_memory = spatial_hash_memory_size(DRONE_COUNT);
    size_t sys_memory = collision_memory_size(DRONE_COUNT, DRONE_COUNT * 2);

    printf("  Spatial hash memory (1024 drones): %zu bytes (%zu KB)\n",
           hash_memory, hash_memory / 1024);
    printf("  Collision system memory (1024 drones): %zu bytes (%zu KB)\n",
           sys_memory, sys_memory / 1024);
    printf("  Target: <100 KB - %s\n", sys_memory < 100 * 1024 ? "PASS" : "FAIL");

    /* Breakdown */
    printf("\n  Memory breakdown estimate:\n");
    printf("    cell_heads: %d KB\n", HASH_TABLE_SIZE * 4 / 1024);
    printf("    entries: %d KB\n", DRONE_COUNT * 8 / 1024);
    printf("    collision_pairs: %d KB\n", DRONE_COUNT * 2 * 2 * 4 / 1024);
    printf("    world_collision: %d KB\n", DRONE_COUNT * 1 / 1024);
    printf("    penetration: %d KB\n", DRONE_COUNT * 4 / 1024);
    printf("    normals: %d KB\n", DRONE_COUNT * 16 / 1024);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("\n========================================\n");
    printf("Collision System Performance Benchmarks\n");
    printf("========================================\n");
    printf("Drone count: %d\n", DRONE_COUNT);
    printf("Iterations: %d (warmup: %d)\n", BENCH_ITERATIONS, WARMUP_ITERATIONS);

    /* Estimate CPU frequency */
    printf("\nEstimating CPU frequency...\n");
    double cpu_ghz = estimate_cpu_ghz();
    printf("Estimated frequency: %.2f GHz\n", cpu_ghz);

    /* Run benchmarks */
    bench_hash_clear(cpu_ghz);
    bench_hash_build(cpu_ghz);
    bench_detect_dd_sparse(cpu_ghz);
    bench_detect_dd_clustered(cpu_ghz);
    bench_response_world(cpu_ghz);
    bench_response_dd(cpu_ghz);
    bench_total_frame(cpu_ghz);
    bench_knn_single(cpu_ghz);
    bench_knn_batch(cpu_ghz);
    bench_memory_usage();

    printf("\n========================================\n");
    printf("Benchmarks Complete\n");
    printf("========================================\n\n");

    return 0;
}
