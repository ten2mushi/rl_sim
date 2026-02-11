/**
 * Reward System Module Performance Benchmarks
 *
 * Performance Targets (1024 drones):
 * - Hover reward: <300 us
 * - Race reward (10 gates): <500 us
 * - Gate crossing check: <200 us
 * - Termination check: <100 us
 * - Total frame: <1 ms
 */

#include "../include/reward_system.h"
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <float.h>

/* Timing utilities */
#ifdef _WIN32
#include <windows.h>
static double get_time_us(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / freq.QuadPart * 1e6;
}
#else
static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}
#endif

#define BENCH_ITERATIONS 1000
#define DRONE_COUNT 1024
#define GATE_COUNT 10

/* Benchmark result tracking */
typedef struct BenchResult {
    const char* name;
    double min_us;
    double max_us;
    double avg_us;
    double target_us;
    int passed;
} BenchResult;

static void print_result(BenchResult* r) {
    const char* status = r->passed ? "PASS" : "FAIL";
    printf("  %-30s  avg: %7.1f us  min: %7.1f us  max: %7.1f us  target: %7.1f us  [%s]\n",
           r->name, r->avg_us, r->min_us, r->max_us, r->target_us, status);
}

/* ============================================================================
 * Benchmark: Hover Reward Computation
 * ============================================================================ */

static BenchResult bench_hover_reward(void) {
    BenchResult result = {
        .name = "Hover Reward (1024 drones)",
        .target_us = 300.0,
        .min_us = 1e9,
        .max_us = 0,
        .avg_us = 0
    };

    Arena* arena = arena_create(16 * 1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, DRONE_COUNT);
    drone_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_HOVER);
    RewardSystem* sys = reward_create(arena, &cfg, DRONE_COUNT, 0);

    /* Initialize random positions */
    PCG32 rng;
    pcg32_seed(&rng, 12345);
    for (uint32_t i = 0; i < DRONE_COUNT; i++) {
        states->pos_x[i] = pcg32_range(&rng, -10.0f, 10.0f);
        states->pos_y[i] = pcg32_range(&rng, -10.0f, 10.0f);
        states->pos_z[i] = pcg32_range(&rng, 0.0f, 10.0f);
    }
    reward_set_targets_random(sys, DRONE_COUNT, VEC3(-10, -10, 0), VEC3(10, 10, 10), &rng);

    float* rewards = arena_alloc_array(arena, float, DRONE_COUNT);
    float* actions = arena_alloc_array(arena, float, DRONE_COUNT * 4);
    memset(actions, 0, DRONE_COUNT * 4 * sizeof(float));

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        reward_compute_hover(sys, states, NULL, actions, NULL, rewards, DRONE_COUNT);
        reward_reset_batch(sys, NULL, 0);
    }

    /* Benchmark */
    double total = 0;
    for (int iter = 0; iter < BENCH_ITERATIONS; iter++) {
        /* Reset for consistent measurements */
        for (uint32_t i = 0; i < DRONE_COUNT; i++) {
            sys->prev_distance[i] = FLT_MAX;
        }

        double start = get_time_us();
        reward_compute_hover(sys, states, NULL, actions, NULL, rewards, DRONE_COUNT);
        double elapsed = get_time_us() - start;

        if (elapsed < result.min_us) result.min_us = elapsed;
        if (elapsed > result.max_us) result.max_us = elapsed;
        total += elapsed;
    }

    result.avg_us = total / BENCH_ITERATIONS;
    result.passed = result.avg_us <= result.target_us;

    arena_destroy(arena);
    return result;
}

/* ============================================================================
 * Benchmark: Race Reward Computation
 * ============================================================================ */

static BenchResult bench_race_reward(void) {
    BenchResult result = {
        .name = "Race Reward (1024 drones, 10 gates)",
        .target_us = 500.0,
        .min_us = 1e9,
        .max_us = 0,
        .avg_us = 0
    };

    Arena* arena = arena_create(16 * 1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, DRONE_COUNT);
    drone_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_RACE);
    RewardSystem* sys = reward_create(arena, &cfg, DRONE_COUNT, GATE_COUNT);

    /* Setup gates */
    Vec3 centers[GATE_COUNT];
    Vec3 normals[GATE_COUNT];
    float radii[GATE_COUNT];
    for (int g = 0; g < GATE_COUNT; g++) {
        centers[g] = VEC3((g + 1) * 10.0f, 0, 5);
        normals[g] = VEC3(1, 0, 0);
        radii[g] = 3.0f;
    }
    reward_set_gates(sys, centers, normals, radii, GATE_COUNT);

    /* Initialize drones */
    PCG32 rng;
    pcg32_seed(&rng, 12345);
    for (uint32_t i = 0; i < DRONE_COUNT; i++) {
        states->pos_x[i] = pcg32_range(&rng, 0.0f, 100.0f);
        states->pos_y[i] = pcg32_range(&rng, -5.0f, 5.0f);
        states->pos_z[i] = pcg32_range(&rng, 2.0f, 8.0f);
        states->vel_x[i] = pcg32_range(&rng, 5.0f, 15.0f);
    }

    float* rewards = arena_alloc_array(arena, float, DRONE_COUNT);
    float* actions = arena_alloc_array(arena, float, DRONE_COUNT * 4);

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        reward_compute_race(sys, states, NULL, actions, NULL, rewards, DRONE_COUNT);
    }

    /* Benchmark */
    double total = 0;
    for (int iter = 0; iter < BENCH_ITERATIONS; iter++) {
        /* Reset gate progress */
        for (uint32_t i = 0; i < DRONE_COUNT; i++) {
            sys->gates->current_gate[i] = 0;
            sys->prev_distance[i] = 50.0f;
        }

        double start = get_time_us();
        reward_compute_race(sys, states, NULL, actions, NULL, rewards, DRONE_COUNT);
        double elapsed = get_time_us() - start;

        if (elapsed < result.min_us) result.min_us = elapsed;
        if (elapsed > result.max_us) result.max_us = elapsed;
        total += elapsed;
    }

    result.avg_us = total / BENCH_ITERATIONS;
    result.passed = result.avg_us <= result.target_us;

    arena_destroy(arena);
    return result;
}

/* ============================================================================
 * Benchmark: Termination Check
 * ============================================================================ */

static BenchResult bench_termination_check(void) {
    BenchResult result = {
        .name = "Termination Check (1024 drones)",
        .target_us = 100.0,
        .min_us = 1e9,
        .max_us = 0,
        .avg_us = 0
    };

    Arena* arena = arena_create(16 * 1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, DRONE_COUNT);
    drone_state_zero(states);
    RewardSystem* sys = reward_create(arena, NULL, DRONE_COUNT, 0);

    /* Setup collision results */
    uint8_t* world_flags = arena_alloc_array(arena, uint8_t, DRONE_COUNT);
    memset(world_flags, 0, DRONE_COUNT);
    for (int i = 0; i < 50; i++) {
        world_flags[i * 20] = 1;  /* Some collisions */
    }

    CollisionResults collisions = {
        .pairs = NULL,
        .pair_count = 0,
        .world_flags = world_flags
    };

    /* Setup termination flags output */
    TerminationFlags flags;
    flags.done = arena_alloc_array(arena, uint8_t, DRONE_COUNT);
    flags.truncated = arena_alloc_array(arena, uint8_t, DRONE_COUNT);
    flags.success = arena_alloc_array(arena, uint8_t, DRONE_COUNT);
    flags.collision = arena_alloc_array(arena, uint8_t, DRONE_COUNT);
    flags.out_of_bounds = arena_alloc_array(arena, uint8_t, DRONE_COUNT);
    flags.timeout = arena_alloc_array(arena, uint8_t, DRONE_COUNT);
    flags.capacity = DRONE_COUNT;

    /* Randomize positions (some out of bounds) */
    PCG32 rng;
    pcg32_seed(&rng, 12345);
    for (uint32_t i = 0; i < DRONE_COUNT; i++) {
        states->pos_x[i] = pcg32_range(&rng, -120.0f, 120.0f);
        states->pos_y[i] = pcg32_range(&rng, -120.0f, 120.0f);
        states->pos_z[i] = pcg32_range(&rng, -20.0f, 120.0f);
        sys->episode_length[i] = pcg32_bounded(&rng, 1500);
    }

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        reward_compute_terminations(sys, states, &collisions,
                                    VEC3(-100, -100, 0), VEC3(100, 100, 100),
                                    1000, &flags, DRONE_COUNT);
    }

    /* Benchmark */
    double total = 0;
    for (int iter = 0; iter < BENCH_ITERATIONS; iter++) {
        double start = get_time_us();
        reward_compute_terminations(sys, states, &collisions,
                                    VEC3(-100, -100, 0), VEC3(100, 100, 100),
                                    1000, &flags, DRONE_COUNT);
        double elapsed = get_time_us() - start;

        if (elapsed < result.min_us) result.min_us = elapsed;
        if (elapsed > result.max_us) result.max_us = elapsed;
        total += elapsed;
    }

    result.avg_us = total / BENCH_ITERATIONS;
    result.passed = result.avg_us <= result.target_us;

    arena_destroy(arena);
    return result;
}

/* ============================================================================
 * Benchmark: Track Reward
 * ============================================================================ */

static BenchResult bench_track_reward(void) {
    BenchResult result = {
        .name = "Track Reward (1024 drones)",
        .target_us = 400.0,
        .min_us = 1e9,
        .max_us = 0,
        .avg_us = 0
    };

    Arena* arena = arena_create(16 * 1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, DRONE_COUNT);
    drone_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_TRACK);
    RewardSystem* sys = reward_create(arena, &cfg, DRONE_COUNT, 0);

    PCG32 rng;
    pcg32_seed(&rng, 12345);
    for (uint32_t i = 0; i < DRONE_COUNT; i++) {
        states->pos_x[i] = pcg32_range(&rng, -10.0f, 10.0f);
        states->pos_y[i] = pcg32_range(&rng, -10.0f, 10.0f);
        states->pos_z[i] = pcg32_range(&rng, 0.0f, 10.0f);
        states->vel_x[i] = pcg32_range(&rng, -5.0f, 5.0f);
        states->vel_y[i] = pcg32_range(&rng, -5.0f, 5.0f);
        states->vel_z[i] = pcg32_range(&rng, -2.0f, 2.0f);

        /* Set moving targets */
        reward_set_target(sys, i,
                          VEC3(pcg32_range(&rng, -10.0f, 10.0f),
                               pcg32_range(&rng, -10.0f, 10.0f),
                               pcg32_range(&rng, 0.0f, 10.0f)),
                          VEC3(pcg32_range(&rng, -2.0f, 2.0f),
                               pcg32_range(&rng, -2.0f, 2.0f),
                               pcg32_range(&rng, -1.0f, 1.0f)),
                          1.0f);
    }

    float* rewards = arena_alloc_array(arena, float, DRONE_COUNT);
    float* actions = arena_alloc_array(arena, float, DRONE_COUNT * 4);

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        reward_compute_track(sys, states, NULL, actions, NULL, rewards, DRONE_COUNT);
    }

    /* Benchmark */
    double total = 0;
    for (int iter = 0; iter < BENCH_ITERATIONS; iter++) {
        for (uint32_t i = 0; i < DRONE_COUNT; i++) {
            sys->prev_distance[i] = FLT_MAX;
        }

        double start = get_time_us();
        reward_compute_track(sys, states, NULL, actions, NULL, rewards, DRONE_COUNT);
        double elapsed = get_time_us() - start;

        if (elapsed < result.min_us) result.min_us = elapsed;
        if (elapsed > result.max_us) result.max_us = elapsed;
        total += elapsed;
    }

    result.avg_us = total / BENCH_ITERATIONS;
    result.passed = result.avg_us <= result.target_us;

    arena_destroy(arena);
    return result;
}

/* ============================================================================
 * Benchmark: Full Frame
 * ============================================================================ */

static BenchResult bench_full_frame(void) {
    BenchResult result = {
        .name = "Full Frame (1024 drones)",
        .target_us = 1000.0,
        .min_us = 1e9,
        .max_us = 0,
        .avg_us = 0
    };

    Arena* arena = arena_create(16 * 1024 * 1024);
    DroneStateSOA* states = drone_state_create(arena, DRONE_COUNT);
    drone_state_zero(states);

    RewardConfig cfg = reward_config_default(TASK_HOVER);
    RewardSystem* sys = reward_create(arena, &cfg, DRONE_COUNT, 0);

    PCG32 rng;
    pcg32_seed(&rng, 12345);
    for (uint32_t i = 0; i < DRONE_COUNT; i++) {
        states->pos_x[i] = pcg32_range(&rng, -10.0f, 10.0f);
        states->pos_y[i] = pcg32_range(&rng, -10.0f, 10.0f);
        states->pos_z[i] = pcg32_range(&rng, 0.0f, 10.0f);
    }
    reward_set_targets_random(sys, DRONE_COUNT, VEC3(-10, -10, 0), VEC3(10, 10, 10), &rng);

    float* rewards = arena_alloc_array(arena, float, DRONE_COUNT);
    float* actions = arena_alloc_array(arena, float, DRONE_COUNT * 4);

    uint8_t* world_flags = arena_alloc_array(arena, uint8_t, DRONE_COUNT);
    memset(world_flags, 0, DRONE_COUNT);
    CollisionResults collisions = {.world_flags = world_flags};

    TerminationFlags flags;
    flags.done = arena_alloc_array(arena, uint8_t, DRONE_COUNT);
    flags.truncated = arena_alloc_array(arena, uint8_t, DRONE_COUNT);
    flags.success = arena_alloc_array(arena, uint8_t, DRONE_COUNT);
    flags.collision = arena_alloc_array(arena, uint8_t, DRONE_COUNT);
    flags.out_of_bounds = arena_alloc_array(arena, uint8_t, DRONE_COUNT);
    flags.timeout = arena_alloc_array(arena, uint8_t, DRONE_COUNT);

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        reward_compute(sys, states, NULL, actions, &collisions, rewards, DRONE_COUNT);
        reward_compute_terminations(sys, states, &collisions,
                                    VEC3(-100, -100, 0), VEC3(100, 100, 100),
                                    1000, &flags, DRONE_COUNT);
    }

    /* Benchmark */
    double total = 0;
    for (int iter = 0; iter < BENCH_ITERATIONS; iter++) {
        for (uint32_t i = 0; i < DRONE_COUNT; i++) {
            sys->prev_distance[i] = FLT_MAX;
        }

        double start = get_time_us();
        reward_compute(sys, states, NULL, actions, &collisions, rewards, DRONE_COUNT);
        reward_compute_terminations(sys, states, &collisions,
                                    VEC3(-100, -100, 0), VEC3(100, 100, 100),
                                    1000, &flags, DRONE_COUNT);
        double elapsed = get_time_us() - start;

        if (elapsed < result.min_us) result.min_us = elapsed;
        if (elapsed > result.max_us) result.max_us = elapsed;
        total += elapsed;
    }

    result.avg_us = total / BENCH_ITERATIONS;
    result.passed = result.avg_us <= result.target_us;

    arena_destroy(arena);
    return result;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("=== Reward System Performance Benchmarks ===\n");
    printf("Configuration: %d drones, %d gates, %d iterations\n\n",
           DRONE_COUNT, GATE_COUNT, BENCH_ITERATIONS);

    BenchResult results[5];
    int total_passed = 0;

    results[0] = bench_hover_reward();
    print_result(&results[0]);
    total_passed += results[0].passed;

    results[1] = bench_race_reward();
    print_result(&results[1]);
    total_passed += results[1].passed;

    results[2] = bench_termination_check();
    print_result(&results[2]);
    total_passed += results[2].passed;

    results[3] = bench_track_reward();
    print_result(&results[3]);
    total_passed += results[3].passed;

    results[4] = bench_full_frame();
    print_result(&results[4]);
    total_passed += results[4].passed;

    printf("\n=== Summary ===\n");
    printf("Benchmarks passed: %d / 5\n", total_passed);

    if (total_passed == 5) {
        printf("\nAll performance targets met!\n");
    } else {
        printf("\nSome targets not met - consider optimization.\n");
    }

    return total_passed == 5 ? 0 : 1;
}
