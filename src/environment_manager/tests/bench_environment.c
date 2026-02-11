/**
 * Environment Manager Benchmarks
 *
 * Performance benchmarks for the batch drone engine.
 * Target: <20ms total step for 1024 drones (50 FPS)
 */

#include "environment_manager.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * Benchmark Configuration
 * ============================================================================ */

#define BENCH_WARMUP_STEPS    10
#define BENCH_MEASURE_STEPS   100
#define BENCH_NUM_ENVS        64
#define BENCH_DRONES_PER_ENV  16
#define BENCH_TOTAL_DRONES    (BENCH_NUM_ENVS * BENCH_DRONES_PER_ENV)

/* Performance targets (milliseconds) */
#define TARGET_PHYSICS_MS     5.0
#define TARGET_COLLISION_MS   1.0
#define TARGET_SENSOR_MS      10.0
#define TARGET_REWARD_MS      1.0
#define TARGET_RESET_MS       2.0
#define TARGET_TOTAL_MS       20.0

/* ============================================================================
 * Benchmark Utilities
 * ============================================================================ */

static int benches_run = 0;
static int benches_passed = 0;

#define BENCHMARK(name) static int bench_##name(void)
#define RUN_BENCHMARK(name) do { \
    benches_run++; \
    printf("\n  Running %s...\n", #name); \
    if (bench_##name() == 0) { \
        benches_passed++; \
        printf("    [PASS]\n"); \
    } else { \
        printf("    [FAIL]\n"); \
    } \
} while(0)

/* Helper: Create benchmark engine */
static BatchDroneEngine* create_bench_engine(void) {
    EngineConfig cfg = engine_config_default();
    cfg.num_envs = BENCH_NUM_ENVS;
    cfg.drones_per_env = BENCH_DRONES_PER_ENV;
    cfg.persistent_arena_size = 256 * 1024 * 1024;
    cfg.frame_arena_size = 64 * 1024 * 1024;
    cfg.seed = 42;

    char error[ENGINE_ERROR_MSG_SIZE];
    return engine_create(&cfg, error);
}

/* ============================================================================
 * Benchmarks
 * ============================================================================ */

/* Benchmark: Full step time */
BENCHMARK(step_1024_drones) {
    BatchDroneEngine* engine = create_bench_engine();
    if (!engine) {
        printf("    Failed to create engine\n");
        return -1;
    }

    engine_reset(engine);

    /* Warmup */
    for (int i = 0; i < BENCH_WARMUP_STEPS; i++) {
        engine_step(engine);
    }

    /* Measure */
    double total_time = 0.0;
    double start = engine_get_time_ms();

    for (int i = 0; i < BENCH_MEASURE_STEPS; i++) {
        engine_step(engine);
    }

    double end = engine_get_time_ms();
    total_time = end - start;

    double avg_time = total_time / BENCH_MEASURE_STEPS;

    printf("    Total drones: %d\n", BENCH_TOTAL_DRONES);
    printf("    Avg step time: %.3f ms (target: <%.1f ms)\n", avg_time, TARGET_TOTAL_MS);
    printf("    Steps/second: %.1f\n", 1000.0 / avg_time);
    printf("    Drone-steps/second: %.1f\n", 1000.0 / avg_time * BENCH_TOTAL_DRONES);

    engine_destroy(engine);

    return (avg_time < TARGET_TOTAL_MS) ? 0 : -1;
}

/* Benchmark: Physics phase */
BENCHMARK(physics_time) {
    BatchDroneEngine* engine = create_bench_engine();
    if (!engine) return -1;

    engine_reset(engine);

    /* Warmup */
    for (int i = 0; i < BENCH_WARMUP_STEPS; i++) {
        engine_step(engine);
    }

    /* Measure physics phase only */
    double total_physics = 0.0;

    for (int i = 0; i < BENCH_MEASURE_STEPS; i++) {
        double start = engine_get_time_ms();
        engine_step_physics(engine);
        double end = engine_get_time_ms();
        total_physics += (end - start);
    }

    double avg_physics = total_physics / BENCH_MEASURE_STEPS;

    printf("    Avg physics time: %.3f ms (target: <%.1f ms)\n", avg_physics, TARGET_PHYSICS_MS);

    engine_destroy(engine);

    return (avg_physics < TARGET_PHYSICS_MS) ? 0 : -1;
}

/* Benchmark: Collision phase */
BENCHMARK(collision_time) {
    BatchDroneEngine* engine = create_bench_engine();
    if (!engine) return -1;

    engine_reset(engine);

    for (int i = 0; i < BENCH_WARMUP_STEPS; i++) {
        engine_step(engine);
    }

    double total_collision = 0.0;

    for (int i = 0; i < BENCH_MEASURE_STEPS; i++) {
        double start = engine_get_time_ms();
        engine_step_collision(engine);
        double end = engine_get_time_ms();
        total_collision += (end - start);
    }

    double avg_collision = total_collision / BENCH_MEASURE_STEPS;

    printf("    Avg collision time: %.3f ms (target: <%.1f ms)\n", avg_collision, TARGET_COLLISION_MS);

    engine_destroy(engine);

    return (avg_collision < TARGET_COLLISION_MS) ? 0 : -1;
}

/* Benchmark: Sensor phase */
BENCHMARK(sensor_time) {
    BatchDroneEngine* engine = create_bench_engine();
    if (!engine) return -1;

    engine_reset(engine);

    for (int i = 0; i < BENCH_WARMUP_STEPS; i++) {
        engine_step(engine);
    }

    double total_sensor = 0.0;

    for (int i = 0; i < BENCH_MEASURE_STEPS; i++) {
        double start = engine_get_time_ms();
        engine_step_sensors(engine);
        double end = engine_get_time_ms();
        total_sensor += (end - start);
    }

    double avg_sensor = total_sensor / BENCH_MEASURE_STEPS;

    printf("    Avg sensor time: %.3f ms (target: <%.1f ms)\n", avg_sensor, TARGET_SENSOR_MS);

    engine_destroy(engine);

    return (avg_sensor < TARGET_SENSOR_MS) ? 0 : -1;
}

/* Benchmark: Reward phase */
BENCHMARK(reward_time) {
    BatchDroneEngine* engine = create_bench_engine();
    if (!engine) return -1;

    engine_reset(engine);

    for (int i = 0; i < BENCH_WARMUP_STEPS; i++) {
        engine_step(engine);
    }

    double total_reward = 0.0;

    for (int i = 0; i < BENCH_MEASURE_STEPS; i++) {
        double start = engine_get_time_ms();
        engine_step_rewards(engine);
        double end = engine_get_time_ms();
        total_reward += (end - start);
    }

    double avg_reward = total_reward / BENCH_MEASURE_STEPS;

    printf("    Avg reward time: %.3f ms (target: <%.1f ms)\n", avg_reward, TARGET_REWARD_MS);

    engine_destroy(engine);

    return (avg_reward < TARGET_REWARD_MS) ? 0 : -1;
}

/* Benchmark: Reset time */
BENCHMARK(reset_time) {
    BatchDroneEngine* engine = create_bench_engine();
    if (!engine) return -1;

    double total_reset = 0.0;

    for (int i = 0; i < BENCH_MEASURE_STEPS; i++) {
        double start = engine_get_time_ms();
        engine_reset(engine);
        double end = engine_get_time_ms();
        total_reset += (end - start);
    }

    double avg_reset = total_reset / BENCH_MEASURE_STEPS;

    /* Per-drone reset time */
    double per_drone_reset = avg_reset / BENCH_TOTAL_DRONES;

    printf("    Avg reset time: %.3f ms\n", avg_reset);
    printf("    Per-drone reset: %.4f ms (target: <%.1f ms/drone)\n",
           per_drone_reset * 1000, TARGET_RESET_MS);

    engine_destroy(engine);

    return (per_drone_reset * 1000 < TARGET_RESET_MS) ? 0 : -1;
}

/* Benchmark: Memory usage */
BENCHMARK(memory_usage) {
    BatchDroneEngine* engine = create_bench_engine();
    if (!engine) return -1;

    engine_reset(engine);

    EngineStats stats;
    engine_get_stats(engine, &stats);

    double persistent_mb = stats.persistent_memory_used / (1024.0 * 1024.0);
    double frame_mb = stats.frame_memory_used / (1024.0 * 1024.0);
    double total_mb = persistent_mb + frame_mb;

    printf("    Persistent arena: %.2f MB\n", persistent_mb);
    printf("    Frame arena: %.2f MB\n", frame_mb);
    printf("    Total: %.2f MB (target: <4096 MB)\n", total_mb);

    engine_destroy(engine);

    return (total_mb < 4096.0) ? 0 : -1;
}

/* Benchmark: Sustained performance */
BENCHMARK(sustained_performance) {
    BatchDroneEngine* engine = create_bench_engine();
    if (!engine) return -1;

    engine_reset(engine);

    /* Run for 1000 steps and check we maintain target FPS */
    double start = engine_get_time_ms();

    for (int i = 0; i < 1000; i++) {
        engine_step(engine);
    }

    double end = engine_get_time_ms();
    double total_time = end - start;
    double avg_time = total_time / 1000.0;
    double fps = 1000.0 / avg_time;

    printf("    1000 steps in %.1f ms\n", total_time);
    printf("    Avg step: %.3f ms\n", avg_time);
    printf("    Sustained FPS: %.1f (target: >50 FPS)\n", fps);

    engine_destroy(engine);

    return (fps >= 50.0) ? 0 : -1;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("\n=== Environment Manager Benchmarks ===\n");
    printf("Target: %d drones at 50 FPS (<20ms step)\n", BENCH_TOTAL_DRONES);

    RUN_BENCHMARK(step_1024_drones);
    RUN_BENCHMARK(physics_time);
    RUN_BENCHMARK(collision_time);
    RUN_BENCHMARK(sensor_time);
    RUN_BENCHMARK(reward_time);
    RUN_BENCHMARK(reset_time);
    RUN_BENCHMARK(memory_usage);
    RUN_BENCHMARK(sustained_performance);

    printf("\n======================================\n");
    printf("Benchmarks: %d/%d passed\n\n", benches_passed, benches_run);

    if (benches_passed == benches_run) {
        printf("All performance targets met!\n\n");
        return 0;
    } else {
        printf("Some performance targets not met.\n\n");
        return 1;
    }
}
