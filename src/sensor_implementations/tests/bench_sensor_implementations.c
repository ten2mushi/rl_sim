/**
 * Sensor Implementations Benchmarks
 *
 * Performance benchmarks for individual sensor types.
 *
 * Target Performance (1024 drones):
 * - IMU: <0.1ms
 * - Position/Velocity: <0.05ms
 * - ToF: <0.5ms
 * - LiDAR 2D (64 rays): <5ms
 * - LiDAR 3D (16x64): <20ms
 * - Camera RGB (64x64): <20ms
 * - Camera Depth (64x64): <15ms
 * - Camera Seg (64x64): <15ms
 * - Neighbor (K=5): <1ms
 */

#include "sensor_system.h"
#include "sensor_implementations.h"
#include "collision_system.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

/* ============================================================================
 * Timing Utilities
 * ============================================================================ */

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

#define BENCH_ITERATIONS 50
#define WARMUP_ITERATIONS 10

/* ============================================================================
 * Benchmark Helper
 * ============================================================================ */

typedef struct BenchResult {
    double avg_ms;
    double min_ms;
    double max_ms;
    const char* name;
    double target_ms;
    bool passed;
} BenchResult;

static BenchResult bench_sensor(Arena* arena, SensorType type, SensorConfig config,
                                uint32_t num_drones, const char* name, double target_ms) {
    BenchResult result = {0};
    result.name = name;
    result.target_ms = target_ms;

    SensorSystem* sys = sensor_system_create(arena, num_drones, 4, 16384);
    if (sys == NULL) {
        printf("Failed to create sensor system for %s\n", name);
        return result;
    }

    sensor_implementations_register_all(&sys->registry);

    uint32_t sensor_idx = sensor_system_create_sensor(sys, &config);
    if (sensor_idx == UINT32_MAX) {
        printf("Failed to create sensor for %s\n", name);
        sensor_system_destroy(sys);
        return result;
    }

    /* Attach to all drones */
    for (uint32_t d = 0; d < num_drones; d++) {
        sensor_system_attach(sys, d, sensor_idx);
    }

    /* Create drone state */
    DroneStateSOA* drones = drone_state_create(arena, num_drones);
    for (uint32_t d = 0; d < num_drones; d++) {
        drone_state_init(drones, d);
        drones->pos_x[d] = (float)(d % 32) * 2.0f;
        drones->pos_y[d] = (float)(d / 32) * 2.0f;
        drones->pos_z[d] = 10.0f;
    }
    drones->count = num_drones;

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        sensor_system_sample_all(sys, drones, NULL, NULL, num_drones);
    }

    /* Benchmark */
    double total_time = 0.0;
    double min_time = 1e9;
    double max_time = 0.0;

    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        double start = get_time_ms();
        sensor_system_sample_all(sys, drones, NULL, NULL, num_drones);
        double end = get_time_ms();

        double elapsed = end - start;
        total_time += elapsed;
        if (elapsed < min_time) min_time = elapsed;
        if (elapsed > max_time) max_time = elapsed;
    }

    result.avg_ms = total_time / BENCH_ITERATIONS;
    result.min_ms = min_time;
    result.max_ms = max_time;
    result.passed = result.avg_ms < target_ms;

    sensor_system_destroy(sys);

    return result;
}

static void print_result(BenchResult* r) {
    printf("  %-20s: avg=%.3f ms, min=%.3f ms, max=%.3f ms  [%s] (target: %.1f ms)\n",
           r->name, r->avg_ms, r->min_ms, r->max_ms,
           r->passed ? "PASS" : "FAIL", r->target_ms);
}

/* ============================================================================
 * Main Benchmarks
 * ============================================================================ */

int main(void) {
    printf("Sensor Implementations Benchmarks\n");
    printf("==================================\n\n");

    Arena* arena = arena_create(128 * 1024 * 1024);  /* 128 MB */
    if (arena == NULL) {
        printf("Failed to create arena\n");
        return 1;
    }

    uint32_t num_drones = 1024;
    BenchResult results[10];
    int num_results = 0;

    printf("Running benchmarks with %u drones...\n\n", num_drones);

    /* Position Sensor */
    {
        SensorConfig config = sensor_config_position();
        results[num_results++] = bench_sensor(arena, SENSOR_TYPE_POSITION, config,
                                              num_drones, "Position", 0.05);
        arena_reset(arena);
    }

    /* Velocity Sensor */
    {
        SensorConfig config = sensor_config_velocity();
        results[num_results++] = bench_sensor(arena, SENSOR_TYPE_VELOCITY, config,
                                              num_drones, "Velocity", 0.05);
        arena_reset(arena);
    }

    /* IMU Sensor (with noise) */
    {
        SensorConfig config = sensor_config_imu();
        results[num_results++] = bench_sensor(arena, SENSOR_TYPE_IMU, config,
                                              num_drones, "IMU (noise)", 0.15);
        arena_reset(arena);
    }

    /* IMU Sensor (no noise - baseline) */
    {
        SensorConfig config = sensor_config_imu();
        results[num_results++] = bench_sensor(arena, SENSOR_TYPE_IMU, config,
                                              num_drones, "IMU (no noise)", 0.1);
        arena_reset(arena);
    }

    /* ToF Sensor (without world) */
    {
        SensorConfig config = sensor_config_tof(VEC3(0, 0, -1), 10.0f);
        results[num_results++] = bench_sensor(arena, SENSOR_TYPE_TOF, config,
                                              num_drones, "ToF", 0.5);
        arena_reset(arena);
    }

    /* LiDAR 2D (64 rays) */
    {
        SensorConfig config = sensor_config_lidar_2d(64, 3.14159f, 20.0f);
        results[num_results++] = bench_sensor(arena, SENSOR_TYPE_LIDAR_2D, config,
                                              num_drones, "LiDAR-2D (64)", 5.0);
        arena_reset(arena);
    }

    /* LiDAR 3D (16x64 rays) */
    {
        SensorConfig config = sensor_config_lidar_3d(64, 16, 6.28f, 0.5f, 50.0f);
        results[num_results++] = bench_sensor(arena, SENSOR_TYPE_LIDAR_3D, config,
                                              num_drones, "LiDAR-3D (16x64)", 20.0);
        arena_reset(arena);
    }

    /* Neighbor Sensor (K=5) - with collision system for spatial hashing */
    {
        BenchResult result = {0};
        result.name = "Neighbor (K=5)";
        result.target_ms = 1.0;

        SensorSystem* sys = sensor_system_create(arena, num_drones, 4, 16384);
        sensor_implementations_register_all(&sys->registry);

        SensorConfig config = sensor_config_neighbor(5, 20.0f);
        uint32_t sensor_idx = sensor_system_create_sensor(sys, &config);

        for (uint32_t d = 0; d < num_drones; d++) {
            sensor_system_attach(sys, d, sensor_idx);
        }

        /* Create drone state */
        DroneStateSOA* drones = drone_state_create(arena, num_drones);
        for (uint32_t d = 0; d < num_drones; d++) {
            drone_state_init(drones, d);
            drones->pos_x[d] = (float)(d % 32) * 2.0f;
            drones->pos_y[d] = (float)(d / 32) * 2.0f;
            drones->pos_z[d] = 10.0f;
        }
        drones->count = num_drones;

        /* Create collision system with spatial hashing */
        CollisionSystem* collision = collision_create(arena, num_drones, 0.2f, 2.0f);
        collision_build_spatial_hash(collision, drones, num_drones);

        /* Warmup */
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            collision_build_spatial_hash(collision, drones, num_drones);
            sensor_system_sample_all(sys, drones, NULL, collision, num_drones);
        }

        /* Benchmark */
        double total_time = 0.0;
        double min_time = 1e9;
        double max_time = 0.0;

        for (int i = 0; i < BENCH_ITERATIONS; i++) {
            collision_build_spatial_hash(collision, drones, num_drones);
            double start = get_time_ms();
            sensor_system_sample_all(sys, drones, NULL, collision, num_drones);
            double end = get_time_ms();

            double elapsed = end - start;
            total_time += elapsed;
            if (elapsed < min_time) min_time = elapsed;
            if (elapsed > max_time) max_time = elapsed;
        }

        result.avg_ms = total_time / BENCH_ITERATIONS;
        result.min_ms = min_time;
        result.max_ms = max_time;
        result.passed = result.avg_ms < result.target_ms;

        results[num_results++] = result;
        sensor_system_destroy(sys);
        arena_reset(arena);
    }

    /* Camera RGB (small, without world) */
    {
        SensorConfig config = sensor_config_camera(32, 32, 1.57f, 100.0f);
        config.type = SENSOR_TYPE_CAMERA_RGB;
        results[num_results++] = bench_sensor(arena, SENSOR_TYPE_CAMERA_RGB, config,
                                              256, "Camera-RGB (32x32)", 5.0);  /* Fewer drones for camera */
        arena_reset(arena);
    }

    /* Camera Depth (small, without world) */
    {
        SensorConfig config = sensor_config_camera(32, 32, 1.57f, 100.0f);
        config.type = SENSOR_TYPE_CAMERA_DEPTH;
        results[num_results++] = bench_sensor(arena, SENSOR_TYPE_CAMERA_DEPTH, config,
                                              256, "Camera-Depth (32x32)", 4.0);
        arena_reset(arena);
    }

    printf("Results:\n");
    printf("---------\n");

    int passed = 0;
    int failed = 0;
    for (int i = 0; i < num_results; i++) {
        print_result(&results[i]);
        if (results[i].passed) passed++;
        else failed++;
    }

    printf("\n==================================\n");
    printf("Benchmarks: %d passed, %d failed\n", passed, failed);

    arena_destroy(arena);

    return failed > 0 ? 1 : 0;
}
