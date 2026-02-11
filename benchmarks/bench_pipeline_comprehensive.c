/**
 * Comprehensive Pipeline Benchmark
 *
 * Measures the entire engine pipeline with per-phase breakdown at multiple
 * drone scales and sensor profiles. Produces a table showing:
 *   Physics | Collision | Sensors(CPU) | Sensors(GPU) | Rewards | Reset | Total
 *
 * Also measures:
 *   - Memory usage (arena utilization)
 *   - Scaling efficiency (throughput per drone)
 *   - GPU dispatch latency (dispatch to wait)
 */

#include "bench_harness.h"
#include "environment_manager.h"
#include "gpu_hal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * Profile Definitions
 * ============================================================================ */

typedef enum {
    P_MINIMAL,
    P_LIGHT,
    P_NAVIGATION,
    P_VISION,
    P_FULL,
    P_STRESS,
    P_COUNT
} Profile;

static const char* profile_names[] = {
    "MINIMAL", "LIGHT", "NAVIGATION", "VISION", "FULL", "STRESS"
};

static EngineConfig profile_config(Profile p, uint32_t total_drones) {
    EngineConfig cfg = engine_config_default();
    cfg.drones_per_env = 16;
    cfg.num_envs = total_drones / cfg.drones_per_env;
    if (cfg.num_envs == 0) cfg.num_envs = 1;
    cfg.total_drones = cfg.num_envs * cfg.drones_per_env;
    cfg.num_threads = 0;
    cfg.seed = 42;
    cfg.persistent_arena_size = 512 * 1024 * 1024;  /* 512MB for large scales */
    cfg.frame_arena_size = 128 * 1024 * 1024;

    SensorConfig sc;
    switch (p) {
    case P_MINIMAL:
        sc = sensor_config_imu();
        engine_config_add_sensor(&cfg, &sc);
        break;
    case P_LIGHT:
        sc = sensor_config_imu();
        engine_config_add_sensor(&cfg, &sc);
        sc = sensor_config_position();
        engine_config_add_sensor(&cfg, &sc);
        sc = sensor_config_velocity();
        engine_config_add_sensor(&cfg, &sc);
        break;
    case P_NAVIGATION:
        sc = sensor_config_imu();
        engine_config_add_sensor(&cfg, &sc);
        sc = sensor_config_lidar_2d(64, (float)M_PI, 20.0f);
        engine_config_add_sensor(&cfg, &sc);
        sc = sensor_config_position();
        engine_config_add_sensor(&cfg, &sc);
        break;
    case P_VISION:
        sc = sensor_config_imu();
        engine_config_add_sensor(&cfg, &sc);
        sc = sensor_config_camera(32, 32, 1.57f, 100.0f);
        sc.type = SENSOR_TYPE_CAMERA_DEPTH;
        engine_config_add_sensor(&cfg, &sc);
        sc = sensor_config_position();
        engine_config_add_sensor(&cfg, &sc);
        break;
    case P_FULL:
        sc = sensor_config_imu();
        engine_config_add_sensor(&cfg, &sc);
        sc = sensor_config_camera(32, 32, 1.57f, 100.0f);
        sc.type = SENSOR_TYPE_CAMERA_DEPTH;
        engine_config_add_sensor(&cfg, &sc);
        sc = sensor_config_lidar_2d(64, (float)M_PI, 20.0f);
        engine_config_add_sensor(&cfg, &sc);
        sc = sensor_config_position();
        engine_config_add_sensor(&cfg, &sc);
        sc = sensor_config_velocity();
        engine_config_add_sensor(&cfg, &sc);
        break;
    case P_STRESS:
        sc = sensor_config_imu();
        engine_config_add_sensor(&cfg, &sc);
        sc = sensor_config_camera(64, 64, 1.57f, 100.0f);
        sc.type = SENSOR_TYPE_CAMERA_DEPTH;
        engine_config_add_sensor(&cfg, &sc);
        sc = sensor_config_camera(64, 64, 1.57f, 100.0f);
        engine_config_add_sensor(&cfg, &sc);
        sc = sensor_config_lidar_3d(16, 64, (float)(2.0 * M_PI), (float)(M_PI / 3.0), 50.0f);
        engine_config_add_sensor(&cfg, &sc);
        sc = sensor_config_position();
        engine_config_add_sensor(&cfg, &sc);
        sc = sensor_config_velocity();
        engine_config_add_sensor(&cfg, &sc);
        break;
    default:
        break;
    }
    return cfg;
}

/* ============================================================================
 * Phase Timing Accumulator
 * ============================================================================ */

typedef struct {
    double physics_ms;
    double collision_ms;
    double sensor_cpu_ms;
    double sensor_gpu_ms;
    double reward_ms;
    double reset_ms;
    double total_ms;
    size_t persistent_used;
    size_t persistent_cap;
    size_t frame_peak;
    size_t frame_cap;
    uint32_t obs_dim;
} PhaseResult;

static PhaseResult bench_pipeline(Profile profile, uint32_t drone_count,
                                  uint32_t warmup, uint32_t iterations) {
    PhaseResult r = {0};

    EngineConfig cfg = profile_config(profile, drone_count);
    char error[ENGINE_ERROR_MSG_SIZE];
    BatchDroneEngine* engine = engine_create(&cfg, error);
    if (!engine) {
        fprintf(stderr, "  [%s %u drones] Failed: %s\n",
                profile_names[profile], drone_count, error);
        return r;
    }

    engine_reset(engine);
    r.obs_dim = engine_get_obs_dim(engine);

    /* Warmup */
    for (uint32_t i = 0; i < warmup; i++) {
        engine_step(engine);
    }

    /* Benchmark iterations */
    double phys = 0, coll = 0, sens = 0, gpu_s = 0, rew = 0, rst = 0, total = 0;

    for (uint32_t i = 0; i < iterations; i++) {
        double t0 = bench_time_ms();
        engine_step(engine);
        double t1 = bench_time_ms();

        total += (t1 - t0);
        phys  += engine->physics_time_ms;
        coll  += engine->collision_time_ms;
        sens  += engine->sensor_time_ms;
        gpu_s += engine->gpu_sensor_time_ms;
        rew   += engine->reward_time_ms;
        rst   += engine->reset_time_ms;
    }

    double n = (double)iterations;
    r.physics_ms   = phys / n;
    r.collision_ms = coll / n;
    r.sensor_cpu_ms = sens / n;
    r.sensor_gpu_ms = gpu_s / n;
    r.reward_ms    = rew / n;
    r.reset_ms     = rst / n;
    r.total_ms     = total / n;

    /* Memory stats */
    r.persistent_used = engine->persistent_arena->used;
    r.persistent_cap  = engine->persistent_arena->capacity;
    r.frame_peak      = engine->frame_arena->used;
    r.frame_cap       = engine->frame_arena->capacity;

    engine_destroy(engine);
    return r;
}

/* ============================================================================
 * Printing
 * ============================================================================ */

static void print_phase_header(void) {
    printf("%-12s %6s %7s %7s %7s %7s %7s %7s %9s %7s\n",
           "PROFILE", "DRONES", "PHYS", "COLL", "SENS_C", "SENS_G", "REWARD",
           "RESET", "TOTAL", "us/dne");
    printf("------------------------------------------------------------------------------\n");
}

static void print_phase_row(const char* name, uint32_t drones, const PhaseResult* r) {
    double us_per_drone = (r->total_ms * 1000.0) / drones;
    printf("%-12s %6u %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %9.3f %7.2f\n",
           name, drones,
           r->physics_ms, r->collision_ms,
           r->sensor_cpu_ms, r->sensor_gpu_ms,
           r->reward_ms, r->reset_ms,
           r->total_ms, us_per_drone);
}

static void print_memory(const PhaseResult* r) {
    printf("  Memory: persistent %zuMB/%zuMB (%.0f%%), frame peak %zuKB/%zuMB\n",
           r->persistent_used / (1024*1024), r->persistent_cap / (1024*1024),
           (double)r->persistent_used / r->persistent_cap * 100.0,
           r->frame_peak / 1024, r->frame_cap / (1024*1024));
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char** argv) {
    BenchCLI cli = bench_parse_cli(argc, argv);

    printf("\n");
    printf("========================================================\n");
    printf("  Comprehensive Pipeline Benchmark\n");
    printf("========================================================\n\n");

    printf("GPU: %s\n", gpu_is_available() ? "YES" : "NO");
    if (gpu_is_available()) {
        GpuDevice* dev = gpu_device_create();
        if (dev) {
            printf("Device: %s\n", gpu_device_name(dev));
            gpu_device_destroy(dev);
        }
    }
    printf("\n");

    uint32_t warmup = cli.warmup > 0 ? cli.warmup : 10;
    uint32_t iters  = cli.iterations > 0 ? cli.iterations : 50;

    /* ====================================================================
     * Part 1: All profiles at 1024 drones
     * ==================================================================== */

    printf("=== All Profiles @ 1024 drones (warmup=%u, iters=%u) ===\n\n", warmup, iters);
    print_phase_header();

    for (int p = 0; p < P_COUNT; p++) {
        PhaseResult r = bench_pipeline((Profile)p, 1024, warmup, iters);
        print_phase_row(profile_names[p], 1024, &r);
    }
    printf("\n");

    /* ====================================================================
     * Part 2: VISION profile scaling
     * ==================================================================== */

    uint32_t scale[] = {256, 512, 1024, 2048, 4096};
    uint32_t num_scale = sizeof(scale) / sizeof(scale[0]);

    printf("=== VISION Scaling (cam_depth_32 + IMU + pos) ===\n\n");
    print_phase_header();

    PhaseResult vision_results[5];
    for (uint32_t i = 0; i < num_scale; i++) {
        vision_results[i] = bench_pipeline(P_VISION, scale[i], warmup, iters);
        print_phase_row("VISION", scale[i], &vision_results[i]);
    }
    printf("\n");

    /* Scaling efficiency */
    printf("=== Scaling Efficiency ===\n");
    printf("  %6s  %9s  %8s  %10s\n", "DRONES", "TOTAL_MS", "us/drone", "EFFICIENCY");
    printf("  ----------------------------------------\n");
    double base_us = 0;
    for (uint32_t i = 0; i < num_scale; i++) {
        double us = (vision_results[i].total_ms * 1000.0) / scale[i];
        if (i == 0) base_us = us;
        double eff = base_us / us * 100.0;
        printf("  %6u  %9.3f  %8.2f  %9.0f%%\n",
               scale[i], vision_results[i].total_ms, us, eff);
    }
    printf("\n");

    /* ====================================================================
     * Part 3: Memory usage at max scale
     * ==================================================================== */

    printf("=== Memory Usage (4096 drones) ===\n");
    for (int p = 0; p < P_COUNT; p++) {
        PhaseResult r = bench_pipeline((Profile)p, 4096, 2, 5);
        printf("  %-12s obs_dim=%4u  ", profile_names[p], r.obs_dim);
        print_memory(&r);
    }
    printf("\n");

    /* ====================================================================
     * Part 4: Summary
     * ==================================================================== */

    PhaseResult v1024 = bench_pipeline(P_VISION, 1024, warmup, iters);
    printf("=== Summary ===\n");
    printf("  Target: 1024 drones at 50 FPS = 20ms/step\n");
    printf("  VISION @ 1024: %.3f ms/step (%.1f FPS)\n",
           v1024.total_ms, 1000.0 / v1024.total_ms);
    printf("  Target met: %s\n\n", v1024.total_ms < 20.0 ? "YES" : "NO");

    return 0;
}
