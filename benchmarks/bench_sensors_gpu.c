/**
 * GPU Sensor Benchmarks
 *
 * Side-by-side CPU vs GPU timing per sensor type.
 * Uses the engine's GPU sensor pipeline to compare:
 *   - CPU path: sensor_system_sample_all()
 *   - GPU path: gpu dispatch + wait + scatter
 *
 * Output: | Sensor | CPU (ms) | GPU (ms) | Speedup | Status |
 */

#include "bench_harness.h"
#include "environment_manager.h"
#include "gpu_hal.h"
#include "sensor_system.h"
#include "sensor_implementations.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * GPU Sensor Benchmark Context
 * ============================================================================ */

typedef struct GpuSensorBenchResult {
    const char* name;
    uint32_t    drone_count;
    double      cpu_ms;
    double      gpu_ms;
    double      speedup;
    bool        passed;
} GpuSensorBenchResult;

/* ============================================================================
 * Benchmark: Individual Sensor CPU vs GPU via Engine
 * ============================================================================ */

static GpuSensorBenchResult bench_sensor_gpu(const char* name,
                                               SensorConfig* configs,
                                               uint32_t num_configs,
                                               uint32_t num_drones,
                                               uint32_t warmup,
                                               uint32_t iterations) {
    GpuSensorBenchResult result = {0};
    result.name = name;
    result.drone_count = num_drones;

    /* Create two engines: one with GPU, one CPU-only */
    EngineConfig cfg = engine_config_default();
    cfg.drones_per_env = 16;
    cfg.num_envs = num_drones / cfg.drones_per_env;
    if (cfg.num_envs == 0) cfg.num_envs = 1;
    cfg.persistent_arena_size = 256 * 1024 * 1024;
    cfg.frame_arena_size = 64 * 1024 * 1024;
    cfg.sensor_configs = configs;
    cfg.num_sensor_configs = num_configs;

    char error[ENGINE_ERROR_MSG_SIZE];

    /* GPU-enabled engine */
    BatchDroneEngine* gpu_engine = engine_create(&cfg, error);
    if (!gpu_engine) {
        fprintf(stderr, "  Failed to create GPU engine: %s\n", error);
        return result;
    }

    engine_reset(gpu_engine);

    /* Benchmark GPU path (full engine step includes GPU dispatch) */
    for (uint32_t i = 0; i < warmup; i++) {
        engine_step(gpu_engine);
    }

    double gpu_total = 0.0;
    for (uint32_t i = 0; i < iterations; i++) {
        double t0 = bench_time_ms();
        engine_step(gpu_engine);
        double t1 = bench_time_ms();
        gpu_total += (t1 - t0);
    }
    result.gpu_ms = gpu_total / iterations;

    /* Now benchmark CPU-only path by disabling GPU context */
    struct GpuSensorContext* saved_ctx = gpu_engine->gpu_sensor_ctx;
    gpu_engine->gpu_sensor_ctx = NULL;  /* Force CPU path */

    for (uint32_t i = 0; i < warmup; i++) {
        engine_step(gpu_engine);
    }

    double cpu_total = 0.0;
    for (uint32_t i = 0; i < iterations; i++) {
        double t0 = bench_time_ms();
        engine_step(gpu_engine);
        double t1 = bench_time_ms();
        cpu_total += (t1 - t0);
    }
    result.cpu_ms = cpu_total / iterations;

    /* Restore GPU context for proper cleanup */
    gpu_engine->gpu_sensor_ctx = saved_ctx;

    /* Compute speedup */
    if (result.gpu_ms > 0.001) {
        result.speedup = result.cpu_ms / result.gpu_ms;
    }
    result.passed = result.gpu_ms <= result.cpu_ms;

    engine_destroy(gpu_engine);
    return result;
}

/* ============================================================================
 * Result Printing
 * ============================================================================ */

#define GPU_HEADER_FMT "%-32s %7s %9s %9s %9s %6s\n"
#define GPU_ROW_FMT    "%-32s %7u %9.3f %9.3f %9.1fx %6s\n"

static void gpu_print_header(void) {
    printf(GPU_HEADER_FMT, "SENSOR", "DRONES", "CPU_MS", "GPU_MS", "SPEEDUP", "STATUS");
    printf("--------------------------------------------------------------------------\n");
}

static void gpu_print_row(const GpuSensorBenchResult* r) {
    printf(GPU_ROW_FMT, r->name, r->drone_count,
           r->cpu_ms, r->gpu_ms, r->speedup,
           r->passed ? "PASS" : "FAIL");
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char** argv) {
    BenchCLI cli = bench_parse_cli(argc, argv);

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║             GPU Sensor Benchmarks                          ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    printf("GPU available: %s\n", gpu_is_available() ? "YES" : "NO");
    if (gpu_is_available()) {
        GpuDevice* dev = gpu_device_create();
        if (dev) {
            printf("Device: %s\n", gpu_device_name(dev));
            gpu_device_destroy(dev);
        }
    }
    printf("\n");

    if (!gpu_is_available()) {
        printf("No GPU available - skipping GPU benchmarks\n");
        return 0;
    }

    /* Sensor configurations */
    SensorConfig cam_depth_32, cam_depth_64, cam_rgb_32;
    SensorConfig lidar3d, lidar2d, tof;
    SensorConfig imu;

    /* Camera Depth 32x32 */
    cam_depth_32 = sensor_config_camera(32, 32, 1.57f, 100.0f);
    cam_depth_32.type = SENSOR_TYPE_CAMERA_DEPTH;

    /* Camera Depth 64x64 */
    cam_depth_64 = sensor_config_camera(64, 64, 1.57f, 100.0f);
    cam_depth_64.type = SENSOR_TYPE_CAMERA_DEPTH;

    /* Camera RGB 32x32 */
    cam_rgb_32 = sensor_config_camera(32, 32, 1.57f, 100.0f);
    /* sensor_config_camera defaults to RGB */

    /* LiDAR 3D */
    lidar3d = sensor_config_lidar_3d(16, 64, (float)(2.0 * M_PI), (float)(M_PI / 3.0), 50.0f);

    /* LiDAR 2D */
    lidar2d = sensor_config_lidar_2d(64, (float)M_PI, 20.0f);

    /* ToF */
    Vec3 tof_dir = {1.0f, 0.0f, 0.0f};
    tof = sensor_config_tof(tof_dir, 50.0f);

    /* CPU-only sensors */
    imu = sensor_config_imu();

    /* ====================================================================
     * Individual Sensor Benchmarks
     * ==================================================================== */

    uint32_t drone_count = 1024;
    if (cli.num_drone_counts > 0) {
        drone_count = cli.drone_counts[0];
    }
    uint32_t warmup = cli.warmup > 0 ? cli.warmup : 10;
    uint32_t iters = cli.iterations > 0 ? cli.iterations : 50;

    printf("=== Individual Sensor GPU Benchmarks (%u drones) ===\n\n", drone_count);
    gpu_print_header();

    GpuSensorBenchResult results[16];
    uint32_t num_results = 0;

    /* Camera depth 32x32 */
    {
        SensorConfig cfgs[2] = {imu, cam_depth_32};
        results[num_results] = bench_sensor_gpu("cam_depth_32+imu", cfgs, 2,
                                                  drone_count, warmup, iters);
        gpu_print_row(&results[num_results++]);
    }

    /* Camera depth 64x64 */
    {
        SensorConfig cfgs[2] = {imu, cam_depth_64};
        results[num_results] = bench_sensor_gpu("cam_depth_64+imu", cfgs, 2,
                                                  drone_count, warmup, iters);
        gpu_print_row(&results[num_results++]);
    }

    /* Camera RGB 32x32 */
    {
        SensorConfig cfgs[2] = {imu, cam_rgb_32};
        results[num_results] = bench_sensor_gpu("cam_rgb_32+imu", cfgs, 2,
                                                  drone_count, warmup, iters);
        gpu_print_row(&results[num_results++]);
    }

    /* LiDAR 3D */
    {
        SensorConfig cfgs[2] = {imu, lidar3d};
        results[num_results] = bench_sensor_gpu("lidar3d_16x64+imu", cfgs, 2,
                                                  drone_count, warmup, iters);
        gpu_print_row(&results[num_results++]);
    }

    /* LiDAR 2D */
    {
        SensorConfig cfgs[2] = {imu, lidar2d};
        results[num_results] = bench_sensor_gpu("lidar2d_64+imu", cfgs, 2,
                                                  drone_count, warmup, iters);
        gpu_print_row(&results[num_results++]);
    }

    /* ToF */
    {
        SensorConfig cfgs[2] = {imu, tof};
        results[num_results] = bench_sensor_gpu("tof+imu", cfgs, 2,
                                                  drone_count, warmup, iters);
        gpu_print_row(&results[num_results++]);
    }

    printf("--------------------------------------------------------------------------\n");

    /* ====================================================================
     * Scaling Test: Camera Depth 32x32
     * ==================================================================== */

    printf("\n=== Scaling: cam_depth_32 (GPU vs CPU) ===\n\n");
    gpu_print_header();

    uint32_t scale_counts[] = {64, 256, 512, 1024};
    uint32_t num_scale = sizeof(scale_counts) / sizeof(scale_counts[0]);

    for (uint32_t i = 0; i < num_scale; i++) {
        SensorConfig cfgs[2] = {imu, cam_depth_32};
        GpuSensorBenchResult r = bench_sensor_gpu("cam_depth_32+imu", cfgs, 2,
                                                    scale_counts[i], warmup / 2, iters / 2);
        gpu_print_row(&r);
    }

    printf("--------------------------------------------------------------------------\n");

    /* Summary */
    printf("\n");
    uint32_t passed = 0;
    for (uint32_t i = 0; i < num_results; i++) {
        if (results[i].passed) passed++;
    }
    printf("Summary: %u/%u GPU faster than CPU\n\n", passed, num_results);

    return 0;
}
