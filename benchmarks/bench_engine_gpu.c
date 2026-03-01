/**
 * GPU-Enabled Engine Benchmarks
 *
 * End-to-end benchmarks of BatchEngine with GPU sensor acceleration.
 * Tests all 6 profiles in two modes: GPU active vs CPU fallback.
 * Shows per-phase breakdown and total step time.
 */

#include "bench_harness.h"
#include "environment_manager.h"
#include "gpu_hal.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * Engine Profile Definitions (mirrors bench_engine.c)
 * ============================================================================ */

typedef enum EngineProfile {
    PROFILE_MINIMAL,
    PROFILE_LIGHT,
    PROFILE_NAVIGATION,
    PROFILE_VISION,
    PROFILE_FULL,
    PROFILE_STRESS,
    PROFILE_COUNT
} EngineProfile;

static const char* profile_names[] = {
    "MINIMAL", "LIGHT", "NAVIGATION", "VISION", "FULL", "STRESS"
};

static EngineConfig make_profile_config(EngineProfile profile, uint32_t total_agents) {
    EngineConfig config = engine_config_default();
    config.agents_per_env = 16;
    config.num_envs = total_agents / config.agents_per_env;
    if (config.num_envs == 0) config.num_envs = 1;
    config.total_agents = config.num_envs * config.agents_per_env;
    config.num_threads = 0;
    config.seed = 42;

    SensorConfig sc;

    switch (profile) {
    case PROFILE_MINIMAL:
        sc = sensor_config_imu();
        engine_config_add_sensor(&config, &sc);
        break;

    case PROFILE_LIGHT:
        sc = sensor_config_imu();
        engine_config_add_sensor(&config, &sc);
        sc = sensor_config_position();
        engine_config_add_sensor(&config, &sc);
        break;

    case PROFILE_NAVIGATION:
        sc = sensor_config_imu();
        engine_config_add_sensor(&config, &sc);
        sc = sensor_config_lidar_2d(64, (float)M_PI, 20.0f);
        engine_config_add_sensor(&config, &sc);
        sc = sensor_config_position();
        engine_config_add_sensor(&config, &sc);
        break;

    case PROFILE_VISION:
        sc = sensor_config_imu();
        engine_config_add_sensor(&config, &sc);
        sc = sensor_config_camera(32, 32, 1.57f, 100.0f);
        sc.type = SENSOR_TYPE_CAMERA_DEPTH;
        engine_config_add_sensor(&config, &sc);
        sc = sensor_config_position();
        engine_config_add_sensor(&config, &sc);
        break;

    case PROFILE_FULL:
        sc = sensor_config_imu();
        engine_config_add_sensor(&config, &sc);
        sc = sensor_config_camera(32, 32, 1.57f, 100.0f);
        sc.type = SENSOR_TYPE_CAMERA_DEPTH;
        engine_config_add_sensor(&config, &sc);
        sc = sensor_config_lidar_2d(64, (float)M_PI, 20.0f);
        engine_config_add_sensor(&config, &sc);
        sc = sensor_config_position();
        engine_config_add_sensor(&config, &sc);
        sc = sensor_config_velocity();
        engine_config_add_sensor(&config, &sc);
        break;

    case PROFILE_STRESS:
        sc = sensor_config_imu();
        engine_config_add_sensor(&config, &sc);
        sc = sensor_config_camera(64, 64, 1.57f, 100.0f);
        sc.type = SENSOR_TYPE_CAMERA_DEPTH;
        engine_config_add_sensor(&config, &sc);
        sc = sensor_config_camera(64, 64, 1.57f, 100.0f);
        /* RGB by default */
        engine_config_add_sensor(&config, &sc);
        sc = sensor_config_lidar_3d(16, 64, (float)(2.0 * M_PI), (float)(M_PI / 3.0), 50.0f);
        engine_config_add_sensor(&config, &sc);
        sc = sensor_config_position();
        engine_config_add_sensor(&config, &sc);
        sc = sensor_config_velocity();
        engine_config_add_sensor(&config, &sc);
        break;

    default:
        break;
    }

    return config;
}

/* ============================================================================
 * Benchmark Result
 * ============================================================================ */

typedef struct ProfileResult {
    const char* name;
    uint32_t    agent_count;
    double      gpu_step_ms;
    double      cpu_step_ms;
    double      gpu_physics_ms;
    double      gpu_collision_ms;
    double      gpu_sensor_ms;
    double      gpu_sensor_gpu_ms;
    double      gpu_reward_ms;
    double      speedup;
} ProfileResult;

/* ============================================================================
 * Run a profile in both modes
 * ============================================================================ */

static ProfileResult bench_profile(EngineProfile profile, uint32_t num_agents,
                                     uint32_t warmup, uint32_t iterations) {
    ProfileResult result = {0};
    result.name = profile_names[profile];
    result.agent_count = num_agents;

    EngineConfig cfg = make_profile_config(profile, num_agents);
    char error[ENGINE_ERROR_MSG_SIZE];

    BatchEngine* engine = engine_create(&cfg, error);
    if (!engine) {
        fprintf(stderr, "  Failed: %s\n", error);
        return result;
    }

    engine_reset(engine);

    /* ----- GPU mode ----- */
    for (uint32_t i = 0; i < warmup; i++) {
        engine_step(engine);
    }

    double gpu_total = 0.0;
    double physics_total = 0.0, collision_total = 0.0;
    double sensor_total = 0.0, sensor_gpu_total = 0.0, reward_total = 0.0;

    for (uint32_t i = 0; i < iterations; i++) {
        double t0 = bench_time_ms();
        engine_step(engine);
        double t1 = bench_time_ms();
        gpu_total += (t1 - t0);
        physics_total += engine->physics_time_ms;
        collision_total += engine->collision_time_ms;
        sensor_total += engine->sensor_time_ms;
        sensor_gpu_total += engine->gpu_sensor_time_ms;
        reward_total += engine->reward_time_ms;
    }

    result.gpu_step_ms = gpu_total / iterations;
    result.gpu_physics_ms = physics_total / iterations;
    result.gpu_collision_ms = collision_total / iterations;
    result.gpu_sensor_ms = sensor_total / iterations;
    result.gpu_sensor_gpu_ms = sensor_gpu_total / iterations;
    result.gpu_reward_ms = reward_total / iterations;

    /* ----- CPU mode ----- */
    struct GpuSensorContext* saved_ctx = engine->gpu_sensor_ctx;
    engine->gpu_sensor_ctx = NULL;

    for (uint32_t i = 0; i < warmup; i++) {
        engine_step(engine);
    }

    double cpu_total = 0.0;
    for (uint32_t i = 0; i < iterations; i++) {
        double t0 = bench_time_ms();
        engine_step(engine);
        double t1 = bench_time_ms();
        cpu_total += (t1 - t0);
    }
    result.cpu_step_ms = cpu_total / iterations;

    /* Restore and cleanup */
    engine->gpu_sensor_ctx = saved_ctx;
    if (result.gpu_step_ms > 0.001) {
        result.speedup = result.cpu_step_ms / result.gpu_step_ms;
    }

    engine_destroy(engine);
    return result;
}

/* ============================================================================
 * Printing
 * ============================================================================ */

static void print_profile_header(void) {
    printf("%-14s %7s %9s %9s %9s | %8s %8s %8s %8s %8s\n",
           "PROFILE", "DRONES", "CPU_MS", "GPU_MS", "SPEEDUP",
           "PHYSICS", "COLLISN", "SENSOR", "GPU_SNS", "REWARD");
    printf("----------------------------------------------------------------------"
           "----------------------------------------------\n");
}

static void print_profile_row(const ProfileResult* r) {
    printf("%-14s %7u %9.3f %9.3f %8.1fx | %8.3f %8.3f %8.3f %8.3f %8.3f\n",
           r->name, r->agent_count,
           r->cpu_step_ms, r->gpu_step_ms, r->speedup,
           r->gpu_physics_ms, r->gpu_collision_ms,
           r->gpu_sensor_ms, r->gpu_sensor_gpu_ms, r->gpu_reward_ms);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char** argv) {
    BenchCLI cli = bench_parse_cli(argc, argv);

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           GPU-Enabled Engine Benchmarks                    ║\n");
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

    uint32_t agent_count = 1024;
    if (cli.num_agent_counts > 0) {
        agent_count = cli.agent_counts[0];
    }
    uint32_t warmup = cli.warmup > 0 ? cli.warmup : 10;
    uint32_t iters = cli.iterations > 0 ? cli.iterations : 50;

    /* ====================================================================
     * All Profiles at Fixed Drone Count
     * ==================================================================== */

    printf("=== All Profiles (%u drones, %u iters) ===\n\n", agent_count, iters);
    print_profile_header();

    ProfileResult all_results[PROFILE_COUNT];
    for (int p = 0; p < PROFILE_COUNT; p++) {
        all_results[p] = bench_profile((EngineProfile)p, agent_count, warmup, iters);
        print_profile_row(&all_results[p]);
    }

    printf("----------------------------------------------------------------------"
           "----------------------------------------------\n");

    /* ====================================================================
     * Scaling: VISION profile
     * ==================================================================== */

    printf("\n=== Scaling: VISION profile (cam_depth_32 + IMU + pos) ===\n\n");
    print_profile_header();

    uint32_t scale_counts[] = {64, 256, 512, 1024};
    uint32_t num_scale = sizeof(scale_counts) / sizeof(scale_counts[0]);

    for (uint32_t i = 0; i < num_scale; i++) {
        ProfileResult r = bench_profile(PROFILE_VISION, scale_counts[i],
                                          warmup / 2, iters / 2);
        print_profile_row(&r);
    }

    printf("----------------------------------------------------------------------"
           "----------------------------------------------\n");

    /* Summary */
    printf("\nTarget: 1024 drones at 50 FPS = 20ms/step\n");
    printf("VISION profile: %.1fms/step (GPU), %.1fms/step (CPU), %.1fx speedup\n",
           all_results[PROFILE_VISION].gpu_step_ms,
           all_results[PROFILE_VISION].cpu_step_ms,
           all_results[PROFILE_VISION].speedup);
    printf("VISION target met: %s\n\n",
           all_results[PROFILE_VISION].gpu_step_ms < 20.0 ? "YES" : "NO");

    return 0;
}
