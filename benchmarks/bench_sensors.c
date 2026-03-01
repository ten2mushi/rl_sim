/**
 * Comprehensive Sensor Benchmarks
 *
 * Tests individual sensors WITH real world geometry (CSG), sensor combinations,
 * scaling across drone counts, memory reporting, and degradation checks.
 * This is the most thorough sensor benchmark, unlike the per-module bench
 * which passes NULL world.
 */

#include "bench_harness.h"
#include "sensor_system.h"
#include "sensor_implementations.h"
#include "collision_system.h"
#include "world_brick_map.h"
#include "platform_quadcopter.h"

/* ============================================================================
 * Fixture
 * ============================================================================ */

typedef struct SensorBenchCtx {
    Arena* persistent;
    Arena* scratch;
    SensorSystem* sys;
    PlatformStateSOA* drones;
    WorldBrickMap* world;
    CollisionSystem* collision;
    uint32_t num_agents;
} SensorBenchCtx;

static SensorBenchCtx* sensor_ctx_create(uint32_t num_agents, bool with_world,
                                          bool with_collision, uint64_t seed) {
    size_t pa_size = 256 * 1024 * 1024;
    size_t sa_size = 64 * 1024 * 1024;
    Arena* persistent = arena_create(pa_size);
    Arena* scratch = arena_create(sa_size);
    if (!persistent || !scratch) return NULL;

    SensorBenchCtx* ctx = arena_alloc_type(persistent, SensorBenchCtx);
    memset(ctx, 0, sizeof(*ctx));
    ctx->persistent = persistent;
    ctx->scratch = scratch;
    ctx->num_agents = num_agents;

    /* Create sensor system with enough obs buffer */
    ctx->sys = sensor_system_create(persistent, num_agents, 8, 16384);
    if (!ctx->sys) return NULL;
    sensor_implementations_register_all(&ctx->sys->registry);

    /* Create drone state */
    ctx->drones = platform_state_create(persistent, num_agents, QUAD_STATE_EXT_COUNT);
    if (!ctx->drones) return NULL;
    bench_init_drones_grid(ctx->drones, num_agents, 10.0f, seed);

    /* Create world with CSG geometry */
    if (with_world) {
        ctx->world = world_create(persistent, VEC3(-50, -50, -10),
                                   VEC3(50, 50, 50), 0.1f, 10000, 0);
        if (ctx->world) {
            world_set_box(ctx->world, VEC3(0, 0, -5), VEC3(50, 50, 5), 1);
            world_set_sphere(ctx->world, VEC3(10, 0, 5), 3.0f, 1);
        }
    }

    /* Create collision system for neighbor sensor */
    if (with_collision) {
        ctx->collision = collision_create(persistent, num_agents, 0.1f, 2.0f);
        if (ctx->collision) {
            collision_build_spatial_hash(ctx->collision, ctx->drones, num_agents);
        }
    }

    return ctx;
}

static void sensor_ctx_destroy(SensorBenchCtx* ctx) {
    if (!ctx) return;
    if (ctx->sys) sensor_system_destroy(ctx->sys);
    arena_destroy(ctx->scratch);
    arena_destroy(ctx->persistent);
}

/* ============================================================================
 * Single-sensor benchmark helper
 * ============================================================================ */

typedef struct SingleSensorCtx {
    SensorSystem* sys;
    PlatformStateSOA* drones;
    WorldBrickMap* world;
    CollisionSystem* collision;
    uint32_t num_agents;
} SingleSensorCtx;

static void fn_sensor_sample_all(void* arg) {
    SingleSensorCtx* ctx = (SingleSensorCtx*)arg;
    sensor_system_sample_all(ctx->sys, ctx->drones, ctx->world,
                              ctx->collision, ctx->num_agents);
}

static BenchStats run_single_sensor(const char* name, SensorConfig config,
                                     uint32_t num_agents, bool needs_world,
                                     bool needs_collision, double target_ms,
                                     uint32_t warmup, uint32_t iterations,
                                     uint64_t seed) {
    SensorBenchCtx* ctx = sensor_ctx_create(num_agents, needs_world, needs_collision, seed);
    BenchStats s = {0};
    s.name = name;
    s.agent_count = num_agents;
    if (!ctx) return s;

    uint32_t sensor_idx = sensor_system_create_sensor(ctx->sys, &config);
    if (sensor_idx == UINT32_MAX) {
        sensor_ctx_destroy(ctx);
        return s;
    }

    for (uint32_t d = 0; d < num_agents; d++) {
        sensor_system_attach(ctx->sys, d, sensor_idx);
    }

    /* Rebuild collision hash if needed */
    if (needs_collision && ctx->collision) {
        collision_build_spatial_hash(ctx->collision, ctx->drones, num_agents);
    }

    SingleSensorCtx sctx = {
        .sys = ctx->sys,
        .drones = ctx->drones,
        .world = ctx->world,
        .collision = ctx->collision,
        .num_agents = num_agents
    };

    s = bench_measure(name, fn_sensor_sample_all, &sctx, warmup, iterations, target_ms);
    s.agent_count = num_agents;
    s.persistent_bytes = ctx->persistent->used;
    s.frame_bytes = ctx->scratch->used;

    sensor_ctx_destroy(ctx);
    return s;
}

/* ============================================================================
 * Combination benchmark helper
 * ============================================================================ */

static BenchStats run_combo_sensor(const char* name, SensorConfig* configs,
                                    uint32_t num_configs, uint32_t num_agents,
                                    bool needs_world, bool needs_collision,
                                    double target_ms, uint32_t warmup,
                                    uint32_t iterations, uint64_t seed) {
    SensorBenchCtx* ctx = sensor_ctx_create(num_agents, needs_world, needs_collision, seed);
    BenchStats s = {0};
    s.name = name;
    s.agent_count = num_agents;
    if (!ctx) return s;

    for (uint32_t c = 0; c < num_configs; c++) {
        uint32_t sensor_idx = sensor_system_create_sensor(ctx->sys, &configs[c]);
        if (sensor_idx == UINT32_MAX) {
            sensor_ctx_destroy(ctx);
            return s;
        }
        for (uint32_t d = 0; d < num_agents; d++) {
            sensor_system_attach(ctx->sys, d, sensor_idx);
        }
    }

    if (needs_collision && ctx->collision) {
        collision_build_spatial_hash(ctx->collision, ctx->drones, num_agents);
    }

    SingleSensorCtx sctx = {
        .sys = ctx->sys,
        .drones = ctx->drones,
        .world = ctx->world,
        .collision = ctx->collision,
        .num_agents = num_agents
    };

    s = bench_measure(name, fn_sensor_sample_all, &sctx, warmup, iterations, target_ms);
    s.agent_count = num_agents;
    s.persistent_bytes = ctx->persistent->used;
    s.frame_bytes = ctx->scratch->used;

    sensor_ctx_destroy(ctx);
    return s;
}

/* ============================================================================
 * Scaling helper
 * ============================================================================ */

static BenchStats bench_lidar2d_scaling(uint32_t agent_count, uint32_t iterations,
                                         uint32_t warmup, uint64_t seed) {
    SensorConfig config = sensor_config_lidar_2d(64, 3.14159f, 20.0f);
    return run_single_sensor("lidar2d_64", config, agent_count, true, false,
                              5.0, warmup, iterations, seed);
}

static BenchStats bench_camera_depth_scaling(uint32_t agent_count, uint32_t iterations,
                                              uint32_t warmup, uint64_t seed) {
    SensorConfig config = sensor_config_camera(32, 32, 1.57f, 100.0f);
    config.type = SENSOR_TYPE_CAMERA_DEPTH;
    return run_single_sensor("camera_depth_32", config, agent_count, true, false,
                              4.0, warmup, iterations, seed);
}

static BenchStats bench_neighbor_scaling(uint32_t agent_count, uint32_t iterations,
                                          uint32_t warmup, uint64_t seed) {
    SensorConfig config = sensor_config_neighbor(5, 20.0f);
    return run_single_sensor("neighbor_k5", config, agent_count, false, true,
                              1.0, warmup, iterations, seed);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char** argv) {
    BenchCLI cli = bench_parse_cli(argc, argv);

    printf("=== Comprehensive Sensor Benchmarks ===\n");
    printf("SIMD width: %d floats\n\n", FOUNDATION_SIMD_WIDTH);

    uint32_t N = 1024;
    BenchStats results[32];
    uint32_t num_results = 0;

    /* --- Individual Sensor Benchmarks --- */
    printf("--- Individual Sensors (%u drones) ---\n", N);
    bench_print_header();

    /* Non-world sensors */
    {
        SensorConfig c;
        BenchStats s;

        c = sensor_config_imu();
        s = run_single_sensor("imu_noise", c, N, false, false, 0.15,
                               cli.warmup, cli.iterations, cli.seed);
        bench_print_row(&s);
        results[num_results++] = s;

        c = sensor_config_imu();
        s = run_single_sensor("imu_clean", c, N, false, false, 0.10,
                               cli.warmup, cli.iterations, cli.seed);
        bench_print_row(&s);
        results[num_results++] = s;

        c = sensor_config_position();
        s = run_single_sensor("position", c, N, false, false, 0.05,
                               cli.warmup, cli.iterations, cli.seed);
        bench_print_row(&s);
        results[num_results++] = s;

        c = sensor_config_velocity();
        s = run_single_sensor("velocity", c, N, false, false, 0.05,
                               cli.warmup, cli.iterations, cli.seed);
        bench_print_row(&s);
        results[num_results++] = s;
    }

    /* World-dependent sensors */
    {
        SensorConfig c;
        BenchStats s;

        c = sensor_config_tof(VEC3(0, 0, -1), 10.0f);
        s = run_single_sensor("tof_world", c, N, true, false, 1.0,
                               cli.warmup, cli.iterations, cli.seed);
        bench_print_row(&s);
        results[num_results++] = s;

        c = sensor_config_lidar_2d(32, 3.14159f, 20.0f);
        s = run_single_sensor("lidar2d_32", c, N, true, false, 3.0,
                               cli.warmup, cli.iterations, cli.seed);
        bench_print_row(&s);
        results[num_results++] = s;

        c = sensor_config_lidar_2d(64, 3.14159f, 20.0f);
        s = run_single_sensor("lidar2d_64", c, N, true, false, 5.0,
                               cli.warmup, cli.iterations, cli.seed);
        bench_print_row(&s);
        results[num_results++] = s;

        c = sensor_config_lidar_2d(128, 3.14159f, 20.0f);
        s = run_single_sensor("lidar2d_128", c, N, true, false, 10.0,
                               cli.warmup, cli.iterations, cli.seed);
        bench_print_row(&s);
        results[num_results++] = s;

        c = sensor_config_lidar_3d(64, 16, 6.28f, 0.5f, 50.0f);
        s = run_single_sensor("lidar3d_16x64", c, N, true, false, 20.0,
                               cli.warmup, cli.iterations, cli.seed);
        bench_print_row(&s);
        results[num_results++] = s;

        c = sensor_config_camera(32, 32, 1.57f, 100.0f);
        c.type = SENSOR_TYPE_CAMERA_RGB;
        s = run_single_sensor("camera_rgb_32", c, N, true, false, 5.0,
                               cli.warmup, cli.iterations, cli.seed);
        bench_print_row(&s);
        results[num_results++] = s;

        c = sensor_config_camera(64, 64, 1.57f, 100.0f);
        c.type = SENSOR_TYPE_CAMERA_RGB;
        s = run_single_sensor("camera_rgb_64", c, N, true, false, 20.0,
                               cli.warmup, cli.iterations, cli.seed);
        bench_print_row(&s);
        results[num_results++] = s;

        c = sensor_config_camera(32, 32, 1.57f, 100.0f);
        c.type = SENSOR_TYPE_CAMERA_DEPTH;
        s = run_single_sensor("camera_depth_32", c, N, true, false, 4.0,
                               cli.warmup, cli.iterations, cli.seed);
        bench_print_row(&s);
        results[num_results++] = s;

        c = sensor_config_camera(64, 64, 1.57f, 100.0f);
        c.type = SENSOR_TYPE_CAMERA_DEPTH;
        s = run_single_sensor("camera_depth_64", c, N, true, false, 15.0,
                               cli.warmup, cli.iterations, cli.seed);
        bench_print_row(&s);
        results[num_results++] = s;

        c = sensor_config_camera(32, 32, 1.57f, 100.0f);
        c.type = SENSOR_TYPE_CAMERA_SEGMENTATION;
        s = run_single_sensor("camera_seg_32", c, N, true, false, 4.0,
                               cli.warmup, cli.iterations, cli.seed);
        bench_print_row(&s);
        results[num_results++] = s;
    }

    /* Neighbor sensors */
    {
        SensorConfig c;
        BenchStats s;

        c = sensor_config_neighbor(5, 20.0f);
        s = run_single_sensor("neighbor_k5", c, N, false, true, 1.0,
                               cli.warmup, cli.iterations, cli.seed);
        bench_print_row(&s);
        results[num_results++] = s;

        c = sensor_config_neighbor(10, 20.0f);
        s = run_single_sensor("neighbor_k10", c, N, false, true, 2.0,
                               cli.warmup, cli.iterations, cli.seed);
        bench_print_row(&s);
        results[num_results++] = s;
    }

    /* --- Combination Benchmarks --- */
    printf("\n--- Sensor Combinations (%u drones) ---\n", N);
    bench_print_header();

    /* combo_basic: IMU + Position + Velocity */
    {
        SensorConfig configs[3];
        configs[0] = sensor_config_imu();
        configs[1] = sensor_config_position();
        configs[2] = sensor_config_velocity();
        BenchStats s = run_combo_sensor("combo_basic", configs, 3, N,
                                         false, false, 0.3,
                                         cli.warmup, cli.iterations, cli.seed);
        bench_print_row(&s);
        results[num_results++] = s;
    }

    /* combo_navigation: IMU + LiDAR-2D(64) + Position */
    {
        SensorConfig configs[3];
        configs[0] = sensor_config_imu();
        configs[1] = sensor_config_lidar_2d(64, 3.14159f, 20.0f);
        configs[2] = sensor_config_position();
        BenchStats s = run_combo_sensor("combo_navigation", configs, 3, N,
                                         true, false, 6.0,
                                         cli.warmup, cli.iterations, cli.seed);
        bench_print_row(&s);
        results[num_results++] = s;
    }

    /* combo_full: IMU + LiDAR-2D(64) + Camera-Depth(32) + Neighbor(5) */
    {
        SensorConfig configs[4];
        configs[0] = sensor_config_imu();
        configs[1] = sensor_config_lidar_2d(64, 3.14159f, 20.0f);
        configs[2] = sensor_config_camera(32, 32, 1.57f, 100.0f);
        configs[2].type = SENSOR_TYPE_CAMERA_DEPTH;
        configs[3] = sensor_config_neighbor(5, 20.0f);
        BenchStats s = run_combo_sensor("combo_full", configs, 4, N,
                                         true, true, 25.0,
                                         cli.warmup, cli.iterations, cli.seed);
        bench_print_row(&s);
        results[num_results++] = s;
    }

    /* combo_stress: IMU + LiDAR-3D(16x64) + Camera-RGB(64) + Neighbor(10) */
    {
        SensorConfig configs[4];
        configs[0] = sensor_config_imu();
        configs[1] = sensor_config_lidar_3d(64, 16, 6.28f, 0.5f, 50.0f);
        configs[2] = sensor_config_camera(64, 64, 1.57f, 100.0f);
        configs[2].type = SENSOR_TYPE_CAMERA_RGB;
        configs[3] = sensor_config_neighbor(10, 20.0f);
        BenchStats s = run_combo_sensor("combo_stress", configs, 4, N,
                                         true, true, 100.0,
                                         cli.warmup, cli.iterations, cli.seed);
        bench_print_row(&s);
        results[num_results++] = s;
    }

    /* --- Scaling Tests --- */
    bench_scaling_test("lidar2d_64", bench_lidar2d_scaling, &cli);
    bench_scaling_test("camera_depth_32", bench_camera_depth_scaling, &cli);
    bench_scaling_test("neighbor_k5", bench_neighbor_scaling, &cli);

    /* --- Memory Report --- */
    printf("\n--- Memory Report (1024 drones) ---\n");
    {
        /* Run combo_navigation and report memory */
        SensorBenchCtx* ctx = sensor_ctx_create(1024, true, true, cli.seed);
        if (ctx) {
            SensorConfig configs[3];
            configs[0] = sensor_config_imu();
            configs[1] = sensor_config_lidar_2d(64, 3.14159f, 20.0f);
            configs[2] = sensor_config_position();
            for (uint32_t c = 0; c < 3; c++) {
                uint32_t si = sensor_system_create_sensor(ctx->sys, &configs[c]);
                if (si != UINT32_MAX) {
                    for (uint32_t d = 0; d < 1024; d++) {
                        sensor_system_attach(ctx->sys, d, si);
                    }
                }
            }
            bench_report_arena("persistent", ctx->persistent);
            bench_report_arena("scratch", ctx->scratch);
            sensor_ctx_destroy(ctx);
        }
    }

    /* --- Degradation Test --- */
    printf("\n--- Degradation Test (combo_navigation, 5000 iters, 500-step blocks) ---\n");
    {
        SensorBenchCtx* ctx = sensor_ctx_create(N, true, false, cli.seed);
        if (ctx) {
            SensorConfig configs[3];
            configs[0] = sensor_config_imu();
            configs[1] = sensor_config_lidar_2d(64, 3.14159f, 20.0f);
            configs[2] = sensor_config_position();
            for (uint32_t c = 0; c < 3; c++) {
                uint32_t si = sensor_system_create_sensor(ctx->sys, &configs[c]);
                if (si != UINT32_MAX) {
                    for (uint32_t d = 0; d < N; d++) {
                        sensor_system_attach(ctx->sys, d, si);
                    }
                }
            }

            SingleSensorCtx sctx = {
                .sys = ctx->sys,
                .drones = ctx->drones,
                .world = ctx->world,
                .collision = ctx->collision,
                .num_agents = N
            };

            bench_check_degradation("combo_navigation", fn_sensor_sample_all, &sctx,
                                     5000, 500, 10.0);
            sensor_ctx_destroy(ctx);
        }
    }

    /* --- Summary --- */
    bench_print_separator();
    bench_print_summary(results, num_results);

    return 0;
}
