/**
 * Full Pipeline Engine Benchmarks
 *
 * End-to-end benchmarks of BatchEngine with different sensor profiles.
 * Tests MINIMAL through STRESS configurations with phase breakdown,
 * scaling tests, memory reporting, and sustained performance checks.
 */

#include "bench_harness.h"
#include "environment_manager.h"

/* ============================================================================
 * Engine Profile Definitions
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

static double profile_targets[] = {
    8.0, 10.0, 20.0, 25.0, 30.0, 100.0
};

static EngineConfig make_profile_config(EngineProfile profile, uint32_t total_agents) {
    EngineConfig config = engine_config_default();
    /* Configure for specified total drones */
    config.agents_per_env = 16;
    config.num_envs = total_agents / config.agents_per_env;
    if (config.num_envs == 0) config.num_envs = 1;
    config.total_agents = config.num_envs * config.agents_per_env;
    config.num_threads = 0; /* auto-detect */
    config.seed = 42;

    /* Sensor configs are allocated on stack - engine_config_add_sensor copies them */
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
        sc = sensor_config_lidar_2d(64, 3.14159f, 20.0f);
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
        sc = sensor_config_lidar_2d(64, 3.14159f, 20.0f);
        engine_config_add_sensor(&config, &sc);
        sc = sensor_config_camera(32, 32, 1.57f, 100.0f);
        sc.type = SENSOR_TYPE_CAMERA_DEPTH;
        engine_config_add_sensor(&config, &sc);
        sc = sensor_config_neighbor(5, 20.0f);
        engine_config_add_sensor(&config, &sc);
        break;

    case PROFILE_STRESS:
        sc = sensor_config_imu();
        engine_config_add_sensor(&config, &sc);
        sc = sensor_config_lidar_3d(64, 16, 6.28f, 0.5f, 50.0f);
        engine_config_add_sensor(&config, &sc);
        sc = sensor_config_camera(64, 64, 1.57f, 100.0f);
        sc.type = SENSOR_TYPE_CAMERA_RGB;
        engine_config_add_sensor(&config, &sc);
        sc = sensor_config_neighbor(10, 20.0f);
        engine_config_add_sensor(&config, &sc);
        break;

    default:
        break;
    }

    return config;
}

/* ============================================================================
 * Benchmark Context
 * ============================================================================ */

typedef struct EngineBenchCtx {
    BatchEngine* engine;
    uint32_t total_agents;
} EngineBenchCtx;

static void fn_engine_step(void* arg) {
    EngineBenchCtx* ctx = (EngineBenchCtx*)arg;
    engine_step(ctx->engine);
}

/* ============================================================================
 * Profile Benchmark Runner
 * ============================================================================ */

static BenchStats run_profile(EngineProfile profile, uint32_t total_agents,
                               uint32_t warmup, uint32_t iterations) {
    BenchStats s = {0};
    s.name = profile_names[profile];
    s.agent_count = total_agents;
    s.target_ms = profile_targets[profile];

    EngineConfig config = make_profile_config(profile, total_agents);

    char error_msg[ENGINE_ERROR_MSG_SIZE];
    BatchEngine* engine = engine_create(&config, error_msg);
    if (!engine) {
        fprintf(stderr, "  Failed to create engine for %s: %s\n",
                profile_names[profile], error_msg);
        return s;
    }

    /* Set random actions */
    float* actions = engine_get_actions(engine);
    PCG32 rng;
    pcg32_seed(&rng, 42);
    for (uint32_t i = 0; i < config.total_agents * engine_get_action_dim(engine); i++) {
        actions[i] = pcg32_range(&rng, 0.2f, 0.8f);
    }

    engine_reset(engine);

    /* Add CSG geometry for profiles that use sensors requiring world */
    if (profile >= PROFILE_NAVIGATION) {
        engine_add_box(engine, VEC3(-50, -50, -10), VEC3(50, 50, 0), 1);
        engine_add_sphere(engine, VEC3(10, 0, 5), 3.0f, 1);
    }

    EngineBenchCtx ctx = { .engine = engine, .total_agents = config.total_agents };
    s = bench_measure(profile_names[profile], fn_engine_step, &ctx,
                       warmup, iterations, profile_targets[profile]);
    s.agent_count = config.total_agents;

    /* Get memory stats */
    EngineStats stats;
    engine_get_stats(engine, &stats);
    s.persistent_bytes = stats.persistent_memory_used;
    s.frame_bytes = stats.frame_memory_used;

    engine_destroy(engine);
    return s;
}

/* ============================================================================
 * Phase Breakdown
 * ============================================================================ */

static void print_phase_breakdown(EngineProfile profile, uint32_t total_agents,
                                   uint32_t num_steps) {
    EngineConfig config = make_profile_config(profile, total_agents);

    char error_msg[ENGINE_ERROR_MSG_SIZE];
    BatchEngine* engine = engine_create(&config, error_msg);
    if (!engine) return;

    /* Set actions */
    float* actions = engine_get_actions(engine);
    PCG32 rng;
    pcg32_seed(&rng, 42);
    for (uint32_t i = 0; i < config.total_agents * engine_get_action_dim(engine); i++) {
        actions[i] = pcg32_range(&rng, 0.2f, 0.8f);
    }

    engine_reset(engine);

    /* Add world geometry */
    if (profile >= PROFILE_NAVIGATION) {
        engine_add_box(engine, VEC3(-50, -50, -10), VEC3(50, 50, 0), 1);
        engine_add_sphere(engine, VEC3(10, 0, 5), 3.0f, 1);
    }

    /* Warmup */
    for (uint32_t i = 0; i < 20; i++) {
        engine_step(engine);
    }

    /* Accumulate phase timings */
    double total_physics = 0, total_collision = 0, total_sensor = 0;
    double total_reward = 0, total_reset = 0, total_step = 0;

    for (uint32_t i = 0; i < num_steps; i++) {
        double t0 = bench_time_ms();

        engine_step_physics(engine);
        double t1 = bench_time_ms();

        engine_step_collision(engine);
        double t2 = bench_time_ms();

        engine_step_sensors(engine);
        double t3 = bench_time_ms();

        engine_step_rewards(engine);
        double t4 = bench_time_ms();

        engine_step_reset_terminated(engine);
        double t5 = bench_time_ms();

        total_physics += (t1 - t0);
        total_collision += (t2 - t1);
        total_sensor += (t3 - t2);
        total_reward += (t4 - t3);
        total_reset += (t5 - t4);
        total_step += (t5 - t0);
    }

    double avg_total = total_step / num_steps;
    double avg_phys = total_physics / num_steps;
    double avg_coll = total_collision / num_steps;
    double avg_sens = total_sensor / num_steps;
    double avg_rew = total_reward / num_steps;
    double avg_rst = total_reset / num_steps;

    printf("\nPhase Breakdown (%s, %u drones, avg over %u steps):\n",
           profile_names[profile], config.total_agents, num_steps);
    printf("  Physics:    %7.3f ms (%5.1f%%)\n", avg_phys, avg_phys / avg_total * 100);
    printf("  Collision:  %7.3f ms (%5.1f%%)\n", avg_coll, avg_coll / avg_total * 100);
    printf("  Sensors:    %7.3f ms (%5.1f%%)\n", avg_sens, avg_sens / avg_total * 100);
    printf("  Rewards:    %7.3f ms (%5.1f%%)\n", avg_rew, avg_rew / avg_total * 100);
    printf("  Reset:      %7.3f ms (%5.1f%%)\n", avg_rst, avg_rst / avg_total * 100);
    printf("  TOTAL:      %7.3f ms\n", avg_total);

    engine_destroy(engine);
}

/* ============================================================================
 * Scaling helper
 * ============================================================================ */

static BenchStats bench_navigation_scaling(uint32_t agent_count, uint32_t iterations,
                                            uint32_t warmup, uint64_t seed) {
    (void)seed;
    return run_profile(PROFILE_NAVIGATION, agent_count, warmup, iterations);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char** argv) {
    BenchCLI cli = bench_parse_cli(argc, argv);

    printf("=== Full Pipeline Engine Benchmarks ===\n");
    printf("SIMD width: %d floats\n\n", FOUNDATION_SIMD_WIDTH);

    uint32_t N = 1024;
    BenchStats results[16];
    uint32_t num_results = 0;

    /* --- Profile Benchmarks at 1024 drones --- */
    printf("--- Engine Profiles (%u drones) ---\n", N);
    bench_print_header();

    for (int p = 0; p < PROFILE_COUNT; p++) {
        BenchStats s = run_profile((EngineProfile)p, N, cli.warmup, cli.iterations);
        bench_print_row(&s);
        results[num_results++] = s;
    }

    /* --- Phase Breakdown --- */
    print_phase_breakdown(PROFILE_NAVIGATION, N, 200);

    /* --- Scaling Test --- */
    bench_scaling_test("NAVIGATION", bench_navigation_scaling, &cli);

    /* --- Memory Report --- */
    printf("\n--- Memory Report (all profiles, %u drones) ---\n", N);
    printf("  %-15s %12s %12s\n", "PROFILE", "PERSISTENT", "FRAME");
    printf("  -----------------------------------------\n");
    for (uint32_t i = 0; i < num_results && i < PROFILE_COUNT; i++) {
        printf("  %-15s %10zu KB %10zu KB\n",
               results[i].name,
               results[i].persistent_bytes / 1024,
               results[i].frame_bytes / 1024);
    }

    /* --- Sustained Performance (NAVIGATION, 10000 steps) --- */
    printf("\n--- Sustained Performance (NAVIGATION, 10000 steps) ---\n");
    {
        EngineConfig config = make_profile_config(PROFILE_NAVIGATION, N);
        char error_msg[ENGINE_ERROR_MSG_SIZE];
        BatchEngine* engine = engine_create(&config, error_msg);
        if (engine) {
            float* actions = engine_get_actions(engine);
            PCG32 rng;
            pcg32_seed(&rng, 42);
            for (uint32_t i = 0; i < config.total_agents * engine_get_action_dim(engine); i++) {
                actions[i] = pcg32_range(&rng, 0.2f, 0.8f);
            }
            engine_reset(engine);
            engine_add_box(engine, VEC3(-50, -50, -10), VEC3(50, 50, 0), 1);

            EngineBenchCtx ctx = { .engine = engine, .total_agents = config.total_agents };

            /* Check avg < target */
            BenchStats s = bench_measure("sustained_nav_10k", fn_engine_step, &ctx,
                                          50, 10000, profile_targets[PROFILE_NAVIGATION]);
            s.agent_count = config.total_agents;
            bench_print_header();
            bench_print_row(&s);

            printf("\n  Avg step: %.3f ms (target: %.1f ms) [%s]\n",
                   s.avg_ms, s.target_ms, s.passed ? "PASS" : "FAIL");

            /* Degradation check */
            bench_check_degradation("sustained_nav", fn_engine_step, &ctx,
                                     10000, 1000, 10.0);

            /* Frame arena stability check */
            size_t frame_before = engine->frame_arena->used;
            for (int i = 0; i < 100; i++) {
                engine_step(engine);
            }
            size_t frame_after = engine->frame_arena->used;
            printf("  Frame arena: before=%zu after=%zu [%s]\n",
                   frame_before, frame_after,
                   (frame_after <= frame_before * 2) ? "STABLE" : "GROWING");

            engine_destroy(engine);
        }
    }

    /* --- Throughput Summary --- */
    printf("\n--- Throughput Summary (drone-steps/second) ---\n");
    printf("  %-15s %15s %15s\n", "PROFILE", "AVG_MS", "DRONES/SEC");
    printf("  -----------------------------------------------\n");
    for (uint32_t i = 0; i < num_results && i < PROFILE_COUNT; i++) {
        double drones_per_sec = 0;
        if (results[i].avg_ms > 0.0001) {
            drones_per_sec = (double)results[i].agent_count / (results[i].avg_ms / 1000.0);
        }
        printf("  %-15s %12.3f ms %13.0f\n",
               results[i].name, results[i].avg_ms, drones_per_sec);
    }

    /* --- Summary --- */
    bench_print_separator();
    bench_print_summary(results, num_results);

    return 0;
}
