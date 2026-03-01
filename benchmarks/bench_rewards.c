/**
 * Reward System Benchmarks
 *
 * Benchmarks for hover, race, track, land rewards, termination checking,
 * and full reward frames. Tests with various task types and gate counts.
 */

#include "bench_harness.h"
#include "reward_system.h"
#include "platform_quadcopter.h"

/* ============================================================================
 * Fixture
 * ============================================================================ */

typedef struct RewardBenchCtx {
    Arena* arena;
    RewardSystem* rewards;
    PlatformStateSOA* drones;
    PlatformParamsSOA* params;
    float* actions;
    float* reward_buf;
    CollisionResults collision_results;
    uint8_t* world_flags;
    float* penetration;
    uint32_t num_agents;
} RewardBenchCtx;

static RewardBenchCtx* reward_ctx_create(Arena* arena, uint32_t num_agents,
                                          TaskType task, uint32_t num_gates,
                                          uint64_t seed) {
    RewardBenchCtx* ctx = arena_alloc_type(arena, RewardBenchCtx);
    memset(ctx, 0, sizeof(*ctx));
    ctx->arena = arena;
    ctx->num_agents = num_agents;

    RewardConfig config = reward_config_default(task);
    ctx->rewards = reward_create(arena, &config, num_agents, num_gates);
    ctx->drones = platform_state_create(arena, num_agents, QUAD_STATE_EXT_COUNT);
    ctx->params = platform_params_create(arena, num_agents, QUAD_PARAMS_EXT_COUNT);
    ctx->actions = arena_alloc_array(arena, float, num_agents * 4);
    ctx->reward_buf = arena_alloc_array(arena, float, num_agents);

    if (!ctx->rewards || !ctx->drones || !ctx->params || !ctx->actions || !ctx->reward_buf)
        return NULL;

    /* Initialize drones with random positions/velocities */
    PCG32 rng;
    pcg32_seed(&rng, seed);
    for (uint32_t i = 0; i < num_agents; i++) {
        platform_state_init(ctx->drones, i);
        platform_params_init(ctx->params, i);
        PLATFORM_QUADCOPTER.init_params(ctx->params->extension, ctx->params->extension_count, i);
        ctx->drones->rigid_body.pos_x[i] = pcg32_range(&rng, -30, 30);
        ctx->drones->rigid_body.pos_y[i] = pcg32_range(&rng, -30, 30);
        ctx->drones->rigid_body.pos_z[i] = pcg32_range(&rng, 2, 20);
        ctx->drones->rigid_body.vel_x[i] = pcg32_range(&rng, -2, 2);
        ctx->drones->rigid_body.vel_y[i] = pcg32_range(&rng, -2, 2);
        ctx->drones->rigid_body.vel_z[i] = pcg32_range(&rng, -1, 1);
        for (uint32_t m = 0; m < 4; m++) {
            ctx->actions[i * 4 + m] = pcg32_range(&rng, 0.2f, 0.8f);
        }
    }
    ctx->drones->rigid_body.count = num_agents;

    /* Set random targets */
    reward_set_targets_random(ctx->rewards, num_agents,
                               VEC3(-30, -30, 2), VEC3(30, 30, 20), &rng);

    /* Set gates if racing */
    if (task == TASK_RACE && num_gates > 0) {
        Vec3* centers = arena_alloc_array(arena, Vec3, num_gates);
        Vec3* normals = arena_alloc_array(arena, Vec3, num_gates);
        float* radii = arena_alloc_array(arena, float, num_gates);
        for (uint32_t g = 0; g < num_gates; g++) {
            centers[g] = VEC3(pcg32_range(&rng, -20, 20),
                              pcg32_range(&rng, -20, 20),
                              pcg32_range(&rng, 5, 15));
            normals[g] = VEC3(1, 0, 0);
            radii[g] = 2.0f;
        }
        reward_set_gates(ctx->rewards, centers, normals, radii, num_gates);
    }

    /* Pre-populated collision results (~5% collision flags) */
    ctx->world_flags = arena_alloc_zero(arena, num_agents);
    ctx->penetration = arena_alloc_array(arena, float, num_agents);
    memset(ctx->penetration, 0, sizeof(float) * num_agents);
    for (uint32_t i = 0; i < num_agents; i++) {
        if (pcg32_bounded(&rng, 20) == 0) { /* ~5% */
            ctx->world_flags[i] = 1;
            ctx->penetration[i] = -0.05f;
        }
    }
    ctx->collision_results.pairs = NULL;
    ctx->collision_results.pair_count = 0;
    ctx->collision_results.world_flags = ctx->world_flags;
    ctx->collision_results.penetration = ctx->penetration;
    ctx->collision_results.normals = NULL;

    return ctx;
}

/* ============================================================================
 * Benchmark Functions
 * ============================================================================ */

static void fn_reward_compute(void* arg) {
    RewardBenchCtx* ctx = (RewardBenchCtx*)arg;
    reward_compute(ctx->rewards, ctx->drones, ctx->params, ctx->actions,
                    &ctx->collision_results, ctx->reward_buf, ctx->num_agents);
}

typedef struct TermBenchCtx {
    RewardBenchCtx* rctx;
    TerminationFlags* flags;
} TermBenchCtx;

static void fn_termination(void* arg) {
    TermBenchCtx* ctx = (TermBenchCtx*)arg;
    reward_compute_terminations(ctx->rctx->rewards, ctx->rctx->drones,
                                 &ctx->rctx->collision_results,
                                 VEC3(-50, -50, -10), VEC3(50, 50, 50),
                                 1000, ctx->flags, ctx->rctx->num_agents);
}

typedef struct FullFrameCtx {
    RewardBenchCtx* rctx;
    TerminationFlags* flags;
} FullFrameCtx;

static void fn_full_frame(void* arg) {
    FullFrameCtx* ctx = (FullFrameCtx*)arg;
    reward_compute(ctx->rctx->rewards, ctx->rctx->drones, ctx->rctx->params,
                    ctx->rctx->actions, &ctx->rctx->collision_results,
                    ctx->rctx->reward_buf, ctx->rctx->num_agents);
    reward_compute_terminations(ctx->rctx->rewards, ctx->rctx->drones,
                                 &ctx->rctx->collision_results,
                                 VEC3(-50, -50, -10), VEC3(50, 50, 50),
                                 1000, ctx->flags, ctx->rctx->num_agents);
}

/* ============================================================================
 * Scaling helper
 * ============================================================================ */

static BenchStats bench_hover_frame_scaling(uint32_t agent_count, uint32_t iterations,
                                             uint32_t warmup, uint64_t seed) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    BenchStats s = {0};
    s.name = "full_hover_frame";

    RewardBenchCtx* rctx = reward_ctx_create(arena, agent_count, TASK_HOVER, 0, seed);
    if (!rctx) {
        arena_destroy(arena);
        return s;
    }

    /* Create termination flags */
    TerminationFlags flags = {0};
    flags.done = arena_alloc_zero(arena, agent_count);
    flags.truncated = arena_alloc_zero(arena, agent_count);
    flags.success = arena_alloc_zero(arena, agent_count);
    flags.collision = arena_alloc_zero(arena, agent_count);
    flags.out_of_bounds = arena_alloc_zero(arena, agent_count);
    flags.timeout = arena_alloc_zero(arena, agent_count);
    flags.capacity = agent_count;

    FullFrameCtx fctx = { .rctx = rctx, .flags = &flags };
    s = bench_measure("full_hover_frame", fn_full_frame, &fctx, warmup, iterations, 1.0);
    s.agent_count = agent_count;

    arena_destroy(arena);
    return s;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char** argv) {
    BenchCLI cli = bench_parse_cli(argc, argv);

    printf("=== Reward System Benchmarks ===\n");
    printf("SIMD width: %d floats\n\n", FOUNDATION_SIMD_WIDTH);

    uint32_t N = 1024;
    BenchStats results[16];
    uint32_t num_results = 0;

    /* --- Task-Specific Reward Benchmarks --- */
    printf("--- Task-Specific Rewards (%u drones) ---\n", N);
    bench_print_header();

    /* Hover */
    {
        Arena* arena = arena_create(64 * 1024 * 1024);
        RewardBenchCtx* ctx = reward_ctx_create(arena, N, TASK_HOVER, 0, cli.seed);
        if (ctx) {
            BenchStats s = bench_measure("hover_reward", fn_reward_compute, ctx,
                                          cli.warmup, cli.iterations, 0.3);
            s.agent_count = N;
            bench_print_row(&s);
            results[num_results++] = s;
        }
        arena_destroy(arena);
    }

    /* Race with 10 gates */
    {
        Arena* arena = arena_create(64 * 1024 * 1024);
        RewardBenchCtx* ctx = reward_ctx_create(arena, N, TASK_RACE, 10, cli.seed);
        if (ctx) {
            BenchStats s = bench_measure("race_reward_10", fn_reward_compute, ctx,
                                          cli.warmup, cli.iterations, 0.5);
            s.agent_count = N;
            bench_print_row(&s);
            results[num_results++] = s;
        }
        arena_destroy(arena);
    }

    /* Race with 50 gates */
    {
        Arena* arena = arena_create(64 * 1024 * 1024);
        RewardBenchCtx* ctx = reward_ctx_create(arena, N, TASK_RACE, 50, cli.seed);
        if (ctx) {
            BenchStats s = bench_measure("race_reward_50", fn_reward_compute, ctx,
                                          cli.warmup, cli.iterations, 0.8);
            s.agent_count = N;
            bench_print_row(&s);
            results[num_results++] = s;
        }
        arena_destroy(arena);
    }

    /* Track */
    {
        Arena* arena = arena_create(64 * 1024 * 1024);
        RewardBenchCtx* ctx = reward_ctx_create(arena, N, TASK_TRACK, 0, cli.seed);
        if (ctx) {
            BenchStats s = bench_measure("track_reward", fn_reward_compute, ctx,
                                          cli.warmup, cli.iterations, 0.4);
            s.agent_count = N;
            bench_print_row(&s);
            results[num_results++] = s;
        }
        arena_destroy(arena);
    }

    /* Land */
    {
        Arena* arena = arena_create(64 * 1024 * 1024);
        RewardBenchCtx* ctx = reward_ctx_create(arena, N, TASK_LAND, 0, cli.seed);
        if (ctx) {
            BenchStats s = bench_measure("land_reward", fn_reward_compute, ctx,
                                          cli.warmup, cli.iterations, 0.3);
            s.agent_count = N;
            bench_print_row(&s);
            results[num_results++] = s;
        }
        arena_destroy(arena);
    }

    /* --- Termination Benchmark --- */
    printf("\n--- Termination & Full Frame (%u drones) ---\n", N);
    bench_print_header();

    {
        Arena* arena = arena_create(64 * 1024 * 1024);
        RewardBenchCtx* rctx = reward_ctx_create(arena, N, TASK_HOVER, 0, cli.seed);
        if (rctx) {
            TerminationFlags flags = {0};
            flags.done = arena_alloc_zero(arena, N);
            flags.truncated = arena_alloc_zero(arena, N);
            flags.success = arena_alloc_zero(arena, N);
            flags.collision = arena_alloc_zero(arena, N);
            flags.out_of_bounds = arena_alloc_zero(arena, N);
            flags.timeout = arena_alloc_zero(arena, N);
            flags.capacity = N;

            TermBenchCtx tctx = { .rctx = rctx, .flags = &flags };
            BenchStats s = bench_measure("termination", fn_termination, &tctx,
                                          cli.warmup, cli.iterations, 0.1);
            s.agent_count = N;
            bench_print_row(&s);
            results[num_results++] = s;

            FullFrameCtx fctx = { .rctx = rctx, .flags = &flags };
            s = bench_measure("full_hover_frame", fn_full_frame, &fctx,
                               cli.warmup, cli.iterations, 1.0);
            s.agent_count = N;
            bench_print_row(&s);
            results[num_results++] = s;
        }
        arena_destroy(arena);
    }

    {
        Arena* arena = arena_create(64 * 1024 * 1024);
        RewardBenchCtx* rctx = reward_ctx_create(arena, N, TASK_RACE, 10, cli.seed);
        if (rctx) {
            TerminationFlags flags = {0};
            flags.done = arena_alloc_zero(arena, N);
            flags.truncated = arena_alloc_zero(arena, N);
            flags.success = arena_alloc_zero(arena, N);
            flags.collision = arena_alloc_zero(arena, N);
            flags.out_of_bounds = arena_alloc_zero(arena, N);
            flags.timeout = arena_alloc_zero(arena, N);
            flags.capacity = N;

            FullFrameCtx fctx = { .rctx = rctx, .flags = &flags };
            BenchStats s = bench_measure("full_race_frame", fn_full_frame, &fctx,
                                          cli.warmup, cli.iterations, 1.2);
            s.agent_count = N;
            bench_print_row(&s);
            results[num_results++] = s;
        }
        arena_destroy(arena);
    }

    /* --- Scaling Test --- */
    bench_scaling_test("full_hover_frame", bench_hover_frame_scaling, &cli);

    /* --- Summary --- */
    bench_print_separator();
    bench_print_summary(results, num_results);

    return 0;
}
