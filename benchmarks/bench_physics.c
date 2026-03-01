/**
 * Physics Engine Benchmarks
 *
 * Comprehensive benchmarks for physics step, derivatives, force/torque
 * computation, RK4 combine, quaternion normalization, and ground effect.
 * Includes scaling tests and degradation checks.
 */

#include "bench_harness.h"
#include "physics.h"
#include "platform_quadcopter.h"

/* Helper: call compute_forces_torques through the quadcopter vtable */
static void physics_compute_forces_torques(PlatformStateSOA* states, PlatformParamsSOA* params,
                                            float* fx, float* fy, float* fz,
                                            float* tx, float* ty, float* tz,
                                            uint32_t count) {
    PLATFORM_QUADCOPTER.compute_forces_torques(
        &states->rigid_body,
        (float* const*)states->extension, states->extension_count,
        (float* const*)params->extension, params->extension_count,
        &params->rigid_body,
        fx, fy, fz, tx, ty, tz, count);
}

/* ============================================================================
 * Fixture
 * ============================================================================ */

typedef struct PhysicsBenchCtx {
    Arena* persistent;
    Arena* scratch;
    PhysicsSystem* physics;
    PlatformStateSOA* states;
    PlatformParamsSOA* params;
    float* actions;
    float* sdf_distances;
    uint32_t num_agents;
} PhysicsBenchCtx;

static PhysicsBenchCtx* physics_ctx_create(Arena* persistent, Arena* scratch,
                                            uint32_t num_agents, bool with_sdf) {
    PhysicsBenchCtx* ctx = arena_alloc_type(persistent, PhysicsBenchCtx);
    memset(ctx, 0, sizeof(*ctx));
    ctx->persistent = persistent;
    ctx->scratch = scratch;
    ctx->num_agents = num_agents;

    PhysicsConfig config = physics_config_default();
    ctx->physics = physics_create(persistent, scratch, &config, num_agents, &PLATFORM_QUADCOPTER);
    ctx->states = platform_state_create(persistent, num_agents, QUAD_STATE_EXT_COUNT);
    ctx->params = platform_params_create(persistent, num_agents, QUAD_PARAMS_EXT_COUNT);
    ctx->actions = arena_alloc_array(persistent, float, num_agents * 4);

    if (!ctx->physics || !ctx->states || !ctx->params || !ctx->actions) return NULL;

    /* Initialize */
    for (uint32_t i = 0; i < num_agents; i++) {
        platform_state_init(ctx->states, i);
        platform_params_init(ctx->params, i);
        PLATFORM_QUADCOPTER.init_params(ctx->params->extension, ctx->params->extension_count, i);
        ctx->states->rigid_body.pos_z[i] = 10.0f;
        ctx->states->rigid_body.quat_w[i] = 1.0f;
        for (uint32_t m = 0; m < 4; m++) {
            ctx->actions[i * 4 + m] = 0.4f;
        }
    }
    ctx->states->rigid_body.count = num_agents;

    if (with_sdf) {
        ctx->sdf_distances = arena_alloc_array(persistent, float, num_agents);
        for (uint32_t i = 0; i < num_agents; i++) {
            ctx->sdf_distances[i] = 0.3f; /* Close to ground */
        }
        ctx->physics->sdf_distances = ctx->sdf_distances;
    }

    return ctx;
}

/* ============================================================================
 * Benchmark Functions
 * ============================================================================ */

static void fn_physics_step(void* arg) {
    PhysicsBenchCtx* ctx = (PhysicsBenchCtx*)arg;
    physics_step(ctx->physics, ctx->states, ctx->params, ctx->actions, ctx->num_agents);
}

static void fn_compute_derivatives(void* arg) {
    PhysicsBenchCtx* ctx = (PhysicsBenchCtx*)arg;
    physics_compute_derivatives(ctx->physics, ctx->states, ctx->params,
                                 ctx->physics->k1, ctx->num_agents);
}

static void fn_compute_forces_torques(void* arg) {
    PhysicsBenchCtx* ctx = (PhysicsBenchCtx*)arg;
    physics_compute_forces_torques(ctx->states, ctx->params,
                                    ctx->physics->forces_x, ctx->physics->forces_y,
                                    ctx->physics->forces_z, ctx->physics->torques_x,
                                    ctx->physics->torques_y, ctx->physics->torques_z,
                                    ctx->num_agents);
}

static void fn_rk4_combine(void* arg) {
    PhysicsBenchCtx* ctx = (PhysicsBenchCtx*)arg;
    physics_rk4_combine(&ctx->states->rigid_body, ctx->physics->k1, ctx->physics->k2,
                         ctx->physics->k3, ctx->physics->k4, 0.02f, ctx->num_agents);
}

static void fn_quaternion_normalize(void* arg) {
    PhysicsBenchCtx* ctx = (PhysicsBenchCtx*)arg;
    physics_normalize_quaternions(ctx->states, ctx->num_agents);
}

/* ============================================================================
 * Scaling helper
 * ============================================================================ */

static BenchStats bench_physics_step_scaling(uint32_t agent_count, uint32_t iterations,
                                              uint32_t warmup, uint64_t seed) {
    (void)seed;
    size_t arena_size = physics_memory_size(agent_count) +
                        platform_state_memory_size(agent_count, QUAD_STATE_EXT_COUNT) +
                        platform_params_memory_size(agent_count, QUAD_PARAMS_EXT_COUNT) +
                        agent_count * 4 * sizeof(float) + 16 * 1024 * 1024;
    Arena* pa = arena_create(arena_size);
    Arena* sa = arena_create(8 * 1024 * 1024);

    PhysicsBenchCtx* ctx = physics_ctx_create(pa, sa, agent_count, false);
    BenchStats s = {0};
    s.name = "physics_step";
    if (!ctx) {
        arena_destroy(sa);
        arena_destroy(pa);
        return s;
    }

    s = bench_measure("physics_step", fn_physics_step, ctx, warmup, iterations, 5.0);
    s.agent_count = agent_count;

    arena_destroy(sa);
    arena_destroy(pa);
    return s;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char** argv) {
    BenchCLI cli = bench_parse_cli(argc, argv);

    printf("=== Physics Engine Benchmarks ===\n");
    printf("SIMD width: %d floats\n\n", FOUNDATION_SIMD_WIDTH);

    uint32_t N = 1024;
    BenchStats results[16];
    uint32_t num_results = 0;

    /* --- Core Benchmarks at 1024 drones --- */
    printf("--- Core Benchmarks (%u drones) ---\n", N);
    bench_print_header();

    /* Physics step (no ground effect) */
    {
        size_t arena_size = physics_memory_size(N) +
                            platform_state_memory_size(N, QUAD_STATE_EXT_COUNT) +
                            platform_params_memory_size(N, QUAD_PARAMS_EXT_COUNT) +
                            N * 4 * sizeof(float) + 16 * 1024 * 1024;
        Arena* pa = arena_create(arena_size);
        Arena* sa = arena_create(8 * 1024 * 1024);
        PhysicsBenchCtx* ctx = physics_ctx_create(pa, sa, N, false);
        if (ctx) {
            BenchStats s = bench_measure("physics_step", fn_physics_step, ctx,
                                          cli.warmup, cli.iterations, 5.0);
            s.agent_count = N;
            bench_print_row(&s);
            results[num_results++] = s;
        }
        arena_destroy(sa);
        arena_destroy(pa);
    }

    /* Physics step with ground effect */
    {
        size_t arena_size = physics_memory_size(N) +
                            platform_state_memory_size(N, QUAD_STATE_EXT_COUNT) +
                            platform_params_memory_size(N, QUAD_PARAMS_EXT_COUNT) +
                            N * 5 * sizeof(float) + 16 * 1024 * 1024;
        Arena* pa = arena_create(arena_size);
        Arena* sa = arena_create(8 * 1024 * 1024);
        PhysicsBenchCtx* ctx = physics_ctx_create(pa, sa, N, true);
        if (ctx) {
            BenchStats s = bench_measure("physics_step_ground_effect", fn_physics_step, ctx,
                                          cli.warmup, cli.iterations, 6.0);
            s.agent_count = N;
            bench_print_row(&s);
            results[num_results++] = s;
        }
        arena_destroy(sa);
        arena_destroy(pa);
    }

    /* Individual components */
    {
        size_t arena_size = physics_memory_size(N) +
                            platform_state_memory_size(N, QUAD_STATE_EXT_COUNT) * 2 +
                            platform_params_memory_size(N, QUAD_PARAMS_EXT_COUNT) +
                            N * 4 * sizeof(float) + 16 * 1024 * 1024;
        Arena* pa = arena_create(arena_size);
        Arena* sa = arena_create(8 * 1024 * 1024);
        PhysicsBenchCtx* ctx = physics_ctx_create(pa, sa, N, false);
        if (ctx) {
            BenchStats s;

            s = bench_measure("compute_derivatives", fn_compute_derivatives, ctx,
                               cli.warmup, cli.iterations, 1.0);
            s.agent_count = N;
            bench_print_row(&s);
            results[num_results++] = s;

            s = bench_measure("compute_forces_torques", fn_compute_forces_torques, ctx,
                               cli.warmup, cli.iterations, 0.5);
            s.agent_count = N;
            bench_print_row(&s);
            results[num_results++] = s;

            s = bench_measure("rk4_combine", fn_rk4_combine, ctx,
                               cli.warmup, cli.iterations, 0.3);
            s.agent_count = N;
            bench_print_row(&s);
            results[num_results++] = s;

            s = bench_measure("quaternion_normalize", fn_quaternion_normalize, ctx,
                               cli.warmup, cli.iterations, 0.2);
            s.agent_count = N;
            bench_print_row(&s);
            results[num_results++] = s;
        }
        arena_destroy(sa);
        arena_destroy(pa);
    }

    /* --- Ground Effect Overhead Comparison --- */
    printf("\n--- Ground Effect Overhead ---\n");
    if (num_results >= 2) {
        double without = results[0].avg_ms;
        double with_ge = results[1].avg_ms;
        double overhead_pct = 0.0;
        if (without > 0.0001) {
            overhead_pct = ((with_ge - without) / without) * 100.0;
        }
        printf("  Without ground effect: %.3f ms\n", without);
        printf("  With ground effect:    %.3f ms\n", with_ge);
        printf("  Overhead:              %.1f%%\n", overhead_pct);
    }

    /* --- Scaling Test --- */
    bench_scaling_test("physics_step", bench_physics_step_scaling, &cli);

    /* --- Memory Report --- */
    printf("\n--- Memory Report (%u drones) ---\n", N);
    printf("  physics_memory_size:           %zu bytes (%.1f KB)\n",
           physics_memory_size(N), physics_memory_size(N) / 1024.0);
    printf("  platform_state_memory_size:    %zu bytes (%.1f KB)\n",
           platform_state_memory_size(N, QUAD_STATE_EXT_COUNT),
           platform_state_memory_size(N, QUAD_STATE_EXT_COUNT) / 1024.0);
    printf("  platform_params_memory_size:   %zu bytes (%.1f KB)\n",
           platform_params_memory_size(N, QUAD_PARAMS_EXT_COUNT),
           platform_params_memory_size(N, QUAD_PARAMS_EXT_COUNT) / 1024.0);

    /* --- Degradation Test --- */
    printf("\n--- Degradation Test (5000 iters, 500-step blocks, 5%% max drift) ---\n");
    {
        size_t arena_size = physics_memory_size(N) +
                            platform_state_memory_size(N, QUAD_STATE_EXT_COUNT) +
                            platform_params_memory_size(N, QUAD_PARAMS_EXT_COUNT) +
                            N * 4 * sizeof(float) + 16 * 1024 * 1024;
        Arena* pa = arena_create(arena_size);
        Arena* sa = arena_create(8 * 1024 * 1024);
        PhysicsBenchCtx* ctx = physics_ctx_create(pa, sa, N, false);
        if (ctx) {
            bench_check_degradation("physics_step", fn_physics_step, ctx,
                                     5000, 500, 5.0);
        }
        arena_destroy(sa);
        arena_destroy(pa);
    }

    /* --- Summary --- */
    bench_print_separator();
    bench_print_summary(results, num_results);

    return 0;
}
