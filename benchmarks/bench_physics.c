/**
 * Physics Engine Benchmarks
 *
 * Comprehensive benchmarks for physics step, derivatives, force/torque
 * computation, RK4 combine, quaternion normalization, and ground effect.
 * Includes scaling tests and degradation checks.
 */

#include "bench_harness.h"
#include "physics.h"

/* ============================================================================
 * Fixture
 * ============================================================================ */

typedef struct PhysicsBenchCtx {
    Arena* persistent;
    Arena* scratch;
    PhysicsSystem* physics;
    DroneStateSOA* states;
    DroneParamsSOA* params;
    float* actions;
    float* sdf_distances;
    uint32_t num_drones;
} PhysicsBenchCtx;

static PhysicsBenchCtx* physics_ctx_create(Arena* persistent, Arena* scratch,
                                            uint32_t num_drones, bool with_sdf) {
    PhysicsBenchCtx* ctx = arena_alloc_type(persistent, PhysicsBenchCtx);
    memset(ctx, 0, sizeof(*ctx));
    ctx->persistent = persistent;
    ctx->scratch = scratch;
    ctx->num_drones = num_drones;

    PhysicsConfig config = physics_config_default();
    ctx->physics = physics_create(persistent, scratch, &config, num_drones);
    ctx->states = drone_state_create(persistent, num_drones);
    ctx->params = drone_params_create(persistent, num_drones);
    ctx->actions = arena_alloc_array(persistent, float, num_drones * 4);

    if (!ctx->physics || !ctx->states || !ctx->params || !ctx->actions) return NULL;

    /* Initialize */
    for (uint32_t i = 0; i < num_drones; i++) {
        drone_state_init(ctx->states, i);
        drone_params_init(ctx->params, i);
        ctx->states->pos_z[i] = 10.0f;
        ctx->states->quat_w[i] = 1.0f;
        for (uint32_t m = 0; m < 4; m++) {
            ctx->actions[i * 4 + m] = 0.4f;
        }
    }
    ctx->states->count = num_drones;

    if (with_sdf) {
        ctx->sdf_distances = arena_alloc_array(persistent, float, num_drones);
        for (uint32_t i = 0; i < num_drones; i++) {
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
    physics_step(ctx->physics, ctx->states, ctx->params, ctx->actions, ctx->num_drones);
}

static void fn_compute_derivatives(void* arg) {
    PhysicsBenchCtx* ctx = (PhysicsBenchCtx*)arg;
    physics_compute_derivatives(ctx->states, ctx->params, ctx->actions,
                                 ctx->physics->k1, ctx->num_drones,
                                 &ctx->physics->config, ctx->sdf_distances);
}

static void fn_compute_forces_torques(void* arg) {
    PhysicsBenchCtx* ctx = (PhysicsBenchCtx*)arg;
    physics_compute_forces_torques(ctx->states, ctx->params,
                                    ctx->physics->forces_x, ctx->physics->forces_y,
                                    ctx->physics->forces_z, ctx->physics->torques_x,
                                    ctx->physics->torques_y, ctx->physics->torques_z,
                                    ctx->num_drones);
}

static void fn_rk4_combine(void* arg) {
    PhysicsBenchCtx* ctx = (PhysicsBenchCtx*)arg;
    physics_rk4_combine(ctx->states, ctx->physics->k1, ctx->physics->k2,
                         ctx->physics->k3, ctx->physics->k4, 0.02f, ctx->num_drones);
}

static void fn_quaternion_normalize(void* arg) {
    PhysicsBenchCtx* ctx = (PhysicsBenchCtx*)arg;
    physics_normalize_quaternions(ctx->states, ctx->num_drones);
}

/* ============================================================================
 * Scaling helper
 * ============================================================================ */

static BenchStats bench_physics_step_scaling(uint32_t drone_count, uint32_t iterations,
                                              uint32_t warmup, uint64_t seed) {
    (void)seed;
    size_t arena_size = physics_memory_size(drone_count) +
                        drone_state_memory_size(drone_count) +
                        drone_params_memory_size(drone_count) +
                        drone_count * 4 * sizeof(float) + 16 * 1024 * 1024;
    Arena* pa = arena_create(arena_size);
    Arena* sa = arena_create(8 * 1024 * 1024);

    PhysicsBenchCtx* ctx = physics_ctx_create(pa, sa, drone_count, false);
    BenchStats s = {0};
    s.name = "physics_step";
    if (!ctx) {
        arena_destroy(sa);
        arena_destroy(pa);
        return s;
    }

    s = bench_measure("physics_step", fn_physics_step, ctx, warmup, iterations, 5.0);
    s.drone_count = drone_count;

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
        size_t arena_size = physics_memory_size(N) + drone_state_memory_size(N) +
                            drone_params_memory_size(N) + N * 4 * sizeof(float) +
                            16 * 1024 * 1024;
        Arena* pa = arena_create(arena_size);
        Arena* sa = arena_create(8 * 1024 * 1024);
        PhysicsBenchCtx* ctx = physics_ctx_create(pa, sa, N, false);
        if (ctx) {
            BenchStats s = bench_measure("physics_step", fn_physics_step, ctx,
                                          cli.warmup, cli.iterations, 5.0);
            s.drone_count = N;
            bench_print_row(&s);
            results[num_results++] = s;
        }
        arena_destroy(sa);
        arena_destroy(pa);
    }

    /* Physics step with ground effect */
    {
        size_t arena_size = physics_memory_size(N) + drone_state_memory_size(N) +
                            drone_params_memory_size(N) + N * 5 * sizeof(float) +
                            16 * 1024 * 1024;
        Arena* pa = arena_create(arena_size);
        Arena* sa = arena_create(8 * 1024 * 1024);
        PhysicsBenchCtx* ctx = physics_ctx_create(pa, sa, N, true);
        if (ctx) {
            BenchStats s = bench_measure("physics_step_ground_effect", fn_physics_step, ctx,
                                          cli.warmup, cli.iterations, 6.0);
            s.drone_count = N;
            bench_print_row(&s);
            results[num_results++] = s;
        }
        arena_destroy(sa);
        arena_destroy(pa);
    }

    /* Individual components */
    {
        size_t arena_size = physics_memory_size(N) + drone_state_memory_size(N) * 2 +
                            drone_params_memory_size(N) + N * 4 * sizeof(float) +
                            16 * 1024 * 1024;
        Arena* pa = arena_create(arena_size);
        Arena* sa = arena_create(8 * 1024 * 1024);
        PhysicsBenchCtx* ctx = physics_ctx_create(pa, sa, N, false);
        if (ctx) {
            BenchStats s;

            s = bench_measure("compute_derivatives", fn_compute_derivatives, ctx,
                               cli.warmup, cli.iterations, 1.0);
            s.drone_count = N;
            bench_print_row(&s);
            results[num_results++] = s;

            s = bench_measure("compute_forces_torques", fn_compute_forces_torques, ctx,
                               cli.warmup, cli.iterations, 0.5);
            s.drone_count = N;
            bench_print_row(&s);
            results[num_results++] = s;

            s = bench_measure("rk4_combine", fn_rk4_combine, ctx,
                               cli.warmup, cli.iterations, 0.3);
            s.drone_count = N;
            bench_print_row(&s);
            results[num_results++] = s;

            s = bench_measure("quaternion_normalize", fn_quaternion_normalize, ctx,
                               cli.warmup, cli.iterations, 0.2);
            s.drone_count = N;
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
    printf("  physics_memory_size:      %zu bytes (%.1f KB)\n",
           physics_memory_size(N), physics_memory_size(N) / 1024.0);
    printf("  drone_state_memory_size:  %zu bytes (%.1f KB)\n",
           drone_state_memory_size(N), drone_state_memory_size(N) / 1024.0);
    printf("  drone_params_memory_size: %zu bytes (%.1f KB)\n",
           drone_params_memory_size(N), drone_params_memory_size(N) / 1024.0);

    /* --- Degradation Test --- */
    printf("\n--- Degradation Test (5000 iters, 500-step blocks, 5%% max drift) ---\n");
    {
        size_t arena_size = physics_memory_size(N) + drone_state_memory_size(N) +
                            drone_params_memory_size(N) + N * 4 * sizeof(float) +
                            16 * 1024 * 1024;
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
