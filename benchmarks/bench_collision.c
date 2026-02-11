/**
 * Collision System Benchmarks
 *
 * Benchmarks for spatial hash, drone-drone/world detection, collision response,
 * and K-nearest neighbor queries. Tests with scattered, clustered, and
 * world-populated distributions.
 */

#include "bench_harness.h"
#include "collision_system.h"
#include "world_brick_map.h"

/* ============================================================================
 * Fixture
 * ============================================================================ */

typedef struct CollisionBenchCtx {
    Arena* persistent;
    Arena* scratch;
    CollisionSystem* collision;
    DroneStateSOA* drones;
    DroneParamsSOA* params;
    WorldBrickMap* world;
    uint32_t num_drones;
} CollisionBenchCtx;

static CollisionBenchCtx* collision_ctx_create(Arena* persistent, Arena* scratch,
                                                uint32_t num_drones, bool with_world) {
    CollisionBenchCtx* ctx = arena_alloc_type(persistent, CollisionBenchCtx);
    memset(ctx, 0, sizeof(*ctx));
    ctx->persistent = persistent;
    ctx->scratch = scratch;
    ctx->num_drones = num_drones;

    ctx->collision = collision_create(persistent, num_drones, 0.1f, 1.0f);
    ctx->drones = drone_state_create(persistent, num_drones);
    ctx->params = drone_params_create(persistent, num_drones);

    if (!ctx->collision || !ctx->drones || !ctx->params) return NULL;

    for (uint32_t i = 0; i < num_drones; i++) {
        drone_params_init(ctx->params, i);
    }

    if (with_world) {
        ctx->world = world_create(persistent, VEC3(-50, -50, -10),
                                   VEC3(50, 50, 50), 0.1f, 10000, 0);
        if (ctx->world) {
            world_set_box(ctx->world, VEC3(0, 0, -5), VEC3(50, 50, 5), 1);
            world_set_sphere(ctx->world, VEC3(10, 0, 10), 5.0f, 1);
            world_set_sphere(ctx->world, VEC3(-15, 10, 8), 3.0f, 1);
        }
    }

    return ctx;
}

/* ============================================================================
 * Benchmark Functions
 * ============================================================================ */

/* Hash build */
static void fn_hash_build(void* arg) {
    CollisionBenchCtx* ctx = (CollisionBenchCtx*)arg;
    collision_build_spatial_hash(ctx->collision, ctx->drones, ctx->num_drones);
}

/* Drone-drone detection */
static void fn_detect_dd(void* arg) {
    CollisionBenchCtx* ctx = (CollisionBenchCtx*)arg;
    collision_reset(ctx->collision);
    collision_build_spatial_hash(ctx->collision, ctx->drones, ctx->num_drones);
    collision_detect_drone_drone(ctx->collision, ctx->drones, ctx->num_drones);
}

/* World collision detection */
static void fn_detect_world(void* arg) {
    CollisionBenchCtx* ctx = (CollisionBenchCtx*)arg;
    collision_detect_drone_world(ctx->collision, ctx->drones, ctx->world, ctx->num_drones);
}

/* Full detection (drone-drone + world) */
static void fn_detect_all(void* arg) {
    CollisionBenchCtx* ctx = (CollisionBenchCtx*)arg;
    collision_detect_all(ctx->collision, ctx->drones, ctx->world, ctx->num_drones);
}

/* Collision response */
static void fn_response(void* arg) {
    CollisionBenchCtx* ctx = (CollisionBenchCtx*)arg;
    collision_apply_response(ctx->collision, ctx->drones, ctx->params,
                              0.5f, 1.0f, ctx->num_drones);
}

/* Full frame: detect + response */
typedef struct TotalFrameCtx {
    CollisionBenchCtx* cctx;
} TotalFrameCtx;

static void fn_total_frame(void* arg) {
    TotalFrameCtx* tctx = (TotalFrameCtx*)arg;
    CollisionBenchCtx* ctx = tctx->cctx;
    collision_detect_all(ctx->collision, ctx->drones, ctx->world, ctx->num_drones);
    collision_apply_response(ctx->collision, ctx->drones, ctx->params,
                              0.5f, 1.0f, ctx->num_drones);
}

/* KNN batch */
typedef struct KNNBenchCtx {
    CollisionSystem* collision;
    DroneStateSOA* drones;
    uint32_t num_drones;
    uint32_t k;
    uint32_t* out_indices;
    float* out_distances;
} KNNBenchCtx;

static void fn_knn_batch(void* arg) {
    KNNBenchCtx* ctx = (KNNBenchCtx*)arg;
    collision_find_k_nearest_batch(ctx->collision, ctx->drones,
                                    ctx->num_drones, ctx->k,
                                    ctx->out_indices, ctx->out_distances);
}

/* ============================================================================
 * Scaling helper
 * ============================================================================ */

static BenchStats bench_total_frame_scaling(uint32_t drone_count, uint32_t iterations,
                                             uint32_t warmup, uint64_t seed) {
    Arena* pa = arena_create(128 * 1024 * 1024);
    Arena* sa = arena_create(32 * 1024 * 1024);
    BenchStats s = {0};
    s.name = "total_frame";
    s.drone_count = drone_count;

    CollisionBenchCtx* ctx = collision_ctx_create(pa, sa, drone_count, true);
    if (!ctx) {
        arena_destroy(sa);
        arena_destroy(pa);
        return s;
    }

    bench_init_drones_scattered(ctx->drones, drone_count, 50.0f, seed);

    TotalFrameCtx tctx = { .cctx = ctx };
    s = bench_measure("total_frame", fn_total_frame, &tctx, warmup, iterations, 1.5);
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

    printf("=== Collision System Benchmarks ===\n");
    printf("SIMD width: %d floats\n\n", FOUNDATION_SIMD_WIDTH);

    uint32_t N = 1024;
    Arena* persistent = arena_create(256 * 1024 * 1024);
    Arena* scratch = arena_create(64 * 1024 * 1024);
    if (!persistent || !scratch) {
        fprintf(stderr, "Failed to create arenas\n");
        return 1;
    }

    BenchStats results[16];
    uint32_t num_results = 0;

    /* --- Scattered Distribution --- */
    printf("--- Scattered Distribution (%u drones) ---\n", N);
    bench_print_header();
    {
        CollisionBenchCtx* ctx = collision_ctx_create(persistent, scratch, N, true);
        if (ctx) {
            bench_init_drones_scattered(ctx->drones, N, 50.0f, cli.seed);

            BenchStats s;

            s = bench_measure("hash_build_scattered", fn_hash_build, ctx,
                               cli.warmup, cli.iterations, 0.1);
            s.drone_count = N;
            bench_print_row(&s);
            results[num_results++] = s;

            s = bench_measure("detect_dd_scattered", fn_detect_dd, ctx,
                               cli.warmup, cli.iterations, 0.3);
            s.drone_count = N;
            bench_print_row(&s);
            results[num_results++] = s;

            s = bench_measure("detect_world", fn_detect_world, ctx,
                               cli.warmup, cli.iterations, 0.2);
            s.drone_count = N;
            bench_print_row(&s);
            results[num_results++] = s;

            s = bench_measure("detect_all_world", fn_detect_all, ctx,
                               cli.warmup, cli.iterations, 1.0);
            s.drone_count = N;
            bench_print_row(&s);
            results[num_results++] = s;

            /* Run detection first so response has collision data */
            collision_detect_all(ctx->collision, ctx->drones, ctx->world, N);
            s = bench_measure("response_all", fn_response, ctx,
                               cli.warmup, cli.iterations, 0.2);
            s.drone_count = N;
            bench_print_row(&s);
            results[num_results++] = s;

            TotalFrameCtx tctx = { .cctx = ctx };
            s = bench_measure("total_frame", fn_total_frame, &tctx,
                               cli.warmup, cli.iterations, 1.5);
            s.drone_count = N;
            bench_print_row(&s);
            results[num_results++] = s;
        }
    }

    /* --- Clustered Distribution (worst case) --- */
    printf("\n--- Clustered Distribution (%u drones in 2x2x2m) ---\n", N);
    bench_print_header();
    {
        arena_reset(persistent);
        arena_reset(scratch);
        CollisionBenchCtx* ctx = collision_ctx_create(persistent, scratch, N, false);
        if (ctx) {
            bench_init_drones_clustered(ctx->drones, N, VEC3(0, 0, 10), 2.0f, cli.seed);

            BenchStats s;

            s = bench_measure("hash_build_clustered", fn_hash_build, ctx,
                               cli.warmup, cli.iterations, 0.2);
            s.drone_count = N;
            bench_print_row(&s);
            results[num_results++] = s;

            s = bench_measure("detect_dd_clustered", fn_detect_dd, ctx,
                               cli.warmup, cli.iterations, 0.5);
            s.drone_count = N;
            bench_print_row(&s);
            results[num_results++] = s;
        }
    }

    /* --- KNN Batch --- */
    printf("\n--- KNN Batch (%u drones) ---\n", N);
    bench_print_header();
    {
        arena_reset(persistent);
        arena_reset(scratch);
        CollisionBenchCtx* ctx = collision_ctx_create(persistent, scratch, N, false);
        if (ctx) {
            bench_init_drones_scattered(ctx->drones, N, 50.0f, cli.seed);
            collision_build_spatial_hash(ctx->collision, ctx->drones, N);

            uint32_t k = 8;
            uint32_t* out_idx = arena_alloc_array(scratch, uint32_t, N * k);
            float* out_dist = arena_alloc_array(scratch, float, N * k);

            KNNBenchCtx kctx = {
                .collision = ctx->collision,
                .drones = ctx->drones,
                .num_drones = N,
                .k = k,
                .out_indices = out_idx,
                .out_distances = out_dist
            };

            BenchStats s = bench_measure("knn_batch_k8", fn_knn_batch, &kctx,
                                          cli.warmup, cli.iterations, 10.0);
            s.drone_count = N;
            bench_print_row(&s);
            results[num_results++] = s;
        }
    }

    /* --- Scaling Test --- */
    bench_scaling_test("total_frame", bench_total_frame_scaling, &cli);

    /* --- Memory Report --- */
    printf("\n--- Memory Report ---\n");
    {
        arena_reset(persistent);
        CollisionSystem* col = collision_create(persistent, N, 0.1f, 1.0f);
        if (col) {
            printf("  Collision system memory (1024 drones):\n");
            printf("    Spatial hash:     %zu bytes\n", spatial_hash_memory_size(N));
            printf("    Collision memory: %zu bytes\n", collision_memory_size(N, N * 2));
        }
        bench_report_arena("persistent", persistent);
    }

    /* --- Summary --- */
    bench_print_separator();
    bench_print_summary(results, num_results);

    arena_destroy(scratch);
    arena_destroy(persistent);

    return 0;
}
