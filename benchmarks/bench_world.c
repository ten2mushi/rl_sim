/**
 * World Brick Map Benchmarks
 *
 * Benchmarks for SDF queries, raymarching, CSG operations, and brick allocation.
 * Tests with varying geometry complexity and reports memory usage.
 */

#include "bench_harness.h"
#include "world_brick_map.h"

/* ============================================================================
 * Fixture: World with CSG geometry
 * ============================================================================ */

typedef struct WorldBenchCtx {
    Arena* arena;
    WorldBrickMap* world;
    Vec3* query_positions;
    Vec3* ray_origins;
    Vec3* ray_directions;
    float* sdf_results;
    RayHit* ray_results;
    uint32_t query_count;
    uint32_t ray_count;
} WorldBenchCtx;

static WorldBenchCtx* world_ctx_create(Arena* arena, uint32_t num_obstacles) {
    WorldBenchCtx* ctx = arena_alloc_type(arena, WorldBenchCtx);
    memset(ctx, 0, sizeof(*ctx));
    ctx->arena = arena;

    /* Create world */
    Vec3 wmin = VEC3(-50, -50, -10);
    Vec3 wmax = VEC3(50, 50, 50);
    ctx->world = world_create(arena, wmin, wmax, 0.1f, 10000, 0);
    if (!ctx->world) return NULL;

    /* Ground plane via box */
    world_set_box(ctx->world, VEC3(0, 0, -5), VEC3(50, 50, 5), 1);

    /* Add sphere obstacles */
    PCG32 rng;
    pcg32_seed(&rng, 42);
    for (uint32_t i = 0; i < num_obstacles; i++) {
        Vec3 center = VEC3(
            pcg32_range(&rng, -30, 30),
            pcg32_range(&rng, -30, 30),
            pcg32_range(&rng, 2, 20)
        );
        float radius = pcg32_range(&rng, 1.0f, 5.0f);
        world_set_sphere(ctx->world, center, radius, 1);
    }

    return ctx;
}

static void world_ctx_init_queries(WorldBenchCtx* ctx, Arena* scratch,
                                    uint32_t query_count, uint64_t seed) {
    ctx->query_count = query_count;
    ctx->query_positions = arena_alloc_array(scratch, Vec3, query_count);
    ctx->sdf_results = arena_alloc_array(scratch, float, query_count);

    PCG32 rng;
    pcg32_seed(&rng, seed);
    for (uint32_t i = 0; i < query_count; i++) {
        ctx->query_positions[i] = VEC3(
            pcg32_range(&rng, -40, 40),
            pcg32_range(&rng, -40, 40),
            pcg32_range(&rng, -5, 40)
        );
    }
}

static void world_ctx_init_rays(WorldBenchCtx* ctx, Arena* scratch,
                                 uint32_t ray_count, uint64_t seed) {
    ctx->ray_count = ray_count;
    ctx->ray_origins = arena_alloc_array(scratch, Vec3, ray_count);
    ctx->ray_directions = arena_alloc_array(scratch, Vec3, ray_count);
    ctx->ray_results = arena_alloc_array(scratch, RayHit, ray_count);

    PCG32 rng;
    pcg32_seed(&rng, seed);
    for (uint32_t i = 0; i < ray_count; i++) {
        ctx->ray_origins[i] = VEC3(
            pcg32_range(&rng, -20, 20),
            pcg32_range(&rng, -20, 20),
            pcg32_range(&rng, 10, 30)
        );
        /* Random downward-ish direction */
        Vec3 dir = VEC3(
            pcg32_range(&rng, -0.5f, 0.5f),
            pcg32_range(&rng, -0.5f, 0.5f),
            pcg32_range(&rng, -1.0f, -0.2f)
        );
        ctx->ray_directions[i] = vec3_normalize(dir);
    }
}

/* ============================================================================
 * Benchmark Functions
 * ============================================================================ */

static void fn_sdf_query_batch(void* arg) {
    WorldBenchCtx* ctx = (WorldBenchCtx*)arg;
    world_sdf_query_batch(ctx->world, ctx->query_positions,
                          ctx->sdf_results, ctx->query_count);
}

static void fn_raymarch_batch(void* arg) {
    WorldBenchCtx* ctx = (WorldBenchCtx*)arg;
    world_raymarch_batch(ctx->world, ctx->ray_origins, ctx->ray_directions,
                         100.0f, ctx->ray_results, ctx->ray_count);
}

/* CSG operation context */
typedef struct CSGBenchCtx {
    WorldBrickMap* world;
    Vec3 center;
    float radius;
    Vec3 half_size;
} CSGBenchCtx;

static void fn_csg_sphere(void* arg) {
    CSGBenchCtx* ctx = (CSGBenchCtx*)arg;
    world_set_sphere(ctx->world, ctx->center, ctx->radius, 1);
}

static void fn_csg_box(void* arg) {
    CSGBenchCtx* ctx = (CSGBenchCtx*)arg;
    world_set_box(ctx->world, ctx->center, ctx->half_size, 1);
}

/* Brick allocation context */
typedef struct BrickAllocCtx {
    WorldBrickMap* world;
    uint32_t alloc_count;
    PCG32 rng;
} BrickAllocCtx;

static void fn_brick_alloc(void* arg) {
    BrickAllocCtx* ctx = (BrickAllocCtx*)arg;
    PCG32 rng = ctx->rng; /* local copy for speed */
    for (uint32_t i = 0; i < ctx->alloc_count; i++) {
        int32_t bx = (int32_t)pcg32_bounded(&rng, ctx->world->grid_x);
        int32_t by = (int32_t)pcg32_bounded(&rng, ctx->world->grid_y);
        int32_t bz = (int32_t)pcg32_bounded(&rng, ctx->world->grid_z);
        world_alloc_brick(ctx->world, bx, by, bz);
    }
    ctx->rng = rng;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char** argv) {
    BenchCLI cli = bench_parse_cli(argc, argv);

    printf("=== World Brick Map Benchmarks ===\n");
    printf("SIMD width: %d floats\n\n", FOUNDATION_SIMD_WIDTH);

    Arena* persistent = arena_create(256 * 1024 * 1024);
    Arena* scratch = arena_create(64 * 1024 * 1024);
    if (!persistent || !scratch) {
        fprintf(stderr, "Failed to create arenas\n");
        return 1;
    }

    BenchStats results[16];
    uint32_t num_results = 0;

    /* --- SDF Query Benchmarks --- */
    printf("--- SDF Query Benchmarks ---\n");
    bench_print_header();

    {
        ArenaScope scope = arena_scope_begin(persistent);
        WorldBenchCtx* ctx = world_ctx_create(persistent, 3);
        if (ctx) {
            world_ctx_init_queries(ctx, scratch, 1024, cli.seed);
            BenchStats s = bench_measure("sdf_query_batch_1024", fn_sdf_query_batch, ctx,
                                          cli.warmup, cli.iterations, 0.025);
            s.drone_count = 1024;
            bench_print_row(&s);
            results[num_results++] = s;
        }
        arena_scope_end(scope);
        arena_reset(scratch);
    }

    {
        ArenaScope scope = arena_scope_begin(persistent);
        WorldBenchCtx* ctx = world_ctx_create(persistent, 3);
        if (ctx) {
            world_ctx_init_queries(ctx, scratch, 4096, cli.seed);
            BenchStats s = bench_measure("sdf_query_batch_4096", fn_sdf_query_batch, ctx,
                                          cli.warmup, cli.iterations, 0.1);
            s.drone_count = 4096;
            bench_print_row(&s);
            results[num_results++] = s;
        }
        arena_scope_end(scope);
        arena_reset(scratch);
    }

    /* --- Raymarch Benchmarks --- */
    printf("\n--- Raymarch Benchmarks ---\n");
    bench_print_header();

    {
        ArenaScope scope = arena_scope_begin(persistent);
        WorldBenchCtx* ctx = world_ctx_create(persistent, 3);
        if (ctx) {
            world_ctx_init_rays(ctx, scratch, 1024, cli.seed);
            BenchStats s = bench_measure("raymarch_batch_1024", fn_raymarch_batch, ctx,
                                          cli.warmup, cli.iterations, 5.0);
            s.drone_count = 1024;
            bench_print_row(&s);
            results[num_results++] = s;
        }
        arena_scope_end(scope);
        arena_reset(scratch);
    }

    /* Raymarch with varying geometry complexity */
    printf("\n--- Raymarch vs Geometry Complexity ---\n");
    bench_print_header();
    {
        uint32_t complexities[] = {1, 3, 5, 10};
        for (uint32_t ci = 0; ci < 4; ci++) {
            arena_reset(persistent);
            arena_reset(scratch);
            WorldBenchCtx* ctx = world_ctx_create(persistent, complexities[ci]);
            if (ctx) {
                world_ctx_init_rays(ctx, scratch, 1024, cli.seed);
                char name[64];
                snprintf(name, sizeof(name), "raymarch_1024_%u_prims", complexities[ci]);
                BenchStats s = bench_measure(name, fn_raymarch_batch, ctx,
                                              cli.warmup, cli.iterations, 10.0);
                s.drone_count = 1024;
                bench_print_row(&s);
                results[num_results++] = s;
            }
        }
    }

    /* --- CSG Operation Benchmarks --- */
    printf("\n--- CSG Operation Benchmarks ---\n");
    bench_print_header();

    {
        arena_reset(persistent);
        arena_reset(scratch);
        WorldBenchCtx* wctx = world_ctx_create(persistent, 0);
        if (wctx) {
            CSGBenchCtx csg_ctx = {
                .world = wctx->world,
                .center = VEC3(10, 10, 10),
                .radius = 5.0f
            };
            BenchStats s = bench_measure("csg_sphere", fn_csg_sphere, &csg_ctx,
                                          5, cli.iterations, 10.0);
            bench_print_row(&s);
            results[num_results++] = s;
        }
    }

    {
        arena_reset(persistent);
        arena_reset(scratch);
        WorldBenchCtx* wctx = world_ctx_create(persistent, 0);
        if (wctx) {
            CSGBenchCtx csg_ctx = {
                .world = wctx->world,
                .center = VEC3(10, 10, 10),
                .half_size = VEC3(5, 5, 5)
            };
            BenchStats s = bench_measure("csg_box", fn_csg_box, &csg_ctx,
                                          5, cli.iterations, 5.0);
            bench_print_row(&s);
            results[num_results++] = s;
        }
    }

    /* --- Brick Allocation Benchmark --- */
    printf("\n--- Brick Allocation Benchmark ---\n");
    bench_print_header();

    {
        arena_reset(persistent);
        WorldBrickMap* world = world_create(persistent, VEC3(-50, -50, -10),
                                             VEC3(50, 50, 50), 0.1f, 20000, 0);
        if (world) {
            BrickAllocCtx ba_ctx = {
                .world = world,
                .alloc_count = 1000
            };
            pcg32_seed(&ba_ctx.rng, cli.seed);
            /* Note: 100 iterations of 1000 allocs = 100K total */
            BenchStats s = bench_measure("brick_alloc_100K", fn_brick_alloc, &ba_ctx,
                                          5, 100, 10.0);
            bench_print_row(&s);
            results[num_results++] = s;
        }
    }

    /* --- Memory Report --- */
    printf("\n--- Memory Report ---\n");
    {
        arena_reset(persistent);
        WorldBrickMap* world = world_create(persistent, VEC3(-50, -50, -10),
                                             VEC3(50, 50, 50), 0.1f, 10000, 0);
        if (world) {
            /* Add geometry */
            world_set_box(world, VEC3(0, 0, -5), VEC3(50, 50, 5), 1);
            world_set_sphere(world, VEC3(10, 10, 10), 5.0f, 1);
            world_set_sphere(world, VEC3(-10, -10, 15), 3.0f, 1);

            WorldStats ws = world_get_stats(world);
            printf("  World Stats:\n");
            printf("    Active bricks:    %u / %u\n", ws.active_bricks, ws.total_bricks);
            printf("    Uniform outside:  %u\n", ws.uniform_outside);
            printf("    Uniform inside:   %u\n", ws.uniform_inside);
            printf("    Grid memory:      %zu bytes (%.1f KB)\n", ws.grid_memory, ws.grid_memory / 1024.0);
            printf("    Atlas memory:     %zu bytes (%.1f KB)\n", ws.atlas_memory, ws.atlas_memory / 1024.0);
            printf("    Total memory:     %zu bytes (%.1f MB)\n", ws.total_memory, ws.total_memory / (1024.0 * 1024.0));
            printf("    Fill ratio:       %.2f%%\n", ws.fill_ratio * 100.0f);
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
