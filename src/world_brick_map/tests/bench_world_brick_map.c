/**
 * World Brick Map Module Benchmarks
 *
 * Performance measurement for all key operations with target thresholds.
 */

#include "../include/world_brick_map.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

/* ============================================================================
 * Timing Utilities
 * ============================================================================ */

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

#define BENCH_START() double _bench_start = get_time_ms()
#define BENCH_END(iterations) ((get_time_ms() - _bench_start) / (double)(iterations))

#define BENCH_RESULT(name, time_ns, target_ns) do { \
    bool passed = (time_ns) <= (target_ns); \
    printf("  %-35s %8.1f ns  (target: %6.0f ns)  [%s]\n", \
           name, time_ns, (double)(target_ns), passed ? "PASS" : "FAIL"); \
    if (!passed) bench_failures++; \
} while(0)

#define BENCH_RESULT_US(name, time_us, target_us) do { \
    bool passed = (time_us) <= (target_us); \
    printf("  %-35s %8.1f us  (target: %6.0f us)  [%s]\n", \
           name, time_us, (double)(target_us), passed ? "PASS" : "FAIL"); \
    if (!passed) bench_failures++; \
} while(0)

#define BENCH_RESULT_MS(name, time_ms, target_ms) do { \
    bool passed = (time_ms) <= (target_ms); \
    printf("  %-35s %8.2f ms  (target: %6.1f ms)  [%s]\n", \
           name, time_ms, (double)(target_ms), passed ? "PASS" : "FAIL"); \
    if (!passed) bench_failures++; \
} while(0)

#define BENCH_RESULT_MB(name, size_mb, target_mb) do { \
    bool passed = (size_mb) <= (target_mb); \
    printf("  %-35s %8.2f MB  (target: %6.1f MB)  [%s]\n", \
           name, size_mb, (double)(target_mb), passed ? "PASS" : "FAIL"); \
    if (!passed) bench_failures++; \
} while(0)

static int bench_failures = 0;

/* ============================================================================
 * Memory Benchmarks
 * ============================================================================ */

static void bench_memory_usage(void) {
    printf("\n=== Memory Usage Benchmarks ===\n");

    Arena* arena = arena_create(64 * 1024 * 1024);

    /* Create 256^3 world with 5% fill */
    WorldBrickMap* world = world_create(arena,
        VEC3(-12.8f, -12.8f, -12.8f), VEC3(12.8f, 12.8f, 12.8f),
        0.1f, 10000, 256);

    /* Add some geometry to simulate 5% fill */
    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 5.0f, 1);
    world_set_box(world, VEC3(5.0f, 0.0f, 0.0f), VEC3(2.0f, 2.0f, 2.0f), 2);
    world_set_box(world, VEC3(0.0f, 0.0f, -1.0f), VEC3(12.0f, 12.0f, 1.0f), 3);  /* Floor */

    /* Compact uniform bricks to reclaim memory */
    uint32_t compacted = world_compact_uniform_bricks(world);
    (void)compacted;

    WorldStats stats = world_get_stats(world);
    double total_mb = (double)stats.total_memory / (1024.0 * 1024.0);

    BENCH_RESULT_MB("Memory (256^3, 5% fill)", total_mb, 2.0);

    printf("    Active bricks: %u / %u (%.1f%%), uniform_outside: %u, uniform_inside: %u\n",
           stats.active_bricks, stats.total_bricks,
           stats.fill_ratio * 100.0f,
           stats.uniform_outside, stats.uniform_inside);

    arena_destroy(arena);
}

static void bench_memory_clipmap(void) {
    printf("\n=== Clip Map Memory Benchmarks ===\n");

    Arena* arena = arena_create(128 * 1024 * 1024);

    ClipMapWorld* clipmap = clipmap_create(arena, 0.1f, 10.0f, 2000);

    /* Calculate total memory used by all levels */
    size_t total_memory = sizeof(ClipMapWorld);
    for (int i = 0; i < CLIPMAP_LEVELS; i++) {
        WorldStats stats = world_get_stats(clipmap->levels[i].map);
        total_memory += stats.total_memory;
    }

    double total_mb = (double)total_memory / (1024.0 * 1024.0);
    BENCH_RESULT_MB("Clip map memory (4 levels)", total_mb, 8.0);

    arena_destroy(arena);
}

/* ============================================================================
 * Core Operation Benchmarks
 * ============================================================================ */

static void bench_sdf_query_single(void) {
    printf("\n=== SDF Query Benchmarks ===\n");

    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 5000, 256);

    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 5.0f, 1);

    /* Warm up */
    for (int i = 0; i < 1000; i++) {
        volatile float sdf = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
        (void)sdf;
    }

    /* Benchmark */
    const int iterations = 1000000;
    PCG32 rng;
    pcg32_seed(&rng, 12345);

    BENCH_START();
    for (int i = 0; i < iterations; i++) {
        float x = pcg32_range(&rng, -8.0f, 8.0f);
        float y = pcg32_range(&rng, -8.0f, 8.0f);
        float z = pcg32_range(&rng, -8.0f, 8.0f);
        volatile float sdf = world_sdf_query(world, VEC3(x, y, z));
        (void)sdf;
    }
    double time_ns = BENCH_END(iterations) * 1000000.0;

    BENCH_RESULT("SDF query (single)", time_ns, 50.0);

    arena_destroy(arena);
}

static void bench_sdf_query_batch(void) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 5000, 256);

    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 5.0f, 1);

    const int batch_size = 1024;
    Vec3* positions = arena_alloc_array(arena, Vec3, batch_size);
    float* sdfs = arena_alloc_array(arena, float, batch_size);

    PCG32 rng;
    pcg32_seed(&rng, 12345);

    for (int i = 0; i < batch_size; i++) {
        positions[i] = VEC3(
            pcg32_range(&rng, -8.0f, 8.0f),
            pcg32_range(&rng, -8.0f, 8.0f),
            pcg32_range(&rng, -8.0f, 8.0f)
        );
    }

    /* Warm up */
    for (int i = 0; i < 100; i++) {
        world_sdf_query_batch(world, positions, sdfs, batch_size);
    }

    const int iterations = 10000;
    BENCH_START();
    for (int i = 0; i < iterations; i++) {
        world_sdf_query_batch(world, positions, sdfs, batch_size);
    }
    double time_us = BENCH_END(iterations) * 1000.0;

    BENCH_RESULT_US("SDF query batch (1024)", time_us, 25.0);

    arena_destroy(arena);
}

static void bench_brick_allocation(void) {
    printf("\n=== Brick Allocation Benchmarks ===\n");

    Arena* arena = arena_create(256 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(100.0f, 100.0f, 100.0f),
        1.0f, 50000, 256);

    const int iterations = 100000;

    BENCH_START();
    for (int i = 0; i < iterations; i++) {
        int bx = i % (int)world->grid_x;
        int by = (i / (int)world->grid_x) % (int)world->grid_y;
        int bz = i / ((int)world->grid_x * (int)world->grid_y);
        bz = bz % (int)world->grid_z;

        int32_t idx = world_alloc_brick(world, bx, by, bz);
        (void)idx;
    }
    double time_ns = BENCH_END(iterations) * 1000000.0;

    BENCH_RESULT("Brick allocation", time_ns, 100.0);

    arena_destroy(arena);
}

/* ============================================================================
 * Raymarch Benchmarks
 * ============================================================================ */

static void bench_raymarch_single(void) {
    printf("\n=== Raymarch Benchmarks ===\n");

    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 5000, 256);

    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 3.0f, 1);
    world_set_box(world, VEC3(0.0f, 0.0f, -1.0f), VEC3(10.0f, 10.0f, 1.0f), 2);  /* Floor */

    /* Warm up */
    for (int i = 0; i < 1000; i++) {
        RayHit hit = world_raymarch(world,
            VEC3(-8.0f, 2.0f, 0.0f), VEC3(1.0f, 0.0f, 0.0f), 20.0f);
        (void)hit;
    }

    const int iterations = 100000;
    PCG32 rng;
    pcg32_seed(&rng, 12345);

    BENCH_START();
    for (int i = 0; i < iterations; i++) {
        Vec3 origin = VEC3(
            pcg32_range(&rng, -8.0f, -6.0f),
            pcg32_range(&rng, 0.0f, 5.0f),
            pcg32_range(&rng, -2.0f, 2.0f)
        );
        Vec3 dir = VEC3(1.0f, -0.1f, 0.0f);
        dir = vec3_normalize(dir);

        RayHit hit = world_raymarch(world, origin, dir, 20.0f);
        (void)hit;
    }
    double time_us = BENCH_END(iterations) * 1000.0;

    BENCH_RESULT_US("Raymarch (single, ~100 steps)", time_us, 5.0);

    arena_destroy(arena);
}

static void bench_raymarch_batch(void) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 5000, 256);

    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 3.0f, 1);
    world_set_box(world, VEC3(0.0f, 0.0f, -1.0f), VEC3(10.0f, 10.0f, 1.0f), 2);  /* Floor */

    const int batch_size = 1024;
    Vec3* origins = arena_alloc_array(arena, Vec3, batch_size);
    Vec3* directions = arena_alloc_array(arena, Vec3, batch_size);
    RayHit* hits = arena_alloc_array(arena, RayHit, batch_size);

    PCG32 rng;
    pcg32_seed(&rng, 12345);

    for (int i = 0; i < batch_size; i++) {
        origins[i] = VEC3(
            pcg32_range(&rng, -8.0f, -6.0f),
            pcg32_range(&rng, 0.0f, 5.0f),
            pcg32_range(&rng, -2.0f, 2.0f)
        );
        directions[i] = vec3_normalize(VEC3(1.0f, -0.1f, pcg32_range(&rng, -0.2f, 0.2f)));
    }

    /* Warm up */
    for (int i = 0; i < 10; i++) {
        world_raymarch_batch(world, origins, directions, 20.0f, hits, batch_size);
    }

    const int iterations = 1000;
    BENCH_START();
    for (int i = 0; i < iterations; i++) {
        world_raymarch_batch(world, origins, directions, 20.0f, hits, batch_size);
    }
    double time_ms = BENCH_END(iterations);

    BENCH_RESULT_MS("Raymarch batch (1024)", time_ms, 5.0);

    arena_destroy(arena);
}

/* ============================================================================
 * Incremental Regeneration Benchmarks
 * ============================================================================ */

static void bench_dirty_mark_region(void) {
    printf("\n=== Incremental Regeneration Benchmarks ===\n");

    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.2f, 5000, 256);
    DirtyTracker* tracker = dirty_tracker_create(arena, world->grid_total);

    const int iterations = 100000;

    BENCH_START();
    for (int i = 0; i < iterations; i++) {
        dirty_tracker_clear(tracker);
        dirty_tracker_mark_region(tracker, world,
            VEC3(-3.0f, -3.0f, -3.0f), VEC3(3.0f, 3.0f, 3.0f));
    }
    double time_us = BENCH_END(iterations) * 1000.0;

    BENCH_RESULT_US("Mark dirty region (~100 bricks)", time_us, 10.0);

    printf("    Dirty bricks marked: %u\n", dirty_tracker_count(tracker));

    arena_destroy(arena);
}

static void bench_regenerate_single_brick(void) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.2f, 5000, 256);
    EditList* edits = edit_list_create(arena, 100);

    /* Add some edits */
    edit_list_add(edits, CSG_UNION, PRIM_SPHERE, VEC3(0,0,0), VEC3(3,0,0), 1);
    edit_list_add(edits, CSG_UNION, PRIM_BOX, VEC3(2,0,0), VEC3(1,1,1), 2);
    edit_list_add(edits, CSG_SUBTRACT, PRIM_SPHERE, VEC3(0,0,0), VEC3(1.5f,0,0), 0);

    const int iterations = 100000;
    uint32_t brick_idx = brick_linear_index(world, 1, 1, 1);

    BENCH_START();
    for (int i = 0; i < iterations; i++) {
        world_regenerate_brick(world, brick_idx, edits);
    }
    double time_us = BENCH_END(iterations) * 1000.0;

    BENCH_RESULT_US("Regenerate single brick", time_us, 50.0);

    arena_destroy(arena);
}

static void bench_regenerate_dirty(void) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.2f, 5000, 256);
    EditList* edits = edit_list_create(arena, 100);
    DirtyTracker* tracker = dirty_tracker_create(arena, world->grid_total);

    edit_list_add(edits, CSG_UNION, PRIM_SPHERE, VEC3(0,0,0), VEC3(3,0,0), 1);

    /* Mark ~100 bricks dirty */
    dirty_tracker_mark_region(tracker, world,
        VEC3(-3.0f, -3.0f, -3.0f), VEC3(3.0f, 3.0f, 3.0f));

    uint32_t num_dirty = dirty_tracker_count(tracker);

    const int iterations = 10000;

    BENCH_START();
    for (int i = 0; i < iterations; i++) {
        /* Re-mark for each iteration */
        dirty_tracker_mark_region(tracker, world,
            VEC3(-3.0f, -3.0f, -3.0f), VEC3(3.0f, 3.0f, 3.0f));
        world_regenerate_dirty(world, tracker, edits);
    }
    double time_ms = BENCH_END(iterations);

    BENCH_RESULT_MS("Regenerate 100 dirty bricks", time_ms, 5.0);

    printf("    Bricks per iteration: %u\n", num_dirty);

    arena_destroy(arena);
}

static void bench_edit_list_append(void) {
    Arena* arena = arena_create(8 * 1024 * 1024);  /* Need space for 100K EditEntry (~4MB) */
    EditList* list = edit_list_create(arena, 100000);

    const int iterations = 10000000;

    BENCH_START();
    for (int i = 0; i < iterations && (uint32_t)i < list->capacity; i++) {
        edit_list_add(list, CSG_UNION, PRIM_SPHERE,
                      VEC3((float)i, 0, 0), VEC3(1, 0, 0), 1);
    }
    double time_ns = BENCH_END(min_u32(iterations, list->capacity)) * 1000000.0;

    BENCH_RESULT("Edit list append", time_ns, 20.0);

    arena_destroy(arena);
}

/* ============================================================================
 * Clip Map Benchmarks
 * ============================================================================ */

static void bench_clipmap_level_selection(void) {
    printf("\n=== Clip Map Benchmarks ===\n");

    Arena* arena = arena_create(64 * 1024 * 1024);
    ClipMapWorld* clipmap = clipmap_create(arena, 0.1f, 10.0f, 1000);

    PCG32 rng;
    pcg32_seed(&rng, 12345);

    /* Warm up */
    for (int i = 0; i < 1000; i++) {
        volatile int level = clipmap_select_level(clipmap, VEC3(
            pcg32_range(&rng, -50.0f, 50.0f),
            pcg32_range(&rng, -50.0f, 50.0f),
            pcg32_range(&rng, -50.0f, 50.0f)
        ));
        (void)level;
    }

    const int iterations = 10000000;

    BENCH_START();
    for (int i = 0; i < iterations; i++) {
        volatile int level = clipmap_select_level(clipmap, VEC3(
            (float)(i % 100) - 50.0f,
            (float)((i / 100) % 100) - 50.0f,
            0.0f
        ));
        (void)level;
    }
    double time_ns = BENCH_END(iterations) * 1000000.0;

    BENCH_RESULT("Level selection", time_ns, 10.0);

    arena_destroy(arena);
}

static void bench_clipmap_sdf_query(void) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    ClipMapWorld* clipmap = clipmap_create(arena, 0.1f, 10.0f, 2000);

    clipmap_set_sphere(clipmap, VEC3(0, 0, 0), 5.0f, 1);

    PCG32 rng;
    pcg32_seed(&rng, 12345);

    const int iterations = 1000000;

    BENCH_START();
    for (int i = 0; i < iterations; i++) {
        volatile float sdf = clipmap_sdf_query(clipmap, VEC3(
            pcg32_range(&rng, -8.0f, 8.0f),
            pcg32_range(&rng, -8.0f, 8.0f),
            pcg32_range(&rng, -8.0f, 8.0f)
        ));
        (void)sdf;
    }
    double time_ns = BENCH_END(iterations) * 1000000.0;

    BENCH_RESULT("Clip map SDF query", time_ns, 100.0);

    arena_destroy(arena);
}

static void bench_clipmap_raymarch(void) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    ClipMapWorld* clipmap = clipmap_create(arena, 0.1f, 10.0f, 2000);

    clipmap_set_sphere(clipmap, VEC3(0, 0, 0), 3.0f, 1);

    PCG32 rng;
    pcg32_seed(&rng, 12345);

    const int iterations = 100000;

    BENCH_START();
    for (int i = 0; i < iterations; i++) {
        Vec3 origin = VEC3(-8.0f, pcg32_range(&rng, -2.0f, 2.0f), 0.0f);
        RayHit hit = clipmap_raymarch(clipmap, origin, VEC3(1, 0, 0), 20.0f);
        (void)hit;
    }
    double time_us = BENCH_END(iterations) * 1000.0;

    BENCH_RESULT_US("Clip map raymarch", time_us, 10.0);

    arena_destroy(arena);
}

static void bench_clipmap_focus_update(void) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    ClipMapWorld* clipmap = clipmap_create(arena, 0.1f, 10.0f, 2000);

    clipmap_set_sphere(clipmap, VEC3(0, 0, 0), 5.0f, 1);

    const int iterations = 10000;

    BENCH_START();
    for (int i = 0; i < iterations; i++) {
        float x = (float)(i % 100) * 0.1f;
        float y = (float)((i / 100) % 100) * 0.1f;
        clipmap_update_focus(clipmap, VEC3(x, y, 0));
    }
    double time_us = BENCH_END(iterations) * 1000.0;

    BENCH_RESULT_US("Focus update (all levels)", time_us, 500.0);

    arena_destroy(arena);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("=== World Brick Map Benchmarks ===\n");
    printf("Platform: ");
#if defined(FOUNDATION_SIMD_AVX2)
    printf("AVX2\n");
#elif defined(FOUNDATION_SIMD_NEON)
    printf("NEON\n");
#else
    printf("Scalar\n");
#endif

    /* Memory benchmarks */
    bench_memory_usage();
    bench_memory_clipmap();

    /* Core operation benchmarks */
    bench_sdf_query_single();
    bench_sdf_query_batch();
    bench_brick_allocation();

    /* Raymarch benchmarks */
    bench_raymarch_single();
    bench_raymarch_batch();

    /* Incremental regeneration benchmarks */
    bench_dirty_mark_region();
    bench_regenerate_single_brick();
    bench_regenerate_dirty();
    bench_edit_list_append();

    /* Clip map benchmarks */
    bench_clipmap_level_selection();
    bench_clipmap_sdf_query();
    bench_clipmap_raymarch();
    bench_clipmap_focus_update();

    printf("\n=== Summary ===\n");
    if (bench_failures == 0) {
        printf("All benchmarks PASSED\n");
    } else {
        printf("FAILED: %d benchmark(s) exceeded target\n", bench_failures);
    }

    return bench_failures > 0 ? 1 : 0;
}
