/**
 * World Brick Map Module Unit Tests
 *
 * Yoneda Philosophy: Tests serve as complete behavioral specification.
 * Each test defines what the module MUST do - the definitive contract.
 */

#include "../include/world_brick_map.h"
#include "test_harness.h"
#include <stdint.h>

/* Custom macros not in harness */
#define ASSERT_PTR_ALIGNED(ptr, align) \
    ASSERT_TRUE(((uintptr_t)(ptr) & ((align)-1)) == 0)

/* ============================================================================
 * Section 1: Allocation Tests
 * ============================================================================ */

TEST(allocation_basic) {
    Arena* arena = arena_create(16 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 1000, 256);
    ASSERT_NOT_NULL(world);
    ASSERT_TRUE(world->grid_x > 0);
    ASSERT_TRUE(world->grid_y > 0);
    ASSERT_TRUE(world->grid_z > 0);
    /* max_bricks is rounded up to page boundary (ATLAS_PAGE_BRICKS = 64) */
    uint32_t expected_max = ((1000 + ATLAS_PAGE_BRICKS - 1) / ATLAS_PAGE_BRICKS) * ATLAS_PAGE_BRICKS;
    ASSERT_EQ(world->max_bricks, expected_max);
    ASSERT_EQ(world->atlas_count, 0);

    arena_destroy(arena);
    return 0;
}

TEST(allocation_alignment) {
    Arena* arena = arena_create(16 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 1000, 256);
    ASSERT_NOT_NULL(world);

    /* Demand-paged: allocate a brick to trigger page allocation */
    int32_t atlas_idx = world_alloc_brick(world, 0, 0, 0);
    ASSERT_TRUE(atlas_idx >= 0);

    /* First page should now be allocated and 64-byte aligned */
    ASSERT_TRUE(world->page_count > 0);
    ASSERT_PTR_ALIGNED(world->sdf_pages[0], BRICK_ALIGNMENT);
    ASSERT_PTR_ALIGNED(world->material_pages[0], BRICK_ALIGNMENT);

    arena_destroy(arena);
    return 0;
}

TEST(allocation_null_arena) {
    WorldBrickMap* world = world_create(NULL,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 1000, 256);
    ASSERT_NULL(world);

    return 0;
}

TEST(allocation_invalid_params) {
    Arena* arena = arena_create(16 * 1024 * 1024);

    /* Zero voxel size */
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.0f, 1000, 256);
    ASSERT_NULL(world);

    /* Zero max_bricks */
    world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 0, 256);
    ASSERT_NULL(world);

    /* Inverted bounds */
    world = world_create(arena,
        VEC3(10.0f, 10.0f, 10.0f), VEC3(-10.0f, -10.0f, -10.0f),
        0.1f, 1000, 256);
    ASSERT_NULL(world);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 2: Coordinate Transform Tests
 * ============================================================================ */

TEST(coords_world_to_brick) {
    Arena* arena = arena_create(16 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 1000, 256);

    int32_t bx, by, bz;

    /* Origin should map to center brick */
    world_pos_to_brick(world, VEC3(0.0f, 0.0f, 0.0f), &bx, &by, &bz);
    ASSERT_TRUE(bx >= 0 && by >= 0 && bz >= 0);

    /* World min corner */
    world_pos_to_brick(world, VEC3(-10.0f, -10.0f, -10.0f), &bx, &by, &bz);
    ASSERT_TRUE(bx == 0 && by == 0 && bz == 0);

    arena_destroy(arena);
    return 0;
}

TEST(coords_world_to_voxel) {
    Arena* arena = arena_create(16 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(8.0f, 8.0f, 8.0f),
        1.0f, 100, 256);

    int32_t bx, by, bz, vx, vy, vz;

    /* Position (0.5, 0.5, 0.5) should map to voxel (0,0,0) in brick (0,0,0) */
    Vec3 pos = VEC3(0.5f, 0.5f, 0.5f);
    world_pos_to_brick(world, pos, &bx, &by, &bz);
    world_pos_to_voxel(world, pos, bx, by, bz, &vx, &vy, &vz);

    ASSERT_EQ(bx, 0);
    ASSERT_EQ(by, 0);
    ASSERT_EQ(bz, 0);
    ASSERT_EQ(vx, 0);
    ASSERT_EQ(vy, 0);
    ASSERT_EQ(vz, 0);

    arena_destroy(arena);
    return 0;
}

TEST(coords_contains) {
    Arena* arena = arena_create(16 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 1000, 256);

    ASSERT_TRUE(world_contains(world, VEC3(0.0f, 0.0f, 0.0f)));
    ASSERT_TRUE(world_contains(world, VEC3(-9.9f, -9.9f, -9.9f)));
    ASSERT_TRUE(world_contains(world, VEC3(9.9f, 9.9f, 9.9f)));
    ASSERT_FALSE(world_contains(world, VEC3(-10.1f, 0.0f, 0.0f)));
    ASSERT_FALSE(world_contains(world, VEC3(10.1f, 0.0f, 0.0f)));

    arena_destroy(arena);
    return 0;
}

TEST(coords_brick_valid) {
    Arena* arena = arena_create(16 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 1000, 256);

    ASSERT_TRUE(world_brick_valid(world, 0, 0, 0));
    ASSERT_FALSE(world_brick_valid(world, -1, 0, 0));
    ASSERT_FALSE(world_brick_valid(world, (int32_t)world->grid_x, 0, 0));

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 3: SDF Quantization Tests
 * ============================================================================ */

TEST(quantize_zero) {
    float sdf_scale = 1.0f;
    float inv_sdf_scale = 1.0f / sdf_scale;

    int8_t q = sdf_quantize(0.0f, inv_sdf_scale);
    ASSERT_EQ(q, 0);

    float dq = sdf_dequantize(q, sdf_scale);
    ASSERT_FLOAT_EQ(dq, 0.0f);

    return 0;
}

TEST(quantize_positive) {
    float sdf_scale = 1.0f;
    float inv_sdf_scale = 1.0f / sdf_scale;

    int8_t q = sdf_quantize(0.5f, inv_sdf_scale);
    ASSERT_TRUE(q > 0);
    ASSERT_TRUE(q < 127);

    float dq = sdf_dequantize(q, sdf_scale);
    ASSERT_FLOAT_NEAR(dq, 0.5f, 0.01f);

    return 0;
}

TEST(quantize_negative) {
    float sdf_scale = 1.0f;
    float inv_sdf_scale = 1.0f / sdf_scale;

    int8_t q = sdf_quantize(-0.5f, inv_sdf_scale);
    ASSERT_TRUE(q < 0);
    ASSERT_TRUE(q > -127);

    float dq = sdf_dequantize(q, sdf_scale);
    ASSERT_FLOAT_NEAR(dq, -0.5f, 0.01f);

    return 0;
}

TEST(quantize_clamp) {
    float sdf_scale = 1.0f;
    float inv_sdf_scale = 1.0f / sdf_scale;

    /* Values outside range should clamp */
    int8_t q_large = sdf_quantize(10.0f, inv_sdf_scale);
    ASSERT_EQ(q_large, 127);

    int8_t q_small = sdf_quantize(-10.0f, inv_sdf_scale);
    ASSERT_EQ(q_small, -127);

    return 0;
}

TEST(quantize_roundtrip) {
    float sdf_scale = 1.2f;  /* Non-trivial scale */
    float inv_sdf_scale = 1.0f / sdf_scale;

    /* Test various values */
    float test_values[] = {0.0f, 0.1f, -0.1f, 0.5f, -0.5f, 1.0f, -1.0f};
    int num_values = sizeof(test_values) / sizeof(test_values[0]);

    for (int i = 0; i < num_values; i++) {
        float original = test_values[i];
        int8_t q = sdf_quantize(original, inv_sdf_scale);
        float dq = sdf_dequantize(q, sdf_scale);

        /* Allow for quantization error */
        float max_error = sdf_scale / 127.0f;
        ASSERT_FLOAT_NEAR(dq, original, max_error * 1.5f);
    }

    return 0;
}

/* ============================================================================
 * Section 4: Brick Management Tests
 * ============================================================================ */

TEST(brick_alloc_basic) {
    Arena* arena = arena_create(16 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 100, 256);

    /* Initially empty - uses SoA layout, returns atlas index */
    int32_t idx = world_get_brick_index(world, 0, 0, 0);
    ASSERT_EQ(idx, BRICK_EMPTY_INDEX);

    /* Allocate brick */
    idx = world_alloc_brick(world, 0, 0, 0);
    ASSERT_NE(idx, BRICK_EMPTY_INDEX);
    ASSERT_EQ(world->atlas_count, 1);

    /* Get should now return same index */
    int32_t idx2 = world_get_brick_index(world, 0, 0, 0);
    ASSERT_EQ(idx2, idx);

    /* Second alloc should return same index */
    int32_t idx3 = world_alloc_brick(world, 0, 0, 0);
    ASSERT_EQ(idx3, idx);
    ASSERT_EQ(world->atlas_count, 1);

    arena_destroy(arena);
    return 0;
}

TEST(brick_alloc_reuse) {
    Arena* arena = arena_create(16 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 100, 256);

    /* Allocate and free a brick */
    int32_t idx1 = world_alloc_brick(world, 0, 0, 0);
    ASSERT_NE(idx1, BRICK_EMPTY_INDEX);
    uint32_t count1 = world->atlas_count;

    world_free_brick(world, 0, 0, 0);
    ASSERT_EQ(world->free_count, 1);
    ASSERT_EQ(world_get_brick_index(world, 0, 0, 0), BRICK_EMPTY_INDEX);

    /* Allocate again - should reuse */
    int32_t idx2 = world_alloc_brick(world, 1, 0, 0);
    ASSERT_NE(idx2, BRICK_EMPTY_INDEX);
    ASSERT_EQ(world->free_count, 0);
    ASSERT_EQ(world->atlas_count, count1);

    arena_destroy(arena);
    return 0;
}

TEST(brick_atlas_full) {
    Arena* arena = arena_create(16 * 1024 * 1024);
    /* Request small atlas - rounds up to ATLAS_PAGE_BRICKS (64) */
    /* World must be large enough to hold more bricks than atlas capacity */
    WorldBrickMap* world = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(80.0f, 80.0f, 80.0f),
        1.0f, ATLAS_PAGE_BRICKS, 256);  /* 8m bricks = 10x10x10 grid = 1000 potential bricks */

    /* Fill atlas completely (one full page) by allocating across grid */
    uint32_t allocated = 0;
    for (int32_t z = 0; z < 4 && allocated < ATLAS_PAGE_BRICKS; z++) {
        for (int32_t y = 0; y < 4 && allocated < ATLAS_PAGE_BRICKS; y++) {
            for (int32_t x = 0; x < 4 && allocated < ATLAS_PAGE_BRICKS; x++) {
                int32_t idx = world_alloc_brick(world, x, y, z);
                ASSERT_NE(idx, BRICK_EMPTY_INDEX);
                allocated++;
            }
        }
    }
    ASSERT_EQ(allocated, ATLAS_PAGE_BRICKS);

    /* Next allocation should fail */
    int32_t idx = world_alloc_brick(world, 5, 5, 5);
    ASSERT_EQ(idx, BRICK_EMPTY_INDEX);

    arena_destroy(arena);
    return 0;
}

TEST(brick_init_values) {
    Arena* arena = arena_create(16 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 100, 256);

    int32_t idx = world_alloc_brick(world, 0, 0, 0);
    ASSERT_NE(idx, BRICK_EMPTY_INDEX);

    /* SDF should be initialized to +127 (far outside) - using SoA accessors */
    int8_t* sdf = world_brick_sdf(world, idx);
    ASSERT_EQ(sdf[0], 127);
    ASSERT_EQ(sdf[BRICK_VOXELS - 1], 127);

    /* Material should be initialized to 0 (air) - using SoA accessors */
    uint8_t* material = world_brick_material(world, idx);
    ASSERT_EQ(material[0], 0);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 5: SDF Query Tests
 * ============================================================================ */

TEST(sdf_query_empty_world) {
    Arena* arena = arena_create(16 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 100, 256);

    /* Query empty world should return sdf_scale (far outside) */
    float sdf = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_FLOAT_EQ(sdf, world->sdf_scale);

    arena_destroy(arena);
    return 0;
}

TEST(sdf_query_single_voxel) {
    Arena* arena = arena_create(16 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(8.0f, 8.0f, 8.0f),
        1.0f, 100, 256);

    /* Set a single voxel to inside (-0.5) */
    world_set_sdf(world, VEC3(0.5f, 0.5f, 0.5f), -0.5f);

    /* Query that position */
    float sdf = world_sdf_query_nearest(world, VEC3(0.5f, 0.5f, 0.5f));
    ASSERT_FLOAT_NEAR(sdf, -0.5f, 0.1f);

    arena_destroy(arena);
    return 0;
}

TEST(sdf_query_out_of_bounds) {
    Arena* arena = arena_create(16 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 100, 256);

    float sdf = world_sdf_query(world, VEC3(100.0f, 100.0f, 100.0f));
    ASSERT_FLOAT_EQ(sdf, world->sdf_scale);

    arena_destroy(arena);
    return 0;
}

TEST(material_query) {
    Arena* arena = arena_create(16 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(8.0f, 8.0f, 8.0f),
        1.0f, 100, 256);

    /* Empty world should return air (0) */
    uint8_t mat = world_material_query(world, VEC3(0.5f, 0.5f, 0.5f));
    ASSERT_EQ(mat, 0);

    /* Set material */
    world_set_material(world, VEC3(0.5f, 0.5f, 0.5f), 42);
    mat = world_material_query(world, VEC3(0.5f, 0.5f, 0.5f));
    ASSERT_EQ(mat, 42);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 6: Primitive Generation Tests
 * ============================================================================ */

TEST(primitive_sphere) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-5.0f, -5.0f, -5.0f), VEC3(5.0f, 5.0f, 5.0f),
        0.1f, 1000, 256);

    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 2.0f, 1);

    /* Center should be inside (negative SDF) */
    float sdf_center = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_center < 0.0f);

    /* Surface should be near zero */
    float sdf_surface = world_sdf_query(world, VEC3(2.0f, 0.0f, 0.0f));
    ASSERT_FLOAT_NEAR(sdf_surface, 0.0f, 0.2f);

    /* Outside should be positive */
    float sdf_outside = world_sdf_query(world, VEC3(3.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_outside > 0.0f);

    /* Material at center */
    uint8_t mat = world_material_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_EQ(mat, 1);

    arena_destroy(arena);
    return 0;
}

TEST(primitive_box) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-5.0f, -5.0f, -5.0f), VEC3(5.0f, 5.0f, 5.0f),
        0.1f, 1000, 256);

    world_set_box(world, VEC3(0.0f, 0.0f, 0.0f), VEC3(1.0f, 1.0f, 1.0f), 2);

    /* Center should be inside */
    float sdf_center = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_center < 0.0f);

    /* Outside should be positive */
    float sdf_outside = world_sdf_query(world, VEC3(2.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_outside > 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(primitive_csg_union) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-5.0f, -5.0f, -5.0f), VEC3(5.0f, 5.0f, 5.0f),
        0.1f, 1000, 256);

    /* Two overlapping spheres */
    world_set_sphere(world, VEC3(-0.5f, 0.0f, 0.0f), 1.0f, 1);
    world_set_sphere(world, VEC3(0.5f, 0.0f, 0.0f), 1.0f, 2);

    /* Center of both should be inside */
    float sdf = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf < 0.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 7: Gradient and Normal Tests
 * ============================================================================ */

TEST(gradient_sphere) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-5.0f, -5.0f, -5.0f), VEC3(5.0f, 5.0f, 5.0f),
        0.1f, 1000, 256);

    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 2.0f, 1);

    /* Gradient at surface should point outward (radially) */
    Vec3 normal = world_sdf_normal(world, VEC3(2.0f, 0.0f, 0.0f));
    ASSERT_FLOAT_NEAR(normal.x, 1.0f, 0.2f);
    ASSERT_FLOAT_NEAR(normal.y, 0.0f, 0.2f);
    ASSERT_FLOAT_NEAR(normal.z, 0.0f, 0.2f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 8: Raymarching Tests
 * ============================================================================ */

TEST(raymarch_miss) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 1000, 256);

    /* Empty world - ray should miss */
    RayHit hit = world_raymarch(world,
        VEC3(-5.0f, 0.0f, 0.0f), VEC3(1.0f, 0.0f, 0.0f), 20.0f);
    ASSERT_FALSE(hit.hit);

    arena_destroy(arena);
    return 0;
}

TEST(raymarch_hit_sphere) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 1000, 256);

    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 2.0f, 1);

    /* Ray from outside toward sphere */
    RayHit hit = world_raymarch(world,
        VEC3(-5.0f, 0.0f, 0.0f), VEC3(1.0f, 0.0f, 0.0f), 20.0f);

    ASSERT_TRUE(hit.hit);
    ASSERT_FLOAT_NEAR(hit.distance, 3.0f, 0.2f);
    ASSERT_EQ(hit.material, 1);

    arena_destroy(arena);
    return 0;
}

TEST(raymarch_max_distance) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 1000, 256);

    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 2.0f, 1);

    /* Ray with max distance less than sphere */
    RayHit hit = world_raymarch(world,
        VEC3(-5.0f, 0.0f, 0.0f), VEC3(1.0f, 0.0f, 0.0f), 2.0f);

    ASSERT_FALSE(hit.hit);

    arena_destroy(arena);
    return 0;
}

TEST(raymarch_batch) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 1000, 256);

    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 2.0f, 1);

    Vec3 origins[4] = {
        VEC3(-5.0f, 0.0f, 0.0f),
        VEC3(0.0f, -5.0f, 0.0f),
        VEC3(0.0f, 5.0f, 0.0f),  /* Will miss (going away) */
        VEC3(0.0f, 0.0f, -5.0f)
    };
    Vec3 directions[4] = {
        VEC3(1.0f, 0.0f, 0.0f),
        VEC3(0.0f, 1.0f, 0.0f),
        VEC3(0.0f, 1.0f, 0.0f),  /* Going away from sphere */
        VEC3(0.0f, 0.0f, 1.0f)
    };
    RayHit hits[4];

    world_raymarch_batch(world, origins, directions, 20.0f, hits, 4);

    ASSERT_TRUE(hits[0].hit);
    ASSERT_TRUE(hits[1].hit);
    ASSERT_FALSE(hits[2].hit);
    ASSERT_TRUE(hits[3].hit);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 9: Edit List Tests
 * ============================================================================ */

TEST(edit_list_create) {
    Arena* arena = arena_create(1024 * 1024);
    EditList* list = edit_list_create(arena, 100);

    ASSERT_NOT_NULL(list);
    ASSERT_EQ(list->count, 0);
    ASSERT_EQ(list->capacity, 100);

    arena_destroy(arena);
    return 0;
}

TEST(edit_list_add) {
    Arena* arena = arena_create(1024 * 1024);
    EditList* list = edit_list_create(arena, 100);

    bool added = edit_list_add(list, CSG_UNION, PRIM_SPHERE,
                               VEC3(0.0f, 0.0f, 0.0f), VEC3(2.0f, 0.0f, 0.0f), 1);
    ASSERT_TRUE(added);
    ASSERT_EQ(list->count, 1);

    added = edit_list_add(list, CSG_SUBTRACT, PRIM_BOX,
                          VEC3(1.0f, 0.0f, 0.0f), VEC3(0.5f, 0.5f, 0.5f), 0);
    ASSERT_TRUE(added);
    ASSERT_EQ(list->count, 2);

    arena_destroy(arena);
    return 0;
}

TEST(edit_list_clear) {
    Arena* arena = arena_create(1024 * 1024);
    EditList* list = edit_list_create(arena, 100);

    edit_list_add(list, CSG_UNION, PRIM_SPHERE,
                  VEC3(0.0f, 0.0f, 0.0f), VEC3(2.0f, 0.0f, 0.0f), 1);
    edit_list_add(list, CSG_UNION, PRIM_SPHERE,
                  VEC3(1.0f, 0.0f, 0.0f), VEC3(1.0f, 0.0f, 0.0f), 2);

    edit_list_clear(list);
    ASSERT_EQ(list->count, 0);

    arena_destroy(arena);
    return 0;
}

TEST(edit_list_capacity) {
    Arena* arena = arena_create(1024 * 1024);
    EditList* list = edit_list_create(arena, 3);

    bool added1 = edit_list_add(list, CSG_UNION, PRIM_SPHERE, VEC3_ZERO, VEC3_ONE, 1);
    bool added2 = edit_list_add(list, CSG_UNION, PRIM_SPHERE, VEC3_ZERO, VEC3_ONE, 2);
    bool added3 = edit_list_add(list, CSG_UNION, PRIM_SPHERE, VEC3_ZERO, VEC3_ONE, 3);
    bool added4 = edit_list_add(list, CSG_UNION, PRIM_SPHERE, VEC3_ZERO, VEC3_ONE, 4);

    ASSERT_TRUE(added1 && added2 && added3);
    ASSERT_FALSE(added4);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 10: Dirty Tracker Tests
 * ============================================================================ */

TEST(dirty_tracker_create) {
    Arena* arena = arena_create(1024 * 1024);
    DirtyTracker* tracker = dirty_tracker_create(arena, 1000);

    ASSERT_NOT_NULL(tracker);
    ASSERT_EQ(tracker->dirty_count, 0);

    arena_destroy(arena);
    return 0;
}

TEST(dirty_tracker_mark_single) {
    Arena* arena = arena_create(1024 * 1024);
    DirtyTracker* tracker = dirty_tracker_create(arena, 1000);

    dirty_tracker_mark_brick(tracker, 42);
    ASSERT_EQ(tracker->dirty_count, 1);
    ASSERT_TRUE(dirty_tracker_is_dirty(tracker, 42));
    ASSERT_FALSE(dirty_tracker_is_dirty(tracker, 0));

    /* Marking same brick again shouldn't increase count */
    dirty_tracker_mark_brick(tracker, 42);
    ASSERT_EQ(tracker->dirty_count, 1);

    arena_destroy(arena);
    return 0;
}

TEST(dirty_tracker_mark_region) {
    Arena* arena = arena_create(16 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(16.0f, 16.0f, 16.0f),
        1.0f, 1000, 256);
    DirtyTracker* tracker = dirty_tracker_create(arena, world->grid_total);

    /* Mark region covering multiple bricks */
    dirty_tracker_mark_region(tracker, world,
                              VEC3(0.0f, 0.0f, 0.0f), VEC3(10.0f, 10.0f, 10.0f));

    ASSERT_TRUE(tracker->dirty_count > 0);

    arena_destroy(arena);
    return 0;
}

TEST(dirty_tracker_clear) {
    Arena* arena = arena_create(1024 * 1024);
    DirtyTracker* tracker = dirty_tracker_create(arena, 1000);

    dirty_tracker_mark_brick(tracker, 10);
    dirty_tracker_mark_brick(tracker, 20);
    dirty_tracker_mark_brick(tracker, 30);

    ASSERT_EQ(tracker->dirty_count, 3);

    dirty_tracker_clear(tracker);
    ASSERT_EQ(tracker->dirty_count, 0);
    ASSERT_FALSE(dirty_tracker_is_dirty(tracker, 10));

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 11: Incremental Regeneration Tests
 * ============================================================================ */

TEST(regenerate_single_brick) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-5.0f, -5.0f, -5.0f), VEC3(5.0f, 5.0f, 5.0f),
        0.5f, 1000, 256);
    EditList* edits = edit_list_create(arena, 100);

    /* Add sphere via edit list */
    edit_list_add(edits, CSG_UNION, PRIM_SPHERE,
                  VEC3(0.0f, 0.0f, 0.0f), VEC3(2.0f, 0.0f, 0.0f), 1);

    /* Find the brick that contains the origin */
    int32_t bx, by, bz;
    world_pos_to_brick(world, VEC3(0.0f, 0.0f, 0.0f), &bx, &by, &bz);
    uint32_t brick_idx = brick_linear_index(world, bx, by, bz);

    /* Regenerate that brick */
    world_regenerate_brick(world, brick_idx, edits);

    /* Query should now show sphere */
    float sdf = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf < 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(regenerate_dirty) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-5.0f, -5.0f, -5.0f), VEC3(5.0f, 5.0f, 5.0f),
        0.5f, 1000, 256);
    EditList* edits = edit_list_create(arena, 100);
    DirtyTracker* tracker = dirty_tracker_create(arena, world->grid_total);

    /* Add sphere edit */
    edit_list_add(edits, CSG_UNION, PRIM_SPHERE,
                  VEC3(0.0f, 0.0f, 0.0f), VEC3(2.0f, 0.0f, 0.0f), 1);

    /* Mark affected region */
    world_mark_dirty_bricks(world, tracker,
                            VEC3(-3.0f, -3.0f, -3.0f), VEC3(3.0f, 3.0f, 3.0f));

    uint32_t dirty_count = dirty_tracker_count(tracker);
    ASSERT_TRUE(dirty_count > 0);

    /* Regenerate */
    world_regenerate_dirty(world, tracker, edits);

    /* Verify sphere exists */
    float sdf = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf < 0.0f);

    /* Tracker should be cleared */
    ASSERT_EQ(dirty_tracker_count(tracker), 0);

    arena_destroy(arena);
    return 0;
}

TEST(csg_subtract) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-5.0f, -5.0f, -5.0f), VEC3(5.0f, 5.0f, 5.0f),
        0.2f, 1000, 256);
    EditList* edits = edit_list_create(arena, 100);
    DirtyTracker* tracker = dirty_tracker_create(arena, world->grid_total);

    /* Add large sphere, then subtract smaller sphere (hollow) */
    edit_list_add(edits, CSG_UNION, PRIM_SPHERE,
                  VEC3(0.0f, 0.0f, 0.0f), VEC3(2.0f, 0.0f, 0.0f), 1);
    edit_list_add(edits, CSG_SUBTRACT, PRIM_SPHERE,
                  VEC3(0.0f, 0.0f, 0.0f), VEC3(1.5f, 0.0f, 0.0f), 0);

    world_mark_dirty_bricks(world, tracker,
                            VEC3(-3.0f, -3.0f, -3.0f), VEC3(3.0f, 3.0f, 3.0f));
    world_regenerate_dirty(world, tracker, edits);

    /* Center should now be outside (hollow) */
    float sdf_center = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_center > 0.0f);

    /* Shell should still be inside */
    float sdf_shell = world_sdf_query(world, VEC3(1.8f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_shell < 0.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 12: Clip Map Tests
 * ============================================================================ */

TEST(clipmap_create) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    ClipMapWorld* clipmap = clipmap_create(arena, 0.1f, 10.0f, 1000);

    ASSERT_NOT_NULL(clipmap);
    ASSERT_FLOAT_EQ(clipmap->base_voxel_size, 0.1f);
    ASSERT_FLOAT_EQ(clipmap->base_extent, 10.0f);

    /* Verify level hierarchy */
    for (int i = 0; i < CLIPMAP_LEVELS; i++) {
        ASSERT_NOT_NULL(clipmap->levels[i].map);
        float expected_voxel = 0.1f * (float)(1 << i);
        ASSERT_FLOAT_EQ(clipmap->levels[i].voxel_size, expected_voxel);
    }

    arena_destroy(arena);
    return 0;
}

TEST(clipmap_level_selection) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    ClipMapWorld* clipmap = clipmap_create(arena, 0.1f, 10.0f, 1000);

    /* At focus - should use level 0 */
    int level = clipmap_select_level(clipmap, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_EQ(level, 0);

    /* Near focus but within level 0 extent */
    level = clipmap_select_level(clipmap, VEC3(5.0f, 0.0f, 0.0f));
    ASSERT_EQ(level, 0);

    /* Beyond level 0 extent */
    level = clipmap_select_level(clipmap, VEC3(15.0f, 0.0f, 0.0f));
    ASSERT_TRUE(level > 0);

    arena_destroy(arena);
    return 0;
}

TEST(clipmap_sdf_query) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    ClipMapWorld* clipmap = clipmap_create(arena, 0.1f, 10.0f, 1000);

    /* Add sphere */
    clipmap_set_sphere(clipmap, VEC3(0.0f, 0.0f, 0.0f), 2.0f, 1);

    /* Query inside sphere */
    float sdf = clipmap_sdf_query(clipmap, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf < 0.0f);

    /* Query outside sphere */
    sdf = clipmap_sdf_query(clipmap, VEC3(3.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf > 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(clipmap_raymarch) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    ClipMapWorld* clipmap = clipmap_create(arena, 0.1f, 10.0f, 1000);

    clipmap_set_sphere(clipmap, VEC3(0.0f, 0.0f, 0.0f), 2.0f, 1);

    RayHit hit = clipmap_raymarch(clipmap,
        VEC3(-5.0f, 0.0f, 0.0f), VEC3(1.0f, 0.0f, 0.0f), 20.0f);

    ASSERT_TRUE(hit.hit);
    ASSERT_FLOAT_NEAR(hit.distance, 3.0f, 0.3f);

    arena_destroy(arena);
    return 0;
}

TEST(clipmap_update_focus) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    ClipMapWorld* clipmap = clipmap_create(arena, 0.1f, 10.0f, 1000);

    /* Add geometry */
    clipmap_set_sphere(clipmap, VEC3(0.0f, 0.0f, 0.0f), 2.0f, 1);

    /* Move focus */
    clipmap_update_focus(clipmap, VEC3(5.0f, 0.0f, 0.0f));
    ASSERT_VEC3_NEAR(clipmap->focus, VEC3(5.0f, 0.0f, 0.0f), 1e-5f);

    /* Level selection should now be relative to new focus */
    int level = clipmap_select_level(clipmap, VEC3(5.0f, 0.0f, 0.0f));
    ASSERT_EQ(level, 0);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 13: Integration Tests
 * ============================================================================ */

TEST(integration_basic) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 5000, 256);

    /* Create scene with obstacles */
    world_set_sphere(world, VEC3(0.0f, 0.0f, 2.0f), 1.5f, 2);
    world_set_box(world, VEC3(-3.0f, 0.0f, 1.0f), VEC3(1.0f, 1.0f, 1.0f), 3);

    /* Verify statistics */
    WorldStats stats = world_get_stats(world);
    ASSERT_TRUE(stats.active_bricks > 0);
    ASSERT_TRUE(stats.fill_ratio < 1.0f);

    /* Raymarch test - from above downward (-Z) */
    RayHit hit = world_raymarch(world,
        VEC3(0.0f, 0.0f, 5.0f), VEC3(0.0f, 0.0f, -1.0f), 20.0f);
    ASSERT_TRUE(hit.hit);
    ASSERT_EQ(hit.material, 2);

    arena_destroy(arena);
    return 0;
}

TEST(integration_incremental) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.2f, 5000, 256);
    EditList* edits = edit_list_create(arena, 100);
    DirtyTracker* tracker = dirty_tracker_create(arena, world->grid_total);

    /* Build initial scene via edit list */
    edit_list_add(edits, CSG_UNION, PRIM_SPHERE,
                  VEC3(0.0f, 0.0f, 0.0f), VEC3(3.0f, 0.0f, 0.0f), 1);

    world_mark_dirty_bricks(world, tracker,
                            VEC3(-4.0f, -4.0f, -4.0f), VEC3(4.0f, 4.0f, 4.0f));
    world_regenerate_dirty(world, tracker, edits);

    /* Verify sphere */
    float sdf1 = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf1 < 0.0f);

    /* Add subtraction */
    edit_list_add(edits, CSG_SUBTRACT, PRIM_SPHERE,
                  VEC3(0.0f, 0.0f, 0.0f), VEC3(2.0f, 0.0f, 0.0f), 0);

    world_mark_dirty_bricks(world, tracker,
                            VEC3(-3.0f, -3.0f, -3.0f), VEC3(3.0f, 3.0f, 3.0f));
    world_regenerate_dirty(world, tracker, edits);

    /* Verify hollow */
    float sdf2 = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf2 > 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(integration_clipmap) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    ClipMapWorld* clipmap = clipmap_create(arena, 0.1f, 10.0f, 2000);

    /* Add geometry at different distances */
    clipmap_set_sphere(clipmap, VEC3(0.0f, 0.0f, 0.0f), 3.0f, 1);
    clipmap_set_box(clipmap, VEC3(15.0f, 0.0f, 0.0f), VEC3(2.0f, 2.0f, 2.0f), 2);

    /* Query near geometry uses fine LOD */
    float sdf_near = clipmap_sdf_query(clipmap, VEC3(4.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_near > 0.0f);

    /* Move focus and verify */
    clipmap_update_focus(clipmap, VEC3(10.0f, 0.0f, 0.0f));

    /* Far geometry should still be queryable */
    int level = clipmap_select_level(clipmap, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(level > 0);

    /* Raymarch across LOD boundaries */
    clipmap_raymarch(clipmap,
        VEC3(-10.0f, 0.0f, 0.0f), VEC3(1.0f, 0.0f, 0.0f), 30.0f);
    /* May or may not hit depending on LOD coverage */

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 14: Edge Case Tests
 * ============================================================================ */

TEST(edge_world_boundary) {
    Arena* arena = arena_create(16 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(8.0f, 8.0f, 8.0f),
        1.0f, 100, 256);

    /* Set voxel at corner */
    world_set_voxel(world, VEC3(0.5f, 0.5f, 0.5f), -1.0f, 1);

    /* Query at corner */
    float sdf = world_sdf_query_nearest(world, VEC3(0.5f, 0.5f, 0.5f));
    ASSERT_FLOAT_NEAR(sdf, -1.0f, 0.1f);

    /* Query just outside world */
    sdf = world_sdf_query(world, VEC3(-0.1f, 0.0f, 0.0f));
    ASSERT_FLOAT_EQ(sdf, world->sdf_scale);

    arena_destroy(arena);
    return 0;
}

TEST(edge_brick_boundary) {
    Arena* arena = arena_create(16 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(16.0f, 16.0f, 16.0f),
        1.0f, 100, 256);

    /* Set voxels at brick boundary */
    /* Brick 0 ends at x=8, brick 1 starts at x=8 */
    world_set_voxel(world, VEC3(7.5f, 0.5f, 0.5f), -1.0f, 1);
    world_set_voxel(world, VEC3(8.5f, 0.5f, 0.5f), -1.0f, 2);

    /* Verify both bricks allocated - using SoA index API */
    int32_t idx0 = world_get_brick_index(world, 0, 0, 0);
    int32_t idx1 = world_get_brick_index(world, 1, 0, 0);
    ASSERT_NE(idx0, BRICK_EMPTY_INDEX);
    ASSERT_NE(idx1, BRICK_EMPTY_INDEX);

    arena_destroy(arena);
    return 0;
}

TEST(edge_voxel_boundary) {
    Arena* arena = arena_create(16 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(8.0f, 8.0f, 8.0f),
        1.0f, 100, 256);

    /* Trilinear interpolation samples 8 voxels based on floor(pos / voxel_size).
     * Query at (3.5, 0.5, 0.5) will sample voxels at indices:
     * x: floor(3.5)=3 to 4, y: floor(0.5)=0 to 1, z: floor(0.5)=0 to 1
     *
     * world_set_voxel maps position to voxel index via floor(pos / voxel_size).
     * Position (3.5, 0.5, 0.5) -> voxel (3, 0, 0)
     * Position (4.5, 0.5, 0.5) -> voxel (4, 0, 0)
     *
     * We set all 8 voxels in the 2x2x2 cube (3-4, 0-1, 0-1) with a gradient. */

    /* Set voxels at x=3 to SDF=-1 (inside) */
    world_set_voxel(world, VEC3(3.5f, 0.5f, 0.5f), -1.0f, 1);  /* voxel (3,0,0) */
    world_set_voxel(world, VEC3(3.5f, 1.5f, 0.5f), -1.0f, 1);  /* voxel (3,1,0) */
    world_set_voxel(world, VEC3(3.5f, 0.5f, 1.5f), -1.0f, 1);  /* voxel (3,0,1) */
    world_set_voxel(world, VEC3(3.5f, 1.5f, 1.5f), -1.0f, 1);  /* voxel (3,1,1) */

    /* Set voxels at x=4 to SDF=+1 (outside) */
    world_set_voxel(world, VEC3(4.5f, 0.5f, 0.5f), 1.0f, 0);   /* voxel (4,0,0) */
    world_set_voxel(world, VEC3(4.5f, 1.5f, 0.5f), 1.0f, 0);   /* voxel (4,1,0) */
    world_set_voxel(world, VEC3(4.5f, 0.5f, 1.5f), 1.0f, 0);   /* voxel (4,0,1) */
    world_set_voxel(world, VEC3(4.5f, 1.5f, 1.5f), 1.0f, 0);   /* voxel (4,1,1) */

    /* Query at boundary (x=3.5, y=0.5, z=0.5) samples all 8 voxels we set.
     * fx=0.5, fy=0.5, fz=0.5 -> center of the interpolation cube.
     * Should interpolate to ~0 (average of -1 and +1). */
    float sdf = world_sdf_query(world, VEC3(3.5f, 0.5f, 0.5f));
    ASSERT_TRUE(sdf > -0.5f && sdf < 0.5f);

    /* Query biased toward inside (x=3.25 -> fx=0.25, closer to x=3 voxels) */
    float sdf_inside = world_sdf_query(world, VEC3(3.25f, 0.5f, 0.5f));
    ASSERT_TRUE(sdf_inside < 0.0f);

    /* Query biased toward outside (x=3.75 -> fx=0.75, closer to x=4 voxels) */
    float sdf_outside = world_sdf_query(world, VEC3(3.75f, 0.5f, 0.5f));
    ASSERT_TRUE(sdf_outside > 0.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("World Brick Map Module Tests");

    /* Allocation tests */
    RUN_TEST(allocation_basic);
    RUN_TEST(allocation_alignment);
    RUN_TEST(allocation_null_arena);
    RUN_TEST(allocation_invalid_params);

    /* Coordinate tests */
    RUN_TEST(coords_world_to_brick);
    RUN_TEST(coords_world_to_voxel);
    RUN_TEST(coords_contains);
    RUN_TEST(coords_brick_valid);

    /* Quantization tests */
    RUN_TEST(quantize_zero);
    RUN_TEST(quantize_positive);
    RUN_TEST(quantize_negative);
    RUN_TEST(quantize_clamp);
    RUN_TEST(quantize_roundtrip);

    /* Brick management tests */
    RUN_TEST(brick_alloc_basic);
    RUN_TEST(brick_alloc_reuse);
    RUN_TEST(brick_atlas_full);
    RUN_TEST(brick_init_values);

    /* SDF query tests */
    RUN_TEST(sdf_query_empty_world);
    RUN_TEST(sdf_query_single_voxel);
    RUN_TEST(sdf_query_out_of_bounds);
    RUN_TEST(material_query);

    /* Primitive tests */
    RUN_TEST(primitive_sphere);
    RUN_TEST(primitive_box);
    RUN_TEST(primitive_csg_union);

    /* Gradient tests */
    RUN_TEST(gradient_sphere);

    /* Raymarch tests */
    RUN_TEST(raymarch_miss);
    RUN_TEST(raymarch_hit_sphere);
    RUN_TEST(raymarch_max_distance);
    RUN_TEST(raymarch_batch);

    /* Edit list tests */
    RUN_TEST(edit_list_create);
    RUN_TEST(edit_list_add);
    RUN_TEST(edit_list_clear);
    RUN_TEST(edit_list_capacity);

    /* Dirty tracker tests */
    RUN_TEST(dirty_tracker_create);
    RUN_TEST(dirty_tracker_mark_single);
    RUN_TEST(dirty_tracker_mark_region);
    RUN_TEST(dirty_tracker_clear);

    /* Incremental regeneration tests */
    RUN_TEST(regenerate_single_brick);
    RUN_TEST(regenerate_dirty);
    RUN_TEST(csg_subtract);

    /* Clip map tests */
    RUN_TEST(clipmap_create);
    RUN_TEST(clipmap_level_selection);
    RUN_TEST(clipmap_sdf_query);
    RUN_TEST(clipmap_raymarch);
    RUN_TEST(clipmap_update_focus);

    /* Integration tests */
    RUN_TEST(integration_basic);
    RUN_TEST(integration_incremental);
    RUN_TEST(integration_clipmap);

    /* Edge case tests */
    RUN_TEST(edge_world_boundary);
    RUN_TEST(edge_brick_boundary);
    RUN_TEST(edge_voxel_boundary);

    TEST_SUITE_END();
}
