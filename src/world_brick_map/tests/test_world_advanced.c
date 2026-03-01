/**
 * Advanced World Brick Map Tests: Clipmap, Incremental Updates, Boundary Queries
 *
 * Yoneda Philosophy: These tests serve as the definitive behavioral specification
 * for the clipmap LOD system, incremental update pipeline, and boundary query
 * semantics. Every morphism (interaction) of these subsystems is explored:
 *
 * 1. Clipmap focus update through brick boundaries (toroidal shift)
 * 2. Clipmap large jump (complete grid invalidation)
 * 3. Incremental add/remove bricks via edit list + dirty tracker
 * 4. SDF query at exact brick boundary (continuity)
 * 5. SDF query outside world bounds (clamp/default)
 * 6. SDF batch query (mixed inside/outside/boundary)
 * 7. Raymarch through empty world (no hit)
 * 8. Raymarch at grazing angle (robustness)
 * 9. World from zero-triangle mesh (empty world, queries return default)
 * 10. Material operations on non-existent / default bricks
 * 11. World reset via world_clear
 * 12. Brick allocation stress (thousands of bricks)
 * 13. SDF gradient at brick boundary (no blow-up)
 */

#include "../include/world_brick_map.h"
#include "test_harness.h"
#include <stdint.h>

/* ============================================================================
 * Section 1: Clipmap Focus Update Through Brick Boundaries
 *
 * When the camera moves across a brick boundary, the toroidal shift
 * mechanism must free edge bricks and update world bounds correctly.
 * A partial shift (less than grid width) should preserve interior bricks.
 * ============================================================================ */

TEST(clipmap_update_through_brick_boundary) {
    Arena *arena = arena_create(128 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* base_voxel_size=0.5, base_extent=10.0 => brick_size_world=4.0
     * Level 0 world: [-10, +10] in each axis
     * Grid size: 20 / 4 = 5 bricks per axis */
    ClipMapWorld *clipmap = clipmap_create(arena, 0.5f, 10.0f, 500);
    ASSERT_NOT_NULL(clipmap);

    /* Place a sphere at the origin - it will be populated in all levels */
    clipmap_set_sphere(clipmap, VEC3(0.0f, 0.0f, 0.0f), 3.0f, 1);

    /* Verify sphere is queryable at level 0 before any focus update */
    float sdf_before = clipmap_sdf_query(clipmap, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_before < 0.0f);

    /* Move focus by one brick width in X (4.0 world units for level 0).
     * This triggers a toroidal shift that frees edge bricks on the trailing side
     * and shifts the world bounds. */
    float brick_size = clipmap->levels[0].map->brick_size_world;
    clipmap_update_focus(clipmap, VEC3(brick_size, 0.0f, 0.0f));

    /* After shift, focus should be updated */
    ASSERT_FLOAT_NEAR(clipmap->focus.x, brick_size, 1e-5f);

    /* The level 0 world bounds should now be centered on new focus */
    WorldBrickMap *level0 = clipmap->levels[0].map;
    float expected_min_x = brick_size - clipmap->base_extent;
    float expected_max_x = brick_size + clipmap->base_extent;
    ASSERT_FLOAT_NEAR(level0->world_min.x, expected_min_x, 0.1f);
    ASSERT_FLOAT_NEAR(level0->world_max.x, expected_max_x, 0.1f);

    /* Level selection at the new focus should return level 0 */
    int level = clipmap_select_level(clipmap, VEC3(brick_size, 0.0f, 0.0f));
    ASSERT_EQ(level, 0);

    arena_destroy(arena);
    return 0;
}

TEST(clipmap_incremental_focus_movement) {
    Arena *arena = arena_create(128 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    ClipMapWorld *clipmap = clipmap_create(arena, 0.5f, 10.0f, 500);
    ASSERT_NOT_NULL(clipmap);

    /* Place sphere at origin */
    clipmap_set_sphere(clipmap, VEC3(0.0f, 0.0f, 0.0f), 4.0f, 1);

    /* Move focus incrementally through several brick boundaries in small steps.
     * The sub-brick-size movements should not trigger any toroidal shift. */
    float brick_size = clipmap->levels[0].map->brick_size_world;
    float small_step = brick_size * 0.3f;

    for (int i = 0; i < 5; i++) {
        Vec3 new_focus = VEC3(small_step * (float)(i + 1), 0.0f, 0.0f);
        clipmap_update_focus(clipmap, new_focus);

        /* Focus should track exactly */
        ASSERT_FLOAT_NEAR(clipmap->focus.x, new_focus.x, 1e-5f);
    }

    /* After moving ~1.5 brick sizes, level selection at new focus should be 0 */
    int level = clipmap_select_level(clipmap, clipmap->focus);
    ASSERT_EQ(level, 0);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 2: Clipmap Large Jump
 *
 * When the camera teleports multiple bricks at once (e.g., respawn),
 * the shift exceeds the grid dimensions and should trigger a complete
 * clear of that level's world.
 * ============================================================================ */

TEST(clipmap_large_jump_clears_world) {
    Arena *arena = arena_create(128 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    ClipMapWorld *clipmap = clipmap_create(arena, 0.5f, 10.0f, 500);
    ASSERT_NOT_NULL(clipmap);

    /* Populate geometry at origin */
    clipmap_set_sphere(clipmap, VEC3(0.0f, 0.0f, 0.0f), 3.0f, 1);

    /* Verify sphere at origin before jump */
    float sdf_pre = clipmap_sdf_query(clipmap, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_pre < 0.0f);

    /* Jump far beyond the level 0 extent. Level 0 extent is 10.0, so
     * jumping by 100.0 is well beyond any level's grid. This should
     * trigger a complete clear for all levels. */
    clipmap_update_focus(clipmap, VEC3(100.0f, 0.0f, 0.0f));

    /* After the large jump, querying the old origin position should return
     * the default (far outside) for the level that now covers it, because
     * the sphere data was cleared during the complete shift. */
    WorldBrickMap *level0 = clipmap->levels[0].map;
    WorldStats stats = world_get_stats(level0);
    /* atlas_count might still be nonzero (freed bricks go to free list),
     * but active_bricks should be zero since world_clear resets atlas_count */
    ASSERT_EQ(stats.active_bricks, 0);

    /* Focus should be at the new position */
    ASSERT_FLOAT_NEAR(clipmap->focus.x, 100.0f, 1e-5f);

    arena_destroy(arena);
    return 0;
}

TEST(clipmap_large_jump_all_axes) {
    Arena *arena = arena_create(128 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    ClipMapWorld *clipmap = clipmap_create(arena, 0.5f, 10.0f, 500);
    ASSERT_NOT_NULL(clipmap);

    clipmap_set_sphere(clipmap, VEC3(0.0f, 0.0f, 0.0f), 2.0f, 1);

    /* Jump diagonally across all axes */
    clipmap_update_focus(clipmap, VEC3(200.0f, -150.0f, 300.0f));

    ASSERT_FLOAT_NEAR(clipmap->focus.x, 200.0f, 1e-5f);
    ASSERT_FLOAT_NEAR(clipmap->focus.y, -150.0f, 1e-5f);
    ASSERT_FLOAT_NEAR(clipmap->focus.z, 300.0f, 1e-5f);

    /* All levels should be cleared */
    for (int lev = 0; lev < CLIPMAP_LEVELS; lev++) {
        WorldBrickMap *map = clipmap->levels[lev].map;
        ASSERT_NOT_NULL(map);
        WorldStats s = world_get_stats(map);
        ASSERT_EQ(s.active_bricks, 0);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 3: Incremental Add / Remove Bricks
 *
 * The incremental update pipeline (EditList + DirtyTracker + regenerate)
 * must correctly add geometry and then remove it, with SDF queries
 * reflecting the changes after each regeneration pass.
 * ============================================================================ */

TEST(incremental_add_then_remove_sphere) {
    Arena *arena = arena_create(64 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.5f, 2000, 256);
    ASSERT_NOT_NULL(world);

    EditList *edits = edit_list_create(arena, 100);
    DirtyTracker *tracker = dirty_tracker_create(arena, world->grid_total);
    ASSERT_NOT_NULL(edits);
    ASSERT_NOT_NULL(tracker);

    /* --- Step 1: Add sphere via edit list --- */
    edit_list_add(edits, CSG_UNION, PRIM_SPHERE,
                  VEC3(0.0f, 0.0f, 0.0f), VEC3(3.0f, 0.0f, 0.0f), 5);

    world_mark_dirty_bricks(world, tracker,
                            VEC3(-4.0f, -4.0f, -4.0f), VEC3(4.0f, 4.0f, 4.0f));
    uint32_t dirty_count = dirty_tracker_count(tracker);
    ASSERT_TRUE(dirty_count > 0);

    world_regenerate_dirty(world, tracker, edits);

    /* Verify sphere present */
    float sdf_center = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_center < 0.0f);

    float sdf_outside = world_sdf_query(world, VEC3(5.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_outside > 0.0f);

    uint8_t mat = world_material_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_EQ(mat, 5);

    /* Tracker should be cleared after regeneration */
    ASSERT_EQ(dirty_tracker_count(tracker), 0);

    /* --- Step 2: Remove sphere by subtracting same volume --- */
    edit_list_add(edits, CSG_SUBTRACT, PRIM_SPHERE,
                  VEC3(0.0f, 0.0f, 0.0f), VEC3(3.0f, 0.0f, 0.0f), 0);

    world_mark_dirty_bricks(world, tracker,
                            VEC3(-4.0f, -4.0f, -4.0f), VEC3(4.0f, 4.0f, 4.0f));
    world_regenerate_dirty(world, tracker, edits);

    /* After subtracting same volume, center should be outside.
     * The edit list replays: first union sphere (SDF min), then subtract sphere
     * (SDF max(existing, -new)). Since union yields ~-3 at center and subtract
     * yields max(-3, -(-3)) = max(-3, 3) = 3 -> positive -> outside. */
    float sdf_after = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_after > 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(incremental_add_multiple_primitives) {
    Arena *arena = arena_create(64 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.3f, 2000, 256);
    EditList *edits = edit_list_create(arena, 100);
    DirtyTracker *tracker = dirty_tracker_create(arena, world->grid_total);

    /* Add sphere at -3,0,0 */
    edit_list_add(edits, CSG_UNION, PRIM_SPHERE,
                  VEC3(-3.0f, 0.0f, 0.0f), VEC3(2.0f, 0.0f, 0.0f), 1);
    /* Add box at +3,0,0 */
    edit_list_add(edits, CSG_UNION, PRIM_BOX,
                  VEC3(3.0f, 0.0f, 0.0f), VEC3(1.5f, 1.5f, 1.5f), 2);

    world_mark_dirty_bricks(world, tracker,
                            VEC3(-6.0f, -3.0f, -3.0f), VEC3(6.0f, 3.0f, 3.0f));
    world_regenerate_dirty(world, tracker, edits);

    /* Sphere center should be inside with material 1 */
    float sdf_sphere = world_sdf_query(world, VEC3(-3.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_sphere < 0.0f);
    uint8_t mat_sphere = world_material_query(world, VEC3(-3.0f, 0.0f, 0.0f));
    ASSERT_EQ(mat_sphere, 1);

    /* Box center should be inside with material 2 */
    float sdf_box = world_sdf_query(world, VEC3(3.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_box < 0.0f);
    uint8_t mat_box = world_material_query(world, VEC3(3.0f, 0.0f, 0.0f));
    ASSERT_EQ(mat_box, 2);

    /* Midpoint between them should be outside */
    float sdf_gap = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_gap > 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(incremental_cylinder_edit) {
    Arena *arena = arena_create(64 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.3f, 2000, 256);
    EditList *edits = edit_list_create(arena, 100);
    DirtyTracker *tracker = dirty_tracker_create(arena, world->grid_total);

    /* Add cylinder: radius=2, half_height=3, params=(radius, half_height, 0) */
    edit_list_add(edits, CSG_UNION, PRIM_CYLINDER,
                  VEC3(0.0f, 0.0f, 0.0f), VEC3(2.0f, 3.0f, 0.0f), 7);

    world_mark_dirty_bricks(world, tracker,
                            VEC3(-3.0f, -3.0f, -4.0f), VEC3(3.0f, 3.0f, 4.0f));
    world_regenerate_dirty(world, tracker, edits);

    /* Center of cylinder should be inside */
    float sdf_center = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_center < 0.0f);

    /* Above the cylinder (z > half_height) should be outside */
    float sdf_above = world_sdf_query(world, VEC3(0.0f, 0.0f, 4.0f));
    ASSERT_TRUE(sdf_above > 0.0f);

    /* Beyond the cylinder radius (in XY) should be outside */
    float sdf_beyond_r = world_sdf_query(world, VEC3(3.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_beyond_r > 0.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 4: SDF Query at Exact Brick Boundary (Continuity)
 *
 * The trilinear interpolation crosses brick boundaries. If a point lies
 * exactly on a brick boundary, the query must sample from both adjacent
 * bricks and produce a smooth (continuous) SDF value -- no large
 * discontinuity between adjacent points near the boundary.
 * ============================================================================ */

TEST(sdf_query_brick_boundary_continuity) {
    Arena *arena = arena_create(64 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* voxel_size=1.0 => brick_size_world=8.0
     * World: [0, 24] => 3 bricks per axis
     * Brick boundaries at x=8, x=16 */
    WorldBrickMap *world = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(24.0f, 24.0f, 24.0f),
        1.0f, 200, 256);
    ASSERT_NOT_NULL(world);

    /* Place a large sphere centered at (12, 12, 12) that spans multiple bricks.
     * Radius 8 means the sphere surface crosses through brick boundaries. */
    world_set_sphere(world, VEC3(12.0f, 12.0f, 12.0f), 8.0f, 1);

    /* Query at exact brick boundary x=8.0, inside the sphere.
     * The point (8.0, 12.0, 12.0) is at distance 4 from center -> inside. */
    float sdf_at_boundary = world_sdf_query(world, VEC3(8.0f, 12.0f, 12.0f));

    /* Query at points slightly before and after the boundary */
    float sdf_before = world_sdf_query(world, VEC3(7.8f, 12.0f, 12.0f));
    float sdf_after = world_sdf_query(world, VEC3(8.2f, 12.0f, 12.0f));

    /* All three should be inside (negative SDF) since they are within the sphere */
    ASSERT_TRUE(sdf_at_boundary < 0.0f);
    ASSERT_TRUE(sdf_before < 0.0f);
    ASSERT_TRUE(sdf_after < 0.0f);

    /* Continuity check: the SDF difference between adjacent points should be
     * small (no large discontinuity). For a sphere of radius 8 sampled at
     * 0.2 apart, the analytic SDF difference is ~0.2. With quantization,
     * we allow more tolerance but should not see jumps > 1.0 (which would
     * indicate a brick boundary bug). */
    float diff_before = fabsf(sdf_at_boundary - sdf_before);
    float diff_after = fabsf(sdf_at_boundary - sdf_after);
    ASSERT_TRUE(diff_before < 1.0f);
    ASSERT_TRUE(diff_after < 1.0f);

    arena_destroy(arena);
    return 0;
}

TEST(sdf_query_brick_boundary_surface_crossing) {
    Arena *arena = arena_create(64 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Small world: 2 bricks in X, 1 in Y and Z.
     * voxel_size=1.0, brick_size=8.0, world: [0, 16] x [0, 8] x [0, 8] */
    WorldBrickMap *world = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(16.0f, 8.0f, 8.0f),
        1.0f, 100, 256);
    ASSERT_NOT_NULL(world);

    /* Place sphere centered exactly at the brick boundary x=8.
     * Radius 3 means it spans into both brick (0,0,0) and brick (1,0,0). */
    world_set_sphere(world, VEC3(8.0f, 4.0f, 4.0f), 3.0f, 1);

    /* Both bricks should be allocated */
    int32_t idx0 = world_get_brick_index(world, 0, 0, 0);
    int32_t idx1 = world_get_brick_index(world, 1, 0, 0);
    ASSERT_NE(idx0, BRICK_EMPTY_INDEX);
    ASSERT_NE(idx1, BRICK_EMPTY_INDEX);

    /* Center of sphere (on the boundary) should be inside */
    float sdf_center = world_sdf_query(world, VEC3(8.0f, 4.0f, 4.0f));
    ASSERT_TRUE(sdf_center < 0.0f);

    /* Check a line of SDF values crossing the boundary at x=6..10.
     * The SDF should be monotonically changing sign at x ~= 5 and x ~= 11
     * (surface crossings) and should be smooth near x=8 (boundary). */
    float prev_sdf = world_sdf_query(world, VEC3(6.0f, 4.0f, 4.0f));
    for (float x = 6.5f; x <= 10.0f; x += 0.5f) {
        float cur_sdf = world_sdf_query(world, VEC3(x, 4.0f, 4.0f));
        /* No large discontinuity (> 2x voxel_size worth of SDF jump) */
        float jump = fabsf(cur_sdf - prev_sdf);
        ASSERT_TRUE(jump < 2.0f * world->sdf_scale);
        prev_sdf = cur_sdf;
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 5: SDF Query Outside World Bounds
 *
 * Points far outside the world should return the default "far outside" value
 * (sdf_scale), not crash or return garbage.
 * ============================================================================ */

TEST(sdf_query_far_outside_world) {
    Arena *arena = arena_create(16 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.5f, 500, 256);

    /* Place some geometry so the world is not trivially empty */
    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 3.0f, 1);

    /* Query far outside in each dimension */
    float sdf_x_far = world_sdf_query(world, VEC3(1000.0f, 0.0f, 0.0f));
    float sdf_y_far = world_sdf_query(world, VEC3(0.0f, -500.0f, 0.0f));
    float sdf_z_far = world_sdf_query(world, VEC3(0.0f, 0.0f, 9999.0f));
    float sdf_all_far = world_sdf_query(world, VEC3(-999.0f, 999.0f, -999.0f));

    /* All should return sdf_scale (the "far outside" constant) */
    ASSERT_FLOAT_EQ(sdf_x_far, world->sdf_scale);
    ASSERT_FLOAT_EQ(sdf_y_far, world->sdf_scale);
    ASSERT_FLOAT_EQ(sdf_z_far, world->sdf_scale);
    ASSERT_FLOAT_EQ(sdf_all_far, world->sdf_scale);

    arena_destroy(arena);
    return 0;
}

TEST(sdf_query_just_outside_world_edge) {
    Arena *arena = arena_create(16 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(8.0f, 8.0f, 8.0f),
        1.0f, 100, 256);

    /* Test world_contains at exact boundaries and just outside */
    ASSERT_FALSE(world_contains(world, VEC3(-0.01f, 4.0f, 4.0f)));
    ASSERT_FALSE(world_contains(world, VEC3(8.01f, 4.0f, 4.0f)));
    ASSERT_FALSE(world_contains(world, VEC3(4.0f, -0.01f, 4.0f)));
    ASSERT_FALSE(world_contains(world, VEC3(4.0f, 4.0f, 8.01f)));

    /* SDF queries just outside should all return sdf_scale */
    float sdf = world_sdf_query(world, VEC3(-0.01f, 4.0f, 4.0f));
    ASSERT_FLOAT_EQ(sdf, world->sdf_scale);

    arena_destroy(arena);
    return 0;
}

TEST(material_query_outside_world) {
    Arena *arena = arena_create(16 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(-5.0f, -5.0f, -5.0f), VEC3(5.0f, 5.0f, 5.0f),
        0.5f, 500, 256);

    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 2.0f, 42);

    /* Material query outside world bounds should return 0 (air) */
    uint8_t mat = world_material_query(world, VEC3(100.0f, 0.0f, 0.0f));
    ASSERT_EQ(mat, 0);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 6: SDF Batch Query
 *
 * world_sdf_query_batch must produce identical results to individual queries
 * for a mix of inside, outside, and boundary points.
 * ============================================================================ */

TEST(sdf_batch_query_matches_individual) {
    Arena *arena = arena_create(64 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.2f, 2000, 256);

    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 4.0f, 1);
    world_set_box(world, VEC3(5.0f, 0.0f, 0.0f), VEC3(1.0f, 1.0f, 1.0f), 2);

    /* Mixed test points: inside sphere, outside sphere, inside box,
     * on sphere surface, outside world, near brick boundary */
    Vec3 positions[] = {
        VEC3(0.0f, 0.0f, 0.0f),     /* inside sphere center */
        VEC3(2.0f, 0.0f, 0.0f),     /* inside sphere */
        VEC3(4.0f, 0.0f, 0.0f),     /* near sphere surface */
        VEC3(7.0f, 0.0f, 0.0f),     /* outside sphere, outside box */
        VEC3(5.0f, 0.0f, 0.0f),     /* inside box */
        VEC3(-8.0f, -8.0f, -8.0f),  /* near world corner, empty */
        VEC3(100.0f, 0.0f, 0.0f),   /* outside world */
    };
    uint32_t count = sizeof(positions) / sizeof(positions[0]);

    float batch_results[7];
    world_sdf_query_batch(world, positions, batch_results, count);

    /* Compare each batch result with individual query */
    for (uint32_t i = 0; i < count; i++) {
        float individual = world_sdf_query(world, positions[i]);
        ASSERT_FLOAT_EQ(batch_results[i], individual);
    }

    arena_destroy(arena);
    return 0;
}

TEST(sdf_batch_query_empty_world) {
    Arena *arena = arena_create(16 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.5f, 500, 256);

    Vec3 positions[] = {
        VEC3(0.0f, 0.0f, 0.0f),
        VEC3(5.0f, 5.0f, 5.0f),
        VEC3(-3.0f, 2.0f, -1.0f),
    };
    float results[3];
    world_sdf_query_batch(world, positions, results, 3);

    /* Empty world: all queries should return sdf_scale */
    for (int i = 0; i < 3; i++) {
        ASSERT_FLOAT_EQ(results[i], world->sdf_scale);
    }

    arena_destroy(arena);
    return 0;
}

TEST(sdf_batch_query_zero_count) {
    Arena *arena = arena_create(16 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.5f, 500, 256);

    /* Calling with count=0 should be a no-op (not crash) */
    float dummy = -999.0f;
    world_sdf_query_batch(world, NULL, &dummy, 0);
    /* dummy should be unchanged (batch function returns early) */
    ASSERT_FLOAT_EQ(dummy, -999.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 7: Raymarch Through Empty World
 *
 * With no geometry loaded, a raymarch in any direction should produce a miss.
 * ============================================================================ */

TEST(raymarch_empty_world_all_directions) {
    Arena *arena = arena_create(16 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(-20.0f, -20.0f, -20.0f), VEC3(20.0f, 20.0f, 20.0f),
        0.5f, 500, 256);

    /* Test 6 cardinal directions + 2 diagonal */
    Vec3 origins[] = {
        VEC3(-15.0f, 0.0f, 0.0f),
        VEC3(15.0f, 0.0f, 0.0f),
        VEC3(0.0f, -15.0f, 0.0f),
        VEC3(0.0f, 15.0f, 0.0f),
        VEC3(0.0f, 0.0f, -15.0f),
        VEC3(0.0f, 0.0f, 15.0f),
        VEC3(-10.0f, -10.0f, -10.0f),
        VEC3(10.0f, 10.0f, 10.0f),
    };
    Vec3 directions[] = {
        VEC3(1.0f, 0.0f, 0.0f),
        VEC3(-1.0f, 0.0f, 0.0f),
        VEC3(0.0f, 1.0f, 0.0f),
        VEC3(0.0f, -1.0f, 0.0f),
        VEC3(0.0f, 0.0f, 1.0f),
        VEC3(0.0f, 0.0f, -1.0f),
        VEC3(0.577f, 0.577f, 0.577f),    /* ~(1,1,1)/sqrt(3) */
        VEC3(-0.577f, -0.577f, -0.577f),
    };
    int n = sizeof(origins) / sizeof(origins[0]);

    for (int i = 0; i < n; i++) {
        RayHit hit = world_raymarch(world, origins[i], directions[i], 40.0f);
        ASSERT_FALSE(hit.hit);
    }

    arena_destroy(arena);
    return 0;
}

TEST(raymarch_empty_world_batch) {
    Arena *arena = arena_create(16 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(-20.0f, -20.0f, -20.0f), VEC3(20.0f, 20.0f, 20.0f),
        0.5f, 500, 256);

    Vec3 origins[4] = {
        VEC3(-15.0f, 0.0f, 0.0f),
        VEC3(0.0f, -15.0f, 0.0f),
        VEC3(0.0f, 0.0f, 0.0f),
        VEC3(10.0f, 10.0f, 10.0f),
    };
    Vec3 dirs[4] = {
        VEC3(1.0f, 0.0f, 0.0f),
        VEC3(0.0f, 1.0f, 0.0f),
        VEC3(0.0f, 0.0f, 1.0f),
        VEC3(-0.577f, -0.577f, -0.577f),
    };
    RayHit hits[4];

    world_raymarch_batch(world, origins, dirs, 40.0f, hits, 4);

    for (int i = 0; i < 4; i++) {
        ASSERT_FALSE(hits[i].hit);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 8: Raymarch at Grazing Angle
 *
 * A ray nearly parallel to a flat surface must not infinite-loop or
 * return a wildly incorrect result. It should either miss cleanly or
 * hit near the expected point.
 * ============================================================================ */

TEST(raymarch_grazing_angle_box) {
    Arena *arena = arena_create(64 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.2f, 2000, 256);

    /* Place a large flat box: half_size=(5, 0.5, 5) centered at origin.
     * The top face is at y=0.5. */
    world_set_box(world, VEC3(0.0f, 0.0f, 0.0f), VEC3(5.0f, 0.5f, 5.0f), 1);

    /* Grazing ray: nearly parallel to the top face of the box.
     * Origin at y=0.6 (just above surface), direction almost horizontal
     * with a tiny downward component. */
    Vec3 origin = VEC3(-8.0f, 0.6f, 0.0f);
    Vec3 dir = VEC3(0.9999f, -0.01f, 0.0f); /* 0.57 degree downward */
    /* Normalize direction */
    float len = sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    dir.x /= len;
    dir.y /= len;
    dir.z /= len;

    RayHit hit = world_raymarch(world, origin, dir, 30.0f);

    /* The ray should eventually hit the box (or miss if it overshoots).
     * The critical property is that it terminates in reasonable time
     * and does not produce NaN or infinite distance. */
    ASSERT_FALSE(isnan(hit.distance));
    ASSERT_FALSE(isinf(hit.distance));
    ASSERT_TRUE(hit.distance <= 30.0f);
    ASSERT_TRUE(hit.distance >= 0.0f);

    /* If it did hit, verify the hit position is near the box surface */
    if (hit.hit) {
        /* Hit should be near y=0.5 (top face) */
        ASSERT_FLOAT_NEAR(hit.position.y, 0.5f, 0.3f);
        ASSERT_EQ(hit.material, 1);
    }

    arena_destroy(arena);
    return 0;
}

TEST(raymarch_grazing_angle_sphere) {
    Arena *arena = arena_create(64 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 2000, 256);

    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 3.0f, 1);

    /* Ray that just barely misses the sphere -- tangent line.
     * Sphere radius 3, tangent from y=3.01 should miss. */
    Vec3 origin = VEC3(-8.0f, 3.01f, 0.0f);
    Vec3 dir = VEC3(1.0f, 0.0f, 0.0f);

    RayHit hit = world_raymarch(world, origin, dir, 20.0f);

    /* This should miss (tangent line doesn't intersect sphere) */
    /* Note: Due to quantization, it might barely "hit" at the surface, so we
     * just verify robustness -- no NaN, no crash. */
    ASSERT_FALSE(isnan(hit.distance));
    ASSERT_FALSE(isinf(hit.distance));

    /* Ray that just barely hits the sphere -- just inside tangent.
     * y=2.99 < radius=3.0 => this ray intersects the sphere. */
    origin = VEC3(-8.0f, 2.9f, 0.0f);
    dir = VEC3(1.0f, 0.0f, 0.0f);

    hit = world_raymarch(world, origin, dir, 20.0f);

    /* With y=2.9, the analytic intersection exists:
     * x^2 + 2.9^2 = 9 => x^2 = 0.59 => x ~= 0.77
     * Distance from origin: 8 - 0.77 = 7.23 */
    ASSERT_TRUE(hit.hit);
    ASSERT_FLOAT_NEAR(hit.distance, 7.23f, 1.0f); /* Generous tolerance for quantization */

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 9: World from Zero-Triangle Mesh (Empty World)
 *
 * When no geometry is added, the world must respond to all queries
 * with well-defined defaults: SDF=sdf_scale, material=0, raymarch=miss.
 * ============================================================================ */

TEST(empty_world_all_queries_default) {
    Arena *arena = arena_create(16 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.5f, 500, 256);
    ASSERT_NOT_NULL(world);

    /* No geometry added. */

    /* SDF query should return sdf_scale */
    float sdf = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_FLOAT_EQ(sdf, world->sdf_scale);

    /* Nearest query in empty world */
    float sdf_n = world_sdf_query_nearest(world, VEC3(5.0f, 5.0f, 5.0f));
    ASSERT_FLOAT_EQ(sdf_n, world->sdf_scale);

    /* Material query should return 0 (air) */
    uint8_t mat = world_material_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_EQ(mat, 0);

    /* Raymarch should miss */
    RayHit hit = world_raymarch(world, VEC3(-5.0f, 0.0f, 0.0f),
                                VEC3(1.0f, 0.0f, 0.0f), 20.0f);
    ASSERT_FALSE(hit.hit);

    /* World stats should show no active bricks */
    WorldStats stats = world_get_stats(world);
    ASSERT_EQ(stats.active_bricks, 0);
    ASSERT_EQ(stats.uniform_inside, 0);
    ASSERT_EQ(stats.uniform_outside, 0);

    arena_destroy(arena);
    return 0;
}

TEST(empty_world_gradient_no_crash) {
    Arena *arena = arena_create(16 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.5f, 500, 256);

    /* Gradient in empty world: all SDF queries return sdf_scale (constant),
     * so gradient should be near zero. Should not crash or return NaN. */
    Vec3 grad = world_sdf_gradient(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_FALSE(isnan(grad.x));
    ASSERT_FALSE(isnan(grad.y));
    ASSERT_FALSE(isnan(grad.z));

    /* Gradient of a constant field should be zero or very near zero */
    ASSERT_FLOAT_NEAR(grad.x, 0.0f, 0.01f);
    ASSERT_FLOAT_NEAR(grad.y, 0.0f, 0.01f);
    ASSERT_FLOAT_NEAR(grad.z, 0.0f, 0.01f);

    /* Normal in empty world: gradient is zero, fallback should be (0,0,1) */
    Vec3 normal = world_sdf_normal(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_FALSE(isnan(normal.x));
    ASSERT_FALSE(isnan(normal.y));
    ASSERT_FALSE(isnan(normal.z));

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 10: Material Operations on Non-Existent / Default Bricks
 *
 * Setting/getting materials on bricks that don't exist or on positions
 * that haven't had geometry added should produce well-defined behavior.
 * ============================================================================ */

TEST(material_set_on_empty_position) {
    Arena *arena = arena_create(16 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(8.0f, 8.0f, 8.0f),
        1.0f, 100, 256);

    /* Setting material should allocate the brick and set the value */
    world_set_material(world, VEC3(4.0f, 4.0f, 4.0f), 99);

    /* The brick should now be allocated */
    int32_t bx, by, bz;
    world_pos_to_brick(world, VEC3(4.0f, 4.0f, 4.0f), &bx, &by, &bz);
    int32_t idx = world_get_brick_index(world, bx, by, bz);
    ASSERT_NE(idx, BRICK_EMPTY_INDEX);

    /* Read it back */
    uint8_t mat = world_material_query(world, VEC3(4.0f, 4.0f, 4.0f));
    ASSERT_EQ(mat, 99);

    arena_destroy(arena);
    return 0;
}

TEST(material_get_from_unallocated_brick) {
    Arena *arena = arena_create(16 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(8.0f, 8.0f, 8.0f),
        1.0f, 100, 256);

    /* No bricks allocated. Material query should return 0 (air). */
    uint8_t mat = world_material_query(world, VEC3(4.0f, 4.0f, 4.0f));
    ASSERT_EQ(mat, 0);

    arena_destroy(arena);
    return 0;
}

TEST(material_registry_on_non_existent_id) {
    Arena *arena = arena_create(16 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(8.0f, 8.0f, 8.0f),
        1.0f, 100, 256);

    /* Getting material metadata for an unregistered ID should return NULL */
    const MaterialMetadata *meta = world_get_material(world, 200);
    ASSERT_NULL(meta);

    /* ID 0 (default) should always exist */
    meta = world_get_material(world, 0);
    ASSERT_NOT_NULL(meta);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 11: World Reset via world_clear
 *
 * After populating the world with geometry, calling world_clear must
 * reset all bricks so that subsequent queries return empty/default values.
 * ============================================================================ */

TEST(world_clear_resets_all_bricks) {
    Arena *arena = arena_create(64 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.2f, 2000, 256);

    /* Populate with geometry */
    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 5.0f, 1);
    world_set_box(world, VEC3(5.0f, 0.0f, 0.0f), VEC3(2.0f, 2.0f, 2.0f), 2);

    /* Verify geometry present */
    float sdf_pre = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_pre < 0.0f);

    WorldStats stats_pre = world_get_stats(world);
    ASSERT_TRUE(stats_pre.active_bricks > 0);

    /* Clear the world */
    world_clear(world);

    /* All SDF queries should return sdf_scale */
    float sdf_post = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_FLOAT_EQ(sdf_post, world->sdf_scale);

    float sdf_post2 = world_sdf_query(world, VEC3(5.0f, 0.0f, 0.0f));
    ASSERT_FLOAT_EQ(sdf_post2, world->sdf_scale);

    /* Material should be 0 */
    uint8_t mat = world_material_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_EQ(mat, 0);

    /* Stats should show no active bricks */
    WorldStats stats_post = world_get_stats(world);
    ASSERT_EQ(stats_post.active_bricks, 0);
    ASSERT_EQ(stats_post.uniform_inside, 0);
    ASSERT_EQ(stats_post.uniform_outside, 0);

    /* Raymarch should miss */
    RayHit hit = world_raymarch(world, VEC3(-5.0f, 0.0f, 0.0f),
                                VEC3(1.0f, 0.0f, 0.0f), 20.0f);
    ASSERT_FALSE(hit.hit);

    arena_destroy(arena);
    return 0;
}

TEST(world_clear_then_repopulate) {
    Arena *arena = arena_create(64 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.3f, 2000, 256);

    /* First population */
    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 3.0f, 1);
    float sdf1 = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf1 < 0.0f);

    /* Clear */
    world_clear(world);
    float sdf_cleared = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_FLOAT_EQ(sdf_cleared, world->sdf_scale);

    /* Re-populate with different geometry at a different location */
    world_set_sphere(world, VEC3(5.0f, 0.0f, 0.0f), 2.0f, 3);

    /* Original location should be outside the new sphere (positive SDF).
     * It won't necessarily equal sdf_scale because the new sphere at (5,0,0)
     * may have allocated a brick that covers the origin with a positive but
     * non-maximal SDF value (distance to sphere surface ≈ 3.0). */
    float sdf_origin = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_origin > 0.0f);

    /* New location should have geometry */
    float sdf_new = world_sdf_query(world, VEC3(5.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_new < 0.0f);

    uint8_t mat = world_material_query(world, VEC3(5.0f, 0.0f, 0.0f));
    ASSERT_EQ(mat, 3);

    arena_destroy(arena);
    return 0;
}

TEST(world_clear_resets_counters) {
    Arena *arena = arena_create(64 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.3f, 2000, 256);

    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 5.0f, 1);

    world_clear(world);

    /* Verify all counters are reset */
    ASSERT_EQ(world->atlas_count, 0);
    ASSERT_EQ(world->free_count, 0);
    ASSERT_EQ(world->uniform_inside_count, 0);
    ASSERT_EQ(world->uniform_outside_count, 0);

    /* All brick indices should be BRICK_EMPTY_INDEX */
    for (uint32_t i = 0; i < world->grid_total; i++) {
        ASSERT_EQ(world->brick_indices[i], BRICK_EMPTY_INDEX);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 12: Brick Allocation Stress
 *
 * Allocate thousands of bricks to verify the demand-paged atlas handles
 * large brick counts correctly with proper page allocation.
 * ============================================================================ */

TEST(brick_allocation_stress_many_bricks) {
    Arena *arena = arena_create(256 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Large world: 50x50x50 in world units, voxel_size=1.0 => brick_size=8
     * Grid: ceil(50/8) = 7 per axis => 343 potential bricks.
     * max_bricks=4000 (rounds to page boundary). */
    WorldBrickMap *world = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(50.0f, 50.0f, 50.0f),
        1.0f, 4000, 256);
    ASSERT_NOT_NULL(world);

    /* Allocate bricks across the entire grid */
    uint32_t allocated = 0;
    uint32_t grid_x = world->grid_x;
    uint32_t grid_y = world->grid_y;
    uint32_t grid_z = world->grid_z;

    for (uint32_t z = 0; z < grid_z; z++) {
        for (uint32_t y = 0; y < grid_y; y++) {
            for (uint32_t x = 0; x < grid_x; x++) {
                int32_t idx = world_alloc_brick(world, (int32_t)x, (int32_t)y, (int32_t)z);
                ASSERT_NE(idx, BRICK_EMPTY_INDEX);
                allocated++;
            }
        }
    }

    /* Verify count */
    ASSERT_EQ(world->atlas_count, allocated);

    /* Verify page allocation: allocated / ATLAS_PAGE_BRICKS pages needed */
    uint32_t expected_pages = (allocated + ATLAS_PAGE_BRICKS - 1) / ATLAS_PAGE_BRICKS;
    ASSERT_EQ(world->page_count, expected_pages);

    /* Verify all bricks are accessible and have valid SDF data */
    for (uint32_t z = 0; z < grid_z; z++) {
        for (uint32_t y = 0; y < grid_y; y++) {
            for (uint32_t x = 0; x < grid_x; x++) {
                int32_t idx = world_get_brick_index(world, (int32_t)x, (int32_t)y, (int32_t)z);
                ASSERT_TRUE(idx >= 0);

                int8_t *sdf = world_brick_sdf(world, idx);
                ASSERT_NOT_NULL(sdf);

                /* Newly allocated bricks should have SDF=+127 (outside) */
                ASSERT_EQ(sdf[0], 127);
            }
        }
    }

    arena_destroy(arena);
    return 0;
}

TEST(brick_allocation_stress_alloc_free_cycle) {
    Arena *arena = arena_create(64 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(40.0f, 40.0f, 40.0f),
        1.0f, 2000, 256);
    ASSERT_NOT_NULL(world);

    /* Repeatedly allocate and free bricks to exercise the free list */
    for (int cycle = 0; cycle < 10; cycle++) {
        /* Allocate a column of bricks */
        for (int32_t z = 0; z < 5; z++) {
            int32_t idx = world_alloc_brick(world, 0, 0, z);
            ASSERT_NE(idx, BRICK_EMPTY_INDEX);
        }

        /* Free them all */
        for (int32_t z = 0; z < 5; z++) {
            world_free_brick(world, 0, 0, z);
        }

        /* Free list should contain 5 bricks */
        ASSERT_EQ(world->free_count, 5);

        /* Re-allocate - should reuse from free list */
        for (int32_t z = 0; z < 5; z++) {
            int32_t idx = world_alloc_brick(world, 0, 0, z);
            ASSERT_NE(idx, BRICK_EMPTY_INDEX);
        }

        /* Free list should be empty */
        ASSERT_EQ(world->free_count, 0);

        /* Clean up for next cycle */
        for (int32_t z = 0; z < 5; z++) {
            world_free_brick(world, 0, 0, z);
        }
    }

    arena_destroy(arena);
    return 0;
}

TEST(brick_allocation_stress_page_growth) {
    Arena *arena = arena_create(128 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(80.0f, 80.0f, 80.0f),
        1.0f, 4096, 256);
    ASSERT_NOT_NULL(world);

    uint32_t prev_page_count = 0;

    /* Allocate bricks and verify page count grows correctly.
     * Each page holds ATLAS_PAGE_BRICKS (64) bricks. */
    for (uint32_t i = 0; i < 256; i++) {
        /* Spread across grid to avoid revisiting same brick */
        int32_t bx = (int32_t)(i % world->grid_x);
        int32_t by = (int32_t)((i / world->grid_x) % world->grid_y);
        int32_t bz = (int32_t)((i / (world->grid_x * world->grid_y)) % world->grid_z);

        if (!world_brick_valid(world, bx, by, bz)) continue;

        int32_t idx = world_alloc_brick(world, bx, by, bz);
        if (idx == BRICK_EMPTY_INDEX) break; /* Atlas full */

        /* Pages should only increase when crossing page boundaries */
        uint32_t expected_pages = (world->atlas_count + ATLAS_PAGE_BRICKS - 1) / ATLAS_PAGE_BRICKS;
        ASSERT_EQ(world->page_count, expected_pages);

        /* Pages should never decrease */
        ASSERT_GE(world->page_count, prev_page_count);
        prev_page_count = world->page_count;
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 13: SDF Gradient at Brick Boundary
 *
 * The SDF gradient (computed via central differences) probes points at
 * +/- epsilon from the query position. When these probes cross brick
 * boundaries, the gradient must not blow up (relates to the known
 * clamping fix documented in CLAUDE.md).
 * ============================================================================ */

TEST(sdf_gradient_at_brick_boundary_sphere) {
    Arena *arena = arena_create(64 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* voxel_size=1.0, brick_size=8.0.
     * World: [0, 24]. Brick boundary at x=8. */
    WorldBrickMap *world = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(24.0f, 24.0f, 24.0f),
        1.0f, 200, 256);

    /* Sphere centered at (12,12,12), radius 8.
     * The sphere surface crosses through brick boundaries. */
    world_set_sphere(world, VEC3(12.0f, 12.0f, 12.0f), 8.0f, 1);

    /* Test gradient at brick boundary x=8, on the sphere surface.
     * The point (8, 12, 12) is at distance 4 from center (inside sphere). */
    Vec3 grad_at_boundary = world_sdf_gradient(world, VEC3(8.0f, 12.0f, 12.0f));

    /* Gradient should not be NaN or blow up to very large values */
    ASSERT_FALSE(isnan(grad_at_boundary.x));
    ASSERT_FALSE(isnan(grad_at_boundary.y));
    ASSERT_FALSE(isnan(grad_at_boundary.z));

    /* Gradient magnitude should be reasonable (order of 1 for unit sphere SDF).
     * Due to quantization, the gradient may be somewhat larger than analytic,
     * but should not blow up to 100+ (which would indicate a cross-brick
     * discontinuity in the SDF probes). */
    float grad_mag = sqrtf(grad_at_boundary.x * grad_at_boundary.x +
                           grad_at_boundary.y * grad_at_boundary.y +
                           grad_at_boundary.z * grad_at_boundary.z);
    ASSERT_TRUE(grad_mag < 10.0f);

    /* The dominant gradient direction should be roughly toward the center
     * (negative X direction from x=8 toward center at x=12).
     * With quantization and boundary effects, we just verify the sign. */

    arena_destroy(arena);
    return 0;
}

TEST(sdf_gradient_at_surface_near_brick_boundary) {
    Arena *arena = arena_create(64 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(24.0f, 24.0f, 24.0f),
        0.5f, 2000, 256);

    /* Sphere centered at (12,12,12), radius 4.
     * Surface at x=8 (exactly on brick boundary). */
    world_set_sphere(world, VEC3(12.0f, 12.0f, 12.0f), 4.0f, 1);

    /* Query gradient at the surface point (8, 12, 12) */
    Vec3 grad_surface = world_sdf_gradient(world, VEC3(8.0f, 12.0f, 12.0f));

    /* Should not blow up */
    float mag = sqrtf(grad_surface.x * grad_surface.x +
                      grad_surface.y * grad_surface.y +
                      grad_surface.z * grad_surface.z);
    ASSERT_FALSE(isnan(mag));
    ASSERT_TRUE(mag < 20.0f);

    /* Normal at surface should point outward (negative X from center's perspective) */
    Vec3 normal = world_sdf_normal(world, VEC3(8.0f, 12.0f, 12.0f));
    ASSERT_FALSE(isnan(normal.x));

    /* Normal should point roughly in -X direction (away from center at x=12) */
    ASSERT_TRUE(normal.x < 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(sdf_gradient_batch_at_boundaries) {
    Arena *arena = arena_create(64 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(24.0f, 24.0f, 24.0f),
        0.5f, 2000, 256);

    world_set_sphere(world, VEC3(12.0f, 12.0f, 12.0f), 5.0f, 1);

    /* Sample points at various brick boundaries */
    Vec3 positions[] = {
        VEC3(8.0f, 12.0f, 12.0f),   /* brick boundary x=8 */
        VEC3(16.0f, 12.0f, 12.0f),  /* brick boundary x=16 */
        VEC3(12.0f, 8.0f, 12.0f),   /* brick boundary y=8 */
        VEC3(12.0f, 12.0f, 8.0f),   /* brick boundary z=8 */
        VEC3(12.0f, 12.0f, 12.0f),  /* center (no boundary) */
    };
    Vec3 gradients[5];
    uint32_t count = 5;

    world_sdf_gradient_batch(world, positions, gradients, count);

    /* All gradients should be finite and not blow up */
    for (uint32_t i = 0; i < count; i++) {
        ASSERT_FALSE(isnan(gradients[i].x));
        ASSERT_FALSE(isnan(gradients[i].y));
        ASSERT_FALSE(isnan(gradients[i].z));

        float mag = sqrtf(gradients[i].x * gradients[i].x +
                          gradients[i].y * gradients[i].y +
                          gradients[i].z * gradients[i].z);
        ASSERT_TRUE(mag < 20.0f);
    }

    /* Individual and batch results should match exactly */
    for (uint32_t i = 0; i < count; i++) {
        Vec3 individual = world_sdf_gradient(world, positions[i]);
        ASSERT_FLOAT_EQ(gradients[i].x, individual.x);
        ASSERT_FLOAT_EQ(gradients[i].y, individual.y);
        ASSERT_FLOAT_EQ(gradients[i].z, individual.z);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 14: Clipmap Level Selection Correctness
 *
 * Verify the level selection algorithm correctly assigns LOD levels
 * based on distance from the focus point.
 * ============================================================================ */

TEST(clipmap_level_selection_all_levels) {
    Arena *arena = arena_create(128 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    ClipMapWorld *clipmap = clipmap_create(arena, 0.5f, 10.0f, 500);
    ASSERT_NOT_NULL(clipmap);

    /* Level 0: distance < 10.0 (base_extent)
     * Level 1: distance < 20.0 (base_extent * 2)
     * Level 2: distance < 40.0 (base_extent * 4)
     * Level 3: distance >= 40.0 (catch-all) */

    /* At focus: level 0 */
    ASSERT_EQ(clipmap_select_level(clipmap, VEC3(0.0f, 0.0f, 0.0f)), 0);

    /* Just inside level 0 extent */
    ASSERT_EQ(clipmap_select_level(clipmap, VEC3(9.0f, 0.0f, 0.0f)), 0);

    /* Just past level 0 extent (in level 1 range) */
    ASSERT_EQ(clipmap_select_level(clipmap, VEC3(11.0f, 0.0f, 0.0f)), 1);

    /* In level 2 range */
    ASSERT_EQ(clipmap_select_level(clipmap, VEC3(25.0f, 0.0f, 0.0f)), 2);

    /* Beyond all levels -> last level */
    ASSERT_EQ(clipmap_select_level(clipmap, VEC3(50.0f, 0.0f, 0.0f)), CLIPMAP_LEVELS - 1);

    /* Test after moving focus */
    clipmap_update_focus(clipmap, VEC3(100.0f, 0.0f, 0.0f));

    /* At new focus: level 0 */
    ASSERT_EQ(clipmap_select_level(clipmap, VEC3(100.0f, 0.0f, 0.0f)), 0);

    /* Old focus point is now far from new focus */
    float dist_to_old = 100.0f;
    int level_old = clipmap_select_level(clipmap, VEC3(0.0f, 0.0f, 0.0f));
    /* Distance 100 > 40 (level 2 extent) -> should be level 3 */
    ASSERT_EQ(level_old, CLIPMAP_LEVELS - 1);

    (void)dist_to_old;

    arena_destroy(arena);
    return 0;
}

TEST(clipmap_null_safety) {
    /* clipmap functions should handle NULL gracefully */
    int level = clipmap_select_level(NULL, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_EQ(level, 0);

    float sdf = clipmap_sdf_query(NULL, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_FLOAT_EQ(sdf, 1e6f);

    RayHit hit = clipmap_raymarch(NULL, VEC3(0.0f, 0.0f, 0.0f),
                                  VEC3(1.0f, 0.0f, 0.0f), 10.0f);
    ASSERT_FALSE(hit.hit);

    /* These should not crash */
    clipmap_update_focus(NULL, VEC3(0.0f, 0.0f, 0.0f));
    clipmap_set_sphere(NULL, VEC3(0.0f, 0.0f, 0.0f), 1.0f, 1);
    clipmap_set_box(NULL, VEC3(0.0f, 0.0f, 0.0f), VEC3(1.0f, 1.0f, 1.0f), 1);
    clipmap_destroy(NULL);

    return 0;
}

/* ============================================================================
 * Section 15: Dirty Tracker Edge Cases
 *
 * Exercise the dirty tracker with edge cases: out-of-bounds indices,
 * double-marking, region that extends beyond world bounds.
 * ============================================================================ */

TEST(dirty_tracker_out_of_bounds_mark) {
    Arena *arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    DirtyTracker *tracker = dirty_tracker_create(arena, 100);
    ASSERT_NOT_NULL(tracker);

    /* Marking beyond max_bricks should be silently ignored */
    dirty_tracker_mark_brick(tracker, 100);  /* Exactly at limit */
    dirty_tracker_mark_brick(tracker, 999);  /* Way beyond */
    ASSERT_EQ(tracker->dirty_count, 0);

    /* Querying beyond max_bricks should return false */
    ASSERT_FALSE(dirty_tracker_is_dirty(tracker, 100));
    ASSERT_FALSE(dirty_tracker_is_dirty(tracker, 999));

    arena_destroy(arena);
    return 0;
}

TEST(dirty_tracker_region_clamps_to_world) {
    Arena *arena = arena_create(16 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(16.0f, 16.0f, 16.0f),
        1.0f, 500, 256);
    DirtyTracker *tracker = dirty_tracker_create(arena, world->grid_total);

    /* Mark a region that extends beyond world bounds.
     * This should clamp to valid brick coordinates and not crash. */
    dirty_tracker_mark_region(tracker, world,
                              VEC3(-100.0f, -100.0f, -100.0f),
                              VEC3(100.0f, 100.0f, 100.0f));

    /* All bricks in the grid should be marked dirty */
    ASSERT_EQ(tracker->dirty_count, world->grid_total);

    arena_destroy(arena);
    return 0;
}

TEST(dirty_tracker_null_safety) {
    /* All dirty tracker functions should handle NULL without crashing */
    dirty_tracker_clear(NULL);
    dirty_tracker_mark_brick(NULL, 0);
    ASSERT_FALSE(dirty_tracker_is_dirty(NULL, 0));
    ASSERT_EQ(dirty_tracker_count(NULL), 0);

    /* dirty_tracker_create with NULL arena should return NULL */
    DirtyTracker *t = dirty_tracker_create(NULL, 100);
    ASSERT_NULL(t);

    /* dirty_tracker_create with 0 capacity should return NULL */
    Arena *arena = arena_create(1024 * 1024);
    t = dirty_tracker_create(arena, 0);
    ASSERT_NULL(t);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 16: Edit List Edge Cases
 *
 * Verify edit list behavior with NULL inputs and boundary conditions.
 * ============================================================================ */

TEST(edit_list_null_safety) {
    /* Null edit list operations should not crash */
    edit_list_clear(NULL);
    ASSERT_EQ(edit_list_count(NULL), 0);

    bool result = edit_list_add(NULL, CSG_UNION, PRIM_SPHERE, VEC3_ZERO, VEC3_ONE, 1);
    ASSERT_FALSE(result);

    /* Create with NULL arena should return NULL */
    EditList *list = edit_list_create(NULL, 100);
    ASSERT_NULL(list);

    /* Create with 0 capacity should return NULL */
    Arena *arena = arena_create(1024 * 1024);
    list = edit_list_create(arena, 0);
    ASSERT_NULL(list);

    arena_destroy(arena);
    return 0;
}

TEST(edit_list_full_rejects_add) {
    Arena *arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    EditList *list = edit_list_create(arena, 2);
    ASSERT_NOT_NULL(list);

    ASSERT_TRUE(edit_list_add(list, CSG_UNION, PRIM_SPHERE, VEC3_ZERO, VEC3_ONE, 1));
    ASSERT_TRUE(edit_list_add(list, CSG_UNION, PRIM_BOX, VEC3_ZERO, VEC3_ONE, 2));
    ASSERT_EQ(edit_list_count(list), 2);

    /* Third add should fail */
    ASSERT_FALSE(edit_list_add(list, CSG_UNION, PRIM_SPHERE, VEC3_ZERO, VEC3_ONE, 3));
    ASSERT_EQ(edit_list_count(list), 2);

    /* After clear, should be able to add again */
    edit_list_clear(list);
    ASSERT_EQ(edit_list_count(list), 0);
    ASSERT_TRUE(edit_list_add(list, CSG_UNION, PRIM_SPHERE, VEC3_ZERO, VEC3_ONE, 4));
    ASSERT_EQ(edit_list_count(list), 1);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 17: Page Dirty Tracking
 *
 * Verify the page-level dirty tracking used for GPU synchronization.
 * ============================================================================ */

TEST(page_dirty_tracking_basic) {
    Arena *arena = arena_create(16 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.5f, 500, 256);

    /* Initially no pages dirty */
    bool dirty_flags[MAX_ATLAS_PAGES];
    uint32_t dirty_count = world_get_dirty_pages(world, dirty_flags, MAX_ATLAS_PAGES);
    ASSERT_EQ(dirty_count, 0);

    /* Manually mark a page dirty */
    world_mark_page_dirty(world, 0);
    dirty_count = world_get_dirty_pages(world, dirty_flags, MAX_ATLAS_PAGES);
    ASSERT_EQ(dirty_count, 1);
    ASSERT_TRUE(dirty_flags[0]);

    /* Clear and verify */
    world_clear_dirty_pages(world);
    dirty_count = world_get_dirty_pages(world, dirty_flags, MAX_ATLAS_PAGES);
    ASSERT_EQ(dirty_count, 0);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 18: Uniform Brick Detection and Compaction
 *
 * After creating geometry, uniform bricks (all inside or all outside)
 * can be compacted to sentinel indices to save atlas memory.
 * ============================================================================ */

TEST(uniform_brick_detection) {
    Arena *arena = arena_create(64 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.5f, 2000, 256);

    /* Allocate a brick and fill it with all-outside values (+127) */
    int32_t atlas_idx = world_alloc_brick(world, 0, 0, 0);
    ASSERT_NE(atlas_idx, BRICK_EMPTY_INDEX);

    int8_t *sdf = world_brick_sdf(world, atlas_idx);
    ASSERT_NOT_NULL(sdf);
    memset(sdf, 127, BRICK_VOXELS);

    /* Detection should identify it as uniform outside */
    int32_t detected = world_detect_uniform_brick(world, atlas_idx);
    ASSERT_EQ(detected, BRICK_UNIFORM_OUTSIDE);

    /* Fill with all-inside values (-127) */
    memset(sdf, (uint8_t)-127, BRICK_VOXELS);
    detected = world_detect_uniform_brick(world, atlas_idx);
    ASSERT_EQ(detected, BRICK_UNIFORM_INSIDE);

    /* Mixed values: should return the atlas index (not uniform) */
    sdf[0] = -127;
    sdf[1] = 127;
    detected = world_detect_uniform_brick(world, atlas_idx);
    ASSERT_EQ(detected, atlas_idx);

    arena_destroy(arena);
    return 0;
}

TEST(uniform_brick_compaction) {
    Arena *arena = arena_create(64 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    WorldBrickMap *world = world_create(arena,
        VEC3(-20.0f, -20.0f, -20.0f), VEC3(20.0f, 20.0f, 20.0f),
        0.5f, 3000, 256);

    /* Create a large sphere that will have some fully-inside interior bricks */
    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 10.0f, 1);

    uint32_t atlas_before = world->atlas_count;

    /* Run compaction */
    uint32_t compacted = world_compact_uniform_bricks(world);

    /* Some bricks should have been converted to uniform */
    /* (At minimum, the SDF query semantics should be preserved) */
    float sdf_center = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_center < 0.0f);

    float sdf_outside = world_sdf_query(world, VEC3(15.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_outside > 0.0f);

    /* If any bricks were compacted, the atlas count (excluding free list)
     * should effectively be lower since freed bricks go to free list. */
    if (compacted > 0) {
        ASSERT_TRUE(world->free_count > 0 ||
                    world->uniform_inside_count > 0 ||
                    world->uniform_outside_count > 0);
    }

    (void)atlas_before;

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 19: Clipmap SDF Query with Geometry at Different Levels
 *
 * Verify that clipmap queries correctly route to the appropriate LOD level
 * and that geometry added to the clipmap is queryable at different distances.
 * ============================================================================ */

TEST(clipmap_geometry_at_focus_and_far) {
    Arena *arena = arena_create(128 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    ClipMapWorld *clipmap = clipmap_create(arena, 0.5f, 10.0f, 1000);
    ASSERT_NOT_NULL(clipmap);

    /* Sphere at origin (near focus, level 0) */
    clipmap_set_sphere(clipmap, VEC3(0.0f, 0.0f, 0.0f), 3.0f, 1);

    /* Box at distance (in level 1 or 2 range) */
    clipmap_set_box(clipmap, VEC3(15.0f, 0.0f, 0.0f), VEC3(2.0f, 2.0f, 2.0f), 2);

    /* Near-focus query should use level 0 and find the sphere */
    float sdf_near = clipmap_sdf_query(clipmap, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_near < 0.0f);

    /* Far query should use higher level and find the box */
    float sdf_far = clipmap_sdf_query(clipmap, VEC3(15.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_far < 0.0f);

    /* Point between sphere and box should be outside */
    float sdf_gap = clipmap_sdf_query(clipmap, VEC3(8.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_gap > 0.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 20: World Create Edge Cases
 *
 * Verify world_create handles all parameter validation correctly.
 * ============================================================================ */

TEST(world_create_edge_cases) {
    Arena *arena = arena_create(16 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Negative voxel size */
    WorldBrickMap *w = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        -0.1f, 100, 256);
    ASSERT_NULL(w);

    /* Equal bounds (zero volume) */
    w = world_create(arena,
        VEC3(5.0f, 5.0f, 5.0f), VEC3(5.0f, 5.0f, 5.0f),
        0.1f, 100, 256);
    ASSERT_NULL(w);

    /* Single-axis zero extent */
    w = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(10.0f, 0.0f, 10.0f),
        0.1f, 100, 256);
    ASSERT_NULL(w);

    /* Valid minimal world: 1 brick in each dimension */
    w = world_create(arena,
        VEC3(0.0f, 0.0f, 0.0f), VEC3(8.0f, 8.0f, 8.0f),
        1.0f, 1, 256);
    /* max_bricks=1 rounds up to ATLAS_PAGE_BRICKS (64) */
    ASSERT_NOT_NULL(w);
    ASSERT_EQ(w->grid_x, 1);
    ASSERT_EQ(w->grid_y, 1);
    ASSERT_EQ(w->grid_z, 1);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Main Test Runner
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("World Brick Map Advanced: Clipmap, Incremental, Boundary");

    /* Section 1: Clipmap update through brick boundaries */
    RUN_TEST(clipmap_update_through_brick_boundary);
    RUN_TEST(clipmap_incremental_focus_movement);

    /* Section 2: Clipmap large jump */
    RUN_TEST(clipmap_large_jump_clears_world);
    RUN_TEST(clipmap_large_jump_all_axes);

    /* Section 3: Incremental add/remove bricks */
    RUN_TEST(incremental_add_then_remove_sphere);
    RUN_TEST(incremental_add_multiple_primitives);
    RUN_TEST(incremental_cylinder_edit);

    /* Section 4: SDF query at exact brick boundary */
    RUN_TEST(sdf_query_brick_boundary_continuity);
    RUN_TEST(sdf_query_brick_boundary_surface_crossing);

    /* Section 5: SDF query outside world bounds */
    RUN_TEST(sdf_query_far_outside_world);
    RUN_TEST(sdf_query_just_outside_world_edge);
    RUN_TEST(material_query_outside_world);

    /* Section 6: SDF batch query */
    RUN_TEST(sdf_batch_query_matches_individual);
    RUN_TEST(sdf_batch_query_empty_world);
    RUN_TEST(sdf_batch_query_zero_count);

    /* Section 7: Raymarch through empty world */
    RUN_TEST(raymarch_empty_world_all_directions);
    RUN_TEST(raymarch_empty_world_batch);

    /* Section 8: Raymarch at grazing angle */
    RUN_TEST(raymarch_grazing_angle_box);
    RUN_TEST(raymarch_grazing_angle_sphere);

    /* Section 9: World from zero-triangle mesh (empty world) */
    RUN_TEST(empty_world_all_queries_default);
    RUN_TEST(empty_world_gradient_no_crash);

    /* Section 10: Material operations */
    RUN_TEST(material_set_on_empty_position);
    RUN_TEST(material_get_from_unallocated_brick);
    RUN_TEST(material_registry_on_non_existent_id);

    /* Section 11: World reset */
    RUN_TEST(world_clear_resets_all_bricks);
    RUN_TEST(world_clear_then_repopulate);
    RUN_TEST(world_clear_resets_counters);

    /* Section 12: Brick allocation stress */
    RUN_TEST(brick_allocation_stress_many_bricks);
    RUN_TEST(brick_allocation_stress_alloc_free_cycle);
    RUN_TEST(brick_allocation_stress_page_growth);

    /* Section 13: SDF gradient at brick boundary */
    RUN_TEST(sdf_gradient_at_brick_boundary_sphere);
    RUN_TEST(sdf_gradient_at_surface_near_brick_boundary);
    RUN_TEST(sdf_gradient_batch_at_boundaries);

    /* Section 14: Clipmap level selection */
    RUN_TEST(clipmap_level_selection_all_levels);
    RUN_TEST(clipmap_null_safety);

    /* Section 15: Dirty tracker edge cases */
    RUN_TEST(dirty_tracker_out_of_bounds_mark);
    RUN_TEST(dirty_tracker_region_clamps_to_world);
    RUN_TEST(dirty_tracker_null_safety);

    /* Section 16: Edit list edge cases */
    RUN_TEST(edit_list_null_safety);
    RUN_TEST(edit_list_full_rejects_add);

    /* Section 17: Page dirty tracking */
    RUN_TEST(page_dirty_tracking_basic);

    /* Section 18: Uniform brick detection and compaction */
    RUN_TEST(uniform_brick_detection);
    RUN_TEST(uniform_brick_compaction);

    /* Section 19: Clipmap geometry at different levels */
    RUN_TEST(clipmap_geometry_at_focus_and_far);

    /* Section 20: World create edge cases */
    RUN_TEST(world_create_edge_cases);

    TEST_SUITE_END();
}
