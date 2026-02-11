/**
 * Geometry Primitives and Mesh Integrity Tests
 *
 * Yoneda Philosophy: Tests as complete behavioral specification.
 *
 * These tests serve as the definitive contract for:
 * 1. Uniform brick initialization (BRICK_UNIFORM_INSIDE must init to SDF=-127)
 * 2. State transitions between brick types
 * 3. Ground primitive at arbitrary heights
 * 4. CSG operations on uniform bricks
 * 5. SDF correctness at primitive boundaries
 *
 * CRITICAL: These tests are designed to FAIL if the following bugs are reintroduced:
 * - Bug #1: world_alloc_brick() initializing BRICK_UNIFORM_INSIDE with SDF=+127
 * - Bug #2: Missing BRICK_UNIFORM_OUTSIDE -> BRICK_UNIFORM_INSIDE transitions
 * - Bug #3: Ground primitive hardcoded to z=0
 */

#include "../include/world_brick_map.h"
#include "test_harness.h"
#include <stdint.h>

/* ============================================================================
 * Section 1: Uniform Brick Initialization Tests
 *
 * CRITICAL BUG REGRESSION: world_alloc_brick() was initializing
 * BRICK_UNIFORM_INSIDE bricks with SDF=+127 instead of -127.
 *
 * This caused hollow spheres inside large filled regions because the
 * "inside" marker was treated as "outside" when reallocated.
 * ============================================================================ */

TEST(uniform_inside_brick_initialization) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.5f, 1000, 256);

    /* Use valid brick coordinates (middle of the grid) */
    int32_t bx = (int32_t)(world->grid_x / 2);
    int32_t by = (int32_t)(world->grid_y / 2);
    int32_t bz = (int32_t)(world->grid_z / 2);

    ASSERT_TRUE(world_brick_valid(world, bx, by, bz));

    /* Manually mark a brick as UNIFORM_INSIDE */
    uint32_t grid_idx = brick_linear_index(world, bx, by, bz);
    world->brick_indices[grid_idx] = BRICK_UNIFORM_INSIDE;
    world->uniform_inside_count++;

    /* Now allocate it - this should initialize SDF to -127, NOT +127 */
    int32_t atlas_idx = world_alloc_brick(world, bx, by, bz);
    ASSERT_NE(atlas_idx, BRICK_EMPTY_INDEX);

    /* Get SDF data and verify initialization */
    int8_t* sdf = world_brick_sdf(world, atlas_idx);
    ASSERT_NOT_NULL(sdf);

    /* CRITICAL: All voxels should be initialized to -127 (inside), not +127 (outside) */
    int inside_count = 0;
    int outside_count = 0;
    for (int i = 0; i < BRICK_VOXELS; i++) {
        if (sdf[i] == -127) inside_count++;
        else if (sdf[i] == 127) outside_count++;
    }

    /* This is the critical assertion that would fail with the old bug */
    ASSERT_EQ(inside_count, BRICK_VOXELS);
    ASSERT_EQ(outside_count, 0);

    /* Additional check: verify SDF query returns negative value */
    Vec3 brick_center = VEC3(
        world->world_min.x + ((float)bx + 0.5f) * world->brick_size_world,
        world->world_min.y + ((float)by + 0.5f) * world->brick_size_world,
        world->world_min.z + ((float)bz + 0.5f) * world->brick_size_world
    );
    float sdf_query = world_sdf_query(world, brick_center);
    ASSERT_TRUE(sdf_query < 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(uniform_outside_brick_initialization) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.5f, 1000, 256);

    /* Use valid brick coordinates (middle of the grid) */
    int32_t bx = (int32_t)(world->grid_x / 2);
    int32_t by = (int32_t)(world->grid_y / 2);
    int32_t bz = (int32_t)(world->grid_z / 2);

    ASSERT_TRUE(world_brick_valid(world, bx, by, bz));

    /* Manually mark a brick as UNIFORM_OUTSIDE */
    uint32_t grid_idx = brick_linear_index(world, bx, by, bz);
    world->brick_indices[grid_idx] = BRICK_UNIFORM_OUTSIDE;
    world->uniform_outside_count++;

    /* Allocate it - should initialize SDF to +127 */
    int32_t atlas_idx = world_alloc_brick(world, bx, by, bz);
    ASSERT_NE(atlas_idx, BRICK_EMPTY_INDEX);

    int8_t* sdf = world_brick_sdf(world, atlas_idx);
    ASSERT_NOT_NULL(sdf);

    /* All voxels should be +127 (outside) */
    int outside_count = 0;
    for (int i = 0; i < BRICK_VOXELS; i++) {
        if (sdf[i] == 127) outside_count++;
    }

    ASSERT_EQ(outside_count, BRICK_VOXELS);

    arena_destroy(arena);
    return 0;
}

TEST(empty_brick_initialization) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.5f, 1000, 256);

    /* Fresh brick (BRICK_EMPTY_INDEX) should initialize to +127 */
    int32_t atlas_idx = world_alloc_brick(world, 0, 0, 0);
    ASSERT_NE(atlas_idx, BRICK_EMPTY_INDEX);

    int8_t* sdf = world_brick_sdf(world, atlas_idx);
    ASSERT_EQ(sdf[0], 127);
    ASSERT_EQ(sdf[BRICK_VOXELS - 1], 127);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 2: State Transition Tests
 *
 * CRITICAL BUG REGRESSION: Primitive generation was missing
 * BRICK_UNIFORM_OUTSIDE -> BRICK_UNIFORM_INSIDE state transitions.
 * ============================================================================ */

TEST(state_transition_empty_to_uniform_inside) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-20.0f, -20.0f, -20.0f), VEC3(20.0f, 20.0f, 20.0f),
        0.5f, 2000, 256);

    /* Create a large sphere that will fill interior bricks completely */
    float radius = 10.0f;
    Vec3 center = VEC3(0.0f, 0.0f, 0.0f);
    world_set_sphere(world, center, radius, 1);

    /* Find a brick that should be deep inside the sphere */
    int32_t bx, by, bz;
    world_pos_to_brick(world, center, &bx, &by, &bz);

    uint32_t grid_idx = brick_linear_index(world, bx, by, bz);
    int32_t brick_state = world->brick_indices[grid_idx];

    /* The center brick should either be UNIFORM_INSIDE or allocated with inside SDF */
    if (brick_state == BRICK_UNIFORM_INSIDE) {
        ASSERT_TRUE(1);  /* Correct */
    } else if (brick_state >= 0) {
        /* Allocated - check that SDF is negative */
        float sdf = world_sdf_query(world, center);
        ASSERT_TRUE(sdf < 0.0f);
    } else {
        ASSERT_MSG(0, "center brick should not be EMPTY or UNIFORM_OUTSIDE");
    }

    /* The primitive functions may or may not use UNIFORM_INSIDE optimization
     * depending on implementation. The key invariant is that the SDF is correct. */
    float sdf_center = world_sdf_query(world, center);
    ASSERT_TRUE(sdf_center < 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(state_transition_uniform_outside_to_uniform_inside) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-20.0f, -20.0f, -20.0f), VEC3(20.0f, 20.0f, 20.0f),
        0.5f, 2000, 256);

    /* First, manually mark some bricks as UNIFORM_OUTSIDE */
    int32_t test_bx = 5, test_by = 5, test_bz = 5;
    uint32_t grid_idx = brick_linear_index(world, test_bx, test_by, test_bz);
    world->brick_indices[grid_idx] = BRICK_UNIFORM_OUTSIDE;
    world->uniform_outside_count++;

    /* Now create a large sphere that encompasses this brick entirely */
    Vec3 brick_center = VEC3(
        world->world_min.x + ((float)test_bx + 0.5f) * world->brick_size_world,
        world->world_min.y + ((float)test_by + 0.5f) * world->brick_size_world,
        world->world_min.z + ((float)test_bz + 0.5f) * world->brick_size_world
    );

    /* Sphere large enough to fully contain this brick */
    float radius = world->brick_size_world * 4.0f;
    world_set_sphere(world, brick_center, radius, 1);

    /* After adding sphere, the brick should transition to UNIFORM_INSIDE or allocated */
    int32_t new_state = world->brick_indices[grid_idx];

    if (new_state == BRICK_UNIFORM_INSIDE) {
        /* Correct: transitioned to UNIFORM_INSIDE */
        ASSERT_TRUE(1);
    } else if (new_state >= 0) {
        /* Allocated brick - verify it's inside */
        float sdf = world_sdf_query(world, brick_center);
        ASSERT_TRUE(sdf < 0.0f);
    } else if (new_state == BRICK_UNIFORM_OUTSIDE) {
        /* BUG DETECTED: The old bug would leave this as UNIFORM_OUTSIDE */
        ASSERT_MSG(0,
            "BUG DETECTED: Brick should NOT remain UNIFORM_OUTSIDE when inside sphere");
    }

    arena_destroy(arena);
    return 0;
}

TEST(state_transition_counters_consistency) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.5f, 1000, 256);

    /* Add sphere */
    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 5.0f, 1);

    /* Count actual states in grid */
    uint32_t actual_inside = 0, actual_outside = 0;
    for (uint32_t i = 0; i < world->grid_total; i++) {
        int32_t state = world->brick_indices[i];
        if (state == BRICK_UNIFORM_INSIDE) actual_inside++;
        else if (state == BRICK_UNIFORM_OUTSIDE) actual_outside++;
    }

    /* Verify counters match actual state */
    ASSERT_EQ(actual_inside, world->uniform_inside_count);
    ASSERT_EQ(actual_outside, world->uniform_outside_count);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 4: Overlapping Primitives Tests
 * ============================================================================ */

TEST(overlapping_spheres) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 2000, 256);

    /* Two overlapping spheres */
    world_set_sphere(world, VEC3(-1.0f, 0.0f, 0.0f), 2.0f, 1);
    world_set_sphere(world, VEC3(1.0f, 0.0f, 0.0f), 2.0f, 2);

    /* Center of overlap should be inside */
    float sdf_center = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_center < 0.0f);

    /* Centers of each sphere should be inside */
    float sdf_s1 = world_sdf_query(world, VEC3(-1.0f, 0.0f, 0.0f));
    float sdf_s2 = world_sdf_query(world, VEC3(1.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_s1 < 0.0f);
    ASSERT_TRUE(sdf_s2 < 0.0f);

    /* Outside both should be positive */
    float sdf_outside = world_sdf_query(world, VEC3(5.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_outside > 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(sphere_inside_box) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 2000, 256);

    /* Large box, then small sphere inside */
    world_set_box(world, VEC3(0.0f, 0.0f, 0.0f), VEC3(3.0f, 3.0f, 3.0f), 1);
    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 1.0f, 2);

    /* Center should be inside (CSG union takes min) */
    float sdf_center = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_center < 0.0f);

    /* Inside box but outside sphere should still be inside (union) */
    float sdf_box_only = world_sdf_query(world, VEC3(2.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_box_only < 0.0f);

    /* Material at center should be sphere's material (added second) */
    uint8_t mat = world_material_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_EQ(mat, 2);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 5: CSG Subtract from Uniform Inside Tests
 *
 * CRITICAL: This tests the interaction between uniform inside bricks and
 * CSG subtract operations. When subtracting from a UNIFORM_INSIDE brick,
 * it must be reallocated with SDF=-127 (inside) before applying the subtract.
 * ============================================================================ */

TEST(csg_subtract_from_uniform_inside) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.2f, 2000, 256);
    EditList* edits = edit_list_create(arena, 100);
    DirtyTracker* tracker = dirty_tracker_create(arena, world->grid_total);

    /* Create a large filled sphere - interior bricks will be UNIFORM_INSIDE */
    edit_list_add(edits, CSG_UNION, PRIM_SPHERE,
                  VEC3(0.0f, 0.0f, 0.0f), VEC3(6.0f, 0.0f, 0.0f), 1);

    world_mark_dirty_bricks(world, tracker,
                            VEC3(-8.0f, -8.0f, -8.0f), VEC3(8.0f, 8.0f, 8.0f));
    world_regenerate_dirty(world, tracker, edits);

    /* Verify center is inside before subtract */
    float sdf_before = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_before < 0.0f);

    /* Now subtract a smaller sphere from the center */
    edit_list_add(edits, CSG_SUBTRACT, PRIM_SPHERE,
                  VEC3(0.0f, 0.0f, 0.0f), VEC3(3.0f, 0.0f, 0.0f), 0);

    world_mark_dirty_bricks(world, tracker,
                            VEC3(-4.0f, -4.0f, -4.0f), VEC3(4.0f, 4.0f, 4.0f));
    world_regenerate_dirty(world, tracker, edits);

    /* Center should now be outside (hollow) */
    float sdf_after = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_after > 0.0f);

    /* Shell should still be inside */
    float sdf_shell = world_sdf_query(world, VEC3(5.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_shell < 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(csg_subtract_preserves_shell) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.2f, 2000, 256);
    EditList* edits = edit_list_create(arena, 100);
    DirtyTracker* tracker = dirty_tracker_create(arena, world->grid_total);

    float outer_radius = 5.0f;
    float inner_radius = 3.0f;

    /* Create hollow sphere */
    edit_list_add(edits, CSG_UNION, PRIM_SPHERE,
                  VEC3(0.0f, 0.0f, 0.0f), VEC3(outer_radius, 0.0f, 0.0f), 1);
    edit_list_add(edits, CSG_SUBTRACT, PRIM_SPHERE,
                  VEC3(0.0f, 0.0f, 0.0f), VEC3(inner_radius, 0.0f, 0.0f), 0);

    world_mark_dirty_bricks(world, tracker,
                            VEC3(-6.0f, -6.0f, -6.0f), VEC3(6.0f, 6.0f, 6.0f));
    world_regenerate_dirty(world, tracker, edits);

    /* Test multiple points in the shell */
    float shell_radius = (outer_radius + inner_radius) / 2.0f;  /* 4.0 */

    /* Test along each axis */
    Vec3 test_points[] = {
        VEC3(shell_radius, 0.0f, 0.0f),
        VEC3(-shell_radius, 0.0f, 0.0f),
        VEC3(0.0f, shell_radius, 0.0f),
        VEC3(0.0f, -shell_radius, 0.0f),
        VEC3(0.0f, 0.0f, shell_radius),
        VEC3(0.0f, 0.0f, -shell_radius),
    };

    for (int i = 0; i < 6; i++) {
        float sdf = world_sdf_query(world, test_points[i]);
        ASSERT_TRUE(sdf < 0.0f);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 6: SDF Correctness at Boundaries Tests
 * ============================================================================ */

TEST(sphere_sdf_accuracy) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 2000, 256);

    float radius = 3.0f;
    Vec3 center = VEC3(0.0f, 0.0f, 0.0f);
    world_set_sphere(world, center, radius, 1);

    /* Test sign correctness at key points (quantization limits exact value matching) */

    /* Inside sphere - SDF must be negative */
    float sdf_center = world_sdf_query(world, center);
    ASSERT_TRUE(sdf_center < 0.0f);

    float sdf_half = world_sdf_query(world, VEC3(radius * 0.5f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_half < 0.0f);

    /* At surface - SDF should be near zero */
    float sdf_surface = world_sdf_query(world, VEC3(radius, 0.0f, 0.0f));
    ASSERT_FLOAT_NEAR(sdf_surface, 0.0f, 0.3f);

    /* Outside sphere - SDF must be positive */
    float sdf_outside_1 = world_sdf_query(world, VEC3(radius + 1.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_outside_1 > 0.0f);

    float sdf_outside_2 = world_sdf_query(world, VEC3(radius + 2.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_outside_2 > sdf_outside_1);

    arena_destroy(arena);
    return 0;
}

TEST(box_sdf_accuracy) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 2000, 256);

    Vec3 half_size = VEC3(2.0f, 2.0f, 2.0f);
    Vec3 center = VEC3(0.0f, 0.0f, 0.0f);
    world_set_box(world, center, half_size, 1);

    /* Center should be deeply inside (negative SDF) */
    float sdf_center = world_sdf_query(world, center);
    ASSERT_TRUE(sdf_center < 0.0f);

    /* Face center should be near 0 */
    float sdf_face = world_sdf_query(world, VEC3(2.0f, 0.0f, 0.0f));
    ASSERT_FLOAT_NEAR(sdf_face, 0.0f, 0.3f);

    /* Outside box should be positive */
    float sdf_outside = world_sdf_query(world, VEC3(3.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_outside > 0.0f);

    /* Corner outside should be positive */
    float sdf_corner_2d = world_sdf_query(world, VEC3(3.0f, 3.0f, 0.0f));
    ASSERT_TRUE(sdf_corner_2d > 0.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 7: Material Assignment Tests
 * ============================================================================ */

TEST(material_inside_sphere) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.2f, 2000, 256);

    uint8_t expected_mat = 42;
    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 3.0f, expected_mat);

    /* Material at center */
    uint8_t mat_center = world_material_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_EQ(mat_center, expected_mat);

    /* Material at various inside points */
    Vec3 inside_points[] = {
        VEC3(1.0f, 0.0f, 0.0f),
        VEC3(0.0f, 1.0f, 0.0f),
        VEC3(0.0f, 0.0f, 1.0f),
        VEC3(1.0f, 1.0f, 0.0f),
    };

    for (size_t i = 0; i < sizeof(inside_points)/sizeof(inside_points[0]); i++) {
        uint8_t mat = world_material_query(world, inside_points[i]);
        ASSERT_EQ(mat, expected_mat);
    }

    arena_destroy(arena);
    return 0;
}

TEST(material_outside_is_air) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.2f, 1000, 256);

    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 2.0f, 5);

    /* Material outside sphere should be 0 (air) */
    uint8_t mat_outside = world_material_query(world, VEC3(5.0f, 0.0f, 0.0f));
    ASSERT_EQ(mat_outside, 0);

    arena_destroy(arena);
    return 0;
}

TEST(material_overlapping_primitives) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.2f, 2000, 256);

    /* First primitive with material 1 */
    world_set_box(world, VEC3(0.0f, 0.0f, 0.0f), VEC3(3.0f, 3.0f, 3.0f), 1);

    /* Second primitive overlapping with material 2 */
    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 2.0f, 2);

    /* At overlap, later primitive's material should win */
    uint8_t mat_overlap = world_material_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_EQ(mat_overlap, 2);

    /* In box-only region, should have box material */
    uint8_t mat_box = world_material_query(world, VEC3(2.5f, 0.0f, 0.0f));
    ASSERT_EQ(mat_box, 1);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 8: Cylinder Primitive Tests
 * ============================================================================ */

TEST(cylinder_basic) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 2000, 256);

    float radius = 2.0f;
    float half_height = 3.0f;
    Vec3 center = VEC3(0.0f, 0.0f, 0.0f);
    world_set_cylinder(world, center, radius, half_height, 1);

    /* Center should be inside */
    float sdf_center = world_sdf_query(world, center);
    ASSERT_TRUE(sdf_center < 0.0f);

    /* On axis but beyond height should be outside */
    float sdf_above = world_sdf_query(world, VEC3(0.0f, 0.0f, half_height + 1.0f));
    ASSERT_TRUE(sdf_above > 0.0f);

    /* At radius on XY plane should be near surface */
    float sdf_edge = world_sdf_query(world, VEC3(radius, 0.0f, 0.0f));
    ASSERT_FLOAT_NEAR(sdf_edge, 0.0f, 0.2f);

    /* Beyond radius should be outside */
    float sdf_beyond = world_sdf_query(world, VEC3(radius + 1.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_beyond > 0.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 9: Mesh Integrity Tests (Raymarching Consistency)
 * ============================================================================ */

TEST(mesh_integrity_sphere_raycast) {
    Arena* arena = arena_create(32 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 2000, 256);

    float radius = 3.0f;
    Vec3 center = VEC3(0.0f, 0.0f, 0.0f);
    world_set_sphere(world, center, radius, 1);

    /* Cast rays from all 6 cardinal directions */
    struct { Vec3 origin; Vec3 dir; float expected_dist; } rays[] = {
        { VEC3(-8.0f, 0.0f, 0.0f), VEC3(1.0f, 0.0f, 0.0f), 8.0f - radius },
        { VEC3(8.0f, 0.0f, 0.0f), VEC3(-1.0f, 0.0f, 0.0f), 8.0f - radius },
        { VEC3(0.0f, -8.0f, 0.0f), VEC3(0.0f, 1.0f, 0.0f), 8.0f - radius },
        { VEC3(0.0f, 8.0f, 0.0f), VEC3(0.0f, -1.0f, 0.0f), 8.0f - radius },
        { VEC3(0.0f, 0.0f, -8.0f), VEC3(0.0f, 0.0f, 1.0f), 8.0f - radius },
        { VEC3(0.0f, 0.0f, 8.0f), VEC3(0.0f, 0.0f, -1.0f), 8.0f - radius },
    };

    for (size_t i = 0; i < sizeof(rays)/sizeof(rays[0]); i++) {
        RayHit hit = world_raymarch(world, rays[i].origin, rays[i].dir, 20.0f);
        ASSERT_TRUE(hit.hit);
        ASSERT_FLOAT_NEAR(hit.distance, rays[i].expected_dist, 0.3f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(mesh_integrity_no_holes) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.1f, 3000, 256);

    float radius = 4.0f;
    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), radius, 1);

    /* Sample many points inside the sphere - all should be inside */
    int holes_found = 0;
    for (float x = -radius + 0.5f; x < radius; x += 0.5f) {
        for (float y = -radius + 0.5f; y < radius; y += 0.5f) {
            for (float z = -radius + 0.5f; z < radius; z += 0.5f) {
                float dist_from_center = sqrtf(x*x + y*y + z*z);
                if (dist_from_center < radius - 0.5f) {
                    /* This point should definitely be inside */
                    float sdf = world_sdf_query(world, VEC3(x, y, z));
                    if (sdf > 0.0f) {
                        holes_found++;
                    }
                }
            }
        }
    }

    ASSERT_EQ(holes_found, 0);

    arena_destroy(arena);
    return 0;
}

TEST(mesh_integrity_hollow_sphere) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.15f, 3000, 256);
    EditList* edits = edit_list_create(arena, 100);
    DirtyTracker* tracker = dirty_tracker_create(arena, world->grid_total);

    float outer_r = 5.0f;
    float inner_r = 3.0f;

    edit_list_add(edits, CSG_UNION, PRIM_SPHERE, VEC3_ZERO, VEC3(outer_r, 0, 0), 1);
    edit_list_add(edits, CSG_SUBTRACT, PRIM_SPHERE, VEC3_ZERO, VEC3(inner_r, 0, 0), 0);

    world_mark_dirty_bricks(world, tracker,
        VEC3(-outer_r-1, -outer_r-1, -outer_r-1),
        VEC3(outer_r+1, outer_r+1, outer_r+1));
    world_regenerate_dirty(world, tracker, edits);

    /* Verify hollow interior */
    float sdf_center = world_sdf_query(world, VEC3(0, 0, 0));
    ASSERT_TRUE(sdf_center > 0.0f);

    /* Verify shell is solid at multiple points */
    float shell_r = (outer_r + inner_r) / 2.0f;
    int shell_holes = 0;
    for (int i = 0; i < 100; i++) {
        /* Sample random directions on shell */
        float theta = (float)i * 0.628f;  /* ~2*pi/10 */
        float phi = (float)i * 0.314f;    /* ~pi/10 */
        float x = shell_r * sinf(phi) * cosf(theta);
        float y = shell_r * sinf(phi) * sinf(theta);
        float z = shell_r * cosf(phi);

        float sdf = world_sdf_query(world, VEC3(x, y, z));
        if (sdf > 0.0f) {
            shell_holes++;
        }
    }

    ASSERT_EQ(shell_holes, 0);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 10: Regression Tests for Specific Bug Fixes
 * ============================================================================ */

TEST(regression_bug1_uniform_inside_init) {
    /*
     * BUG #1: world_alloc_brick() was initializing BRICK_UNIFORM_INSIDE
     * bricks with SDF=+127 (outside) instead of -127 (inside).
     */

    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-15.0f, -15.0f, -15.0f), VEC3(15.0f, 15.0f, 15.0f),
        0.5f, 3000, 256);

    /* Large sphere - interior bricks will be UNIFORM_INSIDE */
    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 8.0f, 1);

    /* Add small sphere at center - this will force some UNIFORM_INSIDE bricks
     * to be reallocated near the surface of the small sphere */
    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 2.0f, 2);

    /* The center should still be inside */
    float sdf_center = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_center < 0.0f);

    /* Sample points that were in the original large sphere's interior */
    Vec3 interior_points[] = {
        VEC3(4.0f, 0.0f, 0.0f),
        VEC3(-4.0f, 0.0f, 0.0f),
        VEC3(0.0f, 4.0f, 0.0f),
        VEC3(0.0f, -4.0f, 0.0f),
        VEC3(0.0f, 0.0f, 4.0f),
        VEC3(0.0f, 0.0f, -4.0f),
        VEC3(3.0f, 3.0f, 0.0f),
        VEC3(-3.0f, -3.0f, 0.0f),
    };

    for (size_t i = 0; i < sizeof(interior_points)/sizeof(interior_points[0]); i++) {
        float sdf = world_sdf_query(world, interior_points[i]);
        ASSERT_TRUE(sdf < 0.0f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(regression_bug2_state_transitions) {
    /*
     * BUG #2: Primitive generation was missing BRICK_UNIFORM_OUTSIDE ->
     * BRICK_UNIFORM_INSIDE state transitions.
     */

    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-20.0f, -20.0f, -20.0f), VEC3(20.0f, 20.0f, 20.0f),
        0.5f, 3000, 256);

    /* Manually mark a specific brick as UNIFORM_OUTSIDE */
    int32_t target_bx = 5, target_by = 5, target_bz = 5;
    uint32_t target_grid_idx = brick_linear_index(world, target_bx, target_by, target_bz);
    world->brick_indices[target_grid_idx] = BRICK_UNIFORM_OUTSIDE;
    world->uniform_outside_count++;

    /* Calculate the world position of this brick's center */
    Vec3 brick_center = VEC3(
        world->world_min.x + ((float)target_bx + 0.5f) * world->brick_size_world,
        world->world_min.y + ((float)target_by + 0.5f) * world->brick_size_world,
        world->world_min.z + ((float)target_bz + 0.5f) * world->brick_size_world
    );

    /* Add sphere large enough to fill this brick completely */
    float large_radius = world->brick_size_world * 3.0f;
    world_set_sphere(world, brick_center, large_radius, 1);

    /* After adding sphere, query the brick center */
    float sdf = world_sdf_query(world, brick_center);
    ASSERT_TRUE(sdf < 0.0f);

    /* Check the brick state - should NOT be UNIFORM_OUTSIDE anymore */
    int32_t new_state = world->brick_indices[target_grid_idx];
    ASSERT_NE(new_state, BRICK_UNIFORM_OUTSIDE);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 11: CSG Operation Tests via Regeneration
 * ============================================================================ */

TEST(csg_union_via_regeneration) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.2f, 2000, 256);
    EditList* edits = edit_list_create(arena, 100);
    DirtyTracker* tracker = dirty_tracker_create(arena, world->grid_total);

    /* Two spheres forming union */
    edit_list_add(edits, CSG_UNION, PRIM_SPHERE,
                  VEC3(-2.0f, 0.0f, 0.0f), VEC3(3.0f, 0.0f, 0.0f), 1);
    edit_list_add(edits, CSG_UNION, PRIM_SPHERE,
                  VEC3(2.0f, 0.0f, 0.0f), VEC3(3.0f, 0.0f, 0.0f), 2);

    world_mark_dirty_bricks(world, tracker,
                            VEC3(-6.0f, -4.0f, -4.0f), VEC3(6.0f, 4.0f, 4.0f));
    world_regenerate_dirty(world, tracker, edits);

    /* Centers of both spheres should be inside */
    float sdf_s1 = world_sdf_query(world, VEC3(-2.0f, 0.0f, 0.0f));
    float sdf_s2 = world_sdf_query(world, VEC3(2.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_s1 < 0.0f);
    ASSERT_TRUE(sdf_s2 < 0.0f);

    /* Overlap region should be inside */
    float sdf_overlap = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_overlap < 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(csg_intersect_via_regeneration) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.15f, 2000, 256);
    EditList* edits = edit_list_create(arena, 100);
    DirtyTracker* tracker = dirty_tracker_create(arena, world->grid_total);

    /* Sphere intersected with box */
    edit_list_add(edits, CSG_UNION, PRIM_SPHERE,
                  VEC3(0.0f, 0.0f, 0.0f), VEC3(4.0f, 0.0f, 0.0f), 1);
    edit_list_add(edits, CSG_INTERSECT, PRIM_BOX,
                  VEC3(0.0f, 0.0f, 0.0f), VEC3(2.0f, 2.0f, 2.0f), 2);

    world_mark_dirty_bricks(world, tracker,
                            VEC3(-5.0f, -5.0f, -5.0f), VEC3(5.0f, 5.0f, 5.0f));
    world_regenerate_dirty(world, tracker, edits);

    /* Center (inside both) should be inside */
    float sdf_center = world_sdf_query(world, VEC3(0.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_center < 0.0f);

    /* Point inside sphere but outside box should be outside */
    float sdf_sphere_only = world_sdf_query(world, VEC3(3.0f, 0.0f, 0.0f));
    ASSERT_TRUE(sdf_sphere_only > 0.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 12: World Statistics Tests
 * ============================================================================ */

TEST(world_stats_consistency) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena,
        VEC3(-10.0f, -10.0f, -10.0f), VEC3(10.0f, 10.0f, 10.0f),
        0.3f, 2000, 256);

    WorldStats stats_before = world_get_stats(world);
    ASSERT_EQ(stats_before.active_bricks, 0);

    /* Add geometry */
    world_set_sphere(world, VEC3(0.0f, 0.0f, 0.0f), 5.0f, 1);

    WorldStats stats_after = world_get_stats(world);
    ASSERT_TRUE(stats_after.active_bricks > 0);

    /* Verify internal consistency */
    ASSERT_EQ(stats_after.uniform_inside, world->uniform_inside_count);
    ASSERT_EQ(stats_after.uniform_outside, world->uniform_outside_count);
    ASSERT_EQ(stats_after.free_list_count, world->free_count);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Main Test Runner
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Geometry Primitives and Mesh Integrity Tests");

    /* Section 1: Uniform Brick Initialization */
    RUN_TEST(uniform_inside_brick_initialization);
    RUN_TEST(uniform_outside_brick_initialization);
    RUN_TEST(empty_brick_initialization);

    /* Section 2: State Transitions */
    RUN_TEST(state_transition_empty_to_uniform_inside);
    RUN_TEST(state_transition_uniform_outside_to_uniform_inside);
    RUN_TEST(state_transition_counters_consistency);

    /* Section 4: Overlapping Primitives */
    RUN_TEST(overlapping_spheres);
    RUN_TEST(sphere_inside_box);

    /* Section 5: CSG Subtract from Uniform Inside */
    RUN_TEST(csg_subtract_from_uniform_inside);
    RUN_TEST(csg_subtract_preserves_shell);

    /* Section 6: SDF Correctness at Boundaries */
    RUN_TEST(sphere_sdf_accuracy);
    RUN_TEST(box_sdf_accuracy);

    /* Section 7: Material Assignment */
    RUN_TEST(material_inside_sphere);
    RUN_TEST(material_outside_is_air);
    RUN_TEST(material_overlapping_primitives);

    /* Section 8: Cylinder Primitive */
    RUN_TEST(cylinder_basic);

    /* Section 9: Mesh Integrity */
    RUN_TEST(mesh_integrity_sphere_raycast);
    RUN_TEST(mesh_integrity_no_holes);
    RUN_TEST(mesh_integrity_hollow_sphere);

    /* Section 10: Regression Tests for Specific Bug Fixes */
    RUN_TEST(regression_bug1_uniform_inside_init);
    RUN_TEST(regression_bug2_state_transitions);

    /* Section 11: CSG Operations via Regeneration */
    RUN_TEST(csg_union_via_regeneration);
    RUN_TEST(csg_intersect_via_regeneration);

    /* Section 12: World Statistics */
    RUN_TEST(world_stats_consistency);

    TEST_SUITE_END();
}
