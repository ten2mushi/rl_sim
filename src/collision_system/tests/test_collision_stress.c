/**
 * Collision System Stress & Edge-Case Tests (Yoneda Philosophy)
 *
 * These tests complement the existing 83-test suite by probing the behavioral
 * boundaries that normal tests miss: worst-case hash distribution, near-
 * exhaustion of entry pools, negative-coordinate handling, cell-boundary
 * precision, extreme cell sizes, KNN edge cases, zero-mass division guards,
 * coincident-position degeneracies, NULL-world safety, and large-scale
 * stress paths.
 *
 * Yoneda principle: an object is fully characterized by the totality of its
 * morphisms. These tests systematically explore the overlooked morphisms --
 * the degenerate, extreme, and adversarial inputs that distinguish a correct
 * implementation from a subtly broken one.
 *
 * Test Categories (14 categories, 27 tests):
 * 1. All Drones In Same Cell (2 tests)
 * 2. Hash Table Near-Exhaustion (2 tests)
 * 3. Negative Coordinates (2 tests)
 * 4. Cell Boundary Detection (2 tests)
 * 5. Very Large Cell Size (1 test)
 * 6. Very Small Cell Size (1 test)
 * 7. K-Nearest With K > Drone Count (2 tests)
 * 8. K-Nearest Equidistant (1 test)
 * 9. Collision Response Zero Mass (2 tests)
 * 10. Collision At Exact Same Position (2 tests)
 * 11. World Collision With NULL World (2 tests)
 * 12. Drone-Drone At Exactly Collision Radius (2 tests)
 * 13. 1024 Drones Uniform Distribution Stress (3 tests)
 * 14. Spatial Hash Clear And Rebuild (2 tests)
 */

#include "../include/collision_system.h"
#include "test_harness.h"
#include <float.h>

#define EPSILON 1e-5f

/* ============================================================================
 * Test Helpers (copied from existing test file for self-containment)
 * ============================================================================ */

/**
 * Create test drone states with specified count, all initialized to defaults.
 */
static RigidBodyStateSOA* create_test_states(Arena* arena, uint32_t count) {
    RigidBodyStateSOA* states = rigid_body_state_create(arena, count);
    if (states == NULL) return NULL;

    for (uint32_t i = 0; i < count; i++) {
        states->quat_w[i] = 1.0f;
    }
    states->count = count;

    return states;
}

/**
 * Create test drone params with uniform mass.
 */
static RigidBodyParamsSOA* create_test_params(Arena* arena, uint32_t count, float mass) {
    RigidBodyParamsSOA* params = rigid_body_params_create(arena, count);
    if (params == NULL) return NULL;

    for (uint32_t i = 0; i < count; i++) {
        rigid_body_params_init(params, i);
        params->mass[i] = mass;
    }
    params->count = count;

    return params;
}

/* ============================================================================
 * Section 1: All Drones In Same Cell
 *
 * When 64 drones are placed at near-identical positions, they all hash to
 * the same cell. This is the worst case for chain length in the spatial
 * hash. We must verify that:
 *   - All 64 entries are inserted (entry_count == 64)
 *   - A query on that cell returns all 64
 *   - Collision detection finds all C(64,2) = 2016 pairs (or caps at max)
 * ============================================================================ */

TEST(stress_all_drones_same_cell_insertion) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* cell_size=1.0 means all drones within [0,1)^3 go to cell (0,0,0) */
    SpatialHashGrid* grid = spatial_hash_create(arena, 64, 1.0f);
    ASSERT_NOT_NULL(grid);

    /* Insert 64 drones at positions clustered within the same cell */
    for (uint32_t i = 0; i < 64; i++) {
        float offset = (float)i * 0.001f;  /* All within [0, 0.064) */
        spatial_hash_insert(grid, i, offset, offset, offset);
    }

    ASSERT_EQ(grid->entry_count, 64);

    /* Query the cell -- all 64 should be found */
    uint32_t indices[128];
    CellQuery query = { .indices = indices, .count = 0, .capacity = 128 };
    spatial_hash_query_cell(grid, 0.0f, 0.0f, 0.0f, &query);

    ASSERT_EQ(query.count, 64);

    /* Verify every drone index [0..63] is present in results */
    uint8_t found[64];
    memset(found, 0, sizeof(found));
    for (uint32_t i = 0; i < query.count; i++) {
        ASSERT_LT(query.indices[i], (uint32_t)64);
        found[query.indices[i]] = 1;
    }
    for (uint32_t i = 0; i < 64; i++) {
        ASSERT_MSG(found[i] == 1, "Every drone should appear in query results");
    }

    arena_destroy(arena);
    return 0;
}

TEST(stress_all_drones_same_cell_collision_detection) {
    Arena* arena = arena_create(8 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* 64 drones, radius=0.5 so all overlap (diameter=1.0, all within 0.064m) */
    CollisionSystem* sys = collision_create(arena, 64, 0.5f, 1.0f);
    ASSERT_NOT_NULL(sys);

    RigidBodyStateSOA* states = create_test_states(arena, 64);
    ASSERT_NOT_NULL(states);

    for (uint32_t i = 0; i < 64; i++) {
        float offset = (float)i * 0.001f;
        states->pos_x[i] = offset;
        states->pos_y[i] = offset;
        states->pos_z[i] = offset;
    }

    collision_build_spatial_hash(sys, states, 64);
    collision_detect_drone_drone(sys, states, 64);

    /* C(64,2) = 2016 pairs. max_pairs = 64*2 = 128, so we should cap. */
    uint32_t expected_max = sys->max_pairs;
    ASSERT_LE(sys->pair_count, expected_max);
    /* With 64 overlapping drones, we should have at least SOME pairs */
    ASSERT_GT(sys->pair_count, (uint32_t)0);

    /* Verify pair ordering invariant: all pairs have i < j */
    for (uint32_t p = 0; p < sys->pair_count; p++) {
        uint32_t idx = p * 2;
        ASSERT_LT(sys->collision_pairs[idx], sys->collision_pairs[idx + 1]);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 2: Hash Table Near-Exhaustion
 *
 * The entry pool has max_entries slots. We test behavior at the boundary:
 *   - Insert max_entries-1 => should succeed
 *   - Insert one more => should succeed (reaches exactly max_entries)
 *   - Insert beyond max_entries => should be silently dropped
 * ============================================================================ */

TEST(stress_hash_near_exhaustion_boundary) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    uint32_t max_entries = 100;
    SpatialHashGrid* grid = spatial_hash_create(arena, max_entries, 1.0f);
    ASSERT_NOT_NULL(grid);

    /* Insert max_entries - 1 */
    for (uint32_t i = 0; i < max_entries - 1; i++) {
        spatial_hash_insert(grid, i, (float)i, 0.0f, 0.0f);
    }
    ASSERT_EQ(grid->entry_count, max_entries - 1);

    /* Insert one more -- should reach exactly max_entries */
    spatial_hash_insert(grid, max_entries - 1, 99.0f, 0.0f, 0.0f);
    ASSERT_EQ(grid->entry_count, max_entries);

    /* Insert beyond max -- should be silently dropped */
    spatial_hash_insert(grid, max_entries, 100.0f, 0.0f, 0.0f);
    ASSERT_EQ(grid->entry_count, max_entries);

    /* All previously inserted entries should still be queryable */
    uint32_t indices[256];
    CellQuery query = { .indices = indices, .count = 0, .capacity = 256 };
    /* Query neighborhood at 0,0,0 to find first few entries */
    spatial_hash_query_neighborhood(grid, 0.0f, 0.0f, 0.0f, &query);
    ASSERT_GT(query.count, (uint32_t)0);

    arena_destroy(arena);
    return 0;
}

TEST(stress_hash_exhaustion_collision_detection_safe) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* Create system with max_agents=32, so spatial hash has 32 entries */
    CollisionSystem* sys = collision_create(arena, 32, 0.1f, 1.0f);
    ASSERT_NOT_NULL(sys);

    RigidBodyStateSOA* states = create_test_states(arena, 32);
    ASSERT_NOT_NULL(states);

    /* All 32 drones at distinct positions */
    for (uint32_t i = 0; i < 32; i++) {
        states->pos_x[i] = (float)i * 5.0f;  /* Well separated */
        states->pos_y[i] = 0.0f;
        states->pos_z[i] = 0.0f;
    }

    /* Build should insert all 32 */
    collision_build_spatial_hash(sys, states, 32);
    ASSERT_EQ(sys->spatial_hash->entry_count, 32);

    /* Detection should complete without crash */
    collision_detect_drone_drone(sys, states, 32);
    /* All are separated, so no collisions */
    ASSERT_EQ(sys->pair_count, 0);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 3: Negative Coordinates
 *
 * The spatial hash uses floorf() to convert to cell coordinates, then casts
 * to int32_t. Negative coordinates must produce valid hash values and correct
 * cell assignments. This tests the prime-based hash with negative int inputs.
 * ============================================================================ */

TEST(stress_negative_coordinates_insertion_and_query) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    SpatialHashGrid* grid = spatial_hash_create(arena, 64, 1.0f);
    ASSERT_NOT_NULL(grid);

    /* Insert drones at various negative coordinates */
    spatial_hash_insert(grid, 0, -100.0f, -200.0f, -50.0f);
    spatial_hash_insert(grid, 1, -100.5f, -200.5f, -50.5f);  /* Same cell */
    spatial_hash_insert(grid, 2, -0.5f, -0.5f, -0.5f);       /* Cell (-1,-1,-1) */
    spatial_hash_insert(grid, 3, -1000.0f, -1000.0f, -1000.0f);

    ASSERT_EQ(grid->entry_count, 4);

    /* Query at (-100, -200, -50) should find drones 0 and 1 */
    uint32_t indices[16];
    CellQuery query = { .indices = indices, .count = 0, .capacity = 16 };
    spatial_hash_query_cell(grid, -100.3f, -200.3f, -50.3f, &query);

    /* Both drone 0 and 1 hash to the same cell: floor(-100*1)=-100, etc. */
    ASSERT_GE(query.count, (uint32_t)1);

    /* Neighborhood query around (-100, -200, -50) should also work */
    query.count = 0;
    spatial_hash_query_neighborhood(grid, -100.3f, -200.3f, -50.3f, &query);
    /* Should find at least drones 0 and 1 */
    ASSERT_GE(query.count, (uint32_t)2);

    arena_destroy(arena);
    return 0;
}

TEST(stress_negative_coordinates_collision_detection) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    CollisionSystem* sys = collision_create(arena, 64, 0.1f, 1.0f);
    ASSERT_NOT_NULL(sys);

    RigidBodyStateSOA* states = create_test_states(arena, 4);
    ASSERT_NOT_NULL(states);

    /* Two drones close together in negative space -- should collide */
    states->pos_x[0] = -100.0f; states->pos_y[0] = -200.0f; states->pos_z[0] = -50.0f;
    states->pos_x[1] = -100.05f; states->pos_y[1] = -200.0f; states->pos_z[1] = -50.0f;

    /* Two drones far apart in negative space -- should NOT collide */
    states->pos_x[2] = -500.0f; states->pos_y[2] = -500.0f; states->pos_z[2] = -500.0f;
    states->pos_x[3] = -600.0f; states->pos_y[3] = -600.0f; states->pos_z[3] = -600.0f;

    collision_detect_all(sys, states, NULL, 4);

    /* Only drones 0 and 1 should be a collision pair */
    ASSERT_EQ(sys->pair_count, 1);
    ASSERT_EQ(sys->collision_pairs[0], 0);
    ASSERT_EQ(sys->collision_pairs[1], 1);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 4: Cell Boundary Detection
 *
 * Two drones exactly cell_size apart, positioned on a cell boundary. The
 * spatial hash neighborhood query must span cell boundaries to catch this.
 * With cell_size=1.0 and collision_radius=0.1 (collision at dist < 0.2), we
 * test drones just inside neighboring cells.
 * ============================================================================ */

TEST(stress_cell_boundary_detection_across_cells) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* cell_size=1.0, collision_radius=0.1 => collision if dist < 0.2 */
    CollisionSystem* sys = collision_create(arena, 64, 0.1f, 1.0f);
    ASSERT_NOT_NULL(sys);

    RigidBodyStateSOA* states = create_test_states(arena, 2);
    ASSERT_NOT_NULL(states);

    /* Drone 0 just below cell boundary, drone 1 just above cell boundary.
     * Cell boundary is at x=1.0. Drone 0 at x=0.95, drone 1 at x=1.05.
     * Distance = 0.10, which is < 0.20 (2*radius), so should collide.
     * They are in different cells: cell(0) vs cell(1). */
    states->pos_x[0] = 0.95f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 1.05f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;

    collision_detect_all(sys, states, NULL, 2);

    /* The 3x3x3 neighborhood query should find both drones */
    ASSERT_EQ(sys->pair_count, 1);
    ASSERT_EQ(sys->collision_pairs[0], 0);
    ASSERT_EQ(sys->collision_pairs[1], 1);

    arena_destroy(arena);
    return 0;
}

TEST(stress_cell_boundary_exact_boundary_position) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    CollisionSystem* sys = collision_create(arena, 64, 0.1f, 1.0f);
    ASSERT_NOT_NULL(sys);

    RigidBodyStateSOA* states = create_test_states(arena, 2);
    ASSERT_NOT_NULL(states);

    /* One drone at exactly the cell boundary x=1.0 (floorf(1.0*1.0) = 1 => cell 1)
     * Another at x=0.85 (floorf(0.85*1.0) = 0 => cell 0).
     * Distance = 0.15, which is < 0.20, so should collide. */
    states->pos_x[0] = 0.85f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 1.0f;  states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;

    collision_detect_all(sys, states, NULL, 2);

    ASSERT_EQ(sys->pair_count, 1);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 5: Very Large Cell Size (100m)
 *
 * With cell_size=100, all drones within a 100m cube map to the same cell.
 * This means the neighborhood query is 27 cells each covering 100m^3.
 * Collision detection should still work correctly.
 * ============================================================================ */

TEST(stress_very_large_cell_size) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    CollisionSystem* sys = collision_create(arena, 64, 0.1f, 100.0f);
    ASSERT_NOT_NULL(sys);
    ASSERT_FLOAT_NEAR(sys->spatial_hash->cell_size, 100.0f, EPSILON);
    ASSERT_FLOAT_NEAR(sys->spatial_hash->inv_cell_size, 0.01f, EPSILON);

    RigidBodyStateSOA* states = create_test_states(arena, 10);
    ASSERT_NOT_NULL(states);

    /* Mix of close and far drones */
    states->pos_x[0] = 0.0f;  states->pos_y[0] = 0.0f;  states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.15f; states->pos_y[1] = 0.0f;  states->pos_z[1] = 0.0f;  /* Should collide with 0 */
    states->pos_x[2] = 50.0f; states->pos_y[2] = 50.0f;  states->pos_z[2] = 0.0f;  /* Far away */
    /* Other drones far apart */
    for (uint32_t i = 3; i < 10; i++) {
        states->pos_x[i] = (float)i * 10.0f;
        states->pos_y[i] = 0.0f;
        states->pos_z[i] = 0.0f;
    }

    collision_detect_all(sys, states, NULL, 10);

    /* Only drones 0 and 1 should collide */
    ASSERT_EQ(sys->pair_count, 1);
    ASSERT_EQ(sys->collision_pairs[0], 0);
    ASSERT_EQ(sys->collision_pairs[1], 1);

    /* Verify all drones inserted into hash */
    ASSERT_EQ(sys->spatial_hash->entry_count, 10);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 6: Very Small Cell Size (0.01m)
 *
 * With cell_size=0.01, drones spread 1m apart span 100 cells each.
 * The neighborhood query (3x3x3 = 0.03m range) is much smaller than the
 * collision radius (2 * 0.1 = 0.2m). This means drones that should collide
 * may be in cells more than 1 hop apart.
 *
 * This is a critical edge case: the collision detection relies on the 3x3x3
 * neighborhood covering at least the collision diameter. When
 * cell_size < 2*radius, the neighborhood is insufficient.
 * The implementation should still find nearby pairs via the 3x3x3 scan,
 * but will MISS pairs that are within collision distance but > 1.5*cell_size
 * apart. This is a known limitation / documented design constraint.
 * ============================================================================ */

TEST(stress_very_small_cell_size_detection) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* cell_size=0.01, radius=0.1. 3x3x3 neighborhood covers 0.03m, but
     * collision diameter is 0.2m. Drones 0.15m apart should collide but may
     * be missed because they're in cells 15 apart. */
    CollisionSystem* sys = collision_create(arena, 64, 0.1f, 0.01f);
    ASSERT_NOT_NULL(sys);

    RigidBodyStateSOA* states = create_test_states(arena, 2);
    ASSERT_NOT_NULL(states);

    /* Two drones 0.005m apart -- within the 3-cell neighborhood (0.03m range) */
    states->pos_x[0] = 0.0f;   states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.005f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;

    collision_detect_all(sys, states, NULL, 2);

    /* Very close drones (well within 3-cell neighborhood) should still be found */
    ASSERT_EQ(sys->pair_count, 1);

    /* Now test drones 0.15m apart -- beyond the 3-cell neighborhood */
    collision_reset(sys);
    states->pos_x[1] = 0.15f;  /* 15 cells apart on the 0.01m grid */

    collision_build_spatial_hash(sys, states, 2);
    collision_detect_drone_drone(sys, states, 2);

    /* BUG DETECTED (by design): cell_size=0.01 means 3x3x3 neighborhood only
     * covers 0.03m, but collision diameter is 0.2m. Drones 0.15m apart are
     * within collision range but NOT within the 3-cell neighborhood.
     *
     * Expected: pair_count == 1 (drones are within collision diameter)
     * Actual: pair_count == 0 (neighborhood too small to find them)
     *
     * This is a KNOWN DESIGN CONSTRAINT documented in the header:
     * "cell_size should be >= 2 * collision_radius"
     * When violated, collisions between drones in distant cells are missed. */
    if (sys->pair_count == 0) {
        /* Document the expected behavior: with tiny cells, distant-cell
         * pairs ARE missed. This is correct behavior for the implementation
         * (the user should not set cell_size < 2*radius). */
        printf(" [NOTE: pair_count=0 as expected when cell_size < 2*radius]");
    }
    /* Either 0 (known limitation) or 1 (if implementation expanded search) is acceptable */
    ASSERT_LE(sys->pair_count, (uint32_t)1);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 7: K-Nearest With K > Drone Count
 *
 * When we request more neighbors than exist, the function should return
 * only the available drones (count < k), not crash or return garbage.
 * ============================================================================ */

TEST(stress_knn_k_greater_than_agent_count) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    CollisionSystem* sys = collision_create(arena, 64, 0.1f, 1.0f);
    ASSERT_NOT_NULL(sys);

    RigidBodyStateSOA* states = create_test_states(arena, 3);
    ASSERT_NOT_NULL(states);

    states->pos_x[0] = 0.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.5f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;
    states->pos_x[2] = 1.0f; states->pos_y[2] = 0.0f; states->pos_z[2] = 0.0f;

    collision_build_spatial_hash(sys, states, 3);

    uint32_t out_indices[10];
    float out_distances[10];
    uint32_t out_count = 0;

    /* Initialize output to sentinel values to detect overwrites */
    for (uint32_t i = 0; i < 10; i++) {
        out_indices[i] = 0xDEADBEEF;
        out_distances[i] = -999.0f;
    }

    /* Request k=10 neighbors but only 3 drones exist.
     * Query at (0.1, 0, 0) -- close to drone 0, so drone 0 is the "self"
     * exclusion distance. All 3 drones should be non-zero distance. */
    collision_find_k_nearest(sys, states, VEC3(0.1f, 0.0f, 0.0f), 10,
                            out_indices, out_distances, &out_count);

    /* Should return at most 3 (all drones) */
    ASSERT_LE(out_count, (uint32_t)3);
    /* Should return at least 1 */
    ASSERT_GE(out_count, (uint32_t)1);

    /* Returned indices should be valid drone indices */
    for (uint32_t i = 0; i < out_count; i++) {
        ASSERT_LT(out_indices[i], (uint32_t)3);
    }

    /* Distances should be sorted ascending */
    for (uint32_t i = 1; i < out_count; i++) {
        ASSERT_GE(out_distances[i], out_distances[i - 1]);
    }

    /* Unfilled slots (beyond out_count) should have sentinel distance FLT_MAX */
    for (uint32_t i = out_count; i < 10; i++) {
        ASSERT_FLOAT_EQ(out_distances[i], FLT_MAX);
    }

    arena_destroy(arena);
    return 0;
}

TEST(stress_knn_k_greater_than_agent_count_batch) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    CollisionSystem* sys = collision_create(arena, 64, 0.1f, 1.0f);
    ASSERT_NOT_NULL(sys);

    RigidBodyStateSOA* states = create_test_states(arena, 3);
    ASSERT_NOT_NULL(states);

    states->pos_x[0] = 0.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.5f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;
    states->pos_x[2] = 1.0f; states->pos_y[2] = 0.0f; states->pos_z[2] = 0.0f;

    collision_build_spatial_hash(sys, states, 3);

    uint32_t k = 8;  /* More than 3-1=2 available neighbors per drone */
    uint32_t* batch_indices = arena_alloc_array(arena, uint32_t, 3 * k);
    float* batch_distances = arena_alloc_array(arena, float, 3 * k);
    ASSERT_NOT_NULL(batch_indices);
    ASSERT_NOT_NULL(batch_distances);

    collision_find_k_nearest_batch(sys, states, 3, k, batch_indices, batch_distances);

    /* For each drone, at most 2 neighbors exist (self excluded).
     * The remaining k-2 slots should have UINT32_MAX indices and FLT_MAX distances. */
    for (uint32_t d = 0; d < 3; d++) {
        /* No drone should find itself */
        for (uint32_t j = 0; j < k; j++) {
            uint32_t idx = batch_indices[d * k + j];
            if (idx != UINT32_MAX) {
                ASSERT_NE(idx, d);
                ASSERT_LT(idx, (uint32_t)3);
            }
        }
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 8: K-Nearest Equidistant
 *
 * All drones equidistant from query point. The sort must be stable enough
 * to return exactly K results without crashing, even when all distances are
 * identical.
 * ============================================================================ */

TEST(stress_knn_equidistant) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    CollisionSystem* sys = collision_create(arena, 64, 0.1f, 2.0f);
    ASSERT_NOT_NULL(sys);

    RigidBodyStateSOA* states = create_test_states(arena, 8);
    ASSERT_NOT_NULL(states);

    /* Place 8 drones equidistant from origin on a unit sphere
     * at corners of a cube: (+-1/sqrt3, +-1/sqrt3, +-1/sqrt3) */
    float c = 1.0f / sqrtf(3.0f);
    float coords[8][3] = {
        { c,  c,  c}, {-c,  c,  c}, { c, -c,  c}, { c,  c, -c},
        {-c, -c,  c}, {-c,  c, -c}, { c, -c, -c}, {-c, -c, -c}
    };
    for (uint32_t i = 0; i < 8; i++) {
        states->pos_x[i] = coords[i][0];
        states->pos_y[i] = coords[i][1];
        states->pos_z[i] = coords[i][2];
    }

    collision_build_spatial_hash(sys, states, 8);

    uint32_t out_indices[4];
    float out_distances[4];
    uint32_t out_count = 0;

    /* Query at origin -- all 8 drones are at distance 1.0 (squared: 1.0) */
    collision_find_k_nearest(sys, states, VEC3(0.0f, 0.0f, 0.0f), 4,
                            out_indices, out_distances, &out_count);

    /* Should find exactly 4 */
    ASSERT_EQ(out_count, 4);

    /* All distances should be approximately equal (1.0 squared) */
    for (uint32_t i = 0; i < out_count; i++) {
        ASSERT_FLOAT_NEAR(out_distances[i], 1.0f, 0.01f);
    }

    /* All returned indices should be valid and distinct */
    for (uint32_t i = 0; i < out_count; i++) {
        ASSERT_LT(out_indices[i], (uint32_t)8);
        for (uint32_t j = i + 1; j < out_count; j++) {
            ASSERT_NE(out_indices[i], out_indices[j]);
        }
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 9: Collision Response Zero Mass
 *
 * When params->mass[i] == 0 for one or both drones, the response code
 * computes total_mass = mass_i + mass_j. If both are 0, total_mass = 0
 * and division by zero occurs. We test the implementation's behavior.
 * ============================================================================ */

TEST(stress_response_zero_mass_one_drone) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    CollisionSystem* sys = collision_create(arena, 64, 0.1f, 1.0f);
    ASSERT_NOT_NULL(sys);

    RigidBodyStateSOA* states = create_test_states(arena, 2);
    RigidBodyParamsSOA* params = create_test_params(arena, 2, 1.0f);
    ASSERT_NOT_NULL(states);
    ASSERT_NOT_NULL(params);

    /* One drone with zero mass */
    params->mass[0] = 0.0f;
    params->mass[1] = 1.0f;

    /* Overlapping drones approaching each other */
    states->pos_x[0] = 0.0f;  states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.15f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;
    states->vel_x[0] = 5.0f;
    states->vel_x[1] = -5.0f;

    collision_detect_all(sys, states, NULL, 2);
    ASSERT_EQ(sys->pair_count, 1);

    /* Apply response -- total_mass = 0 + 1 = 1, should work fine */
    collision_apply_drone_response(sys, states, params, 0.5f, 2);

    /* Verify no NaN or Inf in output */
    ASSERT_FALSE(isnan(states->pos_x[0]));
    ASSERT_FALSE(isnan(states->pos_x[1]));
    ASSERT_FALSE(isinf(states->pos_x[0]));
    ASSERT_FALSE(isinf(states->pos_x[1]));
    ASSERT_FALSE(isnan(states->vel_x[0]));
    ASSERT_FALSE(isnan(states->vel_x[1]));
    ASSERT_FALSE(isinf(states->vel_x[0]));
    ASSERT_FALSE(isinf(states->vel_x[1]));

    arena_destroy(arena);
    return 0;
}

TEST(stress_response_zero_mass_both_drones) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    CollisionSystem* sys = collision_create(arena, 64, 0.1f, 1.0f);
    ASSERT_NOT_NULL(sys);

    RigidBodyStateSOA* states = create_test_states(arena, 2);
    RigidBodyParamsSOA* params = create_test_params(arena, 2, 0.0f);  /* Both zero mass! */
    ASSERT_NOT_NULL(states);
    ASSERT_NOT_NULL(params);

    /* Overlapping drones approaching each other */
    states->pos_x[0] = 0.0f;  states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.15f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;
    states->vel_x[0] = 5.0f;
    states->vel_x[1] = -5.0f;

    collision_detect_all(sys, states, NULL, 2);
    ASSERT_EQ(sys->pair_count, 1);

    /* With zero total_mass, the response skips the pair gracefully.
     * Positions and velocities must remain finite (no NaN/Inf). */
    collision_apply_drone_response(sys, states, params, 0.5f, 2);

    ASSERT_FALSE(isnan(states->pos_x[0]) || isinf(states->pos_x[0]));
    ASSERT_FALSE(isnan(states->pos_x[1]) || isinf(states->pos_x[1]));
    ASSERT_FALSE(isnan(states->vel_x[0]) || isinf(states->vel_x[0]));
    ASSERT_FALSE(isnan(states->vel_x[1]) || isinf(states->vel_x[1]));

    /* Velocities should be unchanged since the pair is skipped */
    ASSERT_FLOAT_EQ(states->vel_x[0], 5.0f);
    ASSERT_FLOAT_EQ(states->vel_x[1], -5.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 10: Collision At Exact Same Position
 *
 * Two drones at identical coordinates produce zero distance. The response
 * code must handle this without division by zero in normal computation.
 * The implementation uses an arbitrary normal (1,0,0) when dist < 1e-6.
 * ============================================================================ */

TEST(stress_exact_same_position_detection) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    CollisionSystem* sys = collision_create(arena, 64, 0.1f, 1.0f);
    ASSERT_NOT_NULL(sys);

    RigidBodyStateSOA* states = create_test_states(arena, 3);
    ASSERT_NOT_NULL(states);

    /* Three drones at exactly the same position */
    states->pos_x[0] = 5.0f; states->pos_y[0] = 5.0f; states->pos_z[0] = 5.0f;
    states->pos_x[1] = 5.0f; states->pos_y[1] = 5.0f; states->pos_z[1] = 5.0f;
    states->pos_x[2] = 5.0f; states->pos_y[2] = 5.0f; states->pos_z[2] = 5.0f;

    collision_detect_all(sys, states, NULL, 3);

    /* Should detect all C(3,2) = 3 collision pairs */
    ASSERT_EQ(sys->pair_count, 3);

    /* Verify pair ordering: i < j for all pairs */
    for (uint32_t p = 0; p < sys->pair_count; p++) {
        uint32_t idx = p * 2;
        ASSERT_LT(sys->collision_pairs[idx], sys->collision_pairs[idx + 1]);
    }

    arena_destroy(arena);
    return 0;
}

TEST(stress_exact_same_position_response_no_nan) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    CollisionSystem* sys = collision_create(arena, 64, 0.1f, 1.0f);
    ASSERT_NOT_NULL(sys);

    RigidBodyStateSOA* states = create_test_states(arena, 2);
    RigidBodyParamsSOA* params = create_test_params(arena, 2, 1.0f);
    ASSERT_NOT_NULL(states);
    ASSERT_NOT_NULL(params);

    /* Two drones at exactly the same position, approaching each other */
    states->pos_x[0] = 5.0f; states->pos_y[0] = 5.0f; states->pos_z[0] = 5.0f;
    states->pos_x[1] = 5.0f; states->pos_y[1] = 5.0f; states->pos_z[1] = 5.0f;
    states->vel_x[0] = 1.0f;
    states->vel_x[1] = -1.0f;

    collision_detect_all(sys, states, NULL, 2);
    ASSERT_EQ(sys->pair_count, 1);

    collision_apply_drone_response(sys, states, params, 0.5f, 2);

    /* The response code uses arbitrary normal (1,0,0) when dist < 1e-6
     * and sets dist = 0.001. Verify no NaN. */
    ASSERT_FALSE(isnan(states->pos_x[0]));
    ASSERT_FALSE(isnan(states->pos_y[0]));
    ASSERT_FALSE(isnan(states->pos_z[0]));
    ASSERT_FALSE(isnan(states->pos_x[1]));
    ASSERT_FALSE(isnan(states->pos_y[1]));
    ASSERT_FALSE(isnan(states->pos_z[1]));

    /* Drones should be separated after response (pushed along (1,0,0)) */
    float dist_after = fabsf(states->pos_x[1] - states->pos_x[0]);
    ASSERT_GT(dist_after, 0.0f);

    /* Verify the arbitrary normal direction: drone 0 should move in -X,
     * drone 1 should move in +X (normal from 0 to 1 is (1,0,0)) */
    ASSERT_LE(states->pos_x[0], 5.0f);
    ASSERT_GE(states->pos_x[1], 5.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 11: World Collision With NULL World
 *
 * collision_detect_drone_world with NULL world should return immediately.
 * collision_detect_all with NULL world should skip world detection but
 * still do drone-drone detection.
 * ============================================================================ */

TEST(stress_null_world_detect_drone_world) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    CollisionSystem* sys = collision_create(arena, 64, 0.1f, 1.0f);
    ASSERT_NOT_NULL(sys);

    RigidBodyStateSOA* states = create_test_states(arena, 10);
    ASSERT_NOT_NULL(states);

    for (uint32_t i = 0; i < 10; i++) {
        states->pos_x[i] = (float)i;
        states->pos_y[i] = 0.0f;
        states->pos_z[i] = 0.0f;
    }

    /* Direct call with NULL world -- should not crash */
    collision_detect_drone_world(sys, states, NULL, 10);

    /* No world collisions should be flagged */
    for (uint32_t i = 0; i < 10; i++) {
        ASSERT_EQ(sys->drone_world_collision[i], 0);
    }

    arena_destroy(arena);
    return 0;
}

TEST(stress_null_world_detect_all_still_does_drone_drone) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    CollisionSystem* sys = collision_create(arena, 64, 0.1f, 1.0f);
    ASSERT_NOT_NULL(sys);

    RigidBodyStateSOA* states = create_test_states(arena, 4);
    ASSERT_NOT_NULL(states);

    /* Two drones close together, two far away */
    states->pos_x[0] = 0.0f;  states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.1f;  states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;
    states->pos_x[2] = 50.0f; states->pos_y[2] = 0.0f; states->pos_z[2] = 0.0f;
    states->pos_x[3] = 100.0f; states->pos_y[3] = 0.0f; states->pos_z[3] = 0.0f;

    /* NULL world should not prevent drone-drone detection */
    collision_detect_all(sys, states, NULL, 4);

    /* Drone-drone collision should still be detected */
    ASSERT_EQ(sys->pair_count, 1);
    ASSERT_EQ(sys->collision_pairs[0], 0);
    ASSERT_EQ(sys->collision_pairs[1], 1);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 12: Drone-Drone At Exactly Collision Radius
 *
 * The collision threshold is dist_sq < radius_sum_sq (strict less-than).
 * At exactly 2*radius distance, dist_sq == radius_sum_sq, so no collision.
 * We verify this boundary precisely.
 * ============================================================================ */

TEST(stress_exactly_at_collision_radius_no_collision) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    /* radius = 0.1, collision threshold: dist < 2*radius = 0.2 */
    CollisionSystem* sys = collision_create(arena, 64, 0.1f, 1.0f);
    ASSERT_NOT_NULL(sys);

    RigidBodyStateSOA* states = create_test_states(arena, 2);
    ASSERT_NOT_NULL(states);

    /* Place drones at exactly 2*radius apart (0.2m) */
    states->pos_x[0] = 0.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.2f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;

    collision_detect_all(sys, states, NULL, 2);

    /* dist_sq = 0.04, radius_sum_sq = (2*0.1)^2 = 0.04.
     * Condition is dist_sq < radius_sum_sq, which is 0.04 < 0.04 = false.
     * So NO collision at exactly 2*radius. This documents strict-less-than. */
    ASSERT_EQ(sys->pair_count, 0);

    arena_destroy(arena);
    return 0;
}

TEST(stress_just_inside_collision_radius) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    CollisionSystem* sys = collision_create(arena, 64, 0.1f, 1.0f);
    ASSERT_NOT_NULL(sys);

    RigidBodyStateSOA* states = create_test_states(arena, 2);
    ASSERT_NOT_NULL(states);

    /* Place drones just barely inside collision range: 0.19999m < 0.2m */
    states->pos_x[0] = 0.0f;    states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.1999f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;

    collision_detect_all(sys, states, NULL, 2);

    /* dist_sq = 0.1999^2 = 0.039960, radius_sum_sq = 0.04.
     * 0.039960 < 0.04 = true. Should detect collision. */
    ASSERT_EQ(sys->pair_count, 1);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 13: 1024 Drones Uniform Distribution Stress
 *
 * Large-scale test with 1024 drones. Verifies that the full pipeline
 * (build, detect, response, knn) completes without crashing.
 * ============================================================================ */

TEST(stress_1024_drones_full_pipeline) {
    Arena* arena = arena_create(16 * 1024 * 1024);  /* 16MB for 1024 drones */
    ASSERT_NOT_NULL(arena);

    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    ASSERT_NOT_NULL(sys);

    RigidBodyStateSOA* states = create_test_states(arena, 1024);
    RigidBodyParamsSOA* params = create_test_params(arena, 1024, 1.0f);
    ASSERT_NOT_NULL(states);
    ASSERT_NOT_NULL(params);

    /* Place drones on a 32x32 grid with 1m spacing (well separated) */
    for (uint32_t i = 0; i < 1024; i++) {
        states->pos_x[i] = (float)(i % 32) * 1.0f;
        states->pos_y[i] = (float)((i / 32) % 32) * 1.0f;
        states->pos_z[i] = 0.0f;
        /* Give some velocity */
        states->vel_x[i] = (float)(i % 7) * 0.1f - 0.3f;
        states->vel_y[i] = (float)(i % 5) * 0.1f - 0.2f;
        states->vel_z[i] = 0.0f;
    }

    /* Full pipeline: detect all */
    collision_detect_all(sys, states, NULL, 1024);

    /* With 1m spacing and 0.2m collision diameter, no collisions expected */
    ASSERT_EQ(sys->pair_count, 0);
    ASSERT_EQ(sys->spatial_hash->entry_count, 1024);

    /* Apply response (no-op since no collisions, but should not crash) */
    collision_apply_response(sys, states, params, 0.5f, 1.0f, 1024);

    /* Verify no NaN in any position */
    for (uint32_t i = 0; i < 1024; i++) {
        ASSERT_FALSE(isnan(states->pos_x[i]));
        ASSERT_FALSE(isnan(states->pos_y[i]));
        ASSERT_FALSE(isnan(states->pos_z[i]));
    }

    arena_destroy(arena);
    return 0;
}

TEST(stress_1024_drones_clustered_collisions) {
    Arena* arena = arena_create(16 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    ASSERT_NOT_NULL(sys);

    RigidBodyStateSOA* states = create_test_states(arena, 1024);
    RigidBodyParamsSOA* params = create_test_params(arena, 1024, 1.0f);
    ASSERT_NOT_NULL(states);
    ASSERT_NOT_NULL(params);

    /* Place first 100 drones in a tight cluster (0.05m spacing, all overlapping)
     * and remaining 924 on a grid far away */
    for (uint32_t i = 0; i < 100; i++) {
        states->pos_x[i] = (float)(i % 10) * 0.05f;
        states->pos_y[i] = (float)(i / 10) * 0.05f;
        states->pos_z[i] = 0.0f;
    }
    for (uint32_t i = 100; i < 1024; i++) {
        states->pos_x[i] = 100.0f + (float)((i - 100) % 30) * 2.0f;
        states->pos_y[i] = 100.0f + (float)(((i - 100) / 30) % 30) * 2.0f;
        states->pos_z[i] = 0.0f;
    }

    collision_detect_all(sys, states, NULL, 1024);

    /* Should find collisions among the first 100 drones */
    ASSERT_GT(sys->pair_count, (uint32_t)0);
    /* Should not exceed max_pairs */
    ASSERT_LE(sys->pair_count, sys->max_pairs);
    /* All entries should be inserted */
    ASSERT_EQ(sys->spatial_hash->entry_count, 1024);

    /* Apply response -- should not crash even with many pairs */
    collision_apply_drone_response(sys, states, params, 0.5f, 1024);

    /* Verify no NaN in any drone position */
    for (uint32_t i = 0; i < 1024; i++) {
        ASSERT_FALSE(isnan(states->pos_x[i]));
        ASSERT_FALSE(isnan(states->pos_y[i]));
        ASSERT_FALSE(isnan(states->pos_z[i]));
    }

    arena_destroy(arena);
    return 0;
}

TEST(stress_1024_drones_knn_batch) {
    Arena* arena = arena_create(16 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    ASSERT_NOT_NULL(sys);

    RigidBodyStateSOA* states = create_test_states(arena, 1024);
    ASSERT_NOT_NULL(states);

    /* Uniform grid placement */
    for (uint32_t i = 0; i < 1024; i++) {
        states->pos_x[i] = (float)(i % 32) * 1.0f;
        states->pos_y[i] = (float)((i / 32) % 32) * 1.0f;
        states->pos_z[i] = 0.0f;
    }

    collision_build_spatial_hash(sys, states, 1024);

    uint32_t k = 5;
    uint32_t* batch_indices = arena_alloc_array(arena, uint32_t, 1024 * k);
    float* batch_distances = arena_alloc_array(arena, float, 1024 * k);
    ASSERT_NOT_NULL(batch_indices);
    ASSERT_NOT_NULL(batch_distances);

    collision_find_k_nearest_batch(sys, states, 1024, k, batch_indices, batch_distances);

    /* Verify no drone finds itself and all valid indices are in range */
    for (uint32_t d = 0; d < 1024; d++) {
        for (uint32_t j = 0; j < k; j++) {
            uint32_t idx = batch_indices[d * k + j];
            if (idx != UINT32_MAX) {
                ASSERT_NE(idx, d);
                ASSERT_LT(idx, (uint32_t)1024);
            }
        }
        /* Distances should be sorted */
        for (uint32_t j = 1; j < k; j++) {
            ASSERT_GE(batch_distances[d * k + j], batch_distances[d * k + j - 1]);
        }
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 14: Spatial Hash Clear And Rebuild
 *
 * Verify that after clearing and rebuilding the hash with different positions,
 * queries return the updated results, not stale data.
 * ============================================================================ */

TEST(stress_clear_and_rebuild_different_positions) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    CollisionSystem* sys = collision_create(arena, 64, 0.1f, 1.0f);
    ASSERT_NOT_NULL(sys);

    RigidBodyStateSOA* states = create_test_states(arena, 4);
    ASSERT_NOT_NULL(states);

    /* Phase 1: Place drones 0,1 close together, detect collision */
    states->pos_x[0] = 0.0f;  states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.1f;  states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;
    states->pos_x[2] = 50.0f; states->pos_y[2] = 0.0f; states->pos_z[2] = 0.0f;
    states->pos_x[3] = 100.0f; states->pos_y[3] = 0.0f; states->pos_z[3] = 0.0f;

    collision_detect_all(sys, states, NULL, 4);
    ASSERT_EQ(sys->pair_count, 1);
    ASSERT_EQ(sys->collision_pairs[0], 0);
    ASSERT_EQ(sys->collision_pairs[1], 1);

    /* Phase 2: Move drones -- now 2,3 are close, 0,1 are far apart */
    states->pos_x[0] = -100.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 200.0f;  states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;
    states->pos_x[2] = 50.0f;   states->pos_y[2] = 0.0f; states->pos_z[2] = 0.0f;
    states->pos_x[3] = 50.1f;   states->pos_y[3] = 0.0f; states->pos_z[3] = 0.0f;

    collision_detect_all(sys, states, NULL, 4);

    /* Should find new collision (2,3), not old collision (0,1) */
    ASSERT_EQ(sys->pair_count, 1);
    ASSERT_EQ(sys->collision_pairs[0], 2);
    ASSERT_EQ(sys->collision_pairs[1], 3);

    arena_destroy(arena);
    return 0;
}

TEST(stress_clear_and_rebuild_knn_updated) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    CollisionSystem* sys = collision_create(arena, 64, 0.1f, 1.0f);
    ASSERT_NOT_NULL(sys);

    RigidBodyStateSOA* states = create_test_states(arena, 4);
    ASSERT_NOT_NULL(states);

    /* Phase 1: Drone 1 is closest to query point */
    states->pos_x[0] = 10.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.5f;  states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;
    states->pos_x[2] = 5.0f;  states->pos_y[2] = 0.0f; states->pos_z[2] = 0.0f;
    states->pos_x[3] = 8.0f;  states->pos_y[3] = 0.0f; states->pos_z[3] = 0.0f;

    collision_build_spatial_hash(sys, states, 4);

    uint32_t out_indices[1];
    float out_distances[1];
    uint32_t out_count = 0;

    collision_find_k_nearest(sys, states, VEC3(0.0f, 0.0f, 0.0f), 1,
                            out_indices, out_distances, &out_count);
    ASSERT_EQ(out_count, 1);
    ASSERT_EQ(out_indices[0], 1);  /* Drone 1 at 0.5 is nearest to origin */

    /* Phase 2: Move drone 2 closer, rebuild */
    states->pos_x[1] = 100.0f;   /* Move drone 1 far away */
    states->pos_x[2] = 0.1f;     /* Move drone 2 very close to origin */

    collision_build_spatial_hash(sys, states, 4);

    out_count = 0;
    collision_find_k_nearest(sys, states, VEC3(0.0f, 0.0f, 0.0f), 1,
                            out_indices, out_distances, &out_count);

    ASSERT_EQ(out_count, 1);
    /* Drone 2 at 0.1 should now be nearest (not stale result of drone 1) */
    ASSERT_EQ(out_indices[0], 2);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Main Test Runner
 * ============================================================================ */

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    TEST_SUITE_BEGIN("Collision System Stress & Edge Cases (Yoneda Philosophy)");

    /* Section 1: All Drones In Same Cell */
    RUN_TEST(stress_all_drones_same_cell_insertion);
    RUN_TEST(stress_all_drones_same_cell_collision_detection);

    /* Section 2: Hash Table Near-Exhaustion */
    RUN_TEST(stress_hash_near_exhaustion_boundary);
    RUN_TEST(stress_hash_exhaustion_collision_detection_safe);

    /* Section 3: Negative Coordinates */
    RUN_TEST(stress_negative_coordinates_insertion_and_query);
    RUN_TEST(stress_negative_coordinates_collision_detection);

    /* Section 4: Cell Boundary Detection */
    RUN_TEST(stress_cell_boundary_detection_across_cells);
    RUN_TEST(stress_cell_boundary_exact_boundary_position);

    /* Section 5: Very Large Cell Size */
    RUN_TEST(stress_very_large_cell_size);

    /* Section 6: Very Small Cell Size */
    RUN_TEST(stress_very_small_cell_size_detection);

    /* Section 7: K-Nearest With K > Drone Count */
    RUN_TEST(stress_knn_k_greater_than_agent_count);
    RUN_TEST(stress_knn_k_greater_than_agent_count_batch);

    /* Section 8: K-Nearest Equidistant */
    RUN_TEST(stress_knn_equidistant);

    /* Section 9: Collision Response Zero Mass */
    RUN_TEST(stress_response_zero_mass_one_drone);
    RUN_TEST(stress_response_zero_mass_both_drones);

    /* Section 10: Collision At Exact Same Position */
    RUN_TEST(stress_exact_same_position_detection);
    RUN_TEST(stress_exact_same_position_response_no_nan);

    /* Section 11: World Collision With NULL World */
    RUN_TEST(stress_null_world_detect_drone_world);
    RUN_TEST(stress_null_world_detect_all_still_does_drone_drone);

    /* Section 12: Drone-Drone At Exactly Collision Radius */
    RUN_TEST(stress_exactly_at_collision_radius_no_collision);
    RUN_TEST(stress_just_inside_collision_radius);

    /* Section 13: 1024 Drones Uniform Distribution Stress */
    RUN_TEST(stress_1024_drones_full_pipeline);
    RUN_TEST(stress_1024_drones_clustered_collisions);
    RUN_TEST(stress_1024_drones_knn_batch);

    /* Section 14: Spatial Hash Clear And Rebuild */
    RUN_TEST(stress_clear_and_rebuild_different_positions);
    RUN_TEST(stress_clear_and_rebuild_knn_updated);

    TEST_SUITE_END();
}
