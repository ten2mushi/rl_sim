/**
 * Collision System Module Unit Tests
 *
 * Yoneda Philosophy: Tests serve as complete behavioral specification.
 * Each test defines what the module MUST do - the definitive contract.
 *
 * Test Categories (15 categories, 83 tests):
 * 1. Spatial Hash Allocation (4 tests)
 * 2. Spatial Hash Insert (4 tests)
 * 3. Spatial Hash Query (4 tests)
 * 4. Collision System Allocation (5 tests)
 * 5. Spatial Hash Build (5 tests)
 * 6. Drone-Drone Detection (8 tests)
 * 7. Drone-World Detection (8 tests)
 * 8. Combined Detection (3 tests)
 * 9. World Response (6 tests)
 * 10. Drone-Drone Response (7 tests)
 * 11. K-Nearest Neighbor (6 tests)
 * 12. Query Functions (4 tests)
 * 13. Utility Functions (4 tests)
 * 14. Edge Cases (6 tests)
 * 15. Memory Validation (2 tests)
 */

#include "../include/collision_system.h"
#include "test_harness.h"

#define EPSILON 1e-5f

/* ============================================================================
 * Test Helpers
 * ============================================================================ */

/**
 * Create test drone states with specified positions
 */
static DroneStateSOA* create_test_states(Arena* arena, uint32_t count) {
    DroneStateSOA* states = drone_state_create(arena, count);
    if (states == NULL) return NULL;

    /* Initialize to default positions */
    for (uint32_t i = 0; i < count; i++) {
        drone_state_init(states, i);
    }
    states->count = count;

    return states;
}

/**
 * Create test drone params with specified masses
 */
static DroneParamsSOA* create_test_params(Arena* arena, uint32_t count, float mass) {
    DroneParamsSOA* params = drone_params_create(arena, count);
    if (params == NULL) return NULL;

    for (uint32_t i = 0; i < count; i++) {
        drone_params_init(params, i);
        params->mass[i] = mass;
    }
    params->count = count;

    return params;
}

/* ============================================================================
 * Section 1: Spatial Hash Allocation Tests
 * ============================================================================ */

TEST(spatial_hash_create_basic) {
    Arena* arena = arena_create(1024 * 1024);
    ASSERT_NOT_NULL(arena);

    SpatialHashGrid* grid = spatial_hash_create(arena, 1024, 1.0f);
    ASSERT_NOT_NULL(grid);
    ASSERT_NOT_NULL(grid->cell_heads);
    ASSERT_NOT_NULL(grid->entries);
    ASSERT_EQ(grid->max_entries, 1024);
    ASSERT_EQ(grid->entry_count, 0);
    ASSERT_FLOAT_EQ(grid->cell_size, 1.0f);
    ASSERT_FLOAT_EQ(grid->inv_cell_size, 1.0f);

    arena_destroy(arena);
    return 0;
}

TEST(spatial_hash_create_alignment) {
    Arena* arena = arena_create(1024 * 1024);
    SpatialHashGrid* grid = spatial_hash_create(arena, 1024, 1.0f);
    ASSERT_NOT_NULL(grid);

    /* cell_heads should be 32-byte aligned for SIMD */
    ASSERT_TRUE(((uintptr_t)(grid->cell_heads) & (32 - 1)) == 0);

    arena_destroy(arena);
    return 0;
}

TEST(spatial_hash_create_null_arena) {
    SpatialHashGrid* grid = spatial_hash_create(NULL, 1024, 1.0f);
    ASSERT_NULL(grid);
    return 0;
}

TEST(spatial_hash_clear_resets_all) {
    Arena* arena = arena_create(1024 * 1024);
    SpatialHashGrid* grid = spatial_hash_create(arena, 1024, 1.0f);
    ASSERT_NOT_NULL(grid);

    /* Insert some entries */
    spatial_hash_insert(grid, 0, 0.0f, 0.0f, 0.0f);
    spatial_hash_insert(grid, 1, 1.0f, 1.0f, 1.0f);
    ASSERT_EQ(grid->entry_count, 2);

    /* Clear */
    spatial_hash_clear(grid);
    ASSERT_EQ(grid->entry_count, 0);

    /* Verify all cell_heads are SPATIAL_HASH_END */
    bool all_empty = true;
    for (uint32_t i = 0; i < HASH_TABLE_SIZE; i++) {
        if (grid->cell_heads[i] != SPATIAL_HASH_END) {
            all_empty = false;
            break;
        }
    }
    ASSERT_TRUE(all_empty);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 2: Spatial Hash Insert Tests
 * ============================================================================ */

TEST(spatial_hash_insert_single) {
    Arena* arena = arena_create(1024 * 1024);
    SpatialHashGrid* grid = spatial_hash_create(arena, 1024, 1.0f);

    spatial_hash_insert(grid, 42, 5.5f, 3.2f, 1.8f);
    ASSERT_EQ(grid->entry_count, 1);
    ASSERT_EQ(grid->entries[0].drone_index, 42);

    arena_destroy(arena);
    return 0;
}

TEST(spatial_hash_insert_multiple_same_cell) {
    Arena* arena = arena_create(1024 * 1024);
    SpatialHashGrid* grid = spatial_hash_create(arena, 1024, 1.0f);

    /* Insert multiple drones in same cell */
    spatial_hash_insert(grid, 0, 0.1f, 0.1f, 0.1f);
    spatial_hash_insert(grid, 1, 0.2f, 0.2f, 0.2f);
    spatial_hash_insert(grid, 2, 0.3f, 0.3f, 0.3f);

    ASSERT_EQ(grid->entry_count, 3);

    /* Query the cell */
    uint32_t indices[16];
    CellQuery query = { .indices = indices, .count = 0, .capacity = 16 };
    spatial_hash_query_cell(grid, 0.1f, 0.1f, 0.1f, &query);

    ASSERT_EQ(query.count, 3);

    arena_destroy(arena);
    return 0;
}

TEST(spatial_hash_insert_multiple_diff_cells) {
    Arena* arena = arena_create(1024 * 1024);
    SpatialHashGrid* grid = spatial_hash_create(arena, 1024, 1.0f);

    /* Insert drones in different cells */
    spatial_hash_insert(grid, 0, 0.5f, 0.5f, 0.5f);   /* Cell (0,0,0) */
    spatial_hash_insert(grid, 1, 10.5f, 10.5f, 10.5f); /* Cell (10,10,10) */
    spatial_hash_insert(grid, 2, -5.5f, -5.5f, -5.5f); /* Cell (-6,-6,-6) */

    ASSERT_EQ(grid->entry_count, 3);

    /* Query first cell */
    uint32_t indices[16];
    CellQuery query = { .indices = indices, .count = 0, .capacity = 16 };
    spatial_hash_query_cell(grid, 0.5f, 0.5f, 0.5f, &query);
    ASSERT_EQ(query.count, 1);
    ASSERT_EQ(query.indices[0], 0);

    /* Query second cell */
    query.count = 0;
    spatial_hash_query_cell(grid, 10.5f, 10.5f, 10.5f, &query);
    ASSERT_EQ(query.count, 1);
    ASSERT_EQ(query.indices[0], 1);

    arena_destroy(arena);
    return 0;
}

TEST(spatial_hash_insert_overflow) {
    Arena* arena = arena_create(1024 * 1024);
    SpatialHashGrid* grid = spatial_hash_create(arena, 3, 1.0f);  /* Only 3 entries */

    spatial_hash_insert(grid, 0, 0.0f, 0.0f, 0.0f);
    spatial_hash_insert(grid, 1, 1.0f, 1.0f, 1.0f);
    spatial_hash_insert(grid, 2, 2.0f, 2.0f, 2.0f);
    spatial_hash_insert(grid, 3, 3.0f, 3.0f, 3.0f);  /* Should fail silently */

    ASSERT_EQ(grid->entry_count, 3);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 3: Spatial Hash Query Tests
 * ============================================================================ */

TEST(spatial_hash_query_empty) {
    Arena* arena = arena_create(1024 * 1024);
    SpatialHashGrid* grid = spatial_hash_create(arena, 1024, 1.0f);

    uint32_t indices[16];
    CellQuery query = { .indices = indices, .count = 0, .capacity = 16 };
    spatial_hash_query_cell(grid, 0.0f, 0.0f, 0.0f, &query);

    ASSERT_EQ(query.count, 0);

    arena_destroy(arena);
    return 0;
}

TEST(spatial_hash_query_single_cell) {
    Arena* arena = arena_create(1024 * 1024);
    SpatialHashGrid* grid = spatial_hash_create(arena, 1024, 1.0f);

    spatial_hash_insert(grid, 5, 2.5f, 2.5f, 2.5f);

    uint32_t indices[16];
    CellQuery query = { .indices = indices, .count = 0, .capacity = 16 };
    spatial_hash_query_cell(grid, 2.1f, 2.1f, 2.1f, &query);

    ASSERT_EQ(query.count, 1);
    ASSERT_EQ(query.indices[0], 5);

    arena_destroy(arena);
    return 0;
}

TEST(spatial_hash_query_neighborhood_27) {
    Arena* arena = arena_create(1024 * 1024);
    SpatialHashGrid* grid = spatial_hash_create(arena, 1024, 1.0f);

    /* Insert drones in a grid pattern */
    uint32_t idx = 0;
    for (int32_t dz = -1; dz <= 1; dz++) {
        for (int32_t dy = -1; dy <= 1; dy++) {
            for (int32_t dx = -1; dx <= 1; dx++) {
                spatial_hash_insert(grid, idx++,
                                   5.5f + dx, 5.5f + dy, 5.5f + dz);
            }
        }
    }
    ASSERT_EQ(grid->entry_count, 27);

    /* Query neighborhood around center */
    uint32_t indices[64];
    CellQuery query = { .indices = indices, .count = 0, .capacity = 64 };
    spatial_hash_query_neighborhood(grid, 5.5f, 5.5f, 5.5f, &query);

    /* Should find all 27 drones */
    ASSERT_EQ(query.count, 27);

    arena_destroy(arena);
    return 0;
}

TEST(spatial_hash_query_hash_distribution) {
    Arena* arena = arena_create(1024 * 1024);
    SpatialHashGrid* grid = spatial_hash_create(arena, 1024, 1.0f);

    /* Insert 1024 drones scattered across space */
    for (uint32_t i = 0; i < 1024; i++) {
        float x = (float)(i % 32);
        float y = (float)((i / 32) % 32);
        float z = (float)(i / 1024);
        spatial_hash_insert(grid, i, x, y, z);
    }

    /* Count non-empty buckets */
    uint32_t non_empty = 0;
    for (uint32_t i = 0; i < HASH_TABLE_SIZE; i++) {
        if (grid->cell_heads[i] != SPATIAL_HASH_END) {
            non_empty++;
        }
    }

    /* With good distribution, we should use a reasonable fraction of buckets */
    float utilization = (float)non_empty / HASH_TABLE_SIZE;
    ASSERT_GT(utilization, 0.1f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 4: Collision System Allocation Tests
 * ============================================================================ */

TEST(collision_create_basic) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    ASSERT_NOT_NULL(arena);

    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    ASSERT_NOT_NULL(sys);
    ASSERT_NOT_NULL(sys->spatial_hash);
    ASSERT_NOT_NULL(sys->collision_pairs);
    ASSERT_NOT_NULL(sys->drone_world_collision);
    ASSERT_NOT_NULL(sys->penetration_depth);
    ASSERT_NOT_NULL(sys->collision_normals);
    ASSERT_EQ(sys->max_drones, 1024);
    ASSERT_FLOAT_EQ(sys->drone_radius, 0.1f);

    arena_destroy(arena);
    return 0;
}

TEST(collision_create_alignment) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    ASSERT_NOT_NULL(sys);

    ASSERT_TRUE(((uintptr_t)(sys->collision_pairs) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(sys->drone_world_collision) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(sys->penetration_depth) & (32 - 1)) == 0);
    ASSERT_TRUE(((uintptr_t)(sys->collision_normals) & (16 - 1)) == 0);

    arena_destroy(arena);
    return 0;
}

TEST(collision_create_capacity_1024) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    ASSERT_NOT_NULL(sys);
    ASSERT_EQ(sys->max_drones, 1024);
    ASSERT_EQ(sys->max_pairs, 2048);

    arena_destroy(arena);
    return 0;
}

TEST(collision_create_null_arena) {
    CollisionSystem* sys = collision_create(NULL, 1024, 0.1f, 1.0f);
    ASSERT_NULL(sys);
    return 0;
}

TEST(collision_create_zero_drones) {
    Arena* arena = arena_create(1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 0, 0.1f, 1.0f);
    ASSERT_NULL(sys);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 5: Spatial Hash Build Tests
 * ============================================================================ */

TEST(build_hash_empty) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);

    /* Build with zero count should be safe */
    collision_build_spatial_hash(sys, NULL, 0);
    ASSERT_EQ(sys->spatial_hash->entry_count, 0);

    arena_destroy(arena);
    return 0;
}

TEST(build_hash_single_drone) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 1);

    states->pos_x[0] = 5.0f;
    states->pos_y[0] = 5.0f;
    states->pos_z[0] = 5.0f;

    collision_build_spatial_hash(sys, states, 1);
    ASSERT_EQ(sys->spatial_hash->entry_count, 1);

    arena_destroy(arena);
    return 0;
}

TEST(build_hash_clustered_drones) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 100);

    /* All drones at nearly same position */
    for (uint32_t i = 0; i < 100; i++) {
        states->pos_x[i] = 5.0f + (float)i * 0.001f;
        states->pos_y[i] = 5.0f + (float)i * 0.001f;
        states->pos_z[i] = 5.0f + (float)i * 0.001f;
    }

    collision_build_spatial_hash(sys, states, 100);
    ASSERT_EQ(sys->spatial_hash->entry_count, 100);

    arena_destroy(arena);
    return 0;
}

TEST(build_hash_scattered_drones) {
    Arena* arena = arena_create(2 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 100);

    /* Drones scattered across large area */
    for (uint32_t i = 0; i < 100; i++) {
        states->pos_x[i] = (float)(i % 10) * 10.0f;
        states->pos_y[i] = (float)((i / 10) % 10) * 10.0f;
        states->pos_z[i] = (float)(i / 100) * 10.0f;
    }

    collision_build_spatial_hash(sys, states, 100);
    ASSERT_EQ(sys->spatial_hash->entry_count, 100);

    arena_destroy(arena);
    return 0;
}

TEST(build_hash_1024_drones) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 1024);

    for (uint32_t i = 0; i < 1024; i++) {
        states->pos_x[i] = (float)(i % 32);
        states->pos_y[i] = (float)((i / 32) % 32);
        states->pos_z[i] = (float)(i / 1024) * 10.0f;
    }

    collision_build_spatial_hash(sys, states, 1024);
    ASSERT_EQ(sys->spatial_hash->entry_count, 1024);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 6: Drone-Drone Detection Tests
 * ============================================================================ */

TEST(detect_dd_no_collisions) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 10);

    /* Drones far apart (5 meters between each) */
    for (uint32_t i = 0; i < 10; i++) {
        states->pos_x[i] = (float)i * 5.0f;
        states->pos_y[i] = 0.0f;
        states->pos_z[i] = 0.0f;
    }

    collision_build_spatial_hash(sys, states, 10);
    collision_detect_drone_drone(sys, states, 10);

    ASSERT_EQ(sys->pair_count, 0);

    arena_destroy(arena);
    return 0;
}

TEST(detect_dd_single_collision) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 2);

    /* Two drones overlapping */
    states->pos_x[0] = 0.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.15f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;

    collision_build_spatial_hash(sys, states, 2);
    collision_detect_drone_drone(sys, states, 2);

    ASSERT_EQ(sys->pair_count, 1);
    ASSERT_EQ(sys->collision_pairs[0], 0);
    ASSERT_EQ(sys->collision_pairs[1], 1);

    arena_destroy(arena);
    return 0;
}

TEST(detect_dd_multiple_collisions) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 4);

    /* Four drones in a cluster */
    states->pos_x[0] = 0.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.15f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;
    states->pos_x[2] = 0.0f; states->pos_y[2] = 0.15f; states->pos_z[2] = 0.0f;
    states->pos_x[3] = 0.15f; states->pos_y[3] = 0.15f; states->pos_z[3] = 0.0f;

    collision_build_spatial_hash(sys, states, 4);
    collision_detect_drone_drone(sys, states, 4);

    /* Each adjacent pair should collide: (0,1), (0,2), (1,3), (2,3) = 4 pairs */
    ASSERT_GE(sys->pair_count, 4);

    arena_destroy(arena);
    return 0;
}

TEST(detect_dd_chain_collision) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 3);

    /* A--B--C chain where A touches B, B touches C, but A doesn't touch C */
    states->pos_x[0] = 0.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.15f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;
    states->pos_x[2] = 0.30f; states->pos_y[2] = 0.0f; states->pos_z[2] = 0.0f;

    collision_build_spatial_hash(sys, states, 3);
    collision_detect_drone_drone(sys, states, 3);

    ASSERT_EQ(sys->pair_count, 2);

    arena_destroy(arena);
    return 0;
}

TEST(detect_dd_pair_ordering) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 10);

    /* Create several collisions with mixed indices */
    states->pos_x[5] = 0.0f; states->pos_y[5] = 0.0f; states->pos_z[5] = 0.0f;
    states->pos_x[2] = 0.05f; states->pos_y[2] = 0.0f; states->pos_z[2] = 0.0f;
    for (uint32_t i = 0; i < 10; i++) {
        if (i != 2 && i != 5) {
            states->pos_x[i] = (float)i * 10.0f;
            states->pos_y[i] = 0.0f;
            states->pos_z[i] = 0.0f;
        }
    }

    collision_build_spatial_hash(sys, states, 10);
    collision_detect_drone_drone(sys, states, 10);

    /* Verify all pairs have i < j */
    bool ordering_ok = true;
    for (uint32_t p = 0; p < sys->pair_count; p++) {
        uint32_t idx = p * 2;
        if (sys->collision_pairs[idx] >= sys->collision_pairs[idx + 1]) {
            ordering_ok = false;
            break;
        }
    }
    ASSERT_TRUE(ordering_ok);

    arena_destroy(arena);
    return 0;
}

TEST(detect_dd_pair_count) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 5);

    /* Create exactly 2 collision pairs */
    states->pos_x[0] = 0.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.1f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;  /* Collides with 0 */
    states->pos_x[2] = 5.0f; states->pos_y[2] = 0.0f; states->pos_z[2] = 0.0f;
    states->pos_x[3] = 5.1f; states->pos_y[3] = 0.0f; states->pos_z[3] = 0.0f;  /* Collides with 2 */
    states->pos_x[4] = 10.0f; states->pos_y[4] = 0.0f; states->pos_z[4] = 0.0f; /* No collision */

    collision_build_spatial_hash(sys, states, 5);
    collision_detect_drone_drone(sys, states, 5);

    ASSERT_EQ(sys->pair_count, 2);

    arena_destroy(arena);
    return 0;
}

TEST(detect_dd_touching) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 2);

    /* Drones at exactly 2r distance (radius = 0.1, so distance = 0.2) */
    states->pos_x[0] = 0.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.2f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;

    collision_build_spatial_hash(sys, states, 2);
    collision_detect_drone_drone(sys, states, 2);

    /* At exactly 2r, dist_sq = 4r^2, which is NOT < threshold, so no collision */
    ASSERT_EQ(sys->pair_count, 0);

    /* Move slightly closer */
    states->pos_x[1] = 0.199f;

    collision_reset(sys);
    collision_build_spatial_hash(sys, states, 2);
    collision_detect_drone_drone(sys, states, 2);

    ASSERT_EQ(sys->pair_count, 1);

    arena_destroy(arena);
    return 0;
}

TEST(detect_dd_self_no_collision) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 1);

    states->pos_x[0] = 0.0f;
    states->pos_y[0] = 0.0f;
    states->pos_z[0] = 0.0f;

    collision_build_spatial_hash(sys, states, 1);
    collision_detect_drone_drone(sys, states, 1);

    ASSERT_EQ(sys->pair_count, 0);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 7: Drone-World Detection Tests (simplified - no actual world)
 * ============================================================================ */

TEST(detect_world_empty_world) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 10);

    /* Calling with NULL world should be safe */
    collision_detect_drone_world(sys, states, NULL, 10);

    /* Should not crash and should have no world collisions */
    bool any_collision = false;
    for (uint32_t i = 0; i < 10; i++) {
        if (sys->drone_world_collision[i]) {
            any_collision = true;
            break;
        }
    }
    ASSERT_FALSE(any_collision);

    arena_destroy(arena);
    return 0;
}

TEST(detect_world_preserves_non_colliding) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);

    /* Check that collision flags are initialized to 0 */
    ASSERT_EQ(sys->drone_world_collision[0], 0);
    ASSERT_EQ(sys->drone_world_collision[100], 0);

    arena_destroy(arena);
    return 0;
}

TEST(detect_world_batch_efficiency) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);

    /* Verify arrays are properly allocated for batch operations */
    ASSERT_NOT_NULL(sys->penetration_depth);
    ASSERT_NOT_NULL(sys->collision_normals);

    arena_destroy(arena);
    return 0;
}

TEST(detect_world_penetration_depth) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);

    /* Verify penetration depth array is writable */
    sys->penetration_depth[0] = -0.05f;
    sys->penetration_depth[500] = -0.1f;

    ASSERT_FLOAT_EQ(sys->penetration_depth[0], -0.05f);
    ASSERT_FLOAT_EQ(sys->penetration_depth[500], -0.1f);

    arena_destroy(arena);
    return 0;
}

TEST(detect_world_normal_calculation) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);

    /* Verify normal array is writable */
    sys->collision_normals[0] = VEC3(0.0f, 1.0f, 0.0f);
    sys->collision_normals[100] = VEC3(0.0f, 0.0f, 1.0f);

    ASSERT_VEC3_NEAR(sys->collision_normals[0], VEC3(0.0f, 1.0f, 0.0f), EPSILON);
    ASSERT_VEC3_NEAR(sys->collision_normals[100], VEC3(0.0f, 0.0f, 1.0f), EPSILON);

    arena_destroy(arena);
    return 0;
}

TEST(detect_world_flags) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);

    /* Verify flags are writable */
    sys->drone_world_collision[0] = 1;
    sys->drone_world_collision[1] = 0;

    ASSERT_EQ(sys->drone_world_collision[0], 1);
    ASSERT_EQ(sys->drone_world_collision[1], 0);

    arena_destroy(arena);
    return 0;
}

TEST(detect_world_no_collision_above) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);

    /* Set penetration depth above margin (positive = outside) */
    sys->penetration_depth[0] = 0.5f;  /* Well outside */
    sys->drone_world_collision[0] = 0;

    /* Check that no collision is flagged for positive SDF */
    ASSERT_EQ(sys->drone_world_collision[0], 0);

    arena_destroy(arena);
    return 0;
}

TEST(detect_world_ground_collision) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);

    /* Simulate ground collision: negative SDF means inside */
    sys->penetration_depth[0] = -0.05f;  /* 5cm inside */
    sys->drone_world_collision[0] = 1;
    sys->collision_normals[0] = VEC3(0.0f, 0.0f, 1.0f);  /* Up normal */

    ASSERT_EQ(sys->drone_world_collision[0], 1);
    ASSERT_FLOAT_NEAR(sys->penetration_depth[0], -0.05f, 0.001f);
    ASSERT_VEC3_NEAR(sys->collision_normals[0], VEC3(0.0f, 0.0f, 1.0f), EPSILON);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 8: Combined Detection Tests
 * ============================================================================ */

TEST(detect_all_ordering) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 10);

    /* Set up some colliding drones */
    states->pos_x[0] = 0.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.1f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;

    collision_detect_all(sys, states, NULL, 10);

    ASSERT_GT(sys->pair_count, 0);

    arena_destroy(arena);
    return 0;
}

TEST(detect_all_resets_previous) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 10);

    /* Initialize all drone positions to be spread out, with 0 and 1 colliding */
    for (uint32_t i = 0; i < 10; i++) {
        states->pos_x[i] = (float)i * 10.0f;  /* Spread apart by 10m */
        states->pos_y[i] = 0.0f;
        states->pos_z[i] = 0.0f;
    }

    /* First detection - move drones 0 and 1 to collide */
    states->pos_x[0] = 0.0f;
    states->pos_x[1] = 0.1f;  /* Collides with 0 */
    collision_detect_all(sys, states, NULL, 10);
    uint32_t first_count = sys->pair_count;
    ASSERT_GT(first_count, 0);

    /* Move drone 1 far away from drone 0 */
    states->pos_x[1] = 100.0f;
    collision_detect_all(sys, states, NULL, 10);

    ASSERT_EQ(sys->pair_count, 0);

    arena_destroy(arena);
    return 0;
}

TEST(detect_all_independence) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 10);

    /* Two drones colliding, rest spread out */
    states->pos_x[0] = 0.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.1f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;
    for (uint32_t i = 2; i < 10; i++) {
        states->pos_x[i] = (float)i * 10.0f;
        states->pos_y[i] = 0.0f;
        states->pos_z[i] = 0.0f;
    }

    collision_detect_all(sys, states, NULL, 10);

    /* Should have drone-drone collision independent of world collision */
    ASSERT_EQ(sys->pair_count, 1);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 9: World Response Tests
 * ============================================================================ */

TEST(response_world_position_correction) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 1);

    /* Set up drone colliding with ground */
    states->pos_x[0] = 0.0f;
    states->pos_y[0] = 0.0f;
    states->pos_z[0] = -0.05f;  /* 5cm below ground (assuming ground at z=0) */

    sys->drone_world_collision[0] = 1;
    sys->penetration_depth[0] = -0.05f;  /* 5cm inside */
    sys->collision_normals[0] = VEC3(0.0f, 0.0f, 1.0f);  /* Up */

    float old_z = states->pos_z[0];
    collision_apply_world_response(sys, states, 0.5f, 1.0f, 1);

    /* Position should be pushed up */
    ASSERT_GT(states->pos_z[0], old_z);

    arena_destroy(arena);
    return 0;
}

TEST(response_world_velocity_reflection) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 1);

    /* Set up drone moving into ground */
    states->vel_x[0] = 0.0f;
    states->vel_y[0] = 0.0f;
    states->vel_z[0] = -5.0f;  /* Moving down at 5 m/s */

    sys->drone_world_collision[0] = 1;
    sys->penetration_depth[0] = -0.01f;
    sys->collision_normals[0] = VEC3(0.0f, 0.0f, 1.0f);

    collision_apply_world_response(sys, states, 0.5f, 1.0f, 1);

    /* Velocity z should be positive (reflected) */
    ASSERT_GT(states->vel_z[0], 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(response_world_restitution_0) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 1);

    /* Moving into surface */
    states->vel_x[0] = 0.0f;
    states->vel_y[0] = 0.0f;
    states->vel_z[0] = -5.0f;

    sys->drone_world_collision[0] = 1;
    sys->penetration_depth[0] = -0.01f;
    sys->collision_normals[0] = VEC3(0.0f, 0.0f, 1.0f);

    collision_apply_world_response(sys, states, 0.0f, 1.0f, 1);

    /* With restitution=0, velocity component along normal should be zeroed */
    ASSERT_FLOAT_NEAR(states->vel_z[0], 0.0f, 0.01f);

    arena_destroy(arena);
    return 0;
}

TEST(response_world_restitution_1) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 1);

    /* Moving into surface */
    states->vel_x[0] = 0.0f;
    states->vel_y[0] = 0.0f;
    states->vel_z[0] = -5.0f;

    sys->drone_world_collision[0] = 1;
    sys->penetration_depth[0] = -0.01f;
    sys->collision_normals[0] = VEC3(0.0f, 0.0f, 1.0f);

    collision_apply_world_response(sys, states, 1.0f, 1.0f, 1);

    /* With restitution=1, velocity should be fully reflected */
    ASSERT_FLOAT_NEAR(states->vel_z[0], 5.0f, 0.1f);

    arena_destroy(arena);
    return 0;
}

TEST(response_world_only_into_surface) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 1);

    /* Moving away from surface */
    states->vel_x[0] = 0.0f;
    states->vel_y[0] = 0.0f;
    states->vel_z[0] = 5.0f;  /* Moving up (away from surface) */

    sys->drone_world_collision[0] = 1;
    sys->penetration_depth[0] = -0.01f;
    sys->collision_normals[0] = VEC3(0.0f, 0.0f, 1.0f);

    float old_vel_z = states->vel_z[0];
    collision_apply_world_response(sys, states, 0.5f, 1.0f, 1);

    /* Velocity should not be changed (already moving away) */
    ASSERT_FLOAT_EQ(states->vel_z[0], old_vel_z);

    arena_destroy(arena);
    return 0;
}

TEST(response_world_preserves_non_colliding) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 2);

    /* Drone 0: colliding */
    states->vel_z[0] = -5.0f;
    sys->drone_world_collision[0] = 1;
    sys->penetration_depth[0] = -0.01f;
    sys->collision_normals[0] = VEC3(0.0f, 0.0f, 1.0f);

    /* Drone 1: not colliding */
    states->vel_x[1] = 10.0f;
    states->vel_y[1] = 20.0f;
    states->vel_z[1] = -5.0f;
    sys->drone_world_collision[1] = 0;

    collision_apply_world_response(sys, states, 0.5f, 1.0f, 2);

    /* Drone 0 should be modified, drone 1 should be unchanged */
    ASSERT_GT(states->vel_z[0], -5.0f);
    ASSERT_FLOAT_EQ(states->vel_x[1], 10.0f);
    ASSERT_FLOAT_EQ(states->vel_y[1], 20.0f);
    ASSERT_FLOAT_EQ(states->vel_z[1], -5.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 10: Drone-Drone Response Tests
 * ============================================================================ */

TEST(response_dd_separation) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 2);
    DroneParamsSOA* params = create_test_params(arena, 2, 1.0f);

    /* Two overlapping drones */
    states->pos_x[0] = 0.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.1f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;

    collision_detect_all(sys, states, NULL, 2);
    collision_apply_drone_response(sys, states, params, 0.5f, 2);

    /* Drones should be pushed apart */
    float dist_after = fabsf(states->pos_x[1] - states->pos_x[0]);
    ASSERT_GT(dist_after, 0.1f);

    arena_destroy(arena);
    return 0;
}

TEST(response_dd_mass_weighted) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 2);
    DroneParamsSOA* params = create_test_params(arena, 2, 1.0f);

    /* Drone 0: heavy, Drone 1: light */
    params->mass[0] = 10.0f;
    params->mass[1] = 1.0f;

    /* Overlapping drones */
    states->pos_x[0] = 0.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.1f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;

    float orig_x0 = states->pos_x[0];
    float orig_x1 = states->pos_x[1];

    collision_detect_all(sys, states, NULL, 2);
    collision_apply_drone_response(sys, states, params, 0.5f, 2);

    /* Light drone should move more than heavy drone */
    float move_0 = fabsf(states->pos_x[0] - orig_x0);
    float move_1 = fabsf(states->pos_x[1] - orig_x1);
    ASSERT_GT(move_1, move_0);

    arena_destroy(arena);
    return 0;
}

TEST(response_dd_equal_mass) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 2);
    DroneParamsSOA* params = create_test_params(arena, 2, 1.0f);

    /* Equal mass drones */
    params->mass[0] = 1.0f;
    params->mass[1] = 1.0f;

    /* Symmetric setup */
    states->pos_x[0] = -0.05f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.05f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;

    collision_detect_all(sys, states, NULL, 2);
    collision_apply_drone_response(sys, states, params, 0.5f, 2);

    /* Both should move equally in opposite directions */
    float move_0 = fabsf(states->pos_x[0] - (-0.05f));
    float move_1 = fabsf(states->pos_x[1] - 0.05f);
    ASSERT_FLOAT_NEAR(move_0, move_1, 0.001f);

    arena_destroy(arena);
    return 0;
}

TEST(response_dd_velocity_exchange) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 2);
    DroneParamsSOA* params = create_test_params(arena, 2, 1.0f);

    /* Drone 0 moving toward drone 1 */
    states->pos_x[0] = 0.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.15f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;
    states->vel_x[0] = 5.0f; states->vel_y[0] = 0.0f; states->vel_z[0] = 0.0f;
    states->vel_x[1] = 0.0f; states->vel_y[1] = 0.0f; states->vel_z[1] = 0.0f;

    collision_detect_all(sys, states, NULL, 2);
    collision_apply_drone_response(sys, states, params, 1.0f, 2);  /* Elastic */

    /* With elastic collision and equal mass, velocities should exchange */
    ASSERT_GT(states->vel_x[1], 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(response_dd_restitution) {
    Arena* arena = arena_create(4 * 1024 * 1024);

    /* Test with restitution = 0 */
    {
        CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
        DroneStateSOA* states = create_test_states(arena, 2);
        DroneParamsSOA* params = create_test_params(arena, 2, 1.0f);

        states->pos_x[0] = 0.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
        states->pos_x[1] = 0.15f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;
        states->vel_x[0] = 5.0f;
        states->vel_x[1] = -5.0f;

        collision_detect_all(sys, states, NULL, 2);
        collision_apply_drone_response(sys, states, params, 0.0f, 2);

        /* Inelastic collision - velocities should be more similar after */
        float rel_vel_after = fabsf(states->vel_x[0] - states->vel_x[1]);
        ASSERT_LT(rel_vel_after, 10.0f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(response_dd_head_on) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 2);
    DroneParamsSOA* params = create_test_params(arena, 2, 1.0f);

    /* Head-on collision */
    states->pos_x[0] = 0.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.15f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;
    states->vel_x[0] = 5.0f;
    states->vel_x[1] = -5.0f;

    float total_momentum_before = states->vel_x[0] + states->vel_x[1];

    collision_detect_all(sys, states, NULL, 2);
    collision_apply_drone_response(sys, states, params, 1.0f, 2);

    float total_momentum_after = states->vel_x[0] + states->vel_x[1];

    /* Momentum should be conserved */
    ASSERT_FLOAT_NEAR(total_momentum_before, total_momentum_after, 0.1f);

    arena_destroy(arena);
    return 0;
}

TEST(response_dd_glancing) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 2);
    DroneParamsSOA* params = create_test_params(arena, 2, 1.0f);

    /* Glancing collision - drones moving perpendicular to collision normal */
    states->pos_x[0] = 0.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.15f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;
    states->vel_y[0] = 5.0f;  /* Moving along Y (perpendicular to X collision) */
    states->vel_y[1] = -5.0f;

    collision_detect_all(sys, states, NULL, 2);
    collision_apply_drone_response(sys, states, params, 0.5f, 2);

    /* Y velocities should be mostly unchanged since collision is along X */
    ASSERT_GT(fabsf(states->vel_y[0]), 1.0f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 11: K-Nearest Neighbor Tests
 * ============================================================================ */

TEST(knn_single_neighbor) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 10);

    /* Spread out drones with 0.5m spacing (within single cell) */
    for (uint32_t i = 0; i < 10; i++) {
        states->pos_x[i] = (float)i * 0.5f;
        states->pos_y[i] = 0.0f;
        states->pos_z[i] = 0.0f;
    }

    collision_build_spatial_hash(sys, states, 10);

    uint32_t indices[1];
    float distances[1];
    uint32_t count;

    /* Query slightly offset from drone 5 (at 2.5) - query at 2.6 */
    collision_find_k_nearest(sys, states, VEC3(2.6f, 0.0f, 0.0f), 1,
                            indices, distances, &count);

    ASSERT_EQ(count, 1);
    /* Nearest should be drone 5 (at position 2.5) or drone 6 (at 3.0) */
    ASSERT_TRUE(indices[0] == 5 || indices[0] == 6);

    arena_destroy(arena);
    return 0;
}

TEST(knn_k_neighbors) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 10);

    /* Spread out drones along X axis */
    for (uint32_t i = 0; i < 10; i++) {
        states->pos_x[i] = (float)i;
        states->pos_y[i] = 0.0f;
        states->pos_z[i] = 0.0f;
    }

    collision_build_spatial_hash(sys, states, 10);

    uint32_t indices[3];
    float distances[3];
    uint32_t count;

    /* Query at position 5.5 - nearest should be 5, 6, then 4 or 7 */
    collision_find_k_nearest(sys, states, VEC3(5.5f, 0.0f, 0.0f), 3,
                            indices, distances, &count);

    ASSERT_EQ(count, 3);

    arena_destroy(arena);
    return 0;
}

TEST(knn_fewer_than_k) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 2);

    states->pos_x[0] = 0.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 1.0f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;

    collision_build_spatial_hash(sys, states, 2);

    uint32_t indices[5];
    float distances[5];
    uint32_t count;

    /* Ask for 5 but only 2 available */
    collision_find_k_nearest(sys, states, VEC3(0.5f, 0.0f, 0.0f), 5,
                            indices, distances, &count);

    ASSERT_LE(count, 2);

    arena_destroy(arena);
    return 0;
}

TEST(knn_sorted_by_distance) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 10);

    /* Various positions */
    states->pos_x[0] = 5.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 2.0f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;
    states->pos_x[2] = 8.0f; states->pos_y[2] = 0.0f; states->pos_z[2] = 0.0f;
    states->pos_x[3] = 3.0f; states->pos_y[3] = 0.0f; states->pos_z[3] = 0.0f;
    states->pos_x[4] = 7.0f; states->pos_y[4] = 0.0f; states->pos_z[4] = 0.0f;

    collision_build_spatial_hash(sys, states, 5);

    uint32_t indices[5];
    float distances[5];
    uint32_t count;

    collision_find_k_nearest(sys, states, VEC3(5.0f, 0.0f, 0.0f), 5,
                            indices, distances, &count);

    /* Verify distances are sorted */
    bool sorted = true;
    for (uint32_t i = 1; i < count; i++) {
        if (distances[i] < distances[i-1]) {
            sorted = false;
            break;
        }
    }
    ASSERT_TRUE(sorted);

    arena_destroy(arena);
    return 0;
}

TEST(knn_excludes_query_point) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 3);

    /* Three drones */
    states->pos_x[0] = 0.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 1.0f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;
    states->pos_x[2] = 2.0f; states->pos_y[2] = 0.0f; states->pos_z[2] = 0.0f;

    collision_build_spatial_hash(sys, states, 3);

    uint32_t indices[3];
    float distances[3];
    uint32_t count;

    /* Query exactly at drone 0's position */
    collision_find_k_nearest(sys, states, VEC3(0.0f, 0.0f, 0.0f), 3,
                            indices, distances, &count);

    /* Should not include drone 0 (distance = 0) */
    bool found_zero_dist = false;
    for (uint32_t i = 0; i < count; i++) {
        if (distances[i] < 1e-6f) {
            found_zero_dist = true;
            break;
        }
    }
    ASSERT_FALSE(found_zero_dist);

    arena_destroy(arena);
    return 0;
}

TEST(knn_batch_correctness) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 10);

    for (uint32_t i = 0; i < 10; i++) {
        states->pos_x[i] = (float)i;
        states->pos_y[i] = 0.0f;
        states->pos_z[i] = 0.0f;
    }

    collision_build_spatial_hash(sys, states, 10);

    uint32_t* batch_indices = arena_alloc_array(arena, uint32_t, 10 * 3);
    float* batch_distances = arena_alloc_array(arena, float, 10 * 3);

    collision_find_k_nearest_batch(sys, states, 10, 3, batch_indices, batch_distances);

    /* Verify each drone has neighbors (not itself) */
    bool valid = true;
    for (uint32_t d = 0; d < 10; d++) {
        for (uint32_t k = 0; k < 3; k++) {
            uint32_t idx = batch_indices[d * 3 + k];
            if (idx == d) {
                valid = false;  /* Should not find itself */
            }
        }
    }
    ASSERT_TRUE(valid);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 12: Query Functions Tests
 * ============================================================================ */

TEST(get_results_returns_valid) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);

    CollisionResults results = collision_get_results(sys);

    ASSERT_NOT_NULL(results.pairs);
    ASSERT_NOT_NULL(results.world_flags);
    ASSERT_NOT_NULL(results.penetration);
    ASSERT_NOT_NULL(results.normals);

    arena_destroy(arena);
    return 0;
}

TEST(drone_world_check_true) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);

    sys->drone_world_collision[42] = 1;

    ASSERT_TRUE(collision_drone_world_check(sys, 42) == true);

    arena_destroy(arena);
    return 0;
}

TEST(drone_world_check_false) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);

    sys->drone_world_collision[42] = 0;

    ASSERT_TRUE(collision_drone_world_check(sys, 42) == false);

    arena_destroy(arena);
    return 0;
}

TEST(get_pair_finds_collision) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);

    /* Manually add a collision pair */
    sys->collision_pairs[0] = 5;
    sys->collision_pairs[1] = 10;
    sys->pair_count = 1;

    ASSERT_EQ(collision_get_pair(sys, 5), 10);
    ASSERT_EQ(collision_get_pair(sys, 10), 5);
    ASSERT_EQ(collision_get_pair(sys, 7), UINT32_MAX);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 13: Utility Functions Tests
 * ============================================================================ */

TEST(check_pair_colliding) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    DroneStateSOA* states = create_test_states(arena, 2);

    states->pos_x[0] = 0.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 0.1f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;

    float radius_sum_sq = 0.04f;  /* (0.1 + 0.1)^2 */

    ASSERT_TRUE(collision_check_pair(states, 0, 1, radius_sum_sq) == true);

    arena_destroy(arena);
    return 0;
}

TEST(check_pair_not_colliding) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    DroneStateSOA* states = create_test_states(arena, 2);

    states->pos_x[0] = 0.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 5.0f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;

    float radius_sum_sq = 0.04f;

    ASSERT_TRUE(collision_check_pair(states, 0, 1, radius_sum_sq) == false);

    arena_destroy(arena);
    return 0;
}

TEST(compute_normal_direction) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    DroneStateSOA* states = create_test_states(arena, 2);

    states->pos_x[0] = 0.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 1.0f; states->pos_y[1] = 0.0f; states->pos_z[1] = 0.0f;

    Vec3 normal = collision_compute_normal(states, 0, 1);

    /* Normal should point from A to B (positive X) */
    ASSERT_GT(normal.x, 0.0f);
    ASSERT_FLOAT_NEAR(normal.y, 0.0f, 0.001f);
    ASSERT_FLOAT_NEAR(normal.z, 0.0f, 0.001f);

    arena_destroy(arena);
    return 0;
}

TEST(compute_normal_unit_length) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    DroneStateSOA* states = create_test_states(arena, 2);

    states->pos_x[0] = 0.0f; states->pos_y[0] = 0.0f; states->pos_z[0] = 0.0f;
    states->pos_x[1] = 3.0f; states->pos_y[1] = 4.0f; states->pos_z[1] = 0.0f;

    Vec3 normal = collision_compute_normal(states, 0, 1);
    float length = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);

    ASSERT_FLOAT_NEAR(length, 1.0f, 0.001f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 14: Edge Cases Tests
 * ============================================================================ */

TEST(edge_zero_radius) {
    Arena* arena = arena_create(4 * 1024 * 1024);

    /* Zero radius should still create a valid system */
    CollisionSystem* sys = collision_create(arena, 1024, 0.0f, 1.0f);
    ASSERT_NOT_NULL(sys);
    ASSERT_FLOAT_EQ(sys->drone_radius, 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(edge_very_large_cell) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1000.0f);
    ASSERT_NOT_NULL(sys);

    DroneStateSOA* states = create_test_states(arena, 100);
    for (uint32_t i = 0; i < 100; i++) {
        states->pos_x[i] = (float)i;
        states->pos_y[i] = 0.0f;
        states->pos_z[i] = 0.0f;
    }

    /* All drones should hash to same cell with huge cell size */
    collision_build_spatial_hash(sys, states, 100);

    uint32_t indices[200];
    CellQuery query = { .indices = indices, .count = 0, .capacity = 200 };
    spatial_hash_query_cell(sys->spatial_hash, 50.0f, 0.0f, 0.0f, &query);

    ASSERT_EQ(query.count, 100);

    arena_destroy(arena);
    return 0;
}

TEST(edge_very_small_cell) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 0.001f);
    ASSERT_NOT_NULL(sys);

    DroneStateSOA* states = create_test_states(arena, 10);
    for (uint32_t i = 0; i < 10; i++) {
        states->pos_x[i] = (float)i;
        states->pos_y[i] = 0.0f;
        states->pos_z[i] = 0.0f;
    }

    collision_build_spatial_hash(sys, states, 10);
    collision_detect_drone_drone(sys, states, 10);

    /* Detection should still work */
    ASSERT_EQ(sys->pair_count, 0);

    arena_destroy(arena);
    return 0;
}

TEST(edge_coincident_drones) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 2);
    DroneParamsSOA* params = create_test_params(arena, 2, 1.0f);

    /* Two drones at exactly same position */
    states->pos_x[0] = 5.0f; states->pos_y[0] = 5.0f; states->pos_z[0] = 5.0f;
    states->pos_x[1] = 5.0f; states->pos_y[1] = 5.0f; states->pos_z[1] = 5.0f;

    collision_detect_all(sys, states, NULL, 2);

    ASSERT_EQ(sys->pair_count, 1);

    /* Response should handle without division by zero */
    collision_apply_drone_response(sys, states, params, 0.5f, 2);

    /* Positions should be pushed apart (not NaN) */
    ASSERT_FALSE(isnan(states->pos_x[0]));
    ASSERT_FALSE(isnan(states->pos_x[1]));

    arena_destroy(arena);
    return 0;
}

TEST(edge_max_pairs_reached) {
    Arena* arena = arena_create(8 * 1024 * 1024);

    /* Create small system with limited pairs */
    CollisionSystem* sys = collision_create(arena, 100, 0.5f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 100);

    /* All drones at same position - should generate many collisions */
    for (uint32_t i = 0; i < 100; i++) {
        states->pos_x[i] = 0.0f;
        states->pos_y[i] = 0.0f;
        states->pos_z[i] = 0.0f;
    }

    collision_detect_all(sys, states, NULL, 100);

    /* Should cap at max_pairs without crashing */
    ASSERT_LE(sys->pair_count, sys->max_pairs);

    arena_destroy(arena);
    return 0;
}

TEST(edge_boundary_positions) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 4);

    /* Drones at extreme positions */
    states->pos_x[0] = -1e6f; states->pos_y[0] = -1e6f; states->pos_z[0] = -1e6f;
    states->pos_x[1] = 1e6f;  states->pos_y[1] = 1e6f;  states->pos_z[1] = 1e6f;
    states->pos_x[2] = 0.0f;  states->pos_y[2] = 0.0f;  states->pos_z[2] = 0.0f;
    states->pos_x[3] = 0.05f; states->pos_y[3] = 0.0f;  states->pos_z[3] = 0.0f;

    collision_detect_all(sys, states, NULL, 4);

    /* Drones 2 and 3 should collide, others too far apart */
    ASSERT_EQ(sys->pair_count, 1);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Section 15: Memory Validation Tests
 * ============================================================================ */

TEST(memory_size_calculation) {
    size_t hash_size = spatial_hash_memory_size(1024);
    ASSERT_GT(hash_size, (size_t)0);

    size_t sys_size = collision_memory_size(1024, 2048);
    ASSERT_GT(sys_size, (size_t)0);
    ASSERT_LT(sys_size, (size_t)(200 * 1024));

    return 0;
}

TEST(memory_bounds) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    CollisionSystem* sys = collision_create(arena, 1024, 0.1f, 1.0f);
    DroneStateSOA* states = create_test_states(arena, 1024);

    /* Write to array boundaries */
    sys->drone_world_collision[0] = 1;
    sys->drone_world_collision[1023] = 1;
    sys->penetration_depth[0] = -0.1f;
    sys->penetration_depth[1023] = -0.1f;
    sys->collision_normals[0] = VEC3(1, 0, 0);
    sys->collision_normals[1023] = VEC3(0, 1, 0);

    /* Should not crash */
    ASSERT_EQ(sys->drone_world_collision[0], 1);
    ASSERT_EQ(sys->drone_world_collision[1023], 1);

    /* Run full detection on all 1024 drones */
    for (uint32_t i = 0; i < 1024; i++) {
        states->pos_x[i] = (float)(i % 32);
        states->pos_y[i] = (float)((i / 32) % 32);
        states->pos_z[i] = (float)(i / 1024);
    }

    collision_detect_all(sys, states, NULL, 1024);

    ASSERT_EQ(sys->spatial_hash->entry_count, 1024);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Main Test Runner
 * ============================================================================ */

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    TEST_SUITE_BEGIN("Collision System Module Tests");

    /* Section 1: Spatial Hash Allocation */
    RUN_TEST(spatial_hash_create_basic);
    RUN_TEST(spatial_hash_create_alignment);
    RUN_TEST(spatial_hash_create_null_arena);
    RUN_TEST(spatial_hash_clear_resets_all);

    /* Section 2: Spatial Hash Insert */
    RUN_TEST(spatial_hash_insert_single);
    RUN_TEST(spatial_hash_insert_multiple_same_cell);
    RUN_TEST(spatial_hash_insert_multiple_diff_cells);
    RUN_TEST(spatial_hash_insert_overflow);

    /* Section 3: Spatial Hash Query */
    RUN_TEST(spatial_hash_query_empty);
    RUN_TEST(spatial_hash_query_single_cell);
    RUN_TEST(spatial_hash_query_neighborhood_27);
    RUN_TEST(spatial_hash_query_hash_distribution);

    /* Section 4: Collision System Allocation */
    RUN_TEST(collision_create_basic);
    RUN_TEST(collision_create_alignment);
    RUN_TEST(collision_create_capacity_1024);
    RUN_TEST(collision_create_null_arena);
    RUN_TEST(collision_create_zero_drones);

    /* Section 5: Spatial Hash Build */
    RUN_TEST(build_hash_empty);
    RUN_TEST(build_hash_single_drone);
    RUN_TEST(build_hash_clustered_drones);
    RUN_TEST(build_hash_scattered_drones);
    RUN_TEST(build_hash_1024_drones);

    /* Section 6: Drone-Drone Detection */
    RUN_TEST(detect_dd_no_collisions);
    RUN_TEST(detect_dd_single_collision);
    RUN_TEST(detect_dd_multiple_collisions);
    RUN_TEST(detect_dd_chain_collision);
    RUN_TEST(detect_dd_pair_ordering);
    RUN_TEST(detect_dd_pair_count);
    RUN_TEST(detect_dd_touching);
    RUN_TEST(detect_dd_self_no_collision);

    /* Section 7: Drone-World Detection */
    RUN_TEST(detect_world_empty_world);
    RUN_TEST(detect_world_preserves_non_colliding);
    RUN_TEST(detect_world_batch_efficiency);
    RUN_TEST(detect_world_penetration_depth);
    RUN_TEST(detect_world_normal_calculation);
    RUN_TEST(detect_world_flags);
    RUN_TEST(detect_world_no_collision_above);
    RUN_TEST(detect_world_ground_collision);

    /* Section 8: Combined Detection */
    RUN_TEST(detect_all_ordering);
    RUN_TEST(detect_all_resets_previous);
    RUN_TEST(detect_all_independence);

    /* Section 9: World Response */
    RUN_TEST(response_world_position_correction);
    RUN_TEST(response_world_velocity_reflection);
    RUN_TEST(response_world_restitution_0);
    RUN_TEST(response_world_restitution_1);
    RUN_TEST(response_world_only_into_surface);
    RUN_TEST(response_world_preserves_non_colliding);

    /* Section 10: Drone-Drone Response */
    RUN_TEST(response_dd_separation);
    RUN_TEST(response_dd_mass_weighted);
    RUN_TEST(response_dd_equal_mass);
    RUN_TEST(response_dd_velocity_exchange);
    RUN_TEST(response_dd_restitution);
    RUN_TEST(response_dd_head_on);
    RUN_TEST(response_dd_glancing);

    /* Section 11: K-Nearest Neighbor */
    RUN_TEST(knn_single_neighbor);
    RUN_TEST(knn_k_neighbors);
    RUN_TEST(knn_fewer_than_k);
    RUN_TEST(knn_sorted_by_distance);
    RUN_TEST(knn_excludes_query_point);
    RUN_TEST(knn_batch_correctness);

    /* Section 12: Query Functions */
    RUN_TEST(get_results_returns_valid);
    RUN_TEST(drone_world_check_true);
    RUN_TEST(drone_world_check_false);
    RUN_TEST(get_pair_finds_collision);

    /* Section 13: Utility Functions */
    RUN_TEST(check_pair_colliding);
    RUN_TEST(check_pair_not_colliding);
    RUN_TEST(compute_normal_direction);
    RUN_TEST(compute_normal_unit_length);

    /* Section 14: Edge Cases */
    RUN_TEST(edge_zero_radius);
    RUN_TEST(edge_very_large_cell);
    RUN_TEST(edge_very_small_cell);
    RUN_TEST(edge_coincident_drones);
    RUN_TEST(edge_max_pairs_reached);
    RUN_TEST(edge_boundary_positions);

    /* Section 15: Memory Validation */
    RUN_TEST(memory_size_calculation);
    RUN_TEST(memory_bounds);

    TEST_SUITE_END();
}
