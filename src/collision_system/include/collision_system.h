/**
 * Collision System Module - Spatial Hash-Based Collision Detection
 *
 * Provides O(n) drone-drone collision detection via spatial hashing,
 * SIMD-optimized batch SDF queries for drone-world collisions,
 * physics-based collision response with position correction and velocity reflection,
 * and K-nearest neighbor queries for observation generation.
 *
 * Performance Targets (1024 drones):
 * - Spatial hash clear: <1us
 * - Spatial hash build: <100us
 * - Drone-drone detection: <500us
 * - World collision batch: <200us
 * - Collision response: <100us
 * - Total collision frame: <1ms
 *
 * Memory Budget (1024 drones): ~61 KB
 * - SpatialHashGrid cell_heads: 16 KB
 * - SpatialHashGrid entries: 8 KB
 * - collision_pairs: 16 KB
 * - drone_world_collision: 1 KB
 * - penetration_depth: 4 KB
 * - collision_normals: 16 KB
 *
 * Dependencies:
 * - foundation: Vec3, Arena, SIMD macros, atomics
 * - drone_state: DroneStateSOA, DroneParamsSOA
 * - world_brick_map: world_sdf_query_batch, world_sdf_gradient
 */

#ifndef COLLISION_SYSTEM_H
#define COLLISION_SYSTEM_H

#include "foundation.h"
#include "drone_state.h"
#include "world_brick_map.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Section 1: Constants
 * ============================================================================ */

/** Hash table size - power of 2 for fast modulo via bitmask */
#define HASH_TABLE_SIZE          4096

/** Bitmask for hash table indexing (HASH_TABLE_SIZE - 1) */
#define HASH_TABLE_MASK          (HASH_TABLE_SIZE - 1)

/** Default spatial hash cell size in meters */
#define DEFAULT_CELL_SIZE        1.0f

/** Maximum neighboring cells to check (3x3x3 neighborhood) */
#define MAX_NEIGHBORS            27

/** Sentinel value indicating end of linked list chain */
#define SPATIAL_HASH_END         UINT32_MAX

/* ============================================================================
 * Section 2: SpatialHashEntry - Linked List Node for Chaining
 * ============================================================================ */

/**
 * Linked list node for chaining drones in the same spatial hash cell.
 *
 * When multiple drones hash to the same cell, they form a linked list.
 * This allows handling arbitrary collision density without pre-allocation.
 */
typedef struct SpatialHashEntry {
    /** Index into DroneStateSOA arrays */
    uint32_t drone_index;

    /** Next entry in chain (SPATIAL_HASH_END = end of list) */
    uint32_t next;
} SpatialHashEntry;

/* ============================================================================
 * Section 3: SpatialHashGrid
 * ============================================================================ */

/**
 * Spatial hash grid for O(n) collision detection.
 *
 * Uses a fixed-size hash table with chaining for collision resolution.
 * Entries are allocated from a pool to avoid per-frame heap allocations.
 *
 * Memory usage (1024 drones):
 * - cell_heads: HASH_TABLE_SIZE * 4B = 16 KB
 * - entries: max_entries * 8B = 8 KB
 * - Total: ~24 KB
 */
typedef struct SpatialHashGrid {
    /** Hash table: head of linked list for each cell.
     *  cell_heads[hash] = first entry index, or SPATIAL_HASH_END if empty.
     *  Array size: HASH_TABLE_SIZE */
    uint32_t* cell_heads;

    /** Entry pool for linked list nodes.
     *  Array size: max_entries */
    SpatialHashEntry* entries;

    /** Current number of entries used in the pool */
    uint32_t entry_count;

    /** Maximum capacity of entry pool */
    uint32_t max_entries;

    /** Spatial cell size in world units (meters) */
    float cell_size;

    /** Precomputed 1.0 / cell_size for faster coordinate conversion */
    float inv_cell_size;
} SpatialHashGrid;

/* ============================================================================
 * Section 4: CellQuery - Query Results Container
 * ============================================================================ */

/**
 * Container for cell query results.
 *
 * Used to return drone indices found within a cell or neighborhood.
 * Caller provides the output buffer; the query fills it and sets count.
 */
typedef struct CellQuery {
    /** Output array of drone indices */
    uint32_t* indices;

    /** Number of drones found */
    uint32_t count;

    /** Maximum output size (capacity of indices array) */
    uint32_t capacity;
} CellQuery;

/* ============================================================================
 * Section 5: CollisionSystem - Main Collision Detection System
 * ============================================================================ */

/**
 * Complete collision detection system for drone simulation.
 *
 * Handles both drone-drone and drone-world collision detection with
 * physics-based response calculations. Uses spatial hashing for efficient
 * broad-phase detection and SDF queries for precise world collision.
 *
 * Memory budget (1024 drones, 2048 max pairs):
 * - SpatialHashGrid: ~24 KB
 * - collision_pairs: 2048 * 2 * 4B = 16 KB
 * - drone_world_collision: 1024 * 1B = 1 KB
 * - penetration_depth: 1024 * 4B = 4 KB
 * - collision_normals: 1024 * 16B = 16 KB
 * - Total: ~61 KB
 */
typedef struct CollisionSystem {
    /** Spatial hash grid for drone-drone detection */
    SpatialHashGrid* spatial_hash;

    /** Scratch arena for per-frame temporary allocations */
    Arena* scratch_arena;

    /** Collision sphere radius per drone (uniform for all drones) */
    float drone_radius;

    /** Precomputed drone_radius squared for distance comparisons */
    float drone_radius_sq;

    /** SDF threshold for world collision detection (positive = margin) */
    float world_collision_margin;

    /** Collision pair storage - compact [drone_a, drone_b] pairs.
     *  Array size: max_pairs * 2 */
    uint32_t* collision_pairs;

    /** Current number of detected collision pairs */
    uint32_t pair_count;

    /** Maximum number of collision pairs to store */
    uint32_t max_pairs;

    /** Per-drone world collision flags.
     *  Array size: max_drones */
    uint8_t* drone_world_collision;

    /** Per-drone SDF penetration depth (negative = inside world geometry).
     *  Array size: max_drones */
    float* penetration_depth;

    /** Per-drone collision surface normal (unit vector).
     *  Array size: max_drones */
    Vec3* collision_normals;

    /** Maximum number of drones this system can handle */
    uint32_t max_drones;
} CollisionSystem;

/* ============================================================================
 * Section 6: CollisionResults - Read-Only Results View
 * ============================================================================ */

/**
 * Read-only view of collision detection results.
 *
 * Returned by collision_get_results() for external consumption
 * without exposing internal mutable state.
 */
typedef struct CollisionResults {
    /** Drone-drone collision pairs: [pair_count * 2] as (drone_a, drone_b) */
    const uint32_t* pairs;

    /** Number of collision pairs detected */
    uint32_t pair_count;

    /** Per-drone world collision flags: [drone_count], 1 = colliding */
    const uint8_t* world_flags;

    /** Per-drone SDF penetration depth: [drone_count], negative = inside */
    const float* penetration;

    /** Per-drone surface normal at collision point: [drone_count] */
    const Vec3* normals;
} CollisionResults;

/* ============================================================================
 * Section 7: Spatial Hash Lifecycle Functions
 * ============================================================================ */

/**
 * Create a spatial hash grid from an arena allocator.
 *
 * Allocates all memory from the arena - no per-frame allocations needed.
 * Hash table is 32-byte aligned for SIMD operations.
 *
 * @param arena      Memory arena for allocation
 * @param max_entries Maximum number of drone entries (typically = max_drones)
 * @param cell_size  Spatial cell size in world units (meters)
 * @return Pointer to SpatialHashGrid, or NULL on failure
 */
SpatialHashGrid* spatial_hash_create(Arena* arena, uint32_t max_entries, float cell_size);

/**
 * Destroy a spatial hash grid.
 *
 * No-op when using arena allocation (memory freed with arena).
 *
 * @param grid Grid to destroy (can be NULL)
 */
void spatial_hash_destroy(SpatialHashGrid* grid);

/**
 * Clear all entries from the spatial hash grid.
 *
 * O(1) reset: memsets cell_heads to SPATIAL_HASH_END, resets entry_count to 0.
 * Target performance: <1us
 *
 * @param grid Grid to clear
 */
void spatial_hash_clear(SpatialHashGrid* grid);

/**
 * Insert a drone into the spatial hash grid.
 *
 * Computes hash from position, allocates entry from pool, inserts at chain head.
 * O(1) operation.
 *
 * @param grid        Grid to insert into
 * @param drone_index Index of drone in DroneStateSOA
 * @param x, y, z     World position of drone
 */
void spatial_hash_insert(SpatialHashGrid* grid, uint32_t drone_index,
                         float x, float y, float z);

/**
 * Query drones in a single cell.
 *
 * Returns all drone indices that hash to the same cell as the given position.
 *
 * @param grid   Grid to query
 * @param x,y,z  World position to query
 * @param result Output query result (caller provides buffer)
 */
void spatial_hash_query_cell(const SpatialHashGrid* grid,
                             float x, float y, float z,
                             CellQuery* result);

/**
 * Query drones in 3x3x3 neighborhood around a position.
 *
 * Returns all drone indices in the 27 cells surrounding the given position.
 * This is the primary query for collision detection.
 *
 * @param grid   Grid to query
 * @param x,y,z  World position to query
 * @param result Output query result (caller provides buffer)
 */
void spatial_hash_query_neighborhood(const SpatialHashGrid* grid,
                                     float x, float y, float z,
                                     CellQuery* result);

/* ============================================================================
 * Section 8: Collision System Lifecycle Functions
 * ============================================================================ */

/**
 * Create a collision detection system.
 *
 * Allocates spatial hash, collision pair storage, and per-drone arrays.
 * All allocations are 32-byte aligned for SIMD operations.
 *
 * @param arena       Memory arena for allocation
 * @param max_drones  Maximum number of drones to handle
 * @param drone_radius Collision sphere radius per drone (meters)
 * @param cell_size   Spatial hash cell size (should be >= 2 * drone_radius)
 * @return Pointer to CollisionSystem, or NULL on failure
 */
CollisionSystem* collision_create(Arena* arena, uint32_t max_drones,
                                  float drone_radius, float cell_size);

/**
 * Destroy a collision system.
 *
 * No-op when using arena allocation (memory freed with arena).
 *
 * @param sys System to destroy (can be NULL)
 */
void collision_destroy(CollisionSystem* sys);

/**
 * Reset collision system for a new frame.
 *
 * Clears spatial hash, resets pair count, clears world collision flags.
 * Call this at the start of each physics frame.
 *
 * @param sys System to reset
 */
void collision_reset(CollisionSystem* sys);

/* ============================================================================
 * Section 9: Collision Detection Functions
 * ============================================================================ */

/**
 * Build spatial hash from drone positions.
 *
 * Clears the hash and inserts all drones. Must be called before detection.
 * Target performance: <100us for 1024 drones.
 *
 * @param sys    Collision system
 * @param states Drone state arrays (reads pos_x, pos_y, pos_z)
 * @param count  Number of drones to process
 */
void collision_build_spatial_hash(CollisionSystem* sys,
                                  const DroneStateSOA* states,
                                  uint32_t count);

/**
 * Detect drone-drone collisions using spatial hash.
 *
 * For each drone, queries 3x3x3 neighborhood and checks sphere-sphere collision.
 * Stores collision pairs in (i, j) format where i < j to avoid duplicates.
 * Target performance: <500us for 1024 drones.
 *
 * @param sys    Collision system
 * @param states Drone state arrays
 * @param count  Number of drones
 */
void collision_detect_drone_drone(CollisionSystem* sys,
                                  const DroneStateSOA* states,
                                  uint32_t count);

/**
 * Detect drone-world collisions using batch SDF queries.
 *
 * Gathers drone positions, performs batch SDF query, computes normals
 * for colliding drones. Uses SIMD-optimized world_sdf_query_batch.
 * Target performance: <200us for 1024 drones.
 *
 * @param sys    Collision system
 * @param states Drone state arrays
 * @param world  World brick map for SDF queries
 * @param count  Number of drones
 */
void collision_detect_drone_world(CollisionSystem* sys,
                                  const DroneStateSOA* states,
                                  const WorldBrickMap* world,
                                  uint32_t count);

/**
 * Perform all collision detection (spatial hash build, drone-drone, drone-world).
 *
 * Convenience function that calls all detection steps in order.
 * Target performance: <1ms total for 1024 drones.
 *
 * @param sys    Collision system
 * @param states Drone state arrays
 * @param world  World brick map (can be NULL to skip world collision)
 * @param count  Number of drones
 */
void collision_detect_all(CollisionSystem* sys,
                          const DroneStateSOA* states,
                          const WorldBrickMap* world,
                          uint32_t count);

/* ============================================================================
 * Section 10: Results Query Functions
 * ============================================================================ */

/**
 * Get read-only view of collision detection results.
 *
 * @param sys Collision system
 * @return CollisionResults struct with pointers to internal data
 */
CollisionResults collision_get_results(const CollisionSystem* sys);

/**
 * Check if a specific drone is colliding with the world.
 *
 * @param sys       Collision system
 * @param drone_idx Drone index to check
 * @return true if drone is colliding with world geometry
 */
bool collision_drone_world_check(const CollisionSystem* sys, uint32_t drone_idx);

/**
 * Find the other drone in a collision pair involving the given drone.
 *
 * Searches collision pairs for any pair containing drone_idx.
 * Returns UINT32_MAX if no collision found.
 *
 * @param sys       Collision system
 * @param drone_idx Drone index to search for
 * @return Index of other drone in collision, or UINT32_MAX if none
 */
uint32_t collision_get_pair(const CollisionSystem* sys, uint32_t drone_idx);

/* ============================================================================
 * Section 11: Collision Response Functions
 * ============================================================================ */

/**
 * Apply all collision responses (world and drone-drone).
 *
 * Convenience function combining world and drone response.
 *
 * @param sys             Collision system
 * @param states          Drone state arrays (modified in place)
 * @param params          Drone parameter arrays (for mass)
 * @param restitution     Coefficient of restitution [0,1]
 * @param separation_force Force multiplier for separation
 * @param count           Number of drones
 */
void collision_apply_response(CollisionSystem* sys,
                              DroneStateSOA* states,
                              const DroneParamsSOA* params,
                              float restitution,
                              float separation_force,
                              uint32_t count);

/**
 * Apply world collision response.
 *
 * For drones colliding with world geometry:
 * - Reflects velocity component along collision normal
 * - Pushes drone out of collision (position correction)
 * Target performance: <100us for 1024 drones.
 *
 * @param sys          Collision system
 * @param states       Drone state arrays (modified in place)
 * @param restitution  Coefficient of restitution [0,1] (0=inelastic, 1=elastic)
 * @param pushout_speed Multiplier for position correction speed
 * @param count        Number of drones
 */
void collision_apply_world_response(CollisionSystem* sys,
                                    DroneStateSOA* states,
                                    float restitution,
                                    float pushout_speed,
                                    uint32_t count);

/**
 * Apply drone-drone collision response.
 *
 * For each collision pair:
 * - Computes mass-weighted position separation
 * - Applies impulse-based velocity exchange with restitution
 * Target performance: <50us for typical collision counts.
 *
 * @param sys         Collision system
 * @param states      Drone state arrays (modified in place)
 * @param params      Drone parameter arrays (for mass)
 * @param restitution Coefficient of restitution [0,1]
 * @param count       Number of drones
 */
void collision_apply_drone_response(CollisionSystem* sys,
                                    DroneStateSOA* states,
                                    const DroneParamsSOA* params,
                                    float restitution,
                                    uint32_t count);

/* ============================================================================
 * Section 12: K-Nearest Neighbor Functions
 * ============================================================================ */

/**
 * Find K nearest drones to a given position.
 *
 * Uses spatial hash for efficient neighborhood query, then sorts by distance.
 * Useful for observation generation (e.g., nearest neighbors for each drone).
 *
 * @param sys           Collision system (must have built spatial hash)
 * @param states        Drone state arrays
 * @param position      Query position
 * @param k             Number of neighbors to find
 * @param out_indices   Output: indices of k nearest drones [k]
 * @param out_distances Output: squared distances to each neighbor [k]
 * @param out_count     Output: actual number found (may be < k)
 */
void collision_find_k_nearest(const CollisionSystem* sys,
                              const DroneStateSOA* states,
                              Vec3 position,
                              uint32_t k,
                              uint32_t* out_indices,
                              float* out_distances,
                              uint32_t* out_count);

/**
 * Find K nearest neighbors for all drones (batch operation).
 *
 * For each drone, finds k nearest other drones.
 * Output arrays: out_indices[drone_count * k], out_distances[drone_count * k]
 *
 * @param sys           Collision system (must have built spatial hash)
 * @param states        Drone state arrays
 * @param drone_count   Number of drones
 * @param k             Number of neighbors per drone
 * @param out_indices   Output: [drone_count * k] neighbor indices
 * @param out_distances Output: [drone_count * k] squared distances
 */
void collision_find_k_nearest_batch(const CollisionSystem* sys,
                                    const DroneStateSOA* states,
                                    uint32_t drone_count,
                                    uint32_t k,
                                    uint32_t* out_indices,
                                    float* out_distances);

/* ============================================================================
 * Section 13: Utility Functions
 * ============================================================================ */

/**
 * Compute spatial hash from world coordinates.
 *
 * Uses prime-based hash combining for good distribution.
 * Inline for performance in tight loops.
 *
 * @param x, y, z       World coordinates
 * @param inv_cell_size Precomputed 1.0 / cell_size
 * @return Hash value in range [0, HASH_TABLE_MASK]
 */
FOUNDATION_INLINE uint32_t spatial_hash_compute(float x, float y, float z,
                                                 float inv_cell_size) {
    /* Convert to cell coordinates */
    int32_t ix = (int32_t)floorf(x * inv_cell_size);
    int32_t iy = (int32_t)floorf(y * inv_cell_size);
    int32_t iz = (int32_t)floorf(z * inv_cell_size);

    /* Prime-based hash combine for good distribution */
    uint32_t h = (uint32_t)ix * 73856093u;
    h ^= (uint32_t)iy * 19349663u;
    h ^= (uint32_t)iz * 83492791u;

    return h & HASH_TABLE_MASK;
}

/**
 * Compute spatial hash from cell coordinates.
 *
 * @param cx, cy, cz Cell coordinates (integers)
 * @return Hash value in range [0, HASH_TABLE_MASK]
 */
FOUNDATION_INLINE uint32_t spatial_hash_compute_cell(int32_t cx, int32_t cy, int32_t cz) {
    uint32_t h = (uint32_t)cx * 73856093u;
    h ^= (uint32_t)cy * 19349663u;
    h ^= (uint32_t)cz * 83492791u;
    return h & HASH_TABLE_MASK;
}

/**
 * Check if two drones are colliding (sphere-sphere test).
 *
 * @param states        Drone state arrays
 * @param idx_a         First drone index
 * @param idx_b         Second drone index
 * @param radius_sum_sq (radius_a + radius_b)^2 for threshold
 * @return true if drones are colliding
 */
bool collision_check_pair(const DroneStateSOA* states,
                          uint32_t idx_a, uint32_t idx_b,
                          float radius_sum_sq);

/**
 * Compute collision normal from drone A to drone B.
 *
 * Returns normalized vector pointing from A's center to B's center.
 *
 * @param states Drone state arrays
 * @param idx_a  First drone index (normal points away from this drone)
 * @param idx_b  Second drone index (normal points toward this drone)
 * @return Unit normal vector
 */
Vec3 collision_compute_normal(const DroneStateSOA* states,
                              uint32_t idx_a, uint32_t idx_b);

/**
 * Calculate required memory for a collision system.
 *
 * @param max_drones Maximum drone count
 * @param max_pairs  Maximum collision pairs
 * @return Total bytes required
 */
size_t collision_memory_size(uint32_t max_drones, uint32_t max_pairs);

/**
 * Calculate required memory for a spatial hash grid.
 *
 * @param max_entries Maximum entry count
 * @return Total bytes required
 */
size_t spatial_hash_memory_size(uint32_t max_entries);

/* ============================================================================
 * Section 14: Type Size Verification
 * ============================================================================ */

FOUNDATION_STATIC_ASSERT(sizeof(SpatialHashEntry) == 8, "SpatialHashEntry must be 8 bytes");

#ifdef __cplusplus
}
#endif

#endif /* COLLISION_SYSTEM_H */
