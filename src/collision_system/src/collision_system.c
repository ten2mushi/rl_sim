/**
 * Collision System Implementation
 *
 * Provides spatial hash-based collision detection for drone simulation.
 * Optimized for 1024+ drones with <1ms total frame time target.
 */

#include "../include/collision_system.h"
#include <string.h>
#include <float.h>

/* ============================================================================
 * Section 1: Memory Size Calculations
 * ============================================================================ */

size_t spatial_hash_memory_size(uint32_t max_entries) {
    size_t size = 0;

    /* SpatialHashGrid struct */
    size += sizeof(SpatialHashGrid);

    /* cell_heads array (32-byte aligned) */
    size = align_up_size(size, 32);
    size += HASH_TABLE_SIZE * sizeof(uint32_t);

    /* entries array (8-byte aligned) */
    size = align_up_size(size, 8);
    size += max_entries * sizeof(SpatialHashEntry);

    return size;
}

size_t collision_memory_size(uint32_t max_drones, uint32_t max_pairs) {
    size_t size = 0;

    /* CollisionSystem struct */
    size += sizeof(CollisionSystem);

    /* SpatialHashGrid */
    size += spatial_hash_memory_size(max_drones);

    /* collision_pairs (32-byte aligned) */
    size = align_up_size(size, 32);
    size += max_pairs * 2 * sizeof(uint32_t);

    /* drone_world_collision (32-byte aligned for SIMD) */
    size = align_up_size(size, 32);
    size += max_drones * sizeof(uint8_t);

    /* penetration_depth (32-byte aligned) */
    size = align_up_size(size, 32);
    size += max_drones * sizeof(float);

    /* collision_normals (16-byte aligned for Vec3) */
    size = align_up_size(size, 16);
    size += max_drones * sizeof(Vec3);

    return size;
}

/* ============================================================================
 * Section 2: Spatial Hash Lifecycle
 * ============================================================================ */

SpatialHashGrid* spatial_hash_create(Arena* arena, uint32_t max_entries, float cell_size) {
    if (arena == NULL || max_entries == 0 || cell_size <= 0.0f) {
        return NULL;
    }

    /* Allocate grid struct */
    SpatialHashGrid* grid = arena_alloc_type(arena, SpatialHashGrid);
    if (grid == NULL) {
        return NULL;
    }

    /* Allocate cell_heads with 32-byte alignment */
    grid->cell_heads = arena_alloc_aligned(arena,
                                           HASH_TABLE_SIZE * sizeof(uint32_t),
                                           32);
    if (grid->cell_heads == NULL) {
        return NULL;
    }

    /* Allocate entries array */
    grid->entries = arena_alloc_aligned(arena,
                                        max_entries * sizeof(SpatialHashEntry),
                                        8);
    if (grid->entries == NULL) {
        return NULL;
    }

    /* Initialize fields */
    grid->entry_count = 0;
    grid->max_entries = max_entries;
    grid->cell_size = cell_size;
    grid->inv_cell_size = 1.0f / cell_size;

    /* Clear cell heads to empty */
    spatial_hash_clear(grid);

    return grid;
}

void spatial_hash_destroy(SpatialHashGrid* grid) {
    /* No-op: arena allocation handles cleanup */
    (void)grid;
}

void spatial_hash_clear(SpatialHashGrid* grid) {
    if (grid == NULL) {
        return;
    }

    /* O(1) reset: memset cell_heads to SPATIAL_HASH_END (0xFFFFFFFF) */
    memset(grid->cell_heads, 0xFF, HASH_TABLE_SIZE * sizeof(uint32_t));
    grid->entry_count = 0;
}

void spatial_hash_insert(SpatialHashGrid* grid, uint32_t drone_index,
                         float x, float y, float z) {
    if (grid == NULL || grid->entry_count >= grid->max_entries) {
        return;
    }

    /* Compute hash */
    uint32_t hash = spatial_hash_compute(x, y, z, grid->inv_cell_size);

    /* Allocate entry from pool */
    uint32_t entry_idx = grid->entry_count++;
    SpatialHashEntry* entry = &grid->entries[entry_idx];

    /* Insert at head of chain */
    entry->drone_index = drone_index;
    entry->next = grid->cell_heads[hash];
    grid->cell_heads[hash] = entry_idx;
}

void spatial_hash_query_cell(const SpatialHashGrid* grid,
                             float x, float y, float z,
                             CellQuery* result) {
    if (grid == NULL || result == NULL || result->indices == NULL) {
        if (result) result->count = 0;
        return;
    }

    result->count = 0;

    /* Compute hash */
    uint32_t hash = spatial_hash_compute(x, y, z, grid->inv_cell_size);

    /* Walk chain */
    uint32_t entry_idx = grid->cell_heads[hash];
    while (entry_idx != SPATIAL_HASH_END && result->count < result->capacity) {
        const SpatialHashEntry* entry = &grid->entries[entry_idx];
        result->indices[result->count++] = entry->drone_index;
        entry_idx = entry->next;
    }
}

void spatial_hash_query_neighborhood(const SpatialHashGrid* grid,
                                     float x, float y, float z,
                                     CellQuery* result) {
    if (grid == NULL || result == NULL || result->indices == NULL) {
        if (result) result->count = 0;
        return;
    }

    result->count = 0;

    /* Convert to cell coordinates */
    int32_t cx = (int32_t)floorf(x * grid->inv_cell_size);
    int32_t cy = (int32_t)floorf(y * grid->inv_cell_size);
    int32_t cz = (int32_t)floorf(z * grid->inv_cell_size);

    /* Query 3x3x3 neighborhood */
    for (int32_t dz = -1; dz <= 1; dz++) {
        for (int32_t dy = -1; dy <= 1; dy++) {
            for (int32_t dx = -1; dx <= 1; dx++) {
                uint32_t hash = spatial_hash_compute_cell(cx + dx, cy + dy, cz + dz);

                /* Walk chain */
                uint32_t entry_idx = grid->cell_heads[hash];
                while (entry_idx != SPATIAL_HASH_END && result->count < result->capacity) {
                    const SpatialHashEntry* entry = &grid->entries[entry_idx];
                    result->indices[result->count++] = entry->drone_index;
                    entry_idx = entry->next;
                }
            }
        }
    }
}

/* ============================================================================
 * Section 3: Collision System Lifecycle
 * ============================================================================ */

CollisionSystem* collision_create(Arena* arena, uint32_t max_drones,
                                  float drone_radius, float cell_size) {
    if (arena == NULL || max_drones == 0) {
        return NULL;
    }

    /* Allocate system struct */
    CollisionSystem* sys = arena_alloc_type(arena, CollisionSystem);
    if (sys == NULL) {
        return NULL;
    }

    /* Create spatial hash */
    sys->spatial_hash = spatial_hash_create(arena, max_drones, cell_size);
    if (sys->spatial_hash == NULL) {
        return NULL;
    }

    /* Store scratch arena reference */
    sys->scratch_arena = arena;

    /* Set collision parameters */
    sys->drone_radius = drone_radius;
    sys->drone_radius_sq = drone_radius * drone_radius;
    sys->world_collision_margin = drone_radius;  /* Collision at surface - radius */

    /* Allocate collision pairs (max_pairs = 2 * max_drones) */
    sys->max_pairs = max_drones * 2;
    sys->collision_pairs = arena_alloc_aligned(arena,
                                               sys->max_pairs * 2 * sizeof(uint32_t),
                                               32);
    if (sys->collision_pairs == NULL) {
        return NULL;
    }
    sys->pair_count = 0;

    /* Allocate per-drone world collision flags */
    sys->drone_world_collision = arena_alloc_aligned(arena,
                                                     max_drones * sizeof(uint8_t),
                                                     32);
    if (sys->drone_world_collision == NULL) {
        return NULL;
    }

    /* Allocate penetration depth array */
    sys->penetration_depth = arena_alloc_aligned(arena,
                                                 max_drones * sizeof(float),
                                                 32);
    if (sys->penetration_depth == NULL) {
        return NULL;
    }

    /* Allocate collision normals array */
    sys->collision_normals = arena_alloc_aligned(arena,
                                                 max_drones * sizeof(Vec3),
                                                 16);
    if (sys->collision_normals == NULL) {
        return NULL;
    }

    sys->max_drones = max_drones;

    /* Initialize to clean state */
    collision_reset(sys);

    return sys;
}

void collision_destroy(CollisionSystem* sys) {
    /* No-op: arena allocation handles cleanup */
    (void)sys;
}

void collision_reset(CollisionSystem* sys) {
    if (sys == NULL) {
        return;
    }

    /* Clear spatial hash */
    spatial_hash_clear(sys->spatial_hash);

    /* Reset pair count */
    sys->pair_count = 0;

    /* Clear world collision flags */
    memset(sys->drone_world_collision, 0, sys->max_drones * sizeof(uint8_t));

    /* Initialize penetration depth to large positive value (far from surface)
     * so first-frame physics doesn't erroneously trigger ground effect */
    for (uint32_t i = 0; i < sys->max_drones; i++) {
        sys->penetration_depth[i] = 100.0f;
    }
}

/* ============================================================================
 * Section 4: Collision Detection
 * ============================================================================ */

void collision_build_spatial_hash(CollisionSystem* sys,
                                  const DroneStateSOA* states,
                                  uint32_t count) {
    if (sys == NULL || states == NULL || count == 0) {
        return;
    }

    /* Clear and rebuild */
    spatial_hash_clear(sys->spatial_hash);

    for (uint32_t i = 0; i < count; i++) {
        spatial_hash_insert(sys->spatial_hash, i,
                           states->pos_x[i],
                           states->pos_y[i],
                           states->pos_z[i]);
    }
}

void collision_detect_drone_drone(CollisionSystem* sys,
                                  const DroneStateSOA* states,
                                  uint32_t count) {
    if (sys == NULL || states == NULL || count == 0) {
        return;
    }

    sys->pair_count = 0;

    /* Collision threshold: (2 * radius)^2 */
    float radius_sum_sq = sys->drone_radius_sq * 4.0f;

    /* For each drone */
    for (uint32_t i = 0; i < count; i++) {
        float px = states->pos_x[i];
        float py = states->pos_y[i];
        float pz = states->pos_z[i];

        /* Convert to cell coordinates */
        int32_t cx = (int32_t)floorf(px * sys->spatial_hash->inv_cell_size);
        int32_t cy = (int32_t)floorf(py * sys->spatial_hash->inv_cell_size);
        int32_t cz = (int32_t)floorf(pz * sys->spatial_hash->inv_cell_size);

        /* Query 3x3x3 neighborhood */
        for (int32_t dz = -1; dz <= 1; dz++) {
            for (int32_t dy = -1; dy <= 1; dy++) {
                for (int32_t dx = -1; dx <= 1; dx++) {
                    uint32_t hash = spatial_hash_compute_cell(cx + dx, cy + dy, cz + dz);

                    /* Walk chain */
                    uint32_t entry_idx = sys->spatial_hash->cell_heads[hash];
                    while (entry_idx != SPATIAL_HASH_END) {
                        const SpatialHashEntry* entry = &sys->spatial_hash->entries[entry_idx];
                        uint32_t j = entry->drone_index;

                        /* Only check pairs once: i < j avoids duplicates */
                        if (i < j) {
                            /* Sphere-sphere collision test */
                            float dx_pos = states->pos_x[j] - px;
                            float dy_pos = states->pos_y[j] - py;
                            float dz_pos = states->pos_z[j] - pz;
                            float dist_sq = dx_pos * dx_pos + dy_pos * dy_pos + dz_pos * dz_pos;

                            if (dist_sq < radius_sum_sq) {
                                /* Record collision pair */
                                if (sys->pair_count < sys->max_pairs) {
                                    uint32_t pair_idx = sys->pair_count * 2;
                                    sys->collision_pairs[pair_idx] = i;
                                    sys->collision_pairs[pair_idx + 1] = j;
                                    sys->pair_count++;
                                }
                            }
                        }

                        entry_idx = entry->next;
                    }
                }
            }
        }
    }
}

void collision_detect_drone_world(CollisionSystem* sys,
                                  const DroneStateSOA* states,
                                  const WorldBrickMap* world,
                                  uint32_t count) {
    if (sys == NULL || states == NULL || world == NULL || count == 0) {
        return;
    }

    /* Use scratch arena for temporary allocations with manual scoping */
    Arena* scratch = sys->scratch_arena;
    ArenaScope scope = arena_scope_begin(scratch);

    /* Gather positions from SoA to Vec3 array */
    Vec3* positions = arena_alloc_array(scratch, Vec3, count);
    if (positions == NULL) {
        arena_scope_end(scope);
        return;
    }

    for (uint32_t i = 0; i < count; i++) {
        positions[i] = VEC3(states->pos_x[i], states->pos_y[i], states->pos_z[i]);
    }

    /* Batch SDF query (SIMD-optimized in world_brick_map) */
    world_sdf_query_batch(world, positions, sys->penetration_depth, count);

    /* Check collision threshold and compute normals */
    float margin = sys->world_collision_margin;

    for (uint32_t i = 0; i < count; i++) {
        float sdf = sys->penetration_depth[i];

        if (sdf < margin) {
            /* Drone is colliding with world */
            sys->drone_world_collision[i] = 1;

            /* Get collision normal from SDF gradient */
            Vec3 gradient = world_sdf_gradient(world, positions[i]);
            float len_sq = vec3_length_sq(gradient);

            if (len_sq > 1e-10f) {
                float inv_len = 1.0f / sqrtf(len_sq);
                sys->collision_normals[i] = vec3_scale(gradient, inv_len);
            } else {
                /* Default to up vector if gradient is zero */
                sys->collision_normals[i] = VEC3_UP;
            }
        } else {
            sys->drone_world_collision[i] = 0;
            sys->collision_normals[i] = VEC3_ZERO;
        }
    }

    arena_scope_end(scope);
}

void collision_detect_all(CollisionSystem* sys,
                          const DroneStateSOA* states,
                          const WorldBrickMap* world,
                          uint32_t count) {
    if (sys == NULL || states == NULL) {
        return;
    }

    /* Reset for new frame */
    collision_reset(sys);

    /* Build spatial hash */
    collision_build_spatial_hash(sys, states, count);

    /* Detect drone-drone collisions */
    collision_detect_drone_drone(sys, states, count);

    /* Detect drone-world collisions (skip if world is NULL) */
    if (world != NULL) {
        collision_detect_drone_world(sys, states, world, count);
    }
}

/* ============================================================================
 * Section 5: Results Query
 * ============================================================================ */

CollisionResults collision_get_results(const CollisionSystem* sys) {
    CollisionResults results = {0};

    if (sys != NULL) {
        results.pairs = sys->collision_pairs;
        results.pair_count = sys->pair_count;
        results.world_flags = sys->drone_world_collision;
        results.penetration = sys->penetration_depth;
        results.normals = sys->collision_normals;
    }

    return results;
}

bool collision_drone_world_check(const CollisionSystem* sys, uint32_t drone_idx) {
    if (sys == NULL || drone_idx >= sys->max_drones) {
        return false;
    }
    return sys->drone_world_collision[drone_idx] != 0;
}

uint32_t collision_get_pair(const CollisionSystem* sys, uint32_t drone_idx) {
    if (sys == NULL) {
        return UINT32_MAX;
    }

    /* Search through collision pairs */
    for (uint32_t p = 0; p < sys->pair_count; p++) {
        uint32_t idx = p * 2;
        if (sys->collision_pairs[idx] == drone_idx) {
            return sys->collision_pairs[idx + 1];
        }
        if (sys->collision_pairs[idx + 1] == drone_idx) {
            return sys->collision_pairs[idx];
        }
    }

    return UINT32_MAX;
}

/* ============================================================================
 * Section 6: Collision Response
 * ============================================================================ */

void collision_apply_world_response(CollisionSystem* sys,
                                    DroneStateSOA* states,
                                    float restitution,
                                    float pushout_speed,
                                    uint32_t count) {
    if (sys == NULL || states == NULL || count == 0) {
        return;
    }

    for (uint32_t i = 0; i < count; i++) {
        if (!sys->drone_world_collision[i]) {
            continue;
        }

        Vec3 normal = sys->collision_normals[i];
        float penetration = sys->penetration_depth[i];

        /* Get velocity */
        Vec3 vel = VEC3(states->vel_x[i], states->vel_y[i], states->vel_z[i]);

        /* Velocity component along normal */
        float v_normal = vec3_dot(vel, normal);

        /* Only respond if moving into surface (negative v_normal means into surface) */
        if (v_normal < 0.0f) {
            /* Reflect velocity with restitution */
            /* v_new = v - (1 + e) * (v . n) * n */
            float impulse = -v_normal * (1.0f + restitution);
            Vec3 v_reflect = vec3_scale(normal, impulse);
            vel = vec3_add(vel, v_reflect);
        }

        /* Push out of collision */
        /* penetration is negative when inside, margin is positive */
        float push_dist = sys->world_collision_margin - penetration;
        if (push_dist > 0.0f) {
            Vec3 pushout = vec3_scale(normal, push_dist * pushout_speed);
            states->pos_x[i] += pushout.x;
            states->pos_y[i] += pushout.y;
            states->pos_z[i] += pushout.z;
        }

        /* Write back velocity */
        states->vel_x[i] = vel.x;
        states->vel_y[i] = vel.y;
        states->vel_z[i] = vel.z;
    }
}

void collision_apply_drone_response(CollisionSystem* sys,
                                    DroneStateSOA* states,
                                    const DroneParamsSOA* params,
                                    float restitution,
                                    uint32_t count) {
    if (sys == NULL || states == NULL || sys->pair_count == 0) {
        return;
    }

    (void)count;  /* Used for bounds checking if needed */

    float radius_sum = sys->drone_radius * 2.0f;

    /* Process each collision pair */
    for (uint32_t p = 0; p < sys->pair_count; p++) {
        uint32_t idx = p * 2;
        uint32_t i = sys->collision_pairs[idx];
        uint32_t j = sys->collision_pairs[idx + 1];

        /* Get positions */
        Vec3 pos_i = VEC3(states->pos_x[i], states->pos_y[i], states->pos_z[i]);
        Vec3 pos_j = VEC3(states->pos_x[j], states->pos_y[j], states->pos_z[j]);

        /* Get velocities */
        Vec3 vel_i = VEC3(states->vel_x[i], states->vel_y[i], states->vel_z[i]);
        Vec3 vel_j = VEC3(states->vel_x[j], states->vel_y[j], states->vel_z[j]);

        /* Collision normal (from i to j) */
        Vec3 delta = vec3_sub(pos_j, pos_i);
        float dist = vec3_length(delta);

        Vec3 normal;
        if (dist > 1e-6f) {
            normal = vec3_scale(delta, 1.0f / dist);
        } else {
            /* Coincident drones - use arbitrary direction */
            normal = VEC3(1.0f, 0.0f, 0.0f);
            dist = 0.001f;
        }

        /* Penetration depth */
        float overlap = radius_sum - dist;
        if (overlap <= 0.0f) {
            continue;  /* No longer colliding */
        }

        /* Get masses */
        float mass_i = (params != NULL) ? params->mass[i] : 1.0f;
        float mass_j = (params != NULL) ? params->mass[j] : 1.0f;
        float total_mass = mass_i + mass_j;

        /* Mass ratios for separation */
        float ratio_i = mass_j / total_mass;
        float ratio_j = mass_i / total_mass;

        /* Position separation (mass-weighted) */
        float sep_i = overlap * ratio_i * 0.5f;
        float sep_j = overlap * ratio_j * 0.5f;

        Vec3 sep_vec_i = vec3_scale(normal, -sep_i);
        Vec3 sep_vec_j = vec3_scale(normal, sep_j);

        states->pos_x[i] += sep_vec_i.x;
        states->pos_y[i] += sep_vec_i.y;
        states->pos_z[i] += sep_vec_i.z;

        states->pos_x[j] += sep_vec_j.x;
        states->pos_y[j] += sep_vec_j.y;
        states->pos_z[j] += sep_vec_j.z;

        /* Relative velocity */
        Vec3 rel_vel = vec3_sub(vel_i, vel_j);
        float v_rel_n = vec3_dot(rel_vel, normal);

        /* Only apply impulse if drones are approaching */
        if (v_rel_n < 0.0f) {
            continue;
        }

        /* Impulse magnitude (coefficient of restitution) */
        float impulse_mag = (1.0f + restitution) * v_rel_n / total_mass;

        /* Apply impulse */
        Vec3 impulse = vec3_scale(normal, impulse_mag);

        Vec3 dv_i = vec3_scale(impulse, -mass_j);
        Vec3 dv_j = vec3_scale(impulse, mass_i);

        states->vel_x[i] += dv_i.x;
        states->vel_y[i] += dv_i.y;
        states->vel_z[i] += dv_i.z;

        states->vel_x[j] += dv_j.x;
        states->vel_y[j] += dv_j.y;
        states->vel_z[j] += dv_j.z;
    }
}

void collision_apply_response(CollisionSystem* sys,
                              DroneStateSOA* states,
                              const DroneParamsSOA* params,
                              float restitution,
                              float separation_force,
                              uint32_t count) {
    if (sys == NULL || states == NULL) {
        return;
    }

    /* Apply world collision response */
    collision_apply_world_response(sys, states, restitution, separation_force, count);

    /* Apply drone-drone collision response */
    collision_apply_drone_response(sys, states, params, restitution, count);
}

/* ============================================================================
 * Section 7: K-Nearest Neighbor
 * ============================================================================ */

/**
 * Internal helper: insertion sort for small k values
 * Returns true if inserted, false if duplicate or not inserted
 */
static bool knn_insert_sorted(uint32_t* indices, float* distances, uint32_t* count,
                              uint32_t capacity, uint32_t new_idx, float new_dist) {
    /* Check for duplicate - drone already in results */
    uint32_t check_limit = (*count < capacity) ? *count : capacity;
    for (uint32_t i = 0; i < check_limit; i++) {
        if (indices[i] == new_idx) {
            return false;  /* Already in results */
        }
    }

    /* Find insertion position */
    uint32_t pos = *count;
    while (pos > 0 && distances[pos - 1] > new_dist) {
        if (pos < capacity) {
            indices[pos] = indices[pos - 1];
            distances[pos] = distances[pos - 1];
        }
        pos--;
    }

    /* Insert if within capacity */
    if (pos < capacity) {
        indices[pos] = new_idx;
        distances[pos] = new_dist;
        if (*count < capacity) {
            (*count)++;
        }
        return true;
    }
    return false;
}

void collision_find_k_nearest(const CollisionSystem* sys,
                              const DroneStateSOA* states,
                              Vec3 position,
                              uint32_t k,
                              uint32_t* out_indices,
                              float* out_distances,
                              uint32_t* out_count) {
    if (sys == NULL || states == NULL || k == 0 ||
        out_indices == NULL || out_distances == NULL || out_count == NULL) {
        if (out_count) *out_count = 0;
        return;
    }

    *out_count = 0;

    /* Initialize distances to infinity */
    for (uint32_t i = 0; i < k; i++) {
        out_distances[i] = FLT_MAX;
        out_indices[i] = UINT32_MAX;
    }

    /* Query neighborhood */
    float x = position.x;
    float y = position.y;
    float z = position.z;

    int32_t cx = (int32_t)floorf(x * sys->spatial_hash->inv_cell_size);
    int32_t cy = (int32_t)floorf(y * sys->spatial_hash->inv_cell_size);
    int32_t cz = (int32_t)floorf(z * sys->spatial_hash->inv_cell_size);

    /* Search expanding neighborhood until we have k neighbors */
    int32_t search_radius = 1;
    uint32_t found = 0;

    /* Start with 3x3x3, expand if needed */
    while (found < k && search_radius <= 10) {
        for (int32_t dz = -search_radius; dz <= search_radius; dz++) {
            for (int32_t dy = -search_radius; dy <= search_radius; dy++) {
                for (int32_t dx = -search_radius; dx <= search_radius; dx++) {
                    /* Skip inner cells already searched (except first iteration) */
                    if (search_radius > 1) {
                        int32_t max_coord = max_i32(max_i32(abs(dx), abs(dy)), abs(dz));
                        if (max_coord < search_radius) {
                            continue;  /* Already searched */
                        }
                    }

                    uint32_t hash = spatial_hash_compute_cell(cx + dx, cy + dy, cz + dz);

                    /* Walk chain */
                    uint32_t entry_idx = sys->spatial_hash->cell_heads[hash];
                    while (entry_idx != SPATIAL_HASH_END) {
                        const SpatialHashEntry* entry = &sys->spatial_hash->entries[entry_idx];
                        uint32_t j = entry->drone_index;

                        /* Compute squared distance */
                        float dx_pos = states->pos_x[j] - x;
                        float dy_pos = states->pos_y[j] - y;
                        float dz_pos = states->pos_z[j] - z;
                        float dist_sq = dx_pos * dx_pos + dy_pos * dy_pos + dz_pos * dz_pos;

                        /* Skip if this is the query point itself (dist ~= 0) */
                        if (dist_sq > 1e-10f) {
                            /* Insert into sorted list if closer than current k-th */
                            if (found < k || dist_sq < out_distances[k - 1]) {
                                knn_insert_sorted(out_indices, out_distances, &found, k, j, dist_sq);
                            }
                        }

                        entry_idx = entry->next;
                    }
                }
            }
        }

        search_radius++;
    }

    *out_count = found;
}

void collision_find_k_nearest_batch(const CollisionSystem* sys,
                                    const DroneStateSOA* states,
                                    uint32_t drone_count,
                                    uint32_t k,
                                    uint32_t* out_indices,
                                    float* out_distances) {
    if (sys == NULL || states == NULL || drone_count == 0 || k == 0 ||
        out_indices == NULL || out_distances == NULL) {
        return;
    }

    /* Process each drone */
    for (uint32_t i = 0; i < drone_count; i++) {
        Vec3 pos = VEC3(states->pos_x[i], states->pos_y[i], states->pos_z[i]);

        uint32_t* idx_ptr = out_indices + (i * k);
        float* dist_ptr = out_distances + (i * k);
        uint32_t found = 0;

        /* Initialize to invalid */
        for (uint32_t j = 0; j < k; j++) {
            idx_ptr[j] = UINT32_MAX;
            dist_ptr[j] = FLT_MAX;
        }

        /* Query neighborhood - for batch, we want neighbors excluding self */
        int32_t cx = (int32_t)floorf(pos.x * sys->spatial_hash->inv_cell_size);
        int32_t cy = (int32_t)floorf(pos.y * sys->spatial_hash->inv_cell_size);
        int32_t cz = (int32_t)floorf(pos.z * sys->spatial_hash->inv_cell_size);

        /* Search 3x3x3 neighborhood */
        for (int32_t dz = -1; dz <= 1; dz++) {
            for (int32_t dy = -1; dy <= 1; dy++) {
                for (int32_t dx = -1; dx <= 1; dx++) {
                    uint32_t hash = spatial_hash_compute_cell(cx + dx, cy + dy, cz + dz);

                    uint32_t entry_idx = sys->spatial_hash->cell_heads[hash];
                    while (entry_idx != SPATIAL_HASH_END) {
                        const SpatialHashEntry* entry = &sys->spatial_hash->entries[entry_idx];
                        uint32_t j = entry->drone_index;

                        /* Skip self */
                        if (j != i) {
                            float dx_pos = states->pos_x[j] - pos.x;
                            float dy_pos = states->pos_y[j] - pos.y;
                            float dz_pos = states->pos_z[j] - pos.z;
                            float dist_sq = dx_pos * dx_pos + dy_pos * dy_pos + dz_pos * dz_pos;

                            if (found < k || dist_sq < dist_ptr[k - 1]) {
                                knn_insert_sorted(idx_ptr, dist_ptr, &found, k, j, dist_sq);
                            }
                        }

                        entry_idx = entry->next;
                    }
                }
            }
        }
    }
}

/* ============================================================================
 * Section 8: Utility Functions
 * ============================================================================ */

bool collision_check_pair(const DroneStateSOA* states,
                          uint32_t idx_a, uint32_t idx_b,
                          float radius_sum_sq) {
    if (states == NULL) {
        return false;
    }

    float dx = states->pos_x[idx_b] - states->pos_x[idx_a];
    float dy = states->pos_y[idx_b] - states->pos_y[idx_a];
    float dz = states->pos_z[idx_b] - states->pos_z[idx_a];
    float dist_sq = dx * dx + dy * dy + dz * dz;

    return dist_sq < radius_sum_sq;
}

Vec3 collision_compute_normal(const DroneStateSOA* states,
                              uint32_t idx_a, uint32_t idx_b) {
    if (states == NULL) {
        return VEC3(1.0f, 0.0f, 0.0f);
    }

    float dx = states->pos_x[idx_b] - states->pos_x[idx_a];
    float dy = states->pos_y[idx_b] - states->pos_y[idx_a];
    float dz = states->pos_z[idx_b] - states->pos_z[idx_a];

    float len_sq = dx * dx + dy * dy + dz * dz;

    if (len_sq > 1e-10f) {
        float inv_len = 1.0f / sqrtf(len_sq);
        return VEC3(dx * inv_len, dy * inv_len, dz * inv_len);
    }

    /* Coincident points - return arbitrary direction */
    return VEC3(1.0f, 0.0f, 0.0f);
}
