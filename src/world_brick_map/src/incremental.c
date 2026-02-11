/**
 * Incremental Regeneration Implementation
 *
 * Provides edit list management, dirty brick tracking, and incremental
 * SDF regeneration for efficient world updates.
 */

#include "../include/world_brick_map.h"

/* ============================================================================
 * Section 1: Edit List Functions
 * ============================================================================ */

EditList* edit_list_create(Arena* arena, uint32_t capacity) {
    if (arena == NULL || capacity == 0) {
        return NULL;
    }

    EditList* list = arena_alloc_type(arena, EditList);
    if (list == NULL) {
        return NULL;
    }

    list->entries = arena_alloc_array(arena, EditEntry, capacity);
    if (list->entries == NULL) {
        return NULL;
    }

    list->count = 0;
    list->capacity = capacity;

    return list;
}

void edit_list_clear(EditList* list) {
    if (list != NULL) {
        list->count = 0;
    }
}

bool edit_list_add(EditList* list, CSGOperation op, PrimitiveType primitive,
                   Vec3 center, Vec3 params, uint8_t material) {
    if (list == NULL || list->count >= list->capacity) {
        return false;
    }

    EditEntry* entry = &list->entries[list->count++];
    entry->op = op;
    entry->primitive = primitive;
    entry->center = center;
    entry->params = params;
    entry->material = material;

    return true;
}

uint32_t edit_list_count(const EditList* list) {
    return list != NULL ? list->count : 0;
}

/* ============================================================================
 * Section 2: Dirty Tracker Functions
 * ============================================================================ */

DirtyTracker* dirty_tracker_create(Arena* arena, uint32_t max_bricks) {
    if (arena == NULL || max_bricks == 0) {
        return NULL;
    }

    DirtyTracker* tracker = arena_alloc_type(arena, DirtyTracker);
    if (tracker == NULL) {
        return NULL;
    }

    tracker->dirty_indices = arena_alloc_array(arena, uint32_t, max_bricks);
    if (tracker->dirty_indices == NULL) {
        return NULL;
    }

    tracker->dirty_flags = arena_alloc_array(arena, bool, max_bricks);
    if (tracker->dirty_flags == NULL) {
        return NULL;
    }

    /* Initialize all flags to false */
    memset(tracker->dirty_flags, 0, max_bricks * sizeof(bool));

    tracker->dirty_count = 0;
    tracker->dirty_capacity = max_bricks;
    tracker->max_bricks = max_bricks;

    return tracker;
}

void dirty_tracker_clear(DirtyTracker* tracker) {
    if (tracker == NULL) return;

    /* Clear flags for dirty bricks */
    for (uint32_t i = 0; i < tracker->dirty_count; i++) {
        uint32_t idx = tracker->dirty_indices[i];
        if (idx < tracker->max_bricks) {
            tracker->dirty_flags[idx] = false;
        }
    }

    tracker->dirty_count = 0;
}

void dirty_tracker_mark_brick(DirtyTracker* tracker, uint32_t brick_index) {
    if (tracker == NULL || brick_index >= tracker->max_bricks) {
        return;
    }

    /* Skip if already dirty */
    if (tracker->dirty_flags[brick_index]) {
        return;
    }

    /* Add to dirty list if capacity allows */
    if (tracker->dirty_count < tracker->dirty_capacity) {
        tracker->dirty_flags[brick_index] = true;
        tracker->dirty_indices[tracker->dirty_count++] = brick_index;
    }
}

void dirty_tracker_mark_region(DirtyTracker* tracker,
                               const WorldBrickMap* world,
                               Vec3 min_pos, Vec3 max_pos) {
    if (tracker == NULL || world == NULL) return;

    /* Convert world positions to brick coordinates */
    int32_t bx0, by0, bz0, bx1, by1, bz1;
    world_pos_to_brick(world, min_pos, &bx0, &by0, &bz0);
    world_pos_to_brick(world, max_pos, &bx1, &by1, &bz1);

    /* Clamp to valid range */
    bx0 = max_i32(bx0, 0);
    by0 = max_i32(by0, 0);
    bz0 = max_i32(bz0, 0);
    bx1 = min_i32(bx1, (int32_t)world->grid_x - 1);
    by1 = min_i32(by1, (int32_t)world->grid_y - 1);
    bz1 = min_i32(bz1, (int32_t)world->grid_z - 1);

    /* Mark all bricks in range */
    for (int32_t bz = bz0; bz <= bz1; bz++) {
        for (int32_t by = by0; by <= by1; by++) {
            for (int32_t bx = bx0; bx <= bx1; bx++) {
                uint32_t idx = brick_linear_index(world, bx, by, bz);
                dirty_tracker_mark_brick(tracker, idx);
            }
        }
    }
}

bool dirty_tracker_is_dirty(const DirtyTracker* tracker, uint32_t brick_index) {
    if (tracker == NULL || brick_index >= tracker->max_bricks) {
        return false;
    }
    return tracker->dirty_flags[brick_index];
}

uint32_t dirty_tracker_count(const DirtyTracker* tracker) {
    return tracker != NULL ? tracker->dirty_count : 0;
}

/* ============================================================================
 * Section 3: Internal Helpers for SDF Evaluation
 * ============================================================================ */

/**
 * Evaluate SDF for a primitive at a position
 */
static float evaluate_primitive_sdf(const EditEntry* entry, Vec3 pos) {
    switch (entry->primitive) {
        case PRIM_BOX:
            return sdf_box(pos, entry->center, entry->params);

        case PRIM_SPHERE:
            return sdf_sphere(pos, entry->center, entry->params.x);

        case PRIM_CYLINDER:
            return sdf_cylinder(pos, entry->center,
                               entry->params.x, entry->params.y);

        default:
            return 1e6f;  /* Far outside */
    }
}

/**
 * Apply CSG operation to combine SDF values
 */
static float apply_csg_operation(CSGOperation op, float existing_sdf, float new_sdf) {
    switch (op) {
        case CSG_UNION:
            return existing_sdf < new_sdf ? existing_sdf : new_sdf;

        case CSG_SUBTRACT:
            return existing_sdf > -new_sdf ? existing_sdf : -new_sdf;

        case CSG_INTERSECT:
            return existing_sdf > new_sdf ? existing_sdf : new_sdf;

        default:
            return existing_sdf;
    }
}

/**
 * Check if a primitive affects a brick region
 */
static bool primitive_affects_brick(const EditEntry* entry,
                                    Vec3 brick_min, Vec3 brick_max,
                                    float brick_size) {
    /* Add margin for SDF falloff */
    float margin = brick_size * 1.5f;

    Vec3 prim_min, prim_max;

    switch (entry->primitive) {
        case PRIM_BOX:
            prim_min = vec3_sub(entry->center, entry->params);
            prim_max = vec3_add(entry->center, entry->params);
            break;

        case PRIM_SPHERE:
            prim_min = vec3_sub(entry->center, VEC3(entry->params.x,
                                                    entry->params.x,
                                                    entry->params.x));
            prim_max = vec3_add(entry->center, VEC3(entry->params.x,
                                                    entry->params.x,
                                                    entry->params.x));
            break;

        case PRIM_CYLINDER:
            prim_min = VEC3(entry->center.x - entry->params.x,
                           entry->center.y - entry->params.x,
                           entry->center.z - entry->params.y);
            prim_max = VEC3(entry->center.x + entry->params.x,
                           entry->center.y + entry->params.x,
                           entry->center.z + entry->params.y);
            break;

        default:
            return false;
    }

    /* Expand primitive bounds by margin */
    prim_min = vec3_sub(prim_min, VEC3(margin, margin, margin));
    prim_max = vec3_add(prim_max, VEC3(margin, margin, margin));

    /* AABB intersection test */
    return !(brick_max.x < prim_min.x || brick_min.x > prim_max.x ||
             brick_max.y < prim_min.y || brick_min.y > prim_max.y ||
             brick_max.z < prim_min.z || brick_min.z > prim_max.z);
}

/* ============================================================================
 * Section 4: Incremental Regeneration Functions
 * ============================================================================ */

void world_mark_dirty_bricks(WorldBrickMap* world, DirtyTracker* tracker,
                             Vec3 edit_min, Vec3 edit_max) {
    if (world == NULL || tracker == NULL) return;

    /* Expand region by brick size for SDF falloff */
    Vec3 margin = VEC3(world->brick_size_world, world->brick_size_world,
                       world->brick_size_world);
    Vec3 expanded_min = vec3_sub(edit_min, margin);
    Vec3 expanded_max = vec3_add(edit_max, margin);

    dirty_tracker_mark_region(tracker, world, expanded_min, expanded_max);
}

void world_regenerate_brick(WorldBrickMap* world, uint32_t brick_index,
                            const EditList* edits) {
    if (world == NULL || edits == NULL || brick_index >= world->grid_total) {
        return;
    }

    /* Precompute constants */
    const float brick_size = world->brick_size_world;
    const float voxel_size = world->voxel_size;
    const float inv_sdf_scale = world->inv_sdf_scale;
    const float sdf_scale_div_127 = world->sdf_scale_div_127;

    /* Convert linear index to brick coordinates */
    int32_t bz = (int32_t)(brick_index / world->stride_z);
    int32_t rem = brick_index - (uint32_t)bz * world->stride_z;
    int32_t by = rem / (int32_t)world->stride_y;
    int32_t bx = rem - by * (int32_t)world->stride_y;

    /* Get or allocate brick - returns atlas index */
    int32_t atlas_idx = world_alloc_brick(world, bx, by, bz);
    if (atlas_idx == BRICK_EMPTY_INDEX) return;

    /* Get SoA pointers for this brick (demand-paged) */
    uint32_t page_idx = (uint32_t)atlas_idx / ATLAS_PAGE_BRICKS;
    uint32_t brick_in_page = (uint32_t)atlas_idx % ATLAS_PAGE_BRICKS;
    size_t brick_offset = (size_t)brick_in_page * BRICK_VOXELS;
    int8_t* sdf = world->sdf_pages[page_idx] + brick_offset;
    uint8_t* material = world->material_pages[page_idx] + brick_offset;

    /* Compute brick bounds in world space */
    float brick_ox = world->world_min.x + (float)bx * brick_size;
    float brick_oy = world->world_min.y + (float)by * brick_size;
    float brick_oz = world->world_min.z + (float)bz * brick_size;

    Vec3 brick_origin = VEC3(brick_ox, brick_oy, brick_oz);
    Vec3 brick_min = brick_origin;
    Vec3 brick_max = vec3_add(brick_origin, VEC3(brick_size, brick_size, brick_size));

    /* Reset brick to far outside (positive SDF) */
    memset(sdf, 127, BRICK_VOXELS);
    memset(material, 0, BRICK_VOXELS);

    /* Replay all edits that affect this brick */
    for (uint32_t e = 0; e < edits->count; e++) {
        const EditEntry* entry = &edits->entries[e];

        /* Skip if primitive doesn't affect this brick */
        if (!primitive_affects_brick(entry, brick_min, brick_max, brick_size)) {
            continue;
        }

        /* Evaluate SDF for each voxel with stride precomputation */
        for (int32_t vz = 0; vz < BRICK_SIZE; vz++) {
            const uint32_t z_stride = (uint32_t)vz << (BRICK_SHIFT * 2);
            const float pz = brick_oz + ((float)vz + 0.5f) * voxel_size;

            for (int32_t vy = 0; vy < BRICK_SIZE; vy++) {
                const uint32_t yz_stride = z_stride + ((uint32_t)vy << BRICK_SHIFT);
                const float py = brick_oy + ((float)vy + 0.5f) * voxel_size;

                for (int32_t vx = 0; vx < BRICK_SIZE; vx++) {
                    const uint32_t idx = yz_stride + (uint32_t)vx;
                    const float px = brick_ox + ((float)vx + 0.5f) * voxel_size;

                    /* Evaluate primitive SDF */
                    float prim_sdf = evaluate_primitive_sdf(entry, VEC3(px, py, pz));

                    /* Get current SDF and apply CSG operation */
                    float current_sdf = (float)sdf[idx] * sdf_scale_div_127;
                    float new_sdf = apply_csg_operation(entry->op, current_sdf, prim_sdf);

                    /* Update SDF */
                    sdf[idx] = sdf_quantize(new_sdf, inv_sdf_scale);

                    /* Update material for union: surface band (one voxel past
                     * zero-crossing) so raymarcher queries always succeed */
                    if (entry->op == CSG_UNION && prim_sdf < voxel_size &&
                        entry->material > 0) {
                        material[idx] = entry->material;
                    }
                    /* Clear material for subtract operations */
                    else if (entry->op == CSG_SUBTRACT && prim_sdf < voxel_size) {
                        material[idx] = 0;
                    }
                }
            }
        }
    }
}

void world_regenerate_dirty(WorldBrickMap* world, DirtyTracker* tracker,
                            const EditList* edits) {
    if (world == NULL || tracker == NULL || edits == NULL) return;

    /* Regenerate each dirty brick */
    for (uint32_t i = 0; i < tracker->dirty_count; i++) {
        uint32_t brick_index = tracker->dirty_indices[i];
        world_regenerate_brick(world, brick_index, edits);
    }

    /* Clear dirty flags after regeneration */
    dirty_tracker_clear(tracker);
}
