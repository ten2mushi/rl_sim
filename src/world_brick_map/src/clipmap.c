/**
 * Clip Map LOD Implementation
 *
 * Provides multi-resolution world representation using nested grids
 * centered on the camera/drone position for efficient large environment
 * support.
 */

#include "../include/world_brick_map.h"

/* ============================================================================
 * Section 1: Clip Map Lifecycle Functions
 * ============================================================================
 */

ClipMapWorld *clipmap_create(Arena *arena, float base_voxel_size,
                             float base_extent, uint32_t bricks_per_level) {
  if (arena == NULL || base_voxel_size <= 0.0f || base_extent <= 0.0f ||
      bricks_per_level == 0) {
    return NULL;
  }

  ClipMapWorld *clipmap = arena_alloc_type(arena, ClipMapWorld);
  if (clipmap == NULL) {
    return NULL;
  }

  clipmap->focus = VEC3_ZERO;
  clipmap->base_voxel_size = base_voxel_size;
  clipmap->base_extent = base_extent;
  clipmap->arena = arena;

  /* Create each LOD level */
  for (int level = 0; level < CLIPMAP_LEVELS; level++) {
    ClipMapLevel *lod = &clipmap->levels[level];

    /* Each level has 2x coarser voxels and 2x larger extent */
    float scale = (float)(1 << level);
    lod->voxel_size = base_voxel_size * scale;
    lod->extent = base_extent * scale;
    lod->center = VEC3_ZERO;
    lod->grid_origin = VEC3_ZERO;

    /* Create world brick map for this level */
    Vec3 half_ext = VEC3(lod->extent, lod->extent, lod->extent);
    Vec3 world_min = vec3_neg(half_ext);
    Vec3 world_max = half_ext;

    lod->map = world_create(arena, world_min, world_max, lod->voxel_size,
                            bricks_per_level, 0);
    if (lod->map == NULL) {
      return NULL;
    }
  }

  return clipmap;
}

void clipmap_destroy(ClipMapWorld *clipmap) {
  /* No-op: arena handles all memory */
  (void)clipmap;
}

/* ============================================================================
 * Section 2: LOD Level Selection
 * ============================================================================
 */

int clipmap_select_level(const ClipMapWorld *clipmap, Vec3 pos) {
  if (clipmap == NULL)
    return 0;

  /* Compute distance from focus point */
  float dist = vec3_length(vec3_sub(pos, clipmap->focus));

  /* Select level based on distance */
  /* Level 0: 0 to base_extent
   * Level 1: base_extent to 2*base_extent
   * Level N: 2^(N-1)*base_extent to 2^N*base_extent */
  for (int level = 0; level < CLIPMAP_LEVELS - 1; level++) {
    float level_extent = clipmap->base_extent * (float)(1 << level);
    if (dist < level_extent) {
      return level;
    }
  }

  return CLIPMAP_LEVELS - 1;
}

/* ============================================================================
 * Section 3: Focus Update and Toroidal Wrapping
 * ============================================================================
 */

/**
 * Internal: Perform toroidal shift for a single level
 *
 * When the focus moves, we shift the grid origin instead of moving data.
 * This allows O(1) focus updates without reallocating bricks.
 */
static void clipmap_level_toroidal_shift(ClipMapLevel *level, Vec3 new_center) {
  if (level == NULL || level->map == NULL)
    return;

  Vec3 delta = vec3_sub(new_center, level->center);

  /* Compute brick-space shift */
  float inv_brick = 1.0f / level->map->brick_size_world;
  int32_t shift_x = (int32_t)floorf(delta.x * inv_brick);
  int32_t shift_y = (int32_t)floorf(delta.y * inv_brick);
  int32_t shift_z = (int32_t)floorf(delta.z * inv_brick);

  /* Skip if no significant shift */
  if (shift_x == 0 && shift_y == 0 && shift_z == 0) {
    level->center = new_center;
    return;
  }

  /* Update grid origin with wrap-around */
  /* Note: For simplicity, we clear bricks that scroll out.
   * A more sophisticated implementation would track which bricks
   * need regeneration at the new edge. */

  WorldBrickMap *map = level->map;
  int32_t grid_x = (int32_t)map->grid_x;
  int32_t grid_y = (int32_t)map->grid_y;
  int32_t grid_z = (int32_t)map->grid_z;

  /* Clear bricks that have scrolled out of bounds */
  if (abs(shift_x) >= grid_x || abs(shift_y) >= grid_y ||
      abs(shift_z) >= grid_z) {
    /* Complete shift - clear everything */
    world_clear(map);
  } else {
    /* Partial shift - clear edge bricks */
    /* For now, clear all bricks on the shifted edges */
    if (shift_x != 0) {
      int32_t x_start = shift_x > 0 ? 0 : grid_x + shift_x;
      int32_t x_end = shift_x > 0 ? shift_x : grid_x;
      for (int32_t z = 0; z < grid_z; z++) {
        for (int32_t y = 0; y < grid_y; y++) {
          for (int32_t x = x_start; x < x_end && x < grid_x; x++) {
            world_free_brick(map, x, y, z);
          }
        }
      }
    }
    if (shift_y != 0) {
      int32_t y_start = shift_y > 0 ? 0 : grid_y + shift_y;
      int32_t y_end = shift_y > 0 ? shift_y : grid_y;
      for (int32_t z = 0; z < grid_z; z++) {
        for (int32_t y = y_start; y < y_end && y < grid_y; y++) {
          for (int32_t x = 0; x < grid_x; x++) {
            world_free_brick(map, x, y, z);
          }
        }
      }
    }
    if (shift_z != 0) {
      int32_t z_start = shift_z > 0 ? 0 : grid_z + shift_z;
      int32_t z_end = shift_z > 0 ? shift_z : grid_z;
      for (int32_t z = z_start; z < z_end && z < grid_z; z++) {
        for (int32_t y = 0; y < grid_y; y++) {
          for (int32_t x = 0; x < grid_x; x++) {
            world_free_brick(map, x, y, z);
          }
        }
      }
    }
  }

  /* Update origin tracking */
  level->grid_origin = vec3_add(level->grid_origin,
                                VEC3((float)shift_x * map->brick_size_world,
                                     (float)shift_y * map->brick_size_world,
                                     (float)shift_z * map->brick_size_world));

  /* Update world bounds to be centered on new position */
  Vec3 half_ext = VEC3(level->extent, level->extent, level->extent);
  map->world_min = vec3_sub(new_center, half_ext);
  map->world_max = vec3_add(new_center, half_ext);
  map->world_size = vec3_scale(half_ext, 2.0f);

  level->center = new_center;
}

void clipmap_update_focus(ClipMapWorld *clipmap, Vec3 new_focus) {
  if (clipmap == NULL)
    return;

  clipmap->focus = new_focus;

  /* Update each level's center position */
  for (int level = 0; level < CLIPMAP_LEVELS; level++) {
    clipmap_level_toroidal_shift(&clipmap->levels[level], new_focus);
  }
}

/* ============================================================================
 * Section 4: SDF Query with LOD
 * ============================================================================
 */

float clipmap_sdf_query(const ClipMapWorld *clipmap, Vec3 pos) {
  if (clipmap == NULL)
    return 1e6f;

  /* Select appropriate LOD level */
  int level = clipmap_select_level(clipmap, pos);
  const ClipMapLevel *lod = &clipmap->levels[level];

  if (lod->map == NULL) {
    return 1e6f;
  }

  /* Query SDF from selected level */
  return world_sdf_query(lod->map, pos);
}

/* ============================================================================
 * Section 5: Raymarching with LOD Transitions
 * ============================================================================
 */

RayHit clipmap_raymarch(const ClipMapWorld *clipmap, Vec3 origin,
                        Vec3 direction, float max_distance) {
  RayHit hit = {0};
  hit.hit = false;
  hit.distance = max_distance;

  if (clipmap == NULL)
    return hit;

  float t = 0.0f;

  for (int step = 0; step < RAYMARCH_MAX_STEPS; step++) {
    Vec3 pos = vec3_add(origin, vec3_scale(direction, t));

    /* Select LOD level based on distance from focus */
    int level = clipmap_select_level(clipmap, pos);
    const ClipMapLevel *lod = &clipmap->levels[level];

    if (lod->map == NULL) {
      t += RAYMARCH_EPSILON;
      continue;
    }

    /* Check if position is within this level's bounds */
    if (!world_contains(lod->map, pos)) {
      /* Move to next level boundary or terminate */
      t += lod->voxel_size;
      if (t > max_distance) {
        hit.distance = max_distance;
        return hit;
      }
      continue;
    }

    /* Query SDF at this level */
    float dist = world_sdf_query(lod->map, pos);

    /* Hit detected */
    if (dist < RAYMARCH_HIT_DIST) {
      hit.hit = true;
      hit.position = pos;
      hit.distance = t;
      hit.material = world_material_query(lod->map, pos);

      /* Compute normal using finest available level */
      int finest_level = clipmap_select_level(clipmap, pos);
      const ClipMapLevel *finest_lod = &clipmap->levels[finest_level];
      if (finest_lod->map != NULL && world_contains(finest_lod->map, pos)) {
        hit.normal = world_sdf_normal(finest_lod->map, pos);
      } else {
        hit.normal = world_sdf_normal(lod->map, pos);
      }

      return hit;
    }

    /* Check max distance */
    if (t > max_distance) {
      hit.distance = max_distance;
      return hit;
    }

    /* Sphere trace step - use level's voxel size as minimum step */
    float min_step = lod->voxel_size * 0.5f;
    t += (dist > min_step) ? dist : min_step;
  }

  hit.distance = t;
  return hit;
}

/* ============================================================================
 * Section 6: Primitive Generation for Clip Maps
 * ============================================================================
 */

void clipmap_set_sphere(ClipMapWorld *clipmap, Vec3 center, float radius,
                        uint8_t material) {
  if (clipmap == NULL || radius <= 0.0f)
    return;

  /* Add sphere to all levels that it intersects */
  for (int level = 0; level < CLIPMAP_LEVELS; level++) {
    ClipMapLevel *lod = &clipmap->levels[level];
    if (lod->map == NULL)
      continue;

    /* Check if sphere intersects this level's bounds */
    Vec3 level_min = lod->map->world_min;
    Vec3 level_max = lod->map->world_max;

    Vec3 sphere_min = vec3_sub(center, VEC3(radius, radius, radius));
    Vec3 sphere_max = vec3_add(center, VEC3(radius, radius, radius));

    /* AABB intersection test */
    if (sphere_max.x < level_min.x || sphere_min.x > level_max.x ||
        sphere_max.y < level_min.y || sphere_min.y > level_max.y ||
        sphere_max.z < level_min.z || sphere_min.z > level_max.z) {
      continue;
    }

    /* Add sphere to this level */
    world_set_sphere(lod->map, center, radius, material);
  }
}

void clipmap_set_box(ClipMapWorld *clipmap, Vec3 center, Vec3 half_size,
                     uint8_t material) {
  if (clipmap == NULL)
    return;

  /* Add box to all levels that it intersects */
  for (int level = 0; level < CLIPMAP_LEVELS; level++) {
    ClipMapLevel *lod = &clipmap->levels[level];
    if (lod->map == NULL)
      continue;

    Vec3 level_min = lod->map->world_min;
    Vec3 level_max = lod->map->world_max;

    Vec3 box_min = vec3_sub(center, half_size);
    Vec3 box_max = vec3_add(center, half_size);

    /* AABB intersection test */
    if (box_max.x < level_min.x || box_min.x > level_max.x ||
        box_max.y < level_min.y || box_min.y > level_max.y ||
        box_max.z < level_min.z || box_min.z > level_max.z) {
      continue;
    }

    /* Add box to this level */
    world_set_box(lod->map, center, half_size, material);
  }
}
