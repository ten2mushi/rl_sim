/**
 * World Brick Map Implementation
 *
 * Core sparse SDF storage with 8x8x8 brick atlas, trilinear interpolation,
 * primitive generation, and SIMD-optimized raymarching.
 */

#include "../include/world_brick_map.h"

/* ============================================================================
 * Section 1: Internal Helper Macros
 * ============================================================================
 */

#define MIN3(a, b, c) simd_min_ps(simd_min_ps(a, b), c)
#define MAX3(a, b, c) simd_max_ps(simd_max_ps(a, b), c)

/* ============================================================================
 * Section 2: World Lifecycle Functions
 * ============================================================================
 */

WorldBrickMap *world_create(Arena *arena, Vec3 world_min, Vec3 world_max,
                            float voxel_size, uint32_t max_bricks,
                            uint32_t max_materials) {
  if (arena == NULL || voxel_size <= 0.0f || max_bricks == 0) {
    return NULL;
  }

  /* Calculate world dimensions */
  Vec3 world_size = vec3_sub(world_max, world_min);
  if (world_size.x <= 0.0f || world_size.y <= 0.0f || world_size.z <= 0.0f) {
    return NULL;
  }

  /* Calculate grid dimensions in bricks */
  float brick_world_size = voxel_size * BRICK_SIZE;
  uint32_t grid_x = (uint32_t)ceilf(world_size.x / brick_world_size);
  uint32_t grid_y = (uint32_t)ceilf(world_size.y / brick_world_size);
  uint32_t grid_z = (uint32_t)ceilf(world_size.z / brick_world_size);
  uint32_t grid_total = grid_x * grid_y * grid_z;

  /* Calculate page counts for demand-paged atlas */
  uint32_t max_pages = (max_bricks + ATLAS_PAGE_BRICKS - 1) / ATLAS_PAGE_BRICKS;
  if (max_pages > MAX_ATLAS_PAGES) {
    max_pages = MAX_ATLAS_PAGES;
  }
  /* Round max_bricks to page boundary */
  max_bricks = max_pages * ATLAS_PAGE_BRICKS;

  /* Allocate world structure */
  WorldBrickMap *world = arena_alloc_type(arena, WorldBrickMap);
  if (world == NULL) {
    return NULL;
  }

  /* Allocate brick indices grid (first level of sparse structure) */
  world->brick_indices = arena_alloc_array(arena, int32_t, grid_total);
  if (world->brick_indices == NULL) {
    return NULL;
  }

  /* Initialize all brick indices to empty */
  for (uint32_t i = 0; i < grid_total; i++) {
    world->brick_indices[i] = BRICK_EMPTY_INDEX;
  }

  /* Allocate page pointer arrays (demand-paged - pages allocated on use) */
  world->sdf_pages = arena_alloc_array(arena, int8_t *, max_pages);
  if (world->sdf_pages == NULL) {
    return NULL;
  }
  world->material_pages = arena_alloc_array(arena, uint8_t *, max_pages);
  if (world->material_pages == NULL) {
    return NULL;
  }

  /* Initialize all page pointers to NULL (pages allocated on demand) */
  for (uint32_t i = 0; i < max_pages; i++) {
    world->sdf_pages[i] = NULL;
    world->material_pages[i] = NULL;
  }

  /* Allocate free list (for max possible bricks) */
  world->free_list = arena_alloc_array(arena, uint32_t, max_bricks);
  if (world->free_list == NULL) {
    return NULL;
  }

  /* Initialize world state */
  world->grid_x = grid_x;
  world->grid_y = grid_y;
  world->grid_z = grid_z;
  world->grid_total = grid_total;
  world->max_pages = max_pages;
  world->page_count = 0;
  world->max_bricks = max_bricks;
  world->atlas_count = 0;
  world->free_count = 0;

  /* Initialize uniform brick counters */
  world->uniform_outside_count = 0;
  world->uniform_inside_count = 0;

  world->world_min = world_min;
  world->world_max = world_max;
  world->world_size = world_size;
  world->voxel_size = voxel_size;
  world->inv_voxel_size = 1.0f / voxel_size;
  world->brick_size_world = brick_world_size;
  world->inv_brick_size = 1.0f / brick_world_size;

  /* Narrow-band distance threshold */
  world->narrow_band_dist = NARROW_BAND_VOXELS * voxel_size;

  /* SDF quantization scale: covers ~1 brick diagonal worth of distance */
  world->sdf_scale = brick_world_size * 1.5f;
  world->inv_sdf_scale = 1.0f / world->sdf_scale;
  world->sdf_scale_div_127 = world->sdf_scale / 127.0f;

  /* Precompute strides */
  world->stride_y = grid_x;
  world->stride_z = grid_x * grid_y;

  /* Initialize material system */
  if (max_materials == 0) {
    max_materials = 256; /* Default to 256 materials */
  }
  world->max_materials = max_materials;
  world->material_count = 0;

  /* Allocate material registry */
  world->materials = arena_alloc_array(arena, MaterialMetadata, max_materials);
  if (world->materials == NULL) {
    return NULL;
  }

  /* Register default material (ID=0, white, named "default") */
  world->material_count = 0;
  world_register_material(world, "default", (Vec3){1.0f, 1.0f, 1.0f});

  /* Initialize feature channel system (arena doesn't zero memory!) */
  world->feature_channel_count = 0;

  /* Initialize page dirty flags (all clean on creation) */
  memset(world->page_dirty, 0, sizeof(world->page_dirty));

  world->arena = arena;

  return world;
}

void world_destroy(WorldBrickMap *world) {
  /* No-op: arena handles all memory */
  (void)world;
}

void world_clear(WorldBrickMap *world) {
  if (world == NULL)
    return;

  /* Reset all brick indices to empty */
  for (uint32_t i = 0; i < world->grid_total; i++) {
    world->brick_indices[i] = BRICK_EMPTY_INDEX;
  }

  /* Reset atlas state (pages remain allocated but unused) */
  world->atlas_count = 0;
  world->free_count = 0;
  world->uniform_outside_count = 0;
  world->uniform_inside_count = 0;
  /* Note: page_count not reset - pages remain allocated for reuse */
}

WorldStats world_get_stats(const WorldBrickMap *world) {
  WorldStats stats = {0};
  if (world == NULL)
    return stats;

  stats.total_bricks = world->max_bricks;
  stats.active_bricks = world->atlas_count - world->free_count;
  stats.uniform_outside = world->uniform_outside_count;
  stats.uniform_inside = world->uniform_inside_count;
  stats.free_list_count = world->free_count;
  stats.pages_allocated = world->page_count;

  /* Grid memory: brick indices array */
  stats.grid_memory = world->grid_total * sizeof(int32_t);

  /* Atlas memory: only count actually allocated pages */
  size_t page_size = (size_t)ATLAS_PAGE_BRICKS * BRICK_VOXELS;
  stats.atlas_memory =
      (size_t)world->page_count * page_size * 2; /* SDF + material */

  /* Total: struct + grid + page pointers + allocated pages + free list */
  stats.total_memory = sizeof(WorldBrickMap) + stats.grid_memory +
                       world->max_pages * sizeof(int8_t *) * 2 +
                       stats.atlas_memory +
                       world->max_bricks * sizeof(uint32_t);

  stats.fill_ratio = (stats.total_bricks > 0) ? (float)stats.active_bricks /
                                                    (float)stats.total_bricks
                                              : 0.0f;

  return stats;
}

size_t world_memory_size(uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                         uint32_t max_bricks) {
  uint32_t grid_total = grid_x * grid_y * grid_z;
  /* Calculate max pages needed */
  uint32_t max_pages = (max_bricks + ATLAS_PAGE_BRICKS - 1) / ATLAS_PAGE_BRICKS;
  if (max_pages > MAX_ATLAS_PAGES)
    max_pages = MAX_ATLAS_PAGES;

  /* Base memory: struct + grid + page pointers + free list */
  /* Note: this is minimum - actual atlas pages allocated on demand */
  return sizeof(WorldBrickMap) + grid_total * sizeof(int32_t) +
         max_pages * sizeof(int8_t *) * 2 + max_bricks * sizeof(uint32_t);
}

/* ============================================================================
 * Section 2b: Material Registry Functions
 * ============================================================================
 */

uint8_t world_register_material(WorldBrickMap *world, const char *name,
                                Vec3 diffuse_color) {
  if (world == NULL || name == NULL || name[0] == '\0') {
    return 255; /* Invalid input */
  }

  /* Check if material already exists - return existing ID (no update) */
  for (uint32_t i = 0; i < world->material_count; i++) {
    if (strncmp(world->materials[i].name, name,
                sizeof(world->materials[i].name)) == 0) {
      return world->materials[i].id; /* Return existing ID */
    }
  }

  /* Check if we have room for a new material */
  if (world->material_count >= world->max_materials ||
      world->material_count >= 256) {
    return 255; /* Material table full */
  }

  /* Register new material */
  uint8_t new_id = (uint8_t)world->material_count;
  strncpy(world->materials[new_id].name, name,
          sizeof(world->materials[new_id].name) - 1);
  world->materials[new_id].name[sizeof(world->materials[new_id].name) - 1] =
      '\0';
  world->materials[new_id].diffuse_color = diffuse_color;
  world->materials[new_id].id = new_id;
  world->material_count++;

  return new_id;
}

uint8_t world_find_material(const WorldBrickMap *world, const char *name) {
  if (world == NULL || name == NULL || name[0] == '\0') {
    return 0; /* Return default material */
  }

  for (uint32_t i = 0; i < world->material_count; i++) {
    if (strncmp(world->materials[i].name, name,
                sizeof(world->materials[i].name)) == 0) {
      return world->materials[i].id;
    }
  }

  return 0; /* Not found, return default material */
}

const MaterialMetadata *world_get_material(const WorldBrickMap *world,
                                           uint8_t material_id) {
  if (world == NULL || material_id >= world->material_count) {
    return NULL;
  }

  return &world->materials[material_id];
}

/* ============================================================================
 * Section 3: Coordinate Transformation Functions
 * ============================================================================
 */

void world_pos_to_brick(const WorldBrickMap *world, Vec3 pos, int32_t *bx,
                        int32_t *by, int32_t *bz) {
  Vec3 rel = vec3_sub(pos, world->world_min);
  float inv_brick = 1.0f / world->brick_size_world;

  *bx = (int32_t)floorf(rel.x * inv_brick);
  *by = (int32_t)floorf(rel.y * inv_brick);
  *bz = (int32_t)floorf(rel.z * inv_brick);
}

void world_pos_to_voxel(const WorldBrickMap *world, Vec3 pos, int32_t bx,
                        int32_t by, int32_t bz, int32_t *vx, int32_t *vy,
                        int32_t *vz) {
  /* Compute brick origin in world space */
  Vec3 brick_origin =
      VEC3(world->world_min.x + (float)bx * world->brick_size_world,
           world->world_min.y + (float)by * world->brick_size_world,
           world->world_min.z + (float)bz * world->brick_size_world);

  /* Convert to local voxel coordinates */
  Vec3 local = vec3_scale(vec3_sub(pos, brick_origin), world->inv_voxel_size);

  *vx = (int32_t)floorf(local.x);
  *vy = (int32_t)floorf(local.y);
  *vz = (int32_t)floorf(local.z);

  /* Clamp to valid range [0, 7] */
  *vx = max_i32(0, min_i32(*vx, BRICK_MASK));
  *vy = max_i32(0, min_i32(*vy, BRICK_MASK));
  *vz = max_i32(0, min_i32(*vz, BRICK_MASK));
}

bool world_contains(const WorldBrickMap *world, Vec3 pos) {
  return pos.x >= world->world_min.x && pos.x < world->world_max.x &&
         pos.y >= world->world_min.y && pos.y < world->world_max.y &&
         pos.z >= world->world_min.z && pos.z < world->world_max.z;
}

bool world_brick_valid(const WorldBrickMap *world, int32_t bx, int32_t by,
                       int32_t bz) {
  return bx >= 0 && bx < (int32_t)world->grid_x && by >= 0 &&
         by < (int32_t)world->grid_y && bz >= 0 && bz < (int32_t)world->grid_z;
}

/* ============================================================================
 * Section 4: Brick Management Functions (SoA Layout)
 * ============================================================================
 */

int32_t world_get_brick_index(const WorldBrickMap *world, int32_t bx,
                              int32_t by, int32_t bz) {
  if (!world_brick_valid(world, bx, by, bz)) {
    return BRICK_EMPTY_INDEX;
  }

  uint32_t grid_idx = brick_linear_index(world, bx, by, bz);
  return world->brick_indices[grid_idx];
}

/**
 * Internal: Allocate a new page if needed
 */
static bool world_ensure_page(WorldBrickMap *world, uint32_t page_idx) {
  if (page_idx >= world->max_pages) {
    return false;
  }

  /* Page already allocated */
  if (world->sdf_pages[page_idx] != NULL) {
    return true;
  }

  /* Allocate new page for SDF data */
  size_t page_size = (size_t)ATLAS_PAGE_BRICKS * BRICK_VOXELS;
  world->sdf_pages[page_idx] =
      (int8_t *)arena_alloc_aligned(world->arena, page_size, BRICK_ALIGNMENT);
  if (world->sdf_pages[page_idx] == NULL) {
    return false;
  }

  /* Allocate new page for material data */
  world->material_pages[page_idx] =
      (uint8_t *)arena_alloc_aligned(world->arena, page_size, BRICK_ALIGNMENT);
  if (world->material_pages[page_idx] == NULL) {
    return false;
  }

  /* Allocate pages for user-defined feature channels */
  for (uint32_t ch = 0; ch < world->feature_channel_count; ch++) {
    VoxelChannel *fc = &world->feature_channels[ch];
    if (fc->pages != NULL && fc->pages[page_idx] == NULL) {
      size_t ch_page_size =
          (size_t)ATLAS_PAGE_BRICKS * fc->bytes_per_brick;
      fc->pages[page_idx] =
          arena_alloc_aligned(world->arena, ch_page_size, BRICK_ALIGNMENT);
      if (fc->pages[page_idx] == NULL) {
        return false;
      }
      memset(fc->pages[page_idx], 0, ch_page_size);
      if (page_idx >= fc->page_count) {
        fc->page_count = page_idx + 1;
      }
    }
  }

  /* Track allocated pages */
  if (page_idx >= world->page_count) {
    world->page_count = page_idx + 1;
  }

  return true;
}

int32_t world_alloc_brick(WorldBrickMap *world, int32_t bx, int32_t by,
                          int32_t bz) {
  if (!world_brick_valid(world, bx, by, bz)) {
    return BRICK_EMPTY_INDEX;
  }

  uint32_t grid_idx = brick_linear_index(world, bx, by, bz);
  int32_t current_idx = world->brick_indices[grid_idx];

  /* Return existing atlas index if already allocated (not uniform/empty) */
  if (current_idx >= 0) {
    /* Mark page dirty - caller will modify this brick's data */
    uint32_t pg = (uint32_t)current_idx / ATLAS_PAGE_BRICKS;
    if (pg < MAX_ATLAS_PAGES) world->page_dirty[pg] = true;
    return current_idx;
  }

  /* Track if brick was uniform inside (needed for correct initialization) */
  bool was_uniform_inside = (current_idx == BRICK_UNIFORM_INSIDE);

  /* If was uniform, decrement the counter */
  if (current_idx == BRICK_UNIFORM_OUTSIDE) {
    world->uniform_outside_count--;
  } else if (was_uniform_inside) {
    world->uniform_inside_count--;
  }

  /* Try to reuse from free list */
  uint32_t new_idx;
  if (world->free_count > 0) {
    new_idx = world->free_list[--world->free_count];
  } else if (world->atlas_count < world->max_bricks) {
    new_idx = world->atlas_count++;
  } else {
    /* Atlas full */
    return BRICK_EMPTY_INDEX;
  }

  /* Ensure the page exists (allocate on demand) */
  uint32_t page_idx = new_idx / ATLAS_PAGE_BRICKS;
  if (!world_ensure_page(world, page_idx)) {
    /* Failed to allocate page, put index back */
    if (world->free_count < world->max_bricks) {
      world->free_list[world->free_count++] = new_idx;
    }
    return BRICK_EMPTY_INDEX;
  }

  /* Mark new page as dirty */
  world->page_dirty[page_idx] = true;

  /* Get brick offset within page */
  uint32_t brick_in_page = new_idx % ATLAS_PAGE_BRICKS;
  size_t brick_offset = (size_t)brick_in_page * BRICK_VOXELS;

  /* Initialize brick based on previous state:
   * - Was uniform inside: SDF = -127 (far inside), preserves "filled" state
   * - Was uniform outside or empty: SDF = +127 (far outside) */
  int8_t init_sdf = was_uniform_inside ? -127 : 127;
  memset(world->sdf_pages[page_idx] + brick_offset, init_sdf, BRICK_VOXELS);
  memset(world->material_pages[page_idx] + brick_offset, 0, BRICK_VOXELS);

  /* Link brick to grid */
  world->brick_indices[grid_idx] = (int32_t)new_idx;

  return (int32_t)new_idx;
}

void world_free_brick(WorldBrickMap *world, int32_t bx, int32_t by,
                      int32_t bz) {
  if (!world_brick_valid(world, bx, by, bz)) {
    return;
  }

  uint32_t grid_idx = brick_linear_index(world, bx, by, bz);
  int32_t atlas_idx = world->brick_indices[grid_idx];

  /* Handle uniform bricks - just decrement counter */
  if (atlas_idx == BRICK_UNIFORM_OUTSIDE) {
    world->uniform_outside_count--;
    world->brick_indices[grid_idx] = BRICK_EMPTY_INDEX;
    return;
  }
  if (atlas_idx == BRICK_UNIFORM_INSIDE) {
    world->uniform_inside_count--;
    world->brick_indices[grid_idx] = BRICK_EMPTY_INDEX;
    return;
  }

  /* Empty brick - nothing to free */
  if (atlas_idx == BRICK_EMPTY_INDEX) {
    return;
  }

  /* Add atlas index to free list */
  if (world->free_count < world->max_bricks) {
    world->free_list[world->free_count++] = (uint32_t)atlas_idx;
  }

  /* Unlink from grid */
  world->brick_indices[grid_idx] = BRICK_EMPTY_INDEX;
}

/* ============================================================================
 * Section 5: SDF Query Functions (SIMD Optimized)
 * ============================================================================
 */

/**
 * Sample a single dequantized SDF voxel, handling cross-brick boundaries.
 *
 * Voxel coordinates (vx, vy, vz) may be outside [0, BRICK_SIZE-1] for
 * the brick at (bx, by, bz). This function wraps them into the correct
 * neighboring brick and returns the dequantized SDF value.
 */
static inline float world_sample_voxel_cross(const WorldBrickMap *world,
                                             int32_t bx, int32_t by,
                                             int32_t bz, int32_t vx,
                                             int32_t vy, int32_t vz) {
  /* Wrap voxel coordinates into neighboring bricks */
  if (vx >= BRICK_SIZE) {
    bx++;
    vx -= BRICK_SIZE;
  } else if (vx < 0) {
    bx--;
    vx += BRICK_SIZE;
  }
  if (vy >= BRICK_SIZE) {
    by++;
    vy -= BRICK_SIZE;
  } else if (vy < 0) {
    by--;
    vy += BRICK_SIZE;
  }
  if (vz >= BRICK_SIZE) {
    bz++;
    vz -= BRICK_SIZE;
  } else if (vz < 0) {
    bz--;
    vz += BRICK_SIZE;
  }

  /* Bounds check */
  if (bx < 0 || bx >= (int32_t)world->grid_x || by < 0 ||
      by >= (int32_t)world->grid_y || bz < 0 ||
      bz >= (int32_t)world->grid_z) {
    return world->sdf_scale;
  }

  uint32_t grid_idx = (uint32_t)bx + (uint32_t)by * world->stride_y +
                      (uint32_t)bz * world->stride_z;
  int32_t atlas_idx = world->brick_indices[grid_idx];

  if (atlas_idx == BRICK_EMPTY_INDEX || atlas_idx == BRICK_UNIFORM_OUTSIDE)
    return world->sdf_scale;
  if (atlas_idx == BRICK_UNIFORM_INSIDE)
    return -world->sdf_scale;

  uint32_t page_idx = (uint32_t)atlas_idx / ATLAS_PAGE_BRICKS;
  uint32_t brick_in_page = (uint32_t)atlas_idx % ATLAS_PAGE_BRICKS;
  const int8_t *sdf =
      world->sdf_pages[page_idx] + (size_t)brick_in_page * BRICK_VOXELS;

  uint32_t idx = (uint32_t)vx + ((uint32_t)vy << BRICK_SHIFT) +
                 ((uint32_t)vz << (BRICK_SHIFT * 2));
  return (float)sdf[idx] * world->sdf_scale_div_127;
}

/**
 * SIMD-optimized SDF query with trilinear interpolation
 *
 * Optimization notes:
 * - Precomputed inv_brick_size eliminates division
 * - Fused dequantize: multiply by sdf_scale_div_127 instead of divide by 127
 * - SIMD trilinear uses 4-wide ops where beneficial
 * - Branch-free clamping with fminf/fmaxf
 * - Uniform bricks return constant values without atlas access
 * - Cross-brick interpolation at brick boundaries for correct SDF continuity
 */
float world_sdf_query(const WorldBrickMap *world, Vec3 pos) {
  if (world == NULL)
    return 1e6f;

  /* Convert world position to brick coordinates using precomputed inverse */
  Vec3 rel = vec3_sub(pos, world->world_min);
  int32_t bx = (int32_t)floorf(rel.x * world->inv_brick_size);
  int32_t by = (int32_t)floorf(rel.y * world->inv_brick_size);
  int32_t bz = (int32_t)floorf(rel.z * world->inv_brick_size);

  /* Branch-free bounds check using precomputed grid dimensions */
  if (FOUNDATION_UNLIKELY(bx < 0 || bx >= (int32_t)world->grid_x || by < 0 ||
                          by >= (int32_t)world->grid_y || bz < 0 ||
                          bz >= (int32_t)world->grid_z)) {
    return world->sdf_scale;
  }

  /* Get atlas index */
  uint32_t grid_idx = (uint32_t)bx + (uint32_t)by * world->stride_y +
                      (uint32_t)bz * world->stride_z;
  int32_t atlas_idx = world->brick_indices[grid_idx];

  /* Handle special indices - no atlas lookup needed */
  if (atlas_idx == BRICK_EMPTY_INDEX || atlas_idx == BRICK_UNIFORM_OUTSIDE) {
    return world->sdf_scale; /* Far outside */
  }
  if (atlas_idx == BRICK_UNIFORM_INSIDE) {
    return -world->sdf_scale; /* Deep inside */
  }

  /* Get SDF data pointer for this brick (demand-paged) */
  uint32_t page_idx = (uint32_t)atlas_idx / ATLAS_PAGE_BRICKS;
  uint32_t brick_in_page = (uint32_t)atlas_idx % ATLAS_PAGE_BRICKS;
  const int8_t *sdf =
      world->sdf_pages[page_idx] + (size_t)brick_in_page * BRICK_VOXELS;

  /* Compute brick origin using precomputed brick_size_world */
  float brick_ox = world->world_min.x + (float)bx * world->brick_size_world;
  float brick_oy = world->world_min.y + (float)by * world->brick_size_world;
  float brick_oz = world->world_min.z + (float)bz * world->brick_size_world;

  /* Compute fractional voxel coordinates for interpolation */
  float local_x = (pos.x - brick_ox) * world->inv_voxel_size;
  float local_y = (pos.y - brick_oy) * world->inv_voxel_size;
  float local_z = (pos.z - brick_oz) * world->inv_voxel_size;

  int32_t x0 = (int32_t)floorf(local_x);
  int32_t y0 = (int32_t)floorf(local_y);
  int32_t z0 = (int32_t)floorf(local_z);

  float fx = local_x - (float)x0;
  float fy = local_y - (float)y0;
  float fz = local_z - (float)z0;

  float c000, c100, c010, c110, c001, c101, c011, c111;

  /* Check if any interpolation corner crosses into a neighboring brick.
   * x0 in [0, BRICK_MASK-1] means x0 and x0+1 are both in [0, BRICK_MASK]. */
  if (FOUNDATION_UNLIKELY(x0 < 0 || x0 > BRICK_MASK - 1 || y0 < 0 ||
                          y0 > BRICK_MASK - 1 || z0 < 0 ||
                          z0 > BRICK_MASK - 1)) {
    /* Slow path: cross-brick interpolation via neighbor lookups */
    int32_t x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
    c000 = world_sample_voxel_cross(world, bx, by, bz, x0, y0, z0);
    c100 = world_sample_voxel_cross(world, bx, by, bz, x1, y0, z0);
    c010 = world_sample_voxel_cross(world, bx, by, bz, x0, y1, z0);
    c110 = world_sample_voxel_cross(world, bx, by, bz, x1, y1, z0);
    c001 = world_sample_voxel_cross(world, bx, by, bz, x0, y0, z1);
    c101 = world_sample_voxel_cross(world, bx, by, bz, x1, y0, z1);
    c011 = world_sample_voxel_cross(world, bx, by, bz, x0, y1, z1);
    c111 = world_sample_voxel_cross(world, bx, by, bz, x1, y1, z1);
  } else {
    /* Fast path: all 8 corners within same brick */
    int32_t x1 = x0 + 1;
    int32_t y1 = y0 + 1;
    int32_t z1 = z0 + 1;

    /* Precompute voxel indices: index = x + (y << 3) + (z << 6) */
    int32_t idx_y0 = y0 << BRICK_SHIFT;
    int32_t idx_y1 = y1 << BRICK_SHIFT;
    int32_t idx_z0 = z0 << (BRICK_SHIFT * 2);
    int32_t idx_z1 = z1 << (BRICK_SHIFT * 2);

    /* Sample 8 corners with fused dequantize (multiply instead of divide) */
    float scale = world->sdf_scale_div_127;
    c000 = (float)sdf[x0 + idx_y0 + idx_z0] * scale;
    c100 = (float)sdf[x1 + idx_y0 + idx_z0] * scale;
    c010 = (float)sdf[x0 + idx_y1 + idx_z0] * scale;
    c110 = (float)sdf[x1 + idx_y1 + idx_z0] * scale;
    c001 = (float)sdf[x0 + idx_y0 + idx_z1] * scale;
    c101 = (float)sdf[x1 + idx_y0 + idx_z1] * scale;
    c011 = (float)sdf[x0 + idx_y1 + idx_z1] * scale;
    c111 = (float)sdf[x1 + idx_y1 + idx_z1] * scale;
  }

  /* Trilinear interpolation using FMA pattern */
  float c00 = c000 + fx * (c100 - c000);
  float c01 = c001 + fx * (c101 - c001);
  float c10 = c010 + fx * (c110 - c010);
  float c11 = c011 + fx * (c111 - c011);

  float c0 = c00 + fy * (c10 - c00);
  float c1 = c01 + fy * (c11 - c01);

  return c0 + fz * (c1 - c0);
}

float world_sdf_query_nearest(const WorldBrickMap *world, Vec3 pos) {
  if (world == NULL)
    return 1e6f;

  /* Convert to brick coords using precomputed inverse */
  Vec3 rel = vec3_sub(pos, world->world_min);
  int32_t bx = (int32_t)floorf(rel.x * world->inv_brick_size);
  int32_t by = (int32_t)floorf(rel.y * world->inv_brick_size);
  int32_t bz = (int32_t)floorf(rel.z * world->inv_brick_size);

  if (FOUNDATION_UNLIKELY(bx < 0 || bx >= (int32_t)world->grid_x || by < 0 ||
                          by >= (int32_t)world->grid_y || bz < 0 ||
                          bz >= (int32_t)world->grid_z)) {
    return world->sdf_scale;
  }

  uint32_t grid_idx = (uint32_t)bx + (uint32_t)by * world->stride_y +
                      (uint32_t)bz * world->stride_z;
  int32_t atlas_idx = world->brick_indices[grid_idx];

  /* Handle special indices */
  if (atlas_idx == BRICK_EMPTY_INDEX || atlas_idx == BRICK_UNIFORM_OUTSIDE) {
    return world->sdf_scale;
  }
  if (atlas_idx == BRICK_UNIFORM_INSIDE) {
    return -world->sdf_scale;
  }

  /* Get voxel coordinates */
  float brick_ox = world->world_min.x + (float)bx * world->brick_size_world;
  float brick_oy = world->world_min.y + (float)by * world->brick_size_world;
  float brick_oz = world->world_min.z + (float)bz * world->brick_size_world;

  int32_t vx = (int32_t)floorf((pos.x - brick_ox) * world->inv_voxel_size);
  int32_t vy = (int32_t)floorf((pos.y - brick_oy) * world->inv_voxel_size);
  int32_t vz = (int32_t)floorf((pos.z - brick_oz) * world->inv_voxel_size);

  /* Clamp to valid range */
  vx = vx < 0 ? 0 : (vx > BRICK_MASK ? BRICK_MASK : vx);
  vy = vy < 0 ? 0 : (vy > BRICK_MASK ? BRICK_MASK : vy);
  vz = vz < 0 ? 0 : (vz > BRICK_MASK ? BRICK_MASK : vz);

  uint32_t voxel_idx = (uint32_t)vx + ((uint32_t)vy << BRICK_SHIFT) +
                       ((uint32_t)vz << (BRICK_SHIFT * 2));

  /* Get SDF data (demand-paged) */
  uint32_t page_idx = (uint32_t)atlas_idx / ATLAS_PAGE_BRICKS;
  uint32_t brick_in_page = (uint32_t)atlas_idx % ATLAS_PAGE_BRICKS;
  const int8_t *sdf =
      world->sdf_pages[page_idx] + (size_t)brick_in_page * BRICK_VOXELS;

  return (float)sdf[voxel_idx] * world->sdf_scale_div_127;
}

uint8_t world_material_query(const WorldBrickMap *world, Vec3 pos) {
  if (world == NULL)
    return 0;

  /* Convert to brick coords using precomputed inverse */
  Vec3 rel = vec3_sub(pos, world->world_min);
  int32_t bx = (int32_t)floorf(rel.x * world->inv_brick_size);
  int32_t by = (int32_t)floorf(rel.y * world->inv_brick_size);
  int32_t bz = (int32_t)floorf(rel.z * world->inv_brick_size);

  if (FOUNDATION_UNLIKELY(bx < 0 || bx >= (int32_t)world->grid_x || by < 0 ||
                          by >= (int32_t)world->grid_y || bz < 0 ||
                          bz >= (int32_t)world->grid_z)) {
    return 0;
  }

  uint32_t grid_idx = (uint32_t)bx + (uint32_t)by * world->stride_y +
                      (uint32_t)bz * world->stride_z;
  int32_t atlas_idx = world->brick_indices[grid_idx];

  /* Uniform/empty bricks have no material (air) */
  if (atlas_idx < 0) {
    return 0;
  }

  /* Get voxel coordinates */
  float brick_ox = world->world_min.x + (float)bx * world->brick_size_world;
  float brick_oy = world->world_min.y + (float)by * world->brick_size_world;
  float brick_oz = world->world_min.z + (float)bz * world->brick_size_world;

  int32_t vx = (int32_t)floorf((pos.x - brick_ox) * world->inv_voxel_size);
  int32_t vy = (int32_t)floorf((pos.y - brick_oy) * world->inv_voxel_size);
  int32_t vz = (int32_t)floorf((pos.z - brick_oz) * world->inv_voxel_size);

  /* Clamp to valid range */
  vx = vx < 0 ? 0 : (vx > BRICK_MASK ? BRICK_MASK : vx);
  vy = vy < 0 ? 0 : (vy > BRICK_MASK ? BRICK_MASK : vy);
  vz = vz < 0 ? 0 : (vz > BRICK_MASK ? BRICK_MASK : vz);

  uint32_t voxel_idx = (uint32_t)vx + ((uint32_t)vy << BRICK_SHIFT) +
                       ((uint32_t)vz << (BRICK_SHIFT * 2));

  /* Get material data (demand-paged) */
  uint32_t page_idx = (uint32_t)atlas_idx / ATLAS_PAGE_BRICKS;
  uint32_t brick_in_page = (uint32_t)atlas_idx % ATLAS_PAGE_BRICKS;
  const uint8_t *material =
      world->material_pages[page_idx] + (size_t)brick_in_page * BRICK_VOXELS;

  return material[voxel_idx];
}

Vec3 world_sdf_gradient(const WorldBrickMap *world, Vec3 pos) {
  float eps = RAYMARCH_NORMAL_EPSILON;

  float dx = world_sdf_query(world, VEC3(pos.x + eps, pos.y, pos.z)) -
             world_sdf_query(world, VEC3(pos.x - eps, pos.y, pos.z));
  float dy = world_sdf_query(world, VEC3(pos.x, pos.y + eps, pos.z)) -
             world_sdf_query(world, VEC3(pos.x, pos.y - eps, pos.z));
  float dz = world_sdf_query(world, VEC3(pos.x, pos.y, pos.z + eps)) -
             world_sdf_query(world, VEC3(pos.x, pos.y, pos.z - eps));

  float inv_eps2 = 0.5f / eps;
  return VEC3(dx * inv_eps2, dy * inv_eps2, dz * inv_eps2);
}

Vec3 world_sdf_normal(const WorldBrickMap *world, Vec3 pos) {
  Vec3 grad = world_sdf_gradient(world, pos);
  float len = vec3_length(grad);
  if (len < 1e-6f) {
    return VEC3(0.0f, 0.0f, 1.0f);
  }
  return vec3_scale(grad, 1.0f / len);
}

/* ============================================================================
 * Section 6: Voxel Modification Functions (SoA Layout)
 * ============================================================================
 */

void world_set_sdf(WorldBrickMap *world, Vec3 pos, float sdf_val) {
  if (world == NULL)
    return;

  int32_t bx, by, bz;
  world_pos_to_brick(world, pos, &bx, &by, &bz);

  if (!world_brick_valid(world, bx, by, bz)) {
    return;
  }

  int32_t atlas_idx = world_alloc_brick(world, bx, by, bz);
  if (atlas_idx == BRICK_EMPTY_INDEX) {
    return;
  }

  int32_t vx, vy, vz;
  world_pos_to_voxel(world, pos, bx, by, bz, &vx, &vy, &vz);

  uint32_t idx = voxel_linear_index(vx, vy, vz);

  /* Get SDF data (demand-paged) */
  uint32_t page_idx = (uint32_t)atlas_idx / ATLAS_PAGE_BRICKS;
  uint32_t brick_in_page = (uint32_t)atlas_idx % ATLAS_PAGE_BRICKS;
  int8_t *sdf =
      world->sdf_pages[page_idx] + (size_t)brick_in_page * BRICK_VOXELS;
  sdf[idx] = sdf_quantize(sdf_val, world->inv_sdf_scale);
}

void world_set_material(WorldBrickMap *world, Vec3 pos, uint8_t mat) {
  if (world == NULL)
    return;

  int32_t bx, by, bz;
  world_pos_to_brick(world, pos, &bx, &by, &bz);

  if (!world_brick_valid(world, bx, by, bz)) {
    return;
  }

  int32_t atlas_idx = world_alloc_brick(world, bx, by, bz);
  if (atlas_idx == BRICK_EMPTY_INDEX) {
    return;
  }

  int32_t vx, vy, vz;
  world_pos_to_voxel(world, pos, bx, by, bz, &vx, &vy, &vz);

  uint32_t idx = voxel_linear_index(vx, vy, vz);

  /* Get material data (demand-paged) */
  uint32_t page_idx = (uint32_t)atlas_idx / ATLAS_PAGE_BRICKS;
  uint32_t brick_in_page = (uint32_t)atlas_idx % ATLAS_PAGE_BRICKS;
  uint8_t *material =
      world->material_pages[page_idx] + (size_t)brick_in_page * BRICK_VOXELS;
  material[idx] = mat;
}

void world_set_voxel(WorldBrickMap *world, Vec3 pos, float sdf_val,
                     uint8_t mat) {
  if (world == NULL)
    return;

  int32_t bx, by, bz;
  world_pos_to_brick(world, pos, &bx, &by, &bz);

  if (!world_brick_valid(world, bx, by, bz)) {
    return;
  }

  int32_t atlas_idx = world_alloc_brick(world, bx, by, bz);
  if (atlas_idx == BRICK_EMPTY_INDEX) {
    return;
  }

  int32_t vx, vy, vz;
  world_pos_to_voxel(world, pos, bx, by, bz, &vx, &vy, &vz);

  uint32_t idx = voxel_linear_index(vx, vy, vz);

  /* Get SDF and material data (demand-paged) */
  uint32_t page_idx = (uint32_t)atlas_idx / ATLAS_PAGE_BRICKS;
  uint32_t brick_in_page = (uint32_t)atlas_idx % ATLAS_PAGE_BRICKS;
  size_t brick_offset = (size_t)brick_in_page * BRICK_VOXELS;
  world->sdf_pages[page_idx][brick_offset + idx] =
      sdf_quantize(sdf_val, world->inv_sdf_scale);
  world->material_pages[page_idx][brick_offset + idx] = mat;
}

/* ============================================================================
 * Section 7: Primitive Generation Functions (SoA + Stride Optimized)
 * ============================================================================
 */

void world_set_box(WorldBrickMap *world, Vec3 center, Vec3 half_size,
                   uint8_t mat) {
  if (world == NULL)
    return;

  /* Precompute constants */
  const float brick_size = world->brick_size_world;
  const float voxel_size = world->voxel_size;
  const float inv_sdf_scale = world->inv_sdf_scale;
  const float sdf_scale = world->sdf_scale;
  /* Brick half-diagonal for conservative distance check */
  const float brick_half_diag = brick_size * 0.866f; /* sqrt(3)/2 */

  /* Compute affected region (expand by 1 brick for SDF falloff) */
  Vec3 min_pos = vec3_sub(vec3_sub(center, half_size),
                          VEC3(brick_size, brick_size, brick_size));
  Vec3 max_pos = vec3_add(vec3_add(center, half_size),
                          VEC3(brick_size, brick_size, brick_size));

  /* Clamp to world bounds */
  min_pos = vec3_max(min_pos, world->world_min);
  max_pos = vec3_min(max_pos, world->world_max);

  /* Get brick range */
  int32_t bx0, by0, bz0, bx1, by1, bz1;
  world_pos_to_brick(world, min_pos, &bx0, &by0, &bz0);
  world_pos_to_brick(world, max_pos, &bx1, &by1, &bz1);

  bx0 = max_i32(bx0, 0);
  by0 = max_i32(by0, 0);
  bz0 = max_i32(bz0, 0);
  bx1 = min_i32(bx1, (int32_t)world->grid_x - 1);
  by1 = min_i32(by1, (int32_t)world->grid_y - 1);
  bz1 = min_i32(bz1, (int32_t)world->grid_z - 1);

  /* Iterate over affected bricks */
  for (int32_t bz = bz0; bz <= bz1; bz++) {
    for (int32_t by = by0; by <= by1; by++) {
      for (int32_t bx = bx0; bx <= bx1; bx++) {
        /* Compute brick center for distance check */
        Vec3 brick_center =
            VEC3(world->world_min.x + ((float)bx + 0.5f) * brick_size,
                 world->world_min.y + ((float)by + 0.5f) * brick_size,
                 world->world_min.z + ((float)bz + 0.5f) * brick_size);

        /* Compute box SDF at brick center */
        float dist_to_surface = sdf_box(brick_center, center, half_size);

        /* Check if brick is entirely outside (beyond SDF scale + safety margin)
         */
        if (dist_to_surface > sdf_scale + brick_half_diag) {
          /* Brick is far outside - skip (already uniform outside by default) */
          continue;
        }

        /* Check if brick is entirely inside (beyond SDF scale + safety margin)
         */
        if (dist_to_surface < -(sdf_scale + brick_half_diag)) {
          /* Mark as uniform inside without allocating */
          uint32_t brick_idx = brick_linear_index(world, bx, by, bz);
          int32_t current = world->brick_indices[brick_idx];
          if (current == BRICK_EMPTY_INDEX) {
            world->brick_indices[brick_idx] = BRICK_UNIFORM_INSIDE;
            world->uniform_inside_count++;
          } else if (current == BRICK_UNIFORM_OUTSIDE) {
            /* Transition from uniform outside to uniform inside */
            world->uniform_outside_count--;
            world->brick_indices[brick_idx] = BRICK_UNIFORM_INSIDE;
            world->uniform_inside_count++;
          }
          /* If already UNIFORM_INSIDE or allocated, leave as is */
          continue;
        }

        /* Brick is near surface - needs full allocation */
        int32_t atlas_idx = world_alloc_brick(world, bx, by, bz);
        if (atlas_idx == BRICK_EMPTY_INDEX)
          continue;

        /* Get SoA pointers for this brick (demand-paged) */
        uint32_t page_idx = (uint32_t)atlas_idx / ATLAS_PAGE_BRICKS;
        uint32_t brick_in_page = (uint32_t)atlas_idx % ATLAS_PAGE_BRICKS;
        size_t brick_offset = (size_t)brick_in_page * BRICK_VOXELS;
        int8_t *sdf = world->sdf_pages[page_idx] + brick_offset;
        uint8_t *material = world->material_pages[page_idx] + brick_offset;

        /* Compute brick origin */
        float brick_ox = world->world_min.x + (float)bx * brick_size;
        float brick_oy = world->world_min.y + (float)by * brick_size;
        float brick_oz = world->world_min.z + (float)bz * brick_size;

        /* Iterate over voxels with precomputed strides */
        for (int32_t vz = 0; vz < BRICK_SIZE; vz++) {
          const uint32_t z_stride = (uint32_t)vz << (BRICK_SHIFT * 2);
          const float pz = brick_oz + ((float)vz + 0.5f) * voxel_size;

          for (int32_t vy = 0; vy < BRICK_SIZE; vy++) {
            const uint32_t yz_stride = z_stride + ((uint32_t)vy << BRICK_SHIFT);
            const float py = brick_oy + ((float)vy + 0.5f) * voxel_size;

            for (int32_t vx = 0; vx < BRICK_SIZE; vx++) {
              const uint32_t idx = yz_stride + (uint32_t)vx;
              const float px = brick_ox + ((float)vx + 0.5f) * voxel_size;

              float sdf_val = sdf_box(VEC3(px, py, pz), center, half_size);
              int8_t new_q = sdf_quantize(sdf_val, inv_sdf_scale);

              /* CSG union: min(existing, new) */
              if (new_q < sdf[idx]) {
                sdf[idx] = new_q;
              }

              /* Set material where inside or on surface band.
               * Extend by one voxel_size past the SDF zero-crossing so
               * that the raymarcher's floor-based material lookup always
               * finds the correct material at hit positions. */
              if (sdf_val < voxel_size && mat > 0) {
                material[idx] = mat;
              }
            }
          }
        }
      }
    }
  }
}

void world_set_sphere(WorldBrickMap *world, Vec3 center, float radius,
                      uint8_t mat) {
  if (world == NULL || radius <= 0.0f)
    return;

  /* Precompute constants */
  const float brick_size = world->brick_size_world;
  const float voxel_size = world->voxel_size;
  const float inv_sdf_scale = world->inv_sdf_scale;
  const float sdf_scale = world->sdf_scale;
  /* Brick half-diagonal for conservative distance check */
  const float brick_half_diag = brick_size * 0.866f; /* sqrt(3)/2 */

  /* Compute affected region */
  Vec3 min_pos = vec3_sub(center, VEC3(radius + brick_size, radius + brick_size,
                                       radius + brick_size));
  Vec3 max_pos = vec3_add(center, VEC3(radius + brick_size, radius + brick_size,
                                       radius + brick_size));

  min_pos = vec3_max(min_pos, world->world_min);
  max_pos = vec3_min(max_pos, world->world_max);

  int32_t bx0, by0, bz0, bx1, by1, bz1;
  world_pos_to_brick(world, min_pos, &bx0, &by0, &bz0);
  world_pos_to_brick(world, max_pos, &bx1, &by1, &bz1);

  bx0 = max_i32(bx0, 0);
  by0 = max_i32(by0, 0);
  bz0 = max_i32(bz0, 0);
  bx1 = min_i32(bx1, (int32_t)world->grid_x - 1);
  by1 = min_i32(by1, (int32_t)world->grid_y - 1);
  bz1 = min_i32(bz1, (int32_t)world->grid_z - 1);

  for (int32_t bz = bz0; bz <= bz1; bz++) {
    for (int32_t by = by0; by <= by1; by++) {
      for (int32_t bx = bx0; bx <= bx1; bx++) {
        /* Compute brick center for distance check */
        float brick_cx = world->world_min.x + ((float)bx + 0.5f) * brick_size;
        float brick_cy = world->world_min.y + ((float)by + 0.5f) * brick_size;
        float brick_cz = world->world_min.z + ((float)bz + 0.5f) * brick_size;

        /* Distance from brick center to sphere surface */
        float dx = brick_cx - center.x;
        float dy = brick_cy - center.y;
        float dz = brick_cz - center.z;
        float dist_to_center = sqrtf(dx * dx + dy * dy + dz * dz);
        float dist_to_surface = dist_to_center - radius;

        /* Check if brick is entirely outside (beyond SDF scale + safety margin)
         */
        if (dist_to_surface > sdf_scale + brick_half_diag) {
          /* Brick is far outside - skip (already uniform outside by default) */
          continue;
        }

        /* Check if brick is entirely inside (beyond SDF scale + safety margin)
         */
        if (dist_to_surface < -(sdf_scale + brick_half_diag)) {
          /* Mark as uniform inside without allocating */
          uint32_t brick_idx = brick_linear_index(world, bx, by, bz);
          int32_t current = world->brick_indices[brick_idx];
          if (current == BRICK_EMPTY_INDEX) {
            world->brick_indices[brick_idx] = BRICK_UNIFORM_INSIDE;
            world->uniform_inside_count++;
          } else if (current == BRICK_UNIFORM_OUTSIDE) {
            /* Transition from uniform outside to uniform inside */
            world->uniform_outside_count--;
            world->brick_indices[brick_idx] = BRICK_UNIFORM_INSIDE;
            world->uniform_inside_count++;
          }
          /* If already UNIFORM_INSIDE or allocated, leave as is */
          continue;
        }

        /* Brick is near surface - needs full allocation */
        int32_t atlas_idx = world_alloc_brick(world, bx, by, bz);
        if (atlas_idx == BRICK_EMPTY_INDEX)
          continue;

        /* Get SoA pointers for this brick (demand-paged) */
        uint32_t page_idx = (uint32_t)atlas_idx / ATLAS_PAGE_BRICKS;
        uint32_t brick_in_page = (uint32_t)atlas_idx % ATLAS_PAGE_BRICKS;
        size_t brick_offset = (size_t)brick_in_page * BRICK_VOXELS;
        int8_t *sdf = world->sdf_pages[page_idx] + brick_offset;
        uint8_t *material = world->material_pages[page_idx] + brick_offset;

        /* Compute brick origin */
        float brick_ox = world->world_min.x + (float)bx * brick_size;
        float brick_oy = world->world_min.y + (float)by * brick_size;
        float brick_oz = world->world_min.z + (float)bz * brick_size;

        /* Iterate over voxels with precomputed strides */
        for (int32_t vz = 0; vz < BRICK_SIZE; vz++) {
          const uint32_t z_stride = (uint32_t)vz << (BRICK_SHIFT * 2);
          const float pz = brick_oz + ((float)vz + 0.5f) * voxel_size;

          for (int32_t vy = 0; vy < BRICK_SIZE; vy++) {
            const uint32_t yz_stride = z_stride + ((uint32_t)vy << BRICK_SHIFT);
            const float py = brick_oy + ((float)vy + 0.5f) * voxel_size;

            for (int32_t vx = 0; vx < BRICK_SIZE; vx++) {
              const uint32_t idx = yz_stride + (uint32_t)vx;
              const float px = brick_ox + ((float)vx + 0.5f) * voxel_size;

              float sdf_val = sdf_sphere(VEC3(px, py, pz), center, radius);
              int8_t new_q = sdf_quantize(sdf_val, inv_sdf_scale);

              if (new_q < sdf[idx]) {
                sdf[idx] = new_q;
              }

              if (sdf_val < voxel_size && mat > 0) {
                material[idx] = mat;
              }
            }
          }
        }
      }
    }
  }
}

void world_set_cylinder(WorldBrickMap *world, Vec3 center, float radius,
                        float half_height, uint8_t mat) {
  if (world == NULL || radius <= 0.0f || half_height <= 0.0f)
    return;

  /* Precompute constants */
  const float brick_size = world->brick_size_world;
  const float voxel_size = world->voxel_size;
  const float inv_sdf_scale = world->inv_sdf_scale;

  Vec3 extent =
      VEC3(radius + brick_size, radius + brick_size, half_height + brick_size);
  Vec3 min_pos = vec3_sub(center, extent);
  Vec3 max_pos = vec3_add(center, extent);

  min_pos = vec3_max(min_pos, world->world_min);
  max_pos = vec3_min(max_pos, world->world_max);

  int32_t bx0, by0, bz0, bx1, by1, bz1;
  world_pos_to_brick(world, min_pos, &bx0, &by0, &bz0);
  world_pos_to_brick(world, max_pos, &bx1, &by1, &bz1);

  bx0 = max_i32(bx0, 0);
  by0 = max_i32(by0, 0);
  bz0 = max_i32(bz0, 0);
  bx1 = min_i32(bx1, (int32_t)world->grid_x - 1);
  by1 = min_i32(by1, (int32_t)world->grid_y - 1);
  bz1 = min_i32(bz1, (int32_t)world->grid_z - 1);

  for (int32_t bz = bz0; bz <= bz1; bz++) {
    for (int32_t by = by0; by <= by1; by++) {
      for (int32_t bx = bx0; bx <= bx1; bx++) {
        int32_t atlas_idx = world_alloc_brick(world, bx, by, bz);
        if (atlas_idx == BRICK_EMPTY_INDEX)
          continue;

        /* Get SoA pointers for this brick (demand-paged) */
        uint32_t page_idx = (uint32_t)atlas_idx / ATLAS_PAGE_BRICKS;
        uint32_t brick_in_page = (uint32_t)atlas_idx % ATLAS_PAGE_BRICKS;
        size_t brick_offset = (size_t)brick_in_page * BRICK_VOXELS;
        int8_t *sdf = world->sdf_pages[page_idx] + brick_offset;
        uint8_t *material = world->material_pages[page_idx] + brick_offset;

        float brick_ox = world->world_min.x + (float)bx * brick_size;
        float brick_oy = world->world_min.y + (float)by * brick_size;
        float brick_oz = world->world_min.z + (float)bz * brick_size;

        for (int32_t vz = 0; vz < BRICK_SIZE; vz++) {
          const uint32_t z_stride = (uint32_t)vz << (BRICK_SHIFT * 2);
          const float pz = brick_oz + ((float)vz + 0.5f) * voxel_size;

          for (int32_t vy = 0; vy < BRICK_SIZE; vy++) {
            const uint32_t yz_stride = z_stride + ((uint32_t)vy << BRICK_SHIFT);
            const float py = brick_oy + ((float)vy + 0.5f) * voxel_size;

            for (int32_t vx = 0; vx < BRICK_SIZE; vx++) {
              const uint32_t idx = yz_stride + (uint32_t)vx;
              const float px = brick_ox + ((float)vx + 0.5f) * voxel_size;

              float sdf_val =
                  sdf_cylinder(VEC3(px, py, pz), center, radius, half_height);
              int8_t new_q = sdf_quantize(sdf_val, inv_sdf_scale);

              if (new_q < sdf[idx]) {
                sdf[idx] = new_q;
              }

              if (sdf_val < voxel_size && mat > 0) {
                material[idx] = mat;
              }
            }
          }
        }
      }
    }
  }
}

/* ============================================================================
 * Section 8: Raymarching Functions
 * ============================================================================
 */

RayHit world_raymarch(const WorldBrickMap *world, Vec3 origin, Vec3 direction,
                      float max_distance) {
  RayHit hit = {0};
  hit.hit = false;
  hit.distance = max_distance;

  if (world == NULL)
    return hit;

  float t = 0.0f;

  for (int step = 0; step < RAYMARCH_MAX_STEPS; step++) {
    Vec3 pos = vec3_add(origin, vec3_scale(direction, t));

    /* Check bounds */
    if (!world_contains(world, pos) && t > 0.0f) {
      hit.distance = t;
      return hit;
    }

    float dist = world_sdf_query(world, pos);

    /* Hit detected */
    if (dist < RAYMARCH_HIT_DIST) {
      hit.hit = true;
      hit.position = pos;
      hit.normal = world_sdf_normal(world, pos);
      hit.distance = t;
      hit.material = world_material_query(world, pos);
      return hit;
    }

    /* Check max distance */
    if (t > max_distance) {
      hit.distance = max_distance;
      return hit;
    }

    /* Sphere trace step */
    t += (dist > RAYMARCH_EPSILON) ? dist : RAYMARCH_EPSILON;
  }

  hit.distance = t;
  return hit;
}

void world_raymarch_batch(const WorldBrickMap *world, const Vec3 *origins,
                          const Vec3 *directions, float max_distance,
                          RayHit *hits, uint32_t count) {
  if (world == NULL || origins == NULL || directions == NULL || hits == NULL ||
      count == 0) {
    return;
  }

  /* Process rays in parallel with SIMD where beneficial */
  /* For now, use scalar loop - SIMD optimization would require
   * careful handling of divergent ray termination */
  for (uint32_t i = 0; i < count; i++) {
    hits[i] = world_raymarch(world, origins[i], directions[i], max_distance);
  }
}

void world_raymarch_camera(const WorldBrickMap *world, Vec3 camera_pos,
                           Vec3 camera_forward, Vec3 camera_up, float fov_h,
                           float fov_v, uint32_t width, uint32_t height,
                           float max_distance, float *depth_buffer,
                           uint8_t *material_buffer) {
  if (world == NULL || depth_buffer == NULL || material_buffer == NULL) {
    return;
  }

  /* Compute camera basis vectors */
  Vec3 forward = vec3_normalize(camera_forward);
  Vec3 right = vec3_normalize(vec3_cross(forward, camera_up));
  Vec3 up = vec3_cross(right, forward);

  float tan_fov_h = tanf(fov_h * 0.5f);
  float tan_fov_v = tanf(fov_v * 0.5f);

  for (uint32_t y = 0; y < height; y++) {
    for (uint32_t x = 0; x < width; x++) {
      /* Compute ray direction */
      float u = (2.0f * ((float)x + 0.5f) / (float)width - 1.0f) * tan_fov_h;
      float v = (1.0f - 2.0f * ((float)y + 0.5f) / (float)height) * tan_fov_v;

      Vec3 dir = vec3_normalize(
          vec3_add(vec3_add(forward, vec3_scale(right, u)), vec3_scale(up, v)));

      RayHit hit = world_raymarch(world, camera_pos, dir, max_distance);

      uint32_t idx = y * width + x;
      depth_buffer[idx] = hit.hit ? hit.distance : max_distance;
      material_buffer[idx] = hit.hit ? hit.material : 0;
    }
  }
}

/* ============================================================================
 * Section 9: Batch SDF Operations
 * ============================================================================
 */

void world_sdf_query_batch(const WorldBrickMap *world, const Vec3 *positions,
                           float *sdfs, uint32_t count) {
  if (world == NULL || positions == NULL || sdfs == NULL || count == 0) {
    return;
  }

  /* Scalar loop for now - complex brick lookup makes SIMD difficult */
  for (uint32_t i = 0; i < count; i++) {
    sdfs[i] = world_sdf_query(world, positions[i]);
  }
}

void world_sdf_gradient_batch(const WorldBrickMap *world, const Vec3 *positions,
                              Vec3 *gradients, uint32_t count) {
  if (world == NULL || positions == NULL || gradients == NULL || count == 0) {
    return;
  }

  for (uint32_t i = 0; i < count; i++) {
    gradients[i] = world_sdf_gradient(world, positions[i]);
  }
}

/* ============================================================================
 * Section 10: Uniform Brick Detection Functions
 * ============================================================================
 */

/**
 * Threshold for uniform brick detection
 * A brick is uniform if all voxels have SDF > threshold (outside)
 * or all voxels have SDF < -threshold (inside)
 */
#define UNIFORM_THRESHOLD 100 /* ~78% of max quantized SDF */

int32_t world_detect_uniform_brick(const WorldBrickMap *world,
                                   int32_t atlas_idx) {
  if (world == NULL || atlas_idx < 0 ||
      (uint32_t)atlas_idx >= world->atlas_count) {
    return atlas_idx;
  }

  /* Get SDF data */
  uint32_t page_idx = (uint32_t)atlas_idx / ATLAS_PAGE_BRICKS;
  uint32_t brick_in_page = (uint32_t)atlas_idx % ATLAS_PAGE_BRICKS;
  if (page_idx >= world->page_count || world->sdf_pages[page_idx] == NULL) {
    return atlas_idx;
  }
  const int8_t *sdf =
      world->sdf_pages[page_idx] + (size_t)brick_in_page * BRICK_VOXELS;

  /* Check first voxel to determine expected sign */
  int8_t first = sdf[0];
  bool all_outside = (first > UNIFORM_THRESHOLD);
  bool all_inside = (first < -UNIFORM_THRESHOLD);

  if (!all_outside && !all_inside) {
    return atlas_idx; /* First voxel is in transition zone */
  }

  /* Check all remaining voxels */
  for (uint32_t i = 1; i < BRICK_VOXELS; i++) {
    int8_t val = sdf[i];
    if (all_outside && val <= UNIFORM_THRESHOLD) {
      return atlas_idx; /* Found non-uniform voxel */
    }
    if (all_inside && val >= -UNIFORM_THRESHOLD) {
      return atlas_idx; /* Found non-uniform voxel */
    }
  }

  /* Brick is uniform */
  return all_outside ? BRICK_UNIFORM_OUTSIDE : BRICK_UNIFORM_INSIDE;
}

uint32_t world_compact_uniform_bricks(WorldBrickMap *world) {
  if (world == NULL)
    return 0;

  uint32_t converted = 0;

  /* Scan all brick indices */
  for (uint32_t i = 0; i < world->grid_total; i++) {
    int32_t atlas_idx = world->brick_indices[i];

    /* Skip empty or already uniform bricks */
    if (atlas_idx < 0)
      continue;

    /* Check if this brick is uniform */
    int32_t result = world_detect_uniform_brick(world, atlas_idx);

    if (result != atlas_idx) {
      /* Brick is uniform - convert it */
      /* Add old atlas index to free list */
      if (world->free_count < world->max_bricks) {
        world->free_list[world->free_count++] = (uint32_t)atlas_idx;
      }

      /* Update brick index to uniform sentinel */
      world->brick_indices[i] = result;

      /* Update counters */
      if (result == BRICK_UNIFORM_OUTSIDE) {
        world->uniform_outside_count++;
      } else {
        world->uniform_inside_count++;
      }

      converted++;
    }
  }

  return converted;
}

void world_mark_brick_uniform_outside(WorldBrickMap *world, int32_t bx,
                                      int32_t by, int32_t bz) {
  if (!world_brick_valid(world, bx, by, bz)) {
    return;
  }

  uint32_t grid_idx = brick_linear_index(world, bx, by, bz);
  int32_t current_idx = world->brick_indices[grid_idx];

  /* Handle transition from different states */
  if (current_idx >= 0) {
    /* Was allocated - free it */
    if (world->free_count < world->max_bricks) {
      world->free_list[world->free_count++] = (uint32_t)current_idx;
    }
  } else if (current_idx == BRICK_UNIFORM_INSIDE) {
    world->uniform_inside_count--;
  } else if (current_idx == BRICK_UNIFORM_OUTSIDE) {
    return; /* Already uniform outside */
  }

  world->brick_indices[grid_idx] = BRICK_UNIFORM_OUTSIDE;
  world->uniform_outside_count++;
}

void world_mark_brick_uniform_inside(WorldBrickMap *world, int32_t bx,
                                     int32_t by, int32_t bz) {
  if (!world_brick_valid(world, bx, by, bz)) {
    return;
  }

  uint32_t grid_idx = brick_linear_index(world, bx, by, bz);
  int32_t current_idx = world->brick_indices[grid_idx];

  /* Handle transition from different states */
  if (current_idx >= 0) {
    /* Was allocated - free it */
    if (world->free_count < world->max_bricks) {
      world->free_list[world->free_count++] = (uint32_t)current_idx;
    }
  } else if (current_idx == BRICK_UNIFORM_OUTSIDE) {
    world->uniform_outside_count--;
  } else if (current_idx == BRICK_UNIFORM_INSIDE) {
    return; /* Already uniform inside */
  }

  world->brick_indices[grid_idx] = BRICK_UNIFORM_INSIDE;
  world->uniform_inside_count++;
}

/* ============================================================================
 * Page-Level Dirty Tracking for GPU Sync
 * ============================================================================ */

uint32_t world_get_dirty_pages(const WorldBrickMap *world, bool *dirty_out,
                                uint32_t max_pages) {
  if (world == NULL || dirty_out == NULL) return 0;

  uint32_t count = 0;
  uint32_t limit = max_pages < world->max_pages ? max_pages : world->max_pages;

  for (uint32_t i = 0; i < limit; i++) {
    dirty_out[i] = world->page_dirty[i];
    if (world->page_dirty[i]) count++;
  }

  /* Clear remaining entries */
  for (uint32_t i = limit; i < max_pages; i++) {
    dirty_out[i] = false;
  }

  return count;
}

void world_clear_dirty_pages(WorldBrickMap *world) {
  if (world == NULL) return;
  memset(world->page_dirty, 0, sizeof(world->page_dirty));
}

void world_mark_page_dirty(WorldBrickMap *world, uint32_t page_idx) {
  if (world == NULL || page_idx >= MAX_ATLAS_PAGES) return;
  world->page_dirty[page_idx] = true;
}

/* ============================================================================
 * Per-Voxel Feature Channel Implementation
 * ============================================================================ */

static uint32_t voxel_type_size(VoxelDataType type) {
  switch (type) {
  case VOXEL_TYPE_INT8:
    return 1;
  case VOXEL_TYPE_UINT8:
    return 1;
  case VOXEL_TYPE_FLOAT32:
    return 4;
  default:
    return 0;
  }
}

int32_t world_add_channel(WorldBrickMap *world, const char *name,
                          VoxelDataType type, uint32_t components) {
  if (world == NULL || name == NULL || components == 0) {
    return -1;
  }
  if (world->feature_channel_count >= MAX_VOXEL_CHANNELS) {
    return -1;
  }

  uint32_t elem_size = voxel_type_size(type);
  if (elem_size == 0) {
    return -1;
  }

  int32_t idx = (int32_t)world->feature_channel_count;
  VoxelChannel *ch = &world->feature_channels[idx];

  /* Copy name (truncate to 31 chars) */
  size_t name_len = strlen(name);
  if (name_len > 31)
    name_len = 31;
  memcpy(ch->name, name, name_len);
  ch->name[name_len] = '\0';

  ch->type = type;
  ch->components = components;
  ch->bytes_per_voxel = elem_size * components;
  ch->bytes_per_brick = ch->bytes_per_voxel * BRICK_VOXELS;
  ch->page_count = 0;

  /* Allocate page pointer array from arena */
  ch->pages =
      (void **)arena_alloc_aligned(world->arena,
                                   world->max_pages * sizeof(void *), 8);
  if (ch->pages == NULL) {
    return -1;
  }
  memset(ch->pages, 0, world->max_pages * sizeof(void *));

  /* Retroactively allocate pages for existing bricks */
  for (uint32_t p = 0; p < world->page_count; p++) {
    if (world->sdf_pages[p] != NULL) {
      size_t ch_page_size = (size_t)ATLAS_PAGE_BRICKS * ch->bytes_per_brick;
      ch->pages[p] = arena_alloc_aligned(world->arena, ch_page_size,
                                         BRICK_ALIGNMENT);
      if (ch->pages[p] == NULL) {
        return -1;
      }
      memset(ch->pages[p], 0, ch_page_size);
      ch->page_count = p + 1;
    }
  }

  world->feature_channel_count++;
  return idx;
}

int32_t world_find_channel(const WorldBrickMap *world, const char *name) {
  if (world == NULL || name == NULL) {
    return -1;
  }
  for (uint32_t i = 0; i < world->feature_channel_count; i++) {
    if (strcmp(world->feature_channels[i].name, name) == 0) {
      return (int32_t)i;
    }
  }
  return -1;
}

const VoxelChannel *world_get_channel(const WorldBrickMap *world,
                                      int32_t channel_idx) {
  if (world == NULL || channel_idx < 0 ||
      (uint32_t)channel_idx >= world->feature_channel_count) {
    return NULL;
  }
  return &world->feature_channels[channel_idx];
}

/**
 * Internal: resolve world position to page/brick/voxel offsets for a channel.
 * Returns false if out of bounds or unallocated.
 */
static bool channel_resolve(const WorldBrickMap *world, int32_t channel_idx,
                            Vec3 pos, uint32_t component,
                            const VoxelChannel **out_ch,
                            uint32_t *out_page, size_t *out_byte_offset) {
  if (world == NULL || channel_idx < 0 ||
      (uint32_t)channel_idx >= world->feature_channel_count) {
    return false;
  }
  const VoxelChannel *ch = &world->feature_channels[channel_idx];
  if (component >= ch->components) {
    return false;
  }

  Vec3 rel = vec3_sub(pos, world->world_min);
  int32_t bx = (int32_t)floorf(rel.x * world->inv_brick_size);
  int32_t by = (int32_t)floorf(rel.y * world->inv_brick_size);
  int32_t bz = (int32_t)floorf(rel.z * world->inv_brick_size);

  if (bx < 0 || bx >= (int32_t)world->grid_x || by < 0 ||
      by >= (int32_t)world->grid_y || bz < 0 ||
      bz >= (int32_t)world->grid_z) {
    return false;
  }

  uint32_t grid_idx = (uint32_t)bx + (uint32_t)by * world->stride_y +
                      (uint32_t)bz * world->stride_z;
  int32_t atlas_idx = world->brick_indices[grid_idx];
  if (atlas_idx < 0) {
    return false;
  }

  /* Voxel coordinates within brick */
  float brick_ox = world->world_min.x + (float)bx * world->brick_size_world;
  float brick_oy = world->world_min.y + (float)by * world->brick_size_world;
  float brick_oz = world->world_min.z + (float)bz * world->brick_size_world;

  int32_t vx = (int32_t)floorf((pos.x - brick_ox) * world->inv_voxel_size);
  int32_t vy = (int32_t)floorf((pos.y - brick_oy) * world->inv_voxel_size);
  int32_t vz = (int32_t)floorf((pos.z - brick_oz) * world->inv_voxel_size);
  vx = vx < 0 ? 0 : (vx > BRICK_MASK ? BRICK_MASK : vx);
  vy = vy < 0 ? 0 : (vy > BRICK_MASK ? BRICK_MASK : vy);
  vz = vz < 0 ? 0 : (vz > BRICK_MASK ? BRICK_MASK : vz);

  uint32_t voxel_idx = (uint32_t)vx + ((uint32_t)vy << BRICK_SHIFT) +
                       ((uint32_t)vz << (BRICK_SHIFT * 2));

  uint32_t page_idx = (uint32_t)atlas_idx / ATLAS_PAGE_BRICKS;
  uint32_t brick_in_page = (uint32_t)atlas_idx % ATLAS_PAGE_BRICKS;

  if (ch->pages == NULL || page_idx >= ch->page_count ||
      ch->pages[page_idx] == NULL) {
    return false;
  }

  *out_ch = ch;
  *out_page = page_idx;
  *out_byte_offset = (size_t)brick_in_page * ch->bytes_per_brick +
                     (size_t)voxel_idx * ch->bytes_per_voxel +
                     (size_t)component * voxel_type_size(ch->type);
  return true;
}

float world_channel_query_f32(const WorldBrickMap *world, int32_t channel_idx,
                              Vec3 pos, uint32_t component) {
  const VoxelChannel *ch;
  uint32_t page;
  size_t offset;
  if (!channel_resolve(world, channel_idx, pos, component, &ch, &page,
                       &offset)) {
    return 0.0f;
  }
  if (ch->type == VOXEL_TYPE_FLOAT32) {
    float val;
    memcpy(&val, (const uint8_t *)ch->pages[page] + offset, sizeof(float));
    return val;
  } else if (ch->type == VOXEL_TYPE_UINT8) {
    return (float)((const uint8_t *)ch->pages[page])[offset];
  } else {
    return (float)((const int8_t *)ch->pages[page])[offset];
  }
}

void world_channel_set_f32(WorldBrickMap *world, int32_t channel_idx, Vec3 pos,
                           uint32_t component, float value) {
  const VoxelChannel *ch;
  uint32_t page;
  size_t offset;
  if (!channel_resolve(world, channel_idx, pos, component, &ch, &page,
                       &offset)) {
    return;
  }
  if (ch->type == VOXEL_TYPE_FLOAT32) {
    memcpy((uint8_t *)ch->pages[page] + offset, &value, sizeof(float));
  } else if (ch->type == VOXEL_TYPE_UINT8) {
    ((uint8_t *)ch->pages[page])[offset] = (uint8_t)value;
  } else {
    ((int8_t *)ch->pages[page])[offset] = (int8_t)value;
  }
}

uint8_t world_channel_query_u8(const WorldBrickMap *world, int32_t channel_idx,
                               Vec3 pos, uint32_t component) {
  const VoxelChannel *ch;
  uint32_t page;
  size_t offset;
  if (!channel_resolve(world, channel_idx, pos, component, &ch, &page,
                       &offset)) {
    return 0;
  }
  return ((const uint8_t *)ch->pages[page])[offset];
}

void world_channel_set_u8(WorldBrickMap *world, int32_t channel_idx, Vec3 pos,
                          uint32_t component, uint8_t value) {
  const VoxelChannel *ch;
  uint32_t page;
  size_t offset;
  if (!channel_resolve(world, channel_idx, pos, component, &ch, &page,
                       &offset)) {
    return;
  }
  ((uint8_t *)ch->pages[page])[offset] = value;
}

void *world_channel_brick_data(const WorldBrickMap *world, int32_t channel_idx,
                               int32_t atlas_idx) {
  if (world == NULL || channel_idx < 0 ||
      (uint32_t)channel_idx >= world->feature_channel_count) {
    return NULL;
  }
  if (atlas_idx < 0 || (uint32_t)atlas_idx >= world->atlas_count) {
    return NULL;
  }
  const VoxelChannel *ch = &world->feature_channels[channel_idx];
  uint32_t page_idx = (uint32_t)atlas_idx / ATLAS_PAGE_BRICKS;
  uint32_t brick_in_page = (uint32_t)atlas_idx % ATLAS_PAGE_BRICKS;
  if (ch->pages == NULL || page_idx >= ch->page_count ||
      ch->pages[page_idx] == NULL) {
    return NULL;
  }
  return (uint8_t *)ch->pages[page_idx] +
         (size_t)brick_in_page * ch->bytes_per_brick;
}
