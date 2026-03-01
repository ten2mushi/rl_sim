/**
 * World Brick Map Module - Sparse SDF Storage with 8x8x8 Brick Atlas
 *
 * This module provides efficient sparse voxel SDF (Signed Distance Field)
 * storage for world representation and raymarching queries.
 *
 * Key Features:
 * 1. Two-level sparse structure with O(1) lookup (grid indices -> brick atlas)
 * 2. 8-bit distance quantization (75% memory reduction vs float32)
 * 3. SIMD-optimized batch raymarching for sensor generation
 * 4. Incremental regeneration with dirty brick tracking
 * 5. Clip map LOD for 10,000x memory reduction in large environments
 *
 * Dependencies:
 * - foundation (Vec3, Quat, Arena, SIMD utilities)
 */

#ifndef WORLD_BRICK_MAP_H
#define WORLD_BRICK_MAP_H

#include "foundation.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Section 1: Constants and Configuration
 * ============================================================================
 */

/* Brick dimensions (8x8x8 voxels per brick) */
#define BRICK_SIZE 8     /* Voxels per brick dimension */
#define BRICK_VOXELS 512 /* 8^3 total voxels per brick */
#define BRICK_SHIFT 3    /* log2(8) for bit shifts */
#define BRICK_MASK 7     /* 0x7 for local coordinate masking */

/* Special brick indices for memory optimization */
#define BRICK_EMPTY_INDEX (-1) /* Sentinel for empty brick slots */
#define BRICK_UNIFORM_OUTSIDE                                                  \
  (-2) /* All voxels far outside geometry (return +sdf_scale) */
#define BRICK_UNIFORM_INSIDE                                                   \
  (-3) /* All voxels deep inside geometry (return -sdf_scale) */

/* Demand-paged atlas configuration */
#define ATLAS_PAGE_BRICKS 64 /* Bricks per page (allocate in chunks) */
#define MAX_ATLAS_PAGES 512  /* Maximum pages (64 * 512 = 32K max bricks) */

/* Narrow-band SDF configuration */
#define NARROW_BAND_VOXELS 2 /* Voxels of margin around surfaces */

/* Raymarching parameters */
#define RAYMARCH_MAX_STEPS 128  /* Maximum sphere trace iterations */
#define RAYMARCH_EPSILON 0.001f /* Minimum step size */
#define RAYMARCH_HIT_DIST 0.01f /* Distance threshold for hit detection */

/* Memory alignment for bricks (cache line friendly) */
#define BRICK_ALIGNMENT 64

/* Clip map configuration */
#define CLIPMAP_LEVELS 4 /* Number of LOD levels */

/* ============================================================================
 * Section 2: Core Data Structures
 * ============================================================================
 */

/**
 * SoA Brick Storage (Structure-of-Arrays)
 *
 * For optimal cache utilization, brick data is stored in separate flat arrays:
 * - sdf_atlas[max_bricks * BRICK_VOXELS]: All SDF data contiguous
 * - material_atlas[max_bricks * BRICK_VOXELS]: All material data contiguous
 *
 * This layout provides:
 * - 100% cache utilization for SDF-only queries (raymarching hot path)
 * - Better SIMD loading patterns for batch operations
 * - Cache line efficiency when material data not needed (50% -> 100%)
 *
 * Voxel addressing within brick: index = x + (y << 3) + (z << 6)
 * Global atlas addressing: atlas_idx * BRICK_VOXELS + voxel_index
 */

/**
 * Vec3 batch structure for SoA batch operations
 * Stores positions as separate x/y/z arrays for SIMD efficiency
 */
typedef struct Vec3Batch {
  float *FOUNDATION_RESTRICT x; /* X coordinates array */
  float *FOUNDATION_RESTRICT y; /* Y coordinates array */
  float *FOUNDATION_RESTRICT z; /* Z coordinates array */
  uint32_t count;               /* Number of positions */
  uint32_t capacity;            /* Allocated capacity */
} Vec3Batch;

/**
 * Material metadata for voxel materials
 *
 * Stores material information for import/export and visualization.
 * Materials are indexed 0-255, with ID 0 reserved for default/air.
 */
typedef struct MaterialMetadata {
  char name[64];       /* Material name (e.g., "terrain_rock") */
  Vec3 diffuse_color;  /* Kd (RGB) diffuse color for visualization */
  uint8_t id;          /* Material ID (0-255, auto-assigned) */
  uint8_t _padding[3]; /* Alignment padding */
} MaterialMetadata;

/**
 * Per-voxel feature channel data types.
 *
 * The world brick map stores per-voxel data in separate SoA channels.
 * Built-in channels: SDF (int8) and material (uint8).
 * Users can add arbitrary channels for semantic features (color, class, etc.).
 */
typedef enum VoxelDataType {
  VOXEL_TYPE_INT8 = 0,    /* int8_t  (1 byte, e.g. quantized SDF) */
  VOXEL_TYPE_UINT8 = 1,   /* uint8_t (1 byte, e.g. material/class ID) */
  VOXEL_TYPE_FLOAT32 = 2, /* float   (4 bytes, e.g. color, temperature) */
} VoxelDataType;

/**
 * Per-voxel feature channel.
 *
 * Each channel stores one feature per voxel, using the same brick/page
 * indexing as SDF and material (demand-paged SoA). Multiple components
 * are supported for vector features (e.g. 3 floats for RGB color).
 *
 * Pages are allocated alongside SDF/material pages in world_alloc_brick().
 * Channel data persists for the lifetime of the WorldBrickMap.
 */
typedef struct VoxelChannel {
  char name[32];              /* Channel name (e.g. "color_r", "class") */
  VoxelDataType type;         /* Element data type */
  uint32_t components;        /* Elements per voxel (1=scalar, 3=RGB) */
  uint32_t bytes_per_voxel;   /* sizeof(type) * components */
  uint32_t bytes_per_brick;   /* bytes_per_voxel * BRICK_VOXELS */
  void **pages;               /* Demand-paged array [max_pages] */
  uint32_t page_count;        /* Currently allocated pages */
} VoxelChannel;

/** Maximum user-defined feature channels per world */
#define MAX_VOXEL_CHANNELS 16

/**
 * Ray hit result from raymarching
 */
typedef struct RayHit {
  Vec3 position;    /* World position of hit point */
  Vec3 normal;      /* Surface normal at hit (unit length) */
  float distance;   /* Distance from ray origin to hit */
  uint8_t material; /* Material ID at hit point */
  bool hit;         /* True if ray hit geometry */
} RayHit;

/**
 * World statistics for memory monitoring
 */
typedef struct WorldStats {
  uint32_t total_bricks;    /* Maximum brick capacity */
  uint32_t active_bricks;   /* Currently allocated bricks (in atlas) */
  uint32_t uniform_outside; /* Bricks marked as uniform outside */
  uint32_t uniform_inside;  /* Bricks marked as uniform inside */
  uint32_t free_list_count; /* Bricks in free list */
  uint32_t pages_allocated; /* Number of atlas pages allocated */
  size_t grid_memory;       /* Memory for brick index grid (bytes) */
  size_t atlas_memory;      /* Memory for brick atlas (bytes) */
  size_t total_memory;      /* Total memory usage (bytes) */
  float fill_ratio;         /* active_bricks / total_bricks */
} WorldStats;

/**
 * Main sparse world brick map structure (SoA layout with demand-paged atlas)
 *
 * Three-level indirection with memory optimization:
 * 1. brick_indices[grid_total]: Maps grid coords -> atlas index
 *    - BRICK_EMPTY_INDEX (-1): Never touched
 *    - BRICK_UNIFORM_OUTSIDE (-2): All voxels far outside (return +sdf_scale)
 *    - BRICK_UNIFORM_INSIDE (-3): All voxels deep inside (return -sdf_scale)
 *    - >=0: Index into atlas
 * 2. sdf_pages[page_count]: Array of pointers to SDF pages
 * 3. material_pages[page_count]: Array of pointers to material pages
 *
 * Demand-Paged Atlas Benefits:
 * - Memory allocated only when bricks are actually used
 * - Uniform bricks use no atlas memory (sentinel indices)
 * - 64-brick pages provide good allocation granularity
 */
typedef struct WorldBrickMap {
  /* Grid of brick indices (first level) */
  int32_t *brick_indices;          /* BRICK_* sentinel or atlas index >=0 */
  uint32_t grid_x, grid_y, grid_z; /* Grid dimensions in bricks */
  uint32_t grid_total; /* Total grid cells (grid_x * grid_y * grid_z) */

  /* Demand-paged SoA brick atlas (second level) */
  int8_t **sdf_pages;       /* Array of page pointers [page_count] */
  uint8_t **material_pages; /* Array of page pointers [page_count] */
  uint32_t max_pages;       /* Maximum number of pages (MAX_ATLAS_PAGES) */
  uint32_t page_count;      /* Currently allocated pages */
  uint32_t
      max_bricks; /* Maximum number of bricks (pages * ATLAS_PAGE_BRICKS) */
  uint32_t atlas_count; /* Current number of allocated bricks in atlas */

  /* Free list for brick reuse */
  uint32_t *free_list; /* Stack of freed brick indices */
  uint32_t free_count; /* Number of bricks in free list */

  /* Uniform brick tracking */
  uint32_t uniform_outside_count; /* Bricks marked BRICK_UNIFORM_OUTSIDE */
  uint32_t uniform_inside_count;  /* Bricks marked BRICK_UNIFORM_INSIDE */

  /* World bounds and voxel scale */
  Vec3 world_min;         /* Minimum world coordinates */
  Vec3 world_max;         /* Maximum world coordinates */
  Vec3 world_size;        /* world_max - world_min */
  float voxel_size;       /* Size of single voxel in world units */
  float inv_voxel_size;   /* 1.0 / voxel_size (precomputed) */
  float brick_size_world; /* Size of one brick in world units */
  float inv_brick_size;   /* 1.0 / brick_size_world (precomputed) */

  /* Narrow-band threshold for SDF (voxels from surface) */
  float narrow_band_dist; /* NARROW_BAND_VOXELS * voxel_size */

  /* SDF quantization scale */
  float sdf_scale;         /* SDF range: [-sdf_scale, +sdf_scale] */
  float inv_sdf_scale;     /* 1.0 / sdf_scale (precomputed) */
  float sdf_scale_div_127; /* sdf_scale / 127.0 (precomputed for dequantize) */

  /* Grid indexing helpers (precomputed) */
  uint32_t stride_y; /* grid_x */
  uint32_t stride_z; /* grid_x * grid_y */

  /* Material registry (for OBJ import/export) */
  MaterialMetadata
      *materials;          /* Array of registered materials [max_materials] */
  uint32_t material_count; /* Number of currently registered materials */
  uint32_t max_materials;  /* Maximum material capacity (default 256) */

  /* User-defined per-voxel feature channels (beyond built-in SDF/material) */
  VoxelChannel feature_channels[MAX_VOXEL_CHANNELS];
  uint32_t feature_channel_count; /* Number of active feature channels */

  /* Page-level dirty tracking for incremental GPU sync.
   * Set when a page's SDF or material data is modified.
   * Cleared by world_clear_dirty_pages() after GPU upload. */
  bool page_dirty[MAX_ATLAS_PAGES];

  /* Arena reference (no individual frees needed) */
  Arena *arena;
} WorldBrickMap;

/* ============================================================================
 * Section 3: Primitive Types for SDF Generation
 * ============================================================================
 */

/**
 * Primitive types for SDF generation
 */
typedef enum PrimitiveType {
  PRIM_BOX,
  PRIM_SPHERE,
  PRIM_CYLINDER,
  PRIM_COUNT
} PrimitiveType;

/**
 * CSG operation types for incremental edits
 */
typedef enum CSGOperation {
  CSG_UNION,    /* Add geometry: min(existing, new) */
  CSG_SUBTRACT, /* Remove geometry: max(existing, -new) */
  CSG_INTERSECT /* Keep intersection: max(existing, new) */
} CSGOperation;

/* ============================================================================
 * Section 4: Edit List for Incremental Regeneration
 * ============================================================================
 */

/**
 * Single edit entry in the ordered edit list
 */
typedef struct EditEntry {
  CSGOperation op;         /* CSG operation type */
  PrimitiveType primitive; /* Primitive shape */
  Vec3 center;             /* Center position in world coordinates */
  Vec3 params;             /* Shape-specific parameters:
                            * Box: half_size (x, y, z)
                            * Sphere: (radius, 0, 0)
                            * Cylinder: (radius, half_height, 0) */
  uint8_t material;        /* Material ID for solid regions */
} EditEntry;

/**
 * Edit list for non-destructive CSG modifications
 *
 * Maintains ordered list of edits that can be replayed to regenerate bricks.
 */
typedef struct EditList {
  EditEntry *entries; /* Array of edit entries */
  uint32_t count;     /* Current number of entries */
  uint32_t capacity;  /* Maximum entries */
} EditList;

/* ============================================================================
 * Section 5: Dirty Tracking for Incremental Regeneration
 * ============================================================================
 */

/**
 * Dirty brick tracker for incremental regeneration
 *
 * Tracks which bricks need regeneration when geometry changes.
 */
typedef struct DirtyTracker {
  uint32_t *dirty_indices; /* Array of dirty brick linear indices */
  uint32_t dirty_count;    /* Number of dirty bricks */
  uint32_t dirty_capacity; /* Maximum tracked dirty bricks */
  bool *dirty_flags;       /* Quick lookup: dirty_flags[brick_index] */
  uint32_t max_bricks;     /* Size of dirty_flags array */
} DirtyTracker;

/* ============================================================================
 * Section 6: Clip Map LOD Structures
 * ============================================================================
 */

/**
 * Single level in the clip map hierarchy
 */
typedef struct ClipMapLevel {
  WorldBrickMap *map; /* Brick map at this resolution */
  float voxel_size;   /* Voxel size at this level */
  float extent;       /* Half-size of level bounds */
  Vec3 center;        /* Current center position */
  Vec3 grid_origin;   /* Origin for toroidal wrapping */
} ClipMapLevel;

/**
 * Complete clip map world with multiple LOD levels
 *
 * Level 0: Highest resolution (base_voxel_size), smallest extent
 * Level N: 2^N coarser resolution, 2^N larger extent
 */
typedef struct ClipMapWorld {
  ClipMapLevel levels[CLIPMAP_LEVELS]; /* LOD levels */
  Vec3 focus;                          /* Current focus point */
  float base_voxel_size;               /* Level 0 voxel size */
  float base_extent;                   /* Level 0 half-extent */
  Arena *arena;                        /* Memory arena */
} ClipMapWorld;

/* ============================================================================
 * Section 7: World Brick Map Lifecycle Functions
 * ============================================================================
 * Create a new world brick map
 *
 * @param arena        Memory arena for allocation
 * @param world_min    Minimum world bounds
 * @param world_max    Maximum world bounds
 * @param voxel_size   Size of single voxel in world units
 * @param max_bricks   Maximum number of bricks to allocate
 * @param max_materials Maximum number of materials (default 256, use 0 for
 * default)
 * @return New world brick map or NULL on failure
 */
WorldBrickMap *world_create(Arena *arena, Vec3 world_min, Vec3 world_max,
                            float voxel_size, uint32_t max_bricks,
                            uint32_t max_materials);

/**
 * Destroy a world brick map (no-op with arena allocation)
 *
 * @param world World to destroy
 */
void world_destroy(WorldBrickMap *world);

/**
 * Clear all bricks from the world, resetting to empty state
 *
 * @param world World to clear
 */
void world_clear(WorldBrickMap *world);

/**
 * Get memory statistics for the world
 *
 * @param world World to query
 * @return Statistics structure
 */
WorldStats world_get_stats(const WorldBrickMap *world);

/**
 * Register a material with the world
 *
 * @param world          World brick map
 * @param name           Material name (max 63 chars, null-terminated)
 * @param diffuse_color  Diffuse color for visualization
 * @return Material ID (0-255) or 255 if table is full or name is NULL/empty
 */
uint8_t world_register_material(WorldBrickMap *world, const char *name,
                                Vec3 diffuse_color);

/**
 * Find material ID by name
 *
 * @param world World brick map
 * @param name  Material name to search for
 * @return Material ID or 0 (default material) if not found
 */
uint8_t world_find_material(const WorldBrickMap *world, const char *name);

/**
 * Get material metadata by ID
 *
 * @param world       World brick map
 * @param material_id Material ID to lookup
 * @return Pointer to material metadata or NULL if invalid ID
 */
const MaterialMetadata *world_get_material(const WorldBrickMap *world,
                                           uint8_t material_id);

/**
 * Calculate required memory for a world brick map
 *
 * @param grid_x      Grid dimensions in bricks
 * @param grid_y
 * @param grid_z
 * @param max_bricks  Maximum number of bricks
 * @return Required memory in bytes
 */
size_t world_memory_size(uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                         uint32_t max_bricks);

/* ============================================================================
 * Section 8: Coordinate Transformation Functions
 * ============================================================================
 */

/**
 * Convert world position to brick coordinates
 *
 * @param world World brick map
 * @param pos   World position
 * @param bx    Output: brick X coordinate
 * @param by    Output: brick Y coordinate
 * @param bz    Output: brick Z coordinate
 */
void world_pos_to_brick(const WorldBrickMap *world, Vec3 pos, int32_t *bx,
                        int32_t *by, int32_t *bz);

/**
 * Convert world position to voxel coordinates within a brick
 *
 * @param world World brick map
 * @param pos   World position
 * @param bx    Brick X coordinate
 * @param by    Brick Y coordinate
 * @param bz    Brick Z coordinate
 * @param vx    Output: local voxel X [0-7]
 * @param vy    Output: local voxel Y [0-7]
 * @param vz    Output: local voxel Z [0-7]
 */
void world_pos_to_voxel(const WorldBrickMap *world, Vec3 pos, int32_t bx,
                        int32_t by, int32_t bz, int32_t *vx, int32_t *vy,
                        int32_t *vz);

/**
 * Check if a position is within world bounds
 *
 * @param world World brick map
 * @param pos   Position to check
 * @return True if position is within bounds
 */
bool world_contains(const WorldBrickMap *world, Vec3 pos);

/**
 * Check if brick coordinates are valid
 *
 * @param world World brick map
 * @param bx    Brick X coordinate
 * @param by    Brick Y coordinate
 * @param bz    Brick Z coordinate
 * @return True if brick coordinates are valid
 */
bool world_brick_valid(const WorldBrickMap *world, int32_t bx, int32_t by,
                       int32_t bz);

/* ============================================================================
 * Section 9: Brick Management Functions (SoA Access)
 * ============================================================================
 */

/**
 * Get the atlas index for a brick at specified coordinates
 *
 * @param world World brick map
 * @param bx    Brick X coordinate
 * @param by    Brick Y coordinate
 * @param bz    Brick Z coordinate
 * @return Atlas index or BRICK_EMPTY_INDEX if empty/invalid
 */
int32_t world_get_brick_index(const WorldBrickMap *world, int32_t bx,
                              int32_t by, int32_t bz);

/**
 * Allocate a brick at the specified coordinates
 *
 * Returns existing atlas index if already allocated, allocates new one
 * otherwise. SDF data is initialized to +127 (far outside), material to 0
 * (air).
 *
 * @param world World brick map
 * @param bx    Brick X coordinate
 * @param by    Brick Y coordinate
 * @param bz    Brick Z coordinate
 * @return Atlas index or BRICK_EMPTY_INDEX if allocation failed
 */
int32_t world_alloc_brick(WorldBrickMap *world, int32_t bx, int32_t by,
                          int32_t bz);

/**
 * Free a brick at the specified coordinates
 *
 * @param world World brick map
 * @param bx    Brick X coordinate
 * @param by    Brick Y coordinate
 * @param bz    Brick Z coordinate
 */
void world_free_brick(WorldBrickMap *world, int32_t bx, int32_t by, int32_t bz);

/**
 * Check if a brick index represents a uniform brick (no atlas storage)
 */
FOUNDATION_INLINE bool brick_is_uniform(int32_t brick_idx) {
  return brick_idx == BRICK_UNIFORM_OUTSIDE ||
         brick_idx == BRICK_UNIFORM_INSIDE;
}

/**
 * Get pointer to SDF data for a brick by atlas index (demand-paged)
 *
 * @param world      World brick map
 * @param atlas_idx  Atlas index (must be >= 0, not a sentinel)
 * @return Pointer to 512 SDF bytes or NULL if invalid
 */
FOUNDATION_INLINE int8_t *world_brick_sdf(WorldBrickMap *world,
                                          int32_t atlas_idx) {
  if (atlas_idx < 0 || (uint32_t)atlas_idx >= world->atlas_count)
    return NULL;
  uint32_t page = (uint32_t)atlas_idx / ATLAS_PAGE_BRICKS;
  uint32_t offset = (uint32_t)atlas_idx % ATLAS_PAGE_BRICKS;
  if (page >= world->page_count || world->sdf_pages[page] == NULL)
    return NULL;
  return world->sdf_pages[page] + (offset * BRICK_VOXELS);
}

/**
 * Get pointer to SDF data (const version)
 */
FOUNDATION_INLINE const int8_t *
world_brick_sdf_const(const WorldBrickMap *world, int32_t atlas_idx) {
  return world_brick_sdf((WorldBrickMap *)world, atlas_idx);
}

/**
 * Get pointer to material data for a brick by atlas index (demand-paged)
 *
 * @param world      World brick map
 * @param atlas_idx  Atlas index (must be >= 0, not a sentinel)
 * @return Pointer to 512 material bytes or NULL if invalid
 */
FOUNDATION_INLINE uint8_t *world_brick_material(WorldBrickMap *world,
                                                int32_t atlas_idx) {
  if (atlas_idx < 0 || (uint32_t)atlas_idx >= world->atlas_count)
    return NULL;
  uint32_t page = (uint32_t)atlas_idx / ATLAS_PAGE_BRICKS;
  uint32_t offset = (uint32_t)atlas_idx % ATLAS_PAGE_BRICKS;
  if (page >= world->page_count || world->material_pages[page] == NULL)
    return NULL;
  return world->material_pages[page] + (offset * BRICK_VOXELS);
}

/**
 * Get pointer to material data (const version)
 */
FOUNDATION_INLINE const uint8_t *
world_brick_material_const(const WorldBrickMap *world, int32_t atlas_idx) {
  return world_brick_material((WorldBrickMap *)world, atlas_idx);
}

/* ============================================================================
 * Section 10: SDF Query Functions
 * ============================================================================
 */

/**
 * Query the SDF value at a world position (with trilinear interpolation)
 *
 * @param world World brick map
 * @param pos   World position
 * @return Interpolated SDF value (positive outside, negative inside)
 */
float world_sdf_query(const WorldBrickMap *world, Vec3 pos);

/**
 * Query SDF without interpolation (nearest neighbor)
 *
 * @param world World brick map
 * @param pos   World position
 * @return Nearest voxel SDF value
 */
float world_sdf_query_nearest(const WorldBrickMap *world, Vec3 pos);

/**
 * Query the material at a world position
 *
 * @param world World brick map
 * @param pos   World position
 * @return Material ID (0 = air)
 */
uint8_t world_material_query(const WorldBrickMap *world, Vec3 pos);

/**
 * Compute SDF gradient (surface normal direction) at a position
 *
 * Uses central differences for numerical gradient estimation.
 *
 * @param world World brick map
 * @param pos   World position
 * @return Gradient vector (not normalized)
 */
Vec3 world_sdf_gradient(const WorldBrickMap *world, Vec3 pos);

/**
 * Compute unit normal at a position
 *
 * @param world World brick map
 * @param pos   World position
 * @return Unit normal vector
 */
Vec3 world_sdf_normal(const WorldBrickMap *world, Vec3 pos);

/* ============================================================================
 * Section 11: Voxel Modification Functions
 * ============================================================================
 */

/**
 * Set the SDF value at a world position
 *
 * Allocates brick if necessary.
 *
 * @param world    World brick map
 * @param pos      World position
 * @param sdf      SDF value to set
 */
void world_set_sdf(WorldBrickMap *world, Vec3 pos, float sdf);

/**
 * Set the material at a world position
 *
 * Allocates brick if necessary.
 *
 * @param world    World brick map
 * @param pos      World position
 * @param material Material ID to set
 */
void world_set_material(WorldBrickMap *world, Vec3 pos, uint8_t material);

/**
 * Set both SDF and material at a world position
 *
 * @param world    World brick map
 * @param pos      World position
 * @param sdf      SDF value to set
 * @param material Material ID to set
 */
void world_set_voxel(WorldBrickMap *world, Vec3 pos, float sdf,
                     uint8_t material);

/* ============================================================================
 * Section 11b: Per-Voxel Feature Channels
 *
 * User-defined per-voxel feature channels provide extensible semantic storage
 * beyond the built-in SDF and material channels. Each channel stores one
 * feature per voxel using the same brick/page indexing as SDF/material.
 *
 * Usage:
 *   int32_t ch = world_add_channel(world, "color_r", VOXEL_TYPE_FLOAT32, 1);
 *   world_channel_set_f32(world, ch, pos, 0, 0.8f);
 *   float r = world_channel_query_f32(world, ch, pos, 0);
 * ============================================================================
 */

/**
 * Add a per-voxel feature channel.
 *
 * Creates storage for a new per-voxel feature. Pages are allocated on demand
 * alongside SDF/material pages when bricks are created. Existing bricks get
 * retroactive page allocation with zero-initialization.
 *
 * @param world      World brick map
 * @param name       Channel name (max 31 chars, e.g. "color_r", "class")
 * @param type       Data type per element
 * @param components Number of elements per voxel (1=scalar, 3=RGB vector)
 * @return Channel index (>=0) on success, -1 on failure
 */
int32_t world_add_channel(WorldBrickMap *world, const char *name,
                          VoxelDataType type, uint32_t components);

/**
 * Find a feature channel by name.
 *
 * @return Channel index (>=0) if found, -1 if not found
 */
int32_t world_find_channel(const WorldBrickMap *world, const char *name);

/**
 * Get a feature channel descriptor.
 *
 * @return Pointer to channel, or NULL if index invalid
 */
const VoxelChannel *world_get_channel(const WorldBrickMap *world,
                                      int32_t channel_idx);

/**
 * Query a float32 feature channel value at a world position.
 *
 * @param world       World brick map
 * @param channel_idx Channel index from world_add_channel()
 * @param pos         World position
 * @param component   Component index (0 for scalar, 0-2 for RGB)
 * @return Feature value, or 0.0f if out of bounds / unallocated
 */
float world_channel_query_f32(const WorldBrickMap *world, int32_t channel_idx,
                              Vec3 pos, uint32_t component);

/**
 * Set a float32 feature channel value at a world position.
 *
 * Allocates brick and channel page if necessary.
 *
 * @param world       World brick map
 * @param channel_idx Channel index from world_add_channel()
 * @param pos         World position
 * @param component   Component index
 * @param value       Value to write
 */
void world_channel_set_f32(WorldBrickMap *world, int32_t channel_idx, Vec3 pos,
                           uint32_t component, float value);

/**
 * Query a uint8 feature channel value at a world position.
 */
uint8_t world_channel_query_u8(const WorldBrickMap *world, int32_t channel_idx,
                               Vec3 pos, uint32_t component);

/**
 * Set a uint8 feature channel value at a world position.
 */
void world_channel_set_u8(WorldBrickMap *world, int32_t channel_idx, Vec3 pos,
                          uint32_t component, uint8_t value);

/**
 * Get raw pointer to a channel's brick data.
 *
 * Returns a pointer to the contiguous bytes_per_brick block for the given
 * brick atlas index. Used for bulk operations and GPU upload.
 *
 * @return Pointer to brick data, or NULL if unallocated
 */
void *world_channel_brick_data(const WorldBrickMap *world, int32_t channel_idx,
                               int32_t atlas_idx);

/* ============================================================================
 * Section 12: Primitive Generation Functions
 * ============================================================================
 */

/**
 * Add a box to the world (CSG union)
 *
 * @param world     World brick map
 * @param center    Box center position
 * @param half_size Half-size in each dimension
 * @param material  Material ID for solid voxels
 */
void world_set_box(WorldBrickMap *world, Vec3 center, Vec3 half_size,
                   uint8_t material);

/**
 * Add a sphere to the world (CSG union)
 *
 * @param world    World brick map
 * @param center   Sphere center position
 * @param radius   Sphere radius
 * @param material Material ID for solid voxels
 */
void world_set_sphere(WorldBrickMap *world, Vec3 center, float radius,
                      uint8_t material);

/**
 * Add a cylinder to the world (CSG union)
 *
 * Cylinder is aligned along the Z axis.
 *
 * @param world       World brick map
 * @param center      Cylinder center position
 * @param radius      Cylinder radius
 * @param half_height Half-height along Z axis
 * @param material    Material ID for solid voxels
 */
void world_set_cylinder(WorldBrickMap *world, Vec3 center, float radius,
                        float half_height, uint8_t material);

/* ============================================================================
 * Section 13: Raymarching Functions
 * ============================================================================
 */

/**
 * Raymarch a single ray through the world
 *
 * @param world        World brick map
 * @param origin       Ray origin
 * @param direction    Ray direction (must be normalized)
 * @param max_distance Maximum ray travel distance
 * @return Hit result
 */
RayHit world_raymarch(const WorldBrickMap *world, Vec3 origin, Vec3 direction,
                      float max_distance);

/**
 * Batch raymarch multiple rays
 *
 * @param world        World brick map
 * @param origins      Array of ray origins
 * @param directions   Array of ray directions (must be normalized)
 * @param max_distance Maximum ray travel distance
 * @param hits         Output: array of hit results
 * @param count        Number of rays
 */
void world_raymarch_batch(const WorldBrickMap *world, const Vec3 *origins,
                          const Vec3 *directions, float max_distance,
                          RayHit *hits, uint32_t count);

/**
 * Raymarch from camera viewpoint for depth/material buffers
 *
 * @param world          World brick map
 * @param camera_pos     Camera position
 * @param camera_forward Camera forward direction
 * @param camera_up      Camera up direction
 * @param fov_h          Horizontal field of view (radians)
 * @param fov_v          Vertical field of view (radians)
 * @param width          Output buffer width
 * @param height         Output buffer height
 * @param max_distance   Maximum ray distance
 * @param depth_buffer   Output: depth values (width * height floats)
 * @param material_buffer Output: material IDs (width * height uint8s)
 */
void world_raymarch_camera(const WorldBrickMap *world, Vec3 camera_pos,
                           Vec3 camera_forward, Vec3 camera_up, float fov_h,
                           float fov_v, uint32_t width, uint32_t height,
                           float max_distance, float *depth_buffer,
                           uint8_t *material_buffer);

/* ============================================================================
 * Section 14: Batch SDF Operations (SIMD-optimized)
 * ============================================================================
 */

/**
 * Batch SDF query at multiple positions
 *
 * @param world     World brick map
 * @param positions Array of world positions
 * @param sdfs      Output: array of SDF values
 * @param count     Number of positions
 */
void world_sdf_query_batch(const WorldBrickMap *world, const Vec3 *positions,
                           float *sdfs, uint32_t count);

/**
 * Batch SDF gradient calculation
 *
 * @param world     World brick map
 * @param positions Array of world positions
 * @param gradients Output: array of gradient vectors
 * @param count     Number of positions
 */
void world_sdf_gradient_batch(const WorldBrickMap *world, const Vec3 *positions,
                              Vec3 *gradients, uint32_t count);

/* ============================================================================
 * Section 15: Edit List Functions (Incremental Regeneration)
 * ============================================================================
 */

/**
 * Create a new edit list
 *
 * @param arena    Memory arena
 * @param capacity Maximum number of edit entries
 * @return New edit list or NULL on failure
 */
EditList *edit_list_create(Arena *arena, uint32_t capacity);

/**
 * Clear all entries from the edit list
 *
 * @param list Edit list to clear
 */
void edit_list_clear(EditList *list);

/**
 * Add an edit entry to the list
 *
 * @param list      Edit list
 * @param op        CSG operation
 * @param primitive Primitive type
 * @param center    Primitive center
 * @param params    Primitive parameters
 * @param material  Material ID
 * @return True if added successfully, false if list full
 */
bool edit_list_add(EditList *list, CSGOperation op, PrimitiveType primitive,
                   Vec3 center, Vec3 params, uint8_t material);

/**
 * Get number of entries in the edit list
 *
 * @param list Edit list
 * @return Number of entries
 */
uint32_t edit_list_count(const EditList *list);

/* ============================================================================
 * Section 16: Dirty Tracking Functions
 * ============================================================================
 */

/**
 * Create a new dirty tracker
 *
 * @param arena      Memory arena
 * @param max_bricks Maximum number of bricks to track
 * @return New dirty tracker or NULL on failure
 */
DirtyTracker *dirty_tracker_create(Arena *arena, uint32_t max_bricks);

/**
 * Clear all dirty flags
 *
 * @param tracker Dirty tracker to clear
 */
void dirty_tracker_clear(DirtyTracker *tracker);

/**
 * Mark a single brick as dirty
 *
 * @param tracker     Dirty tracker
 * @param brick_index Linear index of brick in grid
 */
void dirty_tracker_mark_brick(DirtyTracker *tracker, uint32_t brick_index);

/**
 * Mark all bricks in a world-space region as dirty
 *
 * @param tracker  Dirty tracker
 * @param world    World brick map
 * @param min_pos  Minimum corner of region
 * @param max_pos  Maximum corner of region
 */
void dirty_tracker_mark_region(DirtyTracker *tracker,
                               const WorldBrickMap *world, Vec3 min_pos,
                               Vec3 max_pos);

/**
 * Check if a brick is marked dirty
 *
 * @param tracker     Dirty tracker
 * @param brick_index Linear index of brick in grid
 * @return True if dirty
 */
bool dirty_tracker_is_dirty(const DirtyTracker *tracker, uint32_t brick_index);

/**
 * Get number of dirty bricks
 *
 * @param tracker Dirty tracker
 * @return Number of dirty bricks
 */
uint32_t dirty_tracker_count(const DirtyTracker *tracker);

/* ============================================================================
 * Section 17: Incremental Regeneration Functions
 * ============================================================================
 */

/**
 * Mark bricks affected by an edit region as dirty
 *
 * @param world    World brick map
 * @param tracker  Dirty tracker
 * @param edit_min Minimum corner of edit region
 * @param edit_max Maximum corner of edit region
 */
void world_mark_dirty_bricks(WorldBrickMap *world, DirtyTracker *tracker,
                             Vec3 edit_min, Vec3 edit_max);

/**
 * Regenerate all dirty bricks by replaying the edit list
 *
 * @param world   World brick map
 * @param tracker Dirty tracker
 * @param edits   Edit list to replay
 */
void world_regenerate_dirty(WorldBrickMap *world, DirtyTracker *tracker,
                            const EditList *edits);

/**
 * Regenerate a single brick by replaying the edit list
 *
 * @param world       World brick map
 * @param brick_index Linear index of brick in grid
 * @param edits       Edit list to replay
 */
void world_regenerate_brick(WorldBrickMap *world, uint32_t brick_index,
                            const EditList *edits);

/* ============================================================================
 * Section 17b: Uniform Brick Detection Functions
 * ============================================================================
 */

/**
 * Check if a brick is uniform (all voxels have same SDF sign with magnitude >
 * threshold)
 *
 * @param world      World brick map
 * @param atlas_idx  Atlas index of brick to check
 * @return BRICK_UNIFORM_OUTSIDE, BRICK_UNIFORM_INSIDE, or atlas_idx if not
 * uniform
 */
int32_t world_detect_uniform_brick(const WorldBrickMap *world,
                                   int32_t atlas_idx);

/**
 * Scan all allocated bricks and convert uniform ones to sentinel indices
 * This releases atlas memory for bricks that don't need per-voxel storage.
 *
 * @param world World brick map
 * @return Number of bricks converted to uniform
 */
uint32_t world_compact_uniform_bricks(WorldBrickMap *world);

/**
 * Mark a brick as uniform outside (all voxels far outside geometry)
 * Frees the brick's atlas storage.
 *
 * @param world       World brick map
 * @param bx, by, bz  Brick coordinates
 */
void world_mark_brick_uniform_outside(WorldBrickMap *world, int32_t bx,
                                      int32_t by, int32_t bz);

/**
 * Mark a brick as uniform inside (all voxels deep inside geometry)
 * Frees the brick's atlas storage.
 *
 * @param world       World brick map
 * @param bx, by, bz  Brick coordinates
 */
void world_mark_brick_uniform_inside(WorldBrickMap *world, int32_t bx,
                                     int32_t by, int32_t bz);

/* ============================================================================
 * Section 17c: Page-Level Dirty Tracking for GPU Sync
 * ============================================================================
 */

/**
 * Get which atlas pages have been modified since last clear.
 *
 * @param world      World brick map
 * @param dirty_out  Output: bool array [max_pages]. True = page was modified.
 * @param max_pages  Size of dirty_out array
 * @return Number of dirty pages
 */
uint32_t world_get_dirty_pages(const WorldBrickMap *world, bool *dirty_out,
                                uint32_t max_pages);

/**
 * Clear all page dirty flags (call after GPU sync completes).
 *
 * @param world World brick map
 */
void world_clear_dirty_pages(WorldBrickMap *world);

/**
 * Mark a specific page as dirty (for external writers like GPU readback).
 *
 * @param world    World brick map
 * @param page_idx Page index to mark dirty
 */
void world_mark_page_dirty(WorldBrickMap *world, uint32_t page_idx);

/* ============================================================================
 * Section 18: Clip Map LOD Functions
 * ============================================================================
 */

/**
 * Create a clip map world with multiple LOD levels
 *
 * @param arena            Memory arena
 * @param base_voxel_size  Voxel size for level 0 (highest detail)
 * @param base_extent      Half-extent for level 0
 * @param bricks_per_level Maximum bricks per level
 * @return New clip map world or NULL on failure
 */
ClipMapWorld *clipmap_create(Arena *arena, float base_voxel_size,
                             float base_extent, uint32_t bricks_per_level);

/**
 * Destroy a clip map world (no-op with arena)
 *
 * @param clipmap Clip map to destroy
 */
void clipmap_destroy(ClipMapWorld *clipmap);

/**
 * Update the focus point for the clip map
 *
 * Triggers toroidal grid updates for levels that have moved.
 *
 * @param clipmap   Clip map world
 * @param new_focus New focus position (camera/drone position)
 */
void clipmap_update_focus(ClipMapWorld *clipmap, Vec3 new_focus);

/**
 * Query SDF with automatic LOD selection
 *
 * @param clipmap Clip map world
 * @param pos     World position
 * @return SDF value from appropriate LOD level
 */
float clipmap_sdf_query(const ClipMapWorld *clipmap, Vec3 pos);

/**
 * Raymarch through clip map with LOD transitions
 *
 * @param clipmap      Clip map world
 * @param origin       Ray origin
 * @param direction    Ray direction (normalized)
 * @param max_distance Maximum ray distance
 * @return Hit result
 */
RayHit clipmap_raymarch(const ClipMapWorld *clipmap, Vec3 origin,
                        Vec3 direction, float max_distance);

/**
 * Select the appropriate LOD level for a position
 *
 * @param clipmap Clip map world
 * @param pos     World position
 * @return Level index (0 = highest detail)
 */
int clipmap_select_level(const ClipMapWorld *clipmap, Vec3 pos);

/**
 * Add a sphere to the clip map (updates all relevant levels)
 *
 * @param clipmap  Clip map world
 * @param center   Sphere center
 * @param radius   Sphere radius
 * @param material Material ID
 */
void clipmap_set_sphere(ClipMapWorld *clipmap, Vec3 center, float radius,
                        uint8_t material);

/**
 * Add a box to the clip map (updates all relevant levels)
 *
 * @param clipmap   Clip map world
 * @param center    Box center
 * @param half_size Box half-size
 * @param material  Material ID
 */
void clipmap_set_box(ClipMapWorld *clipmap, Vec3 center, Vec3 half_size,
                     uint8_t material);

/* ============================================================================
 * Section 19: SDF Quantization Helpers (Inline)
 * ============================================================================
 */

/**
 * Quantize a float SDF value to int8
 *
 * Maps [-sdf_scale, +sdf_scale] to [-127, +127]
 *
 * C truncation maps (-sdf_scale/127, +sdf_scale/127) to int8=0 (dequant 0.0).
 * A post-voxelization sweep eliminates phantom zeros (isolated int8=0 voxels
 * with no negative neighbor) to prevent false raymarcher hits.
 */
FOUNDATION_INLINE int8_t sdf_quantize(float sdf, float inv_sdf_scale) {
  float normalized = clampf(sdf * inv_sdf_scale, -1.0f, 1.0f);
  return (int8_t)(normalized * 127.0f);
}

/**
 * Dequantize an int8 SDF value to float
 *
 * Maps [-127, +127] to [-sdf_scale, +sdf_scale]
 */
FOUNDATION_INLINE float sdf_dequantize(int8_t q, float sdf_scale) {
  return ((float)q / 127.0f) * sdf_scale;
}

/* ============================================================================
 * Section 20: Voxel Index Helpers (Inline)
 * ============================================================================
 */

/**
 * Compute linear index within a brick from local voxel coordinates
 */
FOUNDATION_INLINE uint32_t voxel_linear_index(int32_t vx, int32_t vy,
                                              int32_t vz) {
  return (uint32_t)(vx + (vy << BRICK_SHIFT) + (vz << (BRICK_SHIFT * 2)));
}

/**
 * Compute linear brick index from brick coordinates
 */
FOUNDATION_INLINE uint32_t brick_linear_index(const WorldBrickMap *world,
                                              int32_t bx, int32_t by,
                                              int32_t bz) {
  return (uint32_t)(bx + by * (int32_t)world->stride_y +
                    bz * (int32_t)world->stride_z);
}

/* ============================================================================
 * Section 21: SDF Primitive Helpers (Inline)
 * ============================================================================
 */

/**
 * Signed distance to a box (rounded corners)
 */
FOUNDATION_INLINE float sdf_box(Vec3 p, Vec3 center, Vec3 half_size) {
  Vec3 d = vec3_sub(vec3_abs(vec3_sub(p, center)), half_size);
  Vec3 d_clamped = vec3_max(d, VEC3_ZERO);
  float outside = vec3_length(d_clamped);
  float inside = minf(maxf(d.x, maxf(d.y, d.z)), 0.0f);
  return outside + inside;
}

/**
 * Signed distance to a sphere
 */
FOUNDATION_INLINE float sdf_sphere(Vec3 p, Vec3 center, float radius) {
  return vec3_length(vec3_sub(p, center)) - radius;
}

/**
 * Signed distance to a vertical cylinder (Z-axis aligned)
 */
FOUNDATION_INLINE float sdf_cylinder(Vec3 p, Vec3 center, float radius,
                                     float half_height) {
  Vec3 rel = vec3_sub(p, center);
  float dist_xy = sqrtf(rel.x * rel.x + rel.y * rel.y) - radius;
  float dist_z = absf(rel.z) - half_height;
  float outside = sqrtf(maxf(dist_xy, 0.0f) * maxf(dist_xy, 0.0f) +
                        maxf(dist_z, 0.0f) * maxf(dist_z, 0.0f));
  float inside = minf(maxf(dist_xy, dist_z), 0.0f);
  return outside + inside;
}

/* ============================================================================
 * Section 22: Vec3Batch Helper Functions (Inline)
 * ============================================================================
 */

/**
 * Create a Vec3Batch with allocated arrays
 */
FOUNDATION_INLINE Vec3Batch vec3_batch_create(Arena *arena, uint32_t capacity) {
  Vec3Batch batch = {0};
  if (arena == NULL || capacity == 0)
    return batch;

  batch.x = arena_alloc_aligned(arena, sizeof(float) * capacity,
                                FOUNDATION_SIMD_ALIGNMENT);
  batch.y = arena_alloc_aligned(arena, sizeof(float) * capacity,
                                FOUNDATION_SIMD_ALIGNMENT);
  batch.z = arena_alloc_aligned(arena, sizeof(float) * capacity,
                                FOUNDATION_SIMD_ALIGNMENT);

  if (batch.x && batch.y && batch.z) {
    batch.capacity = capacity;
    batch.count = 0;
  }
  return batch;
}

/**
 * Convert AoS Vec3 array to SoA Vec3Batch
 */
FOUNDATION_INLINE void
vec3_batch_from_aos(Vec3Batch *batch, const Vec3 *positions, uint32_t count) {
  if (batch == NULL || positions == NULL || count > batch->capacity)
    return;

  for (uint32_t i = 0; i < count; i++) {
    batch->x[i] = positions[i].x;
    batch->y[i] = positions[i].y;
    batch->z[i] = positions[i].z;
  }
  batch->count = count;
}

/* ============================================================================
 * Section 23: Type Size Verification
 * ============================================================================
 */

/* SoA layout: each brick uses BRICK_VOXELS bytes for SDF, BRICK_VOXELS for
 * material */
FOUNDATION_STATIC_ASSERT(BRICK_VOXELS == 512, "Brick must contain 512 voxels");
FOUNDATION_STATIC_ASSERT(BRICK_SIZE == 8, "Brick size must be 8");

#ifdef __cplusplus
}
#endif

#endif /* WORLD_BRICK_MAP_H */
