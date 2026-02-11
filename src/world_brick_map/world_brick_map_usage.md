# World Brick Map Usage Guide

Sparse SDF (Signed Distance Field) storage using 8×8×8 brick atlas for efficient world representation and raymarching queries.

## Quick Start

```c
#include "world_brick_map.h"

// Create arena and world
Arena* arena = arena_create(16 * 1024 * 1024);  // 16 MB
WorldBrickMap* world = world_create(arena,
    VEC3(-50, -10, -50),  // world_min
    VEC3(50, 30, 50),     // world_max
    0.1f,                 // voxel_size (10cm)
    10000);               // max_bricks

// Add geometry
world_set_ground(world, 0.0f, 1);                          // Ground at z=0
world_set_sphere(world, VEC3(0, 0, 5), 2.0f, 2);          // Sphere
world_set_box(world, VEC3(10, 0, 1), VEC3(2, 3, 1), 3);   // Box

// Query SDF
float dist = world_sdf_query(world, VEC3(0, 0, 8));  // Distance to nearest surface

// Raymarching
RayHit hit = world_raymarch(world, VEC3(0, 0, 20), VEC3(0, 0, -1), 50.0f);
if (hit.hit) {
    printf("Hit at z=%.2f, material=%d\n", hit.position.z, hit.material);
}

// Cleanup
arena_destroy(arena);  // Frees all allocations
```

## Key Concepts

### SDF (Signed Distance Field)
- **Positive**: Outside geometry (distance to nearest surface)
- **Negative**: Inside geometry
- **Zero**: On surface

### Brick Atlas
- World divided into 8×8×8 voxel bricks
- Only bricks near surfaces are allocated (sparse storage)
- Special sentinel values for uniform regions:
  - `BRICK_EMPTY_INDEX (-1)`: Never touched
  - `BRICK_UNIFORM_OUTSIDE (-2)`: All voxels far outside
  - `BRICK_UNIFORM_INSIDE (-3)`: All voxels deep inside

### Memory Optimization
- Demand-paged atlas: 64 bricks per page, allocated on first use
- 8-bit SDF quantization: 75% memory reduction vs float32
- Uniform brick detection: No storage for homogeneous regions

---

## API Reference

### Lifecycle

| Function | Description |
|----------|-------------|
| `world_create` | Create a new world brick map |
| `world_destroy` | Destroy (no-op with arena) |
| `world_clear` | Reset all bricks to empty |
| `world_get_stats` | Get memory statistics |
| `world_memory_size` | Calculate required memory |

```
WorldBrickMap* world_create(Arena* arena, Vec3 world_min, Vec3 world_max, float voxel_size, uint32_t max_bricks) -> WorldBrickMap*
void world_destroy(WorldBrickMap* world) -> void
void world_clear(WorldBrickMap* world) -> void
WorldStats world_get_stats(const WorldBrickMap* world) -> WorldStats
size_t world_memory_size(uint32_t grid_x, uint32_t grid_y, uint32_t grid_z, uint32_t max_bricks) -> size_t
```

### Coordinate Transforms

| Function | Description |
|----------|-------------|
| `world_pos_to_brick` | World position → brick coordinates |
| `world_pos_to_voxel` | World position → local voxel coordinates |
| `world_contains` | Check if position is within bounds |
| `world_brick_valid` | Check if brick coordinates are valid |

```
void world_pos_to_brick(const WorldBrickMap* world, Vec3 pos, int32_t* bx, int32_t* by, int32_t* bz) -> void
void world_pos_to_voxel(const WorldBrickMap* world, Vec3 pos, int32_t bx, int32_t by, int32_t bz, int32_t* vx, int32_t* vy, int32_t* vz) -> void
bool world_contains(const WorldBrickMap* world, Vec3 pos) -> bool
bool world_brick_valid(const WorldBrickMap* world, int32_t bx, int32_t by, int32_t bz) -> bool
```

### Brick Management

| Function | Description |
|----------|-------------|
| `world_get_brick_index` | Get atlas index for brick |
| `world_alloc_brick` | Allocate brick (or return existing) |
| `world_free_brick` | Free brick to free list |
| `world_brick_sdf` | Get SDF data pointer for brick |
| `world_brick_material` | Get material data pointer for brick |
| `brick_is_uniform` | Check if brick is uniform (no storage) |

```
int32_t world_get_brick_index(const WorldBrickMap* world, int32_t bx, int32_t by, int32_t bz) -> int32_t (atlas index or BRICK_EMPTY_INDEX)
int32_t world_alloc_brick(WorldBrickMap* world, int32_t bx, int32_t by, int32_t bz) -> int32_t (atlas index or BRICK_EMPTY_INDEX)
void world_free_brick(WorldBrickMap* world, int32_t bx, int32_t by, int32_t bz) -> void
int8_t* world_brick_sdf(WorldBrickMap* world, int32_t atlas_idx) -> int8_t* (512 bytes)
uint8_t* world_brick_material(WorldBrickMap* world, int32_t atlas_idx) -> uint8_t* (512 bytes)
bool brick_is_uniform(int32_t brick_idx) -> bool
```

### SDF Queries

| Function | Description |
|----------|-------------|
| `world_sdf_query` | Query SDF with trilinear interpolation |
| `world_sdf_query_nearest` | Query SDF (nearest neighbor) |
| `world_material_query` | Query material at position |
| `world_sdf_gradient` | Compute SDF gradient (unnormalized) |
| `world_sdf_normal` | Compute surface normal (unit length) |

```
float world_sdf_query(const WorldBrickMap* world, Vec3 pos) -> float
float world_sdf_query_nearest(const WorldBrickMap* world, Vec3 pos) -> float
uint8_t world_material_query(const WorldBrickMap* world, Vec3 pos) -> uint8_t (0 = air)
Vec3 world_sdf_gradient(const WorldBrickMap* world, Vec3 pos) -> Vec3
Vec3 world_sdf_normal(const WorldBrickMap* world, Vec3 pos) -> Vec3
```

### Voxel Modification

| Function | Description |
|----------|-------------|
| `world_set_sdf` | Set SDF at position |
| `world_set_material` | Set material at position |
| `world_set_voxel` | Set both SDF and material |

```
void world_set_sdf(WorldBrickMap* world, Vec3 pos, float sdf) -> void
void world_set_material(WorldBrickMap* world, Vec3 pos, uint8_t material) -> void
void world_set_voxel(WorldBrickMap* world, Vec3 pos, float sdf, uint8_t material) -> void
```

### Primitive Generation

| Function | Description |
|----------|-------------|
| `world_set_box` | Add axis-aligned box |
| `world_set_sphere` | Add sphere |
| `world_set_cylinder` | Add Z-axis cylinder |
| `world_set_ground` | Add ground plane at height |

```
void world_set_box(WorldBrickMap* world, Vec3 center, Vec3 half_size, uint8_t material) -> void
void world_set_sphere(WorldBrickMap* world, Vec3 center, float radius, uint8_t material) -> void
void world_set_cylinder(WorldBrickMap* world, Vec3 center, float radius, float half_height, uint8_t material) -> void
void world_set_ground(WorldBrickMap* world, float ground_z, uint8_t material) -> void
```

### Raymarching

| Function | Description |
|----------|-------------|
| `world_raymarch` | Single ray sphere tracing |
| `world_raymarch_batch` | Batch raymarch multiple rays |
| `world_raymarch_camera` | Generate depth/material buffers |

```
RayHit world_raymarch(const WorldBrickMap* world, Vec3 origin, Vec3 direction, float max_distance) -> RayHit
void world_raymarch_batch(const WorldBrickMap* world, const Vec3* origins, const Vec3* directions, float max_distance, RayHit* hits, uint32_t count) -> void
void world_raymarch_camera(const WorldBrickMap* world, Vec3 camera_pos, Vec3 camera_forward, Vec3 camera_up, float fov_h, float fov_v, uint32_t width, uint32_t height, float max_distance, float* depth_buffer, uint8_t* material_buffer) -> void
```

### Batch Operations (SIMD)

| Function | Description |
|----------|-------------|
| `world_sdf_query_batch` | Batch SDF query |
| `world_sdf_gradient_batch` | Batch gradient calculation |

```
void world_sdf_query_batch(const WorldBrickMap* world, const Vec3* positions, float* sdfs, uint32_t count) -> void
void world_sdf_gradient_batch(const WorldBrickMap* world, const Vec3* positions, Vec3* gradients, uint32_t count) -> void
```

### Edit List (Incremental Regeneration)

| Function | Description |
|----------|-------------|
| `edit_list_create` | Create edit list |
| `edit_list_clear` | Clear all entries |
| `edit_list_add` | Add edit entry |
| `edit_list_count` | Get entry count |

```
EditList* edit_list_create(Arena* arena, uint32_t capacity) -> EditList*
void edit_list_clear(EditList* list) -> void
bool edit_list_add(EditList* list, CSGOperation op, PrimitiveType primitive, Vec3 center, Vec3 params, uint8_t material) -> bool
uint32_t edit_list_count(const EditList* list) -> uint32_t
```

**CSGOperation enum:** `CSG_UNION`, `CSG_SUBTRACT`, `CSG_INTERSECT`

**PrimitiveType enum:** `PRIM_BOX`, `PRIM_SPHERE`, `PRIM_CYLINDER`, `PRIM_GROUND`

**EditEntry params field:**
- Box: `(half_x, half_y, half_z)`
- Sphere: `(radius, 0, 0)`
- Cylinder: `(radius, half_height, 0)`
- Ground: `(ground_z, 0, 0)`

### Dirty Tracking

| Function | Description |
|----------|-------------|
| `dirty_tracker_create` | Create dirty tracker |
| `dirty_tracker_clear` | Clear all dirty flags |
| `dirty_tracker_mark_brick` | Mark single brick dirty |
| `dirty_tracker_mark_region` | Mark region dirty |
| `dirty_tracker_is_dirty` | Check if brick is dirty |
| `dirty_tracker_count` | Get dirty brick count |

```
DirtyTracker* dirty_tracker_create(Arena* arena, uint32_t max_bricks) -> DirtyTracker*
void dirty_tracker_clear(DirtyTracker* tracker) -> void
void dirty_tracker_mark_brick(DirtyTracker* tracker, uint32_t brick_index) -> void
void dirty_tracker_mark_region(DirtyTracker* tracker, const WorldBrickMap* world, Vec3 min_pos, Vec3 max_pos) -> void
bool dirty_tracker_is_dirty(const DirtyTracker* tracker, uint32_t brick_index) -> bool
uint32_t dirty_tracker_count(const DirtyTracker* tracker) -> uint32_t
```

### Incremental Regeneration

| Function | Description |
|----------|-------------|
| `world_mark_dirty_bricks` | Mark bricks in edit region |
| `world_regenerate_dirty` | Regenerate all dirty bricks |
| `world_regenerate_brick` | Regenerate single brick |

```
void world_mark_dirty_bricks(WorldBrickMap* world, DirtyTracker* tracker, Vec3 edit_min, Vec3 edit_max) -> void
void world_regenerate_dirty(WorldBrickMap* world, DirtyTracker* tracker, const EditList* edits) -> void
void world_regenerate_brick(WorldBrickMap* world, uint32_t brick_index, const EditList* edits) -> void
```

### Uniform Brick Detection

| Function | Description |
|----------|-------------|
| `world_detect_uniform_brick` | Check if brick is uniform |
| `world_compact_uniform_bricks` | Convert uniform bricks to sentinels |
| `world_mark_brick_uniform_outside` | Mark brick as uniform outside |
| `world_mark_brick_uniform_inside` | Mark brick as uniform inside |

```
int32_t world_detect_uniform_brick(const WorldBrickMap* world, int32_t atlas_idx) -> int32_t (BRICK_UNIFORM_* or atlas_idx)
uint32_t world_compact_uniform_bricks(WorldBrickMap* world) -> uint32_t (bricks converted)
void world_mark_brick_uniform_outside(WorldBrickMap* world, int32_t bx, int32_t by, int32_t bz) -> void
void world_mark_brick_uniform_inside(WorldBrickMap* world, int32_t bx, int32_t by, int32_t bz) -> void
```

### Clip Map LOD

| Function | Description |
|----------|-------------|
| `clipmap_create` | Create clip map with LOD levels |
| `clipmap_destroy` | Destroy clip map |
| `clipmap_update_focus` | Update focus point (camera pos) |
| `clipmap_sdf_query` | Query SDF with auto LOD |
| `clipmap_raymarch` | Raymarch with LOD transitions |
| `clipmap_select_level` | Get LOD level for position |
| `clipmap_set_sphere` | Add sphere to all levels |
| `clipmap_set_box` | Add box to all levels |

```
ClipMapWorld* clipmap_create(Arena* arena, float base_voxel_size, float base_extent, uint32_t bricks_per_level) -> ClipMapWorld*
void clipmap_destroy(ClipMapWorld* clipmap) -> void
void clipmap_update_focus(ClipMapWorld* clipmap, Vec3 new_focus) -> void
float clipmap_sdf_query(const ClipMapWorld* clipmap, Vec3 pos) -> float
RayHit clipmap_raymarch(const ClipMapWorld* clipmap, Vec3 origin, Vec3 direction, float max_distance) -> RayHit
int clipmap_select_level(const ClipMapWorld* clipmap, Vec3 pos) -> int (0 = highest detail)
void clipmap_set_sphere(ClipMapWorld* clipmap, Vec3 center, float radius, uint8_t material) -> void
void clipmap_set_box(ClipMapWorld* clipmap, Vec3 center, Vec3 half_size, uint8_t material) -> void
```

### SDF Quantization (Inline)

| Function | Description |
|----------|-------------|
| `sdf_quantize` | Float → int8 |
| `sdf_dequantize` | int8 → float |

```
int8_t sdf_quantize(float sdf, float inv_sdf_scale) -> int8_t [-127, +127]
float sdf_dequantize(int8_t q, float sdf_scale) -> float
```

### Index Helpers (Inline)

| Function | Description |
|----------|-------------|
| `voxel_linear_index` | Local voxel coords → index |
| `brick_linear_index` | Brick coords → grid index |

```
uint32_t voxel_linear_index(int32_t vx, int32_t vy, int32_t vz) -> uint32_t [0, 511]
uint32_t brick_linear_index(const WorldBrickMap* world, int32_t bx, int32_t by, int32_t bz) -> uint32_t
```

### SDF Primitives (Inline)

| Function | Description |
|----------|-------------|
| `sdf_box` | Box signed distance |
| `sdf_sphere` | Sphere signed distance |
| `sdf_cylinder` | Z-axis cylinder signed distance |
| `sdf_ground` | Ground plane signed distance |

```
float sdf_box(Vec3 p, Vec3 center, Vec3 half_size) -> float
float sdf_sphere(Vec3 p, Vec3 center, float radius) -> float
float sdf_cylinder(Vec3 p, Vec3 center, float radius, float half_height) -> float
float sdf_ground(Vec3 p, float ground_z) -> float
```

### Vec3Batch Helpers (Inline)

| Function | Description |
|----------|-------------|
| `vec3_batch_create` | Create SoA batch |
| `vec3_batch_from_aos` | Convert AoS → SoA |

```
Vec3Batch vec3_batch_create(Arena* arena, uint32_t capacity) -> Vec3Batch
void vec3_batch_from_aos(Vec3Batch* batch, const Vec3* positions, uint32_t count) -> void
```

---

## Data Structures

### RayHit
```c
typedef struct RayHit {
    Vec3     position;    // World position of hit
    Vec3     normal;      // Surface normal (unit length)
    float    distance;    // Distance from ray origin
    uint8_t  material;    // Material ID at hit
    bool     hit;         // True if ray hit geometry
} RayHit;
```

### WorldStats
```c
typedef struct WorldStats {
    uint32_t total_bricks;       // Maximum capacity
    uint32_t active_bricks;      // Allocated in atlas
    uint32_t uniform_outside;    // Bricks marked uniform outside
    uint32_t uniform_inside;     // Bricks marked uniform inside
    uint32_t free_list_count;    // Bricks in free list
    uint32_t pages_allocated;    // Atlas pages allocated
    size_t   grid_memory;        // Grid memory (bytes)
    size_t   atlas_memory;       // Atlas memory (bytes)
    size_t   total_memory;       // Total memory (bytes)
    float    fill_ratio;         // active_bricks / total_bricks
} WorldStats;
```

### EditEntry
```c
typedef struct EditEntry {
    CSGOperation    op;         // CSG_UNION, CSG_SUBTRACT, CSG_INTERSECT
    PrimitiveType   primitive;  // PRIM_BOX, PRIM_SPHERE, PRIM_CYLINDER, PRIM_GROUND
    Vec3            center;     // Primitive center
    Vec3            params;     // Shape-specific (see above)
    uint8_t         material;   // Material ID
} EditEntry;
```

---

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `BRICK_SIZE` | 8 | Voxels per brick dimension |
| `BRICK_VOXELS` | 512 | Total voxels per brick (8³) |
| `BRICK_SHIFT` | 3 | log₂(8) for bit shifts |
| `BRICK_MASK` | 7 | Mask for local coords |
| `BRICK_EMPTY_INDEX` | -1 | Empty brick sentinel |
| `BRICK_UNIFORM_OUTSIDE` | -2 | All voxels outside |
| `BRICK_UNIFORM_INSIDE` | -3 | All voxels inside |
| `ATLAS_PAGE_BRICKS` | 64 | Bricks per atlas page |
| `MAX_ATLAS_PAGES` | 512 | Max pages (32K bricks) |
| `RAYMARCH_MAX_STEPS` | 128 | Max sphere trace steps |
| `RAYMARCH_HIT_DIST` | 0.01f | Hit distance threshold |
| `CLIPMAP_LEVELS` | 4 | LOD hierarchy levels |

---

## Common Patterns

### Incremental World Editing

```c
// Setup
EditList* edits = edit_list_create(arena, 100);
DirtyTracker* tracker = dirty_tracker_create(arena, world->grid_total);

// Add initial geometry
edit_list_add(edits, CSG_UNION, PRIM_SPHERE, VEC3(0,0,0), VEC3(5,0,0), 1);
world_mark_dirty_bricks(world, tracker, VEC3(-6,-6,-6), VEC3(6,6,6));
world_regenerate_dirty(world, tracker, edits);

// Later: carve a hole
edit_list_add(edits, CSG_SUBTRACT, PRIM_SPHERE, VEC3(0,0,0), VEC3(2,0,0), 0);
world_mark_dirty_bricks(world, tracker, VEC3(-3,-3,-3), VEC3(3,3,3));
world_regenerate_dirty(world, tracker, edits);
```

### Clip Map for Large Worlds

```c
// Create: 10cm base voxels, 20m extent at level 0
ClipMapWorld* clip = clipmap_create(arena, 0.1f, 20.0f, 5000);

// Add geometry
clipmap_set_ground(clip, 0.0f, 1);  // Not implemented, use per-level
for (int i = 0; i < CLIPMAP_LEVELS; i++) {
    world_set_ground(clip->levels[i].map, 0.0f, 1);
}

// Update focus as camera moves
clipmap_update_focus(clip, camera_position);

// Query with automatic LOD
float dist = clipmap_sdf_query(clip, query_pos);
RayHit hit = clipmap_raymarch(clip, ray_origin, ray_dir, 100.0f);
```

### Batch Sensor Generation

```c
// Allocate ray arrays
Vec3* origins = arena_alloc_array(arena, Vec3, 1024);
Vec3* directions = arena_alloc_array(arena, Vec3, 1024);
RayHit* hits = arena_alloc_array(arena, RayHit, 1024);

// Fill ray data...

// Batch raymarch (SIMD optimized)
world_raymarch_batch(world, origins, directions, 50.0f, hits, 1024);

// Process results
for (int i = 0; i < 1024; i++) {
    if (hits[i].hit) {
        depths[i] = hits[i].distance;
    }
}
```
