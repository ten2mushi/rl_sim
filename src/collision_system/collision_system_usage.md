# Collision System Module Usage Guide

## Overview

The collision system provides efficient collision detection and response for drone swarm simulation. It uses spatial hashing for O(n) drone-drone collision detection and integrates with the world brick map for drone-world collision via batch SDF queries.

## Key Features

- **Spatial Hash Grid**: O(n) collision detection using prime-based hash combining
- **Drone-Drone Collision**: Sphere-sphere collision detection with pair deduplication
- **Drone-World Collision**: SIMD-optimized batch SDF queries via world_brick_map
- **Physics Response**: Position correction and velocity reflection with configurable restitution
- **K-Nearest Neighbor**: Efficient neighbor queries for observation generation

## Performance Targets (1024 drones)

| Operation | Target Time |
|-----------|-------------|
| Spatial hash clear | <1 μs |
| Spatial hash build | <100 μs |
| Drone-drone detection | <500 μs |
| World collision batch | <200 μs |
| Collision response | <100 μs |
| **Total frame** | **<1 ms** |

## Memory Budget

~61 KB for 1024 drones:
- SpatialHashGrid cell_heads: 16 KB
- SpatialHashGrid entries: 8 KB
- collision_pairs: 16 KB
- drone_world_collision: 1 KB
- penetration_depth: 4 KB
- collision_normals: 16 KB

## Basic Usage

### 1. Create Collision System

```c
#include "collision_system.h"

// Create arena with sufficient memory
Arena* arena = arena_create(4 * 1024 * 1024);  // 4 MB

// Create collision system
// - max_drones: Maximum number of drones
// - drone_radius: Collision sphere radius per drone (meters)
// - cell_size: Spatial hash cell size (should be >= 2 * drone_radius)
CollisionSystem* collision = collision_create(arena, 1024, 0.1f, 1.0f);
```

### 2. Per-Frame Collision Detection

```c
// Run full collision detection pipeline
collision_detect_all(collision, drone_states, world, drone_count);

// Or run individual stages for more control:
collision_reset(collision);
collision_build_spatial_hash(collision, drone_states, drone_count);
collision_detect_drone_drone(collision, drone_states, drone_count);
collision_detect_drone_world(collision, drone_states, world, drone_count);
```

### 3. Query Collision Results

```c
// Get collision results
CollisionResults results = collision_get_results(collision);

// Check drone-drone collision pairs
for (uint32_t p = 0; p < results.pair_count; p++) {
    uint32_t drone_a = results.pairs[p * 2];
    uint32_t drone_b = results.pairs[p * 2 + 1];
    printf("Collision: drone %u and drone %u\n", drone_a, drone_b);
}

// Check drone-world collisions
for (uint32_t i = 0; i < drone_count; i++) {
    if (results.world_flags[i]) {
        printf("Drone %u colliding with world, penetration: %f\n",
               i, results.penetration[i]);
    }
}

// Or use helper functions
if (collision_drone_world_check(collision, drone_idx)) {
    // Handle world collision
}

uint32_t other = collision_get_pair(collision, drone_idx);
if (other != UINT32_MAX) {
    // Handle drone-drone collision with 'other'
}
```

### 4. Apply Collision Response

```c
// Apply all collision responses
collision_apply_response(collision, drone_states, drone_params,
                         restitution, separation_force, drone_count);

// Or apply separately:
// World response: position correction + velocity reflection
collision_apply_world_response(collision, drone_states,
                               restitution, pushout_speed, drone_count);

// Drone-drone response: mass-weighted separation + impulse exchange
collision_apply_drone_response(collision, drone_states, drone_params,
                               restitution, drone_count);
```

### 5. K-Nearest Neighbor Queries

```c
// Find k nearest neighbors for a single position
uint32_t indices[8];
float distances[8];
uint32_t found;

collision_find_k_nearest(collision, drone_states, query_pos, 8,
                         indices, distances, &found);

// Batch KNN for all drones (for observation generation)
uint32_t* all_indices = arena_alloc_array(arena, uint32_t, drone_count * 8);
float* all_distances = arena_alloc_array(arena, float, drone_count * 8);

collision_find_k_nearest_batch(collision, drone_states, drone_count, 8,
                               all_indices, all_distances);

// Access neighbors for drone i:
for (uint32_t k = 0; k < 8; k++) {
    uint32_t neighbor_idx = all_indices[i * 8 + k];
    float neighbor_dist_sq = all_distances[i * 8 + k];
}
```

## Complete Frame Example

```c
void physics_frame(CollisionSystem* collision,
                   DroneStateSOA* states,
                   DroneParamsSOA* params,
                   WorldBrickMap* world,
                   uint32_t count,
                   float dt) {
    // 1. Detect all collisions
    collision_detect_all(collision, states, world, count);

    // 2. Apply collision responses
    float restitution = 0.5f;      // Bounce factor
    float pushout_speed = 1.0f;    // Position correction speed

    collision_apply_response(collision, states, params,
                             restitution, pushout_speed, count);

    // 3. Check for terminal conditions (crashed drones)
    for (uint32_t i = 0; i < count; i++) {
        if (collision_drone_world_check(collision, i)) {
            float penetration = collision->penetration_depth[i];
            if (penetration < -0.1f) {  // Significant penetration
                // Mark drone as crashed
            }
        }
    }
}
```

## Tuning Parameters

### Cell Size
- Should be at least 2× drone radius
- Larger cells = more drones per cell to check
- Smaller cells = more hash collisions from neighborhood queries
- Recommended: 1.0m for 10cm radius drones

### Restitution
- 0.0 = Perfectly inelastic (no bounce)
- 1.0 = Perfectly elastic (full energy conservation)
- Recommended: 0.3-0.7 for realistic drone behavior

### World Collision Margin
- Default: Equal to drone radius
- Larger margin = earlier detection but potential false positives
- Set via `sys->world_collision_margin` after creation

## Thread Safety

The collision system is **not thread-safe** by design for performance. Each simulation environment should have its own CollisionSystem instance.

For parallel environments:
```c
// Create one collision system per environment
CollisionSystem* collision_systems[NUM_ENVS];
for (int i = 0; i < NUM_ENVS; i++) {
    collision_systems[i] = collision_create(arena, drones_per_env, 0.1f, 1.0f);
}

// Each thread processes its own environment
void process_env(int env_id) {
    CollisionSystem* collision = collision_systems[env_id];
    // ... collision detection and response
}
```

## Debugging

### Enable Debug Assertions
```c
// Define before including header or in CMake
#define FOUNDATION_DEBUG 1
```

### Visualize Spatial Hash
```c
// Count drones per cell for visualization
void visualize_hash(const CollisionSystem* sys) {
    uint32_t max_chain = 0;
    uint32_t non_empty = 0;

    for (uint32_t i = 0; i < HASH_TABLE_SIZE; i++) {
        uint32_t count = 0;
        uint32_t entry = sys->spatial_hash->cell_heads[i];
        while (entry != SPATIAL_HASH_END) {
            count++;
            entry = sys->spatial_hash->entries[entry].next;
        }
        if (count > 0) non_empty++;
        if (count > max_chain) max_chain = count;
    }

    printf("Hash utilization: %u/%u cells (%.1f%%)\n",
           non_empty, HASH_TABLE_SIZE,
           100.0f * non_empty / HASH_TABLE_SIZE);
    printf("Max chain length: %u\n", max_chain);
}
```

## API Reference

See `include/collision_system.h` for complete API documentation.

### Key Types
- `SpatialHashGrid`: Spatial hash grid for O(n) detection
- `CollisionSystem`: Main collision detection system
- `CollisionResults`: Read-only view of detection results
- `CellQuery`: Query result container

### Key Functions
- `collision_create()`: Create collision system
- `collision_detect_all()`: Run full detection pipeline
- `collision_apply_response()`: Apply all collision responses
- `collision_get_results()`: Get detection results
- `collision_find_k_nearest_batch()`: Batch KNN queries
