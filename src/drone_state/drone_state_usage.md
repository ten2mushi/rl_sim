# Drone State Module Usage Guide

Concise reference for developers working with the drone state submodule.

## Overview

The drone-state module provides Structure-of-Arrays (SoA) data structures for drone state and parameters, optimized for SIMD vectorization and cache-efficient batch processing.

**Key characteristics:**
- 32-byte aligned arrays for AVX2/NEON
- 68 bytes per drone (state), 60 bytes (params), 28 bytes (episode)
- 1024 drones fit in L2 cache (~156 KB total)

## Include and Link

```c
#include "drone_state.h"
```

```cmake
target_link_libraries(your_target PRIVATE drone_state)
```

---

## Data Structures

### DroneStateSOA

Hot data for physics simulation (17 float arrays).

| Field | Type | Description |
|-------|------|-------------|
| `pos_x`, `pos_y`, `pos_z` | `float*` | Position (world frame, meters) |
| `vel_x`, `vel_y`, `vel_z` | `float*` | Velocity (world frame, m/s) |
| `quat_w`, `quat_x`, `quat_y`, `quat_z` | `float*` | Orientation quaternion (world-to-body) |
| `omega_x`, `omega_y`, `omega_z` | `float*` | Angular velocity (body frame, rad/s) |
| `rpm_0`, `rpm_1`, `rpm_2`, `rpm_3` | `float*` | Motor RPMs (FR-CW, RL-CW, FL-CCW, RR-CCW) |
| `capacity` | `uint32_t` | Maximum drone count |
| `count` | `uint32_t` | Active drone count |

### DroneParamsSOA

Physics constants (15 float arrays).

| Field | Type | Description |
|-------|------|-------------|
| `mass` | `float*` | Mass (kg) |
| `ixx`, `iyy`, `izz` | `float*` | Moments of inertia (kg·m²) |
| `arm_length` | `float*` | Motor arm length (m) |
| `collision_radius` | `float*` | Collision sphere radius (m) |
| `k_thrust` | `float*` | Thrust coefficient N/(rad/s)² |
| `k_torque` | `float*` | Torque coefficient N·m/(rad/s)² |
| `k_drag` | `float*` | Linear drag coefficient |
| `k_ang_damp` | `float*` | Angular damping coefficient |
| `motor_tau` | `float*` | Motor time constant (s) |
| `max_rpm` | `float*` | Maximum motor RPM (rad/s) |
| `max_vel` | `float*` | Maximum velocity (m/s) |
| `max_omega` | `float*` | Maximum angular velocity (rad/s) |
| `gravity` | `float*` | Gravity magnitude (m/s²) |
| `capacity`, `count` | `uint32_t` | Capacity and active count |

### DroneEpisodeData

RL episode tracking (AoS, 28 bytes).

| Field | Type | Description |
|-------|------|-------------|
| `episode_return` | `float` | Cumulative reward |
| `best_episode_return` | `float` | Best return seen |
| `episode_length` | `uint32_t` | Steps in episode |
| `total_episodes` | `uint32_t` | Episodes completed |
| `env_id` | `uint32_t` | Environment index |
| `drone_id` | `uint32_t` | Drone index |
| `done` | `uint8_t` | Episode terminated |
| `truncated` | `uint8_t` | Episode truncated |

### DroneStateAoS / DroneParamsAoS

Single-drone views for debugging and accessors.

```c
typedef struct DroneStateAoS {
    Vec3 position;
    Vec3 velocity;
    Quat orientation;
    Vec3 omega;
    float rpm[4];
} DroneStateAoS;
```

---

## API Reference

### Lifecycle Functions

```c
DroneStateSOA* drone_state_create(Arena* arena, uint32_t capacity)
```
- **arena**: Arena allocator to use
- **capacity**: Maximum number of drones
- **Returns**: Pointer to DroneStateSOA, or NULL on failure

```c
DroneParamsSOA* drone_params_create(Arena* arena, uint32_t capacity)
```
- **arena**: Arena allocator to use
- **capacity**: Maximum number of drones
- **Returns**: Pointer to DroneParamsSOA, or NULL on failure

```c
DroneEpisodeData* drone_episode_create(Arena* arena, uint32_t capacity)
```
- **arena**: Arena allocator to use
- **capacity**: Maximum number of drones
- **Returns**: Pointer to DroneEpisodeData array, or NULL on failure

---

### Initialization Functions

```c
void drone_state_init(DroneStateSOA* states, uint32_t index)
```
- **states**: DroneStateSOA to modify
- **index**: Drone index to initialize
- **Effect**: Sets position=0, velocity=0, orientation=identity, omega=0, rpms=0

```c
void drone_params_init(DroneParamsSOA* params, uint32_t index)
```
- **params**: DroneParamsSOA to modify
- **index**: Drone index to initialize
- **Effect**: Sets default parameters for ~0.5kg quadcopter

```c
void drone_episode_init(DroneEpisodeData* episodes, uint32_t index, uint32_t env_id, uint32_t drone_id)
```
- **episodes**: Episode data array
- **index**: Episode index to initialize
- **env_id**: Environment ID
- **drone_id**: Drone ID within environment
- **Effect**: Resets episode return, length, done flags

---

### Batch Operations

```c
void drone_state_zero(DroneStateSOA* states)
```
- **states**: DroneStateSOA to zero
- **Effect**: SIMD-optimized zeroing of all arrays; quaternion set to identity (w=1)

```c
void drone_state_reset_batch(DroneStateSOA* states, const uint32_t* indices, const Vec3* positions, const Quat* orientations, uint32_t count)
```
- **states**: DroneStateSOA to modify
- **indices**: Array of drone indices to reset
- **positions**: Array of reset positions
- **orientations**: Array of reset orientations
- **count**: Number of drones to reset
- **Effect**: Sets position/orientation at indices; zeros velocity, omega, rpms

```c
void drone_state_copy(DroneStateSOA* dst, const DroneStateSOA* src, uint32_t dst_offset, uint32_t src_offset, uint32_t count)
```
- **dst**: Destination DroneStateSOA
- **src**: Source DroneStateSOA
- **dst_offset**: Starting index in destination
- **src_offset**: Starting index in source
- **count**: Number of drones to copy
- **Effect**: Copies all 17 arrays from src to dst

---

### Single-Drone Accessors

```c
DroneStateAoS drone_state_get(const DroneStateSOA* states, uint32_t index)
```
- **states**: DroneStateSOA to read from
- **index**: Drone index
- **Returns**: DroneStateAoS with gathered state data

```c
void drone_state_set(DroneStateSOA* states, uint32_t index, const DroneStateAoS* state)
```
- **states**: DroneStateSOA to modify
- **index**: Drone index
- **state**: State to write
- **Effect**: Scatters AoS data to SoA arrays

```c
DroneParamsAoS drone_params_get(const DroneParamsSOA* params, uint32_t index)
```
- **params**: DroneParamsSOA to read from
- **index**: Drone index
- **Returns**: DroneParamsAoS with gathered parameter data

```c
void drone_params_set(DroneParamsSOA* params, uint32_t index, const DroneParamsAoS* param)
```
- **params**: DroneParamsSOA to modify
- **index**: Drone index
- **param**: Parameters to write
- **Effect**: Scatters AoS data to SoA arrays

---

### Utility Functions

```c
size_t drone_state_memory_size(uint32_t capacity)
```
- **capacity**: Number of drones
- **Returns**: Total bytes required for state arrays

```c
size_t drone_params_memory_size(uint32_t capacity)
```
- **capacity**: Number of drones
- **Returns**: Total bytes required for parameter arrays

```c
bool drone_state_validate(const DroneStateSOA* states, uint32_t index)
```
- **states**: DroneStateSOA to validate
- **index**: Drone index to check
- **Returns**: true if valid (no NaN, unit quaternion, non-negative RPMs)

```c
void drone_state_print(const DroneStateSOA* states, uint32_t index)
```
- **states**: DroneStateSOA to print from
- **index**: Drone index to print
- **Effect**: Prints formatted state to stdout

```c
void drone_params_print(const DroneParamsSOA* params, uint32_t index)
```
- **params**: DroneParamsSOA to print from
- **index**: Drone index to print
- **Effect**: Prints formatted parameters to stdout

---

## Usage Examples

### Basic Setup

```c
#include "drone_state.h"

int main(void) {
    // Create arena (2MB for 1024 drones with margin)
    Arena* arena = arena_create(2 * 1024 * 1024);

    // Allocate state and params
    DroneStateSOA* states = drone_state_create(arena, 1024);
    DroneParamsSOA* params = drone_params_create(arena, 1024);
    DroneEpisodeData* episodes = drone_episode_create(arena, 1024);

    // All arrays are initialized to defaults
    // States: position=0, velocity=0, orientation=identity, rpms=0
    // Params: mass=0.5kg, gravity=9.81, etc.

    arena_destroy(arena);
    return 0;
}
```

### Physics Integration Pattern

```c
void physics_step(DroneStateSOA* states, DroneParamsSOA* params, float dt) {
    uint32_t count = states->capacity;

    // SIMD-friendly loop: process 8 drones at a time (AVX2)
    for (uint32_t i = 0; i < count; i += FOUNDATION_SIMD_WIDTH) {
        // Load position and velocity
        simd_float px = simd_load_ps(&states->pos_x[i]);
        simd_float vx = simd_load_ps(&states->vel_x[i]);

        // Integrate: p += v * dt
        simd_float dt_vec = simd_set1_ps(dt);
        px = simd_fmadd_ps(vx, dt_vec, px);

        // Store result
        simd_store_ps(&states->pos_x[i], px);
    }
}
```

### Batch Reset on Episode Done

```c
void reset_done_environments(DroneStateSOA* states, DroneEpisodeData* episodes,
                             uint32_t num_drones) {
    // Collect done indices
    uint32_t done_indices[1024];
    Vec3 reset_positions[1024];
    Quat reset_orientations[1024];
    uint32_t done_count = 0;

    for (uint32_t i = 0; i < num_drones; i++) {
        if (episodes[i].done || episodes[i].truncated) {
            done_indices[done_count] = i;
            reset_positions[done_count] = VEC3(0.0f, 0.0f, 1.0f);  // 1m altitude
            reset_orientations[done_count] = QUAT_IDENTITY;
            done_count++;

            // Reset episode tracking
            drone_episode_init(episodes, i, episodes[i].env_id, episodes[i].drone_id);
        }
    }

    // Batch reset all done drones
    if (done_count > 0) {
        drone_state_reset_batch(states, done_indices, reset_positions,
                                reset_orientations, done_count);
    }
}
```

### Reading/Writing Single Drone State

```c
void debug_drone(DroneStateSOA* states, uint32_t index) {
    // Get as AoS for convenient access
    DroneStateAoS state = drone_state_get(states, index);

    printf("Drone %u: pos=(%.2f, %.2f, %.2f) vel=(%.2f, %.2f, %.2f)\n",
           index,
           state.position.x, state.position.y, state.position.z,
           state.velocity.x, state.velocity.y, state.velocity.z);

    // Modify and write back
    state.position.z += 1.0f;  // Raise 1 meter
    drone_state_set(states, index, &state);
}
```

### Direct Array Access (Preferred for Performance)

```c
void apply_gravity(DroneStateSOA* states, DroneParamsSOA* params, float dt) {
    uint32_t count = states->capacity;

    // Direct array access is faster than get/set
    for (uint32_t i = 0; i < count; i++) {
        states->vel_z[i] -= params->gravity[i] * dt;
    }
}
```

### Validation After Physics Step

```c
void validate_all_states(DroneStateSOA* states) {
    for (uint32_t i = 0; i < states->capacity; i++) {
        if (!drone_state_validate(states, i)) {
            printf("WARNING: Invalid state at drone %u\n", i);
            drone_state_print(states, i);

            // Reset to safe state
            drone_state_init(states, i);
        }
    }
}
```

---

## Best Practices

1. **Use direct array access** for performance-critical code, not get/set accessors
2. **Process in SIMD-width batches** (8 for AVX2, 4 for NEON) for vectorization
3. **Use `drone_state_reset_batch()`** for scattered resets, not individual `drone_state_init()`
4. **Validate periodically** to catch NaN propagation early
5. **Pre-allocate arena** with `drone_state_memory_size()` + `drone_params_memory_size()` + margin
6. **Keep episode data separate** - it's cold data accessed less frequently

---

## Memory Budget

| Drones | State | Params | Episode | Total |
|--------|-------|--------|---------|-------|
| 256 | 17 KB | 15 KB | 7 KB | 39 KB |
| 1024 | 68 KB | 60 KB | 28 KB | 156 KB |
| 4096 | 272 KB | 240 KB | 112 KB | 624 KB |
| 10000 | 664 KB | 586 KB | 274 KB | 1.5 MB |
