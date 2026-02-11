# Sensor System Usage Guide

## Overview

The sensor system provides a vtable-based polymorphic framework for generating drone observations. Key features:

- **Batch-by-type processing**: Single vtable dispatch per sensor type (not per drone)
- **Zero-copy observations**: 32-byte aligned buffer for direct numpy interop
- **Deterministic noise**: Per-sensor PCG32 RNG for reproducibility
- **Arena allocation**: Zero per-frame heap allocations

## Quick Start

```c
#include "sensor_system.h"
#include "sensor_implementations.h"

// 1. Create arena and sensor system
Arena* arena = arena_create(64 * 1024 * 1024);  // 64 MB
SensorSystem* sys = sensor_system_create(arena, 1024, 64, 128);

// 2. Register sensor implementations
sensor_implementations_register_all(&sys->registry);

// 3. Create sensors
uint32_t imu = sensor_system_create_sensor(sys, &(SensorConfig){
    .type = SENSOR_TYPE_IMU,
    .imu = { .accel_noise = 0.01f, .gyro_noise = 0.001f }
});
uint32_t lidar = sensor_system_create_sensor(sys,
    &sensor_config_lidar_2d(64, 3.14159f, 20.0f));

// 4. Attach sensors to drones
for (uint32_t d = 0; d < num_drones; d++) {
    sensor_system_attach(sys, d, imu);
    sensor_system_attach(sys, d, lidar);
}

// 5. Sample all sensors (call each frame)
sensor_system_sample_all(sys, drones, world, collision, num_drones);

// 6. Access observations (zero-copy)
float* obs = sensor_system_get_observations(sys);
// obs layout: [drone_0_obs, drone_1_obs, ..., drone_N_obs]
// Each drone_obs: [imu_6_floats, lidar_64_floats, ...]

// 7. Cleanup
sensor_system_destroy(sys);
arena_destroy(arena);
```

## Sensor Types

| Type | Output | Description |
|------|--------|-------------|
| `SENSOR_TYPE_IMU` | 6 floats | ax, ay, az, gx, gy, gz (body frame) |
| `SENSOR_TYPE_TOF` | 1 float | Distance along sensing direction |
| `SENSOR_TYPE_LIDAR_2D` | N floats | Distances for N rays in horizontal plane |
| `SENSOR_TYPE_LIDAR_3D` | N×M floats | Distances for N horizontal × M vertical rays |
| `SENSOR_TYPE_CAMERA_RGB` | W×H×3 floats | RGB values per pixel |
| `SENSOR_TYPE_CAMERA_DEPTH` | W×H floats | Normalized depth [0,1] |
| `SENSOR_TYPE_CAMERA_SEGMENTATION` | W×H floats | Material IDs per pixel |
| `SENSOR_TYPE_POSITION` | 3 floats | x, y, z world position (oracle) |
| `SENSOR_TYPE_VELOCITY` | 6 floats | vx, vy, vz, wx, wy, wz (oracle) |
| `SENSOR_TYPE_NEIGHBOR` | K×4 floats | dx, dy, dz, dist for K nearest neighbors |

## Performance Targets (1024 drones)

| Sensor | Target |
|--------|--------|
| IMU | <0.15ms |
| Position/Velocity | <0.05ms |
| ToF | <0.5ms |
| LiDAR 2D (64 rays) | <5ms |
| LiDAR 3D (16×64) | <20ms |
| Camera RGB (64×64) | <20ms |
| Camera Depth (64×64) | <15ms |
| Neighbor (K=5) | <1ms |

---

## API Reference

### Lifecycle Functions

```
sensor_system_create(arena, max_drones, max_sensors, max_obs_dim) -> SensorSystem*
```
- `arena`: Arena* - Memory arena for allocation
- `max_drones`: uint32_t - Maximum drone count
- `max_sensors`: uint32_t - Maximum unique sensors
- `max_obs_dim`: size_t - Maximum observation floats per drone
- **Returns**: SensorSystem* or NULL on failure

```
sensor_system_destroy(sys) -> void
```
- `sys`: SensorSystem* - System to destroy (can be NULL)

```
sensor_system_reset(sys) -> void
```
- `sys`: SensorSystem* - System to reset (clears obs buffer, keeps sensors)

### Registry Functions

```
sensor_registry_init(registry) -> void
```
- `registry`: SensorRegistry* - Registry to initialize

```
sensor_registry_get(registry, type) -> const SensorVTable*
```
- `registry`: const SensorRegistry* - Registry to query
- `type`: SensorType - Type to look up
- **Returns**: Vtable pointer or NULL if not registered

```
sensor_registry_register(registry, type, vtable) -> void
```
- `registry`: SensorRegistry* - Registry to modify
- `type`: SensorType - Type to register
- `vtable`: const SensorVTable* - Implementation vtable

```
sensor_implementations_register_all(registry) -> void
```
- `registry`: SensorRegistry* - Registry to populate with all 10 sensor types

### Sensor Management

```
sensor_system_create_sensor(sys, config) -> uint32_t
```
- `sys`: SensorSystem* - Sensor system
- `config`: const SensorConfig* - Sensor configuration
- **Returns**: Sensor index (0 to sensor_count-1), or UINT32_MAX on failure

```
sensor_system_attach(sys, drone_idx, sensor_idx) -> uint32_t
```
- `sys`: SensorSystem* - Sensor system
- `drone_idx`: uint32_t - Drone to attach to
- `sensor_idx`: uint32_t - Sensor to attach
- **Returns**: Output offset in observation buffer, or UINT32_MAX on failure

```
sensor_system_detach(sys, drone_idx, attachment_idx) -> void
```
- `sys`: SensorSystem* - Sensor system
- `drone_idx`: uint32_t - Drone index
- `attachment_idx`: uint32_t - Attachment index (0 to attachment_count-1)

```
sensor_system_get_sensor(sys, sensor_idx) -> Sensor*
```
- `sys`: SensorSystem* - Sensor system
- `sensor_idx`: uint32_t - Sensor index
- **Returns**: Pointer to sensor or NULL if invalid

```
sensor_system_get_attachment_count(sys, drone_idx) -> uint32_t
```
- `sys`: const SensorSystem* - Sensor system
- `drone_idx`: uint32_t - Drone index
- **Returns**: Number of attached sensors

### Batch Processing (Critical Path)

```
sensor_system_sample_all(sys, drones, world, collision, drone_count) -> void
```
- `sys`: SensorSystem* - Sensor system
- `drones`: const DroneStateSOA* - Drone state arrays
- `world`: const WorldBrickMap* - World for raymarching (can be NULL)
- `collision`: const CollisionSystem* - For neighbor queries (can be NULL)
- `drone_count`: uint32_t - Number of drones to process

```
sensor_system_sample_sensor(sys, sensor_idx, drones, world, collision, drone_count) -> void
```
- `sys`: SensorSystem* - Sensor system
- `sensor_idx`: uint32_t - Specific sensor to sample
- `drones`: const DroneStateSOA* - Drone state arrays
- `world`: const WorldBrickMap* - World for raymarching (can be NULL)
- `collision`: const CollisionSystem* - For neighbor queries (can be NULL)
- `drone_count`: uint32_t - Number of drones

### Observation Access

```
sensor_system_get_observations(sys) -> float*
```
- `sys`: SensorSystem* - Sensor system
- **Returns**: Pointer to observation buffer [max_drones × obs_dim], 32-byte aligned

```
sensor_system_get_observations_const(sys) -> const float*
```
- `sys`: const SensorSystem* - Sensor system
- **Returns**: Const pointer to observation buffer

```
sensor_system_get_obs_dim(sys) -> size_t
```
- `sys`: const SensorSystem* - Sensor system
- **Returns**: Total floats per drone observation

```
sensor_system_get_drone_obs(sys, drone_idx) -> float*
```
- `sys`: SensorSystem* - Sensor system
- `drone_idx`: uint32_t - Drone index
- **Returns**: Pointer to drone's observation (obs_dim floats)

```
sensor_system_get_drone_obs_const(sys, drone_idx) -> const float*
```
- `sys`: const SensorSystem* - Sensor system
- `drone_idx`: uint32_t - Drone index
- **Returns**: Const pointer to drone's observation

### Configuration Helpers

```
sensor_config_default(type) -> SensorConfig
```
- `type`: SensorType - Sensor type
- **Returns**: Default configuration for type

```
sensor_config_imu(accel_noise, gyro_noise) -> SensorConfig
```
- `accel_noise`: float - Accelerometer noise stddev (m/s²)
- `gyro_noise`: float - Gyroscope noise stddev (rad/s)
- **Returns**: IMU configuration

```
sensor_config_tof(direction, max_range) -> SensorConfig
```
- `direction`: Vec3 - Sensing direction in body frame (will be normalized)
- `max_range`: float - Maximum sensing range (meters)
- **Returns**: ToF configuration

```
sensor_config_lidar_2d(num_rays, fov, max_range) -> SensorConfig
```
- `num_rays`: uint32_t - Number of rays
- `fov`: float - Field of view (radians)
- `max_range`: float - Maximum range (meters)
- **Returns**: LiDAR 2D configuration

```
sensor_config_lidar_3d(horizontal_rays, vertical_layers, horizontal_fov, vertical_fov, max_range) -> SensorConfig
```
- `horizontal_rays`: uint32_t - Rays per horizontal sweep
- `vertical_layers`: uint32_t - Number of vertical layers
- `horizontal_fov`: float - Horizontal FOV (radians)
- `vertical_fov`: float - Vertical FOV (radians)
- `max_range`: float - Maximum range (meters)
- **Returns**: LiDAR 3D configuration

```
sensor_config_camera(width, height, fov, max_range) -> SensorConfig
```
- `width`: uint32_t - Image width (pixels)
- `height`: uint32_t - Image height (pixels)
- `fov`: float - Horizontal FOV (radians)
- `max_range`: float - Maximum depth (meters)
- **Returns**: Camera configuration (set .type for RGB/Depth/Seg)

```
sensor_config_neighbor(k, max_range) -> SensorConfig
```
- `k`: uint32_t - Number of nearest neighbors
- `max_range`: float - Maximum search range (meters)
- **Returns**: Neighbor configuration

```
sensor_config_position() -> SensorConfig
```
- **Returns**: Position sensor configuration

```
sensor_config_velocity() -> SensorConfig
```
- **Returns**: Velocity sensor configuration

### Utility Functions

```
sensor_type_name(type) -> const char*
```
- `type`: SensorType - Sensor type
- **Returns**: Human-readable name string

```
sensor_system_memory_size(max_drones, max_sensors, max_obs_dim) -> size_t
```
- `max_drones`: uint32_t - Maximum drones
- `max_sensors`: uint32_t - Maximum sensors
- `max_obs_dim`: size_t - Maximum observation dimensions
- **Returns**: Required bytes

```
sensor_system_compute_obs_dim(sys, drone_idx) -> size_t
```
- `sys`: const SensorSystem* - Sensor system
- `drone_idx`: uint32_t - Drone index
- **Returns**: Total observation dimension for drone

### Ray Precomputation (Advanced)

```
precompute_lidar_2d_rays(arena, num_rays, fov) -> Vec3*
```
- `arena`: Arena* - Memory arena
- `num_rays`: uint32_t - Number of rays
- `fov`: float - Field of view (radians)
- **Returns**: Array of normalized ray directions, or NULL

```
precompute_lidar_3d_rays(arena, horizontal_rays, vertical_layers, horizontal_fov, vertical_fov) -> Vec3*
```
- `arena`: Arena* - Memory arena
- `horizontal_rays`: uint32_t - Rays per horizontal sweep
- `vertical_layers`: uint32_t - Number of vertical layers
- `horizontal_fov`: float - Horizontal FOV (radians)
- `vertical_fov`: float - Vertical FOV (radians)
- **Returns**: Array of normalized ray directions, or NULL

```
precompute_camera_rays(arena, width, height, fov_horizontal, fov_vertical) -> Vec3*
```
- `arena`: Arena* - Memory arena
- `width`: uint32_t - Image width
- `height`: uint32_t - Image height
- `fov_horizontal`: float - Horizontal FOV (radians)
- `fov_vertical`: float - Vertical FOV (radians)
- **Returns**: Array of normalized ray directions, or NULL

```
pcg32_gaussian(rng) -> float
```
- `rng`: PCG32* - RNG state
- **Returns**: Standard normal random value (mean=0, stddev=1)

---

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_SENSORS_PER_DRONE` | 8 | Maximum sensors attachable per drone |
| `SENSOR_OBS_ALIGNMENT` | 32 | Observation buffer alignment (bytes) |
| `SENSOR_TYPE_COUNT` | 10 | Total sensor types |

## Memory Budget

For 1024 drones, 64 sensors, 128-dim observations:

| Component | Size |
|-----------|------|
| Sensor array | 4.5 KB |
| Attachments | 96 KB |
| Attachment counts | 4 KB |
| Observation buffer | 512 KB |
| Drone-by-sensor lists | 256 KB |
| **Total** | **~873 KB** |

## Dependencies

- `foundation`: Vec3, Quat, Arena, PCG32, SIMD utilities
- `drone_state`: DroneStateSOA
- `world_brick_map`: RayHit, world_raymarch (for raymarching sensors)
- `collision_system`: collision_find_k_nearest (for neighbor sensor)
