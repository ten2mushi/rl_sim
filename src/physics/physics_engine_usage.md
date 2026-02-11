# Physics Engine Module Usage Guide

Concise reference for developers working with the physics engine submodule.

## Overview

The physics module provides RK4-integrated quadcopter physics simulation with SIMD-vectorized batch processing.

**Key characteristics:**
- RK4 integration for numerical accuracy (O(dt⁴) error)
- SIMD vectorization (AVX2/NEON) for parallel processing
- X-configuration quadcopter model with 4 motors
- Quaternion-based orientation (no gimbal lock)
- Configurable: drag, ground effect, motor dynamics

**Performance (1024 drones):**
- Full physics step: <0.3 ms (target: <5 ms)
- Derivative computation: <0.01 ms
- Quaternion normalize: <0.001 ms

## Include and Link

```c
#include "physics.h"
```

```cmake
target_link_libraries(your_target PRIVATE physics)
```

---

## Data Structures

### PhysicsConfig

Simulation configuration parameters.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dt` | `float` | 0.02 | Base timestep (seconds, 50 Hz) |
| `dt_variance` | `float` | 0.0 | Domain randomization variance [0, 0.1] |
| `substeps` | `uint32_t` | 4 | Physics substeps per frame |
| `gravity` | `float` | 9.81 | Gravitational acceleration (m/s²) |
| `air_density` | `float` | 1.225 | Air density (kg/m³) |
| `enable_drag` | `bool` | true | Enable aerodynamic drag |
| `enable_ground_effect` | `bool` | true | Enable ground effect thrust boost |
| `enable_motor_dynamics` | `bool` | true | Enable first-order motor lag |
| `enable_gyroscopic` | `bool` | false | Enable gyroscopic precession |
| `ground_effect_height` | `float` | 0.5 | Reference height for ground effect (m) |
| `ground_effect_coeff` | `float` | 1.15 | Maximum ground effect multiplier |
| `max_linear_accel` | `float` | 100.0 | Max linear acceleration clamp (m/s²) |
| `max_angular_accel` | `float` | 200.0 | Max angular acceleration clamp (rad/s²) |

### PhysicsSystem

Main physics simulation system.

| Field | Type | Description |
|-------|------|-------------|
| `config` | `PhysicsConfig` | Current configuration |
| `scratch_arena` | `Arena*` | Scratch memory for RK4 temporaries |
| `k1`, `k2`, `k3`, `k4` | `DroneStateSOA*` | RK4 derivative storage |
| `temp_state` | `DroneStateSOA*` | Temporary state for intermediate steps |
| `forces_x/y/z` | `float*` | World-frame force buffers [capacity] |
| `torques_x/y/z` | `float*` | Body-frame torque buffers [capacity] |
| `step_count` | `uint64_t` | Total physics steps executed |
| `total_integration_time` | `double` | Cumulative integration time (s) |
| `max_drones` | `uint32_t` | Maximum drone capacity |

---

## API Reference

### Lifecycle Functions

```c
PhysicsConfig physics_config_default(void)
```
- **Returns**: PhysicsConfig with sensible default values

```c
PhysicsSystem* physics_create(Arena* persistent_arena, Arena* scratch_arena, const PhysicsConfig* config, uint32_t max_drones)
```
- **persistent_arena**: Arena for long-lived allocations (PhysicsSystem struct)
- **scratch_arena**: Arena for temporary allocations (reset each step)
- **config**: Physics configuration (NULL for defaults)
- **max_drones**: Maximum number of drones to simulate
- **Returns**: Pointer to PhysicsSystem, or NULL on failure

```c
void physics_destroy(PhysicsSystem* physics)
```
- **physics**: Physics system to destroy (NULL safe)
- **Effect**: Resets statistics and nullifies pointers

```c
size_t physics_memory_size(uint32_t max_drones)
```
- **max_drones**: Maximum drone capacity
- **Returns**: Total bytes required for physics system

---

### Main Physics Step

```c
void physics_step(PhysicsSystem* physics, DroneStateSOA* states, const DroneParamsSOA* params, const float* actions, uint32_t count)
```
- **physics**: Physics system
- **states**: Drone states to update (modified in place)
- **params**: Drone physical parameters
- **actions**: Motor commands [count × 4] normalized [0, 1]
- **count**: Number of drones to simulate
- **Effect**: Advances simulation by config.dt with config.substeps substeps

```c
void physics_step_dt(PhysicsSystem* physics, DroneStateSOA* states, const DroneParamsSOA* params, const float* actions, uint32_t count, float dt)
```
- **physics**: Physics system
- **states**: Drone states to update (modified in place)
- **params**: Drone physical parameters
- **actions**: Motor commands [count × 4] normalized [0, 1]
- **count**: Number of drones to simulate
- **dt**: Custom timestep in seconds
- **Effect**: Advances simulation by specified dt

---

### Derivative Computation

```c
void physics_compute_derivatives(const DroneStateSOA* states, const DroneParamsSOA* params, const float* actions, DroneStateSOA* derivatives, uint32_t count, const PhysicsConfig* config)
```
- **states**: Current drone states (input)
- **params**: Drone physical parameters (input)
- **actions**: Motor commands [count × 4] (input)
- **derivatives**: Output derivatives (same layout as DroneStateSOA)
- **count**: Number of drones
- **config**: Physics configuration
- **Effect**: Computes ṗ=v, v̇=F/m, q̇=½q⊗ω, ω̇=I⁻¹(τ-ω×Iω)

---

### Motor Dynamics

```c
void physics_motor_dynamics(const float* rpm_commands, float* actual_rpms, const DroneParamsSOA* params, float dt, uint32_t count)
```
- **rpm_commands**: Target RPM commands [count × 4] (input)
- **actual_rpms**: Current RPMs to update [count × 4] (modified in place)
- **params**: Drone parameters with motor_tau, max_rpm
- **dt**: Timestep in seconds
- **count**: Number of drones
- **Effect**: First-order lag update: rpm += (target - rpm) × min(dt/tau, 1)

---

### Force and Torque Computation

```c
void physics_compute_forces_torques(const DroneStateSOA* states, const DroneParamsSOA* params, float* forces_x, float* forces_y, float* forces_z, float* torques_x, float* torques_y, float* torques_z, uint32_t count)
```
- **states**: Drone states with RPMs (input)
- **params**: Drone parameters with k_thrust, k_torque, arm_length (input)
- **forces_x/y/z**: Output world-frame forces [count]
- **torques_x/y/z**: Output body-frame torques [count]
- **count**: Number of drones
- **Effect**: Computes thrust and torques from motor RPMs

```c
void physics_apply_drag(const DroneStateSOA* states, const DroneParamsSOA* params, float* forces_x, float* forces_y, float* forces_z, uint32_t count, float air_density)
```
- **states**: Drone states with velocities (input)
- **params**: Drone parameters with k_drag (input)
- **forces_x/y/z**: Forces to modify [count] (modified in place)
- **count**: Number of drones
- **air_density**: Air density kg/m³
- **Effect**: Adds drag force: F_drag = -k_drag × |v| × v

```c
void physics_apply_ground_effect(const DroneStateSOA* states, const DroneParamsSOA* params, float* forces_z, uint32_t count, float ground_height, float effect_coeff)
```
- **states**: Drone states with pos_z (input)
- **params**: Drone parameters (input)
- **forces_z**: Vertical force to modify [count] (modified in place)
- **count**: Number of drones
- **ground_height**: Reference height for ground effect (m)
- **effect_coeff**: Maximum thrust multiplier
- **Effect**: Multiplies thrust near ground: k = 1 + (coeff-1) × exp(-z/h)

---

### RK4 Integration

```c
void physics_rk4_integrate(PhysicsSystem* physics, DroneStateSOA* states, const DroneParamsSOA* params, const float* actions, float dt, uint32_t count)
```
- **physics**: Physics system with scratch buffers
- **states**: Drone states to integrate (modified in place)
- **params**: Drone physical parameters
- **actions**: Motor commands [count × 4]
- **dt**: Timestep in seconds
- **count**: Number of drones
- **Effect**: Full RK4 step: y_new = y + (k1+2k2+2k3+k4)×dt/6

```c
void physics_rk4_substep(const DroneStateSOA* current, const DroneStateSOA* derivative, DroneStateSOA* output, float dt_scale, uint32_t count)
```
- **current**: Current state (input)
- **derivative**: State derivative (input)
- **output**: Output state = current + derivative × dt_scale
- **dt_scale**: Timestep scale (0.5 for k2/k3, 1.0 for k4)
- **count**: Number of drones

```c
void physics_rk4_combine(DroneStateSOA* states, const DroneStateSOA* k1, const DroneStateSOA* k2, const DroneStateSOA* k3, const DroneStateSOA* k4, float dt, uint32_t count)
```
- **states**: State to update (modified in place)
- **k1/k2/k3/k4**: RK4 derivatives
- **dt**: Timestep in seconds
- **count**: Number of drones
- **Effect**: Weighted combination: (k1 + 2×k2 + 2×k3 + k4) / 6

---

### Numerical Stability

```c
void physics_normalize_quaternions(DroneStateSOA* states, uint32_t count)
```
- **states**: Drone states to normalize (modified in place)
- **count**: Number of drones
- **Effect**: Renormalizes quaternions to unit length

```c
void physics_clamp_velocities(DroneStateSOA* states, const DroneParamsSOA* params, uint32_t count)
```
- **states**: Drone states to clamp (modified in place)
- **params**: Drone parameters with max_vel, max_omega
- **count**: Number of drones
- **Effect**: Clamps linear and angular velocities to limits

```c
void physics_clamp_accelerations(float* accel_x, float* accel_y, float* accel_z, float max_accel, uint32_t count)
```
- **accel_x/y/z**: Accelerations to clamp [count] (modified in place)
- **max_accel**: Maximum allowed acceleration magnitude
- **count**: Number of drones

```c
uint32_t physics_sanitize_state(DroneStateSOA* states, uint32_t count)
```
- **states**: Drone states to sanitize (modified in place)
- **count**: Number of drones
- **Returns**: Number of drones reset due to NaN/Inf values
- **Effect**: Detects invalid values; resets to origin, identity quat, zero velocity

---

### Inline Utility Functions

```c
float action_to_rpm(float action, float max_rpm)
```
- **action**: Normalized action [0, 1]
- **max_rpm**: Maximum RPM
- **Returns**: RPM = clamp(action, 0, 1) × max_rpm

```c
float rpm_to_thrust(float rpm, float k_thrust)
```
- **rpm**: Motor RPM
- **k_thrust**: Thrust coefficient
- **Returns**: Thrust = k_thrust × rpm²

```c
float rpm_to_torque(float rpm, float k_torque)
```
- **rpm**: Motor RPM
- **k_torque**: Torque coefficient
- **Returns**: Torque = k_torque × rpm²

```c
Vec3 quat_rotate_body_z_to_world(Quat q, float fz_body)
```
- **q**: Orientation quaternion
- **fz_body**: Body-frame Z force (thrust)
- **Returns**: World-frame force vector

```c
Quat quat_derivative(Quat q, Vec3 omega)
```
- **q**: Current quaternion
- **omega**: Angular velocity (body frame)
- **Returns**: Quaternion derivative q̇ = ½q⊗ω

---

## Usage Examples

### Basic Setup

```c
#include "physics.h"

int main(void) {
    // Create arenas
    Arena* persistent = arena_create(1024 * 1024);  // 1MB
    Arena* scratch = arena_create(512 * 1024);      // 512KB

    // Create physics system with defaults
    PhysicsSystem* physics = physics_create(persistent, scratch, NULL, 1024);

    // Create drone state and params
    DroneStateSOA* states = drone_state_create(persistent, 1024);
    DroneParamsSOA* params = drone_params_create(persistent, 1024);

    // Initialize all drones
    for (uint32_t i = 0; i < 1024; i++) {
        drone_state_init(states, i);
        drone_params_init(params, i);
    }

    // Actions: 4 motors per drone, normalized [0, 1]
    float* actions = arena_alloc_aligned(persistent, 1024 * 4 * sizeof(float), 32);

    // Simulation loop
    for (int step = 0; step < 1000; step++) {
        // Set motor commands (e.g., hover at ~55% throttle)
        for (uint32_t i = 0; i < 1024 * 4; i++) {
            actions[i] = 0.55f;
        }

        // Advance physics
        physics_step(physics, states, params, actions, 1024);
    }

    arena_destroy(scratch);
    arena_destroy(persistent);
    return 0;
}
```

### Custom Configuration

```c
PhysicsConfig config = physics_config_default();
config.dt = 0.01f;                    // 100 Hz
config.substeps = 8;                  // Higher accuracy
config.enable_ground_effect = false;  // Disable ground effect
config.enable_gyroscopic = true;      // Enable gyroscopic precession

PhysicsSystem* physics = physics_create(persistent, scratch, &config, 1024);
```

### Variable Timestep

```c
void update(PhysicsSystem* physics, DroneStateSOA* states,
            DroneParamsSOA* params, float* actions, uint32_t count, float dt) {
    // Use custom timestep for frame-rate independent simulation
    physics_step_dt(physics, states, params, actions, count, dt);
}
```

### Manual RK4 Integration

```c
// For custom physics pipelines, call RK4 directly
void custom_physics_step(PhysicsSystem* physics, DroneStateSOA* states,
                         DroneParamsSOA* params, float* actions,
                         uint32_t count, float dt) {
    // Apply motor dynamics first
    float rpm_commands[4096];
    for (uint32_t i = 0; i < count; i++) {
        for (int m = 0; m < 4; m++) {
            rpm_commands[i * 4 + m] = action_to_rpm(actions[i * 4 + m],
                                                     params->max_rpm[i]);
        }
    }

    float* actual_rpms = &states->rpm_0[0];  // Interleaved access
    physics_motor_dynamics(rpm_commands, actual_rpms, params, dt, count);

    // RK4 integration
    physics_rk4_integrate(physics, states, params, actions, dt, count);

    // Post-processing
    physics_normalize_quaternions(states, count);
    physics_clamp_velocities(states, params, count);
    physics_sanitize_state(states, count);
}
```

### Hover Thrust Calculation

```c
// Calculate throttle for hover
float calculate_hover_throttle(const DroneParamsSOA* params, uint32_t index) {
    float mass = params->mass[index];
    float gravity = params->gravity[index];
    float k_thrust = params->k_thrust[index];
    float max_rpm = params->max_rpm[index];

    // At hover: 4 × k_thrust × rpm² = mass × gravity
    float hover_rpm = sqrtf((mass * gravity) / (4.0f * k_thrust));
    float hover_throttle = hover_rpm / max_rpm;

    return clampf(hover_throttle, 0.0f, 1.0f);
}
```

---

## Motor Configuration (X-Configuration)

```
    Motor 2 (FL, CCW)      Motor 0 (FR, CW)
          \                   /
           \                 /
            +-------+-------+
            |      CG       |
            +-------+-------+
           /                 \
          /                   \
    Motor 1 (RL, CW)       Motor 3 (RR, CCW)
```

**Motor indices:**
- `rpm_0`: Front-Right, Clockwise
- `rpm_1`: Rear-Left, Clockwise
- `rpm_2`: Front-Left, Counter-Clockwise
- `rpm_3`: Rear-Right, Counter-Clockwise

**Torque generation:**
- +Roll: Increase rpm_1, rpm_3; Decrease rpm_0, rpm_2
- +Pitch: Increase rpm_0, rpm_1; Decrease rpm_2, rpm_3
- +Yaw: Increase CW motors (rpm_0, rpm_1); Decrease CCW motors (rpm_2, rpm_3)

---

## Physics Equations

| Quantity | Equation |
|----------|----------|
| Motor thrust | T = k_thrust × rpm² |
| Motor torque | τ = k_torque × rpm² |
| Total thrust | F_z = T₀ + T₁ + T₂ + T₃ |
| Roll torque | τ_x = arm × (T₁ + T₃ - T₀ - T₂) |
| Pitch torque | τ_y = arm × (T₀ + T₁ - T₂ - T₃) |
| Yaw torque | τ_z = k_torque × (rpm₀² + rpm₁² - rpm₂² - rpm₃²) |
| Linear accel | a = F/m - g |
| Angular accel | ω̇ = I⁻¹(τ - ω × Iω) |
| Quat derivative | q̇ = ½q ⊗ [0, ω] |
| Drag force | F_drag = -k_drag × \|v\| × v |
| Ground effect | k_ge = 1 + (k_max - 1) × exp(-z/h) |

---

## Memory Budget

| Drones | Physics System | Total with States |
|--------|----------------|-------------------|
| 256 | 93 KB | 132 KB |
| 1024 | 365 KB | 521 KB |
| 4096 | 1.4 MB | 2.0 MB |
| 10000 | 3.5 MB | 5.0 MB |

---

## Best Practices

1. **Use `physics_step()` for standard simulation** - handles all substeps, normalization, and clamping
2. **Pre-allocate arenas** using `physics_memory_size()` + margin
3. **Normalize actions to [0, 1]** - clamped internally but best to provide clean input
4. **Monitor `physics_sanitize_state()` return value** - non-zero indicates numerical issues
5. **Use 4 substeps minimum** for stable RK4 integration at 50 Hz
6. **Keep scratch arena separate** - allows reset between frames without reallocating
7. **Validate `k_thrust` against mass** - hover should be achievable below max RPM
