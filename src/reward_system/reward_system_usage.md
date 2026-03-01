# Reward System Module - API Reference

## Overview

The Reward System module provides task-specific reward computation and termination condition checking for drone reinforcement learning training. It supports multiple task types with SIMD-optimized batch processing.

## Performance Targets (1024 drones)

| Operation | Target | Typical |
|-----------|--------|---------|
| Hover reward compute | <300 us | ~200 us |
| Race reward (10 gates) | <500 us | ~350 us |
| Termination check | <100 us | ~50 us |
| Total reward frame | <1 ms | ~600 us |

## Memory Budget (~85 KB for 1024 drones, 10 gates)

- TargetSOA arrays: 28 KB
- GateSOA arrays: ~15 KB
- Previous state tracking: 20 KB
- Episode tracking: 16 KB
- TerminationFlags: 6 KB

## Quick Start

```c
#include "reward_system.h"

// Create arena and dependencies
Arena* arena = arena_create(16 * 1024 * 1024);
DroneStateSOA* states = drone_state_create(arena, 1024);
CollisionSystem* collision = collision_create(arena, 1024, 0.1f, 1.0f);

// Create reward system with hover task
RewardConfig config = reward_config_default(TASK_HOVER);
RewardSystem* reward = reward_create(arena, &config, 1024, 0);

// Set targets for all drones
PCG32 rng;
pcg32_seed(&rng, 12345);
reward_set_targets_random(reward, 1024,
    VEC3(-10, -10, 0), VEC3(10, 10, 10), &rng);

// In training loop:
float rewards[1024];
float actions[1024 * 4];

// Compute collision detection first
collision_detect_all(collision, states, world, 1024);
CollisionResults results = collision_get_results(collision);

// Compute rewards
reward_compute(reward, states, params, actions, &results, rewards, 1024);

// Check terminations
TerminationFlags flags = { /* allocate arrays */ };
reward_compute_terminations(reward, states, &results,
    VEC3(-100, -100, 0), VEC3(100, 100, 100), 1000, &flags, 1024);

// Reset done drones
for (uint32_t i = 0; i < 1024; i++) {
    if (flags.done[i]) {
        reward_reset(reward, i);
        // Also reset drone state...
    }
}
```

## Task Types

### TASK_HOVER
Maintain position at target with upright orientation.

```c
RewardConfig cfg = reward_config_default(TASK_HOVER);
cfg.distance_scale = 2.0f;      // Penalty per meter
cfg.uprightness_scale = 1.0f;   // Reward for being upright
cfg.reach_bonus = 10.0f;        // Bonus for reaching target
cfg.reach_radius = 0.5f;        // Meters to consider "reached"
```

### TASK_RACE
Navigate through gates in order.

```c
RewardConfig cfg = reward_config_default(TASK_RACE);
RewardSystem* reward = reward_create(arena, &cfg, 1024, 10);

// Setup racing gates
Vec3 centers[10] = { ... };
Vec3 normals[10] = { ... };  // Gate facing direction
float radii[10] = { ... };
reward_set_gates(reward, centers, normals, radii, 10);

cfg.gate_pass_bonus = 20.0f;    // Bonus per gate
cfg.progress_scale = 1.0f;      // Reward for approaching next gate
```

### TASK_TRACK
Follow a moving target with velocity matching.

```c
RewardConfig cfg = reward_config_default(TASK_TRACK);
cfg.velocity_match_scale = 0.5f;  // Reward for matching target velocity
cfg.distance_scale = 1.5f;

// Set moving target
reward_set_target(reward, drone_idx,
    VEC3(10, 0, 5),           // Position
    VEC3(2, 0, 0),            // Velocity (moving in +X)
    1.0f);                    // Radius

// Update target positions each step
reward_update_targets(reward, dt, count);
```

### TASK_LAND
Soft landing at target location.

```c
RewardConfig cfg = reward_config_default(TASK_LAND);
cfg.landing_velocity_scale = 10.0f;  // Penalty for hard landing
cfg.uprightness_scale = 2.0f;        // Must stay upright
cfg.success_bonus = 100.0f;          // Bonus for soft landing
```

### TASK_FORMATION
Multi-agent formation flying.

```c
RewardConfig cfg = reward_config_default(TASK_FORMATION);
cfg.formation_position_scale = 2.0f;

// Set relative positions (drone 0 is leader)
reward_set_target(reward, 0, VEC3(0, 0, 5), VEC3_ZERO, 1.0f);  // Leader target
reward_set_target(reward, 1, VEC3(2, 0, 0), VEC3_ZERO, 0.5f);  // +2m X from leader
reward_set_target(reward, 2, VEC3(-2, 0, 0), VEC3_ZERO, 0.5f); // -2m X from leader
```

## Reward Configuration

### Distance Rewards
```c
config.distance_scale = 1.0f;     // Multiplier for distance penalty
config.distance_exp = 1.0f;       // Exponent (1=linear, 2=quadratic)
config.delta_distance_scale = 1.0f; // Reward for distance improvement
```

### Reach/Success
```c
config.reach_bonus = 10.0f;       // Bonus for reaching target
config.reach_radius = 0.5f;       // Radius to consider reached
config.success_bonus = 100.0f;    // Bonus for completing task
```

### Orientation
```c
config.uprightness_scale = 0.5f;  // Reward for upright orientation
config.heading_scale = 0.2f;      // Reward for correct heading
```

### Energy/Smoothness
```c
config.energy_scale = 0.01f;      // Penalty for motor power usage
config.jerk_scale = 0.05f;        // Penalty for action changes
```

### Collision Penalties
```c
config.collision_penalty = 10.0f;        // Base collision penalty
config.world_collision_penalty = 5.0f;   // Additional for world collision
config.drone_collision_penalty = 5.0f;   // Additional for drone-drone
```

### Clipping
```c
config.reward_min = -100.0f;      // Minimum reward per step
config.reward_max = 100.0f;       // Maximum reward per step
```

## Termination Conditions

```c
TerminationFlags flags;
// Allocate flag arrays...

reward_compute_terminations(reward, states, &collisions,
    bounds_min, bounds_max, max_steps, &flags, count);

// Check individual conditions:
if (flags.collision[i])    // Hit something
if (flags.out_of_bounds[i]) // Left world bounds
if (flags.timeout[i])      // Exceeded max_steps
if (flags.success[i])      // Completed task
if (flags.truncated[i])    // Episode truncated (timeout)
if (flags.done[i])         // Any termination
```

## Episode Statistics

```c
EpisodeStats stats = reward_get_episode_stats(reward, drone_idx);

printf("Return: %.2f\n", stats.episode_return);
printf("Length: %u\n", stats.episode_length);
printf("Gates: %u\n", stats.gates_passed);
printf("Best dist: %.2f\n", stats.best_distance);
printf("Success: %s\n", stats.success ? "yes" : "no");
```

## Resetting Drones

```c
// Reset single drone
reward_reset(reward, drone_idx);

// Reset multiple drones (batch)
uint32_t done_indices[256];
uint32_t done_count = 0;
for (uint32_t i = 0; i < count; i++) {
    if (flags.done[i]) {
        done_indices[done_count++] = i;
    }
}
reward_reset_batch(reward, done_indices, done_count);

// Reset gate progress specifically
reward_reset_gates(reward, drone_idx);
```

## Gate Crossing Detection

```c
// Manual gate crossing check
Vec3 prev_pos = /* previous position */;
bool crossed = reward_check_gate_crossing(reward, states,
    drone_idx, gate_idx, prev_pos);

// The race reward function handles this automatically
```

## Memory Calculation

```c
size_t mem = reward_memory_size(1024, 10);
printf("Memory required: %zu bytes\n", mem);

// Ensure arena has enough space
Arena* arena = arena_create(mem + overhead);
```

## Integration Example

```c
void training_step(TrainingContext* ctx) {
    // 1. Get actions from policy
    policy_get_actions(ctx->policy, ctx->observations, ctx->actions);

    // 2. Step physics
    physics_step(ctx->physics, ctx->states, ctx->params, ctx->actions, ctx->dt);

    // 3. Collision detection
    collision_detect_all(ctx->collision, ctx->states, ctx->world, ctx->count);
    CollisionResults col = collision_get_results(ctx->collision);

    // 4. Collision response
    collision_apply_response(ctx->collision, ctx->states, ctx->params,
                             0.5f, 1.0f, ctx->count);

    // 5. Compute rewards
    reward_compute(ctx->reward, ctx->states, ctx->params,
                   ctx->actions, &col, ctx->rewards, ctx->count);

    // 6. Check terminations
    reward_compute_terminations(ctx->reward, ctx->states, &col,
                                ctx->bounds_min, ctx->bounds_max,
                                ctx->max_steps, &ctx->term_flags, ctx->count);

    // 7. Handle resets
    for (uint32_t i = 0; i < ctx->count; i++) {
        if (ctx->term_flags.done[i]) {
            // Log episode
            EpisodeStats stats = reward_get_episode_stats(ctx->reward, i);
            log_episode(stats);

            // Reset drone
            reward_reset(ctx->reward, i);
            reset_drone_state(ctx->states, i, ctx->rng);
        }
    }

    // 8. Generate observations for next step
    generate_observations(ctx->states, ctx->observations, ctx->count);
}
```

## Thread Safety

The reward system is **not thread-safe**. Each thread should have its own RewardSystem instance, or external synchronization must be used.

For parallel environments, create separate RewardSystem instances per environment batch, or use a single instance with non-overlapping drone index ranges per thread.

## SIMD Optimization

The module automatically uses AVX2 (x86_64) or NEON (ARM) when available:

- Distance computation: 8-wide SIMD on AVX2
- Reward accumulation: Vectorized operations
- Termination checks: Batch processing

Ensure arrays are 32-byte aligned (handled automatically by arena allocation).
