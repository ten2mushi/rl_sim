/**
 * @file drone_rl.h
 * @brief Unified public API for the Drone RL Engine
 *
 * This header provides a single include for all drone RL engine functionality.
 * Include this header to access the complete API for drone simulation,
 * physics, sensors, rewards, and environment management.
 *
 * Usage:
 *   #include "drone_rl.h"
 *
 * Link against: libdronerl.a (unified static library)
 */

#ifndef DRONE_RL_H
#define DRONE_RL_H

#ifdef __cplusplus
extern "C" {
#endif

/* Core foundation: Arena allocators, Vec3, Quat, PCG32 RNG, SIMD utilities */
#include "foundation.h"

/* SoA data structures for drone state and parameters */
#include "drone_state.h"

/* RK4 quadcopter physics integration */
#include "physics.h"

/* Sparse SDF world representation */
#include "world_brick_map.h"

/* Spatial hash collision detection */
#include "collision_system.h"

/* Sensor framework with vtable polymorphism */
#include "sensor_system.h"

/* Concrete sensor implementations (raycast, IMU, GPS, etc.) */
#include "sensor_implementations.h"

/* Task rewards and termination conditions */
#include "reward_system.h"

/* Thread pool and work-stealing scheduler */
#include "threading.h"

/* TOML configuration parsing and validation */
#include "configuration.h"

/* Top-level orchestration - BatchDroneEngine API */
#include "environment_manager.h"

#ifdef __cplusplus
}
#endif

#endif /* DRONE_RL_H */
