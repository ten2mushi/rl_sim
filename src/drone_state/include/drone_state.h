/**
 * Drone State Module - SoA data structures for drone state and parameters
 *
 * Provides Structure-of-Arrays (SoA) data structures enabling SIMD vectorization
 * and cache-efficient batch processing for drone physics simulation.
 *
 * Memory Layout:
 * - PlatformStateSOA: RigidBodyStateSOA (13 arrays) + extension arrays
 * - PlatformParamsSOA: RigidBodyParamsSOA (8 arrays) + extension arrays
 * - AgentEpisodeData: AoS structure (28 bytes per drone, cold data)
 *
 * All arrays are 32-byte aligned for AVX2 aligned loads (2x faster than unaligned).
 *
 * Coordinate Frames:
 * - Position/velocity: World frame (ENU: X=Forward, Y=Left, Z=Up)
 * - Angular velocity (omega): Body frame
 * - Quaternion: World-to-body rotation
 */

#ifndef PLATFORM_STATE_H
#define PLATFORM_STATE_H

#include "foundation.h"
#include "rigid_body_state.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Section 1: Type Aliases
 * ============================================================================ */

/* PlatformStateSOA and PlatformParamsSOA are defined in rigid_body_state.h.
 * DroneStateSOA is a convenience alias for quadcopter-specific code. */
typedef PlatformStateSOA DroneStateSOA;
typedef PlatformParamsSOA DroneParamsSOA;

/* ============================================================================
 * Section 2: Episode Tracking (AoS, cold data)
 * ============================================================================ */

/**
 * Episode tracking data (AoS, cold data, 28 bytes)
 *
 * Used for RL episode management and statistics.
 * Accessed infrequently, not performance-critical.
 */
typedef struct AgentEpisodeData {
    float episode_return;       /* Cumulative reward this episode */
    float best_episode_return;  /* Best return seen */
    uint32_t episode_length;    /* Steps in current episode */
    uint32_t total_episodes;    /* Total episodes completed */
    uint32_t env_id;            /* Environment index */
    uint32_t agent_id;          /* Drone index within environment */
    uint8_t done;               /* Episode terminated */
    uint8_t truncated;          /* Episode truncated (time limit) */
    uint8_t _pad[2];            /* Padding for alignment */
} AgentEpisodeData;

/* Verify struct sizes at compile time */
FOUNDATION_STATIC_ASSERT(sizeof(AgentEpisodeData) == 28, "AgentEpisodeData must be 28 bytes");

/* ============================================================================
 * Section 3: Array-of-Structures (AoS) Accessor Types
 * ============================================================================ */

/**
 * Single agent state (AoS view for debugging and single-agent access)
 *
 * Contains only rigid body fields. Platform-specific fields (e.g. RPMs)
 * are accessed via extensions.
 */
typedef struct PlatformStateAoS {
    Vec3 position;      /* World frame position */
    Vec3 velocity;      /* World frame velocity */
    Quat orientation;   /* World-to-body quaternion */
    Vec3 omega;         /* Body frame angular velocity */
} PlatformStateAoS;

/**
 * Single agent parameters (AoS view)
 *
 * Contains only rigid body parameters. Platform-specific parameters
 * (arm_length, k_thrust, etc.) are in extensions.
 */
typedef struct PlatformParamsAoS {
    float mass;
    float ixx, iyy, izz;
    float collision_radius;
    float max_vel;
    float max_omega;
    float gravity;
} PlatformParamsAoS;

/* ============================================================================
 * Section 4: Lifecycle Functions
 * ============================================================================ */

/**
 * Create drone episode data array from arena allocator.
 *
 * @param arena Arena allocator to use
 * @param capacity Maximum number of drones
 * @return Pointer to AgentEpisodeData array, or NULL on failure
 */
AgentEpisodeData* agent_episode_create(Arena* arena, uint32_t capacity);

/* ============================================================================
 * Section 5: Initialization Functions
 * ============================================================================ */

/**
 * Initialize a single agent state to default values (rigid body only).
 *
 * Default: position=origin, velocity=0, orientation=identity, omega=0
 * Does NOT touch extension arrays (platform-specific init is via vtable).
 *
 * @param states PlatformStateSOA to modify
 * @param index Agent index to initialize
 */
void platform_state_init(PlatformStateSOA* states, uint32_t index);

/**
 * Initialize a single agent's rigid body parameters to default values.
 *
 * Default values approximate a small robot (~0.5kg):
 * - mass=0.5kg, inertia diagonal, collision_radius=0.15m
 * - max_vel=20 m/s, max_omega=10 rad/s, gravity=9.81 m/s^2
 *
 * Does NOT touch extension arrays (platform-specific init is via vtable).
 *
 * @param params PlatformParamsSOA to modify
 * @param index Agent index to initialize
 */
void platform_params_init(PlatformParamsSOA* params, uint32_t index);

/**
 * Initialize episode data to defaults.
 *
 * @param episodes Episode data array
 * @param index Episode index to initialize
 * @param env_id Environment ID
 * @param agent_id Drone ID within environment
 */
void agent_episode_init(AgentEpisodeData* episodes, uint32_t index,
                        uint32_t env_id, uint32_t agent_id);

/* ============================================================================
 * Section 6: Single-Agent Accessors
 * ============================================================================ */

/**
 * Get agent state as AoS structure (rigid body fields only).
 *
 * Gathers data from SoA arrays into a single struct for debugging/single access.
 *
 * @param states PlatformStateSOA to read from
 * @param index Agent index
 * @return PlatformStateAoS containing the agent's state
 */
PlatformStateAoS platform_state_get(const PlatformStateSOA* states, uint32_t index);

/**
 * Set agent state from AoS structure (rigid body fields only).
 *
 * @param states PlatformStateSOA to modify
 * @param index Agent index
 * @param state State to write
 */
void platform_state_set(PlatformStateSOA* states, uint32_t index, const PlatformStateAoS* state);

/**
 * Get agent parameters as AoS structure (rigid body fields only).
 *
 * @param params PlatformParamsSOA to read from
 * @param index Agent index
 * @return PlatformParamsAoS containing the agent's parameters
 */
PlatformParamsAoS platform_params_get(const PlatformParamsSOA* params, uint32_t index);

/**
 * Set agent parameters from AoS structure (rigid body fields only).
 *
 * @param params PlatformParamsSOA to modify
 * @param index Agent index
 * @param param Parameters to write
 */
void platform_params_set(PlatformParamsSOA* params, uint32_t index, const PlatformParamsAoS* param);

/* ============================================================================
 * Section 7: Utility Functions
 * ============================================================================ */

/**
 * Validate agent state for consistency (rigid body fields only).
 *
 * Checks:
 * - No NaN values in any field
 * - Quaternion is unit normalized (|q|^2 ~ 1.0, tolerance 1e-4)
 *
 * @param states PlatformStateSOA to validate
 * @param index Agent index to check
 * @return true if valid, false otherwise
 */
bool platform_state_validate(const PlatformStateSOA* states, uint32_t index);

/**
 * Print agent state for debugging.
 *
 * @param states PlatformStateSOA to print from
 * @param index Agent index to print
 */
void platform_state_print(const PlatformStateSOA* states, uint32_t index);

/**
 * Print agent parameters for debugging.
 *
 * @param params PlatformParamsSOA to print from
 * @param index Agent index to print
 */
void platform_params_print(const PlatformParamsSOA* params, uint32_t index);

#ifdef __cplusplus
}
#endif

#endif /* PLATFORM_STATE_H */
