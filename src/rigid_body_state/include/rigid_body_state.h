/**
 * Rigid Body State Module - Robot-agnostic SoA data structures
 *
 * Provides the platform abstraction layer's base types:
 * - RigidBodyStateSOA: 13 float arrays for position, velocity, orientation, angular velocity
 * - RigidBodyParamsSOA: 8 float arrays for mass, inertia, limits, gravity
 * - PlatformStateSOA: RigidBodyStateSOA + platform-specific extension arrays
 * - PlatformParamsSOA: RigidBodyParamsSOA + platform-specific extension arrays
 *
 * All arrays are 32-byte aligned for AVX2 aligned loads.
 *
 * Coordinate Frames:
 * - Position/velocity: World frame (ENU: X=Forward, Y=Left, Z=Up)
 * - Angular velocity (omega): Body frame
 * - Quaternion: World-to-body rotation
 */

#ifndef RIGID_BODY_STATE_H
#define RIGID_BODY_STATE_H

#include "foundation.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Section 1: Constants
 * ============================================================================ */

#define RIGID_BODY_STATE_ARRAY_COUNT  13
#define RIGID_BODY_PARAMS_ARRAY_COUNT  8

/* ============================================================================
 * Section 2: RigidBodyStateSOA
 * ============================================================================ */

/**
 * Rigid body state in Structure-of-Arrays layout (13 arrays)
 *
 * Per-agent memory: 13 floats x 4 bytes = 52 bytes
 * 1024 agents: 52 KB (fits in L2 cache)
 */
typedef struct RigidBodyStateSOA {
    /* Position (world frame, meters) */
    float* pos_x;
    float* pos_y;
    float* pos_z;

    /* Linear velocity (world frame, m/s) */
    float* vel_x;
    float* vel_y;
    float* vel_z;

    /* Orientation quaternion (world-to-body, w is scalar) */
    float* quat_w;
    float* quat_x;
    float* quat_y;
    float* quat_z;

    /* Angular velocity (body frame, rad/s) */
    float* omega_x;
    float* omega_y;
    float* omega_z;

    /* Metadata */
    uint32_t capacity;
    uint32_t count;
} RigidBodyStateSOA;

/* ============================================================================
 * Section 3: RigidBodyParamsSOA
 * ============================================================================ */

/**
 * Rigid body physical parameters in Structure-of-Arrays layout (8 arrays)
 *
 * Contains physics constants that rarely change during simulation.
 * Platform-specific parameters (arm_length, k_thrust, etc.) go in extensions.
 *
 * Per-agent memory: 8 floats x 4 bytes = 32 bytes
 */
typedef struct RigidBodyParamsSOA {
    /* Mass and inertia */
    float* mass;        /* kg */
    float* ixx;         /* kg*m^2 (moment of inertia about x) */
    float* iyy;         /* kg*m^2 (moment of inertia about y) */
    float* izz;         /* kg*m^2 (moment of inertia about z) */

    /* Geometry */
    float* collision_radius; /* m (for collision detection) */

    /* Physical limits */
    float* max_vel;     /* m/s - maximum linear velocity */
    float* max_omega;   /* rad/s - maximum angular velocity */

    /* Environment */
    float* gravity;     /* m/s^2 (typically 9.81) */

    /* Metadata */
    uint32_t capacity;
    uint32_t count;
} RigidBodyParamsSOA;

/* ============================================================================
 * Section 4: PlatformStateSOA and PlatformParamsSOA
 * ============================================================================ */

/**
 * Platform state: rigid body core + platform-specific extension arrays.
 *
 * Extension arrays are platform-defined (e.g., quadcopter: rpm_0..rpm_3).
 * The vtable determines extension_count and semantics.
 */
typedef struct PlatformStateSOA {
    RigidBodyStateSOA rigid_body;
    float** extension;          /* [extension_count] pointers to float[capacity] arrays */
    uint32_t extension_count;
} PlatformStateSOA;

/**
 * Platform parameters: rigid body core + platform-specific extension arrays.
 *
 * Extension arrays are platform-defined (e.g., quadcopter: arm_length, k_thrust, etc.).
 */
typedef struct PlatformParamsSOA {
    RigidBodyParamsSOA rigid_body;
    float** extension;          /* [extension_count] pointers to float[capacity] arrays */
    uint32_t extension_count;
} PlatformParamsSOA;

/* ============================================================================
 * Section 5: RigidBodyStateSOA Lifecycle
 * ============================================================================ */

/**
 * Create rigid body state arrays from arena allocator.
 *
 * Allocates 13 float arrays with 32-byte alignment.
 *
 * @param arena Arena allocator to use
 * @param capacity Maximum number of agents
 * @return Pointer to RigidBodyStateSOA, or NULL on failure
 */
RigidBodyStateSOA* rigid_body_state_create(Arena* arena, uint32_t capacity);

/**
 * Zero all rigid body states.
 *
 * Sets all values to zero except quaternion w-component (set to 1 for identity).
 *
 * @param states RigidBodyStateSOA to zero
 */
void rigid_body_state_zero(RigidBodyStateSOA* states);

/**
 * Reset agents at scattered indices to specified positions/orientations.
 *
 * Velocities are set to zero.
 *
 * @param states RigidBodyStateSOA to modify
 * @param indices Array of agent indices to reset
 * @param positions Array of reset positions
 * @param orientations Array of reset orientations
 * @param count Number of agents to reset
 */
void rigid_body_state_reset_batch(RigidBodyStateSOA* states,
                                  const uint32_t* indices,
                                  const Vec3* positions,
                                  const Quat* orientations,
                                  uint32_t count);

/**
 * Copy state data between SoA structures.
 *
 * @param dst Destination RigidBodyStateSOA
 * @param src Source RigidBodyStateSOA
 * @param dst_offset Starting index in destination
 * @param src_offset Starting index in source
 * @param count Number of agents to copy
 */
void rigid_body_state_copy(RigidBodyStateSOA* dst, const RigidBodyStateSOA* src,
                           uint32_t dst_offset, uint32_t src_offset, uint32_t count);

/**
 * Calculate total memory size for rigid body state arrays.
 *
 * @param capacity Number of agents
 * @return Total bytes required
 */
size_t rigid_body_state_memory_size(uint32_t capacity);

/* ============================================================================
 * Section 6: RigidBodyParamsSOA Lifecycle
 * ============================================================================ */

/**
 * Create rigid body parameter arrays from arena allocator.
 *
 * @param arena Arena allocator to use
 * @param capacity Maximum number of agents
 * @return Pointer to RigidBodyParamsSOA, or NULL on failure
 */
RigidBodyParamsSOA* rigid_body_params_create(Arena* arena, uint32_t capacity);

/**
 * Initialize a single agent's rigid body parameters to defaults.
 *
 * Default: mass=0.5kg, diagonal inertia, collision_radius=0.15m,
 * max_vel=20 m/s, max_omega=10 rad/s, gravity=9.81 m/s^2
 *
 * @param params RigidBodyParamsSOA to modify
 * @param index Agent index to initialize
 */
void rigid_body_params_init(RigidBodyParamsSOA* params, uint32_t index);

/**
 * Calculate total memory size for rigid body parameter arrays.
 *
 * @param capacity Number of agents
 * @return Total bytes required
 */
size_t rigid_body_params_memory_size(uint32_t capacity);

/* ============================================================================
 * Section 7: PlatformStateSOA Lifecycle
 * ============================================================================ */

/**
 * Create platform state with rigid body core + extension arrays.
 *
 * @param arena Arena allocator to use
 * @param capacity Maximum number of agents
 * @param extension_count Number of platform-specific float arrays
 * @return Pointer to PlatformStateSOA, or NULL on failure
 */
PlatformStateSOA* platform_state_create(Arena* arena, uint32_t capacity,
                                        uint32_t extension_count);

/**
 * Zero all platform states (rigid body + extensions).
 *
 * Sets rigid body to default (identity quaternion), zeros all extensions.
 *
 * @param states PlatformStateSOA to zero
 */
void platform_state_zero(PlatformStateSOA* states);

/**
 * Calculate total memory size for platform state.
 *
 * @param capacity Number of agents
 * @param extension_count Number of extension arrays
 * @return Total bytes required
 */
size_t platform_state_memory_size(uint32_t capacity, uint32_t extension_count);

/**
 * Copy platform state (rigid body + extensions) between SOA arrays.
 *
 * Both src and dst must have the same extension_count.
 *
 * @param dst Destination platform state
 * @param src Source platform state
 * @param dst_offset Start index in destination
 * @param src_offset Start index in source
 * @param count Number of agents to copy
 */
void platform_state_copy(PlatformStateSOA* dst, const PlatformStateSOA* src,
                          uint32_t dst_offset, uint32_t src_offset, uint32_t count);

/* ============================================================================
 * Section 8: PlatformParamsSOA Lifecycle
 * ============================================================================ */

/**
 * Create platform parameters with rigid body core + extension arrays.
 *
 * @param arena Arena allocator to use
 * @param capacity Maximum number of agents
 * @param extension_count Number of platform-specific float arrays
 * @return Pointer to PlatformParamsSOA, or NULL on failure
 */
PlatformParamsSOA* platform_params_create(Arena* arena, uint32_t capacity,
                                          uint32_t extension_count);

/**
 * Calculate total memory size for platform parameters.
 *
 * @param capacity Number of agents
 * @param extension_count Number of extension arrays
 * @return Total bytes required
 */
size_t platform_params_memory_size(uint32_t capacity, uint32_t extension_count);

#ifdef __cplusplus
}
#endif

#endif /* RIGID_BODY_STATE_H */
