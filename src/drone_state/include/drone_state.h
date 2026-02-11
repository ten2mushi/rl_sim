/**
 * Drone State Module - SoA data structures for drone state and parameters
 *
 * Provides Structure-of-Arrays (SoA) data structures enabling SIMD vectorization
 * and cache-efficient batch processing for drone physics simulation.
 *
 * Memory Layout:
 * - DroneStateSOA: 17 float arrays (68 bytes per drone)
 * - DroneParamsSOA: 15 float arrays (60 bytes per drone)
 * - DroneEpisodeData: AoS structure (28 bytes per drone, cold data)
 *
 * All arrays are 32-byte aligned for AVX2 aligned loads (2x faster than unaligned).
 *
 * Motor Convention (X-configuration quadcopter):
 *   M0 = Front-Right (CW)
 *   M1 = Rear-Left (CW)
 *   M2 = Front-Left (CCW)
 *   M3 = Rear-Right (CCW)
 *
 * Coordinate Frames:
 * - Position/velocity: World frame (ENU: X=Forward, Y=Left, Z=Up)
 * - Angular velocity (omega): Body frame
 * - Quaternion: World-to-body rotation
 */

#ifndef DRONE_STATE_H
#define DRONE_STATE_H

#include "foundation.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Section 1: Structure-of-Arrays (SoA) Data Types
 * ============================================================================ */

/**
 * Drone state in Structure-of-Arrays layout (hot data, 17 arrays)
 *
 * Optimized for SIMD batch processing in physics integration.
 * All float arrays are 32-byte aligned for AVX2 operations.
 *
 * Per-drone memory: 17 floats × 4 bytes = 68 bytes
 * 1024 drones: 68 KB (fits in L2 cache)
 */
typedef struct DroneStateSOA {
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

    /* Motor RPMs (M0=FR, M1=RL, M2=FL, M3=RR) */
    float* rpm_0;
    float* rpm_1;
    float* rpm_2;
    float* rpm_3;

    /* Metadata */
    uint32_t capacity;  /* Maximum number of drones */
    uint32_t count;     /* Current active drone count */
} DroneStateSOA;

/**
 * Drone physical parameters in Structure-of-Arrays layout (15 arrays)
 *
 * Contains physics constants that rarely change during simulation.
 * Can be shared across drones with identical physical properties.
 *
 * Per-drone memory: 15 floats × 4 bytes = 60 bytes
 */
typedef struct DroneParamsSOA {
    /* Mass and inertia */
    float* mass;        /* kg */
    float* ixx;         /* kg·m² (moment of inertia about x) */
    float* iyy;         /* kg·m² (moment of inertia about y) */
    float* izz;         /* kg·m² (moment of inertia about z) */

    /* Geometry */
    float* arm_length;      /* m (motor arm length from center) */
    float* collision_radius; /* m (for collision detection) */

    /* Thrust and torque coefficients */
    float* k_thrust;    /* N/(rad/s)² - thrust coefficient */
    float* k_torque;    /* N·m/(rad/s)² - torque coefficient */

    /* Damping coefficients */
    float* k_drag;      /* N/(m/s) - linear drag */
    float* k_ang_damp;  /* N·m/(rad/s) - angular damping */

    /* Motor dynamics */
    float* motor_tau;   /* s - motor time constant */
    float* max_rpm;     /* rad/s - maximum motor angular velocity */

    /* Physical limits */
    float* max_vel;     /* m/s - maximum linear velocity */
    float* max_omega;   /* rad/s - maximum angular velocity */

    /* Environment */
    float* gravity;     /* m/s² (typically 9.81) */

    /* Metadata */
    uint32_t capacity;
    uint32_t count;
} DroneParamsSOA;

/**
 * Episode tracking data (AoS, cold data, 28 bytes)
 *
 * Used for RL episode management and statistics.
 * Accessed infrequently, not performance-critical.
 */
typedef struct DroneEpisodeData {
    float episode_return;       /* Cumulative reward this episode */
    float best_episode_return;  /* Best return seen */
    uint32_t episode_length;    /* Steps in current episode */
    uint32_t total_episodes;    /* Total episodes completed */
    uint32_t env_id;            /* Environment index */
    uint32_t drone_id;          /* Drone index within environment */
    uint8_t done;               /* Episode terminated */
    uint8_t truncated;          /* Episode truncated (time limit) */
    uint8_t _pad[2];            /* Padding for alignment */
} DroneEpisodeData;

/* Verify struct sizes at compile time */
FOUNDATION_STATIC_ASSERT(sizeof(DroneEpisodeData) == 28, "DroneEpisodeData must be 28 bytes");

/* ============================================================================
 * Section 2: Array-of-Structures (AoS) Accessor Types
 * ============================================================================ */

/**
 * Single drone state (AoS view for debugging and single-drone access)
 */
typedef struct DroneStateAoS {
    Vec3 position;      /* World frame position */
    Vec3 velocity;      /* World frame velocity */
    Quat orientation;   /* World-to-body quaternion */
    Vec3 omega;         /* Body frame angular velocity */
    float rpm[4];       /* Motor RPMs */
} DroneStateAoS;

/**
 * Single drone parameters (AoS view)
 */
typedef struct DroneParamsAoS {
    float mass;
    float ixx, iyy, izz;
    float arm_length;
    float collision_radius;
    float k_thrust;
    float k_torque;
    float k_drag;
    float k_ang_damp;
    float motor_tau;
    float max_rpm;
    float max_vel;
    float max_omega;
    float gravity;
} DroneParamsAoS;

/* ============================================================================
 * Section 3: Lifecycle Functions
 * ============================================================================ */

/**
 * Create drone state arrays from arena allocator.
 *
 * Allocates 17 float arrays with 32-byte alignment for AVX2.
 * Memory comes from the arena (no malloc in hot paths).
 *
 * @param arena Arena allocator to use
 * @param capacity Maximum number of drones
 * @return Pointer to DroneStateSOA, or NULL on failure
 */
DroneStateSOA* drone_state_create(Arena* arena, uint32_t capacity);

/**
 * Create drone parameter arrays from arena allocator.
 *
 * @param arena Arena allocator to use
 * @param capacity Maximum number of drones
 * @return Pointer to DroneParamsSOA, or NULL on failure
 */
DroneParamsSOA* drone_params_create(Arena* arena, uint32_t capacity);

/**
 * Create drone episode data array from arena allocator.
 *
 * @param arena Arena allocator to use
 * @param capacity Maximum number of drones
 * @return Pointer to DroneEpisodeData array, or NULL on failure
 */
DroneEpisodeData* drone_episode_create(Arena* arena, uint32_t capacity);

/* ============================================================================
 * Section 4: Initialization Functions
 * ============================================================================ */

/**
 * Initialize a single drone state to default values.
 *
 * Default: position=origin, velocity=0, orientation=identity, omega=0, rpms=0
 *
 * @param states DroneStateSOA to modify
 * @param index Drone index to initialize
 */
void drone_state_init(DroneStateSOA* states, uint32_t index);

/**
 * Initialize a single drone's parameters to default values.
 *
 * Default values approximate a small quadcopter (~0.5kg):
 * - mass=0.5kg, inertia diagonal, arm_length=0.1m
 * - Standard thrust/torque coefficients
 * - gravity=9.81 m/s²
 *
 * @param params DroneParamsSOA to modify
 * @param index Drone index to initialize
 */
void drone_params_init(DroneParamsSOA* params, uint32_t index);

/**
 * Initialize episode data to defaults.
 *
 * @param episodes Episode data array
 * @param index Episode index to initialize
 * @param env_id Environment ID
 * @param drone_id Drone ID within environment
 */
void drone_episode_init(DroneEpisodeData* episodes, uint32_t index,
                        uint32_t env_id, uint32_t drone_id);

/* ============================================================================
 * Section 5: Batch Operations (SIMD-optimized)
 * ============================================================================ */

/**
 * Zero all drone states (SIMD-optimized).
 *
 * Sets all values to zero except quaternion w-component (set to 1 for identity).
 * Uses AVX2 8-wide or NEON 4-wide operations.
 *
 * @param states DroneStateSOA to zero
 */
void drone_state_zero(DroneStateSOA* states);

/**
 * Reset drones at scattered indices to specified positions/orientations.
 *
 * Velocities and RPMs are set to zero. Uses scalar scatter pattern
 * (SIMD gather/scatter not beneficial for scattered indices).
 *
 * @param states DroneStateSOA to modify
 * @param indices Array of drone indices to reset
 * @param positions Array of reset positions (same length as indices)
 * @param orientations Array of reset orientations (same length as indices)
 * @param count Number of drones to reset
 */
void drone_state_reset_batch(DroneStateSOA* states,
                             const uint32_t* indices,
                             const Vec3* positions,
                             const Quat* orientations,
                             uint32_t count);

/**
 * Copy state data between SoA structures.
 *
 * Uses memcpy per array (compiler-optimized for large copies).
 *
 * @param dst Destination DroneStateSOA
 * @param src Source DroneStateSOA
 * @param dst_offset Starting index in destination
 * @param src_offset Starting index in source
 * @param count Number of drones to copy
 */
void drone_state_copy(DroneStateSOA* dst, const DroneStateSOA* src,
                      uint32_t dst_offset, uint32_t src_offset, uint32_t count);

/* ============================================================================
 * Section 6: Single-Drone Accessors
 * ============================================================================ */

/**
 * Get drone state as AoS structure.
 *
 * Gathers data from SoA arrays into a single struct for debugging/single access.
 * May incur cache misses across multiple arrays.
 *
 * @param states DroneStateSOA to read from
 * @param index Drone index
 * @return DroneStateAoS containing the drone's state
 */
DroneStateAoS drone_state_get(const DroneStateSOA* states, uint32_t index);

/**
 * Set drone state from AoS structure.
 *
 * Scatters data from struct to SoA arrays.
 *
 * @param states DroneStateSOA to modify
 * @param index Drone index
 * @param state State to write
 */
void drone_state_set(DroneStateSOA* states, uint32_t index, const DroneStateAoS* state);

/**
 * Get drone parameters as AoS structure.
 *
 * @param params DroneParamsSOA to read from
 * @param index Drone index
 * @return DroneParamsAoS containing the drone's parameters
 */
DroneParamsAoS drone_params_get(const DroneParamsSOA* params, uint32_t index);

/**
 * Set drone parameters from AoS structure.
 *
 * @param params DroneParamsSOA to modify
 * @param index Drone index
 * @param param Parameters to write
 */
void drone_params_set(DroneParamsSOA* params, uint32_t index, const DroneParamsAoS* param);

/* ============================================================================
 * Section 7: Utility Functions
 * ============================================================================ */

/**
 * Calculate total memory size for drone state arrays.
 *
 * Formula: sizeof(DroneStateSOA) + 17 × aligned_array_size
 * where aligned_array_size = (capacity × 4 + 31) & ~31
 *
 * @param capacity Number of drones
 * @return Total bytes required
 */
size_t drone_state_memory_size(uint32_t capacity);

/**
 * Calculate total memory size for drone parameter arrays.
 *
 * Formula: sizeof(DroneParamsSOA) + 15 × aligned_array_size
 *
 * @param capacity Number of drones
 * @return Total bytes required
 */
size_t drone_params_memory_size(uint32_t capacity);

/**
 * Validate drone state for consistency.
 *
 * Checks:
 * - No NaN values in any field
 * - Quaternion is unit normalized (|q|² ≈ 1.0, tolerance 1e-4)
 * - RPMs are non-negative
 *
 * @param states DroneStateSOA to validate
 * @param index Drone index to check
 * @return true if valid, false otherwise
 */
bool drone_state_validate(const DroneStateSOA* states, uint32_t index);

/**
 * Print drone state for debugging.
 *
 * @param states DroneStateSOA to print from
 * @param index Drone index to print
 */
void drone_state_print(const DroneStateSOA* states, uint32_t index);

/**
 * Print drone parameters for debugging.
 *
 * @param params DroneParamsSOA to print from
 * @param index Drone index to print
 */
void drone_params_print(const DroneParamsSOA* params, uint32_t index);

#ifdef __cplusplus
}
#endif

#endif /* DRONE_STATE_H */
