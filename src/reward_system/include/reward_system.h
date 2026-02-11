/**
 * Reward System Module - Task-Specific Reward Computation and Termination Conditions
 *
 * Provides SIMD-optimized reward computation for various drone tasks:
 * - HOVER: Maintain position at target
 * - RACE: Navigate through gates
 * - TRACK: Follow moving target
 * - LAND: Soft landing
 * - FORMATION: Multi-agent formation
 * - EXPLORE: Maximize coverage
 * - CUSTOM: User-defined rewards
 *
 * Performance Targets (1024 drones):
 * - Reward compute: <0.5ms
 * - Gate crossing check (10 gates): <0.2ms
 * - Termination check: <0.1ms
 * - Total reward frame: <1ms
 *
 * Memory Budget (1024 drones, 10 gates): ~85 KB
 * - TargetSOA arrays: 28 KB
 * - GateSOA arrays: ~15 KB
 * - Previous state tracking: 20 KB
 * - Episode tracking: 16 KB
 * - TerminationFlags: 6 KB
 *
 * Dependencies:
 * - foundation: Vec3, Arena, SIMD macros, PCG32, math utilities
 * - drone_state: DroneStateSOA, DroneParamsSOA
 * - collision_system: CollisionResults
 */

#ifndef REWARD_SYSTEM_H
#define REWARD_SYSTEM_H

#include "foundation.h"
#include "drone_state.h"
#include "collision_system.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Section 1: Task Type Enumeration
 * ============================================================================ */

/**
 * Task types for different drone RL objectives.
 */
typedef enum TaskType {
    TASK_HOVER,      /**< Maintain position at target */
    TASK_RACE,       /**< Navigate through gates */
    TASK_TRACK,      /**< Follow moving target */
    TASK_LAND,       /**< Soft landing */
    TASK_FORMATION,  /**< Multi-agent formation */
    TASK_EXPLORE,    /**< Maximize coverage */
    TASK_CUSTOM,     /**< User-defined */
    TASK_TYPE_COUNT
} TaskType;

/**
 * Get string name for a task type.
 *
 * @param type Task type
 * @return String name (e.g., "HOVER", "RACE")
 */
const char* task_type_name(TaskType type);

/* ============================================================================
 * Section 2: AoS Helper Structures
 * ============================================================================ */

/**
 * Single target point (for HOVER, LAND tasks).
 */
typedef struct TargetPoint {
    Vec3 position;     /**< Target position in world frame */
    Vec3 velocity;     /**< Target velocity (usually zero for hover) */
    float radius;      /**< Success radius (meters) */
} TargetPoint;

/**
 * Target trajectory (for TRACK task).
 */
typedef struct TargetTrajectory {
    Vec3* positions;       /**< [num_points] waypoints */
    Vec3* velocities;      /**< [num_points] target velocities */
    float* timestamps;     /**< [num_points] time for each waypoint */
    uint32_t num_points;   /**< Number of waypoints */
    uint32_t current_idx;  /**< Current waypoint index */
    bool loop;             /**< Loop back to start */
} TargetTrajectory;

/**
 * Racing gate (for RACE task).
 */
typedef struct RaceGate {
    Vec3 center;       /**< Gate center position */
    Vec3 normal;       /**< Gate facing direction (unit vector) */
    float radius;      /**< Gate opening radius */
    bool passed;       /**< Whether gate has been passed */
} RaceGate;

/**
 * Formation target (for FORMATION task).
 */
typedef struct FormationTarget {
    Vec3* relative_positions;  /**< [num_drones] relative to leader */
    uint32_t leader_idx;       /**< Index of leader drone */
    float separation_dist;     /**< Minimum separation distance */
} FormationTarget;

/**
 * Episode statistics for a single drone.
 */
typedef struct EpisodeStats {
    float episode_return;      /**< Cumulative reward this episode */
    float best_distance;       /**< Best distance achieved */
    uint32_t episode_length;   /**< Steps in current episode */
    uint32_t gates_passed;     /**< Number of gates passed (for RACE) */
    bool success;              /**< Episode ended in success */
} EpisodeStats;

/* ============================================================================
 * Section 3: RewardConfig Structure
 * ============================================================================ */

/**
 * Reward configuration parameters.
 *
 * Contains all tunable reward function parameters.
 * Size: ~140 bytes
 */
typedef struct RewardConfig {
    /* Task identification */
    TaskType task_type;        /**< Type of task */

    /* Distance rewards */
    float distance_scale;      /**< Scale factor for distance penalty */
    float distance_exp;        /**< Exponent for distance (1.0 = linear, 2.0 = quadratic) */
    float reach_bonus;         /**< Bonus for reaching target */
    float reach_radius;        /**< Radius to consider target reached */

    /* Velocity rewards */
    float velocity_match_scale;   /**< Scale for velocity matching (TRACK task) */
    float max_velocity_penalty;   /**< Penalty for exceeding max velocity */

    /* Orientation rewards */
    float uprightness_scale;   /**< Scale for upright orientation bonus */
    float heading_scale;       /**< Scale for heading alignment (RACE/TRACK) */

    /* Energy/efficiency rewards */
    float energy_scale;        /**< Scale for energy penalty */
    float jerk_scale;          /**< Scale for action smoothness penalty */

    /* Collision penalties */
    float collision_penalty;         /**< Base collision penalty */
    float world_collision_penalty;   /**< Additional penalty for world collision */
    float drone_collision_penalty;   /**< Additional penalty for drone-drone collision */

    /* Survival rewards */
    float alive_bonus;         /**< Per-step bonus for staying alive */
    float success_bonus;       /**< Bonus for completing task */

    /* Reward clipping */
    float reward_min;          /**< Minimum reward value */
    float reward_max;          /**< Maximum reward value */

    /* Task-specific parameters */
    float gate_pass_bonus;          /**< Bonus for passing a gate (RACE) */
    float landing_velocity_scale;   /**< Scale for soft landing (LAND) */
    float formation_position_scale; /**< Scale for formation position error */
    float exploration_coverage_scale; /**< Scale for exploration coverage */

    /* Progress rewards */
    float progress_scale;      /**< Scale for progress toward goal */
    float delta_distance_scale; /**< Scale for distance improvement */

    /* Padding for alignment */
    float _pad[2];
} RewardConfig;

/* ============================================================================
 * Section 4: TargetSOA Structure
 * ============================================================================ */

/**
 * Target positions in Structure-of-Arrays layout.
 *
 * All arrays are 32-byte aligned for AVX2 operations.
 * Memory per drone: 7 floats × 4 bytes = 28 bytes
 */
typedef struct TargetSOA {
    /* Position (world frame) */
    float* target_x;      /**< [capacity] Target X position */
    float* target_y;      /**< [capacity] Target Y position */
    float* target_z;      /**< [capacity] Target Z position */

    /* Velocity (for moving targets) */
    float* target_vx;     /**< [capacity] Target X velocity */
    float* target_vy;     /**< [capacity] Target Y velocity */
    float* target_vz;     /**< [capacity] Target Z velocity */

    /* Success radius */
    float* target_radius; /**< [capacity] Success radius per target */

    /* Metadata */
    uint32_t capacity;    /**< Maximum number of targets */
    uint32_t count;       /**< Active target count */
} TargetSOA;

/* ============================================================================
 * Section 5: GateSOA Structure
 * ============================================================================ */

/**
 * Racing gates in Structure-of-Arrays layout.
 *
 * For RACE task with multiple gates per course.
 * Memory: ~15 KB for 1024 drones, 10 gates
 */
typedef struct GateSOA {
    /* Gate center positions */
    float* center_x;      /**< [num_gates] Gate center X */
    float* center_y;      /**< [num_gates] Gate center Y */
    float* center_z;      /**< [num_gates] Gate center Z */

    /* Gate normals (facing direction) */
    float* normal_x;      /**< [num_gates] Gate normal X */
    float* normal_y;      /**< [num_gates] Gate normal Y */
    float* normal_z;      /**< [num_gates] Gate normal Z */

    /* Gate radii */
    float* radius;        /**< [num_gates] Gate opening radius */

    /* Per-drone progress tracking */
    uint8_t* passed;      /**< [max_drones * num_gates] Passed flags bitmap */
    uint32_t* current_gate; /**< [max_drones] Current gate index per drone */

    /* Metadata */
    uint32_t num_gates;   /**< Number of gates in course */
    uint32_t max_drones;  /**< Maximum drones tracked */
} GateSOA;

/* ============================================================================
 * Section 6: TerminationFlags Structure
 * ============================================================================ */

/**
 * Termination condition flags in SoA layout.
 *
 * All arrays are uint8_t for compact storage.
 * Memory: 6 KB for 1024 drones
 */
typedef struct TerminationFlags {
    uint8_t* done;         /**< [capacity] Episode done flag */
    uint8_t* truncated;    /**< [capacity] Episode truncated (time limit) */
    uint8_t* success;      /**< [capacity] Episode ended in success */
    uint8_t* collision;    /**< [capacity] Terminated due to collision */
    uint8_t* out_of_bounds; /**< [capacity] Terminated due to out of bounds */
    uint8_t* timeout;      /**< [capacity] Terminated due to timeout */

    uint32_t capacity;     /**< Maximum number of drones */
} TerminationFlags;

/* ============================================================================
 * Section 7: RewardSystem Main Structure
 * ============================================================================ */

/**
 * Main reward system structure.
 *
 * Manages targets, gates, episode tracking, and termination conditions
 * for computing task-specific rewards during RL training.
 */
typedef struct RewardSystem {
    /* Configuration */
    RewardConfig config;           /**< Reward function parameters */

    /* Target management */
    TargetSOA* targets;            /**< Per-drone targets */

    /* Gate management (for RACE task) */
    GateSOA* gates;                /**< Racing gates (NULL if not racing) */

    /* Previous state tracking (for delta rewards) */
    float* prev_distance;          /**< [max_drones] Previous distance to target */
    float* prev_actions;           /**< [max_drones * 4] Previous actions (for jerk) */

    /* Episode tracking */
    float* episode_return;         /**< [max_drones] Cumulative reward */
    uint32_t* episode_length;      /**< [max_drones] Steps in episode */
    uint32_t* gates_passed;        /**< [max_drones] Gates passed count */
    float* best_distance;          /**< [max_drones] Best distance achieved */

    /* Termination conditions */
    TerminationFlags* termination; /**< Termination flags */

    /* Memory management */
    Arena* arena;                  /**< Arena allocator (reference) */
    uint32_t max_drones;           /**< Maximum drone capacity */
    uint32_t max_gates;            /**< Maximum gates (0 if not racing) */
} RewardSystem;

/* ============================================================================
 * Section 8: Lifecycle Functions
 * ============================================================================ */

/**
 * Create a reward system from arena allocator.
 *
 * Allocates all arrays with 32-byte alignment for SIMD operations.
 *
 * @param arena Arena allocator to use
 * @param config Reward configuration (NULL for default based on task)
 * @param max_drones Maximum number of drones
 * @param max_gates Maximum number of gates (0 for non-racing tasks)
 * @return Pointer to RewardSystem, or NULL on failure
 */
RewardSystem* reward_create(Arena* arena, const RewardConfig* config,
                            uint32_t max_drones, uint32_t max_gates);

/**
 * Destroy a reward system.
 *
 * No-op when using arena allocation (memory freed with arena).
 *
 * @param sys System to destroy (can be NULL)
 */
void reward_destroy(RewardSystem* sys);

/**
 * Reset a single drone's reward state.
 *
 * Clears episode tracking, resets termination flags, clears gate progress.
 *
 * @param sys Reward system
 * @param drone_idx Drone index to reset
 */
void reward_reset(RewardSystem* sys, uint32_t drone_idx);

/**
 * Reset multiple drones' reward states.
 *
 * Batch version of reward_reset for efficiency.
 *
 * @param sys Reward system
 * @param indices Array of drone indices to reset
 * @param count Number of indices
 */
void reward_reset_batch(RewardSystem* sys, const uint32_t* indices, uint32_t count);

/* ============================================================================
 * Section 9: Target Management Functions
 * ============================================================================ */

/**
 * Set target for a single drone.
 *
 * @param sys Reward system
 * @param drone_idx Drone index
 * @param position Target position
 * @param velocity Target velocity (usually zero)
 * @param radius Success radius
 */
void reward_set_target(RewardSystem* sys, uint32_t drone_idx,
                       Vec3 position, Vec3 velocity, float radius);

/**
 * Set random targets for multiple drones.
 *
 * Generates random positions within bounds.
 *
 * @param sys Reward system
 * @param count Number of drones to set targets for
 * @param bounds_min Minimum bounds
 * @param bounds_max Maximum bounds
 * @param rng Random number generator
 */
void reward_set_targets_random(RewardSystem* sys, uint32_t count,
                               Vec3 bounds_min, Vec3 bounds_max, PCG32* rng);

/**
 * Update targets for moving target tracking (TRACK task).
 *
 * Advances targets along their trajectories.
 *
 * @param sys Reward system
 * @param dt Time step (seconds)
 */
void reward_update_targets(RewardSystem* sys, float dt);

/**
 * Set racing gates.
 *
 * @param sys Reward system
 * @param centers Array of gate centers [num_gates]
 * @param normals Array of gate normals [num_gates]
 * @param radii Array of gate radii [num_gates]
 * @param num_gates Number of gates
 */
void reward_set_gates(RewardSystem* sys, const Vec3* centers,
                      const Vec3* normals, const float* radii, uint32_t num_gates);

/**
 * Reset gate progress for a drone.
 *
 * @param sys Reward system
 * @param drone_idx Drone index
 */
void reward_reset_gates(RewardSystem* sys, uint32_t drone_idx);

/* ============================================================================
 * Section 10: Reward Computation Functions
 * ============================================================================ */

/**
 * Compute rewards for all drones.
 *
 * Dispatches to task-specific reward function based on config.
 * SIMD-optimized for batch processing.
 *
 * @param sys Reward system
 * @param states Drone state arrays
 * @param params Drone parameter arrays
 * @param actions Current actions [count * 4]
 * @param collisions Collision detection results
 * @param rewards Output: computed rewards [count]
 * @param count Number of drones
 */
void reward_compute(RewardSystem* sys, const DroneStateSOA* states,
                    const DroneParamsSOA* params, const float* actions,
                    const CollisionResults* collisions,
                    float* rewards, uint32_t count);

/**
 * Compute hover task rewards.
 *
 * Rewards maintaining position at target with upright orientation.
 *
 * @param sys Reward system
 * @param states Drone state arrays
 * @param params Drone parameter arrays
 * @param actions Current actions
 * @param collisions Collision results
 * @param rewards Output rewards
 * @param count Number of drones
 */
void reward_compute_hover(RewardSystem* sys, const DroneStateSOA* states,
                          const DroneParamsSOA* params, const float* actions,
                          const CollisionResults* collisions,
                          float* rewards, uint32_t count);

/**
 * Compute race task rewards.
 *
 * Rewards passing gates in order, penalizes collisions.
 *
 * @param sys Reward system
 * @param states Drone state arrays
 * @param params Drone parameter arrays
 * @param actions Current actions
 * @param collisions Collision results
 * @param rewards Output rewards
 * @param count Number of drones
 */
void reward_compute_race(RewardSystem* sys, const DroneStateSOA* states,
                         const DroneParamsSOA* params, const float* actions,
                         const CollisionResults* collisions,
                         float* rewards, uint32_t count);

/**
 * Compute track task rewards.
 *
 * Rewards following moving target with velocity matching.
 *
 * @param sys Reward system
 * @param states Drone state arrays
 * @param params Drone parameter arrays
 * @param actions Current actions
 * @param collisions Collision results
 * @param rewards Output rewards
 * @param count Number of drones
 */
void reward_compute_track(RewardSystem* sys, const DroneStateSOA* states,
                          const DroneParamsSOA* params, const float* actions,
                          const CollisionResults* collisions,
                          float* rewards, uint32_t count);

/**
 * Compute land task rewards.
 *
 * Rewards soft landing with low velocity at touchdown.
 *
 * @param sys Reward system
 * @param states Drone state arrays
 * @param params Drone parameter arrays
 * @param actions Current actions
 * @param collisions Collision results
 * @param rewards Output rewards
 * @param count Number of drones
 */
void reward_compute_land(RewardSystem* sys, const DroneStateSOA* states,
                         const DroneParamsSOA* params, const float* actions,
                         const CollisionResults* collisions,
                         float* rewards, uint32_t count);

/**
 * Compute formation task rewards.
 *
 * Rewards maintaining relative positions in formation.
 *
 * @param sys Reward system
 * @param states Drone state arrays
 * @param params Drone parameter arrays
 * @param actions Current actions
 * @param collisions Collision results
 * @param rewards Output rewards
 * @param count Number of drones
 */
void reward_compute_formation(RewardSystem* sys, const DroneStateSOA* states,
                              const DroneParamsSOA* params, const float* actions,
                              const CollisionResults* collisions,
                              float* rewards, uint32_t count);

/* ============================================================================
 * Section 11: Termination Functions
 * ============================================================================ */

/**
 * Compute termination conditions for all drones.
 *
 * Checks collision, out-of-bounds, timeout, and success conditions.
 *
 * @param sys Reward system
 * @param states Drone state arrays
 * @param collisions Collision detection results
 * @param bounds_min Minimum world bounds
 * @param bounds_max Maximum world bounds
 * @param max_steps Maximum steps before truncation
 * @param flags Output: termination flags
 * @param count Number of drones
 */
void reward_compute_terminations(RewardSystem* sys, const DroneStateSOA* states,
                                 const CollisionResults* collisions,
                                 Vec3 bounds_min, Vec3 bounds_max,
                                 uint32_t max_steps, TerminationFlags* flags,
                                 uint32_t count);

/**
 * Check if a drone's episode is done.
 *
 * @param sys Reward system
 * @param drone_idx Drone index
 * @return true if episode is terminated
 */
bool reward_is_done(const RewardSystem* sys, uint32_t drone_idx);

/**
 * Check if a drone achieved success.
 *
 * @param sys Reward system
 * @param drone_idx Drone index
 * @return true if episode ended in success
 */
bool reward_is_success(const RewardSystem* sys, uint32_t drone_idx);

/* ============================================================================
 * Section 12: Utility Functions
 * ============================================================================ */

/**
 * Compute distance from drone to its target.
 *
 * @param sys Reward system
 * @param states Drone state arrays
 * @param drone_idx Drone index
 * @return Distance to target (meters)
 */
float reward_distance_to_target(const RewardSystem* sys,
                                const DroneStateSOA* states,
                                uint32_t drone_idx);

/**
 * Check if drone has reached its target.
 *
 * @param sys Reward system
 * @param states Drone state arrays
 * @param drone_idx Drone index
 * @return true if within target radius
 */
bool reward_reached_target(const RewardSystem* sys,
                           const DroneStateSOA* states,
                           uint32_t drone_idx);

/**
 * Check if drone crossed a gate.
 *
 * Uses line-segment/plane intersection test.
 *
 * @param sys Reward system
 * @param states Drone state arrays
 * @param drone_idx Drone index
 * @param gate_idx Gate index
 * @param prev_pos Previous position
 * @return true if gate was crossed in correct direction
 */
bool reward_check_gate_crossing(const RewardSystem* sys,
                                const DroneStateSOA* states,
                                uint32_t drone_idx, uint32_t gate_idx,
                                Vec3 prev_pos);

/**
 * Get episode statistics for a drone.
 *
 * @param sys Reward system
 * @param drone_idx Drone index
 * @return Episode statistics
 */
EpisodeStats reward_get_episode_stats(const RewardSystem* sys, uint32_t drone_idx);

/**
 * Get default reward configuration for a task type.
 *
 * @param task Task type
 * @return Default configuration
 */
RewardConfig reward_config_default(TaskType task);

/**
 * Calculate memory required for reward system.
 *
 * @param max_drones Maximum drone count
 * @param max_gates Maximum gate count
 * @return Total bytes required
 */
size_t reward_memory_size(uint32_t max_drones, uint32_t max_gates);

/* ============================================================================
 * Section 13: Type Size Verification
 * ============================================================================ */

/* Verify RewardConfig size at compile time (should be around 140 bytes) */
FOUNDATION_STATIC_ASSERT(sizeof(RewardConfig) <= 160, "RewardConfig too large");

#ifdef __cplusplus
}
#endif

#endif /* REWARD_SYSTEM_H */
