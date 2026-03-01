/**
 * Environment Manager Module - Top-Level Orchestration Layer
 *
 * The Environment Manager coordinates all subsystems and provides the Python/C API
 * for PufferLib integration. It is the final module (09) that brings together all
 * dependencies (00-08, 10) to create a complete batch drone RL environment.
 *
 * Performance Targets:
 * - <20ms total step for 1024 drones (50 FPS = 51,200 drone-steps/second)
 * - Physics: <5ms
 * - Collision: <1ms
 * - Sensors: <10ms
 * - Rewards: <1ms
 * - Reset: <2ms per drone
 *
 * Memory Budget (1024 drones):
 * - Persistent arena: ~256 MB
 * - Frame arena: ~64 MB
 * - External buffers: ~534 KB (observations, actions, rewards, dones)
 * - Total: <325 MB (well under 4 GB budget)
 *
 * Dependencies:
 * - foundation: Arena, Vec3, Quat, PCG32, SIMD utilities
 * - drone_state: PlatformStateSOA, PlatformParamsSOA
 * - physics: PhysicsSystem, physics_step
 * - world_brick_map: WorldBrickMap, SDF queries
 * - collision_system: CollisionSystem, collision_detect_all
 * - sensor_system: SensorSystem, sensor_system_sample_all
 * - reward_system: RewardSystem, reward_compute
 * - threading: ThreadPool, Scheduler
 * - configuration: Config, config_load
 */

#ifndef ENVIRONMENT_MANAGER_H
#define ENVIRONMENT_MANAGER_H

#include "foundation.h"
#include "drone_state.h"
#include "physics.h"
#include "world_brick_map.h"
#include "collision_system.h"
#include "sensor_system.h"
#include "sensor_implementations.h"
#include "reward_system.h"
#include "threading.h"
#include "platform.h"
#include "platform_quadcopter.h"
/* configuration.h included for TOML bridge (PhysicsConfig name collision
 * resolved by renaming TOML-facing type to ConfigPhysics) */
#include "configuration.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Section 1: Constants
 * ============================================================================ */

/** Default configuration values */
#define ENGINE_DEFAULT_NUM_ENVS             64
#define ENGINE_DEFAULT_AGENTS_PER_ENV       16
#define ENGINE_DEFAULT_TIMESTEP             0.02f
#define ENGINE_DEFAULT_PHYSICS_SUBSTEPS     4
#define ENGINE_DEFAULT_GRAVITY              9.81f
#define ENGINE_DEFAULT_MAX_EPISODE_STEPS    1000
#define ENGINE_DEFAULT_VOXEL_SIZE           0.1f
#define ENGINE_DEFAULT_MAX_BRICKS           10000
#define ENGINE_DEFAULT_PERSISTENT_ARENA_MB  512
#define ENGINE_DEFAULT_FRAME_ARENA_MB       64
#define ENGINE_DEFAULT_SEED                 12345

/* ENGINE_ACTION_DIM removed: use engine->action_dim or config->platform_vtable->action_dim */

/** Error message buffer size */
#define ENGINE_ERROR_MSG_SIZE               256

/* ============================================================================
 * Section 2: Forward Declarations
 * ============================================================================ */

typedef struct EngineConfig EngineConfig;
typedef struct BatchEngine BatchEngine;
typedef struct AgentStateQuery AgentStateQuery;
typedef struct EngineStats EngineStats;
typedef struct PufferEnv PufferEnv;

/* ============================================================================
 * Section 3: EngineConfig Structure
 * ============================================================================ */

/**
 * Engine configuration structure.
 *
 * Contains all parameters needed to create a BatchEngine.
 * Use engine_config_default() to get sensible defaults.
 */
struct EngineConfig {
    /* Environment dimensions */
    uint32_t num_envs;              /**< Number of parallel environments */
    uint32_t agents_per_env;        /**< Drones per environment */
    uint32_t total_agents;          /**< Computed: num_envs * agents_per_env */

    /* World bounds */
    Vec3 world_min;                 /**< Minimum world coordinates */
    Vec3 world_max;                 /**< Maximum world coordinates */
    float voxel_size;               /**< SDF voxel size (meters) */
    uint32_t max_bricks;            /**< Maximum SDF bricks to allocate */

    /* Physics parameters */
    float timestep;                 /**< Simulation timestep (seconds) */
    uint32_t physics_substeps;      /**< Physics substeps per frame */
    float gravity;                  /**< Gravitational acceleration (m/s²) */

    /* Sensor configuration */
    SensorConfig* sensor_configs;   /**< Array of sensor configurations */
    uint32_t num_sensor_configs;    /**< Number of sensor configs */

    /* Reward configuration */
    RewardConfig reward_config;     /**< Reward function parameters */

    /* Episode limits */
    uint32_t max_episode_steps;     /**< Maximum steps before truncation */

    /* Threading */
    uint32_t num_threads;           /**< Worker threads (0 = auto-detect) */

    /* Memory management */
    size_t persistent_arena_size;   /**< Persistent arena size (bytes) */
    size_t frame_arena_size;        /**< Frame arena size (bytes) */

    /* Random seed */
    uint64_t seed;                  /**< Random seed for reproducibility */

    /* Domain randomization */
    float domain_randomization;     /**< Spawn randomization factor [0,1] */

    /* Termination bounds (optional, defaults to world bounds) */
    Vec3 termination_min;           /**< Minimum termination bounds */
    Vec3 termination_max;           /**< Maximum termination bounds */
    bool use_custom_termination;    /**< Use custom termination bounds */

    /* Spawn region (optional, defaults to world bounds with margin) */
    Vec3 spawn_min;                 /**< Minimum spawn region */
    Vec3 spawn_max;                 /**< Maximum spawn region */
    bool use_custom_spawn;          /**< Use custom spawn region */

    /* World bounds override flag */
    bool use_custom_bounds;         /**< Explicit flag for custom world_min/world_max */

    /* OBJ file path (NULL = empty world) */
    const char* obj_path;           /**< Path to .obj file for world geometry */
    bool use_gpu_voxelization;      /**< Use GPU for Phase 3 voxelization (default: true) */

    /* Physics tunables */
    float air_density;              /**< Air density kg/m^3 (default: 1.225) */
    bool enable_drag;               /**< Enable aerodynamic drag (default: true) */
    bool enable_ground_effect;      /**< Enable SDF-proximity ground effect (default: true) */
    bool enable_motor_dynamics;     /**< Enable first-order motor lag (default: true) */
    bool enable_gyroscopic;         /**< Enable gyroscopic precession (default: false) */
    float ground_effect_height;     /**< SDF distance for full effect (default: 0.5) */
    float ground_effect_coeff;      /**< Maximum thrust multiplier (default: 1.15) */
    float max_linear_accel;         /**< Clamp linear acceleration m/s^2 (default: 100.0) */
    float max_angular_accel;        /**< Clamp angular acceleration rad/s^2 (default: 200.0) */

    /* Collision tunables */
    float collision_radius;             /**< Drone collision radius (default: 0.1) */
    float collision_cell_size;      /**< Spatial hash cell size (default: 1.0) */

    /* Platform vtable */
    const PlatformVTable* platform_vtable; /**< Platform vtable for forces/torques */

    /* Debug options */
    bool enable_profiling;          /**< Enable detailed timing */
    bool verbose_logging;           /**< Enable verbose output */
};

/* ============================================================================
 * Section 4: BatchEngine Structure
 * ============================================================================ */

/**
 * Main batch drone engine structure.
 *
 * Contains all subsystems, state storage, and external buffers for
 * high-performance batch simulation of multiple drone environments.
 */
struct BatchEngine {
    /* Configuration */
    EngineConfig config;

    /* Subsystems */
    PhysicsSystem* physics;         /**< RK4 physics integration */
    WorldBrickMap* world;           /**< Sparse SDF world representation */
    CollisionSystem* collision;     /**< Spatial hash collision detection */
    SensorSystem* sensors;          /**< Polymorphic sensor system */
    RewardSystem* rewards;          /**< Task-specific rewards */
    Scheduler* scheduler;           /**< Hybrid work scheduler */
    ThreadPool* thread_pool;        /**< Work-stealing thread pool */
    struct GpuSensorContext* gpu_sensor_ctx; /**< GPU sensor context (nullable) */

    /* State storage (SoA layout) */
    PlatformStateSOA* states;          /**< Drone positions, velocities, etc. */
    PlatformParamsSOA* params;         /**< Drone physical parameters */

    /* Episode tracking */
    float* episode_returns;         /**< [total_agents] Cumulative rewards */
    uint32_t* episode_lengths;      /**< [total_agents] Current episode length */
    uint32_t* env_ids;              /**< [total_agents] Environment ID per drone */

    /* Memory management */
    Arena* persistent_arena;        /**< Long-lived allocations */
    Arena* frame_arena;             /**< Per-step scratch allocations */

    /* External buffers (zero-copy for Python/numpy) */
    float* observations;            /**< [total_agents × obs_dim] 32-byte aligned */
    float* actions;                 /**< [total_agents × action_dim] 32-byte aligned */
    float* rewards_buffer;          /**< [total_agents] 32-byte aligned */
    uint8_t* dones;                 /**< [total_agents] 32-byte aligned */
    uint8_t* truncations;           /**< [total_agents] 32-byte aligned */

    /* Detailed termination flags (for analysis/debugging) */
    uint8_t* term_success;          /**< [total_agents] Task completed successfully */
    uint8_t* term_collision;        /**< [total_agents] Terminated by collision */
    uint8_t* term_out_of_bounds;    /**< [total_agents] Terminated by out of bounds */
    uint8_t* term_timeout;          /**< [total_agents] Terminated by timeout */

    /* Buffer dimensions */
    uint32_t obs_dim;               /**< Observation dimension per drone */
    uint32_t action_dim;            /**< Action dimension per drone (4 for quadcopter) */

    /* Statistics */
    uint64_t total_steps;           /**< Total steps executed */
    uint64_t total_episodes;        /**< Total episodes completed */
    double physics_time_ms;         /**< Last physics step time */
    double collision_time_ms;       /**< Last collision step time */
    double sensor_time_ms;          /**< Last sensor step time */
    double reward_time_ms;          /**< Last reward step time */
    double gpu_sensor_time_ms;      /**< Last GPU sensor dispatch+wait time */
    double reset_time_ms;           /**< Last reset time */

    /* Random number generator */
    PCG32 rng;                      /**< For spawn randomization */

    /* State flags */
    bool initialized;               /**< Engine is initialized */
    bool needs_reset;               /**< Engine needs reset before step */
};

/* ============================================================================
 * Section 5: AgentStateQuery Structure
 * ============================================================================ */

/**
 * Query result for single drone state.
 *
 * Gathers data from SoA arrays into a convenient struct for
 * debugging, visualization, or single-drone queries.
 */
struct AgentStateQuery {
    Vec3 position;                  /**< World position */
    Vec3 velocity;                  /**< Linear velocity */
    Quat orientation;               /**< Orientation quaternion */
    Vec3 angular_velocity;          /**< Angular velocity (body frame) */
    uint32_t env_id;                /**< Environment index */
    uint32_t agent_id;              /**< Drone index within environment */
    bool is_done;                   /**< Episode terminated */
    bool is_truncated;              /**< Episode truncated */
};

/* ============================================================================
 * Section 6: EngineStats Structure
 * ============================================================================ */

/**
 * Engine performance and episode statistics.
 */
struct EngineStats {
    /* Timing (milliseconds) */
    double physics_time_ms;         /**< Physics integration time */
    double collision_time_ms;       /**< Collision detection time */
    double sensor_time_ms;          /**< Sensor sampling time */
    double reward_time_ms;          /**< Reward computation time */
    double reset_time_ms;           /**< Reset/spawn time */
    double avg_step_time_ms;        /**< Total step time */

    /* Counts */
    uint64_t total_steps;           /**< Steps executed since reset */
    uint64_t total_episodes;        /**< Episodes completed since reset */

    /* Episode averages (computed across all drones) */
    float avg_episode_return;       /**< Average episode return */
    float avg_episode_length;       /**< Average episode length */

    /* Performance metrics */
    double steps_per_second;        /**< Steps per second (engine) */
    double drones_per_second;       /**< Drone-steps per second */

    /* Memory usage (bytes) */
    size_t persistent_memory_used;  /**< Persistent arena usage */
    size_t frame_memory_used;       /**< Frame arena usage (peak) */
};

/* ============================================================================
 * Section 7: PufferEnv Structure
 * ============================================================================ */

/**
 * PufferLib environment wrapper.
 *
 * Provides a standard RL environment interface compatible with PufferLib.
 * All buffer pointers alias the engine's internal buffers for zero-copy access.
 */
struct PufferEnv {
    BatchEngine* engine;       /**< Underlying engine */

    /* Buffer aliases (zero-copy) */
    float* observations;            /**< [num_envs × num_agents × obs_size] */
    float* actions;                 /**< [num_envs × num_agents × action_size] */
    float* rewards;                 /**< [num_envs × num_agents] */
    uint8_t* terminals;             /**< [num_envs × num_agents] */
    uint8_t* truncations;           /**< [num_envs × num_agents] */

    /* Dimensions */
    int num_envs;                   /**< Number of environments */
    int num_agents;                 /**< Agents per environment */
    int obs_size;                   /**< Observation size per agent */
    int action_size;                /**< Action size per agent */

    /* Metadata */
    const char* name;               /**< Environment name */
    const char* version;            /**< Version string */
};

/* ============================================================================
 * Section 8: Configuration Functions
 * ============================================================================ */

/**
 * Get default engine configuration.
 *
 * Returns sensible defaults:
 * - 64 envs × 16 drones = 1024 total drones
 * - World: (-50,-50,-10) to (50,50,50)
 * - Timestep: 0.02s (50 Hz) with 4 substeps
 * - No sensors configured (add with engine_config_add_sensor)
 *
 * @return EngineConfig with default values
 */
EngineConfig engine_config_default(void);

/**
 * Validate engine configuration.
 *
 * Checks all constraints:
 * - num_envs > 0, agents_per_env > 0
 * - world_min < world_max
 * - timestep > 0, gravity > 0
 * - arena sizes sufficient
 *
 * @param config Configuration to validate
 * @param error_msg Output buffer for error message (ENGINE_ERROR_MSG_SIZE)
 * @return 0 on success, negative error code on failure
 */
int engine_config_validate(const EngineConfig* config, char* error_msg);

/**
 * Load engine configuration from TOML file.
 *
 * Uses the configuration module to parse the file and populate
 * the EngineConfig structure.
 *
 * @param path Path to TOML configuration file
 * @param config Output configuration (caller provides)
 * @param error_msg Output buffer for error message
 * @return 0 on success, negative error code on failure
 */
int engine_config_load(const char* path, EngineConfig* config, char* error_msg);

/**
 * Add a sensor configuration to the engine config.
 *
 * @param config Engine configuration to modify
 * @param sensor_config Sensor configuration to add
 * @return 0 on success, -1 if max sensors exceeded
 */
int engine_config_add_sensor(EngineConfig* config, const SensorConfig* sensor_config);

/* ============================================================================
 * Section 9: Lifecycle Functions
 * ============================================================================ */

/**
 * Create a new batch drone engine.
 *
 * Allocates all subsystems in dependency order:
 * 1. Thread pool and scheduler
 * 2. Drone state and parameter arrays
 * 3. World brick map
 * 4. Physics system
 * 5. Collision system
 * 6. Sensor system
 * 7. Reward system
 * 8. Episode tracking arrays
 * 9. External buffers (32-byte aligned)
 *
 * @param config Engine configuration
 * @param error_msg Output buffer for error message (ENGINE_ERROR_MSG_SIZE)
 * @return New engine, or NULL on failure
 */
BatchEngine* engine_create(const EngineConfig* config, char* error_msg);

/**
 * Destroy a batch drone engine.
 *
 * Cleans up all subsystems and frees arenas.
 * Safe to call with NULL.
 *
 * @param engine Engine to destroy
 */
void engine_destroy(BatchEngine* engine);

/**
 * Check if engine is valid and ready for use.
 *
 * @param engine Engine to check
 * @return true if engine is initialized and valid
 */
bool engine_is_valid(const BatchEngine* engine);

/* ============================================================================
 * Section 10: Step Functions
 * ============================================================================ */

/**
 * Advance simulation by one timestep with auto-reset.
 *
 * Executes the full step pipeline:
 * 1. Reset frame arena
 * 2. Physics integration (RK4)
 * 3. Collision detection and response
 * 4. Sensor sampling
 * 5. Reward computation
 * 6. Termination checking
 * 7. Episode tracking update
 * 8. Auto-reset terminated environments
 *
 * @param engine Engine to step
 */
void engine_step(BatchEngine* engine);

/**
 * Advance simulation without auto-resetting terminated drones.
 *
 * Same as engine_step but leaves done/truncated drones in terminal state.
 * Useful for manual reset control.
 *
 * @param engine Engine to step
 */
void engine_step_no_reset(BatchEngine* engine);

/**
 * Execute only the physics phase.
 *
 * For debugging or custom step pipelines.
 *
 * @param engine Engine to step
 */
void engine_step_physics(BatchEngine* engine);

/**
 * Execute only the collision phase.
 *
 * @param engine Engine to step
 */
void engine_step_collision(BatchEngine* engine);

/**
 * Execute only the sensor phase.
 *
 * @param engine Engine to step
 */
void engine_step_sensors(BatchEngine* engine);

/**
 * Execute only the reward phase.
 *
 * @param engine Engine to step
 */
void engine_step_rewards(BatchEngine* engine);

/* ============================================================================
 * Section 11: Reset Functions
 * ============================================================================ */

/**
 * Reset all environments to initial state.
 *
 * - Spawns all drones at randomized positions
 * - Zeros velocities and RPMs
 * - Sets identity orientations
 * - Clears done/truncation flags
 * - Resets episode tracking
 * - Computes initial observations
 *
 * @param engine Engine to reset
 */
void engine_reset(BatchEngine* engine);

/**
 * Reset specific environments.
 *
 * Only resets drones in the specified environment indices.
 *
 * @param engine Engine to reset
 * @param env_indices Array of environment indices to reset
 * @param count Number of environments to reset
 */
void engine_reset_envs(BatchEngine* engine, const uint32_t* env_indices, uint32_t count);

/**
 * Reset a single drone.
 *
 * @param engine Engine
 * @param agent_idx Global drone index to reset
 * @param position Spawn position
 * @param orientation Spawn orientation
 */
void engine_reset_agent(BatchEngine* engine, uint32_t agent_idx,
                        Vec3 position, Quat orientation);

/**
 * Reset terminated drones (internal, called by engine_step).
 *
 * Finds all done/truncated drones and resets them.
 *
 * @param engine Engine
 */
void engine_step_reset_terminated(BatchEngine* engine);

/* ============================================================================
 * Section 12: Buffer Access Functions
 * ============================================================================ */

/**
 * Get observation buffer pointer.
 *
 * Layout: [total_agents × obs_dim] contiguous float32
 * 32-byte aligned for SIMD operations.
 *
 * @param engine Engine
 * @return Pointer to observation buffer
 */
float* engine_get_observations(BatchEngine* engine);

/**
 * Get action buffer pointer.
 *
 * Layout: [total_agents × action_dim] contiguous float32
 * Python/numpy writes actions here before calling engine_step.
 *
 * @param engine Engine
 * @return Pointer to action buffer
 */
float* engine_get_actions(BatchEngine* engine);

/**
 * Get rewards buffer pointer.
 *
 * Layout: [total_agents] contiguous float32
 *
 * @param engine Engine
 * @return Pointer to rewards buffer
 */
float* engine_get_rewards(BatchEngine* engine);

/**
 * Get done flags buffer pointer.
 *
 * Layout: [total_agents] contiguous uint8
 * 1 = episode terminated, 0 = ongoing
 *
 * @param engine Engine
 * @return Pointer to dones buffer
 */
uint8_t* engine_get_dones(BatchEngine* engine);

/**
 * Get truncation flags buffer pointer.
 *
 * Layout: [total_agents] contiguous uint8
 * 1 = episode truncated (timeout), 0 = not truncated
 *
 * @param engine Engine
 * @return Pointer to truncations buffer
 */
uint8_t* engine_get_truncations(BatchEngine* engine);

/* ============================================================================
 * Section 13: Dimension Getters
 * ============================================================================ */

/**
 * Get number of environments.
 */
uint32_t engine_get_num_envs(const BatchEngine* engine);

/**
 * Get drones per environment.
 */
uint32_t engine_get_agents_per_env(const BatchEngine* engine);

/**
 * Get total number of drones.
 */
uint32_t engine_get_total_agents(const BatchEngine* engine);

/**
 * Get observation dimension.
 */
uint32_t engine_get_obs_dim(const BatchEngine* engine);

/**
 * Get action dimension.
 */
uint32_t engine_get_action_dim(const BatchEngine* engine);

/* ============================================================================
 * Section 14: State Query Functions
 * ============================================================================ */

/**
 * Get state of a single drone.
 *
 * Gathers from SoA arrays into AgentStateQuery struct.
 *
 * @param engine Engine
 * @param agent_idx Global drone index
 * @param out Output query result
 */
void engine_get_agent_state(const BatchEngine* engine, uint32_t agent_idx,
                            AgentStateQuery* out);

/**
 * Get all drone positions.
 *
 * Exports positions for rendering or analysis.
 *
 * @param engine Engine
 * @param positions_xyz Output: [total_agents × 3] XYZ positions
 */
void engine_get_all_positions(const BatchEngine* engine, float* positions_xyz);

/**
 * Get all drone orientations.
 *
 * @param engine Engine
 * @param quats_wxyz Output: [total_agents × 4] WXYZ quaternions
 */
void engine_get_all_orientations(const BatchEngine* engine, float* quats_wxyz);

/**
 * Get all drone velocities.
 *
 * @param engine Engine
 * @param velocities_xyz Output: [total_agents × 3] XYZ velocities
 */
void engine_get_all_velocities(const BatchEngine* engine, float* velocities_xyz);

/* ============================================================================
 * Section 15: Statistics Functions
 * ============================================================================ */

/**
 * Get engine statistics.
 *
 * @param engine Engine
 * @param stats Output statistics struct
 */
void engine_get_stats(const BatchEngine* engine, EngineStats* stats);

/**
 * Reset engine statistics counters.
 *
 * Zeros total_steps, total_episodes, and timing accumulators.
 *
 * @param engine Engine
 */
void engine_reset_stats(BatchEngine* engine);

/**
 * Print statistics to stdout.
 *
 * @param engine Engine
 */
void engine_print_stats(const BatchEngine* engine);

/* ============================================================================
 * Section 16: World Manipulation Functions
 * ============================================================================ */

/**
 * Add a box obstacle to the world.
 *
 * @param engine Engine
 * @param min_corner Box minimum corner
 * @param max_corner Box maximum corner
 * @param material Material ID (0 = air, >0 = solid)
 */
void engine_add_box(BatchEngine* engine, Vec3 min_corner, Vec3 max_corner,
                    uint8_t material);

/**
 * Add a sphere obstacle to the world.
 *
 * @param engine Engine
 * @param center Sphere center
 * @param radius Sphere radius
 * @param material Material ID
 */
void engine_add_sphere(BatchEngine* engine, Vec3 center, float radius,
                       uint8_t material);

/**
 * Add a cylinder obstacle to the world (Y-axis aligned).
 *
 * @param engine Engine
 * @param center Cylinder center
 * @param radius Cylinder radius
 * @param half_height Half-height along Y axis
 * @param material Material ID
 */
void engine_add_cylinder(BatchEngine* engine, Vec3 center,
                         float radius, float half_height, uint8_t material);

/**
 * Load world geometry from an OBJ file.
 *
 * Replaces current world with geometry from the OBJ file.
 * Updates world bounds from loaded geometry.
 *
 * @param engine Engine
 * @param obj_path Path to .obj file
 * @return 0 on success, negative error code on failure
 */
int engine_load_obj(BatchEngine* engine, const char* obj_path);

/**
 * Clear all world geometry.
 *
 * @param engine Engine
 */
void engine_clear_world(BatchEngine* engine);

/**
 * Set target position for a drone.
 *
 * @param engine Engine
 * @param agent_idx Drone index
 * @param target Target position
 */
void engine_set_target(BatchEngine* engine, uint32_t agent_idx, Vec3 target);

/**
 * Set target positions for all drones.
 *
 * @param engine Engine
 * @param targets Array of target positions [total_agents]
 */
void engine_set_targets(BatchEngine* engine, const Vec3* targets);

/* ============================================================================
 * Section 17: PufferLib Integration Functions
 * ============================================================================ */

/**
 * Create a PufferEnv wrapper.
 *
 * @param config_path Path to TOML config (NULL for defaults)
 * @return New PufferEnv, or NULL on failure
 */
PufferEnv* puffer_env_create(const char* config_path);

/**
 * Create PufferEnv from existing engine config.
 *
 * @param config Engine configuration
 * @param error_msg Optional buffer for error message (ENGINE_ERROR_MSG_SIZE), or NULL
 * @return New PufferEnv, or NULL on failure
 */
PufferEnv* puffer_env_create_from_config(const EngineConfig* config, char* error_msg);

/**
 * Reset the PufferLib environment.
 *
 * @param env PufferEnv to reset
 */
void puffer_env_reset(PufferEnv* env);

/**
 * Step the PufferLib environment.
 *
 * Actions should be written to env->actions before calling.
 *
 * @param env PufferEnv to step
 */
void puffer_env_step(PufferEnv* env);

/**
 * Close the PufferLib environment.
 *
 * @param env PufferEnv to close
 */
void puffer_env_close(PufferEnv* env);

/**
 * Get observation space shape.
 *
 * @param env PufferEnv
 * @param shape Output: shape array (must have capacity for 4 dims)
 * @param ndim Output: number of dimensions
 */
void puffer_env_get_observation_space(PufferEnv* env, int* shape, int* ndim);

/**
 * Get action space shape.
 *
 * @param env PufferEnv
 * @param shape Output: shape array (must have capacity for 4 dims)
 * @param ndim Output: number of dimensions
 */
void puffer_env_get_action_space(PufferEnv* env, int* shape, int* ndim);

/**
 * Render current state (placeholder for visualization).
 *
 * @param env PufferEnv
 * @param mode Render mode ("human", "rgb_array", etc.)
 */
void puffer_env_render(PufferEnv* env, const char* mode);

/* ============================================================================
 * Section 18: Memory Size Helpers
 * ============================================================================ */

/**
 * Calculate total memory required for an engine.
 *
 * @param config Engine configuration
 * @return Total bytes required
 */
size_t engine_memory_size(const EngineConfig* config);

/**
 * Calculate observation buffer size.
 *
 * @param total_agents Total number of drones
 * @param obs_dim Observation dimension
 * @return Bytes required (32-byte aligned)
 */
size_t engine_observation_buffer_size(uint32_t total_agents, uint32_t obs_dim);

/**
 * Calculate action buffer size.
 *
 * @param total_agents Total number of drones
 * @param action_dim Action dimension
 * @return Bytes required (32-byte aligned)
 */
size_t engine_action_buffer_size(uint32_t total_agents, uint32_t action_dim);

/* ============================================================================
 * Section 19: Utility Functions
 * ============================================================================ */

/**
 * Sync engine config bounds from the current world brick map.
 *
 * Copies world_min/world_max from engine->world into engine->config,
 * and updates termination bounds if not using custom termination.
 * Called after loading OBJ or replacing the world.
 *
 * @param engine Engine with a valid world
 */
void sync_bounds_from_world(BatchEngine* engine);

/**
 * Get high-resolution time in milliseconds.
 *
 * @return Current time in milliseconds
 */
double engine_get_time_ms(void);

/**
 * Convert drone index to (env_id, local_agent_id).
 *
 * @param engine Engine
 * @param agent_idx Global drone index
 * @param env_id Output: environment index
 * @param local_id Output: drone index within environment
 */
void engine_agent_idx_to_env(const BatchEngine* engine, uint32_t agent_idx,
                             uint32_t* env_id, uint32_t* local_id);

/**
 * Convert (env_id, local_agent_id) to global drone index.
 *
 * @param engine Engine
 * @param env_id Environment index
 * @param local_id Drone index within environment
 * @return Global drone index
 */
uint32_t engine_env_to_agent_idx(const BatchEngine* engine,
                                 uint32_t env_id, uint32_t local_id);

/* ============================================================================
 * Section 20: Type Size Verification
 * ============================================================================ */

/* Static assert for ENGINE_ACTION_DIM removed: action_dim is now runtime from vtable */

#ifdef __cplusplus
}
#endif

#endif /* ENVIRONMENT_MANAGER_H */
