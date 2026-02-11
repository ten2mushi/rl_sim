/**
 * Configuration Module - TOML-Based Configuration Parsing and Validation
 *
 * Provides TOML-based configuration parsing, validation, and conversion for
 * the drone RL engine. Bridges human-readable config files to the SoA data
 * structures used by physics, sensors, and reward systems.
 *
 * Key Features:
 * - TOML parsing with tomlc99 library
 * - Three-phase validation (parse -> schema -> semantic)
 * - Conversion to DroneParamsSOA for physics simulation
 * - FNV-1a hashing for change detection
 * - Default Crazyflie 2.0 parameters
 *
 * Performance Targets:
 * - Config load: <10ms (startup-only)
 * - Config validate: <1ms
 * - Config hash: <100us
 * - Config to params (1024 drones): <1ms
 *
 * Dependencies:
 * - foundation: Arena, FOUNDATION_ASSERT, arena_alloc_aligned
 * - drone_state: DroneParamsSOA, drone_params_create
 *
 * Provides to Environment Manager (09):
 * - Config: Complete configuration struct
 * - config_load/config_load_string: Load from file or string
 * - config_validate: Three-phase validation
 * - drone_config_to_params: Convert to DroneParamsSOA
 */

#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include "foundation.h"
#include "drone_state.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Section 1: Constants
 * ============================================================================ */

/** Maximum length of string fields */
#define CONFIG_NAME_MAX 64
#define CONFIG_PATH_MAX 256
#define CONFIG_ERROR_MSG_MAX 256

/** Maximum number of sensors in a config */
#define CONFIG_MAX_SENSORS 32

/** Maximum number of validation errors returned */
#define CONFIG_MAX_ERRORS 32

/* ============================================================================
 * Section 2: Drone Configuration
 * ============================================================================ */

/**
 * Drone physical parameters (from URDF or config).
 * Maps to DroneParamsSOA for physics simulation.
 *
 * Default values are for Crazyflie 2.0.
 */
typedef struct DroneConfig {
    /* Identity */
    char name[CONFIG_NAME_MAX];          /**< Drone identifier */
    char model_path[CONFIG_PATH_MAX];    /**< Path to URDF/mesh */

    /* Physical properties */
    float mass;                          /**< kg (default: 0.027 Crazyflie 2.0) */
    float arm_length;                    /**< m, motor to center (default: 0.046) */

    /* Inertia tensor (diagonal) */
    float ixx;                           /**< kg*m^2 (default: 1.4e-5) */
    float iyy;                           /**< kg*m^2 (default: 1.4e-5) */
    float izz;                           /**< kg*m^2 (default: 2.17e-5) */

    /* Motor parameters */
    float k_thrust;                      /**< N/(rad/s)^2 (default: 2.88e-8) */
    float k_torque;                      /**< N*m/(rad/s)^2 (default: 7.24e-10) */
    float motor_tau;                     /**< Motor time constant s (default: 0.02) */
    float max_rpm;                       /**< Maximum motor RPM (default: 21702) */

    /* Geometry */
    float collision_radius;              /**< m (default: arm_length + 0.01) */

    /* Aerodynamics */
    float k_drag;                        /**< Linear drag coefficient (default: 0.0) */
    float k_drag_angular;                /**< Angular drag coefficient (default: 0.0) */

    /* Limits */
    float max_velocity;                  /**< m/s (default: 10.0) */
    float max_angular_velocity;          /**< rad/s (default: 20.0) */
    float max_tilt_angle;                /**< rad (default: 1.0) */

    /* Visual */
    float color[3];                      /**< RGB [0,1] (default: [0.2, 0.6, 1.0]) */
    float scale;                         /**< Model scale factor (default: 1.0) */
} DroneConfig;

/* ============================================================================
 * Section 3: Environment Configuration
 * ============================================================================ */

/**
 * Environment configuration.
 */
typedef struct EnvironmentConfig {
    /* Dimensions */
    uint32_t num_envs;                   /**< Number of parallel environments */
    uint32_t drones_per_env;             /**< Drones per environment */

    /* World bounds */
    float world_size[3];                 /**< Width, depth, height (m) */
    float world_origin[3];               /**< Center offset */

    /* Voxel settings */
    float voxel_size;                    /**< m (default: 0.1) */
    uint32_t max_bricks;                 /**< Maximum allocated bricks */

    /* Spawning */
    float spawn_radius;                  /**< Spawn area radius */
    float spawn_height_min;
    float spawn_height_max;
    float min_separation;                /**< Minimum drone separation */

    /* Episode */
    uint32_t max_episode_steps;          /**< (default: 1000) */
    bool auto_reset;                     /**< (default: true) */

    /* World generation */
    char world_type[32];                 /**< "empty", "obstacles", "maze", "race" */
    uint32_t num_obstacles;
    uint32_t seed;                       /**< Random seed (default: 42) */
} EnvironmentConfig;

/* ============================================================================
 * Section 4: Physics Configuration
 * ============================================================================ */

/**
 * Physics configuration.
 */
typedef struct ConfigPhysics {
    float timestep;                      /**< Simulation dt (s) (default: 0.02) */
    uint32_t substeps;                   /**< Physics substeps per frame (default: 4) */
    float gravity;                       /**< m/s^2 positive down (default: 9.81) */

    /* Integration */
    char integrator[16];                 /**< "euler" or "rk4" (default: "rk4") */

    /* Stability */
    float velocity_clamp;                /**< (default: 20.0) */
    float angular_velocity_clamp;        /**< (default: 30.0) */
    bool normalize_quaternions;          /**< (default: true) */

    /* Ground effect */
    bool enable_ground_effect;           /**< (default: true) */
    float ground_effect_height;          /**< (default: 0.5) */
    float ground_effect_strength;        /**< (default: 1.5) */

    /* Domain randomization */
    float dt_variance;                   /**< Timestep variance (default: 0.0) */
    float mass_variance;                 /**< Mass variance (default: 0.0) */
    float thrust_variance;               /**< Thrust variance (default: 0.0) */
} ConfigPhysics;

/* ============================================================================
 * Section 5: Noise Configuration Entries (TOML-facing)
 * ============================================================================ */

/** Maximum noise stages per group in config */
#define CONFIG_MAX_NOISE_STAGES 8

/** Maximum noise groups per sensor in config */
#define CONFIG_MAX_NOISE_GROUPS 4

/** Maximum bias values per constant_bias stage */
#define CONFIG_MAX_NOISE_BIAS_VALUES 8

/**
 * Single noise stage entry (parsed from TOML [[sensors.noise_groups.stages]]).
 */
typedef struct NoiseStageEntry {
    char type[32];            /**< "white_gaussian", "constant_bias", etc. */
    float stddev;             /**< white_gaussian */
    float values[CONFIG_MAX_NOISE_BIAS_VALUES]; /**< constant_bias per-element */
    uint32_t value_count;     /**< number of bias values specified */
    float tau;                /**< bias_drift OU time constant */
    float sigma;              /**< bias_drift OU noise strength */
    float error;              /**< scale_factor */
    float coeff;              /**< distance_dependent coefficient */
    float power;              /**< distance_dependent exponent */
    float step;               /**< quantization step size */
    float probability;        /**< dropout probability */
    float replacement;        /**< dropout replacement value */
    float min_val;            /**< saturation min */
    float max_val;            /**< saturation max */
} NoiseStageEntry;

/**
 * Noise group entry (parsed from TOML [[sensors.noise_groups]]).
 */
typedef struct NoiseGroupEntry {
    uint32_t channels[2];     /**< [start, count], [0,0] = all channels */
    NoiseStageEntry stages[CONFIG_MAX_NOISE_STAGES];
    uint32_t num_stages;
} NoiseGroupEntry;

/* ============================================================================
 * Section 6: Sensor Configuration Entry
 * ============================================================================ */

/**
 * Sensor configuration entry (parsed from TOML [[sensors]] array).
 */
typedef struct SensorConfigEntry {
    char type[32];                       /**< "imu", "tof", "lidar_2d", etc. */
    char name[CONFIG_NAME_MAX];          /**< User-defined name */

    /* Mounting */
    float position[3];                   /**< Offset from drone center */
    float orientation[4];                /**< Quaternion wxyz */

    /* Common */
    float sample_rate;                   /**< Hz (0 = every frame) */

    /* Composable noise pipeline */
    NoiseGroupEntry noise_groups[CONFIG_MAX_NOISE_GROUPS];
    uint32_t num_noise_groups;

    /* Type-specific (ToF/LiDAR) */
    float max_range;
    uint32_t num_rays;
    float fov;                           /**< Horizontal FOV (rad) */
    float fov_vertical;                  /**< Vertical FOV (rad) */
    uint32_t vertical_layers;

    /* Type-specific (Camera) */
    uint32_t width;
    uint32_t height;
    float near_clip;
    float far_clip;
    uint32_t num_classes;                /**< Segmentation classes */

    /* Type-specific (Neighbor sensor) */
    uint32_t k_neighbors;
} SensorConfigEntry;

/* ============================================================================
 * Section 7: Reward Configuration
 * ============================================================================ */

/**
 * Reward configuration (from 07-reward-system).
 */
typedef struct RewardConfigData {
    char task[32];                       /**< "hover", "waypoint", "race", etc. */
    float distance_scale;                /**< Position error weight */
    float distance_exp;                  /**< Distance exponent */
    float reach_bonus;                   /**< Target reach bonus */
    float reach_radius;                  /**< Target reach radius */
    float velocity_match_scale;          /**< Velocity matching weight */
    float uprightness_scale;             /**< Orientation upright weight */
    float energy_scale;                  /**< Energy penalty weight */
    float jerk_scale;                    /**< Jerk penalty weight */
    float collision_penalty;             /**< Generic collision penalty */
    float world_collision_penalty;
    float drone_collision_penalty;
    float alive_bonus;                   /**< Per-step survival bonus */
    float success_bonus;                 /**< Task completion bonus */
    float reward_min;                    /**< Reward clipping min */
    float reward_max;                    /**< Reward clipping max */
} RewardConfigData;

/* ============================================================================
 * Section 7: Training Configuration
 * ============================================================================ */

/**
 * Training configuration (RL hyperparameters).
 */
typedef struct TrainingConfig {
    char algorithm[32];                  /**< "ppo", "sac", etc. */
    float learning_rate;                 /**< (default: 3e-4) */
    float gamma;                         /**< Discount factor (default: 0.99) */
    float gae_lambda;                    /**< GAE parameter (default: 0.95) */
    float clip_range;                    /**< PPO clip range (default: 0.2) */
    float entropy_coef;                  /**< (default: 0.01) */
    float value_coef;                    /**< (default: 0.5) */
    float max_grad_norm;                 /**< (default: 0.5) */
    uint32_t batch_size;                 /**< (default: 2048) */
    uint32_t num_epochs;                 /**< (default: 10) */
    uint32_t rollout_length;             /**< (default: 128) */
    uint32_t log_interval;               /**< (default: 10) */
    uint32_t save_interval;              /**< (default: 100) */
    char checkpoint_dir[CONFIG_PATH_MAX]; /**< (default: "checkpoints") */
} TrainingConfig;

/* ============================================================================
 * Section 8: Complete Configuration
 * ============================================================================ */

/**
 * Complete configuration aggregating all subsystems.
 */
typedef struct Config {
    DroneConfig drone;
    EnvironmentConfig environment;
    ConfigPhysics physics;
    RewardConfigData reward;
    TrainingConfig training;

    /* Sensor array (variable length, dynamically allocated) */
    SensorConfigEntry* sensors;
    uint32_t num_sensors;

    /* Metadata */
    char config_path[CONFIG_PATH_MAX];   /**< Source file path */
    uint64_t config_hash;                /**< FNV-1a hash for change detection */
} Config;

/* ============================================================================
 * Section 9: Configuration Error
 * ============================================================================ */

/**
 * Configuration error for validation reporting.
 */
typedef struct ConfigError {
    char field[CONFIG_NAME_MAX];         /**< Field path e.g., "drone.mass" */
    char message[CONFIG_ERROR_MSG_MAX];  /**< Error description */
    int line_number;                     /**< TOML line number (-1 if unknown) */
} ConfigError;

/* ============================================================================
 * Section 10: Parsing API
 * ============================================================================ */

/**
 * Parse complete configuration from TOML file.
 * Sets defaults first, then overrides with file values.
 *
 * @param path Path to TOML configuration file
 * @param config Output config structure (caller allocated)
 * @param error_msg Buffer for error message (at least 256 bytes)
 * @return 0 on success, negative error code on failure
 */
int config_load(const char* path, Config* config, char* error_msg);

/**
 * Parse from TOML string (for embedded configs or testing).
 *
 * @param toml_str TOML content as string
 * @param config Output config structure (caller allocated)
 * @param error_msg Buffer for error message (at least 256 bytes)
 * @return 0 on success, negative error code on failure
 */
int config_load_string(const char* toml_str, Config* config, char* error_msg);

/**
 * Free dynamically allocated config memory (sensors array).
 *
 * @param config Config to free
 */
void config_free(Config* config);

/* ============================================================================
 * Section 11: Validation API
 * ============================================================================ */

/**
 * Validate configuration (returns number of errors).
 * Three-phase: parse -> schema -> semantic
 *
 * @param config Config to validate
 * @param errors Output array for error details
 * @param max_errors Maximum errors to report
 * @return Number of errors found (0 = valid)
 */
int config_validate(const Config* config, ConfigError* errors, uint32_t max_errors);

/**
 * Validate drone section.
 */
int config_validate_drone(const DroneConfig* config,
                          ConfigError* errors, uint32_t max_errors);

/**
 * Validate environment section.
 */
int config_validate_environment(const EnvironmentConfig* config,
                                ConfigError* errors, uint32_t max_errors);

/**
 * Validate physics section.
 */
int config_validate_physics(const ConfigPhysics* config,
                            ConfigError* errors, uint32_t max_errors);

/**
 * Validate sensors array.
 */
int config_validate_sensors(const SensorConfigEntry* sensors, uint32_t count,
                            ConfigError* errors, uint32_t max_errors);

/* ============================================================================
 * Section 12: Default Values API
 * ============================================================================ */

/**
 * Set all fields to sensible defaults (Crazyflie 2.0 base).
 *
 * @param config Config to initialize
 */
void config_set_defaults(Config* config);

/**
 * Set drone config to Crazyflie 2.0 defaults.
 */
void drone_config_set_defaults(DroneConfig* config);

/**
 * Set environment config to defaults.
 */
void environment_config_set_defaults(EnvironmentConfig* config);

/**
 * Set physics config to defaults.
 */
void physics_config_set_defaults(ConfigPhysics* config);

/**
 * Set reward config to defaults.
 */
void reward_config_data_set_defaults(RewardConfigData* config);

/**
 * Set training config to defaults.
 */
void training_config_set_defaults(TrainingConfig* config);

/**
 * Get default sensor config for type.
 *
 * @param type Sensor type string ("imu", "tof", etc.)
 * @return Default sensor config entry
 */
SensorConfigEntry sensor_config_entry_default(const char* type);

/* ============================================================================
 * Section 13: Serialization API
 * ============================================================================ */

/**
 * Save configuration to TOML file.
 *
 * @param path Output file path
 * @param config Config to save
 * @return 0 on success, negative on error
 */
int config_save(const char* path, const Config* config);

/**
 * Export configuration as JSON (for logging/tracking).
 *
 * @param config Config to export
 * @param buffer Output buffer
 * @param buffer_size Buffer capacity
 * @return 0 on success, -1 if buffer too small
 */
int config_to_json(const Config* config, char* buffer, size_t buffer_size);

/**
 * Compute FNV-1a hash of configuration (for change detection).
 *
 * @param config Config to hash
 * @return 64-bit hash value
 */
uint64_t config_hash(const Config* config);

/* ============================================================================
 * Section 14: Conversion API
 * ============================================================================ */

/**
 * Convert DroneConfig + ConfigPhysics to DroneParamsSOA entries.
 *
 * Maps config fields to SoA arrays, broadcasting to [start_index, start_index+count).
 * Gravity comes from ConfigPhysics since it's an environment property.
 *
 * Field name mappings (config -> SoA):
 *   k_drag_angular -> k_ang_damp
 *   max_velocity -> max_vel
 *   max_angular_velocity -> max_omega
 *
 * @param drone_cfg Drone configuration
 * @param physics_cfg Physics configuration (for gravity)
 * @param params Target DroneParamsSOA
 * @param start_index Starting index in params arrays
 * @param count Number of drones to initialize
 */
void drone_config_to_params(const DroneConfig* drone_cfg,
                            const ConfigPhysics* physics_cfg,
                            DroneParamsSOA* params,
                            uint32_t start_index,
                            uint32_t count);

/**
 * Extract DroneConfig from DroneParamsSOA at specified index.
 * Note: gravity is NOT extracted (belongs to ConfigPhysics).
 *
 * @param params Source DroneParamsSOA
 * @param index Index to extract from
 * @return DroneConfig with extracted values
 */
DroneConfig drone_params_to_config(const DroneParamsSOA* params, uint32_t index);

/**
 * Convenience wrapper to initialize all drones from complete Config.
 *
 * @param config Complete configuration
 * @param params Target DroneParamsSOA
 * @param num_drones Number of drones to initialize
 */
void config_init_drone_params(const Config* config,
                              DroneParamsSOA* params,
                              uint32_t num_drones);

/* ============================================================================
 * Section 15: Utility API
 * ============================================================================ */

/**
 * Print configuration summary to stdout.
 *
 * @param config Config to print
 */
void config_print(const Config* config);

/**
 * Compare two configurations (returns 0 if identical).
 *
 * @param a First config
 * @param b Second config
 * @return 0 if identical, non-zero if different
 */
int config_compare(const Config* a, const Config* b);

/**
 * Clone configuration (deep copy with arena allocation).
 *
 * @param src Source config
 * @param dst Destination config (caller allocated struct)
 * @param arena Arena for sensor array allocation
 */
void config_clone(const Config* src, Config* dst, Arena* arena);

/**
 * Calculate memory required for a config with N sensors.
 *
 * @param num_sensors Number of sensors
 * @return Required bytes
 */
size_t config_memory_size(uint32_t num_sensors);

/* ============================================================================
 * Section 16: Type Size Verification
 * ============================================================================ */

/* Verify reasonable struct sizes */
FOUNDATION_STATIC_ASSERT(sizeof(DroneConfig) < 512, "DroneConfig too large");
FOUNDATION_STATIC_ASSERT(sizeof(EnvironmentConfig) < 256, "EnvironmentConfig too large");
FOUNDATION_STATIC_ASSERT(sizeof(ConfigPhysics) < 128, "ConfigPhysics too large");
FOUNDATION_STATIC_ASSERT(sizeof(SensorConfigEntry) < 4096, "SensorConfigEntry too large");
FOUNDATION_STATIC_ASSERT(sizeof(Config) < 2048, "Config too large");

#ifdef __cplusplus
}
#endif

#endif /* CONFIGURATION_H */
