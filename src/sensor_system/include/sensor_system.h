/**
 * Sensor System Module - Vtable-Based Polymorphic Sensor Framework
 *
 * Provides a high-performance sensor framework with batch-by-type processing
 * for cache-efficient observation generation. Supports 10 sensor types with
 * SIMD-optimized batch sampling and zero runtime allocation.
 *
 * Key Design Principles:
 * - Single vtable dispatch per sensor TYPE (not per drone) - 3-5x faster
 * - Precomputed ray directions for cameras/LiDAR (32-byte aligned)
 * - Zero-copy observation buffer for Python/numpy interop
 * - Per-sensor PCG32 RNG for deterministic noise
 * - Arena-allocated, no per-frame heap allocations
 *
 * Performance Targets (1024 drones):
 * - Vtable dispatch: <5ns (single pointer indirection)
 * - Drone grouping: <100us (O(n * sensors_per_drone))
 * - Scatter to obs buffer: <200us (memcpy per drone per sensor)
 * - Total system overhead: <500us (excluding sensor batch_sample)
 *
 * Memory Budget (1024 drones, 64 sensors, 128-dim obs): ~873 KB
 *
 * Dependencies:
 * - foundation: Vec3, Quat, Arena, PCG32, SIMD utilities
 * - drone_state: DroneStateSOA
 */

#ifndef SENSOR_SYSTEM_H
#define SENSOR_SYSTEM_H

#include "foundation.h"
#include "drone_state.h"
#include "noise.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Section 1: Forward Declarations
 * ============================================================================ */

/* Forward declarations for optional dependencies */
struct WorldBrickMap;
struct CollisionSystem;
struct GpuSensorContext;

/* Forward declarations for sensor types */
typedef struct Sensor Sensor;
typedef struct SensorConfig SensorConfig;
typedef struct SensorVTable SensorVTable;
typedef struct SensorContext SensorContext;
typedef struct SensorSystem SensorSystem;

/* ============================================================================
 * Section 2: Constants
 * ============================================================================ */

/** Maximum sensors attachable to a single drone */
#define MAX_SENSORS_PER_DRONE 8

/** Default observation buffer alignment for SIMD operations */
#define SENSOR_OBS_ALIGNMENT 32

/* ============================================================================
 * Section 3: Sensor Type Enumeration (10 types)
 * ============================================================================ */

/**
 * Enumeration of all supported sensor types.
 *
 * Each type has a corresponding implementation with specific output format:
 * - IMU: 6 floats (ax, ay, az, gx, gy, gz)
 * - ToF: 1 float (distance)
 * - LiDAR 2D: N floats (distances)
 * - LiDAR 3D: N*M floats (distances)
 * - Camera RGB: W*H*3 floats
 * - Camera Depth: W*H floats (normalized [0,1])
 * - Camera Segmentation: W*H floats (material IDs)
 * - Position: 3 floats (x, y, z) - oracle sensor
 * - Velocity: 6 floats (vx, vy, vz, wx, wy, wz) - oracle sensor
 * - Neighbor: K*4 floats (dx, dy, dz, dist)
 */
typedef enum SensorType {
    SENSOR_TYPE_IMU,                    /* 6 floats: ax, ay, az, gx, gy, gz */
    SENSOR_TYPE_TOF,                    /* 1 float: distance */
    SENSOR_TYPE_LIDAR_2D,               /* N floats: distances */
    SENSOR_TYPE_LIDAR_3D,               /* N*M floats: distances */
    SENSOR_TYPE_CAMERA_RGB,             /* W*H*3 floats: RGB values */
    SENSOR_TYPE_CAMERA_DEPTH,           /* W*H floats: normalized depth [0,1] */
    SENSOR_TYPE_CAMERA_SEGMENTATION,    /* W*H floats: material IDs */
    SENSOR_TYPE_POSITION,               /* 3 floats: x, y, z (oracle) */
    SENSOR_TYPE_VELOCITY,               /* 6 floats: vx, vy, vz, wx, wy, wz (oracle) */
    SENSOR_TYPE_NEIGHBOR,               /* K*4 floats: dx, dy, dz, dist */
    SENSOR_TYPE_COUNT                   /* Total number of sensor types */
} SensorType;

/* ============================================================================
 * Section 4: Sensor Context (passed to batch_sample)
 * ============================================================================ */

/**
 * Context structure passed to sensor batch_sample functions.
 *
 * Contains all information needed to sample observations for a batch of drones.
 * World and collision pointers may be NULL if not needed by the sensor type.
 */
struct SensorContext {
    const DroneStateSOA* drones;        /* Drone state arrays */
    const uint32_t* drone_indices;      /* Which drones to sample */
    uint32_t drone_count;               /* Number of drones in this batch */
    const struct WorldBrickMap* world;  /* World for raymarching (nullable) */
    const struct CollisionSystem* collision; /* For neighbor queries (nullable) */
    Arena* scratch;                     /* Per-frame scratch memory */
    struct GpuSensorContext* gpu;       /* GPU context for accelerated sensors (nullable) */
};

/* ============================================================================
 * Section 5: Sensor VTable (Polymorphic Interface)
 * ============================================================================ */

/**
 * Virtual function table for sensor polymorphism.
 *
 * Each sensor type implements these functions once. The batch_sample function
 * processes ALL drones with this sensor type in a single call, enabling
 * cache-efficient batch processing and minimizing vtable dispatch overhead.
 *
 * Target dispatch overhead: <5ns (single pointer indirection)
 */
struct SensorVTable {
    /** Human-readable sensor name (e.g., "IMU", "ToF", "LiDAR-2D") */
    const char* name;

    /** Sensor type enum for fast type checking */
    SensorType type;

    /**
     * Initialize sensor-specific data.
     * Called once when sensor is created.
     *
     * @param sensor The sensor being initialized
     * @param config Configuration parameters
     * @param arena  Arena for persistent allocations (ray tables, etc.)
     */
    void (*init)(Sensor* sensor, const SensorConfig* config, Arena* arena);

    /**
     * Get the number of floats in sensor output.
     *
     * @param sensor The sensor to query
     * @return Number of floats per drone observation
     */
    size_t (*get_output_size)(const Sensor* sensor);

    /**
     * Get the output data type string.
     * Always returns "float32" for this implementation.
     *
     * @param sensor The sensor to query
     * @return Data type string (always "float32")
     */
    const char* (*get_output_dtype)(const Sensor* sensor);

    /**
     * Get the output tensor shape.
     * Writes shape dimensions to the provided array.
     *
     * @param sensor The sensor to query
     * @param shape  Output array for shape dimensions (must have capacity for 4 dims)
     * @return Number of dimensions written
     */
    uint32_t (*get_output_shape)(const Sensor* sensor, uint32_t* shape);

    /**
     * Sample observations for a batch of drones.
     * This is the critical path - called once per sensor type per frame.
     *
     * @param sensor        The sensor to sample
     * @param ctx           Context with drone state, indices, world, etc.
     * @param output_buffer Output buffer sized [ctx->drone_count * output_size]
     */
    void (*batch_sample)(Sensor* sensor, const SensorContext* ctx, float* output_buffer);

    /**
     * GPU batch sample dispatch (optional, NULL = CPU-only sensor).
     * Non-NULL indicates this sensor type supports GPU acceleration.
     * Returns 0 on success, negative on error (matches GpuResult).
     */
    int32_t (*batch_sample_gpu)(Sensor* sensor, const SensorContext* ctx, float* output_buffer);

    /**
     * Reset sensor state for a specific drone (optional).
     * Called when a drone is reset (e.g., episode end).
     *
     * @param sensor     The sensor to reset
     * @param drone_index Index of drone being reset
     */
    void (*reset)(Sensor* sensor, uint32_t drone_index);

    /**
     * Clean up sensor resources (optional).
     * Usually no-op with arena allocation.
     *
     * @param sensor The sensor to destroy
     */
    void (*destroy)(Sensor* sensor);
};

/* ============================================================================
 * Section 6: Sensor Base Structure
 * ============================================================================ */

/**
 * Base sensor structure.
 *
 * Contains common fields for all sensor types. The vtable pointer must be
 * first for efficient dispatch. Type-specific data is stored in impl pointer.
 */
struct Sensor {
    const SensorVTable* vtable;         /* Must be first for fast dispatch */
    void* impl;                         /* Type-specific implementation data */
    uint32_t sensor_id;                 /* Unique identifier within system */
    SensorType type;                    /* Cached type for fast checks */
    Vec3 position_offset;               /* Body-frame mounting position */
    Quat orientation_offset;            /* Body-frame mounting orientation */
    float sample_rate;                  /* Sampling rate in Hz (0 = every step) */
    float last_sample_time;             /* Time of last sample (for rate limiting) */
    NoiseConfig noise_config;           /* Composable noise pipeline configuration */
    NoiseState* noise_state;            /* Per-drone noise state (NULL = no noise) */
    size_t output_size;                 /* Cached output size in floats */
};

/* ============================================================================
 * Section 7: Sensor Configuration
 * ============================================================================ */

/**
 * Sensor configuration structure.
 *
 * Uses a union for type-specific parameters to minimize memory usage.
 * Create using sensor_config_* helper functions.
 */
struct SensorConfig {
    SensorType type;                    /* Sensor type to create */
    Vec3 position_offset;               /* Body-frame mounting position */
    Quat orientation_offset;            /* Body-frame mounting orientation */
    float sample_rate;                  /* Sampling rate in Hz (0 = every step) */
    NoiseConfig noise_config;           /* Composable noise pipeline configuration */

    /** Type-specific parameters */
    union {
        /* IMU parameters (noise configured via noise_config) */
        struct {
            uint8_t _reserved;          /* Empty - noise now in NoiseConfig */
        } imu;

        /* ToF (Time-of-Flight) parameters */
        struct {
            Vec3 direction;             /* Sensing direction (body frame) */
            float max_range;            /* Maximum sensing range (meters) */
        } tof;

        /* LiDAR 2D parameters */
        struct {
            uint32_t num_rays;          /* Number of rays */
            float fov;                  /* Field of view (radians) */
            float max_range;            /* Maximum range (meters) */
        } lidar_2d;

        /* LiDAR 3D parameters */
        struct {
            uint32_t horizontal_rays;   /* Rays per horizontal sweep */
            uint32_t vertical_layers;   /* Number of vertical layers */
            float horizontal_fov;       /* Horizontal FOV (radians) */
            float vertical_fov;         /* Vertical FOV (radians) */
            float max_range;            /* Maximum range (meters) */
        } lidar_3d;

        /* Camera parameters (RGB, Depth, Segmentation) */
        struct {
            uint32_t width;             /* Image width in pixels */
            uint32_t height;            /* Image height in pixels */
            float fov_horizontal;       /* Horizontal FOV (radians) */
            float fov_vertical;         /* Vertical FOV (radians) */
            float near_clip;            /* Near clipping plane */
            float far_clip;             /* Far clipping plane */
            uint32_t num_classes;       /* Number of segmentation classes */
        } camera;

        /* Neighbor sensor parameters */
        struct {
            uint32_t k;                 /* Number of neighbors to find */
            float max_range;            /* Maximum search range (meters) */
            bool include_self;          /* Include self in neighbor list */
        } neighbor;
    };
};

/* ============================================================================
 * Section 8: Sensor Attachment
 * ============================================================================ */

/**
 * Sensor attachment linking a sensor to a drone.
 *
 * Tracks which sensor is attached and where its output goes
 * in the drone's observation buffer.
 */
typedef struct SensorAttachment {
    uint32_t sensor_idx;                /* Index in SensorSystem.sensors */
    uint32_t output_offset;             /* Float offset in drone's observation */
    size_t output_size;                 /* Size of this sensor's output */
} SensorAttachment;

/* ============================================================================
 * Section 9: Sensor Registry
 * ============================================================================ */

/**
 * Registry of sensor type vtables.
 *
 * Maps SensorType enum values to their implementations.
 * Must be initialized before creating sensors.
 */
typedef struct SensorRegistry {
    const SensorVTable* vtables[SENSOR_TYPE_COUNT]; /* Vtable per type */
    bool initialized;                   /* True after sensor_registry_init() */
} SensorRegistry;

/* ============================================================================
 * Section 10: Sensor System (Main Structure)
 * ============================================================================ */

/**
 * Main sensor system structure.
 *
 * Manages all sensors, attachments, and the shared observation buffer.
 * Implements batch-by-type processing for optimal cache efficiency.
 *
 * Memory budget (1024 drones, 64 sensors, 128-dim obs):
 * - Sensor Array: 64 * 72 bytes = 4.5 KB
 * - Attachments: 1024 * 8 * 12 bytes = 96 KB
 * - Attachment Counts: 1024 * 4 bytes = 4 KB
 * - Observation Buffer: 1024 * 128 * 4 bytes = 512 KB
 * - Drones-by-Sensor Lists: 64 * 1024 * 4 bytes = 256 KB
 * - Drones-per-Sensor Counts: 64 * 4 bytes = 256 B
 * - Total: ~873 KB
 */
struct SensorSystem {
    /** Sensor type registry */
    SensorRegistry registry;

    /** Array of all sensors in the system */
    Sensor* sensors;
    uint32_t sensor_count;              /* Current number of sensors */
    uint32_t max_sensors;               /* Maximum sensor capacity */

    /** Per-drone sensor attachments: [max_drones * MAX_SENSORS_PER_DRONE] */
    SensorAttachment* attachments;

    /** Number of sensors attached per drone: [max_drones] */
    uint32_t* attachment_counts;

    /** Shared observation buffer: [max_drones * obs_dim], 32-byte aligned */
    float* observation_buffer;
    size_t obs_dim;                     /* Total observation dimensions */

    /** Batch processing data: drones grouped by sensor */
    uint32_t** drones_by_sensor;        /* [max_sensors][max_drones] */
    uint32_t* drones_per_sensor;        /* [max_sensors] count per sensor */

    /** Memory arenas */
    Arena* persistent_arena;            /* For permanent allocations */
    Arena* scratch_arena;               /* For per-frame temporary data */

    /** Simulation timestep for noise drift integration */
    float dt;

    /** Capacity limits */
    uint32_t max_drones;
};

/* ============================================================================
 * Section 11: Lifecycle Functions
 * ============================================================================ */

/**
 * Create a new sensor system.
 *
 * Allocates all memory from the provided arena. The observation buffer
 * is 32-byte aligned for SIMD operations.
 *
 * @param arena       Memory arena for allocation
 * @param max_drones  Maximum number of drones to support
 * @param max_sensors Maximum number of unique sensors
 * @param max_obs_dim Maximum observation dimensions per drone
 * @return New sensor system, or NULL on failure
 */
SensorSystem* sensor_system_create(Arena* arena, uint32_t max_drones,
                                   uint32_t max_sensors, size_t max_obs_dim);

/**
 * Destroy a sensor system.
 *
 * Calls destroy on all sensors. No-op with arena allocation.
 *
 * @param sys System to destroy (can be NULL)
 */
void sensor_system_destroy(SensorSystem* sys);

/**
 * Reset sensor system state.
 *
 * Clears observation buffer and resets scratch arena.
 * Does not remove sensors or attachments.
 *
 * @param sys System to reset
 */
void sensor_system_reset(SensorSystem* sys);

/* ============================================================================
 * Section 12: Registry Functions
 * ============================================================================ */

/**
 * Initialize the sensor registry with default vtables.
 *
 * Must be called before creating sensors. Registers all built-in sensor types.
 *
 * @param registry Registry to initialize
 */
void sensor_registry_init(SensorRegistry* registry);

/**
 * Get the vtable for a sensor type.
 *
 * @param registry Registry to query
 * @param type     Sensor type to look up
 * @return Vtable pointer, or NULL if not registered
 */
const SensorVTable* sensor_registry_get(const SensorRegistry* registry, SensorType type);

/**
 * Register a custom vtable for a sensor type.
 *
 * Can be used to override built-in implementations or add new sensor types.
 *
 * @param registry Registry to modify
 * @param type     Sensor type to register
 * @param vtable   Vtable to register
 */
void sensor_registry_register(SensorRegistry* registry, SensorType type,
                              const SensorVTable* vtable);

/* ============================================================================
 * Section 13: Sensor Management Functions
 * ============================================================================ */

/**
 * Create a new sensor in the system.
 *
 * The sensor is initialized using the vtable from the registry.
 *
 * @param sys    Sensor system
 * @param config Sensor configuration
 * @return Sensor index (0 to sensor_count-1), or UINT32_MAX on failure
 */
uint32_t sensor_system_create_sensor(SensorSystem* sys, const SensorConfig* config);

/**
 * Attach a sensor to a drone.
 *
 * The sensor's output will be written to the drone's observation buffer
 * at the returned offset.
 *
 * @param sys        Sensor system
 * @param drone_idx  Drone index to attach to
 * @param sensor_idx Sensor index to attach
 * @return Output offset in observation buffer, or UINT32_MAX on failure
 */
uint32_t sensor_system_attach(SensorSystem* sys, uint32_t drone_idx, uint32_t sensor_idx);

/**
 * Detach a sensor from a drone.
 *
 * @param sys            Sensor system
 * @param drone_idx      Drone index
 * @param attachment_idx Attachment index (0 to attachment_count-1)
 */
void sensor_system_detach(SensorSystem* sys, uint32_t drone_idx, uint32_t attachment_idx);

/**
 * Get a sensor by index.
 *
 * @param sys        Sensor system
 * @param sensor_idx Sensor index
 * @return Pointer to sensor, or NULL if invalid index
 */
Sensor* sensor_system_get_sensor(SensorSystem* sys, uint32_t sensor_idx);

/**
 * Get the number of sensors attached to a drone.
 *
 * @param sys       Sensor system
 * @param drone_idx Drone index
 * @return Number of attached sensors
 */
uint32_t sensor_system_get_attachment_count(const SensorSystem* sys, uint32_t drone_idx);

/* ============================================================================
 * Section 14: Batch Processing Functions (Critical Path)
 * ============================================================================ */

/**
 * Sample all sensors for all drones.
 *
 * This is the main entry point for sensor processing. It:
 * 1. Groups drones by sensor type for cache-efficient batch processing
 * 2. Calls each sensor's batch_sample once with all relevant drones
 * 3. Scatters results to per-drone observation buffers
 *
 * Target performance: <500us overhead (excluding sensor batch_sample time)
 *
 * @param sys         Sensor system
 * @param drones      Drone state arrays
 * @param world       World for raymarching sensors (can be NULL)
 * @param collision   Collision system for neighbor queries (can be NULL)
 * @param drone_count Number of drones to process
 */
void sensor_system_sample_all(SensorSystem* sys, const DroneStateSOA* drones,
                              const struct WorldBrickMap* world,
                              const struct CollisionSystem* collision,
                              uint32_t drone_count);

/**
 * Sample only CPU sensors (those with batch_sample_gpu == NULL).
 *
 * Groups all drones by sensor, then processes only non-GPU sensor types.
 * Used in the GPU pipeline: GPU sensors are dispatched separately via
 * gpu_sensors_dispatch(), while CPU-only sensors (IMU, position, velocity,
 * neighbor) run here in parallel with GPU work.
 *
 * @param sys         Sensor system
 * @param drones      Drone state arrays
 * @param world       World for raymarching sensors (can be NULL)
 * @param collision   Collision system for neighbor queries (can be NULL)
 * @param drone_count Number of drones to process
 */
void sensor_system_sample_cpu_only(SensorSystem* sys, const DroneStateSOA* drones,
                                    const struct WorldBrickMap* world,
                                    const struct CollisionSystem* collision,
                                    uint32_t drone_count);

/**
 * Sample a specific sensor for all drones that have it attached.
 *
 * @param sys        Sensor system
 * @param sensor_idx Sensor index to sample
 * @param drones     Drone state arrays
 * @param world      World for raymarching sensors (can be NULL)
 * @param collision  Collision system for neighbor queries (can be NULL)
 * @param drone_count Number of drones
 */
void sensor_system_sample_sensor(SensorSystem* sys, uint32_t sensor_idx,
                                 const DroneStateSOA* drones,
                                 const struct WorldBrickMap* world,
                                 const struct CollisionSystem* collision,
                                 uint32_t drone_count);

/* ============================================================================
 * Section 15: Observation Access Functions
 * ============================================================================ */

/**
 * Get the observation buffer pointer.
 *
 * Returns the contiguous observation buffer for all drones.
 * Buffer is sized [max_drones * obs_dim] and 32-byte aligned.
 *
 * @param sys Sensor system
 * @return Pointer to observation buffer
 */
float* sensor_system_get_observations(SensorSystem* sys);

/**
 * Get the observation buffer (const version).
 *
 * @param sys Sensor system
 * @return Const pointer to observation buffer
 */
const float* sensor_system_get_observations_const(const SensorSystem* sys);

/**
 * Set an external observation buffer for zero-copy writes.
 *
 * After calling this, sensor sampling writes directly into the provided buffer,
 * eliminating a memcpy from the internal buffer. The external buffer must be
 * at least max_drones * obs_dim * sizeof(float) bytes and 32-byte aligned.
 *
 * @param sys    Sensor system
 * @param buffer External observation buffer (must outlive sensor system)
 */
void sensor_system_set_external_buffer(SensorSystem* sys, float* buffer);

/**
 * Get the observation dimension.
 *
 * @param sys Sensor system
 * @return Total floats per drone observation
 */
size_t sensor_system_get_obs_dim(const SensorSystem* sys);

/**
 * Get a specific drone's observation.
 *
 * @param sys       Sensor system
 * @param drone_idx Drone index
 * @return Pointer to drone's observation (obs_dim floats)
 */
float* sensor_system_get_drone_obs(SensorSystem* sys, uint32_t drone_idx);

/**
 * Get a specific drone's observation (const version).
 *
 * @param sys       Sensor system
 * @param drone_idx Drone index
 * @return Const pointer to drone's observation
 */
const float* sensor_system_get_drone_obs_const(const SensorSystem* sys, uint32_t drone_idx);

/* ============================================================================
 * Section 16: Configuration Helper Functions
 * ============================================================================ */

/**
 * Create a default configuration for a sensor type.
 *
 * @param type Sensor type
 * @return Default configuration
 */
SensorConfig sensor_config_default(SensorType type);

/**
 * Create an IMU sensor configuration.
 *
 * Noise is configured via NoiseConfig on the returned SensorConfig.
 *
 * @return IMU configuration
 */
SensorConfig sensor_config_imu(void);

/**
 * Create a ToF sensor configuration.
 *
 * @param direction Sensing direction in body frame (will be normalized)
 * @param max_range Maximum sensing range in meters
 * @return ToF configuration
 */
SensorConfig sensor_config_tof(Vec3 direction, float max_range);

/**
 * Create a 2D LiDAR configuration.
 *
 * @param num_rays  Number of rays in the sweep
 * @param fov       Field of view in radians
 * @param max_range Maximum range in meters
 * @return LiDAR 2D configuration
 */
SensorConfig sensor_config_lidar_2d(uint32_t num_rays, float fov, float max_range);

/**
 * Create a 3D LiDAR configuration.
 *
 * @param horizontal_rays Rays per horizontal sweep
 * @param vertical_layers Number of vertical layers
 * @param horizontal_fov  Horizontal FOV in radians
 * @param vertical_fov    Vertical FOV in radians
 * @param max_range       Maximum range in meters
 * @return LiDAR 3D configuration
 */
SensorConfig sensor_config_lidar_3d(uint32_t horizontal_rays, uint32_t vertical_layers,
                                    float horizontal_fov, float vertical_fov,
                                    float max_range);

/**
 * Create a camera configuration.
 *
 * @param width     Image width in pixels
 * @param height    Image height in pixels
 * @param fov       Horizontal field of view in radians
 * @param max_range Maximum depth range in meters
 * @return Camera configuration
 */
SensorConfig sensor_config_camera(uint32_t width, uint32_t height,
                                  float fov, float max_range);

/**
 * Create a neighbor sensor configuration.
 *
 * @param k         Number of nearest neighbors to find
 * @param max_range Maximum search range in meters
 * @return Neighbor configuration
 */
SensorConfig sensor_config_neighbor(uint32_t k, float max_range);

/**
 * Create a position sensor configuration (oracle).
 *
 * @return Position configuration
 */
SensorConfig sensor_config_position(void);

/**
 * Create a velocity sensor configuration (oracle).
 *
 * @return Velocity configuration
 */
SensorConfig sensor_config_velocity(void);

/* ============================================================================
 * Section 17: Utility Functions
 * ============================================================================ */

/**
 * Get the name of a sensor type.
 *
 * @param type Sensor type
 * @return Human-readable name string
 */
const char* sensor_type_name(SensorType type);

/**
 * Calculate memory required for a sensor system.
 *
 * @param max_drones  Maximum drones
 * @param max_sensors Maximum sensors
 * @param max_obs_dim Maximum observation dimensions
 * @return Required bytes
 */
size_t sensor_system_memory_size(uint32_t max_drones, uint32_t max_sensors,
                                 size_t max_obs_dim);

/**
 * Compute total observation dimension from attachments.
 *
 * @param sys       Sensor system
 * @param drone_idx Drone index
 * @return Total observation dimension for this drone
 */
size_t sensor_system_compute_obs_dim(const SensorSystem* sys, uint32_t drone_idx);

/* ============================================================================
 * Section 18: Type Size Verification
 * ============================================================================ */

FOUNDATION_STATIC_ASSERT(SENSOR_TYPE_COUNT == 10, "Must have exactly 10 sensor types");
FOUNDATION_STATIC_ASSERT(MAX_SENSORS_PER_DRONE == 8, "MAX_SENSORS_PER_DRONE must be 8");

#ifdef __cplusplus
}
#endif

#endif /* SENSOR_SYSTEM_H */
