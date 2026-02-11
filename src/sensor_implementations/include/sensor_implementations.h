/**
 * Sensor Implementations Module - Vtable Declarations for All Sensor Types
 *
 * Provides vtable implementations for all 10 sensor types:
 * - IMU: 6-axis inertial measurement unit
 * - ToF: Time-of-flight distance sensor
 * - LiDAR 2D/3D: Multi-ray distance scanning
 * - Camera RGB/Depth/Segmentation: Visual sensors
 * - Position/Velocity: Oracle sensors (direct state access)
 * - Neighbor: K-nearest neighbor detection
 *
 * All implementations use batch-by-type processing for cache efficiency
 * and per-sensor PCG32 RNG for deterministic noise.
 *
 * Dependencies:
 * - sensor_system: Sensor, SensorVTable, SensorConfig, etc.
 * - foundation: Vec3, Quat, PCG32, Arena
 * - world_brick_map: RayHit, world_raymarch, world_raymarch_batch
 * - collision_system: collision_find_k_nearest
 */

#ifndef SENSOR_IMPLEMENTATIONS_H
#define SENSOR_IMPLEMENTATIONS_H

#include "sensor_system.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Section 1: Vtable Declarations
 * ============================================================================ */

/** IMU sensor vtable - 6 floats output (ax, ay, az, gx, gy, gz) */
extern const SensorVTable SENSOR_VTABLE_IMU;

/** ToF sensor vtable - 1 float output (distance) */
extern const SensorVTable SENSOR_VTABLE_TOF;

/** LiDAR 2D sensor vtable - N floats output (distances) */
extern const SensorVTable SENSOR_VTABLE_LIDAR_2D;

/** LiDAR 3D sensor vtable - N*M floats output (distances) */
extern const SensorVTable SENSOR_VTABLE_LIDAR_3D;

/** Camera RGB sensor vtable - W*H*3 floats output */
extern const SensorVTable SENSOR_VTABLE_CAMERA_RGB;

/** Camera Depth sensor vtable - W*H floats output (normalized [0,1]) */
extern const SensorVTable SENSOR_VTABLE_CAMERA_DEPTH;

/** Camera Segmentation sensor vtable - W*H floats output (material IDs) */
extern const SensorVTable SENSOR_VTABLE_CAMERA_SEGMENTATION;

/** Position sensor vtable - 3 floats output (x, y, z) */
extern const SensorVTable SENSOR_VTABLE_POSITION;

/** Velocity sensor vtable - 6 floats output (vx, vy, vz, wx, wy, wz) */
extern const SensorVTable SENSOR_VTABLE_VELOCITY;

/** Neighbor sensor vtable - K*4 floats output (dx, dy, dz, dist) */
extern const SensorVTable SENSOR_VTABLE_NEIGHBOR;

/* ============================================================================
 * Section 2: Registry Initialization
 * ============================================================================ */

/**
 * Register all sensor implementations with a sensor registry.
 *
 * Call this after sensor_registry_init() to register all built-in sensor types.
 *
 * @param registry Registry to populate with sensor vtables
 */
void sensor_implementations_register_all(SensorRegistry* registry);

/* ============================================================================
 * Section 2b: GPU Acceleration
 * ============================================================================ */

/**
 * Register GPU-accelerated sensor implementations.
 * Copies CPU vtables, sets batch_sample_gpu for GPU-capable types,
 * and re-registers them. No-op when GPU is unavailable.
 *
 * @param registry Registry to update with GPU vtables
 */
void sensor_implementations_register_gpu(SensorRegistry* registry);

/**
 * Create a GPU sensor context for accelerated dispatch.
 * Returns NULL if GPU is unavailable or creation fails.
 */
struct GpuSensorContext* gpu_sensor_context_create(uint32_t max_drones);

/** Destroy a GPU sensor context. */
void gpu_sensor_context_destroy(struct GpuSensorContext* ctx);

/**
 * Sync SDF atlas and drone poses for current frame.
 * Must be called before gpu_sensors_dispatch().
 */
int32_t gpu_sensor_context_sync_frame(struct GpuSensorContext* ctx,
                                       const struct WorldBrickMap* world,
                                       const DroneStateSOA* drones,
                                       uint32_t drone_count);

/**
 * Initialize a GPU sensor slot for a given sensor type.
 * Lazy: called once per sensor type, creates ray table + output buffer.
 */
int32_t gpu_sensor_context_init_sensor(struct GpuSensorContext* ctx,
                                        const Sensor* sensor,
                                        uint32_t drone_count);

/**
 * Dispatch all GPU-accelerated sensors (returns immediately).
 * GPU work runs asynchronously while CPU sensors can be processed.
 */
int32_t gpu_sensors_dispatch(struct GpuSensorContext* gpu_ctx,
                              SensorSystem* sys,
                              const struct WorldBrickMap* world,
                              uint32_t drone_count);

/**
 * Wait for all GPU sensor dispatches to complete.
 */
int32_t gpu_sensors_wait(struct GpuSensorContext* gpu_ctx);

/**
 * Scatter GPU results to observation buffer.
 * Call after gpu_sensors_wait().
 */
int32_t gpu_sensors_scatter_results(struct GpuSensorContext* gpu_ctx,
                                     SensorSystem* sys,
                                     uint32_t drone_count);

/* ============================================================================
 * Section 3: IMU Implementation Types
 * ============================================================================ */

/**
 * IMU sensor implementation data.
 *
 * Stores calibration parameters and per-sensor RNG state.
 */
typedef struct IMUImpl {
    uint8_t _reserved;          /* Noise now handled by composable pipeline */
} IMUImpl;

/* ============================================================================
 * Section 4: ToF Implementation Types
 * ============================================================================ */

/**
 * ToF sensor implementation data.
 */
typedef struct ToFImpl {
    Vec3 direction;             /* Sensing direction (body frame, normalized) */
    float max_range;            /* Maximum sensing range */
} ToFImpl;

/* ============================================================================
 * Section 5: LiDAR Implementation Types
 * ============================================================================ */

/**
 * LiDAR 2D sensor implementation data.
 *
 * Stores precomputed ray directions for cache-efficient batch processing.
 */
typedef struct LiDAR2DImpl {
    Vec3* ray_directions;       /* Precomputed ray directions [num_rays], 32-byte aligned */
    uint32_t num_rays;          /* Number of rays */
    float fov;                  /* Field of view (radians) */
    float max_range;            /* Maximum sensing range */
} LiDAR2DImpl;

/**
 * LiDAR 3D sensor implementation data.
 */
typedef struct LiDAR3DImpl {
    Vec3* ray_directions;       /* Precomputed ray directions [h_rays * v_layers], 32-byte aligned */
    uint32_t horizontal_rays;   /* Rays per horizontal sweep */
    uint32_t vertical_layers;   /* Number of vertical layers */
    uint32_t total_rays;        /* horizontal_rays * vertical_layers */
    float horizontal_fov;       /* Horizontal FOV (radians) */
    float vertical_fov;         /* Vertical FOV (radians) */
    float max_range;            /* Maximum sensing range */
} LiDAR3DImpl;

/* ============================================================================
 * Section 6: Camera Implementation Types
 * ============================================================================ */

/**
 * Camera sensor implementation data.
 *
 * Used for RGB, Depth, and Segmentation cameras.
 * Stores precomputed ray directions for each pixel.
 */
typedef struct CameraImpl {
    Vec3* ray_directions;       /* Precomputed ray directions [width * height], 32-byte aligned */
    uint32_t width;             /* Image width */
    uint32_t height;            /* Image height */
    uint32_t total_pixels;      /* width * height */
    float fov_horizontal;       /* Horizontal FOV (radians) */
    float fov_vertical;         /* Vertical FOV (radians) */
    float near_clip;            /* Near clipping distance */
    float far_clip;             /* Far clipping distance */
    float inv_depth_range;      /* 1.0 / (far_clip - near_clip) for normalization */
    uint32_t num_classes;       /* Number of segmentation classes */
} CameraImpl;

/* ============================================================================
 * Section 7: Neighbor Implementation Types
 * ============================================================================ */

/**
 * Neighbor sensor implementation data.
 */
typedef struct NeighborImpl {
    uint32_t k;                 /* Number of neighbors to find */
    float max_range;            /* Maximum search range */
    float max_range_sq;         /* max_range^2 for distance comparisons */
    bool include_self;          /* Include self in neighbor list */
} NeighborImpl;

/* ============================================================================
 * Section 8: Ray Direction Precomputation
 * ============================================================================ */

/**
 * Precompute ray directions for a 2D LiDAR sweep.
 *
 * Rays are distributed evenly across the FOV in the XY plane.
 *
 * @param arena      Arena for allocation
 * @param num_rays   Number of rays
 * @param fov        Field of view (radians)
 * @return Array of normalized ray directions, or NULL on failure
 */
Vec3* precompute_lidar_2d_rays(Arena* arena, uint32_t num_rays, float fov);

/**
 * Precompute ray directions for a 3D LiDAR.
 *
 * Rays form a spherical grid with horizontal and vertical layers.
 *
 * @param arena           Arena for allocation
 * @param horizontal_rays Rays per horizontal sweep
 * @param vertical_layers Number of vertical layers
 * @param horizontal_fov  Horizontal FOV (radians)
 * @param vertical_fov    Vertical FOV (radians)
 * @return Array of normalized ray directions, or NULL on failure
 */
Vec3* precompute_lidar_3d_rays(Arena* arena, uint32_t horizontal_rays,
                               uint32_t vertical_layers,
                               float horizontal_fov, float vertical_fov);

/**
 * Precompute ray directions for a camera.
 *
 * Rays are computed for a pinhole camera model.
 *
 * @param arena          Arena for allocation
 * @param width          Image width
 * @param height         Image height
 * @param fov_horizontal Horizontal FOV (radians)
 * @param fov_vertical   Vertical FOV (radians)
 * @return Array of normalized ray directions, or NULL on failure
 */
Vec3* precompute_camera_rays(Arena* arena, uint32_t width, uint32_t height,
                             float fov_horizontal, float fov_vertical);

#ifdef __cplusplus
}
#endif

#endif /* SENSOR_IMPLEMENTATIONS_H */
