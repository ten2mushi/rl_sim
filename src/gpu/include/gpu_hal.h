/**
 * GPU Hardware Abstraction Layer (HAL)
 *
 * C-only API providing opaque handles for GPU resources and a backend vtable
 * for portable compute dispatch. Currently supports Metal (Apple) with a
 * compile-time CPU fallback (GPU_BACKEND_NONE).
 *
 * Key types:
 *   GpuDevice       - GPU device handle
 *   GpuBuffer       - GPU buffer (vertex/index/storage)
 *   GpuKernel       - Compute pipeline + bound buffer slots
 *   GpuCommandQueue - Command buffer submission + encoding
 *   GpuEvent        - GPU/CPU synchronization fence
 *
 * SDF-specific helpers:
 *   GpuSdfAtlas     - Flattened SDF brick atlas for GPU access
 *   GpuDronePoses   - Packed drone positions + orientations
 *   GpuRayTable     - Precomputed ray directions (float4-padded)
 *   GpuSensorOutput - Double-buffered sensor output
 *
 * Dependencies: foundation.h (for Vec3, Quat, Arena, static_assert)
 */

#ifndef GPU_HAL_H
#define GPU_HAL_H

#include "foundation.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Section 1: Backend Detection and Compile-Time Configuration
 * ============================================================================ */

/* GPU_BACKEND is set by CMake: METAL or NONE */
#if !defined(GPU_BACKEND_METAL) && !defined(GPU_BACKEND_NONE)
  #if defined(__APPLE__)
    #define GPU_BACKEND_METAL 1
  #else
    #define GPU_BACKEND_NONE 1
  #endif
#endif

#if defined(GPU_BACKEND_METAL)
  #define GPU_AVAILABLE 1
#else
  #define GPU_AVAILABLE 0
#endif

/* ============================================================================
 * Section 2: Error Codes and Result Type
 * ============================================================================ */

typedef int32_t GpuResult;

#define GPU_SUCCESS           ((GpuResult)0)
#define GPU_ERROR_NO_DEVICE   ((GpuResult)-1)
#define GPU_ERROR_NO_MEMORY   ((GpuResult)-2)
#define GPU_ERROR_INVALID_ARG ((GpuResult)-3)
#define GPU_ERROR_COMPILE     ((GpuResult)-4)
#define GPU_ERROR_DISPATCH    ((GpuResult)-5)
#define GPU_ERROR_TIMEOUT     ((GpuResult)-6)
#define GPU_ERROR_NOT_READY   ((GpuResult)-7)
#define GPU_ERROR_BACKEND     ((GpuResult)-8)

/**
 * Get human-readable error string for a GpuResult code.
 */
const char* gpu_error_string(GpuResult result);

/* ============================================================================
 * Section 3: Opaque Handle Types
 * ============================================================================ */

/** GPU device (wraps MTLDevice on Metal) */
typedef struct GpuDevice GpuDevice;

/** GPU buffer (wraps MTLBuffer on Metal) */
typedef struct GpuBuffer GpuBuffer;

/** Compute kernel with bound buffer slots (wraps MTLComputePipelineState) */
typedef struct GpuKernel GpuKernel;

/** Command queue + encoder for dispatch (wraps MTLCommandQueue + encoder) */
typedef struct GpuCommandQueue GpuCommandQueue;

/** GPU/CPU synchronization event (wraps MTLSharedEvent) */
typedef struct GpuEvent GpuEvent;

/* ============================================================================
 * Section 4: Memory Mode
 * ============================================================================ */

typedef enum GpuMemoryMode {
    GPU_MEMORY_SHARED,     /**< CPU+GPU shared (Apple Silicon zero-copy) */
    GPU_MEMORY_PRIVATE,    /**< GPU-only (fastest for GPU, no CPU access) */
    GPU_MEMORY_MANAGED     /**< System-managed synchronization */
} GpuMemoryMode;

/* ============================================================================
 * Section 5: Device Functions
 * ============================================================================ */

/**
 * Check if any GPU backend is available at runtime.
 */
bool gpu_is_available(void);

/**
 * Create a GPU device (uses default system GPU).
 * Returns NULL if no GPU is available or backend is NONE.
 */
GpuDevice* gpu_device_create(void);

/**
 * Destroy a GPU device and all associated resources.
 */
void gpu_device_destroy(GpuDevice* device);

/**
 * Get the device name string (e.g. "Apple M1 Pro").
 */
const char* gpu_device_name(const GpuDevice* device);

/**
 * Get maximum threadgroup size supported.
 */
uint32_t gpu_device_max_threadgroup_size(const GpuDevice* device);

/* ============================================================================
 * Section 6: Buffer Functions
 * ============================================================================ */

/**
 * Create a GPU buffer.
 *
 * @param device   GPU device
 * @param size     Buffer size in bytes
 * @param mode     Memory mode (shared/private/managed)
 * @return Buffer handle or NULL on failure
 */
GpuBuffer* gpu_buffer_create(GpuDevice* device, size_t size, GpuMemoryMode mode);

/**
 * Destroy a GPU buffer.
 */
void gpu_buffer_destroy(GpuBuffer* buffer);

/**
 * Get CPU-accessible pointer to buffer contents.
 * Only valid for GPU_MEMORY_SHARED and GPU_MEMORY_MANAGED modes.
 * Returns NULL for private buffers.
 */
void* gpu_buffer_map(GpuBuffer* buffer);

/**
 * Get buffer size in bytes.
 */
size_t gpu_buffer_size(const GpuBuffer* buffer);

/**
 * Upload data from CPU memory to GPU buffer.
 * Buffer must be shared or managed mode.
 *
 * @param buffer GPU buffer
 * @param data   Source CPU data
 * @param size   Bytes to copy
 * @param offset Byte offset into buffer
 * @return GPU_SUCCESS or error code
 */
GpuResult gpu_buffer_upload(GpuBuffer* buffer, const void* data, size_t size,
                            size_t offset);

/**
 * Read data from GPU buffer to CPU memory.
 * Buffer must be shared or managed mode.
 *
 * @param buffer GPU buffer
 * @param data   Destination CPU buffer
 * @param size   Bytes to copy
 * @param offset Byte offset into GPU buffer
 * @return GPU_SUCCESS or error code
 */
GpuResult gpu_buffer_readback(const GpuBuffer* buffer, void* data, size_t size,
                              size_t offset);

/* ============================================================================
 * Section 7: Kernel Functions
 * ============================================================================ */

/** Maximum buffer bindings per kernel */
#define GPU_MAX_BUFFER_BINDINGS 16

/** Maximum constant bindings per kernel */
#define GPU_MAX_CONSTANT_BINDINGS 4

/** Maximum constant data size per binding */
#define GPU_MAX_CONSTANT_SIZE 256

/**
 * Create a compute kernel from a function name in the default library.
 *
 * @param device        GPU device
 * @param function_name Metal function name
 * @return Kernel handle or NULL on failure
 */
GpuKernel* gpu_kernel_create(GpuDevice* device, const char* function_name);

/**
 * Destroy a compute kernel.
 */
void gpu_kernel_destroy(GpuKernel* kernel);

/**
 * Bind a buffer to a kernel slot.
 * Bindings are stored in the kernel and applied at dispatch time.
 *
 * @param kernel Compute kernel
 * @param index  Buffer index (0 to GPU_MAX_BUFFER_BINDINGS-1)
 * @param buffer Buffer to bind (NULL to unbind)
 */
void gpu_kernel_set_buffer(GpuKernel* kernel, uint32_t index, GpuBuffer* buffer);

/**
 * Set constant data on a kernel slot.
 * Data is copied and applied at dispatch time.
 *
 * @param kernel Compute kernel
 * @param index  Constant index (0 to GPU_MAX_CONSTANT_BINDINGS-1)
 * @param data   Pointer to constant data
 * @param size   Size in bytes (must be <= GPU_MAX_CONSTANT_SIZE)
 */
void gpu_kernel_set_constant(GpuKernel* kernel, uint32_t index,
                             const void* data, size_t size);

/* ============================================================================
 * Section 8: Command Queue Functions
 * ============================================================================ */

/**
 * Create a command queue for submitting GPU work.
 */
GpuCommandQueue* gpu_queue_create(GpuDevice* device);

/**
 * Destroy a command queue.
 */
void gpu_queue_destroy(GpuCommandQueue* queue);

/**
 * Dispatch a compute kernel.
 * Applies all bound buffers and constants, encodes dispatch, commits.
 *
 * @param queue   Command queue
 * @param kernel  Compute kernel with bindings set
 * @param grid_x  Global grid width
 * @param grid_y  Global grid height
 * @param grid_z  Global grid depth
 * @param group_x Threadgroup width
 * @param group_y Threadgroup height
 * @param group_z Threadgroup depth
 * @return GPU_SUCCESS or error code
 */
GpuResult gpu_queue_dispatch(GpuCommandQueue* queue, GpuKernel* kernel,
                             uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                             uint32_t group_x, uint32_t group_y, uint32_t group_z);

/**
 * Wait for all submitted work on this queue to complete.
 */
GpuResult gpu_queue_wait(GpuCommandQueue* queue);

/**
 * Check if all submitted work has completed (non-blocking).
 * Returns GPU_SUCCESS if done, GPU_ERROR_NOT_READY if still running.
 */
GpuResult gpu_queue_poll(GpuCommandQueue* queue);

/* ============================================================================
 * Section 9: Event Functions (GPU/CPU Synchronization)
 * ============================================================================ */

/**
 * Create a synchronization event.
 */
GpuEvent* gpu_event_create(GpuDevice* device);

/**
 * Destroy an event.
 */
void gpu_event_destroy(GpuEvent* event);

/**
 * Signal the event from the GPU at the end of current command buffer.
 */
GpuResult gpu_event_signal(GpuEvent* event, GpuCommandQueue* queue,
                           uint64_t value);

/**
 * Wait on the CPU until the event reaches the specified value.
 * @param timeout_ms Maximum wait in milliseconds (0 = no wait, UINT64_MAX = infinite)
 */
GpuResult gpu_event_wait(GpuEvent* event, uint64_t value, uint64_t timeout_ms);

/**
 * Get current event value (non-blocking).
 */
uint64_t gpu_event_value(const GpuEvent* event);

/* ============================================================================
 * Section 10: Forward Declarations for Domain Types
 * ============================================================================ */

struct WorldBrickMap;
struct RigidBodyStateSOA;
struct MeshBVH;
struct TriangleMesh;

/* ============================================================================
 * Section 11: SDF Atlas GPU Structure
 * ============================================================================ */

/**
 * Flattened SDF atlas for GPU consumption.
 *
 * The CPU world uses demand-paged storage (sdf_pages[]).
 * This structure provides a contiguous GPU buffer with the SDF data
 * and a brick index remapping table.
 */
/** Maximum feature channel GPU buffers in atlas */
#define GPU_MAX_FEATURE_CHANNELS 16

typedef struct GpuSdfAtlas {
    GpuBuffer* sdf_data;        /**< Contiguous int8 SDF [brick_count * 512] */
    GpuBuffer* material_data;   /**< Contiguous uint8 material [brick_count * 512] */
    GpuBuffer* brick_indices;   /**< int32 grid [grid_total], remapped to flat atlas */
    uint32_t brick_count;       /**< Number of active bricks in GPU atlas */
    uint32_t grid_total;        /**< Total grid cells (same as WorldBrickMap) */

    /* User-defined feature channel buffers (flattened from VoxelChannel pages) */
    GpuBuffer* channel_data[GPU_MAX_FEATURE_CHANNELS];
    uint32_t channel_count;     /**< Number of uploaded feature channels */

    /* Dirty tracking for incremental sync */
    uint64_t last_sync_frame;   /**< Frame counter at last full sync */
    bool needs_full_sync;       /**< Force full re-upload on next sync */
} GpuSdfAtlas;

/**
 * Create and upload a flattened SDF atlas from a WorldBrickMap.
 *
 * Iterates all demand-paged bricks and copies to contiguous GPU buffers.
 * Brick indices are remapped from page-local to flat-atlas addressing.
 *
 * @param device GPU device
 * @param world  Source world brick map
 * @return GpuSdfAtlas or {NULL} on failure
 */
GpuSdfAtlas gpu_sdf_atlas_upload(GpuDevice* device,
                                  const struct WorldBrickMap* world);

/**
 * Incrementally sync dirty bricks from CPU to GPU.
 * Only re-uploads bricks that changed since last sync.
 *
 * @param atlas  GPU SDF atlas
 * @param world  Source world brick map
 * @return GPU_SUCCESS or error code
 */
GpuResult gpu_sdf_atlas_sync_dirty(GpuSdfAtlas* atlas,
                                    const struct WorldBrickMap* world);

/**
 * Destroy a GPU SDF atlas.
 */
void gpu_sdf_atlas_destroy(GpuSdfAtlas* atlas);

/* ============================================================================
 * Section 12: Drone Poses GPU Structure
 * ============================================================================ */

/**
 * Packed drone positions and orientations for GPU consumption.
 * Layout: separate float arrays for pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z
 */
typedef struct GpuDronePoses {
    GpuBuffer* pos_x;     /**< float[max_agents] */
    GpuBuffer* pos_y;     /**< float[max_agents] */
    GpuBuffer* pos_z;     /**< float[max_agents] */
    GpuBuffer* quat_w;    /**< float[max_agents] */
    GpuBuffer* quat_x;    /**< float[max_agents] */
    GpuBuffer* quat_y;    /**< float[max_agents] */
    GpuBuffer* quat_z;    /**< float[max_agents] */
    uint32_t max_agents;
} GpuDronePoses;

/**
 * Create drone pose buffers.
 */
GpuDronePoses gpu_agent_poses_create(GpuDevice* device, uint32_t max_agents);

/**
 * Upload drone poses from RigidBodyStateSOA to GPU buffers.
 */
GpuResult gpu_agent_poses_upload(GpuDronePoses* poses,
                                  const struct RigidBodyStateSOA* agents,
                                  uint32_t agent_count);

/**
 * Destroy drone pose buffers.
 */
void gpu_agent_poses_destroy(GpuDronePoses* poses);

/* ============================================================================
 * Section 13: Ray Table GPU Structure
 * ============================================================================ */

/**
 * Precomputed ray directions packed as float4 (xyz + padding).
 */
typedef struct GpuRayTable {
    GpuBuffer* rays;       /**< float4[ray_count] packed ray directions */
    uint32_t ray_count;    /**< Number of rays */
} GpuRayTable;

/**
 * Create a ray table from Vec3 ray directions.
 * Converts Vec3 (12 bytes) to float4 (16 bytes) with zero padding.
 *
 * @param device     GPU device
 * @param directions Array of Vec3 ray directions
 * @param count      Number of rays
 * @return GpuRayTable or {NULL} on failure
 */
GpuRayTable gpu_ray_table_create(GpuDevice* device, const Vec3* directions,
                                  uint32_t count);

/**
 * Destroy a ray table.
 */
void gpu_ray_table_destroy(GpuRayTable* table);

/* ============================================================================
 * Section 14: Sensor Output GPU Structure
 * ============================================================================ */

/**
 * GPU sensor output buffer with swap for async readback.
 */
typedef struct GpuSensorOutput {
    GpuBuffer* buffer;     /**< Output buffer [total_floats] */
    uint32_t total_floats; /**< Buffer size in floats */
} GpuSensorOutput;

/**
 * Create a sensor output buffer.
 *
 * @param device       GPU device
 * @param total_floats Total float elements in output
 * @return GpuSensorOutput or {NULL} on failure
 */
GpuSensorOutput gpu_sensor_output_create(GpuDevice* device,
                                          uint32_t total_floats);

/**
 * Destroy a sensor output buffer.
 */
void gpu_sensor_output_destroy(GpuSensorOutput* output);

/* ============================================================================
 * Section 15: Linearized BVH for GPU Voxelization
 * ============================================================================ */

/**
 * GPU-friendly BVH: single buffer of GpuBVHNode structs + face index buffer.
 * Converted from CPU MeshBVH's AoS BVHNode array.
 */
typedef struct GpuLinearBVH {
    GpuBuffer* nodes;           /**< GpuBVHNode[node_count] (48 bytes each) */
    GpuBuffer* face_indices;    /**< uint32[face_count] reordered face indices */
    uint32_t node_count;
    uint32_t face_count;
    float avg_normal_x, avg_normal_y, avg_normal_z;
    float normal_coherence;
} GpuLinearBVH;

/**
 * Create a GPU BVH from a CPU MeshBVH.
 */
GpuLinearBVH gpu_linear_bvh_create(GpuDevice* device,
                                    const struct MeshBVH* bvh);

/**
 * Destroy a GPU BVH.
 */
void gpu_linear_bvh_destroy(GpuLinearBVH* bvh);

/* ============================================================================
 * Section 16: Triangle Data for GPU Voxelization
 * ============================================================================ */

/**
 * Triangle mesh data uploaded to GPU buffers.
 * Vertex positions as SoA, face indices and materials.
 */
typedef struct GpuTriangleData {
    GpuBuffer* vertices_x;     /**< float[vertex_count] */
    GpuBuffer* vertices_y;     /**< float[vertex_count] */
    GpuBuffer* vertices_z;     /**< float[vertex_count] */
    GpuBuffer* face_v;         /**< uint32[face_count * 3] vertex indices */
    GpuBuffer* face_mat;       /**< uint8[face_count] material IDs */
    uint32_t vertex_count;
    uint32_t face_count;
    float bbox_min_x, bbox_min_y, bbox_min_z;
    float bbox_max_x, bbox_max_y, bbox_max_z;
} GpuTriangleData;

/**
 * Create GPU triangle data from a CPU TriangleMesh.
 */
GpuTriangleData gpu_triangle_data_create(GpuDevice* device,
                                          const struct TriangleMesh* mesh);

/**
 * Destroy GPU triangle data.
 */
void gpu_triangle_data_destroy(GpuTriangleData* data);

/* ============================================================================
 * Section 17: GPU Voxelization
 * ============================================================================ */

/**
 * GPU-accelerated Phase 3 voxelization: compute per-voxel SDF for surface bricks.
 * Phases 1-2 (classification) remain on CPU. This function handles the expensive
 * per-voxel closest_point + inside_outside queries on GPU.
 *
 * @param device              GPU device
 * @param bvh                 CPU BVH (will be uploaded to GPU)
 * @param mesh                CPU mesh (will be uploaded to GPU)
 * @param surface_brick_list  Array of (bx,by,bz) packed as uint32[num_bricks*3]
 * @param num_surface_bricks  Number of surface bricks
 * @param world               WorldBrickMap to populate with SDF data
 * @param options             Voxelization options
 * @return GPU_SUCCESS or error code
 */
GpuResult gpu_voxelize_surface_bricks(GpuDevice* device,
                                       const struct MeshBVH* bvh,
                                       const struct TriangleMesh* mesh,
                                       const uint32_t* surface_brick_list,
                                       uint32_t num_surface_bricks,
                                       struct WorldBrickMap* world,
                                       const void* options);

/* ============================================================================
 * Section 18: Forward Declaration for Sensor Context
 * ============================================================================ */

struct GpuSensorContext;

/* ============================================================================
 * Section 19: Static Assertions
 * ============================================================================ */

FOUNDATION_STATIC_ASSERT(sizeof(GpuResult) == sizeof(int32_t),
                         "GpuResult must be int32_t");
FOUNDATION_STATIC_ASSERT(GPU_MAX_BUFFER_BINDINGS == 16,
                         "Must support 16 buffer bindings");

#ifdef __cplusplus
}
#endif

#endif /* GPU_HAL_H */
