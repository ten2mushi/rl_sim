/**
 * GPU Backend: NONE (CPU Fallback)
 *
 * All functions return failure or no-op. Allows the engine to compile
 * and run on platforms without GPU support. GPU sensors will fall back
 * to CPU batch_sample automatically.
 */

#include "gpu_hal.h"

#if !GPU_AVAILABLE

#include <stddef.h>

/* Device */
bool gpu_is_available(void) { return false; }
GpuDevice* gpu_device_create(void) { return NULL; }
void gpu_device_destroy(GpuDevice* device) { (void)device; }
const char* gpu_device_name(const GpuDevice* device) { (void)device; return "None"; }
uint32_t gpu_device_max_threadgroup_size(const GpuDevice* device) { (void)device; return 0; }

/* Buffer */
GpuBuffer* gpu_buffer_create(GpuDevice* device, size_t size, GpuMemoryMode mode) {
    (void)device; (void)size; (void)mode; return NULL;
}
void gpu_buffer_destroy(GpuBuffer* buffer) { (void)buffer; }
void* gpu_buffer_map(GpuBuffer* buffer) { (void)buffer; return NULL; }
size_t gpu_buffer_size(const GpuBuffer* buffer) { (void)buffer; return 0; }
GpuResult gpu_buffer_upload(GpuBuffer* buffer, const void* data, size_t size, size_t offset) {
    (void)buffer; (void)data; (void)size; (void)offset; return GPU_ERROR_BACKEND;
}
GpuResult gpu_buffer_readback(const GpuBuffer* buffer, void* data, size_t size, size_t offset) {
    (void)buffer; (void)data; (void)size; (void)offset; return GPU_ERROR_BACKEND;
}

/* Kernel */
GpuKernel* gpu_kernel_create(GpuDevice* device, const char* function_name) {
    (void)device; (void)function_name; return NULL;
}
void gpu_kernel_destroy(GpuKernel* kernel) { (void)kernel; }
void gpu_kernel_set_buffer(GpuKernel* kernel, uint32_t index, GpuBuffer* buffer) {
    (void)kernel; (void)index; (void)buffer;
}
void gpu_kernel_set_constant(GpuKernel* kernel, uint32_t index, const void* data, size_t size) {
    (void)kernel; (void)index; (void)data; (void)size;
}

/* Command Queue */
GpuCommandQueue* gpu_queue_create(GpuDevice* device) { (void)device; return NULL; }
void gpu_queue_destroy(GpuCommandQueue* queue) { (void)queue; }
GpuResult gpu_queue_dispatch(GpuCommandQueue* queue, GpuKernel* kernel,
                             uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                             uint32_t group_x, uint32_t group_y, uint32_t group_z) {
    (void)queue; (void)kernel;
    (void)grid_x; (void)grid_y; (void)grid_z;
    (void)group_x; (void)group_y; (void)group_z;
    return GPU_ERROR_BACKEND;
}
GpuResult gpu_queue_wait(GpuCommandQueue* queue) { (void)queue; return GPU_ERROR_BACKEND; }
GpuResult gpu_queue_poll(GpuCommandQueue* queue) { (void)queue; return GPU_ERROR_BACKEND; }

/* Event */
GpuEvent* gpu_event_create(GpuDevice* device) { (void)device; return NULL; }
void gpu_event_destroy(GpuEvent* event) { (void)event; }
GpuResult gpu_event_signal(GpuEvent* event, GpuCommandQueue* queue, uint64_t value) {
    (void)event; (void)queue; (void)value; return GPU_ERROR_BACKEND;
}
GpuResult gpu_event_wait(GpuEvent* event, uint64_t value, uint64_t timeout_ms) {
    (void)event; (void)value; (void)timeout_ms; return GPU_ERROR_BACKEND;
}
uint64_t gpu_event_value(const GpuEvent* event) { (void)event; return 0; }

/* SDF Atlas */
GpuSdfAtlas gpu_sdf_atlas_upload(GpuDevice* device, const struct WorldBrickMap* world) {
    (void)device; (void)world;
    GpuSdfAtlas atlas = {0};
    return atlas;
}
GpuResult gpu_sdf_atlas_sync_dirty(GpuSdfAtlas* atlas, const struct WorldBrickMap* world) {
    (void)atlas; (void)world; return GPU_ERROR_BACKEND;
}
void gpu_sdf_atlas_destroy(GpuSdfAtlas* atlas) { (void)atlas; }

/* Drone Poses */
GpuDronePoses gpu_drone_poses_create(GpuDevice* device, uint32_t max_drones) {
    (void)device; (void)max_drones;
    GpuDronePoses poses = {0};
    return poses;
}
GpuResult gpu_drone_poses_upload(GpuDronePoses* poses, const struct DroneStateSOA* drones,
                                  uint32_t drone_count) {
    (void)poses; (void)drones; (void)drone_count; return GPU_ERROR_BACKEND;
}
void gpu_drone_poses_destroy(GpuDronePoses* poses) { (void)poses; }

/* Ray Table */
GpuRayTable gpu_ray_table_create(GpuDevice* device, const Vec3* directions, uint32_t count) {
    (void)device; (void)directions; (void)count;
    GpuRayTable table = {0};
    return table;
}
void gpu_ray_table_destroy(GpuRayTable* table) { (void)table; }

/* Sensor Output */
GpuSensorOutput gpu_sensor_output_create(GpuDevice* device, uint32_t total_floats) {
    (void)device; (void)total_floats;
    GpuSensorOutput output = {0};
    return output;
}
void gpu_sensor_output_destroy(GpuSensorOutput* output) { (void)output; }

/* ============================================================================
 * GPU Sensor Context + Dispatch Stubs
 * ============================================================================ */

#include "sensor_system.h"

struct GpuSensorContext;

void sensor_implementations_register_gpu(SensorRegistry* registry) {
    (void)registry;
}

struct GpuSensorContext* gpu_sensor_context_create(uint32_t max_drones) {
    (void)max_drones;
    return NULL;
}

void gpu_sensor_context_destroy(struct GpuSensorContext* ctx) {
    (void)ctx;
}

int32_t gpu_sensor_context_sync_frame(struct GpuSensorContext* ctx,
                                       const struct WorldBrickMap* world,
                                       const struct DroneStateSOA* drones,
                                       uint32_t drone_count) {
    (void)ctx; (void)world; (void)drones; (void)drone_count;
    return GPU_ERROR_BACKEND;
}

int32_t gpu_sensor_context_init_sensor(struct GpuSensorContext* ctx,
                                        const Sensor* sensor,
                                        uint32_t drone_count) {
    (void)ctx; (void)sensor; (void)drone_count;
    return GPU_ERROR_BACKEND;
}

int32_t gpu_sensors_dispatch(struct GpuSensorContext* gpu_ctx,
                              SensorSystem* sys,
                              const struct WorldBrickMap* world,
                              uint32_t drone_count) {
    (void)gpu_ctx; (void)sys; (void)world; (void)drone_count;
    return GPU_ERROR_BACKEND;
}

int32_t gpu_sensors_wait(struct GpuSensorContext* gpu_ctx) {
    (void)gpu_ctx;
    return GPU_ERROR_BACKEND;
}

int32_t gpu_sensors_scatter_results(struct GpuSensorContext* gpu_ctx,
                                     SensorSystem* sys,
                                     uint32_t drone_count) {
    (void)gpu_ctx; (void)sys; (void)drone_count;
    return GPU_ERROR_BACKEND;
}

#endif /* !GPU_AVAILABLE */
