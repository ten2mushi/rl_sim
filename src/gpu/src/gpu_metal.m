/**
 * Metal GPU Backend Implementation
 *
 * Implements the GPU HAL using Apple Metal compute shaders.
 * Uses Objective-C for Metal API interaction.
 *
 * Key implementation details:
 * - GpuDevice wraps MTLDevice + default MTLLibrary
 * - GpuBuffer wraps MTLBuffer with GPU_MEMORY_SHARED for Apple Silicon zero-copy
 * - GpuKernel stores bound_buffers[16] and bound_constants[4], applied at dispatch
 * - GpuCommandQueue creates a new MTLCommandBuffer per dispatch
 * - GpuEvent wraps MTLSharedEvent for CPU/GPU synchronization
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <mach/mach_time.h>
#include <unistd.h>
#include "gpu_hal.h"
#include "world_brick_map.h"
#include "drone_state.h"
#include <string.h>

/* ============================================================================
 * Section 1: Opaque Type Definitions
 * ============================================================================ */

struct GpuDevice {
    id<MTLDevice> mtl_device;
    id<MTLLibrary> mtl_library;
    char name[256];
};

struct GpuBuffer {
    id<MTLBuffer> mtl_buffer;
    size_t size;
    GpuMemoryMode mode;
};

typedef struct GpuConstantBinding {
    uint8_t data[GPU_MAX_CONSTANT_SIZE];
    size_t size;
    bool bound;
} GpuConstantBinding;

struct GpuKernel {
    id<MTLComputePipelineState> mtl_pipeline;
    GpuBuffer* bound_buffers[GPU_MAX_BUFFER_BINDINGS];
    GpuConstantBinding bound_constants[GPU_MAX_CONSTANT_BINDINGS];
    char function_name[128];
};

struct GpuCommandQueue {
    id<MTLCommandQueue> mtl_queue;
    id<MTLCommandBuffer> mtl_last_command_buffer;
    GpuDevice* device;
};

struct GpuEvent {
    id<MTLSharedEvent> mtl_event;
};

/* ============================================================================
 * Section 2: Helper - Convert GpuMemoryMode to MTLResourceOptions
 * ============================================================================ */

static MTLResourceOptions memory_mode_to_options(GpuMemoryMode mode) {
    switch (mode) {
        case GPU_MEMORY_SHARED:
            return MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;
        case GPU_MEMORY_PRIVATE:
            return MTLResourceStorageModePrivate;
        case GPU_MEMORY_MANAGED:
            #if TARGET_OS_OSX
            return MTLResourceStorageModeManaged;
            #else
            return MTLResourceStorageModeShared;
            #endif
        default:
            return MTLResourceStorageModeShared;
    }
}

/* ============================================================================
 * Section 3: Device Functions
 * ============================================================================ */

/**
 * Get Metal device, preferring MTLCopyAllDevices() which works in CLI/headless
 * contexts where MTLCreateSystemDefaultDevice() returns nil.
 */
static id<MTLDevice> get_metal_device(void) {
    /* MTLCopyAllDevices works in headless/CLI environments */
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (devices != nil && [devices count] > 0) {
        return devices[0];
    }
    /* Fallback to system default */
    return MTLCreateSystemDefaultDevice();
}

bool gpu_is_available(void) {
    @autoreleasepool {
        id<MTLDevice> device = get_metal_device();
        return device != nil;
    }
}

GpuDevice* gpu_device_create(void) {
    @autoreleasepool {
        id<MTLDevice> mtl_device = get_metal_device();
        if (mtl_device == nil) {
            return NULL;
        }

        GpuDevice* device = calloc(1, sizeof(GpuDevice));
        if (device == NULL) {
            return NULL;
        }

        device->mtl_device = mtl_device;

        /* Try loading pre-compiled metallib first (fast path) */
        NSError* error = nil;
        NSString* search_dirs[] = {
            [[NSBundle mainBundle] resourcePath],
            [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent],
            [[NSFileManager defaultManager] currentDirectoryPath],
        };
        for (int i = 0; i < 3 && device->mtl_library == nil; i++) {
            if (search_dirs[i] == nil) continue;
            NSString* path = [search_dirs[i] stringByAppendingPathComponent:@"default.metallib"];
            device->mtl_library = [mtl_device newLibraryWithURL:[NSURL fileURLWithPath:path]
                                                          error:&error];
        }

        /* Fallback: compile .metal source at runtime (no Xcode needed) */
        if (device->mtl_library == nil) {
            /* Search for raymarch.metal source in known locations */
            NSString* src_search[] = {
                /* Relative to CWD (typical: build/ -> ../src/gpu/shaders/) */
                @"../src/gpu/shaders/raymarch.metal",
                @"src/gpu/shaders/raymarch.metal",
                /* Relative to executable */
                [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent],
            };
            NSString* shader_src = nil;
            NSString* include_dir = nil;
            for (int i = 0; i < 3 && shader_src == nil; i++) {
                NSString* path;
                if (i < 2) {
                    path = src_search[i];
                } else {
                    if (src_search[i] == nil) continue;
                    path = [src_search[i] stringByAppendingPathComponent:
                            @"../src/gpu/shaders/raymarch.metal"];
                }
                NSString* full = [[[NSFileManager defaultManager] currentDirectoryPath]
                                  stringByAppendingPathComponent:path];
                shader_src = [NSString stringWithContentsOfFile:full
                                                      encoding:NSUTF8StringEncoding
                                                         error:nil];
                if (shader_src != nil) {
                    include_dir = [full stringByDeletingLastPathComponent];
                }
            }

            if (shader_src != nil) {
                MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
                opts.languageVersion = MTLLanguageVersion3_0;
                /* Set preprocessor macros for include path workaround:
                 * Metal runtime compiler doesn't support -I, so we
                 * prepend the sdf_types.h content. */
                NSString* types_path = [include_dir stringByAppendingPathComponent:
                                        @"sdf_types.h"];
                NSString* types_src = [NSString stringWithContentsOfFile:types_path
                                                               encoding:NSUTF8StringEncoding
                                                                  error:nil];
                if (types_src != nil) {
                    /* Replace #include "sdf_types.h" with actual content */
                    shader_src = [shader_src stringByReplacingOccurrencesOfString:
                                  @"#include \"sdf_types.h\""
                                                                      withString:types_src];
                }

                /* Also load voxelize.metal and concatenate */
                NSString* voxelize_path = [include_dir stringByAppendingPathComponent:
                                           @"voxelize.metal"];
                NSString* voxelize_src = [NSString stringWithContentsOfFile:voxelize_path
                                                                  encoding:NSUTF8StringEncoding
                                                                     error:nil];
                if (voxelize_src != nil && types_src != nil) {
                    voxelize_src = [voxelize_src stringByReplacingOccurrencesOfString:
                                    @"#include \"sdf_types.h\""
                                                                          withString:types_src];
                    /* Remove duplicate metal_stdlib include */
                    voxelize_src = [voxelize_src stringByReplacingOccurrencesOfString:
                                    @"#include <metal_stdlib>\nusing namespace metal;"
                                                                          withString:@""];
                    shader_src = [shader_src stringByAppendingString:@"\n"];
                    shader_src = [shader_src stringByAppendingString:voxelize_src];
                }

                error = nil;
                device->mtl_library = [mtl_device newLibraryWithSource:shader_src
                                                               options:opts
                                                                 error:&error];
                if (device->mtl_library == nil && error != nil) {
                    NSLog(@"Metal shader compile error: %@", error);
                }
            }
        }

        /* Library may be NULL if no shaders found - HAL tests still work */

        /* Store device name */
        const char* mtl_name = [[mtl_device name] UTF8String];
        if (mtl_name) {
            strncpy(device->name, mtl_name, sizeof(device->name) - 1);
        } else {
            strncpy(device->name, "Metal Device", sizeof(device->name) - 1);
        }

        return device;
    }
}

void gpu_device_destroy(GpuDevice* device) {
    if (device == NULL) return;
    /* ARC handles MTL object release */
    device->mtl_device = nil;
    device->mtl_library = nil;
    free(device);
}

const char* gpu_device_name(const GpuDevice* device) {
    if (device == NULL) return "None";
    return device->name;
}

uint32_t gpu_device_max_threadgroup_size(const GpuDevice* device) {
    if (device == NULL) return 0;
    return (uint32_t)[device->mtl_device maxThreadsPerThreadgroup].width;
}

/* ============================================================================
 * Section 4: Buffer Functions
 * ============================================================================ */

GpuBuffer* gpu_buffer_create(GpuDevice* device, size_t size, GpuMemoryMode mode) {
    if (device == NULL || size == 0) return NULL;

    @autoreleasepool {
        MTLResourceOptions options = memory_mode_to_options(mode);
        id<MTLBuffer> mtl_buffer = [device->mtl_device newBufferWithLength:size
                                                                   options:options];
        if (mtl_buffer == nil) return NULL;

        GpuBuffer* buffer = calloc(1, sizeof(GpuBuffer));
        if (buffer == NULL) return NULL;

        buffer->mtl_buffer = mtl_buffer;
        buffer->size = size;
        buffer->mode = mode;
        return buffer;
    }
}

void gpu_buffer_destroy(GpuBuffer* buffer) {
    if (buffer == NULL) return;
    buffer->mtl_buffer = nil;
    free(buffer);
}

void* gpu_buffer_map(GpuBuffer* buffer) {
    if (buffer == NULL) return NULL;
    if (buffer->mode == GPU_MEMORY_PRIVATE) return NULL;
    return [buffer->mtl_buffer contents];
}

size_t gpu_buffer_size(const GpuBuffer* buffer) {
    if (buffer == NULL) return 0;
    return buffer->size;
}

GpuResult gpu_buffer_upload(GpuBuffer* buffer, const void* data, size_t size,
                            size_t offset) {
    if (buffer == NULL || data == NULL) return GPU_ERROR_INVALID_ARG;
    if (offset + size > buffer->size) return GPU_ERROR_INVALID_ARG;
    if (buffer->mode == GPU_MEMORY_PRIVATE) return GPU_ERROR_INVALID_ARG;

    void* contents = [buffer->mtl_buffer contents];
    memcpy((uint8_t*)contents + offset, data, size);

    #if TARGET_OS_OSX
    if (buffer->mode == GPU_MEMORY_MANAGED) {
        [buffer->mtl_buffer didModifyRange:NSMakeRange(offset, size)];
    }
    #endif

    return GPU_SUCCESS;
}

GpuResult gpu_buffer_readback(const GpuBuffer* buffer, void* data, size_t size,
                              size_t offset) {
    if (buffer == NULL || data == NULL) return GPU_ERROR_INVALID_ARG;
    if (offset + size > buffer->size) return GPU_ERROR_INVALID_ARG;
    if (buffer->mode == GPU_MEMORY_PRIVATE) return GPU_ERROR_INVALID_ARG;

    const void* contents = [buffer->mtl_buffer contents];
    memcpy(data, (const uint8_t*)contents + offset, size);
    return GPU_SUCCESS;
}

/* ============================================================================
 * Section 5: Kernel Functions
 * ============================================================================ */

GpuKernel* gpu_kernel_create(GpuDevice* device, const char* function_name) {
    if (device == NULL || function_name == NULL) return NULL;
    if (device->mtl_library == nil) return NULL;

    @autoreleasepool {
        NSString* name = [NSString stringWithUTF8String:function_name];
        id<MTLFunction> function = [device->mtl_library newFunctionWithName:name];
        if (function == nil) return NULL;

        NSError* error = nil;
        id<MTLComputePipelineState> pipeline =
            [device->mtl_device newComputePipelineStateWithFunction:function
                                                             error:&error];
        if (pipeline == nil) return NULL;

        GpuKernel* kernel = calloc(1, sizeof(GpuKernel));
        if (kernel == NULL) return NULL;

        kernel->mtl_pipeline = pipeline;
        strncpy(kernel->function_name, function_name,
                sizeof(kernel->function_name) - 1);

        return kernel;
    }
}

void gpu_kernel_destroy(GpuKernel* kernel) {
    if (kernel == NULL) return;
    kernel->mtl_pipeline = nil;
    free(kernel);
}

void gpu_kernel_set_buffer(GpuKernel* kernel, uint32_t index, GpuBuffer* buffer) {
    if (kernel == NULL || index >= GPU_MAX_BUFFER_BINDINGS) return;
    kernel->bound_buffers[index] = buffer;
}

void gpu_kernel_set_constant(GpuKernel* kernel, uint32_t index,
                             const void* data, size_t size) {
    if (kernel == NULL || index >= GPU_MAX_CONSTANT_BINDINGS) return;
    if (data == NULL || size == 0 || size > GPU_MAX_CONSTANT_SIZE) return;

    memcpy(kernel->bound_constants[index].data, data, size);
    kernel->bound_constants[index].size = size;
    kernel->bound_constants[index].bound = true;
}

/* ============================================================================
 * Section 6: Command Queue Functions
 * ============================================================================ */

GpuCommandQueue* gpu_queue_create(GpuDevice* device) {
    if (device == NULL) return NULL;

    @autoreleasepool {
        id<MTLCommandQueue> mtl_queue = [device->mtl_device newCommandQueue];
        if (mtl_queue == nil) return NULL;

        GpuCommandQueue* queue = calloc(1, sizeof(GpuCommandQueue));
        if (queue == NULL) return NULL;

        queue->mtl_queue = mtl_queue;
        queue->device = device;
        return queue;
    }
}

void gpu_queue_destroy(GpuCommandQueue* queue) {
    if (queue == NULL) return;
    /* Wait for any pending work */
    if (queue->mtl_last_command_buffer != nil) {
        [queue->mtl_last_command_buffer waitUntilCompleted];
    }
    queue->mtl_queue = nil;
    queue->mtl_last_command_buffer = nil;
    free(queue);
}

GpuResult gpu_queue_dispatch(GpuCommandQueue* queue, GpuKernel* kernel,
                             uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                             uint32_t group_x, uint32_t group_y, uint32_t group_z) {
    if (queue == NULL || kernel == NULL) return GPU_ERROR_INVALID_ARG;
    if (grid_x == 0 || grid_y == 0 || grid_z == 0) return GPU_ERROR_INVALID_ARG;
    if (group_x == 0 || group_y == 0 || group_z == 0) return GPU_ERROR_INVALID_ARG;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [queue->mtl_queue commandBuffer];
        if (cmd == nil) return GPU_ERROR_DISPATCH;

        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        if (encoder == nil) return GPU_ERROR_DISPATCH;

        [encoder setComputePipelineState:kernel->mtl_pipeline];

        /* Apply buffer bindings */
        for (uint32_t i = 0; i < GPU_MAX_BUFFER_BINDINGS; i++) {
            if (kernel->bound_buffers[i] != NULL) {
                [encoder setBuffer:kernel->bound_buffers[i]->mtl_buffer
                            offset:0
                           atIndex:i];
            }
        }

        /* Apply constant bindings (after buffer slots) */
        for (uint32_t i = 0; i < GPU_MAX_CONSTANT_BINDINGS; i++) {
            if (kernel->bound_constants[i].bound) {
                [encoder setBytes:kernel->bound_constants[i].data
                           length:kernel->bound_constants[i].size
                          atIndex:GPU_MAX_BUFFER_BINDINGS + i];
            }
        }

        /* Compute grid and group sizes */
        MTLSize grid_size = MTLSizeMake(grid_x, grid_y, grid_z);
        MTLSize group_size = MTLSizeMake(group_x, group_y, group_z);

        [encoder dispatchThreads:grid_size
           threadsPerThreadgroup:group_size];

        [encoder endEncoding];
        [cmd commit];

        queue->mtl_last_command_buffer = cmd;

        return GPU_SUCCESS;
    }
}

GpuResult gpu_queue_wait(GpuCommandQueue* queue) {
    if (queue == NULL) return GPU_ERROR_INVALID_ARG;
    if (queue->mtl_last_command_buffer != nil) {
        [queue->mtl_last_command_buffer waitUntilCompleted];

        /* Check for errors */
        if ([queue->mtl_last_command_buffer status] == MTLCommandBufferStatusError) {
            return GPU_ERROR_DISPATCH;
        }
    }
    return GPU_SUCCESS;
}

GpuResult gpu_queue_poll(GpuCommandQueue* queue) {
    if (queue == NULL) return GPU_ERROR_INVALID_ARG;
    if (queue->mtl_last_command_buffer == nil) return GPU_SUCCESS;

    MTLCommandBufferStatus status = [queue->mtl_last_command_buffer status];
    if (status == MTLCommandBufferStatusCompleted) return GPU_SUCCESS;
    if (status == MTLCommandBufferStatusError) return GPU_ERROR_DISPATCH;
    return GPU_ERROR_NOT_READY;
}

/* ============================================================================
 * Section 7: Event Functions
 * ============================================================================ */

GpuEvent* gpu_event_create(GpuDevice* device) {
    if (device == NULL) return NULL;

    @autoreleasepool {
        id<MTLSharedEvent> mtl_event = [device->mtl_device newSharedEvent];
        if (mtl_event == nil) return NULL;

        GpuEvent* event = calloc(1, sizeof(GpuEvent));
        if (event == NULL) return NULL;

        event->mtl_event = mtl_event;
        return event;
    }
}

void gpu_event_destroy(GpuEvent* event) {
    if (event == NULL) return;
    event->mtl_event = nil;
    free(event);
}

GpuResult gpu_event_signal(GpuEvent* event, GpuCommandQueue* queue,
                           uint64_t value) {
    if (event == NULL || queue == NULL) return GPU_ERROR_INVALID_ARG;
    if (queue->mtl_last_command_buffer == nil) return GPU_ERROR_INVALID_ARG;

    /* Encode signal at end of last command buffer.
       Note: For proper async, signal should be encoded before commit.
       In practice we re-create command buffers per dispatch, so we
       signal on the queue's next command buffer. */
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [queue->mtl_queue commandBuffer];
        [cmd encodeSignalEvent:event->mtl_event value:value];
        [cmd commit];
        queue->mtl_last_command_buffer = cmd;
    }

    return GPU_SUCCESS;
}

GpuResult gpu_event_wait(GpuEvent* event, uint64_t value, uint64_t timeout_ms) {
    if (event == NULL) return GPU_ERROR_INVALID_ARG;

    /* Spin-wait on event value (MTLSharedEvent has no blocking wait API) */
    uint64_t start = mach_absolute_time();
    mach_timebase_info_data_t timebase = {0};
    mach_timebase_info(&timebase);

    while ([event->mtl_event signaledValue] < value) {
        if (timeout_ms == 0) return GPU_ERROR_TIMEOUT;
        if (timeout_ms != UINT64_MAX) {
            uint64_t elapsed_ns = (mach_absolute_time() - start) *
                                  timebase.numer / timebase.denom;
            if (elapsed_ns / 1000000ULL >= timeout_ms) {
                return GPU_ERROR_TIMEOUT;
            }
        }
        /* Yield to avoid spinning too hard */
        usleep(10);
    }

    return GPU_SUCCESS;
}

uint64_t gpu_event_value(const GpuEvent* event) {
    if (event == NULL) return 0;
    return [event->mtl_event signaledValue];
}
