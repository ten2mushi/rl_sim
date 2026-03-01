/**
 * GPU HAL Tests
 *
 * Tests the hardware abstraction layer:
 * - Device creation and query
 * - Buffer create/map/upload/readback
 * - SDF atlas flatten and readback comparison
 * - Drone pose upload/readback
 * - Ray table creation
 *
 * Works with both METAL and NONE backends.
 * NONE backend: verifies graceful failure (all NULL returns).
 */

#include "gpu_hal.h"
#include "world_brick_map.h"
#include "drone_state.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "test_harness.h"

/* ============================================================================
 * Test: Backend Availability
 * ============================================================================ */

/* Global: tracks if Metal actually works at runtime */
static bool g_metal_runtime_available = false;

TEST(availability) {
    bool available = gpu_is_available();
    g_metal_runtime_available = available;
    printf("(available=%s) ", available ? "yes" : "no");

    #if GPU_AVAILABLE
    /* Metal backend compiled, but device may not be available at runtime
     * (e.g., headless/sandboxed environment) */
    if (!available) {
        printf("(runtime unavailable - headless?) ");
    }
    #else
    ASSERT_MSG(!available, "GPU should not be available with NONE backend");
    #endif

    return 0;
}

/* ============================================================================
 * Test: Device Creation
 * ============================================================================ */

TEST(device_create) {
    GpuDevice* device = gpu_device_create();

    #if GPU_AVAILABLE
    if (!g_metal_runtime_available) {
        ASSERT_MSG(device == NULL, "Device should be NULL when runtime unavailable");
        printf("(skipped - no runtime) ");
        return 0;
    }

    ASSERT_MSG(device != NULL, "Device creation should succeed on Metal");

    const char* name = gpu_device_name(device);
    ASSERT_MSG(name != NULL, "Device name should not be NULL");
    printf("(name='%s') ", name);

    uint32_t max_tg = gpu_device_max_threadgroup_size(device);
    ASSERT_MSG(max_tg > 0, "Max threadgroup size should be > 0");
    printf("(max_tg=%u) ", max_tg);

    gpu_device_destroy(device);
    #else
    ASSERT_MSG(device == NULL, "Device should be NULL with NONE backend");
    #endif

    return 0;
}

/* ============================================================================
 * Test: Buffer Create/Map/Upload/Readback
 * ============================================================================ */

TEST(buffer_operations) {
    #if !GPU_AVAILABLE
    /* NONE backend: verify graceful failure */
    GpuBuffer* buf = gpu_buffer_create(NULL, 1024, GPU_MEMORY_SHARED);
    ASSERT_MSG(buf == NULL, "Buffer should be NULL with NONE backend");
    return 0;
    #endif

    GpuDevice* device = gpu_device_create();
    ASSERT_MSG(device != NULL, "Need device for buffer test");

    /* Create shared buffer */
    size_t buf_size = 4096;
    GpuBuffer* buffer = gpu_buffer_create(device, buf_size, GPU_MEMORY_SHARED);
    ASSERT_MSG(buffer != NULL, "Buffer creation should succeed");
    ASSERT_MSG(gpu_buffer_size(buffer) == buf_size, "Buffer size mismatch");

    /* Map and verify writable */
    void* ptr = gpu_buffer_map(buffer);
    ASSERT_MSG(ptr != NULL, "Map should return non-NULL for shared buffer");

    /* Upload test data */
    float test_data[256];
    for (int i = 0; i < 256; i++) {
        test_data[i] = (float)i * 0.5f;
    }
    GpuResult r = gpu_buffer_upload(buffer, test_data, sizeof(test_data), 0);
    ASSERT_MSG(r == GPU_SUCCESS, "Upload should succeed");

    /* Readback and verify */
    float readback[256];
    r = gpu_buffer_readback(buffer, readback, sizeof(readback), 0);
    ASSERT_MSG(r == GPU_SUCCESS, "Readback should succeed");

    for (int i = 0; i < 256; i++) {
        if (readback[i] != test_data[i]) {
            gpu_buffer_destroy(buffer);
            gpu_device_destroy(device);
            ASSERT_MSG(0, "Readback data mismatch");
        }
    }

    /* Upload with offset */
    float offset_data[16] = {99.0f, 98.0f, 97.0f, 96.0f,
                             95.0f, 94.0f, 93.0f, 92.0f,
                             91.0f, 90.0f, 89.0f, 88.0f,
                             87.0f, 86.0f, 85.0f, 84.0f};
    r = gpu_buffer_upload(buffer, offset_data, sizeof(offset_data), 512);
    ASSERT_MSG(r == GPU_SUCCESS, "Upload with offset should succeed");

    float offset_readback[16];
    r = gpu_buffer_readback(buffer, offset_readback, sizeof(offset_readback), 512);
    ASSERT_MSG(r == GPU_SUCCESS, "Readback with offset should succeed");
    ASSERT_MSG(memcmp(offset_data, offset_readback, sizeof(offset_data)) == 0,
                "Offset readback mismatch");

    /* Error cases */
    r = gpu_buffer_upload(buffer, test_data, buf_size + 1, 0);
    ASSERT_MSG(r != GPU_SUCCESS, "Oversized upload should fail");

    gpu_buffer_destroy(buffer);

    /* Private buffer: map should return NULL */
    GpuBuffer* priv_buf = gpu_buffer_create(device, 1024, GPU_MEMORY_PRIVATE);
    if (priv_buf != NULL) {
        void* priv_ptr = gpu_buffer_map(priv_buf);
        ASSERT_MSG(priv_ptr == NULL, "Map of private buffer should return NULL");
        gpu_buffer_destroy(priv_buf);
    }

    gpu_device_destroy(device);
    return 0;
}

/* ============================================================================
 * Test: SDF Atlas Flatten and Readback
 * ============================================================================ */

TEST(sdf_atlas) {
    #if !GPU_AVAILABLE
    GpuSdfAtlas atlas = gpu_sdf_atlas_upload(NULL, NULL);
    ASSERT_MSG(atlas.sdf_data == NULL, "Atlas should be empty with NONE backend");
    return 0;
    #endif

    GpuDevice* device = gpu_device_create();
    ASSERT_MSG(device != NULL, "Need device for atlas test");

    /* Create a small test world */
    Arena* arena = arena_create(64 * 1024 * 1024);
    ASSERT_MSG(arena != NULL, "Arena creation should succeed");

    WorldBrickMap* world = world_create(arena, VEC3(-10, -10, -10),
                                         VEC3(10, 10, 10), 0.5f, 5000, 0);
    ASSERT_MSG(world != NULL, "World creation should succeed");

    /* Add some geometry */
    world_set_sphere(world, VEC3(0, 0, 0), 3.0f, 1);
    world_set_box(world, VEC3(5, 0, 0), VEC3(1, 1, 1), 2);

    WorldStats stats = world_get_stats(world);
    printf("(bricks=%u) ", stats.active_bricks);

    /* Upload to GPU */
    GpuSdfAtlas atlas = gpu_sdf_atlas_upload(device, world);
    ASSERT_MSG(atlas.sdf_data != NULL, "Atlas SDF data should not be NULL");
    ASSERT_MSG(atlas.material_data != NULL, "Atlas material data should not be NULL");
    ASSERT_MSG(atlas.brick_indices != NULL, "Atlas indices should not be NULL");
    ASSERT_MSG(atlas.brick_count > 0, "Atlas should have active bricks");
    ASSERT_MSG(atlas.grid_total == world->grid_total, "Grid total mismatch");

    printf("(gpu_bricks=%u) ", atlas.brick_count);

    /* Readback and compare a few bricks */
    bool data_match = true;
    for (uint32_t i = 0; i < atlas.brick_count && i < 10; i++) {
        int8_t gpu_sdf[BRICK_VOXELS];
        GpuResult r = gpu_buffer_readback(atlas.sdf_data, gpu_sdf,
                                          BRICK_VOXELS, (size_t)i * BRICK_VOXELS);
        if (r != GPU_SUCCESS) {
            data_match = false;
            break;
        }

        /* Compare with CPU data */
        const int8_t* cpu_sdf = world_brick_sdf_const(world, (int32_t)i);
        if (cpu_sdf != NULL) {
            if (memcmp(gpu_sdf, cpu_sdf, BRICK_VOXELS) != 0) {
                data_match = false;
                break;
            }
        }
    }
    ASSERT_MSG(data_match, "GPU atlas data should match CPU data");

    /* Compare brick indices */
    int32_t* gpu_indices = malloc(world->grid_total * sizeof(int32_t));
    ASSERT_MSG(gpu_indices != NULL, "Malloc for indices");
    GpuResult r = gpu_buffer_readback(atlas.brick_indices, gpu_indices,
                                      world->grid_total * sizeof(int32_t), 0);
    ASSERT_MSG(r == GPU_SUCCESS, "Index readback should succeed");
    ASSERT_MSG(memcmp(gpu_indices, world->brick_indices,
                       world->grid_total * sizeof(int32_t)) == 0,
                "Brick indices should match");
    free(gpu_indices);

    gpu_sdf_atlas_destroy(&atlas);
    arena_destroy(arena);
    gpu_device_destroy(device);
    return 0;
}

/* ============================================================================
 * Test: Drone Poses Upload/Readback
 * ============================================================================ */

TEST(agent_poses) {
    #if !GPU_AVAILABLE
    GpuDronePoses poses = gpu_agent_poses_create(NULL, 32);
    ASSERT_MSG(poses.pos_x == NULL, "Poses should be empty with NONE backend");
    return 0;
    #endif

    GpuDevice* device = gpu_device_create();
    ASSERT_MSG(device != NULL, "Need device");

    uint32_t num_agents = 64;

    /* Create test drone state */
    Arena* arena = arena_create(16 * 1024 * 1024);
    RigidBodyStateSOA* drones = rigid_body_state_create(arena, num_agents);
    ASSERT_MSG(drones != NULL, "Drone state creation");

    /* Fill with test data */
    for (uint32_t i = 0; i < num_agents; i++) {
        drones->pos_x[i] = (float)i * 0.1f;
        drones->pos_y[i] = (float)i * 0.2f;
        drones->pos_z[i] = (float)i * 0.3f;
        drones->quat_w[i] = 1.0f;
        drones->quat_x[i] = 0.0f;
        drones->quat_y[i] = 0.0f;
        drones->quat_z[i] = 0.0f;
    }

    /* Upload to GPU */
    GpuDronePoses poses = gpu_agent_poses_create(device, num_agents);
    ASSERT_MSG(poses.pos_x != NULL, "Pose buffers should be created");
    ASSERT_MSG(poses.max_agents == num_agents, "Max drones mismatch");

    GpuResult r = gpu_agent_poses_upload(&poses, drones, num_agents);
    ASSERT_MSG(r == GPU_SUCCESS, "Pose upload should succeed");

    /* Readback and verify */
    float readback[64];
    r = gpu_buffer_readback(poses.pos_x, readback, num_agents * sizeof(float), 0);
    ASSERT_MSG(r == GPU_SUCCESS, "Pos_x readback");

    bool match = true;
    for (uint32_t i = 0; i < num_agents; i++) {
        if (fabsf(readback[i] - drones->pos_x[i]) > 1e-6f) {
            match = false;
            break;
        }
    }
    ASSERT_MSG(match, "Pose pos_x data should match");

    gpu_agent_poses_destroy(&poses);
    arena_destroy(arena);
    gpu_device_destroy(device);
    return 0;
}

/* ============================================================================
 * Test: Ray Table Creation
 * ============================================================================ */

TEST(ray_table) {
    #if !GPU_AVAILABLE
    GpuRayTable table = gpu_ray_table_create(NULL, NULL, 0);
    ASSERT_MSG(table.rays == NULL, "Table should be empty with NONE backend");
    return 0;
    #endif

    GpuDevice* device = gpu_device_create();
    ASSERT_MSG(device != NULL, "Need device");

    /* Create test ray directions */
    uint32_t num_rays = 128;
    Vec3 dirs[128];
    for (uint32_t i = 0; i < num_rays; i++) {
        float angle = (float)i / (float)num_rays * 6.28318f;
        dirs[i] = VEC3(cosf(angle), sinf(angle), 0.0f);
    }

    GpuRayTable table = gpu_ray_table_create(device, dirs, num_rays);
    ASSERT_MSG(table.rays != NULL, "Ray table should be created");
    ASSERT_MSG(table.ray_count == num_rays, "Ray count mismatch");

    /* Readback and verify (float4 layout) */
    float readback[128 * 4];
    GpuResult r = gpu_buffer_readback(table.rays, readback,
                                      num_rays * 4 * sizeof(float), 0);
    ASSERT_MSG(r == GPU_SUCCESS, "Ray readback");

    bool match = true;
    for (uint32_t i = 0; i < num_rays; i++) {
        if (fabsf(readback[i*4+0] - dirs[i].x) > 1e-6f ||
            fabsf(readback[i*4+1] - dirs[i].y) > 1e-6f ||
            fabsf(readback[i*4+2] - dirs[i].z) > 1e-6f ||
            readback[i*4+3] != 0.0f) {
            match = false;
            break;
        }
    }
    ASSERT_MSG(match, "Ray directions should match with float4 layout");

    gpu_ray_table_destroy(&table);
    gpu_device_destroy(device);
    return 0;
}

/* ============================================================================
 * Test: Error Strings
 * ============================================================================ */

TEST(error_strings) {
    const char* s;
    s = gpu_error_string(GPU_SUCCESS);
    ASSERT_MSG(s != NULL && strlen(s) > 0, "Success string");

    s = gpu_error_string(GPU_ERROR_NO_DEVICE);
    ASSERT_MSG(s != NULL && strlen(s) > 0, "No device string");

    s = gpu_error_string(GPU_ERROR_BACKEND);
    ASSERT_MSG(s != NULL && strlen(s) > 0, "Backend string");

    s = gpu_error_string(-999);
    ASSERT_MSG(s != NULL && strlen(s) > 0, "Unknown error string");

    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("Backend: %s\n", GPU_AVAILABLE ? "METAL" : "NONE");

    TEST_SUITE_BEGIN("GPU HAL Tests");

    RUN_TEST(error_strings);
    RUN_TEST(availability);
    RUN_TEST(device_create);
    RUN_TEST(buffer_operations);
    RUN_TEST(sdf_atlas);
    RUN_TEST(agent_poses);
    RUN_TEST(ray_table);

    TEST_SUITE_END();
}
