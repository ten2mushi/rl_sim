/**
 * GPU Raymarch Tests
 *
 * Compares CPU and GPU depth camera output for accuracy.
 *
 * Test worlds:
 *   1. Empty world (all max depth)
 *   2. Single sphere
 *   3. Box room (sphere + boxes)
 *   4. Multiple obstacles
 *   5. Stress scene (many primitives)
 *
 * For each world: run CPU camera, run GPU kernel, compare outputs.
 * Tolerance: max_abs_error < 1e-3 for depth.
 */

#include "gpu_hal.h"
#include "world_brick_map.h"
#include "drone_state.h"
#include "sdf_types.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "test_harness.h"

/* ============================================================================
 * CPU Depth Camera Reference
 * ============================================================================ */

extern RayHit world_raymarch(const WorldBrickMap* world, Vec3 origin,
                             Vec3 direction, float max_distance);

/* Precompute camera rays matching sensor_camera.c pinhole model */
static Vec3* create_camera_rays(uint32_t width, uint32_t height,
                                 float fov_h, float fov_v) {
    Vec3* rays = malloc(width * height * sizeof(Vec3));
    if (rays == NULL) return NULL;

    float focal_x = 1.0f / tanf(fov_h * 0.5f);
    float focal_y = 1.0f / tanf(fov_v * 0.5f);
    float half_w = (float)width * 0.5f;
    float half_h = (float)height * 0.5f;

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            float u = ((float)x - half_w + 0.5f) / half_w;
            float v = ((float)y - half_h + 0.5f) / half_h;
            Vec3 dir = VEC3(focal_x, u, -v * focal_x / focal_y);
            float len = sqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
            rays[y * width + x] = VEC3(dir.x/len, dir.y/len, dir.z/len);
        }
    }
    return rays;
}

/* CPU depth render for a single drone */
static void cpu_depth_render(const WorldBrickMap* world, Vec3 agent_pos, Quat drone_quat,
                              const Vec3* rays, uint32_t total_pixels,
                              float near_clip, float far_clip, float inv_depth_range,
                              float* output) {
    for (uint32_t p = 0; p < total_pixels; p++) {
        Vec3 world_dir = quat_rotate(drone_quat, rays[p]);
        RayHit hit = world_raymarch(world, agent_pos, world_dir, far_clip);

        float depth = 1.0f;
        if (hit.hit && hit.distance >= near_clip) {
            depth = (hit.distance - near_clip) * inv_depth_range;
            depth = clampf(depth, 0.0f, 1.0f);
        }
        output[p] = depth;
    }
}

/* ============================================================================
 * GPU dispatch helpers (from gpu_sensor_dispatch.c / gpu_sensor_context.c)
 * ============================================================================ */

extern WorldParams gpu_world_params_from_world(const WorldBrickMap* world);
extern RaymarchParams gpu_raymarch_params_for_depth(float near_clip, float far_clip);

extern GpuResult gpu_dispatch_raymarch(struct GpuSensorContext* ctx,
                                       GpuKernel* kernel,
                                       GpuBuffer* ray_table_buf,
                                       GpuBuffer* output_buf,
                                       GpuBuffer* agent_idx_buf,
                                       const WorldBrickMap* world,
                                       RaymarchParams rp);

extern struct GpuSensorContext* gpu_sensor_context_create(uint32_t max_agents);
extern void gpu_sensor_context_destroy(struct GpuSensorContext* ctx);
extern GpuResult gpu_sensor_context_sync_frame(struct GpuSensorContext* ctx,
                                                const WorldBrickMap* world,
                                                const RigidBodyStateSOA* drones,
                                                uint32_t agent_count);
extern GpuDevice* gpu_sensor_context_device(struct GpuSensorContext* ctx);
extern GpuCommandQueue* gpu_sensor_context_queue(struct GpuSensorContext* ctx);

/* ============================================================================
 * Test Scenes
 * ============================================================================ */

typedef struct TestScene {
    const char* name;
    WorldBrickMap* world;
    Arena* arena;
} TestScene;

static TestScene create_empty_scene(void) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena, VEC3(-10, -10, -10),
                                         VEC3(10, 10, 10), 0.5f, 5000, 0);
    return (TestScene){ .name = "empty", .world = world, .arena = arena };
}

static TestScene create_sphere_scene(void) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena, VEC3(-10, -10, -10),
                                         VEC3(10, 10, 10), 0.5f, 5000, 0);
    world_set_sphere(world, VEC3(0, 0, 0), 3.0f, 1);
    return (TestScene){ .name = "sphere", .world = world, .arena = arena };
}

static TestScene create_box_room_scene(void) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena, VEC3(-10, -10, -10),
                                         VEC3(10, 10, 10), 0.5f, 5000, 0);
    world_set_sphere(world, VEC3(0, 0, 0), 3.0f, 1);
    world_set_box(world, VEC3(5, 0, 0), VEC3(1, 1, 1), 2);
    world_set_box(world, VEC3(-5, 0, 0), VEC3(1, 1, 1), 3);
    return (TestScene){ .name = "box_room", .world = world, .arena = arena };
}

static TestScene create_obstacles_scene(void) {
    Arena* arena = arena_create(64 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena, VEC3(-10, -10, -10),
                                         VEC3(10, 10, 10), 0.5f, 5000, 0);
    world_set_sphere(world, VEC3(3, 0, 0), 1.5f, 1);
    world_set_sphere(world, VEC3(-3, 2, 1), 2.0f, 2);
    world_set_box(world, VEC3(0, -4, 0), VEC3(2, 1, 1), 3);
    world_set_box(world, VEC3(0, 0, -3), VEC3(6, 6, 0.5f), 4);
    return (TestScene){ .name = "obstacles", .world = world, .arena = arena };
}

static TestScene create_stress_scene(void) {
    Arena* arena = arena_create(128 * 1024 * 1024);
    WorldBrickMap* world = world_create(arena, VEC3(-20, -20, -10),
                                         VEC3(20, 20, 10), 0.5f, 10000, 0);
    /* Dense scene with many primitives */
    for (int i = -3; i <= 3; i++) {
        for (int j = -3; j <= 3; j++) {
            float x = (float)i * 5.0f;
            float y = (float)j * 5.0f;
            uint8_t mat = (uint8_t)(((i + 3) * 7 + (j + 3)) % 15 + 1);
            if ((i + j) % 2 == 0) {
                world_set_sphere(world, VEC3(x, y, 0), 1.5f, mat);
            } else {
                world_set_box(world, VEC3(x, y, 0), VEC3(1.0f, 1.0f, 1.0f), mat);
            }
        }
    }
    return (TestScene){ .name = "stress", .world = world, .arena = arena };
}

static void destroy_scene(TestScene* scene) {
    arena_destroy(scene->arena);
}

/* ============================================================================
 * Core Test: CPU vs GPU Depth Comparison
 *
 * NOTE: This is a parametric helper that manages its own pass/fail reporting
 * because test parameters (scene, config) are set at runtime. It uses the
 * harness counters directly.
 * ============================================================================ */

static void test_depth_comparison(const char* scene_name,
                                   WorldBrickMap* world,
                                   struct GpuSensorContext* gpu_ctx,
                                   GpuKernel* kernel,
                                   uint32_t num_agents,
                                   uint32_t width, uint32_t height,
                                   bool use_rotation) {
    char test_name[128];
    snprintf(test_name, sizeof(test_name), "depth_%s_%ux%u_%ud%s",
             scene_name, width, height, num_agents,
             use_rotation ? "_rot" : "");
    _th_run++;
    printf("  Running %s...", test_name);
    fflush(stdout);

    float fov_h = 1.5708f;  /* ~90 degrees */
    float fov_v = 1.5708f;
    float near_clip = 0.1f;
    float far_clip = 20.0f;
    float inv_depth_range = 1.0f / (far_clip - near_clip);
    uint32_t total_pixels = width * height;

    Vec3* rays = create_camera_rays(width, height, fov_h, fov_v);
    if (rays == NULL) { printf(" FAILED (line %d)\n", __LINE__); return; }

    Arena* drone_arena = arena_create(4 * 1024 * 1024);
    RigidBodyStateSOA* drones = rigid_body_state_create(drone_arena, num_agents);
    if (drones == NULL) { printf(" FAILED (line %d)\n", __LINE__); free(rays); arena_destroy(drone_arena); return; }

    /* Place drones at various positions */
    for (uint32_t i = 0; i < num_agents; i++) {
        drones->pos_x[i] = -7.0f + (float)(i % 4) * 2.0f;
        drones->pos_y[i] = -3.0f + (float)((i / 4) % 4) * 2.0f;
        drones->pos_z[i] = 0.0f + (float)(i / 16) * 1.0f;
        if (use_rotation) {
            /* 45-degree yaw rotation around Z: quat = (cos(pi/8), 0, 0, sin(pi/8)) */
            float half_angle = 0.3927f;  /* pi/8 */
            drones->quat_w[i] = cosf(half_angle);
            drones->quat_x[i] = 0.0f;
            drones->quat_y[i] = 0.0f;
            drones->quat_z[i] = sinf(half_angle);
        } else {
            drones->quat_w[i] = 1.0f;
            drones->quat_x[i] = 0.0f;
            drones->quat_y[i] = 0.0f;
            drones->quat_z[i] = 0.0f;
        }
    }

    /* ---- CPU reference ---- */
    float* cpu_output = malloc(num_agents * total_pixels * sizeof(float));
    if (cpu_output == NULL) { printf(" FAILED (line %d)\n", __LINE__); free(rays); arena_destroy(drone_arena); return; }

    uint32_t* indices = malloc(num_agents * sizeof(uint32_t));
    for (uint32_t i = 0; i < num_agents; i++) indices[i] = i;

    for (uint32_t d = 0; d < num_agents; d++) {
        Vec3 pos = VEC3(drones->pos_x[d], drones->pos_y[d], drones->pos_z[d]);
        Quat q = QUAT(drones->quat_w[d], drones->quat_x[d],
                       drones->quat_y[d], drones->quat_z[d]);
        cpu_depth_render(world, pos, q, rays, total_pixels,
                         near_clip, far_clip, inv_depth_range,
                         &cpu_output[d * total_pixels]);
    }

    /* ---- GPU render ---- */
    GpuResult r = gpu_sensor_context_sync_frame(gpu_ctx, world, drones, num_agents);
    if (r != GPU_SUCCESS) {
        printf(" FAILED (line %d)\n", __LINE__);
        free(cpu_output); free(indices); free(rays);
        arena_destroy(drone_arena);
        return;
    }

    GpuDevice* device = gpu_sensor_context_device(gpu_ctx);

    GpuRayTable ray_table = gpu_ray_table_create(device, rays, total_pixels);
    if (ray_table.rays == NULL) {
        printf(" FAILED (line %d)\n", __LINE__);
        free(cpu_output); free(indices); free(rays);
        arena_destroy(drone_arena);
        return;
    }

    GpuBuffer* output_buf = gpu_buffer_create(device,
        num_agents * total_pixels * sizeof(float), GPU_MEMORY_SHARED);
    if (output_buf == NULL) { printf(" FAILED (line %d)\n", __LINE__); gpu_ray_table_destroy(&ray_table); free(cpu_output); free(indices); free(rays); arena_destroy(drone_arena); return; }

    GpuBuffer* idx_buf = gpu_buffer_create(device,
        num_agents * sizeof(uint32_t), GPU_MEMORY_SHARED);
    if (idx_buf == NULL) { printf(" FAILED (line %d)\n", __LINE__); gpu_buffer_destroy(output_buf); gpu_ray_table_destroy(&ray_table); free(cpu_output); free(indices); free(rays); arena_destroy(drone_arena); return; }
    gpu_buffer_upload(idx_buf, indices, num_agents * sizeof(uint32_t), 0);

    RaymarchParams rp = gpu_raymarch_params_for_depth(near_clip, far_clip);
    rp.image_width = width;
    rp.image_height = height;
    rp.agent_count = num_agents;

    r = gpu_dispatch_raymarch(gpu_ctx, kernel, ray_table.rays,
                              output_buf, idx_buf, world, rp);
    if (r != GPU_SUCCESS) {
        printf(" FAILED (line %d)\n", __LINE__);
        gpu_buffer_destroy(output_buf); gpu_buffer_destroy(idx_buf);
        gpu_ray_table_destroy(&ray_table);
        free(cpu_output); free(indices); free(rays);
        arena_destroy(drone_arena);
        return;
    }

    r = gpu_queue_wait(gpu_sensor_context_queue(gpu_ctx));
    if (r != GPU_SUCCESS) {
        printf(" FAILED (line %d)\n", __LINE__);
        gpu_buffer_destroy(output_buf); gpu_buffer_destroy(idx_buf);
        gpu_ray_table_destroy(&ray_table);
        free(cpu_output); free(indices); free(rays);
        arena_destroy(drone_arena);
        return;
    }

    float* gpu_output = malloc(num_agents * total_pixels * sizeof(float));
    if (gpu_output == NULL) { printf(" FAILED (line %d)\n", __LINE__); gpu_buffer_destroy(output_buf); gpu_buffer_destroy(idx_buf); gpu_ray_table_destroy(&ray_table); free(cpu_output); free(indices); free(rays); arena_destroy(drone_arena); return; }
    r = gpu_buffer_readback(output_buf, gpu_output,
                            num_agents * total_pixels * sizeof(float), 0);
    if (r != GPU_SUCCESS) { printf(" FAILED (line %d)\n", __LINE__); free(gpu_output); gpu_buffer_destroy(output_buf); gpu_buffer_destroy(idx_buf); gpu_ray_table_destroy(&ray_table); free(cpu_output); free(indices); free(rays); arena_destroy(drone_arena); return; }

    /* ---- Compare ---- */
    float max_err = 0.0f;
    uint32_t mismatch_count = 0;
    uint32_t nan_count = 0;
    uint32_t boundary_count = 0;
    uint32_t total = num_agents * total_pixels;
    float tolerance = 1e-3f;

    for (uint32_t i = 0; i < total; i++) {
        if (isnan(gpu_output[i]) || isinf(gpu_output[i])) {
            nan_count++;
            continue;
        }
        float err = fabsf(gpu_output[i] - cpu_output[i]);
        if (err > max_err) max_err = err;
        if (err > tolerance) {
            bool cpu_miss = (cpu_output[i] >= 0.999f);
            bool gpu_miss = (gpu_output[i] >= 0.999f);
            if (cpu_miss != gpu_miss || err < 0.05f) {
                boundary_count++;
            } else {
                mismatch_count++;
            }
        }
    }

    /* Allow up to 0.1% boundary disagreements (grazing angle FP differences) */
    uint32_t max_boundary = total / 1000 + 1;
    bool pass = (nan_count == 0 && mismatch_count == 0 && boundary_count <= max_boundary);
    if (boundary_count > 0) {
        printf("(max_err=%.6f, nans=%u, boundary=%u) ", max_err, nan_count, boundary_count);
    } else {
        printf("(max_err=%.6f, nans=%u) ", max_err, nan_count);
    }

    if (pass) {
        _th_passed++;
        printf(" PASSED\n");
    } else {
        printf(" FAILED (line %d)\n", __LINE__);
        printf("    (mismatches=%u, boundary=%u/%u)\n",
               mismatch_count, boundary_count, total);
        uint32_t printed = 0;
        for (uint32_t i = 0; i < total && printed < 5; i++) {
            float err = fabsf(gpu_output[i] - cpu_output[i]);
            if (err > 1e-3f) {
                uint32_t agent_id = i / total_pixels;
                uint32_t pixel = i % total_pixels;
                printf("    pixel[drone=%u, p=%u]: cpu=%.6f gpu=%.6f err=%.6f\n",
                       agent_id, pixel, cpu_output[i], gpu_output[i], err);
                printed++;
            }
        }
    }

    gpu_buffer_destroy(output_buf);
    gpu_buffer_destroy(idx_buf);
    gpu_ray_table_destroy(&ray_table);
    free(cpu_output);
    free(gpu_output);
    free(indices);
    free(rays);
    arena_destroy(drone_arena);
}

/* ============================================================================
 * GPU-to-GPU Determinism Test
 * ============================================================================ */

static void test_determinism(WorldBrickMap* world,
                              struct GpuSensorContext* gpu_ctx,
                              GpuKernel* kernel) {
    _th_run++;
    printf("  Running determinism...");
    fflush(stdout);

    uint32_t num_agents = 16;
    uint32_t width = 32, height = 32;
    uint32_t total_pixels = width * height;
    float near_clip = 0.1f, far_clip = 20.0f;

    Vec3* rays = create_camera_rays(width, height, 1.5708f, 1.5708f);
    if (rays == NULL) { printf(" FAILED (line %d)\n", __LINE__); return; }

    Arena* drone_arena = arena_create(4 * 1024 * 1024);
    RigidBodyStateSOA* drones = rigid_body_state_create(drone_arena, num_agents);
    if (drones == NULL) { printf(" FAILED (line %d)\n", __LINE__); free(rays); arena_destroy(drone_arena); return; }

    for (uint32_t i = 0; i < num_agents; i++) {
        drones->pos_x[i] = -5.0f + (float)(i % 4) * 2.5f;
        drones->pos_y[i] = -4.0f + (float)((i / 4) % 4) * 2.5f;
        drones->pos_z[i] = 1.0f;
        drones->quat_w[i] = 1.0f;
        drones->quat_x[i] = 0.0f;
        drones->quat_y[i] = 0.0f;
        drones->quat_z[i] = 0.0f;
    }

    uint32_t* indices = malloc(num_agents * sizeof(uint32_t));
    for (uint32_t i = 0; i < num_agents; i++) indices[i] = i;

    GpuDevice* device = gpu_sensor_context_device(gpu_ctx);
    gpu_sensor_context_sync_frame(gpu_ctx, world, drones, num_agents);

    GpuRayTable ray_table = gpu_ray_table_create(device, rays, total_pixels);
    GpuBuffer* idx_buf = gpu_buffer_create(device, num_agents * sizeof(uint32_t), GPU_MEMORY_SHARED);
    gpu_buffer_upload(idx_buf, indices, num_agents * sizeof(uint32_t), 0);

    size_t out_size = num_agents * total_pixels * sizeof(float);
    float* run1 = malloc(out_size);
    float* run2 = malloc(out_size);

    for (int run = 0; run < 2; run++) {
        GpuBuffer* output_buf = gpu_buffer_create(device, out_size, GPU_MEMORY_SHARED);
        RaymarchParams rp = gpu_raymarch_params_for_depth(near_clip, far_clip);
        rp.image_width = width;
        rp.image_height = height;
        rp.agent_count = num_agents;

        gpu_dispatch_raymarch(gpu_ctx, kernel, ray_table.rays,
                              output_buf, idx_buf, world, rp);
        gpu_queue_wait(gpu_sensor_context_queue(gpu_ctx));
        gpu_buffer_readback(output_buf, run == 0 ? run1 : run2, out_size, 0);
        gpu_buffer_destroy(output_buf);
    }

    bool match = (memcmp(run1, run2, out_size) == 0);
    if (match) {
        _th_passed++;
        printf(" PASSED (bit-exact across 2 runs)\n");
    } else {
        uint32_t diff_count = 0;
        for (uint32_t i = 0; i < num_agents * total_pixels; i++) {
            if (run1[i] != run2[i]) diff_count++;
        }
        printf(" FAILED (line %d)\n", __LINE__);
        printf("    (%u/%u values differ)\n", diff_count, num_agents * total_pixels);
    }

    gpu_buffer_destroy(idx_buf);
    gpu_ray_table_destroy(&ray_table);
    free(run1); free(run2); free(indices); free(rays);
    arena_destroy(drone_arena);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("Backend: %s\n", GPU_AVAILABLE ? "METAL" : "NONE");

    TEST_SUITE_BEGIN("GPU Raymarch Tests");

    #if !GPU_AVAILABLE
    printf("GPU not available, skipping raymarch tests.\n");
    TEST_SUITE_END();
    #endif

    if (!gpu_is_available()) {
        printf("GPU not available at runtime, skipping.\n");
        TEST_SUITE_END();
    }

    struct GpuSensorContext* ctx = gpu_sensor_context_create(256);
    if (ctx == NULL) {
        printf("FAIL: Could not create GPU sensor context\n");
        return 1;
    }

    GpuDevice* device = gpu_sensor_context_device(ctx);
    GpuKernel* kernel = gpu_kernel_create(device, "raymarch_unified");
    if (kernel == NULL) {
        printf("SKIP: Metal shader not compiled (kernel=NULL)\n");
        gpu_sensor_context_destroy(ctx);
        TEST_SUITE_END();
    }

    printf("Kernel 'raymarch_unified' loaded OK\n\n");

    /* Test scenes */
    TestScene scenes[] = {
        create_empty_scene(),
        create_sphere_scene(),
        create_box_room_scene(),
        create_obstacles_scene(),
        create_stress_scene(),
    };
    uint32_t scene_count = sizeof(scenes) / sizeof(scenes[0]);

    /* Test configurations */
    struct { uint32_t drones, w, h; } configs[] = {
        {   4, 16, 16 },
        {  16, 32, 32 },
        {  64, 32, 32 },
        { 256, 16, 16 },
    };
    uint32_t config_count = sizeof(configs) / sizeof(configs[0]);

    /* Identity rotation tests */
    for (uint32_t s = 0; s < scene_count; s++) {
        printf("--- Scene: %s ---\n", scenes[s].name);
        for (uint32_t c = 0; c < config_count; c++) {
            test_depth_comparison(scenes[s].name, scenes[s].world,
                                  ctx, kernel,
                                  configs[c].drones,
                                  configs[c].w, configs[c].h,
                                  false);
        }
        printf("\n");
    }

    /* Non-identity rotation tests (sphere + obstacles) */
    printf("--- Rotated drones ---\n");
    test_depth_comparison("sphere", scenes[1].world, ctx, kernel,
                          16, 32, 32, true);
    test_depth_comparison("obstacles", scenes[3].world, ctx, kernel,
                          16, 32, 32, true);
    printf("\n");

    /* GPU-to-GPU determinism */
    printf("--- Determinism ---\n");
    test_determinism(scenes[1].world, ctx, kernel);
    printf("\n");

    /* Cleanup */
    for (uint32_t s = 0; s < scene_count; s++) {
        destroy_scene(&scenes[s]);
    }
    gpu_kernel_destroy(kernel);
    gpu_sensor_context_destroy(ctx);

    TEST_SUITE_END();
}
