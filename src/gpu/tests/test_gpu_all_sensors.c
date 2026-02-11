/**
 * GPU All-Sensors Tests
 *
 * Tests all 6 GPU-accelerated sensor types against CPU reference AND ground truth.
 *
 * Test strategy:
 *   1. GPU vs CPU comparison: run same rays through both, compare outputs
 *   2. Ground truth: use known geometry to verify both CPU and GPU are correct
 *
 * Sensor types tested:
 *   - Camera Depth   (OUTPUT_MODE_DEPTH)
 *   - Camera RGB     (OUTPUT_MODE_RGB)
 *   - Camera Seg     (OUTPUT_MODE_MATERIAL)
 *   - LiDAR 3D       (OUTPUT_MODE_DISTANCE)
 *   - LiDAR 2D       (OUTPUT_MODE_DISTANCE)
 *   - ToF             (OUTPUT_MODE_DISTANCE)
 *
 * Ground truth scene: sphere at (5,0,0) radius 2, material 5 (red).
 * Drone at origin, identity orientation, camera looks along +X.
 * Forward ray should hit at distance ~3.0.
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * CPU Reference Functions
 * ============================================================================ */

extern RayHit world_raymarch(const WorldBrickMap* world, Vec3 origin,
                             Vec3 direction, float max_distance);
extern Vec3 world_sdf_normal(const WorldBrickMap* world, Vec3 pos);
extern uint8_t world_material_query(const WorldBrickMap* world, Vec3 pos);

/* Material palette (must match sdf_types.h MATERIAL_PALETTE_*) */
static const float MAT_R[16] = MATERIAL_PALETTE_R;
static const float MAT_G[16] = MATERIAL_PALETTE_G;
static const float MAT_B[16] = MATERIAL_PALETTE_B;

/* Light direction (precomputed, matches raymarch.metal) */
static const Vec3 LIGHT_DIR_VEC = {0.43193421f, 0.25916053f, 0.86386843f};

/* ============================================================================
 * Ray Generation Helpers
 * ============================================================================ */

/* Pinhole camera rays matching precompute_camera_rays in sensor_camera.c */
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

/* LiDAR 2D: horizontal plane rays */
static Vec3* create_lidar_2d_rays(uint32_t num_rays, float fov) {
    Vec3* rays = malloc(num_rays * sizeof(Vec3));
    if (rays == NULL) return NULL;

    float half_fov = fov * 0.5f;
    float step = (num_rays > 1) ? fov / (float)(num_rays - 1) : 0.0f;

    for (uint32_t i = 0; i < num_rays; i++) {
        float angle = -half_fov + step * (float)i;
        rays[i] = VEC3(cosf(angle), sinf(angle), 0.0f);
    }
    return rays;
}

/* LiDAR 3D: spherical grid rays */
static Vec3* create_lidar_3d_rays(uint32_t h_rays, uint32_t v_layers,
                                   float h_fov, float v_fov) {
    uint32_t total = h_rays * v_layers;
    Vec3* rays = malloc(total * sizeof(Vec3));
    if (rays == NULL) return NULL;

    float half_h = h_fov * 0.5f;
    float half_v = v_fov * 0.5f;
    float h_step = (h_rays > 1) ? h_fov / (float)(h_rays - 1) : 0.0f;
    float v_step = (v_layers > 1) ? v_fov / (float)(v_layers - 1) : 0.0f;

    for (uint32_t v = 0; v < v_layers; v++) {
        float el = -half_v + v_step * (float)v;
        float cos_el = cosf(el);
        float sin_el = sinf(el);
        for (uint32_t h = 0; h < h_rays; h++) {
            float az = -half_h + h_step * (float)h;
            uint32_t idx = v * h_rays + h;
            rays[idx] = VEC3(cos_el * cosf(az), cos_el * sinf(az), sin_el);
        }
    }
    return rays;
}

/* ============================================================================
 * CPU Rendering Functions (match GPU shader exactly)
 * ============================================================================ */

static void cpu_render_depth(const WorldBrickMap* world, Vec3 pos, Quat q,
                              const Vec3* rays, uint32_t total_rays,
                              float near_clip, float far_clip, float* output) {
    float inv_range = 1.0f / (far_clip - near_clip);
    for (uint32_t p = 0; p < total_rays; p++) {
        Vec3 world_dir = quat_rotate(q, rays[p]);
        RayHit hit = world_raymarch(world, pos, world_dir, far_clip);
        float depth = 1.0f;
        if (hit.hit && hit.distance >= near_clip) {
            depth = (hit.distance - near_clip) * inv_range;
            depth = clampf(depth, 0.0f, 1.0f);
        }
        output[p] = depth;
    }
}

static void cpu_render_rgb(const WorldBrickMap* world, Vec3 pos, Quat q,
                            const Vec3* rays, uint32_t total_rays,
                            float near_clip, float far_clip, float* output) {
    for (uint32_t p = 0; p < total_rays; p++) {
        Vec3 world_dir = quat_rotate(q, rays[p]);
        RayHit hit = world_raymarch(world, pos, world_dir, far_clip);
        float* pixel = &output[p * 3];

        if (hit.hit && hit.distance >= near_clip) {
            uint8_t mat = hit.material;
            if (mat > 15) mat = 0;

            Vec3 normal = world_sdf_normal(world, hit.position);
            float ndotl = clampf(vec3_dot(normal, LIGHT_DIR_VEC), 0.0f, 1.0f);
            float lighting = 0.3f + 0.7f * ndotl;

            pixel[0] = MAT_R[mat] * lighting;
            pixel[1] = MAT_G[mat] * lighting;
            pixel[2] = MAT_B[mat] * lighting;
        } else {
            pixel[0] = SKY_COLOR_R;
            pixel[1] = SKY_COLOR_G;
            pixel[2] = SKY_COLOR_B;
        }
    }
}

static void cpu_render_material(const WorldBrickMap* world, Vec3 pos, Quat q,
                                 const Vec3* rays, uint32_t total_rays,
                                 float near_clip, float far_clip, float* output) {
    for (uint32_t p = 0; p < total_rays; p++) {
        Vec3 world_dir = quat_rotate(q, rays[p]);
        RayHit hit = world_raymarch(world, pos, world_dir, far_clip);
        if (hit.hit && hit.distance >= near_clip) {
            output[p] = (float)hit.material;
        } else {
            output[p] = 0.0f;
        }
    }
}

static void cpu_render_distance(const WorldBrickMap* world, Vec3 pos, Quat q,
                                 const Vec3* rays, uint32_t total_rays,
                                 float max_range, float* output) {
    for (uint32_t p = 0; p < total_rays; p++) {
        Vec3 world_dir = quat_rotate(q, rays[p]);
        RayHit hit = world_raymarch(world, pos, world_dir, max_range);
        output[p] = hit.hit ? hit.distance : max_range;
    }
}

/* ============================================================================
 * GPU Dispatch Externs
 * ============================================================================ */

extern WorldParams gpu_world_params_from_world(const WorldBrickMap* world);
extern RaymarchParams gpu_raymarch_params_for_depth(float near_clip, float far_clip);
extern RaymarchParams gpu_raymarch_params_for_rgb(float near_clip, float far_clip);
extern RaymarchParams gpu_raymarch_params_for_material(float near_clip, float far_clip);
extern RaymarchParams gpu_raymarch_params_for_distance(float max_range);

extern GpuResult gpu_dispatch_raymarch(struct GpuSensorContext* ctx,
                                       GpuKernel* kernel,
                                       GpuBuffer* ray_table_buf,
                                       GpuBuffer* output_buf,
                                       GpuBuffer* drone_idx_buf,
                                       const WorldBrickMap* world,
                                       RaymarchParams rp);

extern struct GpuSensorContext* gpu_sensor_context_create(uint32_t max_drones);
extern void gpu_sensor_context_destroy(struct GpuSensorContext* ctx);
extern GpuResult gpu_sensor_context_sync_frame(struct GpuSensorContext* ctx,
                                                const WorldBrickMap* world,
                                                const DroneStateSOA* drones,
                                                uint32_t drone_count);
extern GpuDevice* gpu_sensor_context_device(struct GpuSensorContext* ctx);
extern GpuCommandQueue* gpu_sensor_context_queue(struct GpuSensorContext* ctx);

/* ============================================================================
 * Test Scene
 * ============================================================================ */

typedef struct TestScene {
    const char* name;
    WorldBrickMap* world;
    Arena* arena;
} TestScene;

/* Sphere at (5,0,0) r=2 material 5 (red) in a +/-10 world */
static TestScene create_sphere_scene(void) {
    TestScene scene = {.name = "sphere"};
    scene.arena = arena_create(1024 * 1024);
    if (scene.arena == NULL) return scene;

    scene.world = world_create(scene.arena, VEC3(-10,-10,-10), VEC3(10,10,10),
                                0.5f, 5000, 0);
    if (scene.world == NULL) return scene;

    world_set_sphere(scene.world, VEC3(5.0f, 0.0f, 0.0f), 2.0f, 5);
    return scene;
}

/* Box room + sphere for multi-surface testing */
static TestScene create_room_scene(void) {
    TestScene scene = {.name = "room"};
    scene.arena = arena_create(2 * 1024 * 1024);
    if (scene.arena == NULL) return scene;

    scene.world = world_create(scene.arena, VEC3(-10,-10,-10), VEC3(10,10,10),
                                0.5f, 5000, 0);
    if (scene.world == NULL) return scene;

    /* Floor at z=-5, wall at x=8, sphere at (3,0,0) */
    world_set_box(scene.world, VEC3(0, 0, -6), VEC3(20, 20, 2), 1);
    world_set_box(scene.world, VEC3(8, 0, 0), VEC3(2, 20, 20), 3);
    world_set_sphere(scene.world, VEC3(3.0f, 0.0f, 0.0f), 1.5f, 5);
    return scene;
}

static void destroy_scene(TestScene* scene) {
    if (scene->arena) arena_destroy(scene->arena);
    scene->world = NULL;
    scene->arena = NULL;
}

/* ============================================================================
 * DroneStateSOA Helper
 * ============================================================================ */

#define MAX_TEST_DRONES 1024

typedef struct TestDrones {
    DroneStateSOA soa;
    float px[MAX_TEST_DRONES], py[MAX_TEST_DRONES], pz[MAX_TEST_DRONES];
    float qw[MAX_TEST_DRONES], qx[MAX_TEST_DRONES];
    float qy[MAX_TEST_DRONES], qz[MAX_TEST_DRONES];
} TestDrones;

static void init_test_drones(TestDrones* td, uint32_t count) {
    memset(td, 0, sizeof(TestDrones));
    td->soa.pos_x = td->px;
    td->soa.pos_y = td->py;
    td->soa.pos_z = td->pz;
    td->soa.quat_w = td->qw;
    td->soa.quat_x = td->qx;
    td->soa.quat_y = td->qy;
    td->soa.quat_z = td->qz;

    for (uint32_t i = 0; i < count; i++) {
        td->qw[i] = 1.0f;
    }
}

/* ============================================================================
 * Comparison Helpers
 * ============================================================================ */

typedef struct CompareResult {
    float max_err;
    uint32_t nan_count;
    uint32_t mismatch_count;
    uint32_t boundary_count;
    bool pass;
} CompareResult;

/* Compare single-float-per-pixel outputs (depth, material, distance) */
static CompareResult compare_1fpp(const float* cpu, const float* gpu,
                                   uint32_t total, float tolerance,
                                   bool exact_match) {
    CompareResult r = {0};
    for (uint32_t i = 0; i < total; i++) {
        if (isnan(gpu[i]) || isinf(gpu[i])) { r.nan_count++; continue; }

        float err = fabsf(cpu[i] - gpu[i]);
        if (err > r.max_err) r.max_err = err;

        if (exact_match) {
            if (cpu[i] != gpu[i]) {
                if (err < 0.05f) r.boundary_count++;
                else r.mismatch_count++;
            }
        } else {
            if (err > tolerance) {
                if (err < 0.05f) r.boundary_count++;
                else r.mismatch_count++;
            }
        }
    }
    uint32_t max_boundary = total / 1000 + 1;
    r.pass = (r.nan_count == 0 && r.mismatch_count == 0 &&
              r.boundary_count <= max_boundary);
    return r;
}

/* Compare RGB outputs (3 floats per pixel) */
static CompareResult compare_rgb(const float* cpu, const float* gpu,
                                  uint32_t total_pixels, float tolerance) {
    CompareResult r = {0};
    uint32_t total_floats = total_pixels * 3;

    for (uint32_t i = 0; i < total_floats; i++) {
        if (isnan(gpu[i]) || isinf(gpu[i])) { r.nan_count++; continue; }
        float err = fabsf(cpu[i] - gpu[i]);
        if (err > r.max_err) r.max_err = err;
        if (err > tolerance) {
            if (err < 0.05f) r.boundary_count++;
            else r.mismatch_count++;
        }
    }
    uint32_t max_boundary = total_floats / 1000 + 1;
    r.pass = (r.nan_count == 0 && r.mismatch_count == 0 &&
              r.boundary_count <= max_boundary);
    return r;
}

/* ============================================================================
 * GPU Test Runner
 * ============================================================================ */

typedef struct GpuTestCtx {
    struct GpuSensorContext* ctx;
    GpuDevice* device;
    GpuKernel* kernel;
} GpuTestCtx;

static GpuTestCtx gpu_test_setup(uint32_t max_drones) {
    GpuTestCtx g = {0};
    g.ctx = gpu_sensor_context_create(max_drones);
    if (g.ctx == NULL) return g;
    g.device = gpu_sensor_context_device(g.ctx);
    g.kernel = gpu_kernel_create(g.device, "raymarch_unified");
    return g;
}

static void gpu_test_teardown(GpuTestCtx* g) {
    gpu_kernel_destroy(g->kernel);
    gpu_sensor_context_destroy(g->ctx);
    memset(g, 0, sizeof(GpuTestCtx));
}

/* Run GPU dispatch and readback */
static float* gpu_dispatch_and_readback(GpuTestCtx* g,
                                         const WorldBrickMap* world,
                                         const DroneStateSOA* drones,
                                         uint32_t drone_count,
                                         const Vec3* rays, uint32_t ray_count,
                                         RaymarchParams rp,
                                         uint32_t total_output_floats) {
    GpuResult r = gpu_sensor_context_sync_frame(g->ctx, world, drones, drone_count);
    if (r != GPU_SUCCESS) return NULL;

    GpuRayTable ray_table = gpu_ray_table_create(g->device, rays, ray_count);
    if (ray_table.rays == NULL) return NULL;

    GpuSensorOutput output = gpu_sensor_output_create(g->device, total_output_floats);
    if (output.buffer == NULL) {
        gpu_ray_table_destroy(&ray_table);
        return NULL;
    }

    GpuBuffer* drone_idx = gpu_buffer_create(g->device,
        drone_count * sizeof(uint32_t), GPU_MEMORY_SHARED);
    if (drone_idx == NULL) {
        gpu_ray_table_destroy(&ray_table);
        gpu_sensor_output_destroy(&output);
        return NULL;
    }
    uint32_t* idx_ptr = (uint32_t*)gpu_buffer_map(drone_idx);
    for (uint32_t i = 0; i < drone_count; i++) idx_ptr[i] = i;

    rp.drone_count = drone_count;

    r = gpu_dispatch_raymarch(g->ctx, g->kernel, ray_table.rays, output.buffer,
                              drone_idx, world, rp);
    if (r != GPU_SUCCESS) {
        gpu_ray_table_destroy(&ray_table);
        gpu_sensor_output_destroy(&output);
        gpu_buffer_destroy(drone_idx);
        return NULL;
    }

    GpuCommandQueue* queue = gpu_sensor_context_queue(g->ctx);
    gpu_queue_wait(queue);

    float* result = malloc(total_output_floats * sizeof(float));
    if (result != NULL) {
        const float* gpu_out = (const float*)gpu_buffer_map(output.buffer);
        if (gpu_out != NULL) {
            memcpy(result, gpu_out, total_output_floats * sizeof(float));
        } else {
            free(result);
            result = NULL;
        }
    }

    gpu_ray_table_destroy(&ray_table);
    gpu_sensor_output_destroy(&output);
    gpu_buffer_destroy(drone_idx);

    return result;
}

/* ============================================================================
 * Parametric sensor tests -- manage their own counters because they're
 * called with runtime parameters from loops/stress functions.
 * ============================================================================ */

static void test_camera_depth(GpuTestCtx* g, TestScene* scene,
                               uint32_t drone_count) {
    const uint32_t W = 32, H = 32;
    const float FOV = (float)(M_PI / 2.0);
    const float NEAR = 0.1f, FAR = 50.0f;
    char name[128];
    snprintf(name, sizeof(name), "camera_depth_%ux%u_%ud", W, H, drone_count);
    _th_run++;

    Vec3* rays = create_camera_rays(W, H, FOV, FOV);
    if (rays == NULL) { printf("  Running %s... FAILED (line %d)\n", name, __LINE__); return; }

    uint32_t total = W * H;
    uint32_t out_total = drone_count * total;

    TestDrones td;
    init_test_drones(&td, drone_count);
    for (uint32_t i = 0; i < drone_count; i++) {
        td.px[i] = -2.0f + (float)(i % 8) * 0.5f;
        td.py[i] = -2.0f + (float)((i / 8) % 8) * 0.5f;
        td.pz[i] = (float)(i % 3) * 0.3f;
    }

    float* cpu_out = malloc(out_total * sizeof(float));
    if (cpu_out == NULL) { printf("  Running %s... FAILED (line %d)\n", name, __LINE__); free(rays); return; }
    for (uint32_t d = 0; d < drone_count; d++) {
        Vec3 pos = VEC3(td.px[d], td.py[d], td.pz[d]);
        Quat q = QUAT(td.qw[d], td.qx[d], td.qy[d], td.qz[d]);
        cpu_render_depth(scene->world, pos, q, rays, total, NEAR, FAR,
                         &cpu_out[d * total]);
    }

    RaymarchParams rp = gpu_raymarch_params_for_depth(NEAR, FAR);
    rp.image_width = W;
    rp.image_height = H;
    float* gpu_out = gpu_dispatch_and_readback(g, scene->world, &td.soa,
                                                drone_count, rays, total, rp, out_total);
    if (gpu_out == NULL) { printf("  Running %s... FAILED (line %d)\n", name, __LINE__); free(cpu_out); free(rays); return; }

    CompareResult cr = compare_1fpp(cpu_out, gpu_out, out_total, 1e-3f, false);
    printf("  Running %s...(max_err=%f, nans=%u, boundary=%u) %s\n",
           name, cr.max_err, cr.nan_count, cr.boundary_count,
           cr.pass ? "PASSED" : "FAILED");
    if (cr.pass) _th_passed++;

    free(cpu_out);
    free(gpu_out);
    free(rays);
}

static void test_camera_rgb(GpuTestCtx* g, TestScene* scene,
                             uint32_t drone_count) {
    const uint32_t W = 32, H = 32;
    const float FOV = (float)(M_PI / 2.0);
    const float NEAR = 0.1f, FAR = 50.0f;
    char name[128];
    snprintf(name, sizeof(name), "camera_rgb_%ux%u_%ud", W, H, drone_count);
    _th_run++;

    Vec3* rays = create_camera_rays(W, H, FOV, FOV);
    if (rays == NULL) { printf("  Running %s... FAILED (line %d)\n", name, __LINE__); return; }

    uint32_t total = W * H;
    uint32_t out_total = drone_count * total * 3;

    TestDrones td;
    init_test_drones(&td, drone_count);
    for (uint32_t i = 0; i < drone_count; i++) {
        td.px[i] = -2.0f + (float)(i % 8) * 0.5f;
        td.py[i] = -2.0f + (float)((i / 8) % 8) * 0.5f;
        td.pz[i] = (float)(i % 3) * 0.3f;
    }

    float* cpu_out = malloc(out_total * sizeof(float));
    if (cpu_out == NULL) { printf("  Running %s... FAILED (line %d)\n", name, __LINE__); free(rays); return; }
    for (uint32_t d = 0; d < drone_count; d++) {
        Vec3 pos = VEC3(td.px[d], td.py[d], td.pz[d]);
        Quat q = QUAT(td.qw[d], td.qx[d], td.qy[d], td.qz[d]);
        cpu_render_rgb(scene->world, pos, q, rays, total, NEAR, FAR,
                       &cpu_out[d * total * 3]);
    }

    RaymarchParams rp = gpu_raymarch_params_for_rgb(NEAR, FAR);
    rp.image_width = W;
    rp.image_height = H;
    float* gpu_out = gpu_dispatch_and_readback(g, scene->world, &td.soa,
                                                drone_count, rays, total, rp, out_total);
    if (gpu_out == NULL) { printf("  Running %s... FAILED (line %d)\n", name, __LINE__); free(cpu_out); free(rays); return; }

    CompareResult cr = compare_rgb(cpu_out, gpu_out, drone_count * total, 1e-2f);
    printf("  Running %s...(max_err=%f, nans=%u, boundary=%u) %s\n",
           name, cr.max_err, cr.nan_count, cr.boundary_count,
           cr.pass ? "PASSED" : "FAILED");
    if (cr.pass) _th_passed++;

    free(cpu_out);
    free(gpu_out);
    free(rays);
}

static void test_camera_segmentation(GpuTestCtx* g, TestScene* scene,
                                      uint32_t drone_count) {
    const uint32_t W = 32, H = 32;
    const float FOV = (float)(M_PI / 2.0);
    const float NEAR = 0.1f, FAR = 50.0f;
    char name[128];
    snprintf(name, sizeof(name), "camera_seg_%ux%u_%ud", W, H, drone_count);
    _th_run++;

    Vec3* rays = create_camera_rays(W, H, FOV, FOV);
    if (rays == NULL) { printf("  Running %s... FAILED (line %d)\n", name, __LINE__); return; }

    uint32_t total = W * H;
    uint32_t out_total = drone_count * total;

    TestDrones td;
    init_test_drones(&td, drone_count);
    for (uint32_t i = 0; i < drone_count; i++) {
        td.px[i] = -2.0f + (float)(i % 8) * 0.5f;
        td.py[i] = -2.0f + (float)((i / 8) % 8) * 0.5f;
        td.pz[i] = (float)(i % 3) * 0.3f;
    }

    float* cpu_out = malloc(out_total * sizeof(float));
    if (cpu_out == NULL) { printf("  Running %s... FAILED (line %d)\n", name, __LINE__); free(rays); return; }
    for (uint32_t d = 0; d < drone_count; d++) {
        Vec3 pos = VEC3(td.px[d], td.py[d], td.pz[d]);
        Quat q = QUAT(td.qw[d], td.qx[d], td.qy[d], td.qz[d]);
        cpu_render_material(scene->world, pos, q, rays, total, NEAR, FAR,
                            &cpu_out[d * total]);
    }

    RaymarchParams rp = gpu_raymarch_params_for_material(NEAR, FAR);
    rp.image_width = W;
    rp.image_height = H;
    float* gpu_out = gpu_dispatch_and_readback(g, scene->world, &td.soa,
                                                drone_count, rays, total, rp, out_total);
    if (gpu_out == NULL) { printf("  Running %s... FAILED (line %d)\n", name, __LINE__); free(cpu_out); free(rays); return; }

    CompareResult cr = compare_1fpp(cpu_out, gpu_out, out_total, 0.0f, true);
    printf("  Running %s...(max_err=%f, nans=%u, mismatches=%u, boundary=%u) %s\n",
           name, cr.max_err, cr.nan_count, cr.mismatch_count, cr.boundary_count,
           cr.pass ? "PASSED" : "FAILED");
    if (cr.pass) _th_passed++;

    free(cpu_out);
    free(gpu_out);
    free(rays);
}

static void test_lidar_3d(GpuTestCtx* g, TestScene* scene,
                           uint32_t drone_count) {
    const uint32_t H_RAYS = 16, V_LAYERS = 8;
    const float H_FOV = (float)(M_PI * 2.0 / 3.0);
    const float V_FOV = (float)(M_PI / 6.0);
    const float MAX_RANGE = 30.0f;
    char name[128];
    snprintf(name, sizeof(name), "lidar_3d_%ux%u_%ud", H_RAYS, V_LAYERS, drone_count);
    _th_run++;

    uint32_t total_rays = H_RAYS * V_LAYERS;
    Vec3* rays = create_lidar_3d_rays(H_RAYS, V_LAYERS, H_FOV, V_FOV);
    if (rays == NULL) { printf("  Running %s... FAILED (line %d)\n", name, __LINE__); return; }

    uint32_t out_total = drone_count * total_rays;

    TestDrones td;
    init_test_drones(&td, drone_count);
    for (uint32_t i = 0; i < drone_count; i++) {
        td.px[i] = -1.0f + (float)(i % 4) * 0.5f;
        td.py[i] = -1.0f + (float)((i / 4) % 4) * 0.5f;
    }

    float* cpu_out = malloc(out_total * sizeof(float));
    if (cpu_out == NULL) { printf("  Running %s... FAILED (line %d)\n", name, __LINE__); free(rays); return; }
    for (uint32_t d = 0; d < drone_count; d++) {
        Vec3 pos = VEC3(td.px[d], td.py[d], td.pz[d]);
        Quat q = QUAT(td.qw[d], td.qx[d], td.qy[d], td.qz[d]);
        cpu_render_distance(scene->world, pos, q, rays, total_rays, MAX_RANGE,
                            &cpu_out[d * total_rays]);
    }

    RaymarchParams rp = gpu_raymarch_params_for_distance(MAX_RANGE);
    rp.image_width = H_RAYS;
    rp.image_height = V_LAYERS;
    float* gpu_out = gpu_dispatch_and_readback(g, scene->world, &td.soa,
                                                drone_count, rays, total_rays,
                                                rp, out_total);
    if (gpu_out == NULL) { printf("  Running %s... FAILED (line %d)\n", name, __LINE__); free(cpu_out); free(rays); return; }

    CompareResult cr = compare_1fpp(cpu_out, gpu_out, out_total, 1e-3f, false);
    printf("  Running %s...(max_err=%f, nans=%u, boundary=%u) %s\n",
           name, cr.max_err, cr.nan_count, cr.boundary_count,
           cr.pass ? "PASSED" : "FAILED");
    if (cr.pass) _th_passed++;

    free(cpu_out);
    free(gpu_out);
    free(rays);
}

static void test_lidar_2d(GpuTestCtx* g, TestScene* scene,
                           uint32_t drone_count) {
    const uint32_t NUM_RAYS = 64;
    const float FOV = (float)M_PI;
    const float MAX_RANGE = 30.0f;
    char name[128];
    snprintf(name, sizeof(name), "lidar_2d_%u_%ud", NUM_RAYS, drone_count);
    _th_run++;

    Vec3* rays = create_lidar_2d_rays(NUM_RAYS, FOV);
    if (rays == NULL) { printf("  Running %s... FAILED (line %d)\n", name, __LINE__); return; }

    uint32_t out_total = drone_count * NUM_RAYS;

    TestDrones td;
    init_test_drones(&td, drone_count);
    for (uint32_t i = 0; i < drone_count; i++) {
        td.px[i] = -1.0f + (float)(i % 4) * 0.5f;
        td.py[i] = -1.0f + (float)((i / 4) % 4) * 0.5f;
    }

    float* cpu_out = malloc(out_total * sizeof(float));
    if (cpu_out == NULL) { printf("  Running %s... FAILED (line %d)\n", name, __LINE__); free(rays); return; }
    for (uint32_t d = 0; d < drone_count; d++) {
        Vec3 pos = VEC3(td.px[d], td.py[d], td.pz[d]);
        Quat q = QUAT(td.qw[d], td.qx[d], td.qy[d], td.qz[d]);
        cpu_render_distance(scene->world, pos, q, rays, NUM_RAYS, MAX_RANGE,
                            &cpu_out[d * NUM_RAYS]);
    }

    RaymarchParams rp = gpu_raymarch_params_for_distance(MAX_RANGE);
    rp.image_width = NUM_RAYS;
    rp.image_height = 1;
    float* gpu_out = gpu_dispatch_and_readback(g, scene->world, &td.soa,
                                                drone_count, rays, NUM_RAYS,
                                                rp, out_total);
    if (gpu_out == NULL) { printf("  Running %s... FAILED (line %d)\n", name, __LINE__); free(cpu_out); free(rays); return; }

    CompareResult cr = compare_1fpp(cpu_out, gpu_out, out_total, 1e-3f, false);
    printf("  Running %s...(max_err=%f, nans=%u, boundary=%u) %s\n",
           name, cr.max_err, cr.nan_count, cr.boundary_count,
           cr.pass ? "PASSED" : "FAILED");
    if (cr.pass) _th_passed++;

    free(cpu_out);
    free(gpu_out);
    free(rays);
}

static void test_tof(GpuTestCtx* g, TestScene* scene, uint32_t drone_count) {
    const float MAX_RANGE = 30.0f;
    char name[128];
    snprintf(name, sizeof(name), "tof_%ud", drone_count);
    _th_run++;

    Vec3 forward_dir = VEC3(1.0f, 0.0f, 0.0f);
    Vec3* rays = malloc(sizeof(Vec3));
    if (rays == NULL) { printf("  Running %s... FAILED (line %d)\n", name, __LINE__); return; }
    rays[0] = forward_dir;

    uint32_t out_total = drone_count;

    TestDrones td;
    init_test_drones(&td, drone_count);
    for (uint32_t i = 0; i < drone_count; i++) {
        td.px[i] = -1.0f + (float)(i % 8) * 0.3f;
        td.py[i] = -1.0f + (float)((i / 8) % 8) * 0.3f;
    }

    float* cpu_out = malloc(out_total * sizeof(float));
    if (cpu_out == NULL) { printf("  Running %s... FAILED (line %d)\n", name, __LINE__); free(rays); return; }
    for (uint32_t d = 0; d < drone_count; d++) {
        Vec3 pos = VEC3(td.px[d], td.py[d], td.pz[d]);
        Quat q = QUAT(td.qw[d], td.qx[d], td.qy[d], td.qz[d]);
        cpu_render_distance(scene->world, pos, q, &forward_dir, 1, MAX_RANGE,
                            &cpu_out[d]);
    }

    RaymarchParams rp = gpu_raymarch_params_for_distance(MAX_RANGE);
    rp.image_width = 1;
    rp.image_height = 1;
    float* gpu_out = gpu_dispatch_and_readback(g, scene->world, &td.soa,
                                                drone_count, rays, 1,
                                                rp, out_total);
    if (gpu_out == NULL) { printf("  Running %s... FAILED (line %d)\n", name, __LINE__); free(cpu_out); free(rays); return; }

    CompareResult cr = compare_1fpp(cpu_out, gpu_out, out_total, 1e-3f, false);
    printf("  Running %s...(max_err=%f, nans=%u) %s\n",
           name, cr.max_err, cr.nan_count, cr.pass ? "PASSED" : "FAILED");
    if (cr.pass) _th_passed++;

    free(cpu_out);
    free(gpu_out);
    free(rays);
}

/* ============================================================================
 * Ground Truth Tests -- also parametric, manage own counters
 * ============================================================================ */

static void test_ground_truth_depth(GpuTestCtx* g, TestScene* scene) {
    _th_run++;
    const float NEAR = 0.1f, FAR = 50.0f;
    const float EXPECTED_HIT = 3.0f;
    const float GT_TOLERANCE = 0.5f;

    TestDrones td;
    init_test_drones(&td, 1);

    Vec3 forward = VEC3(1.0f, 0.0f, 0.0f);

    float cpu_depth;
    cpu_render_depth(scene->world, VEC3(0,0,0), QUAT(1,0,0,0),
                     &forward, 1, NEAR, FAR, &cpu_depth);

    RaymarchParams rp = gpu_raymarch_params_for_depth(NEAR, FAR);
    rp.image_width = 1;
    rp.image_height = 1;
    float* gpu_out = gpu_dispatch_and_readback(g, scene->world, &td.soa,
                                                1, &forward, 1, rp, 1);
    if (gpu_out == NULL) { printf("  Running gt_depth... FAILED (line %d)\n", __LINE__); return; }

    float cpu_dist = cpu_depth * (FAR - NEAR) + NEAR;
    float gpu_dist = gpu_out[0] * (FAR - NEAR) + NEAR;

    bool cpu_close = fabsf(cpu_dist - EXPECTED_HIT) < GT_TOLERANCE;
    bool gpu_close = fabsf(gpu_dist - EXPECTED_HIT) < GT_TOLERANCE;
    bool cpu_gpu_match = fabsf(cpu_depth - gpu_out[0]) < 1e-3f;

    bool pass = cpu_close && gpu_close && cpu_gpu_match;
    printf("  Running gt_depth...(expected=%.2f cpu=%.4f gpu=%.4f) %s\n",
           EXPECTED_HIT, cpu_dist, gpu_dist, pass ? "PASSED" : "FAILED");
    if (pass) _th_passed++;
    free(gpu_out);
}

static void test_ground_truth_distance(GpuTestCtx* g, TestScene* scene) {
    _th_run++;
    const float MAX_RANGE = 30.0f;
    const float EXPECTED_HIT = 3.0f;
    const float GT_TOLERANCE = 0.5f;

    TestDrones td;
    init_test_drones(&td, 1);
    Vec3 forward = VEC3(1.0f, 0.0f, 0.0f);

    float cpu_dist;
    cpu_render_distance(scene->world, VEC3(0,0,0), QUAT(1,0,0,0),
                        &forward, 1, MAX_RANGE, &cpu_dist);

    RaymarchParams rp = gpu_raymarch_params_for_distance(MAX_RANGE);
    rp.image_width = 1;
    rp.image_height = 1;
    float* gpu_out = gpu_dispatch_and_readback(g, scene->world, &td.soa,
                                                1, &forward, 1, rp, 1);
    if (gpu_out == NULL) { printf("  Running gt_distance... FAILED (line %d)\n", __LINE__); return; }

    bool cpu_close = fabsf(cpu_dist - EXPECTED_HIT) < GT_TOLERANCE;
    bool gpu_close = fabsf(gpu_out[0] - EXPECTED_HIT) < GT_TOLERANCE;
    bool match = fabsf(cpu_dist - gpu_out[0]) < 1e-3f;

    bool pass = cpu_close && gpu_close && match;
    printf("  Running gt_distance...(expected=%.2f cpu=%.4f gpu=%.4f) %s\n",
           EXPECTED_HIT, cpu_dist, gpu_out[0], pass ? "PASSED" : "FAILED");
    if (pass) _th_passed++;
    free(gpu_out);
}

static void test_ground_truth_segmentation(GpuTestCtx* g, TestScene* scene) {
    _th_run++;
    const float NEAR = 0.1f, FAR = 50.0f;
    const float EXPECTED_MATERIAL = 5.0f;

    TestDrones td;
    init_test_drones(&td, 1);
    Vec3 forward = VEC3(1.0f, 0.0f, 0.0f);

    float cpu_mat;
    cpu_render_material(scene->world, VEC3(0,0,0), QUAT(1,0,0,0),
                        &forward, 1, NEAR, FAR, &cpu_mat);

    RaymarchParams rp = gpu_raymarch_params_for_material(NEAR, FAR);
    rp.image_width = 1;
    rp.image_height = 1;
    float* gpu_out = gpu_dispatch_and_readback(g, scene->world, &td.soa,
                                                1, &forward, 1, rp, 1);
    if (gpu_out == NULL) { printf("  Running gt_seg... FAILED (line %d)\n", __LINE__); return; }

    bool cpu_ok = (cpu_mat == EXPECTED_MATERIAL);
    bool gpu_ok = (gpu_out[0] == EXPECTED_MATERIAL);

    bool pass = cpu_ok && gpu_ok;
    printf("  Running gt_seg...(expected=%.0f cpu=%.0f gpu=%.0f) %s\n",
           EXPECTED_MATERIAL, cpu_mat, gpu_out[0], pass ? "PASSED" : "FAILED");
    if (pass) _th_passed++;
    free(gpu_out);
}

static void test_ground_truth_rgb(GpuTestCtx* g, TestScene* scene) {
    _th_run++;
    const float NEAR = 0.1f, FAR = 50.0f;

    TestDrones td;
    init_test_drones(&td, 1);
    Vec3 forward = VEC3(1.0f, 0.0f, 0.0f);

    float cpu_rgb[3];
    cpu_render_rgb(scene->world, VEC3(0,0,0), QUAT(1,0,0,0),
                   &forward, 1, NEAR, FAR, cpu_rgb);

    RaymarchParams rp = gpu_raymarch_params_for_rgb(NEAR, FAR);
    rp.image_width = 1;
    rp.image_height = 1;
    float* gpu_out = gpu_dispatch_and_readback(g, scene->world, &td.soa,
                                                1, &forward, 1, rp, 3);
    if (gpu_out == NULL) { printf("  Running gt_rgb... FAILED (line %d)\n", __LINE__); return; }

    bool is_red = (cpu_rgb[0] > cpu_rgb[1]) && (cpu_rgb[0] > cpu_rgb[2]);
    bool not_sky = (cpu_rgb[0] != SKY_COLOR_R || cpu_rgb[1] != SKY_COLOR_G ||
                    cpu_rgb[2] != SKY_COLOR_B);
    bool match = (fabsf(cpu_rgb[0] - gpu_out[0]) < 1e-2f &&
                  fabsf(cpu_rgb[1] - gpu_out[1]) < 1e-2f &&
                  fabsf(cpu_rgb[2] - gpu_out[2]) < 1e-2f);

    bool pass = is_red && not_sky && match;
    printf("  Running gt_rgb...(cpu=(%.4f,%.4f,%.4f) gpu=(%.4f,%.4f,%.4f)) %s\n",
           cpu_rgb[0], cpu_rgb[1], cpu_rgb[2],
           gpu_out[0], gpu_out[1], gpu_out[2],
           pass ? "PASSED" : "FAILED");
    if (pass) _th_passed++;
    free(gpu_out);
}

static void test_ground_truth_lidar_forward(GpuTestCtx* g, TestScene* scene) {
    _th_run++;
    const float MAX_RANGE = 30.0f;
    const float EXPECTED_HIT = 3.0f;
    const float GT_TOLERANCE = 0.5f;

    uint32_t num_rays = 16;
    Vec3* rays = create_lidar_2d_rays(num_rays, (float)(M_PI / 2.0));
    if (rays == NULL) { printf("  Running gt_lidar... FAILED (line %d)\n", __LINE__); return; }

    TestDrones td;
    init_test_drones(&td, 1);

    float cpu_out[16];
    cpu_render_distance(scene->world, VEC3(0,0,0), QUAT(1,0,0,0),
                        rays, num_rays, MAX_RANGE, cpu_out);

    RaymarchParams rp = gpu_raymarch_params_for_distance(MAX_RANGE);
    rp.image_width = num_rays;
    rp.image_height = 1;
    float* gpu_out = gpu_dispatch_and_readback(g, scene->world, &td.soa,
                                                1, rays, num_rays, rp, num_rays);
    if (gpu_out == NULL) { printf("  Running gt_lidar... FAILED (line %d)\n", __LINE__); free(rays); return; }

    /* Center rays see the sphere; edge rays miss */
    uint32_t center = num_rays / 2;
    float cpu_center = cpu_out[center];
    float gpu_center = gpu_out[center];

    bool cpu_hit = fabsf(cpu_center - EXPECTED_HIT) < GT_TOLERANCE;
    bool gpu_hit = fabsf(gpu_center - EXPECTED_HIT) < GT_TOLERANCE;
    bool match = fabsf(cpu_center - gpu_center) < 1e-3f;
    bool side_ok = (cpu_out[0] > 5.0f && cpu_out[num_rays-1] > 5.0f);

    bool pass = cpu_hit && gpu_hit && match && side_ok;
    printf("  Running gt_lidar...(center cpu=%.4f gpu=%.4f) %s\n",
           cpu_center, gpu_center, pass ? "PASSED" : "FAILED");
    if (pass) _th_passed++;
    free(gpu_out);
    free(rays);
}

/* ============================================================================
 * Test: Non-square FOV (verifies camera ray fix)
 * ============================================================================ */

static void test_nonsquare_fov(GpuTestCtx* g, TestScene* scene) {
    _th_run++;
    const uint32_t W = 32, H = 16;
    const float FOV_H = (float)(M_PI / 2.0);   /* 90 deg */
    const float FOV_V = (float)(M_PI / 3.0);   /* 60 deg */
    const float NEAR = 0.1f, FAR = 50.0f;

    Vec3* rays = create_camera_rays(W, H, FOV_H, FOV_V);
    if (rays == NULL) { printf("  Running nonsquare_%ux%u... FAILED (line %d)\n", W, H, __LINE__); return; }

    float focal_x = 1.0f / tanf(FOV_H * 0.5f);
    float focal_y = 1.0f / tanf(FOV_V * 0.5f);

    uint32_t mid_y = H / 2;
    Vec3 right_ray = rays[mid_y * W + (W - 1)];
    float h_angle = atan2f(right_ray.y, right_ray.x);
    float u_edge = 1.0f - 1.0f / (float)W;
    float expected_h = atanf(u_edge / focal_x);
    bool h_ok = fabsf(h_angle - expected_h) < 0.01f;

    uint32_t mid_x = W / 2;
    Vec3 bottom_ray = rays[(H - 1) * W + mid_x];
    float v_angle = -atan2f(bottom_ray.z, bottom_ray.x);
    float v_edge = 1.0f - 1.0f / (float)H;
    float expected_v = atanf(v_edge / focal_y);
    bool v_ok = fabsf(v_angle - expected_v) < 0.01f;

    /* GPU vs CPU consistency */
    TestDrones td;
    init_test_drones(&td, 4);
    uint32_t total = W * H;
    uint32_t out_total = 4 * total;

    float* cpu_out = malloc(out_total * sizeof(float));
    if (cpu_out == NULL) { printf("  Running nonsquare_%ux%u... FAILED (line %d)\n", W, H, __LINE__); free(rays); return; }
    for (uint32_t d = 0; d < 4; d++) {
        cpu_render_depth(scene->world, VEC3(0,0,0), QUAT(1,0,0,0),
                         rays, total, NEAR, FAR, &cpu_out[d * total]);
    }

    RaymarchParams rp = gpu_raymarch_params_for_depth(NEAR, FAR);
    rp.image_width = W;
    rp.image_height = H;
    float* gpu_out = gpu_dispatch_and_readback(g, scene->world, &td.soa,
                                                4, rays, total, rp, out_total);
    if (gpu_out == NULL) { printf("  Running nonsquare_%ux%u... FAILED (line %d)\n", W, H, __LINE__); free(cpu_out); free(rays); return; }

    CompareResult cr = compare_1fpp(cpu_out, gpu_out, out_total, 1e-3f, false);
    bool pass = h_ok && v_ok && cr.pass;

    printf("  Running nonsquare_%ux%u...(h=%.3f(exp=%.3f) v=%.3f(exp=%.3f) err=%f) %s\n",
           W, H, h_angle, expected_h, v_angle, expected_v,
           cr.max_err, pass ? "PASSED" : "FAILED");
    if (!cr.pass) {
        uint32_t printed = 0;
        for (uint32_t i = 0; i < out_total && printed < 5; i++) {
            float err = fabsf(cpu_out[i] - gpu_out[i]);
            if (err > 1e-3f) {
                uint32_t drone = i / (W * H);
                uint32_t px = i % (W * H);
                printf("    pixel[d=%u, p=%u (%u,%u)]: cpu=%.6f gpu=%.6f "
                       "err=%.6f\n",
                       drone, px, px % W, px / W,
                       cpu_out[i], gpu_out[i], err);
                printed++;
            }
        }
    }

    if (pass) _th_passed++;
    free(cpu_out);
    free(gpu_out);
    free(rays);
}

/* ============================================================================
 * Stress: 256 drones, all sensor modes on room scene
 * ============================================================================ */

static void test_stress_all_modes(GpuTestCtx* g, TestScene* scene) {
    const uint32_t N = 256;
    printf("\n--- Stress %u drones ---\n", N);

    test_camera_depth(g, scene, N);
    test_camera_rgb(g, scene, N);
    test_camera_segmentation(g, scene, N);
    test_lidar_3d(g, scene, N);
    test_lidar_2d(g, scene, N);
    test_tof(g, scene, N);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    if (!gpu_is_available()) {
        printf("GPU not available - skipping\n");
        return 0;
    }
    printf("Backend: %s\n", GPU_AVAILABLE ? "METAL" : "NONE");

    TEST_SUITE_BEGIN("GPU All-Sensors Tests");

    GpuTestCtx g = gpu_test_setup(MAX_TEST_DRONES);
    if (g.ctx == NULL) {
        printf("FAIL: Could not create GPU context\n");
        return 1;
    }
    if (g.kernel == NULL) {
        printf("FAIL: Could not load raymarch kernel\n");
        gpu_test_teardown(&g);
        return 1;
    }
    printf("Kernel 'raymarch_unified' loaded OK\n");

    TestScene sphere = create_sphere_scene();
    TestScene room = create_room_scene();

    if (sphere.world == NULL || room.world == NULL) {
        printf("FAIL: Could not create test scenes\n");
        gpu_test_teardown(&g);
        return 1;
    }

    /* Ground truth tests (sphere scene) */
    printf("\n--- Ground Truth ---\n");
    test_ground_truth_depth(&g, &sphere);
    test_ground_truth_distance(&g, &sphere);
    test_ground_truth_segmentation(&g, &sphere);
    test_ground_truth_rgb(&g, &sphere);
    test_ground_truth_lidar_forward(&g, &sphere);

    /* GPU vs CPU: sphere scene, 16 drones */
    printf("\n--- GPU vs CPU: sphere, 16d ---\n");
    test_camera_depth(&g, &sphere, 16);
    test_camera_rgb(&g, &sphere, 16);
    test_camera_segmentation(&g, &sphere, 16);
    test_lidar_3d(&g, &sphere, 16);
    test_lidar_2d(&g, &sphere, 16);
    test_tof(&g, &sphere, 16);

    /* GPU vs CPU: room scene, 16 drones */
    printf("\n--- GPU vs CPU: room, 16d ---\n");
    test_camera_depth(&g, &room, 16);
    test_camera_rgb(&g, &room, 16);
    test_camera_segmentation(&g, &room, 16);
    test_lidar_3d(&g, &room, 16);
    test_lidar_2d(&g, &room, 16);
    test_tof(&g, &room, 16);

    /* Non-square FOV */
    printf("\n--- Non-square FOV ---\n");
    test_nonsquare_fov(&g, &sphere);

    /* Stress test */
    test_stress_all_modes(&g, &room);

    /* Cleanup */
    destroy_scene(&sphere);
    destroy_scene(&room);
    gpu_test_teardown(&g);

    TEST_SUITE_END();
}
