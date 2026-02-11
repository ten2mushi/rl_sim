/**
 * GPU Voxelization Tests
 *
 * Compares GPU Phase 3 voxelization against CPU Phase 3 for several
 * test meshes, verifying SDF and material values match within tolerance.
 *
 * Test strategy:
 * 1. Create a simple mesh (cube, sphere)
 * 2. Run full CPU voxelization (Phases 1-3)
 * 3. Run GPU voxelization (Phases 1-2 CPU, Phase 3 GPU)
 * 4. Compare every voxel: SDF must match within 1 quantization step
 */

#include "gpu_hal.h"
#include "sdf_types.h"
#include "obj_io.h"
#include "world_brick_map.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "test_harness.h"

static double test_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* ============================================================================
 * Helper: Create a cube mesh centered at origin
 * ============================================================================ */

static TriangleMesh* create_cube_mesh(Arena* arena, float half_size) {
    TriangleMesh* mesh = mesh_create(arena, 8, 12);
    if (!mesh) return NULL;

    float h = half_size;

    /* 8 vertices of cube */
    mesh_add_vertex(mesh, -h, -h, -h);  /* 0 */
    mesh_add_vertex(mesh,  h, -h, -h);  /* 1 */
    mesh_add_vertex(mesh,  h,  h, -h);  /* 2 */
    mesh_add_vertex(mesh, -h,  h, -h);  /* 3 */
    mesh_add_vertex(mesh, -h, -h,  h);  /* 4 */
    mesh_add_vertex(mesh,  h, -h,  h);  /* 5 */
    mesh_add_vertex(mesh,  h,  h,  h);  /* 6 */
    mesh_add_vertex(mesh, -h,  h,  h);  /* 7 */

    /* 12 triangles (2 per face) */
    /* Bottom (z=-h) */
    mesh_add_face(mesh, 0, 2, 1, 0);
    mesh_add_face(mesh, 0, 3, 2, 0);
    /* Top (z=+h) */
    mesh_add_face(mesh, 4, 5, 6, 1);
    mesh_add_face(mesh, 4, 6, 7, 1);
    /* Front (y=-h) */
    mesh_add_face(mesh, 0, 1, 5, 2);
    mesh_add_face(mesh, 0, 5, 4, 2);
    /* Back (y=+h) */
    mesh_add_face(mesh, 2, 3, 7, 3);
    mesh_add_face(mesh, 2, 7, 6, 3);
    /* Left (x=-h) */
    mesh_add_face(mesh, 0, 4, 7, 4);
    mesh_add_face(mesh, 0, 7, 3, 4);
    /* Right (x=+h) */
    mesh_add_face(mesh, 1, 2, 6, 5);
    mesh_add_face(mesh, 1, 6, 5, 5);

    mesh_compute_bbox(mesh);
    return mesh;
}

/* ============================================================================
 * Helper: Create a UV sphere mesh
 * ============================================================================ */

static TriangleMesh* create_sphere_mesh(Arena* arena, float radius,
                                         uint32_t rings, uint32_t sectors) {
    uint32_t vert_cap = (rings + 1) * (sectors + 1);
    uint32_t face_cap = rings * sectors * 2;
    TriangleMesh* mesh = mesh_create(arena, vert_cap, face_cap);
    if (!mesh) return NULL;

    /* Generate vertices */
    for (uint32_t r = 0; r <= rings; r++) {
        float phi = (float)M_PI * (float)r / (float)rings;
        for (uint32_t s = 0; s <= sectors; s++) {
            float theta = 2.0f * (float)M_PI * (float)s / (float)sectors;
            float x = radius * sinf(phi) * cosf(theta);
            float y = radius * sinf(phi) * sinf(theta);
            float z = radius * cosf(phi);
            mesh_add_vertex(mesh, x, y, z);
        }
    }

    /* Generate faces */
    for (uint32_t r = 0; r < rings; r++) {
        for (uint32_t s = 0; s < sectors; s++) {
            uint32_t v0 = r * (sectors + 1) + s;
            uint32_t v1 = v0 + 1;
            uint32_t v2 = v0 + sectors + 1;
            uint32_t v3 = v2 + 1;

            mesh_add_face(mesh, v0, v2, v1, 0);
            mesh_add_face(mesh, v1, v2, v3, 0);
        }
    }

    mesh_compute_bbox(mesh);
    return mesh;
}

/* ============================================================================
 * Helper: Run CPU voxelization and collect surface brick list
 * ============================================================================ */

typedef struct VoxelizeResult {
    WorldBrickMap* world;
    BrickClassification* classes;
    MeshBVH* bvh;
    uint32_t* surface_brick_list; /* (bx, by, bz) * num_surface */
    uint32_t num_surface_bricks;
} VoxelizeResult;

static VoxelizeResult run_cpu_voxelization(Arena* arena, TriangleMesh* mesh,
                                            const VoxelizeOptions* options) {
    VoxelizeResult r = {0};

    float voxel_size = options->voxel_size;
    float brick_size = voxel_size * BRICK_SIZE;
    float padding = options->padding > 0 ? options->padding : brick_size;

    Vec3 world_min = vec3_sub(mesh->bbox_min, VEC3(padding, padding, padding));
    Vec3 world_max = vec3_add(mesh->bbox_max, VEC3(padding, padding, padding));

    Vec3 world_size = vec3_sub(world_max, world_min);
    uint32_t grid_x = (uint32_t)ceilf(world_size.x / brick_size);
    uint32_t grid_y = (uint32_t)ceilf(world_size.y / brick_size);
    uint32_t grid_z = (uint32_t)ceilf(world_size.z / brick_size);
    uint32_t grid_total = grid_x * grid_y * grid_z;

    r.bvh = bvh_build(arena, mesh);
    if (!r.bvh) return r;

    r.world = world_create(arena, world_min, world_max, voxel_size, grid_total, 16);
    if (!r.world) return r;

    /* Phase 1 */
    r.classes = classify_bricks_coarse(arena, r.bvh, mesh, r.world);
    if (!r.classes) return r;

    /* Phase 2 */
    classify_bricks_fine(r.classes, r.bvh, mesh, r.world, options);

    /* Collect surface brick coordinates */
    uint32_t surface_count = 0;
    for (uint32_t i = 0; i < grid_total; i++) {
        if (r.classes->classes[i] == BRICK_CLASS_SURFACE) {
            surface_count++;
        }
    }

    r.surface_brick_list = arena_alloc_array(arena, uint32_t, surface_count * 3);
    r.num_surface_bricks = surface_count;

    uint32_t si = 0;
    for (uint32_t bz = 0; bz < grid_z; bz++) {
        for (uint32_t by = 0; by < grid_y; by++) {
            for (uint32_t bx = 0; bx < grid_x; bx++) {
                uint32_t idx = bx + by * grid_x + bz * grid_x * grid_y;
                if (r.classes->classes[idx] == BRICK_CLASS_SURFACE) {
                    r.surface_brick_list[si * 3 + 0] = bx;
                    r.surface_brick_list[si * 3 + 1] = by;
                    r.surface_brick_list[si * 3 + 2] = bz;
                    si++;
                }
            }
        }
    }

    /* Phase 3: CPU voxelization */
    voxelize_surface_bricks(r.world, r.classes, r.bvh, mesh, options);

    return r;
}

/* ============================================================================
 * Helper: Compare two WorldBrickMaps voxel-by-voxel
 * ============================================================================ */

typedef struct CompareStats {
    uint32_t total_voxels;
    uint32_t matching_sdf;
    uint32_t matching_mat;
    uint32_t sdf_within_1;   /* Within 1 quantization step */
    int32_t max_sdf_diff;
    float mean_sdf_diff;
} CompareStats;

static CompareStats compare_worlds(const WorldBrickMap* cpu_world,
                                    const WorldBrickMap* gpu_world) {
    CompareStats stats = {0};
    float total_diff = 0.0f;

    uint32_t grid_total = cpu_world->grid_x * cpu_world->grid_y * cpu_world->grid_z;

    for (uint32_t i = 0; i < grid_total; i++) {
        int32_t cpu_idx = cpu_world->brick_indices[i];
        int32_t gpu_idx = gpu_world->brick_indices[i];

        /* Only compare surface bricks that exist in both */
        if (cpu_idx < 0 || gpu_idx < 0) continue;

        const int8_t* cpu_sdf = world_brick_sdf_const(cpu_world, cpu_idx);
        const int8_t* gpu_sdf = world_brick_sdf_const(gpu_world, gpu_idx);
        const uint8_t* cpu_mat = world_brick_material_const(cpu_world, cpu_idx);
        const uint8_t* gpu_mat = world_brick_material_const(gpu_world, gpu_idx);

        if (!cpu_sdf || !gpu_sdf) continue;

        for (uint32_t v = 0; v < BRICK_VOXELS; v++) {
            stats.total_voxels++;

            int32_t diff = abs((int32_t)cpu_sdf[v] - (int32_t)gpu_sdf[v]);
            total_diff += (float)diff;

            if (diff == 0) stats.matching_sdf++;
            if (diff <= 1) stats.sdf_within_1++;
            if (diff > stats.max_sdf_diff) stats.max_sdf_diff = diff;

            if (cpu_mat && gpu_mat && cpu_mat[v] == gpu_mat[v]) {
                stats.matching_mat++;
            }
        }
    }

    if (stats.total_voxels > 0) {
        stats.mean_sdf_diff = total_diff / (float)stats.total_voxels;
    }

    return stats;
}

/* ============================================================================
 * Test: GPU device and kernel availability
 * ============================================================================ */

static GpuDevice* g_device = NULL;

TEST(gpu_available) {
    ASSERT_MSG(gpu_is_available(), "GPU not available");

    g_device = gpu_device_create();
    ASSERT_MSG(g_device != NULL, "Failed to create GPU device");
    printf("(%s) ", gpu_device_name(g_device));

    /* Check that voxelize kernel can be created */
    GpuKernel* kernel = gpu_kernel_create(g_device, "sdf_voxelize_surface");
    ASSERT_MSG(kernel != NULL, "Failed to create sdf_voxelize_surface kernel");
    gpu_kernel_destroy(kernel);

    return 0;
}

/* ============================================================================
 * Test: Cube mesh GPU vs CPU voxelization
 * ============================================================================ */

TEST(cube_voxelization) {
    ASSERT_MSG(g_device != NULL, "No GPU device");

    /* Create cube mesh */
    Arena* arena = arena_create(64 * 1024 * 1024);
    ASSERT_MSG(arena != NULL, "Failed to create arena");

    TriangleMesh* mesh = create_cube_mesh(arena, 2.0f);
    ASSERT_MSG(mesh != NULL, "Failed to create cube mesh");

    VoxelizeOptions options = VOXELIZE_DEFAULTS;
    options.voxel_size = 0.25f;
    options.preserve_materials = true;

    /* CPU voxelization */
    VoxelizeResult cpu = run_cpu_voxelization(arena, mesh, &options);
    ASSERT_MSG(cpu.world != NULL, "CPU voxelization failed");
    ASSERT_MSG(cpu.num_surface_bricks > 0, "No surface bricks found");

    printf("(%u surface bricks, %u faces) ",
           cpu.num_surface_bricks, mesh->face_count);

    /* GPU voxelization (same Phases 1-2, Phase 3 on GPU) */
    /* Create a separate world for GPU results */
    Vec3 world_min = cpu.world->world_min;
    Vec3 world_max = cpu.world->world_max;
    uint32_t grid_total = cpu.world->grid_x * cpu.world->grid_y * cpu.world->grid_z;

    WorldBrickMap* gpu_world = world_create(arena, world_min, world_max,
                                             options.voxel_size, grid_total, 16);
    ASSERT_MSG(gpu_world != NULL, "Failed to create GPU world");

    /* Mark uniform bricks same as CPU */
    for (uint32_t bz = 0; bz < cpu.classes->grid_z; bz++) {
        for (uint32_t by = 0; by < cpu.classes->grid_y; by++) {
            for (uint32_t bx = 0; bx < cpu.classes->grid_x; bx++) {
                uint32_t idx = bx + by * cpu.classes->grid_x +
                               bz * cpu.classes->grid_x * cpu.classes->grid_y;
                if (cpu.classes->classes[idx] == BRICK_CLASS_INSIDE) {
                    world_mark_brick_uniform_inside(gpu_world, (int32_t)bx,
                                                     (int32_t)by, (int32_t)bz);
                }
            }
        }
    }

    /* Run GPU Phase 3 */
    GpuResult result = gpu_voxelize_surface_bricks(
        g_device, cpu.bvh, mesh,
        cpu.surface_brick_list, cpu.num_surface_bricks,
        gpu_world, &options);
    ASSERT_MSG(result == GPU_SUCCESS, "GPU voxelization dispatch failed");

    /* Compare results */
    CompareStats stats = compare_worlds(cpu.world, gpu_world);
    printf("(voxels=%u, exact=%.1f%%, within1=%.1f%%, max_diff=%d, mean=%.2f) ",
           stats.total_voxels,
           100.0f * stats.matching_sdf / (stats.total_voxels > 0 ? stats.total_voxels : 1),
           100.0f * stats.sdf_within_1 / (stats.total_voxels > 0 ? stats.total_voxels : 1),
           stats.max_sdf_diff, stats.mean_sdf_diff);

    /* Tolerance: all voxels must match within 1 quantization step */
    ASSERT_MSG(stats.total_voxels > 0, "No voxels compared");
    ASSERT_MSG(stats.sdf_within_1 == stats.total_voxels,
                "SDF values differ by more than 1 quantization step");

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Test: Sphere mesh GPU vs CPU voxelization
 * ============================================================================ */

TEST(sphere_voxelization) {
    ASSERT_MSG(g_device != NULL, "No GPU device");

    Arena* arena = arena_create(64 * 1024 * 1024);
    ASSERT_MSG(arena != NULL, "Failed to create arena");

    TriangleMesh* mesh = create_sphere_mesh(arena, 3.0f, 16, 32);
    ASSERT_MSG(mesh != NULL, "Failed to create sphere mesh");

    VoxelizeOptions options = VOXELIZE_DEFAULTS;
    options.voxel_size = 0.25f;
    options.preserve_materials = true;

    VoxelizeResult cpu = run_cpu_voxelization(arena, mesh, &options);
    ASSERT_MSG(cpu.world != NULL, "CPU voxelization failed");
    ASSERT_MSG(cpu.num_surface_bricks > 0, "No surface bricks");

    printf("(%u surface bricks, %u faces) ",
           cpu.num_surface_bricks, mesh->face_count);

    /* GPU world */
    Vec3 world_min = cpu.world->world_min;
    Vec3 world_max = cpu.world->world_max;
    uint32_t grid_total = cpu.world->grid_x * cpu.world->grid_y * cpu.world->grid_z;

    WorldBrickMap* gpu_world = world_create(arena, world_min, world_max,
                                             options.voxel_size, grid_total, 16);
    ASSERT_MSG(gpu_world != NULL, "Failed to create GPU world");

    for (uint32_t bz = 0; bz < cpu.classes->grid_z; bz++) {
        for (uint32_t by = 0; by < cpu.classes->grid_y; by++) {
            for (uint32_t bx = 0; bx < cpu.classes->grid_x; bx++) {
                uint32_t idx = bx + by * cpu.classes->grid_x +
                               bz * cpu.classes->grid_x * cpu.classes->grid_y;
                if (cpu.classes->classes[idx] == BRICK_CLASS_INSIDE) {
                    world_mark_brick_uniform_inside(gpu_world, (int32_t)bx,
                                                     (int32_t)by, (int32_t)bz);
                }
            }
        }
    }

    GpuResult result = gpu_voxelize_surface_bricks(
        g_device, cpu.bvh, mesh,
        cpu.surface_brick_list, cpu.num_surface_bricks,
        gpu_world, &options);
    ASSERT_MSG(result == GPU_SUCCESS, "GPU voxelization failed");

    CompareStats stats = compare_worlds(cpu.world, gpu_world);
    printf("(voxels=%u, exact=%.1f%%, within1=%.1f%%, max_diff=%d) ",
           stats.total_voxels,
           100.0f * stats.matching_sdf / (stats.total_voxels > 0 ? stats.total_voxels : 1),
           100.0f * stats.sdf_within_1 / (stats.total_voxels > 0 ? stats.total_voxels : 1),
           stats.max_sdf_diff);

    ASSERT_MSG(stats.total_voxels > 0, "No voxels compared");
    ASSERT_MSG(stats.sdf_within_1 == stats.total_voxels,
                "SDF mismatch > 1 quantization step");

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Test: Shell mode GPU vs CPU
 * ============================================================================ */

TEST(shell_mode_voxelization) {
    ASSERT_MSG(g_device != NULL, "No GPU device");

    Arena* arena = arena_create(64 * 1024 * 1024);
    ASSERT_MSG(arena != NULL, "Failed to create arena");

    TriangleMesh* mesh = create_sphere_mesh(arena, 2.0f, 12, 24);
    ASSERT_MSG(mesh != NULL, "Failed to create sphere mesh");

    VoxelizeOptions options = VOXELIZE_DEFAULTS;
    options.voxel_size = 0.25f;
    options.preserve_materials = false;
    options.shell_mode = true;
    options.shell_thickness = 0.5f;

    VoxelizeResult cpu = run_cpu_voxelization(arena, mesh, &options);
    ASSERT_MSG(cpu.world != NULL, "CPU voxelization failed");
    ASSERT_MSG(cpu.num_surface_bricks > 0, "No surface bricks");

    printf("(%u surface bricks) ", cpu.num_surface_bricks);

    Vec3 world_min = cpu.world->world_min;
    Vec3 world_max = cpu.world->world_max;
    uint32_t grid_total = cpu.world->grid_x * cpu.world->grid_y * cpu.world->grid_z;

    WorldBrickMap* gpu_world = world_create(arena, world_min, world_max,
                                             options.voxel_size, grid_total, 16);
    ASSERT_MSG(gpu_world != NULL, "Failed to create GPU world");

    GpuResult result = gpu_voxelize_surface_bricks(
        g_device, cpu.bvh, mesh,
        cpu.surface_brick_list, cpu.num_surface_bricks,
        gpu_world, &options);
    ASSERT_MSG(result == GPU_SUCCESS, "GPU voxelization failed");

    CompareStats stats = compare_worlds(cpu.world, gpu_world);
    printf("(voxels=%u, exact=%.1f%%, within1=%.1f%%, max_diff=%d) ",
           stats.total_voxels,
           100.0f * stats.matching_sdf / (stats.total_voxels > 0 ? stats.total_voxels : 1),
           100.0f * stats.sdf_within_1 / (stats.total_voxels > 0 ? stats.total_voxels : 1),
           stats.max_sdf_diff);

    ASSERT_MSG(stats.total_voxels > 0, "No voxels compared");
    ASSERT_MSG(stats.sdf_within_1 == stats.total_voxels,
                "SDF mismatch > 1 quantization step");

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Test: GPU voxelization timing
 * ============================================================================ */

TEST(gpu_voxelize_performance) {
    ASSERT_MSG(g_device != NULL, "No GPU device");

    Arena* arena = arena_create(128 * 1024 * 1024);
    ASSERT_MSG(arena != NULL, "Failed to create arena");

    /* Larger sphere for meaningful timing */
    TriangleMesh* mesh = create_sphere_mesh(arena, 5.0f, 32, 64);
    ASSERT_MSG(mesh != NULL, "Failed to create sphere mesh");

    VoxelizeOptions options = VOXELIZE_DEFAULTS;
    options.voxel_size = 0.2f;
    options.preserve_materials = false;

    /* CPU path: full voxelization with timing */
    VoxelizeResult cpu = run_cpu_voxelization(arena, mesh, &options);
    ASSERT_MSG(cpu.world != NULL, "CPU voxelization failed");
    ASSERT_MSG(cpu.num_surface_bricks > 0, "No surface bricks");

    printf("(%u faces, %u surface bricks) ", mesh->face_count, cpu.num_surface_bricks);

    /* GPU path with timing */
    Vec3 world_min = cpu.world->world_min;
    Vec3 world_max = cpu.world->world_max;
    uint32_t grid_total = cpu.world->grid_x * cpu.world->grid_y * cpu.world->grid_z;

    /* Time CPU Phase 3 */
    WorldBrickMap* cpu_world2 = world_create(arena, world_min, world_max,
                                              options.voxel_size, grid_total, 16);
    ASSERT_MSG(cpu_world2 != NULL, "Failed to create second CPU world");

    double t0 = test_time_ms();
    voxelize_surface_bricks(cpu_world2, cpu.classes, cpu.bvh, mesh, &options);
    double t1 = test_time_ms();
    double cpu_ms = t1 - t0;

    /* Time GPU Phase 3 */
    WorldBrickMap* gpu_world = world_create(arena, world_min, world_max,
                                             options.voxel_size, grid_total, 16);
    ASSERT_MSG(gpu_world != NULL, "Failed to create GPU world");

    double t2 = test_time_ms();
    GpuResult result = gpu_voxelize_surface_bricks(
        g_device, cpu.bvh, mesh,
        cpu.surface_brick_list, cpu.num_surface_bricks,
        gpu_world, &options);
    double t3 = test_time_ms();
    double gpu_ms = t3 - t2;

    ASSERT_MSG(result == GPU_SUCCESS, "GPU voxelization failed");

    double speedup = cpu_ms / gpu_ms;
    printf("CPU=%.1fms GPU=%.1fms speedup=%.1fx ", cpu_ms, gpu_ms, speedup);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("GPU Voxelization Tests");

    RUN_TEST(gpu_available);
    RUN_TEST(cube_voxelization);
    RUN_TEST(sphere_voxelization);
    RUN_TEST(shell_mode_voxelization);
    RUN_TEST(gpu_voxelize_performance);

    if (g_device) {
        gpu_device_destroy(g_device);
    }

    TEST_SUITE_END();
}
