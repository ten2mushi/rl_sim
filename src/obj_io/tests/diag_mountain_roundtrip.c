/**
 * OBJ roundtrip diagnostic: load OBJ, voxelize, extract, export.
 * Reports timing, brick stats, and Hausdorff distance.
 *
 * Usage: diag_mountain_roundtrip <input.obj> <output.obj> [voxel_size] [cpu|gpu] [shell]
 *
 * The optional "cpu" or "gpu" argument selects the voxelization backend.
 * Default is "gpu" (Phase 3 on Metal GPU, Phases 1-2 on CPU).
 *
 * The optional "shell" argument enables shell mode for thin surfaces
 * (gyroid, single-sheet terrain) that have no well-defined interior.
 */

#include "../include/obj_io.h"
#include "gpu_hal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/**
 * GPU-accelerated mesh_to_sdf: Phases 1-2 on CPU, Phase 3 on GPU.
 * Falls back to CPU if GPU is unavailable.
 */
static ObjIOResult mesh_to_sdf_gpu(Arena* arena, const TriangleMesh* mesh,
                                    const VoxelizeOptions* options,
                                    WorldBrickMap** out_world, char* error) {
#if GPU_AVAILABLE
    if (!gpu_is_available()) {
        printf("  [GPU unavailable, falling back to CPU]\n");
        return mesh_to_sdf(arena, mesh, options, out_world, error);
    }

    if (!options) options = &VOXELIZE_DEFAULTS;
    *out_world = NULL;

    if (mesh->vertex_count == 0 || mesh->face_count == 0) {
        if (error) snprintf(error, 256, "Empty mesh");
        return OBJ_IO_ERROR_EMPTY_MESH;
    }

    /* Compute world bounds with padding (same as mesh_to_sdf) */
    float voxel_size = options->voxel_size;
    float brick_size = voxel_size * BRICK_SIZE;
    float padding = options->padding > 0 ? options->padding : brick_size;

    Vec3 world_min = vec3_sub(mesh->bbox_min, VEC3(padding, padding, padding));
    Vec3 world_max = vec3_add(mesh->bbox_max, VEC3(padding, padding, padding));

    if (options->world_min.x != 0.0f || options->world_min.y != 0.0f ||
        options->world_min.z != 0.0f || options->world_max.x != 0.0f ||
        options->world_max.y != 0.0f || options->world_max.z != 0.0f) {
        world_min = vec3_min(world_min, options->world_min);
        world_max = vec3_max(world_max, options->world_max);
    }

    Vec3 world_size = vec3_sub(world_max, world_min);
    uint32_t grid_x = (uint32_t)ceilf(world_size.x / brick_size);
    uint32_t grid_y = (uint32_t)ceilf(world_size.y / brick_size);
    uint32_t grid_z = (uint32_t)ceilf(world_size.z / brick_size);
    uint32_t grid_total = grid_x * grid_y * grid_z;

    uint32_t max_bricks = options->max_bricks;
    if (max_bricks == 0) {
        max_bricks = grid_total;
        if (max_bricks < 1024) max_bricks = 1024;
    }

    /* Build BVH */
    MeshBVH* bvh = bvh_build(arena, mesh);
    if (!bvh) {
        if (error) snprintf(error, 256, "Failed to build BVH");
        return OBJ_IO_ERROR_BVH_BUILD_FAILED;
    }

    /* Auto-detect shell mode for thin surfaces (open or closed) */
    VoxelizeOptions effective_opts = *options;
    bool was_shell = effective_opts.shell_mode;
    voxelize_options_auto_detect(&effective_opts, bvh, mesh);
    if (!was_shell && effective_opts.shell_mode) {
        printf("    [Auto-detected thin surface → shell mode]\n");
    }
    options = &effective_opts;

    /* Create world */
    WorldBrickMap* world = world_create(arena, world_min, world_max,
                                         voxel_size, max_bricks, 256);
    if (!world) {
        if (error) snprintf(error, 256, "Failed to create world brick map");
        return OBJ_IO_ERROR_OUT_OF_MEMORY;
    }

    /* Register materials (same as mesh_to_sdf) */
    if (mesh->material_names && mesh->material_name_count > 0) {
        for (uint32_t i = 0; i < mesh->material_name_count; i++) {
            if (mesh->material_names[i]) {
                world_register_material(world, mesh->material_names[i],
                                         VEC3(1.0f, 1.0f, 1.0f));
            }
        }
    }

    /* Phase 1: Coarse classification (CPU) */
    clock_t t1 = clock();
    BrickClassification* classes = classify_bricks_coarse(arena, bvh, mesh, world);
    if (!classes) {
        if (error) snprintf(error, 256, "Failed to classify bricks (phase 1)");
        return OBJ_IO_ERROR_VOXELIZE_FAILED;
    }
    double t_phase1 = (double)(clock() - t1) / CLOCKS_PER_SEC;

    /* Phase 2: Fine classification (CPU) */
    clock_t t2 = clock();
    classify_bricks_fine(classes, bvh, mesh, world, options);
    double t_phase2 = (double)(clock() - t2) / CLOCKS_PER_SEC;

    /* Mark uniform INSIDE bricks */
    for (uint32_t bz = 0; bz < classes->grid_z; bz++) {
        for (uint32_t by = 0; by < classes->grid_y; by++) {
            for (uint32_t bx = 0; bx < classes->grid_x; bx++) {
                uint32_t idx = bx + by * classes->grid_x +
                               bz * classes->grid_x * classes->grid_y;
                if (classes->classes[idx] == BRICK_CLASS_INSIDE) {
                    world_mark_brick_uniform_inside(world, (int32_t)bx,
                                                     (int32_t)by, (int32_t)bz);
                }
            }
        }
    }

    /* Collect surface brick list */
    uint32_t surface_count = 0;
    for (uint32_t i = 0; i < grid_total; i++) {
        if (classes->classes[i] == BRICK_CLASS_SURFACE)
            surface_count++;
    }

    uint32_t* surface_brick_list = arena_alloc_array(arena, uint32_t,
                                                       surface_count * 3);
    if (!surface_brick_list) {
        if (error) snprintf(error, 256, "Failed to allocate surface brick list");
        return OBJ_IO_ERROR_OUT_OF_MEMORY;
    }

    uint32_t si = 0;
    for (uint32_t bz = 0; bz < grid_z; bz++) {
        for (uint32_t by = 0; by < grid_y; by++) {
            for (uint32_t bx = 0; bx < grid_x; bx++) {
                uint32_t idx = bx + by * grid_x + bz * grid_x * grid_y;
                if (classes->classes[idx] == BRICK_CLASS_SURFACE) {
                    surface_brick_list[si * 3 + 0] = bx;
                    surface_brick_list[si * 3 + 1] = by;
                    surface_brick_list[si * 3 + 2] = bz;
                    si++;
                }
            }
        }
    }

    printf("    Phase 1 (coarse): %.2fs  Phase 2 (fine): %.2fs\n",
           t_phase1, t_phase2);
    printf("    Bricks: %u surface, %u inside, %u outside\n",
           classes->surface_count, classes->inside_count, classes->outside_count);

    /* Phase 3: GPU voxelization */
    GpuDevice* device = gpu_device_create();
    if (!device) {
        printf("  [GPU device creation failed, falling back to CPU Phase 3]\n");
        voxelize_surface_bricks(world, classes, bvh, mesh, options);
        *out_world = world;
        return OBJ_IO_SUCCESS;
    }

    printf("    GPU: %s\n", gpu_device_name(device));

    clock_t t3 = clock();
    GpuResult gpu_result = gpu_voxelize_surface_bricks(device, bvh, mesh,
                                                        surface_brick_list,
                                                        surface_count,
                                                        world, options);
    double t_phase3 = (double)(clock() - t3) / CLOCKS_PER_SEC;

    gpu_device_destroy(device);

    if (gpu_result != GPU_SUCCESS) {
        printf("  [GPU Phase 3 failed: %s, falling back to CPU]\n",
               gpu_error_string(gpu_result));
        voxelize_surface_bricks(world, classes, bvh, mesh, options);
    } else {
        printf("    Phase 3 (GPU): %.2fs (%u surface bricks)\n",
               t_phase3, surface_count);
    }

    *out_world = world;
    return OBJ_IO_SUCCESS;
#else
    printf("  [GPU not available in this build, using CPU]\n");
    return mesh_to_sdf(arena, mesh, options, out_world, error);
#endif
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr,
                "Usage: %s <input.obj> <output.obj> [voxel_size] [cpu|gpu] [shell]\n"
                "\n"
                "  voxel_size  Voxel size in world units (default: 0.5)\n"
                "  cpu|gpu     Voxelization backend (default: gpu)\n"
                "  shell       Enable shell mode for thin surfaces\n",
                argv[0]);
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_path = argv[2];
    float voxel_size = (argc > 3) ? (float)atof(argv[3]) : 0.5f;
    bool shell_mode = false;
    bool use_gpu = true; /* Default: GPU */

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "shell") == 0) shell_mode = true;
        else if (strcmp(argv[i], "cpu") == 0) use_gpu = false;
        else if (strcmp(argv[i], "gpu") == 0) use_gpu = true;
    }
    char error[256] = {0};

    /* Create arena */
    Arena* arena = arena_create(1024ULL * 1024 * 1024 * 2); /* 2 GB */
    if (!arena) {
        fprintf(stderr, "Failed to create arena\n");
        return 1;
    }

    printf("=== OBJ Roundtrip (voxel_size=%.2f, %s%s) ===\n",
           voxel_size, use_gpu ? "GPU" : "CPU",
           shell_mode ? ", shell mode" : "");

    /* Parse input mesh */
    clock_t t0 = clock();
    TriangleMesh* input_mesh = NULL;
    MtlLibrary* mtl = NULL;
    ObjIOResult result = obj_parse_file(arena, input_path, &OBJ_PARSE_DEFAULTS,
                                         &input_mesh, &mtl, error);
    if (result != OBJ_IO_SUCCESS) {
        fprintf(stderr, "Parse failed: %s\n", error);
        arena_destroy(arena);
        return 1;
    }
    double t_parse = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("  Parse: %.1fs (%u verts, %u faces)\n",
           t_parse, input_mesh->vertex_count, input_mesh->face_count);
    printf("  BBox: (%.1f, %.1f, %.1f) to (%.1f, %.1f, %.1f)\n",
           input_mesh->bbox_min.x, input_mesh->bbox_min.y, input_mesh->bbox_min.z,
           input_mesh->bbox_max.x, input_mesh->bbox_max.y, input_mesh->bbox_max.z);

    /* Voxelize */
    t0 = clock();
    VoxelizeOptions vox_opts = VOXELIZE_DEFAULTS;
    vox_opts.voxel_size = voxel_size;
    vox_opts.shell_mode = shell_mode;
    WorldBrickMap* world = NULL;

    if (use_gpu) {
        result = mesh_to_sdf_gpu(arena, input_mesh, &vox_opts, &world, error);
    } else {
        result = mesh_to_sdf(arena, input_mesh, &vox_opts, &world, error);
    }

    if (result != OBJ_IO_SUCCESS) {
        fprintf(stderr, "Voxelize failed: %s\n", error);
        arena_destroy(arena);
        return 1;
    }
    double t_voxelize = (double)(clock() - t0) / CLOCKS_PER_SEC;

    WorldStats stats = world_get_stats(world);
    printf("  Voxelize: %.1fs (%u active bricks, %u total)\n",
           t_voxelize, stats.active_bricks, stats.total_bricks);

    /* Register materials from MTL file */
    if (mtl) {
        mtl_register_materials(world, mtl);
    }

    /* Extract mesh */
    t0 = clock();
    TriangleMesh* output_mesh = NULL;
    result = sdf_to_mesh(arena, world, &MARCHING_CUBES_DEFAULTS, &output_mesh, error);
    if (result != OBJ_IO_SUCCESS) {
        fprintf(stderr, "Extract failed: %s\n", error);
        arena_destroy(arena);
        return 1;
    }
    double t_extract = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("  Extract: %.1fs (%u verts, %u faces)\n",
           t_extract, output_mesh->vertex_count, output_mesh->face_count);
    printf("  Output BBox: (%.1f, %.1f, %.1f) to (%.1f, %.1f, %.1f)\n",
           output_mesh->bbox_min.x, output_mesh->bbox_min.y, output_mesh->bbox_min.z,
           output_mesh->bbox_max.x, output_mesh->bbox_max.y, output_mesh->bbox_max.z);

    /* Export */
    t0 = clock();
    result = obj_export_file(output_path, output_mesh, world,
                              &OBJ_EXPORT_DEFAULTS, error);
    if (result != OBJ_IO_SUCCESS) {
        fprintf(stderr, "Export failed: %s\n", error);
        arena_destroy(arena);
        return 1;
    }
    double t_export = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("  Export: %.1fs\n", t_export);

    /* Compare */
    t0 = clock();
    MeshCompareResult cmp = mesh_compare(arena, input_mesh, output_mesh, voxel_size * 2);
    double t_compare = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("  Compare: %.1fs\n", t_compare);
    printf("  Hausdorff: %.2f  Mean: %.3f  RMS: %.3f\n",
           cmp.hausdorff_distance, cmp.mean_distance, cmp.rms_distance);
    printf("  Total: %.1fs\n", t_parse + t_voxelize + t_extract + t_export + t_compare);

    arena_destroy(arena);
    return 0;
}
