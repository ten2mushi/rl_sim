/**
 * OBJ I/O Command Line Tool
 *
 * Commands:
 *   roundtrip - Full pipeline validation (OBJ → SDF → OBJ)
 *   info      - Display mesh statistics
 *
 * Usage:
 *   obj_io_tool roundtrip input.obj [-o output.obj] [--voxel-size 0.1] [--report]
 *   obj_io_tool info input.obj
 */

#include "../include/obj_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ============================================================================
 * Command Line Parsing
 * ============================================================================ */

typedef struct {
    const char* command;
    const char* input_path;
    const char* output_path;
    float voxel_size;
    bool report;
    bool verbose;
    bool help;
    bool shell;
    float shell_thickness;
} ToolOptions;

static void print_usage(const char* program) {
    printf("OBJ I/O Tool - Convert between OBJ mesh and SDF formats\n\n");
    printf("Usage: %s <command> [options]\n\n", program);
    printf("Commands:\n");
    printf("  roundtrip Full pipeline validation (OBJ -> SDF -> OBJ)\n");
    printf("  info      Display mesh statistics\n");
    printf("\n");
    printf("Options:\n");
    printf("  -o, --output <path>    Output file path\n");
    printf("  -v, --voxel-size <f>   Voxel size for roundtrip (default: 0.1)\n");
    printf("  -r, --report           Generate detailed comparison report\n");
    printf("  --shell                Shell mode for thin surfaces (gyroid, etc.)\n");
    printf("  --shell-thickness <f>  Shell thickness (default: 2 * voxel_size)\n");
    printf("  --verbose              Print verbose output\n");
    printf("  -h, --help             Show this help message\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s roundtrip model.obj -o output.obj --voxel-size 0.05 --report\n", program);
    printf("  %s info model.obj\n", program);
}

static ToolOptions parse_args(int argc, char** argv) {
    ToolOptions opts = {
        .command = NULL,
        .input_path = NULL,
        .output_path = NULL,
        .voxel_size = 0.1f,
        .report = false,
        .verbose = false,
        .help = false,
        .shell = false,
        .shell_thickness = 0.0f
    };

    if (argc < 2) {
        opts.help = true;
        return opts;
    }

    int i = 1;

    /* First non-option argument is the command */
    if (argv[i][0] != '-') {
        opts.command = argv[i++];
    }

    /* Next non-option argument is the input path */
    while (i < argc && argv[i][0] != '-') {
        if (!opts.input_path) {
            opts.input_path = argv[i];
        }
        i++;
    }

    /* Parse options */
    while (i < argc) {
        const char* arg = argv[i];

        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            opts.help = true;
        } else if (strcmp(arg, "-o") == 0 || strcmp(arg, "--output") == 0) {
            if (i + 1 < argc) {
                opts.output_path = argv[++i];
            }
        } else if (strcmp(arg, "-v") == 0 || strcmp(arg, "--voxel-size") == 0) {
            if (i + 1 < argc) {
                opts.voxel_size = (float)atof(argv[++i]);
            }
        } else if (strcmp(arg, "-r") == 0 || strcmp(arg, "--report") == 0) {
            opts.report = true;
        } else if (strcmp(arg, "--shell") == 0) {
            opts.shell = true;
        } else if (strcmp(arg, "--shell-thickness") == 0) {
            if (i + 1 < argc) {
                opts.shell_thickness = (float)atof(argv[++i]);
                opts.shell = true;
            }
        } else if (strcmp(arg, "--verbose") == 0) {
            opts.verbose = true;
        }
        i++;
    }

    return opts;
}

/* ============================================================================
 * Timing Utilities
 * ============================================================================ */

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* ============================================================================
 * Roundtrip Command
 * ============================================================================ */

static int cmd_roundtrip(const ToolOptions* opts) {
    if (!opts->input_path) {
        fprintf(stderr, "Error: No input file specified\n");
        return 1;
    }

    printf("Roundtrip validation: %s\n", opts->input_path);
    printf("Voxel size: %.4f\n", opts->voxel_size);
    if (opts->shell) {
        printf("Shell mode: ON (thickness: %s)\n",
               opts->shell_thickness > 0 ? "custom" : "auto (2x voxel)");
    }

    Arena* arena = arena_create(500 * 1024 * 1024);
    if (!arena) {
        fprintf(stderr, "Error: Failed to allocate memory\n");
        return 1;
    }

    char error[256] = {0};
    double total_start = get_time_sec();

    /* Parse original */
    printf("\n[1/4] Parsing original OBJ...\n");
    double step_start = get_time_sec();
    TriangleMesh* original = NULL;
    MtlLibrary* mtl = NULL;
    ObjIOResult result = obj_parse_file(arena, opts->input_path, NULL, &original, &mtl, error);
    if (result != OBJ_IO_SUCCESS) {
        fprintf(stderr, "Error: %s\n", error);
        arena_destroy(arena);
        return 1;
    }
    mesh_compute_bbox(original);
    printf("  Vertices: %u, Faces: %u (%.2f sec)\n",
           original->vertex_count, original->face_count, get_time_sec() - step_start);
    if (mtl && mtl->count > 0) {
        printf("  Materials: %u\n", mtl->count);
    }

    /* Voxelize */
    printf("\n[2/4] Voxelizing...\n");
    step_start = get_time_sec();
    WorldBrickMap* world = NULL;
    VoxelizeOptions vox_opts = VOXELIZE_DEFAULTS;
    vox_opts.voxel_size = opts->voxel_size;
    vox_opts.shell_mode = opts->shell;
    vox_opts.shell_thickness = opts->shell_thickness;
    result = mesh_to_sdf(arena, original, &vox_opts, &world, error);
    if (result != OBJ_IO_SUCCESS) {
        fprintf(stderr, "Error: %s\n", error);
        arena_destroy(arena);
        return 1;
    }
    printf("  Bricks: %u (%.2f sec)\n", world->atlas_count, get_time_sec() - step_start);

    /* Register materials if present */
    if (mtl && mtl->count > 0) {
        mtl_register_materials(world, mtl);
        printf("  Materials registered: %u\n", world->material_count);
    }

    /* Extract mesh */
    printf("\n[3/4] Extracting mesh (marching cubes)...\n");
    step_start = get_time_sec();
    TriangleMesh* extracted = NULL;
    result = sdf_to_mesh(arena, world, &MARCHING_CUBES_DEFAULTS, &extracted, error);
    if (result != OBJ_IO_SUCCESS) {
        fprintf(stderr, "Error: %s\n", error);
        arena_destroy(arena);
        return 1;
    }
    mesh_compute_bbox(extracted);
    printf("  Vertices: %u, Faces: %u (%.2f sec)\n",
           extracted->vertex_count, extracted->face_count, get_time_sec() - step_start);

    /* Compare */
    printf("\n[4/4] Comparing meshes...\n");
    step_start = get_time_sec();
    float tolerance = opts->voxel_size * 2.0f;
    MeshCompareResult cmp = mesh_compare(arena, original, extracted, tolerance);
    printf("  Tolerance: %.4f (2x voxel size)\n", tolerance);
    printf("  Hausdorff distance: %.6f\n", cmp.hausdorff_distance);
    printf("  Mean distance: %.6f\n", cmp.mean_distance);
    printf("  RMS distance: %.6f\n", cmp.rms_distance);
    printf("  Samples: %u (%.2f sec)\n", cmp.sample_count, get_time_sec() - step_start);

    /* Report */
    if (opts->report) {
        printf("\n--- Detailed Report ---\n");
        printf("Original mesh:\n");
        printf("  Bbox min: (%.4f, %.4f, %.4f)\n",
               original->bbox_min.x, original->bbox_min.y, original->bbox_min.z);
        printf("  Bbox max: (%.4f, %.4f, %.4f)\n",
               original->bbox_max.x, original->bbox_max.y, original->bbox_max.z);

        printf("Extracted mesh:\n");
        printf("  Bbox min: (%.4f, %.4f, %.4f)\n",
               extracted->bbox_min.x, extracted->bbox_min.y, extracted->bbox_min.z);
        printf("  Bbox max: (%.4f, %.4f, %.4f)\n",
               extracted->bbox_max.x, extracted->bbox_max.y, extracted->bbox_max.z);

        printf("Bbox difference:\n");
        printf("  Min: (%.4f, %.4f, %.4f)\n",
               extracted->bbox_min.x - original->bbox_min.x,
               extracted->bbox_min.y - original->bbox_min.y,
               extracted->bbox_min.z - original->bbox_min.z);
        printf("  Max: (%.4f, %.4f, %.4f)\n",
               extracted->bbox_max.x - original->bbox_max.x,
               extracted->bbox_max.y - original->bbox_max.y,
               extracted->bbox_max.z - original->bbox_max.z);
    }

    /* Summary */
    double total_elapsed = get_time_sec() - total_start;
    printf("\n=== Summary ===\n");
    printf("Total time: %.2f seconds\n", total_elapsed);
    if (cmp.passed) {
        printf("Result: PASSED (Hausdorff <= tolerance)\n");
    } else {
        printf("Result: FAILED (Hausdorff > tolerance)\n");
    }

    /* Optionally save output */
    if (opts->output_path) {
        printf("\nSaving output to: %s\n", opts->output_path);
        result = obj_export_file(opts->output_path, extracted, world, &OBJ_EXPORT_DEFAULTS, error);
        if (result != OBJ_IO_SUCCESS) {
            fprintf(stderr, "Error: %s\n", error);
        } else {
            printf("Export successful!\n");
        }
    }

    arena_destroy(arena);
    return cmp.passed ? 0 : 1;
}

/* ============================================================================
 * Info Command
 * ============================================================================ */

static int cmd_info(const ToolOptions* opts) {
    if (!opts->input_path) {
        fprintf(stderr, "Error: No input file specified\n");
        return 1;
    }

    const char* path = opts->input_path;
    size_t len = strlen(path);

    Arena* arena = arena_create(100 * 1024 * 1024);
    if (!arena) {
        fprintf(stderr, "Error: Failed to allocate memory\n");
        return 1;
    }

    char error[256] = {0};

    /* Check file extension */
    if (len > 4 && strcmp(path + len - 4, ".obj") == 0) {
        /* OBJ file */
        printf("File: %s (OBJ mesh)\n\n", path);

        TriangleMesh* mesh = NULL;
        MtlLibrary* mtl = NULL;
        ObjIOResult result = obj_parse_file(arena, path, NULL, &mesh, &mtl, error);

        if (result != OBJ_IO_SUCCESS) {
            fprintf(stderr, "Error: %s\n", error);
            arena_destroy(arena);
            return 1;
        }

        mesh_compute_bbox(mesh);

        printf("Vertices: %u\n", mesh->vertex_count);
        printf("Faces: %u\n", mesh->face_count);
        printf("Has normals: %s\n", mesh->has_normals ? "yes" : "no");
        printf("\n");
        printf("Bounding box:\n");
        printf("  Min: (%.4f, %.4f, %.4f)\n",
               mesh->bbox_min.x, mesh->bbox_min.y, mesh->bbox_min.z);
        printf("  Max: (%.4f, %.4f, %.4f)\n",
               mesh->bbox_max.x, mesh->bbox_max.y, mesh->bbox_max.z);
        printf("  Size: (%.4f, %.4f, %.4f)\n",
               mesh->bbox_max.x - mesh->bbox_min.x,
               mesh->bbox_max.y - mesh->bbox_min.y,
               mesh->bbox_max.z - mesh->bbox_min.z);

        if (mtl && mtl->count > 0) {
            printf("\nMaterials: %u\n", mtl->count);
            for (uint32_t i = 0; i < mtl->count && i < 10; i++) {
                printf("  [%u] %s", i, mtl->materials[i].name);
                if (mtl->materials[i].has_Kd) {
                    printf(" (Kd: %.2f, %.2f, %.2f)",
                           mtl->materials[i].Kd.x,
                           mtl->materials[i].Kd.y,
                           mtl->materials[i].Kd.z);
                }
                printf("\n");
            }
            if (mtl->count > 10) {
                printf("  ... and %u more\n", mtl->count - 10);
            }
        }

    } else {
        fprintf(stderr, "Error: Unknown file type (expected .obj)\n");
        arena_destroy(arena);
        return 1;
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char** argv) {
    ToolOptions opts = parse_args(argc, argv);

    if (opts.help || !opts.command) {
        print_usage(argv[0]);
        return opts.help ? 0 : 1;
    }

    if (strcmp(opts.command, "roundtrip") == 0) {
        return cmd_roundtrip(&opts);
    } else if (strcmp(opts.command, "info") == 0) {
        return cmd_info(&opts);
    } else {
        fprintf(stderr, "Error: Unknown command '%s'\n", opts.command);
        print_usage(argv[0]);
        return 1;
    }
}
