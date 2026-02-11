/**
 * OBJ I/O Performance Benchmarks
 *
 * Benchmarks:
 * - OBJ parsing throughput
 * - BVH construction time
 * - Brick classification (sparse optimization)
 * - Voxelization throughput
 * - Marching cubes extraction
 * - Full roundtrip pipeline
 */

#include "../include/obj_io.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================================================
 * Benchmark Utilities
 * ============================================================================ */

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

#define BENCH_START() double _bench_start = get_time_sec()
#define BENCH_END() (get_time_sec() - _bench_start)

#define BENCH(name, iterations, code) do { \
    printf("  %-40s", name); \
    fflush(stdout); \
    BENCH_START(); \
    for (int _i = 0; _i < (iterations); _i++) { code; } \
    double _elapsed = BENCH_END(); \
    printf("%8.3f ms (%.1f/s)\n", _elapsed * 1000.0 / (iterations), (iterations) / _elapsed); \
} while(0)

#define BENCH_SINGLE(name, code) do { \
    printf("  %-40s", name); \
    fflush(stdout); \
    BENCH_START(); \
    code; \
    double _elapsed = BENCH_END(); \
    printf("%8.3f ms\n", _elapsed * 1000.0); \
} while(0)

/* ============================================================================
 * Helper: Generate test meshes
 * ============================================================================ */

static void write_test_cube_obj(const char* path) {
    const char* content =
        "v -0.5 -0.5 -0.5\n"
        "v  0.5 -0.5 -0.5\n"
        "v  0.5  0.5 -0.5\n"
        "v -0.5  0.5 -0.5\n"
        "v -0.5 -0.5  0.5\n"
        "v  0.5 -0.5  0.5\n"
        "v  0.5  0.5  0.5\n"
        "v -0.5  0.5  0.5\n"
        "f 1 2 3 4\n"
        "f 5 6 7 8\n"
        "f 1 2 6 5\n"
        "f 2 3 7 6\n"
        "f 3 4 8 7\n"
        "f 4 1 5 8\n";
    FILE* f = fopen(path, "w");
    if (f) {
        fputs(content, f);
        fclose(f);
    }
}

/* Generate a larger mesh for stress testing */
static void write_large_mesh_obj(const char* path, int subdivisions) {
    FILE* f = fopen(path, "w");
    if (!f) return;

    /* Generate grid of vertices */
    int n = subdivisions + 1;
    for (int z = 0; z < n; z++) {
        for (int x = 0; x < n; x++) {
            float fx = (float)x / subdivisions - 0.5f;
            float fz = (float)z / subdivisions - 0.5f;
            /* Wavy height */
            float y = 0.1f * sinf(fx * 10) * cosf(fz * 10);
            fprintf(f, "v %.6f %.6f %.6f\n", fx, y, fz);
        }
    }

    /* Generate faces */
    for (int z = 0; z < subdivisions; z++) {
        for (int x = 0; x < subdivisions; x++) {
            int v0 = z * n + x + 1;
            int v1 = v0 + 1;
            int v2 = v0 + n + 1;
            int v3 = v0 + n;
            fprintf(f, "f %d %d %d %d\n", v0, v1, v2, v3);
        }
    }

    fclose(f);
}

/* ============================================================================
 * Benchmarks
 * ============================================================================ */

static void bench_obj_parsing(void) {
    printf("\n--- OBJ Parsing ---\n");

    const char* small_path = "/tmp/bench_small.obj";
    const char* medium_path = "/tmp/bench_medium.obj";
    const char* large_path = "/tmp/bench_large.obj";

    write_test_cube_obj(small_path);
    write_large_mesh_obj(medium_path, 50);  /* 2500 faces */
    write_large_mesh_obj(large_path, 200);  /* 40000 faces */

    Arena* arena = arena_create(100 * 1024 * 1024);
    char error[256];

    /* Small file parsing */
    BENCH("Small OBJ (12 faces)", 100, {
        arena_reset(arena);
        TriangleMesh* mesh = NULL;
        obj_parse_file(arena, small_path, NULL, &mesh, NULL, error);
    });

    /* Medium file parsing */
    BENCH("Medium OBJ (~5K faces)", 10, {
        arena_reset(arena);
        TriangleMesh* mesh = NULL;
        obj_parse_file(arena, medium_path, NULL, &mesh, NULL, error);
    });

    /* Large file parsing */
    BENCH("Large OBJ (~80K faces)", 3, {
        arena_reset(arena);
        TriangleMesh* mesh = NULL;
        obj_parse_file(arena, large_path, NULL, &mesh, NULL, error);
    });

    arena_destroy(arena);
    remove(small_path);
    remove(medium_path);
    remove(large_path);
}

static void bench_bvh_construction(void) {
    printf("\n--- BVH Construction ---\n");

    const char* path = "/tmp/bench_bvh.obj";
    write_large_mesh_obj(path, 100);  /* 10000 faces */

    Arena* arena = arena_create(100 * 1024 * 1024);
    char error[256];

    TriangleMesh* mesh = NULL;
    obj_parse_file(arena, path, NULL, &mesh, NULL, error);

    if (mesh) {
        printf("  Mesh: %u vertices, %u faces\n", mesh->vertex_count, mesh->face_count);

        BENCH("BVH build (20K faces)", 5, {
            MeshBVH* bvh = bvh_build(arena, mesh);
            (void)bvh;
        });
    }

    arena_destroy(arena);
    remove(path);
}

static void bench_brick_classification(void) {
    printf("\n--- Brick Classification ---\n");

    Arena* arena = arena_create(200 * 1024 * 1024);

    /* Create cube mesh */
    TriangleMesh* mesh = mesh_create(arena, 8, 12);
    mesh_add_vertex(mesh, -0.5f, -0.5f, -0.5f);
    mesh_add_vertex(mesh,  0.5f, -0.5f, -0.5f);
    mesh_add_vertex(mesh,  0.5f,  0.5f, -0.5f);
    mesh_add_vertex(mesh, -0.5f,  0.5f, -0.5f);
    mesh_add_vertex(mesh, -0.5f, -0.5f,  0.5f);
    mesh_add_vertex(mesh,  0.5f, -0.5f,  0.5f);
    mesh_add_vertex(mesh,  0.5f,  0.5f,  0.5f);
    mesh_add_vertex(mesh, -0.5f,  0.5f,  0.5f);
    /* Outward-facing normals */
    mesh_add_face(mesh, 0, 2, 1, 0); mesh_add_face(mesh, 0, 3, 2, 0);
    mesh_add_face(mesh, 4, 5, 6, 0); mesh_add_face(mesh, 4, 6, 7, 0);
    mesh_add_face(mesh, 3, 6, 2, 0); mesh_add_face(mesh, 3, 7, 6, 0);
    mesh_add_face(mesh, 0, 1, 5, 0); mesh_add_face(mesh, 0, 5, 4, 0);
    mesh_add_face(mesh, 1, 2, 6, 0); mesh_add_face(mesh, 1, 6, 5, 0);
    mesh_add_face(mesh, 0, 4, 7, 0); mesh_add_face(mesh, 0, 7, 3, 0);
    mesh_compute_bbox(mesh);

    MeshBVH* bvh = bvh_build(arena, mesh);

    /* Small world (few bricks) */
    WorldBrickMap* world_small = world_create(arena,
                                               VEC3(-1, -1, -1), VEC3(1, 1, 1),
                                               0.2f, 256, 64);

    BENCH("Classify small world", 20, {
        BrickClassification* classes = classify_bricks_coarse(arena, bvh, mesh, world_small);
        (void)classes;
    });

    /* Large world (many bricks, sparse) */
    WorldBrickMap* world_large = world_create(arena,
                                               VEC3(-5, -5, -5), VEC3(5, 5, 5),
                                               0.2f, 4096, 512);

    BENCH_SINGLE("Classify large world (sparse benefit)", {
        BrickClassification* classes = classify_bricks_coarse(arena, bvh, mesh, world_large);
        printf("\n    [outside=%u, surface=%u, ratio=%.1fx]",
               classes->outside_count, classes->surface_count,
               (float)classes->outside_count / (classes->surface_count > 0 ? classes->surface_count : 1));
    });

    arena_destroy(arena);
}

static void bench_voxelization(void) {
    printf("\n--- Voxelization ---\n");

    const char* path = "/tmp/bench_vox.obj";
    write_test_cube_obj(path);

    Arena* arena = arena_create(200 * 1024 * 1024);
    char error[256];

    TriangleMesh* mesh = NULL;
    obj_parse_file(arena, path, NULL, &mesh, NULL, error);

    if (mesh) {
        VoxelizeOptions opts = VOXELIZE_DEFAULTS;

        /* Coarse voxelization */
        opts.voxel_size = 0.1f;
        BENCH("Voxelize cube (0.1 voxel)", 5, {
            arena_reset(arena);
            mesh = NULL;
            obj_parse_file(arena, path, NULL, &mesh, NULL, error);
            WorldBrickMap* world = NULL;
            mesh_to_sdf(arena, mesh, &opts, &world, error);
        });

        /* Fine voxelization */
        opts.voxel_size = 0.05f;
        BENCH("Voxelize cube (0.05 voxel)", 3, {
            arena_reset(arena);
            mesh = NULL;
            obj_parse_file(arena, path, NULL, &mesh, NULL, error);
            WorldBrickMap* world = NULL;
            mesh_to_sdf(arena, mesh, &opts, &world, error);
        });
    }

    arena_destroy(arena);
    remove(path);
}

static void bench_marching_cubes(void) {
    printf("\n--- Marching Cubes ---\n");

    const char* path = "/tmp/bench_mc.obj";
    write_test_cube_obj(path);

    Arena* arena = arena_create(200 * 1024 * 1024);
    char error[256];

    TriangleMesh* mesh = NULL;
    obj_parse_file(arena, path, NULL, &mesh, NULL, error);

    if (mesh) {
        VoxelizeOptions vox_opts = VOXELIZE_DEFAULTS;
        vox_opts.voxel_size = 0.05f;

        WorldBrickMap* world = NULL;
        mesh_to_sdf(arena, mesh, &vox_opts, &world, error);

        if (world && world->atlas_count > 0) {
            printf("  World: %u bricks allocated\n", world->atlas_count);

            BENCH("Extract mesh (marching cubes)", 10, {
                TriangleMesh* extracted = NULL;
                sdf_to_mesh(arena, world, &MARCHING_CUBES_DEFAULTS, &extracted, error);
                if (extracted) {
                    (void)extracted->face_count;
                }
            });
        }
    }

    arena_destroy(arena);
    remove(path);
}

static void bench_full_roundtrip(void) {
    printf("\n--- Full Roundtrip ---\n");

    const char* input_path = "/tmp/bench_roundtrip_in.obj";
    const char* output_path = "/tmp/bench_roundtrip_out.obj";

    /* Small mesh */
    write_test_cube_obj(input_path);

    Arena* arena = arena_create(200 * 1024 * 1024);
    char error[256];

    VoxelizeOptions bench_opts = VOXELIZE_DEFAULTS;
    bench_opts.voxel_size = 0.05f;

    BENCH("Roundtrip small mesh", 5, {
        arena_reset(arena);
        WorldBrickMap* world = NULL;
        obj_to_world(arena, input_path, &bench_opts, &world, error);
        if (world) {
            world_to_obj(world, output_path, error);
        }
    });

    /* Medium mesh */
    write_large_mesh_obj(input_path, 30);  /* ~900 faces */

    BENCH("Roundtrip medium mesh (~1.8K faces)", 3, {
        arena_reset(arena);
        WorldBrickMap* world = NULL;
        obj_to_world(arena, input_path, &bench_opts, &world, error);
        if (world) {
            world_to_obj(world, output_path, error);
        }
    });

    arena_destroy(arena);
    remove(input_path);
    remove(output_path);
    remove("/tmp/bench_roundtrip_out.mtl");
}

static void bench_bvh_queries(void) {
    printf("\n--- BVH Queries ---\n");

    Arena* arena = arena_create(100 * 1024 * 1024);

    /* Create cube mesh */
    TriangleMesh* mesh = mesh_create(arena, 8, 12);
    mesh_add_vertex(mesh, -0.5f, -0.5f, -0.5f);
    mesh_add_vertex(mesh,  0.5f, -0.5f, -0.5f);
    mesh_add_vertex(mesh,  0.5f,  0.5f, -0.5f);
    mesh_add_vertex(mesh, -0.5f,  0.5f, -0.5f);
    mesh_add_vertex(mesh, -0.5f, -0.5f,  0.5f);
    mesh_add_vertex(mesh,  0.5f, -0.5f,  0.5f);
    mesh_add_vertex(mesh,  0.5f,  0.5f,  0.5f);
    mesh_add_vertex(mesh, -0.5f,  0.5f,  0.5f);
    /* Outward-facing normals */
    mesh_add_face(mesh, 0, 2, 1, 0); mesh_add_face(mesh, 0, 3, 2, 0);
    mesh_add_face(mesh, 4, 5, 6, 0); mesh_add_face(mesh, 4, 6, 7, 0);
    mesh_add_face(mesh, 3, 6, 2, 0); mesh_add_face(mesh, 3, 7, 6, 0);
    mesh_add_face(mesh, 0, 1, 5, 0); mesh_add_face(mesh, 0, 5, 4, 0);
    mesh_add_face(mesh, 1, 2, 6, 0); mesh_add_face(mesh, 1, 6, 5, 0);
    mesh_add_face(mesh, 0, 4, 7, 0); mesh_add_face(mesh, 0, 7, 3, 0);
    mesh_compute_bbox(mesh);

    MeshBVH* bvh = bvh_build(arena, mesh);

    /* Ray intersection benchmark */
    Vec3 origin = VEC3(-2, 0, 0);
    Vec3 dir = VEC3(1, 0, 0);
    float hit_t;
    uint32_t hit_face;

    BENCH("Ray intersect (1M queries)", 1, {
        for (int i = 0; i < 1000000; i++) {
            bvh_ray_intersect(bvh, mesh, origin, dir, 100.0f, &hit_t, &hit_face);
        }
    });

    /* Closest point benchmark */
    Vec3 point = VEC3(0.7f, 0.3f, 0.2f);
    Vec3 closest;
    uint32_t face_idx;

    BENCH("Closest point (100K queries)", 1, {
        for (int i = 0; i < 100000; i++) {
            bvh_closest_point(bvh, mesh, point, &closest, &face_idx);
        }
    });

    /* Inside/outside benchmark */
    Vec3 test_point = VEC3(0.1f, 0.1f, 0.1f);

    BENCH("Inside/outside (100K queries)", 1, {
        for (int i = 0; i < 100000; i++) {
            bvh_inside_outside(bvh, mesh, test_point);
        }
    });

    arena_destroy(arena);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("=== OBJ I/O Performance Benchmarks ===\n");
    printf("Note: Times are per-operation averages\n");

    bench_obj_parsing();
    bench_bvh_construction();
    bench_bvh_queries();
    bench_brick_classification();
    bench_voxelization();
    bench_marching_cubes();
    bench_full_roundtrip();

    printf("\n=== Benchmarks Complete ===\n");
    return 0;
}
