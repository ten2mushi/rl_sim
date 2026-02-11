/**
 * Marching Cubes Unit Tests
 *
 * Tests:
 * - Basic mesh extraction from SDF
 * - Vertex interpolation
 * - Normal computation
 * - Material preservation
 * - Edge cases (empty world, uniform regions)
 */

#include "../include/obj_io.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "test_harness.h"

/* ============================================================================
 * Helper: Create test worlds with known SDF
 * ============================================================================ */

static WorldBrickMap* create_sphere_sdf_world(Arena* arena, float radius) {
    /* Create world that contains a sphere SDF */
    float extent = radius * 2.0f;
    WorldBrickMap* world = world_create(arena,
                                         VEC3(-extent, -extent, -extent),
                                         VEC3(extent, extent, extent),
                                         0.1f, 512, 128);
    if (!world) return NULL;

    /* Fill with sphere SDF: distance = length(p) - radius */
    float brick_size = world->brick_size_world;

    for (float bz = world->world_min.z; bz < world->world_max.z; bz += brick_size) {
        for (float by = world->world_min.y; by < world->world_max.y; by += brick_size) {
            for (float bx = world->world_min.x; bx < world->world_max.x; bx += brick_size) {
                /* Check if brick is near surface */
                Vec3 brick_center = VEC3(bx + brick_size/2, by + brick_size/2, bz + brick_size/2);
                float dist_to_surface = fabsf(vec3_length(brick_center) - radius);

                if (dist_to_surface < brick_size * 2) {
                    /* Allocate and fill brick */
                    int32_t bxi, byi, bzi;
                    world_pos_to_brick(world, brick_center, &bxi, &byi, &bzi);
                    int32_t atlas_idx = world_alloc_brick(world, bxi, byi, bzi);
                    if (atlas_idx < 0) continue;

                    int8_t* sdf = world_brick_sdf(world, atlas_idx);

                    for (int vz = 0; vz < 8; vz++) {
                        for (int vy = 0; vy < 8; vy++) {
                            for (int vx = 0; vx < 8; vx++) {
                                Vec3 voxel_pos = VEC3(
                                    bx + (vx + 0.5f) * world->voxel_size,
                                    by + (vy + 0.5f) * world->voxel_size,
                                    bz + (vz + 0.5f) * world->voxel_size
                                );
                                float d = vec3_length(voxel_pos) - radius;
                                int idx = vx + (vy << 3) + (vz << 6);
                                sdf[idx] = sdf_quantize(d, world->inv_sdf_scale);
                            }
                        }
                    }
                }
            }
        }
    }

    return world;
}

/* ============================================================================
 * Tests
 * ============================================================================ */

TEST(extract_sphere_mesh) {
    Arena* arena = arena_create(50 * 1024 * 1024);

    WorldBrickMap* world = create_sphere_sdf_world(arena, 0.5f);
    ASSERT_TRUE(world != NULL);
    ASSERT_GT(world->atlas_count, 0);

    TriangleMesh* mesh = NULL;
    char error[256];
    MarchingCubesOptions opts = MARCHING_CUBES_DEFAULTS;

    ObjIOResult result = sdf_to_mesh(arena, world, &opts, &mesh, error);
    ASSERT_EQ(result, OBJ_IO_SUCCESS);
    ASSERT_TRUE(mesh != NULL);

    /* Should have extracted triangles */
    ASSERT_GT(mesh->face_count, 0);
    ASSERT_GT(mesh->vertex_count, 0);

    /* Vertices should be near the sphere surface (radius 0.5) */
    for (uint32_t i = 0; i < mesh->vertex_count; i++) {
        Vec3 v = VEC3(mesh->vx[i], mesh->vy[i], mesh->vz[i]);
        float dist = fabsf(vec3_length(v) - 0.5f);
        /* Allow tolerance based on voxel size */
        ASSERT_LT(dist, world->voxel_size * 2);
    }

    arena_destroy(arena);
    return 0;
}

TEST(mesh_bbox_matches_sdf) {
    Arena* arena = arena_create(50 * 1024 * 1024);

    WorldBrickMap* world = create_sphere_sdf_world(arena, 0.5f);
    ASSERT_TRUE(world != NULL);

    TriangleMesh* mesh = NULL;
    char error[256];
    ObjIOResult result = sdf_to_mesh(arena, world, &MARCHING_CUBES_DEFAULTS, &mesh, error);
    ASSERT_EQ(result, OBJ_IO_SUCCESS);

    mesh_compute_bbox(mesh);

    /* Mesh bbox should approximately match sphere bounds */
    ASSERT_FLOAT_NEAR(mesh->bbox_min.x, -0.5f, world->voxel_size * 2);
    ASSERT_FLOAT_NEAR(mesh->bbox_max.x,  0.5f, world->voxel_size * 2);
    ASSERT_FLOAT_NEAR(mesh->bbox_min.y, -0.5f, world->voxel_size * 2);
    ASSERT_FLOAT_NEAR(mesh->bbox_max.y,  0.5f, world->voxel_size * 2);

    arena_destroy(arena);
    return 0;
}

TEST(normals_computed) {
    Arena* arena = arena_create(50 * 1024 * 1024);

    WorldBrickMap* world = create_sphere_sdf_world(arena, 0.5f);
    ASSERT_TRUE(world != NULL);

    TriangleMesh* mesh = NULL;
    char error[256];
    MarchingCubesOptions opts = MARCHING_CUBES_DEFAULTS;
    opts.compute_normals = true;

    ObjIOResult result = sdf_to_mesh(arena, world, &opts, &mesh, error);
    ASSERT_EQ(result, OBJ_IO_SUCCESS);
    ASSERT_TRUE(mesh->has_normals);

    /* For a sphere, normals should point outward (same direction as position) */
    for (uint32_t i = 0; i < mesh->vertex_count && i < 10; i++) {
        Vec3 pos = VEC3(mesh->vx[i], mesh->vy[i], mesh->vz[i]);
        Vec3 normal = VEC3(mesh->nx[i], mesh->ny[i], mesh->nz[i]);

        /* Normalize position to get expected normal direction */
        float len = vec3_length(pos);
        if (len > 0.001f) {
            Vec3 expected = vec3_scale(pos, 1.0f / len);
            float dot = vec3_dot(normal, expected);
            /* Normal should roughly align with outward direction */
            ASSERT_GT(dot, 0.5f);
        }
    }

    arena_destroy(arena);
    return 0;
}

TEST(empty_world_produces_empty_mesh) {
    Arena* arena = arena_create(10 * 1024 * 1024);

    /* Create world with no allocated bricks */
    WorldBrickMap* world = world_create(arena,
                                         VEC3(-1, -1, -1), VEC3(1, 1, 1),
                                         0.1f, 256, 64);
    ASSERT_TRUE(world != NULL);
    ASSERT_EQ(world->atlas_count, 0);

    TriangleMesh* mesh = NULL;
    char error[256];
    ObjIOResult result = sdf_to_mesh(arena, world, &MARCHING_CUBES_DEFAULTS, &mesh, error);

    /* Should return empty mesh error or produce mesh with zero faces */
    if (result == OBJ_IO_SUCCESS) {
        ASSERT_EQ(mesh->face_count, 0);
    } else {
        ASSERT_EQ(result, OBJ_IO_ERROR_EMPTY_MESH);
    }

    arena_destroy(arena);
    return 0;
}

TEST(iso_value_affects_surface) {
    Arena* arena = arena_create(50 * 1024 * 1024);

    WorldBrickMap* world = create_sphere_sdf_world(arena, 0.5f);
    ASSERT_TRUE(world != NULL);

    /* Extract with default iso_value (0) */
    TriangleMesh* mesh1 = NULL;
    char error[256];
    MarchingCubesOptions opts1 = MARCHING_CUBES_DEFAULTS;
    opts1.iso_value = 0.0f;
    sdf_to_mesh(arena, world, &opts1, &mesh1, error);

    /* Extract with positive iso_value (smaller surface) */
    TriangleMesh* mesh2 = NULL;
    MarchingCubesOptions opts2 = MARCHING_CUBES_DEFAULTS;
    opts2.iso_value = 0.1f;
    sdf_to_mesh(arena, world, &opts2, &mesh2, error);

    if (mesh1 && mesh2 && mesh1->face_count > 0 && mesh2->face_count > 0) {
        /* Mesh with higher iso_value should be smaller (fewer faces or smaller bbox) */
        mesh_compute_bbox(mesh1);
        mesh_compute_bbox(mesh2);

        float extent1 = mesh1->bbox_max.x - mesh1->bbox_min.x;
        float extent2 = mesh2->bbox_max.x - mesh2->bbox_min.x;

        /* Positive iso extracts inside the original surface */
        ASSERT_LT(extent2, extent1);
    }

    arena_destroy(arena);
    return 0;
}

TEST(roundtrip_preserves_topology) {
    Arena* arena = arena_create(100 * 1024 * 1024);

    /* Create simple cube OBJ with outward-facing normals (CCW winding from outside) */
    const char* obj_content =
        "v -0.5 -0.5 -0.5\n"  /* 1: front-bottom-left */
        "v  0.5 -0.5 -0.5\n"  /* 2: front-bottom-right */
        "v  0.5  0.5 -0.5\n"  /* 3: front-top-right */
        "v -0.5  0.5 -0.5\n"  /* 4: front-top-left */
        "v -0.5 -0.5  0.5\n"  /* 5: back-bottom-left */
        "v  0.5 -0.5  0.5\n"  /* 6: back-bottom-right */
        "v  0.5  0.5  0.5\n"  /* 7: back-top-right */
        "v -0.5  0.5  0.5\n"  /* 8: back-top-left */
        "f 1 4 3 2\n"         /* Front face (z=-0.5, normal -Z) */
        "f 5 6 7 8\n"         /* Back face (z=0.5, normal +Z) */
        "f 1 2 6 5\n"         /* Bottom face (y=-0.5, normal -Y) */
        "f 4 8 7 3\n"         /* Top face (y=0.5, normal +Y) */
        "f 1 5 8 4\n"         /* Left face (x=-0.5, normal -X) */
        "f 2 3 7 6\n";        /* Right face (x=0.5, normal +X) */

    const char* temp_path = "/tmp/test_mc_cube.obj";
    FILE* f = fopen(temp_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(obj_content, f);
    fclose(f);

    /* Parse original */
    TriangleMesh* original = NULL;
    char error[256];
    obj_parse_file(arena, temp_path, NULL, &original, NULL, error);
    ASSERT_TRUE(original != NULL);
    mesh_compute_bbox(original);

    /* Voxelize */
    WorldBrickMap* world = NULL;
    VoxelizeOptions vox_opts = VOXELIZE_DEFAULTS;
    vox_opts.voxel_size = 0.05f;
    mesh_to_sdf(arena, original, &vox_opts, &world, error);
    ASSERT_TRUE(world != NULL);

    /* Extract mesh */
    TriangleMesh* extracted = NULL;
    sdf_to_mesh(arena, world, &MARCHING_CUBES_DEFAULTS, &extracted, error);
    ASSERT_TRUE(extracted != NULL);
    mesh_compute_bbox(extracted);

    /* Compare bounding boxes */
    float tol = vox_opts.voxel_size * 2;
    ASSERT_FLOAT_NEAR(extracted->bbox_min.x, original->bbox_min.x, tol);
    ASSERT_FLOAT_NEAR(extracted->bbox_max.x, original->bbox_max.x, tol);
    ASSERT_FLOAT_NEAR(extracted->bbox_min.y, original->bbox_min.y, tol);
    ASSERT_FLOAT_NEAR(extracted->bbox_max.y, original->bbox_max.y, tol);
    ASSERT_FLOAT_NEAR(extracted->bbox_min.z, original->bbox_min.z, tol);
    ASSERT_FLOAT_NEAR(extracted->bbox_max.z, original->bbox_max.z, tol);

    arena_destroy(arena);
    remove(temp_path);
    return 0;
}

TEST(large_mesh_extraction) {
    Arena* arena = arena_create(200 * 1024 * 1024);

    /* Create larger sphere for more triangles */
    WorldBrickMap* world = create_sphere_sdf_world(arena, 1.0f);
    ASSERT_TRUE(world != NULL);

    TriangleMesh* mesh = NULL;
    char error[256];
    ObjIOResult result = sdf_to_mesh(arena, world, &MARCHING_CUBES_DEFAULTS, &mesh, error);
    ASSERT_EQ(result, OBJ_IO_SUCCESS);

    /* Should produce a reasonable number of triangles */
    ASSERT_GT(mesh->face_count, 100);
    ASSERT_GT(mesh->vertex_count, 50);

    arena_destroy(arena);
    return 0;
}

TEST(vertex_welding_reduces_count) {
    Arena* arena = arena_create(100 * 1024 * 1024);

    WorldBrickMap* world = create_sphere_sdf_world(arena, 0.5f);
    ASSERT_TRUE(world != NULL);

    /* Extract WITHOUT welding */
    TriangleMesh* mesh_noweld = NULL;
    char error[256];
    MarchingCubesOptions opts_noweld = MARCHING_CUBES_DEFAULTS;
    opts_noweld.weld_vertices = false;
    ObjIOResult r1 = sdf_to_mesh(arena, world, &opts_noweld, &mesh_noweld, error);
    ASSERT_EQ(r1, OBJ_IO_SUCCESS);
    ASSERT_GT(mesh_noweld->vertex_count, 0);

    /* Extract WITH welding */
    TriangleMesh* mesh_weld = NULL;
    MarchingCubesOptions opts_weld = MARCHING_CUBES_DEFAULTS;
    opts_weld.weld_vertices = true;
    opts_weld.weld_tolerance = 0.0001f;
    ObjIOResult r2 = sdf_to_mesh(arena, world, &opts_weld, &mesh_weld, error);
    ASSERT_EQ(r2, OBJ_IO_SUCCESS);
    ASSERT_GT(mesh_weld->vertex_count, 0);

    /* Welding should significantly reduce vertex count */
    ASSERT_LT(mesh_weld->vertex_count, mesh_noweld->vertex_count);

    /* Face count must be identical (welding doesn't remove faces) */
    ASSERT_EQ(mesh_weld->face_count, mesh_noweld->face_count);

    arena_destroy(arena);
    return 0;
}

TEST(vertex_welding_preserves_geometry) {
    Arena* arena = arena_create(100 * 1024 * 1024);

    WorldBrickMap* world = create_sphere_sdf_world(arena, 0.5f);
    ASSERT_TRUE(world != NULL);

    /* Extract with welding */
    TriangleMesh* mesh = NULL;
    char error[256];
    MarchingCubesOptions opts = MARCHING_CUBES_DEFAULTS;
    opts.weld_vertices = true;
    opts.weld_tolerance = 0.0001f;
    ObjIOResult result = sdf_to_mesh(arena, world, &opts, &mesh, error);
    ASSERT_EQ(result, OBJ_IO_SUCCESS);
    mesh_compute_bbox(mesh);

    /* BBox should still match sphere (r=0.5) */
    ASSERT_FLOAT_NEAR(mesh->bbox_min.x, -0.5f, world->voxel_size * 2);
    ASSERT_FLOAT_NEAR(mesh->bbox_max.x,  0.5f, world->voxel_size * 2);
    ASSERT_FLOAT_NEAR(mesh->bbox_min.y, -0.5f, world->voxel_size * 2);
    ASSERT_FLOAT_NEAR(mesh->bbox_max.y,  0.5f, world->voxel_size * 2);

    /* Vertices should be near sphere surface */
    for (uint32_t i = 0; i < mesh->vertex_count; i++) {
        Vec3 v = VEC3(mesh->vx[i], mesh->vy[i], mesh->vz[i]);
        float dist = fabsf(vec3_length(v) - 0.5f);
        ASSERT_LT(dist, world->voxel_size * 2);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Marching Cubes Tests");

    RUN_TEST(extract_sphere_mesh);
    RUN_TEST(mesh_bbox_matches_sdf);
    RUN_TEST(normals_computed);
    RUN_TEST(empty_world_produces_empty_mesh);
    RUN_TEST(iso_value_affects_surface);
    RUN_TEST(roundtrip_preserves_topology);
    RUN_TEST(large_mesh_extraction);
    RUN_TEST(vertex_welding_reduces_count);
    RUN_TEST(vertex_welding_preserves_geometry);

    TEST_SUITE_END();
}
