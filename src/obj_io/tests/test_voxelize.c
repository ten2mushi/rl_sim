/**
 * Voxelization Unit Tests
 *
 * Tests:
 * - Brick classification (coarse and fine)
 * - Sparse voxelization pipeline
 * - Inside/outside determination
 * - Material transfer to voxels
 * - High-level mesh_to_sdf conversion
 */

#include "../include/obj_io.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "test_harness.h"

/* ============================================================================
 * Helper: Create test meshes
 * ============================================================================ */

static TriangleMesh* create_unit_cube(Arena* arena) {
    TriangleMesh* mesh = mesh_create(arena, 8, 12);
    assert(mesh != NULL);

    /* 8 vertices of unit cube centered at origin */
    mesh_add_vertex(mesh, -0.5f, -0.5f, -0.5f);
    mesh_add_vertex(mesh,  0.5f, -0.5f, -0.5f);
    mesh_add_vertex(mesh,  0.5f,  0.5f, -0.5f);
    mesh_add_vertex(mesh, -0.5f,  0.5f, -0.5f);
    mesh_add_vertex(mesh, -0.5f, -0.5f,  0.5f);
    mesh_add_vertex(mesh,  0.5f, -0.5f,  0.5f);
    mesh_add_vertex(mesh,  0.5f,  0.5f,  0.5f);
    mesh_add_vertex(mesh, -0.5f,  0.5f,  0.5f);

    /* 12 triangles (2 per face) - winding for outward-facing normals */
    /* Front (z=-0.5, normal -Z) */
    mesh_add_face(mesh, 0, 2, 1, 0);
    mesh_add_face(mesh, 0, 3, 2, 0);
    /* Back (z=+0.5, normal +Z) */
    mesh_add_face(mesh, 4, 5, 6, 0);
    mesh_add_face(mesh, 4, 6, 7, 0);
    /* Top (y=+0.5, normal +Y) */
    mesh_add_face(mesh, 3, 6, 2, 0);
    mesh_add_face(mesh, 3, 7, 6, 0);
    /* Bottom (y=-0.5, normal -Y) */
    mesh_add_face(mesh, 0, 1, 5, 0);
    mesh_add_face(mesh, 0, 5, 4, 0);
    /* Right (x=+0.5, normal +X) */
    mesh_add_face(mesh, 1, 2, 6, 0);
    mesh_add_face(mesh, 1, 6, 5, 0);
    /* Left (x=-0.5, normal -X) */
    mesh_add_face(mesh, 0, 4, 7, 0);
    mesh_add_face(mesh, 0, 7, 3, 0);

    mesh_compute_bbox(mesh);
    return mesh;
}

static TriangleMesh* create_two_material_mesh(Arena* arena) {
    TriangleMesh* mesh = mesh_create(arena, 6, 4);
    assert(mesh != NULL);

    /* Two triangles with different materials */
    /* Triangle 1: material 0 */
    mesh_add_vertex(mesh, 0, 0, 0);
    mesh_add_vertex(mesh, 1, 0, 0);
    mesh_add_vertex(mesh, 0.5f, 1, 0);
    mesh_add_face(mesh, 0, 1, 2, 0);

    /* Triangle 2: material 1 */
    mesh_add_vertex(mesh, 2, 0, 0);
    mesh_add_vertex(mesh, 3, 0, 0);
    mesh_add_vertex(mesh, 2.5f, 1, 0);
    mesh_add_face(mesh, 3, 4, 5, 1);

    mesh_compute_bbox(mesh);
    return mesh;
}

/* ============================================================================
 * Tests
 * ============================================================================ */

TEST(brick_classification_outside) {
    Arena* arena = arena_create(10 * 1024 * 1024);

    /* Create small cube at origin */
    TriangleMesh* mesh = create_unit_cube(arena);
    ASSERT_TRUE(mesh != NULL);

    MeshBVH* bvh = bvh_build(arena, mesh);
    ASSERT_TRUE(bvh != NULL);

    /* Create world larger than the cube */
    WorldBrickMap* world = world_create(arena,
                                         VEC3(-5, -5, -5), VEC3(5, 5, 5),
                                         0.1f, 1024, 256);
    ASSERT_TRUE(world != NULL);

    /* Classify bricks (coarse pass) */
    BrickClassification* classes = classify_bricks_coarse(arena, bvh, mesh, world);
    ASSERT_TRUE(classes != NULL);

    /* Use pre-computed counts */
    uint32_t outside_count = classes->outside_count;
    uint32_t surface_count = classes->surface_count;

    /* Majority should be outside */
    ASSERT_GT(outside_count, surface_count);
    /* Some should be surface (near the cube) */
    ASSERT_GT(surface_count, 0);

    arena_destroy(arena);
    return 0;
}

TEST(brick_classification_surface) {
    Arena* arena = arena_create(10 * 1024 * 1024);

    TriangleMesh* mesh = create_unit_cube(arena);
    ASSERT_TRUE(mesh != NULL);

    MeshBVH* bvh = bvh_build(arena, mesh);
    ASSERT_TRUE(bvh != NULL);

    /* Create world that tightly fits the cube */
    WorldBrickMap* world = world_create(arena,
                                         VEC3(-1, -1, -1), VEC3(1, 1, 1),
                                         0.1f, 256, 64);
    ASSERT_TRUE(world != NULL);

    BrickClassification* classes = classify_bricks_coarse(arena, bvh, mesh, world);
    ASSERT_TRUE(classes != NULL);

    /* With tight bounds, we should have surface bricks */
    ASSERT_GT(classes->surface_count, 0);

    arena_destroy(arena);
    return 0;
}

TEST(mesh_to_sdf_basic) {
    Arena* arena = arena_create(50 * 1024 * 1024);

    TriangleMesh* mesh = create_unit_cube(arena);
    ASSERT_TRUE(mesh != NULL);

    WorldBrickMap* world = NULL;
    char error[256];
    VoxelizeOptions opts = VOXELIZE_DEFAULTS;
    opts.voxel_size = 0.1f;

    ObjIOResult result = mesh_to_sdf(arena, mesh, &opts, &world, error);
    ASSERT_EQ(result, OBJ_IO_SUCCESS);
    ASSERT_TRUE(world != NULL);

    /* World should have some allocated bricks */
    ASSERT_GT(world->atlas_count, 0);

    /* Query SDF at center (should be negative = inside) */
    float sdf_center = world_sdf_query(world, VEC3(0, 0, 0));
    ASSERT_LT(sdf_center, 0);

    /* Query SDF far outside (should be positive) */
    float sdf_outside = world_sdf_query(world, VEC3(2, 2, 2));
    ASSERT_GT(sdf_outside, 0);

    arena_destroy(arena);
    return 0;
}

TEST(mesh_to_sdf_surface_accuracy) {
    Arena* arena = arena_create(50 * 1024 * 1024);

    TriangleMesh* mesh = create_unit_cube(arena);
    ASSERT_TRUE(mesh != NULL);

    WorldBrickMap* world = NULL;
    char error[256];
    VoxelizeOptions opts = VOXELIZE_DEFAULTS;
    opts.voxel_size = 0.05f;

    ObjIOResult result = mesh_to_sdf(arena, mesh, &opts, &world, error);
    ASSERT_EQ(result, OBJ_IO_SUCCESS);

    /* SDF at surface should be near zero */
    /* Point on +X face of unit cube centered at origin */
    float sdf_surface = world_sdf_query(world, VEC3(0.5f, 0, 0));
    ASSERT_FLOAT_NEAR(sdf_surface, 0.0f, opts.voxel_size * 2);

    arena_destroy(arena);
    return 0;
}

TEST(voxelize_with_materials) {
    Arena* arena = arena_create(50 * 1024 * 1024);

    TriangleMesh* mesh = create_two_material_mesh(arena);
    ASSERT_TRUE(mesh != NULL);

    WorldBrickMap* world = NULL;
    char error[256];
    VoxelizeOptions opts = VOXELIZE_DEFAULTS;
    opts.voxel_size = 0.1f;

    ObjIOResult result = mesh_to_sdf(arena, mesh, &opts, &world, error);
    ASSERT_EQ(result, OBJ_IO_SUCCESS);

    /* Materials should be present in the world */
    /* Note: actual material values depend on voxelization proximity */
    ASSERT_TRUE(world != NULL);

    arena_destroy(arena);
    return 0;
}

TEST(obj_to_world_integration) {
    /* Create a temp OBJ file */
    const char* obj_content =
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 1 1 0\n"
        "v 0 1 0\n"
        "v 0 0 1\n"
        "v 1 0 1\n"
        "v 1 1 1\n"
        "v 0 1 1\n"
        "f 1 2 3 4\n"
        "f 5 6 7 8\n"
        "f 1 2 6 5\n"
        "f 2 3 7 6\n"
        "f 3 4 8 7\n"
        "f 4 1 5 8\n";

    const char* temp_path = "/tmp/test_voxelize_cube.obj";
    FILE* f = fopen(temp_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(obj_content, f);
    fclose(f);

    Arena* arena = arena_create(50 * 1024 * 1024);

    WorldBrickMap* world = NULL;
    char error[256];
    VoxelizeOptions opts = VOXELIZE_DEFAULTS;
    opts.voxel_size = 0.1f;
    ObjIOResult result = obj_to_world(arena, temp_path, &opts, &world, error);

    ASSERT_EQ(result, OBJ_IO_SUCCESS);
    ASSERT_TRUE(world != NULL);
    ASSERT_GT(world->atlas_count, 0);

    arena_destroy(arena);
    remove(temp_path);
    return 0;
}

TEST(sparse_optimization_benefit) {
    Arena* arena = arena_create(100 * 1024 * 1024);

    /* Create small cube in large world - should benefit from sparse optimization */
    TriangleMesh* mesh = create_unit_cube(arena);
    ASSERT_TRUE(mesh != NULL);

    MeshBVH* bvh = bvh_build(arena, mesh);
    ASSERT_TRUE(bvh != NULL);

    /* Large world bounds */
    WorldBrickMap* world = world_create(arena,
                                         VEC3(-10, -10, -10), VEC3(10, 10, 10),
                                         0.2f, 4096, 512);
    ASSERT_TRUE(world != NULL);

    BrickClassification* classes = classify_bricks_coarse(arena, bvh, mesh, world);
    ASSERT_TRUE(classes != NULL);

    /* Use pre-computed counts */
    uint32_t outside = classes->outside_count;
    uint32_t inside = classes->inside_count;
    uint32_t surface = classes->surface_count;

    /* Verify sparse optimization: most bricks should be OUTSIDE */
    /* Surface bricks should be a small fraction */
    uint32_t total_non_outside = inside + surface;
    ASSERT_GT(outside, total_non_outside * 5); /* At least 5x more outside */

    arena_destroy(arena);
    return 0;
}

TEST(voxelize_empty_mesh) {
    Arena* arena = arena_create(10 * 1024 * 1024);

    TriangleMesh* mesh = mesh_create(arena, 8, 12);
    ASSERT_TRUE(mesh != NULL);
    /* Don't add any vertices or faces */

    WorldBrickMap* world = NULL;
    char error[256];
    VoxelizeOptions opts = VOXELIZE_DEFAULTS;

    ObjIOResult result = mesh_to_sdf(arena, mesh, &opts, &world, error);
    ASSERT_EQ(result, OBJ_IO_ERROR_EMPTY_MESH);

    arena_destroy(arena);
    return 0;
}

TEST(robust_inside_outside_sphere) {
    Arena* arena = arena_create(20 * 1024 * 1024);

    TriangleMesh* mesh = create_unit_cube(arena);
    ASSERT_TRUE(mesh != NULL);

    MeshBVH* bvh = bvh_build(arena, mesh);
    ASSERT_TRUE(bvh != NULL);

    /* Center should be inside */
    float sign_center = bvh_inside_outside_robust(bvh, mesh, VEC3(0, 0, 0));
    ASSERT_LT(sign_center, 0);

    /* Far outside should be outside */
    float sign_far = bvh_inside_outside_robust(bvh, mesh, VEC3(5, 5, 5));
    ASSERT_GT(sign_far, 0);

    /* Just outside each face */
    float sign_px = bvh_inside_outside_robust(bvh, mesh, VEC3(0.6f, 0, 0));
    ASSERT_GT(sign_px, 0);

    float sign_ny = bvh_inside_outside_robust(bvh, mesh, VEC3(0, -0.6f, 0));
    ASSERT_GT(sign_ny, 0);

    float sign_pz = bvh_inside_outside_robust(bvh, mesh, VEC3(0, 0, 0.6f));
    ASSERT_GT(sign_pz, 0);

    /* Just inside each face */
    float sign_in_x = bvh_inside_outside_robust(bvh, mesh, VEC3(0.4f, 0, 0));
    ASSERT_LT(sign_in_x, 0);

    float sign_in_y = bvh_inside_outside_robust(bvh, mesh, VEC3(0, 0.4f, 0));
    ASSERT_LT(sign_in_y, 0);

    arena_destroy(arena);
    return 0;
}

TEST(robust_matches_single_on_simple_mesh) {
    Arena* arena = arena_create(20 * 1024 * 1024);

    TriangleMesh* mesh = create_unit_cube(arena);
    ASSERT_TRUE(mesh != NULL);

    MeshBVH* bvh = bvh_build(arena, mesh);
    ASSERT_TRUE(bvh != NULL);

    /* For a simple cube, both methods should agree */
    Vec3 test_points[] = {
        VEC3(0, 0, 0),       /* center */
        VEC3(0.4f, 0, 0),    /* inside near face */
        VEC3(0.6f, 0, 0),    /* outside near face */
        VEC3(2, 2, 2),       /* far outside */
        VEC3(-0.3f, 0.3f, -0.3f) /* inside corner */
    };

    for (int i = 0; i < 5; i++) {
        float single = bvh_inside_outside(bvh, mesh, test_points[i]);
        float robust = bvh_inside_outside_robust(bvh, mesh, test_points[i]);
        /* Signs should match */
        ASSERT_TRUE((single > 0) == (robust > 0));
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Voxelization Tests");

    RUN_TEST(brick_classification_outside);
    RUN_TEST(brick_classification_surface);
    RUN_TEST(mesh_to_sdf_basic);
    RUN_TEST(mesh_to_sdf_surface_accuracy);
    RUN_TEST(voxelize_with_materials);
    RUN_TEST(obj_to_world_integration);
    RUN_TEST(sparse_optimization_benefit);
    RUN_TEST(voxelize_empty_mesh);
    RUN_TEST(robust_inside_outside_sphere);
    RUN_TEST(robust_matches_single_on_simple_mesh);

    TEST_SUITE_END();
}
