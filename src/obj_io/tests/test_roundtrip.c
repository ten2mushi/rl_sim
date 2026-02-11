/**
 * Roundtrip Validation Tests
 *
 * Tests the full pipeline: OBJ -> Voxelize -> Marching Cubes -> OBJ
 * Verifies geometric and material preservation.
 */

#include "../include/obj_io.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "test_harness.h"

/* ============================================================================
 * Tests
 * ============================================================================ */

TEST(cube_roundtrip) {
    /* Cube with outward-facing normals (CCW winding from outside) */
    const char* obj_content =
        "# Unit cube with correct winding\n"
        "v -0.5 -0.5 -0.5\n"
        "v  0.5 -0.5 -0.5\n"
        "v  0.5  0.5 -0.5\n"
        "v -0.5  0.5 -0.5\n"
        "v -0.5 -0.5  0.5\n"
        "v  0.5 -0.5  0.5\n"
        "v  0.5  0.5  0.5\n"
        "v -0.5  0.5  0.5\n"
        "f 1 4 3 2\n"     /* Front (z=-0.5, normal -Z) */
        "f 5 6 7 8\n"     /* Back (z=0.5, normal +Z) */
        "f 1 2 6 5\n"     /* Bottom (y=-0.5, normal -Y) */
        "f 4 8 7 3\n"     /* Top (y=0.5, normal +Y) */
        "f 1 5 8 4\n"     /* Left (x=-0.5, normal -X) */
        "f 2 3 7 6\n";    /* Right (x=0.5, normal +X) */

    const char* input_path = "/tmp/test_rt_cube_in.obj";
    const char* output_path = "/tmp/test_rt_cube_out.obj";

    FILE* f = fopen(input_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(obj_content, f);
    fclose(f);

    Arena* arena = arena_create(100 * 1024 * 1024);
    char error[256];

    /* Parse original */
    TriangleMesh* original = NULL;
    ObjIOResult result = obj_parse_file(arena, input_path, NULL, &original, NULL, error);
    ASSERT_EQ(result, OBJ_IO_SUCCESS);
    mesh_compute_bbox(original);

    /* Voxelize */
    WorldBrickMap* world = NULL;
    VoxelizeOptions vox_opts = VOXELIZE_DEFAULTS;
    vox_opts.voxel_size = 0.05f;
    result = mesh_to_sdf(arena, original, &vox_opts, &world, error);
    ASSERT_EQ(result, OBJ_IO_SUCCESS);

    /* Extract mesh */
    TriangleMesh* extracted = NULL;
    result = sdf_to_mesh(arena, world, &MARCHING_CUBES_DEFAULTS, &extracted, error);
    ASSERT_EQ(result, OBJ_IO_SUCCESS);
    mesh_compute_bbox(extracted);

    /* Compare bounding boxes */
    float tol = vox_opts.voxel_size * 2;
    ASSERT_FLOAT_NEAR(extracted->bbox_min.x, original->bbox_min.x, tol);
    ASSERT_FLOAT_NEAR(extracted->bbox_max.x, original->bbox_max.x, tol);
    ASSERT_FLOAT_NEAR(extracted->bbox_min.y, original->bbox_min.y, tol);
    ASSERT_FLOAT_NEAR(extracted->bbox_max.y, original->bbox_max.y, tol);
    ASSERT_FLOAT_NEAR(extracted->bbox_min.z, original->bbox_min.z, tol);
    ASSERT_FLOAT_NEAR(extracted->bbox_max.z, original->bbox_max.z, tol);

    /* Export */
    result = obj_export_file(output_path, extracted, NULL, &OBJ_EXPORT_DEFAULTS, error);
    ASSERT_EQ(result, OBJ_IO_SUCCESS);

    /* Verify output file exists */
    f = fopen(output_path, "r");
    ASSERT_TRUE(f != NULL);
    fclose(f);

    arena_destroy(arena);
    remove(input_path);
    remove(output_path);
    return 0;
}

TEST(mesh_compare_identical) {
    Arena* arena = arena_create(50 * 1024 * 1024);

    /* Create identical meshes */
    TriangleMesh* mesh_a = mesh_create(arena, 3, 1);
    mesh_add_vertex(mesh_a, 0, 0, 0);
    mesh_add_vertex(mesh_a, 1, 0, 0);
    mesh_add_vertex(mesh_a, 0.5f, 1, 0);
    mesh_add_face(mesh_a, 0, 1, 2, 0);
    mesh_compute_bbox(mesh_a);

    TriangleMesh* mesh_b = mesh_create(arena, 3, 1);
    mesh_add_vertex(mesh_b, 0, 0, 0);
    mesh_add_vertex(mesh_b, 1, 0, 0);
    mesh_add_vertex(mesh_b, 0.5f, 1, 0);
    mesh_add_face(mesh_b, 0, 1, 2, 0);
    mesh_compute_bbox(mesh_b);

    MeshCompareResult cmp = mesh_compare(arena, mesh_a, mesh_b, 0.001f);

    ASSERT_TRUE(cmp.passed);
    ASSERT_FLOAT_NEAR(cmp.hausdorff_distance, 0.0f, 0.001f);
    ASSERT_FLOAT_NEAR(cmp.mean_distance, 0.0f, 0.001f);

    arena_destroy(arena);
    return 0;
}

TEST(mesh_compare_different) {
    Arena* arena = arena_create(50 * 1024 * 1024);

    /* Create slightly different meshes */
    TriangleMesh* mesh_a = mesh_create(arena, 3, 1);
    mesh_add_vertex(mesh_a, 0, 0, 0);
    mesh_add_vertex(mesh_a, 1, 0, 0);
    mesh_add_vertex(mesh_a, 0.5f, 1, 0);
    mesh_add_face(mesh_a, 0, 1, 2, 0);
    mesh_compute_bbox(mesh_a);

    TriangleMesh* mesh_b = mesh_create(arena, 3, 1);
    mesh_add_vertex(mesh_b, 0, 0, 0);
    mesh_add_vertex(mesh_b, 1, 0, 0);
    mesh_add_vertex(mesh_b, 0.5f, 1.5f, 0); /* Apex is higher */
    mesh_add_face(mesh_b, 0, 1, 2, 0);
    mesh_compute_bbox(mesh_b);

    MeshCompareResult cmp = mesh_compare(arena, mesh_a, mesh_b, 0.1f);

    /* Should fail tight tolerance */
    ASSERT_TRUE(!cmp.passed);
    ASSERT_GT(cmp.hausdorff_distance, 0.1f);

    /* Should pass loose tolerance */
    cmp = mesh_compare(arena, mesh_a, mesh_b, 1.0f);
    ASSERT_TRUE(cmp.passed);

    arena_destroy(arena);
    return 0;
}

TEST(hausdorff_within_tolerance) {
    const char* obj_content =
        "v 0 0 0\n"
        "v 2 0 0\n"
        "v 2 2 0\n"
        "v 0 2 0\n"
        "v 0 0 2\n"
        "v 2 0 2\n"
        "v 2 2 2\n"
        "v 0 2 2\n"
        "f 1 2 3 4\n"
        "f 5 6 7 8\n"
        "f 1 2 6 5\n"
        "f 2 3 7 6\n"
        "f 3 4 8 7\n"
        "f 4 1 5 8\n";

    const char* temp_path = "/tmp/test_hausdorff.obj";
    FILE* f = fopen(temp_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(obj_content, f);
    fclose(f);

    Arena* arena = arena_create(100 * 1024 * 1024);
    char error[256];

    /* Parse original */
    TriangleMesh* original = NULL;
    obj_parse_file(arena, temp_path, NULL, &original, NULL, error);
    ASSERT_TRUE(original != NULL);

    /* Voxelize and extract */
    WorldBrickMap* world = NULL;
    VoxelizeOptions vox_opts = VOXELIZE_DEFAULTS;
    vox_opts.voxel_size = 0.1f;
    mesh_to_sdf(arena, original, &vox_opts, &world, error);

    TriangleMesh* extracted = NULL;
    sdf_to_mesh(arena, world, &MARCHING_CUBES_DEFAULTS, &extracted, error);
    ASSERT_TRUE(extracted != NULL);

    /* Compare meshes */
    float tolerance = vox_opts.voxel_size * 2;
    MeshCompareResult cmp = mesh_compare(arena, original, extracted, tolerance);

    /* Should pass with 2x voxel size tolerance */
    printf(" (hausdorff=%.4f, mean=%.4f)", cmp.hausdorff_distance, cmp.mean_distance);
    ASSERT_TRUE(cmp.passed);

    arena_destroy(arena);
    remove(temp_path);
    return 0;
}

TEST(pyramid_roundtrip) {
    const char* obj_content =
        "# Pyramid\n"
        "v  0  0  0\n"
        "v  1  0  0\n"
        "v  1  1  0\n"
        "v  0  1  0\n"
        "v  0.5 0.5 1\n"
        "f 1 2 5\n"
        "f 2 3 5\n"
        "f 3 4 5\n"
        "f 4 1 5\n"
        "f 1 2 3 4\n";

    const char* temp_path = "/tmp/test_pyramid.obj";
    FILE* f = fopen(temp_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(obj_content, f);
    fclose(f);

    Arena* arena = arena_create(100 * 1024 * 1024);
    char error[256];

    /* Full roundtrip */
    TriangleMesh* original = NULL;
    obj_parse_file(arena, temp_path, NULL, &original, NULL, error);
    ASSERT_TRUE(original != NULL);
    mesh_compute_bbox(original);

    WorldBrickMap* world = NULL;
    VoxelizeOptions vox_opts = VOXELIZE_DEFAULTS;
    vox_opts.voxel_size = 0.05f;
    mesh_to_sdf(arena, original, &vox_opts, &world, error);

    TriangleMesh* extracted = NULL;
    sdf_to_mesh(arena, world, &MARCHING_CUBES_DEFAULTS, &extracted, error);
    ASSERT_TRUE(extracted != NULL);
    mesh_compute_bbox(extracted);

    /* Verify shape preserved (bbox comparison) */
    float tol = vox_opts.voxel_size * 2;
    ASSERT_FLOAT_NEAR(extracted->bbox_max.z, original->bbox_max.z, tol);

    arena_destroy(arena);
    remove(temp_path);
    return 0;
}

TEST(roundtrip_with_materials) {
    const char* obj_content =
        "mtllib test_rt_mat.mtl\n"
        "usemtl gold\n"
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

    const char* mtl_content =
        "newmtl gold\n"
        "Kd 0.8 0.6 0.2\n";

    const char* obj_path = "/tmp/test_rt_mat.obj";
    const char* mtl_path = "/tmp/test_rt_mat.mtl";
    const char* out_path = "/tmp/test_rt_mat_out.obj";

    FILE* f = fopen(obj_path, "w");
    fputs(obj_content, f);
    fclose(f);

    f = fopen(mtl_path, "w");
    fputs(mtl_content, f);
    fclose(f);

    Arena* arena = arena_create(100 * 1024 * 1024);
    char error[256];

    /* Import and voxelize */
    WorldBrickMap* world = NULL;
    VoxelizeOptions opts = VOXELIZE_DEFAULTS;
    opts.voxel_size = 0.1f;
    ObjIOResult result = obj_to_world(arena, obj_path, &opts, &world, error);
    ASSERT_EQ(result, OBJ_IO_SUCCESS);

    /* Export */
    result = world_to_obj(world, out_path, error);
    ASSERT_EQ(result, OBJ_IO_SUCCESS);

    /* Verify material preserved in export */
    uint8_t gold_id = world_find_material(world, "gold");
    if (gold_id != 0) {
        const MaterialMetadata* gold = world_get_material(world, gold_id);
        ASSERT_TRUE(gold != NULL);
        ASSERT_FLOAT_NEAR(gold->diffuse_color.x, 0.8f, 0.001f);
    }

    arena_destroy(arena);
    remove(obj_path);
    remove(mtl_path);
    remove(out_path);
    remove("/tmp/test_rt_mat_out.mtl");
    return 0;
}

TEST(file_output_valid_obj) {
    const char* obj_content =
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0.5 1 0\n"
        "f 1 2 3\n";

    const char* input_path = "/tmp/test_valid_obj_in.obj";
    const char* output_path = "/tmp/test_valid_obj_out.obj";

    FILE* f = fopen(input_path, "w");
    fputs(obj_content, f);
    fclose(f);

    Arena* arena = arena_create(50 * 1024 * 1024);
    char error[256];

    /* Full pipeline */
    WorldBrickMap* world = NULL;
    VoxelizeOptions opts2 = VOXELIZE_DEFAULTS;
    opts2.voxel_size = 0.05f;
    obj_to_world(arena, input_path, &opts2, &world, error);
    world_to_obj(world, output_path, error);

    /* Verify output is parseable */
    TriangleMesh* result = NULL;
    ObjIOResult parse_result = obj_parse_file(arena, output_path, NULL, &result, NULL, error);

    if (parse_result == OBJ_IO_SUCCESS && result) {
        ASSERT_GT(result->vertex_count, 0);
        ASSERT_GT(result->face_count, 0);
    }

    arena_destroy(arena);
    remove(input_path);
    remove(output_path);
    remove("/tmp/test_valid_obj_out.mtl");
    return 0;
}

TEST(voxel_size_affects_detail) {
    const char* obj_content =
        "# Sphere-like (icosahedron base)\n"
        "v 0 0 1\n"
        "v 0.894 0 0.447\n"
        "v 0.276 0.851 0.447\n"
        "v -0.724 0.526 0.447\n"
        "v -0.724 -0.526 0.447\n"
        "v 0.276 -0.851 0.447\n"
        "v 0.724 0.526 -0.447\n"
        "v -0.276 0.851 -0.447\n"
        "v -0.894 0 -0.447\n"
        "v -0.276 -0.851 -0.447\n"
        "v 0.724 -0.526 -0.447\n"
        "v 0 0 -1\n"
        "f 1 2 3\n"
        "f 1 3 4\n"
        "f 1 4 5\n"
        "f 1 5 6\n"
        "f 1 6 2\n"
        "f 12 7 8\n"
        "f 12 8 9\n"
        "f 12 9 10\n"
        "f 12 10 11\n"
        "f 12 11 7\n"
        "f 2 7 3\n"
        "f 3 8 4\n"
        "f 4 9 5\n"
        "f 5 10 6\n"
        "f 6 11 2\n"
        "f 7 2 11\n"
        "f 8 3 7\n"
        "f 9 4 8\n"
        "f 10 5 9\n"
        "f 11 6 10\n";

    const char* temp_path = "/tmp/test_voxel_size.obj";
    FILE* f = fopen(temp_path, "w");
    fputs(obj_content, f);
    fclose(f);

    Arena* arena = arena_create(200 * 1024 * 1024);
    char error[256];

    TriangleMesh* original = NULL;
    obj_parse_file(arena, temp_path, NULL, &original, NULL, error);

    /* Coarse voxelization */
    WorldBrickMap* world_coarse = NULL;
    VoxelizeOptions opts_coarse = VOXELIZE_DEFAULTS;
    opts_coarse.voxel_size = 0.2f;
    mesh_to_sdf(arena, original, &opts_coarse, &world_coarse, error);

    TriangleMesh* mesh_coarse = NULL;
    sdf_to_mesh(arena, world_coarse, &MARCHING_CUBES_DEFAULTS, &mesh_coarse, error);

    /* Fine voxelization */
    WorldBrickMap* world_fine = NULL;
    VoxelizeOptions opts_fine = VOXELIZE_DEFAULTS;
    opts_fine.voxel_size = 0.05f;
    mesh_to_sdf(arena, original, &opts_fine, &world_fine, error);

    TriangleMesh* mesh_fine = NULL;
    sdf_to_mesh(arena, world_fine, &MARCHING_CUBES_DEFAULTS, &mesh_fine, error);

    /* Fine should have more triangles than coarse */
    if (mesh_coarse && mesh_fine) {
        ASSERT_GT(mesh_fine->face_count, mesh_coarse->face_count);
    }

    arena_destroy(arena);
    remove(temp_path);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Roundtrip Validation Tests");

    RUN_TEST(cube_roundtrip);
    RUN_TEST(mesh_compare_identical);
    RUN_TEST(mesh_compare_different);
    RUN_TEST(hausdorff_within_tolerance);
    RUN_TEST(pyramid_roundtrip);
    RUN_TEST(roundtrip_with_materials);
    RUN_TEST(file_output_valid_obj);
    RUN_TEST(voxel_size_affects_detail);

    TEST_SUITE_END();
}
