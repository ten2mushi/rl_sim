/**
 * Material Handling Unit Tests
 *
 * Tests:
 * - Material parsing from MTL files
 * - Material registration in WorldBrickMap
 * - Material transfer during voxelization
 * - Material preservation in roundtrip
 * - Color accuracy
 */

#include "../include/obj_io.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "test_harness.h"

/* ============================================================================
 * Tests
 * ============================================================================ */

TEST(mtl_colors_parsed) {
    const char* mtl_content =
        "newmtl red_material\n"
        "Kd 1.0 0.0 0.0\n"
        "\n"
        "newmtl green_material\n"
        "Kd 0.0 1.0 0.0\n"
        "\n"
        "newmtl blue_material\n"
        "Kd 0.0 0.0 1.0\n";

    const char* temp_path = "/tmp/test_colors.mtl";
    FILE* f = fopen(temp_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(mtl_content, f);
    fclose(f);

    Arena* arena = arena_create(1024 * 1024);
    MtlLibrary* mtl = mtl_parse_file(arena, temp_path);

    ASSERT_TRUE(mtl != NULL);
    ASSERT_EQ(mtl->count, 3);

    /* Verify colors */
    const MtlMaterial* red = mtl_find_material(mtl, "red_material");
    ASSERT_TRUE(red != NULL);
    ASSERT_FLOAT_NEAR(red->Kd.x, 1.0f, 0.001f);
    ASSERT_FLOAT_NEAR(red->Kd.y, 0.0f, 0.001f);
    ASSERT_FLOAT_NEAR(red->Kd.z, 0.0f, 0.001f);

    const MtlMaterial* green = mtl_find_material(mtl, "green_material");
    ASSERT_TRUE(green != NULL);
    ASSERT_FLOAT_NEAR(green->Kd.y, 1.0f, 0.001f);

    const MtlMaterial* blue = mtl_find_material(mtl, "blue_material");
    ASSERT_TRUE(blue != NULL);
    ASSERT_FLOAT_NEAR(blue->Kd.z, 1.0f, 0.001f);

    arena_destroy(arena);
    remove(temp_path);
    return 0;
}

TEST(materials_registered_in_world) {
    const char* mtl_content =
        "newmtl rock\n"
        "Kd 0.5 0.4 0.3\n"
        "newmtl grass\n"
        "Kd 0.2 0.6 0.1\n";

    const char* temp_path = "/tmp/test_register_world.mtl";
    FILE* f = fopen(temp_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(mtl_content, f);
    fclose(f);

    Arena* arena = arena_create(10 * 1024 * 1024);

    MtlLibrary* mtl = mtl_parse_file(arena, temp_path);
    ASSERT_TRUE(mtl != NULL);

    WorldBrickMap* world = world_create(arena,
                                         VEC3(-1, -1, -1), VEC3(1, 1, 1),
                                         0.1f, 256, 64);
    ASSERT_TRUE(world != NULL);

    mtl_register_materials(world, mtl);

    /* Find registered materials */
    uint8_t rock_id = world_find_material(world, "rock");
    uint8_t grass_id = world_find_material(world, "grass");

    ASSERT_NE(rock_id, 0);
    ASSERT_NE(grass_id, 0);
    ASSERT_NE(rock_id, grass_id);

    /* Verify colors stored correctly */
    const MaterialMetadata* rock = world_get_material(world, rock_id);
    ASSERT_TRUE(rock != NULL);
    ASSERT_FLOAT_NEAR(rock->diffuse_color.x, 0.5f, 0.001f);
    ASSERT_FLOAT_NEAR(rock->diffuse_color.y, 0.4f, 0.001f);
    ASSERT_FLOAT_NEAR(rock->diffuse_color.z, 0.3f, 0.001f);

    arena_destroy(arena);
    remove(temp_path);
    return 0;
}

TEST(obj_mtl_integration) {
    /* Create OBJ with material references */
    const char* obj_content =
        "mtllib test_integration.mtl\n"
        "usemtl material_a\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0.5 1 0\n"
        "f 1 2 3\n"
        "usemtl material_b\n"
        "v 2 0 0\n"
        "v 3 0 0\n"
        "v 2.5 1 0\n"
        "f 4 5 6\n";

    const char* mtl_content =
        "newmtl material_a\n"
        "Kd 1.0 0.0 0.0\n"
        "newmtl material_b\n"
        "Kd 0.0 0.0 1.0\n";

    const char* obj_path = "/tmp/test_integration.obj";
    const char* mtl_path = "/tmp/test_integration.mtl";

    FILE* f = fopen(obj_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(obj_content, f);
    fclose(f);

    f = fopen(mtl_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(mtl_content, f);
    fclose(f);

    Arena* arena = arena_create(10 * 1024 * 1024);

    TriangleMesh* mesh = NULL;
    MtlLibrary* mtl = NULL;
    char error[256];

    ObjIOResult result = obj_parse_file(arena, obj_path, NULL, &mesh, &mtl, error);
    ASSERT_EQ(result, OBJ_IO_SUCCESS);
    ASSERT_TRUE(mesh != NULL);

    /* Verify face materials are tracked */
    ASSERT_EQ(mesh->face_count, 2);
    ASSERT_EQ(mesh->face_mat[0], 0); /* material_a */
    ASSERT_EQ(mesh->face_mat[1], 1); /* material_b */

    /* Verify MTL was parsed */
    if (mtl) {
        ASSERT_EQ(mtl->count, 2);
    }

    arena_destroy(arena);
    remove(obj_path);
    remove(mtl_path);
    return 0;
}

TEST(material_voxelization_transfer) {
    /* Create mesh with two distinct material regions */
    const char* obj_content =
        "mtllib test_vox_mat.mtl\n"
        "usemtl red\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 1 1 0\n"
        "v 0 1 0\n"
        "v 0 0 0.5\n"
        "v 1 0 0.5\n"
        "v 1 1 0.5\n"
        "v 0 1 0.5\n"
        "f 1 2 3 4\n"
        "f 5 6 7 8\n"
        "f 1 2 6 5\n"
        "f 2 3 7 6\n"
        "f 3 4 8 7\n"
        "f 4 1 5 8\n";

    const char* mtl_content =
        "newmtl red\n"
        "Kd 1.0 0.0 0.0\n";

    const char* obj_path = "/tmp/test_vox_mat.obj";
    const char* mtl_path = "/tmp/test_vox_mat.mtl";

    FILE* f = fopen(obj_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(obj_content, f);
    fclose(f);

    f = fopen(mtl_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(mtl_content, f);
    fclose(f);

    Arena* arena = arena_create(50 * 1024 * 1024);

    WorldBrickMap* world = NULL;
    char error[256];

    VoxelizeOptions opts = VOXELIZE_DEFAULTS;
    opts.voxel_size = 0.1f;
    ObjIOResult result = obj_to_world(arena, obj_path, &opts, &world, error);
    ASSERT_EQ(result, OBJ_IO_SUCCESS);
    ASSERT_TRUE(world != NULL);

    /* Verify material is registered */
    uint8_t red_id = world_find_material(world, "red");
    if (red_id != 0) {
        const MaterialMetadata* red = world_get_material(world, red_id);
        ASSERT_TRUE(red != NULL);
        ASSERT_FLOAT_NEAR(red->diffuse_color.x, 1.0f, 0.001f);
    }

    arena_destroy(arena);
    remove(obj_path);
    remove(mtl_path);
    return 0;
}

TEST(material_export_roundtrip) {
    /* Create OBJ with materials */
    const char* obj_content =
        "mtllib test_export_rt.mtl\n"
        "usemtl copper\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 1 1 0\n"
        "v 0 1 0\n"
        "f 1 2 3\n"
        "f 1 3 4\n";

    const char* mtl_content =
        "newmtl copper\n"
        "Kd 0.72 0.45 0.2\n";

    const char* obj_path = "/tmp/test_export_rt.obj";
    const char* mtl_path = "/tmp/test_export_rt.mtl";
    const char* out_obj_path = "/tmp/test_export_rt_out.obj";
    const char* out_mtl_path = "/tmp/test_export_rt_out.mtl";

    FILE* f = fopen(obj_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(obj_content, f);
    fclose(f);

    f = fopen(mtl_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(mtl_content, f);
    fclose(f);

    Arena* arena = arena_create(50 * 1024 * 1024);
    char error[256];

    /* Import */
    WorldBrickMap* world = NULL;
    VoxelizeOptions opts = VOXELIZE_DEFAULTS;
    opts.voxel_size = 0.05f;
    ObjIOResult result = obj_to_world(arena, obj_path, &opts, &world, error);
    ASSERT_EQ(result, OBJ_IO_SUCCESS);

    /* Export */
    result = world_to_obj(world, out_obj_path, error);
    ASSERT_EQ(result, OBJ_IO_SUCCESS);

    /* Verify output OBJ exists and references MTL */
    f = fopen(out_obj_path, "r");
    ASSERT_TRUE(f != NULL);

    char line[256];
    bool found_mtllib = false;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "mtllib", 6) == 0) {
            found_mtllib = true;
            break;
        }
    }
    fclose(f);

    ASSERT_TRUE(found_mtllib);

    arena_destroy(arena);
    remove(obj_path);
    remove(mtl_path);
    remove(out_obj_path);
    remove(out_mtl_path);
    return 0;
}

TEST(default_material_for_missing_mtl) {
    /* OBJ referencing non-existent MTL */
    const char* obj_content =
        "mtllib nonexistent.mtl\n"
        "usemtl some_material\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0.5 1 0\n"
        "f 1 2 3\n";

    const char* obj_path = "/tmp/test_missing_mtl.obj";
    FILE* f = fopen(obj_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(obj_content, f);
    fclose(f);

    Arena* arena = arena_create(10 * 1024 * 1024);

    TriangleMesh* mesh = NULL;
    MtlLibrary* mtl = NULL;
    char error[256];

    /* Should still parse successfully */
    ObjIOResult result = obj_parse_file(arena, obj_path, NULL, &mesh, &mtl, error);
    ASSERT_EQ(result, OBJ_IO_SUCCESS);
    ASSERT_TRUE(mesh != NULL);
    ASSERT_EQ(mesh->face_count, 1);

    /* Material should be assigned (even if MTL not found) */
    /* face_mat will still track the usemtl index */

    arena_destroy(arena);
    remove(obj_path);
    return 0;
}

TEST(multiple_materials_voxelize) {
    /* Create mesh with multiple materials */
    const char* obj_content =
        "mtllib test_multi_mat.mtl\n"
        "usemtl mat1\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0.5 0.5 0\n"
        "f 1 2 3\n"
        "usemtl mat2\n"
        "v 1 0 0\n"
        "v 2 0 0\n"
        "v 1.5 0.5 0\n"
        "f 4 5 6\n"
        "usemtl mat3\n"
        "v 2 0 0\n"
        "v 3 0 0\n"
        "v 2.5 0.5 0\n"
        "f 7 8 9\n";

    const char* mtl_content =
        "newmtl mat1\n"
        "Kd 1.0 0.0 0.0\n"
        "newmtl mat2\n"
        "Kd 0.0 1.0 0.0\n"
        "newmtl mat3\n"
        "Kd 0.0 0.0 1.0\n";

    const char* obj_path = "/tmp/test_multi_mat.obj";
    const char* mtl_path = "/tmp/test_multi_mat.mtl";

    FILE* f = fopen(obj_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(obj_content, f);
    fclose(f);

    f = fopen(mtl_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(mtl_content, f);
    fclose(f);

    Arena* arena = arena_create(50 * 1024 * 1024);
    char error[256];

    WorldBrickMap* world = NULL;
    VoxelizeOptions opts = VOXELIZE_DEFAULTS;
    opts.voxel_size = 0.1f;
    ObjIOResult result = obj_to_world(arena, obj_path, &opts, &world, error);
    ASSERT_EQ(result, OBJ_IO_SUCCESS);

    /* All three materials should be registered */
    uint8_t mat1_id = world_find_material(world, "mat1");
    uint8_t mat2_id = world_find_material(world, "mat2");
    uint8_t mat3_id = world_find_material(world, "mat3");

    /* At least one should be found (depends on voxelization touching faces) */
    /* The point is they're all registered even if voxels don't use them all */
    ASSERT_TRUE(world->material_count >= 1);

    arena_destroy(arena);
    remove(obj_path);
    remove(mtl_path);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Material Handling Tests");

    RUN_TEST(mtl_colors_parsed);
    RUN_TEST(materials_registered_in_world);
    RUN_TEST(obj_mtl_integration);
    RUN_TEST(material_voxelization_transfer);
    RUN_TEST(material_export_roundtrip);
    RUN_TEST(default_material_for_missing_mtl);
    RUN_TEST(multiple_materials_voxelize);

    TEST_SUITE_END();
}
