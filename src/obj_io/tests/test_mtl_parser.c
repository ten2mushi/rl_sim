/**
 * MTL Parser Unit Tests
 *
 * Tests:
 * - Basic MTL parsing
 * - Diffuse color extraction (Kd)
 * - Multiple materials
 * - Missing MTL file handling
 * - Material registration in WorldBrickMap
 */

#include "../include/obj_io.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "test_harness.h"

/* ============================================================================
 * Tests
 * ============================================================================ */

TEST(parse_single_material) {
    const char* mtl_content =
        "newmtl rock\n"
        "Kd 0.5 0.3 0.2\n"
        "Ks 0.1 0.1 0.1\n"
        "Ns 10.0\n";

    const char* temp_path = "/tmp/test_single.mtl";
    FILE* f = fopen(temp_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(mtl_content, f);
    fclose(f);

    Arena* arena = arena_create(1024 * 1024);
    MtlLibrary* mtl = mtl_parse_file(arena, temp_path);

    ASSERT_TRUE(mtl != NULL);
    ASSERT_EQ(mtl->count, 1);
    ASSERT_STR_EQ(mtl->materials[0].name, "rock");
    ASSERT_TRUE(mtl->materials[0].has_Kd);
    ASSERT_FLOAT_NEAR(mtl->materials[0].Kd.x, 0.5f, 0.001f);
    ASSERT_FLOAT_NEAR(mtl->materials[0].Kd.y, 0.3f, 0.001f);
    ASSERT_FLOAT_NEAR(mtl->materials[0].Kd.z, 0.2f, 0.001f);

    arena_destroy(arena);
    remove(temp_path);
    return 0;
}

TEST(parse_multiple_materials) {
    const char* mtl_content =
        "newmtl grass\n"
        "Kd 0.2 0.6 0.1\n"
        "\n"
        "newmtl dirt\n"
        "Kd 0.4 0.3 0.2\n"
        "\n"
        "newmtl water\n"
        "Kd 0.1 0.3 0.8\n";

    const char* temp_path = "/tmp/test_multi.mtl";
    FILE* f = fopen(temp_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(mtl_content, f);
    fclose(f);

    Arena* arena = arena_create(1024 * 1024);
    MtlLibrary* mtl = mtl_parse_file(arena, temp_path);

    ASSERT_TRUE(mtl != NULL);
    ASSERT_EQ(mtl->count, 3);

    const MtlMaterial* grass = mtl_find_material(mtl, "grass");
    ASSERT_TRUE(grass != NULL);
    ASSERT_FLOAT_NEAR(grass->Kd.y, 0.6f, 0.001f);

    const MtlMaterial* water = mtl_find_material(mtl, "water");
    ASSERT_TRUE(water != NULL);
    ASSERT_FLOAT_NEAR(water->Kd.z, 0.8f, 0.001f);

    arena_destroy(arena);
    remove(temp_path);
    return 0;
}

TEST(parse_material_no_kd) {
    /* Material with texture but no explicit Kd */
    const char* mtl_content =
        "newmtl textured\n"
        "map_Kd texture.png\n"
        "Ks 0.5 0.5 0.5\n";

    const char* temp_path = "/tmp/test_no_kd.mtl";
    FILE* f = fopen(temp_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(mtl_content, f);
    fclose(f);

    Arena* arena = arena_create(1024 * 1024);
    MtlLibrary* mtl = mtl_parse_file(arena, temp_path);

    ASSERT_TRUE(mtl != NULL);
    ASSERT_EQ(mtl->count, 1);
    /* Should default to white */
    ASSERT_TRUE(!mtl->materials[0].has_Kd);
    ASSERT_FLOAT_NEAR(mtl->materials[0].Kd.x, 1.0f, 0.001f);

    arena_destroy(arena);
    remove(temp_path);
    return 0;
}

TEST(parse_missing_file) {
    Arena* arena = arena_create(1024 * 1024);
    MtlLibrary* mtl = mtl_parse_file(arena, "/nonexistent/path.mtl");

    /* Missing MTL should return empty library, not NULL */
    ASSERT_TRUE(mtl != NULL);
    ASSERT_EQ(mtl->count, 0);

    arena_destroy(arena);
    return 0;
}

TEST(find_material) {
    const char* mtl_content =
        "newmtl alpha\n"
        "Kd 1 0 0\n"
        "newmtl beta\n"
        "Kd 0 1 0\n"
        "newmtl gamma\n"
        "Kd 0 0 1\n";

    const char* temp_path = "/tmp/test_find.mtl";
    FILE* f = fopen(temp_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(mtl_content, f);
    fclose(f);

    Arena* arena = arena_create(1024 * 1024);
    MtlLibrary* mtl = mtl_parse_file(arena, temp_path);

    ASSERT_TRUE(mtl != NULL);

    /* Find existing */
    const MtlMaterial* beta = mtl_find_material(mtl, "beta");
    ASSERT_TRUE(beta != NULL);
    ASSERT_FLOAT_NEAR(beta->Kd.y, 1.0f, 0.001f);

    /* Find non-existing */
    const MtlMaterial* notfound = mtl_find_material(mtl, "notfound");
    ASSERT_TRUE(notfound == NULL);

    arena_destroy(arena);
    remove(temp_path);
    return 0;
}

TEST(register_materials) {
    const char* mtl_content =
        "newmtl red_mat\n"
        "Kd 1 0 0\n"
        "newmtl green_mat\n"
        "Kd 0 1 0\n";

    const char* temp_path = "/tmp/test_register.mtl";
    FILE* f = fopen(temp_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(mtl_content, f);
    fclose(f);

    Arena* arena = arena_create(1024 * 1024);
    MtlLibrary* mtl = mtl_parse_file(arena, temp_path);

    /* Create world to register materials */
    WorldBrickMap* world = world_create(arena,
                                         VEC3(0, 0, 0), VEC3(10, 10, 10),
                                         0.1f, 1024, 256);
    ASSERT_TRUE(world != NULL);

    /* Register materials */
    mtl_register_materials(world, mtl);

    /* Find materials in world */
    uint8_t red_id = world_find_material(world, "red_mat");
    uint8_t green_id = world_find_material(world, "green_mat");

    ASSERT_NE(red_id, 0); /* 0 is default material */
    ASSERT_NE(green_id, 0);
    ASSERT_NE(red_id, green_id);

    /* Check colors */
    const MaterialMetadata* red = world_get_material(world, red_id);
    ASSERT_TRUE(red != NULL);
    ASSERT_FLOAT_NEAR(red->diffuse_color.x, 1.0f, 0.001f);

    arena_destroy(arena);
    remove(temp_path);
    return 0;
}

TEST(parse_specular) {
    const char* mtl_content =
        "newmtl shiny\n"
        "Kd 0.5 0.5 0.5\n"
        "Ks 1.0 1.0 1.0\n"
        "Ns 100.0\n";

    const char* temp_path = "/tmp/test_specular.mtl";
    FILE* f = fopen(temp_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(mtl_content, f);
    fclose(f);

    Arena* arena = arena_create(1024 * 1024);
    MtlLibrary* mtl = mtl_parse_file(arena, temp_path);

    ASSERT_TRUE(mtl != NULL);
    ASSERT_EQ(mtl->count, 1);
    ASSERT_FLOAT_NEAR(mtl->materials[0].Ks.x, 1.0f, 0.001f);
    ASSERT_FLOAT_NEAR(mtl->materials[0].Ns, 100.0f, 0.1f);

    arena_destroy(arena);
    remove(temp_path);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("MTL Parser Tests");

    RUN_TEST(parse_single_material);
    RUN_TEST(parse_multiple_materials);
    RUN_TEST(parse_material_no_kd);
    RUN_TEST(parse_missing_file);
    RUN_TEST(find_material);
    RUN_TEST(register_materials);
    RUN_TEST(parse_specular);

    TEST_SUITE_END();
}
