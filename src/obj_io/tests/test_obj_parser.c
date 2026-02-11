/**
 * OBJ Parser Unit Tests
 *
 * Tests:
 * - Basic OBJ parsing
 * - Vertex and face extraction
 * - Material tracking (usemtl)
 * - Bounding box computation
 * - Large file handling
 * - Error handling
 */

#include "../include/obj_io.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "test_harness.h"

/* ============================================================================
 * Tests
 * ============================================================================ */

TEST(parse_simple_cube) {
    /* Create a simple cube OBJ in memory (write to temp file) */
    const char* obj_content =
        "# Simple cube\n"
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

    /* Write to temp file */
    const char* temp_path = "/tmp/test_cube.obj";
    FILE* f = fopen(temp_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(obj_content, f);
    fclose(f);

    /* Parse */
    Arena* arena = arena_create(4 * 1024 * 1024);
    ASSERT_TRUE(arena != NULL);

    TriangleMesh* mesh = NULL;
    char error[256];
    ObjIOResult result = obj_parse_file(arena, temp_path, NULL, &mesh, NULL, error);

    ASSERT_EQ(result, OBJ_IO_SUCCESS);
    ASSERT_TRUE(mesh != NULL);
    ASSERT_EQ(mesh->vertex_count, 8);
    /* 6 quads = 12 triangles */
    ASSERT_EQ(mesh->face_count, 12);

    /* Check bbox */
    ASSERT_FLOAT_NEAR(mesh->bbox_min.x, 0.0f, 0.001f);
    ASSERT_FLOAT_NEAR(mesh->bbox_max.x, 1.0f, 0.001f);

    arena_destroy(arena);
    remove(temp_path);
    return 0;
}

TEST(parse_triangulated) {
    /* OBJ with pre-triangulated faces */
    const char* obj_content =
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0.5 1 0\n"
        "f 1 2 3\n";

    const char* temp_path = "/tmp/test_tri.obj";
    FILE* f = fopen(temp_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(obj_content, f);
    fclose(f);

    Arena* arena = arena_create(4 * 1024 * 1024);
    TriangleMesh* mesh = NULL;
    char error[256];
    ObjIOResult result = obj_parse_file(arena, temp_path, NULL, &mesh, NULL, error);

    ASSERT_EQ(result, OBJ_IO_SUCCESS);
    ASSERT_EQ(mesh->vertex_count, 3);
    ASSERT_EQ(mesh->face_count, 1);

    arena_destroy(arena);
    remove(temp_path);
    return 0;
}

TEST(parse_with_normals_format) {
    /* OBJ with v//vn format (normals without texture coords) */
    const char* obj_content =
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vn 0 0 1\n"
        "f 1//1 2//1 3//1\n";

    const char* temp_path = "/tmp/test_normals.obj";
    FILE* f = fopen(temp_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(obj_content, f);
    fclose(f);

    Arena* arena = arena_create(4 * 1024 * 1024);
    TriangleMesh* mesh = NULL;
    char error[256];
    ObjIOResult result = obj_parse_file(arena, temp_path, NULL, &mesh, NULL, error);

    ASSERT_EQ(result, OBJ_IO_SUCCESS);
    ASSERT_EQ(mesh->vertex_count, 3);
    ASSERT_EQ(mesh->face_count, 1);

    arena_destroy(arena);
    remove(temp_path);
    return 0;
}

TEST(parse_with_texcoords_format) {
    /* OBJ with v/vt/vn format */
    const char* obj_content =
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 0 1\n"
        "vn 0 0 1\n"
        "f 1/1/1 2/2/1 3/3/1\n";

    const char* temp_path = "/tmp/test_texcoords.obj";
    FILE* f = fopen(temp_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(obj_content, f);
    fclose(f);

    Arena* arena = arena_create(4 * 1024 * 1024);
    TriangleMesh* mesh = NULL;
    char error[256];
    ObjIOResult result = obj_parse_file(arena, temp_path, NULL, &mesh, NULL, error);

    ASSERT_EQ(result, OBJ_IO_SUCCESS);
    ASSERT_EQ(mesh->vertex_count, 3);
    ASSERT_EQ(mesh->face_count, 1);

    arena_destroy(arena);
    remove(temp_path);
    return 0;
}

TEST(parse_materials) {
    /* OBJ with usemtl directives */
    const char* obj_content =
        "mtllib test.mtl\n"
        "usemtl red\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f 1 2 3\n"
        "usemtl blue\n"
        "v 0 0 1\n"
        "v 1 0 1\n"
        "v 0 1 1\n"
        "f 4 5 6\n";

    const char* temp_path = "/tmp/test_materials.obj";
    FILE* f = fopen(temp_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(obj_content, f);
    fclose(f);

    Arena* arena = arena_create(4 * 1024 * 1024);
    TriangleMesh* mesh = NULL;
    char error[256];
    ObjIOResult result = obj_parse_file(arena, temp_path, NULL, &mesh, NULL, error);

    ASSERT_EQ(result, OBJ_IO_SUCCESS);
    ASSERT_EQ(mesh->face_count, 2);

    /* First face should have material 0 (red), second material 1 (blue) */
    ASSERT_EQ(mesh->face_mat[0], 0);
    ASSERT_EQ(mesh->face_mat[1], 1);

    arena_destroy(arena);
    remove(temp_path);
    return 0;
}

TEST(parse_negative_indices) {
    /* OBJ with negative (relative) indices */
    const char* obj_content =
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f -3 -2 -1\n";

    const char* temp_path = "/tmp/test_neg.obj";
    FILE* f = fopen(temp_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(obj_content, f);
    fclose(f);

    Arena* arena = arena_create(4 * 1024 * 1024);
    TriangleMesh* mesh = NULL;
    char error[256];
    ObjIOResult result = obj_parse_file(arena, temp_path, NULL, &mesh, NULL, error);

    ASSERT_EQ(result, OBJ_IO_SUCCESS);
    ASSERT_EQ(mesh->face_count, 1);

    /* Check face vertices are correct */
    uint32_t v0 = mesh->face_v[0];
    uint32_t v1 = mesh->face_v[1];
    uint32_t v2 = mesh->face_v[2];
    ASSERT_EQ(v0, 0);
    ASSERT_EQ(v1, 1);
    ASSERT_EQ(v2, 2);

    arena_destroy(arena);
    remove(temp_path);
    return 0;
}

TEST(parse_empty_file) {
    const char* obj_content = "# Empty file\n";

    const char* temp_path = "/tmp/test_empty.obj";
    FILE* f = fopen(temp_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(obj_content, f);
    fclose(f);

    Arena* arena = arena_create(4 * 1024 * 1024);
    TriangleMesh* mesh = NULL;
    char error[256];
    ObjIOResult result = obj_parse_file(arena, temp_path, NULL, &mesh, NULL, error);

    ASSERT_EQ(result, OBJ_IO_ERROR_EMPTY_MESH);

    arena_destroy(arena);
    remove(temp_path);
    return 0;
}

TEST(parse_missing_file) {
    Arena* arena = arena_create(4 * 1024 * 1024);
    TriangleMesh* mesh = NULL;
    char error[256];
    ObjIOResult result = obj_parse_file(arena, "/nonexistent/path.obj", NULL, &mesh, NULL, error);

    ASSERT_EQ(result, OBJ_IO_ERROR_FILE_NOT_FOUND);

    arena_destroy(arena);
    return 0;
}

TEST(computed_normals) {
    /* Simple triangle - normals should be computed */
    const char* obj_content =
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f 1 2 3\n";

    const char* temp_path = "/tmp/test_computed_normals.obj";
    FILE* f = fopen(temp_path, "w");
    ASSERT_TRUE(f != NULL);
    fputs(obj_content, f);
    fclose(f);

    Arena* arena = arena_create(4 * 1024 * 1024);
    TriangleMesh* mesh = NULL;
    char error[256];
    ObjParseOptions opts = OBJ_PARSE_DEFAULTS;
    opts.compute_normals = true;
    ObjIOResult result = obj_parse_file(arena, temp_path, &opts, &mesh, NULL, error);

    ASSERT_EQ(result, OBJ_IO_SUCCESS);
    ASSERT_TRUE(mesh->has_normals);

    /* Normal should point in +Z direction */
    ASSERT_FLOAT_NEAR(mesh->nz[0], 1.0f, 0.01f);

    arena_destroy(arena);
    remove(temp_path);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("OBJ Parser Tests");

    RUN_TEST(parse_simple_cube);
    RUN_TEST(parse_triangulated);
    RUN_TEST(parse_with_normals_format);
    RUN_TEST(parse_with_texcoords_format);
    RUN_TEST(parse_materials);
    RUN_TEST(parse_negative_indices);
    RUN_TEST(parse_empty_file);
    RUN_TEST(parse_missing_file);
    RUN_TEST(computed_normals);

    TEST_SUITE_END();
}
