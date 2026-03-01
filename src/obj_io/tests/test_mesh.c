/**
 * Mesh Operations Unit Tests
 *
 * Tests:
 * - Mesh creation and vertex/face addition
 * - Bounding box computation
 * - Normal computation
 * - Triangle queries
 * - BVH construction and queries
 */

#include "../include/obj_io.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "test_harness.h"

/* ============================================================================
 * Helper: Create a simple cube mesh
 * ============================================================================ */

static TriangleMesh* create_cube_mesh(Arena* arena) {
    TriangleMesh* mesh = mesh_create(arena, 8, 12);
    assert(mesh != NULL);

    /* 8 vertices of unit cube */
    mesh_add_vertex(mesh, 0, 0, 0);
    mesh_add_vertex(mesh, 1, 0, 0);
    mesh_add_vertex(mesh, 1, 1, 0);
    mesh_add_vertex(mesh, 0, 1, 0);
    mesh_add_vertex(mesh, 0, 0, 1);
    mesh_add_vertex(mesh, 1, 0, 1);
    mesh_add_vertex(mesh, 1, 1, 1);
    mesh_add_vertex(mesh, 0, 1, 1);

    /* 12 triangles (2 per face) - winding for outward-facing normals */
    /* Front (z=0, normal -Z) */
    mesh_add_face(mesh, 0, 2, 1, 0);
    mesh_add_face(mesh, 0, 3, 2, 0);
    /* Back (z=1, normal +Z) */
    mesh_add_face(mesh, 4, 5, 6, 0);
    mesh_add_face(mesh, 4, 6, 7, 0);
    /* Top (y=1, normal +Y) */
    mesh_add_face(mesh, 3, 6, 2, 0);
    mesh_add_face(mesh, 3, 7, 6, 0);
    /* Bottom (y=0, normal -Y) */
    mesh_add_face(mesh, 0, 1, 5, 0);
    mesh_add_face(mesh, 0, 5, 4, 0);
    /* Right (x=1, normal +X) */
    mesh_add_face(mesh, 1, 2, 6, 0);
    mesh_add_face(mesh, 1, 6, 5, 0);
    /* Left (x=0, normal -X) */
    mesh_add_face(mesh, 0, 4, 7, 0);
    mesh_add_face(mesh, 0, 7, 3, 0);

    mesh_compute_bbox(mesh);
    return mesh;
}

/* ============================================================================
 * Tests
 * ============================================================================ */

TEST(mesh_creation) {
    Arena* arena = arena_create(1024 * 1024);
    TriangleMesh* mesh = mesh_create(arena, 100, 200);

    ASSERT_TRUE(mesh != NULL);
    ASSERT_TRUE(mesh->vx != NULL);
    ASSERT_TRUE(mesh->vy != NULL);
    ASSERT_TRUE(mesh->vz != NULL);
    ASSERT_TRUE(mesh->face_v != NULL);
    ASSERT_TRUE(mesh->face_mat != NULL);
    ASSERT_EQ(mesh->vertex_count, 0);
    ASSERT_EQ(mesh->face_count, 0);
    ASSERT_EQ(mesh->vertex_capacity, 100);
    ASSERT_EQ(mesh->face_capacity, 200);

    arena_destroy(arena);
    return 0;
}

TEST(add_vertices) {
    Arena* arena = arena_create(1024 * 1024);
    TriangleMesh* mesh = mesh_create(arena, 8, 12);

    uint32_t v0 = mesh_add_vertex(mesh, 1.0f, 2.0f, 3.0f);
    uint32_t v1 = mesh_add_vertex(mesh, 4.0f, 5.0f, 6.0f);

    ASSERT_EQ(v0, 0);
    ASSERT_EQ(v1, 1);
    ASSERT_EQ(mesh->vertex_count, 2);

    ASSERT_FLOAT_NEAR(mesh->vx[0], 1.0f, 0.001f);
    ASSERT_FLOAT_NEAR(mesh->vy[0], 2.0f, 0.001f);
    ASSERT_FLOAT_NEAR(mesh->vz[0], 3.0f, 0.001f);

    arena_destroy(arena);
    return 0;
}

TEST(add_faces) {
    Arena* arena = arena_create(1024 * 1024);
    TriangleMesh* mesh = mesh_create(arena, 8, 12);

    mesh_add_vertex(mesh, 0, 0, 0);
    mesh_add_vertex(mesh, 1, 0, 0);
    mesh_add_vertex(mesh, 0, 1, 0);

    uint32_t f0 = mesh_add_face(mesh, 0, 1, 2, 5);

    ASSERT_EQ(f0, 0);
    ASSERT_EQ(mesh->face_count, 1);
    ASSERT_EQ(mesh->face_v[0], 0);
    ASSERT_EQ(mesh->face_v[1], 1);
    ASSERT_EQ(mesh->face_v[2], 2);
    ASSERT_EQ(mesh->face_mat[0], 5);

    arena_destroy(arena);
    return 0;
}

TEST(bbox_computation) {
    Arena* arena = arena_create(1024 * 1024);
    TriangleMesh* mesh = create_cube_mesh(arena);

    ASSERT_FLOAT_NEAR(mesh->bbox_min.x, 0.0f, 0.001f);
    ASSERT_FLOAT_NEAR(mesh->bbox_min.y, 0.0f, 0.001f);
    ASSERT_FLOAT_NEAR(mesh->bbox_min.z, 0.0f, 0.001f);
    ASSERT_FLOAT_NEAR(mesh->bbox_max.x, 1.0f, 0.001f);
    ASSERT_FLOAT_NEAR(mesh->bbox_max.y, 1.0f, 0.001f);
    ASSERT_FLOAT_NEAR(mesh->bbox_max.z, 1.0f, 0.001f);

    arena_destroy(arena);
    return 0;
}

TEST(normal_computation) {
    Arena* arena = arena_create(1024 * 1024);
    TriangleMesh* mesh = mesh_create(arena, 3, 1);

    /* Triangle in XY plane */
    mesh_add_vertex(mesh, 0, 0, 0);
    mesh_add_vertex(mesh, 1, 0, 0);
    mesh_add_vertex(mesh, 0, 1, 0);
    mesh_add_face(mesh, 0, 1, 2, 0);

    mesh_compute_normals(mesh);

    ASSERT_TRUE(mesh->has_normals);
    /* All normals should point in +Z */
    ASSERT_FLOAT_NEAR(mesh->nz[0], 1.0f, 0.01f);
    ASSERT_FLOAT_NEAR(mesh->nz[1], 1.0f, 0.01f);
    ASSERT_FLOAT_NEAR(mesh->nz[2], 1.0f, 0.01f);

    arena_destroy(arena);
    return 0;
}

TEST(triangle_query) {
    Arena* arena = arena_create(1024 * 1024);
    TriangleMesh* mesh = mesh_create(arena, 3, 1);

    mesh_add_vertex(mesh, 0, 0, 0);
    mesh_add_vertex(mesh, 1, 0, 0);
    mesh_add_vertex(mesh, 0, 1, 0);
    mesh_add_face(mesh, 0, 1, 2, 0);

    Vec3 v0, v1, v2;
    mesh_get_triangle(mesh, 0, &v0, &v1, &v2);

    ASSERT_FLOAT_NEAR(v0.x, 0.0f, 0.001f);
    ASSERT_FLOAT_NEAR(v1.x, 1.0f, 0.001f);
    ASSERT_FLOAT_NEAR(v2.y, 1.0f, 0.001f);

    arena_destroy(arena);
    return 0;
}

TEST(face_area) {
    Arena* arena = arena_create(1024 * 1024);
    TriangleMesh* mesh = mesh_create(arena, 3, 1);

    /* Right triangle with legs 1 and 1 -> area = 0.5 */
    mesh_add_vertex(mesh, 0, 0, 0);
    mesh_add_vertex(mesh, 1, 0, 0);
    mesh_add_vertex(mesh, 0, 1, 0);
    mesh_add_face(mesh, 0, 1, 2, 0);

    float area = mesh_face_area(mesh, 0);
    ASSERT_FLOAT_NEAR(area, 0.5f, 0.001f);

    arena_destroy(arena);
    return 0;
}

TEST(bvh_construction) {
    Arena* arena = arena_create(1024 * 1024);
    TriangleMesh* mesh = create_cube_mesh(arena);

    MeshBVH* bvh = bvh_build(arena, mesh);

    ASSERT_TRUE(bvh != NULL);
    ASSERT_TRUE(bvh->nodes != NULL);
    ASSERT_TRUE(bvh->node_count > 0);
    ASSERT_TRUE(bvh->face_indices != NULL);

    arena_destroy(arena);
    return 0;
}

TEST(bvh_ray_intersect) {
    Arena* arena = arena_create(1024 * 1024);
    TriangleMesh* mesh = create_cube_mesh(arena);
    MeshBVH* bvh = bvh_build(arena, mesh);

    /* Ray from outside hitting cube */
    Vec3 origin = VEC3(-1.0f, 0.5f, 0.5f);
    Vec3 dir = VEC3(1.0f, 0.0f, 0.0f);

    float hit_t;
    uint32_t hit_face;
    bool hit = bvh_ray_intersect(bvh, mesh, origin, dir, 100.0f, &hit_t, &hit_face);

    ASSERT_TRUE(hit);
    ASSERT_FLOAT_NEAR(hit_t, 1.0f, 0.01f); /* Should hit at x=0 */

    /* Ray missing cube */
    origin = VEC3(-1.0f, 2.0f, 2.0f);
    hit = bvh_ray_intersect(bvh, mesh, origin, dir, 100.0f, &hit_t, &hit_face);
    ASSERT_TRUE(!hit);

    arena_destroy(arena);
    return 0;
}

TEST(bvh_closest_point) {
    Arena* arena = arena_create(1024 * 1024);
    TriangleMesh* mesh = create_cube_mesh(arena);
    MeshBVH* bvh = bvh_build(arena, mesh);

    /* Point outside cube */
    Vec3 point = VEC3(0.5f, 0.5f, 2.0f);
    Vec3 closest;
    uint32_t face_idx;

    float dist = bvh_closest_point(bvh, mesh, point, &closest, &face_idx);

    ASSERT_FLOAT_NEAR(dist, 1.0f, 0.01f); /* 1 unit from z=1 face */
    ASSERT_FLOAT_NEAR(closest.z, 1.0f, 0.01f);

    /* Point inside cube */
    point = VEC3(0.5f, 0.5f, 0.5f);
    dist = bvh_closest_point(bvh, mesh, point, &closest, &face_idx);
    ASSERT_FLOAT_NEAR(dist, 0.5f, 0.01f); /* 0.5 from any face */

    arena_destroy(arena);
    return 0;
}

TEST(bvh_aabb_intersect) {
    Arena* arena = arena_create(1024 * 1024);
    TriangleMesh* mesh = create_cube_mesh(arena);
    MeshBVH* bvh = bvh_build(arena, mesh);

    /* Overlapping AABB */
    bool intersects = bvh_aabb_intersect(bvh, mesh,
                                          VEC3(0.5f, 0.5f, 0.5f),
                                          VEC3(1.5f, 1.5f, 1.5f));
    ASSERT_TRUE(intersects);

    /* Non-overlapping AABB */
    intersects = bvh_aabb_intersect(bvh, mesh,
                                     VEC3(2.0f, 2.0f, 2.0f),
                                     VEC3(3.0f, 3.0f, 3.0f));
    ASSERT_TRUE(!intersects);

    arena_destroy(arena);
    return 0;
}

TEST(bvh_inside_outside) {
    Arena* arena = arena_create(1024 * 1024);
    TriangleMesh* mesh = create_cube_mesh(arena);
    MeshBVH* bvh = bvh_build(arena, mesh);

    /* Point inside cube */
    float sign = bvh_inside_outside(bvh, mesh, VEC3(0.5f, 0.5f, 0.5f));
    ASSERT_TRUE(sign < 0); /* Inside = negative */

    /* Point outside cube */
    sign = bvh_inside_outside(bvh, mesh, VEC3(2.0f, 0.5f, 0.5f));
    ASSERT_TRUE(sign > 0); /* Outside = positive */

    arena_destroy(arena);
    return 0;
}

TEST(mesh_capacity_enforcement) {
    Arena* arena = arena_create(10 * 1024 * 1024);
    TriangleMesh* mesh = mesh_create(arena, 4, 2);

    /* Fill to capacity — should succeed */
    for (int i = 0; i < 4; i++) {
        uint32_t v = mesh_add_vertex(mesh, (float)i, (float)i, (float)i);
        ASSERT_NE(v, UINT32_MAX);
    }
    ASSERT_EQ(mesh->vertex_count, 4);

    /* Exceeding capacity — should fail (no dynamic growth) */
    uint32_t v = mesh_add_vertex(mesh, 99.0f, 99.0f, 99.0f);
    ASSERT_EQ(v, UINT32_MAX);
    ASSERT_EQ(mesh->vertex_count, 4);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Mesh Operations Tests");

    RUN_TEST(mesh_creation);
    RUN_TEST(add_vertices);
    RUN_TEST(add_faces);
    RUN_TEST(bbox_computation);
    RUN_TEST(normal_computation);
    RUN_TEST(triangle_query);
    RUN_TEST(face_area);
    RUN_TEST(bvh_construction);
    RUN_TEST(bvh_ray_intersect);
    RUN_TEST(bvh_closest_point);
    RUN_TEST(bvh_aabb_intersect);
    RUN_TEST(bvh_inside_outside);
    RUN_TEST(mesh_capacity_enforcement);

    TEST_SUITE_END();
}
