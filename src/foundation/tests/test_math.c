/**
 * Vector and Quaternion Math Unit Tests
 *
 * Tests vector operations, quaternion operations, and matrix operations
 * for correctness and numerical stability.
 */

#include "../include/foundation.h"
#include <stdio.h>
#include <math.h>
#include "test_harness.h"

/* Tolerance for floating point comparisons */
#define EPSILON 1e-5f
#define PI 3.14159265358979323846f

/* Quaternion near-equality helper */
#define ASSERT_QUAT_NEAR(a, b, tol) do { \
    if (fabsf((a).w - (b).w) > (tol) || \
        fabsf((a).x - (b).x) > (tol) || \
        fabsf((a).y - (b).y) > (tol) || \
        fabsf((a).z - (b).z) > (tol)) { \
        printf("\n    ASSERT_QUAT_NEAR failed: (%g,%g,%g,%g) != (%g,%g,%g,%g) (tol=%g)\n    at %s:%d", \
               (double)(a).w, (double)(a).x, (double)(a).y, (double)(a).z, \
               (double)(b).w, (double)(b).x, (double)(b).y, (double)(b).z, \
               (double)(tol), __FILE__, __LINE__); \
        return -__LINE__; \
    } \
} while(0)

/* ============================================================================
 * Vector Tests
 * ============================================================================ */

TEST(vec3_creation) {
    Vec3 v1 = VEC3(1.0f, 2.0f, 3.0f);
    ASSERT_FLOAT_EQ(v1.x, 1.0f);
    ASSERT_FLOAT_EQ(v1.y, 2.0f);
    ASSERT_FLOAT_EQ(v1.z, 3.0f);

    Vec3 zero = vec3_zero();
    ASSERT_VEC3_NEAR(zero, VEC3_ZERO, EPSILON);

    Vec3 one = vec3_one();
    ASSERT_VEC3_NEAR(one, VEC3_ONE, EPSILON);

    Vec3 up = vec3_up();
    ASSERT_VEC3_NEAR(up, VEC3_UP, EPSILON);

    return 0;
}

TEST(vec3_arithmetic) {
    Vec3 a = VEC3(1.0f, 2.0f, 3.0f);
    Vec3 b = VEC3(4.0f, 5.0f, 6.0f);

    Vec3 sum = vec3_add(a, b);
    ASSERT_VEC3_NEAR(sum, VEC3(5.0f, 7.0f, 9.0f), EPSILON);

    Vec3 diff = vec3_sub(b, a);
    ASSERT_VEC3_NEAR(diff, VEC3(3.0f, 3.0f, 3.0f), EPSILON);

    Vec3 scaled = vec3_scale(a, 2.0f);
    ASSERT_VEC3_NEAR(scaled, VEC3(2.0f, 4.0f, 6.0f), EPSILON);

    Vec3 mul = vec3_mul(a, b);
    ASSERT_VEC3_NEAR(mul, VEC3(4.0f, 10.0f, 18.0f), EPSILON);

    Vec3 neg = vec3_neg(a);
    ASSERT_VEC3_NEAR(neg, VEC3(-1.0f, -2.0f, -3.0f), EPSILON);

    return 0;
}

TEST(vec3_products) {
    Vec3 a = VEC3(1.0f, 2.0f, 3.0f);
    Vec3 b = VEC3(4.0f, 5.0f, 6.0f);

    float dot = vec3_dot(a, b);
    ASSERT_FLOAT_EQ(dot, 32.0f);

    Vec3 cross = vec3_cross(a, b);
    ASSERT_VEC3_NEAR(cross, VEC3(-3.0f, 6.0f, -3.0f), EPSILON);

    Vec3 cross_rev = vec3_cross(b, a);
    ASSERT_VEC3_NEAR(cross_rev, VEC3(3.0f, -6.0f, 3.0f), EPSILON);

    return 0;
}

TEST(vec3_length) {
    Vec3 v = VEC3(3.0f, 4.0f, 0.0f);

    float len_sq = vec3_length_sq(v);
    ASSERT_FLOAT_EQ(len_sq, 25.0f);

    float len = vec3_length(v);
    ASSERT_FLOAT_EQ(len, 5.0f);

    Vec3 norm = vec3_normalize(v);
    ASSERT_VEC3_NEAR(norm, VEC3(0.6f, 0.8f, 0.0f), EPSILON);

    ASSERT_FLOAT_EQ(vec3_length(norm), 1.0f);

    Vec3 zero_norm = vec3_normalize(VEC3_ZERO);
    ASSERT_VEC3_NEAR(zero_norm, VEC3_ZERO, EPSILON);

    Vec3 a = VEC3(1.0f, 2.0f, 3.0f);
    Vec3 b = VEC3(4.0f, 6.0f, 3.0f);
    float dist = vec3_distance(a, b);
    ASSERT_FLOAT_EQ(dist, 5.0f);

    return 0;
}

TEST(vec3_interpolation) {
    Vec3 a = VEC3(0.0f, 0.0f, 0.0f);
    Vec3 b = VEC3(10.0f, 20.0f, 30.0f);

    Vec3 l0 = vec3_lerp(a, b, 0.0f);
    ASSERT_VEC3_NEAR(l0, a, EPSILON);

    Vec3 l1 = vec3_lerp(a, b, 1.0f);
    ASSERT_VEC3_NEAR(l1, b, EPSILON);

    Vec3 l05 = vec3_lerp(a, b, 0.5f);
    ASSERT_VEC3_NEAR(l05, VEC3(5.0f, 10.0f, 15.0f), EPSILON);

    return 0;
}

TEST(vec3_clamping) {
    Vec3 v = VEC3(-5.0f, 15.0f, 5.0f);

    Vec3 clamped = vec3_clamp(v, 0.0f, 10.0f);
    ASSERT_VEC3_NEAR(clamped, VEC3(0.0f, 10.0f, 5.0f), EPSILON);

    Vec3 min_v = VEC3(-1.0f, 0.0f, 0.0f);
    Vec3 max_v = VEC3(1.0f, 10.0f, 10.0f);
    Vec3 clamped_v = vec3_clamp_vec(v, min_v, max_v);
    ASSERT_VEC3_NEAR(clamped_v, VEC3(-1.0f, 10.0f, 5.0f), EPSILON);

    return 0;
}

TEST(vec3_minmax) {
    Vec3 a = VEC3(1.0f, 5.0f, 3.0f);
    Vec3 b = VEC3(4.0f, 2.0f, 6.0f);

    Vec3 min_v = vec3_min(a, b);
    ASSERT_VEC3_NEAR(min_v, VEC3(1.0f, 2.0f, 3.0f), EPSILON);

    Vec3 max_v = vec3_max(a, b);
    ASSERT_VEC3_NEAR(max_v, VEC3(4.0f, 5.0f, 6.0f), EPSILON);

    return 0;
}

/* ============================================================================
 * Quaternion Tests
 * ============================================================================ */

TEST(quat_creation) {
    Quat q = QUAT(1.0f, 0.0f, 0.0f, 0.0f);
    ASSERT_FLOAT_EQ(q.w, 1.0f);
    ASSERT_FLOAT_EQ(q.x, 0.0f);

    Quat id = quat_identity();
    ASSERT_QUAT_NEAR(id, QUAT_IDENTITY, EPSILON);

    return 0;
}

TEST(quat_normalize) {
    Quat q = QUAT(2.0f, 0.0f, 0.0f, 0.0f);
    Quat norm = quat_normalize(q);
    ASSERT_FLOAT_EQ(quat_length_sq(norm), 1.0f);

    Quat unit = QUAT_IDENTITY;
    Quat unit_norm = quat_normalize(unit);
    ASSERT_QUAT_NEAR(unit_norm, unit, EPSILON);

    Quat complex = QUAT(1.0f, 2.0f, 3.0f, 4.0f);
    Quat complex_norm = quat_normalize(complex);
    ASSERT_FLOAT_EQ(quat_length_sq(complex_norm), 1.0f);

    return 0;
}

TEST(quat_multiply) {
    Quat q = quat_from_axis_angle(VEC3_UP, PI / 4.0f);
    Quat id = QUAT_IDENTITY;

    Quat q_id = quat_multiply(q, id);
    ASSERT_QUAT_NEAR(q_id, q, EPSILON);

    Quat id_q = quat_multiply(id, q);
    ASSERT_QUAT_NEAR(id_q, q, EPSILON);

    Quat a = quat_from_axis_angle(VEC3_UP, PI / 4.0f);
    Quat b = quat_from_axis_angle(VEC3(1.0f, 0.0f, 0.0f), PI / 3.0f);
    Quat c = quat_from_axis_angle(VEC3(0.0f, 1.0f, 0.0f), PI / 6.0f);

    Quat ab_c = quat_multiply(quat_multiply(a, b), c);
    Quat a_bc = quat_multiply(a, quat_multiply(b, c));

    ASSERT_FLOAT_NEAR(ab_c.w, a_bc.w, EPSILON);
    ASSERT_FLOAT_NEAR(ab_c.x, a_bc.x, EPSILON);
    ASSERT_FLOAT_NEAR(ab_c.y, a_bc.y, EPSILON);
    ASSERT_FLOAT_NEAR(ab_c.z, a_bc.z, EPSILON);

    return 0;
}

TEST(quat_conjugate) {
    Quat q = QUAT(1.0f, 2.0f, 3.0f, 4.0f);
    Quat conj = quat_conjugate(q);

    ASSERT_FLOAT_EQ(conj.w, 1.0f);
    ASSERT_FLOAT_EQ(conj.x, -2.0f);
    ASSERT_FLOAT_EQ(conj.y, -3.0f);
    ASSERT_FLOAT_EQ(conj.z, -4.0f);

    Quat unit = quat_normalize(q);
    Quat unit_conj = quat_conjugate(unit);
    Quat product = quat_multiply(unit, unit_conj);
    ASSERT_FLOAT_NEAR(product.w, 1.0f, EPSILON);
    ASSERT_FLOAT_NEAR(product.x, 0.0f, EPSILON);

    return 0;
}

TEST(quat_rotate) {
    Quat rot_z = quat_from_axis_angle(VEC3_UP, PI / 2.0f);
    Vec3 v = VEC3(1.0f, 0.0f, 0.0f);
    Vec3 rotated = quat_rotate(rot_z, v);

    ASSERT_FLOAT_NEAR(rotated.x, 0.0f, EPSILON);
    ASSERT_FLOAT_NEAR(rotated.y, 1.0f, EPSILON);
    ASSERT_FLOAT_NEAR(rotated.z, 0.0f, EPSILON);

    Vec3 v2 = VEC3(1.0f, 2.0f, 3.0f);
    Vec3 id_rot = quat_rotate(QUAT_IDENTITY, v2);
    ASSERT_VEC3_NEAR(id_rot, v2, EPSILON);

    Quat arbitrary_rot = quat_from_euler(0.5f, 0.7f, 1.2f);
    Vec3 v3 = VEC3(3.0f, 4.0f, 5.0f);
    float original_len = vec3_length(v3);
    Vec3 rotated_v3 = quat_rotate(arbitrary_rot, v3);
    float rotated_len = vec3_length(rotated_v3);
    ASSERT_FLOAT_NEAR(original_len, rotated_len, EPSILON);

    return 0;
}

TEST(quat_from_axis_angle) {
    Quat zero_rot = quat_from_axis_angle(VEC3_UP, 0.0f);
    ASSERT_QUAT_NEAR(zero_rot, QUAT_IDENTITY, EPSILON);

    Quat half_rot = quat_from_axis_angle(VEC3_UP, PI);
    ASSERT_FLOAT_NEAR(half_rot.w, 0.0f, EPSILON);
    ASSERT_FLOAT_NEAR(half_rot.z, 1.0f, EPSILON);

    return 0;
}

TEST(quat_from_euler) {
    Quat zero_euler = quat_from_euler(0.0f, 0.0f, 0.0f);
    ASSERT_QUAT_NEAR(zero_euler, QUAT_IDENTITY, EPSILON);

    Quat roll = quat_from_euler(PI / 2.0f, 0.0f, 0.0f);
    Quat roll_expected = quat_from_axis_angle(VEC3(1.0f, 0.0f, 0.0f), PI / 2.0f);
    ASSERT_FLOAT_NEAR(fabsf(quat_dot(roll, roll_expected)), 1.0f, EPSILON);

    return 0;
}

TEST(quat_to_mat3) {
    Mat3 id_mat = quat_to_mat3(QUAT_IDENTITY);
    Mat3 expected_id = mat3_identity();

    for (int i = 0; i < 9; i++) {
        ASSERT_FLOAT_NEAR(id_mat.m[i], expected_id.m[i], EPSILON);
    }

    Quat rot = quat_from_euler(0.3f, 0.5f, 0.7f);
    Mat3 rot_mat = quat_to_mat3(rot);
    Vec3 v = VEC3(1.0f, 2.0f, 3.0f);

    Vec3 quat_rotated = quat_rotate(rot, v);
    Vec3 mat_rotated = mat3_transform(rot_mat, v);

    ASSERT_VEC3_NEAR(quat_rotated, mat_rotated, EPSILON);

    return 0;
}

TEST(quat_slerp) {
    Quat a = QUAT_IDENTITY;
    Quat b = quat_from_axis_angle(VEC3_UP, PI / 2.0f);

    Quat s0 = quat_slerp(a, b, 0.0f);
    ASSERT_FLOAT_NEAR(fabsf(quat_dot(s0, a)), 1.0f, EPSILON);

    Quat s1 = quat_slerp(a, b, 1.0f);
    ASSERT_FLOAT_NEAR(fabsf(quat_dot(s1, b)), 1.0f, EPSILON);

    Quat s05 = quat_slerp(a, b, 0.5f);
    ASSERT_FLOAT_NEAR(quat_length_sq(s05), 1.0f, EPSILON);

    return 0;
}

/* ============================================================================
 * Matrix Tests
 * ============================================================================ */

TEST(mat3_identity) {
    Mat3 id = mat3_identity();

    ASSERT_FLOAT_EQ(id.m[0], 1.0f);
    ASSERT_FLOAT_EQ(id.m[4], 1.0f);
    ASSERT_FLOAT_EQ(id.m[8], 1.0f);
    ASSERT_FLOAT_EQ(id.m[1], 0.0f);
    ASSERT_FLOAT_EQ(id.m[3], 0.0f);

    return 0;
}

TEST(mat3_transform) {
    Mat3 id = mat3_identity();
    Vec3 v = VEC3(1.0f, 2.0f, 3.0f);
    Vec3 transformed = mat3_transform(id, v);
    ASSERT_VEC3_NEAR(transformed, v, EPSILON);

    return 0;
}

TEST(mat4_identity) {
    Mat4 id = mat4_identity();

    ASSERT_FLOAT_EQ(id.m[0], 1.0f);
    ASSERT_FLOAT_EQ(id.m[5], 1.0f);
    ASSERT_FLOAT_EQ(id.m[10], 1.0f);
    ASSERT_FLOAT_EQ(id.m[15], 1.0f);
    ASSERT_FLOAT_EQ(id.m[1], 0.0f);

    return 0;
}

TEST(mat4_translate) {
    Vec3 t = VEC3(5.0f, 10.0f, 15.0f);
    Mat4 trans = mat4_translate(t);

    ASSERT_FLOAT_EQ(trans.m[12], 5.0f);
    ASSERT_FLOAT_EQ(trans.m[13], 10.0f);
    ASSERT_FLOAT_EQ(trans.m[14], 15.0f);
    ASSERT_FLOAT_EQ(trans.m[15], 1.0f);

    return 0;
}

TEST(mat4_scale) {
    Vec3 s = VEC3(2.0f, 3.0f, 4.0f);
    Mat4 scale_mat = mat4_scale(s);

    ASSERT_FLOAT_EQ(scale_mat.m[0], 2.0f);
    ASSERT_FLOAT_EQ(scale_mat.m[5], 3.0f);
    ASSERT_FLOAT_EQ(scale_mat.m[10], 4.0f);
    ASSERT_FLOAT_EQ(scale_mat.m[15], 1.0f);

    return 0;
}

/* ============================================================================
 * Utility Function Tests
 * ============================================================================ */

TEST(utility_functions) {
    ASSERT_FLOAT_EQ(clampf(-5.0f, 0.0f, 10.0f), 0.0f);
    ASSERT_FLOAT_EQ(clampf(15.0f, 0.0f, 10.0f), 10.0f);
    ASSERT_FLOAT_EQ(clampf(5.0f, 0.0f, 10.0f), 5.0f);

    ASSERT_FLOAT_EQ(lerpf(0.0f, 10.0f, 0.0f), 0.0f);
    ASSERT_FLOAT_EQ(lerpf(0.0f, 10.0f, 1.0f), 10.0f);
    ASSERT_FLOAT_EQ(lerpf(0.0f, 10.0f, 0.5f), 5.0f);

    ASSERT_FLOAT_EQ(smoothstep(0.0f, 1.0f, -0.5f), 0.0f);
    ASSERT_FLOAT_EQ(smoothstep(0.0f, 1.0f, 1.5f), 1.0f);
    ASSERT_FLOAT_NEAR(smoothstep(0.0f, 1.0f, 0.5f), 0.5f, EPSILON);

    ASSERT_MSG(align_up(0, 16) == 0, "align_up 0");
    ASSERT_MSG(align_up(1, 16) == 16, "align_up 1 to 16");
    ASSERT_MSG(align_up(16, 16) == 16, "align_up already aligned");
    ASSERT_MSG(align_up(17, 16) == 32, "align_up 17 to 32");

    ASSERT_MSG(is_power_of_two(1), "1 is power of 2");
    ASSERT_MSG(is_power_of_two(2), "2 is power of 2");
    ASSERT_MSG(is_power_of_two(16), "16 is power of 2");
    ASSERT_MSG(!is_power_of_two(0), "0 is not power of 2");
    ASSERT_MSG(!is_power_of_two(3), "3 is not power of 2");

    ASSERT_MSG(next_power_of_two(0) == 1, "next_pow2 of 0");
    ASSERT_MSG(next_power_of_two(1) == 1, "next_pow2 of 1");
    ASSERT_MSG(next_power_of_two(5) == 8, "next_pow2 of 5");
    ASSERT_MSG(next_power_of_two(16) == 16, "next_pow2 of 16");

    return 0;
}

TEST(type_sizes) {
    ASSERT_MSG(sizeof(Vec3) == 16, "Vec3 size is 16 bytes");
    ASSERT_MSG(sizeof(Quat) == 16, "Quat size is 16 bytes");
    ASSERT_MSG(sizeof(Mat3) == 48, "Mat3 size is 48 bytes");
    ASSERT_MSG(sizeof(Mat4) == 64, "Mat4 size is 64 bytes");
    ASSERT_MSG(sizeof(PCG32) == 16, "PCG32 size is 16 bytes");

    ASSERT_MSG(alignof(Vec3) == 16, "Vec3 alignment is 16");
    ASSERT_MSG(alignof(Quat) == 16, "Quat alignment is 16");
    ASSERT_MSG(alignof(Mat3) == 16, "Mat3 alignment is 16");
    ASSERT_MSG(alignof(Mat4) == 32, "Mat4 alignment is 32");

    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Math Utilities Tests");

    RUN_TEST(vec3_creation);
    RUN_TEST(vec3_arithmetic);
    RUN_TEST(vec3_products);
    RUN_TEST(vec3_length);
    RUN_TEST(vec3_interpolation);
    RUN_TEST(vec3_clamping);
    RUN_TEST(vec3_minmax);
    RUN_TEST(quat_creation);
    RUN_TEST(quat_normalize);
    RUN_TEST(quat_multiply);
    RUN_TEST(quat_conjugate);
    RUN_TEST(quat_rotate);
    RUN_TEST(quat_from_axis_angle);
    RUN_TEST(quat_from_euler);
    RUN_TEST(quat_to_mat3);
    RUN_TEST(quat_slerp);
    RUN_TEST(mat3_identity);
    RUN_TEST(mat3_transform);
    RUN_TEST(mat4_identity);
    RUN_TEST(mat4_translate);
    RUN_TEST(mat4_scale);
    RUN_TEST(utility_functions);
    RUN_TEST(type_sizes);

    TEST_SUITE_END();
}
