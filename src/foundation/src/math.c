/**
 * Math Utilities Implementation
 *
 * Vector, quaternion, and matrix operations that require sqrt/trig functions.
 * Simple operations are inlined in foundation.h for zero overhead.
 */

#include "../include/foundation.h"

/* Small epsilon for floating point comparisons */
#define MATH_EPSILON 1e-6f

/* ============================================================================
 * Vector Operations
 * ============================================================================ */

/**
 * Calculate the length (magnitude) of a vector.
 *
 * @param v The vector
 * @return The length of the vector
 */
float vec3_length(Vec3 v) {
    return sqrtf(vec3_length_sq(v));
}

/**
 * Normalize a vector to unit length.
 *
 * If the vector length is near zero, returns a zero vector to avoid division
 * by zero.
 *
 * @param v The vector to normalize
 * @return The normalized vector
 */
Vec3 vec3_normalize(Vec3 v) {
    float len_sq = vec3_length_sq(v);
    if (len_sq < MATH_EPSILON * MATH_EPSILON) {
        return VEC3_ZERO;
    }
    float inv_len = 1.0f / sqrtf(len_sq);
    return vec3_scale(v, inv_len);
}

/**
 * Calculate the distance between two points.
 *
 * @param a First point
 * @param b Second point
 * @return The Euclidean distance between the points
 */
float vec3_distance(Vec3 a, Vec3 b) {
    return vec3_length(vec3_sub(b, a));
}

/* ============================================================================
 * Quaternion Operations
 * ============================================================================ */

/**
 * Normalize a quaternion to unit length.
 *
 * @param q The quaternion to normalize
 * @return The normalized quaternion (unit quaternion)
 */
Quat quat_normalize(Quat q) {
    float len_sq = quat_length_sq(q);
    if (len_sq < MATH_EPSILON * MATH_EPSILON) {
        return QUAT_IDENTITY;
    }
    float inv_len = 1.0f / sqrtf(len_sq);
    return QUAT(q.w * inv_len, q.x * inv_len, q.y * inv_len, q.z * inv_len);
}

/**
 * Rotate a vector by a quaternion.
 *
 * Uses the formula: v' = q * v * q^(-1)
 * For unit quaternions, q^(-1) = conjugate(q)
 *
 * @param q The rotation quaternion (should be unit length)
 * @param v The vector to rotate
 * @return The rotated vector
 */
Vec3 quat_rotate(Quat q, Vec3 v) {
    /* v' = v + 2 * cross(u, cross(u, v) + w * v)
     * Algebraically identical to: v + 2*w*(u×v) + 2*(u×(u×v))
     * This specific computation order matches the GPU shader (raymarch.metal)
     * to ensure CPU and GPU produce identical ray directions. */
    Vec3 u = VEC3(q.x, q.y, q.z);
    float w = q.w;
    Vec3 cross_uv = vec3_cross(u, v);
    Vec3 inner = vec3_add(cross_uv, vec3_scale(v, w));
    Vec3 outer = vec3_cross(u, inner);
    return vec3_add(v, vec3_scale(outer, 2.0f));
}

/**
 * Create a quaternion from an axis-angle representation.
 *
 * @param axis The rotation axis (should be unit length)
 * @param angle The rotation angle in radians
 * @return The quaternion representing the rotation
 */
Quat quat_from_axis_angle(Vec3 axis, float angle) {
    float half_angle = angle * 0.5f;
    float s = sinf(half_angle);
    float c = cosf(half_angle);

    return QUAT(c, axis.x * s, axis.y * s, axis.z * s);
}

/**
 * Create a quaternion from Euler angles.
 *
 * Uses ZYX convention (yaw, pitch, roll):
 * - Roll (X): rotation about the forward axis
 * - Pitch (Y): rotation about the right axis
 * - Yaw (Z): rotation about the up axis
 *
 * @param roll Rotation about X axis in radians
 * @param pitch Rotation about Y axis in radians
 * @param yaw Rotation about Z axis in radians
 * @return The quaternion representing the combined rotation
 */
Quat quat_from_euler(float roll, float pitch, float yaw) {
    float cr = cosf(roll * 0.5f);
    float sr = sinf(roll * 0.5f);
    float cp = cosf(pitch * 0.5f);
    float sp = sinf(pitch * 0.5f);
    float cy = cosf(yaw * 0.5f);
    float sy = sinf(yaw * 0.5f);

    return QUAT(
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    );
}

/**
 * Convert a quaternion to a 3x3 rotation matrix.
 *
 * @param q The quaternion (should be unit length)
 * @return The equivalent rotation matrix
 */
Mat3 quat_to_mat3(Quat q) {
    float xx = q.x * q.x;
    float yy = q.y * q.y;
    float zz = q.z * q.z;
    float xy = q.x * q.y;
    float xz = q.x * q.z;
    float yz = q.y * q.z;
    float wx = q.w * q.x;
    float wy = q.w * q.y;
    float wz = q.w * q.z;

    Mat3 m;
    /* Column 0 */
    m.m[0] = 1.0f - 2.0f * (yy + zz);
    m.m[1] = 2.0f * (xy + wz);
    m.m[2] = 2.0f * (xz - wy);

    /* Column 1 */
    m.m[3] = 2.0f * (xy - wz);
    m.m[4] = 1.0f - 2.0f * (xx + zz);
    m.m[5] = 2.0f * (yz + wx);

    /* Column 2 */
    m.m[6] = 2.0f * (xz + wy);
    m.m[7] = 2.0f * (yz - wx);
    m.m[8] = 1.0f - 2.0f * (xx + yy);

    /* Zero padding */
    m._pad[0] = m._pad[1] = m._pad[2] = 0.0f;

    return m;
}

/**
 * Convert a quaternion to a 4x4 transformation matrix (rotation only).
 *
 * @param q The quaternion (should be unit length)
 * @return The equivalent 4x4 rotation matrix
 */
Mat4 quat_to_mat4(Quat q) {
    Mat3 m3 = quat_to_mat3(q);

    Mat4 m = {0};
    /* Copy rotation from Mat3 */
    m.m[0] = m3.m[0];  m.m[1] = m3.m[1];  m.m[2] = m3.m[2];   m.m[3] = 0.0f;
    m.m[4] = m3.m[3];  m.m[5] = m3.m[4];  m.m[6] = m3.m[5];   m.m[7] = 0.0f;
    m.m[8] = m3.m[6];  m.m[9] = m3.m[7];  m.m[10] = m3.m[8];  m.m[11] = 0.0f;
    m.m[12] = 0.0f;    m.m[13] = 0.0f;    m.m[14] = 0.0f;     m.m[15] = 1.0f;

    return m;
}

/**
 * Spherical linear interpolation between two quaternions.
 *
 * Interpolates along the shortest path on the unit sphere.
 *
 * @param a Start quaternion (t=0)
 * @param b End quaternion (t=1)
 * @param t Interpolation parameter [0, 1]
 * @return The interpolated quaternion
 */
Quat quat_slerp(Quat a, Quat b, float t) {
    float dot = quat_dot(a, b);

    /* If dot is negative, negate one quaternion to take shorter path */
    if (dot < 0.0f) {
        b = QUAT(-b.w, -b.x, -b.y, -b.z);
        dot = -dot;
    }

    /* If quaternions are very close, use linear interpolation */
    if (dot > 0.9995f) {
        Quat result = QUAT(
            a.w + t * (b.w - a.w),
            a.x + t * (b.x - a.x),
            a.y + t * (b.y - a.y),
            a.z + t * (b.z - a.z)
        );
        return quat_normalize(result);
    }

    /* Spherical interpolation */
    float theta_0 = acosf(dot);
    float theta = theta_0 * t;

    float sin_theta = sinf(theta);
    float sin_theta_0 = sinf(theta_0);

    float s0 = cosf(theta) - dot * sin_theta / sin_theta_0;
    float s1 = sin_theta / sin_theta_0;

    return QUAT(
        s0 * a.w + s1 * b.w,
        s0 * a.x + s1 * b.x,
        s0 * a.y + s1 * b.y,
        s0 * a.z + s1 * b.z
    );
}

/**
 * Convert a 3x3 rotation matrix to a quaternion.
 *
 * @param m The rotation matrix
 * @return The equivalent quaternion
 */
Quat quat_from_mat3(Mat3 m) {
    /* Shepperd's method for numerical stability */
    float trace = m.m[0] + m.m[4] + m.m[8];
    Quat q;

    if (trace > 0.0f) {
        float s = sqrtf(trace + 1.0f) * 2.0f;
        q.w = 0.25f * s;
        q.x = (m.m[5] - m.m[7]) / s;
        q.y = (m.m[6] - m.m[2]) / s;
        q.z = (m.m[1] - m.m[3]) / s;
    } else if (m.m[0] > m.m[4] && m.m[0] > m.m[8]) {
        float s = sqrtf(1.0f + m.m[0] - m.m[4] - m.m[8]) * 2.0f;
        q.w = (m.m[5] - m.m[7]) / s;
        q.x = 0.25f * s;
        q.y = (m.m[3] + m.m[1]) / s;
        q.z = (m.m[6] + m.m[2]) / s;
    } else if (m.m[4] > m.m[8]) {
        float s = sqrtf(1.0f + m.m[4] - m.m[0] - m.m[8]) * 2.0f;
        q.w = (m.m[6] - m.m[2]) / s;
        q.x = (m.m[3] + m.m[1]) / s;
        q.y = 0.25f * s;
        q.z = (m.m[7] + m.m[5]) / s;
    } else {
        float s = sqrtf(1.0f + m.m[8] - m.m[0] - m.m[4]) * 2.0f;
        q.w = (m.m[1] - m.m[3]) / s;
        q.x = (m.m[6] + m.m[2]) / s;
        q.y = (m.m[7] + m.m[5]) / s;
        q.z = 0.25f * s;
    }

    return quat_normalize(q);
}

/* ============================================================================
 * Matrix Operations
 * ============================================================================ */

/**
 * Multiply two 3x3 matrices.
 *
 * @param a Left matrix
 * @param b Right matrix
 * @return The product a * b
 */
Mat3 mat3_multiply(Mat3 a, Mat3 b) {
    Mat3 result;

    for (int col = 0; col < 3; col++) {
        for (int row = 0; row < 3; row++) {
            result.m[col * 3 + row] =
                a.m[0 * 3 + row] * b.m[col * 3 + 0] +
                a.m[1 * 3 + row] * b.m[col * 3 + 1] +
                a.m[2 * 3 + row] * b.m[col * 3 + 2];
        }
    }

    result._pad[0] = result._pad[1] = result._pad[2] = 0.0f;
    return result;
}

/**
 * Transform a vector by a 3x3 matrix.
 *
 * @param m The transformation matrix
 * @param v The vector to transform
 * @return The transformed vector m * v
 */
Vec3 mat3_transform(Mat3 m, Vec3 v) {
    return VEC3(
        m.m[0] * v.x + m.m[3] * v.y + m.m[6] * v.z,
        m.m[1] * v.x + m.m[4] * v.y + m.m[7] * v.z,
        m.m[2] * v.x + m.m[5] * v.y + m.m[8] * v.z
    );
}

/**
 * Transpose a 3x3 matrix.
 *
 * @param m The matrix to transpose
 * @return The transposed matrix
 */
Mat3 mat3_transpose(Mat3 m) {
    Mat3 result;
    result.m[0] = m.m[0];  result.m[1] = m.m[3];  result.m[2] = m.m[6];
    result.m[3] = m.m[1];  result.m[4] = m.m[4];  result.m[5] = m.m[7];
    result.m[6] = m.m[2];  result.m[7] = m.m[5];  result.m[8] = m.m[8];
    result._pad[0] = result._pad[1] = result._pad[2] = 0.0f;
    return result;
}

/**
 * Multiply two 4x4 matrices.
 *
 * @param a Left matrix
 * @param b Right matrix
 * @return The product a * b
 */
Mat4 mat4_multiply(Mat4 a, Mat4 b) {
    Mat4 result = {0};

    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
            result.m[col * 4 + row] =
                a.m[0 * 4 + row] * b.m[col * 4 + 0] +
                a.m[1 * 4 + row] * b.m[col * 4 + 1] +
                a.m[2 * 4 + row] * b.m[col * 4 + 2] +
                a.m[3 * 4 + row] * b.m[col * 4 + 3];
        }
    }

    return result;
}

/**
 * Transpose a 4x4 matrix.
 *
 * @param m The matrix to transpose
 * @return The transposed matrix
 */
Mat4 mat4_transpose(Mat4 m) {
    Mat4 result;
    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
            result.m[col * 4 + row] = m.m[row * 4 + col];
        }
    }
    return result;
}

/**
 * Create a translation matrix.
 *
 * @param translation The translation vector
 * @return The 4x4 translation matrix
 */
Mat4 mat4_translate(Vec3 translation) {
    Mat4 m = mat4_identity();
    m.m[12] = translation.x;
    m.m[13] = translation.y;
    m.m[14] = translation.z;
    return m;
}

/**
 * Create a scale matrix.
 *
 * @param scale The scale factors for each axis
 * @return The 4x4 scale matrix
 */
Mat4 mat4_scale(Vec3 scale) {
    Mat4 m = {0};
    m.m[0] = scale.x;
    m.m[5] = scale.y;
    m.m[10] = scale.z;
    m.m[15] = 1.0f;
    return m;
}

/**
 * Create a rotation matrix from a quaternion.
 *
 * @param rotation The rotation quaternion
 * @return The 4x4 rotation matrix
 */
Mat4 mat4_rotate(Quat rotation) {
    return quat_to_mat4(rotation);
}

/**
 * Create a transformation matrix from translation, rotation, and scale.
 *
 * @param translation The translation vector
 * @param rotation The rotation quaternion
 * @param scale The scale factors
 * @return The combined TRS transformation matrix
 */
Mat4 mat4_from_trs(Vec3 translation, Quat rotation, Vec3 scale) {
    Mat3 rot = quat_to_mat3(rotation);

    Mat4 m = {0};

    /* Rotation * Scale (combined) */
    m.m[0] = rot.m[0] * scale.x;
    m.m[1] = rot.m[1] * scale.x;
    m.m[2] = rot.m[2] * scale.x;
    m.m[3] = 0.0f;

    m.m[4] = rot.m[3] * scale.y;
    m.m[5] = rot.m[4] * scale.y;
    m.m[6] = rot.m[5] * scale.y;
    m.m[7] = 0.0f;

    m.m[8] = rot.m[6] * scale.z;
    m.m[9] = rot.m[7] * scale.z;
    m.m[10] = rot.m[8] * scale.z;
    m.m[11] = 0.0f;

    /* Translation */
    m.m[12] = translation.x;
    m.m[13] = translation.y;
    m.m[14] = translation.z;
    m.m[15] = 1.0f;

    return m;
}
