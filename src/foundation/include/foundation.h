/**
 * Foundation Module - Core types, allocators, SIMD utilities, and math primitives
 *
 * This is the base layer providing foundational components that all other
 * submodules (01-10) depend on. This module has zero external dependencies.
 *
 * Components:
 * - Basic types (Vec3, Quat, Mat3, Mat4) with SIMD-aligned padding
 * - Arena allocator with O(1) bump allocation and scoped allocation
 * - SIMD abstraction layer (AVX2, ARM NEON, scalar fallback)
 * - Math utilities (vector/quaternion operations)
 * - Atomics and lock-free primitives
 * - PCG32 random number generation
 * - Debug assertions and logging
 */

#ifndef FOUNDATION_H
#define FOUNDATION_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <stdalign.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Section 1: Platform Detection and Configuration
 * ============================================================================ */

/* Detect SIMD capabilities */
#if defined(__AVX2__)
    #define FOUNDATION_SIMD_AVX2 1
    #define FOUNDATION_SIMD_WIDTH 8
    #define FOUNDATION_SIMD_ALIGNMENT 32
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define FOUNDATION_SIMD_NEON 1
    #define FOUNDATION_SIMD_WIDTH 4
    #define FOUNDATION_SIMD_ALIGNMENT 16
#else
    #define FOUNDATION_SIMD_SCALAR 1
    #define FOUNDATION_SIMD_WIDTH 1
    #define FOUNDATION_SIMD_ALIGNMENT 4
#endif

/* Default alignment for arena allocations */
#define FOUNDATION_DEFAULT_ALIGNMENT 16

/* Compiler hints */
#if defined(__GNUC__) || defined(__clang__)
    #define FOUNDATION_INLINE static inline __attribute__((always_inline))
    #define FOUNDATION_LIKELY(x)   __builtin_expect(!!(x), 1)
    #define FOUNDATION_UNLIKELY(x) __builtin_expect(!!(x), 0)
    #define FOUNDATION_RESTRICT __restrict__
#else
    #define FOUNDATION_INLINE static inline
    #define FOUNDATION_LIKELY(x)   (x)
    #define FOUNDATION_UNLIKELY(x) (x)
    #define FOUNDATION_RESTRICT
#endif

/* ============================================================================
 * Section 2: Basic Types (16/32-byte SIMD aligned)
 * ============================================================================ */

/* Alignment attribute macro for portable alignment specification */
#if defined(__GNUC__) || defined(__clang__)
    #define FOUNDATION_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)
    #define FOUNDATION_ALIGN(n) __declspec(align(n))
#else
    #define FOUNDATION_ALIGN(n)
#endif

/**
 * 3D Vector with SIMD-friendly padding
 * 16 bytes total (12 bytes data + 4 bytes padding)
 */
typedef struct Vec3 {
    float x, y, z;
    float _pad;
} FOUNDATION_ALIGN(16) Vec3;

/**
 * Quaternion for rotations
 * 16 bytes, 16-byte aligned
 * Convention: w is the scalar component
 */
typedef struct Quat {
    float w, x, y, z;
} FOUNDATION_ALIGN(16) Quat;

/**
 * 3x3 Matrix (column-major order)
 * 48 bytes total (36 bytes data + 12 bytes padding)
 * Note: 16-byte aligned (48 is not a multiple of 32)
 */
typedef struct Mat3 {
    float m[9];
    float _pad[3];
} FOUNDATION_ALIGN(16) Mat3;

/**
 * 4x4 Matrix (column-major order)
 * 64 bytes, 32-byte aligned
 */
typedef struct Mat4 {
    float m[16];
} FOUNDATION_ALIGN(32) Mat4;

/* Convenience constructors */
#define VEC3(x_, y_, z_) ((Vec3){.x = (x_), .y = (y_), .z = (z_), ._pad = 0.0f})
#define VEC3_ZERO ((Vec3){.x = 0.0f, .y = 0.0f, .z = 0.0f, ._pad = 0.0f})
#define VEC3_ONE  ((Vec3){.x = 1.0f, .y = 1.0f, .z = 1.0f, ._pad = 0.0f})
#define VEC3_UP   ((Vec3){.x = 0.0f, .y = 0.0f, .z = 1.0f, ._pad = 0.0f})

#define QUAT(w_, x_, y_, z_) ((Quat){.w = (w_), .x = (x_), .y = (y_), .z = (z_)})
#define QUAT_IDENTITY ((Quat){.w = 1.0f, .x = 0.0f, .y = 0.0f, .z = 0.0f})

/* ============================================================================
 * Section 3: Arena Allocator
 * ============================================================================ */

/**
 * Arena allocator with O(1) bump allocation
 * Single contiguous memory region with a growing watermark
 */
typedef struct Arena {
    void*  data;      /* Start of arena memory */
    size_t capacity;  /* Total arena size in bytes */
    size_t used;      /* Current watermark position */
} Arena;

/**
 * Scoped allocation context for RAII-style automatic cleanup
 */
typedef struct ArenaScope {
    Arena* arena;
    size_t saved_used;
} ArenaScope;

/* Arena lifecycle */
Arena* arena_create(size_t capacity);
void   arena_destroy(Arena* arena);

/* Allocation (O(1) bump allocation) */
void*  arena_alloc(Arena* arena, size_t size);
void*  arena_alloc_aligned(Arena* arena, size_t size, size_t alignment);
void*  arena_alloc_zero(Arena* arena, size_t size);

/* Reset (O(1) - sets watermark to 0) */
void   arena_reset(Arena* arena);

/* Query */
size_t arena_remaining(const Arena* arena);
float  arena_utilization(const Arena* arena);

/* Scoped allocation */
ArenaScope arena_scope_begin(Arena* arena);
void       arena_scope_end(ArenaScope scope);

/* Macro for scoped allocation (RAII-style) */
#define ARENA_SCOPE(arena) \
    for (ArenaScope _scope = arena_scope_begin(arena); \
         _scope.arena != NULL; \
         arena_scope_end(_scope), _scope.arena = NULL)

/* Typed allocation macros */
#define arena_alloc_type(arena, T) \
    ((T*)arena_alloc_aligned(arena, sizeof(T), alignof(T)))

#define arena_alloc_array(arena, T, count) \
    ((T*)arena_alloc_aligned(arena, sizeof(T) * (count), alignof(T)))

/* ============================================================================
 * Section 4: SIMD Abstraction Layer
 * ============================================================================ */

#if defined(FOUNDATION_SIMD_AVX2)

#include <immintrin.h>

typedef __m256  simd_float;
typedef __m256i simd_int;

#define simd_load_ps(p)         _mm256_load_ps(p)
#define simd_loadu_ps(p)        _mm256_loadu_ps(p)
#define simd_store_ps(p, v)     _mm256_store_ps(p, v)
#define simd_storeu_ps(p, v)    _mm256_storeu_ps(p, v)
#define simd_set1_ps(x)         _mm256_set1_ps(x)
#define simd_setzero_ps()       _mm256_setzero_ps()
#define simd_add_ps(a, b)       _mm256_add_ps(a, b)
#define simd_sub_ps(a, b)       _mm256_sub_ps(a, b)
#define simd_mul_ps(a, b)       _mm256_mul_ps(a, b)
#define simd_div_ps(a, b)       _mm256_div_ps(a, b)
#define simd_fmadd_ps(a, b, c)  _mm256_fmadd_ps(a, b, c)
#define simd_fmsub_ps(a, b, c)  _mm256_fmsub_ps(a, b, c)
#define simd_sqrt_ps(a)         _mm256_sqrt_ps(a)
#define simd_rsqrt_ps(a)        _mm256_rsqrt_ps(a)
#define simd_min_ps(a, b)       _mm256_min_ps(a, b)
#define simd_max_ps(a, b)       _mm256_max_ps(a, b)
#define simd_and_ps(a, b)       _mm256_and_ps(a, b)
#define simd_or_ps(a, b)        _mm256_or_ps(a, b)
#define simd_xor_ps(a, b)       _mm256_xor_ps(a, b)
#define simd_cmp_lt_ps(a, b)    _mm256_cmp_ps(a, b, _CMP_LT_OQ)
#define simd_cmp_gt_ps(a, b)    _mm256_cmp_ps(a, b, _CMP_GT_OQ)
#define simd_cmp_le_ps(a, b)    _mm256_cmp_ps(a, b, _CMP_LE_OQ)
#define simd_cmp_ge_ps(a, b)    _mm256_cmp_ps(a, b, _CMP_GE_OQ)
#define simd_blendv_ps(a, b, m) _mm256_blendv_ps(a, b, m)

/* Integer operations */
#define simd_load_si(p)         _mm256_load_si256((const __m256i*)(p))
#define simd_store_si(p, v)     _mm256_store_si256((__m256i*)(p), v)
#define simd_set1_epi32(x)      _mm256_set1_epi32(x)
#define simd_add_epi32(a, b)    _mm256_add_epi32(a, b)
#define simd_sub_epi32(a, b)    _mm256_sub_epi32(a, b)
#define simd_mullo_epi32(a, b)  _mm256_mullo_epi32(a, b)

/* Conversion */
#define simd_cvt_ps_epi32(a)    _mm256_cvtps_epi32(a)
#define simd_cvt_epi32_ps(a)    _mm256_cvtepi32_ps(a)

#elif defined(FOUNDATION_SIMD_NEON)

#include <arm_neon.h>

typedef float32x4_t simd_float;
typedef int32x4_t   simd_int;

#define simd_load_ps(p)         vld1q_f32(p)
#define simd_loadu_ps(p)        vld1q_f32(p)
#define simd_store_ps(p, v)     vst1q_f32(p, v)
#define simd_storeu_ps(p, v)    vst1q_f32(p, v)
#define simd_set1_ps(x)         vdupq_n_f32(x)
#define simd_setzero_ps()       vdupq_n_f32(0.0f)
#define simd_add_ps(a, b)       vaddq_f32(a, b)
#define simd_sub_ps(a, b)       vsubq_f32(a, b)
#define simd_mul_ps(a, b)       vmulq_f32(a, b)
#define simd_div_ps(a, b)       vdivq_f32(a, b)
#define simd_fmadd_ps(a, b, c)  vfmaq_f32(c, a, b)
#define simd_fmsub_ps(a, b, c)  vfmsq_f32(c, a, b)
#define simd_sqrt_ps(a)         vsqrtq_f32(a)
#define simd_rsqrt_ps(a)        vrsqrteq_f32(a)
#define simd_min_ps(a, b)       vminq_f32(a, b)
#define simd_max_ps(a, b)       vmaxq_f32(a, b)
#define simd_and_ps(a, b)       vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)))
#define simd_or_ps(a, b)        vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)))
#define simd_xor_ps(a, b)       vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)))
#define simd_cmp_lt_ps(a, b)    vreinterpretq_f32_u32(vcltq_f32(a, b))
#define simd_cmp_gt_ps(a, b)    vreinterpretq_f32_u32(vcgtq_f32(a, b))
#define simd_cmp_le_ps(a, b)    vreinterpretq_f32_u32(vcleq_f32(a, b))
#define simd_cmp_ge_ps(a, b)    vreinterpretq_f32_u32(vcgeq_f32(a, b))
#define simd_blendv_ps(a, b, m) vbslq_f32(vreinterpretq_u32_f32(m), b, a)

/* Integer operations */
#define simd_load_si(p)         vld1q_s32((const int32_t*)(p))
#define simd_store_si(p, v)     vst1q_s32((int32_t*)(p), v)
#define simd_set1_epi32(x)      vdupq_n_s32(x)
#define simd_add_epi32(a, b)    vaddq_s32(a, b)
#define simd_sub_epi32(a, b)    vsubq_s32(a, b)
#define simd_mullo_epi32(a, b)  vmulq_s32(a, b)

/* Conversion */
#define simd_cvt_ps_epi32(a)    vcvtq_s32_f32(a)
#define simd_cvt_epi32_ps(a)    vcvtq_f32_s32(a)

#else /* Scalar fallback */

typedef float   simd_float;
typedef int32_t simd_int;

#define simd_load_ps(p)         (*(p))
#define simd_loadu_ps(p)        (*(p))
#define simd_store_ps(p, v)     (*(p) = (v))
#define simd_storeu_ps(p, v)    (*(p) = (v))
#define simd_set1_ps(x)         (x)
#define simd_setzero_ps()       (0.0f)
#define simd_add_ps(a, b)       ((a) + (b))
#define simd_sub_ps(a, b)       ((a) - (b))
#define simd_mul_ps(a, b)       ((a) * (b))
#define simd_div_ps(a, b)       ((a) / (b))
#define simd_fmadd_ps(a, b, c)  ((a) * (b) + (c))
#define simd_fmsub_ps(a, b, c)  ((a) * (b) - (c))
#define simd_sqrt_ps(a)         sqrtf(a)
#define simd_rsqrt_ps(a)        (1.0f / sqrtf(a))
#define simd_min_ps(a, b)       ((a) < (b) ? (a) : (b))
#define simd_max_ps(a, b)       ((a) > (b) ? (a) : (b))

/* Integer operations */
#define simd_load_si(p)         (*(const int32_t*)(p))
#define simd_store_si(p, v)     (*(int32_t*)(p) = (v))
#define simd_set1_epi32(x)      (x)
#define simd_add_epi32(a, b)    ((a) + (b))
#define simd_sub_epi32(a, b)    ((a) - (b))
#define simd_mullo_epi32(a, b)  ((a) * (b))

/* Conversion */
#define simd_cvt_ps_epi32(a)    ((int32_t)(a))
#define simd_cvt_epi32_ps(a)    ((float)(a))

#endif /* SIMD platform selection */

/* Helper macros for SIMD loops */
#define SIMD_LOOP_START(count) \
    const uint32_t _simd_count = (count) & ~(FOUNDATION_SIMD_WIDTH - 1)

#define SIMD_LOOP_REMAINDER(i, count) \
    for (uint32_t i = _simd_count; i < (count); i++)

/* ============================================================================
 * Section 5: Atomics and Lock-Free Primitives
 * ============================================================================ */

typedef atomic_uint_fast32_t atomic_u32;
typedef atomic_uint_fast64_t atomic_u64;
typedef atomic_int_fast32_t  atomic_i32;
typedef atomic_bool          atomic_flag_t;

/* Relaxed operations (counters, statistics) */
#define atomic_load_relaxed(p)      atomic_load_explicit(p, memory_order_relaxed)
#define atomic_store_relaxed(p, v)  atomic_store_explicit(p, v, memory_order_relaxed)
#define atomic_add_relaxed(p, v)    atomic_fetch_add_explicit(p, v, memory_order_relaxed)
#define atomic_sub_relaxed(p, v)    atomic_fetch_sub_explicit(p, v, memory_order_relaxed)

/* Acquire-release operations (synchronization) */
#define atomic_load_acquire(p)      atomic_load_explicit(p, memory_order_acquire)
#define atomic_store_release(p, v)  atomic_store_explicit(p, v, memory_order_release)
#define atomic_add_release(p, v)    atomic_fetch_add_explicit(p, v, memory_order_release)

/* Compare-and-swap */
#define atomic_cas_weak(p, expected, desired) \
    atomic_compare_exchange_weak_explicit(p, expected, desired, \
        memory_order_acq_rel, memory_order_acquire)

#define atomic_cas_strong(p, expected, desired) \
    atomic_compare_exchange_strong_explicit(p, expected, desired, \
        memory_order_acq_rel, memory_order_acquire)

/* Memory fences */
#define atomic_fence_acquire()  atomic_thread_fence(memory_order_acquire)
#define atomic_fence_release()  atomic_thread_fence(memory_order_release)
#define atomic_fence_seq_cst()  atomic_thread_fence(memory_order_seq_cst)

/* ============================================================================
 * Section 6: PCG32 Random Number Generation
 * ============================================================================ */

/**
 * PCG32 random number generator state
 * PCG (Permuted Congruential Generator) - fast, statistically excellent
 */
typedef struct PCG32 {
    uint64_t state;
    uint64_t inc;
} PCG32;

/* Seeding */
void pcg32_seed(PCG32* rng, uint64_t seed);
void pcg32_seed_dual(PCG32* rng, uint64_t seed, uint64_t seq);

/* Random generation */
uint32_t pcg32_random(PCG32* rng);
uint32_t pcg32_bounded(PCG32* rng, uint32_t bound);

/* Float generation */
float pcg32_float(PCG32* rng);
float pcg32_range(PCG32* rng, float min, float max);

/* Vector generation */
Vec3 pcg32_vec3_unit(PCG32* rng);
Vec3 pcg32_vec3_range(PCG32* rng, Vec3 min, Vec3 max);

/* Global thread-local RNG */
PCG32* pcg32_thread_local(void);

/* ============================================================================
 * Section 7: Utility Functions
 * ============================================================================ */

/* Floating-point utilities */
FOUNDATION_INLINE float minf(float a, float b) { return fminf(a, b); }
FOUNDATION_INLINE float maxf(float a, float b) { return fmaxf(a, b); }

FOUNDATION_INLINE float clampf(float x, float min_val, float max_val) {
    return fminf(fmaxf(x, min_val), max_val);
}

FOUNDATION_INLINE float lerpf(float a, float b, float t) {
    return a + t * (b - a);
}

FOUNDATION_INLINE float smoothstep(float edge0, float edge1, float x) {
    float t = clampf((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

FOUNDATION_INLINE float absf(float x) {
    return fabsf(x);
}

/* Integer utilities */
FOUNDATION_INLINE uint32_t align_up(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

FOUNDATION_INLINE size_t align_up_size(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

FOUNDATION_INLINE bool is_power_of_two(uint32_t x) {
    return x && !(x & (x - 1));
}

FOUNDATION_INLINE uint32_t next_power_of_two(uint32_t x) {
    if (x == 0) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

FOUNDATION_INLINE int32_t min_i32(int32_t a, int32_t b) {
    return a < b ? a : b;
}

FOUNDATION_INLINE int32_t max_i32(int32_t a, int32_t b) {
    return a > b ? a : b;
}

FOUNDATION_INLINE uint32_t min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

FOUNDATION_INLINE uint32_t max_u32(uint32_t a, uint32_t b) {
    return a > b ? a : b;
}

/* ============================================================================
 * Section 8: Debug and Assertions
 * ============================================================================ */

#ifdef FOUNDATION_DEBUG

#define FOUNDATION_ASSERT(cond, msg) do { \
    if (FOUNDATION_UNLIKELY(!(cond))) { \
        fprintf(stderr, "ASSERT FAILED: %s\n  at %s:%d\n  %s\n", \
                #cond, __FILE__, __LINE__, msg); \
        abort(); \
    } \
} while(0)

#define FOUNDATION_LOG(fmt, ...) \
    fprintf(stderr, "[FOUNDATION] " fmt "\n", ##__VA_ARGS__)

#define FOUNDATION_LOG_ERROR(fmt, ...) \
    fprintf(stderr, "[FOUNDATION ERROR] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#else

#define FOUNDATION_ASSERT(cond, msg) ((void)0)
#define FOUNDATION_LOG(fmt, ...) ((void)0)
#define FOUNDATION_LOG_ERROR(fmt, ...) ((void)0)

#endif /* FOUNDATION_DEBUG */

/* Compile-time assertions */
#define FOUNDATION_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)

/* ============================================================================
 * Section 9: Math Utilities - Vector Operations (Inline)
 * ============================================================================ */

/* Vector creation */
FOUNDATION_INLINE Vec3 vec3_zero(void) { return VEC3_ZERO; }
FOUNDATION_INLINE Vec3 vec3_one(void)  { return VEC3_ONE; }
FOUNDATION_INLINE Vec3 vec3_up(void)   { return VEC3_UP; }

/* Arithmetic */
FOUNDATION_INLINE Vec3 vec3_add(Vec3 a, Vec3 b) {
    return VEC3(a.x + b.x, a.y + b.y, a.z + b.z);
}

FOUNDATION_INLINE Vec3 vec3_sub(Vec3 a, Vec3 b) {
    return VEC3(a.x - b.x, a.y - b.y, a.z - b.z);
}

FOUNDATION_INLINE Vec3 vec3_scale(Vec3 v, float s) {
    return VEC3(v.x * s, v.y * s, v.z * s);
}

FOUNDATION_INLINE Vec3 vec3_mul(Vec3 a, Vec3 b) {
    return VEC3(a.x * b.x, a.y * b.y, a.z * b.z);
}

FOUNDATION_INLINE Vec3 vec3_neg(Vec3 v) {
    return VEC3(-v.x, -v.y, -v.z);
}

FOUNDATION_INLINE Vec3 vec3_abs(Vec3 v) {
    return VEC3(absf(v.x), absf(v.y), absf(v.z));
}

/* Products */
FOUNDATION_INLINE float vec3_dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

FOUNDATION_INLINE Vec3 vec3_cross(Vec3 a, Vec3 b) {
    return VEC3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

/* Length operations */
FOUNDATION_INLINE float vec3_length_sq(Vec3 v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

/* Implemented in math.c (uses sqrt) */
float vec3_length(Vec3 v);
Vec3  vec3_normalize(Vec3 v);
float vec3_distance(Vec3 a, Vec3 b);

/* Interpolation */
FOUNDATION_INLINE Vec3 vec3_lerp(Vec3 a, Vec3 b, float t) {
    return VEC3(
        a.x + t * (b.x - a.x),
        a.y + t * (b.y - a.y),
        a.z + t * (b.z - a.z)
    );
}

/* Clamping */
FOUNDATION_INLINE Vec3 vec3_clamp(Vec3 v, float min_val, float max_val) {
    return VEC3(
        clampf(v.x, min_val, max_val),
        clampf(v.y, min_val, max_val),
        clampf(v.z, min_val, max_val)
    );
}

FOUNDATION_INLINE Vec3 vec3_clamp_vec(Vec3 v, Vec3 min_v, Vec3 max_v) {
    return VEC3(
        clampf(v.x, min_v.x, max_v.x),
        clampf(v.y, min_v.y, max_v.y),
        clampf(v.z, min_v.z, max_v.z)
    );
}

/* Min/Max */
FOUNDATION_INLINE Vec3 vec3_min(Vec3 a, Vec3 b) {
    return VEC3(
        a.x < b.x ? a.x : b.x,
        a.y < b.y ? a.y : b.y,
        a.z < b.z ? a.z : b.z
    );
}

FOUNDATION_INLINE Vec3 vec3_max(Vec3 a, Vec3 b) {
    return VEC3(
        a.x > b.x ? a.x : b.x,
        a.y > b.y ? a.y : b.y,
        a.z > b.z ? a.z : b.z
    );
}

/* ============================================================================
 * Section 10: Math Utilities - Quaternion Operations
 * ============================================================================ */

/* Quaternion creation */
FOUNDATION_INLINE Quat quat_identity(void) { return QUAT_IDENTITY; }

/* Implemented in math.c (need sqrt/trig) */
Quat quat_normalize(Quat q);
Vec3 quat_rotate(Quat q, Vec3 v);
Quat quat_from_axis_angle(Vec3 axis, float angle);
Quat quat_from_euler(float roll, float pitch, float yaw);
Mat3 quat_to_mat3(Quat q);
Mat4 quat_to_mat4(Quat q);
Quat quat_slerp(Quat a, Quat b, float t);
Quat quat_from_mat3(Mat3 m);

/* Hamilton product */
FOUNDATION_INLINE Quat quat_multiply(Quat a, Quat b) {
    return QUAT(
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
    );
}

/* Conjugate (inverse for unit quaternions) */
FOUNDATION_INLINE Quat quat_conjugate(Quat q) {
    return QUAT(q.w, -q.x, -q.y, -q.z);
}

/* Dot product */
FOUNDATION_INLINE float quat_dot(Quat a, Quat b) {
    return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
}

/* Length squared */
FOUNDATION_INLINE float quat_length_sq(Quat q) {
    return q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
}

/* ============================================================================
 * Section 11: Matrix Operations
 * ============================================================================ */

/* Matrix creation */
FOUNDATION_INLINE Mat3 mat3_identity(void) {
    Mat3 m = {0};
    m.m[0] = 1.0f; m.m[4] = 1.0f; m.m[8] = 1.0f;
    return m;
}

FOUNDATION_INLINE Mat4 mat4_identity(void) {
    Mat4 m = {0};
    m.m[0] = 1.0f; m.m[5] = 1.0f; m.m[10] = 1.0f; m.m[15] = 1.0f;
    return m;
}

/* Mat3 operations - implemented in math.c */
Mat3 mat3_multiply(Mat3 a, Mat3 b);
Vec3 mat3_transform(Mat3 m, Vec3 v);
Mat3 mat3_transpose(Mat3 m);

/* Mat4 operations - implemented in math.c */
Mat4 mat4_multiply(Mat4 a, Mat4 b);
Mat4 mat4_transpose(Mat4 m);
Mat4 mat4_translate(Vec3 translation);
Mat4 mat4_scale(Vec3 scale);
Mat4 mat4_rotate(Quat rotation);
Mat4 mat4_from_trs(Vec3 translation, Quat rotation, Vec3 scale);

/* ============================================================================
 * Section 12: Type Size Verification
 * ============================================================================ */

FOUNDATION_STATIC_ASSERT(sizeof(Vec3) == 16, "Vec3 must be 16 bytes");
FOUNDATION_STATIC_ASSERT(sizeof(Quat) == 16, "Quat must be 16 bytes");
FOUNDATION_STATIC_ASSERT(sizeof(Mat3) == 48, "Mat3 must be 48 bytes");
FOUNDATION_STATIC_ASSERT(sizeof(Mat4) == 64, "Mat4 must be 64 bytes");
FOUNDATION_STATIC_ASSERT(alignof(Vec3) == 16, "Vec3 must be 16-byte aligned");
FOUNDATION_STATIC_ASSERT(alignof(Quat) == 16, "Quat must be 16-byte aligned");
FOUNDATION_STATIC_ASSERT(alignof(Mat3) == 16, "Mat3 must be 16-byte aligned");
FOUNDATION_STATIC_ASSERT(alignof(Mat4) == 32, "Mat4 must be 32-byte aligned");
FOUNDATION_STATIC_ASSERT(sizeof(PCG32) == 16, "PCG32 must be 16 bytes");

#ifdef __cplusplus
}
#endif

#endif /* FOUNDATION_H */
