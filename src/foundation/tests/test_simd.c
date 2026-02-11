/**
 * SIMD Abstraction Layer Unit Tests
 *
 * Tests that the SIMD abstraction macros produce correct results
 * regardless of the underlying implementation (AVX2, NEON, or scalar).
 */

#include "../include/foundation.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "test_harness.h"

/* ============================================================================
 * Platform Detection Test
 * ============================================================================ */

TEST(simd_platform) {
#if defined(FOUNDATION_SIMD_AVX2)
    ASSERT_MSG(FOUNDATION_SIMD_WIDTH == 8, "AVX2 SIMD width is 8");
    ASSERT_MSG(FOUNDATION_SIMD_ALIGNMENT == 32, "AVX2 alignment is 32");
#elif defined(FOUNDATION_SIMD_NEON)
    ASSERT_MSG(FOUNDATION_SIMD_WIDTH == 4, "NEON SIMD width is 4");
    ASSERT_MSG(FOUNDATION_SIMD_ALIGNMENT == 16, "NEON alignment is 16");
#else
    ASSERT_MSG(FOUNDATION_SIMD_WIDTH == 1, "Scalar SIMD width is 1");
    ASSERT_MSG(FOUNDATION_SIMD_ALIGNMENT == 4, "Scalar alignment is 4");
#endif

    return 0;
}

/* ============================================================================
 * SIMD Operation Tests
 * ============================================================================ */

TEST(simd_load_store) {
    /* Allocate aligned memory */
    alignas(32) float src[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    alignas(32) float dst[8] = {0};

#if FOUNDATION_SIMD_WIDTH >= 8
    /* AVX2: load/store 8 floats */
    simd_float v = simd_load_ps(src);
    simd_store_ps(dst, v);

    for (int i = 0; i < 8; i++) {
        ASSERT_MSG(fabsf(dst[i] - src[i]) < 1e-4f, "AVX2 load/store");
    }
#elif FOUNDATION_SIMD_WIDTH == 4
    /* NEON: load/store 4 floats */
    simd_float v = simd_load_ps(src);
    simd_store_ps(dst, v);

    for (int i = 0; i < 4; i++) {
        ASSERT_MSG(fabsf(dst[i] - src[i]) < 1e-4f, "NEON load/store");
    }
#else
    /* Scalar: load/store 1 float */
    simd_float v = simd_load_ps(src);
    simd_store_ps(dst, v);
    ASSERT_MSG(fabsf(dst[0] - src[0]) < 1e-4f, "Scalar load/store");
#endif

    return 0;
}

TEST(simd_arithmetic) {
    alignas(32) float a_data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    alignas(32) float b_data[8] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    alignas(32) float result[8] = {0};

    /* Test addition */
#if FOUNDATION_SIMD_WIDTH >= 8
    {
        simd_float a = simd_load_ps(a_data);
        simd_float b = simd_load_ps(b_data);
        simd_float sum = simd_add_ps(a, b);
        simd_store_ps(result, sum);

        for (int i = 0; i < 8; i++) {
            ASSERT_MSG(fabsf(result[i] - (a_data[i] + b_data[i])) < 1e-4f, "SIMD add");
        }
    }
#elif FOUNDATION_SIMD_WIDTH == 4
    {
        simd_float a = simd_load_ps(a_data);
        simd_float b = simd_load_ps(b_data);
        simd_float sum = simd_add_ps(a, b);
        simd_store_ps(result, sum);

        for (int i = 0; i < 4; i++) {
            ASSERT_MSG(fabsf(result[i] - (a_data[i] + b_data[i])) < 1e-4f, "SIMD add");
        }
    }
#else
    {
        simd_float a = simd_load_ps(a_data);
        simd_float b = simd_load_ps(b_data);
        simd_float sum = simd_add_ps(a, b);
        simd_store_ps(result, sum);
        ASSERT_MSG(fabsf(result[0] - (a_data[0] + b_data[0])) < 1e-4f, "SIMD add");
    }
#endif

    /* Test subtraction */
    memset(result, 0, sizeof(result));
#if FOUNDATION_SIMD_WIDTH >= 4
    {
        simd_float a = simd_load_ps(a_data);
        simd_float b = simd_load_ps(b_data);
        simd_float diff = simd_sub_ps(a, b);
        simd_store_ps(result, diff);

        for (int i = 0; i < FOUNDATION_SIMD_WIDTH && i < 8; i++) {
            ASSERT_MSG(fabsf(result[i] - (a_data[i] - b_data[i])) < 1e-4f, "SIMD sub");
        }
    }
#endif

    /* Test multiplication */
    memset(result, 0, sizeof(result));
#if FOUNDATION_SIMD_WIDTH >= 4
    {
        simd_float a = simd_load_ps(a_data);
        simd_float b = simd_load_ps(b_data);
        simd_float prod = simd_mul_ps(a, b);
        simd_store_ps(result, prod);

        for (int i = 0; i < FOUNDATION_SIMD_WIDTH && i < 8; i++) {
            ASSERT_MSG(fabsf(result[i] - (a_data[i] * b_data[i])) < 1e-4f, "SIMD mul");
        }
    }
#endif

    return 0;
}

TEST(simd_set1) {
    alignas(32) float result[8] = {0};
    float value = 3.14159f;

#if FOUNDATION_SIMD_WIDTH >= 8
    {
        simd_float v = simd_set1_ps(value);
        simd_store_ps(result, v);

        for (int i = 0; i < 8; i++) {
            ASSERT_MSG(fabsf(result[i] - value) < 1e-4f, "SIMD set1 AVX2");
        }
    }
#elif FOUNDATION_SIMD_WIDTH == 4
    {
        simd_float v = simd_set1_ps(value);
        simd_store_ps(result, v);

        for (int i = 0; i < 4; i++) {
            ASSERT_MSG(fabsf(result[i] - value) < 1e-4f, "SIMD set1 NEON");
        }
    }
#else
    {
        simd_float v = simd_set1_ps(value);
        simd_store_ps(result, v);
        ASSERT_MSG(fabsf(result[0] - value) < 1e-4f, "SIMD set1 scalar");
    }
#endif

    return 0;
}

TEST(simd_fmadd) {
    alignas(32) float a_data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    alignas(32) float b_data[8] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    alignas(32) float c_data[8] = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
    alignas(32) float result[8] = {0};

    /* a * b + c */
#if FOUNDATION_SIMD_WIDTH >= 4
    {
        simd_float a = simd_load_ps(a_data);
        simd_float b = simd_load_ps(b_data);
        simd_float c = simd_load_ps(c_data);
        simd_float fma = simd_fmadd_ps(a, b, c);
        simd_store_ps(result, fma);

        for (int i = 0; i < FOUNDATION_SIMD_WIDTH && i < 8; i++) {
            float expected = a_data[i] * b_data[i] + c_data[i];
            ASSERT_MSG(fabsf(result[i] - expected) < 1e-4f, "SIMD fmadd");
        }
    }
#else
    {
        simd_float a = simd_load_ps(a_data);
        simd_float b = simd_load_ps(b_data);
        simd_float c = simd_load_ps(c_data);
        simd_float fma = simd_fmadd_ps(a, b, c);
        simd_store_ps(result, fma);
        ASSERT_MSG(fabsf(result[0] - (a_data[0] * b_data[0] + c_data[0])) < 1e-4f, "SIMD fmadd");
    }
#endif

    return 0;
}

TEST(simd_sqrt_rsqrt) {
    alignas(32) float src[8] = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f};
    alignas(32) float result[8] = {0};

    /* sqrt */
#if FOUNDATION_SIMD_WIDTH >= 4
    {
        simd_float v = simd_load_ps(src);
        simd_float sq = simd_sqrt_ps(v);
        simd_store_ps(result, sq);

        for (int i = 0; i < FOUNDATION_SIMD_WIDTH && i < 8; i++) {
            ASSERT_MSG(fabsf(result[i] - sqrtf(src[i])) < 1e-4f, "SIMD sqrt");
        }
    }
#else
    {
        simd_float v = simd_load_ps(src);
        simd_float sq = simd_sqrt_ps(v);
        simd_store_ps(result, sq);
        ASSERT_MSG(fabsf(result[0] - sqrtf(src[0])) < 1e-4f, "SIMD sqrt");
    }
#endif

    /* rsqrt (approximate) */
#if FOUNDATION_SIMD_WIDTH >= 4
    {
        simd_float v = simd_load_ps(src);
        simd_float rsq = simd_rsqrt_ps(v);
        simd_store_ps(result, rsq);

        for (int i = 0; i < FOUNDATION_SIMD_WIDTH && i < 8; i++) {
            float expected = 1.0f / sqrtf(src[i]);
            /* rsqrt is approximate, allow more tolerance */
            ASSERT_MSG(fabsf(result[i] - expected) < 0.01f, "SIMD rsqrt (approximate)");
        }
    }
#endif

    return 0;
}

TEST(simd_minmax) {
    alignas(32) float a_data[8] = {1.0f, 5.0f, 3.0f, 7.0f, 2.0f, 6.0f, 4.0f, 8.0f};
    alignas(32) float b_data[8] = {4.0f, 2.0f, 6.0f, 3.0f, 8.0f, 1.0f, 5.0f, 0.0f};
    alignas(32) float result[8] = {0};

#if FOUNDATION_SIMD_WIDTH >= 4
    /* min */
    {
        simd_float a = simd_load_ps(a_data);
        simd_float b = simd_load_ps(b_data);
        simd_float min_v = simd_min_ps(a, b);
        simd_store_ps(result, min_v);

        for (int i = 0; i < FOUNDATION_SIMD_WIDTH && i < 8; i++) {
            float expected = a_data[i] < b_data[i] ? a_data[i] : b_data[i];
            ASSERT_MSG(fabsf(result[i] - expected) < 1e-4f, "SIMD min");
        }
    }

    /* max */
    {
        simd_float a = simd_load_ps(a_data);
        simd_float b = simd_load_ps(b_data);
        simd_float max_v = simd_max_ps(a, b);
        simd_store_ps(result, max_v);

        for (int i = 0; i < FOUNDATION_SIMD_WIDTH && i < 8; i++) {
            float expected = a_data[i] > b_data[i] ? a_data[i] : b_data[i];
            ASSERT_MSG(fabsf(result[i] - expected) < 1e-4f, "SIMD max");
        }
    }
#endif

    return 0;
}

TEST(simd_loop_macros) {
    /* Test SIMD_LOOP_START */
    uint32_t count = 17;  /* Not divisible by SIMD_WIDTH */
    SIMD_LOOP_START(count);

    /* _simd_count should be largest multiple of SIMD_WIDTH <= count */
    uint32_t expected_simd_count = count & ~(FOUNDATION_SIMD_WIDTH - 1);
    ASSERT_MSG(_simd_count == expected_simd_count, "SIMD_LOOP_START calculation");

    /* Test that we can process remainder */
    int remainder_count = 0;
    SIMD_LOOP_REMAINDER(i, count) {
        remainder_count++;
    }

    uint32_t expected_remainder = count - expected_simd_count;
    ASSERT_MSG((uint32_t)remainder_count == expected_remainder, "SIMD_LOOP_REMAINDER count");

    return 0;
}

TEST(simd_practical_usage) {
    /* Example: Sum an array of floats using SIMD */
    const uint32_t count = 1000;
    alignas(32) float data[1000];

    /* Initialize with known values */
    for (uint32_t i = 0; i < count; i++) {
        data[i] = (float)i;
    }

    /* Expected sum: 0 + 1 + 2 + ... + 999 = 999 * 1000 / 2 = 499500 */
    float expected_sum = 499500.0f;

    /* SIMD sum */
    float simd_sum = 0.0f;

#if FOUNDATION_SIMD_WIDTH >= 4
    simd_float acc = simd_setzero_ps();
    SIMD_LOOP_START(count);

    for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
        simd_float v = simd_load_ps(&data[i]);
        acc = simd_add_ps(acc, v);
    }

    /* Reduce accumulator */
    alignas(32) float acc_arr[8];
    simd_store_ps(acc_arr, acc);
    for (int i = 0; i < FOUNDATION_SIMD_WIDTH; i++) {
        simd_sum += acc_arr[i];
    }

    /* Handle remainder */
    SIMD_LOOP_REMAINDER(i, count) {
        simd_sum += data[i];
    }
#else
    for (uint32_t i = 0; i < count; i++) {
        simd_sum += data[i];
    }
#endif

    ASSERT_MSG(fabsf(simd_sum - expected_sum) < 1e-4f, "SIMD array sum");

    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("SIMD Abstraction Layer Tests");

    RUN_TEST(simd_platform);
    RUN_TEST(simd_load_store);
    RUN_TEST(simd_arithmetic);
    RUN_TEST(simd_set1);
    RUN_TEST(simd_fmadd);
    RUN_TEST(simd_sqrt_rsqrt);
    RUN_TEST(simd_minmax);
    RUN_TEST(simd_loop_macros);
    RUN_TEST(simd_practical_usage);

    TEST_SUITE_END();
}
