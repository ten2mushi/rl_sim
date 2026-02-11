/**
 * Unified Test Harness
 *
 * Provides shared macros for all test binaries in rl_engine.
 * Pattern: int-returning test functions (0 = pass, -__LINE__ = fail).
 *
 * Usage:
 *   #include "test_harness.h"
 *
 *   TEST(my_feature) {
 *       ASSERT_TRUE(1 + 1 == 2);
 *       ASSERT_EQ(42, 42);
 *       return 0;
 *   }
 *
 *   int main(void) {
 *       TEST_SUITE_BEGIN("My Module");
 *       RUN_TEST(my_feature);
 *       TEST_SUITE_END();
 *   }
 */

#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ============================================================================
 * Counters
 * ============================================================================ */

static int _th_run = 0;
static int _th_passed = 0;

/* ============================================================================
 * Test Declaration
 * ============================================================================ */

#define TEST(name) static int test_##name(void)

/* ============================================================================
 * Test Runner
 * ============================================================================ */

#define RUN_TEST(name) do { \
    _th_run++; \
    printf("  Running %s...", #name); \
    fflush(stdout); \
    int _th_rc = test_##name(); \
    if (_th_rc == 0) { \
        _th_passed++; \
        printf(" PASSED\n"); \
    } else { \
        printf(" FAILED (line %d)\n", -_th_rc); \
    } \
} while(0)

/* ============================================================================
 * Suite Bookends
 * ============================================================================ */

#define TEST_SUITE_BEGIN(name) \
    printf("\n=== %s ===\n\n", name)

#define TEST_SUITE_END() do { \
    printf("\n===========================\n"); \
    printf("Tests: %d/%d passed\n", _th_passed, _th_run); \
    return (_th_passed == _th_run) ? 0 : 1; \
} while(0)

/* ============================================================================
 * Assertions — all return -__LINE__ on failure
 * ============================================================================ */

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        printf("\n    ASSERT_TRUE failed: %s\n    at %s:%d", \
               #cond, __FILE__, __LINE__); \
        return -__LINE__; \
    } \
} while(0)

#define ASSERT_FALSE(cond) do { \
    if (cond) { \
        printf("\n    ASSERT_FALSE failed: %s\n    at %s:%d", \
               #cond, __FILE__, __LINE__); \
        return -__LINE__; \
    } \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        printf("\n    ASSERT_EQ failed: %s != %s\n    at %s:%d", \
               #a, #b, __FILE__, __LINE__); \
        return -__LINE__; \
    } \
} while(0)

#define ASSERT_NE(a, b) do { \
    if ((a) == (b)) { \
        printf("\n    ASSERT_NE failed: %s == %s\n    at %s:%d", \
               #a, #b, __FILE__, __LINE__); \
        return -__LINE__; \
    } \
} while(0)

#define ASSERT_GT(a, b) do { \
    if (!((a) > (b))) { \
        printf("\n    ASSERT_GT failed: %s <= %s\n    at %s:%d", \
               #a, #b, __FILE__, __LINE__); \
        return -__LINE__; \
    } \
} while(0)

#define ASSERT_GE(a, b) do { \
    if (!((a) >= (b))) { \
        printf("\n    ASSERT_GE failed: %s < %s\n    at %s:%d", \
               #a, #b, __FILE__, __LINE__); \
        return -__LINE__; \
    } \
} while(0)

#define ASSERT_LT(a, b) do { \
    if (!((a) < (b))) { \
        printf("\n    ASSERT_LT failed: %s >= %s\n    at %s:%d", \
               #a, #b, __FILE__, __LINE__); \
        return -__LINE__; \
    } \
} while(0)

#define ASSERT_LE(a, b) do { \
    if (!((a) <= (b))) { \
        printf("\n    ASSERT_LE failed: %s > %s\n    at %s:%d", \
               #a, #b, __FILE__, __LINE__); \
        return -__LINE__; \
    } \
} while(0)

#define ASSERT_NULL(ptr) do { \
    if ((ptr) != NULL) { \
        printf("\n    ASSERT_NULL failed: %s\n    at %s:%d", \
               #ptr, __FILE__, __LINE__); \
        return -__LINE__; \
    } \
} while(0)

#define ASSERT_NOT_NULL(ptr) do { \
    if ((ptr) == NULL) { \
        printf("\n    ASSERT_NOT_NULL failed: %s\n    at %s:%d", \
               #ptr, __FILE__, __LINE__); \
        return -__LINE__; \
    } \
} while(0)

#define ASSERT_STR_EQ(a, b) do { \
    if (strcmp((a), (b)) != 0) { \
        printf("\n    ASSERT_STR_EQ failed: \"%s\" != \"%s\"\n    at %s:%d", \
               (a), (b), __FILE__, __LINE__); \
        return -__LINE__; \
    } \
} while(0)

#define ASSERT_FLOAT_EQ(a, b) do { \
    if (fabsf((float)(a) - (float)(b)) > 1e-5f) { \
        printf("\n    ASSERT_FLOAT_EQ failed: %s=%g != %s=%g\n    at %s:%d", \
               #a, (double)(a), #b, (double)(b), __FILE__, __LINE__); \
        return -__LINE__; \
    } \
} while(0)

#define ASSERT_FLOAT_NEAR(a, b, tol) do { \
    if (fabsf((float)(a) - (float)(b)) > (tol)) { \
        printf("\n    ASSERT_FLOAT_NEAR failed: %s=%g != %s=%g (tol=%g)\n    at %s:%d", \
               #a, (double)(a), #b, (double)(b), (double)(tol), __FILE__, __LINE__); \
        return -__LINE__; \
    } \
} while(0)

#define ASSERT_VEC3_NEAR(a, b, tol) do { \
    if (fabsf((a).x - (b).x) > (tol) || \
        fabsf((a).y - (b).y) > (tol) || \
        fabsf((a).z - (b).z) > (tol)) { \
        printf("\n    ASSERT_VEC3_NEAR failed: (%g,%g,%g) != (%g,%g,%g) (tol=%g)\n    at %s:%d", \
               (double)(a).x, (double)(a).y, (double)(a).z, \
               (double)(b).x, (double)(b).y, (double)(b).z, \
               (double)(tol), __FILE__, __LINE__); \
        return -__LINE__; \
    } \
} while(0)

#define ASSERT_MSG(cond, msg) do { \
    if (!(cond)) { \
        printf("\n    FAIL: %s\n    at %s:%d\n    %s", \
               msg, __FILE__, __LINE__, #cond); \
        return -__LINE__; \
    } \
} while(0)

#endif /* TEST_HARNESS_H */
