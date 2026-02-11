/**
 * PCG32 Random Number Generator Unit Tests
 *
 * Tests distribution uniformity, bounded random, float generation,
 * and unit vector generation.
 */

#include "../include/foundation.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "test_harness.h"

/* ============================================================================
 * Test Cases
 * ============================================================================ */

TEST(pcg32_seeding) {
    PCG32 rng1, rng2;

    /* Same seed should produce same sequence */
    pcg32_seed(&rng1, 12345);
    pcg32_seed(&rng2, 12345);

    for (int i = 0; i < 100; i++) {
        uint32_t r1 = pcg32_random(&rng1);
        uint32_t r2 = pcg32_random(&rng2);
        ASSERT_MSG(r1 == r2, "same seed produces same sequence");
    }

    /* Different seeds should produce different sequences */
    pcg32_seed(&rng1, 12345);
    pcg32_seed(&rng2, 54321);

    int different_count = 0;
    for (int i = 0; i < 100; i++) {
        uint32_t r1 = pcg32_random(&rng1);
        uint32_t r2 = pcg32_random(&rng2);
        if (r1 != r2) different_count++;
    }
    ASSERT_MSG(different_count > 90, "different seeds produce different sequences");

    return 0;
}

TEST(pcg32_dual_seeding) {
    PCG32 rng1, rng2;

    /* Same seed, different stream */
    pcg32_seed_dual(&rng1, 12345, 1);
    pcg32_seed_dual(&rng2, 12345, 2);

    int different_count = 0;
    for (int i = 0; i < 100; i++) {
        uint32_t r1 = pcg32_random(&rng1);
        uint32_t r2 = pcg32_random(&rng2);
        if (r1 != r2) different_count++;
    }
    ASSERT_MSG(different_count > 90, "different streams produce different sequences");

    return 0;
}

TEST(pcg32_bounded) {
    PCG32 rng;
    pcg32_seed(&rng, 42);

    /* Test that all values are within bounds */
    uint32_t bounds[] = {1, 2, 10, 100, 1000, 65536, 1000000};
    size_t num_bounds = sizeof(bounds) / sizeof(bounds[0]);

    for (size_t b = 0; b < num_bounds; b++) {
        uint32_t bound = bounds[b];
        for (int i = 0; i < 1000; i++) {
            uint32_t r = pcg32_bounded(&rng, bound);
            ASSERT_MSG(r < bound, "bounded random is within bounds");
        }
    }

    /* Test distribution for small bound (chi-squared-like) */
    pcg32_seed(&rng, 42);
    uint32_t counts[10] = {0};
    int num_samples = 100000;

    for (int i = 0; i < num_samples; i++) {
        uint32_t r = pcg32_bounded(&rng, 10);
        counts[r]++;
    }

    /* Each bucket should have approximately num_samples/10 = 10000 */
    /* With normal deviation, expect each to be within 1000 of expected */
    float expected = (float)num_samples / 10.0f;
    int all_in_range = 1;
    for (int i = 0; i < 10; i++) {
        float deviation = fabsf((float)counts[i] - expected);
        if (deviation > expected * 0.1f) {  /* More than 10% deviation */
            all_in_range = 0;
        }
    }
    ASSERT_MSG(all_in_range, "bounded random has uniform distribution");

    return 0;
}

TEST(pcg32_float) {
    PCG32 rng;
    pcg32_seed(&rng, 42);

    /* Test that all values are in [0, 1) */
    for (int i = 0; i < 10000; i++) {
        float f = pcg32_float(&rng);
        ASSERT_MSG(f >= 0.0f && f < 1.0f, "float in [0, 1)");
    }

    /* Test distribution */
    pcg32_seed(&rng, 42);
    int buckets[10] = {0};
    int num_samples = 100000;

    for (int i = 0; i < num_samples; i++) {
        float f = pcg32_float(&rng);
        int bucket = (int)(f * 10.0f);
        if (bucket >= 10) bucket = 9;
        buckets[bucket]++;
    }

    /* Check distribution uniformity */
    float expected = (float)num_samples / 10.0f;
    int all_in_range = 1;
    for (int i = 0; i < 10; i++) {
        float deviation = fabsf((float)buckets[i] - expected);
        if (deviation > expected * 0.1f) {
            all_in_range = 0;
        }
    }
    ASSERT_MSG(all_in_range, "float has uniform distribution");

    return 0;
}

TEST(pcg32_range) {
    PCG32 rng;
    pcg32_seed(&rng, 42);

    /* Test various ranges */
    struct {
        float min, max;
    } ranges[] = {
        {0.0f, 1.0f},
        {-1.0f, 1.0f},
        {10.0f, 20.0f},
        {-100.0f, -50.0f},
        {0.0f, 1000.0f}
    };
    size_t num_ranges = sizeof(ranges) / sizeof(ranges[0]);

    for (size_t r = 0; r < num_ranges; r++) {
        float min = ranges[r].min;
        float max = ranges[r].max;

        for (int i = 0; i < 1000; i++) {
            float f = pcg32_range(&rng, min, max);
            ASSERT_MSG(f >= min && f < max, "range value within bounds");
        }
    }

    return 0;
}

TEST(pcg32_vec3_unit) {
    PCG32 rng;
    pcg32_seed(&rng, 42);

    /* All unit vectors should have length 1 */
    for (int i = 0; i < 1000; i++) {
        Vec3 v = pcg32_vec3_unit(&rng);
        float len = vec3_length(v);
        ASSERT_MSG(fabsf(len - 1.0f) < 0.001f, "unit vector has length 1");
    }

    /* Test distribution on sphere (statistical test) */
    /* Count vectors in each octant */
    pcg32_seed(&rng, 42);
    int octants[8] = {0};
    int num_samples = 10000;

    for (int i = 0; i < num_samples; i++) {
        Vec3 v = pcg32_vec3_unit(&rng);
        int idx = (v.x >= 0 ? 1 : 0) + (v.y >= 0 ? 2 : 0) + (v.z >= 0 ? 4 : 0);
        octants[idx]++;
    }

    /* Each octant should have approximately num_samples/8 vectors */
    float expected = (float)num_samples / 8.0f;
    int all_in_range = 1;
    for (int i = 0; i < 8; i++) {
        float deviation = fabsf((float)octants[i] - expected);
        if (deviation > expected * 0.15f) {  /* Allow 15% deviation */
            all_in_range = 0;
        }
    }
    ASSERT_MSG(all_in_range, "unit vectors uniformly distributed on sphere");

    return 0;
}

TEST(pcg32_vec3_range) {
    PCG32 rng;
    pcg32_seed(&rng, 42);

    Vec3 min = VEC3(-10.0f, 0.0f, 5.0f);
    Vec3 max = VEC3(10.0f, 20.0f, 100.0f);

    for (int i = 0; i < 1000; i++) {
        Vec3 v = pcg32_vec3_range(&rng, min, max);
        ASSERT_MSG(v.x >= min.x && v.x < max.x, "vec3 range x in bounds");
        ASSERT_MSG(v.y >= min.y && v.y < max.y, "vec3 range y in bounds");
        ASSERT_MSG(v.z >= min.z && v.z < max.z, "vec3 range z in bounds");
    }

    return 0;
}

TEST(pcg32_thread_local) {
    /* Get thread-local RNG */
    PCG32* tls_rng = pcg32_thread_local();
    ASSERT_MSG(tls_rng != NULL, "thread-local RNG not NULL");

    /* Should be the same pointer when called again */
    PCG32* tls_rng2 = pcg32_thread_local();
    ASSERT_MSG(tls_rng == tls_rng2, "thread-local RNG returns same pointer");

    /* Should produce valid random numbers */
    uint32_t r1 = pcg32_random(tls_rng);
    uint32_t r2 = pcg32_random(tls_rng);
    ASSERT_MSG(r1 != r2 || r1 == 0, "thread-local RNG produces values");

    return 0;
}

TEST(pcg32_chi_squared) {
    /* More rigorous uniformity test using chi-squared */
    PCG32 rng;
    pcg32_seed(&rng, 12345);

    const int num_buckets = 256;
    const int num_samples = 1000000;
    int buckets[256] = {0};

    /* Generate samples and count in buckets */
    for (int i = 0; i < num_samples; i++) {
        uint32_t r = pcg32_random(&rng);
        int bucket = (r >> 24) & 0xFF;  /* Use top 8 bits */
        buckets[bucket]++;
    }

    /* Calculate chi-squared statistic */
    float expected = (float)num_samples / (float)num_buckets;
    float chi_squared = 0.0f;
    for (int i = 0; i < num_buckets; i++) {
        float diff = (float)buckets[i] - expected;
        chi_squared += (diff * diff) / expected;
    }

    /* For 255 degrees of freedom, chi-squared should be less than ~300 */
    /* at 95% confidence level */
    ASSERT_MSG(chi_squared < 350.0f, "chi-squared test passed");

    return 0;
}

TEST(pcg32_period_partial) {
    /* We can't test the full period (2^64), but we can verify
     * that we get many unique values without repeating */
    PCG32 rng;
    pcg32_seed(&rng, 42);

    /* Record first value */
    uint64_t initial_state = rng.state;
    pcg32_random(&rng);

    /* Generate many values and verify state changes */
    int repetitions = 0;
    for (int i = 0; i < 1000000; i++) {
        pcg32_random(&rng);
        if (rng.state == initial_state) {
            repetitions++;
        }
    }

    ASSERT_MSG(repetitions == 0, "no state repetitions in 1M samples");

    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("PCG32 Random Number Generator Tests");

    RUN_TEST(pcg32_seeding);
    RUN_TEST(pcg32_dual_seeding);
    RUN_TEST(pcg32_bounded);
    RUN_TEST(pcg32_float);
    RUN_TEST(pcg32_range);
    RUN_TEST(pcg32_vec3_unit);
    RUN_TEST(pcg32_vec3_range);
    RUN_TEST(pcg32_thread_local);
    RUN_TEST(pcg32_chi_squared);
    RUN_TEST(pcg32_period_partial);

    TEST_SUITE_END();
}
