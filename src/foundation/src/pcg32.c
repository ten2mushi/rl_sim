/**
 * PCG32 Random Number Generator Implementation
 *
 * PCG (Permuted Congruential Generator) is a family of simple, fast,
 * space-efficient, statistically excellent random number generators.
 *
 * Reference: https://www.pcg-random.org/
 */

#include "../include/foundation.h"

#if defined(__APPLE__) || defined(__linux__)
#include <time.h>
#endif

/* PCG constants */
#define PCG32_MULT  6364136223846793005ULL
#define PCG32_INC   1442695040888963407ULL

/* For converting to float [0, 1) */
#define PCG32_FLOAT_SCALE (1.0f / 4294967296.0f)

/* Thread-local RNG instance */
static _Thread_local PCG32 tls_rng = {0, 0};
static _Thread_local bool tls_rng_initialized = false;

/**
 * Seed the RNG with a single seed value.
 *
 * Uses a default increment value.
 *
 * @param rng The RNG state to seed
 * @param seed The seed value
 */
void pcg32_seed(PCG32* rng, uint64_t seed) {
    pcg32_seed_dual(rng, seed, PCG32_INC);
}

/**
 * Seed the RNG with both seed and sequence values.
 *
 * Different sequences produce entirely different streams even with
 * the same seed. The sequence must be odd (will be made odd if even).
 *
 * @param rng The RNG state to seed
 * @param seed The seed value
 * @param seq The sequence selector
 */
void pcg32_seed_dual(PCG32* rng, uint64_t seed, uint64_t seq) {
    rng->state = 0U;
    rng->inc = (seq << 1U) | 1U;  /* Ensure increment is odd */
    pcg32_random(rng);
    rng->state += seed;
    pcg32_random(rng);
}

/**
 * Generate a random 32-bit unsigned integer.
 *
 * This is the core PCG32 algorithm.
 *
 * @param rng The RNG state
 * @return A random 32-bit value
 */
uint32_t pcg32_random(PCG32* rng) {
    uint64_t oldstate = rng->state;

    /* Advance internal state */
    rng->state = oldstate * PCG32_MULT + rng->inc;

    /* Calculate output function (XSH-RR) */
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18U) ^ oldstate) >> 27U);
    uint32_t rot = (uint32_t)(oldstate >> 59U);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

/**
 * Generate a random integer in [0, bound).
 *
 * Uses rejection sampling to eliminate modulo bias.
 *
 * @param rng The RNG state
 * @param bound The exclusive upper bound
 * @return A random value in [0, bound)
 */
uint32_t pcg32_bounded(PCG32* rng, uint32_t bound) {
    if (bound == 0) {
        return 0;
    }

    /* Calculate threshold for rejection sampling to eliminate bias */
    uint32_t threshold = (-bound) % bound;

    /* Loop until we get a value above the threshold */
    for (;;) {
        uint32_t r = pcg32_random(rng);
        if (r >= threshold) {
            return r % bound;
        }
    }
}

/**
 * Generate a random float in [0, 1).
 *
 * Uses all 32 bits of randomness for maximum precision.
 *
 * @param rng The RNG state
 * @return A random float in [0, 1)
 */
float pcg32_float(PCG32* rng) {
    return (float)pcg32_random(rng) * PCG32_FLOAT_SCALE;
}

/**
 * Generate a random float in [min, max).
 *
 * @param rng The RNG state
 * @param min The minimum value (inclusive)
 * @param max The maximum value (exclusive)
 * @return A random float in [min, max)
 */
float pcg32_range(PCG32* rng, float min, float max) {
    return min + pcg32_float(rng) * (max - min);
}

/**
 * Generate a random unit vector on the sphere surface.
 *
 * Uses Marsaglia's method for uniform distribution on sphere.
 *
 * @param rng The RNG state
 * @return A random unit vector
 */
Vec3 pcg32_vec3_unit(PCG32* rng) {
    /* Marsaglia's method: generate two uniform randoms in [-1, 1],
     * reject if x^2 + y^2 >= 1, then project to sphere */
    float x, y, s;

    do {
        x = pcg32_range(rng, -1.0f, 1.0f);
        y = pcg32_range(rng, -1.0f, 1.0f);
        s = x * x + y * y;
    } while (s >= 1.0f || s < 1e-10f);

    float z = 1.0f - 2.0f * s;
    float r = 2.0f * sqrtf(1.0f - s);

    return VEC3(x * r, y * r, z);
}

/**
 * Generate a random vector with components in the given range.
 *
 * Each component is independently sampled from its range.
 *
 * @param rng The RNG state
 * @param min The minimum values for each component
 * @param max The maximum values for each component
 * @return A random vector with components in [min, max)
 */
Vec3 pcg32_vec3_range(PCG32* rng, Vec3 min, Vec3 max) {
    return VEC3(
        pcg32_range(rng, min.x, max.x),
        pcg32_range(rng, min.y, max.y),
        pcg32_range(rng, min.z, max.z)
    );
}

/**
 * Get the thread-local RNG instance.
 *
 * Lazily initializes with a seed derived from the thread ID and
 * a high-resolution timestamp if available.
 *
 * @return Pointer to the thread-local RNG
 */
PCG32* pcg32_thread_local(void) {
    if (!tls_rng_initialized) {
        /* Generate a seed from available entropy sources */
        /* Using pointer address and a simple hash */
        uintptr_t ptr_entropy = (uintptr_t)&tls_rng;
        uint64_t seed = ptr_entropy ^ 0x853C49E6748FEA9BULL;  /* Golden ratio prime */

        /* Mix in more entropy if we have clock */
#if defined(__APPLE__) || defined(__linux__)
        struct timespec ts;
        if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
            seed ^= (uint64_t)ts.tv_nsec;
            seed ^= (uint64_t)ts.tv_sec << 32;
        }
#endif

        pcg32_seed(&tls_rng, seed);
        tls_rng_initialized = true;
    }

    return &tls_rng;
}
