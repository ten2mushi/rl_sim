/**
 * Composable Noise Pipeline Tests
 *
 * Tests for all noise types, pipeline composition, channel groups,
 * per-drone independence, and determinism guarantees.
 */

#include "test_harness.h"
#include "noise.h"
#include <math.h>
#include <string.h>

/* ============================================================================
 * Test Helpers
 * ============================================================================ */

#define NUM_SAMPLES 2000
#define OUTPUT_SIZE 6

static Arena* test_arena;

static void setup(void) {
    test_arena = arena_create(1024 * 1024);
}

static void teardown(void) {
    if (test_arena) arena_destroy(test_arena);
    test_arena = NULL;
}

/** Fill data buffer with a known pattern */
static void fill_data(float* data, uint32_t drone_count, uint32_t output_size) {
    for (uint32_t d = 0; d < drone_count; d++) {
        for (uint32_t c = 0; c < output_size; c++) {
            data[d * output_size + c] = 1.0f + (float)c * 0.1f;
        }
    }
}

/* ============================================================================
 * White Gaussian Tests
 * ============================================================================ */

TEST(white_gaussian_mean_stddev) {
    setup();

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].channel_count = 0;  /* all channels */
    config.groups[0].stages[0].type = NOISE_WHITE_GAUSSIAN;
    config.groups[0].stages[0].params.white.stddev = 0.5f;

    NoiseState* state = noise_state_create(test_arena, &config, 1, 0);
    ASSERT_NOT_NULL(state);

    /* Accumulate noise by applying to zero-initialized data */
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int total = NUM_SAMPLES;

    for (int i = 0; i < total; i++) {
        float data[1] = {0.0f};
        uint32_t idx = 0;
        noise_apply(&config, state, data, &idx, 1, 1, 0.02f);
        sum += data[0];
        sum_sq += data[0] * data[0];
    }

    float mean = sum / (float)total;
    float variance = sum_sq / (float)total - mean * mean;
    float stddev = sqrtf(variance);

    /* Mean should be near 0 */
    ASSERT_TRUE(fabsf(mean) < 0.1f);
    /* Stddev should be near configured 0.5 */
    ASSERT_TRUE(fabsf(stddev - 0.5f) < 0.15f);

    teardown();
    return 0;
}

/* ============================================================================
 * Constant Bias Tests
 * ============================================================================ */

TEST(constant_bias_exact) {
    setup();

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].channel_count = 3;
    config.groups[0].stages[0].type = NOISE_CONSTANT_BIAS;
    config.groups[0].stages[0].params.bias.values[0] = 0.1f;
    config.groups[0].stages[0].params.bias.values[1] = -0.2f;
    config.groups[0].stages[0].params.bias.values[2] = 0.3f;
    config.groups[0].stages[0].params.bias.count = 3;

    float data[OUTPUT_SIZE];
    fill_data(data, 1, OUTPUT_SIZE);
    float orig[OUTPUT_SIZE];
    memcpy(orig, data, sizeof(data));

    noise_apply(&config, NULL, data, NULL, 1, OUTPUT_SIZE, 0.02f);

    ASSERT_FLOAT_NEAR(data[0], orig[0] + 0.1f, 1e-6f);
    ASSERT_FLOAT_NEAR(data[1], orig[1] - 0.2f, 1e-6f);
    ASSERT_FLOAT_NEAR(data[2], orig[2] + 0.3f, 1e-6f);
    /* Channels beyond group range should be untouched */
    ASSERT_FLOAT_EQ(data[3], orig[3]);
    ASSERT_FLOAT_EQ(data[4], orig[4]);
    ASSERT_FLOAT_EQ(data[5], orig[5]);

    teardown();
    return 0;
}

/* ============================================================================
 * Bias Drift Tests
 * ============================================================================ */

TEST(bias_drift_mean_reversion) {
    setup();

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].channel_count = 1;
    config.groups[0].stages[0].type = NOISE_BIAS_DRIFT;
    config.groups[0].stages[0].params.drift.tau = 1.0f;    /* 1s time constant */
    config.groups[0].stages[0].params.drift.sigma = 0.01f;  /* small diffusion */

    NoiseState* state = noise_state_create(test_arena, &config, 1, 0);
    ASSERT_NOT_NULL(state);

    /* Manually push drift state far from zero */
    state->groups[0].drift_state[0] = 5.0f;

    /* Run many steps — drift should revert toward 0 */
    float dt = 0.02f;
    for (int i = 0; i < 500; i++) {
        float data[1] = {0.0f};
        uint32_t idx = 0;
        noise_apply(&config, state, data, &idx, 1, 1, dt);
    }

    /* After 10 seconds (500 * 0.02), drift should be much closer to 0 */
    float drift_val = state->groups[0].drift_state[0];
    ASSERT_TRUE(fabsf(drift_val) < 1.0f);

    teardown();
    return 0;
}

/* ============================================================================
 * Scale Factor Tests
 * ============================================================================ */

TEST(scale_factor_multiplicative) {
    setup();

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].channel_count = 0;  /* all */
    config.groups[0].stages[0].type = NOISE_SCALE_FACTOR;
    config.groups[0].stages[0].params.scale.error = 0.05f;

    float data[3] = {10.0f, 20.0f, 30.0f};
    noise_apply(&config, NULL, data, NULL, 1, 3, 0.02f);

    ASSERT_FLOAT_NEAR(data[0], 10.5f, 1e-5f);
    ASSERT_FLOAT_NEAR(data[1], 21.0f, 1e-5f);
    ASSERT_FLOAT_NEAR(data[2], 31.5f, 1e-5f);

    teardown();
    return 0;
}

/* ============================================================================
 * Distance-Dependent Tests
 * ============================================================================ */

TEST(distance_dependent_scaling) {
    setup();

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stages[0].type = NOISE_DISTANCE_DEPENDENT;
    config.groups[0].stages[0].params.distance.coeff = 0.1f;
    config.groups[0].stages[0].params.distance.power = 2.0f;

    NoiseState* state = noise_state_create(test_arena, &config, 1, 0);
    ASSERT_NOT_NULL(state);

    /* Measure noise variance at small vs large values */
    float sum_sq_small = 0.0f;
    float sum_sq_large = 0.0f;
    int N = NUM_SAMPLES;

    for (int i = 0; i < N; i++) {
        /* Reset to clean signal each iteration */
        float data_small[1] = {1.0f};  /* |v|=1 → stddev = 0.1*1 = 0.1 */
        float data_large[1] = {10.0f}; /* |v|=10 → stddev = 0.1*100 = 10.0 */
        uint32_t idx = 0;
        noise_apply(&config, state, data_small, &idx, 1, 1, 0.02f);
        noise_apply(&config, state, data_large, &idx, 1, 1, 0.02f);

        float noise_small = data_small[0] - 1.0f;
        float noise_large = data_large[0] - 10.0f;
        sum_sq_small += noise_small * noise_small;
        sum_sq_large += noise_large * noise_large;
    }

    float var_small = sum_sq_small / (float)N;
    float var_large = sum_sq_large / (float)N;

    /* Large values should produce much more variance (ratio ~10000x) */
    ASSERT_TRUE(var_large > var_small * 100.0f);

    teardown();
    return 0;
}

/* ============================================================================
 * Quantization Tests
 * ============================================================================ */

TEST(quantization_step) {
    setup();

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stages[0].type = NOISE_QUANTIZATION;
    config.groups[0].stages[0].params.quantize.step = 0.25f;

    float data[4] = {0.1f, 0.3f, 0.6f, -0.15f};
    noise_apply(&config, NULL, data, NULL, 1, 4, 0.02f);

    ASSERT_FLOAT_NEAR(data[0], 0.0f, 1e-6f);   /* 0.1 → 0.0 */
    ASSERT_FLOAT_NEAR(data[1], 0.25f, 1e-6f);   /* 0.3 → 0.25 */
    ASSERT_FLOAT_NEAR(data[2], 0.5f, 1e-6f);    /* 0.6 → 0.5 (nearest) or 0.75 */
    ASSERT_FLOAT_NEAR(data[3], -0.25f, 1e-6f);  /* -0.15 → -0.25 */

    teardown();
    return 0;
}

/* ============================================================================
 * Dropout Tests
 * ============================================================================ */

TEST(dropout_rate) {
    setup();

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stages[0].type = NOISE_DROPOUT;
    config.groups[0].stages[0].params.dropout.probability = 0.5f;
    config.groups[0].stages[0].params.dropout.replacement = 0.0f;

    NoiseState* state = noise_state_create(test_arena, &config, 1, 0);
    ASSERT_NOT_NULL(state);

    int dropped = 0;
    int total = NUM_SAMPLES;
    uint32_t idx = 0;

    for (int i = 0; i < total; i++) {
        float data[1] = {1.0f};
        noise_apply(&config, state, data, &idx, 1, 1, 0.02f);
        if (data[0] == 0.0f) dropped++;
    }

    /* Expect ~50% dropout (allow +-10%) */
    float rate = (float)dropped / (float)total;
    ASSERT_TRUE(rate > 0.35f);
    ASSERT_TRUE(rate < 0.65f);

    teardown();
    return 0;
}

/* ============================================================================
 * Saturation Tests
 * ============================================================================ */

TEST(saturation_clamp) {
    setup();

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stages[0].type = NOISE_SATURATION;
    config.groups[0].stages[0].params.saturate.min_val = -1.0f;
    config.groups[0].stages[0].params.saturate.max_val = 1.0f;

    float data[4] = {-5.0f, 0.5f, 3.0f, -0.3f};
    noise_apply(&config, NULL, data, NULL, 1, 4, 0.02f);

    ASSERT_FLOAT_EQ(data[0], -1.0f);
    ASSERT_FLOAT_EQ(data[1], 0.5f);
    ASSERT_FLOAT_EQ(data[2], 1.0f);
    ASSERT_FLOAT_EQ(data[3], -0.3f);

    teardown();
    return 0;
}

/* ============================================================================
 * Pipeline Composition Tests
 * ============================================================================ */

TEST(pipeline_composition) {
    setup();

    /* 3-stage chain: constant_bias → scale_factor → saturation */
    NoiseConfig config = {0};
    config.group_count = 1;
    NoisePipeline* pipe = &config.groups[0];
    pipe->stage_count = 3;
    pipe->channel_count = 0;

    /* Stage 1: add 1.0 bias */
    pipe->stages[0].type = NOISE_CONSTANT_BIAS;
    pipe->stages[0].params.bias.values[0] = 1.0f;
    pipe->stages[0].params.bias.count = 1;

    /* Stage 2: scale by 1.5x (error=0.5) */
    pipe->stages[1].type = NOISE_SCALE_FACTOR;
    pipe->stages[1].params.scale.error = 0.5f;

    /* Stage 3: clamp to [0, 5] */
    pipe->stages[2].type = NOISE_SATURATION;
    pipe->stages[2].params.saturate.min_val = 0.0f;
    pipe->stages[2].params.saturate.max_val = 5.0f;

    /* Input: 2.0 → +1.0 = 3.0 → *1.5 = 4.5 → clamp = 4.5 */
    float data1[1] = {2.0f};
    noise_apply(&config, NULL, data1, NULL, 1, 1, 0.02f);
    ASSERT_FLOAT_NEAR(data1[0], 4.5f, 1e-5f);

    /* Input: 4.0 → +1.0 = 5.0 → *1.5 = 7.5 → clamp = 5.0 */
    float data2[1] = {4.0f};
    noise_apply(&config, NULL, data2, NULL, 1, 1, 0.02f);
    ASSERT_FLOAT_NEAR(data2[0], 5.0f, 1e-5f);

    teardown();
    return 0;
}

/* ============================================================================
 * Channel Group Tests
 * ============================================================================ */

TEST(channel_groups) {
    setup();

    /* Two groups: different noise on channels [0,3) vs [3,6) */
    NoiseConfig config = {0};
    config.group_count = 2;

    /* Group 0: channels [0,3) — add bias 0.1 */
    config.groups[0].channel_start = 0;
    config.groups[0].channel_count = 3;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_CONSTANT_BIAS;
    config.groups[0].stages[0].params.bias.values[0] = 0.1f;
    config.groups[0].stages[0].params.bias.count = 1;

    /* Group 1: channels [3,6) — add bias -0.5 */
    config.groups[1].channel_start = 3;
    config.groups[1].channel_count = 3;
    config.groups[1].stage_count = 1;
    config.groups[1].stages[0].type = NOISE_CONSTANT_BIAS;
    config.groups[1].stages[0].params.bias.values[0] = -0.5f;
    config.groups[1].stages[0].params.bias.count = 1;

    float data[OUTPUT_SIZE];
    fill_data(data, 1, OUTPUT_SIZE);
    float orig[OUTPUT_SIZE];
    memcpy(orig, data, sizeof(data));

    noise_apply(&config, NULL, data, NULL, 1, OUTPUT_SIZE, 0.02f);

    /* Group 0: channels 0-2 shifted by +0.1 */
    ASSERT_FLOAT_NEAR(data[0], orig[0] + 0.1f, 1e-6f);
    ASSERT_FLOAT_NEAR(data[1], orig[1] + 0.1f, 1e-6f);
    ASSERT_FLOAT_NEAR(data[2], orig[2] + 0.1f, 1e-6f);

    /* Group 1: channels 3-5 shifted by -0.5 */
    ASSERT_FLOAT_NEAR(data[3], orig[3] - 0.5f, 1e-6f);
    ASSERT_FLOAT_NEAR(data[4], orig[4] - 0.5f, 1e-6f);
    ASSERT_FLOAT_NEAR(data[5], orig[5] - 0.5f, 1e-6f);

    teardown();
    return 0;
}

/* ============================================================================
 * Per-Drone Independence Tests
 * ============================================================================ */

TEST(per_drone_independence) {
    setup();

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stages[0].type = NOISE_WHITE_GAUSSIAN;
    config.groups[0].stages[0].params.white.stddev = 1.0f;

    NoiseState* state = noise_state_create(test_arena, &config, 2, 0);
    ASSERT_NOT_NULL(state);

    /* Two drones, same input */
    float data[2] = {0.0f, 0.0f};
    uint32_t indices[2] = {0, 1};
    noise_apply(&config, state, data, indices, 2, 1, 0.02f);

    /* Drones should get different noise values */
    ASSERT_TRUE(fabsf(data[0] - data[1]) > 1e-6f);

    teardown();
    return 0;
}

/* ============================================================================
 * Determinism Tests
 * ============================================================================ */

TEST(determinism_same_seed) {
    setup();

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stages[0].type = NOISE_WHITE_GAUSSIAN;
    config.groups[0].stages[0].params.white.stddev = 1.0f;

    /* Run 1 */
    NoiseState* state1 = noise_state_create(test_arena, &config, 1, 42);
    float data1[OUTPUT_SIZE];
    fill_data(data1, 1, OUTPUT_SIZE);
    uint32_t idx = 0;
    noise_apply(&config, state1, data1, &idx, 1, OUTPUT_SIZE, 0.02f);

    /* Run 2 — fresh arena, same sensor_id */
    Arena* arena2 = arena_create(1024 * 1024);
    NoiseState* state2 = noise_state_create(arena2, &config, 1, 42);
    float data2[OUTPUT_SIZE];
    fill_data(data2, 1, OUTPUT_SIZE);
    noise_apply(&config, state2, data2, &idx, 1, OUTPUT_SIZE, 0.02f);

    /* Results must be identical */
    for (int c = 0; c < OUTPUT_SIZE; c++) {
        ASSERT_FLOAT_EQ(data1[c], data2[c]);
    }

    arena_destroy(arena2);
    teardown();
    return 0;
}

TEST(determinism_across_reset) {
    setup();

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stages[0].type = NOISE_WHITE_GAUSSIAN;
    config.groups[0].stages[0].params.white.stddev = 1.0f;

    NoiseState* state = noise_state_create(test_arena, &config, 1, 0);
    uint32_t idx = 0;

    /* Run 1: sample some noise */
    float data1[OUTPUT_SIZE];
    fill_data(data1, 1, OUTPUT_SIZE);
    noise_apply(&config, state, data1, &idx, 1, OUTPUT_SIZE, 0.02f);

    /* Reset drone 0 */
    noise_state_reset_drone(state, 0);

    /* Run 2: sample again after reset — should get same noise */
    float data2[OUTPUT_SIZE];
    fill_data(data2, 1, OUTPUT_SIZE);
    noise_apply(&config, state, data2, &idx, 1, OUTPUT_SIZE, 0.02f);

    for (int c = 0; c < OUTPUT_SIZE; c++) {
        ASSERT_FLOAT_EQ(data1[c], data2[c]);
    }

    teardown();
    return 0;
}

/* ============================================================================
 * No-Noise Passthrough Test
 * ============================================================================ */

TEST(no_noise_clean_signal) {
    setup();

    NoiseConfig config = {0};
    config.group_count = 0;  /* No noise */

    float data[OUTPUT_SIZE];
    fill_data(data, 1, OUTPUT_SIZE);
    float orig[OUTPUT_SIZE];
    memcpy(orig, data, sizeof(data));

    noise_apply(&config, NULL, data, NULL, 1, OUTPUT_SIZE, 0.02f);

    /* Output must be unchanged */
    for (int c = 0; c < OUTPUT_SIZE; c++) {
        ASSERT_FLOAT_EQ(data[c], orig[c]);
    }

    teardown();
    return 0;
}

/* ============================================================================
 * Main Test Runner
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Noise Pipeline");

    /* Primitive noise types */
    RUN_TEST(white_gaussian_mean_stddev);
    RUN_TEST(constant_bias_exact);
    RUN_TEST(bias_drift_mean_reversion);
    RUN_TEST(scale_factor_multiplicative);
    RUN_TEST(distance_dependent_scaling);
    RUN_TEST(quantization_step);
    RUN_TEST(dropout_rate);
    RUN_TEST(saturation_clamp);

    /* Composition and grouping */
    RUN_TEST(pipeline_composition);
    RUN_TEST(channel_groups);

    /* Per-drone behavior */
    RUN_TEST(per_drone_independence);

    /* Determinism guarantees */
    RUN_TEST(determinism_same_seed);
    RUN_TEST(determinism_across_reset);

    /* Passthrough */
    RUN_TEST(no_noise_clean_signal);

    TEST_SUITE_END();
}
