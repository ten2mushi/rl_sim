/**
 * Noise Pipeline Deep Tests -- Yoneda-philosophy comprehensive specification
 *
 * These tests define the noise module's complete behavioral contract.
 * They explore every interaction, edge case, and invariant so that
 * any implementation passing all tests is functionally equivalent
 * to the intended design.
 *
 * Categories:
 *  1. Multi-group pipeline isolation
 *  2. Stage composition ordering
 *  3. MAX_NOISE_STAGES saturation
 *  4. Channel boundary validation
 *  5. Dropout edge cases (0.0, 1.0)
 *  6. Quantization edge cases (step=0, huge step, step > range)
 *  7. Saturation edge cases (min>max, min==max)
 *  8. Bias drift O-U mean reversion (statistical)
 *  9. Distance-dependent noise with zero input
 * 10. Per-drone RNG independence (statistical divergence)
 * 11. noise_state_reset_drone clears drift and reseeds RNG
 * 12. noise_type_from_string exhaustive parsing
 */

#include "test_harness.h"
#include "noise.h"
#include <math.h>
#include <string.h>

/* ============================================================================
 * Constants
 * ============================================================================ */

#define DEEP_OUTPUT_SIZE 16
#define STAT_ITERATIONS  2000
#define ARENA_SIZE       (2 * 1024 * 1024)  /* 2 MB */

/* ============================================================================
 * Helpers
 * ============================================================================ */

/** Fill a float buffer with a known ramp: base + channel * step */
static void fill_ramp(float* data, uint32_t count, float base, float step) {
    for (uint32_t i = 0; i < count; i++) {
        data[i] = base + (float)i * step;
    }
}

/** Check that all floats in a range are exactly equal to expected */
static int check_exact(const float* data, const float* expected, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        if (data[i] != expected[i]) return 0;
    }
    return 1;
}

/* ============================================================================
 * 1. Multi-Group Pipeline Isolation
 *
 * 4 NoiseGroups target non-overlapping channel ranges in a 16-channel output.
 * Verify that each group modifies only its own channels and leaves others
 * completely untouched.
 * ============================================================================ */

TEST(multi_group_isolation_4_groups) {
    Arena* arena = arena_create(ARENA_SIZE);

    /* 16-channel output, 4 groups each targeting 4 channels */
    NoiseConfig config = {0};
    config.group_count = 4;

    /* Group 0: channels [0,4) -- constant bias +1.0 */
    config.groups[0].channel_start = 0;
    config.groups[0].channel_count = 4;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_CONSTANT_BIAS;
    config.groups[0].stages[0].params.bias.values[0] = 1.0f;
    config.groups[0].stages[0].params.bias.count = 1;

    /* Group 1: channels [4,8) -- scale factor 2x (error=1.0) */
    config.groups[1].channel_start = 4;
    config.groups[1].channel_count = 4;
    config.groups[1].stage_count = 1;
    config.groups[1].stages[0].type = NOISE_SCALE_FACTOR;
    config.groups[1].stages[0].params.scale.error = 1.0f;

    /* Group 2: channels [8,12) -- saturation clamp to [-0.5, 0.5] */
    config.groups[2].channel_start = 8;
    config.groups[2].channel_count = 4;
    config.groups[2].stage_count = 1;
    config.groups[2].stages[0].type = NOISE_SATURATION;
    config.groups[2].stages[0].params.saturate.min_val = -0.5f;
    config.groups[2].stages[0].params.saturate.max_val = 0.5f;

    /* Group 3: channels [12,16) -- quantization step=1.0 */
    config.groups[3].channel_start = 12;
    config.groups[3].channel_count = 4;
    config.groups[3].stage_count = 1;
    config.groups[3].stages[0].type = NOISE_QUANTIZATION;
    config.groups[3].stages[0].params.quantize.step = 1.0f;

    float data[DEEP_OUTPUT_SIZE];
    float orig[DEEP_OUTPUT_SIZE];
    /* Fill with values that make effects visible */
    for (uint32_t i = 0; i < DEEP_OUTPUT_SIZE; i++) {
        data[i] = 0.3f + (float)i * 0.25f;
    }
    memcpy(orig, data, sizeof(data));

    noise_apply(&config, NULL, data, NULL, 1, DEEP_OUTPUT_SIZE, 0.02f);

    /* Group 0: channels 0-3 each get +1.0 bias */
    for (int i = 0; i < 4; i++) {
        ASSERT_FLOAT_NEAR(data[i], orig[i] + 1.0f, 1e-6f);
    }

    /* Group 1: channels 4-7 each get scaled by 2.0 */
    for (int i = 4; i < 8; i++) {
        ASSERT_FLOAT_NEAR(data[i], orig[i] * 2.0f, 1e-5f);
    }

    /* Group 2: channels 8-11 clamped to [-0.5, 0.5] */
    for (int i = 8; i < 12; i++) {
        ASSERT_TRUE(data[i] >= -0.5f);
        ASSERT_TRUE(data[i] <= 0.5f);
        /* Original values are 2.3, 2.55, 2.8, 3.05 -- all > 0.5 */
        ASSERT_FLOAT_EQ(data[i], 0.5f);
    }

    /* Group 3: channels 12-15 quantized to nearest integer */
    for (int i = 12; i < 16; i++) {
        float expected = roundf(orig[i]);
        ASSERT_FLOAT_NEAR(data[i], expected, 1e-6f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(multi_group_no_cross_contamination) {
    /*
     * Verify that a group targeting channels [4,8) does NOT modify
     * channels [0,4) or [8,16) under any circumstance.
     * Use a single group to isolate the test from multi-group interactions.
     */
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_start = 4;
    config.groups[0].channel_count = 4;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_CONSTANT_BIAS;
    config.groups[0].stages[0].params.bias.values[0] = 999.0f;
    config.groups[0].stages[0].params.bias.count = 1;

    float data[DEEP_OUTPUT_SIZE];
    float orig[DEEP_OUTPUT_SIZE];
    fill_ramp(data, DEEP_OUTPUT_SIZE, 1.0f, 0.1f);
    memcpy(orig, data, sizeof(data));

    noise_apply(&config, NULL, data, NULL, 1, DEEP_OUTPUT_SIZE, 0.02f);

    /* Channels before group range must be untouched */
    for (int i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(data[i], orig[i]);
    }
    /* Channels in group range must be modified */
    for (int i = 4; i < 8; i++) {
        ASSERT_FLOAT_NEAR(data[i], orig[i] + 999.0f, 1e-4f);
    }
    /* Channels after group range must be untouched */
    for (int i = 8; i < 16; i++) {
        ASSERT_FLOAT_EQ(data[i], orig[i]);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * 2. Stage Composition Ordering
 *
 * Verify that the order of stages within a pipeline matters:
 * (bias -> scale) != (scale -> bias) for the same parameters.
 * Also: gaussian -> bias -> drift -> saturation chain produces
 * a valid, order-dependent result.
 * ============================================================================ */

TEST(stage_ordering_bias_then_scale_vs_scale_then_bias) {
    Arena* arena = arena_create(ARENA_SIZE);

    /* Config A: bias +2.0 then scale *1.5 (error=0.5) */
    NoiseConfig config_a = {0};
    config_a.group_count = 1;
    config_a.groups[0].channel_count = 0;  /* all */
    config_a.groups[0].stage_count = 2;
    config_a.groups[0].stages[0].type = NOISE_CONSTANT_BIAS;
    config_a.groups[0].stages[0].params.bias.values[0] = 2.0f;
    config_a.groups[0].stages[0].params.bias.count = 1;
    config_a.groups[0].stages[1].type = NOISE_SCALE_FACTOR;
    config_a.groups[0].stages[1].params.scale.error = 0.5f;

    /* Config B: scale *1.5 then bias +2.0 */
    NoiseConfig config_b = {0};
    config_b.group_count = 1;
    config_b.groups[0].channel_count = 0;
    config_b.groups[0].stage_count = 2;
    config_b.groups[0].stages[0].type = NOISE_SCALE_FACTOR;
    config_b.groups[0].stages[0].params.scale.error = 0.5f;
    config_b.groups[0].stages[1].type = NOISE_CONSTANT_BIAS;
    config_b.groups[0].stages[1].params.bias.values[0] = 2.0f;
    config_b.groups[0].stages[1].params.bias.count = 1;

    float input_val = 10.0f;

    /* Config A: (10 + 2) * 1.5 = 18.0 */
    float data_a[1] = {input_val};
    noise_apply(&config_a, NULL, data_a, NULL, 1, 1, 0.02f);
    ASSERT_FLOAT_NEAR(data_a[0], 18.0f, 1e-5f);

    /* Config B: (10 * 1.5) + 2 = 17.0 */
    float data_b[1] = {input_val};
    noise_apply(&config_b, NULL, data_b, NULL, 1, 1, 0.02f);
    ASSERT_FLOAT_NEAR(data_b[0], 17.0f, 1e-5f);

    /* They must differ -- ordering matters */
    ASSERT_TRUE(fabsf(data_a[0] - data_b[0]) > 0.5f);

    arena_destroy(arena);
    return 0;
}

TEST(stage_ordering_four_stage_chain) {
    /*
     * Chain: constant_bias -> scale_factor -> quantization -> saturation
     * Input: 3.7
     * Step 1: 3.7 + 0.5 = 4.2
     * Step 2: 4.2 * 1.1 = 4.62
     * Step 3: quantize to step=0.5 -> round(4.62/0.5)*0.5 = round(9.24)*0.5 = 9*0.5 = 4.5
     * Step 4: clamp to [0, 4.0] -> 4.0
     */
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    NoisePipeline* pipe = &config.groups[0];
    pipe->channel_count = 0;
    pipe->stage_count = 4;

    pipe->stages[0].type = NOISE_CONSTANT_BIAS;
    pipe->stages[0].params.bias.values[0] = 0.5f;
    pipe->stages[0].params.bias.count = 1;

    pipe->stages[1].type = NOISE_SCALE_FACTOR;
    pipe->stages[1].params.scale.error = 0.1f;

    pipe->stages[2].type = NOISE_QUANTIZATION;
    pipe->stages[2].params.quantize.step = 0.5f;

    pipe->stages[3].type = NOISE_SATURATION;
    pipe->stages[3].params.saturate.min_val = 0.0f;
    pipe->stages[3].params.saturate.max_val = 4.0f;

    float data[1] = {3.7f};
    noise_apply(&config, NULL, data, NULL, 1, 1, 0.02f);
    ASSERT_FLOAT_NEAR(data[0], 4.0f, 1e-5f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * 3. MAX_NOISE_STAGES Pipeline
 *
 * Fill a pipeline with all 8 stages (the maximum), each a different type.
 * Verify all 8 are applied and the final result matches manual computation.
 * ============================================================================ */

TEST(max_noise_stages_all_types) {
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    NoisePipeline* pipe = &config.groups[0];
    pipe->channel_count = 1;
    pipe->stage_count = MAX_NOISE_STAGES;  /* 8 */

    /*
     * We use only deterministic stages to verify exact computation.
     * 8 stages: bias, scale, bias, scale, quantize, bias, scale, saturate
     *
     * Input: 1.0
     * Stage 0 (bias +0.5):       1.0 + 0.5 = 1.5
     * Stage 1 (scale *2):        1.5 * 2.0 = 3.0
     * Stage 2 (bias -1.0):       3.0 - 1.0 = 2.0
     * Stage 3 (scale *1.25):     2.0 * 1.25 = 2.5
     * Stage 4 (quantize 0.3):    round(2.5/0.3)*0.3 = round(8.333)*0.3 = 8*0.3 = 2.4
     * Stage 5 (bias +0.1):       2.4 + 0.1 = 2.5
     * Stage 6 (scale *0.8):      2.5 * 0.8 = 2.0
     * Stage 7 (saturate [0,1.5]):clamp(2.0, 0, 1.5) = 1.5
     */

    pipe->stages[0].type = NOISE_CONSTANT_BIAS;
    pipe->stages[0].params.bias.values[0] = 0.5f;
    pipe->stages[0].params.bias.count = 1;

    pipe->stages[1].type = NOISE_SCALE_FACTOR;
    pipe->stages[1].params.scale.error = 1.0f;  /* factor = 2.0 */

    pipe->stages[2].type = NOISE_CONSTANT_BIAS;
    pipe->stages[2].params.bias.values[0] = -1.0f;
    pipe->stages[2].params.bias.count = 1;

    pipe->stages[3].type = NOISE_SCALE_FACTOR;
    pipe->stages[3].params.scale.error = 0.25f;  /* factor = 1.25 */

    pipe->stages[4].type = NOISE_QUANTIZATION;
    pipe->stages[4].params.quantize.step = 0.3f;

    pipe->stages[5].type = NOISE_CONSTANT_BIAS;
    pipe->stages[5].params.bias.values[0] = 0.1f;
    pipe->stages[5].params.bias.count = 1;

    pipe->stages[6].type = NOISE_SCALE_FACTOR;
    pipe->stages[6].params.scale.error = -0.2f;  /* factor = 0.8 */

    pipe->stages[7].type = NOISE_SATURATION;
    pipe->stages[7].params.saturate.min_val = 0.0f;
    pipe->stages[7].params.saturate.max_val = 1.5f;

    float data[1] = {1.0f};
    noise_apply(&config, NULL, data, NULL, 1, 1, 0.02f);
    ASSERT_FLOAT_NEAR(data[0], 1.5f, 1e-5f);

    arena_destroy(arena);
    return 0;
}

TEST(max_noise_stages_count_is_eight) {
    /* Compile-time property: MAX_NOISE_STAGES == 8 */
    ASSERT_EQ(MAX_NOISE_STAGES, 8);
    return 0;
}

/* ============================================================================
 * 4. Channel Group Boundary Validation
 *
 * When channel_start + channel_count > output_size, the implementation should
 * gracefully clamp (not overrun). Verify this by checking that no data beyond
 * output_size is touched.
 * ============================================================================ */

TEST(channel_boundary_clamped_when_exceeding_output) {
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_start = 6;
    config.groups[0].channel_count = 20;  /* 6+20=26, but output_size=8 */
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_CONSTANT_BIAS;
    config.groups[0].stages[0].params.bias.values[0] = 100.0f;
    config.groups[0].stages[0].params.bias.count = 1;

    /* Allocate with padding to detect overflow writes */
    float data[16];
    float sentinel[16];
    memset(data, 0, sizeof(data));
    memset(sentinel, 0, sizeof(sentinel));

    /* Fill the "real" 8 channels with known values */
    for (int i = 0; i < 8; i++) {
        data[i] = (float)i;
        sentinel[i] = (float)i;
    }
    /* Channels 8-15 are sentinel padding -- must remain zero */
    for (int i = 8; i < 16; i++) {
        data[i] = -999.0f;
        sentinel[i] = -999.0f;
    }

    /* output_size = 8, so effective range is channels [6, 8), count=2 */
    noise_apply(&config, NULL, data, NULL, 1, 8, 0.02f);

    /* Channels [0,6): untouched */
    for (int i = 0; i < 6; i++) {
        ASSERT_FLOAT_EQ(data[i], sentinel[i]);
    }
    /* Channels [6,8): biased by +100 */
    ASSERT_FLOAT_NEAR(data[6], 6.0f + 100.0f, 1e-5f);
    ASSERT_FLOAT_NEAR(data[7], 7.0f + 100.0f, 1e-5f);
    /* Sentinel padding beyond output_size: must remain untouched */
    for (int i = 8; i < 16; i++) {
        ASSERT_FLOAT_EQ(data[i], sentinel[i]);
    }

    arena_destroy(arena);
    return 0;
}

TEST(channel_start_at_output_size_means_zero_channels) {
    /*
     * If channel_start == output_size, then ch_count = output_size - ch_start = 0.
     * No channels should be modified.
     */
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_start = 8;  /* == output_size */
    config.groups[0].channel_count = 4;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_CONSTANT_BIAS;
    config.groups[0].stages[0].params.bias.values[0] = 500.0f;
    config.groups[0].stages[0].params.bias.count = 1;

    float data[8];
    float orig[8];
    fill_ramp(data, 8, 1.0f, 1.0f);
    memcpy(orig, data, sizeof(data));

    /* channel_start=8 >= output_size=8, so clamped ch_count = 8-8 = 0 */
    noise_apply(&config, NULL, data, NULL, 1, 8, 0.02f);

    /* Everything untouched */
    ASSERT_TRUE(check_exact(data, orig, 8));

    arena_destroy(arena);
    return 0;
}

TEST(channel_count_zero_means_all_channels) {
    /*
     * channel_count=0 is a sentinel meaning "all channels" (from channel_start).
     * With channel_start=0, this means the entire output.
     */
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_start = 0;
    config.groups[0].channel_count = 0;  /* sentinel: all */
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_SCALE_FACTOR;
    config.groups[0].stages[0].params.scale.error = 1.0f;  /* double */

    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    noise_apply(&config, NULL, data, NULL, 1, 4, 0.02f);

    ASSERT_FLOAT_NEAR(data[0], 2.0f, 1e-5f);
    ASSERT_FLOAT_NEAR(data[1], 4.0f, 1e-5f);
    ASSERT_FLOAT_NEAR(data[2], 6.0f, 1e-5f);
    ASSERT_FLOAT_NEAR(data[3], 8.0f, 1e-5f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * 5. Dropout Edge Cases
 *
 * probability=0.0: no drops ever.
 * probability=1.0: all drops always.
 * ============================================================================ */

TEST(dropout_probability_zero_no_drops) {
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_DROPOUT;
    config.groups[0].stages[0].params.dropout.probability = 0.0f;
    config.groups[0].stages[0].params.dropout.replacement = -999.0f;

    NoiseState* state = noise_state_create(arena, &config, 1, 0);
    ASSERT_NOT_NULL(state);

    uint32_t idx = 0;
    for (int iter = 0; iter < 500; iter++) {
        float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        noise_apply(&config, state, data, &idx, 1, 4, 0.02f);
        /* pcg32_float returns [0, 1) so it is never < 0.0 */
        for (int c = 0; c < 4; c++) {
            ASSERT_TRUE(data[c] != -999.0f);
        }
    }

    arena_destroy(arena);
    return 0;
}

TEST(dropout_probability_one_all_drops) {
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_DROPOUT;
    config.groups[0].stages[0].params.dropout.probability = 1.0f;
    config.groups[0].stages[0].params.dropout.replacement = -42.0f;

    NoiseState* state = noise_state_create(arena, &config, 1, 0);
    ASSERT_NOT_NULL(state);

    uint32_t idx = 0;
    for (int iter = 0; iter < 500; iter++) {
        float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        noise_apply(&config, state, data, &idx, 1, 4, 0.02f);
        /* pcg32_float returns [0, 1), always < 1.0 */
        for (int c = 0; c < 4; c++) {
            ASSERT_FLOAT_EQ(data[c], -42.0f);
        }
    }

    arena_destroy(arena);
    return 0;
}

TEST(dropout_replacement_value_preserved) {
    /*
     * Verify the replacement value is exactly what gets written,
     * not zero or some default.
     */
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_DROPOUT;
    config.groups[0].stages[0].params.dropout.probability = 1.0f;
    config.groups[0].stages[0].params.dropout.replacement = 3.14159f;

    NoiseState* state = noise_state_create(arena, &config, 1, 0);
    ASSERT_NOT_NULL(state);

    float data[2] = {100.0f, 200.0f};
    uint32_t idx = 0;
    noise_apply(&config, state, data, &idx, 1, 2, 0.02f);

    ASSERT_FLOAT_NEAR(data[0], 3.14159f, 1e-5f);
    ASSERT_FLOAT_NEAR(data[1], 3.14159f, 1e-5f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * 6. Quantization Edge Cases
 *
 * step=0: degenerate, apply_quantization returns early (no modification).
 * step < 0: also returns early per the (step <= 0.0f) guard.
 * Very large step: everything rounds to zero or +/-step.
 * step > value range: everything collapses to nearest multiple.
 * ============================================================================ */

TEST(quantization_step_zero_no_modification) {
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_QUANTIZATION;
    config.groups[0].stages[0].params.quantize.step = 0.0f;

    float data[3] = {1.234f, -5.678f, 0.001f};
    float orig[3];
    memcpy(orig, data, sizeof(data));

    noise_apply(&config, NULL, data, NULL, 1, 3, 0.02f);

    /* step=0 triggers early return, no modification */
    for (int i = 0; i < 3; i++) {
        ASSERT_FLOAT_EQ(data[i], orig[i]);
    }

    arena_destroy(arena);
    return 0;
}

TEST(quantization_step_negative_no_modification) {
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_QUANTIZATION;
    config.groups[0].stages[0].params.quantize.step = -1.0f;

    float data[2] = {2.7f, -3.3f};
    float orig[2];
    memcpy(orig, data, sizeof(data));

    noise_apply(&config, NULL, data, NULL, 1, 2, 0.02f);

    /* step <= 0 triggers early return */
    for (int i = 0; i < 2; i++) {
        ASSERT_FLOAT_EQ(data[i], orig[i]);
    }

    arena_destroy(arena);
    return 0;
}

TEST(quantization_very_large_step) {
    /*
     * step=1000: all values in range [-10,10] round to 0.
     * Value 600.0 rounds to 1000.
     */
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_QUANTIZATION;
    config.groups[0].stages[0].params.quantize.step = 1000.0f;

    float data[4] = {0.1f, -5.0f, 499.9f, 500.1f};
    noise_apply(&config, NULL, data, NULL, 1, 4, 0.02f);

    /* 0.1 / 1000 = 0.0001, round -> 0, * 1000 -> 0 */
    ASSERT_FLOAT_NEAR(data[0], 0.0f, 1e-3f);
    /* -5 / 1000 = -0.005, round -> 0, * 1000 -> 0 */
    ASSERT_FLOAT_NEAR(data[1], 0.0f, 1e-3f);
    /* 499.9 / 1000 = 0.4999, round -> 0, * 1000 -> 0 */
    ASSERT_FLOAT_NEAR(data[2], 0.0f, 1e-3f);
    /* 500.1 / 1000 = 0.5001, round -> 1, * 1000 -> 1000 */
    ASSERT_FLOAT_NEAR(data[3], 1000.0f, 1e-1f);

    arena_destroy(arena);
    return 0;
}

TEST(quantization_exact_multiples_unchanged) {
    /*
     * Values that are already exact multiples of step should be unchanged.
     */
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_QUANTIZATION;
    config.groups[0].stages[0].params.quantize.step = 0.25f;

    float data[4] = {0.0f, 0.25f, -0.5f, 1.0f};
    noise_apply(&config, NULL, data, NULL, 1, 4, 0.02f);

    ASSERT_FLOAT_NEAR(data[0], 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR(data[1], 0.25f, 1e-6f);
    ASSERT_FLOAT_NEAR(data[2], -0.5f, 1e-6f);
    ASSERT_FLOAT_NEAR(data[3], 1.0f, 1e-6f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * 7. Saturation Edge Cases
 *
 * min_val > max_val: both clamps apply, but since min>max the result is
 *   the max_val check last, so effectively everything <= max_val stays at max_val.
 *   Implementation: if (v < min) v = min; if (v > max) v = max;
 *   When min > max, for any v:
 *     if v < min (big): v = min
 *     if min > max: then v = min > max, so v > max -> v = max
 *   Result: everything becomes max_val when min > max.
 *
 * min_val == max_val: all outputs become that single value.
 * ============================================================================ */

TEST(saturation_min_equals_max) {
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_SATURATION;
    config.groups[0].stages[0].params.saturate.min_val = 5.0f;
    config.groups[0].stages[0].params.saturate.max_val = 5.0f;

    float data[4] = {-100.0f, 0.0f, 5.0f, 100.0f};
    noise_apply(&config, NULL, data, NULL, 1, 4, 0.02f);

    for (int i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(data[i], 5.0f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(saturation_min_greater_than_max) {
    /*
     * Implementation: if (v < min) v = min; if (v > max) v = max;
     * When min=10, max=2:
     *   v=0: 0 < 10 -> v=10; 10 > 2 -> v=2. Result: 2
     *   v=5: 5 < 10 -> v=10; 10 > 2 -> v=2. Result: 2
     *   v=15: 15 >= 10 ok; 15 > 2 -> v=2. Result: 2
     *   v=-100: -100 < 10 -> v=10; 10 > 2 -> v=2. Result: 2
     *
     * Everything becomes max_val when min > max.
     */
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_SATURATION;
    config.groups[0].stages[0].params.saturate.min_val = 10.0f;
    config.groups[0].stages[0].params.saturate.max_val = 2.0f;

    float data[4] = {-100.0f, 0.0f, 5.0f, 100.0f};
    noise_apply(&config, NULL, data, NULL, 1, 4, 0.02f);

    /* All values end up at max_val=2.0 due to the second clamp winning */
    for (int i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(data[i], 2.0f);
    }

    arena_destroy(arena);
    return 0;
}

TEST(saturation_values_within_range_unchanged) {
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_SATURATION;
    config.groups[0].stages[0].params.saturate.min_val = -10.0f;
    config.groups[0].stages[0].params.saturate.max_val = 10.0f;

    float data[4] = {-5.0f, 0.0f, 3.14f, 9.99f};
    float orig[4];
    memcpy(orig, data, sizeof(data));

    noise_apply(&config, NULL, data, NULL, 1, 4, 0.02f);

    for (int i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(data[i], orig[i]);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * 8. Bias Drift (Ornstein-Uhlenbeck) Mean Reversion -- Statistical
 *
 * The OU process dx = -x/tau*dt + sigma*sqrt(dt)*dW has stationary variance
 * sigma^2 * tau / 2. Over many steps starting from 0, the drift state should
 * have mean near 0 and variance near sigma^2*tau/2.
 *
 * Additionally: if we push drift far from equilibrium, it should revert.
 * ============================================================================ */

TEST(bias_drift_ou_stationary_distribution) {
    Arena* arena = arena_create(ARENA_SIZE);

    float tau = 2.0f;
    float sigma = 0.1f;
    float dt = 0.01f;

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_BIAS_DRIFT;
    config.groups[0].stages[0].params.drift.tau = tau;
    config.groups[0].stages[0].params.drift.sigma = sigma;

    NoiseState* state = noise_state_create(arena, &config, 1, 0);
    ASSERT_NOT_NULL(state);

    /* Burn in: run 5000 steps to reach stationarity (50s >> tau=2s) */
    uint32_t idx = 0;
    for (int i = 0; i < 5000; i++) {
        float data[1] = {0.0f};
        noise_apply(&config, state, data, &idx, 1, 1, dt);
    }

    /* Collect 5000 samples of drift state after stationarity */
    double sum = 0.0;
    double sum_sq = 0.0;
    int N = 5000;
    for (int i = 0; i < N; i++) {
        float data[1] = {0.0f};
        noise_apply(&config, state, data, &idx, 1, 1, dt);
        float drift_val = state->groups[0].drift_state[0];
        sum += drift_val;
        sum_sq += (double)drift_val * drift_val;
    }

    double mean = sum / N;
    double var = sum_sq / N - mean * mean;

    /* Theoretical stationary variance = sigma^2 * tau / 2 = 0.01 * 2 / 2 = 0.01 */
    /* Mean should be near 0 (lenient: within 0.05) */
    ASSERT_TRUE(fabs(mean) < 0.05);
    /* Variance should be in the right ballpark: 0.001 to 0.05 (theoretical 0.01)
     * The Irwin-Hall approximation is not a true Gaussian, so expect some deviation. */
    ASSERT_TRUE(var > 0.001);
    ASSERT_TRUE(var < 0.1);

    arena_destroy(arena);
    return 0;
}

TEST(bias_drift_large_displacement_reverts) {
    /*
     * Push drift state to +100.0, then run many steps with strong mean reversion
     * (small tau). After enough time, drift should be close to zero.
     */
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_BIAS_DRIFT;
    config.groups[0].stages[0].params.drift.tau = 0.5f;   /* fast reversion */
    config.groups[0].stages[0].params.drift.sigma = 0.01f; /* tiny diffusion */

    NoiseState* state = noise_state_create(arena, &config, 1, 0);
    ASSERT_NOT_NULL(state);

    /* Push drift far from zero */
    state->groups[0].drift_state[0] = 100.0f;

    float dt = 0.02f;
    uint32_t idx = 0;
    /* Run 1000 steps = 20 seconds >> tau=0.5 (40 time constants) */
    for (int i = 0; i < 1000; i++) {
        float data[1] = {0.0f};
        noise_apply(&config, state, data, &idx, 1, 1, dt);
    }

    float drift_val = state->groups[0].drift_state[0];
    /* After 40 time constants, e^{-40} ~ 4e-18, so drift should be near 0 */
    ASSERT_TRUE(fabsf(drift_val) < 1.0f);

    arena_destroy(arena);
    return 0;
}

TEST(bias_drift_adds_to_channel_data) {
    /*
     * Verify drift is additive: channel_data[c] += drift[c].
     * After one step from drift=0 with large sigma, the output should differ
     * from the input.
     */
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 2;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_BIAS_DRIFT;
    config.groups[0].stages[0].params.drift.tau = 1.0f;
    config.groups[0].stages[0].params.drift.sigma = 10.0f;

    NoiseState* state = noise_state_create(arena, &config, 1, 0);
    ASSERT_NOT_NULL(state);

    /* Confirm drift starts at zero */
    ASSERT_FLOAT_EQ(state->groups[0].drift_state[0], 0.0f);
    ASSERT_FLOAT_EQ(state->groups[0].drift_state[1], 0.0f);

    float data[2] = {5.0f, 10.0f};
    uint32_t idx = 0;
    noise_apply(&config, state, data, &idx, 1, 2, 0.02f);

    /* After one step, drift[c] = 0 * (1 - dt/tau) + sigma*sqrt(dt)*N = sigma*sqrt(dt)*N.
     * With sigma=10, sqrt(0.02)~0.14, so drift ~ 1.4 * noise_gaussian.
     * The output should differ from input. */
    ASSERT_TRUE(fabsf(data[0] - 5.0f) > 1e-6f || fabsf(data[1] - 10.0f) > 1e-6f);

    /* Verify drift state is non-zero now */
    ASSERT_TRUE(fabsf(state->groups[0].drift_state[0]) > 1e-6f ||
                fabsf(state->groups[0].drift_state[1]) > 1e-6f);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * 9. Distance-Dependent Noise: Zero Input
 *
 * stddev = coeff * |v|^power. When v=0, stddev=0, so no noise should be added.
 * ============================================================================ */

TEST(distance_dependent_zero_input_no_noise) {
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_DISTANCE_DEPENDENT;
    config.groups[0].stages[0].params.distance.coeff = 100.0f;
    config.groups[0].stages[0].params.distance.power = 2.0f;

    NoiseState* state = noise_state_create(arena, &config, 1, 0);
    ASSERT_NOT_NULL(state);

    uint32_t idx = 0;
    for (int iter = 0; iter < 500; iter++) {
        float data[3] = {0.0f, 0.0f, 0.0f};
        noise_apply(&config, state, data, &idx, 1, 3, 0.02f);
        /* coeff * |0|^2 = 0 -> no noise */
        for (int c = 0; c < 3; c++) {
            ASSERT_FLOAT_EQ(data[c], 0.0f);
        }
    }

    arena_destroy(arena);
    return 0;
}

TEST(distance_dependent_power_one_linear) {
    /*
     * With power=1, stddev = coeff * |v|. Noise variance should scale
     * linearly with |v|^2. Compare v=1 vs v=10: variance ratio should be ~100.
     */
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_DISTANCE_DEPENDENT;
    config.groups[0].stages[0].params.distance.coeff = 0.5f;
    config.groups[0].stages[0].params.distance.power = 1.0f;

    NoiseState* state = noise_state_create(arena, &config, 1, 0);
    ASSERT_NOT_NULL(state);

    double sum_sq_small = 0.0;
    double sum_sq_large = 0.0;
    int N = STAT_ITERATIONS;
    uint32_t idx = 0;

    for (int i = 0; i < N; i++) {
        float d1[1] = {1.0f};
        float d2[1] = {10.0f};
        noise_apply(&config, state, d1, &idx, 1, 1, 0.02f);
        noise_apply(&config, state, d2, &idx, 1, 1, 0.02f);
        double n1 = d1[0] - 1.0;
        double n2 = d2[0] - 10.0;
        sum_sq_small += n1 * n1;
        sum_sq_large += n2 * n2;
    }

    double var_small = sum_sq_small / N;
    double var_large = sum_sq_large / N;

    /* Variance ratio should be ~100 (10^2) since stddev scales linearly.
     * Use lenient bounds: ratio in [20, 500]. */
    double ratio = var_large / var_small;
    ASSERT_TRUE(ratio > 20.0);
    ASSERT_TRUE(ratio < 500.0);

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * 10. Per-Drone RNG Independence
 *
 * Same config, different agent_indices should produce different noise.
 * Same config, same agent_index after reset should produce same noise.
 * ============================================================================ */

TEST(per_drone_rng_divergence_statistical) {
    /*
     * Run 100 drones through white gaussian noise simultaneously.
     * Collect per-drone means. They should NOT all be the same (that would
     * indicate shared RNG state).
     */
    Arena* arena = arena_create(ARENA_SIZE);

    uint32_t num_agents = 100;
    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_WHITE_GAUSSIAN;
    config.groups[0].stages[0].params.white.stddev = 1.0f;

    NoiseState* state = noise_state_create(arena, &config, num_agents, 0);
    ASSERT_NOT_NULL(state);

    /* Accumulate per-drone sums over 100 iterations */
    float sums[100];
    memset(sums, 0, sizeof(sums));

    for (int iter = 0; iter < 100; iter++) {
        float* batch = (float*)arena_alloc_array(arena, float, num_agents);
        memset(batch, 0, num_agents * sizeof(float));
        uint32_t indices[100];
        for (uint32_t d = 0; d < num_agents; d++) indices[d] = d;

        noise_apply(&config, state, batch, indices, num_agents, 1, 0.02f);

        for (uint32_t d = 0; d < num_agents; d++) {
            sums[d] += batch[d];
        }
    }

    /* Count how many drones have unique sum values (not equal to drone 0) */
    int unique = 0;
    for (uint32_t d = 1; d < num_agents; d++) {
        if (fabsf(sums[d] - sums[0]) > 0.01f) unique++;
    }

    /* With independent RNG, all 99 other drones should differ from drone 0 */
    ASSERT_TRUE(unique > 90);

    arena_destroy(arena);
    return 0;
}

TEST(per_drone_same_index_deterministic) {
    /*
     * Two separate state objects with same sensor_id, same agent_index
     * should produce identical noise sequences.
     */
    Arena* arena1 = arena_create(ARENA_SIZE);
    Arena* arena2 = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 3;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_WHITE_GAUSSIAN;
    config.groups[0].stages[0].params.white.stddev = 1.0f;

    NoiseState* s1 = noise_state_create(arena1, &config, 4, /*sensor_id=*/7);
    NoiseState* s2 = noise_state_create(arena2, &config, 4, /*sensor_id=*/7);
    ASSERT_NOT_NULL(s1);
    ASSERT_NOT_NULL(s2);

    uint32_t idx = 2;  /* use drone index 2 */
    for (int iter = 0; iter < 50; iter++) {
        float d1[3] = {0.0f, 0.0f, 0.0f};
        float d2[3] = {0.0f, 0.0f, 0.0f};
        noise_apply(&config, s1, d1, &idx, 1, 3, 0.02f);
        noise_apply(&config, s2, d2, &idx, 1, 3, 0.02f);
        for (int c = 0; c < 3; c++) {
            ASSERT_FLOAT_EQ(d1[c], d2[c]);
        }
    }

    arena_destroy(arena1);
    arena_destroy(arena2);
    return 0;
}

TEST(per_drone_different_sensor_id_diverges) {
    /*
     * Two state objects with different sensor_ids but same agent_index
     * should produce different noise sequences.
     */
    Arena* arena1 = arena_create(ARENA_SIZE);
    Arena* arena2 = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_WHITE_GAUSSIAN;
    config.groups[0].stages[0].params.white.stddev = 1.0f;

    NoiseState* s1 = noise_state_create(arena1, &config, 1, /*sensor_id=*/0);
    NoiseState* s2 = noise_state_create(arena2, &config, 1, /*sensor_id=*/1);
    ASSERT_NOT_NULL(s1);
    ASSERT_NOT_NULL(s2);

    uint32_t idx = 0;
    float d1[1] = {0.0f};
    float d2[1] = {0.0f};
    noise_apply(&config, s1, d1, &idx, 1, 1, 0.02f);
    noise_apply(&config, s2, d2, &idx, 1, 1, 0.02f);

    /* Different seeds should produce different first sample */
    ASSERT_TRUE(fabsf(d1[0] - d2[0]) > 1e-6f);

    arena_destroy(arena1);
    arena_destroy(arena2);
    return 0;
}

/* ============================================================================
 * 11. noise_state_reset_drone
 *
 * After reset, drift state should be zero and RNG should be reseeded
 * to produce the same sequence as initial state.
 * ============================================================================ */

TEST(reset_drone_zeros_drift) {
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 2;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_BIAS_DRIFT;
    config.groups[0].stages[0].params.drift.tau = 1.0f;
    config.groups[0].stages[0].params.drift.sigma = 1.0f;

    NoiseState* state = noise_state_create(arena, &config, 2, 0);
    ASSERT_NOT_NULL(state);
    ASSERT_EQ(state->groups[0].drift_channels, (uint32_t)2);

    /* Run drone 0 for many steps to build up drift */
    uint32_t idx = 0;
    for (int i = 0; i < 100; i++) {
        float data[2] = {0.0f, 0.0f};
        noise_apply(&config, state, data, &idx, 1, 2, 0.02f);
    }

    /* Drift should be non-zero after 100 steps */
    float drift0_before = state->groups[0].drift_state[0];
    float drift1_before = state->groups[0].drift_state[1];
    ASSERT_TRUE(fabsf(drift0_before) > 1e-6f || fabsf(drift1_before) > 1e-6f);

    /* Reset drone 0 */
    noise_state_reset_drone(state, 0);

    /* Drift should be exactly zero */
    ASSERT_FLOAT_EQ(state->groups[0].drift_state[0], 0.0f);
    ASSERT_FLOAT_EQ(state->groups[0].drift_state[1], 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(reset_drone_reseeds_rng_deterministically) {
    /*
     * After reset, the RNG for that drone should produce the exact same
     * sequence as right after creation.
     */
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_WHITE_GAUSSIAN;
    config.groups[0].stages[0].params.white.stddev = 1.0f;

    NoiseState* state = noise_state_create(arena, &config, 1, 42);
    ASSERT_NOT_NULL(state);

    /* Record first 10 samples from fresh state */
    float first_run[10];
    uint32_t idx = 0;
    for (int i = 0; i < 10; i++) {
        float data[1] = {0.0f};
        noise_apply(&config, state, data, &idx, 1, 1, 0.02f);
        first_run[i] = data[0];
    }

    /* Advance RNG many more steps to make state diverge */
    for (int i = 0; i < 500; i++) {
        float data[1] = {0.0f};
        noise_apply(&config, state, data, &idx, 1, 1, 0.02f);
    }

    /* Reset */
    noise_state_reset_drone(state, 0);

    /* Re-record 10 samples -- should match first_run exactly */
    for (int i = 0; i < 10; i++) {
        float data[1] = {0.0f};
        noise_apply(&config, state, data, &idx, 1, 1, 0.02f);
        ASSERT_FLOAT_EQ(data[0], first_run[i]);
    }

    arena_destroy(arena);
    return 0;
}

TEST(reset_drone_does_not_affect_other_drones) {
    /*
     * Reset drone 0 should not affect drone 1's RNG or drift.
     */
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_BIAS_DRIFT;
    config.groups[0].stages[0].params.drift.tau = 1.0f;
    config.groups[0].stages[0].params.drift.sigma = 1.0f;

    NoiseState* state = noise_state_create(arena, &config, 2, 0);
    ASSERT_NOT_NULL(state);

    /* Run both drones */
    for (int i = 0; i < 50; i++) {
        float batch[2] = {0.0f, 0.0f};
        uint32_t indices[2] = {0, 1};
        noise_apply(&config, state, batch, indices, 2, 1, 0.02f);
    }

    /* Record drone 1 drift state */
    float drone1_drift_before = state->groups[0].drift_state[1];

    /* Reset only drone 0 */
    noise_state_reset_drone(state, 0);

    /* Drone 1 drift should be unchanged */
    ASSERT_FLOAT_EQ(state->groups[0].drift_state[1], drone1_drift_before);

    /* Drone 0 drift should be zero */
    ASSERT_FLOAT_EQ(state->groups[0].drift_state[0], 0.0f);

    arena_destroy(arena);
    return 0;
}

TEST(reset_all_resets_every_drone) {
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_BIAS_DRIFT;
    config.groups[0].stages[0].params.drift.tau = 1.0f;
    config.groups[0].stages[0].params.drift.sigma = 5.0f;

    uint32_t num_agents = 8;
    NoiseState* state = noise_state_create(arena, &config, num_agents, 0);
    ASSERT_NOT_NULL(state);

    /* Run all drones to build up drift */
    for (int i = 0; i < 100; i++) {
        float batch[8];
        memset(batch, 0, sizeof(batch));
        uint32_t indices[8];
        for (uint32_t d = 0; d < num_agents; d++) indices[d] = d;
        noise_apply(&config, state, batch, indices, num_agents, 1, 0.02f);
    }

    /* Reset all */
    noise_state_reset_all(state);

    /* All drift states should be zero */
    for (uint32_t d = 0; d < num_agents; d++) {
        ASSERT_FLOAT_EQ(state->groups[0].drift_state[d], 0.0f);
    }

    arena_destroy(arena);
    return 0;
}

/* ============================================================================
 * 12. noise_type_from_string: Exhaustive Parsing
 *
 * All valid type strings, invalid strings, NULL, empty string.
 * ============================================================================ */

TEST(noise_type_from_string_all_valid) {
    ASSERT_EQ(noise_type_from_string("none"), NOISE_NONE);
    ASSERT_EQ(noise_type_from_string("white_gaussian"), NOISE_WHITE_GAUSSIAN);
    ASSERT_EQ(noise_type_from_string("constant_bias"), NOISE_CONSTANT_BIAS);
    ASSERT_EQ(noise_type_from_string("bias_drift"), NOISE_BIAS_DRIFT);
    ASSERT_EQ(noise_type_from_string("scale_factor"), NOISE_SCALE_FACTOR);
    ASSERT_EQ(noise_type_from_string("distance_dependent"), NOISE_DISTANCE_DEPENDENT);
    ASSERT_EQ(noise_type_from_string("quantization"), NOISE_QUANTIZATION);
    ASSERT_EQ(noise_type_from_string("dropout"), NOISE_DROPOUT);
    ASSERT_EQ(noise_type_from_string("saturation"), NOISE_SATURATION);
    return 0;
}

TEST(noise_type_from_string_invalid) {
    ASSERT_EQ(noise_type_from_string("garbage"), NOISE_NONE);
    ASSERT_EQ(noise_type_from_string("WHITE_GAUSSIAN"), NOISE_NONE);  /* case sensitive */
    ASSERT_EQ(noise_type_from_string("white gaussian"), NOISE_NONE);  /* space not underscore */
    ASSERT_EQ(noise_type_from_string("gaussian"), NOISE_NONE);        /* partial match */
    ASSERT_EQ(noise_type_from_string(""), NOISE_NONE);                /* empty string */
    return 0;
}

TEST(noise_type_from_string_null) {
    ASSERT_EQ(noise_type_from_string(NULL), NOISE_NONE);
    return 0;
}

TEST(noise_type_name_all_valid) {
    ASSERT_STR_EQ(noise_type_name(NOISE_NONE), "none");
    ASSERT_STR_EQ(noise_type_name(NOISE_WHITE_GAUSSIAN), "white_gaussian");
    ASSERT_STR_EQ(noise_type_name(NOISE_CONSTANT_BIAS), "constant_bias");
    ASSERT_STR_EQ(noise_type_name(NOISE_BIAS_DRIFT), "bias_drift");
    ASSERT_STR_EQ(noise_type_name(NOISE_SCALE_FACTOR), "scale_factor");
    ASSERT_STR_EQ(noise_type_name(NOISE_DISTANCE_DEPENDENT), "distance_dependent");
    ASSERT_STR_EQ(noise_type_name(NOISE_QUANTIZATION), "quantization");
    ASSERT_STR_EQ(noise_type_name(NOISE_DROPOUT), "dropout");
    ASSERT_STR_EQ(noise_type_name(NOISE_SATURATION), "saturation");
    return 0;
}

TEST(noise_type_name_out_of_range) {
    ASSERT_STR_EQ(noise_type_name(NOISE_TYPE_COUNT), "unknown");
    ASSERT_STR_EQ(noise_type_name((NoiseType)99), "unknown");
    return 0;
}

TEST(noise_type_roundtrip) {
    /*
     * For every valid type, name -> string -> from_string should return
     * the original type (except NOISE_NONE which maps to "none" and back).
     */
    for (int i = 0; i < NOISE_TYPE_COUNT; i++) {
        NoiseType t = (NoiseType)i;
        const char* name = noise_type_name(t);
        NoiseType parsed = noise_type_from_string(name);
        ASSERT_EQ(parsed, t);
    }
    return 0;
}

/* ============================================================================
 * Additional Edge Cases and Robustness Tests
 * ============================================================================ */

TEST(noise_apply_null_config_no_crash) {
    /* NULL config should be a no-op */
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float orig[4];
    memcpy(orig, data, sizeof(data));
    noise_apply(NULL, NULL, data, NULL, 1, 4, 0.02f);
    ASSERT_TRUE(check_exact(data, orig, 4));
    return 0;
}

TEST(noise_apply_null_data_no_crash) {
    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_CONSTANT_BIAS;
    config.groups[0].stages[0].params.bias.values[0] = 1.0f;
    config.groups[0].stages[0].params.bias.count = 1;
    /* Should not crash */
    noise_apply(&config, NULL, NULL, NULL, 1, 4, 0.02f);
    return 0;
}

TEST(noise_apply_zero_agent_count_no_crash) {
    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_CONSTANT_BIAS;
    config.groups[0].stages[0].params.bias.values[0] = 1.0f;
    config.groups[0].stages[0].params.bias.count = 1;
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float orig[4];
    memcpy(orig, data, sizeof(data));
    noise_apply(&config, NULL, data, NULL, 0, 4, 0.02f);
    ASSERT_TRUE(check_exact(data, orig, 4));
    return 0;
}

TEST(noise_state_create_null_arena) {
    NoiseConfig config = {0};
    config.group_count = 1;
    NoiseState* state = noise_state_create(NULL, &config, 1, 0);
    ASSERT_NULL(state);
    return 0;
}

TEST(noise_state_create_null_config) {
    Arena* arena = arena_create(ARENA_SIZE);
    NoiseState* state = noise_state_create(arena, NULL, 1, 0);
    ASSERT_NULL(state);
    arena_destroy(arena);
    return 0;
}

TEST(noise_state_create_zero_groups) {
    Arena* arena = arena_create(ARENA_SIZE);
    NoiseConfig config = {0};
    config.group_count = 0;
    NoiseState* state = noise_state_create(arena, &config, 1, 0);
    ASSERT_NULL(state);
    arena_destroy(arena);
    return 0;
}

TEST(noise_state_create_zero_drones) {
    Arena* arena = arena_create(ARENA_SIZE);
    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_WHITE_GAUSSIAN;
    NoiseState* state = noise_state_create(arena, &config, 0, 0);
    ASSERT_NULL(state);
    arena_destroy(arena);
    return 0;
}

TEST(noise_state_reset_drone_out_of_range_no_crash) {
    Arena* arena = arena_create(ARENA_SIZE);
    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_WHITE_GAUSSIAN;
    NoiseState* state = noise_state_create(arena, &config, 4, 0);
    ASSERT_NOT_NULL(state);
    /* agent_idx=10 is out of range for max_agents=4, should be a no-op */
    noise_state_reset_drone(state, 10);
    /* Should not crash */
    arena_destroy(arena);
    return 0;
}

TEST(noise_state_reset_null_state_no_crash) {
    noise_state_reset_drone(NULL, 0);
    noise_state_reset_all(NULL);
    return 0;
}

TEST(constant_bias_broadcast_cycling) {
    /*
     * When bias.count < channel_count, values should cycle.
     * 2 bias values applied to 5 channels: [v0, v1, v0, v1, v0]
     */
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 5;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_CONSTANT_BIAS;
    config.groups[0].stages[0].params.bias.values[0] = 10.0f;
    config.groups[0].stages[0].params.bias.values[1] = 20.0f;
    config.groups[0].stages[0].params.bias.count = 2;

    float data[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    noise_apply(&config, NULL, data, NULL, 1, 5, 0.02f);

    ASSERT_FLOAT_EQ(data[0], 10.0f);  /* index 0 % 2 = 0 -> values[0] */
    ASSERT_FLOAT_EQ(data[1], 20.0f);  /* index 1 % 2 = 1 -> values[1] */
    ASSERT_FLOAT_EQ(data[2], 10.0f);  /* index 2 % 2 = 0 -> values[0] */
    ASSERT_FLOAT_EQ(data[3], 20.0f);  /* index 3 % 2 = 1 -> values[1] */
    ASSERT_FLOAT_EQ(data[4], 10.0f);  /* index 4 % 2 = 0 -> values[0] */

    arena_destroy(arena);
    return 0;
}

TEST(constant_bias_zero_count_no_modification) {
    /*
     * When bias.count == 0, apply_constant_bias returns early.
     */
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_CONSTANT_BIAS;
    config.groups[0].stages[0].params.bias.count = 0;

    float data[3] = {1.0f, 2.0f, 3.0f};
    float orig[3];
    memcpy(orig, data, sizeof(data));

    noise_apply(&config, NULL, data, NULL, 1, 3, 0.02f);

    for (int i = 0; i < 3; i++) {
        ASSERT_FLOAT_EQ(data[i], orig[i]);
    }

    arena_destroy(arena);
    return 0;
}

TEST(scale_factor_zero_error_identity) {
    /*
     * error=0 means factor=1.0, so no change.
     */
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_SCALE_FACTOR;
    config.groups[0].stages[0].params.scale.error = 0.0f;

    float data[3] = {1.5f, -3.7f, 0.0f};
    float orig[3];
    memcpy(orig, data, sizeof(data));

    noise_apply(&config, NULL, data, NULL, 1, 3, 0.02f);

    for (int i = 0; i < 3; i++) {
        ASSERT_FLOAT_EQ(data[i], orig[i]);
    }

    arena_destroy(arena);
    return 0;
}

TEST(scale_factor_negative_error_shrinks) {
    /*
     * error=-0.5 means factor=0.5, halves the values.
     */
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_SCALE_FACTOR;
    config.groups[0].stages[0].params.scale.error = -0.5f;

    float data[3] = {10.0f, -6.0f, 0.0f};
    noise_apply(&config, NULL, data, NULL, 1, 3, 0.02f);

    ASSERT_FLOAT_NEAR(data[0], 5.0f, 1e-5f);
    ASSERT_FLOAT_NEAR(data[1], -3.0f, 1e-5f);
    ASSERT_FLOAT_NEAR(data[2], 0.0f, 1e-5f);

    arena_destroy(arena);
    return 0;
}

TEST(noise_none_stage_is_noop) {
    /*
     * A pipeline with NOISE_NONE stages should not modify data.
     */
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stage_count = 3;
    config.groups[0].stages[0].type = NOISE_NONE;
    config.groups[0].stages[1].type = NOISE_NONE;
    config.groups[0].stages[2].type = NOISE_NONE;

    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float orig[4];
    memcpy(orig, data, sizeof(data));

    noise_apply(&config, NULL, data, NULL, 1, 4, 0.02f);

    ASSERT_TRUE(check_exact(data, orig, 4));

    arena_destroy(arena);
    return 0;
}

TEST(multi_drone_batch_output_layout) {
    /*
     * Verify that with 3 drones and output_size=4, noise applies
     * correctly to each drone's slice of the contiguous buffer.
     * Buffer layout: [drone0_ch0..ch3, drone1_ch0..ch3, drone2_ch0..ch3]
     */
    Arena* arena = arena_create(ARENA_SIZE);

    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_start = 1;
    config.groups[0].channel_count = 2;  /* channels [1,3) */
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_CONSTANT_BIAS;
    config.groups[0].stages[0].params.bias.values[0] = 100.0f;
    config.groups[0].stages[0].params.bias.count = 1;

    uint32_t agent_count = 3;
    uint32_t output_size = 4;
    float data[12];  /* 3 * 4 */
    for (uint32_t i = 0; i < 12; i++) {
        data[i] = (float)i;
    }
    float orig[12];
    memcpy(orig, data, sizeof(data));

    uint32_t indices[3] = {0, 1, 2};
    noise_apply(&config, NULL, data, indices, agent_count, output_size, 0.02f);

    for (uint32_t d = 0; d < agent_count; d++) {
        uint32_t base = d * output_size;
        /* Channel 0: untouched */
        ASSERT_FLOAT_EQ(data[base + 0], orig[base + 0]);
        /* Channels 1-2: biased by +100 */
        ASSERT_FLOAT_NEAR(data[base + 1], orig[base + 1] + 100.0f, 1e-5f);
        ASSERT_FLOAT_NEAR(data[base + 2], orig[base + 2] + 100.0f, 1e-5f);
        /* Channel 3: untouched */
        ASSERT_FLOAT_EQ(data[base + 3], orig[base + 3]);
    }

    arena_destroy(arena);
    return 0;
}

TEST(white_gaussian_requires_state_with_rng) {
    /*
     * White gaussian noise requires a valid RNG (from NoiseState).
     * Without state, it should skip the stage entirely (no modification).
     */
    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stage_count = 1;
    config.groups[0].stages[0].type = NOISE_WHITE_GAUSSIAN;
    config.groups[0].stages[0].params.white.stddev = 100.0f;

    float data[3] = {1.0f, 2.0f, 3.0f};
    float orig[3];
    memcpy(orig, data, sizeof(data));

    /* No state provided -> rng is NULL -> stage is skipped */
    noise_apply(&config, NULL, data, NULL, 1, 3, 0.02f);

    for (int i = 0; i < 3; i++) {
        ASSERT_FLOAT_EQ(data[i], orig[i]);
    }

    return 0;
}

TEST(pipeline_empty_stage_count_is_noop) {
    /*
     * A group with stage_count=0 should skip entirely.
     */
    NoiseConfig config = {0};
    config.group_count = 1;
    config.groups[0].channel_count = 0;
    config.groups[0].stage_count = 0;

    float data[3] = {1.0f, 2.0f, 3.0f};
    float orig[3];
    memcpy(orig, data, sizeof(data));

    noise_apply(&config, NULL, data, NULL, 1, 3, 0.02f);

    ASSERT_TRUE(check_exact(data, orig, 3));
    return 0;
}

TEST(noise_gaussian_distribution_properties) {
    /*
     * noise_gaussian(rng) uses Irwin-Hall with 2 uniform samples:
     * sum = U1 + U2 (in [0,2]), centered at 1, scaled by sqrt(6) ~ 2.449.
     * Output range: [-2.449, +2.449] (bounded, not true Gaussian).
     * Mean should be ~0, variance should be ~1.
     */
    Arena* arena = arena_create(ARENA_SIZE);
    PCG32 rng;
    pcg32_seed(&rng, 12345);

    double sum = 0.0;
    double sum_sq = 0.0;
    float min_val = 1e9f;
    float max_val = -1e9f;
    int N = 10000;

    for (int i = 0; i < N; i++) {
        float v = noise_gaussian(&rng);
        sum += v;
        sum_sq += (double)v * v;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    double mean = sum / N;
    double var = sum_sq / N - mean * mean;

    /* Mean near 0 */
    ASSERT_TRUE(fabs(mean) < 0.1);
    /* Variance near 1 (Irwin-Hall n=2 scaled to unit variance) */
    ASSERT_TRUE(var > 0.7);
    ASSERT_TRUE(var < 1.3);
    /* Bounded output -- not exceeding sqrt(6) ~ 2.449 */
    ASSERT_TRUE(min_val >= -2.5f);
    ASSERT_TRUE(max_val <= 2.5f);

    arena_destroy(arena);
    return 0;
}

TEST(max_noise_groups_is_four) {
    ASSERT_EQ(MAX_NOISE_GROUPS, 4);
    return 0;
}

TEST(max_noise_bias_values_is_eight) {
    ASSERT_EQ(MAX_NOISE_BIAS_VALUES, 8);
    return 0;
}

/* ============================================================================
 * Main Test Runner
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Noise Pipeline Deep");

    /* 1. Multi-group pipeline isolation */
    RUN_TEST(multi_group_isolation_4_groups);
    RUN_TEST(multi_group_no_cross_contamination);

    /* 2. Stage composition ordering */
    RUN_TEST(stage_ordering_bias_then_scale_vs_scale_then_bias);
    RUN_TEST(stage_ordering_four_stage_chain);

    /* 3. MAX_NOISE_STAGES pipeline */
    RUN_TEST(max_noise_stages_all_types);
    RUN_TEST(max_noise_stages_count_is_eight);

    /* 4. Channel group boundary validation */
    RUN_TEST(channel_boundary_clamped_when_exceeding_output);
    RUN_TEST(channel_start_at_output_size_means_zero_channels);
    RUN_TEST(channel_count_zero_means_all_channels);

    /* 5. Dropout edge cases */
    RUN_TEST(dropout_probability_zero_no_drops);
    RUN_TEST(dropout_probability_one_all_drops);
    RUN_TEST(dropout_replacement_value_preserved);

    /* 6. Quantization edge cases */
    RUN_TEST(quantization_step_zero_no_modification);
    RUN_TEST(quantization_step_negative_no_modification);
    RUN_TEST(quantization_very_large_step);
    RUN_TEST(quantization_exact_multiples_unchanged);

    /* 7. Saturation edge cases */
    RUN_TEST(saturation_min_equals_max);
    RUN_TEST(saturation_min_greater_than_max);
    RUN_TEST(saturation_values_within_range_unchanged);

    /* 8. Bias drift OU process */
    RUN_TEST(bias_drift_ou_stationary_distribution);
    RUN_TEST(bias_drift_large_displacement_reverts);
    RUN_TEST(bias_drift_adds_to_channel_data);

    /* 9. Distance-dependent noise */
    RUN_TEST(distance_dependent_zero_input_no_noise);
    RUN_TEST(distance_dependent_power_one_linear);

    /* 10. Per-drone RNG independence */
    RUN_TEST(per_drone_rng_divergence_statistical);
    RUN_TEST(per_drone_same_index_deterministic);
    RUN_TEST(per_drone_different_sensor_id_diverges);

    /* 11. noise_state_reset_drone */
    RUN_TEST(reset_drone_zeros_drift);
    RUN_TEST(reset_drone_reseeds_rng_deterministically);
    RUN_TEST(reset_drone_does_not_affect_other_drones);
    RUN_TEST(reset_all_resets_every_drone);

    /* 12. noise_type_from_string */
    RUN_TEST(noise_type_from_string_all_valid);
    RUN_TEST(noise_type_from_string_invalid);
    RUN_TEST(noise_type_from_string_null);
    RUN_TEST(noise_type_name_all_valid);
    RUN_TEST(noise_type_name_out_of_range);
    RUN_TEST(noise_type_roundtrip);

    /* Additional robustness */
    RUN_TEST(noise_apply_null_config_no_crash);
    RUN_TEST(noise_apply_null_data_no_crash);
    RUN_TEST(noise_apply_zero_agent_count_no_crash);
    RUN_TEST(noise_state_create_null_arena);
    RUN_TEST(noise_state_create_null_config);
    RUN_TEST(noise_state_create_zero_groups);
    RUN_TEST(noise_state_create_zero_drones);
    RUN_TEST(noise_state_reset_drone_out_of_range_no_crash);
    RUN_TEST(noise_state_reset_null_state_no_crash);
    RUN_TEST(constant_bias_broadcast_cycling);
    RUN_TEST(constant_bias_zero_count_no_modification);
    RUN_TEST(scale_factor_zero_error_identity);
    RUN_TEST(scale_factor_negative_error_shrinks);
    RUN_TEST(noise_none_stage_is_noop);
    RUN_TEST(multi_drone_batch_output_layout);
    RUN_TEST(white_gaussian_requires_state_with_rng);
    RUN_TEST(pipeline_empty_stage_count_is_noop);
    RUN_TEST(noise_gaussian_distribution_properties);
    RUN_TEST(max_noise_groups_is_four);
    RUN_TEST(max_noise_bias_values_is_eight);

    TEST_SUITE_END();
}
