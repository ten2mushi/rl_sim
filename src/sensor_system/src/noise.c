/**
 * Composable Sensor Noise Pipeline - Implementation
 *
 * Applies configurable noise chains to sensor output buffers.
 * Each noise group targets a channel range and chains multiple stages.
 * Per-drone independent RNG ensures sim2real fidelity.
 */

#include "noise.h"
#include <string.h>
#include <math.h>

/* ============================================================================
 * Gaussian Noise (Irwin-Hall approximation using 2 uniform samples)
 * ============================================================================ */

float noise_gaussian(PCG32* rng) {
    float sum = pcg32_float(rng) + pcg32_float(rng);
    return (sum - 1.0f) * 2.4494897f;  /* center and scale: sqrt(6) */
}

/* ============================================================================
 * Noise Type Name Table
 * ============================================================================ */

static const char* NOISE_TYPE_NAMES[NOISE_TYPE_COUNT] = {
    "none",
    "white_gaussian",
    "constant_bias",
    "bias_drift",
    "scale_factor",
    "distance_dependent",
    "quantization",
    "dropout",
    "saturation"
};

const char* noise_type_name(NoiseType type) {
    if (type >= NOISE_TYPE_COUNT) return "unknown";
    return NOISE_TYPE_NAMES[type];
}

NoiseType noise_type_from_string(const char* str) {
    if (!str) return NOISE_NONE;
    for (int i = 0; i < NOISE_TYPE_COUNT; i++) {
        if (strcmp(str, NOISE_TYPE_NAMES[i]) == 0) return (NoiseType)i;
    }
    return NOISE_NONE;
}

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

/** Golden ratio constant for seed mixing */
#define NOISE_SEED_MIX 0x9E3779B97F4A7C15ULL

NoiseState* noise_state_create(Arena* arena, const NoiseConfig* config,
                                uint32_t max_agents, uint32_t sensor_id) {
    if (!arena || !config || config->group_count == 0 || max_agents == 0) {
        return NULL;
    }

    NoiseState* state = arena_alloc_type(arena, NoiseState);
    if (!state) return NULL;
    memset(state, 0, sizeof(NoiseState));

    state->group_count = config->group_count;
    state->max_agents = max_agents;

    for (uint32_t g = 0; g < config->group_count; g++) {
        NoiseGroupState* gs = &state->groups[g];
        const NoisePipeline* pipe = &config->groups[g];

        /* Derive unique base seed from sensor_id and group index */
        gs->base_seed = 0x853c49e6748fea9bULL + (uint64_t)sensor_id * NOISE_SEED_MIX
                        + (uint64_t)g * 0xda3e39cb94b95bdbULL;

        /* Allocate per-drone RNGs */
        gs->rngs = arena_alloc_array(arena, PCG32, max_agents);
        if (!gs->rngs) return NULL;

        /* Seed each drone's RNG independently */
        for (uint32_t d = 0; d < max_agents; d++) {
            pcg32_seed_dual(&gs->rngs[d],
                           gs->base_seed,
                           (uint64_t)d * NOISE_SEED_MIX);
        }

        /* Count drift channels for OU state allocation */
        gs->drift_channels = 0;
        for (uint32_t s = 0; s < pipe->stage_count; s++) {
            if (pipe->stages[s].type == NOISE_BIAS_DRIFT) {
                /* Drift applies to all channels in the group */
                uint32_t ch_count = pipe->channel_count;
                if (ch_count == 0) ch_count = 1;  /* Will be resolved at apply time */
                gs->drift_channels += ch_count;
            }
        }

        if (gs->drift_channels > 0) {
            size_t drift_size = (size_t)max_agents * gs->drift_channels;
            gs->drift_state = arena_alloc_array(arena, float, drift_size);
            if (!gs->drift_state) return NULL;
            memset(gs->drift_state, 0, drift_size * sizeof(float));
        }
    }

    return state;
}

void noise_state_reset_drone(NoiseState* state, uint32_t agent_idx) {
    if (!state || agent_idx >= state->max_agents) return;

    for (uint32_t g = 0; g < state->group_count; g++) {
        NoiseGroupState* gs = &state->groups[g];

        /* Reseed RNG */
        pcg32_seed_dual(&gs->rngs[agent_idx],
                       gs->base_seed,
                       (uint64_t)agent_idx * NOISE_SEED_MIX);

        /* Zero drift state */
        if (gs->drift_state && gs->drift_channels > 0) {
            float* drift = gs->drift_state + (size_t)agent_idx * gs->drift_channels;
            memset(drift, 0, gs->drift_channels * sizeof(float));
        }
    }
}

void noise_state_reset_all(NoiseState* state) {
    if (!state) return;
    for (uint32_t d = 0; d < state->max_agents; d++) {
        noise_state_reset_drone(state, d);
    }
}

/* ============================================================================
 * Stage Application Functions
 * ============================================================================ */

static void apply_white_gaussian(float* channel_data, uint32_t count,
                                  float stddev, PCG32* rng) {
    for (uint32_t c = 0; c < count; c++) {
        channel_data[c] += noise_gaussian(rng) * stddev;
    }
}

static void apply_constant_bias(float* channel_data, uint32_t count,
                                 const float* values, uint32_t value_count) {
    if (value_count == 0) return;
    for (uint32_t c = 0; c < count; c++) {
        /* Broadcast: if value_count < count, cycle through values */
        channel_data[c] += values[c % value_count];
    }
}

static void apply_bias_drift(float* channel_data, uint32_t count,
                              float tau, float sigma, float dt,
                              float* drift, PCG32* rng) {
    /* Ornstein-Uhlenbeck process: dx = -x/tau*dt + sigma*sqrt(dt)*N(0,1) */
    float decay = dt / tau;
    float diffusion = sigma * sqrtf(dt);
    for (uint32_t c = 0; c < count; c++) {
        drift[c] += -drift[c] * decay + diffusion * noise_gaussian(rng);
        channel_data[c] += drift[c];
    }
}

static void apply_scale_factor(float* channel_data, uint32_t count, float error) {
    float factor = 1.0f + error;
    for (uint32_t c = 0; c < count; c++) {
        channel_data[c] *= factor;
    }
}

static void apply_distance_dependent(float* channel_data, uint32_t count,
                                      float coeff, float power, PCG32* rng) {
    for (uint32_t c = 0; c < count; c++) {
        float val = channel_data[c];
        float abs_val = fabsf(val);
        float stddev = coeff * powf(abs_val, power);
        channel_data[c] += noise_gaussian(rng) * stddev;
    }
}

static void apply_quantization(float* channel_data, uint32_t count, float step) {
    if (step <= 0.0f) return;
    float inv_step = 1.0f / step;
    for (uint32_t c = 0; c < count; c++) {
        channel_data[c] = roundf(channel_data[c] * inv_step) * step;
    }
}

static void apply_dropout(float* channel_data, uint32_t count,
                           float probability, float replacement, PCG32* rng) {
    for (uint32_t c = 0; c < count; c++) {
        if (pcg32_float(rng) < probability) {
            channel_data[c] = replacement;
        }
    }
}

static void apply_saturation(float* channel_data, uint32_t count,
                              float min_val, float max_val) {
    for (uint32_t c = 0; c < count; c++) {
        channel_data[c] = clampf(channel_data[c], min_val, max_val);
    }
}

/* ============================================================================
 * Core Application
 * ============================================================================ */

void noise_apply(const NoiseConfig* config, NoiseState* state,
                 float* data, const uint32_t* agent_indices,
                 uint32_t agent_count, uint32_t output_size, float dt) {
    if (!config || !data || config->group_count == 0 || agent_count == 0) {
        return;
    }

    for (uint32_t g = 0; g < config->group_count; g++) {
        const NoisePipeline* pipe = &config->groups[g];
        if (pipe->stage_count == 0) continue;

        NoiseGroupState* gs = (state && g < state->group_count) ? &state->groups[g] : NULL;

        /* Resolve channel range */
        uint32_t ch_start = pipe->channel_start;
        uint32_t ch_count = pipe->channel_count;
        if (ch_start >= output_size) continue;  /* Entirely out of range */
        if (ch_count == 0) ch_count = output_size - ch_start;  /* 0 = remaining channels */
        if (ch_start + ch_count > output_size) {
            ch_count = output_size - ch_start;
        }

        /* Process each drone */
        for (uint32_t i = 0; i < agent_count; i++) {
            uint32_t agent_idx = agent_indices ? agent_indices[i] : i;
            float* drone_data = data + (size_t)i * output_size + ch_start;

            PCG32* rng = (gs && gs->rngs) ? &gs->rngs[agent_idx] : NULL;

            uint32_t drift_offset = 0;

            /* Apply stages in order */
            for (uint32_t s = 0; s < pipe->stage_count; s++) {
                const NoiseStage* stage = &pipe->stages[s];

                switch (stage->type) {
                    case NOISE_WHITE_GAUSSIAN:
                        if (rng) apply_white_gaussian(drone_data, ch_count,
                                                       stage->params.white.stddev, rng);
                        break;

                    case NOISE_CONSTANT_BIAS:
                        apply_constant_bias(drone_data, ch_count,
                                           stage->params.bias.values,
                                           stage->params.bias.count);
                        break;

                    case NOISE_BIAS_DRIFT:
                        if (rng && gs && gs->drift_state) {
                            float* drift = gs->drift_state +
                                          (size_t)agent_idx * gs->drift_channels + drift_offset;
                            apply_bias_drift(drone_data, ch_count,
                                           stage->params.drift.tau,
                                           stage->params.drift.sigma,
                                           dt, drift, rng);
                            drift_offset += ch_count;
                        }
                        break;

                    case NOISE_SCALE_FACTOR:
                        apply_scale_factor(drone_data, ch_count,
                                          stage->params.scale.error);
                        break;

                    case NOISE_DISTANCE_DEPENDENT:
                        if (rng) apply_distance_dependent(drone_data, ch_count,
                                                           stage->params.distance.coeff,
                                                           stage->params.distance.power, rng);
                        break;

                    case NOISE_QUANTIZATION:
                        apply_quantization(drone_data, ch_count,
                                          stage->params.quantize.step);
                        break;

                    case NOISE_DROPOUT:
                        if (rng) apply_dropout(drone_data, ch_count,
                                               stage->params.dropout.probability,
                                               stage->params.dropout.replacement, rng);
                        break;

                    case NOISE_SATURATION:
                        apply_saturation(drone_data, ch_count,
                                        stage->params.saturate.min_val,
                                        stage->params.saturate.max_val);
                        break;

                    case NOISE_NONE:
                    case NOISE_TYPE_COUNT:
                    default:
                        break;
                }
            }
        }
    }
}
