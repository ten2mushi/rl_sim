/**
 * Composable Sensor Noise Pipeline
 *
 * Provides configurable noise chains (white gaussian, bias, drift, scale factor,
 * distance-dependent, quantization, dropout, saturation) applied per-channel-group
 * after sensor sampling. Per-drone independent RNG for sim2real fidelity.
 *
 * Dependencies: foundation.h (Arena, PCG32, Vec3)
 */

#ifndef NOISE_H
#define NOISE_H

#include "foundation.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Constants
 * ============================================================================ */

#define MAX_NOISE_STAGES 8
#define MAX_NOISE_GROUPS 4
#define MAX_NOISE_BIAS_VALUES 8

/* ============================================================================
 * Noise Type Enumeration
 * ============================================================================ */

typedef enum NoiseType {
    NOISE_NONE = 0,
    NOISE_WHITE_GAUSSIAN,      /* additive N(0, stddev) */
    NOISE_CONSTANT_BIAS,       /* additive per-element bias values[count] */
    NOISE_BIAS_DRIFT,          /* Ornstein-Uhlenbeck: dx = -x/tau*dt + sigma*sqrt(dt)*N(0,1) */
    NOISE_SCALE_FACTOR,        /* multiplicative: v *= (1 + error) */
    NOISE_DISTANCE_DEPENDENT,  /* stddev = coeff * |v|^power */
    NOISE_QUANTIZATION,        /* v = round(v/step)*step */
    NOISE_DROPOUT,             /* P(probability) -> replacement value */
    NOISE_SATURATION,          /* clamp to [min_val, max_val] */
    NOISE_TYPE_COUNT
} NoiseType;

/* ============================================================================
 * Noise Parameters (union for space efficiency)
 * ============================================================================ */

typedef union NoiseParams {
    struct { float stddev; }                         white;
    struct { float values[MAX_NOISE_BIAS_VALUES]; uint32_t count; } bias;
    struct { float tau; float sigma; }               drift;
    struct { float error; }                          scale;
    struct { float coeff; float power; }             distance;
    struct { float step; }                           quantize;
    struct { float probability; float replacement; } dropout;
    struct { float min_val; float max_val; }         saturate;
} NoiseParams;

/* ============================================================================
 * Noise Stage (single operation in a pipeline)
 * ============================================================================ */

typedef struct NoiseStage {
    NoiseType type;
    NoiseParams params;
} NoiseStage;

/* ============================================================================
 * Noise Pipeline (chain of stages applied to a channel group)
 * ============================================================================ */

typedef struct NoisePipeline {
    NoiseStage stages[MAX_NOISE_STAGES];
    uint32_t stage_count;
    uint32_t channel_start;   /* 0-based index into sensor output */
    uint32_t channel_count;   /* 0 = all channels */
} NoisePipeline;

/* ============================================================================
 * Noise Configuration (per-sensor, multiple channel groups)
 * ============================================================================ */

typedef struct NoiseConfig {
    NoisePipeline groups[MAX_NOISE_GROUPS];
    uint32_t group_count;
} NoiseConfig;

/* ============================================================================
 * Noise State (runtime state for stateful noise types)
 * ============================================================================ */

typedef struct NoiseGroupState {
    PCG32*   rngs;            /* [max_agents] per-drone independent RNG */
    float*   drift_state;     /* [max_agents * drift_channels] OU process state */
    uint32_t drift_channels;  /* number of channels with NOISE_BIAS_DRIFT stages */
    uint64_t base_seed;       /* for deterministic reseed */
} NoiseGroupState;

typedef struct NoiseState {
    NoiseGroupState groups[MAX_NOISE_GROUPS];
    uint32_t group_count;
    uint32_t max_agents;
} NoiseState;

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

/**
 * Create noise state from config. Returns NULL if group_count == 0.
 *
 * @param arena       Persistent arena for allocation
 * @param config      Noise configuration
 * @param max_agents  Maximum number of drones
 * @param sensor_id   Sensor ID (used to derive unique seeds)
 * @return Allocated NoiseState, or NULL
 */
NoiseState* noise_state_create(Arena* arena, const NoiseConfig* config,
                                uint32_t max_agents, uint32_t sensor_id);

/**
 * Reset noise state for a single drone (reseed RNG, zero drift).
 */
void noise_state_reset_drone(NoiseState* state, uint32_t agent_idx);

/**
 * Reset noise state for all drones.
 */
void noise_state_reset_all(NoiseState* state);

/* ============================================================================
 * Core Application API
 * ============================================================================ */

/**
 * Apply noise pipeline in-place to a batch output buffer.
 *
 * @param config        Noise configuration (pipeline definition)
 * @param state         Noise state (RNGs, drift state) - can be NULL if no stateful noise
 * @param data          Batch output buffer [agent_count * output_size]
 * @param agent_indices Which global drone indices correspond to each batch entry
 * @param agent_count   Number of drones in this batch
 * @param output_size   Number of floats per drone
 * @param dt            Simulation timestep (for drift integration)
 */
void noise_apply(const NoiseConfig* config, NoiseState* state,
                 float* data, const uint32_t* agent_indices,
                 uint32_t agent_count, uint32_t output_size, float dt);

/* ============================================================================
 * Utility API
 * ============================================================================ */

/** Get human-readable name for a noise type. */
const char* noise_type_name(NoiseType type);

/** Parse noise type from string (e.g., "white_gaussian"). Returns NOISE_NONE on failure. */
NoiseType noise_type_from_string(const char* str);

/** Fast Gaussian approximation (Irwin-Hall, 2 uniform samples). */
float noise_gaussian(PCG32* rng);

#ifdef __cplusplus
}
#endif

#endif /* NOISE_H */
