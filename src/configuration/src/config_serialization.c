/**
 * Configuration Serialization - Save, Hash, Export, Clone
 *
 * Provides functions for:
 * - Saving configuration to TOML files
 * - Exporting to JSON for logging
 * - FNV-1a hashing for change detection
 * - Deep copy/clone operations
 * - Configuration comparison
 * - Debug printing
 */

#include "configuration.h"
#include "platform_quadcopter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * FNV-1a Hash Implementation
 * ============================================================================ */

/* FNV-1a hash constants */
#define FNV_OFFSET 14695981039346656037ULL
#define FNV_PRIME 1099511628211ULL

/**
 * FNV-1a hash function for arbitrary data.
 */
static uint64_t fnv1a_hash(const void* data, size_t len) {
    const uint8_t* bytes = (const uint8_t*)data;
    uint64_t hash = FNV_OFFSET;
    for (size_t i = 0; i < len; i++) {
        hash ^= bytes[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

uint64_t config_hash(const Config* config) {
    if (!config) return 0;

    uint64_t hash = FNV_OFFSET;

    /* Hash platform config (excluding pointer fields) */
    /* Hash type, name, model_path, and common numeric fields */
    hash ^= fnv1a_hash(config->platform.type, sizeof(config->platform.type));
    hash *= FNV_PRIME;
    hash ^= fnv1a_hash(config->platform.name, sizeof(config->platform.name));
    hash *= FNV_PRIME;
    hash ^= fnv1a_hash(config->platform.model_path, sizeof(config->platform.model_path));
    hash *= FNV_PRIME;
    hash ^= fnv1a_hash(&config->platform.mass, sizeof(float));
    hash *= FNV_PRIME;
    hash ^= fnv1a_hash(&config->platform.ixx, sizeof(float) * 3); /* ixx, iyy, izz */
    hash *= FNV_PRIME;
    hash ^= fnv1a_hash(&config->platform.collision_radius, sizeof(float) * 5); /* collision_radius through max_tilt_angle */
    hash *= FNV_PRIME;
    hash ^= fnv1a_hash(config->platform.color, sizeof(float) * 3);
    hash *= FNV_PRIME;
    hash ^= fnv1a_hash(&config->platform.scale, sizeof(float));
    hash *= FNV_PRIME;

    /* Hash platform_specific CONTENTS (not the pointer!) */
    if (config->platform.platform_specific && config->platform.platform_specific_size > 0) {
        hash ^= fnv1a_hash(config->platform.platform_specific,
                           config->platform.platform_specific_size);
        hash *= FNV_PRIME;
    }

    /* Hash environment config */
    hash ^= fnv1a_hash(&config->environment, sizeof(EnvironmentConfig));
    hash *= FNV_PRIME;

    /* Hash physics config */
    hash ^= fnv1a_hash(&config->physics, sizeof(ConfigPhysics));
    hash *= FNV_PRIME;

    /* Hash reward config */
    hash ^= fnv1a_hash(&config->reward, sizeof(RewardConfigData));
    hash *= FNV_PRIME;

    /* Hash training config */
    hash ^= fnv1a_hash(&config->training, sizeof(TrainingConfig));
    hash *= FNV_PRIME;

    /* Hash sensors (variable length) */
    if (config->sensors && config->num_sensors > 0) {
        hash ^= fnv1a_hash(config->sensors,
                          config->num_sensors * sizeof(SensorConfigEntry));
        hash *= FNV_PRIME;
    }

    return hash;
}

/* ============================================================================
 * TOML Serialization
 * ============================================================================ */

int config_save(const char* path, const Config* config) {
    if (!path || !config) return -1;

    FILE* fp = fopen(path, "w");
    if (!fp) return -1;

    const PlatformConfig* plat = &config->platform;

    /* [platform] section */
    fprintf(fp, "[platform]\n");
    fprintf(fp, "type = \"%s\"\n", plat->type);
    fprintf(fp, "name = \"%s\"\n", plat->name);
    if (plat->model_path[0]) {
        fprintf(fp, "model_path = \"%s\"\n", plat->model_path);
    }
    fprintf(fp, "mass = %.6f\n", plat->mass);
    fprintf(fp, "ixx = %.6e\n", plat->ixx);
    fprintf(fp, "iyy = %.6e\n", plat->iyy);
    fprintf(fp, "izz = %.6e\n", plat->izz);
    fprintf(fp, "collision_radius = %.4f\n", plat->collision_radius);
    fprintf(fp, "max_velocity = %.1f\n", plat->max_velocity);
    fprintf(fp, "max_angular_velocity = %.1f\n", plat->max_angular_velocity);
    fprintf(fp, "max_tilt_angle = %.2f\n", plat->max_tilt_angle);
    fprintf(fp, "color = [%.2f, %.2f, %.2f]\n",
            plat->color[0], plat->color[1], plat->color[2]);
    fprintf(fp, "scale = %.2f\n", plat->scale);

    /* [platform.quadcopter] sub-section */
    if (strcmp(plat->type, "quadcopter") == 0 && plat->platform_specific) {
        const QuadcopterConfig* quad = (const QuadcopterConfig*)plat->platform_specific;
        fprintf(fp, "\n[platform.quadcopter]\n");
        fprintf(fp, "arm_length = %.4f\n", quad->arm_length);
        fprintf(fp, "k_thrust = %.6e\n", quad->k_thrust);
        fprintf(fp, "k_torque = %.6e\n", quad->k_torque);
        fprintf(fp, "motor_tau = %.4f\n", quad->motor_tau);
        fprintf(fp, "max_rpm = %.1f\n", quad->max_rpm);
        fprintf(fp, "k_drag = %.6f\n", quad->k_drag);
        fprintf(fp, "k_drag_angular = %.6f\n", quad->k_ang_damp);
    }

    /* [environment] section */
    fprintf(fp, "\n[environment]\n");
    fprintf(fp, "num_envs = %u\n", config->environment.num_envs);
    fprintf(fp, "agents_per_env = %u\n", config->environment.agents_per_env);
    fprintf(fp, "world_size = [%.1f, %.1f, %.1f]\n",
            config->environment.world_size[0],
            config->environment.world_size[1],
            config->environment.world_size[2]);
    fprintf(fp, "world_origin = [%.1f, %.1f, %.1f]\n",
            config->environment.world_origin[0],
            config->environment.world_origin[1],
            config->environment.world_origin[2]);
    fprintf(fp, "voxel_size = %.2f\n", config->environment.voxel_size);
    fprintf(fp, "max_bricks = %u\n", config->environment.max_bricks);
    fprintf(fp, "spawn_radius = %.1f\n", config->environment.spawn_radius);
    fprintf(fp, "spawn_height_min = %.1f\n", config->environment.spawn_height_min);
    fprintf(fp, "spawn_height_max = %.1f\n", config->environment.spawn_height_max);
    fprintf(fp, "min_separation = %.1f\n", config->environment.min_separation);
    fprintf(fp, "max_episode_steps = %u\n", config->environment.max_episode_steps);
    fprintf(fp, "auto_reset = %s\n", config->environment.auto_reset ? "true" : "false");
    fprintf(fp, "world_type = \"%s\"\n", config->environment.world_type);
    fprintf(fp, "num_obstacles = %u\n", config->environment.num_obstacles);
    fprintf(fp, "seed = %u\n", config->environment.seed);

    /* [physics] section */
    fprintf(fp, "\n[physics]\n");
    fprintf(fp, "timestep = %.6f\n", config->physics.timestep);
    fprintf(fp, "substeps = %u\n", config->physics.substeps);
    fprintf(fp, "gravity = %.2f\n", config->physics.gravity);
    fprintf(fp, "integrator = \"%s\"\n", config->physics.integrator);
    fprintf(fp, "velocity_clamp = %.1f\n", config->physics.velocity_clamp);
    fprintf(fp, "angular_velocity_clamp = %.1f\n", config->physics.angular_velocity_clamp);
    fprintf(fp, "normalize_quaternions = %s\n",
            config->physics.normalize_quaternions ? "true" : "false");
    fprintf(fp, "enable_ground_effect = %s\n",
            config->physics.enable_ground_effect ? "true" : "false");
    fprintf(fp, "ground_effect_height = %.2f\n", config->physics.ground_effect_height);
    fprintf(fp, "ground_effect_strength = %.2f\n", config->physics.ground_effect_strength);
    fprintf(fp, "dt_variance = %.4f\n", config->physics.dt_variance);
    fprintf(fp, "mass_variance = %.4f\n", config->physics.mass_variance);
    fprintf(fp, "thrust_variance = %.4f\n", config->physics.thrust_variance);

    /* [reward] section */
    fprintf(fp, "\n[reward]\n");
    fprintf(fp, "task = \"%s\"\n", config->reward.task);
    fprintf(fp, "distance_scale = %.2f\n", config->reward.distance_scale);
    fprintf(fp, "distance_exp = %.2f\n", config->reward.distance_exp);
    fprintf(fp, "reach_bonus = %.1f\n", config->reward.reach_bonus);
    fprintf(fp, "reach_radius = %.2f\n", config->reward.reach_radius);
    fprintf(fp, "velocity_match_scale = %.2f\n", config->reward.velocity_match_scale);
    fprintf(fp, "uprightness_scale = %.2f\n", config->reward.uprightness_scale);
    fprintf(fp, "energy_scale = %.4f\n", config->reward.energy_scale);
    fprintf(fp, "jerk_scale = %.6f\n", config->reward.jerk_scale);
    fprintf(fp, "collision_penalty = %.1f\n", config->reward.collision_penalty);
    fprintf(fp, "world_collision_penalty = %.1f\n", config->reward.world_collision_penalty);
    fprintf(fp, "drone_collision_penalty = %.1f\n", config->reward.drone_collision_penalty);
    fprintf(fp, "alive_bonus = %.3f\n", config->reward.alive_bonus);
    fprintf(fp, "success_bonus = %.1f\n", config->reward.success_bonus);
    fprintf(fp, "reward_min = %.1f\n", config->reward.reward_min);
    fprintf(fp, "reward_max = %.1f\n", config->reward.reward_max);

    /* [training] section */
    fprintf(fp, "\n[training]\n");
    fprintf(fp, "algorithm = \"%s\"\n", config->training.algorithm);
    fprintf(fp, "learning_rate = %.2e\n", config->training.learning_rate);
    fprintf(fp, "gamma = %.4f\n", config->training.gamma);
    fprintf(fp, "gae_lambda = %.4f\n", config->training.gae_lambda);
    fprintf(fp, "clip_range = %.2f\n", config->training.clip_range);
    fprintf(fp, "entropy_coef = %.4f\n", config->training.entropy_coef);
    fprintf(fp, "value_coef = %.2f\n", config->training.value_coef);
    fprintf(fp, "max_grad_norm = %.2f\n", config->training.max_grad_norm);
    fprintf(fp, "batch_size = %u\n", config->training.batch_size);
    fprintf(fp, "num_epochs = %u\n", config->training.num_epochs);
    fprintf(fp, "rollout_length = %u\n", config->training.rollout_length);
    fprintf(fp, "log_interval = %u\n", config->training.log_interval);
    fprintf(fp, "save_interval = %u\n", config->training.save_interval);
    fprintf(fp, "checkpoint_dir = \"%s\"\n", config->training.checkpoint_dir);

    /* [[sensors]] array */
    for (uint32_t i = 0; i < config->num_sensors; i++) {
        const SensorConfigEntry* s = &config->sensors[i];
        fprintf(fp, "\n[[sensors]]\n");
        fprintf(fp, "type = \"%s\"\n", s->type);
        fprintf(fp, "name = \"%s\"\n", s->name);
        fprintf(fp, "position = [%.4f, %.4f, %.4f]\n",
                s->position[0], s->position[1], s->position[2]);
        fprintf(fp, "orientation = [%.4f, %.4f, %.4f, %.4f]\n",
                s->orientation[0], s->orientation[1], s->orientation[2], s->orientation[3]);
        fprintf(fp, "sample_rate = %.1f\n", s->sample_rate);

        /* Noise groups */
        for (uint32_t g = 0; g < s->num_noise_groups; g++) {
            const NoiseGroupEntry* ge = &s->noise_groups[g];
            fprintf(fp, "\n[[sensors.noise_groups]]\n");
            if (ge->channels[0] != 0 || ge->channels[1] != 0) {
                fprintf(fp, "channels = [%u, %u]\n", ge->channels[0], ge->channels[1]);
            }
            for (uint32_t st = 0; st < ge->num_stages; st++) {
                const NoiseStageEntry* se = &ge->stages[st];
                fprintf(fp, "[[sensors.noise_groups.stages]]\n");
                fprintf(fp, "type = \"%s\"\n", se->type);
                if (strcmp(se->type, "white_gaussian") == 0) {
                    fprintf(fp, "stddev = %.6f\n", se->stddev);
                } else if (strcmp(se->type, "constant_bias") == 0) {
                    fprintf(fp, "values = [");
                    for (uint32_t v = 0; v < se->value_count; v++) {
                        fprintf(fp, "%.6f%s", se->values[v], v + 1 < se->value_count ? ", " : "");
                    }
                    fprintf(fp, "]\n");
                } else if (strcmp(se->type, "bias_drift") == 0) {
                    fprintf(fp, "tau = %.4f\nsigma = %.6f\n", se->tau, se->sigma);
                } else if (strcmp(se->type, "scale_factor") == 0) {
                    fprintf(fp, "error = %.6f\n", se->error);
                } else if (strcmp(se->type, "distance_dependent") == 0) {
                    fprintf(fp, "coeff = %.6f\npower = %.2f\n", se->coeff, se->power);
                } else if (strcmp(se->type, "quantization") == 0) {
                    fprintf(fp, "step = %.6f\n", se->step);
                } else if (strcmp(se->type, "dropout") == 0) {
                    fprintf(fp, "probability = %.4f\nreplacement = %.4f\n", se->probability, se->replacement);
                } else if (strcmp(se->type, "saturation") == 0) {
                    fprintf(fp, "min_val = %.4f\nmax_val = %.4f\n", se->min_val, se->max_val);
                }
            }
        }

        /* Type-specific fields */
        if (strcmp(s->type, "tof") == 0) {
            fprintf(fp, "max_range = %.1f\n", s->max_range);
        } else if (strcmp(s->type, "lidar_2d") == 0) {
            fprintf(fp, "num_rays = %u\n", s->num_rays);
            fprintf(fp, "fov = %.4f\n", s->fov);
            fprintf(fp, "max_range = %.1f\n", s->max_range);
        } else if (strcmp(s->type, "lidar_3d") == 0) {
            fprintf(fp, "num_rays = %u\n", s->num_rays);
            fprintf(fp, "fov = %.4f\n", s->fov);
            fprintf(fp, "fov_vertical = %.4f\n", s->fov_vertical);
            fprintf(fp, "vertical_layers = %u\n", s->vertical_layers);
            fprintf(fp, "max_range = %.1f\n", s->max_range);
        } else if (strcmp(s->type, "camera_rgb") == 0 ||
                   strcmp(s->type, "camera_depth") == 0 ||
                   strcmp(s->type, "camera_segmentation") == 0) {
            fprintf(fp, "width = %u\n", s->width);
            fprintf(fp, "height = %u\n", s->height);
            fprintf(fp, "near_clip = %.2f\n", s->near_clip);
            fprintf(fp, "far_clip = %.1f\n", s->far_clip);
            if (strcmp(s->type, "camera_segmentation") == 0) {
                fprintf(fp, "num_classes = %u\n", s->num_classes);
            }
        } else if (strcmp(s->type, "neighbor") == 0) {
            fprintf(fp, "k_neighbors = %u\n", s->k_neighbors);
            fprintf(fp, "max_range = %.1f\n", s->max_range);
        }
    }

    fclose(fp);
    return 0;
}

/* ============================================================================
 * JSON Export
 * ============================================================================ */

int config_to_json(const Config* config, char* buffer, size_t buffer_size) {
    if (!config || !buffer || buffer_size == 0) return -1;

    const PlatformConfig* plat = &config->platform;

    int written = snprintf(buffer, buffer_size,
        "{\n"
        "  \"platform\": {\n"
        "    \"type\": \"%s\",\n"
        "    \"name\": \"%s\",\n"
        "    \"mass\": %.6f\n"
        "  },\n"
        "  \"environment\": {\n"
        "    \"num_envs\": %u,\n"
        "    \"agents_per_env\": %u,\n"
        "    \"max_episode_steps\": %u\n"
        "  },\n"
        "  \"physics\": {\n"
        "    \"timestep\": %.6f,\n"
        "    \"gravity\": %.2f,\n"
        "    \"integrator\": \"%s\"\n"
        "  },\n"
        "  \"reward\": {\n"
        "    \"task\": \"%s\"\n"
        "  },\n"
        "  \"training\": {\n"
        "    \"algorithm\": \"%s\",\n"
        "    \"learning_rate\": %.2e\n"
        "  },\n"
        "  \"num_sensors\": %u,\n"
        "  \"config_hash\": \"0x%016llx\"\n"
        "}",
        plat->type,
        plat->name,
        plat->mass,
        config->environment.num_envs,
        config->environment.agents_per_env,
        config->environment.max_episode_steps,
        config->physics.timestep,
        config->physics.gravity,
        config->physics.integrator,
        config->reward.task,
        config->training.algorithm,
        config->training.learning_rate,
        config->num_sensors,
        (unsigned long long)config->config_hash);

    return (written >= 0 && (size_t)written < buffer_size) ? 0 : -1;
}

/* ============================================================================
 * Configuration Comparison
 * ============================================================================ */

int config_compare(const Config* a, const Config* b) {
    if (!a || !b) return 1;

    /* Quick hash comparison first */
    uint64_t hash_a = config_hash(a);
    uint64_t hash_b = config_hash(b);
    if (hash_a != hash_b) return 1;

    /* Deep comparison - can't memcmp PlatformConfig because of pointer field.
     * Compare field by field for the platform section. */
    if (memcmp(a->platform.type, b->platform.type, sizeof(a->platform.type)) != 0) return 1;
    if (memcmp(a->platform.name, b->platform.name, sizeof(a->platform.name)) != 0) return 1;
    if (memcmp(a->platform.model_path, b->platform.model_path, sizeof(a->platform.model_path)) != 0) return 1;
    if (a->platform.mass != b->platform.mass) return 1;
    if (a->platform.ixx != b->platform.ixx) return 1;
    if (a->platform.iyy != b->platform.iyy) return 1;
    if (a->platform.izz != b->platform.izz) return 1;
    if (a->platform.collision_radius != b->platform.collision_radius) return 1;
    if (a->platform.max_velocity != b->platform.max_velocity) return 1;
    if (a->platform.max_angular_velocity != b->platform.max_angular_velocity) return 1;
    if (a->platform.max_tilt_angle != b->platform.max_tilt_angle) return 1;
    if (memcmp(a->platform.color, b->platform.color, sizeof(a->platform.color)) != 0) return 1;
    if (a->platform.scale != b->platform.scale) return 1;
    if (a->platform.platform_specific_size != b->platform.platform_specific_size) return 1;

    /* Compare platform_specific contents */
    if (a->platform.platform_specific_size > 0) {
        if (!a->platform.platform_specific || !b->platform.platform_specific) return 1;
        if (memcmp(a->platform.platform_specific, b->platform.platform_specific,
                   a->platform.platform_specific_size) != 0) return 1;
    }

    if (memcmp(&a->environment, &b->environment, sizeof(EnvironmentConfig)) != 0) return 1;
    if (memcmp(&a->physics, &b->physics, sizeof(ConfigPhysics)) != 0) return 1;
    if (memcmp(&a->reward, &b->reward, sizeof(RewardConfigData)) != 0) return 1;
    if (memcmp(&a->training, &b->training, sizeof(TrainingConfig)) != 0) return 1;

    if (a->num_sensors != b->num_sensors) return 1;
    if (a->num_sensors > 0) {
        if (!a->sensors || !b->sensors) return 1;
        if (memcmp(a->sensors, b->sensors,
                   a->num_sensors * sizeof(SensorConfigEntry)) != 0) return 1;
    }

    return 0;  /* Identical */
}

/* ============================================================================
 * Configuration Clone
 * ============================================================================ */

void config_clone(const Config* src, Config* dst, Arena* arena) {
    if (!src || !dst) return;

    /* Copy fixed-size fields */
    *dst = *src;

    /* Deep copy sensors array */
    if (src->num_sensors > 0 && src->sensors) {
        size_t sensors_size = src->num_sensors * sizeof(SensorConfigEntry);
        if (arena) {
            dst->sensors = (SensorConfigEntry*)arena_alloc_aligned(arena, sensors_size, 8);
        } else {
            dst->sensors = (SensorConfigEntry*)malloc(sensors_size);
        }
        if (dst->sensors) {
            memcpy(dst->sensors, src->sensors, sensors_size);
        } else {
            dst->num_sensors = 0;
        }
    } else {
        dst->sensors = NULL;
        dst->num_sensors = 0;
    }

    /* Deep copy platform_specific */
    if (src->platform.platform_specific && src->platform.platform_specific_size > 0) {
        void* new_ps = malloc(src->platform.platform_specific_size);
        if (new_ps) {
            memcpy(new_ps, src->platform.platform_specific, src->platform.platform_specific_size);
            dst->platform.platform_specific = new_ps;
        } else {
            dst->platform.platform_specific = NULL;
            dst->platform.platform_specific_size = 0;
        }
    } else {
        dst->platform.platform_specific = NULL;
        dst->platform.platform_specific_size = 0;
    }

    /* Clear path and recompute hash */
    memset(dst->config_path, 0, sizeof(dst->config_path));
    dst->config_hash = config_hash(dst);
}

/* ============================================================================
 * Configuration Print
 * ============================================================================ */

void config_print(const Config* config) {
    if (!config) {
        printf("Config: NULL\n");
        return;
    }

    printf("=== Configuration ===\n");
    printf("Source: %s\n", config->config_path[0] ? config->config_path : "(in-memory)");
    printf("Hash: 0x%016llx\n\n", (unsigned long long)config->config_hash);

    const PlatformConfig* plat = &config->platform;
    printf("[platform]\n");
    printf("  type: %s\n", plat->type);
    printf("  name: %s\n", plat->name);
    printf("  mass: %.4f kg\n", plat->mass);
    printf("  inertia: [%.2e, %.2e, %.2e] kg*m^2\n",
           plat->ixx, plat->iyy, plat->izz);
    printf("  collision_radius: %.4f m\n", plat->collision_radius);

    if (strcmp(plat->type, "quadcopter") == 0 && plat->platform_specific) {
        const QuadcopterConfig* quad = (const QuadcopterConfig*)plat->platform_specific;
        printf("  [quadcopter]\n");
        printf("    arm_length: %.4f m\n", quad->arm_length);
        printf("    k_thrust: %.2e N/(rad/s)^2\n", quad->k_thrust);
        printf("    k_torque: %.2e N*m/(rad/s)^2\n", quad->k_torque);
        printf("    motor_tau: %.4f s\n", quad->motor_tau);
        printf("    max_rpm: %.0f\n", quad->max_rpm);
    }

    printf("\n[environment]\n");
    printf("  num_envs: %u\n", config->environment.num_envs);
    printf("  agents_per_env: %u\n", config->environment.agents_per_env);
    printf("  world_size: [%.1f, %.1f, %.1f] m\n",
           config->environment.world_size[0],
           config->environment.world_size[1],
           config->environment.world_size[2]);
    printf("  world_type: %s\n", config->environment.world_type);
    printf("  max_episode_steps: %u\n", config->environment.max_episode_steps);
    printf("  seed: %u\n", config->environment.seed);

    printf("\n[physics]\n");
    printf("  timestep: %.6f s (%.0f Hz)\n",
           config->physics.timestep, 1.0f / config->physics.timestep);
    printf("  substeps: %u (effective %.0f Hz)\n",
           config->physics.substeps,
           (float)config->physics.substeps / config->physics.timestep);
    printf("  gravity: %.2f m/s^2\n", config->physics.gravity);
    printf("  integrator: %s\n", config->physics.integrator);
    printf("  ground_effect: %s (height=%.2f, strength=%.2f)\n",
           config->physics.enable_ground_effect ? "enabled" : "disabled",
           config->physics.ground_effect_height,
           config->physics.ground_effect_strength);

    printf("\n[reward]\n");
    printf("  task: %s\n", config->reward.task);
    printf("  distance_scale: %.2f\n", config->reward.distance_scale);
    printf("  collision_penalty: %.1f\n", config->reward.collision_penalty);
    printf("  alive_bonus: %.3f\n", config->reward.alive_bonus);

    printf("\n[training]\n");
    printf("  algorithm: %s\n", config->training.algorithm);
    printf("  learning_rate: %.2e\n", config->training.learning_rate);
    printf("  gamma: %.4f\n", config->training.gamma);
    printf("  batch_size: %u\n", config->training.batch_size);

    printf("\nSensors: %u configured\n", config->num_sensors);
    for (uint32_t i = 0; i < config->num_sensors; i++) {
        printf("  [%u] %s (%s) @ [%.2f, %.2f, %.2f]\n",
               i, config->sensors[i].name, config->sensors[i].type,
               config->sensors[i].position[0],
               config->sensors[i].position[1],
               config->sensors[i].position[2]);
    }
    printf("\n");
}
