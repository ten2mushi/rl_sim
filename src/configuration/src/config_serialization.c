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
#include <stdio.h>
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

    /* Hash drone config */
    hash ^= fnv1a_hash(&config->drone, sizeof(DroneConfig));
    hash *= FNV_PRIME;

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

    /* [drone] section */
    fprintf(fp, "[drone]\n");
    fprintf(fp, "name = \"%s\"\n", config->drone.name);
    if (config->drone.model_path[0]) {
        fprintf(fp, "model_path = \"%s\"\n", config->drone.model_path);
    }
    fprintf(fp, "mass = %.6f\n", config->drone.mass);
    fprintf(fp, "arm_length = %.4f\n", config->drone.arm_length);
    fprintf(fp, "ixx = %.6e\n", config->drone.ixx);
    fprintf(fp, "iyy = %.6e\n", config->drone.iyy);
    fprintf(fp, "izz = %.6e\n", config->drone.izz);
    fprintf(fp, "collision_radius = %.4f\n", config->drone.collision_radius);
    fprintf(fp, "k_thrust = %.6e\n", config->drone.k_thrust);
    fprintf(fp, "k_torque = %.6e\n", config->drone.k_torque);
    fprintf(fp, "motor_tau = %.4f\n", config->drone.motor_tau);
    fprintf(fp, "max_rpm = %.1f\n", config->drone.max_rpm);
    fprintf(fp, "k_drag = %.6f\n", config->drone.k_drag);
    fprintf(fp, "k_drag_angular = %.6f\n", config->drone.k_drag_angular);
    fprintf(fp, "max_velocity = %.1f\n", config->drone.max_velocity);
    fprintf(fp, "max_angular_velocity = %.1f\n", config->drone.max_angular_velocity);
    fprintf(fp, "max_tilt_angle = %.2f\n", config->drone.max_tilt_angle);
    fprintf(fp, "color = [%.2f, %.2f, %.2f]\n",
            config->drone.color[0], config->drone.color[1], config->drone.color[2]);
    fprintf(fp, "scale = %.2f\n", config->drone.scale);

    /* [environment] section */
    fprintf(fp, "\n[environment]\n");
    fprintf(fp, "num_envs = %u\n", config->environment.num_envs);
    fprintf(fp, "drones_per_env = %u\n", config->environment.drones_per_env);
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

    int written = snprintf(buffer, buffer_size,
        "{\n"
        "  \"drone\": {\n"
        "    \"name\": \"%s\",\n"
        "    \"mass\": %.6f,\n"
        "    \"arm_length\": %.4f\n"
        "  },\n"
        "  \"environment\": {\n"
        "    \"num_envs\": %u,\n"
        "    \"drones_per_env\": %u,\n"
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
        config->drone.name,
        config->drone.mass,
        config->drone.arm_length,
        config->environment.num_envs,
        config->environment.drones_per_env,
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

    /* Deep comparison for hash collision detection */
    if (memcmp(&a->drone, &b->drone, sizeof(DroneConfig)) != 0) return 1;
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

    printf("[drone]\n");
    printf("  name: %s\n", config->drone.name);
    printf("  mass: %.4f kg\n", config->drone.mass);
    printf("  arm_length: %.4f m\n", config->drone.arm_length);
    printf("  inertia: [%.2e, %.2e, %.2e] kg*m^2\n",
           config->drone.ixx, config->drone.iyy, config->drone.izz);
    printf("  k_thrust: %.2e N/(rad/s)^2\n", config->drone.k_thrust);
    printf("  k_torque: %.2e N*m/(rad/s)^2\n", config->drone.k_torque);
    printf("  motor_tau: %.4f s\n", config->drone.motor_tau);
    printf("  max_rpm: %.0f\n", config->drone.max_rpm);
    printf("  collision_radius: %.4f m\n", config->drone.collision_radius);

    printf("\n[environment]\n");
    printf("  num_envs: %u\n", config->environment.num_envs);
    printf("  drones_per_env: %u\n", config->environment.drones_per_env);
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
