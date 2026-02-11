/**
 * Engine Lifecycle Implementation
 *
 * Handles creation, destruction, and validation of BatchDroneEngine.
 * Allocates all subsystems in correct dependency order.
 */

#include "environment_manager.h"
#include "obj_io.h"
#include "gpu_hal.h"
#include <stdio.h>
#include <string.h>

/* ============================================================================
 * Internal Helper: GPU-accelerated OBJ to SDF voxelization
 * Phases 1-2 on CPU, Phase 3 on GPU. Falls back to CPU on failure.
 * ============================================================================ */

#if GPU_AVAILABLE
static ObjIOResult obj_to_world_gpu(Arena* arena, const char* path,
                                     const VoxelizeOptions* options,
                                     WorldBrickMap** out_world, char* error) {
    if (!arena || !path || !out_world || !options) {
        if (error) snprintf(error, 256, "Invalid parameters");
        return OBJ_IO_ERROR_INVALID_PARAMETER;
    }
    *out_world = NULL;

    /* Parse OBJ file */
    TriangleMesh* mesh = NULL;
    MtlLibrary* mtl = NULL;
    ObjIOResult result = obj_parse_file(arena, path, &OBJ_PARSE_DEFAULTS,
                                         &mesh, &mtl, error);
    if (result != OBJ_IO_SUCCESS) return result;

    /* Compute world bounds (same as mesh_to_sdf) */
    float voxel_size = options->voxel_size;
    float brick_size = voxel_size * BRICK_SIZE;
    float padding = options->padding > 0 ? options->padding : brick_size;

    Vec3 world_min = vec3_sub(mesh->bbox_min, VEC3(padding, padding, padding));
    Vec3 world_max = vec3_add(mesh->bbox_max, VEC3(padding, padding, padding));

    if (options->world_min.x != 0.0f || options->world_min.y != 0.0f ||
        options->world_min.z != 0.0f || options->world_max.x != 0.0f ||
        options->world_max.y != 0.0f || options->world_max.z != 0.0f) {
        world_min = vec3_min(world_min, options->world_min);
        world_max = vec3_max(world_max, options->world_max);
    }

    Vec3 world_size = vec3_sub(world_max, world_min);
    uint32_t grid_x = (uint32_t)ceilf(world_size.x / brick_size);
    uint32_t grid_y = (uint32_t)ceilf(world_size.y / brick_size);
    uint32_t grid_z = (uint32_t)ceilf(world_size.z / brick_size);
    uint32_t grid_total = grid_x * grid_y * grid_z;

    uint32_t max_bricks = options->max_bricks;
    if (max_bricks == 0) {
        max_bricks = grid_total;
        if (max_bricks < 1024) max_bricks = 1024;
    }

    /* Build BVH */
    MeshBVH* bvh = bvh_build(arena, mesh);
    if (!bvh) {
        if (error) snprintf(error, 256, "Failed to build BVH");
        return OBJ_IO_ERROR_BVH_BUILD_FAILED;
    }

    /* Create world */
    WorldBrickMap* world = world_create(arena, world_min, world_max,
                                         voxel_size, max_bricks, 256);
    if (!world) {
        if (error) snprintf(error, 256, "Failed to create world brick map");
        return OBJ_IO_ERROR_OUT_OF_MEMORY;
    }

    /* Register materials */
    if (mesh->material_names && mesh->material_name_count > 0) {
        for (uint32_t i = 0; i < mesh->material_name_count; i++) {
            if (mesh->material_names[i])
                world_register_material(world, mesh->material_names[i],
                                         VEC3(1.0f, 1.0f, 1.0f));
        }
    }

    /* Phase 1: Coarse classification (CPU) */
    BrickClassification* classes = classify_bricks_coarse(arena, bvh, mesh, world);
    if (!classes) {
        if (error) snprintf(error, 256, "Failed to classify bricks (phase 1)");
        return OBJ_IO_ERROR_VOXELIZE_FAILED;
    }

    /* Phase 2: Fine classification (CPU) */
    classify_bricks_fine(classes, bvh, mesh, world, options);

    /* Mark uniform INSIDE bricks */
    for (uint32_t bz = 0; bz < classes->grid_z; bz++) {
        for (uint32_t by = 0; by < classes->grid_y; by++) {
            for (uint32_t bx = 0; bx < classes->grid_x; bx++) {
                uint32_t idx = bx + by * classes->grid_x +
                               bz * classes->grid_x * classes->grid_y;
                if (classes->classes[idx] == BRICK_CLASS_INSIDE)
                    world_mark_brick_uniform_inside(world, (int32_t)bx,
                                                     (int32_t)by, (int32_t)bz);
            }
        }
    }

    /* Collect surface brick list */
    uint32_t surface_count = 0;
    for (uint32_t i = 0; i < grid_total; i++) {
        if (classes->classes[i] == BRICK_CLASS_SURFACE)
            surface_count++;
    }

    uint32_t* surface_brick_list = arena_alloc_array(arena, uint32_t,
                                                       surface_count * 3);
    if (!surface_brick_list) {
        if (error) snprintf(error, 256, "Failed to allocate surface brick list");
        return OBJ_IO_ERROR_OUT_OF_MEMORY;
    }

    uint32_t si = 0;
    for (uint32_t bz = 0; bz < grid_z; bz++) {
        for (uint32_t by = 0; by < grid_y; by++) {
            for (uint32_t bx = 0; bx < grid_x; bx++) {
                uint32_t idx = bx + by * grid_x + bz * grid_x * grid_y;
                if (classes->classes[idx] == BRICK_CLASS_SURFACE) {
                    surface_brick_list[si * 3 + 0] = bx;
                    surface_brick_list[si * 3 + 1] = by;
                    surface_brick_list[si * 3 + 2] = bz;
                    si++;
                }
            }
        }
    }

    /* Phase 3: GPU voxelization */
    GpuDevice* device = gpu_device_create();
    if (!device) {
        /* GPU device creation failed - fall back to CPU Phase 3 */
        voxelize_surface_bricks(world, classes, bvh, mesh, options);
    } else {
        GpuResult gpu_result = gpu_voxelize_surface_bricks(device, bvh, mesh,
                                                            surface_brick_list,
                                                            surface_count,
                                                            world, options);
        gpu_device_destroy(device);

        if (gpu_result != GPU_SUCCESS) {
            /* GPU Phase 3 failed - fall back to CPU */
            voxelize_surface_bricks(world, classes, bvh, mesh, options);
        }
    }

    /* Register materials from MTL */
    if (mtl) mtl_register_materials(world, mtl);

    *out_world = world;
    return OBJ_IO_SUCCESS;
}
#endif /* GPU_AVAILABLE */

/* ============================================================================
 * Internal Helper: Create Default Sensors
 * ============================================================================ */

static void create_default_sensors(SensorSystem* sensors, uint32_t total_drones) {
    /* Add IMU sensor to all drones */
    SensorConfig imu_config = sensor_config_imu();
    uint32_t imu_idx = sensor_system_create_sensor(sensors, &imu_config);

    /* Add position sensor (oracle) to all drones */
    SensorConfig pos_config = sensor_config_position();
    uint32_t pos_idx = sensor_system_create_sensor(sensors, &pos_config);

    /* Add velocity sensor (oracle) to all drones */
    SensorConfig vel_config = sensor_config_velocity();
    uint32_t vel_idx = sensor_system_create_sensor(sensors, &vel_config);

    /* Attach sensors to all drones */
    for (uint32_t i = 0; i < total_drones; i++) {
        sensor_system_attach(sensors, i, imu_idx);
        sensor_system_attach(sensors, i, pos_idx);
        sensor_system_attach(sensors, i, vel_idx);
    }
}

/* ============================================================================
 * Engine Creation
 * ============================================================================ */

BatchDroneEngine* engine_create(const EngineConfig* config, char* error_msg) {
    /* Validate configuration */
    if (engine_config_validate(config, error_msg) != 0) {
        return NULL;
    }

    /* Auto-size persistent arena based on sensor memory requirements.
     * Users can override by setting persistent_arena_size larger. */
    size_t arena_size = config->persistent_arena_size;
    if (config->num_sensor_configs > 0 && config->sensor_configs != NULL) {
        uint32_t td = config->num_envs * config->drones_per_env;
        size_t sensor_mem = 0;
        for (uint32_t s = 0; s < config->num_sensor_configs; s++) {
            const SensorConfig* sc = &config->sensor_configs[s];
            size_t pixels = 0;
            size_t output_floats = 0;
            switch (sc->type) {
                case SENSOR_TYPE_CAMERA_RGB:
                    pixels = (size_t)sc->camera.width * sc->camera.height;
                    output_floats = pixels * 3;
                    break;
                case SENSOR_TYPE_CAMERA_DEPTH:
                case SENSOR_TYPE_CAMERA_SEGMENTATION:
                    pixels = (size_t)sc->camera.width * sc->camera.height;
                    output_floats = pixels;
                    break;
                case SENSOR_TYPE_LIDAR_3D:
                    output_floats = (size_t)sc->lidar_3d.horizontal_rays *
                                    sc->lidar_3d.vertical_layers;
                    break;
                default:
                    output_floats = 64;
                    break;
            }
            /* Per-sensor: obs buffer + ray_directions (Vec3 per pixel) */
            sensor_mem += output_floats * sizeof(float) * td;
            if (pixels > 0) {
                sensor_mem += pixels * sizeof(Vec3);  /* ray direction table */
            }
        }
        /* Need: base systems + sensor buffers (obs + scratch) with headroom */
        size_t needed = (size_t)ENGINE_DEFAULT_PERSISTENT_ARENA_MB * 1024 * 1024
                        + sensor_mem * 2;
        if (arena_size < needed) {
            arena_size = needed;
        }
    }

    /* Create persistent arena */
    Arena* persistent = arena_create(arena_size);
    if (persistent == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to create persistent arena (%zu bytes)",
                               arena_size);
        return NULL;
    }

    /* Allocate engine struct from persistent arena */
    BatchDroneEngine* engine = (BatchDroneEngine*)arena_alloc_aligned(
        persistent, sizeof(BatchDroneEngine), 64);
    if (engine == NULL) {
        arena_destroy(persistent);
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to allocate BatchDroneEngine");
        return NULL;
    }

    /* Initialize engine structure */
    memset(engine, 0, sizeof(BatchDroneEngine));
    engine->config = *config;
    engine->config.total_drones = config->num_envs * config->drones_per_env;
    engine->persistent_arena = persistent;

    uint32_t total_drones = engine->config.total_drones;

    /* Create frame arena */
    engine->frame_arena = arena_create(config->frame_arena_size);
    if (engine->frame_arena == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to create frame arena (%zu bytes)",
                               config->frame_arena_size);
        arena_destroy(persistent);
        return NULL;
    }

    /* Initialize RNG */
    pcg32_seed(&engine->rng, config->seed);

    /* --------------------------------------------------------------------------
     * Create subsystems in dependency order
     * -------------------------------------------------------------------------- */

    /* 1. Thread pool and scheduler */
    ThreadPoolConfig pool_config = {
        .num_threads = config->num_threads,
        .queue_capacity = 0  /* Default */
    };
    engine->thread_pool = threadpool_create(&pool_config);
    if (engine->thread_pool == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to create thread pool");
        goto cleanup;
    }

    engine->scheduler = scheduler_create(engine->thread_pool);
    if (engine->scheduler == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to create scheduler");
        goto cleanup;
    }

    /* 2. Drone state and parameter arrays */
    engine->states = drone_state_create(persistent, total_drones);
    if (engine->states == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to create drone state arrays");
        goto cleanup;
    }

    engine->params = drone_params_create(persistent, total_drones);
    if (engine->params == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to create drone params arrays");
        goto cleanup;
    }

    /* Initialize drone parameters with defaults */
    for (uint32_t i = 0; i < total_drones; i++) {
        drone_params_init(engine->params, i);
    }

    /* 3. World brick map */
    if (config->obj_path != NULL) {
        /* Load world geometry from OBJ file */
        char obj_error[256] = {0};
        VoxelizeOptions vox_opts = VOXELIZE_DEFAULTS;
        vox_opts.voxel_size = config->voxel_size;
        vox_opts.world_min = config->world_min;
        vox_opts.world_max = config->world_max;
        ObjIOResult obj_result = OBJ_IO_ERROR_VOXELIZE_FAILED;

#if GPU_AVAILABLE
        if (config->use_gpu_voxelization && gpu_is_available()) {
            obj_result = obj_to_world_gpu(persistent, config->obj_path,
                                           &vox_opts, &engine->world, obj_error);
        }
#endif
        /* Fallback to CPU if GPU path not used or failed */
        if (obj_result != OBJ_IO_SUCCESS) {
            obj_result = obj_to_world(persistent, config->obj_path,
                                       &vox_opts, &engine->world, obj_error);
        }

        if (obj_result != OBJ_IO_SUCCESS || engine->world == NULL) {
            if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                                   "Failed to load OBJ: %s", obj_error);
            goto cleanup;
        }
        /* Update config bounds from loaded world */
        engine->config.world_min = engine->world->world_min;
        engine->config.world_max = engine->world->world_max;
        /* Sync termination bounds if not custom */
        if (!config->use_custom_termination) {
            engine->config.termination_min = engine->world->world_min;
            engine->config.termination_max = engine->world->world_max;
        }
    } else {
        engine->world = world_create(persistent,
                                     config->world_min,
                                     config->world_max,
                                     config->voxel_size,
                                     config->max_bricks,
                                     256);  /* max_materials */
        if (engine->world == NULL) {
            if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                                   "Failed to create world brick map");
            goto cleanup;
        }
    }

    /* 4. Physics system */
    PhysicsConfig phys_config = {
        .dt = config->timestep / (float)config->physics_substeps,
        .dt_variance = 0.0f,
        .substeps = config->physics_substeps,
        .gravity = config->gravity,
        .air_density = config->air_density,
        .enable_drag = config->enable_drag,
        .enable_ground_effect = config->enable_ground_effect,
        .enable_motor_dynamics = config->enable_motor_dynamics,
        .enable_gyroscopic = config->enable_gyroscopic,
        .ground_effect_height = config->ground_effect_height,
        .ground_effect_coeff = config->ground_effect_coeff,
        .max_linear_accel = config->max_linear_accel,
        .max_angular_accel = config->max_angular_accel
    };

    engine->physics = physics_create(persistent, engine->frame_arena,
                                     &phys_config, total_drones);
    if (engine->physics == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to create physics system");
        goto cleanup;
    }

    /* 5. Collision system */
    engine->collision = collision_create(persistent, total_drones,
                                         config->drone_radius,
                                         config->collision_cell_size);
    if (engine->collision == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to create collision system");
        goto cleanup;
    }

    /* 6. Sensor system */
    uint32_t max_sensors = 64;
    size_t max_obs_dim = 256;  /* Default for IMU + Position + Velocity */
    if (config->num_sensor_configs > 0 && config->sensor_configs != NULL) {
        size_t estimated = 0;
        for (uint32_t s = 0; s < config->num_sensor_configs; s++) {
            const SensorConfig* sc = &config->sensor_configs[s];
            switch (sc->type) {
                case SENSOR_TYPE_CAMERA_RGB:
                    estimated += (size_t)sc->camera.width * sc->camera.height * 3; break;
                case SENSOR_TYPE_CAMERA_DEPTH:
                case SENSOR_TYPE_CAMERA_SEGMENTATION:
                    estimated += (size_t)sc->camera.width * sc->camera.height; break;
                case SENSOR_TYPE_LIDAR_3D:
                    estimated += (size_t)sc->lidar_3d.horizontal_rays * sc->lidar_3d.vertical_layers; break;
                case SENSOR_TYPE_LIDAR_2D:
                    estimated += sc->lidar_2d.num_rays; break;
                case SENSOR_TYPE_IMU: estimated += 6; break;
                case SENSOR_TYPE_TOF: estimated += 1; break;
                case SENSOR_TYPE_POSITION: estimated += 3; break;
                case SENSOR_TYPE_VELOCITY: estimated += 6; break;
                case SENSOR_TYPE_NEIGHBOR: estimated += (size_t)sc->neighbor.k * 4; break;
                default: estimated += 64; break;
            }
        }
        if (estimated > max_obs_dim) max_obs_dim = estimated;
    }
    engine->sensors = sensor_system_create(persistent, total_drones,
                                           max_sensors, max_obs_dim);
    if (engine->sensors == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to create sensor system");
        goto cleanup;
    }
    engine->sensors->dt = config->timestep;

    /* Initialize sensor registry and register all implementations */
    sensor_registry_init(&engine->sensors->registry);
    sensor_implementations_register_all(&engine->sensors->registry);

    /* Add sensors from config or use defaults */
    if (config->num_sensor_configs > 0 && config->sensor_configs != NULL) {
        for (uint32_t s = 0; s < config->num_sensor_configs; s++) {
            uint32_t sensor_idx = sensor_system_create_sensor(
                engine->sensors, &config->sensor_configs[s]);
            if (sensor_idx != UINT32_MAX) {
                /* Attach to all drones */
                for (uint32_t d = 0; d < total_drones; d++) {
                    sensor_system_attach(engine->sensors, d, sensor_idx);
                }
            }
        }
    } else {
        /* Use default sensors */
        create_default_sensors(engine->sensors, total_drones);
    }

    /* GPU sensor acceleration */
    sensor_implementations_register_gpu(&engine->sensors->registry);

    /* Update existing sensors' vtable pointers to the GPU-augmented versions.
     * sensor_implementations_register_gpu registers NEW vtable copies in the
     * registry, but sensors created above still point to the old vtables.
     * Re-resolve each sensor's vtable from the updated registry. */
    for (uint32_t s = 0; s < engine->sensors->sensor_count; s++) {
        Sensor* sensor = &engine->sensors->sensors[s];
        const SensorVTable* updated = sensor_registry_get(
            &engine->sensors->registry, sensor->type);
        if (updated != NULL) {
            sensor->vtable = updated;
        }
    }

    engine->gpu_sensor_ctx = gpu_sensor_context_create(total_drones);
    if (engine->gpu_sensor_ctx != NULL) {
        /* Initialize GPU slots for all GPU-capable sensors */
        for (uint32_t s = 0; s < engine->sensors->sensor_count; s++) {
            Sensor* sensor = &engine->sensors->sensors[s];
            if (sensor->vtable->batch_sample_gpu != NULL) {
                gpu_sensor_context_init_sensor(engine->gpu_sensor_ctx,
                                                sensor, total_drones);
            }
        }
    }

    /* Compute observation dimension from actual attached sensor output sizes.
     * sensor_system_get_obs_dim returns the buffer capacity (max_obs_dim),
     * but engine->obs_dim must be the actual used dimension for external
     * buffer sizing and memcpy in the binding layer. */
    {
        uint32_t actual_obs_dim = 0;
        if (total_drones > 0) {
            uint32_t attach_count = engine->sensors->attachment_counts[0];
            for (uint32_t a = 0; a < attach_count; a++) {
                actual_obs_dim += (uint32_t)engine->sensors->attachments[a].output_size;
            }
        }
        engine->obs_dim = actual_obs_dim > 0 ? actual_obs_dim : 15;
    }

    /* 7. Reward system */
    uint32_t max_gates = 0;  /* No racing gates by default */
    engine->rewards = reward_create(persistent, &config->reward_config,
                                    total_drones, max_gates);
    if (engine->rewards == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to create reward system");
        goto cleanup;
    }

    /* --------------------------------------------------------------------------
     * Allocate episode tracking arrays
     * -------------------------------------------------------------------------- */

    engine->episode_returns = (float*)arena_alloc_aligned(
        persistent, total_drones * sizeof(float), 32);
    if (engine->episode_returns == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to allocate episode_returns");
        goto cleanup;
    }
    memset(engine->episode_returns, 0, total_drones * sizeof(float));

    engine->episode_lengths = (uint32_t*)arena_alloc_aligned(
        persistent, total_drones * sizeof(uint32_t), 32);
    if (engine->episode_lengths == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to allocate episode_lengths");
        goto cleanup;
    }
    memset(engine->episode_lengths, 0, total_drones * sizeof(uint32_t));

    engine->env_ids = (uint32_t*)arena_alloc_aligned(
        persistent, total_drones * sizeof(uint32_t), 32);
    if (engine->env_ids == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to allocate env_ids");
        goto cleanup;
    }

    /* Initialize env_ids */
    for (uint32_t env = 0; env < config->num_envs; env++) {
        for (uint32_t d = 0; d < config->drones_per_env; d++) {
            uint32_t idx = env * config->drones_per_env + d;
            engine->env_ids[idx] = env;
        }
    }

    /* --------------------------------------------------------------------------
     * Allocate external buffers (32-byte aligned for SIMD)
     * -------------------------------------------------------------------------- */

    engine->action_dim = ENGINE_ACTION_DIM;

    /* Observations buffer */
    size_t obs_size = total_drones * engine->obs_dim * sizeof(float);
    obs_size = (obs_size + 31) & ~31;  /* Round up to 32 bytes */
    engine->observations = (float*)arena_alloc_aligned(persistent, obs_size, 32);
    if (engine->observations == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to allocate observations buffer");
        goto cleanup;
    }
    memset(engine->observations, 0, obs_size);

    /* Zero-copy: point sensor system's observation buffer directly at
     * engine->observations to avoid a per-frame memcpy in engine_step.
     * CRITICAL: also update the sensor system's obs_dim to match the actual
     * observation dimension. The sensor system was created with max_obs_dim
     * (buffer capacity), but the external buffer is sized for the actual
     * obs_dim. Using the wrong stride causes buffer overflow that corrupts
     * everything allocated after observations in the arena. */
    sensor_system_set_external_buffer(engine->sensors, engine->observations);
    engine->sensors->obs_dim = engine->obs_dim;

    /* Actions buffer */
    size_t act_size = total_drones * engine->action_dim * sizeof(float);
    act_size = (act_size + 31) & ~31;
    engine->actions = (float*)arena_alloc_aligned(persistent, act_size, 32);
    if (engine->actions == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to allocate actions buffer");
        goto cleanup;
    }
    memset(engine->actions, 0, act_size);

    /* Rewards buffer */
    size_t rew_size = total_drones * sizeof(float);
    rew_size = (rew_size + 31) & ~31;
    engine->rewards_buffer = (float*)arena_alloc_aligned(persistent, rew_size, 32);
    if (engine->rewards_buffer == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to allocate rewards buffer");
        goto cleanup;
    }
    memset(engine->rewards_buffer, 0, rew_size);

    /* Dones buffer */
    size_t done_size = total_drones * sizeof(uint8_t);
    done_size = (done_size + 31) & ~31;
    engine->dones = (uint8_t*)arena_alloc_aligned(persistent, done_size, 32);
    if (engine->dones == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to allocate dones buffer");
        goto cleanup;
    }
    memset(engine->dones, 0, done_size);

    /* Truncations buffer */
    engine->truncations = (uint8_t*)arena_alloc_aligned(persistent, done_size, 32);
    if (engine->truncations == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to allocate truncations buffer");
        goto cleanup;
    }
    memset(engine->truncations, 0, done_size);

    /* Detailed termination flag buffers */
    engine->term_success = (uint8_t*)arena_alloc_aligned(persistent, done_size, 32);
    if (engine->term_success == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to allocate term_success buffer");
        goto cleanup;
    }
    memset(engine->term_success, 0, done_size);

    engine->term_collision = (uint8_t*)arena_alloc_aligned(persistent, done_size, 32);
    if (engine->term_collision == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to allocate term_collision buffer");
        goto cleanup;
    }
    memset(engine->term_collision, 0, done_size);

    engine->term_out_of_bounds = (uint8_t*)arena_alloc_aligned(persistent, done_size, 32);
    if (engine->term_out_of_bounds == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to allocate term_out_of_bounds buffer");
        goto cleanup;
    }
    memset(engine->term_out_of_bounds, 0, done_size);

    engine->term_timeout = (uint8_t*)arena_alloc_aligned(persistent, done_size, 32);
    if (engine->term_timeout == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to allocate term_timeout buffer");
        goto cleanup;
    }
    memset(engine->term_timeout, 0, done_size);

    /* --------------------------------------------------------------------------
     * Final initialization
     * -------------------------------------------------------------------------- */

    engine->total_steps = 0;
    engine->total_episodes = 0;
    engine->physics_time_ms = 0.0;
    engine->collision_time_ms = 0.0;
    engine->sensor_time_ms = 0.0;
    engine->reward_time_ms = 0.0;
    engine->reset_time_ms = 0.0;

    engine->initialized = true;
    engine->needs_reset = true;

    if (error_msg) error_msg[0] = '\0';
    return engine;

cleanup:
    /* Cleanup on failure */
    if (engine->thread_pool) {
        threadpool_destroy(engine->thread_pool);
    }
    if (engine->scheduler) {
        scheduler_destroy(engine->scheduler);
    }
    if (engine->frame_arena) {
        arena_destroy(engine->frame_arena);
    }
    arena_destroy(persistent);
    return NULL;
}

/* ============================================================================
 * Engine Destruction
 * ============================================================================ */

void engine_destroy(BatchDroneEngine* engine) {
    if (engine == NULL) {
        return;
    }

    /* Destroy GPU sensor context first */
    if (engine->gpu_sensor_ctx) {
        gpu_sensor_context_destroy(engine->gpu_sensor_ctx);
        engine->gpu_sensor_ctx = NULL;
    }

    /* Destroy thread pool (waits for pending work) */
    if (engine->thread_pool) {
        threadpool_destroy(engine->thread_pool);
        engine->thread_pool = NULL;
    }

    /* Destroy scheduler */
    if (engine->scheduler) {
        scheduler_destroy(engine->scheduler);
        engine->scheduler = NULL;
    }

    /* Destroy sensor system (frees scratch arena which is a separate malloc) */
    if (engine->sensors) {
        sensor_system_destroy(engine->sensors);
        engine->sensors = NULL;
    }

    /* Destroy frame arena */
    if (engine->frame_arena) {
        arena_destroy(engine->frame_arena);
        engine->frame_arena = NULL;
    }

    /* Mark as uninitialized */
    engine->initialized = false;

    /* Destroy persistent arena last — frees everything allocated from it,
     * including the engine struct itself. Do NOT access engine after this. */
    Arena* persistent = engine->persistent_arena;
    if (persistent) {
        arena_destroy(persistent);
    }
}

/* ============================================================================
 * Engine Validation
 * ============================================================================ */

bool engine_is_valid(const BatchDroneEngine* engine) {
    if (engine == NULL) {
        return false;
    }

    if (!engine->initialized) {
        return false;
    }

    /* Check essential subsystems */
    if (engine->states == NULL ||
        engine->params == NULL ||
        engine->world == NULL ||
        engine->physics == NULL ||
        engine->collision == NULL ||
        engine->sensors == NULL ||
        engine->rewards == NULL) {
        return false;
    }

    /* Check buffers */
    if (engine->observations == NULL ||
        engine->actions == NULL ||
        engine->rewards_buffer == NULL ||
        engine->dones == NULL ||
        engine->truncations == NULL) {
        return false;
    }

    /* Check arenas */
    if (engine->persistent_arena == NULL ||
        engine->frame_arena == NULL) {
        return false;
    }

    return true;
}

/* ============================================================================
 * Memory Size Calculation
 * ============================================================================ */

size_t engine_memory_size(const EngineConfig* config) {
    if (config == NULL) {
        return 0;
    }

    uint32_t total_drones = config->num_envs * config->drones_per_env;

    size_t total = 0;

    /* BatchDroneEngine struct */
    total += sizeof(BatchDroneEngine);

    /* Drone state arrays (17 floats per drone) */
    total += total_drones * 17 * sizeof(float);

    /* Drone params arrays (15 floats per drone) */
    total += total_drones * 15 * sizeof(float);

    /* Episode tracking */
    total += total_drones * sizeof(float);    /* episode_returns */
    total += total_drones * sizeof(uint32_t); /* episode_lengths */
    total += total_drones * sizeof(uint32_t); /* env_ids */

    /* External buffers (assume obs_dim = 15 default) */
    uint32_t obs_dim = 15;
    total += total_drones * obs_dim * sizeof(float);  /* observations */
    total += total_drones * 4 * sizeof(float);        /* actions */
    total += total_drones * sizeof(float);            /* rewards */
    total += total_drones * sizeof(uint8_t) * 2;      /* dones + truncations */

    /* Add overhead for alignment */
    total = (total * 120) / 100;  /* ~20% overhead */

    return total;
}

size_t engine_observation_buffer_size(uint32_t total_drones, uint32_t obs_dim) {
    size_t size = total_drones * obs_dim * sizeof(float);
    return (size + 31) & ~31;  /* 32-byte aligned */
}

size_t engine_action_buffer_size(uint32_t total_drones, uint32_t action_dim) {
    size_t size = total_drones * action_dim * sizeof(float);
    return (size + 31) & ~31;
}
