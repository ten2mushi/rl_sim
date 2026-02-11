/**
 * Sensor Registry Implementation
 *
 * Registers all built-in sensor vtables with the sensor system registry.
 * Call sensor_implementations_register_all() after sensor_registry_init()
 * to make all sensor types available.
 */

#include "sensor_implementations.h"

/* ============================================================================
 * Registry Initialization
 * ============================================================================ */

void sensor_implementations_register_all(SensorRegistry* registry) {
    if (registry == NULL) {
        return;
    }

    /* Ensure registry is initialized */
    if (!registry->initialized) {
        sensor_registry_init(registry);
    }

    /* Register all built-in sensor types */
    sensor_registry_register(registry, SENSOR_TYPE_IMU, &SENSOR_VTABLE_IMU);
    sensor_registry_register(registry, SENSOR_TYPE_TOF, &SENSOR_VTABLE_TOF);
    sensor_registry_register(registry, SENSOR_TYPE_LIDAR_2D, &SENSOR_VTABLE_LIDAR_2D);
    sensor_registry_register(registry, SENSOR_TYPE_LIDAR_3D, &SENSOR_VTABLE_LIDAR_3D);
    sensor_registry_register(registry, SENSOR_TYPE_CAMERA_RGB, &SENSOR_VTABLE_CAMERA_RGB);
    sensor_registry_register(registry, SENSOR_TYPE_CAMERA_DEPTH, &SENSOR_VTABLE_CAMERA_DEPTH);
    sensor_registry_register(registry, SENSOR_TYPE_CAMERA_SEGMENTATION, &SENSOR_VTABLE_CAMERA_SEGMENTATION);
    sensor_registry_register(registry, SENSOR_TYPE_POSITION, &SENSOR_VTABLE_POSITION);
    sensor_registry_register(registry, SENSOR_TYPE_VELOCITY, &SENSOR_VTABLE_VELOCITY);
    sensor_registry_register(registry, SENSOR_TYPE_NEIGHBOR, &SENSOR_VTABLE_NEIGHBOR);
}
