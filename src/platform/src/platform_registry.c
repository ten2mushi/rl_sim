/**
 * Platform Registry Implementation
 *
 * Manages registration and lookup of platform vtables by name.
 */

#include "../include/platform.h"
#include <string.h>

void platform_registry_init(PlatformRegistry* registry) {
    if (registry == NULL) return;

    memset(registry, 0, sizeof(PlatformRegistry));

    /* Register built-in platforms */
    platform_registry_register(registry, &PLATFORM_QUADCOPTER);
    platform_registry_register(registry, &PLATFORM_DIFF_DRIVE);
}

int platform_registry_register(PlatformRegistry* registry, const PlatformVTable* vtable) {
    if (registry == NULL || vtable == NULL) return -1;
    if (registry->count >= PLATFORM_REGISTRY_MAX_SLOTS) return -1;

    registry->vtables[registry->count++] = vtable;
    return 0;
}

const PlatformVTable* platform_registry_find(const PlatformRegistry* registry, const char* name) {
    if (registry == NULL || name == NULL) return NULL;

    for (uint32_t i = 0; i < registry->count; i++) {
        if (strcmp(registry->vtables[i]->name, name) == 0) {
            return registry->vtables[i];
        }
    }
    return NULL;
}
