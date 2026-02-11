/**
 * Arena Allocator Implementation
 *
 * O(1) bump allocation with scoped allocation support.
 * Single contiguous memory region with a growing watermark.
 */

#include "../include/foundation.h"

/**
 * Create a new arena with the specified capacity.
 *
 * The arena is allocated as a single contiguous block containing both
 * the Arena header and the data buffer. The data buffer starts immediately
 * after the header, aligned to FOUNDATION_DEFAULT_ALIGNMENT.
 *
 * @param capacity The size of the data buffer in bytes
 * @return Pointer to the new arena, or NULL if allocation fails
 */
Arena* arena_create(size_t capacity) {
    if (capacity == 0) {
        return NULL;
    }

    /* Calculate total allocation size with proper alignment */
    size_t header_size = align_up_size(sizeof(Arena), FOUNDATION_DEFAULT_ALIGNMENT);
    size_t total_size = header_size + capacity;

    /* aligned_alloc requires size to be a multiple of alignment */
    total_size = align_up_size(total_size, FOUNDATION_DEFAULT_ALIGNMENT);

    /* Allocate single block for header + data */
    void* block = aligned_alloc(FOUNDATION_DEFAULT_ALIGNMENT, total_size);
    if (block == NULL) {
        return NULL;
    }

    /* Initialize arena header */
    Arena* arena = (Arena*)block;
    arena->data = (char*)block + header_size;
    arena->capacity = capacity;
    arena->used = 0;

    return arena;
}

/**
 * Destroy an arena and free all associated memory.
 *
 * @param arena The arena to destroy (may be NULL)
 */
void arena_destroy(Arena* arena) {
    if (arena != NULL) {
        /* Single free since we allocated header + data as one block */
        free(arena);
    }
}

/**
 * Allocate memory from the arena with default alignment (16 bytes).
 *
 * This is an O(1) bump allocation - simply increments the watermark.
 *
 * @param arena The arena to allocate from
 * @param size The number of bytes to allocate
 * @return Pointer to the allocated memory, or NULL if insufficient space
 */
void* arena_alloc(Arena* arena, size_t size) {
    return arena_alloc_aligned(arena, size, FOUNDATION_DEFAULT_ALIGNMENT);
}

/**
 * Allocate memory from the arena with specified alignment.
 *
 * @param arena The arena to allocate from
 * @param size The number of bytes to allocate
 * @param alignment The required alignment (must be a power of 2)
 * @return Pointer to the allocated memory, or NULL if insufficient space
 */
void* arena_alloc_aligned(Arena* arena, size_t size, size_t alignment) {
    FOUNDATION_ASSERT(arena != NULL, "arena is NULL");
    FOUNDATION_ASSERT(size > 0, "allocation size must be positive");
    FOUNDATION_ASSERT(is_power_of_two((uint32_t)alignment), "alignment must be power of 2");

    if (arena == NULL || size == 0) {
        return NULL;
    }

    /* Calculate aligned offset */
    uintptr_t current = (uintptr_t)arena->data + arena->used;
    uintptr_t aligned = (current + alignment - 1) & ~(alignment - 1);
    size_t padding = aligned - current;

    /* Check for overflow */
    size_t total_size = padding + size;
    if (arena->used + total_size > arena->capacity) {
        FOUNDATION_LOG_ERROR("arena overflow: requested %zu bytes, only %zu remaining",
                            size, arena_remaining(arena));
        return NULL;
    }

    /* Bump the watermark */
    arena->used += total_size;

    return (void*)aligned;
}

/**
 * Allocate zero-initialized memory from the arena.
 *
 * @param arena The arena to allocate from
 * @param size The number of bytes to allocate
 * @return Pointer to zero-initialized memory, or NULL if insufficient space
 */
void* arena_alloc_zero(Arena* arena, size_t size) {
    void* ptr = arena_alloc(arena, size);
    if (ptr != NULL) {
        memset(ptr, 0, size);
    }
    return ptr;
}

/**
 * Reset the arena, freeing all allocations in O(1).
 *
 * Simply sets the watermark back to 0. All previously allocated pointers
 * become invalid after this call.
 *
 * @param arena The arena to reset
 */
void arena_reset(Arena* arena) {
    FOUNDATION_ASSERT(arena != NULL, "arena is NULL");
    if (arena != NULL) {
        arena->used = 0;
    }
}

/**
 * Get the remaining capacity in the arena.
 *
 * @param arena The arena to query
 * @return The number of bytes remaining (not accounting for alignment)
 */
size_t arena_remaining(const Arena* arena) {
    FOUNDATION_ASSERT(arena != NULL, "arena is NULL");
    if (arena == NULL) {
        return 0;
    }
    return arena->capacity - arena->used;
}

/**
 * Get the utilization ratio of the arena.
 *
 * @param arena The arena to query
 * @return The utilization as a value between 0.0 and 1.0
 */
float arena_utilization(const Arena* arena) {
    FOUNDATION_ASSERT(arena != NULL, "arena is NULL");
    if (arena == NULL || arena->capacity == 0) {
        return 0.0f;
    }
    return (float)arena->used / (float)arena->capacity;
}

/**
 * Begin a scoped allocation context.
 *
 * Saves the current watermark position so it can be restored later.
 * Use with arena_scope_end() for RAII-style temporary allocations.
 *
 * @param arena The arena to create a scope for
 * @return An ArenaScope containing the saved state
 */
ArenaScope arena_scope_begin(Arena* arena) {
    FOUNDATION_ASSERT(arena != NULL, "arena is NULL");
    ArenaScope scope;
    scope.arena = arena;
    scope.saved_used = (arena != NULL) ? arena->used : 0;
    return scope;
}

/**
 * End a scoped allocation context.
 *
 * Restores the watermark to the position saved when the scope was created,
 * effectively freeing all allocations made within the scope.
 *
 * @param scope The scope to end
 */
void arena_scope_end(ArenaScope scope) {
    if (scope.arena != NULL) {
        FOUNDATION_ASSERT(scope.arena->used >= scope.saved_used,
                         "arena corruption: used < saved_used");
        scope.arena->used = scope.saved_used;
    }
}
