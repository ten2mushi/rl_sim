# Foundation Module Usage Guide

Quick reference for developers working with the Foundation submodule.

## Include

```c
#include "foundation.h"
```

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

---

## Types

| Type | Size | Alignment | Description |
|------|------|-----------|-------------|
| `Vec3` | 16 bytes | 16-byte | 3D vector (x, y, z) + padding |
| `Quat` | 16 bytes | 16-byte | Quaternion (w, x, y, z) |
| `Mat3` | 48 bytes | 16-byte | 3×3 matrix, column-major |
| `Mat4` | 64 bytes | 32-byte | 4×4 matrix, column-major |
| `Arena` | 24 bytes | - | Arena allocator state |
| `ArenaScope` | 16 bytes | - | Scoped allocation context |
| `PCG32` | 16 bytes | 8-byte | Random number generator state |

---

## Type Constructors (Macros)

```c
VEC3(x, y, z)           // → Vec3
VEC3_ZERO               // → Vec3{0, 0, 0}
VEC3_ONE                // → Vec3{1, 1, 1}
VEC3_UP                 // → Vec3{0, 0, 1}
QUAT(w, x, y, z)        // → Quat
QUAT_IDENTITY           // → Quat{1, 0, 0, 0}
```

---

## Arena Allocator API

### Lifecycle

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `arena_create` | `size_t capacity` | `Arena*` (NULL on failure) | Create arena with given capacity |
| `arena_destroy` | `Arena* arena` | `void` | Free arena memory |

### Allocation

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `arena_alloc` | `Arena* arena, size_t size` | `void*` (NULL if full) | Allocate with 16-byte alignment |
| `arena_alloc_aligned` | `Arena* arena, size_t size, size_t alignment` | `void*` (NULL if full) | Allocate with custom alignment |
| `arena_alloc_zero` | `Arena* arena, size_t size` | `void*` (NULL if full) | Allocate zero-initialized memory |

### Typed Allocation Macros

```c
arena_alloc_type(arena, T)          // → T* - Allocate single instance of type T
arena_alloc_array(arena, T, count)  // → T* - Allocate array of count elements
```

### Reset & Query

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `arena_reset` | `Arena* arena` | `void` | Free all allocations (O(1)) |
| `arena_remaining` | `const Arena* arena` | `size_t` | Bytes remaining |
| `arena_utilization` | `const Arena* arena` | `float` | Usage ratio [0.0, 1.0] |

### Scoped Allocation

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `arena_scope_begin` | `Arena* arena` | `ArenaScope` | Save current watermark |
| `arena_scope_end` | `ArenaScope scope` | `void` | Restore saved watermark |

```c
// RAII-style scoped allocation macro
ARENA_SCOPE(arena) {
    void* temp = arena_alloc(arena, 1024);
    // Use temp...
} // Memory automatically reclaimed
```

---

## Vector Operations API

### Creation (Inline)

| Function | Input | Output |
|----------|-------|--------|
| `vec3_zero` | `void` | `Vec3{0, 0, 0}` |
| `vec3_one` | `void` | `Vec3{1, 1, 1}` |
| `vec3_up` | `void` | `Vec3{0, 0, 1}` |

### Arithmetic (Inline)

| Function | Input | Output |
|----------|-------|--------|
| `vec3_add` | `Vec3 a, Vec3 b` | `Vec3` (a + b) |
| `vec3_sub` | `Vec3 a, Vec3 b` | `Vec3` (a - b) |
| `vec3_scale` | `Vec3 v, float s` | `Vec3` (v × s) |
| `vec3_mul` | `Vec3 a, Vec3 b` | `Vec3` (component-wise a × b) |
| `vec3_div` | `Vec3 a, Vec3 b` | `Vec3` (component-wise a / b) |
| `vec3_neg` | `Vec3 v` | `Vec3` (-v) |
| `vec3_abs` | `Vec3 v` | `Vec3` (|v|) |

### Products (Inline)

| Function | Input | Output |
|----------|-------|--------|
| `vec3_dot` | `Vec3 a, Vec3 b` | `float` (dot product) |
| `vec3_cross` | `Vec3 a, Vec3 b` | `Vec3` (cross product) |

### Length Operations

| Function | Input | Output | Location |
|----------|-------|--------|----------|
| `vec3_length_sq` | `Vec3 v` | `float` | Inline |
| `vec3_length` | `Vec3 v` | `float` | math.c |
| `vec3_normalize` | `Vec3 v` | `Vec3` (unit vector) | math.c |
| `vec3_distance` | `Vec3 a, Vec3 b` | `float` | math.c |

### Interpolation & Clamping (Inline)

| Function | Input | Output |
|----------|-------|--------|
| `vec3_lerp` | `Vec3 a, Vec3 b, float t` | `Vec3` (interpolated) |
| `vec3_clamp` | `Vec3 v, float min, float max` | `Vec3` (clamped) |
| `vec3_clamp_vec` | `Vec3 v, Vec3 min, Vec3 max` | `Vec3` (clamped per-component) |
| `vec3_min` | `Vec3 a, Vec3 b` | `Vec3` (component-wise min) |
| `vec3_max` | `Vec3 a, Vec3 b` | `Vec3` (component-wise max) |
| `vec3_reflect` | `Vec3 v, Vec3 n` | `Vec3` (reflected) |

---

## Quaternion Operations API

### Creation

| Function | Input | Output | Location |
|----------|-------|--------|----------|
| `quat_identity` | `void` | `Quat{1, 0, 0, 0}` | Inline |
| `quat_from_axis_angle` | `Vec3 axis, float angle` | `Quat` | math.c |
| `quat_from_euler` | `float roll, float pitch, float yaw` | `Quat` | math.c |
| `quat_from_mat3` | `Mat3 m` | `Quat` | math.c |

### Operations

| Function | Input | Output | Location |
|----------|-------|--------|----------|
| `quat_normalize` | `Quat q` | `Quat` (unit quaternion) | math.c |
| `quat_multiply` | `Quat a, Quat b` | `Quat` (Hamilton product) | Inline |
| `quat_conjugate` | `Quat q` | `Quat` (inverse for unit quat) | Inline |
| `quat_dot` | `Quat a, Quat b` | `float` | Inline |
| `quat_length_sq` | `Quat q` | `float` | Inline |
| `quat_rotate` | `Quat q, Vec3 v` | `Vec3` (rotated vector) | math.c |
| `quat_slerp` | `Quat a, Quat b, float t` | `Quat` (interpolated) | math.c |

### Conversion

| Function | Input | Output | Location |
|----------|-------|--------|----------|
| `quat_to_mat3` | `Quat q` | `Mat3` | math.c |
| `quat_to_mat4` | `Quat q` | `Mat4` | math.c |

---

## Matrix Operations API

### Creation (Inline)

| Function | Input | Output |
|----------|-------|--------|
| `mat3_identity` | `void` | `Mat3` (identity) |
| `mat4_identity` | `void` | `Mat4` (identity) |

### Mat3 Operations (math.c)

| Function | Input | Output |
|----------|-------|--------|
| `mat3_multiply` | `Mat3 a, Mat3 b` | `Mat3` (a × b) |
| `mat3_transform` | `Mat3 m, Vec3 v` | `Vec3` (m × v) |
| `mat3_transpose` | `Mat3 m` | `Mat3` (transposed) |

### Mat4 Operations (math.c)

| Function | Input | Output |
|----------|-------|--------|
| `mat4_multiply` | `Mat4 a, Mat4 b` | `Mat4` (a × b) |
| `mat4_transpose` | `Mat4 m` | `Mat4` (transposed) |
| `mat4_translate` | `Vec3 translation` | `Mat4` (translation matrix) |
| `mat4_scale` | `Vec3 scale` | `Mat4` (scale matrix) |
| `mat4_rotate` | `Quat rotation` | `Mat4` (rotation matrix) |
| `mat4_from_trs` | `Vec3 t, Quat r, Vec3 s` | `Mat4` (TRS composition) |

---

## PCG32 Random Number Generator API

### Seeding

| Function | Input | Output |
|----------|-------|--------|
| `pcg32_seed` | `PCG32* rng, uint64_t seed` | `void` |
| `pcg32_seed_dual` | `PCG32* rng, uint64_t seed, uint64_t seq` | `void` |

### Integer Generation

| Function | Input | Output |
|----------|-------|--------|
| `pcg32_random` | `PCG32* rng` | `uint32_t` [0, 2³²) |
| `pcg32_bounded` | `PCG32* rng, uint32_t bound` | `uint32_t` [0, bound) |

### Float Generation

| Function | Input | Output |
|----------|-------|--------|
| `pcg32_float` | `PCG32* rng` | `float` [0.0, 1.0) |
| `pcg32_range` | `PCG32* rng, float min, float max` | `float` [min, max) |

### Vector Generation

| Function | Input | Output |
|----------|-------|--------|
| `pcg32_vec3_unit` | `PCG32* rng` | `Vec3` (random unit vector) |
| `pcg32_vec3_range` | `PCG32* rng, Vec3 min, Vec3 max` | `Vec3` (random in range) |

### Thread-Local RNG

| Function | Input | Output |
|----------|-------|--------|
| `pcg32_thread_local` | `void` | `PCG32*` (thread-local instance) |

---

## Utility Functions API

### Float Utilities (Inline)

| Function | Input | Output |
|----------|-------|--------|
| `clampf` | `float x, float min, float max` | `float` |
| `lerpf` | `float a, float b, float t` | `float` |
| `smoothstep` | `float edge0, float edge1, float x` | `float` |
| `signf` | `float x` | `float` (-1, 0, or 1) |
| `absf` | `float x` | `float` |

### Integer Utilities (Inline)

| Function | Input | Output |
|----------|-------|--------|
| `align_up` | `uint32_t value, uint32_t alignment` | `uint32_t` |
| `align_up_size` | `size_t value, size_t alignment` | `size_t` |
| `is_power_of_two` | `uint32_t x` | `bool` |
| `next_power_of_two` | `uint32_t x` | `uint32_t` |
| `min_i32` | `int32_t a, int32_t b` | `int32_t` |
| `max_i32` | `int32_t a, int32_t b` | `int32_t` |
| `min_u32` | `uint32_t a, uint32_t b` | `uint32_t` |
| `max_u32` | `uint32_t a, uint32_t b` | `uint32_t` |

---

## SIMD Abstraction API

### Platform Constants

| Constant | AVX2 | NEON | Scalar |
|----------|------|------|--------|
| `FOUNDATION_SIMD_WIDTH` | 8 | 4 | 1 |
| `FOUNDATION_SIMD_ALIGNMENT` | 32 | 16 | 4 |

### SIMD Types

| Type | AVX2 | NEON | Scalar |
|------|------|------|--------|
| `simd_float` | `__m256` | `float32x4_t` | `float` |
| `simd_int` | `__m256i` | `int32x4_t` | `int32_t` |

### Float Operations

| Macro | Input | Output |
|-------|-------|--------|
| `simd_load_ps` | `const float* p` | `simd_float` (aligned load) |
| `simd_loadu_ps` | `const float* p` | `simd_float` (unaligned load) |
| `simd_store_ps` | `float* p, simd_float v` | `void` (aligned store) |
| `simd_storeu_ps` | `float* p, simd_float v` | `void` (unaligned store) |
| `simd_set1_ps` | `float x` | `simd_float` (broadcast) |
| `simd_setzero_ps` | `void` | `simd_float` (zeros) |
| `simd_add_ps` | `simd_float a, simd_float b` | `simd_float` |
| `simd_sub_ps` | `simd_float a, simd_float b` | `simd_float` |
| `simd_mul_ps` | `simd_float a, simd_float b` | `simd_float` |
| `simd_div_ps` | `simd_float a, simd_float b` | `simd_float` |
| `simd_fmadd_ps` | `simd_float a, simd_float b, simd_float c` | `simd_float` (a×b+c) |
| `simd_fmsub_ps` | `simd_float a, simd_float b, simd_float c` | `simd_float` (a×b-c) |
| `simd_sqrt_ps` | `simd_float a` | `simd_float` |
| `simd_rsqrt_ps` | `simd_float a` | `simd_float` (1/√a approx) |
| `simd_min_ps` | `simd_float a, simd_float b` | `simd_float` |
| `simd_max_ps` | `simd_float a, simd_float b` | `simd_float` |
| `simd_and_ps` | `simd_float a, simd_float b` | `simd_float` |
| `simd_or_ps` | `simd_float a, simd_float b` | `simd_float` |
| `simd_xor_ps` | `simd_float a, simd_float b` | `simd_float` |
| `simd_cmp_lt_ps` | `simd_float a, simd_float b` | `simd_float` (mask) |
| `simd_cmp_gt_ps` | `simd_float a, simd_float b` | `simd_float` (mask) |
| `simd_cmp_le_ps` | `simd_float a, simd_float b` | `simd_float` (mask) |
| `simd_cmp_ge_ps` | `simd_float a, simd_float b` | `simd_float` (mask) |
| `simd_blendv_ps` | `simd_float a, simd_float b, simd_float mask` | `simd_float` |

### Integer Operations

| Macro | Input | Output |
|-------|-------|--------|
| `simd_load_si` | `const void* p` | `simd_int` |
| `simd_store_si` | `void* p, simd_int v` | `void` |
| `simd_set1_epi32` | `int32_t x` | `simd_int` |
| `simd_add_epi32` | `simd_int a, simd_int b` | `simd_int` |
| `simd_sub_epi32` | `simd_int a, simd_int b` | `simd_int` |
| `simd_mullo_epi32` | `simd_int a, simd_int b` | `simd_int` |

### Conversion

| Macro | Input | Output |
|-------|-------|--------|
| `simd_cvt_ps_epi32` | `simd_float a` | `simd_int` |
| `simd_cvt_epi32_ps` | `simd_int a` | `simd_float` |

### Loop Helpers

```c
SIMD_LOOP_START(count);  // Declares _simd_count (aligned count)
for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
    simd_float v = simd_load_ps(&data[i]);
    // Process SIMD_WIDTH elements
}
SIMD_LOOP_REMAINDER(i, count) {
    // Handle remaining elements (scalar)
}
```

---

## Atomic Operations API

### Types

| Type | Underlying |
|------|------------|
| `atomic_u32` | `atomic_uint_fast32_t` |
| `atomic_u64` | `atomic_uint_fast64_t` |
| `atomic_i32` | `atomic_int_fast32_t` |
| `atomic_flag_t` | `atomic_bool` |

### Relaxed Operations (Counters/Statistics)

| Macro | Input | Output |
|-------|-------|--------|
| `atomic_load_relaxed` | `atomic_T* p` | `T` |
| `atomic_store_relaxed` | `atomic_T* p, T v` | `void` |
| `atomic_add_relaxed` | `atomic_T* p, T v` | `T` (previous value) |
| `atomic_sub_relaxed` | `atomic_T* p, T v` | `T` (previous value) |

### Acquire-Release Operations (Synchronization)

| Macro | Input | Output |
|-------|-------|--------|
| `atomic_load_acquire` | `atomic_T* p` | `T` |
| `atomic_store_release` | `atomic_T* p, T v` | `void` |
| `atomic_add_release` | `atomic_T* p, T v` | `T` (previous value) |

### Compare-and-Swap

| Macro | Input | Output |
|-------|-------|--------|
| `atomic_cas_weak` | `atomic_T* p, T* expected, T desired` | `bool` |
| `atomic_cas_strong` | `atomic_T* p, T* expected, T desired` | `bool` |

### Memory Fences

| Macro | Description |
|-------|-------------|
| `atomic_fence_acquire()` | Acquire fence |
| `atomic_fence_release()` | Release fence |
| `atomic_fence_seq_cst()` | Sequentially consistent fence |

---

## Debug Macros

Enabled when `FOUNDATION_DEBUG` is defined.

| Macro | Usage |
|-------|-------|
| `FOUNDATION_ASSERT(cond, msg)` | Runtime assertion with message |
| `FOUNDATION_LOG(fmt, ...)` | Debug logging (printf-style) |
| `FOUNDATION_LOG_ERROR(fmt, ...)` | Error logging with file:line |
| `FOUNDATION_STATIC_ASSERT(cond, msg)` | Compile-time assertion |

---

## Compiler Hints

| Macro | Description |
|-------|-------------|
| `FOUNDATION_INLINE` | Force inline hint |
| `FOUNDATION_LIKELY(x)` | Branch prediction hint (likely true) |
| `FOUNDATION_UNLIKELY(x)` | Branch prediction hint (likely false) |
| `FOUNDATION_RESTRICT` | Restrict pointer hint |
| `FOUNDATION_ALIGN(n)` | Alignment attribute |

---

## Quick Examples

### Arena Allocation

```c
Arena* arena = arena_create(1024 * 1024);  // 1MB arena

Vec3* positions = arena_alloc_array(arena, Vec3, 1000);
Quat* rotations = arena_alloc_array(arena, Quat, 1000);

arena_reset(arena);      // Free all at once
arena_destroy(arena);    // Release arena
```

### Vector Math

```c
Vec3 a = VEC3(1.0f, 2.0f, 3.0f);
Vec3 b = VEC3(4.0f, 5.0f, 6.0f);

Vec3 sum = vec3_add(a, b);
float dot = vec3_dot(a, b);
Vec3 cross = vec3_cross(a, b);
Vec3 norm = vec3_normalize(a);
```

### Quaternion Rotation

```c
Quat rotation = quat_from_euler(0.0f, 0.0f, 3.14159f / 2.0f);  // 90° yaw
Vec3 forward = VEC3(1.0f, 0.0f, 0.0f);
Vec3 rotated = quat_rotate(rotation, forward);  // Now points along Y
```

### Random Numbers

```c
PCG32 rng;
pcg32_seed(&rng, 12345);

uint32_t dice = pcg32_bounded(&rng, 6) + 1;   // 1-6
float t = pcg32_float(&rng);                   // [0, 1)
Vec3 dir = pcg32_vec3_unit(&rng);              // Random direction
```

### SIMD Loop

```c
alignas(32) float data[1000];

SIMD_LOOP_START(1000);
for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
    simd_float v = simd_load_ps(&data[i]);
    v = simd_mul_ps(v, simd_set1_ps(2.0f));
    simd_store_ps(&data[i], v);
}
SIMD_LOOP_REMAINDER(i, 1000) {
    data[i] *= 2.0f;
}
```
