# Foundation Module (00-Foundation)

Core types, arena allocators, SIMD utilities, and math primitives that all other RL Engine submodules depend on.

## Overview

The Foundation module provides the base layer of the RL Engine with **zero external dependencies**. It includes:

- **Basic Types**: Vec3, Quat, Mat3, Mat4 with SIMD-aligned padding
- **Arena Allocator**: O(1) bump allocation with scoped allocation support
- **SIMD Abstraction**: Unified API for AVX2, ARM NEON, and scalar fallback
- **Math Utilities**: Vector, quaternion, and matrix operations
- **Atomics**: Lock-free primitives with memory ordering semantics
- **PCG32 RNG**: Fast, statistically excellent random number generation
- **Debug Utilities**: Assertions and logging macros

## Directory Structure

```
foundation/
├── include/
│   └── foundation.h      # Single master header
├── src/
│   ├── arena.c           # Arena allocator implementation
│   ├── math.c            # Math utilities (sqrt/trig operations)
│   └── pcg32.c           # PCG32 random number generator
├── tests/
│   ├── test_arena.c      # Arena allocator tests
│   ├── test_math.c       # Math operation tests
│   ├── test_pcg32.c      # RNG distribution tests
│   ├── test_simd.c       # SIMD abstraction tests
│   └── bench_foundation.c # Performance benchmarks
├── CMakeLists.txt        # Build configuration
└── README.md             # This file
```

## Building

### Using CMake

```bash
# Configure
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build .

# Run tests
ctest --output-on-failure

# Run benchmarks
./bench_foundation
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `FOUNDATION_DEBUG` | OFF | Enable debug assertions and logging |
| `FOUNDATION_BUILD_TESTS` | ON | Build unit tests |
| `FOUNDATION_BUILD_BENCHMARKS` | ON | Build performance benchmarks |
| `FOUNDATION_ENABLE_AVX2` | ON | Enable AVX2 on x86_64 |
| `FOUNDATION_ENABLE_NEON` | ON | Enable NEON on ARM |

## Usage

### Including the Header

```c
#include "foundation.h"
```

### Basic Types

```c
// Vectors (16-byte aligned)
Vec3 position = VEC3(1.0f, 2.0f, 3.0f);
Vec3 velocity = vec3_zero();

// Quaternions (16-byte aligned)
Quat rotation = QUAT_IDENTITY;
Quat q = quat_from_euler(roll, pitch, yaw);

// Matrices (32-byte aligned)
Mat3 rot_mat = quat_to_mat3(rotation);
Mat4 transform = mat4_from_trs(position, rotation, VEC3_ONE);
```

### Arena Allocator

```c
// Create arena with 1MB capacity
Arena* arena = arena_create(1024 * 1024);

// Allocate memory (O(1) bump allocation)
Vec3* positions = arena_alloc_array(arena, Vec3, 1000);
Quat* rotations = arena_alloc_array(arena, Quat, 1000);

// Scoped allocation (automatically resets on scope exit)
ARENA_SCOPE(arena) {
    float* temp = arena_alloc_array(arena, float, 10000);
    // Use temp...
} // Memory automatically reclaimed here

// Reset all allocations
arena_reset(arena);

// Cleanup
arena_destroy(arena);
```

### Vector Operations

```c
Vec3 a = VEC3(1.0f, 2.0f, 3.0f);
Vec3 b = VEC3(4.0f, 5.0f, 6.0f);

// Arithmetic (all inlined)
Vec3 sum = vec3_add(a, b);
Vec3 diff = vec3_sub(a, b);
Vec3 scaled = vec3_scale(a, 2.0f);

// Products
float dot = vec3_dot(a, b);
Vec3 cross = vec3_cross(a, b);

// Length operations
float len = vec3_length(a);
Vec3 normalized = vec3_normalize(a);
float dist = vec3_distance(a, b);

// Interpolation
Vec3 lerped = vec3_lerp(a, b, 0.5f);
```

### Quaternion Operations

```c
// Create rotations
Quat from_euler = quat_from_euler(roll, pitch, yaw);
Quat from_axis = quat_from_axis_angle(VEC3_UP, PI / 2.0f);

// Combine rotations
Quat combined = quat_multiply(q1, q2);

// Rotate vectors
Vec3 rotated = quat_rotate(rotation, position);

// Interpolate rotations
Quat slerped = quat_slerp(q1, q2, t);

// Convert to matrix
Mat3 rot_mat = quat_to_mat3(rotation);
```

### PCG32 Random Number Generation

```c
// Create and seed RNG
PCG32 rng;
pcg32_seed(&rng, 12345);

// Generate random numbers
uint32_t rand_int = pcg32_random(&rng);
uint32_t bounded = pcg32_bounded(&rng, 100);  // [0, 100)

// Generate floats
float f = pcg32_float(&rng);                   // [0, 1)
float ranged = pcg32_range(&rng, -10.0f, 10.0f);

// Generate random vectors
Vec3 unit = pcg32_vec3_unit(&rng);             // Random unit vector
Vec3 ranged_vec = pcg32_vec3_range(&rng, min, max);

// Thread-local RNG
PCG32* tls_rng = pcg32_thread_local();
```

### SIMD Abstraction

```c
// SIMD types (automatically selected for platform)
alignas(32) float data[FOUNDATION_SIMD_WIDTH];

// Load/store
simd_float v = simd_load_ps(data);
simd_store_ps(data, v);

// Arithmetic
simd_float sum = simd_add_ps(a, b);
simd_float prod = simd_mul_ps(a, b);
simd_float fma = simd_fmadd_ps(a, b, c);  // a * b + c

// Loop helpers
SIMD_LOOP_START(count);
for (uint32_t i = 0; i < _simd_count; i += FOUNDATION_SIMD_WIDTH) {
    simd_float v = simd_load_ps(&array[i]);
    // Process SIMD_WIDTH elements at once
}
SIMD_LOOP_REMAINDER(i, count) {
    // Handle remaining elements
}
```

### Atomics

```c
atomic_u32 counter = ATOMIC_VAR_INIT(0);

// Relaxed operations (counters, statistics)
atomic_add_relaxed(&counter, 1);
uint32_t val = atomic_load_relaxed(&counter);

// Acquire-release (synchronization)
atomic_store_release(&flag, 1);
while (!atomic_load_acquire(&flag)) { /* spin */ }

// Compare-and-swap
uint32_t expected = 0;
bool success = atomic_cas_strong(&counter, &expected, 1);
```

## Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Arena allocation | <10 cycles | O(1) bump pointer |
| Arena reset | <5 cycles | Single write |
| Vec3 operations | Native | Inlined |
| Quat normalize | <20 cycles | With rsqrt |
| PCG32 random | <15 cycles | Per 32-bit output |

Run `bench_foundation` to verify performance on your system.

## Type Sizes and Alignment

| Type | Size | Alignment | Notes |
|------|------|-----------|-------|
| Vec3 | 16 bytes | 16-byte | 12 bytes data + 4 padding |
| Quat | 16 bytes | 16-byte | |
| Mat3 | 48 bytes | 32-byte | 36 bytes data + 12 padding |
| Mat4 | 64 bytes | 32-byte | |
| PCG32 | 16 bytes | 8-byte | |

## Platform Support

| Platform | SIMD | Width |
|----------|------|-------|
| x86_64 + AVX2 | AVX2 | 8 |
| ARM64 (M-series, etc) | NEON | 4 |
| Other | Scalar | 1 |

## Debug Mode

Enable debug assertions by defining `FOUNDATION_DEBUG`:

```c
#define FOUNDATION_DEBUG 1
#include "foundation.h"
```

Or via CMake:

```bash
cmake .. -DFOUNDATION_DEBUG=ON
```

This enables:
- `FOUNDATION_ASSERT(cond, msg)` - Runtime assertions
- `FOUNDATION_LOG(fmt, ...)` - Debug logging

## Dependencies

**None** - The foundation module is self-contained and has zero external dependencies.

## Modules That Depend On Foundation

| Module | Components Used |
|--------|-----------------|
| 01-SoA Arrays | Vec3, Quat, Arena, SIMD |
| 02-Physics | Vec3, Quat, Mat3, math |
| 03-SDF World | Vec3, Arena, align_up |
| 04-Collision | Vec3, SIMD loops, Arena |
| 05-Sensor | Vec3, Arena, PCG32 |
| 06-Batch Rendering | Vec3, Quat, SIMD, Arena |
| 07-RL Interface | Arena, basic types |
| 08-Threading | Atomics, Arena |
| 09-Config | Basic types |
| 10-Python Binding | All types (marshalling) |


  Bonus Features Beyond Spec

  - Complete 3×3 and 4×4 matrix operations
  - Additional SIMD comparison operations (simd_or_ps, simd_xor_ps, simd_cmp_le_ps, simd_cmp_ge_ps)
  - Enhanced vector operations (vec3_div, vec3_abs, vec3_lerp, vec3_reflect)
  - Enhanced quaternion operations (quat_dot, quat_to_mat4, quat_from_mat3)
  - Typed allocation macros (arena_alloc_type, arena_alloc_array)