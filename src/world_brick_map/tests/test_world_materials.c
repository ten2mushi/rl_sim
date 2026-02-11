/**
 * Tests for WorldBrickMap Material Registry
 *
 * Tests material registration, lookup, and metadata management.
 */

#include "../include/world_brick_map.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "test_harness.h"

/* Test 1: Default material initialization */
TEST(default_material) {
  Arena *arena = arena_create(10 * 1024 * 1024); /* 10 MB */
  WorldBrickMap *world =
      world_create(arena, VEC3(0.0f, 0.0f, 0.0f), VEC3(100.0f, 100.0f, 100.0f),
                   0.1f, 1000, 256);

  ASSERT_NOT_NULL(world);
  ASSERT_TRUE(world->material_count == 1); /* Default material exists */
  ASSERT_TRUE(world->max_materials == 256);

  /* Check default material properties */
  const MaterialMetadata *default_mat = world_get_material(world, 0);
  ASSERT_NOT_NULL(default_mat);
  ASSERT_TRUE(strcmp(default_mat->name, "default") == 0);
  ASSERT_TRUE(default_mat->id == 0);
  ASSERT_FLOAT_NEAR(default_mat->diffuse_color.x, 1.0f, 0.001f);
  ASSERT_FLOAT_NEAR(default_mat->diffuse_color.y, 1.0f, 0.001f);
  ASSERT_FLOAT_NEAR(default_mat->diffuse_color.z, 1.0f, 0.001f);

  arena_destroy(arena);
  return 0;
}

/* Test 2: Material registration */
TEST(material_registration) {
  Arena *arena = arena_create(10 * 1024 * 1024);
  WorldBrickMap *world =
      world_create(arena, VEC3(0.0f, 0.0f, 0.0f), VEC3(100.0f, 100.0f, 100.0f),
                   0.1f, 1000, 256);

  /* Register a new material */
  Vec3 red = VEC3(1.0f, 0.0f, 0.0f);
  uint8_t mat_id = world_register_material(world, "terrain_rock", red);

  ASSERT_TRUE(mat_id == 1); /* First registered material after default */
  ASSERT_TRUE(world->material_count == 2);

  /* Verify material metadata */
  const MaterialMetadata *mat = world_get_material(world, mat_id);
  ASSERT_NOT_NULL(mat);
  ASSERT_TRUE(mat->id == 1);
  ASSERT_TRUE(strcmp(mat->name, "terrain_rock") == 0);
  ASSERT_FLOAT_NEAR(mat->diffuse_color.x, 1.0f, 0.001f);
  ASSERT_FLOAT_NEAR(mat->diffuse_color.y, 0.0f, 0.001f);
  ASSERT_FLOAT_NEAR(mat->diffuse_color.z, 0.0f, 0.001f);

  arena_destroy(arena);
  return 0;
}

/* Test 3: Duplicate material registration */
TEST(duplicate_material) {
  Arena *arena = arena_create(10 * 1024 * 1024);
  WorldBrickMap *world =
      world_create(arena, VEC3(0.0f, 0.0f, 0.0f), VEC3(100.0f, 100.0f, 100.0f),
                   0.1f, 1000, 256);

  Vec3 red = VEC3(1.0f, 0.0f, 0.0f);
  Vec3 blue = VEC3(0.0f, 0.0f, 1.0f);

  /* Register material */
  uint8_t id1 = world_register_material(world, "terrain_rock", red);
  ASSERT_TRUE(id1 == 1);
  ASSERT_TRUE(world->material_count == 2);

  /* Register with same name - should return existing ID */
  uint8_t id2 = world_register_material(world, "terrain_rock",
                                        blue); /* Different color */
  ASSERT_TRUE(id2 == id1);                          /* Same ID returned */
  ASSERT_TRUE(world->material_count == 2);          /* Count unchanged */

  /* Original color should be unchanged */
  const MaterialMetadata *mat = world_get_material(world, id1);
  ASSERT_FLOAT_NEAR(mat->diffuse_color.x, 1.0f, 0.001f); /* Still red */

  arena_destroy(arena);
  return 0;
}

/* Test 4: Material lookup by name */
TEST(material_find) {
  Arena *arena = arena_create(10 * 1024 * 1024);
  WorldBrickMap *world =
      world_create(arena, VEC3(0.0f, 0.0f, 0.0f), VEC3(100.0f, 100.0f, 100.0f),
                   0.1f, 1000, 256);

  /* Register materials */
  world_register_material(world, "terrain_rock", VEC3(0.5f, 0.4f, 0.3f));
  world_register_material(world, "terrain_grass", VEC3(0.2f, 0.8f, 0.1f));
  world_register_material(world, "terrain_water", VEC3(0.1f, 0.3f, 0.9f));

  /* Find existing materials */
  ASSERT_TRUE(world_find_material(world, "terrain_rock") == 1);
  ASSERT_TRUE(world_find_material(world, "terrain_grass") == 2);
  ASSERT_TRUE(world_find_material(world, "terrain_water") == 3);

  /* Find non-existent material - should return 0 (default) */
  ASSERT_TRUE(world_find_material(world, "nonexistent") == 0);

  /* Find with NULL name - should return 0 */
  ASSERT_TRUE(world_find_material(world, NULL) == 0);

  /* Find with empty name - should return 0 */
  ASSERT_TRUE(world_find_material(world, "") == 0);

  arena_destroy(arena);
  return 0;
}

/* Test 5: Material table overflow */
TEST(material_overflow) {
  Arena *arena = arena_create(10 * 1024 * 1024);
  WorldBrickMap *world =
      world_create(arena, VEC3(0.0f, 0.0f, 0.0f), VEC3(100.0f, 100.0f, 100.0f),
                   0.1f, 1000, 10); /* Only 10 materials max */

  ASSERT_TRUE(world->max_materials == 10);

  /* Fill up the material table */
  for (uint32_t i = 0; i < 9; i++) {
    char name[64];
    snprintf(name, sizeof(name), "material_%u", i);
    uint8_t id = world_register_material(world, name, VEC3(1.0f, 1.0f, 1.0f));
    ASSERT_TRUE(id == i + 1); /* IDs start from 1 (0 is default) */
  }

  ASSERT_TRUE(world->material_count == 10); /* Default + 9 registered */

  /* Try to register one more - should fail */
  uint8_t overflow_id =
      world_register_material(world, "overflow", VEC3(1.0f, 0.0f, 0.0f));
  ASSERT_TRUE(overflow_id == 255);          /* Indicates failure */
  ASSERT_TRUE(world->material_count == 10); /* Count unchanged */

  arena_destroy(arena);
  return 0;
}

/* Test 6: Invalid material ID query */
TEST(invalid_material_query) {
  Arena *arena = arena_create(10 * 1024 * 1024);
  WorldBrickMap *world =
      world_create(arena, VEC3(0.0f, 0.0f, 0.0f), VEC3(100.0f, 100.0f, 100.0f),
                   0.1f, 1000, 256);

  /* Query invalid IDs */
  ASSERT_NULL(world_get_material(world, 1));   /* No material at ID 1 yet */
  ASSERT_NULL(world_get_material(world, 255)); /* Out of range */

  /* Register a material */
  world_register_material(world, "test", VEC3(1.0f, 1.0f, 1.0f));

  /* Valid query */
  ASSERT_NOT_NULL(world_get_material(world, 1));
  /* Still invalid */
  ASSERT_NULL(world_get_material(world, 2));

  arena_destroy(arena);
  return 0;
}

/* Test 7: Material name truncation */
TEST(material_name_truncation) {
  Arena *arena = arena_create(10 * 1024 * 1024);
  WorldBrickMap *world =
      world_create(arena, VEC3(0.0f, 0.0f, 0.0f), VEC3(100.0f, 100.0f, 100.0f),
                   0.1f, 1000, 256);

  /* Create a very long name (>64 chars) */
  char long_name[100];
  memset(long_name, 'A', sizeof(long_name) - 1);
  long_name[sizeof(long_name) - 1] = '\0';

  uint8_t mat_id =
      world_register_material(world, long_name, VEC3(1.0f, 0.0f, 0.0f));
  ASSERT_TRUE(mat_id == 1);

  const MaterialMetadata *mat = world_get_material(world, mat_id);
  ASSERT_NOT_NULL(mat);
  /* Name should be truncated to 63 chars + null terminator */
  ASSERT_TRUE(strlen(mat->name) == 63);
  ASSERT_TRUE(mat->name[63] == '\0'); /* Null terminated */

  arena_destroy(arena);
  return 0;
}

/* Test 8: Zero max_materials uses default */
TEST(zero_max_materials) {
  Arena *arena = arena_create(10 * 1024 * 1024);
  WorldBrickMap *world =
      world_create(arena, VEC3(0.0f, 0.0f, 0.0f), VEC3(100.0f, 100.0f, 100.0f),
                   0.1f, 1000, 0); /* 0 --> default 256 */

  ASSERT_NOT_NULL(world);
  ASSERT_TRUE(world->max_materials == 256);
  ASSERT_TRUE(world->material_count == 1); /* Default material */

  arena_destroy(arena);
  return 0;
}

int main(void) {
  TEST_SUITE_BEGIN("World Brick Map Material Registry Tests");

  RUN_TEST(default_material);
  RUN_TEST(material_registration);
  RUN_TEST(duplicate_material);
  RUN_TEST(material_find);
  RUN_TEST(material_overflow);
  RUN_TEST(invalid_material_query);
  RUN_TEST(material_name_truncation);
  RUN_TEST(zero_max_materials);

  TEST_SUITE_END();
}
