/**
 * @file test_urdf.c
 * @brief Unit tests for URDF parser
 */

#include "test_harness.h"
#include "urdf_parser.h"
#include "configuration.h"
#include <stdlib.h>

/* ============================================================================
 * Test URDF Content
 * ============================================================================ */

static const char* MINIMAL_URDF =
    "<?xml version=\"1.0\"?>\n"
    "<robot name=\"test_drone\">\n"
    "  <link name=\"base\">\n"
    "    <inertial>\n"
    "      <mass value=\"0.5\"/>\n"
    "      <inertia ixx=\"0.001\" iyy=\"0.002\" izz=\"0.003\"/>\n"
    "    </inertial>\n"
    "  </link>\n"
    "</robot>\n";

static const char* FULL_URDF =
    "<?xml version=\"1.0\"?>\n"
    "<robot name=\"quadcopter\">\n"
    "  <link name=\"base_link\">\n"
    "    <inertial>\n"
    "      <mass value=\"1.5\"/>\n"
    "      <inertia ixx=\"0.01\" iyy=\"0.01\" izz=\"0.02\"\n"
    "               ixy=\"0\" ixz=\"0\" iyz=\"0\"/>\n"
    "    </inertial>\n"
    "    <collision>\n"
    "      <geometry>\n"
    "        <cylinder radius=\"0.25\" length=\"0.1\"/>\n"
    "      </geometry>\n"
    "    </collision>\n"
    "    <properties arm=\"0.15\" kf=\"1e-6\" km=\"1e-8\"\n"
    "                motor_tau=\"0.05\" max_rpm=\"10000\"/>\n"
    "  </link>\n"
    "</robot>\n";

static const char* SPHERE_COLLISION_URDF =
    "<?xml version=\"1.0\"?>\n"
    "<robot name=\"sphere_bot\">\n"
    "  <link name=\"body\">\n"
    "    <collision>\n"
    "      <geometry>\n"
    "        <sphere radius=\"0.5\"/>\n"
    "      </geometry>\n"
    "    </collision>\n"
    "  </link>\n"
    "</robot>\n";

static const char* BOX_COLLISION_URDF =
    "<?xml version=\"1.0\"?>\n"
    "<robot name=\"box_bot\">\n"
    "  <link name=\"body\">\n"
    "    <collision>\n"
    "      <geometry>\n"
    "        <box size=\"0.4 0.3 0.2\"/>\n"
    "      </geometry>\n"
    "    </collision>\n"
    "  </link>\n"
    "</robot>\n";

static const char* INVALID_XML =
    "<?xml version=\"1.0\"?>\n"
    "<robot name=\"broken\"\n"  /* Missing closing > */
    "  <link name=\"base\"/>\n"
    "</robot>\n";

/* ============================================================================
 * Tests
 * ============================================================================ */

TEST(properties_init) {
    URDFProperties props;
    urdf_properties_init(&props);

    ASSERT_EQ(props.mass, 0.0f);
    ASSERT_EQ(props.ixx, 0.0f);
    ASSERT_FALSE(props.has_inertial);
    ASSERT_FALSE(props.has_collision);
    ASSERT_FALSE(props.has_properties);

    return 0;
}

TEST(parse_minimal_urdf) {
    URDFProperties props;
    char error[256] = {0};

    int result = urdf_parse_string(MINIMAL_URDF, &props, error);
    ASSERT_EQ(result, 0);

    ASSERT_STR_EQ(props.robot_name, "test_drone");
    ASSERT_TRUE(props.has_inertial);
    ASSERT_FLOAT_NEAR(props.mass, 0.5f, 1e-6f);
    ASSERT_FLOAT_NEAR(props.ixx, 0.001f, 1e-6f);
    ASSERT_FLOAT_NEAR(props.iyy, 0.002f, 1e-6f);
    ASSERT_FLOAT_NEAR(props.izz, 0.003f, 1e-6f);

    return 0;
}

TEST(parse_full_urdf) {
    URDFProperties props;
    char error[256] = {0};

    int result = urdf_parse_string(FULL_URDF, &props, error);
    ASSERT_EQ(result, 0);

    ASSERT_STR_EQ(props.robot_name, "quadcopter");

    /* Inertial */
    ASSERT_TRUE(props.has_inertial);
    ASSERT_FLOAT_NEAR(props.mass, 1.5f, 1e-6f);
    ASSERT_FLOAT_NEAR(props.ixx, 0.01f, 1e-6f);
    ASSERT_FLOAT_NEAR(props.iyy, 0.01f, 1e-6f);
    ASSERT_FLOAT_NEAR(props.izz, 0.02f, 1e-6f);

    /* Collision */
    ASSERT_TRUE(props.has_collision);
    ASSERT_FLOAT_NEAR(props.collision_radius, 0.25f, 1e-6f);
    ASSERT_FLOAT_NEAR(props.collision_length, 0.1f, 1e-6f);

    /* Custom properties */
    ASSERT_TRUE(props.has_properties);
    ASSERT_FLOAT_NEAR(props.arm_length, 0.15f, 1e-6f);
    ASSERT_FLOAT_NEAR(props.k_thrust, 1e-6f, 1e-10f);
    ASSERT_FLOAT_NEAR(props.k_torque, 1e-8f, 1e-12f);
    ASSERT_FLOAT_NEAR(props.motor_tau, 0.05f, 1e-6f);
    ASSERT_FLOAT_NEAR(props.max_rpm, 10000.0f, 1.0f);

    return 0;
}

TEST(parse_sphere_collision) {
    URDFProperties props;
    char error[256] = {0};

    int result = urdf_parse_string(SPHERE_COLLISION_URDF, &props, error);
    ASSERT_EQ(result, 0);

    ASSERT_TRUE(props.has_collision);
    ASSERT_FLOAT_NEAR(props.collision_radius, 0.5f, 1e-6f);

    return 0;
}

TEST(parse_box_collision) {
    URDFProperties props;
    char error[256] = {0};

    int result = urdf_parse_string(BOX_COLLISION_URDF, &props, error);
    ASSERT_EQ(result, 0);

    ASSERT_TRUE(props.has_collision);
    /* Box size is 0.4 x 0.3 x 0.2, min dimension is 0.2, radius = 0.1 */
    ASSERT_FLOAT_NEAR(props.collision_radius, 0.1f, 1e-6f);

    return 0;
}

TEST(parse_invalid_xml) {
    URDFProperties props;
    char error[256] = {0};

    int result = urdf_parse_string(INVALID_XML, &props, error);
    ASSERT_EQ(result, -4);  /* XML parse error */
    ASSERT_TRUE(strlen(error) > 0);

    return 0;
}

TEST(parse_empty_string) {
    URDFProperties props;
    char error[256] = {0};

    int result = urdf_parse_string("", &props, error);
    ASSERT_EQ(result, -4);  /* Incomplete XML */

    return 0;
}

TEST(parse_null_arguments) {
    URDFProperties props;
    char error[256] = {0};

    ASSERT_EQ(urdf_parse_string(NULL, &props, error), -1);
    ASSERT_EQ(urdf_parse_string(MINIMAL_URDF, NULL, error), -1);

    return 0;
}

TEST(apply_to_drone_config) {
    URDFProperties urdf;
    urdf_properties_init(&urdf);

    /* Set up URDF properties */
    strcpy(urdf.robot_name, "test_quad");
    urdf.has_inertial = true;
    urdf.mass = 2.0f;
    urdf.ixx = 0.05f;
    urdf.iyy = 0.06f;
    urdf.izz = 0.07f;
    urdf.has_collision = true;
    urdf.collision_radius = 0.3f;
    urdf.has_properties = true;
    urdf.arm_length = 0.2f;
    urdf.k_thrust = 5e-6f;
    urdf.k_torque = 3e-8f;
    urdf.motor_tau = 0.03f;
    urdf.max_rpm = 15000.0f;

    /* Create drone config with defaults */
    DroneConfig config;
    drone_config_set_defaults(&config);

    /* Apply URDF */
    urdf_apply_to_drone_config(&urdf, &config);

    /* Verify mapping */
    ASSERT_STR_EQ(config.name, "test_quad");
    ASSERT_FLOAT_NEAR(config.mass, 2.0f, 1e-6f);
    ASSERT_FLOAT_NEAR(config.ixx, 0.05f, 1e-6f);
    ASSERT_FLOAT_NEAR(config.iyy, 0.06f, 1e-6f);
    ASSERT_FLOAT_NEAR(config.izz, 0.07f, 1e-6f);
    ASSERT_FLOAT_NEAR(config.collision_radius, 0.3f, 1e-6f);
    ASSERT_FLOAT_NEAR(config.arm_length, 0.2f, 1e-6f);
    ASSERT_FLOAT_NEAR(config.k_thrust, 5e-6f, 1e-10f);
    ASSERT_FLOAT_NEAR(config.k_torque, 3e-8f, 1e-12f);
    ASSERT_FLOAT_NEAR(config.motor_tau, 0.03f, 1e-6f);
    ASSERT_FLOAT_NEAR(config.max_rpm, 15000.0f, 1.0f);

    return 0;
}

TEST(apply_partial_urdf) {
    /* Test that only found properties are overwritten */
    URDFProperties urdf;
    urdf_properties_init(&urdf);

    urdf.has_inertial = true;
    urdf.mass = 3.0f;
    /* ixx/iyy/izz left at 0, should not overwrite */

    DroneConfig config;
    drone_config_set_defaults(&config);
    float original_ixx = config.ixx;  /* Save default value */

    urdf_apply_to_drone_config(&urdf, &config);

    ASSERT_FLOAT_NEAR(config.mass, 3.0f, 1e-6f);
    ASSERT_FLOAT_NEAR(config.ixx, original_ixx, 1e-10f);  /* Unchanged */

    return 0;
}

TEST(validate_valid_properties) {
    URDFProperties props;
    urdf_properties_init(&props);
    char error[256] = {0};

    props.has_inertial = true;
    props.mass = 1.0f;
    props.ixx = props.iyy = props.izz = 0.01f;

    ASSERT_EQ(urdf_properties_validate(&props, error), 0);

    return 0;
}

TEST(validate_invalid_mass) {
    URDFProperties props;
    urdf_properties_init(&props);
    char error[256] = {0};

    props.has_inertial = true;
    props.mass = -1.0f;  /* Invalid */
    props.ixx = props.iyy = props.izz = 0.01f;

    ASSERT_EQ(urdf_properties_validate(&props, error), -1);
    ASSERT_TRUE(strstr(error, "mass") != NULL);

    return 0;
}

TEST(validate_invalid_inertia) {
    URDFProperties props;
    urdf_properties_init(&props);
    char error[256] = {0};

    props.has_inertial = true;
    props.mass = 1.0f;
    props.ixx = 0.0f;  /* Invalid */
    props.iyy = props.izz = 0.01f;

    ASSERT_EQ(urdf_properties_validate(&props, error), -1);
    ASSERT_TRUE(strstr(error, "inertia") != NULL);

    return 0;
}

TEST(parse_crazyflie_urdf_file) {
    /* Test parsing actual crazyflie.urdf file */
    URDFProperties props;
    char error[256] = {0};

    int result = urdf_parse_file("test_data/crazyflie.urdf", &props, error);
    if (result != 0) {
        /* Try alternate path if running from different directory */
        result = urdf_parse_file("rl_engine/src/configuration/test_data/crazyflie.urdf", &props, error);
    }
    ASSERT_EQ(result, 0);

    ASSERT_STR_EQ(props.robot_name, "crazyflie2");

    /* Verify Crazyflie 2.0 parameters */
    ASSERT_TRUE(props.has_inertial);
    ASSERT_FLOAT_NEAR(props.mass, 0.027f, 1e-6f);
    ASSERT_FLOAT_NEAR(props.ixx, 1.4e-5f, 1e-8f);
    ASSERT_FLOAT_NEAR(props.iyy, 1.4e-5f, 1e-8f);
    ASSERT_FLOAT_NEAR(props.izz, 2.17e-5f, 1e-8f);

    ASSERT_TRUE(props.has_collision);
    ASSERT_FLOAT_NEAR(props.collision_radius, 0.056f, 1e-4f);

    ASSERT_TRUE(props.has_properties);
    ASSERT_FLOAT_NEAR(props.arm_length, 0.046f, 1e-4f);
    ASSERT_FLOAT_NEAR(props.k_thrust, 2.88e-8f, 1e-12f);
    ASSERT_FLOAT_NEAR(props.k_torque, 7.24e-10f, 1e-14f);
    ASSERT_FLOAT_NEAR(props.motor_tau, 0.02f, 1e-4f);
    ASSERT_FLOAT_NEAR(props.max_rpm, 21702.0f, 1.0f);

    return 0;
}

TEST(parse_nonexistent_file) {
    URDFProperties props;
    char error[256] = {0};

    int result = urdf_parse_file("/nonexistent/path/to/file.urdf", &props, error);
    ASSERT_EQ(result, -1);  /* File open error */
    ASSERT_TRUE(strlen(error) > 0);

    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("URDF Parser");

    /* Initialization tests */
    RUN_TEST(properties_init);

    /* Parsing tests */
    RUN_TEST(parse_minimal_urdf);
    RUN_TEST(parse_full_urdf);
    RUN_TEST(parse_sphere_collision);
    RUN_TEST(parse_box_collision);
    RUN_TEST(parse_invalid_xml);
    RUN_TEST(parse_empty_string);
    RUN_TEST(parse_null_arguments);

    /* Configuration mapping tests */
    RUN_TEST(apply_to_drone_config);
    RUN_TEST(apply_partial_urdf);

    /* Validation tests */
    RUN_TEST(validate_valid_properties);
    RUN_TEST(validate_invalid_mass);
    RUN_TEST(validate_invalid_inertia);

    /* File tests */
    RUN_TEST(parse_crazyflie_urdf_file);
    RUN_TEST(parse_nonexistent_file);

    TEST_SUITE_END();
}
