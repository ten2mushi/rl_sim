"""
Setup script for rl_engine Python package

Builds the C extension binding for PufferLib integration.

Prerequisites:
    1. Build the rl_engine C library first:
       mkdir -p build && cd build && cmake .. && make -j8

    2. Install the Python package:
       pip install -e .

Usage:
    pip install -e .                    # Development install
    pip install -e . --verbose          # Verbose output
    python setup.py build_ext --inplace # Build extension in place
"""

import os
import sys
import platform
import subprocess
from pathlib import Path

import numpy
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeBuildExt(build_ext):
    """Custom build command that runs CMake first if needed."""

    def run(self):
        # Ensure the C library is built
        source_dir = Path(__file__).parent.absolute()
        build_dir = source_dir / "build"

        if not (build_dir / "lib" / "libdronerl.a").exists():
            print("=" * 60)
            print("Building rl_engine C library with CMake...")
            print("=" * 60)

            build_dir.mkdir(exist_ok=True)

            # Run CMake configure
            cmake_args = [
                "cmake",
                str(source_dir),
                f"-DCMAKE_BUILD_TYPE=Release",
                f"-DBUILD_TESTING=OFF",
                f"-DBUILD_BENCHMARKS=OFF",
            ]

            # Add architecture-specific flags
            if platform.system() == "Darwin":
                # macOS
                cmake_args.append("-DCMAKE_OSX_DEPLOYMENT_TARGET=10.14")
            elif platform.system() == "Linux":
                pass  # Linux defaults are fine

            subprocess.check_call(cmake_args, cwd=build_dir)

            # Run CMake build
            subprocess.check_call(
                ["cmake", "--build", ".", "--config", "Release", "-j8"],
                cwd=build_dir,
            )

            print("=" * 60)
            print("C library built successfully!")
            print("=" * 60)

        # Now build the Python extension
        super().run()


# Paths
SOURCE_DIR = Path(__file__).parent.absolute()
BUILD_DIR = SOURCE_DIR / "build"

# PufferLib env_binding.h location (adjust based on your PufferLib install)
PUFFERLIB_INCLUDE = Path(os.environ.get(
    "PUFFERLIB_INCLUDE",
    SOURCE_DIR.parent / "source" / "PufferLib" / "pufferlib" / "ocean"
))

# Include directories
INCLUDE_DIRS = [
    numpy.get_include(),
    str(SOURCE_DIR / "include"),
    str(SOURCE_DIR / "src" / "foundation" / "include"),
    str(SOURCE_DIR / "src" / "drone_state" / "include"),
    str(SOURCE_DIR / "src" / "physics" / "include"),
    str(SOURCE_DIR / "src" / "world_brick_map" / "include"),
    str(SOURCE_DIR / "src" / "collision_system" / "include"),
    str(SOURCE_DIR / "src" / "sensor_system" / "include"),
    str(SOURCE_DIR / "src" / "sensor_implementations" / "include"),
    str(SOURCE_DIR / "src" / "reward_system" / "include"),
    str(SOURCE_DIR / "src" / "threading" / "include"),
    str(SOURCE_DIR / "src" / "configuration" / "include"),
    str(SOURCE_DIR / "src" / "configuration" / "external"),
    str(SOURCE_DIR / "src" / "environment_manager" / "include"),
    str(SOURCE_DIR / "src" / "gpu" / "include"),
    str(PUFFERLIB_INCLUDE),
]

# Library directories
LIBRARY_DIRS = [
    str(BUILD_DIR / "lib"),
]

# Libraries to link
LIBRARIES = ["dronerl"]

# Platform-specific configuration
EXTRA_COMPILE_ARGS = [
    "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
    "-Wall",
    "-Wno-unused-parameter",
]

EXTRA_LINK_ARGS = []

if platform.system() == "Darwin":
    # macOS
    EXTRA_COMPILE_ARGS.extend([
        "-Wno-error=int-conversion",
        "-Wno-error=incompatible-function-pointer-types",
    ])
    EXTRA_LINK_ARGS.extend([
        "-framework", "Accelerate",  # For vecLib BLAS
        "-framework", "Metal",       # GPU compute shaders
        "-framework", "Foundation",  # Metal runtime dependency
    ])
    # Check for ARM vs x86
    if platform.machine() == "arm64":
        # Apple Silicon
        pass  # NEON is default
    else:
        # Intel Mac
        EXTRA_COMPILE_ARGS.extend(["-mavx2", "-mfma"])

elif platform.system() == "Linux":
    # Linux
    EXTRA_COMPILE_ARGS.extend([
        "-Wno-alloc-size-larger-than",
    ])
    LIBRARIES.extend(["pthread", "m"])
    # Assume x86_64 with AVX2 (can be made configurable)
    if platform.machine() == "x86_64":
        EXTRA_COMPILE_ARGS.extend(["-mavx2", "-mfma"])

# Release optimizations
if os.environ.get("DEBUG", "0") != "1":
    EXTRA_COMPILE_ARGS.extend(["-O3", "-DNDEBUG"])
else:
    EXTRA_COMPILE_ARGS.extend(["-O0", "-g", "-DFOUNDATION_DEBUG=1"])

# Extension module
# Name is "binding" (not "rl_engine.binding") because setup.py lives inside
# the rl_engine/ package directory. --inplace places the .so right here,
# next to __init__.py and drone.py.
binding_extension = Extension(
    name="binding",
    sources=[str(SOURCE_DIR / "binding.c")],
    include_dirs=INCLUDE_DIRS,
    library_dirs=LIBRARY_DIRS,
    libraries=LIBRARIES,
    extra_compile_args=EXTRA_COMPILE_ARGS,
    extra_link_args=EXTRA_LINK_ARGS,
    extra_objects=[str(BUILD_DIR / "lib" / "libdronerl.a")],  # Static link
)

# Package setup
setup(
    name="rl_engine",
    version="1.0.0",
    description="High-performance drone swarm RL environment with PufferLib integration",
    long_description=open(SOURCE_DIR / "README.md").read()
    if (SOURCE_DIR / "README.md").exists()
    else "High-performance drone swarm RL environment",
    long_description_content_type="text/markdown",
    author="Drone RL Team",
    python_requires=">=3.8",
    packages=find_packages(),
    ext_modules=[binding_extension],
    cmdclass={"build_ext": CMakeBuildExt},
    install_requires=[
        "numpy>=1.21,<2.0",
        "gymnasium>=0.29.1",
        "pufferlib>=2.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-benchmark",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: C",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
