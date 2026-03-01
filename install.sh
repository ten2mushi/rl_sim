#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# install.sh — One-command build for rl_engine
#
# Usage: ./install.sh [--debug] [--clean] [--help]
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Colors ----------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { printf "${BLUE}[INFO]${NC}  %s\n" "$*"; }
ok()    { printf "${GREEN}[OK]${NC}    %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
err()   { printf "${RED}[ERROR]${NC} %s\n" "$*" >&2; }

trap 'err "Build failed at line $LINENO. Run with bash -x install.sh for details."' ERR

# --- Parse args ------------------------------------------------------------
BUILD_TYPE="Release"
DEBUG_FLAG=0
CLEAN=0

for arg in "$@"; do
    case "$arg" in
        --debug) BUILD_TYPE="Debug"; DEBUG_FLAG=1 ;;
        --clean) CLEAN=1 ;;
        --help|-h)
            cat <<'EOF'
Usage: ./install.sh [OPTIONS]

Options:
  --debug   Build C library with Debug type; Python extension with -O0 -g
  --clean   Wipe build/ directory before building (full rebuild)
  --help    Show this message

Steps performed:
  1. Check prerequisites (cmake, poetry, python3, make)
  2. Clean stale build artifacts
  3. Poetry install (Python dependencies)
  4. Auto-detect PUFFERLIB_INCLUDE path
  5. Build libdronerl.a via CMake
  6. Build binding.cpython-*.so via setuptools
  7. Verify import works
EOF
            exit 0
            ;;
        *) err "Unknown option: $arg"; exit 1 ;;
    esac
done

# --- 1. Check prerequisites ------------------------------------------------
info "Checking prerequisites..."

missing=()
for cmd in cmake poetry python3 make; do
    if ! command -v "$cmd" &>/dev/null; then
        missing+=("$cmd")
    fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
    err "Missing required tools: ${missing[*]}"
    echo "  Install hints:"
    echo "    cmake:   brew install cmake"
    echo "    poetry:  pipx install poetry"
    echo "    python3: brew install python@3.13"
    echo "    make:    xcode-select --install"
    exit 1
fi

# Check cmake version >= 3.16
CMAKE_VER=$(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+')
CMAKE_MAJOR=$(echo "$CMAKE_VER" | cut -d. -f1)
CMAKE_MINOR=$(echo "$CMAKE_VER" | cut -d. -f2)
if [[ "$CMAKE_MAJOR" -lt 3 ]] || { [[ "$CMAKE_MAJOR" -eq 3 ]] && [[ "$CMAKE_MINOR" -lt 16 ]]; }; then
    err "cmake >= 3.16 required (found $CMAKE_VER)"
    exit 1
fi

ok "Prerequisites satisfied (cmake $CMAKE_VER, python3 $(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+'))"

# --- 2. Clean stale artifacts ----------------------------------------------
info "Cleaning stale build artifacts..."

rm -f binding.cpython-*.so binding.*.pyd
rm -rf build/lib.macosx-* build/lib.linux-* build/temp.macosx-* build/temp.linux-*
rm -rf *.egg-info dist/

if [[ "$CLEAN" -eq 1 ]]; then
    info "Full clean: removing build/ directory"
    rm -rf build/
fi

ok "Stale artifacts removed"

# --- 3. Poetry install -----------------------------------------------------
info "Installing Python dependencies via Poetry..."
poetry install --no-interaction --quiet
ok "Python dependencies installed"

# --- 3b. Patch PufferLib env_binding.h (truncation pointer) ----------------
# PufferLib ships with truncation pointer assignment commented out.
# Without this patch, PPO misinterprets timeouts as terminations.
info "Patching env_binding.h (truncation pointer)..."
ENV_BINDING_H=$(poetry run python3 -c "
import importlib.util, pathlib
spec = importlib.util.find_spec('pufferlib')
if spec and spec.origin:
    p = pathlib.Path(spec.origin).parent / 'ocean' / 'env_binding.h'
    if p.exists(): print(p)
" 2>/dev/null || true)

if [[ -n "$ENV_BINDING_H" ]] && [[ -f "$ENV_BINDING_H" ]]; then
    # Uncomment the two truncation assignment lines
    sed -i.bak \
        's|// env->truncations = PyArray_DATA(truncations);|env->truncations = PyArray_DATA(truncations);|' \
        "$ENV_BINDING_H"
    sed -i.bak \
        's|// env->truncations = (void\*)((char\*)PyArray_DATA(truncations) + i\*PyArray_STRIDE(truncations, 0));|env->truncations = (void*)((char*)PyArray_DATA(truncations) + i*PyArray_STRIDE(truncations, 0));|' \
        "$ENV_BINDING_H"
    rm -f "${ENV_BINDING_H}.bak"
    ok "env_binding.h patched (truncation pointer enabled)"
else
    warn "Could not find env_binding.h — truncation patch skipped"
fi

# --- 4. Auto-detect PUFFERLIB_INCLUDE --------------------------------------
if [[ -z "${PUFFERLIB_INCLUDE:-}" ]]; then
    info "Auto-detecting PUFFERLIB_INCLUDE..."
    PUFFERLIB_INCLUDE=$(poetry run python3 -c "
import importlib.util, pathlib
spec = importlib.util.find_spec('pufferlib')
if spec and spec.origin:
    ocean = pathlib.Path(spec.origin).parent / 'ocean'
    if (ocean / 'env_binding.h').exists():
        print(ocean)
" 2>/dev/null || true)

    if [[ -z "$PUFFERLIB_INCLUDE" ]]; then
        # Fallback: sibling directory (setup.py default)
        FALLBACK="$SCRIPT_DIR/../source/PufferLib/pufferlib/ocean"
        if [[ -f "$FALLBACK/env_binding.h" ]]; then
            PUFFERLIB_INCLUDE="$FALLBACK"
            warn "Using fallback PUFFERLIB_INCLUDE: $PUFFERLIB_INCLUDE"
        else
            err "Cannot find env_binding.h. Set PUFFERLIB_INCLUDE manually."
            exit 1
        fi
    fi
fi
export PUFFERLIB_INCLUDE
ok "PUFFERLIB_INCLUDE=$PUFFERLIB_INCLUDE"

# --- 5. Build C library ----------------------------------------------------
BUILD_DIR="$SCRIPT_DIR/build"
LIBDRONERL="$BUILD_DIR/lib/libdronerl.a"

# Detect if build type changed
NEED_RECONFIGURE=0
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    CACHED_TYPE=$(grep -oP '(?<=CMAKE_BUILD_TYPE:STRING=).*' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null || echo "")
    if [[ "$CACHED_TYPE" != "$BUILD_TYPE" ]]; then
        warn "Build type changed ($CACHED_TYPE -> $BUILD_TYPE), reconfiguring"
        NEED_RECONFIGURE=1
        rm -rf "$BUILD_DIR"
    fi
fi

if [[ ! -f "$LIBDRONERL" ]] || [[ "$CLEAN" -eq 1 ]] || [[ "$NEED_RECONFIGURE" -eq 1 ]]; then
    info "Building libdronerl.a ($BUILD_TYPE)..."
    mkdir -p "$BUILD_DIR"

    # Detect parallel jobs
    if command -v sysctl &>/dev/null; then
        JOBS=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
    elif [[ -f /proc/cpuinfo ]]; then
        JOBS=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || echo 4)
    else
        JOBS=4
    fi

    cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DBUILD_TESTING=OFF \
        -DBUILD_BENCHMARKS=OFF

    cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j"$JOBS"

    ok "libdronerl.a built ($BUILD_DIR/lib/)"
else
    ok "libdronerl.a already up to date"
fi

# --- 6. Build Python extension ---------------------------------------------
info "Building Python extension..."
export DEBUG="$DEBUG_FLAG"
poetry run python setup.py build_ext --inplace 2>&1 | tail -5
ok "Python extension built"

# --- 7. Verify -------------------------------------------------------------
SO_FILE=$(ls binding.cpython-*.so 2>/dev/null | head -1 || true)
if [[ -z "$SO_FILE" ]]; then
    err "binding.cpython-*.so not found after build"
    exit 1
fi

info "Verifying import..."
if poetry run python3 -c "import binding; print('  binding module loaded')" 2>&1; then
    ok "Build complete!"
    echo ""
    echo "  Run the demo:"
    echo "    PYTHONPATH=.. poetry run python scripts/demo_gyroid_orbit.py"
    echo ""
else
    warn "Import verification failed (may need PYTHONPATH set)"
    echo "  The .so file exists: $SO_FILE"
    echo "  Try: PYTHONPATH=. poetry run python3 -c 'import binding'"
fi
