#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# uninstall.sh — Remove all build artifacts from rl_engine
#
# Usage: ./uninstall.sh [--venv] [--help]
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

trap 'err "Uninstall failed at line $LINENO."' ERR

# --- Parse args ------------------------------------------------------------
REMOVE_VENV=0

for arg in "$@"; do
    case "$arg" in
        --venv) REMOVE_VENV=1 ;;
        --help|-h)
            cat <<'EOF'
Usage: ./uninstall.sh [OPTIONS]

Options:
  --venv    Also remove .venv/ and run poetry env remove --all
  --help    Show this message

Always removes:
  build/              CMake output, libdronerl.a, test/diag binaries
  binding.cpython-*   Python extension .so/.pyd
  *.egg-info/         Setuptools metadata
  dist/               Distribution archives
  __pycache__/        Bytecode caches (top 2 levels, skips .venv)
  .pytest_cache/      Pytest artifacts
  Testing/            CTest artifacts

With --venv:
  .venv/              Poetry virtual environment
  resources           Symlink (may point into .venv)
  poetry env remove   Deregisters all Poetry environments
EOF
            exit 0
            ;;
        *) err "Unknown option: $arg"; exit 1 ;;
    esac
done

# --- Always remove ---------------------------------------------------------
info "Removing build artifacts..."

rm -rf build/
rm -f binding.cpython-*.so binding.*.pyd
rm -rf *.egg-info dist/
rm -rf .pytest_cache/ Testing/

# __pycache__ in top 2 levels only (skip .venv internals)
find "$SCRIPT_DIR" -maxdepth 2 -type d -name __pycache__ ! -path '*/.venv/*' -exec rm -rf {} + 2>/dev/null || true

ok "Build artifacts removed"

# --- Optional: remove venv -------------------------------------------------
if [[ "$REMOVE_VENV" -eq 1 ]]; then
    info "Removing virtual environment..."

    rm -rf .venv/

    # Remove resources symlink if it points into .venv (dangling)
    if [[ -L resources ]]; then
        rm -f resources
        ok "Removed dangling 'resources' symlink"
    fi

    # Deregister Poetry environments
    if command -v poetry &>/dev/null; then
        poetry env remove --all 2>/dev/null || true
    fi

    ok "Virtual environment removed"
fi

ok "Clean complete. Source code untouched."
