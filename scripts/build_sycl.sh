#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build script for the LMCache SYCL/XPU extension.
#
# This script supports two build modes:
#   1. pip / setuptools  (recommended – integrates with the rest of LMCache)
#   2. Standalone CMake  (useful for development / debugging)
#
# Prerequisites:
#   - Intel oneAPI Base Toolkit (provides icpx, SYCL runtime, Level-Zero)
#     https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html
#   - PyTorch with XPU support (torch >= 2.4 with intel-extension-for-pytorch
#     or a nightly build that includes native XPU backend)
#   - pybind11 (installed automatically by pip when building with setuptools)
#
# Usage:
#   # --- Mode 1: pip / setuptools (recommended) ---
#   BUILD_WITH_SYCL=1 pip install -e . --no-build-isolation
#
#   # --- Mode 2: standalone CMake ---
#   bash scripts/build_sycl.sh [--cmake]
#
# Environment variables (optional):
#   CXX               Override the C++ compiler   (default: icpx)
#   SYCL_TARGETS      Override -fsycl-targets      (default: spir64)
#   BUILD_DIR         Override CMake build dir      (default: build_sycl)
#   INSTALL_PREFIX    Override CMake install prefix (default: <site-packages>)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Colours for pretty output ────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Check prerequisites ─────────────────────────────────────────────────
check_prerequisites() {
  # 1. oneAPI environment
  if [ -z "${CMPLR_ROOT:-}" ]; then
    if [ -f /opt/intel/oneapi/setvars.sh ]; then
      info "Sourcing Intel oneAPI environment (/opt/intel/oneapi/setvars.sh)"
      # shellcheck disable=SC1091
      source /opt/intel/oneapi/setvars.sh --force 2>/dev/null || true
    else
      warn "CMPLR_ROOT is not set and /opt/intel/oneapi/setvars.sh not found."
      warn "Make sure the Intel oneAPI toolkit is installed and sourced."
    fi
  fi

  # 2. icpx compiler
  local cxx="${CXX:-icpx}"
  if ! command -v "$cxx" &>/dev/null; then
    error "Cannot find '$cxx'. Please install the Intel oneAPI DPC++ compiler."
    error "  https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html"
    exit 1
  fi
  info "Using compiler: $(command -v "$cxx")"

  # 3. PyTorch (with XPU support)
  if ! python3 -c "import torch" 2>/dev/null; then
    error "PyTorch is not installed.  pip install torch"
    exit 1
  fi
  info "PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"

  # 4. pybind11
  if ! python3 -c "import pybind11" 2>/dev/null; then
    warn "pybind11 not found – installing via pip"
    pip install pybind11
  fi
}

# ── Build with pip / setuptools ──────────────────────────────────────────
build_pip() {
  info "Building LMCache with SYCL extension via pip …"
  cd "$REPO_ROOT"

  export CXX="${CXX:-icpx}"
  export BUILD_WITH_SYCL=1

  pip install -e . --no-build-isolation
  info "Done.  The xpu_ops extension is now available as lmcache.xpu_ops"
}

# ── Build with CMake (standalone) ────────────────────────────────────────
build_cmake() {
  info "Building LMCache SYCL extension via CMake …"
  local build_dir="${BUILD_DIR:-${REPO_ROOT}/build_sycl}"
  local cxx="${CXX:-icpx}"

  mkdir -p "$build_dir"
  cd "$build_dir"

  cmake "${REPO_ROOT}/csrc/sycl" \
    -DCMAKE_CXX_COMPILER="$cxx" \
    -DCMAKE_BUILD_TYPE=Release

  make -j"$(nproc)"

  info "Build artifacts in: $build_dir"
  info "To install:  cd $build_dir && make install"
  info "Or copy the .so into your lmcache package directory."
}

# ── Main ─────────────────────────────────────────────────────────────────
main() {
  check_prerequisites

  if [[ "${1:-}" == "--cmake" ]]; then
    build_cmake
  else
    build_pip
  fi
}

main "$@"
