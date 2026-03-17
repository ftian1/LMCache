# SPDX-License-Identifier: Apache-2.0
#
# CMakeLists.txt for building the LMCache SYCL/XPU extension standalone.
#
# Prerequisites:
#   - Intel oneAPI DPC++ compiler (icpx) or compatible SYCL compiler
#   - PyTorch with XPU support
#   - pybind11 (bundled with PyTorch)
#
# Usage:
#   mkdir build && cd build
#   cmake -DCMAKE_CXX_COMPILER=icpx ..
#   make -j$(nproc)
#
# The resulting shared library (xpu_ops.<ext>) can be placed on PYTHONPATH
# so that `import lmcache.xpu_ops` works.

cmake_minimum_required(VERSION 3.18)
project(lmcache_sycl LANGUAGES CXX)

# --- C++ standard --------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- SYCL flags ----------------------------------------------------------
# -fsycl enables the SYCL compilation flow.
# -fsycl-targets=spir64 targets Intel GPUs through Level-Zero / OpenCL.
set(SYCL_FLAGS "-fsycl")

# Optional: restrict to Intel GPU targets for smaller binaries.
# Uncomment the next line to limit to SPIR-V (Intel GPU only).
# set(SYCL_FLAGS "${SYCL_FLAGS} -fsycl-targets=spir64")

# --- Find PyTorch --------------------------------------------------------
# Torch ships its own CMake config; we use it to locate headers & libraries.
execute_process(
  COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)

# --- Find pybind11 (shipped with PyTorch or installed separately) --------
execute_process(
  COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)

# --- Find Python ---------------------------------------------------------
find_package(Python3 COMPONENTS Development REQUIRED)

# --- Target: xpu_ops shared library -------------------------------------
pybind11_add_module(xpu_ops MODULE
  ${CMAKE_CURRENT_SOURCE_DIR}/pybind_sycl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mem_kernels_sycl.cpp
)

target_include_directories(xpu_ops PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(xpu_ops PRIVATE
  ${TORCH_LIBRARIES}
)

# Apply SYCL compilation / link flags.
target_compile_options(xpu_ops PRIVATE ${SYCL_FLAGS} -O3)
target_link_options(xpu_ops PRIVATE ${SYCL_FLAGS})

# Intel XPU specific optimizations:
#   -ftarget-register-alloc-mode=pvc:auto  — register allocation tuned for
#       Ponte Vecchio / Data Center GPU Max series.
#   -fno-sycl-id-queries-fit-in-int — allow 64-bit index arithmetic.
#   -ffast-math — aggressive FP optimisations inside copy kernels
#       (safe for current implementation: we only move data, no FP
#       arithmetic.  Review if FP math is added to these files).
#   -funroll-loops — unroll inner copy loops for better instruction packing.
target_compile_options(xpu_ops PRIVATE
  -ftarget-register-alloc-mode=pvc:auto
  -fno-sycl-id-queries-fit-in-int
  -ffast-math
  -funroll-loops
)

# --- Installation --------------------------------------------------------
# Install into the lmcache package directory so
# `from lmcache import xpu_ops` works.
install(TARGETS xpu_ops
  LIBRARY DESTINATION "${Python3_SITELIB}/lmcache"
)
