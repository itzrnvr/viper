# Viper Inference Engine — root CMake
# Megakernel-first design: NO CUDA Graphs anywhere. Persistent grid-sync kernel
# replaces graph capture. Targets RTX 3070 Ti Laptop (sm_86), CUDA 12.8, MSVC
# 2019, C++20.
#
# Scope split (parent directive):
#   - This file: infrastructure + binding + per-op stub .cpp files
#   - kernels/ops/*.cu and kernels/ops/*.h: main thread owns (REAL CUDA kernels)
#   - tests/*.cu: main thread owns (kernel unit tests)

cmake_minimum_required(VERSION 3.21)
project(viper LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Build settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Force CUDA 12.8 if multiple kits are present.
set(CUDA_TOOLKIT_ROOT_POINT "$ENV{CUDA_PATH}" CACHE PATH "CUDA toolkit root")
if(NOT CUDA_TOOLKIT_ROOT_POINT)
    set(CUDA_TOOLKIT_ROOT_POINT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
endif()

list(APPEND CMAKE_PREFIX_PATH "${CUDA_TOOLKIT_ROOT_POINT}")

find_package(CUDAToolkit 12.8 REQUIRED)

# Suppress noisy warnings; surface real errors.
if(MSVC)
    add_compile_options(/W3 /permissive- /Zc:__cplusplus /utf-8
        /bigobj /EHsc)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        VIPER_ENABLE_ASSERTS=1)
endif()

# ---------------------------------------------------------------------------
# CUDA-specific settings
# ---------------------------------------------------------------------------
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -Xptxas -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 -lineinfo -DNDEBUG")

# ---------------------------------------------------------------------------
# Viper core library (C++ host) — one stub .cpp per op, real signatures
# matching the interface contract in include/viper/ops.h. Main thread
# supplies real CUDA kernels under kernels/ops/*.cu and replaces this
# surface as each kernel lands.
# ---------------------------------------------------------------------------
add_library(viper_core STATIC
    src/viper/common.cpp
    src/viper/tensor.cpp
    src/viper/safety.cpp
    src/viper/kv_cache.cpp
    src/viper/model.cpp
    src/viper/quant.cpp
    src/viper/rmsnorm.cpp
    src/viper/rope.cpp
    src/viper/embedding.cpp
    src/viper/linear.cpp
    src/viper/gqa_repeat.cpp
    src/viper/sdpa.cpp
    src/viper/swiglu.cpp
    src/viper/residual.cpp
    src/viper/sampling.cpp
    src/viper/megakernel.cpp
)
target_include_directories(viper_core PUBLIC include kernels)
target_link_libraries(viper_core PUBLIC CUDA::cudart)

# ---------------------------------------------------------------------------
# Viper kernels library — REAL CUDA kernels (sm_86, C++20, CUDA 12.8).
# Each kernel has a .h interface (declared contract) and a .cu
# implementation (the actual GPU code). The host-side stub in
# src/viper/<op>.cpp forwards into the kernel via a wrapper.
# ---------------------------------------------------------------------------
add_library(viper_kernels STATIC
    kernels/ops/rmsnorm_kernel.cu
    kernels/ops/rope_kernel.cu
    kernels/ops/embedding_kernel.cu
    kernels/ops/swiglu_kernel.cu
    kernels/ops/residual_kernel.cu
)
target_include_directories(viper_kernels PUBLIC include kernels)
target_link_libraries(viper_kernels PUBLIC CUDA::cudart)
# sm_86 only. We hard-fail at configure time on any other arch.
set_target_properties(viper_kernels PROPERTIES
    CUDA_ARCHITECTURES "86"
    POSITION_INDEPENDENT_CODE OFF
)
target_link_libraries(viper_core PUBLIC viper_kernels)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
add_executable(viper_cli tools/cli/main.cpp)
target_link_libraries(viper_cli PRIVATE viper_core viper_kernels)
target_include_directories(viper_cli PRIVATE include)
set_target_properties(viper_cli PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ---------------------------------------------------------------------------
# HTTP serve tool (placeholder for M5)
# ---------------------------------------------------------------------------
add_executable(viper_serve tools/serve/main.cpp)
target_link_libraries(viper_serve PRIVATE viper_core viper_kernels)
target_include_directories(viper_serve PRIVATE include)
set_target_properties(viper_serve PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ---------------------------------------------------------------------------
# Convert tool (placeholder for M1)
# ---------------------------------------------------------------------------
add_executable(viper_convert tools/convert/main.cpp)
target_link_libraries(viper_convert PRIVATE viper_core)
target_include_directories(viper_convert PRIVATE include)
set_target_properties(viper_convert PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Note: tests target is owned by the main thread (tests/*.cu).