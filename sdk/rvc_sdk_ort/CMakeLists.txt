cmake_minimum_required(VERSION 3.20)

project(rvc_sdk_ort LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(RVC_SDK_ORT_BUILD_DEMO "Build demo app" ON)
option(RVC_SDK_ORT_BUILD_REALTIME_DEMO "Build realtime demo app (miniaudio)" ON)

# ---- Dependencies (prebuilt) ----
# 为了避免强依赖特定包管理器，这里采用“指定 ROOT 路径”的方式查找。
#
# 你需要准备：
# - onnxruntime (建议使用 GPU 版，它同时包含 CPU EP，方便单 DLL 支持 CPU/GPU)
# - faiss (C++ 版库，必须，用于读取/搜索 *.index)
#
# 例如：
#   cmake -S sdk/rvc_sdk_ort -B build -G "Visual Studio 17 2022" -A x64 ^
#     -DONNXRUNTIME_ROOT=C:/deps/onnxruntime ^
#     -DFAISS_ROOT=C:/deps/faiss
#
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to onnxruntime install root (contains include/ and lib/)")
set(FAISS_ROOT "" CACHE PATH "Path to faiss install root (contains include/ and lib/)")

if(NOT ONNXRUNTIME_ROOT)
  message(FATAL_ERROR "ONNXRUNTIME_ROOT is required (path to onnxruntime root).")
endif()
if(NOT FAISS_ROOT)
  message(FATAL_ERROR "FAISS_ROOT is required (path to faiss root).")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  PATHS "${ONNXRUNTIME_ROOT}/include"
  NO_DEFAULT_PATH
)
find_library(ONNXRUNTIME_LIB
  NAMES onnxruntime
  PATHS "${ONNXRUNTIME_ROOT}/lib"
  PATH_SUFFIXES "" "Release" "Debug"
  NO_DEFAULT_PATH
)

# 可选：CUDA provider（如果你使用 onnxruntime GPU 包，通常会有这个 import lib）
find_library(ONNXRUNTIME_PROVIDERS_CUDA_LIB
  NAMES onnxruntime_providers_cuda
  PATHS "${ONNXRUNTIME_ROOT}/lib"
  PATH_SUFFIXES "" "Release" "Debug"
  NO_DEFAULT_PATH
)

find_path(FAISS_INCLUDE_DIR
  NAMES faiss/Index.h
  PATHS "${FAISS_ROOT}/include"
  NO_DEFAULT_PATH
)
find_library(FAISS_LIB
  NAMES faiss
  PATHS "${FAISS_ROOT}/lib"
  PATH_SUFFIXES "" "Release" "Debug"
  NO_DEFAULT_PATH
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR "Failed to find onnxruntime. Check ONNXRUNTIME_ROOT (need include/ + lib/).")
endif()
if(NOT FAISS_INCLUDE_DIR OR NOT FAISS_LIB)
  message(FATAL_ERROR "Failed to find faiss. Check FAISS_ROOT (need include/ + lib/).")
endif()

add_library(rvc_sdk_ort SHARED
  src/rvc_sdk_ort.cpp
  src/rvc_engine.cpp
  src/dsp/linear_resampler.cpp
  src/dsp/sola.cpp
  src/f0/yin_f0.cpp
  src/f0/rmvpe_f0.cpp
  third_party/kissfft/kiss_fft.c
  third_party/kissfft/kiss_fftr.c
)

target_include_directories(rvc_sdk_ort
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/kissfft"
    "${ONNXRUNTIME_INCLUDE_DIR}"
    "${FAISS_INCLUDE_DIR}"
)

target_compile_definitions(rvc_sdk_ort
  PRIVATE
    RVC_SDK_ORT_EXPORTS=1
)

if(MSVC)
  target_compile_definitions(rvc_sdk_ort PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  target_compile_options(rvc_sdk_ort PRIVATE /W4 /permissive-)
else()
  target_compile_options(rvc_sdk_ort PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_link_libraries(rvc_sdk_ort
  PRIVATE
    "${ONNXRUNTIME_LIB}"
    "${FAISS_LIB}"
)

if(ONNXRUNTIME_PROVIDERS_CUDA_LIB)
  target_link_libraries(rvc_sdk_ort PRIVATE "${ONNXRUNTIME_PROVIDERS_CUDA_LIB}")
endif()

set_target_properties(rvc_sdk_ort PROPERTIES OUTPUT_NAME "rvc_sdk_ort")

if(RVC_SDK_ORT_BUILD_DEMO)
  add_executable(rvc_sdk_ort_demo
    demo/main.cpp
  )
  target_include_directories(rvc_sdk_ort_demo PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_link_libraries(rvc_sdk_ort_demo PRIVATE rvc_sdk_ort)
endif()

if(RVC_SDK_ORT_BUILD_REALTIME_DEMO)
  add_executable(rvc_sdk_ort_realtime
    demo_realtime/main.cpp
  )
  target_include_directories(rvc_sdk_ort_realtime
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/third_party/miniaudio"
  )
  target_link_libraries(rvc_sdk_ort_realtime PRIVATE rvc_sdk_ort)
endif()
