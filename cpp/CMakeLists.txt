cmake_minimum_required(VERSION 3.20)
project(rvc_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  add_compile_options(/O2 /EHsc /DNOMINMAX)
endif()

set(ORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/ortdml)
set(ORT_INC ${ORT_DIR}/build/native/include)
set(ORT_LIB ${ORT_DIR}/runtimes/win-x64/native/onnxruntime.lib)
set(ORT_DLL ${ORT_DIR}/runtimes/win-x64/native/onnxruntime.dll)
set(ORT_SHR ${ORT_DIR}/runtimes/win-x64/native/onnxruntime_providers_shared.dll)

add_executable(rvc_selftest src/main.cpp src/engine.cpp src/audio_io.cpp)
target_include_directories(rvc_selftest PRIVATE ${ORT_INC} src)
target_link_libraries(rvc_selftest PRIVATE ${ORT_LIB})

add_custom_command(TARGET rvc_selftest POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORT_DLL} $<TARGET_FILE_DIR:rvc_selftest>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORT_SHR} $<TARGET_FILE_DIR:rvc_selftest>)

# ---- plugin DLL ----
add_library(rvc_plugin SHARED src/plugin.cpp src/engine.cpp)
target_include_directories(rvc_plugin PRIVATE ${ORT_INC} src)
target_compile_definitions(rvc_plugin PRIVATE RVC_EXPORTS)
target_link_libraries(rvc_plugin PRIVATE ${ORT_LIB})
add_custom_command(TARGET rvc_plugin POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORT_DLL} $<TARGET_FILE_DIR:rvc_plugin>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORT_SHR} $<TARGET_FILE_DIR:rvc_plugin>)

