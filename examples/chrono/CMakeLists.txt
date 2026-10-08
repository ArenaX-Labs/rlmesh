cmake_minimum_required(VERSION 3.18)
project(chrono_reach_env LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Project Chrono: point CMAKE_PREFIX_PATH (or Chrono_DIR) at its install.
find_package(Chrono CONFIG REQUIRED)

# The RLMesh C ABI. Defaults to this repository's headers and a cargo build of
# rlmesh-capi (`cargo build -p rlmesh-capi --release`); override both to use a
# packaged install.
get_filename_component(RLMESH_REPO "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(RLMESH_INCLUDE_DIR "${RLMESH_REPO}/crates/rlmesh-capi/include"
    CACHE PATH "Directory holding rlmesh.h and rlmesh.hpp")
set(RLMESH_LIBRARY_DIR "${RLMESH_REPO}/target/release"
    CACHE PATH "Directory holding librlmesh_capi")
find_library(RLMESH_CAPI rlmesh_capi PATHS "${RLMESH_LIBRARY_DIR}" NO_DEFAULT_PATH REQUIRED)

find_package(Threads REQUIRED)

# The camera's optional Vulkan backend (`--renderer vulkan`). Needs the Vulkan
# headers and loader, and glslangValidator to compile the shaders to SPIR-V.
option(CHRONO_REACH_VULKAN "Build the offscreen Vulkan renderer" OFF)

add_executable(chrono_reach_env src/main.cpp src/reach_env.cpp src/renderer.cpp)
target_include_directories(chrono_reach_env PRIVATE "${RLMESH_INCLUDE_DIR}")
target_link_libraries(chrono_reach_env PRIVATE
  Chrono::Chrono_core Chrono::ChronoModels_robot "${RLMESH_CAPI}" Threads::Threads ${CMAKE_DL_LIBS})
target_compile_options(chrono_reach_env PRIVATE -Wall -Wextra)
if(CHRONO_REACH_VULKAN)
  find_package(Vulkan 1.1 REQUIRED)
  find_program(GLSLANG_VALIDATOR NAMES glslangValidator REQUIRED)
  set(shader_out "${CMAKE_CURRENT_BINARY_DIR}/shaders")
  set(shader_headers)
  # Each shader becomes a header holding its SPIR-V as `const uint32_t <name>[]`.
  foreach(pair IN ITEMS scene.vert:kSceneVert scene.frag:kSceneFrag shadow.vert:kShadowVert
                        sky.vert:kSkyVert sky.frag:kSkyFrag)
    string(REPLACE ":" ";" pair "${pair}")
    list(GET pair 0 shader)
    list(GET pair 1 symbol)
    string(REPLACE "." "_" stem "${shader}")
    set(header "${shader_out}/${stem}.spv.h")
    add_custom_command(
      OUTPUT "${header}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${shader_out}"
      COMMAND "${GLSLANG_VALIDATOR}" -V --target-env vulkan1.1 --quiet --vn "${symbol}"
              -o "${header}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}"
      DEPENDS "shaders/${shader}" shaders/common.glsl
      COMMENT "Compiling ${shader} to SPIR-V"
      VERBATIM)
    list(APPEND shader_headers "${header}")
  endforeach()
  target_sources(chrono_reach_env PRIVATE src/vulkan_renderer.cpp ${shader_headers})
  # `VkFooInfo info{VK_STRUCTURE_TYPE_...}` zero-fills the rest on purpose.
  set_source_files_properties(src/vulkan_renderer.cpp PROPERTIES
    COMPILE_OPTIONS -Wno-missing-field-initializers)
  target_include_directories(chrono_reach_env PRIVATE "${shader_out}")
  target_compile_definitions(chrono_reach_env PRIVATE CHRONO_REACH_VULKAN)
  target_link_libraries(chrono_reach_env PRIVATE Vulkan::Vulkan)
endif()
set_target_properties(chrono_reach_env PROPERTIES
  BUILD_RPATH "${RLMESH_LIBRARY_DIR}"
  INSTALL_RPATH "$ORIGIN/../lib")

# The same task's model, in C++: no Python on either side of the wire.
add_executable(chrono_reach_model src/scripted_model.cpp)
target_include_directories(chrono_reach_model PRIVATE "${RLMESH_INCLUDE_DIR}")
target_link_libraries(chrono_reach_model PRIVATE "${RLMESH_CAPI}" Threads::Threads ${CMAKE_DL_LIBS})
target_compile_options(chrono_reach_model PRIVATE -Wall -Wextra)
set_target_properties(chrono_reach_model PROPERTIES
  BUILD_RPATH "${RLMESH_LIBRARY_DIR}"
  INSTALL_RPATH "$ORIGIN/../lib")

install(TARGETS chrono_reach_env chrono_reach_model RUNTIME DESTINATION bin)
