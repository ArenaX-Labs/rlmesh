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

add_executable(chrono_reach_env src/main.cpp src/reach_env.cpp)
target_include_directories(chrono_reach_env PRIVATE "${RLMESH_INCLUDE_DIR}")
target_link_libraries(chrono_reach_env PRIVATE
  Chrono::Chrono_core Chrono::ChronoModels_robot "${RLMESH_CAPI}" Threads::Threads ${CMAKE_DL_LIBS})
target_compile_options(chrono_reach_env PRIVATE -Wall -Wextra)
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
