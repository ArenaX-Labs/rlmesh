# Consumer project that builds against the INSTALLED package, not the source tree.
# Point CMAKE_PREFIX_PATH (or -Drlmesh_DIR) at an unpacked rlmesh-<ver>-<triple>/
# tree, or pass RLMESH_PACKAGE_URL/RLMESH_PACKAGE_SHA256 to fetch the tarball;
# `mise run test:cxx:pkg` drives both.
cmake_minimum_required(VERSION 3.20)
project(rlmesh_consumer C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# With RLMESH_PACKAGE_URL set, fetch the tarball at configure time the way a
# project pinning a release would; otherwise find the unpacked tree.
set(RLMESH_PACKAGE_URL "" CACHE STRING "rlmesh package tarball to fetch instead of find_package")
set(RLMESH_PACKAGE_SHA256 "" CACHE STRING "SHA-256 of RLMESH_PACKAGE_URL")
if(RLMESH_PACKAGE_URL)
  include(FetchContent)
  FetchContent_Declare(rlmesh
    URL "${RLMESH_PACKAGE_URL}"
    URL_HASH "SHA256=${RLMESH_PACKAGE_SHA256}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(rlmesh)
else()
  find_package(rlmesh CONFIG REQUIRED)
endif()

add_executable(cpp_model ../cpp_model.cpp)
target_link_libraries(cpp_model PRIVATE rlmesh::rlmesh)

add_executable(cpp_env ../cpp_env.cpp)
target_link_libraries(cpp_env PRIVATE rlmesh::rlmesh)

add_executable(c_env ../c_env.c)
target_link_libraries(c_env PRIVATE rlmesh::rlmesh)
