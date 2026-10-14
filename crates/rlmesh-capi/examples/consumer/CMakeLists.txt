# Consumer project that builds against the INSTALLED package, not the source tree.
# Point CMAKE_PREFIX_PATH (or -Drlmesh_DIR) at an unpacked rlmesh-<ver>-<triple>/
# tree; `mise run test:cxx:pkg` drives it.
cmake_minimum_required(VERSION 3.20)
project(rlmesh_consumer C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(rlmesh CONFIG REQUIRED)

add_executable(cpp_model ../cpp_model.cpp)
target_link_libraries(cpp_model PRIVATE rlmesh::rlmesh)

add_executable(cpp_env ../cpp_env.cpp)
target_link_libraries(cpp_env PRIVATE rlmesh::rlmesh)

add_executable(c_env ../c_env.c)
target_link_libraries(c_env PRIVATE rlmesh::rlmesh)
