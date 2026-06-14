# Minimal consumer project — proves the vendoring story against the INSTALLED
# package, not the source tree. Point CMAKE_PREFIX_PATH (or -Drlmesh_DIR) at an
# unpacked rlmesh-<ver>-<triple>/ tree; `mise run test:cxx-pkg` drives it.
cmake_minimum_required(VERSION 3.20)
project(rlmesh_consumer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(rlmesh CONFIG REQUIRED)

# Reuse the existing C++ smoke source verbatim — the same surface test:cxx trusts.
add_executable(cpp_model ../cpp_model/model.cpp)
target_link_libraries(cpp_model PRIVATE rlmesh::rlmesh)
