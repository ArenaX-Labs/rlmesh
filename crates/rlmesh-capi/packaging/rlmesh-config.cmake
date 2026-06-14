# rlmesh-config.cmake — find_package(rlmesh CONFIG) for the prebuilt RLMesh C/C++ ABI.
#
# Defines the IMPORTED target rlmesh::rlmesh over the shipped cdylib. This file never
# builds rlmesh; it points an imported library at lib/librlmesh_capi.so already in the
# package. Relocatable: every path is derived from this file's own location, so the
# unpacked tarball works wherever it lives. Shipped verbatim by `mise run pack:capi`
# alongside the generated rlmesh-config-version.cmake.

get_filename_component(_rlmesh_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
include(CMakeFindDependencyMacro)
find_dependency(Threads)

if(NOT TARGET rlmesh::rlmesh)
  # Shared (cdylib): the default, recommended consumer target.
  add_library(rlmesh::rlmesh SHARED IMPORTED)
  set_target_properties(rlmesh::rlmesh PROPERTIES
    IMPORTED_LOCATION             "${_rlmesh_root}/lib/librlmesh_capi.so"
    INTERFACE_INCLUDE_DIRECTORIES "${_rlmesh_root}/include"
    # Native libs the Rust cdylib pulls in (Linux): pthread (via Threads), dl, m.
    INTERFACE_LINK_LIBRARIES      "Threads::Threads;${CMAKE_DL_LIBS};m")

  # Static (.a): opt-in, only for fully-static engine targets with no other Rust.
  # A Rust staticlib carries no system libs, so it re-declares the full native set
  # (rustc --print=native-static-libs). Present only when the .a was packaged.
  if(EXISTS "${_rlmesh_root}/lib/librlmesh_capi.a")
    add_library(rlmesh::rlmesh_static STATIC IMPORTED)
    set_target_properties(rlmesh::rlmesh_static PROPERTIES
      IMPORTED_LOCATION             "${_rlmesh_root}/lib/librlmesh_capi.a"
      INTERFACE_INCLUDE_DIRECTORIES "${_rlmesh_root}/include"
      INTERFACE_LINK_LIBRARIES      "gcc_s;util;rt;pthread;m;dl;c")
  endif()
endif()

unset(_rlmesh_root)
