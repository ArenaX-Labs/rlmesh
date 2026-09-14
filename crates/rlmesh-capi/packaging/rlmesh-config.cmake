# find_package(rlmesh CONFIG) for the prebuilt RLMesh C/C++ ABI. Never builds
# rlmesh — points an IMPORTED target at the cdylib already in the package. Every
# path derives from this file's own location, so the unpacked tarball is relocatable.

get_filename_component(_rlmesh_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
include(CMakeFindDependencyMacro)
find_dependency(Threads)

# Library file names come from the consuming toolchain, never a hardcoded .so:
# librlmesh_capi.so / .dylib on ELF+Mach-O, rlmesh_capi.dll plus its .lib import
# library on Windows.
set(_rlmesh_shared
  "${_rlmesh_root}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}rlmesh_capi${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(_rlmesh_static
  "${_rlmesh_root}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}rlmesh_capi${CMAKE_STATIC_LIBRARY_SUFFIX}")

if(NOT TARGET rlmesh::rlmesh)
  # Shared (cdylib): the default, recommended consumer target.
  add_library(rlmesh::rlmesh SHARED IMPORTED)
  set_target_properties(rlmesh::rlmesh PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_rlmesh_root}/include")
  if(WIN32)
    # A DLL is linked through its import library and loaded from beside the exe.
    set_target_properties(rlmesh::rlmesh PROPERTIES
      IMPORTED_LOCATION "${_rlmesh_root}/bin/rlmesh_capi.dll"
      IMPORTED_IMPLIB   "${_rlmesh_root}/lib/rlmesh_capi.lib")
  else()
    # Native libs the Rust cdylib pulls in: pthread (via Threads), dl, m.
    set_target_properties(rlmesh::rlmesh PROPERTIES
      IMPORTED_LOCATION        "${_rlmesh_shared}"
      INTERFACE_LINK_LIBRARIES "Threads::Threads;${CMAKE_DL_LIBS};m")
  endif()

  # Static: opt-in, only for fully-static engine targets with no other Rust. A
  # Rust staticlib carries no system libs, so it re-declares the full native set
  # (rustc --print=native-static-libs). Present only when the archive was packaged.
  if(EXISTS "${_rlmesh_static}")
    add_library(rlmesh::rlmesh_static STATIC IMPORTED)
    set_target_properties(rlmesh::rlmesh_static PROPERTIES
      IMPORTED_LOCATION             "${_rlmesh_static}"
      INTERFACE_INCLUDE_DIRECTORIES "${_rlmesh_root}/include"
      # Direct symbols, so the header must not declare them __declspec(dllimport).
      INTERFACE_COMPILE_DEFINITIONS "RLMESH_STATIC")
    if(NOT WIN32)
      set_target_properties(rlmesh::rlmesh_static PROPERTIES
        INTERFACE_LINK_LIBRARIES "gcc_s;util;rt;pthread;m;dl;c")
    endif()
  endif()
endif()

unset(_rlmesh_root)
unset(_rlmesh_shared)
unset(_rlmesh_static)
