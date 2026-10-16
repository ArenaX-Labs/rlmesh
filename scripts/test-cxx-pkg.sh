#!/usr/bin/env bash
# Builds the native examples against the INSTALLED package: a consumer that
# knows only the unpacked tarball runs cpp_model over real gRPC and checks the
# C and C++ env describe runtimes. Local/release check — deliberately not in the `test` /
# `check` aggregates. The pkg-config leg always runs; the find_package and
# FetchContent legs run if cmake is on PATH.
set -euo pipefail

CC="${CC:-zig cc}"
CXX="${CXX:-zig c++}"
# zig's bundled libc++ emits -Wnullability-completeness noise on <variant> etc.;
# this smoke isn't a -Werror gate (test:cxx is), so just keep zig output clean.
quiet=()
case "$CXX" in *zig*) quiet=(-Wno-nullability-completeness -Wno-unused-command-line-argument) ;; esac

# 1. Produce the package, then locate the host triple's tarball (a Windows
#    package from `release:cxx:package --target windows` may sit beside it).
mise run release:cxx:package
tarball="$(echo target/capi-pkg/rlmesh-*-"$(rustc -vV | sed -n 's/host: //p')".tar.gz)"
name="$(basename "$tarball" .tar.gz)"

# 2. Unpack into a throwaway dir — the consumer sees only this tree.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
tar -C "$tmp" -xzf "$tarball"
pkg="$tmp/$name"
src="crates/rlmesh-capi/examples/cpp_model.cpp"

# The e2e harness binds a live SmokeEnv and drives "<bin> <addr> 1" (one episode).
cargo build -p rlmesh-capi --bin e2e_harness

# 3. pkg-config leg: resolve flags from the package, build against the shipped .so.
echo "==> pkg-config: build cpp_model against the installed package"
flags="$(PKG_CONFIG_PATH="$pkg/lib/pkgconfig" pkg-config --cflags --libs rlmesh)"
$CXX -std=c++20 "${quiet[@]}" $flags -Wl,-rpath,"$pkg/lib" "$src" -o "$tmp/cpp_model_pc"
echo "==> pkg-config: end-to-end vs a live env"
target/debug/e2e_harness "$tmp/cpp_model_pc"

# 4. find_package leg: only when cmake is available (the box / CI may lack it).
if command -v cmake >/dev/null 2>&1; then
  echo "==> cmake find_package: configure + build native examples against the package"
  cmake -S crates/rlmesh-capi/examples/consumer -B "$tmp/build" \
    -DCMAKE_PREFIX_PATH="$pkg" -DCMAKE_BUILD_RPATH="$pkg/lib" \
    -DCMAKE_C_COMPILER="${CC// /;}" -DCMAKE_CXX_COMPILER="${CXX// /;}" >/dev/null
  cmake --build "$tmp/build" >/dev/null
  echo "==> cmake find_package: end-to-end vs a live env"
  target/debug/e2e_harness "$tmp/build/cpp_model"
  echo "==> cmake find_package: native env describe runtimes"
  "$tmp/build/cpp_env" --describe | python scripts/check-native-runtime.py c++ 20
  "$tmp/build/c_env" | python scripts/check-native-runtime.py c 11

  # 5. FetchContent leg: the tarball by URL and its published checksum, the way a
  #    project pinning a release consumes it. No CMAKE_BUILD_RPATH: CMake's own
  #    build rpath must find the fetched .so.
  echo "==> cmake FetchContent: configure + build native examples from the tarball URL"
  sha="$(cut -d' ' -f1 "$tarball.sha256")"
  cmake -S crates/rlmesh-capi/examples/consumer -B "$tmp/build-fetch" \
    -DRLMESH_PACKAGE_URL="file://$PWD/$tarball" -DRLMESH_PACKAGE_SHA256="$sha" \
    -DCMAKE_C_COMPILER="${CC// /;}" -DCMAKE_CXX_COMPILER="${CXX// /;}" >/dev/null
  cmake --build "$tmp/build-fetch" >/dev/null
  echo "==> cmake FetchContent: end-to-end vs a live env"
  target/debug/e2e_harness "$tmp/build-fetch/cpp_model"
  "$tmp/build-fetch/c_env" | python scripts/check-native-runtime.py c 11

  # 6. A parent that already ran find_package(rlmesh) (and found Threads) also
  #    pulls the package in as a subdirectory: the wrapper must leave the
  #    parent's targets alone, since CMake refuses to promote them from there.
  echo "==> cmake: find_package(rlmesh) then add_subdirectory of the same package"
  mkdir -p "$tmp/both"
  cat > "$tmp/both/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.20)
project(rlmesh_both C CXX)
find_package(rlmesh CONFIG REQUIRED)
add_subdirectory("${RLMESH_PACKAGE_DIR}" rlmesh_package)
add_executable(c_env "${RLMESH_EXAMPLES}/c_env.c")
target_link_libraries(c_env PRIVATE rlmesh::rlmesh)
EOF
  cmake -S "$tmp/both" -B "$tmp/build-both" -DCMAKE_PREFIX_PATH="$pkg" \
    -DRLMESH_PACKAGE_DIR="$pkg" -DRLMESH_EXAMPLES="$PWD/crates/rlmesh-capi/examples" \
    -DCMAKE_C_COMPILER="${CC// /;}" -DCMAKE_CXX_COMPILER="${CXX// /;}" >/dev/null
else
  echo "==> cmake not found — skipping find_package leg (pkg-config leg proved the package)"
fi
echo "==> test:cxx:pkg OK"
