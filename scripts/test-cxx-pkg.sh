#!/usr/bin/env bash
# Builds the C++ smoke against the INSTALLED package, not the source tree: a
# consumer that knows only the unpacked tarball builds cpp_model and runs an
# episode over real gRPC. Local/release check — deliberately not in the `test` /
# `check` aggregates. The pkg-config leg always runs; find_package runs if cmake
# is on PATH.
set -euo pipefail

CXX="${CXX:-zig c++}"
# zig's bundled libc++ emits -Wnullability-completeness noise on <variant> etc.;
# this smoke isn't a -Werror gate (test:cxx is), so just keep zig output clean.
quiet=()
case "$CXX" in *zig*) quiet=(-Wno-nullability-completeness -Wno-unused-command-line-argument) ;; esac

# 1. Produce the package, then locate it by globbing its single tarball.
mise run release:cxx:package
tarball="$(echo target/capi-pkg/rlmesh-*.tar.gz)"
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
$CXX -std=c++17 "${quiet[@]}" $flags -Wl,-rpath,"$pkg/lib" "$src" -o "$tmp/cpp_model_pc"
echo "==> pkg-config: end-to-end vs a live env"
target/debug/e2e_harness "$tmp/cpp_model_pc"

# 4. find_package leg: only when cmake is available (the box / CI may lack it).
if command -v cmake >/dev/null 2>&1; then
  echo "==> cmake find_package: configure + build cpp_model against the package"
  cmake -S crates/rlmesh-capi/examples/consumer -B "$tmp/build" \
    -DCMAKE_PREFIX_PATH="$pkg" -DCMAKE_BUILD_RPATH="$pkg/lib" \
    -DCMAKE_CXX_COMPILER="${CXX// /;}" >/dev/null
  cmake --build "$tmp/build" >/dev/null
  echo "==> cmake find_package: end-to-end vs a live env"
  target/debug/e2e_harness "$tmp/build/cpp_model"
else
  echo "==> cmake not found — skipping find_package leg (pkg-config leg proved the package)"
fi
echo "==> test:cxx:pkg OK"
