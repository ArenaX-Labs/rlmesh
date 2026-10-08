#!/usr/bin/env bash
# Build the native Chrono env against this checkout, serve it, and drive it from
# the Python model. Extra arguments go to run_model.py, e.g.:
#
#   CHRONO_DIR=~/opt/chrono examples/chrono/demo.sh --view http:9000 --episodes 8
#
# CHRONO_DIR is a Chrono install prefix (it holds lib/cmake/Chrono); add
# EIGEN_DIR when Eigen is not on the default search path. PORT picks the env's
# port (default 50051). RENDERER=vulkan builds and serves the Vulkan camera;
# add VULKAN_DIR when the Vulkan headers, loader and glslangValidator are not on
# the default search path.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build="$repo/target/chrono-env"
port="${PORT:-50051}"
: "${CHRONO_DIR:?set CHRONO_DIR to a Project Chrono install prefix}"
renderer="${RENDERER:-raytrace}"
prefix="$CHRONO_DIR${EIGEN_DIR:+;$EIGEN_DIR}${VULKAN_DIR:+;$VULKAN_DIR}"
vulkan=OFF
[[ "$renderer" == vulkan ]] && vulkan=ON

cd "$repo"
# A release cohort, so the env and the editable Python package negotiate the
# same workflow edition however dirty the checkout is.
export RLMESH_RELEASE_BUILD=1
echo "==> building rlmesh-capi (release)"
cargo build -p rlmesh-capi --release --lib
echo "==> building the Chrono env"
cmake -S examples/chrono -B "$build" -DCMAKE_PREFIX_PATH="$prefix" \
  -DCHRONO_REACH_VULKAN="$vulkan" >/dev/null
cmake --build "$build" -j"$(nproc)"

echo "==> serving on 127.0.0.1:$port"
"$build/chrono_reach_env" --address "127.0.0.1:$port" --renderer "$renderer" &
env_pid=$!
trap 'kill -TERM "$env_pid" 2>/dev/null; wait "$env_pid" 2>/dev/null || true' EXIT
sleep 1

echo "==> driving it from Python"
uv run python examples/chrono/run_model.py "127.0.0.1:$port" "$@"
