#!/usr/bin/env bash
# Usage: CHRONO_DIR=<chrono install prefix> examples/chrono/demo.sh [run_model.py args]
# Optional: EIGEN_DIR, VULKAN_DIR, PORT (default 50051), RENDERER=vulkan.
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
