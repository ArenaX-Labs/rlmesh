#!/bin/sh
set -eu
if [ -n "${RLMESH_RUNTIME_DIR:-}" ]; then
  exit 0
fi
root=$(cd "$(dirname "$0")/../.." && pwd)
target=${CARGO_TARGET_DIR:-$root/target}
case $target in
  /*) ;;
  *) target=$root/$target ;;
esac
dir=$target/rlmesh-test-runtime
mkdir -p "$dir"
chmod 700 "$dir"
echo "RLMESH_RUNTIME_DIR=$dir" >> "$NEXTEST_ENV"
