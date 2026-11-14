#!/bin/sh
# nextest setup script: points every test process's RLMESH_RUNTIME_DIR (where
# a loopback host leaves its key and link-gate files) at a directory of the
# test run's own, under the target directory, instead of the user's real
# runtime directory. A value already set is kept.
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
