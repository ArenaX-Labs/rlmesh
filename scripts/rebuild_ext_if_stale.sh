#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

so="python/rlmesh/src/rlmesh/_rlmesh.abi3.so"

# A fresh clone or worktree has no build yet; setup and the test tasks make one.
[ -f "$so" ] || exit 0

stale="$(
  find crates python/rlmesh/rust Cargo.toml Cargo.lock proto 2>/dev/null \
    \( -name '*.rs' -o -name 'Cargo.toml' -o -name 'Cargo.lock' -o -name '*.proto' \) \
    -newer "$so" -print -quit || true
)"

if [ -n "$stale" ]; then
  echo "rlmesh: rust changed since last build ($stale) — rebuilding _rlmesh…"
  exec mise run build:python:develop
fi

exit 0
