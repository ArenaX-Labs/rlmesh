#!/usr/bin/env bash
# Rebuild the editable _rlmesh extension iff it is older than the Rust sources.
#
# The editable install imports python/rlmesh/src/rlmesh/_rlmesh.abi3.so straight
# from the tree, and `maturin build` never refreshes it — so after a pull or
# branch switch that touched Rust, the CLI loads a stale binary (missing symbols
# like `Advisory`). This runs from the post-merge / post-checkout git hooks.
#
# post-checkout/post-merge hooks get no changed-file list from pre-commit/prek,
# so we can't filter by path there. Instead we guard on mtime: git bumps the
# mtime of exactly the files it rewrites during a checkout/merge, so anything
# newer than the built .so means the Rust changed. When nothing changed this is
# a sub-millisecond `find` and exits without building.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

so="python/rlmesh/src/rlmesh/_rlmesh.abi3.so"

if [ ! -f "$so" ]; then
  echo "rlmesh: _rlmesh extension missing — building…"
  exec mise run build:python:develop
fi

# Any Rust source, Cargo manifest, or proto newer than the built extension.
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
