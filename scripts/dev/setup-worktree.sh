#!/usr/bin/env bash
# Prepare a fresh git worktree; T3 Code runs this from t3.json when it creates
# one. Links the main checkout's gitignored local config, installs the pinned
# tools, then syncs the Python environment (which builds the native extension).
# Safe to rerun.
set -euo pipefail

root=$(git rev-parse --show-toplevel)
main=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd -P)
cd "$root"

if [ "$main" != "$(pwd -P)" ]; then
  for file in mise.local.toml .mise.local.toml fnox.local.toml .fnox.local.toml; do
    if [ -e "$main/$file" ] && [ ! -e "$file" ]; then ln -s "$main/$file" "$file"; fi
  done
fi

mise trust --quiet
mise install
# The first sync compiles the extension; queue behind other heavy jobs.
mise exec -- python scripts/dev/machine_lock.py heavy mise run setup
