#!/usr/bin/env bash
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
mise exec -- python scripts/dev/machine_lock.py heavy mise run setup
