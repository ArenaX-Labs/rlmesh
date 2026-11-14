"""Make sure the Python dev environment (.venv) holds every locked dev dependency.

usage: python_synced.py [--sync]

Exits 0 when it does. Otherwise, with --sync (for a task that already holds the
heavy lock) it syncs it as `mise run setup:python` does; without, it prints one
line saying to run that and exits 1. A new worktree's .venv is empty: the Rust
tests that embed Python then fail on a missing module, and `uv run --no-sync
ruff` falls back to whatever ruff is on PATH.

The check leaves out the workspace's own packages: the editable rlmesh install
is rebuilt whenever its Rust sources move, which is `setup:python`'s business,
not a sign of an unsynced environment.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CHECK = [
    "uv",
    "sync",
    "--group",
    "dev",
    "--check",
    "--inexact",
    "--no-install-workspace",
    "--frozen",
]
SYNC = ["mise", "run", "setup:python"]
HINT = "the Python dev environment (.venv) is not synced: run `mise run setup:python`"

Run = Callable[[Sequence[str]], int]


def _run(command: Sequence[str]) -> int:
    return subprocess.run(command, cwd=ROOT, capture_output=True, check=False).returncode


def _run_shown(command: Sequence[str]) -> int:
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def synced(run: Run = _run) -> bool:
    """Whether every locked dev dependency is installed as locked."""
    return run(CHECK) == 0


def main(argv: Sequence[str] | None = None, run: Run = _run, sync: Run = _run_shown) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sync", action="store_true", help="sync it instead of stopping")
    args = parser.parse_args(argv)
    if synced(run):
        return 0
    if not args.sync:
        print(HINT, file=sys.stderr)
        return 1
    print("the Python dev environment (.venv) is not synced: syncing it", file=sys.stderr)
    return sync(SYNC)


if __name__ == "__main__":
    sys.exit(main())
