"""Run only the tests a change can affect: the fast local loop for `test:affected`.

usage: affected.py [--base REF] [--dry-run]

Changed files are everything that differs from the merge base with REF
(default origin/main), including uncommitted and untracked files. A changed
file under a workspace crate selects that crate and every workspace crate that
depends on it; Rust tests for the selection run under nextest. Python tests run
when Python sources change or when the selection reaches the extension crate.
Workspace-wide inputs (lockfile, root manifest, toolchain) select everything.

This is a local shortcut, not a gate: doctests, the system harness, and wheel
builds are left to `test:ci` (or `ci:remote`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]

# Changes here can alter any crate's build or tests.
WORKSPACE_INPUTS = {
    "Cargo.toml",
    "Cargo.lock",
    "mise.toml",
    "mise.lock",
    "clippy.toml",
    ".config/nextest.toml",
}
EXTENSION_PACKAGE = "rlmesh-python"
PYTHON_PREFIXES = ("python/", "pyproject.toml", "uv.lock")


@dataclass(frozen=True)
class Plan:
    rust: list[str] | None  # None means the whole workspace
    python: bool

    def describe(self) -> str:
        rust = "all" if self.rust is None else (" ".join(self.rust) or "none")
        return f"rust: {rust}\npython: {'yes' if self.python else 'no'}"


@dataclass(frozen=True)
class Workspace:
    dirs: dict[str, PurePosixPath]  # package -> manifest dir, relative to root
    dependents: dict[str, set[str]]  # package -> workspace packages using it


def load_workspace() -> Workspace:
    raw = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--locked"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    meta = json.loads(raw)
    members = set(meta["workspace_members"])
    names = {p["id"]: p["name"] for p in meta["packages"] if p["id"] in members}
    dirs = {
        names[p["id"]]: PurePosixPath(
            Path(p["manifest_path"]).parent.relative_to(ROOT).as_posix()
        )
        for p in meta["packages"]
        if p["id"] in members
    }
    dependents: dict[str, set[str]] = {name: set() for name in dirs}
    for node in meta["resolve"]["nodes"]:
        if node["id"] not in names:
            continue
        for dep in node["deps"]:
            if dep["pkg"] in names:
                dependents[names[dep["pkg"]]].add(names[node["id"]])
    return Workspace(dirs, dependents)


def owner(path: PurePosixPath, workspace: Workspace) -> str | None:
    """The workspace package whose directory most specifically contains path."""
    best: tuple[int, str] | None = None
    for name, directory in workspace.dirs.items():
        if directory == PurePosixPath(".") or path.is_relative_to(directory):
            depth = len(directory.parts)
            if best is None or depth > best[0]:
                best = (depth, name)
    return None if best is None or best[0] == 0 else best[1]


def plan(changed: list[str], workspace: Workspace) -> Plan:
    if any(path in WORKSPACE_INPUTS for path in changed):
        return Plan(rust=None, python=True)

    selected: set[str] = set()
    for path in changed:
        name = owner(PurePosixPath(path), workspace)
        if name is not None:
            selected.add(name)

    # Close over reverse dependencies: a change reaches every crate using it.
    queue = list(selected)
    while queue:
        for dependent in workspace.dependents.get(queue.pop(), ()):
            if dependent not in selected:
                selected.add(dependent)
                queue.append(dependent)

    python = EXTENSION_PACKAGE in selected or any(
        path.startswith(PYTHON_PREFIXES) for path in changed
    )
    return Plan(rust=sorted(selected), python=python)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout


def changed_files(base: str) -> list[str]:
    merge_base = git("merge-base", "HEAD", base).strip()
    tracked = git("diff", "--name-only", merge_base).splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    return sorted(set(tracked) | set(untracked))


def run(command: list[str], env: dict[str, str] | None = None) -> int:
    print(f"$ {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=ROOT, env=env).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    changed = changed_files(args.base)
    selection = plan(changed, load_workspace())
    print(f"{len(changed)} changed file(s) since {args.base}\n{selection.describe()}")
    if args.dry_run:
        return

    # Same as test:rust: tests that embed Python need the venv's site-packages.
    env = dict(os.environ, PYTHONPATH=sysconfig.get_path("purelib"))
    status = 0
    if selection.rust is None:
        status |= run(["cargo", "nextest", "run", "--workspace", "--all-targets"], env)
    elif selection.rust:
        packages = [arg for name in selection.rust for arg in ("-p", name)]
        status |= run(["cargo", "nextest", "run", "--all-targets", *packages], env)
    if selection.python:
        # The editable install loads the in-tree .so; rebuild it if Rust moved.
        status |= run(["scripts/rebuild_ext_if_stale.sh"])
        status |= run(["mise", "run", "test:python"])
    sys.exit(1 if status else 0)


if __name__ == "__main__":
    main()
