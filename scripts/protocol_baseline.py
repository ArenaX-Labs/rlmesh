#!/usr/bin/env python3
"""Manage baselines for the governed realtime protocols in ``rlmesh.toml``.

Provisional protocols are linted and checked for forbidden reserved fields.
At freeze, ``regen NAME`` snapshots the baseline; verify and breaking then
check it against the frozen release.
"""

from __future__ import annotations

import argparse
import filecmp
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import tomllib

RESERVED_RE = re.compile(r"(?:^|[{};])\s*reserved\b")
WIRE_REVISION_RE = re.compile(r"pub const WIRE_REVISION: u32 = (\d+);")
WIRE_VERSION = "crates/rlmesh-session-core/src/wire/version.rs"
WIRE_VECTORS = "crates/rlmesh-session-core/tests/vectors/wire"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _protocols(root: Path) -> dict[str, dict]:
    manifest = tomllib.loads((root / "rlmesh.toml").read_text(encoding="utf-8"))
    protocols = manifest.get("protocols", {})
    if not isinstance(protocols, dict):
        raise SystemExit("[protocols] must be a table of protocol tables")
    for name, entry in protocols.items():
        for key in ("proto", "baseline", "status", "frozen_in"):
            if not isinstance(entry.get(key), str):
                raise SystemExit(f"[protocols.{name!r}].{key} must be a string")
        if entry["status"] not in ("provisional", "frozen"):
            raise SystemExit(
                f"[protocols.{name!r}].status must be provisional or frozen"
            )
        if (entry["status"] == "frozen") != bool(entry["frozen_in"]):
            raise SystemExit(
                f"[protocols.{name!r}]: a frozen protocol names its release tag in "
                "frozen_in, a provisional one leaves it empty"
            )
    return protocols


def _protocol_paths(root: Path, entry: dict) -> tuple[Path, Path]:
    return root / entry["proto"] / "rlmesh", root / entry["baseline"] / "rlmesh"


def _reserved(root: Path, live: Path) -> list[str]:
    errors = []
    for proto in sorted(live.rglob("*.proto")):
        for lineno, line in enumerate(
            proto.read_text(encoding="utf-8").splitlines(), 1
        ):
            if RESERVED_RE.search(line):
                errors.append(
                    f"`reserved` keyword at {proto.relative_to(root)}:{lineno} "
                    "(clean-break policy forbids `reserved` anywhere)"
                )
    return errors


def _compare(live: Path, base: Path, errors: list[str]) -> None:
    _collect_diffs(filecmp.dircmp(str(live), str(base)), "", errors)
    for path in sorted(p for p in live.rglob("*") if p.is_file()):
        rel = path.relative_to(live)
        if (base / rel).is_file() and not filecmp.cmp(path, base / rel, shallow=False):
            msg = f"differs from baseline (content): {rel}"
            if msg not in errors:
                errors.append(msg)


def regen_protocol(root: Path, name: str) -> int:
    protocols = _protocols(root)
    if name not in protocols:
        raise SystemExit(f"no [protocols.{name!r}] in rlmesh.toml")
    live, base = _protocol_paths(root, protocols[name])
    if base.exists():
        shutil.rmtree(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(live, base)
    print(f"baseline re-snapshotted: {base.relative_to(root)} (protocol {name!r})")
    return 0


def verify_protocols(root: Path) -> list[str]:
    errors: list[str] = []
    for name, entry in _protocols(root).items():
        live, base = _protocol_paths(root, entry)
        if not live.is_dir():
            errors.append(f"{name}: live proto tree missing: {live}")
            continue
        errors.extend(f"{name}: {e}" for e in _reserved(root, live))
        if entry["status"] == "provisional":
            errors.extend(f"{name}: {e}" for e in _unbumped_wire_change(root, entry))
            if base.exists():
                errors.append(
                    f"{name}: a provisional protocol has no baseline yet "
                    f"({base.relative_to(root)}); freeze it in rlmesh.toml first"
                )
            continue
        if not base.is_dir():
            errors.append(
                f"{name}: baseline missing: {base} (protocol:baseline-regen {name})"
            )
            continue
        diffs: list[str] = []
        _compare(live, base, diffs)
        errors.extend(f"{name}: {e}" for e in diffs)
    return errors


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True)


def _last_release_tag(root: Path) -> str:
    try:
        return _git(root, "describe", "--tags", "--abbrev=0", "--match", "v*").strip()
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            "protocol:baseline-verify needs the last release tag to tell a wire change "
            "from none, and none is reachable (a shallow clone?): run "
            "`git fetch --tags --unshallow`"
        ) from error


def _wire_revision(text: str) -> int | None:
    match = WIRE_REVISION_RE.search(text)
    return int(match.group(1)) if match else None


def _wire_break(proto_changes: list[str], vector_diff: str) -> list[str]:
    """What changed the wire: any changed proto, and any vector changed or removed.

    A vector only added (a new message, or a new case of an existing one) is no
    wire change.
    """
    vectors = sorted(path for path, n in _removed_lines(vector_diff).items() if n)
    return [*proto_changes, *vectors]


def _removed_lines(diff: str) -> dict[str, int]:
    removed: dict[str, int] = {}
    current = ""
    for line in diff.splitlines():
        if line.startswith("--- a/"):
            current = line[len("--- a/") :]
            removed.setdefault(current, 0)
        elif line.startswith("-") and not line.startswith("---") and current:
            removed[current] += 1
    return removed


def _unbumped_wire_change(root: Path, entry: dict) -> list[str]:
    """A provisional wire that changed since the last release must bump WIRE_REVISION."""
    tag = _last_release_tag(root)
    proto_changes = _git(root, "diff", "--name-only", tag, "--", entry["proto"]).split()
    vector_diff = _git(root, "diff", "-U0", tag, "--", WIRE_VECTORS)
    changed = _wire_break(proto_changes, vector_diff)
    if not changed:
        return []
    try:
        released = _wire_revision(_git(root, "show", f"{tag}:{WIRE_VERSION}"))
    except subprocess.CalledProcessError:
        released = None
    current = _wire_revision((root / WIRE_VERSION).read_text(encoding="utf-8"))
    if released is None or current != released:
        return []
    return [
        f"the wire changed since {tag} ({', '.join(changed)}) but WIRE_REVISION is still "
        f"{current}; bump it in {WIRE_VERSION}, with a REVISIONS line naming the change"
    ]


def breaking_protocols(root: Path) -> int:
    status = 0
    for name, entry in _protocols(root).items():
        if entry["status"] != "frozen":
            print(f"{name}: provisional, no breaking-change check until it freezes")
            continue
        status |= _breaking_against(root, entry["proto"], entry["frozen_in"])
    return status


def _collect_diffs(cmp: filecmp.dircmp, prefix: str, errors: list[str]) -> None:
    for name in cmp.left_only:
        errors.append(f"only in live proto tree: {prefix}{name}")
    for name in cmp.right_only:
        errors.append(f"only in baseline: {prefix}{name}")
    for name in cmp.diff_files:
        errors.append(f"differs from baseline: {prefix}{name}")
    for name in cmp.funny_files:
        errors.append(f"could not compare: {prefix}{name}")
    for name, sub in cmp.subdirs.items():
        _collect_diffs(sub, f"{prefix}{name}/", errors)


def verify(root: Path) -> int:
    errors = verify_protocols(root)

    if errors:
        print("protocol:baseline-verify FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("baseline-verify OK: governed realtime protocols verified")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    regen_parser = sub.add_parser("regen", help="re-snapshot the baseline (idempotent)")
    regen_parser.add_argument("protocol", help="a [protocols.NAME] entry")
    sub.add_parser("verify", help="verify baseline == live and no `reserved`")
    sub.add_parser("breaking", help="check wire compatibility against the release tag")
    args = parser.parse_args(argv)

    root = _repo_root()
    if args.command == "regen":
        return regen_protocol(root, args.protocol)
    if args.command == "verify":
        return verify(root)
    if args.command == "breaking":
        return breaking_protocols(root)
    parser.error(f"unknown command {args.command!r}")
    return 2


def _breaking_against(root: Path, prefix: str, tag: str) -> int:
    """`buf breaking` of the live tree under `prefix` against `tag`'s."""
    try:
        paths = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", tag, "--", prefix],
            cwd=root,
            text=True,
        ).splitlines()
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"cannot read {tag}; fetch release tags before checking compatibility"
        ) from error
    if not paths:
        raise SystemExit(f"{tag} has no protocol tree")
    with tempfile.TemporaryDirectory(prefix="rlmesh-wire-") as temporary:
        baseline = Path(temporary)
        for name in paths:
            target = baseline / Path(name).relative_to(prefix)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(
                subprocess.check_output(["git", "show", f"{tag}:{name}"], cwd=root)
            )
        print(f"checking wire compatibility against {tag}", flush=True)
        return subprocess.run(
            ["buf", "breaking", str(root / prefix), "--against", str(baseline)],
            cwd=root,
            check=False,
        ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
