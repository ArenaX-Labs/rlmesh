#!/usr/bin/env python3
"""Manage the buf breaking-change baseline for the current protocol generation.

The baseline directory is keyed on ``rlmesh.toml`` ``[protocol].current_generation``
(never hardcoded): ``crates/rlmesh-proto/baselines/<current_generation>/rlmesh``.
It must mirror the live proto tree (``crates/rlmesh-proto/proto/rlmesh``)
byte-for-byte once the generation is frozen, so ``buf breaking`` is a no-op diff.

Subcommands:
  regen   Idempotently re-snapshot the baseline: remove
          ``baselines/<gen>/rlmesh`` and copy ``proto/rlmesh`` into it. Running
          it twice in a row leaves the tree identical. Use this at a refreeze /
          generation mint so the refreeze is a one-command operation.
  verify  Assert (a) the baseline tree is byte-for-byte identical to the live
          proto tree and (b) the clean-break policy holds: NO ``reserved``
          keyword statement exists anywhere under the live proto tree.
"""

from __future__ import annotations

import argparse
import filecmp
import re
import shutil
import sys
import tomllib
from pathlib import Path

# Match a `reserved` statement keyword at any statement boundary: line start
# (after indentation) OR right after a `{`/`;`/`}` on the same line. `buf format`
# normally puts `reserved` on its own line (the line-anchored case the C10
# checklist names), but catch the inline form too so the clean-break gate cannot
# be smuggled past on a single line.
RESERVED_RE = re.compile(r"(?:^|[{};])\s*reserved\b")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _current_generation(root: Path) -> str:
    manifest = tomllib.loads((root / "rlmesh.toml").read_text(encoding="utf-8"))
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise SystemExit("rlmesh.toml is missing the [protocol] table")
    gen = protocol.get("current_generation")
    if not isinstance(gen, str) or not gen:
        raise SystemExit("[protocol].current_generation must be a non-empty string")
    return gen


def _paths(root: Path) -> tuple[Path, Path, str]:
    gen = _current_generation(root)
    live = root / "crates/rlmesh-proto/proto/rlmesh"
    base = root / "crates/rlmesh-proto/baselines" / gen / "rlmesh"
    return live, base, gen


def regen(root: Path) -> int:
    live, base, gen = _paths(root)
    if not live.is_dir():
        raise SystemExit(f"live proto tree missing: {live}")
    if base.exists():
        shutil.rmtree(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(live, base)
    print(
        f"baseline re-snapshotted: {base.relative_to(root)} (generation {gen!r})"
    )
    return 0


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
    live, base, gen = _paths(root)
    errors: list[str] = []

    # (a) baseline must mirror the live proto tree byte-for-byte.
    if not live.is_dir():
        errors.append(f"live proto tree missing: {live}")
    if not base.is_dir():
        errors.append(
            f"baseline tree missing: {base} "
            "(run `mise run protocol:baseline-regen`)"
        )
    if not errors:
        # filecmp.dircmp uses a shallow (os.stat) compare by default; force a
        # full byte comparison of every same-named file.
        _collect_diffs(filecmp.dircmp(str(live), str(base)), "", errors)
        live_files = {p.relative_to(live) for p in live.rglob("*") if p.is_file()}
        for rel in sorted(live_files):
            if not filecmp.cmp(live / rel, base / rel, shallow=False):
                msg = f"differs from baseline (content): {rel}"
                if msg not in errors:
                    errors.append(msg)

    # (b) clean-break policy: NO `reserved` keyword statement anywhere.
    if live.is_dir():
        for proto in sorted(live.rglob("*.proto")):
            for lineno, line in enumerate(
                proto.read_text(encoding="utf-8").splitlines(), 1
            ):
                if RESERVED_RE.search(line):
                    errors.append(
                        f"`reserved` keyword at {proto.relative_to(root)}:{lineno} "
                        "(clean-break policy forbids `reserved` anywhere)"
                    )

    if errors:
        print("protocol:baseline-verify FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(
        "baseline-verify OK: baseline mirrors the live proto tree byte-for-byte "
        f"and no `reserved` keyword is present (generation {gen!r})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("regen", help="re-snapshot the baseline (idempotent)")
    sub.add_parser("verify", help="verify baseline == live and no `reserved`")
    args = parser.parse_args(argv)

    root = _repo_root()
    if args.command == "regen":
        return regen(root)
    if args.command == "verify":
        return verify(root)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
