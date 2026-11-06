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

Every ``[protocols.<name>]`` entry of ``rlmesh.toml`` (the real-time session
protocol) is governed the same way once its ``status`` is ``frozen``:
``regen`` snapshots its ``proto`` tree into its ``baseline``, ``verify``
checks the two match, and ``breaking`` checks the live tree against its
``frozen_in`` tag. A ``provisional`` one only has to stay free of ``reserved``
and have no baseline yet: ``regen NAME`` at the freeze makes it.
"""

from __future__ import annotations

import argparse
import filecmp
import re
import shutil
import subprocess
import sys
import tempfile
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
        for lineno, line in enumerate(proto.read_text(encoding="utf-8").splitlines(), 1):
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


def breaking_protocols(root: Path) -> int:
    status = 0
    for name, entry in _protocols(root).items():
        if entry["status"] != "frozen":
            print(f"{name}: provisional, no breaking-change check until it freezes")
            continue
        status |= _breaking_against(root, entry["proto"], entry["frozen_in"])
    return status


def regen(root: Path) -> int:
    live, base, gen = _paths(root)
    if not live.is_dir():
        raise SystemExit(f"live proto tree missing: {live}")
    if base.exists():
        shutil.rmtree(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(live, base)
    print(f"baseline re-snapshotted: {base.relative_to(root)} (generation {gen!r})")
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
            f"baseline tree missing: {base} (run `mise run protocol:baseline-regen`)"
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
        errors.extend(_reserved(root, live))
    # Every other governed protocol (the real-time session protocol).
    errors.extend(verify_protocols(root))

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
    regen_parser = sub.add_parser("regen", help="re-snapshot the baseline (idempotent)")
    regen_parser.add_argument(
        "protocol", nargs="?", help="a [protocols.NAME] entry (default: the eval wire)"
    )
    sub.add_parser("verify", help="verify baseline == live and no `reserved`")
    sub.add_parser("breaking", help="check wire compatibility against the release tag")
    args = parser.parse_args(argv)

    root = _repo_root()
    if args.command == "regen":
        if args.protocol:
            return regen_protocol(root, args.protocol)
        return regen(root)
    if args.command == "verify":
        return verify(root)
    if args.command == "breaking":
        return breaking(root) | breaking_protocols(root)
    parser.error(f"unknown command {args.command!r}")
    return 2


def breaking(root: Path) -> int:
    tag = "v0.1.0"
    exists = (
        subprocess.run(
            ["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
            cwd=root,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )
    if not exists:
        version = tomllib.loads((root / "Cargo.toml").read_text())["workspace"][
            "package"
        ]["version"]
        if not re.fullmatch(r"0\.1\.0(-rc\.(1[2-9]|[2-9]\d))?", version):
            raise SystemExit(
                "v0.1.0 tag is required; fetch release tags before checking compatibility"
            )
        # Bootstrap the seal before its tag exists; later releases must use v0.1.0.
        tag = "v0.1.0-rc.12"
        print(f"v0.1.0 tag absent: bootstrapping against {tag}", file=sys.stderr)
    return _breaking_against(root, "crates/rlmesh-proto/proto", tag)


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
