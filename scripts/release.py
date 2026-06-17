#!/usr/bin/env python3
"""Drive an RLMesh release: the mechanical steps around the hand-written prose.

This glues existing `mise` tasks together and adds the irreversible-step guardrails.
It does not write changelog entries or pick the version — those are yours. The two
irreversible actions (pushing the tag, publishing to the registries) stay manual
unless you pass --publish, and even then nothing is pushed to git for you.

Flow:
    preflight -> bump -> clean wheels -> release:check -> signed commit + signed tag -> print push command
    --dry-run : stop after release:check, make no commit or tag (tree left untouched)
    --publish : a separate mode; tag vX must already exist on HEAD (you pushed it),
                skips bump/commit/tag and only publishes crates + wheels + GitHub Release

Usage:
    python scripts/release.py X.Y.Z[-{alpha,beta,rc}.N] [--dry-run]
    python scripts/release.py X.Y.Z[-{alpha,beta,rc}.N] --publish   # after pushing the tag
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
SEMVER = re.compile(r"^\d+\.\d+\.\d+(-(?:alpha|beta|rc)\.\d+)?$")
# crates.io publish order: dependencies before dependents.
CRATE_ORDER = [
    "rlmesh-proto",
    "rlmesh-spaces",
    "rlmesh-adapters",
    "rlmesh-cli",
    "rlmesh-runtime",
    "rlmesh-grpc",
    "rlmesh-sandbox",
    "rlmesh",
]


def run(*cmd: str) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def out(*cmd: str) -> str:
    return subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def is_prerelease(version: str) -> bool:
    return "-" in version


def changelog_section(version: str) -> str:
    """Extract the body of the `## [version]` changelog section (the tag message)."""
    text = CHANGELOG.read_text()
    m = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## |^\[[^\]]+\]: http|\Z)",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if not m or not m.group(1).strip():
        sys.exit(f"CHANGELOG.md has no curated '## [{version}]' section yet")
    return m.group(1).strip()


def preflight(version: str) -> None:
    if not SEMVER.match(version):
        sys.exit(f"not a SemVer version: {version!r}")
    if out("git", "status", "--porcelain").strip():
        sys.exit("working tree is not clean; commit or stash first")
    tag = f"v{version}"
    if tag in out("git", "tag", "--list", tag).split():
        sys.exit(f"tag {tag} already exists")
    latest = [
        t for t in out("git", "tag", "--sort=-creatordate", "--list", "v*").split() if t
    ]
    if latest:
        print(f"latest release tag: {latest[0]} (new: {tag})")
    if "<!-- DRAFT" in CHANGELOG.read_text():
        sys.exit(
            "CHANGELOG.md still has <!-- DRAFT --> markers; curate them before releasing"
        )
    changelog_section(version)  # fails if the version section is missing/empty
    print("preflight ok")


def publish(version: str) -> None:
    pep440 = (
        version.replace("-beta.", "b").replace("-rc.", "rc").replace("-alpha.", "a")
    )
    dist = ROOT / "python/rlmesh/dist"
    # Fail before the irreversible `cargo publish` if the wheels for THIS version
    # aren't built: an empty/stale dist would otherwise upload nothing (or the
    # wrong version) to PyPI after crates.io is already published.
    if not [p for p in dist.glob("*.whl") if pep440 in p.name]:
        found = sorted(p.name for p in dist.glob("*.whl"))
        sys.exit(
            f"no wheel for {pep440} in {dist}/ (build wheels before --publish); found: {found}"
        )
    for crate in CRATE_ORDER:
        run("cargo", "publish", "-p", crate)
    run("maturin", "upload", *[str(p) for p in dist.glob("*")])
    args = [
        "gh",
        "release",
        "create",
        f"v{version}",
        "--title",
        f"v{version}",
        "--notes",
        changelog_section(version),
    ]
    if is_prerelease(version):
        args.append("--prerelease")
    run(*args)
    print("published to crates.io + PyPI and cut the GitHub Release")
    print(f"smoke: python -m pip install rlmesh=={pep440}")


def selfcheck() -> None:
    text = "## [1.0.0]\n- a change\n\n[Unreleased]: http://x\n[1.0.0]: http://x\n"
    m = re.search(
        r"^## \[1\.0\.0\][^\n]*\n(.*?)(?=^## |^\[[^\]]+\]: http|\Z)",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert "http" not in m.group(1), "changelog_section leaked link-reference lines"
    print("release self-check passed")


def main() -> None:
    if sys.argv[1:2] == ["--check"]:
        selfcheck()
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) != 1 or flags - {"--dry-run", "--publish"}:
        sys.exit(__doc__)
    version = args[0]

    if "--publish" in flags:
        tag = f"v{version}"
        if tag not in out("git", "tag", "--list", tag).split():
            sys.exit(
                f"tag {tag} does not exist; run release.py {version} (then push) first"
            )
        if tag not in out("git", "tag", "--points-at", "HEAD").split():
            sys.exit(f"tag {tag} is not on HEAD; check out the tagged commit first")
        publish(version)
        return

    preflight(version)
    run("mise", "run", "bump", version)
    run("mise", "run", "release:python:clean")
    run("mise", "run", "release:check")

    if "--dry-run" in flags:
        run("git", "checkout", "--", ".")  # dry-run must leave the tree as it found it
        print("dry run: build verified; no commit, tag, or publish made")
        return

    if out("git", "status", "--porcelain").strip():
        run("git", "commit", "-S", "-am", f"chore(release): {version}")
    else:
        print(f"tree already at {version}; nothing to commit, tagging current HEAD")
    run("git", "tag", "-s", f"v{version}", "-m", changelog_section(version))
    print(f"\ntagged v{version}. To publish, push then release:")
    print(f"  git push origin HEAD --tags")
    print(
        f"  python scripts/release.py {version} --publish   # crates.io + PyPI + GitHub Release"
    )


if __name__ == "__main__":
    main()
