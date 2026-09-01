#!/usr/bin/env python3
"""Drive an RLMesh release: the mechanical steps around the hand-written prose.

This glues existing `mise` tasks together and adds the irreversible-step guardrails.
It does not write changelog entries or pick the version — those are yours. The one
irreversible local action (pushing the tag) stays manual; the tag push triggers
.github/workflows/release.yml, which builds the wheel matrix on Linux and macOS
runners and publishes to crates.io + PyPI (both via OIDC trusted publishing) and
cuts the GitHub Release.

Flow:
    preflight -> bump -> clean wheels -> release:check -> signed commit + signed tag -> print push command
    --dry-run : stop after release:check, make no commit or tag (tree left untouched)
    --publish-crates / --github-release : the CI publish steps; tag vX must be on HEAD.
                Both are idempotent (already-published crates and an existing Release
                are skipped) so a failed publish job can be re-run. Break-glass local
                use needs CARGO_REGISTRY_TOKEN / gh auth; PyPI has no local path —
                CI uploads wheels via pypa/gh-action-pypi-publish.

Usage:
    python scripts/release.py X.Y.Z[-{alpha,beta,rc}.N] [--dry-run]
    python scripts/release.py X.Y.Z[-{alpha,beta,rc}.N] --publish-crates
    python scripts/release.py X.Y.Z[-{alpha,beta,rc}.N] --github-release
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

from bump_version import pep440

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
SEMVER = re.compile(r"^\d+\.\d+\.\d+(-(?:alpha|beta|rc)\.\d+)?$")
# crates.io publish order: dependencies before dependents.
CRATE_ORDER = [
    "rlmesh-proto",
    "rlmesh-spaces",
    "rlmesh-viewer",
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


def require_changelog_date(version: str) -> None:
    """Reject a placeholder release date: the `## [version]` heading needs YYYY-MM-DD."""
    text = CHANGELOG.read_text()
    m = re.search(rf"^## \[{re.escape(version)}\]([^\n]*)$", text, re.MULTILINE)
    if not m:
        sys.exit(f"CHANGELOG.md has no '## [{version}]' heading yet")
    if not re.fullmatch(r" - \d{4}-\d{2}-\d{2}", m.group(1)):
        sys.exit(
            f"CHANGELOG.md heading '## [{version}]{m.group(1)}' needs a real "
            f"'- YYYY-MM-DD' release date before releasing"
        )


def preflight(version: str) -> None:
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
    require_changelog_date(version)
    print("preflight ok")


def check_wheels(version: str) -> None:
    wheel_version = pep440(version)
    dist = ROOT / "python/rlmesh/dist"
    # Fail before the irreversible `cargo publish` if the wheels for THIS version
    # aren't built: an empty/stale dist would otherwise let crates.io publish while
    # the PyPI upload has nothing (or the wrong version) to ship. Exact-match the
    # filename version segment: "0.1.0" is a substring of "0.1.0rc6".
    if not [p for p in dist.glob("*.whl") if p.name.split("-")[1] == wheel_version]:
        found = sorted(p.name for p in dist.glob("*.whl"))
        sys.exit(
            f"no wheel for {wheel_version} in {dist}/ (build wheels first); found: {found}"
        )
    run(
        sys.executable,
        "scripts/check_python_wheels.py",
        str(dist),
        "--platform-set",
        "all",
    )


def require_workspace_version(version: str) -> None:
    """The tag names the release; Cargo.toml is what cargo publish actually ships."""
    manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    cargo_version = manifest["workspace"]["package"]["version"]
    if cargo_version != version:
        sys.exit(
            f"version {version} does not match Cargo.toml [workspace.package] "
            f"version {cargo_version}; the tag is not on the bumped commit"
        )


def crate_published(crate: str, version: str) -> bool:
    """Ask the crates.io sparse index whether crate@version exists (a local
    `cargo info` would resolve the workspace member and always say yes)."""
    # Index path scheme for names of 4+ chars; every CRATE_ORDER name qualifies.
    url = f"https://index.crates.io/{crate[:2]}/{crate[2:4]}/{crate}"
    try:
        with urllib.request.urlopen(url) as response:
            lines = response.read().decode("utf-8").splitlines()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise
    return any(json.loads(line)["vers"] == version for line in lines if line)


def publish_crates(version: str) -> None:
    require_workspace_version(version)
    check_wheels(version)
    for crate in CRATE_ORDER:
        if crate_published(crate, version):
            print(f"skip: {crate} {version} already on crates.io")
            continue
        run("cargo", "publish", "-p", crate)
    print(f"crates.io publish complete for {version}")


def github_release(version: str) -> None:
    tag = f"v{version}"
    if (
        subprocess.run(
            ["gh", "release", "view", tag], cwd=ROOT, capture_output=True
        ).returncode
        == 0
    ):
        print(f"skip: GitHub Release {tag} already exists")
        return
    args = [
        "gh",
        "release",
        "create",
        tag,
        "--title",
        tag,
        "--notes",
        changelog_section(version),
    ]
    if is_prerelease(version):
        args.append("--prerelease")
    run(*args)
    print(f"cut GitHub Release {tag}")


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
    publish_flags = flags & {"--publish-crates", "--github-release"}
    if (
        len(args) != 1
        or flags - {"--dry-run"} - publish_flags
        or (publish_flags and "--dry-run" in flags)
    ):
        sys.exit(__doc__)
    version = args[0]
    if not SEMVER.match(version):
        sys.exit(f"not a SemVer version: {version!r}")
    os.environ.setdefault("RLMESH_RELEASE_BUILD", "1")

    if publish_flags:
        tag = f"v{version}"
        if tag not in out("git", "tag", "--points-at", "HEAD").split():
            sys.exit(f"tag {tag} is not on HEAD; check out the tagged commit first")
        if "--publish-crates" in flags:
            publish_crates(version)
        if "--github-release" in flags:
            github_release(version)
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
    print(f"\ntagged v{version}. Push to release:")
    print(
        "  git push origin HEAD --tags   # tag push triggers .github/workflows/release.yml"
        " (wheels + crates.io + PyPI + GitHub Release)"
    )


if __name__ == "__main__":
    main()
