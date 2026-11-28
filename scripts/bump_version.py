#!/usr/bin/env python3
"""Bump the RLMesh version across every manifest and install snippet.

SemVer is the source of truth: `Cargo.toml [workspace.package].version`. The PEP 440
spelling for the Python package is derived from it (mirrors the Rust
`python_package_version` in `python/rlmesh/rust/src/sandbox.rs`). `policy:check` is the backstop — it fails
loudly if any version-bearing spot is missed.

Usage:
    python scripts/bump_version.py X.Y.Z[-{alpha,beta,rc}.N]
    python scripts/bump_version.py --check   # run self-tests, change nothing
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^\d+\.\d+\.\d+(-(?:alpha|beta|rc)\.\d+)?$")
SEMVER_PARTS = re.compile(r"^\d+\.\d+\.\d+(?:-(?P<channel>alpha|beta|rc)\.\d+)?$")
HISTORY_DOCS = {"docs/changelog.md", "docs/compatibility.md", "docs/versioning.md"}
WORKFLOW_EDITION_BLOCK = re.compile(
    r'^\[workflow\.editions\."(?P<edition>[^"]+)"\]\n.*?(?=^\[[^\]]+\]|\Z)',
    re.DOTALL | re.MULTILINE,
)


def pep440(version: str) -> str:
    for tag, short in (("-alpha.", "a"), ("-beta.", "b"), ("-rc.", "rc")):
        if tag in version:
            base, suffix = version.split(tag, 1)
            return f"{base}{short}{suffix}"
    return version


def _release_status(version: str) -> str:
    match = SEMVER_PARTS.match(version)
    if not match:
        sys.exit(f"not a SemVer version: {version!r}")
    return match.group("channel") or "stable"


def _package_family(version: str) -> str:
    """`0.<minor>` before 1.0, then the major: the rule policy:check enforces."""
    major, minor = (int(part) for part in version.split(".")[:2])
    return f"0.{minor}" if major == 0 else str(major)


def _canonical_spec_sha256(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        sys.exit(f"{path.relative_to(ROOT)} has a UTF-8 BOM")
    text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    text = text.rstrip("\n") + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _workflow_value(text: str, key: str) -> str:
    match = re.search(rf'^{key} = "([^"]+)"$', text, re.MULTILINE)
    if not match:
        sys.exit(f"rlmesh.toml missing [workflow].{key}")
    return match.group(1)


def _workflow_edition_block(text: str, edition: str) -> re.Match[str] | None:
    for match in WORKFLOW_EDITION_BLOCK.finditer(text):
        if match.group("edition") == edition:
            return match
    return None


def _workflow_edition_status(block: str) -> str | None:
    match = re.search(r'^status = "([^"]+)"$', block, re.MULTILINE)
    return match.group(1) if match else None


def _update_workflow_manifest(text: str, version: str) -> str:
    """Set the release status; a stable release seals a provisional current edition.

    Editions do not follow versions: a new edition is cut by hand, only when a
    default changes.
    """
    status = _release_status(version)
    text = re.sub(
        r'^status = "(alpha|beta|rc|stable)"$',
        f'status = "{status}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if status != "stable":
        return text
    edition = _workflow_value(text, "current_edition")
    match = _workflow_edition_block(text, edition)
    if match is None or _workflow_edition_status(match.group(0)) != "provisional":
        return text
    spec = f"docs/editions/{edition}.md"
    block = (
        f'[workflow.editions."{edition}"]\n'
        'status = "sealed"\n'
        f'spec = "{spec}"\n'
        f'sealed_in = "{version}"\n'
        f'spec_sha256 = "{_canonical_spec_sha256(ROOT / spec)}"\n\n'
    )
    return text[: match.start()] + block + text[match.end() :]


def current_version() -> str:
    text = (ROOT / "Cargo.toml").read_text()
    m = re.search(
        r'\[workspace\.package\][^\[]*?\nversion = "([^"]+)"', text, re.DOTALL
    )
    if not m:
        sys.exit("could not find [workspace.package] version in Cargo.toml")
    return m.group(1)


def prose_version_files(root: Path) -> list[Path]:
    """Prose whose literal version mentions track the current release: docs
    (except the history docs and the version-stamped edition specs), example
    READMEs and the Python package's README (its PyPI page). The top-level
    README is version-neutral by design; crate READMEs carry precise Cargo
    install snippets bumped separately.
    """
    files = [
        md
        for md in sorted((root / "docs").rglob("*.md"))
        if md.relative_to(root).as_posix() not in HISTORY_DOCS
        and not md.relative_to(root).as_posix().startswith("docs/editions/")
    ]
    files += sorted((root / "examples").rglob("README.md"))
    files.append(root / "python/rlmesh/README.md")
    return files


def member_manifests(root: Path) -> list[Path]:
    """The workspace members' own Cargo.toml files."""
    members = tomllib.loads((root / "Cargo.toml").read_text())["workspace"]["members"]
    return sorted(
        manifest
        for pattern in members
        for manifest in root.glob(f"{pattern}/Cargo.toml")
    )


def sub_file(path: Path, pairs: list[tuple[str, str]]) -> None:
    text = path.read_text()
    new = text
    for pat, repl in pairs:
        new = re.sub(pat, repl, new)
    if new != text:
        path.write_text(new)
        print(f"  {path.relative_to(ROOT)}")


def bump(old: str, new: str) -> None:
    old_pep, new_pep = pep440(old), pep440(new)
    o, op = re.escape(old), re.escape(old_pep)
    print(f"bump {old} -> {new}  (PEP 440 {old_pep} -> {new_pep})")

    for manifest in [ROOT / "Cargo.toml", *member_manifests(ROOT)]:
        sub_file(manifest, [(rf'version = "(=?){o}"', rf'version = "\g<1>{new}"')])

    sub_file(
        ROOT / "crates/rlmesh-capi/include/rlmesh.h",
        [(rf'#define RLMESH_VERSION "{o}"', f'#define RLMESH_VERSION "{new}"')],
    )

    for py in ("pyproject.toml", "python/rlmesh/pyproject.toml"):
        sub_file(ROOT / py, [(rf'version = "{op}"', f'version = "{new_pep}"')])

    sub_file(
        ROOT / "csharp/src/RLMesh/RLMesh.csproj",
        [(rf"<Version>{o}</Version>", f"<Version>{new}</Version>")],
    )

    for dockerfile in sorted((ROOT / "examples").rglob("Dockerfile")):
        sub_file(
            dockerfile,
            [(rf"(?m)^ARG RLMESH_VERSION={op}$", f"ARG RLMESH_VERSION={new_pep}")],
        )

    for npm in (
        "javascript/rlmesh/package.json",
        "javascript/rlmesh/package-lock.json",
    ):
        sub_file(ROOT / npm, [(rf'"version": "{o}"', f'"version": "{new}"')])

    sub_file(
        ROOT / "integrations/unity/package.json",
        [(rf'"(version|com\.rlmesh\.rlmesh)": "{o}"', rf'"\g<1>": "{new}"')],
    )

    rlmesh_toml = ROOT / "rlmesh.toml"
    sub_file(rlmesh_toml, [(rf'version = "{o}"', f'version = "{new}"')])
    text = rlmesh_toml.read_text()
    text = re.sub(
        r'(id = "pypi:rlmesh".*?version = ")[^"]*(")',
        rf"\g<1>{new_pep}\g<2>",
        text,
        count=1,
        flags=re.DOTALL,
    )
    text = re.sub(
        r'^package_family = "[^"]*"$',
        f'package_family = "{_package_family(new)}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = _update_workflow_manifest(text, new)
    rlmesh_toml.write_text(text)

    for readme in sorted((ROOT / "crates").glob("*/README.md")):
        sub_file(
            readme,
            [(rf'= "{o}"', f'= "{new}"'), (rf"--version {o}", f"--version {new}")],
        )

    literal = re.compile(rf'(?P<pep>(?<===)|(?<="package_version": "))?(?:{op}|{o})')

    def prose_spelling(match: re.Match[str]) -> str:
        if old != old_pep:
            return new_pep if match.group(0) == old_pep else new
        return new_pep if match.group("pep") is not None else new

    for prose in prose_version_files(ROOT):
        text = prose.read_text()
        rewritten = literal.sub(prose_spelling, text)
        if rewritten != text:
            prose.write_text(rewritten)
            print(f"  {prose.relative_to(ROOT)}")

    print("sync lockfiles + policy check")
    subprocess.run(["cargo", "update", "--workspace"], cwd=ROOT, check=True)
    subprocess.run(["uv", "lock"], cwd=ROOT, check=True)
    subprocess.run(
        [sys.executable, "scripts/check_rlmesh_policy.py"], cwd=ROOT, check=True
    )
    print(f"done ({new}) — review the diff, then commit")


def selfcheck() -> None:
    assert pep440("0.1.0-alpha.1") == "0.1.0a1"
    assert pep440("0.1.0-beta.2") == "0.1.0b2"
    assert pep440("0.1.0-rc.3") == "0.1.0rc3"
    assert pep440("0.1.0") == "0.1.0"
    stable_to_rc = re.compile(r'(?P<pep>(?<===)|(?<="package_version": "))?0\.1\.0')
    assert (
        stable_to_rc.sub(
            lambda m: "0.1.0rc13" if m.group("pep") is not None else "0.1.0-rc.13",
            'rlmesh==0.1.0 "package_version": "0.1.0" 2026.06-0.1.0',
        )
        == 'rlmesh==0.1.0rc13 "package_version": "0.1.0rc13" 2026.06-0.1.0-rc.13'
    )
    assert _release_status("0.1.0-alpha.1") == "alpha"
    assert _release_status("0.1.0-beta.2") == "beta"
    assert _release_status("0.1.0-rc.3") == "rc"
    assert _release_status("0.1.0") == "stable"
    assert _package_family("0.1.1") == "0.1"
    assert _package_family("0.2.0-rc.1") == "0.2"
    assert _package_family("1.4.0") == "1"
    assert SEMVER.match("0.2.0") and SEMVER.match("1.0.0-rc.1")
    assert not SEMVER.match("0.1") and not SEMVER.match("0.1.0b3")

    prose = {path.relative_to(ROOT).as_posix() for path in prose_version_files(ROOT)}
    assert not (prose & HISTORY_DOCS), sorted(prose & HISTORY_DOCS)
    assert not any(name.startswith("docs/editions/") for name in prose)
    assert "docs/installation.md" in prose
    assert "python/rlmesh/README.md" in prose
    members = {path.relative_to(ROOT).as_posix() for path in member_manifests(ROOT)}
    assert "crates/rlmesh-device/Cargo.toml" in members

    manifest = """[release]
status = "rc"

[workflow]
current_edition = "2026.10"

[workflow.editions."2026.06"]
status = "sealed"
spec = "docs/editions/2026.06.md"
sealed_in = "0.1.0"
spec_sha256 = "already-sealed"

[workflow.editions."2026.10"]
status = "provisional"
spec = "docs/editions/2026.10.md"

[protocols."rlmesh-realtime-v1"]
status = "provisional"
"""
    prerelease = _update_workflow_manifest(manifest, "0.2.0-beta.1")
    assert 'status = "beta"' in prerelease
    assert prerelease.count('status = "provisional"') == 2
    stable = _update_workflow_manifest(manifest, "0.2.0")
    assert '[workflow.editions."2026.10"]\nstatus = "sealed"' in stable
    assert 'sealed_in = "0.2.0"' in stable
    assert 'spec_sha256 = "already-sealed"' in stable
    assert _update_workflow_manifest(stable, "0.2.1") == stable
    print("bump_version self-check passed")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    arg = sys.argv[1]
    if arg == "--check":
        selfcheck()
        return
    if not SEMVER.match(arg):
        sys.exit(
            f"not a SemVer version: {arg!r} (expected X.Y.Z or X.Y.Z-{{alpha,beta,rc}}.N)"
        )
    bump(current_version(), arg)


if __name__ == "__main__":
    main()
