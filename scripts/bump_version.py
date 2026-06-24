#!/usr/bin/env python3
"""Bump the RLMesh version across every manifest and install snippet.

SemVer is the source of truth: `Cargo.toml [workspace.package].version`. The PEP 440
spelling for the Python package is derived from it (mirrors the Rust
`rlmesh-sandbox::python_package_version`). `policy:check` is the backstop — it fails
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

ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^\d+\.\d+\.\d+(-(?:alpha|beta|rc)\.\d+)?$")
SEMVER_PARTS = re.compile(
    r"^\d+\.\d+\.\d+(?:-(?P<channel>alpha|beta|rc)\.\d+)?$"
)
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


def _sealed_workflow_editions(text: str) -> list[str]:
    sealed: list[str] = []
    for match in WORKFLOW_EDITION_BLOCK.finditer(text):
        if _workflow_edition_status(match.group(0)) == "sealed":
            sealed.append(match.group("edition"))
    return sealed


def _workflow_supported_value(editions: list[str]) -> str:
    return "[" + ", ".join(f'"{edition}"' for edition in editions) + "]"


def _replace_workflow_edition_block(
    text: str, match: re.Match[str], block: str
) -> str:
    return text[: match.start()] + block + text[match.end() :]


def _insert_workflow_edition_block(text: str, block: str) -> str:
    seen_workflow_edition = False
    for match in re.finditer(r"^\[[^\]]+\]\n", text, re.MULTILINE):
        header = match.group(0).strip()
        if header.startswith("[workflow.editions."):
            seen_workflow_edition = True
            continue
        if seen_workflow_edition:
            return text[: match.start()] + block + text[match.start() :]
    return text.rstrip() + "\n\n" + block


def _update_workflow_manifest(text: str, version: str) -> str:
    status = _release_status(version)
    base = _workflow_value(text, "base_edition")
    old_edition = _workflow_value(text, "current_edition")
    new_edition = base if status == "stable" else f"{base}-{version}"
    spec = f"docs/editions/{base}.md"
    supported = [new_edition]
    supported.extend(
        edition
        for edition in _sealed_workflow_editions(text)
        if edition != new_edition
    )

    text = re.sub(
        r'^status = "(alpha|beta|rc|stable)"$',
        f'status = "{status}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^current_edition = "[^"]+"$',
        f'current_edition = "{new_edition}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^supported_editions = \[[^\]]*\]$',
        f"supported_editions = {_workflow_supported_value(supported)}",
        text,
        count=1,
        flags=re.MULTILINE,
    )

    if status == "stable":
        block = (
            f'[workflow.editions."{new_edition}"]\n'
            'status = "sealed"\n'
            f'spec = "{spec}"\n'
            f'sealed_in = "{version}"\n'
            f'spec_sha256 = "{_canonical_spec_sha256(ROOT / spec)}"\n\n'
        )
    else:
        block = (
            f'[workflow.editions."{new_edition}"]\n'
            'status = "provisional"\n'
            f'spec = "{spec}"\n\n'
        )

    new_match = _workflow_edition_block(text, new_edition)
    if new_match is not None:
        if (
            status == "stable"
            and _workflow_edition_status(new_match.group(0)) == "sealed"
        ):
            return text
        return _replace_workflow_edition_block(text, new_match, block)

    old_match = _workflow_edition_block(text, old_edition)
    if old_match is None:
        return _insert_workflow_edition_block(text, block)
    if _workflow_edition_status(old_match.group(0)) == "sealed":
        return _insert_workflow_edition_block(text, block)
    return _replace_workflow_edition_block(text, old_match, block)


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
    (except the changelog and the version-stamped edition specs) plus example
    READMEs. The top-level README is version-neutral by design; crate READMEs
    carry precise Cargo install snippets bumped separately.
    """
    files = [
        md
        for md in sorted((root / "docs").rglob("*.md"))
        if md.relative_to(root).as_posix() != "docs/changelog.md"
        and not md.relative_to(root).as_posix().startswith("docs/editions/")
    ]
    files += sorted((root / "examples").rglob("README.md"))
    return files


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

    # Cargo workspace version + the "=X" path-dependency pins.
    sub_file(
        ROOT / "Cargo.toml",
        [(rf'version = "(=?){o}"', rf'version = "\g<1>{new}"')],
    )

    # Python package manifests use the PEP 440 spelling.
    for py in ("pyproject.toml", "python/rlmesh/pyproject.toml"):
        sub_file(ROOT / py, [(rf'version = "{op}"', f'version = "{new_pep}"')])

    # Policy manifest: cargo artifacts use SemVer; the pypi artifact uses PEP 440.
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
    text = _update_workflow_manifest(text, new)
    rlmesh_toml.write_text(text)

    # Crate README install snippets: cargo dependency + `cargo install --version`.
    for readme in sorted((ROOT / "crates").glob("*/README.md")):
        sub_file(readme, [(rf'= "{o}"', f'= "{new}"'), (rf"--version {o}", f"--version {new}")])

    # Prose (docs + example READMEs): every literal version mention — pip specs,
    # cohort examples, "this documents X" claims — tracks the release. Both the
    # SemVer and PEP 440 spellings are rewritten; the bare-SemVer swap also fixes
    # the `BASE-X.Y.Z` cohort strings. Excludes the changelog and the
    # version-stamped edition specs. policy:check's doc-version guard fails loudly
    # if any stale literal survives.
    for prose in prose_version_files(ROOT):
        sub_file(prose, [(o, new), (op, new_pep)])

    print("sync lockfiles + policy check")
    subprocess.run(["cargo", "update", "--workspace"], cwd=ROOT, check=True)
    subprocess.run(["uv", "lock"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/check_rlmesh_policy.py"], cwd=ROOT, check=True)
    print(f"done ({new}) — review the diff, then commit")


def selfcheck() -> None:
    assert pep440("0.1.0-alpha.1") == "0.1.0a1"
    assert pep440("0.1.0-beta.2") == "0.1.0b2"
    assert pep440("0.1.0-rc.3") == "0.1.0rc3"
    assert pep440("0.1.0") == "0.1.0"
    assert _release_status("0.1.0-alpha.1") == "alpha"
    assert _release_status("0.1.0-beta.2") == "beta"
    assert _release_status("0.1.0-rc.3") == "rc"
    assert _release_status("0.1.0") == "stable"
    assert SEMVER.match("0.2.0") and SEMVER.match("1.0.0-rc.1")
    assert not SEMVER.match("0.1") and not SEMVER.match("0.1.0b3")

    sealed_manifest = """[release]
status = "stable"

[workflow]
base_edition = "2026.09"
current_edition = "2026.06"
supported_editions = ["2026.06"]

[workflow.editions."2026.06"]
status = "sealed"
spec = "docs/editions/2026.06.md"
sealed_in = "0.1.0"
spec_sha256 = "already-sealed"

[protocol]
current_generation = "rlmesh-protocol-1"
"""
    prerelease_manifest = _update_workflow_manifest(
        sealed_manifest,
        "0.2.0-alpha.1",
    )
    assert (
        'supported_editions = ["2026.09-0.2.0-alpha.1", "2026.06"]'
        in prerelease_manifest
    )
    assert '[workflow.editions."2026.06"]' in prerelease_manifest
    assert 'sealed_in = "0.1.0"' in prerelease_manifest
    assert '[workflow.editions."2026.09-0.2.0-alpha.1"]' in prerelease_manifest

    stable_patch_manifest = sealed_manifest.replace(
        'base_edition = "2026.09"',
        'base_edition = "2026.06"',
    )
    stable_patch_manifest = _update_workflow_manifest(
        stable_patch_manifest,
        "0.1.1",
    )
    assert 'sealed_in = "0.1.0"' in stable_patch_manifest
    assert 'sealed_in = "0.1.1"' not in stable_patch_manifest
    print("bump_version self-check passed")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    arg = sys.argv[1]
    if arg == "--check":
        selfcheck()
        return
    if not SEMVER.match(arg):
        sys.exit(f"not a SemVer version: {arg!r} (expected X.Y.Z or X.Y.Z-beta.N)")
    bump(current_version(), arg)


if __name__ == "__main__":
    main()
