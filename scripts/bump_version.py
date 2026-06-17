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

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^\d+\.\d+\.\d+(-(?:alpha|beta|rc)\.\d+)?$")


def pep440(version: str) -> str:
    for tag, short in (("-alpha.", "a"), ("-beta.", "b"), ("-rc.", "rc")):
        if tag in version:
            base, suffix = version.split(tag, 1)
            return f"{base}{short}{suffix}"
    return version


def current_version() -> str:
    text = (ROOT / "Cargo.toml").read_text()
    m = re.search(
        r'\[workspace\.package\][^\[]*?\nversion = "([^"]+)"', text, re.DOTALL
    )
    if not m:
        sys.exit("could not find [workspace.package] version in Cargo.toml")
    return m.group(1)


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
    sub_file(ROOT / "Cargo.toml", [(rf'version = "(=?){o}"', rf'version = "\g<1>{new}"')])

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
    rlmesh_toml.write_text(text)

    # Crate README install snippets: cargo dependency + `cargo install --version`.
    for readme in sorted((ROOT / "crates").glob("*/README.md")):
        sub_file(readme, [(rf'= "{o}"', f'= "{new}"'), (rf"--version {o}", f"--version {new}")])

    # Docs / examples pip specifiers use the PEP 440 spelling.
    for spec in ("docs/user-guide/sandbox.md", "examples/python/sandbox/README.md"):
        sub_file(ROOT / spec, [(rf"rlmesh=={op}", f"rlmesh=={new_pep}")])

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
    assert SEMVER.match("0.2.0") and SEMVER.match("1.0.0-rc.1")
    assert not SEMVER.match("0.1") and not SEMVER.match("0.1.0b3")
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
