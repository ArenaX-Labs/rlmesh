"""Assemble com.rlmesh.unity as a UPM directory and tarball for Install from disk/tarball."""

from __future__ import annotations

import json
import pathlib
import shutil
import tarfile

import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[3]
SOURCE = ROOT / "integrations/unity"
OUT = ROOT / "target/csharp-pkg"
INCLUDE = (
    "package.json",
    "package.json.meta",
    "README.md",
    "README.md.meta",
    "ProjectVersion.txt",
    "ProjectVersion.txt.meta",
    "Runtime",
    "Runtime.meta",
    "Editor",
    "Editor.meta",
    "Samples~",
    "Samples~.meta",
)


def version() -> str:
    """Use the Cargo workspace version, as the core package does."""
    return tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))[
        "workspace"
    ]["package"]["version"]


def package() -> pathlib.Path:
    """Copy shippable UPM members; exclude Tests~ and repo-only files."""
    ver = version()
    stage = OUT / "com.rlmesh.unity"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)

    for name in INCLUDE:
        source = SOURCE / name
        if not source.exists():
            raise SystemExit(f"missing UPM member: {source}")
        destination = stage / name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copyfile(source, destination)

    shutil.copyfile(ROOT / "LICENSE-MIT", stage / "LICENSE-MIT")
    shutil.copyfile(ROOT / "LICENSE-APACHE", stage / "LICENSE-APACHE")

    manifest = json.loads((stage / "package.json").read_text(encoding="utf-8"))
    manifest["version"] = ver
    manifest["dependencies"] = {"com.rlmesh.rlmesh": ver}
    manifest["license"] = "MIT OR Apache-2.0"
    manifest["author"] = {"name": "ArenaX Labs", "url": "https://rlmesh.dev"}
    (stage / "package.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    archive = OUT / f"com.rlmesh.unity-{ver}.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(stage, arcname="package")
    return archive


if __name__ == "__main__":
    print(package())
