"""Prepare verification using an existing D-40 core UPM artifact, never build it."""

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
PROJECT = PACKAGE / "Tests~/Project"


def install_core(source: Path, destination: Path) -> None:
    """Accept a D-40 UPM directory or the standard package/ UPM tarball."""
    scratch = Path.home() / ".cache/rlmesh-unity"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as temporary:
        if source.is_file():
            with tarfile.open(source) as archive:
                archive.extractall(temporary, filter="data")
            source = Path(temporary) / "package"
        manifest = json.loads((source / "package.json").read_text())
        if manifest["name"] != "com.rlmesh.rlmesh":
            raise SystemExit("Expected the separate com.rlmesh.rlmesh core package")
        for relative in (
            "Runtime/RLMesh.dll",
            "Runtime/Plugins/x86_64/linux/librlmesh_capi.so",
        ):
            if not (source / relative).is_file():
                raise SystemExit(f"D-40 core artifact is missing {relative}")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)


def main() -> None:
    """Stage ignored project inputs and compile the native-thread wedge driver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-package", type=Path, required=True)
    args = parser.parse_args()
    core = PROJECT / "Packages/com.rlmesh.rlmesh"
    install_core(args.core_package.resolve(), core)
    integration = PROJECT / "Packages/com.rlmesh.unity"
    if integration.exists():
        shutil.rmtree(integration)
    integration.mkdir()
    for directory in ("Runtime", "Editor"):
        shutil.copytree(PACKAGE / directory, integration / directory)
        shutil.copyfile(
            PACKAGE / f"{directory}.meta", integration / f"{directory}.meta"
        )
    for name in ("package.json", "package.json.meta", "README.md", "README.md.meta"):
        shutil.copyfile(PACKAGE / name, integration / name)
    sample = PROJECT / "Assets/ReachTarget"
    if sample.exists():
        shutil.rmtree(sample)
    shutil.copytree(PACKAGE / "Samples~/ReachTarget", sample)
    # Give the sample its own assembly; the smoke bootstrap references it.
    (sample / "ReachTarget.asmdef").write_text(
        json.dumps(
            {
                "name": "RLMesh.Unity.ReachTarget",
                "references": ["RLMesh.Unity"],
                "overrideReferences": True,
                "precompiledReferences": ["RLMesh.dll"],
            },
            indent=2,
        )
        + "\n"
    )
    settings = PROJECT / "ProjectSettings"
    settings.mkdir(exist_ok=True)
    shutil.copyfile(PACKAGE / "ProjectVersion.txt", settings / "ProjectVersion.txt")
    tools = PROJECT / "Assets/Tools"
    tools.mkdir(exist_ok=True)
    scratch = Path.home() / ".cache/rlmesh-unity"
    subprocess.run(
        [
            "mise",
            "exec",
            "zig@0.13.0",
            "--",
            "zig",
            "c++",
            "-target",
            "x86_64-linux-gnu.2.17",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "crates/rlmesh-capi/include"),
            str(PACKAGE / "Tests~/step_client.cpp"),
            "librlmesh_capi.so",
            "-o",
            str(tools / "step-client"),
        ],
        env=dict(os.environ, TMPDIR=str(scratch)),
        cwd=core / "Runtime/Plugins/x86_64/linux",
        check=True,
    )
    print(f"Prepared {PROJECT}; core package consumed from {args.core_package}")


if __name__ == "__main__":
    main()
