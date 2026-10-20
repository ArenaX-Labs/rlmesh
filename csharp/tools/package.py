"""Assemble SDK NuGet and Unity packages from the same release payloads."""

import argparse
import json
import pathlib
import shutil
import subprocess
import tarfile
import uuid

import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "target/csharp-pkg"
PROJECT = ROOT / "csharp/src/RLMesh/RLMesh.csproj"
NATIVES = {
    "linux-x64": (
        "x86_64-unknown-linux-gnu",
        "lib/librlmesh_capi.so",
        "librlmesh_capi.so",
    ),
    "win-x64": ("x86_64-pc-windows-msvc", "bin/rlmesh_capi.dll", "rlmesh_capi.dll"),
    "osx-arm64": (
        "universal-apple-darwin",
        "lib/librlmesh_capi.dylib",
        "librlmesh_capi.dylib",
    ),
    "osx-x64": (
        "universal-apple-darwin",
        "lib/librlmesh_capi.dylib",
        "librlmesh_capi.dylib",
    ),
}
PLUGINS = {
    "linux-x64": ("x86_64/linux/librlmesh_capi.so", "Linux", "Linux64", "x86_64"),
    "win-x64": ("x86_64/windows/rlmesh_capi.dll", "Windows", "Win64", "x86_64"),
    "osx-arm64": ("macOS/librlmesh_capi.dylib", "OSX", "OSXUniversal", "AnyCPU"),
}


def version():
    """Use the Cargo workspace version, as release:cxx:package does."""
    return tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"][
        "version"
    ]


def run(*args):
    """Run a checked build without persistent MSBuild workers."""
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def stage_native(require_all=False):
    """Read only known library members; never extract arbitrary archive paths."""
    stage = OUT / "native"
    shutil.rmtree(stage, ignore_errors=True)
    for rid, (triple, member, filename) in NATIVES.items():
        archive = ROOT / f"target/capi-pkg/rlmesh-{version()}-{triple}.tar.gz"
        data = None
        if archive.exists():
            with tarfile.open(archive) as tar:
                path = f"rlmesh-{version()}-{triple}/{member}"
                item = tar.getmember(path)
                # Linux's unversioned .so is a symlink within the same root.
                if item.issym():
                    item = tar.getmember(
                        str(pathlib.PurePosixPath(path).parent / item.linkname)
                    )
                if not item.isfile():
                    raise ValueError(f"not a regular native library: {path}")
                with tar.extractfile(item) as source:
                    data = source.read()
        elif rid == "linux-x64" and not require_all:
            for base in ("target/csharp-native/release", "target/release"):
                local = ROOT / base / filename
                if local.exists():
                    data = local.read_bytes()
                    break
        if data is None:
            if require_all:
                raise SystemExit(f"missing {rid}: build/download {archive}")
            continue
        destination = stage / rid / "native" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    if not stage.exists():
        raise SystemExit(
            "no native assets: run test:csharp or release:cxx:package first"
        )
    return stage


def pack(require_all=False):
    """Build both assemblies and portable symbols with SDK packaging metadata."""
    native = stage_native(require_all)
    run(
        "dotnet",
        "pack",
        PROJECT,
        "-c",
        "Release",
        "--disable-build-servers",
        "-p:UseSharedCompilation=false",
        f"-p:PackageVersion={version()}",
        f"-p:Version={version()}",
        f"-p:NativePackageRoot={native}",
        "-o",
        OUT,
    )
    return OUT / f"RLMesh.{version()}.nupkg"


def importer(path, os_name=None, platform=None, cpu=None):
    """Stable Unity PluginImporter metadata (managed assembly is auto-referenced)."""
    guid = uuid.uuid5(uuid.NAMESPACE_URL, "https://rlmesh.dev/upm/" + path).hex
    managed = os_name is None
    text = f"""fileFormatVersion: 2
guid: {guid}
PluginImporter:
  externalObjects: {{}}
  serializedVersion: 2
  iconMap: {{}}
  executionOrder: {{}}
  defineConstraints: []
  isPreloaded: 0
  isOverridable: 0
  isExplicitlyReferenced: 0
  validateReferences: 1
  platformData:
  - first:
      Any:
    second:
      enabled: {int(managed)}
      settings: {{}}
"""
    if not managed:
        text += f"""  - first:
      Editor: Editor
    second:
      enabled: 1
      settings:
        CPU: {cpu}
        DefaultValueInitialized: true
        OS: {os_name}
  - first:
      Standalone: {platform}
    second:
      enabled: 1
      settings:
        CPU: {cpu}
"""
    return text + "  userData:\n  assetBundleName:\n  assetBundleVariant:\n"


def upm(require_all=False):
    """Use precisely the netstandard2.1 DLL from the produced NuGet package."""
    import zipfile

    package = pack(require_all)
    stage = OUT / "com.rlmesh.rlmesh"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "Runtime").mkdir(parents=True)
    with zipfile.ZipFile(package) as nupkg:
        (stage / "Runtime/RLMesh.dll").write_bytes(
            nupkg.read("lib/netstandard2.1/RLMesh.dll")
        )
    (stage / "Runtime/RLMesh.dll.meta").write_text(importer("Runtime/RLMesh.dll"))
    for rid, (path, os_name, platform, cpu) in PLUGINS.items():
        source = OUT / "native" / rid / "native" / NATIVES[rid][2]
        if source.exists():
            destination = stage / "Runtime/Plugins" / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            destination.with_name(destination.name + ".meta").write_text(
                importer("Runtime/Plugins/" + path, os_name, platform, cpu)
            )
    (stage / "package.json").write_text(
        json.dumps(
            {
                "name": "com.rlmesh.rlmesh",
                "version": version(),
                "displayName": "RLMesh",
                "description": "Experimental RLMesh environment serving for 64-bit desktop hosts.",
                "unity": "2021.3",
                "license": "MIT OR Apache-2.0",
                "author": {"name": "ArenaX Labs", "url": "https://rlmesh.dev"},
            },
            indent=2,
        )
        + "\n"
    )
    for source, destination in (
        (ROOT / "csharp/README.md", "README.md"),
        (ROOT / "LICENSE-MIT", "LICENSE-MIT"),
        (ROOT / "LICENSE-APACHE", "LICENSE-APACHE"),
    ):
        shutil.copyfile(source, stage / destination)
    # Folder and ordinary asset metas avoid regenerated GUIDs on every import.
    for path in sorted(stage.rglob("*")):
        if path.suffix == ".meta" or path.with_name(path.name + ".meta").exists():
            continue
        relative = path.relative_to(stage).as_posix()
        guid = uuid.uuid5(uuid.NAMESPACE_URL, "https://rlmesh.dev/upm/" + relative).hex
        folder = "folderAsset: yes\n" if path.is_dir() else ""
        path.with_name(path.name + ".meta").write_text(
            f"fileFormatVersion: 2\nguid: {guid}\n{folder}DefaultImporter:\n  externalObjects: {{}}\n  userData:\n  assetBundleName:\n  assetBundleVariant:\n"
        )
    archive = OUT / f"com.rlmesh.rlmesh-{version()}.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(stage, arcname="package")
    return archive


def main():
    """Allow partial host packages locally; final release assembly requires all RIDs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("pack", "upm"))
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    print((pack if args.kind == "pack" else upm)(args.require_all))


if __name__ == "__main__":
    main()
