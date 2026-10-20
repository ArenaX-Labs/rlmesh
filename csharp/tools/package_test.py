"""Validate distribution contents and restore a fresh console app from the nupkg."""

import json
import pathlib
import shutil
import subprocess
import tarfile
import tempfile
import time
import unittest
import xml.etree.ElementTree as ET
import zipfile

import package
from harness import drive, test_env
from test import CSHARP, ROOT, run

# D-40's distribution contract, independently spelled out so a packaging-path
# or CPU-filter change cannot update its own expected result.
EXPECTED_PLUGINS = {
    "linux-x64": ("x86_64/linux/librlmesh_capi.so", "Linux", "Linux64", "x86_64"),
    "win-x64": ("x86_64/windows/rlmesh_capi.dll", "Windows", "Win64", "x86_64"),
    "osx-arm64": ("macOS/librlmesh_capi.dylib", "OSX", "OSXUniversal", "AnyCPU"),
}


class DistributionTests(unittest.TestCase):
    """No Unity installation is needed for importer and payload validation."""

    def test_complete_layout(self):
        """Exercise every RID with synthetic native bytes, without cross-compiling."""
        with tempfile.TemporaryDirectory(
            dir=pathlib.Path.home() / ".cache/rlmesh-csharp/tmp"
        ) as tmp:
            original_root, original_out = package.ROOT, package.OUT
            root = pathlib.Path(tmp)
            try:
                package.ROOT, package.OUT = root, root / "target/csharp-pkg"
                shutil.copyfile(ROOT / "Cargo.toml", root / "Cargo.toml")
                for triple, _member, _ in set(package.NATIVES.values()):
                    directory = root / f"rlmesh-{package.version()}-{triple}"
                    for _, (other, path, _) in package.NATIVES.items():
                        if other == triple:
                            dest = directory / path
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            if other == "x86_64-unknown-linux-gnu":
                                versioned = dest.with_name(dest.name + ".5")
                                versioned.write_bytes(b"fixture native")
                                dest.symlink_to(versioned.name)
                            else:
                                dest.write_bytes(b"fixture native")
                    archives = root / "target/capi-pkg"
                    archives.mkdir(parents=True, exist_ok=True)
                    with tarfile.open(
                        archives / f"{directory.name}.tar.gz", "w:gz"
                    ) as tar:
                        tar.add(directory, arcname=directory.name)
                self.assertEqual(
                    set(package.NATIVES),
                    {"linux-x64", "win-x64", "osx-arm64", "osx-x64"},
                )
                package.stage_native(require_all=True)
                (root / "csharp").mkdir()
                shutil.copyfile(ROOT / "csharp/README.md", root / "csharp/README.md")
                for name in ("LICENSE-MIT", "LICENSE-APACHE"):
                    shutil.copyfile(ROOT / name, root / name)
                archive = package.upm(require_all=True)
                with zipfile.ZipFile(
                    package.OUT / f"RLMesh.{package.version()}.nupkg"
                ) as nuget:
                    for rid, (_, _, filename) in package.NATIVES.items():
                        self.assertEqual(
                            nuget.read(f"runtimes/{rid}/native/{filename}"),
                            b"fixture native",
                        )
                with tarfile.open(archive) as upm:
                    names = {item.name for item in upm if item.isfile()}
                    payload = {
                        "package/package.json",
                        "package/README.md",
                        "package/LICENSE-MIT",
                        "package/LICENSE-APACHE",
                        "package/Runtime/RLMesh.dll",
                    }
                    payload.update(
                        "package/Runtime/Plugins/" + values[0]
                        for values in EXPECTED_PLUGINS.values()
                    )
                    self.assertEqual(
                        {name for name in names if not name.endswith(".meta")}, payload
                    )
                    self.assertTrue(all(name + ".meta" in names for name in payload))
                for rid, (_, _, filename) in package.NATIVES.items():
                    self.assertEqual(
                        (
                            package.OUT / "native" / rid / "native" / filename
                        ).read_bytes(),
                        b"fixture native",
                    )
                missing = (
                    root
                    / f"target/capi-pkg/rlmesh-{package.version()}-universal-apple-darwin.tar.gz"
                )
                missing.unlink()
                with self.assertRaisesRegex(SystemExit, "missing osx-arm64"):
                    package.stage_native(require_all=True)
            finally:
                package.ROOT, package.OUT = original_root, original_out

    def test_version_bump(self):
        """A bumped package accepts its matching native ABI fixture, without hand edits."""
        with tempfile.TemporaryDirectory(
            dir=pathlib.Path.home() / ".cache/rlmesh-csharp/tmp"
        ) as tmp:
            root = pathlib.Path(tmp)
            original = package.ROOT, package.OUT, package.PROJECT
            try:
                # Model a workspace version bump without touching the checkout.
                bumped = "99.0.1-rc.2"
                (root / "Cargo.toml").write_text(
                    f'[workspace.package]\nversion = "{bumped}"\n'
                )
                shutil.copytree(
                    CSHARP / "src",
                    root / "csharp/src",
                    ignore=shutil.ignore_patterns("bin", "obj"),
                )
                (root / "csharp/tools").mkdir()
                shutil.copyfile(
                    CSHARP / "tools/version.py", root / "csharp/tools/version.py"
                )
                shutil.copyfile(CSHARP / "README.md", root / "csharp/README.md")
                for name in ("LICENSE-MIT", "LICENSE-APACHE"):
                    shutil.copyfile(ROOT / name, root / name)
                native = root / "target/csharp-native/release"
                native.mkdir(parents=True)
                header = ROOT / "crates/rlmesh-capi/include/rlmesh.h"
                # Match a version bump's C header too. Keep the ABI fixture small;
                # the real-library/package serving smoke below covers behavior.
                (native / "rlmesh.h").write_text(
                    header.read_text().replace(
                        f'#define RLMESH_VERSION "{package.version()}"',
                        f'#define RLMESH_VERSION "{bumped}"',
                    )
                )
                source = native / "stub.c"
                source.write_text(
                    '#include "rlmesh.h"\n'
                    "uint32_t rlmesh_abi_version(void){return RLMESH_ABI_VERSION;}\n"
                    "RLMeshStatus rlmesh_build_info(RLMeshBuildInfo* out){out->version=RLMESH_VERSION;return RLMESH_OK;}\n"
                    "RLMeshStatus rlmesh_value_none(RLMeshValue** out){*out=0;return RLMESH_OK;}\n"
                    "void rlmesh_value_free(RLMeshValue* value){}\n"
                )
                run(
                    [
                        "zig",
                        "cc",
                        "-shared",
                        "-fPIC",
                        str(source),
                        "-o",
                        str(native / "librlmesh_capi.so"),
                    ]
                )
                package.ROOT, package.OUT = root, root / "target/csharp-pkg"
                package.PROJECT = root / "csharp/src/RLMesh/RLMesh.csproj"
                archive = package.pack()
                with zipfile.ZipFile(archive) as nuget:
                    self.assertIn("lib/netstandard2.1/RLMesh.dll", nuget.namelist())
                    self.assertIn("lib/net10.0/RLMesh.dll", nuget.namelist())
                consumer = root / "consumer"
                consumer.mkdir()
                (consumer / "Consumer.csproj").write_text(
                    '<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType>'
                    "<TargetFramework>net10.0</TargetFramework></PropertyGroup><ItemGroup>"
                    f'<PackageReference Include="RLMesh" Version="{bumped}"/>'
                    "</ItemGroup></Project>"
                )
                (consumer / "NuGet.Config").write_text(
                    f'<configuration><packageSources><clear/><add key="local" value="{package.OUT}"/></packageSources></configuration>'
                )
                (consumer / "Program.cs").write_text(
                    '_ = RLMesh.Value.None; System.Console.WriteLine("matching bumped native accepted");'
                )
                env = test_env()
                env.pop("LD_LIBRARY_PATH", None)
                env["NUGET_PACKAGES"] = str(root / "packages")
                run(
                    [
                        "dotnet",
                        "build",
                        str(consumer / "Consumer.csproj"),
                        "--disable-build-servers",
                        "-p:UseSharedCompilation=false",
                    ],
                    env=env,
                )
                result = subprocess.run(
                    ["dotnet", str(consumer / "bin/Debug/net10.0/Consumer.dll")],
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=30,
                    check=True,
                )
                self.assertEqual(
                    result.stdout.strip(), "matching bumped native accepted"
                )
            finally:
                package.ROOT, package.OUT, package.PROJECT = original

    def test_importers(self):
        """Assert enabled platforms, CPU filters, auto-reference and stable GUIDs."""
        import json
        import subprocess

        def load(text):
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    "-c",
                    "import json, sys, yaml; print(json.dumps(yaml.safe_load(sys.stdin.read())))",
                ],
                input=text,
                text=True,
                capture_output=True,
                env=test_env(),
                check=True,
            )
            return json.loads(result.stdout)

        self.assertEqual(package.PLUGINS, EXPECTED_PLUGINS)
        guids = set()
        for rid, (path, os_name, platform, cpu) in EXPECTED_PLUGINS.items():
            with self.subTest(rid=rid):
                meta = load(package.importer(path, os_name, platform, cpu))
                self.assertNotIn(meta["guid"], guids)
                guids.add(meta["guid"])
                settings = meta["PluginImporter"]
                self.assertEqual(settings["isExplicitlyReferenced"], 0)
                data = settings["platformData"]
                self.assertEqual(len(data), 3)
                self.assertEqual(data[0]["second"]["enabled"], 0)
                self.assertEqual(data[1]["first"], {"Editor": "Editor"})
                self.assertEqual(data[1]["second"]["enabled"], 1)
                self.assertEqual(
                    data[1]["second"]["settings"],
                    {"CPU": cpu, "OS": os_name, "DefaultValueInitialized": True},
                )
                self.assertEqual(
                    data[2],
                    {
                        "first": {"Standalone": platform},
                        "second": {"enabled": 1, "settings": {"CPU": cpu}},
                    },
                )
        managed = load(package.importer("Runtime/RLMesh.dll"))["PluginImporter"]
        self.assertEqual(managed["platformData"][0]["second"]["enabled"], 1)
        self.assertEqual(managed["isExplicitlyReferenced"], 0)


def main():
    """Check actual SDK packages and isolate consumer restore from all caches."""
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(DistributionTests)
    )
    if not result.wasSuccessful():
        raise SystemExit(1)
    archive = package.upm()
    nupkg = package.OUT / f"RLMesh.{package.version()}.nupkg"
    with zipfile.ZipFile(nupkg) as nuget:
        files = set(nuget.namelist())
        assert {
            "lib/netstandard2.1/RLMesh.dll",
            "lib/net10.0/RLMesh.dll",
            "README.md",
            "LICENSE-MIT",
            "LICENSE-APACHE",
        } <= files
        assert "runtimes/linux-x64/native/librlmesh_capi.so" in files
        spec = ET.fromstring(nuget.read("RLMesh.nuspec"))
        ns = {"n": spec.tag.split("}")[0][1:]}
        assert spec.find("n:metadata/n:version", ns).text == package.version()
        assert spec.find("n:metadata/n:repository", ns).get("type") == "git"
        managed = nuget.read("lib/netstandard2.1/RLMesh.dll")
    with zipfile.ZipFile(nupkg.with_suffix(".snupkg")) as symbols:
        assert {"lib/netstandard2.1/RLMesh.pdb", "lib/net10.0/RLMesh.pdb"} <= set(
            symbols.namelist()
        )
    with tarfile.open(archive) as upm:
        names = {m.name for m in upm if m.isfile()}
        assert upm.extractfile("package/Runtime/RLMesh.dll").read() == managed
        manifest = json.load(upm.extractfile("package/package.json"))
        assert manifest["name"] == "com.rlmesh.rlmesh"
        assert manifest["version"] == package.version()
        assert not any(n.endswith((".cs", ".asmdef")) for n in names)
        expected = {
            "package/package.json",
            "package/README.md",
            "package/LICENSE-MIT",
            "package/LICENSE-APACHE",
            "package/Runtime/RLMesh.dll",
        }
        expected.update(
            "package/Runtime/Plugins/" + path
            for rid, (path, *_) in package.PLUGINS.items()
            if (package.OUT / "native" / rid).exists()
        )
        assert {n for n in names if not n.endswith(".meta")} == expected
        assert all(n + ".meta" in names for n in expected)
    scratch = pathlib.Path.home() / ".cache/rlmesh-csharp/tmp"
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        directory = pathlib.Path(tmp)
        env = test_env()
        env["NUGET_PACKAGES"] = str(directory / "packages")
        run(
            [
                "dotnet",
                "new",
                "console",
                "--framework",
                "net10.0",
                "--no-restore",
                "-n",
                "Consumer",
                "-o",
                str(directory),
            ],
            env=env,
        )
        shutil.copyfile(CSHARP / "tests/Harness/Program.cs", directory / "Program.cs")
        (directory / "NuGet.Config").write_text(
            f'<configuration><packageSources><clear/><add key="local" value="{package.OUT}"/></packageSources></configuration>'
        )
        run(
            [
                "dotnet",
                "add",
                str(directory / "Consumer.csproj"),
                "package",
                "RLMesh",
                "--version",
                package.version(),
                "--no-restore",
            ],
            env=env,
        )
        run(
            [
                "dotnet",
                "build",
                str(directory / "Consumer.csproj"),
                "-c",
                "Release",
                "--disable-build-servers",
                "-p:UseSharedCompilation=false",
            ],
            env=env,
        )
        drive(
            ["dotnet", str(directory / "bin/Release/net10.0/Consumer.dll")],
            directory,
            env,
        )
    print(
        f"NuGet fresh restore + Python serving, symbols and UPM structure: passed ({time.monotonic() - started:.2f}s)"
    )


if __name__ == "__main__":
    main()
