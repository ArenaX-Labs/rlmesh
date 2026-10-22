"""Run the C# foundation suite (caller owns the shared heavy-job lock)."""

import os
import pathlib
import re
import subprocess

import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]
CSHARP = ROOT / "csharp"
DOTNET_BUILD = [
    "dotnet",
    "build",
    "--disable-build-servers",
    "-p:UseSharedCompilation=false",
    "--nologo",
]
CC = ["zig", "cc"]


def run(args, *, env=None, expected=0, cwd=ROOT):
    """Run one checked command from the repository root."""
    print("$ " + " ".join(map(str, args)), flush=True)
    result = subprocess.run(args, cwd=cwd, env=env)
    if result.returncode != expected:
        raise SystemExit(result.returncode or 1)


def gate_tests(env):
    """Check loader and version failures in isolated native-library processes."""
    project = CSHARP / "tests/GateProbe/GateProbe.csproj"
    run([*DOTNET_BUILD, str(project)], env=env)
    probe = project.parent / "bin/Debug/net10.0/GateProbe.dll"
    # A stub traps any value call when the gate should have refused admission.
    cases = {
        "missing": (None, ("rlmesh_capi", "Unable to load")),
        "bad_image": ("not an ELF image" * 20, ("rlmesh_capi", "invalid ELF")),
        "missing_symbol": (
            "int unrelated(void) {return 0;}\n",
            ("rlmesh_capi", "rlmesh_abi_version"),
        ),
        "generation": (
            '#include "rlmesh.h"\nuint32_t rlmesh_abi_version(void){return 777;}\n',
            ("777", "5"),
        ),
        "version": (
            '#include "rlmesh.h"\nuint32_t rlmesh_abi_version(void){return 5;}\nRLMeshStatus rlmesh_build_info(RLMeshBuildInfo* out){out->version="wrong-version"; return RLMESH_OK;}\n',
            (
                "wrong-version",
                tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"][
                    "package"
                ]["version"],
            ),
        ),
        "later_symbol": (
            '#include "rlmesh.h"\nuint32_t rlmesh_abi_version(void){return 5;}\nRLMeshStatus rlmesh_build_info(RLMeshBuildInfo* out){out->version=RLMESH_VERSION; return RLMESH_OK;}\n',
            ("rlmesh_capi", "rlmesh_value_none"),
        ),
    }
    cases["dict_readback_symbol"] = (
        '#include "rlmesh.h"\n'
        "uint32_t rlmesh_abi_version(void){return 5;}\n"
        "RLMeshStatus rlmesh_build_info(RLMeshBuildInfo* out){out->version=RLMESH_VERSION; return RLMESH_OK;}\n"
        "RLMeshStatus rlmesh_dict_new(const RLMeshStr* keys, const RLMeshSpaceSpec* const* spaces, size_t n, RLMeshDict** out){*out=(RLMeshDict*)1; return RLMESH_OK;}\n"
        "void rlmesh_space_free(RLMeshSpace* space){}\n",
        ("rlmesh_capi", "rlmesh_dict_len"),
    )
    for name, (source, needles) in cases.items():
        directory = ROOT / "target/csharp-gate" / name
        directory.mkdir(parents=True, exist_ok=True)
        library = directory / "librlmesh_capi.so"
        if name == "bad_image":
            library.write_text(source)
        elif source is not None:
            file = directory / "stub.c"
            file.write_text(
                source + "\nvoid rlmesh_value_none(void) { __builtin_trap(); }\n"
                if name in ("generation", "version")
                else source
            )
            # Avoid the header prototype for the trap: use the real signature.
            file.write_text(
                file.read_text().replace(
                    "void rlmesh_value_none(void)",
                    "RLMeshStatus rlmesh_value_none(RLMeshValue** out)",
                )
            )
            run(
                [
                    *CC,
                    "-shared",
                    "-fPIC",
                    "-I",
                    "crates/rlmesh-capi/include",
                    str(file),
                    "-o",
                    str(library),
                ],
                env=env,
            )
        isolated = dict(env, LD_LIBRARY_PATH=str(directory))
        result = subprocess.run(
            [
                "dotnet",
                str(probe),
                *(["dict"] if name == "dict_readback_symbol" else []),
            ],
            env=isolated,
            cwd=directory,
            capture_output=True,
            text=True,
        )
        if (
            result.returncode
            or not all(text in result.stdout for text in needles)
            or len(result.stdout.splitlines()) != 2
        ):
            raise SystemExit(
                f"NativeGate {name} failed: {result.stdout}\n{result.stderr}"
            )
        print(f"NativeGate {name}: passed")


def native_gate(env, project, assembly):
    """The Gymnasium gate's rows over the project's env, driven from Python 0.2."""
    run([*DOTNET_BUILD, str(project)], env=env)
    program = project.parent / "bin/Debug/net10.0" / assembly
    run(
        [
            "uv",
            "run",
            "--no-sync",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "python/rlmesh/tests/conformance/native_gate.py",
        ],
        env=dict(env, RLMESH_NATIVE_GATE=f"dotnet {program}"),
    )


def compile_tests(env):
    """Compile C# 9 fixtures and check the two intentionally refused conversions."""
    project = CSHARP / "tests/Compile/Compile.csproj"
    for fixture in ("Accepted", "Char", "UInt64"):
        result = subprocess.run(
            [*DOTNET_BUILD, str(project), f"-p:Fixture={fixture}"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        if fixture == "Accepted":
            if result.returncode:
                raise SystemExit(result.stdout + result.stderr)
        elif result.returncode == 0 or "CS0619" not in result.stdout:
            raise SystemExit(
                f"{fixture} should fail with CS0619:\n{result.stdout}\n{result.stderr}"
            )
        print(f"C# 9 compile fixture {fixture}: passed")


def main():
    """Run the task with the pinned repository tools."""
    scratch = pathlib.Path.home() / ".cache/rlmesh-csharp/tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        TMPDIR=str(scratch),
        RLMESH_RELEASE_BUILD="1",
        RLMESH_TEST_ROOT=str(ROOT),
        MSBUILDDISABLENODEREUSE="1",
        DOTNET_CLI_TELEMETRY_OPTOUT="1",
        DOTNET_NOLOGO="1",
    )
    # Build a non-editable package in an isolated uv environment. It uses the
    # project lock and the same release cohort as the C library, without writing
    # to the author's in-tree extension or development environment.
    env.pop("VIRTUAL_ENV", None)
    env["UV_PROJECT_ENVIRONMENT"] = str(ROOT / "target/csharp-python")
    env["CARGO_TARGET_DIR"] = str(ROOT / "target/csharp-native")
    run(
        [
            "uv",
            "sync",
            "--locked",
            "--no-editable",
            "--no-default-groups",
            "--group",
            "csharp",
            # Rebuild the package even if uv cached a dev-cohort wheel from
            # another task. Cargo still reuses this isolated build directory.
            "--reinstall-package",
            "rlmesh",
        ],
        env=env,
    )
    run(["cargo", "build", "-p", "rlmesh-capi", "--release"], env=env)
    env["LD_LIBRARY_PATH"] = (
        str(ROOT / "target/csharp-native/release")
        + os.pathsep
        + env.get("LD_LIBRARY_PATH", "")
    )
    run([*DOTNET_BUILD, str(CSHARP / "src/RLMesh/RLMesh.csproj")], env=env)
    compile_tests(env)
    gate_tests(env)
    run(
        [
            *CC,
            "-std=c11",
            "-I",
            "crates/rlmesh-capi/include",
            "csharp/tests/init.c",
            "-o",
            "target/csharp-init",
        ],
        env=env,
    )
    project = CSHARP / "tests/RLMesh.Tests/RLMesh.Tests.csproj"
    for target in ("net10.0", "netstandard2.1"):
        # Separate output directories prevent stale project-reference resolution
        # from causing the standard-target run to silently load net10.0.
        output = ROOT / f"target/csharp-tests/{target}"
        run(
            [
                *DOTNET_BUILD,
                str(project),
                f"-p:FoundationTarget={target}",
                "-o",
                str(output),
            ],
            env=env,
        )
        run(
            [
                "dotnet",
                "vstest",
                str(output / "RLMesh.Tests.dll"),
                "--logger:console;verbosity=normal",
            ],
            env=env,
        )
    native_gate(env, CSHARP / "tests/Gate/Gate.csproj", "Gate.dll")
    run(["python", str(CSHARP / "tools/benchmark.py")], env=env)
    print(
        "C# foundation: both assembly suites, ABI/layout, gate and C# 9 fixtures passed"
    )
    unity = ROOT / "integrations/unity/Tests~/DotNet/Tests/Tests.csproj"
    run([*DOTNET_BUILD, str(unity)], env=env)
    run(
        [
            "dotnet",
            "vstest",
            str(unity.parent / "bin/Debug/net10.0/RLMesh.Unity.Tests.dll"),
        ],
        env=env,
    )
    unity_scratch = pathlib.Path.home() / ".cache/rlmesh-unity"
    unity_scratch.mkdir(parents=True, exist_ok=True)
    run(
        [
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
            str(ROOT / "integrations/unity/Tests~/step_client.cpp"),
            "librlmesh_capi.so",
            "-o",
            str(unity_scratch / "step-client"),
        ],
        env=env,
        cwd=ROOT / "target/csharp-native/release",
    )
    dynamic = subprocess.run(
        ["readelf", "-d", str(unity_scratch / "step-client")],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", dynamic)
    if "librlmesh_capi.so" not in needed or any(
        "/" in name or "libstdc++" in name or "libc++" in name for name in needed
    ):
        raise SystemExit(f"Unity step client has non-portable dependencies: {needed}")
    versions = subprocess.run(
        ["readelf", "--version-info", str(unity_scratch / "step-client")],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if any(
        tuple(map(int, version.split("."))) > (2, 17)
        for version in re.findall(r"Name: GLIBC_([\d.]+)", versions)
    ):
        raise SystemExit("Unity step client exceeds the glibc 2.17 baseline")
    print(
        "Unity step client: glibc <= 2.17; basename native dependency; static C++ runtime"
    )


if __name__ == "__main__":
    main()
