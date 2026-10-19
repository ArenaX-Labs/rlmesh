"""Generate the internal ABI layer, or compare it without changing the checkout."""

import argparse
import os
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
CSHARP = ROOT / "csharp"
OUTPUT = CSHARP / "src/RLMesh/Interop/Native.g.cs"


def main():
    """Run the task with the pinned repository tools."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    dotnet_scratch = pathlib.Path.home() / ".cache/rlmesh-csharp/tmp"
    dotnet_scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(dotnet_scratch)
    zig = subprocess.check_output(
        ["mise", "where", "zig"], cwd=CSHARP, text=True
    ).strip()
    subprocess.run(["dotnet", "tool", "restore"], cwd=CSHARP, check=True)
    # Keep generator scratch in the checkout, not the host's small /tmp tmpfs.
    scratch = ROOT / "target/csharp-interop"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        generated = pathlib.Path(directory) / "Native.g.cs"
        result = subprocess.run(
            [
                "dotnet",
                "tool",
                "run",
                "ClangSharpPInvokeGenerator",
                "--",
                "-f",
                str(ROOT / "crates/rlmesh-capi/include/rlmesh.h"),
                "-o",
                str(generated),
                "-x",
                "c",
                "-std",
                "c11",
                "-n",
                "RLMesh.Interop",
                "-m",
                "Native",
                "-l",
                "rlmesh_capi",
                "-c",
                "codegen=compatible",
                "--generate",
                "fnptr-codegen=false",
                "--generate",
                "funcs-with-body=false",
                "--generate",
                "helper-types",
                "--with-access-specifier",
                "*=Internal",
                "--no-resource-directory-detection",
                "-I",
                str(pathlib.Path(zig) / "lib/include"),
            ],
            cwd=CSHARP,
            capture_output=True,
            text=True,
        )
        # 21.1.8.4 returns the diagnostic count (truncated to 8 bits on POSIX).
        # Visibility affects exports, not their ABI; accept only this exact
        # known diagnostic, independent of the number of exported functions.
        warnings = [line for line in result.stdout.splitlines() if "Warning (" in line]
        if result.returncode and not (
            result.returncode
            == (len(warnings) if os.name == "nt" else len(warnings) % 256)
            and warnings
            and all("Unsupported attribute: 'Visibility'." in line for line in warnings)
            and "Error (" not in result.stdout
            and not result.stderr.strip()
            and generated.exists()
        ):
            raise SystemExit(
                f"ClangSharp exited {result.returncode}; output exists: {generated.exists()}\n"
                + result.stdout
                + "\nstderr:\n"
                + result.stderr
            )
        if warnings:
            print(
                f"ClangSharp: {len(warnings)} export-visibility warnings (ABI unaffected)"
            )
        content = generated.read_bytes()
        if args.check:
            if not OUTPUT.exists() or OUTPUT.read_bytes() != content:
                raise SystemExit(
                    "C# interop is stale; run mise run csharp:interop:generate"
                )
        else:
            OUTPUT.write_bytes(content)


if __name__ == "__main__":
    main()
