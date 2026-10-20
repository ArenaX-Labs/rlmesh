"""Run E-3's allocation gates and optional interleaved advisory timing batches."""

import argparse
import os
import statistics
import subprocess
from contextlib import ExitStack

from test import CSHARP, ROOT, run


def main():
    """Caller holds dev:heavy, covering both builds and every timing batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    env = dict(
        os.environ,
        RLMESH_RELEASE_BUILD="1",
        LD_LIBRARY_PATH=str(ROOT / "target/csharp-native/release"),
    )
    project = CSHARP / "tests/Benchmark/Benchmark.csproj"
    run(
        [
            "dotnet",
            "build",
            str(project),
            "-c",
            "Release",
            "--disable-build-servers",
            "-p:UseSharedCompilation=false",
        ],
        env=env,
    )
    command = [
        "dotnet",
        str(project.parent / "bin/Release/net10.0/RLMesh.Benchmark.dll"),
    ]
    run(command, env=env)
    if not args.timing:
        return
    binary = ROOT / "target/csharp-benchmark-cpp"
    run(
        [
            "zig",
            "c++",
            "-std=c++17",
            "-O3",
            "-Icrates/rlmesh-capi/include",
            str(project.parent / "runner.cpp"),
            "-Ltarget/csharp-native/release",
            "-lrlmesh_capi",
            "-o",
            str(binary),
        ],
        env=env,
    )
    samples = [[], []]
    with ExitStack() as stack:
        processes = [
            stack.enter_context(
                subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True,
                )
            )
            for cmd in ([*command, "--timing"], [str(binary)])
        ]
        try:
            for process in processes:
                assert process.stdout.readline().strip() == "ready"
            for _ in range(5):
                for process, values in zip(processes, samples, strict=True):
                    process.stdin.write("batch\n")
                    process.stdin.flush()
                    values.append(float(process.stdout.readline()))
            medians = [statistics.median(values) for values in samples]
            print(
                f"Advisory timing (5 interleaved x 10000): C# {medians[0]:.1f} ns/step; C++ {medians[1]:.1f} ns/step; ratio {medians[0] / medians[1]:.3f}x (target 1.25x; never gating)"
            )
            print(f"Raw ns/step: C# {samples[0]}; C++ {samples[1]}")
        finally:
            for process in processes:
                process.stdin.close()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
