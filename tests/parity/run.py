"""Build and execute all four E-3 runners using the CI-installed toolchain."""

from __future__ import annotations

import argparse
import os
import pathlib
import shlex
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def run(command, **kwargs):
    """Run an argv without a shell, propagating failures and diagnostics."""
    print("+", shlex.join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, **kwargs)


def main():
    """Compile once, or reuse the binaries for repeated stress runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--load",
        action="store_true",
        help="Keep one CPU worker active during repeated verification",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    target = pathlib.Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    if not target.is_absolute():
        target = ROOT / target
    out = pathlib.Path(os.environ.get("RLMESH_PARITY_BIN", target / "parity"))
    out.mkdir(parents=True, exist_ok=True)
    scratch = target / "parity-tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    build_env = {**os.environ, "TMPDIR": str(scratch), "RLMESH_WORKFLOW_EDITION": ""}
    run([sys.executable, ROOT / "tests/parity/generate_defaults.py", "--check"])
    if not args.no_build:
        run(["cargo", "build", "-p", "rlmesh-capi", "--lib"], env=build_env)
        include = ROOT / "crates/rlmesh-capi/include"
        link = [
            "-L",
            str(target / "debug"),
            "-lrlmesh_capi",
            f"-Wl,-rpath,{target / 'debug'}",
            "-lpthread",
            "-ldl",
            "-lm",
        ]
        for language, compiler, standard, source in [
            ("c", os.environ.get("CC", "zig cc"), "c11", "c_runner.c"),
            ("cpp", os.environ.get("CXX", "zig c++"), "c++17", "cpp_runner.cpp"),
        ]:
            quiet = (
                ["-Wno-unused-command-line-argument", "-Wno-nullability-completeness"]
                if "zig" in compiler
                else []
            )
            run(
                [
                    *shlex.split(compiler),
                    f"-std={standard}",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    *quiet,
                    "-I",
                    str(include),
                    str(ROOT / "tests/parity" / source),
                    *link,
                    "-o",
                    str(out / language),
                ],
                env=build_env,
            )
    test_env = {**os.environ, "RLMESH_PARITY_BIN": str(out)}
    # Python tests need the system temp directory: some contracts run outside a checkout.
    test_env.pop("TMPDIR", None)
    # One bounded worker stays inside the task's heavy lock; it is always reaped.
    load = None
    if args.load:
        load = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import hashlib; data=b'x'*65536\nwhile True: hashlib.sha256(data).digest()",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    try:
        for iteration in range(1, args.repeat + 1):
            print(
                f"==> parity iteration {iteration}/{args.repeat} (load={bool(load)})",
                flush=True,
            )
            run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/parity",
                    "-q",
                    *(["-k", "not python"] if args.native_only else []),
                ],
                env=test_env,
            )
            if not args.native_only:
                run(
                    [
                        "cargo",
                        "test",
                        "-p",
                        "rlmesh",
                        "--test",
                        "parity_fixtures",
                        "--test",
                        "parity_defaults",
                        "--",
                        "--nocapture",
                    ],
                    env=build_env,
                )
    finally:
        if load:
            load.terminate()
            load.wait(timeout=10)


if __name__ == "__main__":
    main()
