"""Enter the scoped C# toolchain; caller owns the shared heavy-job lock."""

import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CSHARP = ROOT / "csharp"


def main():
    """Install locked C# tools only on demand and run the selected checks."""
    scratch = pathlib.Path.home() / ".cache/rlmesh-csharp/tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TMPDIR=str(scratch))
    subprocess.run(["mise", "trust", str(CSHARP / "mise.toml")], check=True, env=env)
    subprocess.run(
        ["mise", "-C", str(CSHARP), "install", "--locked", "dotnet", "zig"],
        check=True,
        env=env,
    )
    mode, *extra = sys.argv[1:]
    commands = []
    if mode in ("check", "test", "interop-check"):
        commands.append(["interop.py", "--check"])
    if mode in ("check", "test"):
        commands.append(["check.py", *extra])
    if mode == "test":
        commands.append(["test.py"])
    if mode == "generate":
        commands.append(["interop.py", *extra])
    if not commands:
        raise SystemExit(f"unknown C# task: {mode}")
    for script, *args in commands:
        subprocess.run(
            [
                "mise",
                "-C",
                str(CSHARP),
                "exec",
                "--",
                "python",
                str(CSHARP / "tools" / script),
                *args,
            ],
            cwd=ROOT,
            check=True,
            env=env,
        )


if __name__ == "__main__":
    main()
