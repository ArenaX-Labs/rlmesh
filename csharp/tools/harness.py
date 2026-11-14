"""Publish the small serving harness with NativeAOT and drive it through Python."""

import os
import selectors
import subprocess
import time

from test import CSHARP, ROOT, run


def test_env():
    """Reuse test:csharp's release-cohort Python client without a rebuild."""
    env = dict(os.environ, RLMESH_RELEASE_BUILD="1", MSBUILDDISABLENODEREUSE="1")
    env.pop("VIRTUAL_ENV", None)
    env["UV_PROJECT_ENVIRONMENT"] = str(ROOT / "target/csharp-python")
    return env


def drive(command, directory, env):
    """Check the real loader without an inherited native search path."""
    env = dict(env)
    env.pop("LD_LIBRARY_PATH", None)
    with subprocess.Popen(
        command,
        cwd=directory,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as server:
        try:
            with selectors.DefaultSelector() as ready:
                ready.register(server.stdout, selectors.EVENT_READ)
                if not ready.select(timeout=30):
                    raise AssertionError(
                        "serving harness did not publish an address in 30s"
                    )
            address = server.stdout.readline().strip()
            assert address.endswith("/env"), address
            subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    str(CSHARP / "tests/python_client.py"),
                    address,
                    "normal",
                ],
                cwd=ROOT,
                env=env,
                timeout=90,
                check=True,
            )
            server.communicate("stop\n", timeout=30)
            assert server.returncode == 0, server.returncode
        finally:
            if server.poll() is None:
                server.kill()
                server.wait()


def main():
    """Time publish and end-to-end execution for the CI runtime report."""
    started = time.monotonic()
    env = test_env()
    output = ROOT / "target/csharp-aot"
    run(
        [
            "dotnet",
            "publish",
            str(CSHARP / "tests/Harness/Harness.csproj"),
            "-c",
            "Release",
            "-r",
            "linux-x64",
            "--framework",
            "net10.0",
            "-p:TargetFrameworks=net10.0",
            "-p:PublishAot=true",
            "-p:UseSharedCompilation=false",
            "--disable-build-servers",
            "-o",
            str(output),
        ],
        env=env,
    )
    import shutil

    shutil.copyfile(
        ROOT / "target/csharp-native/release/librlmesh_capi.so",
        output / "librlmesh_capi.so",
    )
    drive([str(output / "Harness")], output, env)
    print(
        f"NativeAOT publish + Python serving supplement: {time.monotonic() - started:.2f}s"
    )


if __name__ == "__main__":
    main()
