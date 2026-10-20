"""Run a built IL2CPP player, drive N episodes and require a clean shutdown."""

import argparse
import json
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from client import drive


def main() -> None:
    """Read readiness from the player's stdout without a fixed startup sleep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    scratch = Path.home() / ".cache/rlmesh-unity"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as temporary:
        complete = Path(temporary) / "complete"
        player = subprocess.Popen(
            [
                str(args.player.resolve()),
                "-batchmode",
                "-nographics",
                "-logFile",
                "-",
                "--rlmesh-complete-file",
                str(complete),
                "--rlmesh-episodes",
                str(args.episodes),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(os.environ, TMPDIR=temporary),
        )
        lines: queue.Queue[str | None] = queue.Queue()

        def read() -> None:
            assert player.stdout is not None
            for line in player.stdout:
                print(line, end="", flush=True)
                lines.put(line)
            lines.put(None)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        try:
            deadline = time.monotonic() + 120
            host_runtime = None
            while True:
                line = lines.get(timeout=max(0.01, deadline - time.monotonic()))
                if line is None:
                    raise RuntimeError("player exited before serving")
                if "RLMesh host runtime: " in line:
                    host_runtime = json.loads(line.split("RLMesh host runtime: ", 1)[1])
                match = re.search(r"Serving Unity-ReachTarget-v0 on (tcp://\S+)", line)
                if match:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "player did not start serving within 120 seconds"
                    )
            assert host_runtime is not None, "player did not report its host runtime"
            assert host_runtime["language"] == "csharp", host_runtime
            versions = host_runtime["framework_versions"]
            assert versions["unity"] and versions["il2cpp"], host_runtime
            assert "mono" not in versions and "dotnet" not in versions, host_runtime
            drive(match[1], args.episodes)
            complete.write_text(str(args.episodes))
            if player.wait(timeout=30) != 0:
                raise RuntimeError(f"player exited with {player.returncode}")
            reader.join(timeout=5)
            print("Unity IL2CPP player gate passed", flush=True)
        finally:
            if player.poll() is None:
                player.kill()
                player.wait()


if __name__ == "__main__":
    main()
