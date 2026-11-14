"""Probe C model serve signal modes over sessions."""

from __future__ import annotations

import os
import select
import subprocess
import sys


def read_checked(process: subprocess.Popen[str]) -> None:
    """Wait for the native signal assertion without hanging on a failed child."""
    assert process.stdout is not None
    output = b""
    while b"checked signals\n" not in output:
        ready, _, _ = select.select([process.stdout], [], [], 10)
        assert ready, "C consumer did not finish its signal assertions"
        chunk = os.read(process.stdout.fileno(), 4096)
        assert chunk, "C consumer exited before checking signals"
        output += chunk


def command(process: subprocess.Popen[str], mode: str) -> None:
    """Drive the C consumer's signal check and stop, and expect a clean exit."""
    assert process.stdin is not None
    if "foreground" not in mode:
        process.stdin.write("i\n")
        process.stdin.flush()
        read_checked(process)
    if mode not in {"signals", "old-signals"}:
        process.stdin.write("q\n")
        process.stdin.flush()
    stdout, _ = process.communicate(timeout=15)
    assert process.returncode == 0, (mode, stdout)


def probe(binary: str, mode: str) -> None:
    """Run one fresh process so signal dispositions cannot leak between cases."""
    if mode in {"no-signals", "signals", "old-signals"}:
        # The model serves on a session the C process hosts; it waits for its
        # own run through the served model before checking signals.
        process = subprocess.Popen(
            [binary, mode, ":0/host"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            command(process, mode)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        return


if __name__ == "__main__":
    for case in (
        "no-signals",
        "signals",
        "old-signals",
    ):
        probe(sys.argv[1], case)
        print(f"C serving probe: {case} OK", flush=True)
