"""Probe C model serve signal modes over sessions."""

from __future__ import annotations

import os
import select
import subprocess
import sys


def read_checked(process: subprocess.Popen[str]) -> None:
    assert process.stdout is not None
    output = b""
    while b"checked signals\n" not in output:
        ready, _, _ = select.select([process.stdout], [], [], 10)
        assert ready, "C consumer did not finish its signal assertions"
        chunk = os.read(process.stdout.fileno(), 4096)
        assert chunk, "C consumer exited before checking signals"
        output += chunk


def command(process: subprocess.Popen[str], mode: str) -> None:
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
    if mode in {"no-signals", "signals", "old-signals"}:
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
