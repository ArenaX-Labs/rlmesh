from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Protocol


class HarnessRenderer(Protocol):
    def step(self, label: str) -> None: ...

    def command(self, command: list[str]) -> None: ...

    def command_output(self, output: str) -> None: ...

    def command_running(self, label: str, elapsed: float, log_path: Path) -> None: ...


class CommandError(Exception):
    def __init__(
        self,
        command: list[str],
        returncode: int,
        log_path: Path,
        output: str,
    ) -> None:
        super().__init__(f"{command[0]} exited with {returncode}")
        self.command = command
        self.returncode = returncode
        self.log_path = log_path
        self.output = output


def output_tail(output: str, *, max_lines: int = 20) -> str:
    lines = output.rstrip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(["... output truncated ...", *lines[-max_lines:]])


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_path: Path,
    label: str,
    verbose: bool,
    renderer: HarnessRenderer,
    timeout_seconds: float | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    renderer.step(label)
    if verbose:
        renderer.command(command)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        started = time.monotonic()
        last_notice = started
        output = ""
        timed_out = False
        while True:
            try:
                wait_seconds = 5.0
                if timeout_seconds is not None:
                    elapsed = time.monotonic() - started
                    remaining = timeout_seconds - elapsed
                    if remaining <= 0:
                        stdout = _terminate_process(process)
                        output += stdout or ""
                        output += f"\ncommand timed out after {timeout_seconds:g}s\n"
                        timed_out = True
                        break
                    wait_seconds = min(wait_seconds, max(0.1, remaining))

                stdout, _stderr = process.communicate(timeout=wait_seconds)
                output += stdout or ""
                break
            except subprocess.TimeoutExpired:
                now = time.monotonic()
                if now - last_notice >= 30:
                    renderer.command_running(label, now - started, log_path)
                    last_notice = now
            except KeyboardInterrupt:
                stdout = _terminate_process(process)
                log.write(stdout or "")
                raise
        log.write(output)
    if verbose and output:
        renderer.command_output(output)
    if timed_out:
        raise CommandError(command, 124, log_path, output)
    if process.returncode != 0:
        raise CommandError(command, process.returncode, log_path, output)


def _terminate_process(process: subprocess.Popen[str]) -> str:
    process.terminate()
    try:
        stdout, _stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, _stderr = process.communicate()
    return stdout or ""
