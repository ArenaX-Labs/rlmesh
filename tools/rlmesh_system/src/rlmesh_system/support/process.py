from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from queue import Empty, Queue

from rlmesh_system.support.command import CommandError, HarnessRenderer


class ManagedProcess:
    def __init__(
        self,
        process: subprocess.Popen[str],
        *,
        command: list[str],
        log_path: Path,
        output: list[str],
        thread: threading.Thread,
    ) -> None:
        self.process = process
        self.command = command
        self.log_path = log_path
        self.output = output
        self.thread = thread

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.thread.join(timeout=5)


def start_process_until(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    label: str,
    verbose: bool,
    renderer: HarnessRenderer,
    timeout_seconds: float,
    predicate: Callable[[str], str | None],
) -> tuple[ManagedProcess, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    renderer.step(label)
    if verbose:
        renderer.command(command)

    queue: Queue[str | None] = Queue()
    output: list[str] = []
    log = log_path.open("w")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def drain() -> None:
        try:
            assert process.stdout is not None
            for line in process.stdout:
                output.append(line)
                log.write(line)
                log.flush()
                queue.put(line)
        finally:
            queue.put(None)
            log.close()

    thread = threading.Thread(target=drain, name=f"{label}-log", daemon=True)
    thread.start()

    started = time.monotonic()
    last_notice = started
    while True:
        elapsed = time.monotonic() - started
        if elapsed > timeout_seconds:
            _terminate(process)
            thread.join(timeout=5)
            raise CommandError(
                command,
                124,
                log_path,
                "".join(output) + f"\ncommand timed out after {timeout_seconds:g}s\n",
            )
        if process.poll() is not None and queue.empty():
            thread.join(timeout=5)
            raise CommandError(
                command,
                process.returncode or 1,
                log_path,
                "".join(output),
            )
        if elapsed - last_notice >= 30:
            renderer.command_running(label, elapsed, log_path)
            last_notice = elapsed
        try:
            line = queue.get(timeout=0.1)
        except Empty:
            continue
        if line is None:
            continue
        value = predicate(line)
        if value is not None:
            return (
                ManagedProcess(
                    process,
                    command=command,
                    log_path=log_path,
                    output=output,
                    thread=thread,
                ),
                value,
            )


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
