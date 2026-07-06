"""Execute the headless examples in ``examples/python`` as real subprocesses.

Every example that runs with only the repository environment (no GPU, no
Docker, no external weights) is exercised here, so a change that breaks one
fails CI. The rest of the set is typechecked but not executed:

- ``byo_container/``: Docker image entrypoints; the env/model servers block forever.
- ``sandbox/``: starts owned Docker containers.
- ``vla_adapters/``: model x simulator study; real runs need weights and simulators.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
EXAMPLES = REPO_ROOT / "examples" / "python"
RUN_TIMEOUT = 120.0
SERVER_STARTUP_TIMEOUT = 30.0


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port
    finally:
        sock.close()


def _run_example(relative: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLES / relative), *args],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT,
        check=False,
    )


def _assert_ok(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        f"exit={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _serve_example(relative: str) -> Iterator[str]:
    """Start a server example on a free port and yield its address."""
    address = f"127.0.0.1:{_free_port()}"
    process = subprocess.Popen(
        [sys.executable, str(EXAMPLES / relative), "--address", address],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_until_listening(address, process)
        yield address
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _wait_until_listening(address: str, process: subprocess.Popen[str]) -> None:
    host, port = address.rsplit(":", 1)
    deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise AssertionError(
                f"server exited early (exit={process.returncode}):\n{output}"
            )
        try:
            with socket.create_connection((host, int(port)), timeout=0.5):
                return
        except OSError:
            time.sleep(0.05)
    raise AssertionError(f"server never listened on {address}")


@pytest.fixture(scope="module")
def counter_address() -> Iterator[str]:
    """quickstart/serve.py: the dependency-light CounterEnv server."""
    yield from _serve_example("quickstart/serve.py")


@pytest.fixture(scope="module")
def cartpole_address() -> Iterator[str]:
    """quickstart/serve_gymnasium.py: Gymnasium CartPole-v1 served over RLMesh."""
    yield from _serve_example("quickstart/serve_gymnasium.py")


def test_eval_against_gymnasium_server(cartpole_address: str) -> None:
    result = _run_example("quickstart/eval.py", "--address", cartpole_address)
    _assert_ok(result)
    assert "connected to" in result.stdout
    assert cartpole_address in result.stdout
    assert "step=1 reward=" in result.stdout
    assert "episode complete" in result.stdout or "stopped after" in result.stdout


def test_eval_against_counter_server(counter_address: str) -> None:
    result = _run_example("quickstart/eval.py", "--address", counter_address)
    _assert_ok(result)
    assert "connected to" in result.stdout
    assert counter_address in result.stdout
    assert "episode complete" in result.stdout


def test_model_worker_against_gymnasium_server(cartpole_address: str) -> None:
    result = _run_example(
        "quickstart/model.py", "--address", cartpole_address, "--episodes", "1"
    )
    _assert_ok(result)


def test_eval_many_across_both_servers(
    counter_address: str, cartpole_address: str
) -> None:
    result = _run_example("quickstart/eval_many.py", counter_address, cartpole_address)
    _assert_ok(result)
    assert f"{counter_address}: connected" in result.stdout
    assert f"{cartpole_address}: connected" in result.stdout
    assert f"{counter_address}: episode complete" in result.stdout


def test_adapters_serve_and_run() -> None:
    result = _run_example("adapters/serve_and_run.py")
    _assert_ok(result)
    assert "Resolved adapter:" in result.stdout
    assert "Running one episode:" in result.stdout
    assert "Done." in result.stdout
