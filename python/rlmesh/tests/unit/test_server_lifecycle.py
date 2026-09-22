"""EnvServer lifecycle: a server must never outlive the interpreter.

A server that is still alive when CPython finalizes kills the process with a
signal -- its serve and lane threads keep calling into an interpreter that is
being torn down -- so ``EnvServer`` stops and joins itself from a finalizer.
The exit paths are exercised in a child process: a negative ``returncode`` is a
signal death (SIGABRT/SIGSEGV), which is exactly the failure being guarded.

Also covers the neighbouring lifecycle contracts: ``wait()`` is interruptible,
the unix bind path never deletes a file it cannot prove is a dead socket, and a
non-finite serve timeout raises instead of panicking.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import rlmesh
from rlmesh import serve as serve_cli

_TINY_ENV = """
import rlmesh


class TinyEnv:
    observation_space = rlmesh.spaces.Discrete(4)
    action_space = rlmesh.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        return 0, {}

    def step(self, action):
        return 0, 0.0, False, False, {}

    def close(self):
        print("CLOSE_CALLED", flush=True)
"""


class TinyEnv:
    """The in-process twin of the child script's env."""

    observation_space = rlmesh.spaces.Discrete(4)
    action_space = rlmesh.spaces.Discrete(2)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]]:
        return 0, {}

    def step(self, action: Any) -> tuple[int, float, bool, bool, dict[str, Any]]:
        return 0, 0.0, False, False, {}

    def close(self) -> None:
        return None


def _run_child(tmp_path: Path, body: str) -> subprocess.CompletedProcess[str]:
    """Run ``body`` (after the ``TinyEnv`` preamble) in a child interpreter."""
    script = tmp_path / "child.py"
    script.write_text(_TINY_ENV + body)
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_a_held_server_exits_cleanly(tmp_path: Path) -> None:
    result = _run_child(
        tmp_path,
        'server = rlmesh.EnvServer(TinyEnv(), "127.0.0.1:0")\nprint(server.address)\n',
    )

    assert result.returncode == 0, result.stderr


def test_a_started_server_that_is_never_shut_down_exits_cleanly(
    tmp_path: Path,
) -> None:
    result = _run_child(
        tmp_path,
        'server = rlmesh.EnvServer(TinyEnv(), "127.0.0.1:0")\nserver.start()\n',
    )

    # The finalizer shuts the background server down and joins it while the
    # interpreter is still alive, so the env's close() hook still runs.
    assert result.returncode == 0, result.stderr
    assert "CLOSE_CALLED" in result.stdout


def test_an_explicitly_stopped_server_exits_cleanly(tmp_path: Path) -> None:
    result = _run_child(
        tmp_path,
        'server = rlmesh.EnvServer(TinyEnv(), "127.0.0.1:0")\n'
        "server.start()\n"
        "server.shutdown()\n",
    )

    assert result.returncode == 0, result.stderr


def test_a_script_that_raises_while_holding_a_server_reports_the_error(
    tmp_path: Path,
) -> None:
    result = _run_child(
        tmp_path,
        'server = rlmesh.EnvServer(TinyEnv(), "127.0.0.1:0")\n'
        'raise RuntimeError("boom")\n',
    )

    # The real traceback and exit 1, not a signal death on top of it.
    assert result.returncode == 1, result.stderr
    assert "RuntimeError: boom" in result.stderr


@pytest.mark.skipif(os.name != "posix", reason="SIGINT delivery is POSIX-only")
def test_ctrl_c_interrupts_wait_and_the_env_still_closes(tmp_path: Path) -> None:
    result = _run_child(
        tmp_path,
        "import os, signal, threading\n"
        'server = rlmesh.EnvServer(TinyEnv(), "127.0.0.1:0")\n'
        "server.start()\n"
        "threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGINT)).start()\n"
        "try:\n"
        "    server.wait()\n"
        "except KeyboardInterrupt:\n"
        '    print("INTERRUPTED", flush=True)\n'
        "finally:\n"
        "    server.shutdown()\n",
    )

    # wait() releases the GIL, so without a per-iteration signal poll Ctrl-C is
    # discarded and a background server can only be killed.
    assert result.returncode == 0, result.stderr
    assert "INTERRUPTED" in result.stdout
    assert "CLOSE_CALLED" in result.stdout


def test_shutdown_detaches_the_finalizer() -> None:
    server = rlmesh.EnvServer(TinyEnv(), "127.0.0.1:0")
    assert server._finalizer.alive

    server.shutdown()

    assert not server._finalizer.alive


@pytest.fixture
def socket_dir(tmp_path: Path) -> Iterator[Path]:
    """A directory short enough for an AF_UNIX path (108 bytes on Linux).

    pytest's ``tmp_path`` follows ``TMPDIR`` and can exceed that budget.
    """
    if len(str(tmp_path)) < 80:
        yield tmp_path
        return
    short = Path(tempfile.mkdtemp(prefix="rlm-", dir="/tmp"))
    yield short
    shutil.rmtree(short, ignore_errors=True)


@pytest.mark.skipif(os.name != "posix", reason="unix sockets are POSIX-only")
def test_binding_over_a_live_unix_socket_is_refused(socket_dir: Path) -> None:
    path = socket_dir / "live.sock"
    live = rlmesh.EnvServer(TinyEnv(), f"unix://{path}")
    live.start()
    try:
        with pytest.raises(ConnectionError, match="already"):
            rlmesh.EnvServer(TinyEnv(), f"unix://{path}")
    finally:
        live.shutdown()


@pytest.mark.skipif(os.name != "posix", reason="unix sockets are POSIX-only")
def test_a_non_socket_file_is_left_in_place(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("precious")

    with pytest.raises(ConnectionError):
        rlmesh.EnvServer(TinyEnv(), f"unix://{path}")

    assert path.read_text() == "precious"


@pytest.mark.skipif(os.name != "posix", reason="unix sockets are POSIX-only")
def test_a_stale_socket_is_reclaimed(socket_dir: Path) -> None:
    path = socket_dir / "stale.sock"
    leftover = socket.socket(socket.AF_UNIX)
    leftover.bind(str(path))
    leftover.close()

    server = rlmesh.EnvServer(TinyEnv(), f"unix://{path}")
    try:
        assert server.address == f"unix://{path}"
    finally:
        server.shutdown()


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_non_finite_serve_timeouts_raise_value_error(value: float) -> None:
    with pytest.raises(ValueError, match="idle_timeout_seconds"):
        rlmesh.ServeOptions(idle_timeout_seconds=value)


def test_serve_cli_reports_a_clean_stop_on_ctrl_c(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupted(*_args: Any, **_kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(serve_cli, "resolve_entrypoint", lambda *a, **k: object())
    monkeypatch.setattr(serve_cli, "serve_env", interrupted)

    # Ctrl-C is how an operator stops a served container: the conventional
    # interrupted status, not a traceback.
    assert serve_cli.main(["--env", "pkg.module:make_env"]) == 130
