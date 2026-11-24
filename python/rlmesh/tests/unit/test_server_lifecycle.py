from __future__ import annotations

import os
import socket
import subprocess
import sys
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
        'server = rlmesh.EnvServer(TinyEnv(), ":0/env")\nprint(server.address)\n',
    )

    assert result.returncode == 0, result.stderr


def test_a_started_server_that_is_never_shut_down_exits_cleanly(
    tmp_path: Path,
) -> None:
    result = _run_child(
        tmp_path,
        'server = rlmesh.EnvServer(TinyEnv(), ":0/env")\nserver.start()\n',
    )

    assert result.returncode == 0, result.stderr
    assert "CLOSE_CALLED" in result.stdout


def test_an_explicitly_stopped_server_exits_cleanly(tmp_path: Path) -> None:
    result = _run_child(
        tmp_path,
        'server = rlmesh.EnvServer(TinyEnv(), ":0/env")\n'
        "server.start()\n"
        "server.shutdown()\n",
    )

    assert result.returncode == 0, result.stderr


def test_a_model_still_serving_at_exit_is_closed_before_finalization(
    tmp_path: Path, session_port: int
) -> None:
    result = _run_child(
        tmp_path,
        "import threading\n"
        f"session = '127.0.0.1:{session_port}'\n"
        "model = rlmesh.Model(\n"
        "    lambda observation: 0,\n"
        "    on_close=lambda: print('MODEL_CLOSED', flush=True),\n"
        ")\n"
        "threading.Thread(\n"
        "    target=model.serve, args=(session,), kwargs={'name': 'm'}, daemon=True\n"
        ").start()\n"
        "rlmesh.Model(f'{session}/m').run(TinyEnv(), episodes=2, max_episode_steps=5)\n",
    )

    assert result.returncode == 0, result.stderr
    assert "MODEL_CLOSED" in result.stdout


def test_a_script_that_raises_while_holding_a_server_reports_the_error(
    tmp_path: Path,
) -> None:
    result = _run_child(
        tmp_path,
        'server = rlmesh.EnvServer(TinyEnv(), ":0/env")\nraise RuntimeError("boom")\n',
    )

    assert result.returncode == 1, result.stderr
    assert "RuntimeError: boom" in result.stderr


@pytest.mark.skipif(os.name != "posix", reason="SIGINT delivery is POSIX-only")
def test_ctrl_c_interrupts_wait_and_the_env_still_closes(tmp_path: Path) -> None:
    result = _run_child(
        tmp_path,
        "import os, signal, threading\n"
        'server = rlmesh.EnvServer(TinyEnv(), ":0/env")\n'
        "server.start()\n"
        "threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGINT)).start()\n"
        "try:\n"
        "    server.wait()\n"
        "except KeyboardInterrupt:\n"
        '    print("INTERRUPTED", flush=True)\n'
        "finally:\n"
        "    server.shutdown()\n",
    )

    assert result.returncode == 0, result.stderr
    assert "INTERRUPTED" in result.stdout
    assert "CLOSE_CALLED" in result.stdout


_THREAD_ENVS = (
    "import threading\n"
    "seen = []\n"
    "main = threading.main_thread().ident\n"
    "class ThreadEnv(TinyEnv):\n"
    "    def reset(self, *, seed=None, options=None):\n"
    "        seen.append(threading.get_ident())\n"
    "        return 0, {}\n"
    "    def step(self, action):\n"
    "        seen.append(threading.get_ident())\n"
    "        return 0, 0.0, False, False, {}\n"
    "class ThreadVectorEnv:\n"
    "    num_envs = 2\n"
    "    single_observation_space = rlmesh.spaces.Discrete(4)\n"
    "    single_action_space = rlmesh.spaces.Discrete(2)\n"
    "    def reset(self, *, seed=None, options=None):\n"
    "        seen.append(threading.get_ident())\n"
    "        return [0, 0], {}\n"
    "    def step(self, actions):\n"
    "        seen.append(threading.get_ident())\n"
    "        return [0, 0], [0.0, 0.0], [False, False], [False, False], {}\n"
    "    def close(self):\n"
    "        print('CLOSE_CALLED', flush=True)\n"
    "def report():\n"
    "    print('ON_MAIN' if seen == [main, main] else f'OFF_MAIN {seen} {main}', flush=True)\n"
)


@pytest.mark.parametrize(
    ("env", "client", "action"),
    [
        ("ThreadEnv", "rlmesh.RemoteEnv", "0"),
        ("ThreadVectorEnv", "rlmesh.RemoteVectorEnv", "[0, 0]"),
    ],
)
def test_serve_keeps_the_env_on_the_calling_thread(
    tmp_path: Path, env: str, client: str, action: str
) -> None:
    result = _run_child(
        tmp_path,
        _THREAD_ENVS + f"server = rlmesh.EnvServer({env}(), ':0/env')\n"
        "def drive():\n"
        f"    handle = {client}(server.address)\n"
        "    handle.reset()\n"
        f"    handle.step({action})\n"
        "    handle.close()\n"
        "    server.shutdown()\n"
        "threading.Thread(target=drive).start()\n"
        "server.serve()\n"
        "report()\n",
    )

    assert result.returncode == 0, result.stderr
    assert "ON_MAIN" in result.stdout, result.stdout
    assert "CLOSE_CALLED" in result.stdout


@pytest.mark.skipif(os.name != "posix", reason="SIGINT delivery is POSIX-only")
def test_ctrl_c_stops_every_foreground_serve_and_the_env_still_closes(
    tmp_path: Path,
) -> None:
    result = _run_child(
        tmp_path,
        "import os, signal, threading\n"
        "for _ in range(2):\n"
        '    server = rlmesh.EnvServer(TinyEnv(), ":0/env")\n'
        "    threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGINT)).start()\n"
        "    try:\n"
        "        server.serve()\n"
        "    except KeyboardInterrupt:\n"
        '        print("INTERRUPTED", flush=True)\n',
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("CLOSE_CALLED") == 2, result.stdout
    assert result.stdout.count("INTERRUPTED") == 2, result.stdout


@pytest.mark.skipif(os.name != "posix", reason="SIGINT delivery is POSIX-only")
def test_a_second_ctrl_c_interrupts_an_env_call_that_never_returns(
    tmp_path: Path,
) -> None:
    result = _run_child(
        tmp_path,
        "import os, signal, threading, time\n"
        "class WedgedEnv(TinyEnv):\n"
        "    def close(self):\n"
        '        print("CLOSE_CALLED", flush=True)\n'
        "        time.sleep(60)\n"
        "server = rlmesh.EnvServer(\n"
        '    WedgedEnv(), ":0/env",\n'
        "    options=rlmesh.ServeOptions(close_timeout_seconds=0.1),\n"
        ")\n"
        "threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGINT)).start()\n"
        "threading.Timer(1.5, lambda: os.kill(os.getpid(), signal.SIGINT)).start()\n"
        "started = time.monotonic()\n"
        "try:\n"
        "    server.serve()\n"
        "except (KeyboardInterrupt, RuntimeError) as exc:\n"
        '    print(f"RETURNED {time.monotonic() - started:.1f} {exc!r}", flush=True)\n',
    )

    assert result.returncode == 0, result.stderr
    assert "CLOSE_CALLED" in result.stdout
    line = next(out for out in result.stdout.splitlines() if out.startswith("RETURNED"))
    assert float(line.split()[1]) < 10, result.stdout
    assert "close timed out" in line, result.stdout


def test_shutdown_detaches_the_finalizer() -> None:
    server = rlmesh.EnvServer(TinyEnv(), ":0/env")
    assert server._finalizer.alive

    server.shutdown()

    assert not server._finalizer.alive


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_non_finite_serve_timeouts_raise_value_error(value: float) -> None:
    with pytest.raises(ValueError, match="close_timeout_seconds"):
        rlmesh.ServeOptions(close_timeout_seconds=value)


def test_serve_cli_reports_a_clean_stop_on_ctrl_c(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupted(*_args: Any, **_kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(serve_cli, "resolve_entrypoint", lambda *a, **k: object())
    monkeypatch.setattr(serve_cli, "serve_env", interrupted)

    assert serve_cli.main(["--env", "pkg.module:make_env"]) == 130


def test_shutting_down_an_unstarted_lockstep_server_stops_its_session_host() -> None:
    server = rlmesh.EnvServer(TinyEnv(), ":0/env")
    host, port = server.address.split("#")[0].rsplit(":", 1)
    socket.create_connection((host, int(port)), timeout=5).close()
    server.shutdown()
    with pytest.raises(OSError):
        socket.create_connection((host, int(port)), timeout=5).close()
