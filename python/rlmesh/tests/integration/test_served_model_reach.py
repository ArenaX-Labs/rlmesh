"""``Model(address).run(local_env)`` with the model served by another process.

The model's process serves on loopback, or in a network namespace of its own
with its port published on this machine's loopback under another number (as
a container's is), where it can dial neither this machine's loopback nor
the address its driver dials.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

# Serves a model as `model` in the session at argv[1] until stdin closes; with argv[2], first relays
# the Unix socket `session` in that directory to the host's port.
_MODEL_SERVER = textwrap.dedent(
    """
    import socket, sys, threading
    import rlmesh

    address = sys.argv[1]

    def pipe(a, b):
        try:
            while data := a.recv(65536):
                b.sendall(data)
        except OSError:
            pass
        finally:
            for s in (a, b):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def relay(path, port):
        listener = socket.socket(socket.AF_UNIX)
        listener.bind(path)
        listener.listen()
        while True:
            peer, _ = listener.accept()
            try:
                host = socket.create_connection(("127.0.0.1", port))
            except OSError:
                peer.close()
                continue
            threading.Thread(target=pipe, args=(peer, host), daemon=True).start()
            threading.Thread(target=pipe, args=(host, peer), daemon=True).start()

    if len(sys.argv) > 2:
        port = int(address.rsplit(":", 1)[1])
        threading.Thread(
            target=relay, args=(f"{sys.argv[2]}/session", port), daemon=True
        ).start()
    model = rlmesh.Model(lambda obs: 1)
    threading.Thread(target=lambda: (sys.stdin.read(), model.close()), daemon=True).start()
    model.serve(address, name="model")
    """
)


class ThreeStepEnv:
    def __init__(self) -> None:
        from rlmesh import spaces

        self.observation_space = spaces.Discrete(4)
        self.action_space = spaces.Discrete(2)
        self.steps = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[int, dict[str, object]]:
        _ = seed, options
        self.steps = 0
        return 0, {}

    def step(self, action: object) -> tuple[int, float, bool, bool, dict[str, object]]:
        self.steps += 1
        return self.steps, 1.0, self.steps >= 3, False, {}

    def close(self) -> None:
        return None


@contextmanager
def _model_process(argv: list[str], **env: str) -> Iterator[None]:
    process = subprocess.Popen(argv, stdin=subprocess.PIPE, env={**os.environ, **env})
    try:
        yield
    finally:
        assert process.stdin is not None
        process.stdin.close()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


# Run outside this process: a native call holding the GIL while it waits on
# the link must not stall the forwarding it waits on.
_PUBLISH = textwrap.dedent(
    """
    import socket, sys, threading

    def pipe(a, b):
        try:
            while data := a.recv(65536):
                b.sendall(data)
        except OSError:
            pass
        finally:
            for s in (a, b):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def accept(listener, path):
        while True:
            peer, _ = listener.accept()
            inside = socket.socket(socket.AF_UNIX)
            try:
                inside.connect(path)
            except OSError:
                peer.close()
                inside.close()
                continue
            threading.Thread(target=pipe, args=(peer, inside), daemon=True).start()
            threading.Thread(target=pipe, args=(inside, peer), daemon=True).start()

    port, relay = int(sys.argv[1]), sys.argv[2]
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen()
    threading.Thread(
        target=accept, args=(listener, f"{relay}/session"), daemon=True
    ).start()
    print("ready", flush=True)
    sys.stdin.read()
    """
)


@contextmanager
def _published(port: int, relay: Path) -> Iterator[None]:
    """Forward loopback ``port`` to the Unix socket ``session`` in ``relay``,
    as a container's published port."""
    script = relay / "publish.py"
    script.write_text(_PUBLISH)
    process = subprocess.Popen(
        [sys.executable, str(script), str(port), str(relay)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        yield
    finally:
        assert process.stdin is not None
        process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _namespaces() -> bool:
    if sys.platform != "linux" or shutil.which("unshare") is None:
        return False
    probe = subprocess.run(
        ["unshare", "-rn", "ip", "link", "set", "lo", "up"],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


def test_a_model_in_another_process_drives_a_local_env_by_its_address_alone(
    tmp_path: Path, session_port: int
) -> None:
    import rlmesh

    script = tmp_path / "serve_model.py"
    script.write_text(_MODEL_SERVER)
    session = f"127.0.0.1:{session_port}"
    with _model_process([sys.executable, str(script), session]):
        result = rlmesh.Model(f"{session}/model").run(ThreeStepEnv(), episodes=2)
    assert result.total_steps == 6


@pytest.mark.skipif(not _namespaces(), reason="no unprivileged network namespaces")
def test_a_model_that_cannot_reach_its_driver_drives_a_local_env(
    tmp_path: Path, session_port: int
) -> None:
    import rlmesh

    script = tmp_path / "serve_model.py"
    script.write_text(_MODEL_SERVER)
    from rlmesh.launch import launch_identity

    # The launcher's identity, through the environment as a sandbox passes it.
    identity, key = launch_identity()
    model = _model_process(
        [
            "unshare",
            "-rn",
            "sh",
            "-c",
            'ip link set lo up && exec "$0" "$@"',
            sys.executable,
            str(script),
            "127.0.0.1:20000",
            str(tmp_path),
        ],
        RLMESH_HOST_IDENTITY=identity,
    )
    with model, _published(session_port, tmp_path):
        address = f"127.0.0.1:{session_port}#{key}/model"
        result = rlmesh.Model(address).run(ThreeStepEnv(), episodes=2)
        assert result.total_steps == 6
        # An env given by address is the model's to reach: it can't,
        # and says so well before the 30 s connect timeout.
        server = rlmesh.EnvServer(ThreeStepEnv(), "127.0.0.1:0/elsewhere/env")
        server.start()
        started = time.monotonic()
        try:
            with pytest.raises(
                ConnectionError, match="could not reach the env's session"
            ):
                rlmesh.Model(address).run(server.address, episodes=1)
        finally:
            server.shutdown()
        assert time.monotonic() - started < 15
