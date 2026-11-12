"""A launcher makes a host identity, starts a served env under it, and dials it.

The way a launcher not written in Rust starts ``python -m rlmesh.serve`` in a
container: ``rlmesh host-identity new`` prints the identity and the keyed
address, the served process hosts its session beyond loopback under that
identity, and only the printed key reaches it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("gymnasium")

ENV_SOURCE = """
import gymnasium as gym
from rlmesh import EnvFactory


class Env(EnvFactory):
    def make(self, **kwargs):
        return gym.make("CartPole-v1")
"""


def test_identity_round_trips_through_the_python_api() -> None:
    from rlmesh import launch

    identity, key = launch.launch_identity()
    assert launch.identity_key(identity) == key
    assert (
        launch.launched_address(identity, "10.0.0.5:7447", "env")
        == f"10.0.0.5:7447#{key}/env"
    )
    assert (
        launch.launched_address(identity, "10.0.0.5:7447/bench")
        == f"10.0.0.5:7447/bench#{key}"
    )
    with pytest.raises(ValueError) as refused:
        launch.identity_key("aaaa.bbbb.cccc")
    assert "aaaa" not in str(refused.value)


def _cli(*args: str, env: dict[str, str] | None = None) -> str:
    done = subprocess.run(
        [sys.executable, "-m", "rlmesh", "host-identity", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def _serving(proc: subprocess.Popen[str]) -> str:
    deadline = time.monotonic() + 60
    lines = []
    while time.monotonic() < deadline:
        line = proc.stdout.readline() if proc.stdout else ""
        lines.append(line)
        if "RLMesh serving env" in line:
            return line
        if proc.poll() is not None:
            break
    proc.kill()
    pytest.fail("the served env never reported serving:\n" + "".join(lines))


def test_a_served_env_is_reachable_at_the_printed_keyed_address(
    tmp_path: Path,
) -> None:
    import rlmesh
    from rlmesh import launch
    from rlmesh._rlmesh import sandbox_session_ports

    (tmp_path / "launched_env.py").write_text(ENV_SOURCE, encoding="utf-8")
    port = sandbox_session_ports()
    identity_file = tmp_path / "identity"
    printed = json.loads(
        _cli(
            "new",
            "--out",
            str(identity_file),
            "--address",
            f"127.0.0.1:{port}",
            "--participant",
            "env",
            "--json",
        )
    )
    identity = identity_file.read_text(encoding="utf-8")
    assert printed["identity"] is None
    assert identity.strip() not in json.dumps(printed)
    assert (identity_file.stat().st_mode & 0o777) == 0o600
    key_lines = _cli(
        "key", env={**os.environ, "RLMESH_HOST_IDENTITY": identity}
    ).splitlines()
    assert key_lines == [f"key={printed['key']}"]

    serve_env = {
        **os.environ,
        "RLMESH_HOST_IDENTITY": identity,
        # Beyond loopback, as a container's published ports are.
        "RLMESH_ADDRESS": f"0.0.0.0:{port}",
        "RLMESH_WORKFLOW_EDITION": "",
        "RLMESH_STATE_DIR": str(tmp_path / "state"),
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "rlmesh.serve", "--env", "launched_env:Env"],
        cwd=str(tmp_path),
        env=serve_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _serving(proc)
        env = rlmesh.RemoteEnv(printed["address"], connect_timeout_seconds=20)
        try:
            observation, _ = env.reset(seed=0)
            assert observation is not None
        finally:
            env.close()

        _, wrong = launch.launch_identity()
        # The link gate refuses the wrong key: a refusal retrying cannot get
        # past (not the ConnectionError a launcher retries until the host
        # listens), said well within the timeout.
        started = time.monotonic()
        with pytest.raises(rlmesh.ProtocolException) as refused:
            rlmesh.RemoteEnv(
                f"127.0.0.1:{port}#{wrong}/env", connect_timeout_seconds=30
            ).reset(seed=0)
        assert time.monotonic() - started < 15
        assert wrong not in str(refused.value)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
