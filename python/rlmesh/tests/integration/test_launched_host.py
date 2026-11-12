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
        serving = _serving(proc)
        # Stdout is the container's log: the key is a credential.
        assert printed["key"] not in serving
        assert "#…/env" in serving
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
            rest, _ = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            rest, _ = proc.communicate()
    assert printed["key"] not in rest
    assert identity.strip() not in rest


# The repo's script that runs a command as a host and as a client on another
# network (two network namespaces, else this machine's non-loopback address).
ACROSS_NETNS = Path(__file__).resolve().parents[4] / "scripts" / "test-across-netns.sh"

ACROSS_DISPATCH = """
if [ "$RLMESH_TEST_ROLE" = host ]; then
    RLMESH_HOST_IDENTITY=$(cat "$RLMESH_TEST_DIR/identity")
    # Every interface, as a container serves; the client dials the host's IP.
    RLMESH_ADDRESS="0.0.0.0:$RLMESH_TEST_PORT"
    export RLMESH_HOST_IDENTITY RLMESH_ADDRESS
    # Kept apart from the client's output, which a failure shows.
    exec "$0" -m rlmesh.serve --env launched_env:Env >"$RLMESH_TEST_DIR/host.log" 2>&1
fi
exec "$0" across_client.py
"""

ACROSS_CLIENT = """
import os
import sys
import time

import rlmesh
from rlmesh import launch

where = f"{os.environ['RLMESH_TEST_HOST_IP']}:{os.environ['RLMESH_TEST_PORT']}"
with open(os.path.join(os.environ["RLMESH_TEST_DIR"], "key"), encoding="utf-8") as f:
    key = f.read().strip()
env = rlmesh.RemoteEnv(f"{where}#{key}/env", connect_timeout_seconds=60)
try:
    observation, _ = env.reset(seed=0)
    assert observation is not None
finally:
    env.close()
print("ACROSS the keyed address joined", flush=True)

_, wrong = launch.launch_identity()
started = time.monotonic()
try:
    rlmesh.RemoteEnv(f"{where}#{wrong}/env", connect_timeout_seconds=20).reset(seed=0)
except rlmesh.ProtocolException as refused:
    took = time.monotonic() - started
    message = str(refused)
    assert wrong[:20] not in message, "the error repeats the key"
    assert "key" in message, message
    assert took < 5, f"refused only after {took:.1f}s: {message}"
    print(f"ACROSS a wrong key refused in {took:.2f}s: {message}", flush=True)
else:
    sys.exit("a wrong key joined")
"""


def test_a_served_env_is_reachable_from_another_machine(tmp_path: Path) -> None:
    """The keyed address alone joins a launched env across a real network."""
    from rlmesh import launch
    from rlmesh._rlmesh import sandbox_session_ports

    (tmp_path / "launched_env.py").write_text(ENV_SOURCE, encoding="utf-8")
    (tmp_path / "across_client.py").write_text(ACROSS_CLIENT, encoding="utf-8")
    identity, key = launch.launch_identity()
    for name, secret in (("identity", identity), ("key", key)):
        path = tmp_path / name
        path.write_text(secret, encoding="utf-8")
        path.chmod(0o600)
    env = {
        key_: value
        for key_, value in os.environ.items()
        if key_ != "RLMESH_HOST_IDENTITY"
    }
    env.update(
        RLMESH_TEST_DIR=str(tmp_path),
        RLMESH_TEST_PORT=str(sandbox_session_ports()),
        RLMESH_WORKFLOW_EDITION="",
        RLMESH_STATE_DIR=str(tmp_path / "state"),
    )
    done = subprocess.run(
        ["sh", str(ACROSS_NETNS), "sh", "-c", ACROSS_DISPATCH, sys.executable],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    secrets = (identity, key)

    def scrubbed(text: str) -> str:
        for secret in secrets:
            text = text.replace(secret, "<secret>")
        return text

    output = done.stdout + done.stderr
    leaked = any(secret[:20] in output for secret in secrets)
    assert not leaked, "a secret reached the client's output"
    if done.returncode == 77:
        pytest.skip(output.strip())
    host_log = tmp_path / "host.log"
    host = scrubbed(host_log.read_text(encoding="utf-8")) if host_log.exists() else ""
    assert done.returncode == 0, f"{output}\nhost:\n{host}"
    assert "ACROSS the keyed address joined" in output, output
