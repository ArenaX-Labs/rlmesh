"""The Gymnasium gate's rows over envs, models and clients written in another
language, served and dialed on v2 sessions against Python 0.2.

Each native language's test task runs this file with the command of its gate
program::

    RLMESH_NATIVE_GATE="<command>" uv run pytest python/rlmesh/tests/conformance/native_gate.py

The rows, spaces and fixtures are the gate's own (``test_gym_conformance``);
the program only serves or drives them. It answers:

``<command> env <row> <address>``
    Serve the echo env of ``row`` at ``address``, print ``listening on
    <participant address>`` and stop at stdin EOF.
``<command> model <address>``
    Print ``serving`` and serve the echo model at the participant ``address``
    until stdin EOF.
``<command> remote-env <address>``, ``<command> run <address>``
    Drive the env at ``address`` for two episodes (seeds 0 and 1), each
    step's action the last observation: by hand through the language's remote
    env, or as a run of its echo model. Print ``episodes=2``.
``<command> remote-model <model address> <env address>``
    Run the served model at the first address against the env at the second
    for two episodes; print ``episodes=2``.

Any call may print ``unsupported: <why>`` and exit 0 instead: the row is
skipped with that reason, which names what the language lacks.

The echo env's observation and action space are ``SPACES[row]`` (a ``float32``
``Box(-1, 1, (3,))`` for the other rows). ``reset(seed, options)`` returns any
value of the space with ``{"options": options}`` as its info when options are
given (else ``{}``); ``step(action)`` returns the action as the observation,
reward ``t`` (the step count), terminated at ``t >= 3`` and info ``{"t": t}``.
Row ``info`` returns ``rich_info(t)`` as the step info, ``both_done`` also
truncates when it terminates, ``error_reset`` / ``error_step`` raise the
language's ValueError ``boom in reset`` / RuntimeError ``boom in step``,
``render`` renders a ``uint8`` ``(4, 5, 3)`` frame whose channel 0 is ``t``,
and ``scalar_box``'s space is a shape-``()`` Box, refused before it serves.
The echo model's action is its observation.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import rlmesh
import rlmesh.numpy

sys.path.insert(0, str(Path(__file__).parent))
from test_gym_conformance import (
    ADDRESS,
    EDITION,
    RESET_OPTIONS,
    SPACES,
    SpaceEnv,
    difference,
    gym,
    rich_info,
    sampled,
)

COMMAND = shlex.split(os.environ.get("RLMESH_NATIVE_GATE", ""))
if not COMMAND:
    pytest.skip("RLMESH_NATIVE_GATE names no gate program", allow_module_level=True)

UNSUPPORTED = "unsupported: "


def program(*args: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [*COMMAND, *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def first_line(child: subprocess.Popen[str]) -> str:
    assert child.stdout is not None
    line = child.stdout.readline().strip()
    if line.startswith(UNSUPPORTED):
        child.communicate(timeout=30)
        pytest.skip(line.removeprefix(UNSUPPORTED))
    return line


def stop(child: subprocess.Popen[str]) -> str:
    """Close the program's stdin, its cue to stop; its exit must be clean."""
    out, err = child.communicate(timeout=30)
    assert child.returncode == 0, err
    return out


def call(*args: str) -> str:
    """One driving call: its output, or the row skipped as unsupported."""
    done = subprocess.run(
        [*COMMAND, *args], capture_output=True, text=True, timeout=60, check=False
    )
    if done.stdout.startswith(UNSUPPORTED):
        pytest.skip(done.stdout.strip().removeprefix(UNSUPPORTED))
    assert done.returncode == 0, done.stderr
    return done.stdout


@contextmanager
def native_env(row: str) -> Iterator[Any]:
    """The program's echo env for ``row``, as the RemoteEnv a Gym user dials."""
    child = program("env", row, ADDRESS)
    try:
        line = first_line(child)
        assert line.startswith("listening on "), line
        remote = rlmesh.numpy.RemoteEnv(line.removeprefix("listening on "))
        try:
            yield remote
        finally:
            remote.close()
        stop(child)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


@pytest.mark.parametrize("name", sorted(SPACES))
def test_a_space_and_its_values_round_trip_bit_exact(name: str) -> None:
    space = SPACES[name]
    with native_env(name) as remote:
        assert remote.observation_space == space
        assert remote.action_space == space
        observation, info = remote.reset(seed=0)
        assert space.contains(observation) and info == {}
        for t, action in enumerate(sampled(space, 3), 1):
            want = (action, float(t), t >= 3, False, {"t": t})
            assert difference(want, remote.step(action)) is None


def test_info_round_trips_as_the_env_produced_it() -> None:
    with native_env("info") as remote:
        remote.reset(seed=0)
        for t in (1, 2, 3):
            _, _, _, _, info = remote.step(remote.action_space.sample())
            assert difference(rich_info(t), info) is None


@pytest.mark.parametrize(
    ("where", "error"), [("reset", ValueError), ("step", RuntimeError)]
)
def test_an_env_error_raises_at_once_with_its_type_and_message(
    where: str, error: type[Exception]
) -> None:
    with native_env(f"error_{where}") as remote:
        started = time.perf_counter()
        with pytest.raises(error, match=f"boom in {where}") as raised:
            remote.reset(seed=0)
            remote.step(remote.action_space.sample())
        assert isinstance(raised.value, rlmesh.EnvironmentException)
        assert str(raised.value) == f"boom in {where}"
        assert raised.value.phase == where
        assert time.perf_counter() - started < 2


def test_render_returns_the_envs_frame() -> None:
    reference = SpaceEnv()
    reference.reset(seed=1)
    reference.step(reference.action_space.sample())
    with native_env("render") as remote:
        remote.reset(seed=1)
        remote.step(remote.action_space.sample())
        assert difference(reference.render(), remote.render()) is None


@pytest.mark.xfail(
    strict=True,
    reason="COR-P10: native metadata carries an array as its nested lists, where "
    "Python 0.2 keeps a NumPy array an array (owner decision)",
)
def test_reset_options_reach_the_env_as_given() -> None:
    with native_env("reset_options") as remote:
        _, info = remote.reset(seed=0, options=RESET_OPTIONS)
        assert difference({"options": RESET_OPTIONS}, info) is None
        assert remote.reset()[1] == {}


def test_a_step_both_terminated_and_truncated_arrives_so() -> None:
    with native_env("both_done") as remote:
        remote.reset(seed=0)
        flags = [remote.step(remote.action_space.sample())[2:4] for _ in range(3)]
    assert flags == [(False, False), (False, False), (True, True)]


def test_a_scalar_box_is_refused_at_once_by_name() -> None:
    started = time.perf_counter()
    child = program("env", "scalar_box", ADDRESS)
    line = first_line(child)
    _, err = child.communicate(timeout=30)
    assert child.returncode != 0 and "shape" in err, (line, err)
    assert time.perf_counter() - started < 2


class Echoed(gym.Wrapper):  # type: ignore[misc]
    """Records each action beside the observation it answered, and the resets."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self.pairs: list[tuple[Any, Any]] = []
        self.resets = 0

    def reset(self, **kwargs: Any) -> Any:
        self.last, info = self.env.reset(**kwargs)
        self.resets += 1
        return self.last, info

    def step(self, action: Any) -> Any:
        self.pairs.append((self.last, action))
        self.last, *rest = self.env.step(action)
        return (self.last, *rest)


def assert_echoed(env: Echoed) -> None:
    """Every action was the observation before it, bit-exact."""
    assert env.resets == 2 and env.pairs
    observations, actions = zip(*env.pairs, strict=True)
    assert difference(list(observations), list(actions), scalars=True) is None


CLIENT_SPACES = [
    "box_f32_bounded",
    "box_u8_image",
    "discrete_start",
    "multi_binary_2d",
    "dict_nested",
    "tuple_nested",
    "text",
]


@contextmanager
def python_env(name: str) -> Iterator[tuple[Echoed, Any]]:
    """The gate's SpaceEnv on ``SPACES[name]`` both ways, served from Python."""
    space = SPACES[name]
    env = Echoed(SpaceEnv(space, space))
    server = rlmesh.EnvServer(env, ADDRESS, options=EDITION)
    server.start()
    try:
        yield env, server
    finally:
        server.shutdown()


@pytest.mark.parametrize("verb", ["remote-env", "run"])
@pytest.mark.parametrize("name", CLIENT_SPACES)
def test_a_native_client_drives_a_python_env(verb: str, name: str) -> None:
    with python_env(name) as (env, server):
        assert "episodes=2" in call(verb, server.address)
    assert_echoed(env)


@contextmanager
def native_model(session: str) -> Iterator[str]:
    """The program's echo model, joined to ``session`` as ``policy``."""
    address = f"{session}/policy"
    child = program("model", address)
    try:
        assert first_line(child) == "serving"
        yield address
        stop(child)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def session_of(address: str) -> str:
    return address.rsplit("/", 1)[0]


@pytest.mark.parametrize("name", CLIENT_SPACES)
def test_python_runs_a_native_model(name: str) -> None:
    with (
        python_env(name) as (env, server),
        native_model(session_of(server.address)) as address,
    ):
        result = rlmesh.Model(address).run(server.address, episodes=2)
    assert len(result.episodes) == 2
    assert_echoed(env)


def test_python_binds_a_native_model() -> None:
    with (
        python_env("dict_nested") as (env, server),
        native_model(session_of(server.address)) as address,
        rlmesh.Model(address).bind(server.address) as binding,
    ):
        for seed in (0, 1):
            binding.reset(seed=seed)
            while not binding.done:
                binding.step()
    assert_echoed(env)


@pytest.mark.parametrize("name", CLIENT_SPACES)
def test_a_native_client_runs_a_python_model(name: str) -> None:
    with python_env(name) as (env, server):
        session = session_of(server.address)
        model = rlmesh.Model(lambda observation: observation)
        serving = threading.Thread(
            target=model.serve, args=(session,), kwargs={"name": "policy"}
        )
        serving.start()
        try:
            out = call("remote-model", f"{session}/policy", server.address)
        finally:
            model.close()
            serving.join(timeout=30)
        assert "episodes=2" in out
    assert_echoed(env)
