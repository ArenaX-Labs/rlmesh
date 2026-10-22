"""The v2 lockstep path (§12) from Python.

- Existing Gymnasium envs and v1 models run unchanged at participant
  addresses: the same env and model over v2 (lockstep) leave the traces the
  v1 runtime left (kept in ``golden/lockstep``), bit for bit, across every
  space kind and v1 dtype, a vector env with NEXT_STEP autoreset, seeds and
  chunking.
- A served model (``Model(address)``) runs a lockstep env as v1 ran a gRPC one.
- ``RemoteEnv`` drives a v2-native robot served in lockstep through its Gym
  view; a lockstep robot needs no rates or kinds.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

import rlmesh  # noqa: E402
import rlmesh.numpy  # noqa: E402
import rlmesh.v2 as v2  # noqa: E402


def _digest(value: Any) -> str:
    """A stable digest of a (nested) observation or action, bytes and dtypes."""
    h = hashlib.sha256()

    def walk(v: Any) -> None:
        if isinstance(v, dict):
            for key in sorted(v):
                h.update(repr(key).encode())
                walk(v[key])
        elif isinstance(v, (tuple, list)):
            h.update(b"(")
            for item in v:
                walk(item)
            h.update(b")")
        elif isinstance(v, str):
            h.update(b"s" + v.encode())
        else:
            a = np.asarray(v)
            h.update(str(a.dtype).encode() + repr(a.shape).encode() + a.tobytes())

    walk(value)
    return h.hexdigest()[:16]


def catalog_spaces() -> tuple[Any, Any]:
    s = gym.spaces
    observation = s.Dict(
        {
            "cam": s.Box(0, 255, (6, 6, 3), np.uint8),
            "state": s.Box(-1.0, 1.0, (7,), np.float32),
            "i8": s.Box(-1, 1, (3,), np.int8),
            "u32": s.Box(0, 9, (2,), np.uint32),
            "u64": s.Box(0, 2**40, (2,), np.uint64),
            "f64": s.Box(-1.0, 1.0, (2, 2), np.float64),
            "mode": s.Discrete(5, start=-2),
            "md": s.MultiDiscrete([2, 3]),
            "mb": s.MultiBinary(4),
            "word": s.Text(16),
            "pair": s.Tuple((s.Box(-1.0, 1.0, (2,), np.float16), s.Discrete(3))),
        }
    )
    action = s.Box(-1.0, 1.0, (3,), np.float32)
    return observation, action


class CatalogEnv(gym.Env):  # type: ignore[misc]
    """Deterministic given its seeds and actions; records every call."""

    def __init__(self, log: list[str] | None = None) -> None:
        self.observation_space, self.action_space = catalog_spaces()
        self.log = log if log is not None else []
        self._rng = np.random.default_rng(0)
        self._t = 0
        self._length = 3

    def _obs(self) -> dict[str, Any]:
        r = self._rng
        return {
            "cam": r.integers(0, 256, (6, 6, 3), dtype=np.uint8),
            "state": r.uniform(-1, 1, 7).astype(np.float32),
            "i8": r.integers(-1, 2, 3, dtype=np.int8),
            "u32": r.integers(0, 10, 2, dtype=np.uint32),
            "u64": r.integers(0, 2**40, 2, dtype=np.uint64),
            "f64": r.uniform(-1, 1, (2, 2)),
            "mode": int(r.integers(-2, 3)),
            "md": r.integers(0, [2, 3]).astype(np.int64),
            "mb": r.integers(0, 2, 4, dtype=np.int8),
            "word": "cup" * int(r.integers(1, 4)),
            "pair": (r.uniform(-1, 1, 2).astype(np.float16), int(r.integers(0, 3))),
        }

    def reset(self, *, seed: int | None = None, options: Any = None) -> Any:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed if seed is not None else 0)
        self._t = 0
        self._length = int(self._rng.integers(2, 6))
        obs = self._obs()
        self.log.append(f"reset seed={seed} options={options} obs={_digest(obs)}")
        return obs, {"length": self._length, "third": 1 / 3}

    def step(self, action: Any) -> Any:
        self._t += 1
        a = np.asarray(action)
        obs = self._obs()
        reward = float(a.astype(np.float64).sum()) / 3 + self._t / 7
        done = self._t >= self._length
        self.log.append(
            f"step action={_digest(a)} reward={reward.hex()} done={done} obs={_digest(obs)}"
        )
        info = {"is_success": done and self._length % 2 == 0, "x": reward * 2}
        return (
            obs,
            reward,
            done and self._length % 2 == 0,
            done and self._length % 2 == 1,
            info,
        )


class Recorder:
    """A v1 model: the action is a pure function of the observation."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def _action(self, obs: Any, frame: int) -> Any:
        x = int(_digest((obs, frame))[:8], 16)
        return np.array([((x >> s) % 200 - 100) / 100 for s in (0, 8, 16)], np.float32)

    def predict(self, obs: Any) -> Any:
        self.seen.append(_digest(obs))
        return self._action(obs, 0)

    def predict_batch(self, observations: Any) -> Any:
        """One batched forward over a vector env's lanes (leaves batch-leading)."""
        lanes = int(np.asarray(observations["state"]).shape[0])
        self.seen.append(f"batch of {lanes} {_digest(observations)}")
        return np.stack([self._action(observations, lane) for lane in range(lanes)])

    def predict_chunk(self, obs: Any, horizon: int) -> Any:
        self.seen.append(f"{horizon}x {_digest(obs)}")
        return np.stack([self._action(obs, i) for i in range(horizon)])


class Hooks(rlmesh.RunHooks):
    def __init__(self) -> None:
        self.steps: list[str] = []
        self.episodes: list[str] = []

    def on_step(self, event: Any) -> None:
        self.steps.append(
            f"{event.episode}/{event.step} action={_digest(event.action)} "
            f"reward={float(event.reward).hex()} term={event.terminated} "
            f"trunc={event.truncated} info={sorted(event.info.items())}"
        )

    def on_episode_end(self, result: Any) -> None:
        self.episodes.append(
            f"seed={result.seed} steps={result.steps} reward={float(result.reward).hex()} "
            f"term={result.terminated} trunc={result.truncated} success={result.success}"
        )


GOLDEN = Path(__file__).parent / "golden" / "lockstep"


def _golden(name: str, traces: Any) -> None:
    """Checks ``traces`` against the golden file ``name``: what the v1 runtime
    left before it was deleted."""
    path = GOLDEN / f"{name}.json"
    assert json.loads(path.read_text()) == json.loads(json.dumps(traces)), (
        f"traces differ from {path}"
    )


def _traces(make_env: Any, address: str, **run: Any) -> tuple[Any, ...]:
    log: list[str] = []
    server = rlmesh.EnvServer(make_env(log), address)
    server.start()
    recorder, hooks = Recorder(), Hooks()
    model = rlmesh.numpy.Model(recorder)
    try:
        result = model.run(
            server.address, hooks=hooks, workflow_edition="2026.06", **run
        )
    finally:
        model.close()
        server.shutdown()
    return log, recorder.seen, hooks.steps, hooks.episodes, result.num_episodes


def test_a_catalog_env_runs_unchanged_on_lockstep_with_seeds_and_chunking() -> None:
    run = {"seeds": [5, 6, 2**40, 8], "execution_horizon": 2}
    traces = _traces(CatalogEnv, ":0/catalog", **run)
    assert traces[4] == 4 and traces[1] and traces[2]
    _golden("catalog_chunked", traces)


def test_a_next_step_vector_env_runs_unchanged_on_lockstep() -> None:
    def make(log: list[str]) -> Any:
        return gym.vector.SyncVectorEnv(
            [lambda: CatalogEnv(log) for _ in range(3)],
            autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
        )

    run = {"episodes": 7}
    traces = _traces(make, ":0/vector", **run)
    assert traces[4] == 7
    _golden("vector_next_step", traces)


def test_a_served_model_runs_a_lockstep_env_as_v1_ran_a_grpc_one(
    session_port: int,
) -> None:
    import threading

    session = f"127.0.0.1:{session_port}"
    model = rlmesh.numpy.Model(Recorder())
    serving = threading.Thread(
        target=model.serve, args=(session,), kwargs={"name": "m"}
    )
    serving.start()

    def trace(address: str) -> tuple[Any, ...]:
        log: list[str] = []
        server = rlmesh.EnvServer(CatalogEnv(log), address)
        server.start()
        hooks = Hooks()
        try:
            rlmesh.numpy.Model(f"{session}/m").run(
                server.address, seeds=[5, 6], hooks=hooks, workflow_edition="2026.06"
            )
        finally:
            server.shutdown()
        # Steps' info carries run-minted episode ids; the env's log and the
        # episodes say what ran.
        return log, len(hooks.steps), hooks.episodes

    try:
        traces = trace(":0/served")
        assert len(traces[2]) == 2
        _golden("served_model", traces)
    finally:
        model.close()
        serving.join(10)


ARM = {
    "arm": {
        "kind": "joints",
        "count": 2,
        "fields": ["pos"],
        "hz": 100,
        "command": "position",
    }
}


def test_remote_env_drives_a_lockstep_robot() -> None:
    with v2.Host() as host:
        robot = v2.Robot("arm1", ARM, session=host, clock="lockstep")
        q = np.zeros(2, np.float32)

        def step(commands: dict[str, Any]) -> Any:
            nonlocal q
            q = q + 0.5 * (commands["arm"] - q)
            robot.publish("arm", pos=q)
            return (
                -float(np.abs(1 - q).sum()),
                bool(q[0] > 0.9),
                False,
                {"q0": float(q[0])},
            )

        def reset(seed: int | None, options: Any) -> None:
            nonlocal q
            q = np.full(2, float(seed or 0), np.float32)
            robot.publish("arm", pos=q)

        robot.serve_lockstep(step, reset, background=True)
        with pytest.raises(v2.SessionException, match="inside its step"):
            robot.publish("arm", pos=q)
        env = rlmesh.numpy.RemoteEnv(f"{host.address}/arm1")
        try:
            assert isinstance(env, gym.Env)
            observations, actions = (
                cast("Any", env.observation_space),
                cast("Any", env.action_space),
            )
            assert set(observations.spaces) == {"arm"}
            assert actions["arm"].shape == (2,)
            obs, info = env.reset(seed=0)
            assert obs["arm"].tolist() == [0.0, 0.0] and info["steps"] == 0
            rewards = []
            terminated = False
            while not terminated:
                obs, reward, terminated, truncated, info = env.step(
                    {"arm": np.ones(2, np.float32)}
                )
                rewards.append(reward)
                assert info["edge_modes"] == {"arm": "driving"}
                assert info["q0"] == float(obs["arm"][0])
            assert rewards == [-1.0, -0.5, -0.25, -0.125]
        finally:
            env.close()
            robot.close()


def test_a_remote_env_view_step_is_not_delayed_by_the_signal_poll() -> None:
    # Blocked calls poll for Ctrl-C on a timer. Its first tick used to be
    # awaited before the call started, and it waited for the timer's next
    # millisecond, so every Gym step took over 1 ms in-process.
    import statistics

    with v2.Host() as host:
        robot = v2.Robot("arm1", ARM, session=host, clock="lockstep")
        q = np.zeros(2, np.float32)

        def step(commands: dict[str, Any]) -> Any:
            robot.publish("arm", pos=q)
            return 0.0, False, False, {}

        def reset(seed: int | None, options: Any) -> None:
            robot.publish("arm", pos=q)

        robot.serve_lockstep(step, reset, background=True)
        env = rlmesh.numpy.RemoteEnv(f"{host.address}/arm1")
        try:
            env.reset(seed=0)
            action = {"arm": np.ones(2, np.float32)}
            latencies = []
            for _ in range(300):
                started = time.perf_counter()
                env.step(action)
                latencies.append(time.perf_counter() - started)
        finally:
            env.close()
            robot.close()
    assert statistics.median(latencies) < 0.0005


def test_a_lockstep_robot_needs_no_rates_or_kinds() -> None:
    # A game: no hz (lockstep steps it), no kinds (each part is its own).
    game = {
        "board": {"fields": {"cells": ["int8", [9]]}},
        "move": {"fields": {"cell": ["int64", []]}, "command": "value"},
    }
    with v2.Host() as host:
        robot = v2.Robot("ttt", game, session=host, clock="lockstep")
        assert robot.parts == {"board": "local/board", "move": "local/move"}
        board = np.zeros(9, np.int8)

        def step(commands: dict[str, Any]) -> Any:
            board[int(commands["move"])] = 1
            robot.publish("board", board)
            return 1 / 3, bool(board.all()), False, {"filled": np.int64(board.sum())}

        def reset(seed: int | None, options: Any) -> Any:
            board[:] = 0
            robot.publish("board", board)
            return {"options": options}

        robot.serve_lockstep(step, reset, background=True)
        env = rlmesh.numpy.RemoteEnv(f"{host.address}/ttt")
        try:
            _, info = env.reset(seed=1, options={"first": np.arange(2)})
            assert info["options"] == {"first": [0, 1]}
            obs, reward, _, _, info = env.step(np.int32(4))  # the one action entry
            assert obs["board"].tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 0]
            assert (reward, info["filled"]) == (1 / 3, 1)  # f64, not f32
        finally:
            env.close()
            robot.close()


def test_remote_env_picks_the_gym_path_for_a_gym_env_and_the_view_for_a_robot() -> None:
    server = rlmesh.EnvServer(CatalogEnv(), ":0/gym")
    server.start()
    try:
        remote = rlmesh.numpy.RemoteEnv(server.address)
        assert remote.observation_space["mode"] == catalog_spaces()[0]["mode"]
        remote.close()
    finally:
        server.shutdown()
    with v2.Host() as host:
        robot = v2.Robot("arm1", ARM, session=host, clock="lockstep")
        q = np.zeros(2, np.float32)

        def step(commands: dict[str, Any]) -> None:
            robot.publish("arm", pos=q)

        def reset(seed: int | None, options: Any) -> None:
            robot.publish("arm", pos=q)

        robot.serve_lockstep(step, reset, background=True)
        env = rlmesh.numpy.RemoteEnv(f"{host.address}/arm1")
        try:
            assert set(cast("Any", env.observation_space).spaces) == {"arm"}
            _, info = env.reset()
            assert {"ages_ns", "stale", "edge_modes"} <= set(info)
        finally:
            env.close()
            robot.close()


class _Raises(gym.Env):  # type: ignore[misc]
    observation_space = gym.spaces.Discrete(2)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def reset(self, *, seed: int | None = None, options: Any = None) -> Any:
        raise self.error


def _undecodable() -> UnicodeDecodeError:
    try:
        b"\xff".decode()
    except UnicodeDecodeError as error:
        return error
    raise AssertionError


@pytest.mark.parametrize(
    ("error", "raised", "typed"),
    [
        (KeyError("k"), KeyError, True),
        (OSError("disk"), OSError, True),
        # A builtin whose own constructor takes more than a message.
        (_undecodable(), UnicodeDecodeError, True),
        (rlmesh.EnvironmentException("own"), rlmesh.EnvironmentException, False),
    ],
)
def test_an_env_s_builtin_exception_raises_its_class_with_its_message(
    error: BaseException, raised: type[BaseException], typed: bool
) -> None:
    server = rlmesh.EnvServer(_Raises(error), ":0/raises")
    server.start()
    try:
        remote = rlmesh.numpy.RemoteEnv(server.address)
        try:
            with pytest.raises(raised) as caught:
                remote.reset()
            assert isinstance(caught.value, rlmesh.EnvironmentException)
            assert (type(caught.value) is not rlmesh.EnvironmentException) == typed
            if typed:
                with pytest.raises(raised) as again:
                    remote.reset()
                assert type(again.value) is type(caught.value), "made once"
                assert str(caught.value) == str(error)
                assert caught.value.phase == "reset"
        finally:
            remote.close()
    finally:
        server.shutdown()


def test_a_spec_that_is_not_json_forwards_its_basic_fields() -> None:
    from gymnasium.envs.registration import EnvSpec

    env = gym.make("CartPole-v1").unwrapped
    # Its own metadata: a vector env elsewhere writes the class's autoreset mode.
    env.metadata = {"render_modes": []}
    env.spec = EnvSpec(
        "Mine-v0", entry_point=type(env), max_episode_steps=9, kwargs={"x": object()}
    )
    server = rlmesh.EnvServer(env, ":0/spec")
    server.start()
    try:
        remote = rlmesh.numpy.RemoteEnv(server.address)
        try:
            assert remote.spec is not None
            assert (remote.spec.id, remote.spec.max_episode_steps) == ("Mine-v0", 9)
            assert remote.spec.kwargs == {} and remote.spec.entry_point is None
            assert "rlmesh.gym.spec" not in remote.metadata
        finally:
            remote.close()
    finally:
        server.shutdown()


def _same_step() -> Any:
    return gym.vector.SyncVectorEnv(
        [lambda: gym.make("CartPole-v1")] * 2,
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def test_same_step_is_served_on_lockstep_and_refused_by_name_by_run() -> None:
    server = rlmesh.EnvServer(_same_step(), ":0/same")
    server.start()
    try:
        model = rlmesh.Model(lambda observation: 0)
        with pytest.raises(ValueError, match="SAME_STEP"):
            model.run(server.address, episodes=1)
        remote = rlmesh.numpy.RemoteVectorEnv(server.address)
        try:
            remote.reset(seed=0)
            remote.step(np.zeros(2, np.int64))
        finally:
            remote.close()
    finally:
        server.shutdown()


def test_addresses_parse_as_written_and_old_forms_name_their_upgrade() -> None:
    assert v2.parse_address("lab:7447/bench-3/cartpole") == {
        "host": "lab",
        "port": 7447,
        "session": "bench-3",
        "participant": "cartpole",
    }
    assert v2.parse_address("robot-lab/ur10")["session"] == "robot-lab"
    assert v2.parse_address("lab:7447", session=True)["participant"] is None
    with pytest.raises(ValueError, match=re.escape('use "127.0.0.1:7447/gym"')):
        rlmesh.EnvServer(CatalogEnv(), "lockstep://127.0.0.1:7447/gym")


SERVE_BY_NAME = """
import sys, gymnasium as gym, rlmesh
server = rlmesh.EnvServer(gym.make("CartPole-v1"), "cartpole")
server.start()
print(server.address, flush=True)
sys.stdin.read()
server.shutdown()
"""


def test_two_processes_meet_by_name_the_first_hosting(
    monkeypatch: pytest.MonkeyPatch, session_port: int
) -> None:
    import subprocess
    import sys

    # A free well-known port of this test's own, so a running rlmeshd or a
    # concurrent run cannot meet it.
    port = session_port
    monkeypatch.setenv("RLMESH_PORT", str(port))
    child = subprocess.Popen(
        [sys.executable, "-c", SERVE_BY_NAME],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None and child.stdin is not None
    try:
        printed = child.stdout.readline().strip()
        assert printed == f"127.0.0.1:{port}/cartpole", printed
        remote = rlmesh.numpy.RemoteEnv("cartpole")
        try:
            obs, _ = remote.reset(seed=0)
            assert obs.shape == (4,)
            remote.step(0)
        finally:
            remote.close()
    finally:
        child.stdin.close()
        child.wait(timeout=10)
