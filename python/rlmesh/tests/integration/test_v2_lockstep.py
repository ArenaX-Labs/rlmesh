"""The v2 lockstep path (§12) from Python.

- Existing Gymnasium envs and v1 models run unchanged at ``lockstep://``
  addresses: the same env and model over v1 (gRPC) and v2 (lockstep) leave
  the same traces, bit for bit, across every space kind and v1 dtype, a
  vector env with NEXT_STEP autoreset, seeds and chunking.
- A served model (``RemoteModel``) runs a lockstep env as it runs a gRPC one.
- :class:`rlmesh.v2.GymView` drives a v2-native robot served in lockstep; a
  lockstep robot needs no rates or kinds.
"""

from __future__ import annotations

import hashlib
import time
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
    v1 = _traces(CatalogEnv, "127.0.0.1:0", **run)
    lockstep = _traces(CatalogEnv, "lockstep://127.0.0.1:0/catalog", **run)
    assert v1[4] == 4 and v1[1] and v1[2]
    assert lockstep == v1


def test_a_next_step_vector_env_runs_unchanged_on_lockstep() -> None:
    def make(log: list[str]) -> Any:
        return gym.vector.SyncVectorEnv(
            [lambda: CatalogEnv(log) for _ in range(3)],
            autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
        )

    run = {"episodes": 7}
    v1 = _traces(make, "127.0.0.1:0", **run)
    lockstep = _traces(make, "lockstep://127.0.0.1:0/vector", **run)
    assert v1[4] == 7
    assert lockstep == v1


def test_a_served_model_runs_a_lockstep_env_as_it_runs_a_grpc_one() -> None:
    import socket
    import threading

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    model_address = f"127.0.0.1:{sock.getsockname()[1]}"
    sock.close()
    model = rlmesh.numpy.Model(Recorder())
    threading.Thread(target=lambda: model.serve(model_address), daemon=True).start()

    def trace(address: str) -> tuple[Any, ...]:
        log: list[str] = []
        server = rlmesh.EnvServer(CatalogEnv(log), address)
        server.start()
        hooks = Hooks()
        try:
            for _ in range(100):
                try:
                    rlmesh.run(
                        rlmesh.RemoteModel(model_address),
                        server.address,
                        seeds=[5, 6],
                        hooks=hooks,
                        workflow_edition="2026.06",
                    )
                    break
                except ConnectionError:  # the model server is still starting
                    time.sleep(0.05)
        finally:
            server.shutdown()
        # Steps' info carries run-minted episode ids; the env's log and the
        # episodes say what ran.
        return log, len(hooks.steps), hooks.episodes

    grpc = trace("127.0.0.1:0")
    assert len(grpc[2]) == 2
    assert trace("lockstep://127.0.0.1:0/served") == grpc


ARM = {
    "arm": {
        "kind": "joints",
        "count": 2,
        "fields": ["pos"],
        "hz": 100,
        "command": "position",
    }
}


def test_the_gym_view_drives_a_lockstep_robot() -> None:
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
        env = v2.GymView(host)
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


def test_a_gym_view_step_is_not_delayed_by_the_signal_poll() -> None:
    # Blocked calls poll for Ctrl-C on a timer. Its first tick used to be
    # awaited before the call started, and it waited for the timer's next
    # millisecond, so every Gym step took over 1 ms in-process.
    import statistics
    import time

    with v2.Host() as host:
        robot = v2.Robot("arm1", ARM, session=host, clock="lockstep")
        q = np.zeros(2, np.float32)

        def step(commands: dict[str, Any]) -> Any:
            robot.publish("arm", pos=q)
            return 0.0, False, False, {}

        def reset(seed: int | None, options: Any) -> None:
            robot.publish("arm", pos=q)

        robot.serve_lockstep(step, reset, background=True)
        env = v2.GymView(host)
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


def test_the_gym_view_keys_a_policy_s_entries_and_plays_its_chunks() -> None:
    policy = {
        "hz": 25,
        "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"]}},
        "actions": {
            "arm": {
                "kind": "joints",
                "at": "arm",
                "command": "position",
                "hz": 100,
                "horizon": 4,
            }
        },
    }
    with v2.Host() as host:
        robot = v2.Robot("arm1", ARM, session=host, clock="lockstep")
        q = np.zeros(2, np.float32)
        calls = []

        def step(commands: dict[str, Any]) -> None:
            nonlocal q
            q = commands["arm"].copy()
            calls.append(q.tolist())
            robot.publish("arm", pos=q)

        def reset(seed: int | None, options: Any) -> None:
            robot.publish("arm", pos=q)

        robot.serve_lockstep(step, reset, background=True)
        env = v2.GymView(host, policy, robot="arm1")
        try:
            assert env.table is not None and "arm" in env.table
            assert cast("Any", env.action_space)["arm"].shape == (4, 2)
            env.reset()
            chunk = np.arange(8, dtype=np.float32).reshape(4, 2)
            obs, _, _, _, info = env.step({"arm": chunk})
            assert info["steps"] == 4
            assert calls == chunk.tolist(), "one lockstep step per action"
            assert obs["q"].tolist() == [6.0, 7.0]
        finally:
            env.close()
            robot.close()


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
        env = v2.GymView(host)
        try:
            _, info = env.reset(seed=1, options={"first": np.arange(2)})
            assert info["options"] == {"first": [0, 1]}
            obs, reward, _, _, info = env.step(np.int32(4))  # the one action entry
            assert obs["board"].tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 0]
            assert (reward, info["filled"]) == (1 / 3, 1)  # f64, not f32
        finally:
            env.close()
            robot.close()


def test_the_gym_view_sends_a_gym_env_to_remote_env() -> None:
    server = rlmesh.EnvServer(CatalogEnv(), "lockstep://127.0.0.1:0/gym")
    server.start()
    try:
        session, robot = server.address[len("lockstep://") :].rsplit("/", 1)
        with pytest.raises(v2.SessionException, match="RemoteEnv"):
            v2.GymView(session, robot=robot)
        remote = rlmesh.numpy.RemoteEnv(server.address)
        assert remote.observation_space["mode"] == catalog_spaces()[0]["mode"]
        remote.close()
    finally:
        server.shutdown()
