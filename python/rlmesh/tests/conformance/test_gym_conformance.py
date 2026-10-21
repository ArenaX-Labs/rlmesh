"""Gymnasium conformance: a served env behaves as the env itself.

Every test runs plain gymnasium (the reference) and ``rlmesh.numpy.RemoteEnv``
(or ``RemoteVectorEnv``) over gRPC (v1) and over a v2 session
(``lockstep://``) on the same seeds and actions, and compares types, dtypes,
shapes and bits.

Documented differences (the wire carries values, not Python types):

- ``reward`` is a Python ``float`` (an env's ``np.float64`` or ``int`` is
  equal in value); a vector env's rewards are ``float64``;
- ``info`` values travel as JSON-like data: NumPy arrays and scalars arrive
  as lists and Python numbers, tuples as lists; a vector env's per-lane
  lists are arrays again;
- a ``Discrete`` observation is the space's NumPy integer even when the env
  returned a Python ``int``; a ``MultiBinary`` action reaches the env as
  ``uint8`` (Gymnasium samples ``int8``; the values are the same);
- ``Text`` with Gymnasium's default charset rebuilds as printable text;
- ``spec`` is ``None``, ``close()`` detaches without closing the served env,
  and an env's exception arrives as ``rlmesh.EnvironmentException`` with
  its message.

Not carried at all (refused): Graph, Sequence, OneOf, a shape-``()`` Box,
``MultiDiscrete(start=...)``, ints beyond int64 in ``info``, ``SAME_STEP``
autoreset, and (on ``lockstep://``) Text actions.
"""

from __future__ import annotations

import copy
import hashlib
import math
import statistics
import time
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, ClassVar

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

import rlmesh  # noqa: E402
import rlmesh.numpy  # noqa: E402

spaces = gym.spaces

TRANSPORTS = {"grpc": "127.0.0.1:0", "lockstep": "lockstep://127.0.0.1:0/gym"}
EDITION = rlmesh.ServeOptions(workflow_edition="2026.06")


def digest(value: Any) -> int:
    """A stable digest of a (nested) value, dtypes included."""
    h = hashlib.sha256()

    def walk(v: Any) -> None:
        if isinstance(v, dict):
            for k in sorted(v):
                h.update(repr(k).encode())
                walk(v[k])
        elif isinstance(v, (tuple, list)):
            for x in v:
                walk(x)
        elif isinstance(v, str):
            h.update(v.encode())
        else:
            a = np.asarray(v)
            h.update(str(a.dtype).encode() + a.tobytes())

    walk(value)
    return int(h.hexdigest()[:8], 16)


class SpaceEnv(gym.Env):  # type: ignore[misc]
    """Samples its observations from its own seeded space; the reward is a
    digest of the action; episodes end, terminated or truncated, at a
    seed-dependent length."""

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        observation_space: Any = None,
        action_space: Any = None,
        *,
        info: Callable[[int], dict[str, Any]] | None = None,
        raise_in: str | None = None,
    ) -> None:
        self.observation_space = observation_space or spaces.Box(
            -1, 1, (3,), np.float32
        )
        self.action_space = action_space or spaces.Box(-1, 1, (3,), np.float32)
        self.render_mode = "rgb_array"
        self.extra_info = info or (lambda t: {})
        self.raise_in = raise_in
        self.actions: list[Any] = []
        self._t = 0
        self._len = 3

    def reset(self, *, seed: int | None = None, options: Any = None) -> Any:
        super().reset(seed=seed)
        if seed is not None:
            self.observation_space.seed(seed)
        if self.raise_in == "reset":
            raise ValueError("boom in reset")
        self._t = 0
        self._len = int(self.np_random.integers(2, 6))
        return self.observation_space.sample(), {"len": self._len, **self.extra_info(0)}

    def step(self, action: Any) -> Any:
        self.actions.append(action)
        self._t += 1
        if self.raise_in == "step":
            raise RuntimeError("boom in step")
        done = self._t >= self._len
        reward = (digest(action) % 1000) / 1000 + self._t / 7
        info = {"t": self._t, **self.extra_info(self._t)}
        return (
            self.observation_space.sample(),
            reward,
            done and self._len % 2 == 0,
            done and self._len != 2,
            info,
        )

    def render(self) -> Any:
        frame = np.zeros((4, 5, 3), np.uint8)
        frame[..., 0] = self._t
        return frame


@contextmanager
def served(env: Any, transport: str) -> Iterator[Any]:
    """``env`` behind an EnvServer, as the RemoteEnv (or RemoteVectorEnv) a
    Gym user connects."""
    server = rlmesh.EnvServer(env, TRANSPORTS[transport], options=EDITION)
    server.start()
    vector = hasattr(env, "num_envs") and env.num_envs > 1
    cls = rlmesh.numpy.RemoteVectorEnv if vector else rlmesh.numpy.RemoteEnv
    remote = cls(server.address, workflow_edition="2026.06")
    try:
        yield remote
    finally:
        remote.close()
        server.shutdown()


def difference(a: Any, b: Any, *, scalars: bool = False, path: str = "") -> str | None:
    """The first difference in type, dtype, shape or bits (``a`` is the
    reference); ``scalars`` lets a Python and a NumPy scalar of equal value
    pass (the documented reward, info and ``Discrete`` differences)."""
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)) or set(a) != set(b):
            return f"{path}: keys {sorted(a) if isinstance(a, dict) else a!r} vs {sorted(b) if isinstance(b, dict) else b!r}"
        for k in a:
            d = difference(a[k], b[k], scalars=scalars, path=f"{path}[{k!r}]")
            if d:
                return d
        return None
    if isinstance(a, (tuple, list)):
        if type(a) is not type(b) or len(a) != len(b):
            return f"{path}: {type(a).__name__} vs {type(b).__name__}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            d = difference(x, y, scalars=scalars, path=f"{path}[{i}]")
            if d:
                return d
        return None
    if (
        scalars
        and np.ndim(a) == 0
        and np.ndim(b) == 0
        and not isinstance(a, (str, bytes))
    ):
        x, y = np.asarray(a).item(), np.asarray(b).item()
        same = x == y or (isinstance(x, float) and math.isnan(x) and math.isnan(y))
        return None if same else f"{path}: {a!r} vs {b!r}"
    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, (np.ndarray, np.generic)):
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            return f"{path}: {a!r} vs {b!r}"
        return None
    if isinstance(a, float) and math.isnan(a) and math.isnan(b):
        return None
    return None if a == b else f"{path}: {a!r} vs {b!r}"


def rollout(
    env: Any, actions: list[Any], seeds: tuple[int, ...] = (3, 11), steps: int = 12
) -> list[Any]:
    trace: list[Any] = []
    i = 0
    for seed in seeds:
        trace.append(env.reset(seed=seed))
        for _ in range(steps):
            transition = env.step(actions[i % len(actions)])
            i += 1
            trace.append(transition)
            if transition[2] or transition[3]:
                break
    return trace


def sampled(space: Any, n: int = 32) -> list[Any]:
    space = copy.deepcopy(space)
    space.seed(123)
    return [space.sample() for _ in range(n)]


def assert_matches(
    make: Callable[[], Any], transport: str, *, scalars: bool = False, **run: Any
) -> None:
    reference = make()
    actions = sampled(reference.action_space)
    want = rollout(reference, actions, **run)
    with served(make(), transport) as remote:
        assert remote.observation_space == reference.observation_space
        assert remote.action_space == reference.action_space
        got = rollout(remote, actions, **run)
    assert difference(want, got, scalars=scalars) is None


@pytest.fixture(params=sorted(TRANSPORTS))
def transport(request: pytest.FixtureRequest) -> str:
    return str(request.param)


SPACES = {
    "box_f32_bounded": spaces.Box(-1.0, 2.0, (3,), np.float32),
    "box_f64_inf": spaces.Box(-np.inf, np.inf, (2, 2), np.float64),
    "box_mixed_bounds": spaces.Box(
        np.array([-1.0, -np.inf], np.float32), np.array([np.inf, 3.0], np.float32)
    ),
    "box_f16": spaces.Box(-1.0, 1.0, (4,), np.float16),
    "box_i8": spaces.Box(-5, 5, (3,), np.int8),
    "box_u8_image": spaces.Box(0, 255, (6, 5, 3), np.uint8),
    "box_u16": spaces.Box(0, 60000, (2,), np.uint16),
    "box_i64": spaces.Box(-(2**40), 2**40, (2,), np.int64),
    "box_u64": spaces.Box(0, 2**40, (2,), np.uint64),
    "box_bool": spaces.Box(0, 1, (3,), np.bool_),
    "discrete": spaces.Discrete(4),
    "discrete_start": spaces.Discrete(5, start=-2),
    "multi_discrete_2d": spaces.MultiDiscrete(np.array([[2, 3], [4, 5]])),
    "multi_binary_2d": spaces.MultiBinary([2, 3]),
    "dict_nested": spaces.Dict(
        {
            "robot": spaces.Dict(
                {"q": spaces.Box(-1, 1, (3,), np.float32), "mode": spaces.Discrete(2)}
            ),
            "scene": spaces.Dict({"img": spaces.Box(0, 255, (4, 4, 3), np.uint8)}),
        }
    ),
    "dict_keys_that_escape_alike": spaces.Dict(
        {"Pos": spaces.Discrete(3), "pos": spaces.Discrete(5)}
    ),
    "dict_key_named_action": spaces.Dict(
        {"action": spaces.Box(-1, 1, (1,), np.float32)}
    ),
    "tuple": spaces.Tuple((spaces.Discrete(2), spaces.Box(-1, 1, (2,), np.float32))),
    "tuple_nested": spaces.Tuple(
        (
            spaces.Dict({"a": spaces.Discrete(2)}),
            spaces.Tuple((spaces.MultiBinary(2), spaces.Box(-1, 1, (2,), np.float32))),
        )
    ),
    "text": spaces.Text(12, min_length=1, charset="abc xyz"),
}
ACTION_SPACES = [
    "box_f32_bounded",
    "box_i8",
    "discrete_start",
    "multi_discrete_2d",
    "dict_nested",
    "tuple",
]


@pytest.mark.parametrize("name", sorted(SPACES))
def test_an_observation_space_and_its_values_round_trip_bit_exact(
    name: str, transport: str
) -> None:
    assert_matches(lambda: SpaceEnv(copy.deepcopy(SPACES[name])), transport)


@pytest.mark.parametrize("name", ACTION_SPACES)
def test_an_action_space_reaches_the_env_bit_exact(name: str, transport: str) -> None:
    assert_matches(
        lambda: SpaceEnv(action_space=copy.deepcopy(SPACES[name])), transport
    )


def test_info_round_trips_as_the_env_produced_it(transport: str) -> None:
    def info(t: int) -> dict[str, Any]:
        return {
            "nested": {"a": {"b": t, "c": [1, 2.5, "x"]}, "flag": t % 2 == 0},
            "nan": float("nan"),
            "inf": -float("inf"),
            "bytes": b"\x00\xff",
            "$dollar": 1,
            "none": None,
            "third": 1 / 3,
            "is_success": t >= 3,
        }

    assert_matches(lambda: SpaceEnv(info=info), transport)


GYM_ENVS = [
    ("CartPole-v1", 200),
    ("Acrobot-v1", 40),
    ("MountainCarContinuous-v0", 40),
    ("Pendulum-v1", 60),
    ("FrozenLake-v1", 30),
    ("Blackjack-v1", 10),
    ("InvertedPendulum-v5", 60),
    ("HalfCheetah-v5", 40),
]


@pytest.mark.parametrize(("env_id", "steps"), GYM_ENVS)
def test_a_gymnasium_env_steps_as_it_does_unserved(
    env_id: str, steps: int, transport: str
) -> None:
    if env_id.endswith("-v5"):
        pytest.importorskip("mujoco")
    assert_matches(
        lambda: gym.make(env_id), transport, scalars=True, seeds=(0, 1), steps=steps
    )


def test_the_remote_env_passes_gymnasiums_env_checker(transport: str) -> None:
    from gymnasium.utils.env_checker import check_env

    def warned(env: Any) -> set[str]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_env(env, skip_render_check=True)
        return {str(w.message) for w in caught}

    expected = warned(gym.make("CartPole-v1").unwrapped)
    with served(gym.make("CartPole-v1").unwrapped, transport) as remote:
        assert warned(remote) == expected


def test_the_remote_env_is_a_gymnasium_env_with_the_envs_attributes(
    transport: str,
) -> None:
    reference = gym.make("CartPole-v1", render_mode="rgb_array")
    with served(gym.make("CartPole-v1", render_mode="rgb_array"), transport) as remote:
        assert isinstance(remote, gym.Env)
        assert remote.unwrapped is remote
        assert remote.metadata == reference.metadata
        assert remote.render_mode == "rgb_array"
        remote.reset(seed=4)
        assert remote.np_random is not None


WRAPPERS = {
    "TimeLimit": lambda e: gym.wrappers.TimeLimit(e, max_episode_steps=7),
    "FrameStackObservation": lambda e: gym.wrappers.FrameStackObservation(e, 3),
    "RecordEpisodeStatistics": gym.wrappers.RecordEpisodeStatistics,
}


@pytest.mark.parametrize("wrapper", sorted(WRAPPERS))
def test_a_client_side_wrapper_behaves_as_on_the_env_itself(
    wrapper: str, transport: str
) -> None:
    wrap = WRAPPERS[wrapper]
    reference = wrap(gym.make("CartPole-v1"))
    actions = sampled(reference.action_space)
    want = rollout(reference, actions, seeds=(0, 1), steps=40)
    with served(gym.make("CartPole-v1"), transport) as remote:
        got = rollout(wrap(remote), actions, seeds=(0, 1), steps=40)
    for trace in (want, got):
        for transition in trace:
            transition[-1].get("episode", {}).pop("t", None)  # wall-clock time
    assert difference(want, got, scalars=True) is None


def test_a_dict_observation_flattens_as_on_the_env_itself(transport: str) -> None:
    space = SPACES["dict_nested"]
    reference = gym.wrappers.FlattenObservation(SpaceEnv(copy.deepcopy(space)))
    actions = sampled(reference.action_space)
    want = rollout(reference, actions)
    with served(SpaceEnv(copy.deepcopy(space)), transport) as remote:
        got = rollout(gym.wrappers.FlattenObservation(remote), actions)
    assert difference(want, got) is None


def test_render_returns_the_envs_frame(transport: str) -> None:
    reference = SpaceEnv()
    reference.reset(seed=1)
    reference.step(reference.action_space.sample())
    with served(SpaceEnv(), transport) as remote:
        remote.reset(seed=1)
        remote.step(remote.action_space.sample())
        assert difference(reference.render(), remote.render()) is None


def test_actions_are_cast_to_the_space_dtype_and_refused_when_the_kind_differs(
    transport: str,
) -> None:
    box, discrete = SpaceEnv(), SpaceEnv(action_space=spaces.Discrete(3))
    with served(box, transport) as remote:
        remote.reset(seed=0)
        remote.step(np.array([0.1, 0.2, 0.3]))  # float64
        remote.step([0.5, -0.5, 1.0])
        with pytest.raises(TypeError, match="shape"):
            remote.step(np.zeros(4))
    assert [a.dtype for a in box.actions] == [np.float32, np.float32]
    assert box.actions[1].tolist() == [0.5, -0.5, 1.0]
    with served(discrete, transport) as remote:
        remote.reset(seed=0)
        remote.step(np.int32(2))
        remote.step(1)
        with pytest.raises(TypeError, match="float64"):
            remote.step(1.0)
    assert discrete.actions == [2, 1]


@pytest.mark.parametrize("where", ["reset", "step"])
def test_an_env_error_raises_at_once_with_its_message(
    where: str, transport: str
) -> None:
    env = SpaceEnv(raise_in=where)
    with served(env, transport) as remote:
        started = time.perf_counter()
        with pytest.raises(rlmesh.EnvironmentException, match=f"boom in {where}"):
            remote.reset(seed=0)
            remote.step(remote.action_space.sample())
        assert time.perf_counter() - started < 2
        env.raise_in = None
        remote.reset(seed=1)


def test_an_env_lockstep_cannot_carry_is_refused_when_served() -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match="Text"):
        rlmesh.EnvServer(SpaceEnv(action_space=spaces.Text(8)), TRANSPORTS["lockstep"])
    assert time.perf_counter() - started < 2


AR = gym.vector.AutoresetMode
VECTORS = {
    "sync_next_step": lambda: gym.vector.SyncVectorEnv(
        [lambda: gym.make("CartPole-v1")] * 3, autoreset_mode=AR.NEXT_STEP
    ),
    "sync_disabled": lambda: gym.vector.SyncVectorEnv(
        [lambda: gym.make("CartPole-v1")] * 3, autoreset_mode=AR.DISABLED
    ),
    "vector_entry_point": lambda: gym.make_vec(
        "CartPole-v1", num_envs=4, vectorization_mode="vector_entry_point"
    ),
    "sync_dict_obs": lambda: gym.vector.SyncVectorEnv(
        [lambda: SpaceEnv(copy.deepcopy(SPACES["dict_nested"]))] * 2
    ),
}
# Under DISABLED a done lane must not step again: stay inside the first episodes.
VECTOR_STEPS = {"sync_disabled": 8}


def vector_rollout(env: Any, actions: list[Any], steps: int) -> list[Any]:
    trace: list[Any] = [env.reset(seed=7)]
    for k in range(steps):
        trace.append(env.step(actions[k % len(actions)]))
    return trace


@pytest.mark.parametrize("case", sorted(VECTORS))
def test_a_vector_env_steps_as_it_does_unserved(case: str, transport: str) -> None:
    reference = VECTORS[case]()
    actions = sampled(reference.action_space)
    steps = VECTOR_STEPS.get(case, 20)
    want = vector_rollout(reference, actions, steps)
    with served(VECTORS[case](), transport) as remote:
        assert isinstance(remote, gym.vector.VectorEnv)
        assert remote.single_observation_space == reference.single_observation_space
        assert remote.observation_space == reference.observation_space
        assert remote.action_space == reference.action_space
        got = vector_rollout(remote, actions, steps)
    if case == "vector_entry_point":
        # CartPole's own vector env returns float32 rewards; the wire's are float64.
        want[1:] = [(obs, r.astype(np.float64), *rest) for obs, r, *rest in want[1:]]
    assert difference(want, got) is None


def test_remote_envs_vectorise_on_the_client_side(transport: str) -> None:
    reference = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")] * 2)
    actions = sampled(reference.action_space)
    want = vector_rollout(reference, actions, 40)
    with (
        served(gym.make("CartPole-v1"), transport) as a,
        served(gym.make("CartPole-v1"), transport) as b,
    ):
        remotes = iter((a, b))
        got = vector_rollout(
            gym.vector.SyncVectorEnv([lambda: next(remotes)] * 2), actions, 40
        )
    assert difference(want, got, scalars=True) is None


def test_lockstep_steps_at_least_as_fast_as_grpc() -> None:
    def median_step(transport: str) -> float:
        with served(gym.make("CartPole-v1"), transport) as remote:
            remote.reset(seed=0)
            latencies = []
            for n in range(500):
                started = time.perf_counter()
                _, _, terminated, truncated, _ = remote.step(n % 2)
                latencies.append(time.perf_counter() - started)
                if terminated or truncated:
                    remote.reset()
        return statistics.median(latencies)

    assert median_step("lockstep") <= median_step("grpc")
