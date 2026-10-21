"""Gymnasium conformance: a served env behaves as the env itself.

Every test runs plain gymnasium (the reference) and the RemoteEnv (or
RemoteVectorEnv) a Gym user connects to the env served on a v2 session, on
the same seeds and actions, and compares types, dtypes, shapes and bits.

Accepted differences: ``reward`` is a Python ``float`` (a vector env's
rewards are ``float64``); a ``Discrete`` observation is the space's NumPy
integer; tuples in ``info`` arrive as lists; an env's exception arrives as an
``EnvironmentException`` that is also its builtin class, its message led by
the call that failed (``reset failed: ...``); ``close()`` detaches, and the
server's ``shutdown()`` closes the served env.

Refused, at once and by name: Graph, Sequence, OneOf, a shape-``()`` Box, and
``info`` values that are neither JSON-like nor arrays (a ``TypeError``), or
ints beyond int64 (a ``ValueError``), each naming its key path.
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

ADDRESS = "lockstep://127.0.0.1:0/gym"
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
        self.options: list[Any] = []
        self._t = 0
        self._len = 3

    def reset(self, *, seed: int | None = None, options: Any = None) -> Any:
        super().reset(seed=seed)
        if seed is not None:
            self.observation_space.seed(seed)
        self.options.append(options)
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


def make(env_id: str, **kwargs: Any) -> Any:
    """``gym.make`` with metadata of its own: gymnasium's ``SyncVectorEnv``
    writes its ``autoreset_mode`` into the env class's shared metadata, which
    would leak one test's vector mode into the next test's single env."""
    env = gym.make(env_id, **kwargs)
    unwrapped: Any = env.unwrapped
    unwrapped.metadata = {
        k: v for k, v in unwrapped.metadata.items() if k != "autoreset_mode"
    }
    return env


@contextmanager
def served(env: Any) -> Iterator[Any]:
    """``env`` behind an EnvServer, as the RemoteEnv (or RemoteVectorEnv) a
    Gym user connects."""
    server = rlmesh.EnvServer(env, ADDRESS, options=EDITION)
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
    if isinstance(a, np.ndarray) and a.dtype == object:
        if b.dtype != object or a.shape != b.shape:
            return f"{path}: {a!r} vs {b!r}"
        return difference(list(a.ravel()), list(b.ravel()), scalars=scalars, path=path)
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
    make: Callable[[], Any], *, scalars: bool = False, **run: Any
) -> None:
    reference = make()
    actions = sampled(reference.action_space)
    want = rollout(reference, actions, **run)
    with served(make()) as remote:
        assert remote.observation_space == reference.observation_space
        assert remote.action_space == reference.action_space
        got = rollout(remote, actions, **run)
    assert difference(want, got, scalars=scalars) is None


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
    "box_i16": spaces.Box(-300, 300, (3,), np.int16),
    "box_i32": spaces.Box(-(2**20), 2**20, (3,), np.int32),
    "box_u32": spaces.Box(0, 2**30, (2,), np.uint32),
    "box_bool": spaces.Box(0, 1, (3,), np.bool_),
    "discrete": spaces.Discrete(4),
    "discrete_start": spaces.Discrete(5, start=-2),
    "multi_discrete": spaces.MultiDiscrete([3, 5, 2]),
    "multi_discrete_2d": spaces.MultiDiscrete(np.array([[2, 3], [4, 5]])),
    "multi_discrete_start": spaces.MultiDiscrete([3, 4], start=[-1, 5]),
    "multi_binary": spaces.MultiBinary(4),
    "multi_binary_2d": spaces.MultiBinary([2, 3]),
    "dict_flat": spaces.Dict(
        {"pos": spaces.Box(-1, 1, (2,), np.float32), "gripper": spaces.Discrete(2)}
    ),
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
    "box_f64_inf",
    "box_i8",
    "box_i32",
    "discrete",
    "discrete_start",
    "multi_discrete_2d",
    "multi_discrete_start",
    "multi_binary",
    "multi_binary_2d",
    "dict_nested",
    "tuple",
    "text",
]


@pytest.mark.parametrize("name", sorted(SPACES))
def test_an_observation_space_and_its_values_round_trip_bit_exact(name: str) -> None:
    assert_matches(lambda: SpaceEnv(copy.deepcopy(SPACES[name])))


@pytest.mark.parametrize("name", sorted(ACTION_SPACES))
def test_an_action_space_reaches_the_env_bit_exact(name: str) -> None:
    assert_matches(lambda: SpaceEnv(action_space=copy.deepcopy(SPACES[name])))


def test_info_round_trips_as_the_env_produced_it() -> None:
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

    assert_matches(lambda: SpaceEnv(info=info))


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
def test_a_gymnasium_env_steps_as_it_does_unserved(env_id: str, steps: int) -> None:
    if env_id.endswith("-v5"):
        pytest.importorskip("mujoco")
    assert_matches(lambda: make(env_id), scalars=True, seeds=(0, 1), steps=steps)


def test_the_remote_env_passes_gymnasiums_env_checker() -> None:
    from gymnasium.utils.env_checker import check_env

    def warned(env: Any) -> set[str]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_env(env, skip_render_check=True)
        return {str(w.message) for w in caught}

    expected = warned(make("CartPole-v1").unwrapped)
    with served(make("CartPole-v1").unwrapped) as remote:
        assert warned(remote) == expected


def test_the_remote_env_is_a_gymnasium_env_with_the_envs_attributes() -> None:
    reference = make("CartPole-v1", render_mode="rgb_array")
    with served(make("CartPole-v1", render_mode="rgb_array")) as remote:
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
def test_a_client_side_wrapper_behaves_as_on_the_env_itself(wrapper: str) -> None:
    wrap = WRAPPERS[wrapper]
    reference = wrap(make("CartPole-v1"))
    actions = sampled(reference.action_space)
    want = rollout(reference, actions, seeds=(0, 1), steps=40)
    with served(make("CartPole-v1")) as remote:
        got = rollout(wrap(remote), actions, seeds=(0, 1), steps=40)
    for trace in (want, got):
        for transition in trace:
            transition[-1].get("episode", {}).pop("t", None)  # wall-clock time
    assert difference(want, got, scalars=True) is None


def test_a_dict_observation_flattens_as_on_the_env_itself() -> None:
    space = SPACES["dict_nested"]
    reference = gym.wrappers.FlattenObservation(SpaceEnv(copy.deepcopy(space)))
    actions = sampled(reference.action_space)
    want = rollout(reference, actions)
    with served(SpaceEnv(copy.deepcopy(space))) as remote:
        got = rollout(gym.wrappers.FlattenObservation(remote), actions)
    assert difference(want, got) is None


def test_render_returns_the_envs_frame() -> None:
    reference = SpaceEnv()
    reference.reset(seed=1)
    reference.step(reference.action_space.sample())
    with served(SpaceEnv()) as remote:
        remote.reset(seed=1)
        remote.step(remote.action_space.sample())
        assert difference(reference.render(), remote.render()) is None


def test_actions_are_cast_to_the_space_dtype_and_refused_when_the_kind_differs() -> (
    None
):
    box, discrete = SpaceEnv(), SpaceEnv(action_space=spaces.Discrete(3))
    with served(box) as remote:
        remote.reset(seed=0)
        remote.step(np.array([0.1, 0.2, 0.3]))  # float64
        remote.step([0.5, -0.5, 1.0])
        with pytest.raises(TypeError, match="shape"):
            remote.step(np.zeros(4))
    assert [a.dtype for a in box.actions] == [np.float32, np.float32]
    assert box.actions[1].tolist() == [0.5, -0.5, 1.0]
    with served(discrete) as remote:
        remote.reset(seed=0)
        remote.step(np.int32(2))
        remote.step(1)
        with pytest.raises(TypeError, match="float64"):
            remote.step(1.0)
    assert discrete.actions == [2, 1]


@pytest.mark.parametrize(
    ("where", "error"),
    [("reset", ValueError), ("step", RuntimeError)],
)
def test_an_env_error_raises_at_once_with_its_type_and_message(
    where: str, error: type[Exception]
) -> None:
    env = SpaceEnv(raise_in=where)
    with served(env) as remote:
        started = time.perf_counter()
        with pytest.raises(error, match=f"boom in {where}") as raised:
            remote.reset(seed=0)
            remote.step(remote.action_space.sample())
        assert isinstance(raised.value, rlmesh.EnvironmentException)
        assert time.perf_counter() - started < 2
        env.raise_in = None
        remote.reset(seed=1)


class Unencodable:
    pass


REFUSED = {
    "graph": (
        lambda: SpaceEnv(
            spaces.Graph(spaces.Box(-1, 1, (2,), np.float32), spaces.Discrete(2))
        ),
        "Graph",
    ),
    "sequence": (lambda: SpaceEnv(spaces.Sequence(spaces.Discrete(3))), "Sequence"),
    "one_of": (
        lambda: SpaceEnv(spaces.OneOf((spaces.Discrete(2), spaces.Discrete(3)))),
        "OneOf",
    ),
    "scalar_box": (lambda: SpaceEnv(spaces.Box(-1, 1, (), np.float32)), "shape"),
    "info_object": (lambda: SpaceEnv(info=lambda t: {"thing": Unencodable()}), "thing"),
    "info_big_int": (lambda: SpaceEnv(info=lambda t: {"big": 2**70}), "big"),
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_what_the_wire_cannot_carry_is_refused_at_once_by_name(case: str) -> None:
    build, name = REFUSED[case]
    started = time.perf_counter()
    with pytest.raises((TypeError, ValueError), match=name):
        with served(build()) as remote:
            remote.reset(seed=0)
            remote.step(remote.action_space.sample())
    assert time.perf_counter() - started < 2


def test_reset_options_reach_the_env_as_given() -> None:
    options = {
        "goal": np.arange(3, dtype=np.int16),
        "scale": np.float32(0.5),
        "mode": "hard",
        "nested": {"weights": np.ones((2, 2))},
    }
    env = SpaceEnv()
    with served(env) as remote:
        remote.reset(seed=0, options=options)
        remote.reset()
    assert difference([options, None], env.options) is None


def test_an_unseeded_reset_continues_the_envs_random_stream() -> None:
    assert_matches(lambda: make("CartPole-v1"), scalars=True, seeds=(5, None, None))


class BothDone(SpaceEnv):
    def step(self, action: Any) -> Any:
        obs, reward, _, _, info = super().step(action)
        return obs, reward, True, True, info


def test_a_step_both_terminated_and_truncated_arrives_so() -> None:
    assert_matches(BothDone)


class Strided(gym.ObservationWrapper):  # type: ignore[misc]
    """Observations as non-contiguous views (a reversed, strided slice)."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(0, 255, (4, 3), np.uint8)

    def observation(self, observation: Any) -> Any:
        frame = np.arange(48, dtype=np.uint8).reshape(8, 6)
        return (frame + np.uint8(abs(observation[0]) * 100))[::-2, ::2]


def test_non_contiguous_observations_and_actions_arrive_by_value() -> None:
    def build() -> Any:
        return Strided(SpaceEnv(action_space=spaces.Box(-1, 1, (3,), np.float32)))

    reference = build()
    strided = np.linspace(-1, 1, 9, dtype=np.float32)[::3]
    want = rollout(reference, [strided])
    with served(build()) as remote:
        got = rollout(remote, [strided])
    assert difference(want, got) is None
    assert not strided.flags.c_contiguous


SERVED_WRAPPED = {
    "RecordEpisodeStatistics": lambda: gym.wrappers.RecordEpisodeStatistics(
        make("CartPole-v1")
    ),
    "FrameStackObservation": lambda: gym.wrappers.FrameStackObservation(
        make("CartPole-v1"), 3
    ),
}


@pytest.mark.parametrize("case", sorted(SERVED_WRAPPED))
def test_an_env_served_already_wrapped_steps_as_it_does_unserved(case: str) -> None:
    build = SERVED_WRAPPED[case]
    reference = build()
    actions = sampled(reference.action_space)
    want = rollout(reference, actions, seeds=(0, 1), steps=40)
    with served(build()) as remote:
        got = rollout(remote, actions, seeds=(0, 1), steps=40)
    for trace in (want, got):
        for transition in trace:
            transition[-1].get("episode", {}).pop("t", None)  # wall-clock time
    assert difference(want, got, scalars=True) is None


def test_the_env_s_spec_is_forwarded_and_close_detaches() -> None:
    reference = make("CartPole-v1")
    env = make("CartPole-v1")
    closed: list[bool] = []
    env.unwrapped.close = lambda: closed.append(True)
    server = rlmesh.EnvServer(env, ADDRESS, options=EDITION)
    server.start()
    try:
        remote = rlmesh.numpy.RemoteEnv(server.address, workflow_edition="2026.06")
        assert remote.spec is not None and reference.spec is not None
        assert remote.spec.id == reference.spec.id == "CartPole-v1"
        assert remote.spec.max_episode_steps == reference.spec.max_episode_steps
        remote.close()
        assert not closed, "close() detaches; the server owns the env"
        again = rlmesh.numpy.RemoteEnv(server.address, workflow_edition="2026.06")
        again.reset(seed=0)
        again.close()
    finally:
        server.shutdown()
    assert closed, "shutdown() closes the served env"


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_every_backend_s_remote_env_is_a_gymnasium_env(backend: str) -> None:
    module = pytest.importorskip(f"rlmesh.{backend}")
    pytest.importorskip(backend)
    reference = make("CartPole-v1", render_mode="rgb_array")
    server = rlmesh.EnvServer(
        make("CartPole-v1", render_mode="rgb_array"), ADDRESS, options=EDITION
    )
    server.start()
    try:
        remote = module.RemoteEnv(server.address, workflow_edition="2026.06")
        try:
            assert isinstance(remote, gym.Env)
            assert remote.unwrapped is remote
            assert remote.metadata == reference.metadata
            assert remote.render_mode == "rgb_array"
            assert remote.observation_space == reference.observation_space
            assert remote.action_space == reference.action_space
            wrapped = gym.wrappers.RecordEpisodeStatistics(remote)
            obs, _ = wrapped.reset(seed=0)
            want, _ = reference.reset(seed=0)
            assert np.asarray(obs).tobytes() == want.tobytes()
            _, reward, *_ = wrapped.step(remote.action_space.sample())
            assert reward == 1.0
        finally:
            remote.close()
    finally:
        server.shutdown()


AR = gym.vector.AutoresetMode
VECTORS = {
    "sync_next_step": lambda: gym.vector.SyncVectorEnv(
        [lambda: make("CartPole-v1")] * 3, autoreset_mode=AR.NEXT_STEP
    ),
    "sync_same_step": lambda: gym.vector.SyncVectorEnv(
        [lambda: make("CartPole-v1")] * 3, autoreset_mode=AR.SAME_STEP
    ),
    "sync_disabled": lambda: gym.vector.SyncVectorEnv(
        [lambda: make("CartPole-v1")] * 3, autoreset_mode=AR.DISABLED
    ),
    "async": lambda: gym.vector.AsyncVectorEnv(
        [lambda: make("CartPole-v1")] * 2, context="spawn"
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
def test_a_vector_env_steps_as_it_does_unserved(case: str) -> None:
    reference = VECTORS[case]()
    actions = sampled(reference.action_space)
    steps = VECTOR_STEPS.get(case, 20)
    want = vector_rollout(reference, actions, steps)
    with served(VECTORS[case]()) as remote:
        assert isinstance(remote, gym.vector.VectorEnv)
        assert remote.single_observation_space == reference.single_observation_space
        assert remote.observation_space == reference.observation_space
        assert remote.action_space == reference.action_space
        got = vector_rollout(remote, actions, steps)
    if case == "vector_entry_point":
        # CartPole's own vector env returns float32 rewards; the wire's are float64.
        want[1:] = [(obs, r.astype(np.float64), *rest) for obs, r, *rest in want[1:]]
    assert difference(want, got) is None


@pytest.mark.parametrize("mode", ["NEXT_STEP", "SAME_STEP"])
def test_a_vector_env_s_info_arrays_arrive_as_arrays_with_their_dtypes(
    mode: str,
) -> None:
    def build() -> Any:
        return gym.wrappers.vector.RecordEpisodeStatistics(
            gym.vector.SyncVectorEnv(
                [lambda: make("CartPole-v1")] * 3, autoreset_mode=AR[mode]
            )
        )

    reference = build()
    actions = sampled(reference.action_space)
    want = vector_rollout(reference, actions, 40)
    with served(build()) as remote:
        got = vector_rollout(remote, actions, 40)
    for trace in (want, got):
        for transition in trace:
            transition[-1].get("episode", {}).pop("t", None)  # wall-clock time
    assert any("episode" in t[-1] for t in want)
    assert difference([t[-1] for t in want], [t[-1] for t in got]) is None


def test_remote_envs_vectorise_on_the_client_side() -> None:
    reference = gym.vector.SyncVectorEnv([lambda: make("CartPole-v1")] * 2)
    actions = sampled(reference.action_space)
    want = vector_rollout(reference, actions, 40)
    with (
        served(make("CartPole-v1")) as a,
        served(make("CartPole-v1")) as b,
    ):
        remotes = iter((a, b))
        got = vector_rollout(
            gym.vector.SyncVectorEnv([lambda: next(remotes)] * 2), actions, 40
        )
    assert difference(want, got, scalars=True) is None


SERVE_IN_A_CHILD = """
import sys, gymnasium as gym, numpy as np, rlmesh

class Frames(gym.Env):
    def __init__(self, shape):
        self.observation_space = gym.spaces.Box(0, 255, shape, np.uint8)
        self.action_space = gym.spaces.Discrete(2)
        self.frame = np.random.default_rng(0).integers(0, 255, shape, np.uint8)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.frame, {}
    def step(self, action):
        return self.frame, 0.0, False, False, {}

env = gym.make("CartPole-v1") if sys.argv[1] == "cartpole" else Frames(eval(sys.argv[1]))
server = rlmesh.EnvServer(env, "lockstep://127.0.0.1:0/gym")
server.start()
print(server.address, flush=True)
sys.stdin.read()
server.shutdown()
"""


@contextmanager
def served_in_a_child(env: str) -> Iterator[Any]:
    import subprocess
    import sys

    child = subprocess.Popen(
        [sys.executable, "-c", SERVE_IN_A_CHILD, env],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None and child.stdin is not None
    try:
        remote = rlmesh.numpy.RemoteEnv(child.stdout.readline().strip())
        try:
            yield remote
        finally:
            remote.close()
    finally:
        child.stdin.close()
        child.wait(timeout=10)


def median_step(env: str, steps: int) -> float:
    with served_in_a_child(env) as remote:
        remote.reset(seed=0)
        latencies = []
        for n in range(steps):
            started = time.perf_counter()
            _, _, terminated, truncated, _ = remote.step(n % 2)
            latencies.append(time.perf_counter() - started)
            if terminated or truncated:
                remote.reset()
    return statistics.median(latencies)


PERF = {  # env, steps, median step budget in seconds
    "cartpole": ("cartpole", 2000, 0.0005),
    "image_224": ("(224, 224, 3)", 500, 0.001),
    "image_1080p": ("(1080, 1920, 3)", 100, 0.02),
}


@pytest.mark.parametrize("case", sorted(PERF))
def test_a_two_process_step_is_fast(case: str) -> None:
    env, steps, budget = PERF[case]
    assert median_step(env, steps) < budget
