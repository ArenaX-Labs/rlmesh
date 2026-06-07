from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

import pytest


class TinyEnv:
    def __init__(self) -> None:
        from rlmesh import spaces

        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(2)
        self.close_calls = 0
        self.step_count = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ):
        self.step_count = 0
        return 0, {"seed": seed, "options": options}

    def step(self, action: object):
        self.step_count += 1
        return 1, 1.0, True, False, {"action": action}

    def close(self) -> None:
        self.close_calls += 1


class TinyVectorEnv:
    def __init__(self) -> None:
        from rlmesh import spaces

        self.num_envs = 2
        self.single_observation_space = spaces.Discrete(2)
        self.single_action_space = spaces.Discrete(2)
        self.close_calls = 0

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, object] | None = None,
    ):
        return [0, 0], {"seed": seed, "options": options}

    def step(self, actions: object):
        return [1, 1], [1.0, 1.0], [True, True], [False, False], {"actions": actions}

    def close(self) -> None:
        self.close_calls += 1


class TinyBoxVectorEnv:
    def __init__(self) -> None:
        from rlmesh import spaces

        self.num_envs = 2
        self.single_observation_space = spaces.Box(
            -10.0,
            10.0,
            shape=(4,),
            dtype="float32",
        )
        self.single_action_space = spaces.Discrete(2)
        self.close_calls = 0
        self.last_actions_shape: tuple[int, ...] | None = None

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, object] | None = None,
    ):
        import numpy as np

        return np.zeros((2, 4), dtype=np.float32), {"seed": seed, "options": options}

    def step(self, actions: object):
        import numpy as np

        action_array = np.asarray(actions)
        self.last_actions_shape = action_array.shape
        return (
            np.ones((2, 4), dtype=np.float32),
            np.asarray([1.0, 2.0], dtype=np.float64),
            np.asarray([False, True], dtype=np.bool_),
            np.asarray([False, False], dtype=np.bool_),
            {"actions": action_array},
        )

    def close(self) -> None:
        self.close_calls += 1


class TinyLegacyGymEnv:
    def __init__(self, *, time_limit: bool = False) -> None:
        from rlmesh import spaces

        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(2)
        self.close_calls = 0
        self.time_limit = time_limit

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ):
        return 0

    def step(self, action: object):
        info = {"action": action}
        if self.time_limit:
            info["TimeLimit.truncated"] = True
        return 1, 1.0, True, info

    def close(self) -> None:
        self.close_calls += 1


class TinyLegacyGymVectorEnv:
    def __init__(self) -> None:
        from rlmesh import spaces

        self.num_envs = 2
        self.single_observation_space = spaces.Discrete(2)
        self.single_action_space = spaces.Discrete(2)
        self.close_calls = 0

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, object] | None = None,
    ):
        return [0, 0]

    def step(self, actions: object):
        return (
            [1, 1],
            [1.0, 1.0],
            [True, True],
            {"TimeLimit.truncated": [True, False], "actions": actions},
        )

    def close(self) -> None:
        self.close_calls += 1


RemoteT = TypeVar("RemoteT")


def connect_with_retry(factory: Callable[[str], RemoteT], address: str) -> RemoteT:
    deadline = time.monotonic() + 3.0
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            return factory(address)
        except BaseException as exc:
            last_error = exc
            time.sleep(0.05)
    raise AssertionError(f"failed to connect to {address}") from last_error


def env_server(env: object, options: object | None = None):
    import rlmesh

    try:
        return rlmesh.EnvServer(env, host="127.0.0.1", port=0, options=options)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise


def test_remote_close_detaches_without_stopping_endpoint() -> None:
    import rlmesh

    env = TinyEnv()
    server = env_server(env)
    server.start()
    try:
        remote = connect_with_retry(rlmesh.RemoteEnv, server.address())
        assert remote.shutdown("default remote shutdown is disabled") is False
        remote.close()

        assert env.close_calls == 0

        second = connect_with_retry(rlmesh.RemoteEnv, server.address())
        try:
            observation, info = second.reset(seed=123)
            assert observation == 0
            assert info["seed"] == 123
        finally:
            second.close()
    finally:
        server.shutdown()

    assert env.close_calls == 1


def test_remote_shutdown_requires_explicit_allow() -> None:
    import rlmesh

    env = TinyEnv()
    server = env_server(
        env,
        rlmesh.ServeOptions(allow_remote_shutdown=True),
    )
    server.start()
    remote = connect_with_retry(rlmesh.RemoteEnv, server.address())
    try:
        assert remote.shutdown("test shutdown") is True
    finally:
        server.shutdown()

    assert env.close_calls == 1


def test_env_server_shutdown_is_idempotent_and_closes_env() -> None:
    env = TinyEnv()
    server = env_server(env)
    server.start()
    server.shutdown()
    server.shutdown()

    assert env.close_calls == 1


def test_env_server_shutdown_before_start_closes_env_once() -> None:
    env = TinyEnv()
    server = env_server(env)
    server.shutdown()
    server.shutdown()

    assert env.close_calls == 1


def test_remote_vector_close_detaches_without_stopping_endpoint() -> None:
    import rlmesh

    env = TinyVectorEnv()
    server = env_server(env)
    server.start()
    try:
        remote = connect_with_retry(rlmesh.RemoteVectorEnv, server.address())
        assert remote.shutdown("default remote shutdown is disabled") is False
        remote.close()

        assert env.close_calls == 0

        second = connect_with_retry(rlmesh.RemoteVectorEnv, server.address())
        try:
            _observations, info = second.reset(seed=[1, 2])
            assert info["seed"] == [1, 2]
        finally:
            second.close()
    finally:
        server.shutdown()

    assert env.close_calls == 1


def test_remote_vector_shutdown_requires_explicit_allow() -> None:
    import rlmesh

    env = TinyVectorEnv()
    server = env_server(
        env,
        rlmesh.ServeOptions(allow_remote_shutdown=True),
    )
    server.start()
    remote = connect_with_retry(rlmesh.RemoteVectorEnv, server.address())
    try:
        assert remote.shutdown("test vector shutdown") is True
    finally:
        server.shutdown()

    assert env.close_calls == 1


def test_numpy_remote_vector_step_accepts_ndarray_action_batch() -> None:
    np = pytest.importorskip("numpy")
    from rlmesh import numpy as rlmesh_numpy

    env = TinyBoxVectorEnv()
    server = env_server(env)
    server.start()
    try:
        remote = connect_with_retry(rlmesh_numpy.RemoteVectorEnv, server.address())
        try:
            observations, info = remote.reset(seed=[1, 2])
            assert observations.shape == (2, 4)
            assert info["seed"] == [1, 2]

            actions = np.zeros(2, dtype=np.int64)
            observations, rewards, terminated, truncated, info = remote.step(actions)

            assert observations.shape == (2, 4)
            np.testing.assert_array_equal(rewards, np.asarray([1.0, 2.0]))
            np.testing.assert_array_equal(terminated, np.asarray([False, True]))
            np.testing.assert_array_equal(truncated, np.asarray([False, False]))
            assert info["actions"] == [0, 0]
            assert env.last_actions_shape == (2,)
        finally:
            remote.close()
    finally:
        server.shutdown()


def test_legacy_gym_single_reset_and_step_shapes_are_normalized() -> None:
    import rlmesh

    env = TinyLegacyGymEnv(time_limit=True)
    server = env_server(env)
    server.start()
    try:
        remote = connect_with_retry(rlmesh.RemoteEnv, server.address())
        try:
            observation, info = remote.reset(seed=123)
            assert observation == 0
            assert len(info["episode_ids"]) == 1

            observation, reward, terminated, truncated, info = remote.step(0)
            assert observation == 1
            assert reward == 1.0
            assert terminated is False
            assert truncated is True
            assert info["TimeLimit.truncated"] is True
        finally:
            remote.close()
    finally:
        server.shutdown()


def test_legacy_gym_single_done_without_time_limit_is_terminated() -> None:
    import rlmesh

    env = TinyLegacyGymEnv(time_limit=False)
    server = env_server(env)
    server.start()
    try:
        remote = connect_with_retry(rlmesh.RemoteEnv, server.address())
        try:
            _observation, _info = remote.reset()
            _observation, _reward, terminated, truncated, _info = remote.step(0)
            assert terminated is True
            assert truncated is False
        finally:
            remote.close()
    finally:
        server.shutdown()


def test_legacy_gym_vector_reset_and_step_shapes_are_normalized() -> None:
    import numpy as np
    import rlmesh
    from rlmesh import numpy as rlmesh_numpy

    env = TinyLegacyGymVectorEnv()
    server = env_server(env)
    server.start()
    try:
        remote = connect_with_retry(rlmesh.RemoteVectorEnv, server.address())
        try:
            observations, info = remote.reset(seed=[1, 2])
            assert observations == [0, 0]
            assert len(info["episode_ids"]) == 2

            observations, rewards, terminated, truncated, info = remote.step([0, 1])
            assert observations == [1, 1]
            np.testing.assert_array_equal(
                rlmesh_numpy.asarray(rewards),
                np.asarray([1.0, 1.0], dtype=np.float64),
            )
            np.testing.assert_array_equal(
                rlmesh_numpy.asarray(terminated),
                np.asarray([False, True], dtype=np.bool_),
            )
            np.testing.assert_array_equal(
                rlmesh_numpy.asarray(truncated),
                np.asarray([True, False], dtype=np.bool_),
            )
            assert info["TimeLimit.truncated"] == [True, False]
        finally:
            remote.close()
    finally:
        server.shutdown()


def test_model_run_close_env_requests_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    import rlmesh

    env = TinyEnv()
    server = env_server(
        env,
        rlmesh.ServeOptions(allow_remote_shutdown=True),
    )
    server.start()
    remote = connect_with_retry(rlmesh.RemoteEnv, server.address())
    shutdown_reasons: list[str] = []
    original_shutdown = remote.shutdown

    def shutdown(reason: str = "owner shutdown") -> bool:
        shutdown_reasons.append(reason)
        return original_shutdown(reason)

    monkeypatch.setattr(remote, "shutdown", shutdown)

    try:
        model = rlmesh.Model(lambda _observation: 0)
        model.run(remote, max_episodes=1, close_env=True)
    finally:
        server.shutdown()

    assert shutdown_reasons == ["local model run complete"]


def test_model_lifecycle_callbacks_are_zero_argument() -> None:
    import rlmesh

    env = TinyEnv()
    server = env_server(env, rlmesh.ServeOptions(allow_remote_shutdown=True))
    server.start()
    calls: list[str] = []

    def on_reset() -> None:
        calls.append("reset")

    def on_episode_end() -> None:
        calls.append("episode_end")

    def on_close() -> None:
        calls.append("close")

    try:
        remote = connect_with_retry(rlmesh.RemoteEnv, server.address())
        model = rlmesh.Model(
            lambda _observation: 0,
            on_reset=on_reset,
            on_episode_end=on_episode_end,
            on_close=on_close,
        )
        model.run(remote, max_episodes=1, close_env=True)
    finally:
        server.shutdown()

    assert calls == ["reset", "episode_end", "close"]
