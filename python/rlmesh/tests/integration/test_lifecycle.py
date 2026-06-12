from __future__ import annotations

import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, TypeVar, cast

import pytest

if TYPE_CHECKING:
    import numpy as np
    from rlmesh import EnvServer, ServeOptions
    from rlmesh.server import EnvLike as ServedEnv

    NumpyArray = np.ndarray[Any, Any]


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


class TinySpecEnv(TinyEnv):
    def __init__(self) -> None:
        super().__init__()
        self.spec = SimpleNamespace(id="TinySpecEnv-v0")


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


class TinyOneVectorEnv:
    def __init__(self) -> None:
        from rlmesh import spaces

        self.num_envs = 1
        self.single_observation_space = spaces.Discrete(2)
        self.single_action_space = spaces.Discrete(2)
        self.close_calls = 0
        self.last_actions_shape: tuple[int, ...] | None = None

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, object] | None = None,
    ):
        return [0], {"seed": seed, "options": options}

    def step(self, actions: object):
        import numpy as np

        action_array = np.asarray(actions)
        self.last_actions_shape = action_array.shape
        return (
            [1],
            np.asarray([1.0], dtype=np.float64),
            np.asarray([False], dtype=np.bool_),
            np.asarray([False], dtype=np.bool_),
            {},
        )

    def close(self) -> None:
        self.close_calls += 1


class TinySpecVectorEnv(TinyVectorEnv):
    def __init__(self) -> None:
        super().__init__()
        self.spec = SimpleNamespace(id="TinySpecVectorEnv-v0")


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
        except Exception as exc:
            last_error = exc
            time.sleep(0.05)
    raise AssertionError(f"failed to connect to {address}") from last_error


def assert_connect_rejected_with_value_error(address: str, pattern: str) -> None:
    import rlmesh

    deadline = time.monotonic() + 3.0
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            with pytest.raises(ValueError, match=pattern):
                rlmesh.RemoteEnv(address)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.05)
    raise AssertionError(
        f"failed to observe RemoteEnv rejection for {address}"
    ) from last_error


def env_server(env: object, options: ServeOptions | None = None) -> EnvServer:
    import rlmesh

    try:
        return rlmesh.EnvServer(
            cast("ServedEnv", env),
            host="127.0.0.1",
            port=0,
            options=options,
        )
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
        remote = connect_with_retry(rlmesh.RemoteEnv, server.address)
        assert remote.shutdown("default remote shutdown is disabled") is False
        remote.close()

        assert env.close_calls == 0

        second = connect_with_retry(rlmesh.RemoteEnv, server.address)
        try:
            observation, info = second.reset(seed=123)
            assert observation == 0
            assert info["seed"] == 123
        finally:
            second.close()
    finally:
        server.shutdown()

    assert env.close_calls == 1


def test_remote_space_properties_load_from_contract() -> None:
    import rlmesh
    from rlmesh import spaces

    env = TinyEnv()
    server = env_server(env)
    server.start()
    try:
        remote = connect_with_retry(rlmesh.RemoteEnv, server.address)
        try:
            assert isinstance(remote.observation_space, spaces.Discrete)
            assert remote.observation_space.n == 2
            assert isinstance(remote.action_space, spaces.Discrete)
            assert remote.action_space.n == 2
        finally:
            remote.close()
    finally:
        server.shutdown()


def test_remote_env_rejects_multi_env_endpoint() -> None:
    env = TinyVectorEnv()
    server = env_server(env)
    server.start()
    try:
        assert_connect_rejected_with_value_error(
            server.address,
            "serves 2 environments",
        )
    finally:
        server.shutdown()


def test_remote_shutdown_requires_explicit_allow() -> None:
    import rlmesh

    env = TinyEnv()
    server = env_server(
        env,
        rlmesh.ServeOptions(allow_remote_shutdown=True),
    )
    server.start()
    remote = connect_with_retry(rlmesh.RemoteEnv, server.address)
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


def test_env_server_exposes_env_contract_before_and_after_lifecycle() -> None:
    from rlmesh.specs import EnvContract

    env = TinySpecEnv()
    server = env_server(env)

    try:
        assert isinstance(server.env_contract, EnvContract)
        assert isinstance(server.spec, EnvContract)
        assert server.env_contract.id == "TinySpecEnv-v0"
        assert server.spec.id == "TinySpecEnv-v0"

        server.start()
        server.shutdown()

        assert server.env_contract.id == "TinySpecEnv-v0"
        assert server.spec.id == "TinySpecEnv-v0"
    finally:
        server.shutdown()

    assert env.close_calls == 1


def test_env_server_vector_contract_reports_num_envs() -> None:
    env = TinySpecVectorEnv()
    server = env_server(env)
    try:
        assert server.spec.id == "TinySpecVectorEnv-v0"
        assert server.env_contract.num_envs == 2
        assert server.spec.num_envs == 2
    finally:
        server.shutdown()

    assert env.close_calls == 1


def test_env_server_wait_timeout_returns_false_while_running() -> None:
    env = TinyEnv()
    server = env_server(env)
    server.start()
    try:
        assert server.wait(0.01) is False
    finally:
        server.shutdown()

    assert env.close_calls == 1


def test_env_server_wait_returns_true_after_remote_shutdown() -> None:
    import rlmesh

    env = TinyEnv()
    server = env_server(env, rlmesh.ServeOptions(allow_remote_shutdown=True))
    server.start()
    try:
        remote = connect_with_retry(rlmesh.RemoteEnv, server.address)
        assert remote.shutdown("test shutdown") is True
        assert server.wait(timeout=3.0) is True
    finally:
        server.shutdown()

    assert env.close_calls == 1


def test_background_server_survives_unrelated_sigint() -> None:
    """Review finding #33: start() must not install a process-wide SIGINT handler.

    A Ctrl+C aimed at an unrelated computation in a notebook/REPL used to trip the
    tokio signal task and silently tear down a backgrounded server. After the fix
    only the blocking serve() path owns signal disposition, so a SIGINT delivered
    while the server runs in the background must leave it serving.
    """
    import os
    import signal as signal_module

    import rlmesh

    if not hasattr(signal_module, "SIGINT"):
        pytest.skip("SIGINT not available on this platform")

    env = TinyEnv()
    server = env_server(env)

    previous_handler = signal_module.getsignal(signal_module.SIGINT)
    # Swallow SIGINT in the test process so the raised KeyboardInterrupt (chained
    # from Python's own handler) does not abort the test runner.
    signal_module.signal(signal_module.SIGINT, lambda *_: None)
    try:
        server.start()
        try:
            remote = connect_with_retry(rlmesh.RemoteEnv, server.address)
            remote.close()

            os.kill(os.getpid(), signal_module.SIGINT)
            time.sleep(0.5)

            # Server should still be serving: a fresh client connects and steps.
            survivor = connect_with_retry(rlmesh.RemoteEnv, server.address)
            try:
                observation, _ = survivor.reset(seed=7)
                assert observation == 0
            finally:
                survivor.close()
            assert server.wait(0.01) is False
        finally:
            server.shutdown()
    finally:
        signal_module.signal(signal_module.SIGINT, previous_handler)

    assert env.close_calls == 1


def test_env_server_wait_after_shutdown_returns_true() -> None:
    env = TinyEnv()
    server = env_server(env)
    server.start()
    server.shutdown()

    assert server.wait(0) is True
    assert env.close_calls == 1


def test_env_server_wait_before_start_raises() -> None:
    env = TinyEnv()
    server = env_server(env)
    try:
        with pytest.raises(RuntimeError, match="before start"):
            server.wait(0)
    finally:
        server.shutdown()

    assert env.close_calls == 1


@pytest.mark.parametrize("timeout", [-1.0, float("nan"), float("inf")])
def test_env_server_wait_rejects_invalid_timeout(timeout: float) -> None:
    env = TinyEnv()
    server = env_server(env)
    try:
        with pytest.raises(ValueError, match="timeout"):
            server.wait(timeout)
    finally:
        server.shutdown()

    assert env.close_calls == 1


def test_remote_env_accepts_one_env_vector_endpoint() -> None:
    import rlmesh

    _ = pytest.importorskip("numpy")

    env = TinyOneVectorEnv()
    server = env_server(env)
    server.start()
    try:
        remote = connect_with_retry(rlmesh.RemoteEnv, server.address)
        try:
            observation, info = remote.reset(seed=123)
            assert observation == 0
            assert info["seed"] == 123

            observation, reward, terminated, truncated, _info = remote.step(0)
            assert observation == 1
            assert reward == 1.0
            assert terminated is False
            assert truncated is False
            assert env.last_actions_shape == (1,)
        finally:
            remote.close()
    finally:
        server.shutdown()


def test_remote_vector_close_detaches_without_stopping_endpoint() -> None:
    import rlmesh

    env = TinyVectorEnv()
    server = env_server(env)
    server.start()
    try:
        remote = connect_with_retry(rlmesh.RemoteVectorEnv, server.address)
        assert remote.shutdown("default remote shutdown is disabled") is False
        remote.close()

        assert env.close_calls == 0

        second = connect_with_retry(rlmesh.RemoteVectorEnv, server.address)
        try:
            _observations, info = second.reset(seed=[1, 2])
            assert info["seed"] == [1, 2]
        finally:
            second.close()
    finally:
        server.shutdown()

    assert env.close_calls == 1


def test_remote_vector_env_keeps_one_env_action_batch_shape() -> None:
    import numpy as np
    from rlmesh import numpy as rlmesh_numpy

    env = TinyOneVectorEnv()
    server = env_server(env)
    server.start()
    try:
        remote = connect_with_retry(rlmesh_numpy.RemoteVectorEnv, server.address)
        try:
            observations, info = remote.reset(seed=123)
            assert observations == [0]
            assert info["seed"] == 123

            observations, rewards, terminated, truncated, _info = remote.step([0])
            assert observations == [1]
            np.testing.assert_array_equal(rewards, np.asarray([1.0]))
            np.testing.assert_array_equal(terminated, np.asarray([False]))
            np.testing.assert_array_equal(truncated, np.asarray([False]))
            assert env.last_actions_shape == (1,)
        finally:
            remote.close()
    finally:
        server.shutdown()


def test_remote_vector_space_properties_load_from_contract() -> None:
    import rlmesh
    from rlmesh import spaces

    env = TinyVectorEnv()
    server = env_server(env)
    server.start()
    try:
        remote = connect_with_retry(rlmesh.RemoteVectorEnv, server.address)
        try:
            assert isinstance(remote.single_observation_space, spaces.Discrete)
            assert remote.single_observation_space.n == 2
            assert remote.observation_space is remote.single_observation_space
            assert isinstance(remote.single_action_space, spaces.Discrete)
            assert remote.single_action_space.n == 2
            assert remote.action_space is remote.single_action_space
        finally:
            remote.close()
    finally:
        server.shutdown()


def test_remote_vector_shutdown_requires_explicit_allow() -> None:
    import rlmesh

    env = TinyVectorEnv()
    server = env_server(
        env,
        rlmesh.ServeOptions(allow_remote_shutdown=True),
    )
    server.start()
    remote = connect_with_retry(rlmesh.RemoteVectorEnv, server.address)
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
        remote = connect_with_retry(rlmesh_numpy.RemoteVectorEnv, server.address)
        try:
            observations, info = remote.reset(seed=[1, 2])
            assert isinstance(observations, np.ndarray)
            observation_array = cast("NumpyArray", observations)
            assert observation_array.shape == (2, 4)
            assert info["seed"] == [1, 2]

            actions = np.zeros(2, dtype=np.int64)
            observations, rewards, terminated, truncated, info = remote.step(actions)

            assert isinstance(observations, np.ndarray)
            observation_array = cast("NumpyArray", observations)
            assert observation_array.shape == (2, 4)
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
        remote = connect_with_retry(rlmesh.RemoteEnv, server.address)
        try:
            observation, info = remote.reset(seed=123)
            assert observation == 0
            episode_ids = info["episode_ids"]
            assert isinstance(episode_ids, list)
            assert len(episode_ids) == 1

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
        remote = connect_with_retry(rlmesh.RemoteEnv, server.address)
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
        remote = connect_with_retry(rlmesh.RemoteVectorEnv, server.address)
        try:
            observations, info = remote.reset(seed=[1, 2])
            assert observations == [0, 0]
            episode_ids = info["episode_ids"]
            assert isinstance(episode_ids, list)
            assert len(episode_ids) == 2

            observations, rewards, terminated, truncated, info = remote.step([0, 1])
            assert observations == [1, 1]
            assert isinstance(rewards, rlmesh.Tensor)
            assert isinstance(terminated, rlmesh.Tensor)
            assert isinstance(truncated, rlmesh.Tensor)
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
    remote = connect_with_retry(rlmesh.RemoteEnv, server.address)
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
        remote = connect_with_retry(rlmesh.RemoteEnv, server.address)
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
