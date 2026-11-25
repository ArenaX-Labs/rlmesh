from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, TypeVar, cast

import pytest

if TYPE_CHECKING:
    import numpy as np
    from rlmesh import EnvServer, ServeOptions
    from rlmesh._server import EnvLike as ServedEnv

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


class SlowStepEnv(TinyEnv):
    def __init__(self, step_delay: float) -> None:
        super().__init__()
        self._step_delay = step_delay

    def step(self, action: object):
        time.sleep(self._step_delay)
        return super().step(action)


class FailingStepEnv(TinyEnv):
    def __init__(self) -> None:
        super().__init__()
        self.fail_with: BaseException | None = None

    def step(self, action: object):
        failure, self.fail_with = self.fail_with, None
        if failure is not None:
            raise failure
        self.step_count += 1
        return 1, 1.0, False, False, {}


class BlockingStepEnv(TinyEnv):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def step(self, action: object):
        self.entered.set()
        self.release.wait(10.0)
        return super().step(action)


class InfoKeyEnv(TinyEnv):
    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ):
        return 0, {"episode_ids": ["env-supplied"]}

    def step(self, action: object):
        return (
            1,
            1.0,
            True,
            False,
            {
                "episode_ids": ["env-supplied"],
                "completed_episodes": "env-value",
            },
        )


class RenderingEnv(TinyEnv):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.render_mode = "rgb_array"
        self._channels = channels

    def render(self) -> NumpyArray:
        import numpy as np

        if self._channels == 1:
            return np.arange(4 * 3, dtype=np.uint8).reshape(4, 3)
        return np.arange(4 * 3 * self._channels, dtype=np.uint8).reshape(
            4, 3, self._channels
        )


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
            with pytest.raises(ValueError, match=pattern) as raised:
                rlmesh.RemoteEnv(address)
            _, _, key = address.partition("#")
            if key:
                assert key.split("/")[0] not in str(raised.value)
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
        return rlmesh.EnvServer(cast("ServedEnv", env), ":0/env", options=options)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise


def test_env_server_auto_detects_vector_env_shape() -> None:
    server = env_server(TinyVectorEnv())
    server.shutdown()


def test_remote_close_detaches_without_stopping_endpoint() -> None:
    import rlmesh

    env = TinyEnv()
    server = env_server(env)
    server.start()
    try:
        remote = connect_with_retry(rlmesh.RemoteEnv, server.address)
        with pytest.raises(AttributeError, match="shutdown"):
            remote.shutdown("an owner shutdown")
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
        assert isinstance(server.env_contract, EnvContract)
        assert server.env_contract.id == "TinySpecEnv-v0"
        assert server.env_contract.id == "TinySpecEnv-v0"

        server.start()
        server.shutdown()

        assert server.env_contract.id == "TinySpecEnv-v0"
        assert server.env_contract.id == "TinySpecEnv-v0"
    finally:
        server.shutdown()

    assert env.close_calls == 1


@pytest.mark.parametrize("spec_id", [None, 123, object()])
def test_env_contract_id_falls_back_for_non_string_spec_id(spec_id: object) -> None:
    env = TinyEnv()
    env.spec = SimpleNamespace(id=spec_id)  # type: ignore[attr-defined]
    server = env_server(env)
    try:
        assert server.env_contract.id == "UnknownEnv-v1"
    finally:
        server.shutdown()


def test_env_server_vector_contract_reports_num_envs() -> None:
    env = TinySpecVectorEnv()
    server = env_server(env)
    try:
        assert server.env_contract.id == "TinySpecVectorEnv-v0"
        assert server.env_contract.num_envs == 2
        assert server.env_contract.num_envs == 2
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


@pytest.mark.parametrize(
    ("channels", "expected_shape"),
    [(3, (4, 3, 3)), (4, (4, 3, 4)), (1, (4, 3))],
)
def test_remote_render_preserves_source_channel_count(
    channels: int, expected_shape: tuple[int, ...]
) -> None:
    import numpy as np
    import rlmesh

    env = RenderingEnv(channels)
    server = env_server(env)
    server.start()
    try:
        remote = connect_with_retry(rlmesh.RemoteEnv, server.address)
        try:
            remote.reset(seed=0)
            frame = remote.render()
            assert frame is not None
            array = np.asarray(frame)
            assert array.shape == expected_shape
            assert array.dtype == np.uint8
        finally:
            remote.close()
    finally:
        server.shutdown()


def test_client_preserves_env_provided_info_keys() -> None:
    from rlmesh._rlmesh import PyEnvClient

    env = InfoKeyEnv()
    server = env_server(env)
    server.start()
    try:
        client = PyEnvClient(server.address)
        try:
            _, reset_info = client.reset()
            assert reset_info["episode_ids"] == ["env-supplied"]

            _, _, _, _, step_info = client.step(0)
            assert step_info["episode_ids"] == ["env-supplied"]
            assert step_info["completed_episodes"] == "env-value"
        finally:
            client.close()
    finally:
        server.shutdown()


def test_client_logs_a_shadowed_runtime_info_key_once(
    capfd: pytest.CaptureFixture[str],
) -> None:
    from rlmesh._rlmesh import PyEnvClient

    env = InfoKeyEnv()
    server = env_server(env)
    server.start()
    try:
        client = PyEnvClient(server.address)
        try:
            client.reset()
            client.step(0)
            client.reset()
            client.step(0)
        finally:
            client.close()
    finally:
        server.shutdown()

    stderr = capfd.readouterr().err
    assert stderr.count("shadows the runtime-reserved info key") == 1


class _TwoStepEnv(TinyEnv):
    def step(self, action: object):
        self.step_count += 1
        return 1, 1.0, self.step_count >= 2, False, {}


def test_scalar_client_reports_completed_episodes_on_every_step() -> None:
    from rlmesh._rlmesh import PyEnvClient

    env = _TwoStepEnv()
    server = env_server(env)
    server.start()
    try:
        client = PyEnvClient(server.address)
        try:
            client.reset()
            *_, mid_info = client.step(0)
            *_, end_info = client.step(0)
        finally:
            client.close()
    finally:
        server.shutdown()

    assert mid_info["completed_episodes"] == 0
    assert end_info["completed_episodes"] == 1
    assert len(cast("list[str]", mid_info["episode_ids"])) == 1


def test_client_injects_episode_ids_when_env_omits_them() -> None:
    from rlmesh._rlmesh import PyEnvClient

    env = TinyEnv()
    server = env_server(env)
    server.start()
    try:
        client = PyEnvClient(server.address)
        try:
            _, reset_info = client.reset()
            assert "episode_ids" in reset_info
        finally:
            client.close()
    finally:
        server.shutdown()


def test_client_constructor_default_timeout_applies() -> None:
    from rlmesh._rlmesh import PyEnvClient

    env = SlowStepEnv(step_delay=3.0)
    server = env_server(env)
    server.start()
    try:
        client = PyEnvClient(server.address, request_timeout_seconds=0.9)
        try:
            client.reset()
            with pytest.raises(
                TimeoutError, match=r"^operation timed out after 900ms$"
            ):
                client.step(0)
        finally:
            client.close()
    finally:
        server.shutdown()


def test_recoverable_env_error_keeps_the_session() -> None:
    import rlmesh

    env = FailingStepEnv()
    server = env_server(env)
    server.start()
    try:
        remote = connect_with_retry(rlmesh.RemoteEnv, server.address)
        remote.reset(seed=0)

        env.fail_with = rlmesh.RecoverableEnvironmentException("solver hiccup")
        with pytest.raises(
            rlmesh.RecoverableEnvironmentException, match="solver hiccup"
        ) as recoverable:
            remote.step(0)
        assert recoverable.value.is_recoverable is True
        assert recoverable.value.code == "INTERNAL"
        assert str(recoverable.value) == "[INTERNAL] step failed: solver hiccup"
        remote.step(0)

        env.fail_with = RuntimeError("solver diverged")
        with pytest.raises(
            rlmesh.EnvironmentException, match="solver diverged"
        ) as fatal:
            remote.step(0)
        assert fatal.value.is_recoverable is False
        assert not isinstance(fatal.value, rlmesh.RecoverableEnvironmentException)

        remote.reset(seed=1)
        remote.step(0)
        assert env.step_count == 1
        remote.close()
    finally:
        server.shutdown()


def test_model_exception_raised_in_an_env_is_not_recoverable() -> None:
    import rlmesh

    env = FailingStepEnv()
    server = env_server(env)
    server.start()
    try:
        remote = connect_with_retry(rlmesh.RemoteEnv, server.address)
        remote.reset(seed=0)
        env.fail_with = rlmesh.ModelException("not an env failure", is_recoverable=True)
        with pytest.raises(
            rlmesh.EnvironmentException, match="not an env failure"
        ) as fatal:
            remote.step(0)
        assert fatal.value.is_recoverable is False
        assert not isinstance(fatal.value, rlmesh.RecoverableEnvironmentException)
        remote.close()
    finally:
        server.shutdown()


def test_close_is_best_effort_when_the_peer_is_already_gone() -> None:
    import rlmesh

    server = env_server(TinyEnv())
    server.start()
    remote = connect_with_retry(rlmesh.RemoteEnv, server.address)
    remote.reset(seed=0)
    server.shutdown()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            remote.step(0)
        except ConnectionError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the client never noticed the shut-down server")

    remote.close()


def test_concurrent_client_calls_name_the_single_caller_rule() -> None:
    import rlmesh

    env = BlockingStepEnv()
    server = env_server(env)
    server.start()
    try:
        remote = connect_with_retry(rlmesh.RemoteEnv, server.address)
        remote.reset(seed=0)
        worker = threading.Thread(target=lambda: remote.step(0))
        worker.start()
        try:
            assert env.entered.wait(5.0), "the background step never reached the env"
            with pytest.raises(RuntimeError, match="one caller at a time"):
                remote.step(1)
        finally:
            env.release.set()
            worker.join(10.0)
        remote.close()
    finally:
        server.shutdown()


def test_client_rejects_invalid_timeout() -> None:
    from rlmesh._rlmesh import PyEnvClient

    env = TinyEnv()
    server = env_server(env)
    server.start()
    try:
        with pytest.raises(ValueError):
            PyEnvClient(server.address, request_timeout_seconds=float("nan"))
    finally:
        server.shutdown()


def test_background_server_survives_unrelated_sigint() -> None:
    import os
    import signal as signal_module

    import rlmesh

    if not hasattr(signal_module, "SIGINT"):
        pytest.skip("SIGINT not available on this platform")

    env = TinyEnv()
    server = env_server(env)

    previous_handler = signal_module.getsignal(signal_module.SIGINT)
    signal_module.signal(signal_module.SIGINT, lambda *_: None)
    try:
        server.start()
        try:
            remote = connect_with_retry(rlmesh.RemoteEnv, server.address)
            remote.close()

            os.kill(os.getpid(), signal_module.SIGINT)
            time.sleep(0.5)

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


def test_remote_vector_close_detaches_without_stopping_endpoint() -> None:
    import rlmesh

    env = TinyVectorEnv()
    server = env_server(env)
    server.start()
    try:
        remote = connect_with_retry(rlmesh.RemoteVectorEnv, server.address)
        with pytest.raises(AttributeError, match="shutdown"):
            remote.shutdown("an owner shutdown")
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


def test_env_server_serves_one_env_vector_shape_as_scalar() -> None:
    import rlmesh
    from rlmesh.numpy import RemoteEnv

    env = TinyOneVectorEnv()
    server = rlmesh.EnvServer(env, ":0/env")
    server.start()
    try:
        client = RemoteEnv(server.address)
        assert client.env_contract.num_envs == 1
        client.reset(seed=0)
        client.step(1)
        assert env.last_actions_shape == (1,)
        client.close()
    finally:
        server.shutdown()


def test_env_server_asserts_the_expected_contract_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlmesh
    from rlmesh.adapters import ENV_BRANCH_METADATA_KEY

    env = TinyEnv()
    env.metadata = {ENV_BRANCH_METADATA_KEY: {"action_type": "abs"}}  # type: ignore[attr-defined]

    monkeypatch.setenv("RLMESH_EXPECTED_ENV_BRANCH", '{"action_type": "abs"}')
    server = rlmesh.EnvServer(env, ":0/env")
    server.shutdown()

    monkeypatch.setenv("RLMESH_EXPECTED_ENV_BRANCH", '{"action_type": "delta"}')
    with pytest.raises(ValueError, match="different contracts"):
        rlmesh.EnvServer(env, ":0/env")

    plain = TinyEnv()
    with pytest.raises(ValueError, match="published null"):
        rlmesh.EnvServer(plain, ":0/env")

    monkeypatch.delenv("RLMESH_EXPECTED_ENV_BRANCH")
    rlmesh.EnvServer(plain, ":0/env").shutdown()


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
            assert np.asarray(info["seed"]).tolist() == [1, 2]

            actions = np.zeros(2, dtype=np.int64)
            observations, rewards, terminated, truncated, info = remote.step(actions)

            assert isinstance(observations, np.ndarray)
            observation_array = cast("NumpyArray", observations)
            assert observation_array.shape == (2, 4)
            np.testing.assert_array_equal(rewards, np.asarray([1.0, 2.0]))
            np.testing.assert_array_equal(terminated, np.asarray([False, True]))
            np.testing.assert_array_equal(truncated, np.asarray([False, False]))
            assert np.asarray(info["actions"]).tolist() == [0, 0]
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


def test_model_lifecycle_callbacks_are_zero_argument() -> None:
    import rlmesh

    env = TinyEnv()
    server = env_server(env)
    server.start()
    calls: list[str] = []

    def on_episode_end() -> None:
        calls.append("episode_end")

    def on_close() -> None:
        calls.append("close")

    try:
        model = rlmesh.Model(
            lambda _observation: 0,
            on_episode_end=on_episode_end,
            on_close=on_close,
        )
        model.run(server.address, episodes=1)
        assert calls == ["episode_end"]
        model.close()
    finally:
        server.shutdown()

    assert calls == ["episode_end", "close"]


def test_a_binding_fires_episode_end_between_episodes() -> None:
    import rlmesh

    env = TinyEnv()
    server = env_server(env)
    server.start()
    ends: list[int] = []

    try:
        model = rlmesh.Model(
            lambda _observation: 0, on_episode_end=lambda: ends.append(1)
        )
        with model.bind(server.address) as binding:
            for _ in range(2):
                binding.reset()
                while not binding.done:
                    binding.step()
        assert sum(ends) == 2
    finally:
        server.shutdown()


def test_server_client_lifecycle_process_exits_promptly() -> None:
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import rlmesh
        from rlmesh import EnvServer

        class TinyEnv:
            def __init__(self):
                from rlmesh import spaces
                self.observation_space = spaces.Discrete(2)
                self.action_space = spaces.Discrete(2)
            def reset(self, *, seed=None, options=None):
                return 0, {}
            def step(self, action):
                return 1, 1.0, True, False, {}
            def close(self):
                pass

        server = EnvServer(
            TinyEnv(),
            ":0/env",
        )
        server.start()
        remote = rlmesh.RemoteEnv(server.address)
        remote.reset(seed=1)
        remote.step(remote.action_space.sample())

        try:
            rlmesh.Model(lambda _o: 0).run(
                remote, episodes=1, close_env=True
            )
        except Exception:
            pass
        finally:
            server.shutdown()

        print("CLEAN-EXIT")
        """
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            "server/client lifecycle process hung and did not exit within 30s"
        ) from exc

    if "Operation not permitted" in result.stderr:
        pytest.skip("local tcp bind is not permitted in this environment")
    assert result.returncode == 0, result.stderr
    assert "CLEAN-EXIT" in result.stdout


_STALLED_DRIVER = """
import sys

import rlmesh
from rlmesh import spaces


class TinyEnv:
    def __init__(self):
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        return 0, {}

    def step(self, action):
        return 1, 1.0, True, False, {}

    def close(self):
        pass


server = None
address = sys.argv[1]
if not address:
    server = rlmesh.EnvServer(TinyEnv(), ":0/env")
    server.start()
    address = server.address
remote = rlmesh.RemoteEnv(address)
print("CONNECTED", flush=True)
sys.stdin.readline()
remote.reset(seed=1)
remote.step(1)
remote.close()
if server is not None:
    server.shutdown()
print("DROVE", flush=True)
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs SIGSTOP (POSIX)")
@pytest.mark.parametrize(
    ("own_server", "stall_seconds"),
    [(True, 1.5), (False, 0.6)],
    ids=["host_and_driver", "driver_alone"],
)
def test_a_driver_keeps_its_env_through_a_process_stall(
    own_server: bool, stall_seconds: float
) -> None:
    import signal
    import subprocess
    import sys

    import rlmesh

    server = None
    address = ""
    if not own_server:
        server = rlmesh.EnvServer(TinyEnv(), ":0/env")
        server.start()
        address = server.address
    child = subprocess.Popen(
        [sys.executable, "-c", _STALLED_DRIVER, address],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        connected = child.stdout.readline()
        if "CONNECTED" not in connected:
            _out, err = child.communicate(timeout=30)
            if "Operation not permitted" in err:
                pytest.skip("local tcp bind is not permitted in this environment")
            raise AssertionError(f"the driver did not connect: {err}")
        os.kill(child.pid, signal.SIGSTOP)
        time.sleep(stall_seconds)
        os.kill(child.pid, signal.SIGCONT)
        out, err = child.communicate("go\n", timeout=30)
        assert child.returncode == 0, err
        assert "DROVE" in out
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        if server is not None:
            server.shutdown()


@pytest.mark.skipif(not os.path.isdir("/dev/fd"), reason="needs /dev/fd (POSIX)")
def test_repeated_evals_do_not_accumulate_fds_or_runtimes() -> None:
    import rlmesh

    def one_eval() -> None:
        server = env_server(TinyEnv())
        server.start()
        try:
            rlmesh.run(rlmesh.Model(lambda _o: 0), server.address, episodes=1)
        finally:
            server.shutdown()

    one_eval()
    baseline = len(os.listdir("/dev/fd"))
    for _ in range(5):
        one_eval()
    grown = len(os.listdir("/dev/fd")) - baseline
    assert grown <= 5, f"fd count grew by {grown} across 5 evals (leak)"


def test_a_closed_client_is_misuse_and_lanes_are_checked_before_any_request() -> None:
    import rlmesh

    env = TinyEnv()
    server = env_server(env)
    server.start()
    try:
        remote = connect_with_retry(rlmesh.RemoteEnv, server.address)
        with pytest.raises(ValueError, match="out of range"):
            remote.render(env_index=2**32)
        remote.close()
        remote.close()
        with pytest.raises(RuntimeError, match=r"^RemoteEnv is closed$"):
            remote.reset()
        assert remote.env_contract is not None
    finally:
        server.shutdown()
