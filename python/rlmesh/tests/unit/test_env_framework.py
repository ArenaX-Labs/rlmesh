"""Framework-aware env serving: the BridgedEnv wrapper, device/host ops, the
framework resolver, and the EnvServer device guard.

Live socket round-trips (the Rust Native backend + the undeclared GPU-obs
host-move) are covered by the integration suite; these are fast, pure-Python
checks of the wrapper and the EnvServer wiring around it.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from rlmesh._rlmesh import Tensor
from rlmesh._value_conversion import resolve_bridge
from rlmesh.numpy import from_array

torch = pytest.importorskip("torch")


def _native_action(values: list[float]) -> Tensor:
    """A native Tensor action, as the Rust Native backend hands BridgedEnv.step."""
    return cast(Tensor, from_array(np.array(values, dtype=np.float32)))


class _RecordingEnv:
    """Minimal env that records the action it was handed."""

    def __init__(self, obs: object, *, reward: object = 0.0) -> None:
        self._obs = obs
        self._reward = reward
        self.seen_action: Any = None

    def reset(self, *, seed: object = None, options: object = None) -> object:
        return self._obs, {}

    def step(self, action: object) -> object:
        self.seen_action = action
        return self._obs, self._reward, False, False, {}

    def close(self) -> None:
        pass


def _bridge() -> Any:
    return resolve_bridge("torch")


def test_step_decodes_action_to_framework_tensor() -> None:
    from rlmesh._server_bridge import BridgedEnv

    env = _RecordingEnv(torch.zeros(2))
    wrapped = BridgedEnv(env, _bridge())
    wrapped.step(_native_action([0.5, -0.5]))

    assert isinstance(env.seen_action, torch.Tensor)
    assert np.allclose(env.seen_action.numpy(), [0.5, -0.5])


def test_obs_and_reset_encode_to_native_tensor() -> None:
    from rlmesh._server_bridge import BridgedEnv

    wrapped = BridgedEnv(_RecordingEnv(torch.ones(3) * 2.0), _bridge())

    obs, _info = cast("tuple[Any, Any]", wrapped.reset(seed=0))
    assert isinstance(obs, Tensor)

    obs = cast("tuple[Any, ...]", wrapped.step(_native_action([0.0, 0.0])))[0]
    assert isinstance(obs, Tensor)


def test_device_places_incoming_action() -> None:
    from rlmesh._server_bridge import BridgedEnv

    env = _RecordingEnv(torch.zeros(2))
    wrapped = BridgedEnv(env, _bridge(), device="cpu")
    wrapped.step(_native_action([1.0, 2.0]))

    assert isinstance(env.seen_action, torch.Tensor)
    assert env.seen_action.device.type == "cpu"


def test_to_host_coerces_tensor_reward_and_done() -> None:
    from rlmesh._server_bridge import BridgedEnv

    # A GPU-style batched reward/done tensor must come back as plain Python
    # scalars/lists (not a tensor) so the native step extractor stays cheap.
    env = _RecordingEnv(torch.zeros(2, 3), reward=torch.tensor([1.0, 2.0]))
    wrapped = BridgedEnv(env, _bridge())
    _obs, reward, terminated, truncated, _info = cast(
        "tuple[Any, Any, Any, Any, Any]", wrapped.step(_native_action([0.0]))
    )

    assert reward == [1.0, 2.0]
    assert not isinstance(reward, torch.Tensor)
    assert terminated is False and truncated is False


def test_warns_once_on_foreign_framework_obs_leaf() -> None:
    from rlmesh._server_bridge import BridgedEnv

    # 'b' is numpy inside an otherwise-torch obs -> one warning across many steps.
    obs = {"a": torch.ones(2), "b": np.ones(2, np.float32)}
    wrapped = BridgedEnv(_RecordingEnv(obs), _bridge())

    with pytest.warns(UserWarning, match="non-torch array leaf"):
        wrapped.step(_native_action([0.0]))

    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wrapped.step(_native_action([0.0]))
    assert [w for w in caught if "non-torch array leaf" in str(w.message)] == []


def test_scalar_python_leaf_does_not_warn() -> None:
    from rlmesh._server_bridge import BridgedEnv

    # A Discrete-style int / python scalar obs leaf is fine, not a foreign array.
    obs = {"a": torch.ones(2), "n": 3}
    wrapped = BridgedEnv(_RecordingEnv(obs), _bridge())

    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wrapped.step(_native_action([0.0]))
    assert [w for w in caught if "array leaf" in str(w.message)] == []


def test_delegates_unknown_attributes_to_inner_env() -> None:
    from rlmesh._server_bridge import BridgedEnv

    env = _RecordingEnv(torch.zeros(2))
    env.metadata = {"render_modes": []}  # type: ignore[attr-defined]
    wrapped = BridgedEnv(env, _bridge())

    assert wrapped.metadata == {"render_modes": []}
    assert wrapped.close() is None


def test_resolve_bridge_names_and_passthrough() -> None:
    assert resolve_bridge("torch").name == "torch"
    assert resolve_bridge("numpy").name == "numpy"
    assert resolve_bridge("np").name == "numpy"
    same = resolve_bridge("torch")
    assert resolve_bridge(same) is same  # a ValueBridge passes through
    with pytest.raises(ValueError, match="unknown framework"):
        resolve_bridge("tensorflow")


def test_envserver_device_requires_a_framework() -> None:
    import gymnasium as gym
    import rlmesh
    from rlmesh import spaces

    class _NumpyEnv:
        observation_space = spaces.from_gymnasium_space(
            gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        )
        action_space = spaces.from_gymnasium_space(gym.spaces.Discrete(2))

        def reset(self, *, seed: object = None, options: object = None) -> object:
            return np.zeros(2, np.float32), {}

        def step(self, action: object) -> object:
            return np.zeros(2, np.float32), 0.0, False, False, {}

        def close(self) -> None:
            pass

    # device= without a torch/jax framework is rejected before any socket binds.
    with pytest.raises(ValueError, match=r"device=.* requires"):
        rlmesh.EnvServer(cast("Any", _NumpyEnv()), "127.0.0.1:0", device="cpu")


def test_step_accepts_legacy_four_tuple() -> None:
    from rlmesh._server_bridge import BridgedEnv

    # A 4-tuple (obs, reward, done, info) is upgraded like native conversion.rs:
    # done -> terminated, truncated False. The bridge always hands native a 5-tuple.
    class _LegacyEnv(_RecordingEnv):
        def step(self, action: object) -> object:
            self.seen_action = action
            return self._obs, 1.0, True, {}

    wrapped = BridgedEnv(_LegacyEnv(torch.zeros(2)), _bridge())
    _obs, reward, terminated, truncated, _info = cast(
        "tuple[Any, Any, Any, Any, Any]", wrapped.step(_native_action([0.0, 0.0]))
    )
    assert reward == 1.0 and terminated is True and truncated is False


def test_reset_keeps_info_array_leaf_unencoded() -> None:
    from rlmesh._server_bridge import BridgedEnv

    # info is freeform metadata: an array leaf inside it must stay numpy (the native
    # metadata path reads it via .tolist()), not become a native Tensor the metadata
    # serializer cannot handle -- only the observation is encoded.
    class _InfoEnv(_RecordingEnv):
        def reset(self, *, seed: object = None, options: object = None) -> object:
            return self._obs, {"goal": np.array([1.0, 2.0], np.float32)}

    wrapped = BridgedEnv(_InfoEnv(torch.ones(2)), _bridge())
    obs, info = cast("tuple[Any, Any]", wrapped.reset())
    assert isinstance(obs, Tensor)
    assert isinstance(info["goal"], np.ndarray)


def test_getattr_does_not_recurse_before_init() -> None:
    import copy

    from rlmesh._server_bridge import BridgedEnv

    # pickle/copy build an instance via __new__ and probe attributes before __init__
    # sets _env; the underscore guard makes __getattr__ raise AttributeError instead
    # of recursing into __getattr__('_env') forever.
    bare = BridgedEnv.__new__(BridgedEnv)
    with pytest.raises(AttributeError):
        _ = bare._env  # type: ignore[attr-defined]

    wrapped = BridgedEnv(_RecordingEnv(torch.zeros(2)), _bridge())
    assert isinstance(copy.deepcopy(wrapped), BridgedEnv)  # no RecursionError


def test_gate_device_drops_ambient_device_without_a_device_framework() -> None:
    from rlmesh.serve import _gate_device

    # --device defaults to RLMESH_DEVICE; on a numpy/None framework that ambient
    # default is dropped, so a GPU node's global var does not crash a numpy serve.
    assert _gate_device("cuda:0", "numpy") is None
    assert _gate_device("cuda:0", None) is None
    assert _gate_device("cuda:0", "torch") == "cuda:0"
    assert _gate_device(None, "torch") is None


def test_reject_vectorized_framework_env() -> None:
    from rlmesh.serve import _reject_vectorized_framework

    # gym vectorization numpy-concatenates observations, discarding framework
    # tensors, so a torch/jax env cannot be fanned out that way; numpy and a scalar
    # serve are fine.
    with pytest.raises(NotImplementedError, match="num_envs>1"):
        _reject_vectorized_framework(True, "torch")
    with pytest.raises(NotImplementedError, match="num_envs>1"):
        _reject_vectorized_framework(True, "jax")
    _reject_vectorized_framework(True, "numpy")
    _reject_vectorized_framework(False, "torch")
