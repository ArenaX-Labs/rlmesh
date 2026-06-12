from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


def _make_numpy_vector_env(*, action_kind: str, num_envs: int) -> Any:
    from rlmesh.client.remote_vector_env import RemoteVectorEnvBase
    from rlmesh.numpy import RemoteVectorEnv

    env = RemoteVectorEnv.__new__(RemoteVectorEnv)
    action_space = SimpleNamespace(kind=action_kind)
    env._env_contract = SimpleNamespace(action_space=action_space)
    env._client = SimpleNamespace(num_envs=lambda: num_envs)
    assert isinstance(env, RemoteVectorEnvBase)
    return env


def test_encode_actions_splits_numpy_batch_into_tensors() -> None:
    np = pytest.importorskip("numpy")
    from rlmesh._rlmesh import Tensor

    env = _make_numpy_vector_env(action_kind="box", num_envs=3)
    actions = np.zeros((3, 4), dtype=np.float32)

    encoded = env._encode_actions(actions)

    assert isinstance(encoded, list)
    assert len(encoded) == 3
    assert all(isinstance(item, Tensor) for item in encoded)


def test_encode_actions_passes_through_dict_action_space() -> None:
    np = pytest.importorskip("numpy")

    env = _make_numpy_vector_env(action_kind="dict", num_envs=2)
    actions = {"move": np.zeros((2, 4), dtype=np.float32)}

    assert env._encode_actions(actions) is actions


def test_encode_actions_passes_through_list_batch() -> None:
    env = _make_numpy_vector_env(action_kind="discrete", num_envs=2)
    actions = [0, 1]

    assert env._encode_actions(actions) is actions


def test_encode_actions_passes_through_on_count_mismatch() -> None:
    np = pytest.importorskip("numpy")

    env = _make_numpy_vector_env(action_kind="box", num_envs=3)
    actions = np.zeros((2, 4), dtype=np.float32)

    assert env._encode_actions(actions) is actions


def test_normalize_autoreset_mode_restores_enum() -> None:
    autoreset = pytest.importorskip("gymnasium.vector").AutoresetMode
    from rlmesh.client.remote_vector_env import _normalize_autoreset_mode

    normalized = _normalize_autoreset_mode({"autoreset_mode": "NextStep"})

    assert isinstance(normalized["autoreset_mode"], autoreset)
    assert normalized["autoreset_mode"] is autoreset.NEXT_STEP


def test_normalize_autoreset_mode_passes_through_other_keys() -> None:
    from rlmesh.client.remote_vector_env import _normalize_autoreset_mode

    metadata = {"render_fps": 30}
    normalized = _normalize_autoreset_mode(metadata)

    assert normalized == metadata


def test_normalize_autoreset_mode_leaves_unknown_string() -> None:
    from rlmesh.client.remote_vector_env import _normalize_autoreset_mode

    normalized = _normalize_autoreset_mode({"autoreset_mode": "bogus"})

    assert normalized["autoreset_mode"] == "bogus"


def test_normalize_autoreset_mode_idempotent_on_enum() -> None:
    autoreset = pytest.importorskip("gymnasium.vector").AutoresetMode
    from rlmesh.client.remote_vector_env import _normalize_autoreset_mode

    normalized = _normalize_autoreset_mode({"autoreset_mode": autoreset.SAME_STEP})

    assert normalized["autoreset_mode"] is autoreset.SAME_STEP
