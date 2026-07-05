from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest


def _make_numpy_vector_env(*, action_kind: str, num_envs: int) -> Any:
    from rlmesh._client._remote_vector_env import RemoteVectorEnvBase
    from rlmesh.numpy import RemoteVectorEnv

    env: Any = RemoteVectorEnv.__new__(RemoteVectorEnv)
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
    from rlmesh._client._remote_vector_env import _normalize_autoreset_mode

    normalized = _normalize_autoreset_mode({"autoreset_mode": "NextStep"})

    assert isinstance(normalized["autoreset_mode"], autoreset)
    assert normalized["autoreset_mode"] is autoreset.NEXT_STEP


def test_normalize_autoreset_mode_passes_through_other_keys() -> None:
    from rlmesh._client._remote_vector_env import _normalize_autoreset_mode

    metadata = {"render_fps": 30}
    normalized = _normalize_autoreset_mode(metadata)

    assert normalized == metadata


def test_normalize_autoreset_mode_leaves_unknown_string() -> None:
    from rlmesh._client._remote_vector_env import _normalize_autoreset_mode

    normalized = _normalize_autoreset_mode({"autoreset_mode": "bogus"})

    assert normalized["autoreset_mode"] == "bogus"


def test_normalize_autoreset_mode_idempotent_on_enum() -> None:
    autoreset = pytest.importorskip("gymnasium.vector").AutoresetMode
    from rlmesh._client._remote_vector_env import _normalize_autoreset_mode

    normalized = _normalize_autoreset_mode({"autoreset_mode": autoreset.SAME_STEP})

    assert normalized["autoreset_mode"] is autoreset.SAME_STEP


class _FakeNativeClient:
    def __init__(
        self,
        *,
        num_envs: int = 2,
        handshake_error: Exception | None = None,
    ) -> None:
        self._num_envs = num_envs
        self._handshake_error = handshake_error
        self.closed = False

    def address(self) -> str:
        return "tcp://127.0.0.1:5555"

    def handshake(self) -> Any:
        if self._handshake_error is not None:
            raise self._handshake_error
        return SimpleNamespace(num_envs=self._num_envs)

    def close(self) -> None:
        self.closed = True


def test_vector_client_rejects_scalar_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The mirror of the scalar client's vector-endpoint rejection: a vector
    # client on a 1-env endpoint fails at handshake, not later, and closes
    # the native client.
    from rlmesh._native import RemoteVectorEnv

    fake = _FakeNativeClient(num_envs=1)
    monkeypatch.setattr(RemoteVectorEnv, "_make_client", lambda self, a, c, r: fake)

    with pytest.raises(ValueError, match="Use RemoteEnv instead"):
        RemoteVectorEnv("127.0.0.1:5555")
    assert fake.closed is True


def test_handshake_failure_closes_client_and_names_handshake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A handshake failure after a successful connect must close the native
    # client (no leak) and must not be mislabeled "is the EnvServer running?".
    from rlmesh._native import RemoteEnv

    fake = _FakeNativeClient(handshake_error=ConnectionError("boom"))
    monkeypatch.setattr(RemoteEnv, "_make_client", lambda self, a, c, r: fake)

    with pytest.raises(ConnectionError, match="handshake failed") as excinfo:
        RemoteEnv("127.0.0.1:5555")
    assert "is the EnvServer running" not in str(excinfo.value)
    assert fake.closed is True

    fake_other = _FakeNativeClient(handshake_error=RuntimeError("proto"))
    monkeypatch.setattr(RemoteEnv, "_make_client", lambda self, a, c, r: fake_other)
    with pytest.raises(RuntimeError, match="proto"):
        RemoteEnv("127.0.0.1:5555")
    assert fake_other.closed is True


def test_client_timeouts_pass_through_to_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # connect_timeout_seconds / request_timeout_seconds ride the constructor
    # down to the native client instead of being hardcoded to None.
    from rlmesh._native import RemoteVectorEnv

    captured: dict[str, Any] = {}

    def fake_make(
        self: Any, address: str, connect: float | None, request: float | None
    ) -> Any:
        captured["connect"], captured["request"] = connect, request
        return _FakeNativeClient(num_envs=3)

    monkeypatch.setattr(RemoteVectorEnv, "_make_client", fake_make)
    RemoteVectorEnv(
        "127.0.0.1:5555", connect_timeout_seconds=1.5, request_timeout_seconds=2.5
    )
    assert captured == {"connect": 1.5, "request": 2.5}


def test_remote_model_supports_endpoint_helper_grammar() -> None:
    # RemoteModel gets the same host=/port=/path=/transport= grammar and
    # address normalization as RemoteEnv/RemoteVectorEnv.
    from rlmesh._native import RemoteModel

    assert RemoteModel(host="10.0.0.1", port=5556).address == "tcp://10.0.0.1:5556"
    assert RemoteModel("127.0.0.1:5556").address == "127.0.0.1:5556"
    with pytest.raises(ValueError, match="cannot be combined"):
        RemoteModel("127.0.0.1:5556", host="10.0.0.1")


def test_remote_model_session_rejects_unhonored_kwargs() -> None:
    # session() must hard-error on instruction/trust_entrypoints rather
    # than silently dropping them (the served model owns its adapter).
    from rlmesh._native import RemoteModel

    model = RemoteModel("127.0.0.1:5556")
    env = SimpleNamespace(env_contract=object())
    with pytest.raises(ValueError, match="instruction= is not supported"):
        model.session(cast("Any", env), instruction="pick up the block")
    with pytest.raises(ValueError, match="trust_entrypoints= applies to local"):
        model.session(cast("Any", env), trust_entrypoints=True)


def test_remote_model_timeouts_pass_through_to_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """connect/request timeouts on the handle reach the native PyModelClient."""
    import rlmesh._load_native as load_native_mod
    from rlmesh._native import RemoteModel

    captured: dict[str, Any] = {}

    class _DialAbortedError(RuntimeError):
        pass

    def fake_client(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        raise _DialAbortedError

    monkeypatch.setattr(load_native_mod, "load_native", lambda name: fake_client)

    model = RemoteModel(
        "127.0.0.1:5556", connect_timeout_seconds=1.5, request_timeout_seconds=2.5
    )
    env = SimpleNamespace(env_contract=object())
    with pytest.raises(_DialAbortedError):
        model.session(cast("Any", env))
    assert captured["connect_timeout_seconds"] == 1.5
    assert captured["request_timeout_seconds"] == 2.5
