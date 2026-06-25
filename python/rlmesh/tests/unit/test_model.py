from __future__ import annotations

from typing import Any, cast

import pytest


def test_connect_uses_an_env_like_object_directly() -> None:
    from rlmesh._models._eval import _connect

    class Env:
        env_contract = "contract"

        def reset(self) -> None: ...
        def step(self, action: object) -> None: ...

    env = Env()
    client, contract, owns = _connect(env, "", None)
    assert client is env
    assert contract == "contract"
    assert owns is False  # caller-owned env objects are not closed by the loop


def test_connect_rejects_unsupported() -> None:
    from rlmesh._models._eval import _connect

    with pytest.raises(
        TypeError, match="env object, a remote-env object, or an address"
    ):
        _connect(object(), "", None)


def test_shutdown_passes_a_reason_when_accepted() -> None:
    from rlmesh._models._eval import _shutdown

    calls: list[tuple[object, ...]] = []

    class WithReason:
        def shutdown(self, reason: str) -> None:
            calls.append((reason,))

    _shutdown(WithReason())
    assert calls == [("model run complete",)]


def test_shutdown_falls_back_to_no_arg_shutdown() -> None:
    from rlmesh._models._eval import _shutdown

    calls: list[tuple[object, ...]] = []

    class NoReason:
        def shutdown(self) -> None:
            calls.append(())

    _shutdown(NoReason())
    assert calls == [()]


def test_backend_models_wire_their_own_remote_env() -> None:
    # A bare-address run() dials type(self)._remote_env_cls; a backend Model that
    # leaves it unset silently falls back to the numpy RemoteEnv (wrong value type
    # and a forced numpy dependency). Pin that each backend wires its own.
    from rlmesh import _native

    assert _native.Model._remote_env_cls is _native.RemoteEnv

    for module_name in ("rlmesh.numpy", "rlmesh.jax", "rlmesh.torch"):
        module = pytest.importorskip(module_name)
        assert module.Model._remote_env_cls is module.RemoteEnv, module_name


def test_reject_vector_env_rejects_num_envs_gt_one() -> None:
    from typing import Any, cast

    from rlmesh._models._eval import _reject_vector_env

    class FourEnvs:
        num_envs = 4

    with pytest.raises(ValueError, match="num_envs=4"):
        _reject_vector_env(cast(Any, FourEnvs()))

    class OneEnv:
        num_envs = 1

    _reject_vector_env(cast(Any, OneEnv()))  # single env is fine
    _reject_vector_env(None)  # an env with no contract is fine


def test_no_adapter_sentinel_skips_adapter_resolution_for_tagged_env() -> None:
    from types import SimpleNamespace

    import rlmesh
    import rlmesh.adapters as adapt
    from rlmesh._models._eval import _resolve_adapter

    tags = adapt.EnvTags(observation={}, action=adapt.ActionLayout())
    contract = SimpleNamespace(metadata=tags.to_metadata())

    assert _resolve_adapter(rlmesh.NO_ADAPTER, cast(Any, contract), False) is None


def test_spec_none_rejects_tagged_env_with_no_adapter_hint() -> None:
    from types import SimpleNamespace

    import rlmesh.adapters as adapt
    from rlmesh._models._eval import _resolve_adapter

    tags = adapt.EnvTags(observation={}, action=adapt.ActionLayout())
    contract = SimpleNamespace(metadata=tags.to_metadata())

    with pytest.raises(adapt.AdapterResolutionError, match="spec=NO_ADAPTER"):
        _resolve_adapter(None, cast(Any, contract), False)


def test_invalid_model_spec_mentions_no_adapter_sentinel() -> None:
    from types import SimpleNamespace

    import rlmesh.adapters as adapt
    from rlmesh._models._eval import _resolve_adapter

    with pytest.raises(adapt.AdapterResolutionError, match="ModelSpec or NO_ADAPTER"):
        _resolve_adapter(object(), cast(Any, SimpleNamespace(metadata={})), False)


def test_rekey_value_converts_between_framework_backends() -> None:
    torch = pytest.importorskip("torch")
    import numpy as np
    from rlmesh._value_conversion import rekey_value
    from rlmesh.numpy import _numpy_bridge
    from rlmesh.torch import _torch_bridge

    payload = {"image": np.zeros((2, 2), dtype="float32")}
    framework = cast(
        "dict[str, object]",
        rekey_value(payload, source_bridge=_numpy_bridge, target_bridge=_torch_bridge),
    )
    assert isinstance(framework["image"], torch.Tensor)

    np_action = rekey_value(
        torch.tensor([1.0, 2.0]),
        source_bridge=_torch_bridge,
        target_bridge=_numpy_bridge,
    )
    assert isinstance(np_action, np.ndarray)
    assert np_action.tolist() == [1.0, 2.0]


def test_adapted_run_uses_env_bridge_for_adapter_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import rlmesh._models._eval as eval_mod
    from rlmesh.types import Value

    class EnvBridge:
        name = "env"

        def ensure_available(self) -> None:
            return None

        def encode(self, value: object) -> Value:
            assert value == "env-obs"
            return "canonical-obs"

        def decode(self, value: object) -> object:
            assert value == "canonical-action"
            return "env-action"

    class ModelBridge:
        name = "model"

        def ensure_available(self) -> None:
            return None

        def encode(self, value: object) -> Value:
            assert value == "model-action"
            return "canonical-model-action"

        def decode(self, value: object) -> object:
            assert value == {"payload": "canonical-obs"}
            return {"payload": "model-obs"}

    env_bridge = EnvBridge()
    model_bridge = ModelBridge()
    seen: dict[str, object] = {}

    class Adapter:
        is_stateful = False

        def reset(self) -> None:
            seen["reset"] = True

        def transform_obs_value(
            self,
            raw_obs: object,
            *,
            input_bridge: object | None = None,
            custom_bridge: object | None = None,
        ) -> object:
            seen["input_bridge"] = input_bridge
            seen["custom_bridge"] = custom_bridge
            assert input_bridge is env_bridge
            assert custom_bridge is env_bridge
            assert raw_obs == "env-obs"
            return {"payload": env_bridge.encode(raw_obs)}

        def transform_action_value(
            self,
            raw_action: object,
            *,
            action_bridge: object | None = None,
        ) -> object:
            seen["action_bridge"] = action_bridge
            assert action_bridge is model_bridge
            assert model_bridge.encode(raw_action) == "canonical-model-action"
            return "canonical-action"

    adapter = Adapter()

    class Client:
        _bridge = env_bridge
        env_contract = SimpleNamespace(num_envs=1, metadata={})

        def reset(self) -> tuple[object, dict[str, object]]:
            return "env-obs", {}

        def step(self, action: object) -> tuple[object, float, bool, bool, dict]:
            seen["step_action"] = action
            return "env-obs", 1.0, True, False, {}

    def predict(payload: object) -> object:
        seen["payload"] = payload
        assert payload == {"payload": "model-obs"}
        return "model-action"

    monkeypatch.setattr(eval_mod, "_resolve_adapter", lambda *_args: adapter)

    result = eval_mod.evaluate(
        predict,
        object(),
        Client(),
        max_episodes=1,
        bridge=model_bridge,
    )

    assert result.num_episodes == 1
    assert seen["reset"] is True
    assert seen["payload"] == {"payload": "model-obs"}
    assert seen["step_action"] == "env-action"


def test_adapted_run_uses_active_bridge_for_bridge_less_native_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import rlmesh
    import rlmesh._models._eval as eval_mod
    from rlmesh._framework_bridge import identity_bridge

    action = rlmesh.Tensor(bytes(range(4)), [4], "uint8")
    seen: dict[str, object] = {}

    class Adapter:
        is_stateful = False

        def reset(self) -> None:
            return None

        def transform_obs_value(
            self,
            raw_obs: object,
            *,
            input_bridge: object | None = None,
            custom_bridge: object | None = None,
        ) -> object:
            assert input_bridge is identity_bridge
            assert custom_bridge is identity_bridge
            assert raw_obs == {"raw": "obs"}
            return {"model": raw_obs}

        def transform_action_value(
            self,
            raw_action: object,
            *,
            action_bridge: object | None = None,
        ) -> object:
            assert action_bridge is identity_bridge
            assert raw_action is action
            return raw_action

    class Client:
        env_contract = SimpleNamespace(num_envs=1, metadata={})

        def reset(self) -> tuple[object, dict[str, object]]:
            return {"raw": "obs"}, {}

        def step(self, step_action: object) -> tuple[object, float, bool, bool, dict]:
            seen["step_action"] = step_action
            return {"raw": "obs"}, 1.0, True, False, {}

    def predict(payload: object) -> object:
        assert payload == {"model": {"raw": "obs"}}
        return action

    monkeypatch.setattr(eval_mod, "_resolve_adapter", lambda *_args: Adapter())

    result = eval_mod.evaluate(
        predict,
        object(),
        Client(),
        max_episodes=1,
        bridge=identity_bridge,
    )

    assert result.num_episodes == 1
    assert seen["step_action"] is action


def _address_run_calls(*, close_env: bool) -> tuple[Any, list[str]]:
    from rlmesh._models._eval import evaluate

    calls: list[str] = []

    class FakeRemoteEnv:
        env_contract = None

        def __init__(self, address: str) -> None:
            self.address = address

        def reset(self, seed: object = None) -> tuple[object, dict[str, object]]:
            return 0, {}

        def step(self, action: object) -> tuple[object, float, bool, bool, dict]:
            return 0, 0.0, True, False, {}

        def shutdown(self, reason: str = "owner shutdown") -> bool:
            calls.append("shutdown")
            return True

        def close(self) -> None:
            calls.append("close")

    result = evaluate(
        lambda obs: 0,
        None,
        "tcp://env:9000",
        max_episodes=1,
        close_env=close_env,
        remote_env_cls=FakeRemoteEnv,
    )
    return result, calls


def test_address_run_honors_close_env_when_loop_owns_the_client() -> None:
    # close_env is the caller's explicit opt-in to stop the env it asked us to
    # dial: even when the loop owns the (self-dialed) client, close_env=True must
    # issue the owner-level shutdown() before detaching the client.
    result, calls = _address_run_calls(close_env=True)
    assert result.num_episodes == 1
    assert calls == ["shutdown", "close"]


def test_address_run_only_detaches_when_close_env_is_false() -> None:
    # Without close_env we must never shut down a possibly-shared remote env —
    # only detach the borrowed client connection.
    result, calls = _address_run_calls(close_env=False)
    assert result.num_episodes == 1
    assert calls == ["close"]
    assert "shutdown" not in calls


def test_split_chunk_stays_in_framework_and_does_not_force_numpy() -> None:
    # Regression: a chunked torch/jax DEVICE tensor must split along its leading
    # axis WITHOUT a numpy conversion (np.asarray on a cuda tensor raises). The
    # split uses iteration so each per-step leaf stays a framework tensor, bridged
    # exactly like the non-chunked path and the serve path.
    from rlmesh._models._chunk import split_chunk

    class FakeDeviceTensor:
        def __init__(self, rows: list[Any]) -> None:
            self._rows = rows

        def __iter__(self) -> Any:
            return iter(self._rows)

        def __array__(self, *args: Any, **kwargs: Any) -> Any:
            raise TypeError("can't convert a device tensor to numpy")

    rows = [object(), object()]
    assert split_chunk(FakeDeviceTensor(rows)) == rows


def test_split_chunk_degenerate_scalar_and_mapping_are_single_step() -> None:
    from rlmesh._models._chunk import split_chunk

    # A bare scalar (not iterable) and a structured (dict) action are each one
    # step, matching the native split_chunk's single-step leaf handling.
    assert split_chunk(5.0) == [5.0]
    assert split_chunk({"a": 1}) == [{"a": 1}]


def test_split_chunk_treats_text_as_a_single_step_like_the_native_side() -> None:
    from rlmesh._models._chunk import split_chunk

    # A str/bytes action is iterable, but the native split_chunk treats a text
    # leaf as ONE step (not per-character); the Python side must match so serve
    # and run(env) never disagree for a text-action model.
    assert split_chunk("go") == ["go"]
    assert split_chunk(b"go") == [b"go"]
    assert split_chunk(bytearray(b"go")) == [bytearray(b"go")]
