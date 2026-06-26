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

    with pytest.raises(TypeError, match="remote-env object, or an address string"):
        _connect(object(), "", None)


def test_local_contract_builds_from_a_bare_env_without_forwarding_lookups() -> None:
    # A local env (no env_contract) yields a contract synthesized from its spaces
    # and metadata; num_envs must NOT be probed via __getattr__, so a gymnasium
    # wrapper does not emit its deprecated attribute-forwarding warning.
    from rlmesh._models._eval import _local_contract

    forwarded: list[str] = []

    class ForwardingEnv:
        observation_space = "obs"
        action_space = "act"

        def __init__(self) -> None:
            self.metadata = {"k": "v"}

        def __getattr__(self, name: str) -> object:  # gymnasium warns in here
            forwarded.append(name)
            raise AttributeError(name)

        def reset(self) -> None: ...
        def step(self, action: object) -> None: ...

    contract = _local_contract(ForwardingEnv())
    assert contract.num_envs == 1
    assert contract.observation_space == "obs"
    assert contract.metadata == {"k": "v"}
    assert forwarded == []  # nothing reached the forwarding __getattr__


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


def test_random_sample_samples_action_space_and_skips_adapter_on_tagged_env() -> None:
    # A RANDOM_SAMPLE baseline adapts nothing, so it must drive a tagged env without
    # raising AdapterResolutionError and return the env's action_space.sample().
    from types import SimpleNamespace

    import rlmesh
    import rlmesh.adapters as adapt

    tags = adapt.EnvTags(observation={}, action=adapt.ActionLayout())
    sentinel = object()

    class Env:
        observation_space = "obs"
        action_space = SimpleNamespace(sample=lambda: sentinel)
        metadata = tags.to_metadata()

        def reset(self, *, seed: object = None) -> tuple[object, dict[str, object]]:
            return "o", {}

        def step(self, action: object) -> tuple[object, float, bool, bool, dict]:
            return "o", 1.0, True, False, {}

    sess = rlmesh.session(rlmesh.RANDOM_SAMPLE, Env())
    obs, _ = sess.reset()
    assert sess.predict(obs) is sentinel


def test_invalid_model_spec_mentions_no_adapter_sentinel() -> None:
    from types import SimpleNamespace

    import rlmesh.adapters as adapt
    from rlmesh._models._eval import _resolve_adapter

    with pytest.raises(adapt.AdapterResolutionError, match="ModelSpec or NO_ADAPTER"):
        _resolve_adapter(object(), cast(Any, SimpleNamespace(metadata={})), False)


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

    result = eval_mod.Session(
        predict=predict,
        spec=object(),
        env=Client(),
        bridge=model_bridge,
    ).run(max_episodes=1)

    assert result.num_episodes == 1
    assert seen["reset"] is True
    assert seen["payload"] == {"payload": "model-obs"}
    assert seen["step_action"] == "env-action"


def test_adapted_run_defaults_a_bridge_less_env_to_the_numpy_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bridge-less env encodes obs with the *numpy* bridge, not the model's.

    The env-side bridge tracks the env's native value type, never the model's
    framework. A raw local env (a gym env hands numpy) exposes no ``_bridge``, so
    the env side must default to numpy -- otherwise a torch/jax model rejects the
    numpy obs at the native plan. Regression for the cross-framework local-driving
    crash: a distinct model bridge must NOT leak onto the env side.
    """
    from types import SimpleNamespace

    import rlmesh._models._eval as eval_mod
    from rlmesh._value_conversion import identity_bridge
    from rlmesh.adapters.adapter import _numpy_value_bridge

    numpy_bridge = _numpy_value_bridge()
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
            seen["input_bridge"] = input_bridge
            seen["custom_bridge"] = custom_bridge
            return {"model": raw_obs}

        def transform_action_value(
            self,
            raw_action: object,
            *,
            action_bridge: object | None = None,
        ) -> object:
            seen["action_bridge"] = action_bridge
            return raw_action

    class Client:
        # A native handle (has env_contract) but, like a raw local env, no _bridge.
        env_contract = SimpleNamespace(num_envs=1, metadata={})

        def reset(self) -> tuple[object, dict[str, object]]:
            return {"raw": "obs"}, {}

        def step(self, step_action: object) -> tuple[object, float, bool, bool, dict]:
            seen["step_action"] = step_action
            return {"raw": "obs"}, 1.0, True, False, {}

    monkeypatch.setattr(eval_mod, "_resolve_adapter", lambda *_args: Adapter())

    result = eval_mod.Session(
        predict=lambda payload: "model-action",
        spec=object(),
        env=Client(),
        bridge=identity_bridge,  # the model bridge -- must not become the env bridge
    ).run(max_episodes=1)

    assert result.num_episodes == 1
    # Env side is numpy (the env's native default), regardless of the model bridge.
    assert seen["input_bridge"] is numpy_bridge
    assert seen["custom_bridge"] is numpy_bridge
    # The action's model-side encode still uses the model bridge (unchanged).
    assert seen["action_bridge"] is identity_bridge


def _address_run_calls(*, close_env: bool) -> tuple[Any, list[str]]:
    from rlmesh._models._eval import Session

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

    result = Session(
        predict=lambda obs: 0,
        spec=None,
        env="tcp://env:9000",
        close_env=close_env,
        remote_env_cls=FakeRemoteEnv,
    ).run(max_episodes=1)
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
