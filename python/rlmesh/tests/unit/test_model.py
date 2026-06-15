from __future__ import annotations

import pytest


def test_connect_uses_an_env_like_object_directly() -> None:
    from rlmesh.models._eval import _connect

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
    from rlmesh.models._eval import _connect

    with pytest.raises(
        TypeError, match="env object, a remote-env object, or an address"
    ):
        _connect(object(), "", None)


def test_shutdown_passes_a_reason_when_accepted() -> None:
    from rlmesh.models._eval import _shutdown

    calls: list[tuple[object, ...]] = []

    class WithReason:
        def shutdown(self, reason: str) -> None:
            calls.append((reason,))

    _shutdown(WithReason())
    assert calls == [("model run complete",)]


def test_shutdown_falls_back_to_no_arg_shutdown() -> None:
    from rlmesh.models._eval import _shutdown

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

    from rlmesh.models._eval import _reject_vector_env

    class FourEnvs:
        num_envs = 4

    with pytest.raises(ValueError, match="num_envs=4"):
        _reject_vector_env(cast(Any, FourEnvs()))

    class OneEnv:
        num_envs = 1

    _reject_vector_env(cast(Any, OneEnv()))  # single env is fine
    _reject_vector_env(None)  # an env with no contract is fine


def test_to_framework_rekeys_adapter_numpy_payload() -> None:
    torch = pytest.importorskip("torch")
    import numpy as np
    from rlmesh.models._eval import _to_framework, _to_numpy
    from rlmesh.numpy import _numpy_bridge
    from rlmesh.torch import _torch_bridge

    payload = {"image": np.zeros((2, 2), dtype="float32")}
    framework = _to_framework(payload, _torch_bridge)
    assert isinstance(framework["image"], torch.Tensor)

    # A numpy model already matches, so the round-trip is skipped entirely.
    assert _to_framework(payload, _numpy_bridge) is payload
    assert _to_framework(payload, None) is payload

    # The action the model returns is converted back to numpy for the adapter.
    np_action = _to_numpy(torch.tensor([1.0, 2.0]), _torch_bridge)
    assert isinstance(np_action, np.ndarray)
    assert np_action.tolist() == [1.0, 2.0]
