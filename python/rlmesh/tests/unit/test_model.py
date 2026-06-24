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


def test_rekey_value_converts_between_framework_backends() -> None:
    torch = pytest.importorskip("torch")
    import numpy as np
    from rlmesh._value_conversion import rekey_value
    from rlmesh.numpy import _numpy_bridge
    from rlmesh.torch import _torch_bridge

    payload = {"image": np.zeros((2, 2), dtype="float32")}
    framework = cast(
        "dict[str, object]",
        rekey_value(
            payload, source_bridge=_numpy_bridge, target_bridge=_torch_bridge
        ),
    )
    assert isinstance(framework["image"], torch.Tensor)

    np_action = rekey_value(
        torch.tensor([1.0, 2.0]),
        source_bridge=_torch_bridge,
        target_bridge=_numpy_bridge,
    )
    assert isinstance(np_action, np.ndarray)
    assert np_action.tolist() == [1.0, 2.0]


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
