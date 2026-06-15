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
