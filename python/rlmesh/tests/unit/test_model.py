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
