from __future__ import annotations

from typing import ClassVar

import pytest


def test_sandbox_cleanup_runs_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh import sandbox

    stopped: list[str] = []
    captured: dict[str, object] = {}

    class InterruptingRemote:
        @classmethod
        def _connect_for_sandbox(
            cls,
            address: str,
            *,
            connect_timeout_seconds: float,
        ) -> object:
            captured["address"] = address
            captured["connect_timeout_seconds"] = connect_timeout_seconds
            raise KeyboardInterrupt

    class SandboxUnderTest(sandbox.SandboxSessionBase[object]):
        _remote_env_cls: ClassVar[type[InterruptingRemote]] = InterruptingRemote

    monkeypatch.setattr(sandbox, "_sandbox_start_env", _start_result)
    monkeypatch.setattr(
        sandbox,
        "_sandbox_stop_env",
        lambda *, container_id: stopped.append(container_id),
    )

    with pytest.raises(KeyboardInterrupt):
        SandboxUnderTest("CartPole-v1")

    assert stopped == ["container-1"]
    assert captured == {
        "address": "tcp://127.0.0.1:50051",
        "connect_timeout_seconds": sandbox.SANDBOX_REMOTE_CONNECT_TIMEOUT_SECONDS,
    }


def test_sandbox_cleanup_runs_on_remote_attach_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh import sandbox

    stopped: list[str] = []

    class FailingRemote:
        def __init__(self, address: str) -> None:
            _ = address
            raise RuntimeError("attach failed")

    class SandboxUnderTest(sandbox.SandboxSessionBase[object]):
        _remote_env_cls: ClassVar[type[FailingRemote]] = FailingRemote

    monkeypatch.setattr(sandbox, "_sandbox_start_env", _start_result)
    monkeypatch.setattr(
        sandbox,
        "_sandbox_stop_env",
        lambda *, container_id: stopped.append(container_id),
    )

    with pytest.raises(RuntimeError, match="attach failed"):
        SandboxUnderTest("CartPole-v1")

    assert stopped == ["container-1"]


def _start_result(*_args: object, **_kwargs: object) -> dict[str, str]:
    return {
        "requested_source": "gym://CartPole-v1",
        "resolved_source": "gym://CartPole-v1",
        "address": "tcp://127.0.0.1:50051",
        "container_id": "container-1",
    }
