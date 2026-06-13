from __future__ import annotations

import json
from typing import Any, ClassVar, cast

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


def test_sandbox_package_spec_alias_sets_rlmesh_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh import sandbox

    captured: dict[str, object] = {}
    stopped: list[str] = []

    class Remote:
        def __init__(self, address: str) -> None:
            self.address = address

        def close(self) -> None:
            pass

    class SandboxUnderTest(sandbox.SandboxSessionBase[object]):
        _remote_env_cls: ClassVar[type[Remote]] = Remote

    def start_result(*_args: object, **kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return _start_result()

    monkeypatch.setattr(sandbox, "_sandbox_start_env", start_result)
    monkeypatch.setattr(
        sandbox,
        "_sandbox_stop_env",
        lambda *, container_id: stopped.append(container_id),
    )

    with SandboxUnderTest(
        "CartPole-v1",
        package_spec="local",
        render_mode="rgb_array",
    ):
        pass

    assert captured["rlmesh_package"] == "local"
    assert json.loads(cast(str, captured["kwargs_json"])) == {
        "render_mode": "rgb_array",
    }
    assert stopped == ["container-1"]


def test_sandbox_package_spec_alias_rejects_ambiguous_rlmesh_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh import sandbox

    class Remote:
        def __init__(self, address: str) -> None:
            self.address = address

    class SandboxUnderTest(sandbox.SandboxSessionBase[object]):
        _remote_env_cls: ClassVar[type[Remote]] = Remote

    monkeypatch.setattr(
        sandbox,
        "_sandbox_start_env",
        lambda *_args, **_kwargs: pytest.fail("sandbox should not start"),
    )

    with pytest.raises(TypeError, match=r"both rlmesh_package=.*package_spec"):
        SandboxUnderTest(
            "CartPole-v1",
            rlmesh_package="local",
            package_spec="wheel.whl",
        )


def test_sandbox_retries_close_after_transient_stop_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh import sandbox

    stop_calls: list[str] = []

    class Remote:
        def __init__(self, address: str) -> None:
            self.address = address

        def close(self) -> None:
            pass

    class SandboxUnderTest(sandbox.SandboxSessionBase[object]):
        _remote_env_cls: ClassVar[type[Remote]] = Remote

    def flaky_stop(*, container_id: str) -> None:
        stop_calls.append(container_id)
        if len(stop_calls) == 1:
            raise RuntimeError("docker daemon unavailable")

    monkeypatch.setattr(sandbox, "_sandbox_start_env", _start_result)
    monkeypatch.setattr(sandbox, "_sandbox_stop_env", flaky_stop)

    session = SandboxUnderTest("CartPole-v1")

    # First close attempt fails while stopping the container.
    with pytest.raises(RuntimeError, match="docker daemon unavailable"):
        session.close()
    # Session must not be marked closed, so the container is not leaked.
    assert session._closed is False

    # A retry succeeds and stops the container.
    session.close()
    assert session._closed is True
    assert stop_calls == ["container-1", "container-1"]


@pytest.mark.parametrize("field", ["packages", "imports"])
def test_sandbox_rejects_bare_str_packages_imports(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh import sandbox

    class Remote:
        def __init__(self, address: str) -> None:
            self.address = address

    class SandboxUnderTest(sandbox.SandboxSessionBase[object]):
        _remote_env_cls: ClassVar[type[Remote]] = Remote

    monkeypatch.setattr(
        sandbox,
        "_sandbox_start_env",
        lambda *_args, **_kwargs: pytest.fail("sandbox should not start"),
    )

    with pytest.raises(TypeError, match=rf"{field}= expects a sequence of strings"):
        kwargs: dict[str, Any] = {field: "ale-py"}
        SandboxUnderTest("CartPole-v1", **kwargs)


@pytest.mark.parametrize("field", ["packages", "imports"])
def test_sandbox_accepts_string_sequence_packages_imports(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh import sandbox

    captured: dict[str, object] = {}
    stopped: list[str] = []

    class Remote:
        def __init__(self, address: str) -> None:
            self.address = address

        def close(self) -> None:
            pass

    class SandboxUnderTest(sandbox.SandboxSessionBase[object]):
        _remote_env_cls: ClassVar[type[Remote]] = Remote

    def start_result(*_args: object, **kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return _start_result()

    monkeypatch.setattr(sandbox, "_sandbox_start_env", start_result)
    monkeypatch.setattr(
        sandbox,
        "_sandbox_stop_env",
        lambda *, container_id: stopped.append(container_id),
    )

    kwargs: dict[str, Any] = {field: ["ale-py"]}
    with SandboxUnderTest("CartPole-v1", **kwargs):
        pass

    assert captured[field] == ["ale-py"]


def _start_result(*_args: object, **_kwargs: object) -> dict[str, str]:
    return {
        "requested_source": "gym://CartPole-v1",
        "resolved_source": "gym://CartPole-v1",
        "address": "tcp://127.0.0.1:50051",
        "container_id": "container-1",
    }
