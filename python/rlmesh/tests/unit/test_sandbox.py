"""Sandbox env lifecycle: a SandboxEnv IS a remote env that owns its container.

These mock the container start (``_sandbox_start_env``) and the client attach
(``PyEnvClient``) so the lifecycle -- start, attach, cleanup-on-failure, close-retry,
and option handling -- is exercised without Docker.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
import rlmesh
import rlmesh._rlmesh as native
from rlmesh._sandbox import env as sandbox_env
from rlmesh._sandbox import session as sandbox


def _start_result(*_args: object, **_kwargs: object) -> dict[str, str]:
    return {
        "requested_source": "gym://CartPole-v1",
        "resolved_source": "gym://CartPole-v1",
        "address": "tcp://127.0.0.1:50051",
        "container_id": "container-1",
    }


class _Contract:
    """Minimal env contract the client handshake returns (single env)."""

    num_envs = 1


class _OkClient:
    """A PyEnvClient that attaches on the first dial."""

    def __init__(
        self, address: str, *, connect_timeout_seconds: float | None = None
    ) -> None:
        self._address = address

    def address(self) -> str:
        return self._address

    def handshake(self) -> _Contract:
        return _Contract()

    def close(self) -> None:
        pass


def _patch_start(
    monkeypatch: pytest.MonkeyPatch, capture: dict[str, object] | None = None
) -> None:
    def start(*_args: object, **kwargs: object) -> dict[str, str]:
        if capture is not None:
            capture.update(kwargs)
        return _start_result()

    monkeypatch.setattr(sandbox, "_sandbox_start_env", start)


def _patch_stop(monkeypatch: pytest.MonkeyPatch, stop: Any) -> None:
    # _stop() uses the session-module alias; the __init__ failure path uses env's.
    monkeypatch.setattr(sandbox, "_sandbox_stop_env", stop)
    monkeypatch.setattr(sandbox_env, "_sandbox_stop_env", stop)


def test_sandbox_cleanup_runs_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped: list[str] = []
    captured: dict[str, object] = {}

    class InterruptingClient:
        def __init__(self, address: str, *, connect_timeout_seconds: float) -> None:
            captured["address"] = address
            captured["connect_timeout_seconds"] = connect_timeout_seconds
            raise KeyboardInterrupt

    _patch_start(monkeypatch)
    _patch_stop(monkeypatch, lambda *, container_id: stopped.append(container_id))
    monkeypatch.setattr(native, "PyEnvClient", InterruptingClient)

    with pytest.raises(KeyboardInterrupt):
        rlmesh.SandboxEnv("CartPole-v1")

    # The started container is stopped before the attach error propagates.
    assert stopped == ["container-1"]
    assert captured["address"] == "tcp://127.0.0.1:50051"
    assert (
        captured["connect_timeout_seconds"]
        == sandbox.SANDBOX_REMOTE_CONNECT_TIMEOUT_SECONDS
    )


def test_sandbox_cleanup_runs_on_remote_attach_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped: list[str] = []

    class FailingClient:
        def __init__(self, address: str, *, connect_timeout_seconds: float) -> None:
            _ = address, connect_timeout_seconds
            raise RuntimeError("attach failed")

    _patch_start(monkeypatch)
    _patch_stop(monkeypatch, lambda *, container_id: stopped.append(container_id))
    monkeypatch.setattr(native, "PyEnvClient", FailingClient)

    with pytest.raises(RuntimeError, match="attach failed"):
        rlmesh.SandboxEnv("CartPole-v1")

    assert stopped == ["container-1"]


def test_sandbox_package_spec_alias_sets_rlmesh_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    stopped: list[str] = []

    _patch_start(monkeypatch, captured)
    _patch_stop(monkeypatch, lambda *, container_id: stopped.append(container_id))
    monkeypatch.setattr(native, "PyEnvClient", _OkClient)

    with rlmesh.SandboxEnv(
        "CartPole-v1",
        package_spec="local",
        render_mode="rgb_array",
    ):
        pass

    assert captured["rlmesh_package"] == "local"
    assert json.loads(cast(str, captured["kwargs_json"])) == {
        "render_mode": "rgb_array"
    }
    assert stopped == ["container-1"]


def test_sandbox_package_spec_alias_rejects_ambiguous_rlmesh_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sandbox,
        "_sandbox_start_env",
        lambda *_args, **_kwargs: pytest.fail("sandbox should not start"),
    )

    with pytest.raises(TypeError, match=r"both rlmesh_package=.*package_spec"):
        rlmesh.SandboxEnv(
            "CartPole-v1",
            rlmesh_package="local",
            package_spec="wheel.whl",
        )


def test_sandbox_retries_close_after_transient_stop_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_calls: list[str] = []

    def flaky_stop(*, container_id: str) -> None:
        stop_calls.append(container_id)
        if len(stop_calls) == 1:
            raise RuntimeError("docker daemon unavailable")

    _patch_start(monkeypatch)
    _patch_stop(monkeypatch, flaky_stop)
    monkeypatch.setattr(native, "PyEnvClient", _OkClient)

    session = rlmesh.SandboxEnv("CartPole-v1")

    # First close attempt fails while stopping the container.
    with pytest.raises(RuntimeError, match="docker daemon unavailable"):
        session.close()
    # Not marked closed, so the container is not leaked -- a retry can stop it.
    assert session._closed is False

    session.close()
    assert session._closed is True
    assert stop_calls == ["container-1", "container-1"]


@pytest.mark.parametrize("field", ["packages", "imports"])
def test_sandbox_rejects_bare_str_packages_imports(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sandbox,
        "_sandbox_start_env",
        lambda *_args, **_kwargs: pytest.fail("sandbox should not start"),
    )

    with pytest.raises(TypeError, match=rf"{field}= expects a sequence of strings"):
        kwargs: dict[str, Any] = {field: "ale-py"}
        rlmesh.SandboxEnv("CartPole-v1", **kwargs)


@pytest.mark.parametrize("field", ["packages", "imports"])
def test_sandbox_accepts_string_sequence_packages_imports(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    stopped: list[str] = []

    _patch_start(monkeypatch, captured)
    _patch_stop(monkeypatch, lambda *, container_id: stopped.append(container_id))
    monkeypatch.setattr(native, "PyEnvClient", _OkClient)

    kwargs: dict[str, Any] = {field: ["ale-py"]}
    with rlmesh.SandboxEnv("CartPole-v1", **kwargs):
        pass

    assert captured[field] == ["ale-py"]


def test_start_sandbox_gym_path_forwards_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On the gym/hf source-string path, imports= is forwarded to _sandbox_start_env.
    captured: dict[str, object] = {}

    def start_result(*_args: object, **kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return _start_result()

    monkeypatch.setattr(sandbox, "_sandbox_start_env", start_result)

    sandbox.start_sandbox_container(
        "CartPole-v1",
        base_image=None,
        rlmesh_package=None,
        packages=None,
        imports=["ale_py"],
        trust_remote_code=False,
        allow_unpinned_hf=False,
        num_envs=1,
        vectorization_mode=None,
        build_memory=None,
        task=None,
        config=None,
        capabilities=None,
        override=None,
        cwd=None,
        repo_root=None,
        gym_make_kwargs={},
    )

    assert captured["imports"] == ["ale_py"]
    assert "recipe_json" not in captured


def test_sandbox_model_shutdown_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SandboxModel.shutdown() must be safe to call more than once: an explicit
    # shutdown() followed by __exit__ (or __del__) must not re-stop an already
    # stopped container. Bypass __init__ (which starts a real container).
    from rlmesh._sandbox._model import SandboxModel

    stops: list[str] = []
    monkeypatch.setattr(
        native, "sandbox_stop_env", lambda *, container_id: stops.append(container_id)
    )

    model = object.__new__(SandboxModel)
    model._address = "0.0.0.0:50051"
    model._container_id = "container-x"
    model._closed = False

    with model:
        model.shutdown()  # explicit early stop
    model.shutdown()  # __exit__ already stopped; these must be no-ops
    assert stops == ["container-x"]


def test_sandbox_model_requires_image_source() -> None:
    # SandboxModel is image:// only; a non-image source is rejected.
    from rlmesh._sandbox._model import SandboxModel

    with pytest.raises(TypeError, match="prebuilt image source"):
        SandboxModel("policy/run-test")
