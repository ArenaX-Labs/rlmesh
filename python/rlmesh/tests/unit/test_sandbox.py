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


def test_sandbox_options_set_rlmesh_package_and_params_are_the_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    stopped: list[str] = []

    _patch_start(monkeypatch, captured)
    _patch_stop(monkeypatch, lambda *, container_id: stopped.append(container_id))
    monkeypatch.setattr(native, "PyEnvClient", _OkClient)

    # Build infra rides in options=; everything else is the make-binding (**params).
    with rlmesh.SandboxEnv(
        "CartPole-v1",
        options=rlmesh.SandboxOptions(rlmesh_package="local"),
        render_mode="rgb_array",
    ):
        pass

    assert captured["rlmesh_package"] == "local"
    assert json.loads(cast(str, captured["kwargs_json"])) == {
        "render_mode": "rgb_array"
    }
    assert stopped == ["container-1"]


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
def test_sandbox_options_reject_bare_str_packages_imports(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sandbox,
        "_sandbox_start_env",
        lambda *_args, **_kwargs: pytest.fail("sandbox should not start"),
    )

    with pytest.raises(TypeError, match=rf"{field}= expects a sequence of strings"):
        options = rlmesh.SandboxOptions(**{field: "ale-py"})  # type: ignore[arg-type]
        rlmesh.SandboxEnv("CartPole-v1", options=options)


@pytest.mark.parametrize("field", ["packages", "imports"])
def test_sandbox_options_accept_string_sequence(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    stopped: list[str] = []

    _patch_start(monkeypatch, captured)
    _patch_stop(monkeypatch, lambda *, container_id: stopped.append(container_id))
    monkeypatch.setattr(native, "PyEnvClient", _OkClient)

    options = rlmesh.SandboxOptions(**{field: ["ale-py"]})  # type: ignore[arg-type]
    with rlmesh.SandboxEnv("CartPole-v1", options=options):
        pass

    assert captured[field] == ["ale-py"]


def test_start_sandbox_gym_path_forwards_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On the gym/hf source-string path, imports= (from options) is forwarded.
    captured: dict[str, object] = {}

    def start_result(*_args: object, **kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return _start_result()

    monkeypatch.setattr(sandbox, "_sandbox_start_env", start_result)

    sandbox.start_sandbox_container(
        "CartPole-v1",
        options=rlmesh.SandboxOptions(imports=["ale_py"]),
        num_envs=1,
        vectorization_mode=None,
        binding={},
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


def test_sandbox_model_source_resolution() -> None:
    # A model is always a prebuilt image: bare tags and explicit schemes resolve;
    # a non-string source and an empty/scheme-only tag are rejected.
    from rlmesh._sandbox._model import SandboxModel

    assert SandboxModel("smolvla:latest")._image == "smolvla:latest"
    assert SandboxModel("policy/run-test")._image == "policy/run-test"
    assert SandboxModel("image://m:latest")._image == "m:latest"
    assert SandboxModel("docker://m:latest")._image == "m:latest"

    with pytest.raises(TypeError, match="prebuilt image source"):
        SandboxModel(123)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="image tag"):
        SandboxModel("image://")
    # A bare gym env id is never a model image -- reject it before docker run.
    with pytest.raises(ValueError, match="gym env id"):
        SandboxModel("CartPole-v1")


# --- regression: source autodetect + swallowed options ----------------------


def test_gym_module_id_with_colon_routes_to_build_not_docker() -> None:
    # `pkg:Env-v0` is a valid gym module-import id (has a colon); it must build,
    # never be probed as a Docker image. Pure -- the version suffix short-circuits.
    assert sandbox.looks_like_gym_id("my_envs:MyEnv-v0") is True
    assert sandbox._is_image_shaped("my_envs:MyEnv-v0") is False
    kind, ref = sandbox.resolve_source_kind("my_envs:MyEnv-v0")
    assert (kind, ref) == ("build", "my_envs:MyEnv-v0")
    # A real tagged image still resolves image-shaped.
    assert sandbox._is_image_shaped("registry/img:v1.2") is True


def test_top_level_sandbox_option_is_rejected_not_swallowed() -> None:
    # Security/build flags moved to options=; passing them top-level must fail loud
    # rather than vanish into the make-binding (a silent security downgrade).
    for name in ("trust_remote_code", "allow_unpinned_hf", "packages"):
        with pytest.raises(TypeError, match="options=SandboxOptions"):
            sandbox.reject_sandbox_option_params({name: True})
