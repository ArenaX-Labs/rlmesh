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
from rlmesh._sandbox import _sources
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
        self,
        address: str,
        *,
        connect_timeout_seconds: float | None = None,
        request_timeout_seconds: float | None = None,
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
    # Both _stop() and the __init__-failure cleanup (session._attach) call the
    # session-module alias, so one patch covers both paths.
    monkeypatch.setattr(sandbox, "_sandbox_stop_env", stop)


def test_sandbox_cleanup_runs_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped: list[str] = []
    captured: dict[str, object] = {}

    class InterruptingClient:
        def __init__(
            self,
            address: str,
            *,
            connect_timeout_seconds: float,
            request_timeout_seconds: float | None = None,
        ) -> None:
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
        def __init__(
            self,
            address: str,
            *,
            connect_timeout_seconds: float,
            request_timeout_seconds: float | None = None,
        ) -> None:
            _ = address, connect_timeout_seconds, request_timeout_seconds
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

    # Build infra rides in build=; everything else is the make-binding (**params).
    with rlmesh.SandboxEnv(
        "CartPole-v1",
        build=rlmesh.SandboxBuild(rlmesh_package="local"),
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
        build = rlmesh.SandboxBuild(**{field: "ale-py"})  # type: ignore[arg-type]
        rlmesh.SandboxEnv("CartPole-v1", build=build)


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

    build = rlmesh.SandboxBuild(**{field: ["ale-py"]})  # type: ignore[arg-type]
    with rlmesh.SandboxEnv("CartPole-v1", build=build):
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
        build=rlmesh.SandboxBuild(imports=["ale_py"]),
        runtime=None,
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
    assert _sources.looks_like_gym_id("my_envs:MyEnv-v0") is True
    assert _sources._is_image_shaped("my_envs:MyEnv-v0") is False
    kind, ref = _sources.resolve_source_kind("my_envs:MyEnv-v0")
    assert (kind, ref) == ("build", "my_envs:MyEnv-v0")
    # A real tagged image still resolves image-shaped.
    assert _sources._is_image_shaped("registry/img:v1.2") is True


def test_tagless_name_with_local_image_resolves_to_prebuilt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_sources.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(_sources, "docker_image_exists", lambda image: True)
    monkeypatch.setattr(
        _sources, "docker_pull", lambda *_a: pytest.fail("tagless names never pull")
    )
    assert _sources.resolve_source_kind("myenv") == ("prebuilt", "myenv")


def test_tagless_name_without_local_image_names_both_spellings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_sources.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(_sources, "docker_image_exists", lambda image: False)
    monkeypatch.setattr(
        _sources, "docker_pull", lambda *_a: pytest.fail("tagless names never pull")
    )
    with pytest.raises(ValueError) as excinfo:
        _sources.resolve_source_kind("myenv")
    assert "docker://myenv" in str(excinfo.value)
    assert "gym://myenv" in str(excinfo.value)


def test_tagless_name_without_docker_cli_mentions_the_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_sources.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        _sources,
        "docker_image_exists",
        lambda *_a: pytest.fail("no CLI means no probe"),
    )
    with pytest.raises(ValueError) as excinfo:
        _sources.resolve_source_kind("myenv")
    assert "Docker CLI not on PATH" in str(excinfo.value)
    assert "docker://myenv" in str(excinfo.value)
    assert "gym://myenv" in str(excinfo.value)


def test_gym_versioned_names_never_probe_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _sources,
        "docker_image_exists",
        lambda *_a: pytest.fail("gym-versioned names must not probe"),
    )
    monkeypatch.setattr(
        _sources.shutil, "which", lambda _name: pytest.fail("no CLI check either")
    )
    assert _sources.resolve_source_kind("CartPole-v1") == ("build", "CartPole-v1")
    assert _sources.resolve_source_kind("pkg:Env-v0") == ("build", "pkg:Env-v0")


def test_tagged_and_digest_sources_still_probe_and_pull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_sources.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(_sources, "docker_image_exists", lambda image: False)
    pulled: list[str] = []
    monkeypatch.setattr(
        _sources, "docker_pull", lambda image: pulled.append(image) or (True, "")
    )
    assert _sources.resolve_source_kind("repo/img:tag") == ("prebuilt", "repo/img:tag")
    assert _sources.resolve_source_kind("img@sha256:abcd") == (
        "prebuilt",
        "img@sha256:abcd",
    )
    assert pulled == ["repo/img:tag", "img@sha256:abcd"]


class _ManualSession(sandbox.SandboxLifecycle):
    """A lifecycle-only session with a hand-wired sandbox, for _attach tests."""

    def __init__(self) -> None:
        self._source = "img:1"
        self._closed = False
        self.sandbox = sandbox.SandboxInfo(
            requested_source="img:1",
            resolved_source="docker://img:1",
            address="127.0.0.1:1",
            container_id="container-1",
        )

    def _initialize(self, *args: object, **kwargs: object) -> None:
        raise OSError("connection refused")

    def _detach(self) -> None:
        pass


def test_attach_failure_with_failing_stop_leaves_session_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the cleanup stop fails after a failed attach, the session must NOT
    be marked closed -- close() can then retry the stop instead of leaking a
    running container for the process lifetime (mirrors _stop)."""
    stop_calls: list[str] = []

    def flaky_stop(*, container_id: str) -> None:
        stop_calls.append(container_id)
        if len(stop_calls) == 1:
            raise RuntimeError("docker daemon unavailable")

    _patch_stop(monkeypatch, flaky_stop)
    monkeypatch.setattr(sandbox, "_container_running", lambda _cid: False)
    monkeypatch.setattr(sandbox, "_container_logs", lambda _cid: "boom")

    session = _ManualSession()
    with pytest.raises(RuntimeError, match="exited before becoming ready"):
        session._attach(connect_timeout_seconds=1.0)

    assert session._closed is False
    assert stop_calls == ["container-1"]

    session.close()
    assert session._closed is True
    assert stop_calls == ["container-1", "container-1"]


def test_attach_reports_inspect_failure_distinctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inspect failure (daemon busy) must not be misreported as the container
    having exited with empty logs."""
    _patch_stop(monkeypatch, lambda *, container_id: None)
    monkeypatch.setattr(sandbox, "_container_running", lambda _cid: None)

    session = _ManualSession()
    with pytest.raises(RuntimeError, match="could not inspect sandbox container"):
        session._attach(connect_timeout_seconds=1.0)


def test_attach_caps_retry_timeout_at_remaining_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first dial gets the full timeout; each retry's native timeout is
    capped at the remaining deadline so the worst case never doubles it."""
    timeouts: list[float] = []

    class FlakyClient:
        calls = 0

        def __init__(
            self,
            address: str,
            *,
            connect_timeout_seconds: float | None = None,
            request_timeout_seconds: float | None = None,
        ) -> None:
            assert connect_timeout_seconds is not None
            timeouts.append(connect_timeout_seconds)
            self._address = address
            type(self).calls += 1
            if type(self).calls == 1:
                raise OSError("connection refused")

        def address(self) -> str:
            return self._address

        def handshake(self) -> _Contract:
            return _Contract()

        def close(self) -> None:
            pass

    FlakyClient.calls = 0
    _patch_start(monkeypatch)
    _patch_stop(monkeypatch, lambda *, container_id: None)
    monkeypatch.setattr(sandbox, "_container_running", lambda _cid: True)
    monkeypatch.setattr(native, "PyEnvClient", FlakyClient)

    with rlmesh.SandboxEnv("CartPole-v1"):
        pass

    assert len(timeouts) == 2
    assert timeouts[0] == sandbox.SANDBOX_REMOTE_CONNECT_TIMEOUT_SECONDS
    assert 0 < timeouts[1] < timeouts[0]


def test_prebuilt_source_rejects_security_build_options() -> None:
    """trust_remote_code/allow_unpinned_hf are trust grants; setting one on a
    prebuilt (never-built) source is a hard error, not a silent drop."""
    for name in ("trust_remote_code", "allow_unpinned_hf"):
        with pytest.raises(ValueError, match=name):
            sandbox._warn_ignored_build_options(
                rlmesh.SandboxBuild(**{name: True})  # type: ignore[arg-type]
            )


def test_binding_json_names_unserializable_param() -> None:
    with pytest.raises(TypeError, match="construction param 'camera'"):
        sandbox.binding_json({"camera": object()})


def test_sandbox_vector_env_rejects_unknown_vectorization_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sandbox,
        "_sandbox_start_env",
        lambda *_args, **_kwargs: pytest.fail("sandbox should not start"),
    )
    with pytest.raises(ValueError, match="vectorization_mode"):
        rlmesh.SandboxVectorEnv("CartPole-v1", 2, vectorization_mode="Async")


def test_sandbox_vector_env_default_mode_is_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default vectorization_mode is None (auto), matching the serve-CLI
    surface instead of silently pinning the sandbox to sync."""
    captured: dict[str, object] = {}
    _patch_start(monkeypatch, captured)
    _patch_stop(monkeypatch, lambda *, container_id: None)

    class _VecContract:
        num_envs = 2

    class _OkVectorClient:
        def __init__(
            self,
            address: str,
            *,
            connect_timeout_seconds: float | None = None,
            request_timeout_seconds: float | None = None,
        ) -> None:
            self._address = address

        def address(self) -> str:
            return self._address

        def handshake(self) -> _VecContract:
            return _VecContract()

        def num_envs(self) -> int:
            return 2

        def close(self) -> None:
            pass

    monkeypatch.setattr(native, "PyVectorEnvClient", _OkVectorClient)

    with rlmesh.SandboxVectorEnv("CartPole-v1", 2):
        pass

    assert captured["vectorization_mode"] is None


def test_top_level_sandbox_option_is_rejected_not_swallowed() -> None:
    # Build/runtime flags live in build=/runtime=; passing them top-level must fail
    # loud rather than vanish into the make-binding (a silent security/config drop).
    for name in (
        "trust_remote_code",
        "allow_unpinned_hf",
        "packages",
        "gpus",
        "volumes",
    ):
        with pytest.raises(TypeError, match="build=SandboxBuild"):
            sandbox.reject_sandbox_option_params({name: True})
