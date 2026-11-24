from __future__ import annotations

import json
import threading
from types import SimpleNamespace
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
        "address": "127.0.0.1:40000/env",
        "container_id": "container-1",
    }


@pytest.fixture(autouse=True)
def _container_port_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_accepts", lambda _address: True)
    monkeypatch.setattr(sandbox, "_answers_v1", lambda _address: False)


class _Contract:
    num_envs = 1


class _OkClient:
    def __init__(
        self,
        address: str,
        *,
        connect_timeout_seconds: float | None = None,
        request_timeout_seconds: float | None = None,
        workflow_edition: str | None = None,
    ) -> None:
        self._address = address

    def address(self) -> str:
        return self._address

    def handshake(self) -> _Contract:
        return _Contract()

    def drives_view(self) -> bool:
        return False

    def env_id(self) -> str:
        return "env-id"

    def edition_cell(self) -> Any:
        return SimpleNamespace(get=lambda: "2026.06")

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
    monkeypatch.setattr(sandbox, "_sandbox_stop_env", stop)


def test_sandbox_cleanup_runs_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped: list[str] = []
    killed: list[bool] = []
    captured: dict[str, object] = {}

    class InterruptingClient:
        def __init__(
            self,
            address: str,
            *,
            connect_timeout_seconds: float,
            request_timeout_seconds: float | None = None,
            workflow_edition: str | None = None,
        ) -> None:
            captured["address"] = address
            captured["connect_timeout_seconds"] = connect_timeout_seconds
            raise KeyboardInterrupt

    def stop(*, container_id: str, kill: bool = False) -> None:
        stopped.append(container_id)
        killed.append(kill)

    _patch_start(monkeypatch)
    _patch_stop(monkeypatch, stop)
    monkeypatch.setattr(native, "PyEnvClient", InterruptingClient)

    with pytest.raises(KeyboardInterrupt):
        rlmesh.SandboxEnv("CartPole-v1")

    assert stopped == ["container-1"]
    assert killed == [True]
    assert captured["address"] == "127.0.0.1:40000/env"


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
            workflow_edition: str | None = None,
        ) -> None:
            _ = address, connect_timeout_seconds, request_timeout_seconds
            raise RuntimeError("attach failed")

    _patch_start(monkeypatch)
    _patch_stop(
        monkeypatch, lambda *, container_id, kill=False: stopped.append(container_id)
    )
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
    _patch_stop(
        monkeypatch, lambda *, container_id, kill=False: stopped.append(container_id)
    )
    monkeypatch.setattr(native, "PyEnvClient", _OkClient)

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

    def flaky_stop(*, container_id: str, kill: bool = False) -> None:
        stop_calls.append(container_id)
        if len(stop_calls) == 1:
            raise RuntimeError("docker daemon unavailable")

    _patch_start(monkeypatch)
    _patch_stop(monkeypatch, flaky_stop)
    monkeypatch.setattr(native, "PyEnvClient", _OkClient)

    session = rlmesh.SandboxEnv("CartPole-v1")

    with pytest.raises(RuntimeError, match="docker daemon unavailable"):
        session.close()
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
    _patch_stop(
        monkeypatch, lambda *, container_id, kill=False: stopped.append(container_id)
    )
    monkeypatch.setattr(native, "PyEnvClient", _OkClient)

    build = rlmesh.SandboxBuild(**{field: ["ale-py"]})  # type: ignore[arg-type]
    with rlmesh.SandboxEnv("CartPole-v1", build=build):
        pass

    assert captured[field] == ["ale-py"]


def test_start_sandbox_gym_path_forwards_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    from rlmesh._sandbox._model import SandboxModel

    stops: list[str] = []
    monkeypatch.setattr(
        native,
        "sandbox_stop_env",
        lambda *, container_id, kill=False: stops.append(container_id),
    )

    model = object.__new__(SandboxModel)
    model._address = "127.0.0.1:40000/model"
    model._container_id = "container-x"
    model._model = None
    model._closed = False
    model._serving = threading.Lock()

    with model:
        model.shutdown()
    model.shutdown()
    assert stops == ["container-x"]


def test_sandbox_model_source_resolution() -> None:
    from rlmesh._sandbox._model import SandboxModel

    assert SandboxModel("smolvla:latest")._image == "smolvla:latest"
    assert SandboxModel("policy/run-test")._image == "policy/run-test"
    assert SandboxModel("image://m:latest")._image == "m:latest"
    assert SandboxModel("docker://m:latest")._image == "m:latest"

    with pytest.raises(TypeError, match="prebuilt image source"):
        SandboxModel(123)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="image tag"):
        SandboxModel("image://")
    with pytest.raises(ValueError, match="gym env id"):
        SandboxModel("CartPole-v1")


def test_gym_module_id_with_colon_routes_to_build_not_docker() -> None:
    assert _sources.looks_like_gym_id("my_envs:MyEnv-v0") is True
    assert _sources._is_image_shaped("my_envs:MyEnv-v0") is False
    kind, ref = _sources.resolve_source_kind("my_envs:MyEnv-v0")
    assert (kind, ref) == ("build", "my_envs:MyEnv-v0")
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


def test_gym_ids_docker_cannot_spell_never_probe_docker(
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
    def __init__(self) -> None:
        self._source = "img:1"
        self._closed = False
        self.sandbox = sandbox.SandboxInfo(
            requested_source="img:1",
            resolved_source="docker://img:1",
            address="127.0.0.1:1/env",
            container_id="container-1",
        )

    def _initialize(self, *args: object, **kwargs: object) -> None:
        raise OSError("connection refused")

    def _detach(self) -> None:
        pass


def test_attach_failure_with_failing_stop_leaves_session_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_calls: list[str] = []

    def flaky_stop(*, container_id: str, kill: bool = False) -> None:
        stop_calls.append(container_id)
        if len(stop_calls) == 1:
            raise RuntimeError("docker daemon unavailable")

    _patch_stop(monkeypatch, flaky_stop)
    monkeypatch.setattr(
        sandbox, "_container_state", lambda _cid: sandbox.ContainerState(False, 3)
    )
    monkeypatch.setattr(sandbox, "_container_logs", lambda _cid: "boom")

    session = _ManualSession()
    with pytest.raises(RuntimeError, match="exited with code 3 during startup"):
        session._attach(connect_timeout_seconds=1.0)

    assert session._closed is False
    assert stop_calls == ["container-1"]

    session.close()
    assert session._closed is True
    assert stop_calls == ["container-1", "container-1"]


def test_a_container_that_exits_during_startup_is_the_error_not_the_dial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = "k" * 52
    docker_calls: list[list[str]] = []

    def docker(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
        docker_calls.append(cmd)
        if cmd[1] == "inspect":
            return SimpleNamespace(returncode=0, stdout="false 1\n", stderr="")
        out = f"listening on 0.0.0.0:40000#{key}/env\n"
        err = "ImportError: libvulkan.so.1: cannot open shared object file\n"
        return SimpleNamespace(returncode=0, stdout=out, stderr=err)

    monkeypatch.setattr(sandbox, "run_docker", docker)
    monkeypatch.setattr(sandbox, "_accepts", lambda _address: True)
    monkeypatch.setattr(sandbox, "_answers_v1", lambda _address: False)
    info = sandbox.SandboxInfo(**_start_result())

    for raised in (
        ConnectionError("timed out after 900ms waiting for the session host"),
        rlmesh.ProtocolException("the session closed"),
    ):

        def dial(_attempt: float, raised: Exception = raised) -> None:
            raise raised

        with pytest.raises(RuntimeError) as failed:
            sandbox.wait_serving(info, 30.0, dial)
        text = str(failed.value)
        assert text.startswith(
            "the env container exited with code 1 during startup ("
        ), text
        assert text.endswith(
            "its last log lines:\nlistening on 0.0.0.0:40000#…/env\n"
            "ImportError: libvulkan.so.1: cannot open shared object file"
        ), text
        assert key not in text and "timed out" not in text, text
        assert failed.value.__cause__ is None and failed.value.__suppress_context__
        assert ["docker", "logs", "--tail", "40"] == docker_calls[-1][:4]


def test_attach_reports_inspect_failure_distinctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_stop(monkeypatch, lambda *, container_id, kill=False: None)
    monkeypatch.setattr(sandbox, "_container_state", lambda _cid: None)

    session = _ManualSession()
    with pytest.raises(RuntimeError, match="could not inspect sandbox container"):
        session._attach(connect_timeout_seconds=1.0)


def test_attach_retries_in_short_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeouts: list[float] = []

    class FlakyClient:
        calls = 0

        def __init__(
            self,
            address: str,
            *,
            connect_timeout_seconds: float | None = None,
            request_timeout_seconds: float | None = None,
            workflow_edition: str | None = None,
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

        def drives_view(self) -> bool:
            return False

        def env_id(self) -> str:
            return "env-id"

        def edition_cell(self) -> Any:
            return SimpleNamespace(get=lambda: "2026.06")

        def close(self) -> None:
            pass

    FlakyClient.calls = 0
    _patch_start(monkeypatch)
    _patch_stop(monkeypatch, lambda *, container_id, kill=False: None)
    monkeypatch.setattr(
        sandbox, "_container_state", lambda _cid: sandbox.ContainerState(True, None)
    )
    monkeypatch.setattr(native, "PyEnvClient", FlakyClient)

    with rlmesh.SandboxEnv("CartPole-v1"):
        pass

    assert timeouts == [1.0, 1.0]


def test_prebuilt_source_rejects_security_build_options() -> None:
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
    captured: dict[str, object] = {}
    _patch_start(monkeypatch, captured)
    _patch_stop(monkeypatch, lambda *, container_id, kill=False: None)

    class _VecContract:
        num_envs = 2

    class _OkVectorClient:
        def __init__(
            self,
            address: str,
            *,
            connect_timeout_seconds: float | None = None,
            request_timeout_seconds: float | None = None,
            workflow_edition: str | None = None,
        ) -> None:
            self._address = address

        def address(self) -> str:
            return self._address

        def handshake(self) -> _VecContract:
            return _VecContract()

        def drives_view(self) -> bool:
            return False

        def env_id(self) -> str:
            return "env-id"

        def edition_cell(self) -> Any:
            return SimpleNamespace(get=lambda: "2026.06")

        def num_envs(self) -> int:
            return 2

        def close(self) -> None:
            pass

    monkeypatch.setattr(native, "PyVectorEnvClient", _OkVectorClient)

    with rlmesh.SandboxVectorEnv("CartPole-v1", 2):
        pass

    assert captured["vectorization_mode"] is None


def test_top_level_sandbox_option_is_rejected_not_swallowed() -> None:
    for name in (
        "trust_remote_code",
        "allow_unpinned_hf",
        "packages",
        "gpus",
        "volumes",
    ):
        with pytest.raises(TypeError, match="build=SandboxBuild"):
            sandbox.reject_sandbox_option_params({name: True})


def test_an_image_built_with_rlmesh_0_1_is_refused_naming_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket
    import threading

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(2)

    def v1_server() -> None:
        for _ in range(2):
            conn, _ = listener.accept()
            with conn:
                if conn.recv(64):
                    conn.sendall(bytes([0, 0, 0, 4, 0, 0, 0, 0, 0]))

    thread = threading.Thread(target=v1_server, daemon=True)
    thread.start()
    monkeypatch.undo()
    info = sandbox.SandboxInfo(
        requested_source="old-model:0.1",
        resolved_source="docker://old-model:0.1",
        address=f"127.0.0.1:{listener.getsockname()[1]}/model",
        container_id="container-1",
    )

    def never(_attempt: float) -> None:
        raise ConnectionError("no session here")

    with pytest.raises(ValueError, match=r"'old-model:0.1' was built with rlmesh 0.1"):
        sandbox.wait_serving(info, 30.0, never)
    thread.join(timeout=5)
    listener.close()


def test_another_container_joins_a_sandbox_at_its_network_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = sandbox.SandboxInfo(**_start_result())
    monkeypatch.setattr(
        sandbox,
        "run_docker",
        lambda argv: SimpleNamespace(returncode=0, stdout="172.17.0.3 \n", stderr=""),
    )
    assert sandbox.peer_session(info) == "172.17.0.3:40000"
    monkeypatch.setattr(
        sandbox,
        "run_docker",
        lambda argv: SimpleNamespace(
            returncode=1, stdout="", stderr="no such container"
        ),
    )
    with pytest.raises(RuntimeError, match="no such container"):
        sandbox.peer_session(info)
