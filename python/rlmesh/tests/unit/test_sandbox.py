from __future__ import annotations

import json
from typing import Any, ClassVar, cast

import pytest


def test_sandbox_cleanup_runs_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh._sandbox import session as sandbox

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
    from rlmesh._sandbox import session as sandbox

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
    from rlmesh._sandbox import session as sandbox

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
    from rlmesh._sandbox import session as sandbox

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
    from rlmesh._sandbox import session as sandbox

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
    from rlmesh._sandbox import session as sandbox

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
    from rlmesh._sandbox import session as sandbox

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


def test_sandbox_model_shutdown_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SandboxModel.shutdown() must be safe to call more than once: an explicit
    # shutdown() followed by __exit__ (or __del__) must not re-stop an already
    # stopped container. Bypass __init__ (which starts a real container).
    import rlmesh._rlmesh as native
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


def test_start_sandbox_gym_path_forwards_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On the gym/hf source-string path, imports= is still forwarded to
    # _sandbox_start_env byte-identically to before (the recipe-merge path does not
    # apply to a non-recipe source).
    from rlmesh._sandbox import session as sandbox

    captured: dict[str, object] = {}

    def start_result(*_args: object, **kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return _start_result()

    monkeypatch.setattr(sandbox, "_sandbox_start_env", start_result)

    sandbox._start_sandbox(
        "CartPole-v1",
        base_image=None,
        rlmesh_package=None,
        packages=None,
        imports=["ale_py"],
        trust_remote_code=False,
        allow_unpinned_hf=False,
        num_envs=1,
        vectorization_mode=None,
        gym_make_kwargs={},
    )

    assert captured["imports"] == ["ale_py"]
    assert "recipe_json" not in captured


def _start_result(*_args: object, **_kwargs: object) -> dict[str, str]:
    return {
        "requested_source": "gym://CartPole-v1",
        "resolved_source": "gym://CartPole-v1",
        "address": "tcp://127.0.0.1:50051",
        "container_id": "container-1",
    }


def test_sandbox_model_requires_image_source() -> None:
    # SandboxModel is image:// only; a non-image source is rejected.
    from rlmesh._sandbox._model import SandboxModel

    with pytest.raises(TypeError, match="prebuilt image source"):
        SandboxModel("policy/run-test")


def test_sandbox_model_image_rejects_artifacts() -> None:
    from rlmesh._sandbox._model import SandboxModel
    from rlmesh._spec._core import ArtifactInput

    with pytest.raises(TypeError, match="does not accept artifacts="):
        SandboxModel(
            "image://m:1",
            artifacts=[ArtifactInput("w", "/w", local_dir="/host")],
        )
