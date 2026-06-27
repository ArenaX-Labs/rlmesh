"""BYO prebuilt container: ``SandboxModel(image://<tag>)`` runs the tag directly.

A prebuilt image carries its own baked ``runtime.json``; the serve path must run
the tag with no recipe build and WITHOUT ``RLMESH_BOOTSTRAP_JSON`` (which would
override the container's baked spec).
"""

from __future__ import annotations

from typing import Any

import pytest
import rlmesh
import rlmesh._sandbox._model as model_mod
import rlmesh._sandbox.session as session_mod


class _StartProc:
    returncode = 0
    stdout = "container-abc\n"
    stderr = ""


class _PortProc:
    returncode = 0
    stdout = "127.0.0.1:49153\n"
    stderr = ""


def _serve_dispatch(
    captured: dict[str, list[str]],
) -> Any:
    """Fake ``subprocess.run`` for the serve path: dispatch on argv.

    ``docker run -d ...`` returns the container id; ``docker port ...`` returns
    the Docker-assigned host mapping that serve reads back (BUG #4: no host-side
    bind/close, so the published port is read from Docker, not guessed).
    """

    def fake_run(cmd: list[str], **_kwargs: Any) -> Any:
        if cmd[:3] == ["docker", "run", "-d"]:
            captured["run"] = cmd
            return _StartProc()
        if cmd[:2] == ["docker", "port"]:
            captured["port"] = cmd
            return _PortProc()
        raise AssertionError(f"unexpected docker call: {cmd}")

    return fake_run


def test_image_source_serve_starts_a_published_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(model_mod.subprocess, "run", _serve_dispatch(captured))

    model = rlmesh.SandboxModel("image://my-model:latest")
    assert model.serve() is model

    cmd = captured["run"]
    assert cmd[:3] == ["docker", "run", "-d"]
    assert cmd[-1] == "my-model:latest"
    # Publishes the serve port on a Docker-assigned host port (BUG #4: no TOCTOU).
    assert f"127.0.0.1:0:{session_mod.CONTAINER_SERVE_PORT}" in cmd
    assert "--cap-drop" in cmd and "no-new-privileges" in cmd
    assert model.container_id == "container-abc"
    # The address is read back from `docker port`, not from a host-side guess.
    assert captured["port"] == [
        "docker",
        "port",
        "container-abc",
        str(session_mod.CONTAINER_SERVE_PORT),
    ]
    assert model.address == "127.0.0.1:49153"


def test_gpus_requests_the_device_on_serve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(model_mod.subprocess, "run", _serve_dispatch(captured))

    # serve, with an int count normalized to a string
    model_mod.SandboxModel("image://m:latest", gpus=2).serve()
    serve_cmd = captured["run"]
    assert serve_cmd[serve_cmd.index("--gpus") + 1] == "2"


def test_no_gpus_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(model_mod.subprocess, "run", _serve_dispatch(captured))
    model_mod.SandboxModel("image://m:latest").serve()
    assert "--gpus" not in captured["run"]


def test_gpus_rejects_empty() -> None:
    with pytest.raises(ValueError, match="gpus="):
        rlmesh.SandboxModel("image://m:latest", gpus="  ")


def test_session_serves_then_binds_to_the_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_mod.subprocess, "run", _serve_dispatch({}))

    captured: dict[str, object] = {}

    class FakePyModelClient:
        def __init__(
            self, address: str, contract: object, *_a: object, **_kw: object
        ) -> None:
            captured["address"] = address
            captured["contract"] = contract

        def address(self) -> str:
            return "127.0.0.1:9"

        def close(self) -> None:
            pass

    import rlmesh._rlmesh as native

    monkeypatch.setattr(native, "PyModelClient", FakePyModelClient)

    sentinel_bridge = object()

    class FakeEnv:
        env_contract = object()
        _bridge = sentinel_bridge

    model = rlmesh.SandboxModel("image://m:latest")
    sess = rlmesh.session(model, FakeEnv())

    assert isinstance(sess, rlmesh.Session)
    # The session is configured from the env's contract and matches its backend.
    assert captured["contract"] is FakeEnv.env_contract
    assert captured["address"] == model.address
    assert sess._bridge is sentinel_bridge
    assert model.address == "127.0.0.1:49153"


def test_image_source_requires_a_tag() -> None:
    with pytest.raises(ValueError, match="image tag"):
        rlmesh.SandboxModel("image://")


class _ExitedProc:
    """Stub for ``docker inspect``/``docker logs`` on a container that exited."""

    returncode = 0
    stdout = "false\n"
    stderr = "boom: model failed to load\n"


def _docker_dispatch(stop_calls: list[str], *, running: bool):
    def fake_run(cmd: list[str], **_kwargs: Any) -> Any:
        if cmd[:3] == ["docker", "run", "-d"]:
            return _StartProc()
        if cmd[:2] == ["docker", "port"]:
            return _PortProc()
        if cmd[:2] == ["docker", "inspect"]:
            proc = _ExitedProc()
            proc.stdout = "true\n" if running else "false\n"
            return proc
        if cmd[:2] == ["docker", "logs"]:
            return _ExitedProc()
        raise AssertionError(f"unexpected docker call: {cmd}")

    return fake_run


def test_session_fails_fast_with_logs_when_container_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_mod.subprocess, "run", _docker_dispatch([], running=False)
    )

    class AlwaysFailingClient:
        def __init__(self, *_a: object, **_kw: object) -> None:
            raise OSError("connection refused")

    import rlmesh._rlmesh as native

    monkeypatch.setattr(native, "PyModelClient", AlwaysFailingClient)

    stopped: list[str] = []
    monkeypatch.setattr(
        native,
        "sandbox_stop_env",
        lambda *, container_id: stopped.append(container_id),
    )

    class FakeEnv:
        env_contract = object()

    model = rlmesh.SandboxModel("image://m:latest")
    with pytest.raises(RuntimeError, match="exited before becoming ready"):
        # connect_timeout is long, but it must fail fast (no 30s wait) on exit.
        model.session(FakeEnv(), connect_timeout_seconds=30.0)

    # #4/#7: the exit is detected and surfaced with the container's recent logs.
    # #3: the started container is stopped before the error propagates.
    assert stopped == ["container-abc"]


def test_session_stops_container_on_missing_env_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_mod.subprocess, "run", _serve_dispatch({}))

    import rlmesh._rlmesh as native

    stopped: list[str] = []
    monkeypatch.setattr(
        native,
        "sandbox_stop_env",
        lambda *, container_id: stopped.append(container_id),
    )

    model = rlmesh.SandboxModel("image://m:latest")
    with pytest.raises(TypeError, match="requires an env client exposing"):
        rlmesh.session(model, object())

    # #3: the container started by serve() is stopped before re-raising.
    assert stopped == ["container-abc"]


class _FakeEnv:
    env_contract = object()


def _patch_ok_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """A PyModelClient that connects on the first dial (no retries)."""

    class OkClient:
        def __init__(
            self, address: str, contract: object, *_a: object, **_kw: object
        ) -> None:
            self._address = address

        def address(self) -> str:
            return self._address

        def close(self) -> None:
            pass

    import rlmesh._rlmesh as native

    monkeypatch.setattr(native, "PyModelClient", OkClient)


def test_session_keeps_owner_alive_so_it_is_not_gc_before_predict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gc
    import weakref

    monkeypatch.setattr(model_mod.subprocess, "run", _serve_dispatch({}))
    _patch_ok_client(monkeypatch)

    stopped: list[str] = []
    monkeypatch.setattr(
        model_mod.SandboxModel,
        "shutdown",
        lambda self: stopped.append(self._container_id),
    )

    # #1: the documented one-liner -- no local ref to the SandboxModel.
    session = rlmesh.session(rlmesh.SandboxModel("image://m:latest"), _FakeEnv())
    owner_ref = weakref.ref(session._owner)
    gc.collect()

    # The session keeps the owner alive, so its container is NOT stopped.
    assert owner_ref() is not None
    assert "container-abc" not in stopped

    # Closing the session stops the container the one-liner started (#1/#8).
    session.close()
    assert "container-abc" in stopped


def test_failed_session_on_reused_handle_does_not_shut_it_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_mod.subprocess, "run", _serve_dispatch({}))
    _patch_ok_client(monkeypatch)

    stopped: list[str] = []
    monkeypatch.setattr(
        model_mod.SandboxModel,
        "shutdown",
        lambda self: stopped.append(self._container_id),
    )

    model = rlmesh.SandboxModel("image://m:latest")
    rlmesh.session(model, _FakeEnv())  # first bind starts the container

    # #7: a second bind that fails must NOT stop a container this call did
    # not start -- the caller is still managing the handle.
    with pytest.raises(TypeError, match="requires an env client exposing"):
        rlmesh.session(model, object())
    assert "container-abc" not in stopped


class _ConfigErrorClient:
    """Mimics PyModelClient rejecting a bad contract before it ever dials."""

    calls = 0

    def __init__(self, *_a: object, **_kw: object) -> None:
        type(self).calls += 1
        raise RuntimeError("env contract missing observation_space")


def test_contract_config_error_fails_fast_without_retrying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_mod.subprocess, "run", _serve_dispatch({}))

    import rlmesh._rlmesh as native

    _ConfigErrorClient.calls = 0
    monkeypatch.setattr(native, "PyModelClient", _ConfigErrorClient)
    monkeypatch.setattr(native, "sandbox_stop_env", lambda *, container_id: None)

    model = rlmesh.SandboxModel("image://m:latest")
    # #8: the deterministic config error propagates as-is, not retried for the
    # whole timeout and masked as "did not become ready".
    with pytest.raises(RuntimeError, match="missing observation_space"):
        model.session(_FakeEnv(), connect_timeout_seconds=30.0)
    assert _ConfigErrorClient.calls == 1


def test_resolve_published_port_accepts_ipv6_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _IPv6PortProc:
        returncode = 0
        stdout = "[::]:51000\n"
        stderr = ""

    monkeypatch.setattr(
        session_mod.subprocess,
        "run",
        lambda *_a, **_k: _IPv6PortProc(),
    )

    # #14: an IPv6 host binding must parse (mirrors the Rust parser tested
    # against `[::]:51000`), not raise "published no host port". The model path
    # now shares this helper with the env path (session._resolve_published_port).
    assert session_mod._resolve_published_port("container-abc") == 51000
