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
import rlmesh._sandbox._sources as _sources
import rlmesh._sandbox.session as session_mod


class _StartProc:
    returncode = 0
    stdout = "container-abc\n"
    stderr = ""


class _PortProc:
    returncode = 0
    stdout = "127.0.0.1:49153\n"
    stderr = ""


class _ImageProc:
    returncode = 0
    stdout = "[]\n"
    stderr = ""


def _serve_dispatch(
    captured: dict[str, list[str]],
) -> Any:
    """Fake ``subprocess.run`` for the serve path: dispatch on argv.

    ``docker image inspect`` reports the image as local (no pre-pull);
    ``docker run -d ...`` returns the container id; ``docker port ...`` returns
    the Docker-assigned host mapping that serve reads back (BUG #4: no host-side
    bind/close, so the published port is read from Docker, not guessed).
    """

    def fake_run(cmd: list[str], **_kwargs: Any) -> Any:
        if cmd[:3] == ["docker", "image", "inspect"]:
            return _ImageProc()
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
    monkeypatch.setattr(_sources.subprocess, "run", _serve_dispatch(captured))

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
    monkeypatch.setattr(_sources.subprocess, "run", _serve_dispatch(captured))

    # serve, with an int count normalized to a string
    model_mod.SandboxModel(
        "image://m:latest", runtime=rlmesh.SandboxRuntime(gpus=2)
    ).serve()
    serve_cmd = captured["run"]
    assert serve_cmd[serve_cmd.index("--gpus") + 1] == "2"


def test_no_gpus_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(_sources.subprocess, "run", _serve_dispatch(captured))
    model_mod.SandboxModel("image://m:latest").serve()
    assert "--gpus" not in captured["run"]


def test_gpus_rejects_empty() -> None:
    with pytest.raises(ValueError, match="gpus="):
        rlmesh.SandboxModel(
            "image://m:latest", runtime=rlmesh.SandboxRuntime(gpus="  ")
        )


def test_image_source_requires_a_tag() -> None:
    with pytest.raises(ValueError, match="image tag"):
        rlmesh.SandboxModel("image://")


def test_resolve_published_port_accepts_ipv6_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _IPv6PortProc:
        returncode = 0
        stdout = "[::]:51000\n"
        stderr = ""

    monkeypatch.setattr(
        _sources.subprocess,
        "run",
        lambda *_a, **_k: _IPv6PortProc(),
    )

    # #14: an IPv6 host binding must parse (mirrors the Rust parser tested
    # against `[::]:51000`), not raise "published no host port". The model path
    # now shares this helper with the env path (session._resolve_published_port).
    assert session_mod._resolve_published_port("container-abc") == 51000
