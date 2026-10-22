"""BYO prebuilt container: ``SandboxModel(image://<tag>)`` runs the tag directly.

A prebuilt image carries its own baked ``runtime.json``; the serve path must run
the tag with no recipe build and WITHOUT ``RLMESH_BOOTSTRAP_JSON`` (which would
override the container's baked spec). The container hosts a session, which
the model serves in.
"""

from __future__ import annotations

from typing import Any

import pytest
import rlmesh
import rlmesh._sandbox._model as model_mod
import rlmesh._sandbox._sources as _sources
import rlmesh._sandbox.session as session_mod


@pytest.fixture(autouse=True)
def _session_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Containers publish on 40000-40003 and serve at once."""
    monkeypatch.setattr(session_mod, "_sandbox_session_ports", lambda: 40000)
    monkeypatch.setattr(model_mod, "wait_serving", lambda *_args: None)


class _StartProc:
    returncode = 0
    stdout = "container-abc\n"
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
    ``docker run -d ...`` returns the container id.
    """

    def fake_run(cmd: list[str], **_kwargs: Any) -> Any:
        if cmd[:3] == ["docker", "image", "inspect"]:
            return _ImageProc()
        if cmd[:3] == ["docker", "run", "-d"]:
            captured["run"] = cmd
            return _StartProc()
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
    # Publishes the session's ports on loopback, and serves in it as model.
    assert "127.0.0.1:40000-40003:40000-40003" in cmd
    assert "RLMESH_NAME=model" in cmd
    assert "--cap-drop" in cmd and "no-new-privileges" in cmd
    assert model.container_id == "container-abc"
    assert model.address == "127.0.0.1:40000/model"


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
