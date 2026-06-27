"""Run-prebuilt source resolution (§7.5) and the binding injected as env vars.

These mock Docker (``subprocess``/image probes) so the resolution table and the
``RLMESH_MAKE_KWARGS`` injection are exercised without a daemon.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import rlmesh
from rlmesh._sandbox import _model as model_mod
from rlmesh._sandbox import session as sandbox

# --- resolve_source_kind -----------------------------------------------------


def test_gym_and_explicit_schemes_skip_docker_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A gym id / gym:// / hf:// never probes Docker; explicit docker://image://
    # resolves to prebuilt without a probe either.
    monkeypatch.setattr(
        sandbox,
        "docker_image_exists",
        lambda *_a: pytest.fail("must not probe Docker"),
    )

    assert sandbox.resolve_source_kind("CartPole-v1") == ("build", "CartPole-v1")
    assert sandbox.resolve_source_kind("gym://Foo-v0") == ("build", "gym://Foo-v0")
    assert sandbox.resolve_source_kind("ALE/Pong-v5") == ("build", "ALE/Pong-v5")
    assert sandbox.resolve_source_kind("docker://lib:1") == ("prebuilt", "lib:1")
    assert sandbox.resolve_source_kind("image://lib:1") == ("prebuilt", "lib:1")


def test_bare_image_tag_resolves_to_local_prebuilt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sandbox, "docker_image_exists", lambda image: True)
    monkeypatch.setattr(
        sandbox, "docker_pull", lambda *_a: pytest.fail("local hit, no pull")
    )
    assert sandbox.resolve_source_kind("libero:latest") == ("prebuilt", "libero:latest")


def test_bare_image_tag_pulls_when_not_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sandbox, "docker_image_exists", lambda image: False)
    pulled: list[str] = []
    monkeypatch.setattr(
        sandbox, "docker_pull", lambda image: pulled.append(image) or True
    )
    assert sandbox.resolve_source_kind("repo/img:tag") == ("prebuilt", "repo/img:tag")
    assert pulled == ["repo/img:tag"]


def test_bare_image_tag_not_found_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "docker_image_exists", lambda image: False)
    monkeypatch.setattr(sandbox, "docker_pull", lambda image: False)
    with pytest.raises(ValueError, match="not found locally or pullable"):
        sandbox.resolve_source_kind("nope:latest")


# --- prebuilt_run_cmd hardening + binding ------------------------------------


def test_prebuilt_run_cmd_is_hardened_with_image_last() -> None:
    cmd = sandbox.prebuilt_run_cmd(
        "img:1",
        env_vars={"RLMESH_MAKE_KWARGS": '{"suite": "a"}'},
        gpus="2",
        container_port=50051,
        owner_pid=123,
        owner_pid_ns=None,
    )
    assert cmd[:3] == ["docker", "run", "-d"]
    assert "--cap-drop" in cmd and "ALL" in cmd
    assert "no-new-privileges" in cmd
    assert cmd[cmd.index("--gpus") + 1] == "2"
    assert cmd[cmd.index("-e") + 1] == 'RLMESH_MAKE_KWARGS={"suite": "a"}'
    assert cmd[-1] == "img:1"  # image always last


# --- start_prebuilt_container injects the binding ----------------------------


def _docker_dispatch(captured: dict[str, list[str]]) -> Any:
    def fake_run(cmd: list[str], **_kwargs: Any) -> Any:
        if cmd[:3] == ["docker", "run", "-d"]:
            captured["run"] = cmd

            class _P:
                returncode = 0
                stdout = "container-9\n"
                stderr = ""

            return _P()
        if cmd[:2] == ["docker", "port"]:

            class _Q:
                returncode = 0
                stdout = "127.0.0.1:49200\n"
                stderr = ""

            return _Q()
        raise AssertionError(f"unexpected docker call: {cmd}")

    return fake_run


def test_start_prebuilt_container_injects_binding_and_reads_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(sandbox, "reap_orphans", lambda: None)
    monkeypatch.setattr(sandbox.subprocess, "run", _docker_dispatch(captured))

    info = sandbox.start_prebuilt_container(
        "libero:latest",
        requested_source="libero:latest",
        binding={"suite": "libero_spatial", "task_id": 3},
        num_envs=4,
        vectorization_mode="sync",
    )

    run_cmd = captured["run"]
    assert run_cmd[-1] == "libero:latest"
    payload = next(
        a[len("RLMESH_MAKE_KWARGS=") :]
        for a in run_cmd
        if a.startswith("RLMESH_MAKE_KWARGS=")
    )
    assert json.loads(payload) == {"suite": "libero_spatial", "task_id": 3}
    # num_envs/mode ride alongside the binding as their own env vars.
    assert any(a.startswith("RLMESH_NUM_ENVS=4") for a in run_cmd)
    assert any(a.startswith("RLMESH_VECTORIZATION_MODE=sync") for a in run_cmd)
    assert info.container_id == "container-9"
    assert info.address == "127.0.0.1:49200"
    assert info.resolved_source == "docker://libero:latest"


# --- SandboxModel forwards **params as the binding ---------------------------


def test_sandbox_model_params_inject_make_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(model_mod, "reap_orphans", lambda: None)
    monkeypatch.setattr(model_mod.subprocess, "run", _docker_dispatch(captured))

    model = rlmesh.SandboxModel(
        "smolvla:latest", checkpoint="lerobot/smolvla_base", dtype="bfloat16"
    )
    model.serve()

    run_cmd = captured["run"]
    assert run_cmd[-1] == "smolvla:latest"
    payload = next(
        a[len("RLMESH_MAKE_KWARGS=") :]
        for a in run_cmd
        if a.startswith("RLMESH_MAKE_KWARGS=")
    )
    assert json.loads(payload) == {
        "checkpoint": "lerobot/smolvla_base",
        "dtype": "bfloat16",
    }


def test_sandbox_model_warns_on_options(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.warns(UserWarning, match="prebuilt image"):
        rlmesh.SandboxModel("m:latest", options=rlmesh.SandboxOptions(base_image="x"))
