"""Run-prebuilt source resolution (§7.5) and the binding injected as env vars.

These mock Docker (``subprocess``/image probes) so the resolution table and the
``RLMESH_MAKE_KWARGS`` injection are exercised without a daemon.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import rlmesh
from rlmesh._sandbox import _sources
from rlmesh._sandbox import session as sandbox

# --- resolve_source_kind -----------------------------------------------------


def test_gym_and_explicit_schemes_skip_docker_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A gym id / gym:// / hf:// never probes Docker; explicit docker://image://
    # resolves to prebuilt without a probe either.
    monkeypatch.setattr(
        _sources,
        "docker_image_exists",
        lambda *_a: pytest.fail("must not probe Docker"),
    )

    assert _sources.resolve_source_kind("CartPole-v1") == ("build", "CartPole-v1")
    assert _sources.resolve_source_kind("gym://Foo-v0") == ("build", "gym://Foo-v0")
    assert _sources.resolve_source_kind("ALE/Pong-v5") == ("build", "ALE/Pong-v5")
    assert _sources.resolve_source_kind("docker://lib:1") == ("prebuilt", "lib:1")
    assert _sources.resolve_source_kind("image://lib:1") == ("prebuilt", "lib:1")


def test_bare_image_tag_resolves_to_local_prebuilt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_sources, "docker_image_exists", lambda image: True)
    monkeypatch.setattr(
        _sources, "docker_pull", lambda *_a: pytest.fail("local hit, no pull")
    )
    assert _sources.resolve_source_kind("libero:latest") == (
        "prebuilt",
        "libero:latest",
    )


def test_bare_image_tag_pulls_when_not_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_sources, "docker_image_exists", lambda image: False)
    pulled: list[str] = []
    monkeypatch.setattr(
        _sources, "docker_pull", lambda image: pulled.append(image) or (True, "")
    )
    assert _sources.resolve_source_kind("repo/img:tag") == ("prebuilt", "repo/img:tag")
    assert pulled == ["repo/img:tag"]


def test_bare_image_tag_not_found_raises_with_pull_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed pull's stderr tail is surfaced so an auth failure never reads as
    a plain "not found"."""
    monkeypatch.setattr(_sources, "docker_image_exists", lambda image: False)
    monkeypatch.setattr(
        _sources,
        "docker_pull",
        lambda image: (False, "unauthorized: authentication required"),
    )
    with pytest.raises(ValueError, match="not found locally or pullable") as excinfo:
        _sources.resolve_source_kind("nope:latest")
    assert "unauthorized: authentication required" in str(excinfo.value)


# --- prebuilt_run_cmd hardening + binding ------------------------------------


def test_prebuilt_run_cmd_is_hardened_with_image_last() -> None:
    cmd = sandbox.prebuilt_run_cmd(
        "img:1",
        env_vars={"RLMESH_MAKE_KWARGS": '{"suite": "a"}'},
        gpus="2",
        container_port=50051,
        owner_pid=123,
        owner_pid_ns=None,
        devices=["nvidia.com/gpu=all"],
        volumes=["/h/a:/c/a", "/h/b:/c/b:ro"],
    )
    assert cmd[:3] == ["docker", "run", "-d"]
    assert "--cap-drop" in cmd and "ALL" in cmd
    assert "no-new-privileges" in cmd
    assert cmd[cmd.index("--gpus") + 1] == "2"
    assert cmd[cmd.index("--device") + 1] == "nvidia.com/gpu=all"
    v_idxs = [i for i, a in enumerate(cmd) if a == "-v"]
    assert [cmd[i + 1] for i in v_idxs] == ["/h/a:/c/a", "/h/b:/c/b:ro"]
    assert cmd[cmd.index("-e") + 1] == 'RLMESH_MAKE_KWARGS={"suite": "a"}'
    assert cmd[-1] == "img:1"  # image always last


# --- start_prebuilt_container injects the binding ----------------------------


def _docker_dispatch(captured: dict[str, list[str]]) -> Any:
    def fake_run(cmd: list[str], **_kwargs: Any) -> Any:
        if cmd[:3] == ["docker", "image", "inspect"]:

            class _I:
                returncode = 0
                stdout = "[]\n"
                stderr = ""

            return _I()
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
    monkeypatch.setattr(_sources.subprocess, "run", _docker_dispatch(captured))

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
    monkeypatch.setattr(sandbox, "reap_orphans", lambda: None)
    monkeypatch.setattr(_sources.subprocess, "run", _docker_dispatch(captured))

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


def test_prebuilt_run_cmd_emits_user_before_image() -> None:
    cmd = sandbox.prebuilt_run_cmd(
        "img:1",
        env_vars={},
        gpus=None,
        container_port=50051,
        owner_pid=123,
        owner_pid_ns=None,
        user="1000",
    )
    user_index = cmd.index("--user")
    assert cmd[user_index + 1] == "1000"
    assert user_index < cmd.index("img:1")


def test_prebuilt_run_cmd_default_emits_no_user_flag() -> None:
    cmd = sandbox.prebuilt_run_cmd(
        "img:1",
        env_vars={},
        gpus=None,
        container_port=50051,
        owner_pid=123,
        owner_pid_ns=None,
    )
    assert "--user" not in cmd


def test_normalize_user_accepts_uid_forms_and_rejects_blank() -> None:
    assert sandbox.normalize_user(None) is None
    assert sandbox.normalize_user(1000) == "1000"
    assert sandbox.normalize_user("1000:1000") == "1000:1000"
    for bad in ("", "  "):
        with pytest.raises(ValueError, match="user="):
            sandbox.normalize_user(bad)
    with pytest.raises(ValueError, match="user="):
        sandbox.normalize_user(-1)


def test_start_prebuilt_container_forwards_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(sandbox, "reap_orphans", lambda: None)
    monkeypatch.setattr(_sources.subprocess, "run", _docker_dispatch(captured))

    sandbox.start_prebuilt_container(
        "libero:latest",
        requested_source="libero:latest",
        binding={},
        user="1000:1000",
    )

    run_cmd = captured["run"]
    assert run_cmd[run_cmd.index("--user") + 1] == "1000:1000"


def test_source_build_rejects_runtime_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sandbox,
        "_sandbox_start_env",
        lambda *_args, **_kwargs: pytest.fail("sandbox should not start"),
    )
    with pytest.raises(ValueError, match="prebuilt"):
        sandbox.start_sandbox_container(
            "CartPole-v1",
            build=None,
            runtime=rlmesh.SandboxRuntime(user=1000),
            num_envs=1,
            vectorization_mode=None,
            binding={},
        )


def test_sandbox_model_forwards_user(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(sandbox, "reap_orphans", lambda: None)
    monkeypatch.setattr(_sources.subprocess, "run", _docker_dispatch(captured))

    model = rlmesh.SandboxModel(
        "smolvla:latest", runtime=rlmesh.SandboxRuntime(user=1000)
    )
    assert model._user == "1000"
    model.serve()

    run_cmd = captured["run"]
    assert run_cmd[run_cmd.index("--user") + 1] == "1000"


def test_sandbox_model_runtime_and_rejects_build_runtime_params() -> None:
    # A model is always a prebuilt image (no build config). Runtime flags ride in
    # runtime=SandboxRuntime(...); construction is inert (no container started).
    m = rlmesh.SandboxModel(
        "m:latest",
        runtime=rlmesh.SandboxRuntime(gpus="all", devices=["nvidia.com/gpu=all"]),
    )
    assert m._gpus == "all"
    assert m._devices == ["nvidia.com/gpu=all"]
    # A colliding build/runtime field in **params fails loud, not silently bound.
    for name in ("gpus", "volumes", "base_image", "packages"):
        with pytest.raises(TypeError, match="build=SandboxBuild"):
            rlmesh.SandboxModel("m:latest", **{name: "x"})  # type: ignore[arg-type]


# --- Docker hang guard, pre-pull, and port-failure diagnostics ----------------


def test_run_docker_converts_timeout_into_daemon_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wedged Docker daemon surfaces as a directive error, never a hang."""
    import subprocess

    def hang(*_a: Any, **_k: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd=["docker", "inspect"], timeout=60.0)

    monkeypatch.setattr(_sources.subprocess, "run", hang)
    with pytest.raises(RuntimeError, match="Docker daemon not responding"):
        _sources.run_docker(["docker", "inspect", "x"])


def test_start_prebuilt_container_pre_pulls_missing_image(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing prebuilt image is pulled up front (announced on stderr), not
    invisibly inside docker run."""
    pulls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> Any:
        if cmd[:3] == ["docker", "image", "inspect"]:

            class _Missing:
                returncode = 1
                stdout = ""
                stderr = "No such image\n"

            return _Missing()
        if cmd[:2] == ["docker", "pull"]:
            pulls.append(cmd)

            class _Pulled:
                returncode = 0
                stdout = ""
                stderr = ""

            return _Pulled()
        return _docker_dispatch({})(cmd)

    monkeypatch.setattr(sandbox, "reap_orphans", lambda: None)
    monkeypatch.setattr(_sources.subprocess, "run", fake_run)

    info = sandbox.start_prebuilt_container(
        "img:9", requested_source="img:9", binding={}
    )

    assert pulls == [["docker", "pull", "img:9"]]
    assert info.container_id == "container-9"
    assert "pulling image 'img:9'" in capsys.readouterr().err


def test_port_failure_on_crashed_container_appends_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An instantly-crashing container loses the docker-port race; the error
    must say it exited and include its recent logs, not just "no host port"."""

    def fake_run(cmd: list[str], **_kwargs: Any) -> Any:
        if cmd[:3] == ["docker", "image", "inspect"]:

            class _I:
                returncode = 0
                stdout = "[]\n"
                stderr = ""

            return _I()
        if cmd[:3] == ["docker", "run", "-d"]:

            class _P:
                returncode = 0
                stdout = "container-9\n"
                stderr = ""

            return _P()
        if cmd[:2] == ["docker", "port"]:

            class _NoPort:
                returncode = 0
                stdout = ""
                stderr = ""

            return _NoPort()
        if cmd[:2] == ["docker", "inspect"]:

            class _Exited:
                returncode = 0
                stdout = "false\n"
                stderr = ""

            return _Exited()
        if cmd[:2] == ["docker", "logs"]:

            class _Logs:
                returncode = 0
                stdout = "traceback: kaboom\n"
                stderr = ""

            return _Logs()
        raise AssertionError(f"unexpected docker call: {cmd}")

    stopped: list[str] = []
    monkeypatch.setattr(sandbox, "reap_orphans", lambda: None)
    monkeypatch.setattr(
        sandbox,
        "_sandbox_stop_env",
        lambda *, container_id: stopped.append(container_id),
    )
    monkeypatch.setattr(_sources.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="exited during startup") as excinfo:
        sandbox.start_prebuilt_container("img:1", requested_source="img:1", binding={})

    assert "traceback: kaboom" in str(excinfo.value)
    assert stopped == ["container-9"]
