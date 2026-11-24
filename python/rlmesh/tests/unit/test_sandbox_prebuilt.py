from __future__ import annotations

import json
from typing import Any, cast

import pytest
import rlmesh
from rlmesh._sandbox import _model, _sources
from rlmesh._sandbox import session as sandbox


@pytest.fixture(autouse=True)
def _session_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_sandbox_session_port", lambda: 40000)
    monkeypatch.setattr(sandbox, "_sandbox_host_identity", lambda: ("secret", "k"))
    monkeypatch.setattr(_model, "wait_serving", lambda *_args, **_kwargs: None)


def test_gym_and_explicit_schemes_skip_docker_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(_sources, "docker_image_exists", lambda image: False)
    monkeypatch.setattr(
        _sources,
        "docker_pull",
        lambda image: (False, "unauthorized: authentication required"),
    )
    with pytest.raises(ValueError, match="not found locally or pullable") as excinfo:
        _sources.resolve_source_kind("nope:latest")
    assert "unauthorized: authentication required" in str(excinfo.value)


def test_a_gym_shaped_name_that_is_also_a_local_image_says_which_it_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_sources.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(
        _sources, "docker_pull", lambda *_a: pytest.fail("never pulls a gym id")
    )
    local: set[str] = set()
    probed: list[str] = []

    def exists(image: str) -> bool:
        probed.append(image)
        return image in local

    monkeypatch.setattr(_sources, "docker_image_exists", exists)
    registered: set[str] = set()
    monkeypatch.setattr(_sources, "gym_id_registered", lambda v: v in registered)

    assert _sources.resolve_source_kind("rlmesh-libero-v1") == (
        "build",
        "rlmesh-libero-v1",
    )
    assert probed == ["rlmesh-libero-v1"]

    local.add("rlmesh-libero-v1")
    assert _sources.resolve_source_kind("rlmesh-libero-v1") == (
        "prebuilt",
        "rlmesh-libero-v1",
    )

    registered.add("rlmesh-libero-v1")
    with pytest.warns(UserWarning, match=r"docker://rlmesh-libero-v1 to run the image"):
        assert _sources.resolve_source_kind("rlmesh-libero-v1") == (
            "build",
            "rlmesh-libero-v1",
        )

    registered.clear()
    with pytest.warns(UserWarning, match="because build options were given"):
        assert _sources.resolve_source_kind("rlmesh-libero-v1", prefer_build=True) == (
            "build",
            "rlmesh-libero-v1",
        )

    probed.clear()
    local.update({"Rlmesh-Libero-v1", "pkg:env-v1"})
    assert _sources.resolve_source_kind("Rlmesh-Libero-v1")[0] == "build"
    assert _sources.resolve_source_kind("pkg:env-v1")[0] == "build"
    assert _sources.resolve_source_kind("gym://rlmesh-libero-v1")[0] == "build"
    assert probed == []


def test_gym_id_registered_reads_the_registry_without_importing() -> None:
    pytest.importorskip("gymnasium")
    assert _sources.gym_id_registered("CartPole-v1")
    assert not _sources.gym_id_registered("rlmesh-libero-v1")


def test_build_options_ask_for_a_build() -> None:
    assert not sandbox._asks_for_build(sandbox.SandboxBuild())
    assert sandbox._asks_for_build(sandbox.SandboxBuild(packages=["libero"]))
    assert sandbox._asks_for_build(sandbox.SandboxBuild(trust_remote_code=True))


def test_prebuilt_run_cmd_is_hardened_with_image_last() -> None:
    cmd = sandbox.prebuilt_run_cmd(
        "img:1",
        env_vars={"RLMESH_MAKE_KWARGS": '{"suite": "a"}'},
        gpus="2",
        host_port=40000,
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
    assert [cmd[i + 1] for i, a in enumerate(cmd) if a == "-p"] == [
        "127.0.0.1:40000:40000"
    ]
    assert cmd[-1] == "img:1"


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
        raise AssertionError(f"unexpected docker call: {cmd}")

    return fake_run


def test_start_prebuilt_container_injects_binding_and_serves_in_a_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(sandbox, "reap_orphans", lambda: None)
    monkeypatch.setattr(_sources.subprocess, "run", _docker_dispatch(captured))

    info = sandbox.start_prebuilt_container(
        "libero:latest",
        requested_source="libero:latest",
        name="env",
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
    assert any(a.startswith("RLMESH_NUM_ENVS=4") for a in run_cmd)
    assert any(a.startswith("RLMESH_VECTORIZATION_MODE=sync") for a in run_cmd)
    assert "RLMESH_ADDRESS=0.0.0.0:40000" in run_cmd
    assert "RLMESH_NAME=env" in run_cmd
    assert run_cmd[run_cmd.index("RLMESH_HOST_IDENTITY") - 1] == "-e"
    assert not any("secret" in a for a in run_cmd)
    assert info.container_id == "container-9"
    assert info.address == "127.0.0.1:40000#k/env"
    assert info.resolved_source == "docker://libero:latest"


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
        host_port=40000,
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
        host_port=40000,
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
        name="env",
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
    m = rlmesh.SandboxModel(
        "m:latest",
        runtime=rlmesh.SandboxRuntime(gpus="all", devices=["nvidia.com/gpu=all"]),
    )
    assert m._gpus == "all"
    assert m._devices == ["nvidia.com/gpu=all"]
    for name in ("gpus", "volumes", "base_image", "packages"):
        with pytest.raises(TypeError, match="build=SandboxBuild"):
            rlmesh.SandboxModel("m:latest", **{name: "x"})  # type: ignore[arg-type]


def test_run_docker_converts_timeout_into_daemon_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        "img:9", requested_source="img:9", name="env", binding={}
    )

    assert pulls == [["docker", "pull", "img:9"]]
    assert info.container_id == "container-9"
    assert "pulling image 'img:9'" in capsys.readouterr().err


def test_a_model_failure_carries_the_containers_log_lines_without_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    key = "a" * 26 + "b2c3d4e5f6g7" + "z" * 14
    logs: list[list[str]] = []

    def docker(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[1] == "inspect":
            return subprocess.CompletedProcess(cmd, 0, "true\n", "")
        logs.append(cmd)
        out = f"serving at 0.0.0.0:7447#{key}/model\nloading weights\n"
        return subprocess.CompletedProcess(cmd, 0, out, "Traceback: boom\n")

    monkeypatch.setattr(sandbox, "run_docker", docker)

    class Failing:
        def run(self, env: object, **options: object) -> object:
            raise rlmesh.ModelException("AssertionError")

        def run_routes(self, envs: object, **options: object) -> list[object]:
            return [rlmesh.ModelException("AssertionError"), ValueError("not mine")]

    model = rlmesh.SandboxModel("policy:latest")
    model._address = "127.0.0.1:40000#k/model"  # pyright: ignore[reportPrivateUsage]
    model._container_id = "c1"  # pyright: ignore[reportPrivateUsage]
    model._model = Failing()  # type: ignore[assignment]  # pyright: ignore[reportPrivateUsage]
    model._closed = True  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(rlmesh.ModelException) as failed:
        model.run(cast(Any, object()))
    text = str(failed.value)
    assert text.startswith(
        "[INTERNAL] AssertionError\nthe model container's last log lines:\n"
    )
    assert "loading weights" in text and "Traceback: boom" in text
    assert key not in text and "0.0.0.0:7447#…/model" in text
    assert logs[-1] == ["docker", "logs", "--tail", "40", "c1"]

    first, other = model.run_routes(cast(Any, [object(), object()]))
    assert "the model container's last log lines" in str(first)
    assert str(other) == "not mine"
