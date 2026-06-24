from __future__ import annotations

from typing import Any

import pytest


def test_serve_env_parser_requires_one_source() -> None:
    from rlmesh._cli import serve_env

    parser = serve_env.create_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--env",
                "CartPole-v1",
                "--entrypoint",
                "rlmesh_system_fixtures.registry:make_env",
            ]
        )


def test_serve_env_parser_accepts_entrypoint_kwargs_json() -> None:
    from rlmesh._cli import serve_env

    parser = serve_env.create_parser()
    namespace = parser.parse_args(
        [
            "--entrypoint",
            "rlmesh_system_fixtures.registry:make_env",
            "--kwargs-json",
            '{"fixture": "counter"}',
        ]
    )

    args = serve_env.serve_args_from_namespace(namespace)

    assert args.env is None
    assert args.entrypoint == "rlmesh_system_fixtures.registry:make_env"
    assert args.kwargs == {"fixture": "counter"}


def test_serve_env_parser_rejects_non_object_kwargs_json() -> None:
    from rlmesh._cli import serve_env

    parser = serve_env.create_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--env", "CartPole-v1", "--kwargs-json", "[]"])


def test_serve_env_dispatches_entrypoint_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlmesh
    from rlmesh._cli import serve_env

    captured: dict[str, object] = {}
    fake_env = object()

    def load_env_entrypoint(
        entrypoint: str,
        package_names: list[str],
        kwargs: dict[str, Any] | None = None,
    ) -> object:
        captured["entrypoint"] = entrypoint
        captured["package_names"] = package_names
        captured["kwargs"] = kwargs
        return fake_env

    monkeypatch.setattr(serve_env, "load_env_entrypoint", load_env_entrypoint)
    monkeypatch.setattr(rlmesh, "EnvServer", _server_factory(captured, "EnvServer"))
    monkeypatch.setattr(
        rlmesh, "VectorEnvServer", _server_factory(captured, "VectorEnvServer")
    )

    code = serve_env.serve_from_args(
        serve_env.ServeArgs(
            env=None,
            entrypoint="rlmesh_system_fixtures.registry:make_env",
            transport="tcp",
            address="127.0.0.1:0",
            num_envs=1,
            vectorization_mode=None,
            package=["rlmesh_system_fixtures.registration"],
            verbose=False,
            kwargs={"fixture": "counter"},
        )
    )

    assert code == 0
    assert captured["entrypoint"] == "rlmesh_system_fixtures.registry:make_env"
    assert captured["package_names"] == ["rlmesh_system_fixtures.registration"]
    assert captured["kwargs"] == {"fixture": "counter"}
    assert captured["server_class"] == "EnvServer"
    assert captured["server_env"] is fake_env
    assert captured["server_args"] == ("127.0.0.1:0",)
    assert captured["served"] is True


def test_serve_env_uses_vector_server_for_vector_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlmesh
    from rlmesh._cli import serve_env

    captured: dict[str, object] = {}

    class VectorEnv:
        num_envs = 2

    fake_env = VectorEnv()

    def load_env_entrypoint(
        entrypoint: str,
        package_names: list[str],
        kwargs: dict[str, Any] | None = None,
    ) -> object:
        captured["entrypoint"] = entrypoint
        captured["package_names"] = package_names
        captured["kwargs"] = kwargs
        return fake_env

    monkeypatch.setattr(serve_env, "load_env_entrypoint", load_env_entrypoint)
    monkeypatch.setattr(rlmesh, "EnvServer", _server_factory(captured, "EnvServer"))
    monkeypatch.setattr(
        rlmesh, "VectorEnvServer", _server_factory(captured, "VectorEnvServer")
    )

    code = serve_env.serve_from_args(
        serve_env.ServeArgs(
            env=None,
            entrypoint="rlmesh_system_fixtures.registry:make_vector_env",
            transport="tcp",
            address="127.0.0.1:0",
            num_envs=1,
            vectorization_mode=None,
            package=[],
            verbose=False,
        )
    )

    assert code == 0
    assert captured["entrypoint"] == "rlmesh_system_fixtures.registry:make_vector_env"
    assert captured["server_class"] == "VectorEnvServer"
    assert captured["server_env"] is fake_env


def test_serve_env_dispatches_gym_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    import rlmesh
    from rlmesh._cli import serve_env

    captured: dict[str, object] = {}
    fake_env = object()

    def load_environment(
        env_id: str,
        package_names: list[str],
        num_envs: int,
        vectorization_mode: str | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> object:
        captured["env_id"] = env_id
        captured["package_names"] = package_names
        captured["num_envs"] = num_envs
        captured["vectorization_mode"] = vectorization_mode
        captured["kwargs"] = kwargs
        return fake_env

    monkeypatch.setattr(serve_env, "load_environment", load_environment)
    monkeypatch.setattr(rlmesh, "EnvServer", _server_factory(captured, "EnvServer"))
    monkeypatch.setattr(
        rlmesh, "VectorEnvServer", _server_factory(captured, "VectorEnvServer")
    )

    code = serve_env.serve_from_args(
        serve_env.ServeArgs(
            env="CartPole-v1",
            entrypoint=None,
            transport="tcp",
            address=None,
            num_envs=2,
            vectorization_mode="sync",
            package=["gymnasium"],
            verbose=False,
            kwargs={"render_mode": "rgb_array"},
        )
    )

    assert code == 0
    assert captured["env_id"] == "CartPole-v1"
    assert captured["package_names"] == ["gymnasium"]
    assert captured["num_envs"] == 2
    assert captured["vectorization_mode"] == "sync"
    assert captured["kwargs"] == {"render_mode": "rgb_array"}
    assert captured["server_class"] == "VectorEnvServer"
    assert captured["server_env"] is fake_env
    assert captured["server_args"] == ()
    assert captured["served"] is True


def test_serve_env_rejects_entrypoint_vectorization_options() -> None:
    from rlmesh._cli import serve_env

    code = serve_env.serve_from_args(
        serve_env.ServeArgs(
            env=None,
            entrypoint="rlmesh_system_fixtures.registry:make_env",
            transport="tcp",
            address=None,
            num_envs=2,
            vectorization_mode=None,
            package=[],
            verbose=False,
        )
    )

    assert code == 1


def _server_factory(
    captured: dict[str, object], name: str = "EnvServer"
) -> type[object]:
    class FakeServer:
        def __init__(self, env: object, *args: object, **kwargs: object) -> None:
            captured["server_class"] = name
            captured["server_env"] = env
            captured["server_args"] = args
            captured["server_kwargs"] = kwargs

        def address(self) -> str:
            return "127.0.0.1:1234"

        def serve(self) -> None:
            captured["served"] = True

    return FakeServer


def test_write_ready_fd_writes_address_line_and_closes() -> None:
    import os

    from rlmesh._cli import serve_env

    read_fd, write_fd = os.pipe()
    try:
        serve_env.write_ready_fd(write_fd, "tcp://127.0.0.1:54321")
        # write_ready_fd takes ownership of and closes write_fd.
        with os.fdopen(read_fd, "r") as reader:
            read_fd = -1  # fdopen owns it now
            contents = reader.read()
        assert contents == "tcp://127.0.0.1:54321\n"
    finally:
        if read_fd != -1:
            os.close(read_fd)
        # write_fd is closed by write_ready_fd; nothing to clean up.


def test_serve_from_args_writes_ready_fd(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    import rlmesh
    from rlmesh._cli import serve_env

    fake_env = object()

    def load_environment(
        env_id: str,
        package_names: list[str],
        num_envs: int,
        vectorization_mode: str | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> object:
        return fake_env

    class FakeServer:
        def __init__(self, env: object, *args: object, **kwargs: object) -> None:
            pass

        @property
        def address(self) -> str:
            return "tcp://127.0.0.1:50051"

        def serve(self) -> None:
            return None

    monkeypatch.setattr(serve_env, "load_environment", load_environment)
    monkeypatch.setattr(rlmesh, "EnvServer", FakeServer)
    monkeypatch.setattr(rlmesh, "VectorEnvServer", FakeServer)

    read_fd, write_fd = os.pipe()
    try:
        code = serve_env.serve_from_args(
            serve_env.ServeArgs(
                env="CartPole-v1",
                entrypoint=None,
                transport="tcp",
                address=None,
                num_envs=1,
                vectorization_mode=None,
                package=[],
                verbose=False,
                ready_fd=write_fd,
            )
        )
        assert code == 0
        with os.fdopen(read_fd, "r") as reader:
            read_fd = -1
            contents = reader.read()
        assert contents == "tcp://127.0.0.1:50051\n"
    finally:
        if read_fd != -1:
            os.close(read_fd)


def test_default_unix_socket_path_uses_xdg_runtime_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import os

    from rlmesh._cli import serve_env

    runtime_dir = tmp_path / "run"
    runtime_dir.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))

    path = serve_env._default_unix_socket_path("CartPole-v1")

    assert path.startswith(str(runtime_dir))
    assert path.endswith("rlmesh-cartpole-v1.sock")
    parent = os.path.dirname(path)
    assert os.path.isdir(parent)
    assert (os.stat(parent).st_mode & 0o777) == 0o700


def test_default_unix_socket_path_avoids_shared_tmp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import stat

    from rlmesh._cli import serve_env

    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)

    path = serve_env._default_unix_socket_path("CartPole-v1")

    parent = os.path.dirname(path)
    # Must not be the predictable, world-readable /tmp/rlmesh-*.sock path.
    assert path != "/tmp/rlmesh-cartpole-v1.sock"
    assert parent != "/tmp"
    mode = stat.S_IMODE(os.stat(parent).st_mode)
    assert mode == 0o700
