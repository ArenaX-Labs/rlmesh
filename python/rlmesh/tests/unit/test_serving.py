from __future__ import annotations

import pytest


def test_serving_is_public_module() -> None:
    import rlmesh

    assert hasattr(rlmesh, "serving")
    assert rlmesh.serving.__all__ == [
        "import_packages",
        "load_env",
        "load_env_entrypoint",
    ]


def test_load_env_entrypoint_loads_factory() -> None:
    from rlmesh import serving

    env = serving.load_env_entrypoint(
        "rlmesh_system_fixtures.registry:make_env",
        kwargs={"fixture": "counter"},
    )

    assert hasattr(env, "reset")
    assert hasattr(env, "step")


def test_cli_serve_env_delegates_to_serving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh._cli import serve_env

    captured: dict[str, object] = {}

    def fake_load_env(env_id: str, **kwargs: object) -> str:
        captured["env_id"] = env_id
        captured.update(kwargs)
        return "env-object"

    monkeypatch.setattr(serve_env, "_serving_load_env", fake_load_env)

    result = serve_env.load_environment("CartPole-v1", ["pkg"], 4, "sync", {"a": 1})

    assert result == "env-object"
    assert captured == {
        "env_id": "CartPole-v1",
        "packages": ["pkg"],
        "num_envs": 4,
        "vectorization_mode": "sync",
        "kwargs": {"a": 1},
    }
