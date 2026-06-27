"""``python -m rlmesh.serve`` forwards a make-binding to the env factory.

Closes the gap where ``serve.main()`` dropped kwargs: a containerized env can
now serve any variation via ``--kwargs-json`` / ``RLMESH_MAKE_KWARGS``, while a
binding aimed at a model (no load seam yet) fails loudly instead of vanishing.
"""

from __future__ import annotations

import inspect
from typing import Any, cast

import pytest
import rlmesh
from rlmesh import serve


def _capture_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    sentinel = object()
    monkeypatch.setattr(serve, "resolve_entrypoint", lambda *a, **k: sentinel)

    def fake_serve_env(
        env: object,
        address: str,
        *,
        num_envs: int = 1,
        vectorization_mode: str | None = None,
        **binding: object,
    ) -> None:
        captured["env"] = env
        captured["address"] = address
        captured["num_envs"] = num_envs
        captured["vectorization_mode"] = vectorization_mode
        captured["binding"] = binding

    monkeypatch.setattr(serve, "serve_env", fake_serve_env)
    return captured


def test_env_forwards_kwargs_json_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_env(monkeypatch)
    code = serve.main(
        ["--env", "pkg:Env", "--kwargs-json", '{"suite": "a", "task_id": 3}']
    )
    assert code == 0
    assert captured["binding"] == {"suite": "a", "task_id": 3}


def test_env_reads_binding_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_env(monkeypatch)
    monkeypatch.setenv("RLMESH_MAKE_KWARGS", '{"suite": "b"}')
    code = serve.main(["--env", "pkg:Env"])
    assert code == 0
    assert captured["binding"] == {"suite": "b"}


def test_env_absent_binding_serves_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_env(monkeypatch)
    monkeypatch.delenv("RLMESH_MAKE_KWARGS", raising=False)
    code = serve.main(["--env", "pkg:Env"])
    assert code == 0
    assert captured["binding"] == {}


def test_explicit_flag_wins_over_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_env(monkeypatch)
    monkeypatch.setenv("RLMESH_MAKE_KWARGS", '{"suite": "env"}')
    serve.main(["--env", "pkg:Env", "--kwargs-json", '{"suite": "flag"}'])
    assert captured["binding"] == {"suite": "flag"}


def test_non_object_kwargs_json_is_rejected() -> None:
    with pytest.raises(SystemExit):
        serve.main(["--env", "pkg:Env", "--kwargs-json", "[]"])


def test_model_binding_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(serve, "resolve_entrypoint", lambda *a, **k: object())

    def fake_serve_model(
        model: object, address: str, *, token: str = "", binding: object = None
    ) -> None:
        captured["binding"] = binding

    monkeypatch.setattr(serve, "serve_model", fake_serve_model)
    code = serve.main(["pkg:Model", "--kwargs-json", '{"checkpoint": "x"}'])
    assert code == 0
    assert captured["binding"] == {"checkpoint": "x"}


def test_serve_env_reserved_names_are_positional_only() -> None:
    # env_source/address must be positional-only so a make-param named "address"
    # (or "env_source") rides in **make_kwargs instead of colliding.
    params = inspect.signature(serve.serve_env).parameters
    assert params["env_source"].kind is inspect.Parameter.POSITIONAL_ONLY
    assert params["address"].kind is inspect.Parameter.POSITIONAL_ONLY


def test_serve_env_binding_named_address_does_not_collide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeServer:
        def __init__(self, env: object, address: str, **_: object) -> None:
            captured["env"], captured["address"] = env, address

        def serve(self) -> None: ...

    monkeypatch.setattr(rlmesh, "EnvServer", FakeServer)

    def make_env(**kw: object) -> object:
        captured["make_kwargs"] = kw
        return object()

    serve.serve_env(make_env, "0.0.0.0:1", address="from-binding")
    assert captured["address"] == "0.0.0.0:1"
    assert captured["make_kwargs"] == {"address": "from-binding"}


def test_main_forwards_runtime_vectorization(monkeypatch: pytest.MonkeyPatch) -> None:
    # RLMESH_NUM_ENVS / RLMESH_VECTORIZATION_MODE flow into serve_env so a prebuilt
    # EnvFactory image serves the requested lanes instead of a lone env.
    captured = _capture_env(monkeypatch)
    monkeypatch.setenv("RLMESH_NUM_ENVS", "4")
    monkeypatch.setenv("RLMESH_VECTORIZATION_MODE", "async")
    code = serve.main(["--env", "pkg:Env"])
    assert code == 0
    assert captured["num_envs"] == 4
    assert captured["vectorization_mode"] == "async"
    assert captured["binding"] == {}  # vector knobs do not leak into the binding


def test_main_rejects_non_integer_num_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve, "resolve_entrypoint", lambda *a, **k: object())
    monkeypatch.setenv("RLMESH_NUM_ENVS", "lots")
    with pytest.raises(SystemExit):
        serve.main(["--env", "pkg:Env"])


def test_serve_env_vectorizes_factory_and_skips_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    g = pytest.importorskip("gymnasium")
    captured: dict[str, Any] = {}

    class FakeServer:
        def __init__(self, env: object, address: str, *, tags: object = None) -> None:
            captured["env"], captured["tags"] = env, tags

        def serve(self) -> None: ...

    monkeypatch.setattr(rlmesh, "EnvServer", FakeServer)

    class _GymEnv(g.Env):
        observation_space = g.spaces.Box(low=0.0, high=1.0, shape=(2,))
        action_space = g.spaces.Discrete(2)

        def reset(self, *, seed: object = None, options: object = None) -> object:
            return self.observation_space.sample(), {}

        def step(self, action: object) -> object:
            return self.observation_space.sample(), 0.0, False, False, {}

    class _FakeTags:
        # Stand-in EnvTags: EnvFactory.make stamps it onto each sub-env (validate=
        # False), and EnvServer would re-stamp it on the scalar path.
        def to_metadata(self) -> dict[str, object]:
            return {"rlmesh.adapters.v1.env_tags": "x"}

    class _Factory(rlmesh.EnvFactory):
        tags = cast("Any", _FakeTags())  # published only on the scalar path
        params = None

        def make(self) -> object:
            return _GymEnv()

    serve.serve_env(_Factory, "0.0.0.0:1", num_envs=2)
    assert getattr(captured["env"], "num_envs", None) == 2
    assert captured["tags"] is None  # vector env serves untagged (adapters per-lane)


def test_resolve_model_enforces_required_param_with_empty_binding() -> None:
    # The model path now always resolves (matching the env path), so a declared
    # required param is enforced even when no binding is supplied.
    from rlmesh.params import MissingParamError

    class _Req(rlmesh.Model):
        params = rlmesh.ParamSpec(rlmesh.Param("checkpoint"))

        def load(self, *, checkpoint: str) -> None: ...

        def predict(self, observation: object) -> int:
            return 0

    with pytest.raises(MissingParamError, match="checkpoint"):
        serve._resolve_model(_Req, None)
