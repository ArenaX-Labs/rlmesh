"""An env's camera dial and its episode-cap requirement, end to end.

`EnvFactory.render_params` names the two `make` params that set the camera
resolution, and `requires_episode_cap` says the env never truncates on its own.
Both ride the env's metadata; a sandbox driver binds the model's
`Image(render=...)` size on the dial before `make`, and a run against an env
that needs a cap is refused when it has none.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
import rlmesh
from rlmesh import adapters as adapt
from rlmesh import serve
from rlmesh._rlmesh import ENV_RENDER_PARAMS_KEY, ENV_REQUIRES_EPISODE_CAP_KEY
from rlmesh._sandbox.session import model_render_size


class _TinyEnv:
    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = rlmesh.spaces.Discrete(1)
        self.action_space = rlmesh.spaces.Discrete(1)

    def reset(self, *, seed: object = None, options: Any = None) -> tuple[int, dict]:
        return 0, {}

    def step(self, action: object) -> tuple[int, float, bool, bool, dict]:
        return 0, 0.0, False, False, {}  # never ends on its own

    def close(self) -> None:
        pass


class _Uncapped(rlmesh.EnvFactory):
    render_params = ("cam_height", "cam_width")
    requires_episode_cap = True

    def make(self, cam_height: int = 128, cam_width: int = 128) -> Any:
        return _TinyEnv()


def test_make_publishes_the_dial_and_the_cap_requirement() -> None:
    metadata = _Uncapped().make().metadata
    assert metadata[ENV_RENDER_PARAMS_KEY] == ["cam_height", "cam_width"]
    assert metadata[ENV_REQUIRES_EPISODE_CAP_KEY] is True


def test_a_dial_that_make_does_not_take_is_refused_at_class_creation() -> None:
    with pytest.raises(TypeError, match="make\\(\\) does not take"):

        class _Bad(rlmesh.EnvFactory):  # pyright: ignore[reportUnusedClass]
            render_params = ("height", "width")

            def make(self, size: int = 1) -> Any:
                return _TinyEnv()


def test_a_run_without_a_cap_is_refused_and_a_capped_run_ends() -> None:
    model = rlmesh.Model(lambda obs: 0)
    with pytest.raises(ValueError, match="requires_episode_cap"):
        model.run(_Uncapped().make(), episodes=1)
    result = model.run(_Uncapped().make(), episodes=2, max_episode_steps=3)
    assert [e.steps for e in result.episodes] == [3, 3]


def test_serve_binds_the_render_size_under_the_explicit_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(serve, "resolve_entrypoint", lambda *a, **k: _Uncapped)
    monkeypatch.setattr(serve, "serve_env", lambda env, addr, **kw: captured.update(kw))
    monkeypatch.setenv("RLMESH_RENDER_SIZE", "224x256")
    monkeypatch.setenv("RLMESH_MAKE_KWARGS", '{"cam_width": 300}')

    assert serve.main(["--env", "pkg:Env"]) == 0
    assert captured["cam_height"] == 224
    assert captured["cam_width"] == 300  # the caller's own param wins


def _spec(**images: adapt.Image) -> adapt.ModelSpec:
    return adapt.ModelSpec(
        input=dict(images),
        output=adapt.Action(adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3)),
    )


def test_the_model_render_size_is_one_size_or_refused() -> None:
    primary = adapt.Image(role=adapt.IMAGE_PRIMARY, render=(224, 256))
    assert model_render_size(_spec(a=primary), 1.0) == (224, 256)
    assert model_render_size(
        rlmesh.Model(lambda obs: 0, spec=_spec(a=primary)), 1.0
    ) == (
        224,
        256,
    )
    assert (
        model_render_size(_spec(a=adapt.Image(role=adapt.IMAGE_PRIMARY)), 1.0) is None
    )
    wrist = adapt.Image(role=adapt.IMAGE_WRIST, render=448)
    with pytest.raises(ValueError, match="different render sizes"):
        model_render_size(_spec(a=primary, b=wrist), 1.0)
