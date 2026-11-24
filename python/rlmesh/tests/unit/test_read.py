from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest
import rlmesh
import rlmesh.adapters as adapt


def _tags() -> adapt.EnvTags:
    return adapt.EnvTags(
        observation={
            "cam": adapt.ImageTag(role=adapt.IMAGE_PRIMARY),
            "eef_pos": adapt.StateTag(role=adapt.EEF_POS),
            "instruction": adapt.TextTag(role=adapt.INSTRUCTION),
        },
        action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )


class _ArmEnv:
    def __init__(self) -> None:
        import gymnasium as gym
        import numpy as np

        self.metadata: dict[str, Any] = {"render_modes": []}
        self.observation_space = gym.spaces.Dict(
            {
                "cam": gym.spaces.Box(0, 255, (8, 8, 3), np.uint8),
                "eef_pos": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
                "instruction": gym.spaces.Text(max_length=64),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

    def _obs(self) -> dict[str, Any]:
        import numpy as np

        return {
            "cam": np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3),
            "eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "instruction": "pick up the cube",
        }

    def reset(
        self, *, seed: object = None, options: object = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._obs(), {}

    def step(
        self, action: object
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return self._obs(), 1.0, True, False, {}

    def close(self) -> None:
        return None


@pytest.fixture
def bind() -> Iterator[Callable[..., Any]]:
    pytest.importorskip("numpy")
    pytest.importorskip("gymnasium")
    bindings: list[Any] = []

    def bind(tagged_env: Any = None) -> Any:
        if tagged_env is None:
            tagged_env = adapt.tag(_ArmEnv(), _tags())
        from rlmesh.numpy import Model

        bindings.append(Model(rlmesh.RANDOM_SAMPLE).bind(tagged_env))
        return bindings[-1]

    yield bind
    for binding in bindings:
        binding.close()


def test_read_overrides_encoding_and_keeps_native_for_bare_roles(
    bind: Callable[..., Any],
) -> None:
    binding = bind()
    obs, _ = binding.reset()

    image = binding.read(obs, adapt.Image(adapt.IMAGE_PRIMARY, layout="chw"))
    assert image.shape == (3, 8, 8)
    assert binding.read(obs, adapt.EEF_POS).shape == (3,)


def test_bare_role_desugars_a_prefixless_text_role(bind: Callable[..., Any]) -> None:
    binding = bind()
    obs, _ = binding.reset()
    assert binding.read(obs, adapt.INSTRUCTION) == "pick up the cube"


def test_read_returns_the_single_value_and_caches(bind: Callable[..., Any]) -> None:
    binding = bind()
    obs, _ = binding.reset()

    ee = binding.read(obs, adapt.EEF_POS)
    assert ee.shape == (3,)
    img = binding.read(obs, adapt.Image(adapt.IMAGE_PRIMARY, layout="chw"))
    assert img.shape == (3, 8, 8)
    assert binding.read(obs, adapt.EEF_POS).shape == (3,)


def test_read_rejects_an_unknown_role(bind: Callable[..., Any]) -> None:
    binding = bind()
    obs, _ = binding.reset()
    with pytest.raises(adapt.AdapterResolutionError):
        binding.read(obs, "proprio/not_a_real_role")


def test_bare_image_role_keeps_env_chw_layout(bind: Callable[..., Any]) -> None:
    import gymnasium as gym
    import numpy as np

    class _ChwEnv(_ArmEnv):
        def __init__(self) -> None:
            super().__init__()
            self.observation_space = gym.spaces.Dict(
                {
                    "cam": gym.spaces.Box(0, 255, (3, 8, 8), np.uint8),
                    "eef_pos": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
                    "instruction": gym.spaces.Text(max_length=64),
                }
            )

        def _obs(self) -> dict[str, Any]:
            return {
                "cam": np.arange(3 * 8 * 8, dtype=np.uint8).reshape(3, 8, 8),
                "eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "instruction": "pick up the cube",
            }

    tags = adapt.EnvTags(
        observation={
            "cam": adapt.ImageTag(role=adapt.IMAGE_PRIMARY, layout="chw"),
            "eef_pos": adapt.StateTag(role=adapt.EEF_POS),
            "instruction": adapt.TextTag(role=adapt.INSTRUCTION),
        },
        action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )
    binding = bind(adapt.tag(_ChwEnv(), tags))
    obs, _ = binding.reset()

    native = binding.read(obs, adapt.IMAGE_PRIMARY)
    assert native.shape == (3, 8, 8)


def _clip_tags() -> adapt.EnvTags:
    return adapt.EnvTags(
        observation={"cam": adapt.ImageTag(role=adapt.IMAGE_PRIMARY)},
        action=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3, range=(-1.0, 1.0), clip=True),
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
            clip=(-1.0, 1.0),
        ),
    )


class _ClipEnv(_ArmEnv):
    def __init__(self) -> None:
        import gymnasium as gym
        import numpy as np

        super().__init__()
        self.observation_space = gym.spaces.Dict(
            {"cam": gym.spaces.Box(0, 255, (8, 8, 3), np.uint8)}
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)

    def _obs(self) -> dict[str, Any]:
        import numpy as np

        return {"cam": np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)}


def test_read_by_role_resolves_against_an_env_that_declares_clip_clamps(
    bind: Callable[..., Any],
) -> None:
    binding = bind(adapt.tag(_ClipEnv(), _clip_tags()))
    obs, _ = binding.reset()

    img = binding.read(obs, adapt.Image(adapt.IMAGE_PRIMARY, layout="hwc"))
    assert img.shape == (8, 8, 3)


def test_read_only_action_drops_env_side_clip_clamps() -> None:
    from rlmesh._models._read import _read_only_action

    stripped = _read_only_action(_clip_tags().action)

    assert stripped.clip is None
    assert [c.clip for c in stripped.components] == [
        False,
        False,
    ]
    assert [c.role for c in stripped.components] == [
        adapt.ACTION_DELTA_POS,
        adapt.ACTION_GRIPPER,
    ]
    assert [c.dim for c in stripped.components] == [3, 1]
