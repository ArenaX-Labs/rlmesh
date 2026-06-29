"""Read-only role inspection: sess.reader / sess.read.

Pins the inline read seam: the same adapter pipeline a model uses, pointed at the
consumer. A read item is a role constant (kept in the env's native encoding) or a
model-input leaf that declares the encoding you want (Image(layout=...)), resolved
once against the env's published tags and reused per step. Bare roles desugar to
the env-native leaf by the env's own tag -- authoritative, so a prefix-less role
(INSTRUCTION) resolves like any other.
"""

from __future__ import annotations

from typing import Any

import pytest
import rlmesh
import rlmesh.adapters as adapt


def _tags() -> adapt.EnvTags:
    return adapt.EnvTags(
        observation={
            "cam": adapt.ImageTag(role=adapt.IMAGE_PRIMARY),  # env stores HWC
            "eef_pos": adapt.StateTag(role=adapt.EEF_POS),
            "instruction": adapt.TextTag(role=adapt.INSTRUCTION),
        },
        action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )


class _ArmEnv:
    """A tagged local env: HWC camera + eef state + instruction, 1-dim action."""

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


def _session(tagged_env: Any = None) -> Any:
    pytest.importorskip("numpy")
    pytest.importorskip("gymnasium")
    if tagged_env is None:
        tagged_env = adapt.tag(_ArmEnv(), _tags())
    return rlmesh.session(rlmesh.RANDOM_SAMPLE, tagged_env)


def test_reader_overrides_encoding_and_keeps_native_for_bare_roles() -> None:
    sess = _session()
    obs, _ = sess.reset()

    read = sess.reader(
        adapt.Image(adapt.IMAGE_PRIMARY, layout="chw"),  # env HWC -> CHW
        adapt.EEF_POS,  # bare role -> env-native state
    )
    out = read(obs)

    assert set(out) == {adapt.IMAGE_PRIMARY, adapt.EEF_POS}
    assert out[adapt.IMAGE_PRIMARY].shape == (3, 8, 8)  # layout converted
    assert out[adapt.EEF_POS].shape == (3,)  # native, untouched
    assert read.roles == (adapt.IMAGE_PRIMARY, adapt.EEF_POS)


def test_bare_role_desugars_a_prefixless_text_role() -> None:
    # INSTRUCTION carries no kind prefix, so desugar must read the env tag, not the
    # role string. This is the case that kills a prefix-matching shortcut.
    sess = _session()
    obs, _ = sess.reset()
    assert sess.reader(adapt.INSTRUCTION)(obs)[adapt.INSTRUCTION] == "pick up the cube"


def test_read_one_shot_returns_the_single_value_and_caches() -> None:
    sess = _session()
    obs, _ = sess.reset()

    ee = sess.read(obs, adapt.EEF_POS)
    assert ee.shape == (3,)
    img = sess.read(obs, adapt.Image(adapt.IMAGE_PRIMARY, layout="chw"))
    assert img.shape == (3, 8, 8)
    # Second call with the same item reuses the resolved reader (no re-resolve).
    assert sess.read(obs, adapt.EEF_POS).shape == (3,)


def test_reader_rejects_an_unknown_role() -> None:
    sess = _session()
    with pytest.raises(adapt.AdapterResolutionError):
        sess.reader("proprio/not_a_real_role")


def test_reader_needs_at_least_one_item() -> None:
    sess = _session()
    with pytest.raises(TypeError):
        sess.reader()


def test_bare_image_role_keeps_env_chw_layout() -> None:
    # A torch-style env stores its camera CHW and tags layout="chw". A bare role read
    # must keep that native layout, not silently transpose to the Image default
    # (hwc), which would move the channel axis and corrupt the frame.
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
    sess = _session(adapt.tag(_ChwEnv(), tags))
    obs, _ = sess.reset()

    native = sess.read(obs, adapt.IMAGE_PRIMARY)  # bare role -> env-native layout
    assert native.shape == (3, 8, 8)  # kept CHW, not transposed to (8, 8, 3)


def _clip_tags() -> adapt.EnvTags:
    """Tags that declare env-side-only ``clip`` clamps: a per-actuator ``clip=True``
    AND a global ``Action.clip``. Mirrors the real envs (libero/metaworld/aloha/
    robocasa clamp the whole vector; simpler_env clamps per actuator)."""
    return adapt.EnvTags(
        observation={"cam": adapt.ImageTag(role=adapt.IMAGE_PRIMARY)},
        action=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3, range=(-1.0, 1.0), clip=True),
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
            clip=(-1.0, 1.0),
        ),
    )


class _ClipEnv(_ArmEnv):
    """A tagged env whose action declares env-side clip clamps (4-dim action)."""

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


def test_read_by_role_resolves_against_an_env_that_declares_clip_clamps() -> None:
    # Regression: a read rides the env's own action along as a *no-op* output. ``clip``
    # is env-side only (the resolver forbids a model action from setting it), so
    # echoing it verbatim made every read against a clip-declaring env raise
    # "clip is an env-side actuator clamp; the model action declaration must leave it
    # unset" -- silently blanking the live viewer and `record`, which both read by
    # role. The read must strip the env-side clamps and resolve normally.
    sess = _session(adapt.tag(_ClipEnv(), _clip_tags()))
    obs, _ = sess.reset()

    img = sess.read(obs, adapt.Image(adapt.IMAGE_PRIMARY, layout="hwc"))
    assert img.shape == (8, 8, 3)


def test_read_only_action_drops_env_side_clip_clamps() -> None:
    # The strip helper underneath: global and per-actuator clip go, while the actuator
    # layout the read needs (roles, dims) stays.
    from rlmesh._models._read import _read_only_action

    stripped = _read_only_action(_clip_tags().action)

    assert stripped.clip is None  # global clamp dropped
    assert [c.clip for c in stripped.components] == [
        False,
        False,
    ]  # per-actuator dropped
    assert [c.role for c in stripped.components] == [
        adapt.ACTION_DELTA_POS,
        adapt.ACTION_GRIPPER,
    ]
    assert [c.dim for c in stripped.components] == [3, 1]  # layout preserved
