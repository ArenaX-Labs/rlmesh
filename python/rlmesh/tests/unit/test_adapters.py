"""Tests for rlmesh.adapters resolved against real VLA/LIBERO formats.

The reference helpers in this module are verbatim ports of the bespoke
per-pair adapters (SmolVLA/OpenVLA/X-VLA/GR00T x LIBERO); the resolved
generic adapters must reproduce their outputs.

Environments are described the rework way: sparse :class:`EnvTags`
over the observation/action *spaces*. Widths, dtypes and keys come from the
gymnasium spaces; the tags carry only roles and the few facts spaces
cannot express (image layout, rotation encoding, explicit ranges).
"""

from __future__ import annotations

import io
import warnings
from types import SimpleNamespace
from typing import Any, NamedTuple, cast

import gymnasium as gym
import numpy as np
import pytest
import rlmesh.adapters as adapt


def ref_quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(quat)
    if norm <= 1e-8:
        return np.zeros(3, dtype=np.float32)
    quat = quat / norm
    xyz, w = quat[:3], float(quat[3])
    sin_half = float(np.linalg.norm(xyz))
    if sin_half <= 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, w)
    return (xyz / sin_half * angle).astype(np.float32)


def ref_quat2rot6d(quat):
    quat = np.asarray(quat, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(quat)
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    x, y, z, w = quat / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rot = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    return rot[:, :2].reshape(-1).astype(np.float32)


def ref_r6d_to_rotvec(r6d):
    r6d = np.asarray(r6d, dtype=np.float32).reshape(6)
    # X-VLA's 6D rotation is row-major over the matrix's first two columns
    # (m[:, :2].reshape(6)), so the two basis vectors de-interleave as the even
    # and odd indices -- matching upstream `rotate6d_to_xyz` (datasets/utils.py)
    # and the Isaac_vla bridge client, not the standard [:3]/[3:] concatenation.
    a1, a2 = r6d[0:5:2], r6d[1:6:2]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    rotation = np.stack([b1, b2, np.cross(b1, b2)], axis=1)
    theta = np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    if abs(theta) < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float32,
    )
    return (axis / (2.0 * np.sin(theta) + 1e-8) * theta).astype(np.float32)


class Env(NamedTuple):
    """A tagged environment: tags plus the gymnasium spaces."""

    tags: adapt.EnvTags
    obs_space: gym.spaces.Space[Any]
    action_space: gym.spaces.Space[Any]


def resolve(env: Env, model: adapt.ModelSpec, **kwargs: Any) -> adapt.Adapter:
    return adapt.resolve(env.tags, env.obs_space, env.action_space, model, **kwargs)


def box(
    *shape: int, dtype: Any = np.float32, low: float = -np.inf, high: float = np.inf
):
    return gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)


def image_space(height: int = 64, width: int = 64) -> gym.spaces.Box:
    return gym.spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)


def text_space() -> gym.spaces.Text:
    return gym.spaces.Text(max_length=256)


ACTION7 = box(7, low=-1.0, high=1.0)
ACTION14 = box(14, low=-1.0, high=1.0)


LIBERO_ACTION = adapt.Action(
    adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
    adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
    adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
    clip=(-1.0, 1.0),
)

# The model-side mirror of LIBERO_ACTION: `clip` is an env-side clamp, so a
# model output layout must leave it unset.
LIBERO_MODEL_ACTION = adapt.Action(*LIBERO_ACTION.components)

LIBERO_ENV = Env(
    tags=adapt.EnvTags(
        observation={
            "agentview_image": adapt.ImageTag(role=adapt.IMAGE_PRIMARY),
            "robot0_eye_in_hand_image": adapt.ImageTag(role=adapt.IMAGE_WRIST),
            "robot0_eef_pos": adapt.StateTag(role=adapt.EEF_POS),
            "robot0_eef_quat": adapt.StateTag(role=adapt.EEF_ROT, encoding="quat_xyzw"),
            "robot0_gripper_qpos": adapt.StateTag(role=adapt.GRIPPER_POS),
            "instruction": adapt.TextTag(role=adapt.INSTRUCTION),
        },
        action=LIBERO_ACTION,
    ),
    obs_space=gym.spaces.Dict(
        {
            "agentview_image": image_space(),
            "robot0_eye_in_hand_image": image_space(),
            "robot0_eef_pos": box(3),
            "robot0_eef_quat": box(4),
            "robot0_gripper_qpos": box(2),
            "instruction": text_space(),
        }
    ),
    action_space=ACTION7,
)


def make_obs(size: int = 64) -> dict[str, object]:
    rng = np.random.default_rng(7)
    quat = rng.normal(size=4).astype(np.float32)
    quat /= np.linalg.norm(quat)
    return {
        "agentview_image": rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": rng.integers(
            0, 256, (size, size, 3), dtype=np.uint8
        ),
        "robot0_eef_pos": rng.normal(size=3).astype(np.float32),
        "robot0_eef_quat": quat,
        "robot0_gripper_qpos": np.array([0.03, -0.03], dtype=np.float32),
        "instruction": "pick up the bowl",
    }


SMOLVLA = adapt.ModelSpec(
    input={
        "observation.images.image": adapt.Image(
            role=adapt.IMAGE_PRIMARY,
            height=64,
            width=64,
        ),
        "observation.images.image2": adapt.Image(
            role=adapt.IMAGE_WRIST,
            height=64,
            width=64,
        ),
        "observation.state": adapt.Concat(
            adapt.EEF_POS,
            adapt.State(adapt.EEF_ROT, encoding="axis_angle"),
            adapt.GRIPPER_POS,
            container="list",
        ),
        "instruction": adapt.Text(role=adapt.INSTRUCTION),
    },
    output=adapt.Action(
        adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
        adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
    ),
)


def test_smolvla_obs_matches_bespoke_adapter():
    obs = make_obs()
    adapter = resolve(LIBERO_ENV, SMOLVLA)
    payload = adapter.transform_obs(obs)

    np.testing.assert_array_equal(
        payload["observation.images.image"], obs["agentview_image"]
    )
    np.testing.assert_array_equal(
        payload["observation.images.image2"], obs["robot0_eye_in_hand_image"]
    )
    expected_state = np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            ref_quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]
    ).tolist()
    assert payload["observation.state"] == pytest.approx(expected_state)
    assert payload["instruction"] == "pick up the bowl"


def test_adapter_value_path_uses_native_tensor_leaves():
    from rlmesh._rlmesh import Tensor
    from rlmesh.numpy import _numpy_bridge

    obs = make_obs()
    adapter = resolve(LIBERO_ENV, SMOLVLA)
    payload = adapter.transform_obs_value(
        obs, input_bridge=_numpy_bridge, custom_bridge=_numpy_bridge
    )

    assert isinstance(payload, dict)
    assert isinstance(payload["observation.images.image"], Tensor)
    assert isinstance(payload["observation.images.image2"], Tensor)
    assert payload["instruction"] == "pick up the bowl"

    action = adapter.transform_action_value(
        np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7], dtype=np.float32),
        action_bridge=_numpy_bridge,
    )
    assert isinstance(action, Tensor)


def test_smolvla_omits_missing_instruction():
    obs = make_obs()
    del obs["instruction"]
    adapter = resolve(LIBERO_ENV, SMOLVLA)
    payload = adapter.transform_obs(obs)
    assert "instruction" not in payload


def test_smolvla_action_passthrough_with_clip():
    adapter = resolve(LIBERO_ENV, SMOLVLA)
    raw = np.array([0.1, -0.2, 0.3, 1.7, -1.7, 0.0, 0.5], dtype=np.float32)
    result = adapter.transform_action(raw)
    np.testing.assert_allclose(result, np.clip(raw, -1.0, 1.0), rtol=1e-6)


def test_env_gripper_invert_and_binary_flips_model_sign():
    # The env declares its gripper actuator is sign-flipped from the model's
    # convention; the resolved adapter applies the flip and binary snap, in place
    # of a hand-rolled invert_gripper_action escape hatch on the model side.
    env = Env(
        tags=adapt.EnvTags(
            observation={"instruction": adapt.TextTag(role=adapt.INSTRUCTION)},
            action=adapt.Action(
                adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, invert=True, binary=True),
            ),
        ),
        obs_space=gym.spaces.Dict({"instruction": text_space()}),
        action_space=box(1, low=-1.0, high=1.0),
    )
    model = adapt.ModelSpec(
        input={"instruction": adapt.Text(role=adapt.INSTRUCTION)},
        output=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )
    adapter = resolve(env, model)
    # The model says "close" (+0.8); the env's opposite sign + binary snap to -1.
    result = adapter.transform_action(np.array([0.8], dtype=np.float32))
    np.testing.assert_allclose(result, [-1.0], rtol=1e-6)


def test_env_per_actuator_clip_clamps_mixed_ranges() -> None:
    # A mixed-range action: x in [-1, 1] but rot in [-2, 2]. The single global
    # Action.clip applies one (low, high) to the whole vector, so it cannot bound
    # both dims; a per-actuator clip clamps each to its own declared range.
    env = Env(
        tags=adapt.EnvTags(
            observation={"instruction": adapt.TextTag(role=adapt.INSTRUCTION)},
            action=adapt.Action(
                adapt.Actuator("action/x", dim=1, range=(-1.0, 1.0), clip=True),
                adapt.Actuator("action/rot", dim=1, range=(-2.0, 2.0), clip=True),
            ),
        ),
        obs_space=gym.spaces.Dict({"instruction": text_space()}),
        action_space=box(2),
    )
    model = adapt.ModelSpec(
        input={"instruction": adapt.Text(role=adapt.INSTRUCTION)},
        output=adapt.Action(
            adapt.Actuator("action/x", dim=1),
            adapt.Actuator("action/rot", dim=1),
        ),
    )
    adapter = resolve(env, model)
    result = adapter.transform_action(np.array([5.0, 5.0], dtype=np.float32))
    # No single global (low, high) could map [5, 5] to [1, 2]: both overshoot the
    # high bound to *different* clamped values. Only per-dim clipping does this.
    np.testing.assert_allclose(result, [1.0, 2.0], rtol=1e-6)


def test_opaque_env_actuator_fills_unmatched_dims() -> None:
    # The env action has dims no model produces (e.g. a control-mode selector);
    # a role-less Actuator(dim=, fill=) occupies them with a constant.
    env = Env(
        tags=adapt.EnvTags(
            observation={"instruction": adapt.TextTag(role=adapt.INSTRUCTION)},
            action=adapt.Action(
                adapt.Actuator(adapt.ACTION_GRIPPER, dim=1),
                adapt.Actuator(dim=2, fill=0.25),  # opaque: no role
            ),
        ),
        obs_space=gym.spaces.Dict({"instruction": text_space()}),
        action_space=box(3),
    )
    model = adapt.ModelSpec(
        input={"instruction": adapt.Text(role=adapt.INSTRUCTION)},
        output=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )
    adapter = resolve(env, model)
    result = adapter.transform_action(np.array([0.8], dtype=np.float32))
    np.testing.assert_allclose(result, [0.8, 0.25, 0.25], rtol=1e-6)


def test_opaque_actuator_rejects_corrections_and_roled_fill() -> None:
    with pytest.raises(ValueError, match="opaque"):
        adapt.Actuator(dim=2, range=(-1.0, 1.0))  # role-less cannot carry range
    with pytest.raises(ValueError, match="fill applies"):
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, fill=0.5)  # roled cannot fill


def test_actuator_clip_requires_range() -> None:
    # clip clamps to range, so declaring clip without a range is a contradiction
    # caught at construction (the resolver enforces it too).
    with pytest.raises(ValueError, match="clip"):
        adapt.Actuator("action/x", dim=1, clip=True)


def test_action_component_positional_binary_is_unchanged() -> None:
    # scale/invert/threshold are appended after binary, so the old positional
    # layout (role, dim, encoding, range, binary) keeps its meaning.
    c = adapt.Actuator(adapt.ACTION_GRIPPER, 1, None, None, True)
    assert c.binary is True
    assert c.scale is None and c.invert is False and c.threshold is None


def test_action_layout_loosely_typed_corrections_rejected() -> None:
    # The authoritative Rust codec (reached via from_dict) rejects a truthy
    # non-bool invert/binary or a numeric-string scale/threshold, so both
    # bindings agree on hand-authored or third-party layout JSON.
    def spec(**correction: Any) -> dict[str, Any]:
        return {
            "input": {},
            "output": {
                "components": [{"role": adapt.ACTION_GRIPPER, "dim": 1, **correction}]
            },
        }

    for bad in (
        {"invert": 1},
        {"invert": "yes"},
        {"scale": "2"},
        {"threshold": True},
        {"binary": 1},
    ):
        with pytest.raises(ValueError):
            adapt.ModelSpec.from_dict(spec(**bad))

    ok = adapt.ModelSpec.from_dict(
        spec(invert=True, scale=2.0, threshold=0.5, binary=True)
    )
    assert ok.output.components[0].invert is True
    assert ok.output.components[0].scale == 2.0
    assert ok.output.components[0].threshold == 0.5


OPENVLA = adapt.ModelSpec(
    input={
        "image": adapt.Image(role=adapt.IMAGE_PRIMARY, height=64, width=64),
        "instruction": adapt.Text(role=adapt.INSTRUCTION),
    },
    output=adapt.Action(
        adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
        adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
    ),
)


def test_openvla_obs_matches_bespoke_adapter():
    obs = make_obs()
    adapter = resolve(LIBERO_ENV, OPENVLA)
    payload = adapter.transform_obs(obs)
    assert set(payload) == {"image", "instruction"}
    np.testing.assert_array_equal(payload["image"], obs["agentview_image"])


XVLA = adapt.ModelSpec(
    input={
        "image": adapt.Image(role=adapt.IMAGE_PRIMARY, height=64, width=64),
        "image2": adapt.Image(role=adapt.IMAGE_WRIST, height=64, width=64),
        "state": adapt.Concat(
            adapt.State(adapt.EEF_POS, dim=3),
            # X-VLA uses row-major rot6d for both proprio and action (the
            # m[:, :2].reshape(6) convention from upstream datasets/utils.py
            # and the Isaac_vla bridge client).
            adapt.State(adapt.EEF_ROT, encoding="rot6d_rowmajor"),
            adapt.State(adapt.GRIPPER_POS, dim=1),
            adapt.State(adapt.EEF_POS_2, dim=3, optional=True),
            adapt.State(adapt.EEF_ROT_2, encoding="rot6d_rowmajor", optional=True),
            adapt.State(adapt.GRIPPER_POS_2, dim=1, optional=True),
            pad_to=20,
            container="list",
        ),
        "instruction": adapt.Text(role=adapt.INSTRUCTION),
    },
    output=adapt.Action(
        adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
        adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=6, encoding="rot6d_rowmajor"),
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
        adapt.Actuator(adapt.ACTION_DELTA_POS_2, dim=3),
        adapt.Actuator(adapt.ACTION_DELTA_ROT_2, dim=6, encoding="rot6d_rowmajor"),
        adapt.Actuator(adapt.ACTION_GRIPPER_2, dim=1, range=(-1.0, 1.0)),
    ),
)


def test_xvla_state_matches_bespoke_adapter():
    obs = make_obs()
    adapter = resolve(LIBERO_ENV, XVLA)
    payload = adapter.transform_obs(obs)

    expected_state = np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            ref_quat2rot6d(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)[:1],
            np.zeros(10, dtype=np.float32),
        ]
    )
    assert len(payload["state"]) == 20
    assert payload["state"] == pytest.approx(expected_state.tolist())


def test_xvla_action_matches_bespoke_adapter():
    obs_adapter = resolve(LIBERO_ENV, XVLA)
    rng = np.random.default_rng(3)
    raw = rng.normal(size=20).astype(np.float32)

    expected = np.clip(
        np.concatenate([raw[:3], ref_r6d_to_rotvec(raw[3:9]), raw[9:10]]),
        -1.0,
        1.0,
    )
    np.testing.assert_allclose(
        obs_adapter.transform_action(raw), expected, rtol=1e-5, atol=1e-6
    )


BIMANUAL_ENV = Env(
    tags=adapt.EnvTags(
        observation={
            "agentview_image": adapt.ImageTag(role=adapt.IMAGE_PRIMARY),
            "robot0_eye_in_hand_image": adapt.ImageTag(role=adapt.IMAGE_WRIST),
            "robot0_eef_pos": adapt.StateTag(role=adapt.EEF_POS),
            "robot0_eef_quat": adapt.StateTag(role=adapt.EEF_ROT, encoding="quat_xyzw"),
            "robot0_gripper_qpos": adapt.StateTag(role=adapt.GRIPPER_POS),
            "robot1_eef_pos": adapt.StateTag(role=adapt.EEF_POS_2),
            "robot1_eef_quat": adapt.StateTag(
                role=adapt.EEF_ROT_2, encoding="quat_xyzw"
            ),
            "robot1_gripper_qpos": adapt.StateTag(role=adapt.GRIPPER_POS_2),
            "instruction": adapt.TextTag(role=adapt.INSTRUCTION),
        },
        action=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
            adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
            adapt.Actuator(adapt.ACTION_DELTA_POS_2, dim=3),
            adapt.Actuator(adapt.ACTION_DELTA_ROT_2, dim=3, encoding="axis_angle"),
            adapt.Actuator(adapt.ACTION_GRIPPER_2, dim=1, range=(-1.0, 1.0)),
            clip=(-1.0, 1.0),
        ),
    ),
    obs_space=gym.spaces.Dict(
        {
            "agentview_image": image_space(),
            "robot0_eye_in_hand_image": image_space(),
            "robot0_eef_pos": box(3),
            "robot0_eef_quat": box(4),
            "robot0_gripper_qpos": box(2),
            "robot1_eef_pos": box(3),
            "robot1_eef_quat": box(4),
            "robot1_gripper_qpos": box(2),
            "instruction": text_space(),
        }
    ),
    action_space=ACTION14,
)


def make_bimanual_obs() -> dict[str, object]:
    obs = make_obs()
    rng = np.random.default_rng(11)
    quat = rng.normal(size=4).astype(np.float32)
    quat /= np.linalg.norm(quat)
    obs["robot1_eef_pos"] = rng.normal(size=3).astype(np.float32)
    obs["robot1_eef_quat"] = quat
    obs["robot1_gripper_qpos"] = np.array([0.02, -0.02], dtype=np.float32)
    return obs


def test_xvla_state_consumes_second_arm_on_bimanual_env():
    """X-VLA's unified 20-dim single/bimanual state and ee6d action layout:
    dims 1-10 are the first arm, dims 11-20 the second; second-arm components
    are optional, so single-arm envs resolve them to zero fill / dropped dims.
    """
    obs = make_bimanual_obs()
    adapter = resolve(BIMANUAL_ENV, XVLA)
    payload = adapter.transform_obs(obs)

    expected_state = np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            ref_quat2rot6d(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)[:1],
            np.asarray(obs["robot1_eef_pos"], dtype=np.float32),
            ref_quat2rot6d(obs["robot1_eef_quat"]),
            np.asarray(obs["robot1_gripper_qpos"], dtype=np.float32)[:1],
        ]
    )
    assert len(payload["state"]) == 20
    assert payload["state"] == pytest.approx(expected_state.tolist())


def test_xvla_action_consumes_second_arm_on_bimanual_env():
    adapter = resolve(BIMANUAL_ENV, XVLA)
    rng = np.random.default_rng(5)
    raw = rng.normal(size=20).astype(np.float32)

    expected = np.clip(
        np.concatenate(
            [
                raw[:3],
                ref_r6d_to_rotvec(raw[3:9]),
                raw[9:10],
                raw[10:13],
                ref_r6d_to_rotvec(raw[13:19]),
                raw[19:20],
            ]
        ),
        -1.0,
        1.0,
    )
    np.testing.assert_allclose(
        adapter.transform_action(raw), expected, rtol=1e-5, atol=1e-6
    )


def test_optional_state_without_width_is_an_error():
    spec = adapt.ModelSpec(
        input={"state": adapt.State("proprio/extra", optional=True)},
        output=SMOLVLA.output,
    )
    with pytest.raises(adapt.AdapterResolutionError, match="zero fill"):
        resolve(LIBERO_ENV, spec)


def test_describe_mentions_zero_fill_for_absent_optional_roles():
    text = resolve(LIBERO_ENV, XVLA).explain()
    assert "zeros(3)" in text and "zeros(6)" in text and "zeros(1)" in text


def test_role_constants_match_rust_crate():
    """Roles are single-sourced from the crate; this catches a role added
    to ``roles/*.rs`` but not exposed through the binding's table."""
    import re
    from pathlib import Path

    roles_dir = (
        Path(__file__).resolve().parents[4]
        / "crates"
        / "rlmesh-adapters"
        / "src"
        / "roles"
    )
    rust_roles: dict[str, str] = {}
    for path in roles_dir.glob("*.rs"):
        for name, value in re.findall(
            r'pub const (\w+): &str = "([^"]+)";', path.read_text()
        ):
            rust_roles[name] = value
    assert rust_roles, "no role constants found in the Rust crate"

    from rlmesh.adapters import constants

    python_roles = {
        name: getattr(constants, name)
        for name in constants.__all__
        if not name.endswith("_METADATA_KEY")
    }
    assert python_roles == rust_roles


def test_custom_adapter_subclass_is_interchangeable():
    class JointSpaceAdapter(adapt.AdapterBase[np.ndarray]):
        """Stateful custom adapter: uses proprio cached at obs time."""

        def __init__(self):
            self._joint_pos = np.zeros(3, dtype=np.float32)

        def transform_obs(self, raw_obs):
            self._joint_pos = np.asarray(raw_obs["robot0_eef_pos"], np.float32)
            return {"state": self._joint_pos}

        def transform_action(self, raw_action) -> np.ndarray:
            return self._joint_pos + np.asarray(raw_action, np.float32)

    adapter = JointSpaceAdapter()
    obs = make_obs()
    action = adapter.wrap_predict(lambda payload: np.ones(3, np.float32))(obs)
    np.testing.assert_allclose(
        action, np.asarray(obs["robot0_eef_pos"], np.float32) + 1.0
    )
    assert "JointSpaceAdapter" in adapter.explain()
    assert isinstance(resolve(LIBERO_ENV, SMOLVLA), adapt.AdapterBase)


def test_custom_adapter_reset_is_a_no_op_by_default():
    adapter = resolve(LIBERO_ENV, SMOLVLA)
    # Resolved adapters are stateless; reset must exist and do nothing.
    adapter.reset()
    payload = adapter.transform_obs(make_obs())
    assert "observation.state" in payload


GR00T = adapt.ModelSpec(
    input={
        "video.image": adapt.Image(
            role=adapt.IMAGE_PRIMARY,
            height=64,
            width=64,
            lead_dims=2,
            upside_down=True,
        ),
        "state.x": adapt.State(adapt.EEF_POS, index=0, reshape=(1, 1, 1)),
        "state.roll": adapt.State(
            adapt.EEF_ROT, encoding="axis_angle", index=0, reshape=(1, 1, 1)
        ),
        "state.gripper": adapt.State(adapt.GRIPPER_POS, index=0, reshape=(1, 1, 1)),
        "tag.human.action.task_description": adapt.Text(
            role=adapt.INSTRUCTION,
            container="list",
            fill="",
        ),
    },
    output=adapt.Action(
        adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
        adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(0.0, 1.0), binary=True),
    ),
)


def test_gr00t_obs_matches_bespoke_adapter():
    obs = make_obs()
    adapter = resolve(LIBERO_ENV, GR00T)
    payload = adapter.transform_obs(obs)

    image = np.asarray(obs["agentview_image"])
    expected_image = np.ascontiguousarray(image[::-1, ::-1])[None, None, ...]
    np.testing.assert_array_equal(payload["video.image"], expected_image)
    assert payload["video.image"].shape == (1, 1, 64, 64, 3)

    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    axisangle = ref_quat2axisangle(obs["robot0_eef_quat"])
    assert payload["state.x"].shape == (1, 1, 1)
    assert payload["state.x"][0, 0, 0] == pytest.approx(pos[0])
    assert payload["state.roll"][0, 0, 0] == pytest.approx(axisangle[0])
    assert payload["state.gripper"][0, 0, 0] == pytest.approx(0.03)
    assert payload["tag.human.action.task_description"] == ["pick up the bowl"]


def test_gr00t_default_instruction_when_missing():
    obs = make_obs()
    del obs["instruction"]
    adapter = resolve(LIBERO_ENV, GR00T)
    payload = adapter.transform_obs(obs)
    assert payload["tag.human.action.task_description"] == [""]


def test_gr00t_action_gripper_sign_matches_bespoke_adapter():
    adapter = resolve(LIBERO_ENV, GR00T)
    raw = np.array([0.1, 0.2, 0.3, 0.0, -0.1, 0.2, 0.8], dtype=np.float32)
    result = adapter.transform_action(raw)
    np.testing.assert_allclose(result[:6], raw[:6], rtol=1e-6)
    assert result[6] == np.sign(2.0 * 0.8 - 1.0)

    raw[6] = 0.2
    assert adapter.transform_action(raw)[6] == np.sign(2.0 * 0.2 - 1.0)


def image_env(height: int, width: int, *, role: str = adapt.IMAGE_PRIMARY) -> Env:
    """A minimal single-image env (plus instruction) over a given image size."""
    return Env(
        tags=adapt.EnvTags(
            observation={
                "rgb": adapt.ImageTag(role=role),
                "instruction": adapt.TextTag(role=adapt.INSTRUCTION),
            },
            action=LIBERO_ACTION,
        ),
        obs_space=gym.spaces.Dict(
            {"rgb": image_space(height, width), "instruction": text_space()}
        ),
        action_space=ACTION7,
    )


def test_image_resize_layout_and_normalize():
    obs = make_obs(size=32)
    spec = adapt.ModelSpec(
        input={
            "pixels": adapt.Image(
                role=adapt.IMAGE_PRIMARY,
                height=16,
                width=16,
                layout="chw",
                dtype="float32",
                normalize=True,
            ),
        },
        output=SMOLVLA.output,
    )
    payload = resolve(LIBERO_ENV, spec).transform_obs(obs)
    pixels = payload["pixels"]
    assert pixels.shape == (3, 16, 16)
    assert pixels.dtype == np.float32
    assert float(pixels.max()) <= 1.0
    assert float(pixels.min()) >= 0.0


def _resized(env: Env, image: np.ndarray, height: int, width: int, resample: str):
    """Our resize of `image`, as int16 so a comparison can go negative."""
    spec = adapt.ModelSpec(
        input={
            "image": adapt.Image(
                role=adapt.IMAGE_PRIMARY,
                height=height,
                width=width,
                # The upscale cases interpolate detail the env image does not
                # have, which the resolver gates behind allow_upscale; these
                # anchors deliberately exercise both directions.
                allow_upscale=True,
                fit="stretch",
                resample=resample,
            )
        },
        output=SMOLVLA.output,
    )
    return resolve(env, spec).transform_obs({"rgb": image})["image"].astype(np.int16)


def _anchor_images(height: int, width: int) -> dict[str, np.ndarray]:
    """A smooth ramp, white noise, and a hard edge (the ringing case)."""
    edge = np.zeros((height, width, 3), np.uint8)
    edge[:, : width // 2] = 255
    return {
        "ramp": (np.arange(height * width * 3, dtype=np.int64) * 7 % 251)
        .astype(np.uint8)
        .reshape(height, width, 3),
        "noise": np.random.default_rng(7).integers(
            0, 256, (height, width, 3), dtype=np.uint8
        ),
        "edge": edge,
    }


@pytest.mark.parametrize(
    ("resample", "pil_filter"),
    [
        ("bilinear_aa", "BILINEAR"),
        ("bicubic_aa", "BICUBIC"),
        ("lanczos3_aa", "LANCZOS"),
    ],
)
def test_aa_resize_matches_pillow_within_one_step(resample: str, pil_filter: str):
    """The `_aa` kernels are PIL's, to one uint8 step, in both directions.

    The hard-edge image is the load-bearing case: cubic and Lanczos ring, and
    PIL clips that overshoot in its 8-bit intermediate between the two passes,
    so a float64 pipeline that clips only at the end drifts by tens of levels.
    """
    pil = pytest.importorskip("PIL.Image")
    theirs_filter = getattr(pil.Resampling, pil_filter)
    for src_height, src_width in ((6, 8), (32, 32)):
        env = image_env(src_height, src_width)
        for image in _anchor_images(src_height, src_width).values():
            for height, width in ((3, 4), (12, 16), (src_height * 2, src_width * 2)):
                ours = _resized(env, image, height, width, resample)
                theirs = np.asarray(
                    pil.fromarray(image).resize((width, height), theirs_filter),
                    dtype=np.int16,
                )
                assert int(np.abs(ours - theirs).max()) <= 1


def test_area_resize_matches_opencv_within_one_step():
    """`area` is cv2's INTER_AREA, to one uint8 step, in both directions.

    Upscaling is the load-bearing half: INTER_AREA does not widen a filter the
    way the `_aa` kernels do, it splits each output pixel's sub-pixel footprint
    across the one or two source pixels it covers.
    """
    cv2 = pytest.importorskip("cv2")
    for src_height, src_width in ((6, 8), (32, 32)):
        env = image_env(src_height, src_width)
        for image in _anchor_images(src_height, src_width).values():
            for height, width in ((3, 4), (12, 16), (src_height * 2, src_width * 2)):
                ours = _resized(env, image, height, width, "area")
                theirs = cv2.resize(
                    image, (width, height), interpolation=cv2.INTER_AREA
                ).astype(np.int16)
                assert int(np.abs(ours - theirs).max()) <= 1


def _cropped(
    env: Env, image: np.ndarray, height: int, width: int, resample: str, **crop: object
):
    """Our crop+resize of `image`, as int16 so a comparison can go negative."""
    spec = adapt.ModelSpec(
        input={
            "image": adapt.Image(
                role=adapt.IMAGE_PRIMARY,
                height=height,
                width=width,
                allow_upscale=True,
                fit="stretch",
                resample=resample,
                **crop,  # type: ignore[arg-type]
            )
        },
        output=SMOLVLA.output,
    )
    return resolve(env, spec).transform_obs({"rgb": image})["image"].astype(np.int16)


@pytest.mark.parametrize(
    ("resample", "pil_filter"),
    [
        ("bilinear_aa", "BILINEAR"),
        ("bicubic_aa", "BICUBIC"),
        ("lanczos3_aa", "LANCZOS"),
    ],
)
def test_zoom_crop_matches_pillow_box_resize_within_one_step(
    resample: str, pil_filter: str
):
    """A zoom crop is PIL's `Image.resize(size, box=...)`, to one uint8 step.

    That is the whole point of the mode: the fractional box is resampled
    straight to the target in one pass, so it must agree with the library the
    training pipelines used -- including at the box edge, where the filter
    still reaches into the neighbouring pixels rather than stopping at the cut.
    """
    pil = pytest.importorskip("PIL.Image")
    theirs_filter = getattr(pil.Resampling, pil_filter)
    for src in (16, 32):
        env = image_env(src, src)
        for image in _anchor_images(src, src).values():
            for fraction in (0.5, 0.9**0.5, 1.0):
                span = src * fraction
                start = (src - span) / 2.0
                box = (start, start, start + span, start + span)
                for size in (8, src, src * 2):
                    ours = _cropped(env, image, size, size, resample, crop=fraction)
                    theirs = np.asarray(
                        pil.fromarray(image).resize(
                            (size, size), theirs_filter, box=box
                        ),
                        dtype=np.int16,
                    )
                    assert int(np.abs(ours - theirs).max()) <= 1


def test_crop_area_is_the_square_of_the_side_fraction():
    """`crop_area=0.9` and `crop=sqrt(0.9)` name the same box, pixel for pixel."""
    env = image_env(32, 32)
    image = _anchor_images(32, 32)["noise"]
    by_area = _cropped(env, image, 16, 16, "bilinear_aa", crop_area=0.9)
    by_side = _cropped(env, image, 16, 16, "bilinear_aa", crop=0.9**0.5)
    assert np.array_equal(by_area, by_side)


def test_slice_crop_cuts_integer_pixels_before_the_resize():
    """`crop_mode="slice"` is a numpy center cut, then a plain resize of it."""
    env = image_env(12, 12)
    image = _anchor_images(12, 12)["ramp"]
    ours = _cropped(env, image, 4, 4, "area", crop=2 / 3, crop_mode="slice")
    # The cut keeps the middle 8x8 (12 * 2/3), and the resize sees only that.
    cut_env = image_env(8, 8)
    cut = image[2:10, 2:10]
    assert np.array_equal(ours, _cropped(cut_env, cut, 4, 4, "area"))


def _photo_frame(height: int, width: int) -> np.ndarray:
    """A natural-looking frame: a luma-dominant gradient plus two hard blocks.

    Deliberately not noise. JPEG is tuned for photographic content, so a noise
    frame would measure the codec somewhere no camera ever puts it -- and the
    chroma planes of a luma-dominant image are smooth, which is what 4:2:0
    subsampling assumes.
    """
    rows = np.arange(height)[:, None] / height
    cols = np.arange(width)[None, :] / width
    luma = 30.0 + 190.0 * (0.6 * cols + 0.4 * rows)
    frame = np.stack([luma * 1.02, luma * 0.96, luma * 0.88], axis=-1)
    frame[height // 3 : height * 2 // 3, width // 4 : width // 2] = (236, 231, 214)
    frame[:, width * 3 // 4 : width * 3 // 4 + max(1, width // 16)] = (26, 25, 22)
    return np.clip(np.round(frame), 0, 255).astype(np.uint8)


def _jpeg_roundtripped(env: Env, image: np.ndarray, quality: int):
    """Our JPEG round-trip of `image`, as int16 so a comparison can go negative."""
    spec = adapt.ModelSpec(
        input={"image": adapt.Image(role=adapt.IMAGE_PRIMARY, jpeg_quality=quality)},
        output=SMOLVLA.output,
    )
    return resolve(env, spec).transform_obs({"rgb": image})["image"].astype(np.int16)


def test_jpeg_roundtrip_is_near_pillows_q95_jpeg():
    """`jpeg_quality` reproduces what Pillow (libjpeg) writes, near-exactly.

    Near, not byte-identical: the encoders agree on the profile (baseline,
    4:2:0, standard IJG tables) but libjpeg's IDCT and the decoder behind the
    `image` crate round differently in the last place. Pinned at the tolerance
    the openvla-oft A/B needs -- within 2 levels on at least 99% of pixels and
    4 anywhere. On the natural frame it is tighter than that: every pixel
    within 2, and 98.7% within 1.
    """
    pil = pytest.importorskip("PIL.Image")
    for src in (32, 64):
        env = image_env(src, src)
        images = dict(_anchor_images(src, src), photo=_photo_frame(src, src))
        for image in images.values():
            ours = _jpeg_roundtripped(env, image, 95)
            buffer = io.BytesIO()
            # Pillow's default subsampling for an RGB save IS 4:2:0, which is
            # the profile we pin; naming it here would hide a change of theirs.
            pil.fromarray(image).save(buffer, "JPEG", quality=95)
            buffer.seek(0)
            theirs = np.asarray(pil.open(buffer).convert("RGB"), dtype=np.int16)
            delta = np.abs(ours - theirs)
            assert float((delta <= 2).mean()) >= 0.99
            assert int(delta.max()) <= 4


def test_jpeg_roundtrip_loses_more_at_a_lower_quality():
    """The quality really is the IJG dial, not a decorative field."""
    env = image_env(64, 64)
    image = _photo_frame(64, 64)
    original = image.astype(np.int16)
    fine = float(np.abs(_jpeg_roundtripped(env, image, 95) - original).mean())
    coarse = float(np.abs(_jpeg_roundtripped(env, image, 10) - original).mean())
    assert coarse > fine


def test_jpeg_quality_needs_a_three_channel_camera():
    """JPEG subsampling is defined on YCbCr; a grayscale camera has none."""
    env = Env(
        tags=adapt.EnvTags(
            observation={"rgb": adapt.ImageTag(role=adapt.IMAGE_PRIMARY)},
            action=LIBERO_ACTION,
        ),
        obs_space=gym.spaces.Dict(
            {"rgb": gym.spaces.Box(low=0, high=255, shape=(8, 8, 1), dtype=np.uint8)}
        ),
        action_space=ACTION7,
    )
    spec = adapt.ModelSpec(
        input={"image": adapt.Image(role=adapt.IMAGE_PRIMARY, jpeg_quality=95)},
        output=SMOLVLA.output,
    )
    with pytest.raises(adapt.AdapterResolutionError, match="3-channel"):
        resolve(env, spec)


def test_jpeg_quality_is_bounded_to_the_ijg_scale_at_construction():
    for bad in (0, 101):
        with pytest.raises(ValueError, match="between 1 and 100"):
            adapt.Image(role=adapt.IMAGE_PRIMARY, jpeg_quality=bad)
    assert adapt.Image(role=adapt.IMAGE_PRIMARY, jpeg_quality=1).jpeg_quality == 1
    assert adapt.Image(role=adapt.IMAGE_PRIMARY, jpeg_quality=100).jpeg_quality == 100


def test_bgr_swaps_red_and_blue_after_the_resize():
    """The swap is the last spatial-adjacent step: resize in RGB, then swap.

    Order matters because the resize is per-channel -- swapping first and
    resizing after would give the same pixels, but swapping before a *crop*
    would not, and one order has to be the contract.
    """
    env = image_env(8, 8)
    image = _anchor_images(8, 8)["noise"]
    rgb = _cropped(env, image, 4, 4, "bilinear_aa")
    bgr = _cropped(env, image, 4, 4, "bilinear_aa", channel_order="bgr")
    assert np.array_equal(bgr, rgb[:, :, ::-1])


def test_bgr_needs_a_three_channel_camera():
    env = Env(
        tags=adapt.EnvTags(
            observation={"rgb": adapt.ImageTag(role=adapt.IMAGE_PRIMARY)},
            action=LIBERO_ACTION,
        ),
        obs_space=gym.spaces.Dict(
            {"rgb": gym.spaces.Box(low=0, high=255, shape=(8, 8, 1), dtype=np.uint8)}
        ),
        action_space=ACTION7,
    )
    spec = adapt.ModelSpec(
        input={"image": adapt.Image(role=adapt.IMAGE_PRIMARY, channel_order="bgr")},
        output=SMOLVLA.output,
    )
    with pytest.raises(adapt.AdapterResolutionError, match="3-channel"):
        resolve(env, spec)


def test_crop_guards_reject_an_impossible_box_at_construction():
    """The two fractions are one box said two ways, and the range is (0, 1]."""
    with pytest.raises(ValueError, match="not both"):
        adapt.Image(role=adapt.IMAGE_PRIMARY, crop=0.5, crop_area=0.25)
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match=r"fraction in \(0, 1\]"):
            adapt.Image(role=adapt.IMAGE_PRIMARY, crop=bad)
        with pytest.raises(ValueError, match=r"fraction in \(0, 1\]"):
            adapt.Image(role=adapt.IMAGE_PRIMARY, crop_area=bad)
    # The inclusive end (the whole frame) is a legal, if inert, box.
    assert adapt.Image(role=adapt.IMAGE_PRIMARY, crop_area=1).crop_area == 1.0


def test_crop_and_channel_order_serialize_omit_when_default():
    """Every new field is omitted at its default and agrees with the core."""
    import json

    from rlmesh._rlmesh import adapters_spec_normalize

    output = adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1))
    plain = adapt.ModelSpec(
        input={"image": adapt.Image(adapt.IMAGE_PRIMARY, size=224)}, output=output
    )
    leaf = plain.to_dict()["input"]["image"]
    for field in ("crop", "crop_area", "crop_mode", "jpeg_quality", "channel_order"):
        assert field not in leaf, f"{field} leaked into a spec that never set it"

    every = adapt.ModelSpec(
        input={
            "image": adapt.Image(
                adapt.IMAGE_PRIMARY,
                size=224,
                crop_area=0.9,
                crop_mode="slice",
                jpeg_quality=95,
                channel_order="bgr",
            )
        },
        output=output,
    )
    doc = every.to_dict()
    assert doc["input"]["image"]["crop_area"] == 0.9
    assert doc["input"]["image"]["crop_mode"] == "slice"
    assert doc["input"]["image"]["jpeg_quality"] == 95
    assert doc["input"]["image"]["channel_order"] == "bgr"
    assert adapt.ModelSpec.from_dict(doc) == every
    # Cross-engine: the core's canonical form of each spec is the spec itself,
    # so Python and Rust cannot disagree on what these fields serialize to.
    for spec in (plain, every):
        canonical = adapters_spec_normalize("model", json.dumps(spec.to_dict()), True)
        assert json.loads(canonical) == spec.to_dict()


def test_render_asserts_the_bound_cameras_size():
    """`render` is an assertion about the camera, not a resize request.

    The platform binds the env's camera dial from it before the run; this is
    the check that the dial actually moved. A camera that renders at the
    asserted size resolves, one that does not fails loudly rather than feeding
    the model frames at the wrong scale.
    """
    spec = adapt.ModelSpec(
        input={"image": adapt.Image(role=adapt.IMAGE_PRIMARY, render=448)},
        output=SMOLVLA.output,
    )
    assert resolve(image_env(448, 448), spec) is not None
    with pytest.raises(adapt.AdapterResolutionError, match="declares render 448x448"):
        resolve(image_env(256, 256), spec)


def test_render_guards_reject_an_impossible_size_at_construction():
    """A square int widens to the pair the wire carries; each axis is 1-4096."""
    assert adapt.Image(role=adapt.IMAGE_PRIMARY, render=448).render == (448, 448)
    assert adapt.Image(role=adapt.IMAGE_PRIMARY, render=(480, 640)).render == (480, 640)
    for bad in (0, 4097, (0, 448), (448, 4097)):
        with pytest.raises(ValueError, match="must be between 1 and 4096"):
            adapt.Image(role=adapt.IMAGE_PRIMARY, render=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"an int or a \(height, width\) pair"):
        adapt.Image(role=adapt.IMAGE_PRIMARY, render=(448, 448, 448))  # type: ignore[arg-type]


def test_render_serializes_as_a_pair_and_is_omitted_when_unset():
    """Omitted at its default, `[height, width]` when set, and core-identical."""
    import json

    from rlmesh._rlmesh import adapters_spec_normalize

    output = adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1))
    plain = adapt.ModelSpec(
        input={"image": adapt.Image(adapt.IMAGE_PRIMARY, size=224)}, output=output
    )
    assert "render" not in plain.to_dict()["input"]["image"]

    asserted = adapt.ModelSpec(
        input={"image": adapt.Image(adapt.IMAGE_PRIMARY, size=224, render=448)},
        output=output,
    )
    doc = asserted.to_dict()
    assert doc["input"]["image"]["render"] == [448, 448]
    assert adapt.ModelSpec.from_dict(doc) == asserted
    # Cross-engine: the core's canonical form of each spec is the spec itself.
    for spec in (plain, asserted):
        canonical = adapters_spec_normalize("model", json.dumps(spec.to_dict()), True)
        assert json.loads(canonical) == spec.to_dict()


def test_bare_bicubic_and_lanczos3_are_not_resample_names():
    """The suffix rule is enforced, not just documented: an un-suffixed cubic
    or Lanczos name would silently pick one library's kernel over the other's,
    so resolution rejects both."""
    env = image_env(6, 8)
    for name in ("bicubic", "lanczos3"):
        spec = adapt.ModelSpec(
            input={
                "image": adapt.Image(
                    role=adapt.IMAGE_PRIMARY, height=3, width=4, resample=name
                )
            },
            output=SMOLVLA.output,
        )
        with pytest.raises(adapt.AdapterResolutionError, match="unsupported resample"):
            resolve(env, spec)


def make_png(pixels: np.ndarray) -> bytes:
    """Minimal RGB8 PNG encoder (stdlib only), for byte-decoding tests."""
    import struct
    import zlib

    height, width, _ = pixels.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def test_encoded_image_bytes_decode_natively():
    pixels = (
        (np.arange(2 * 2 * 3, dtype=np.int64) * 9 % 251)
        .astype(np.uint8)
        .reshape(2, 2, 3)
    )
    env = image_env(2, 2)
    spec = adapt.ModelSpec(
        input={"image": adapt.Image(role=adapt.IMAGE_PRIMARY)},
        output=SMOLVLA.output,
    )
    payload = resolve(env, spec).transform_obs({"rgb": make_png(pixels)})
    np.testing.assert_array_equal(payload["image"], pixels)


def test_undecodable_image_bytes_is_an_error():
    env = image_env(2, 2)
    spec = adapt.ModelSpec(
        input={"image": adapt.Image(role=adapt.IMAGE_PRIMARY)},
        output=SMOLVLA.output,
    )
    with pytest.raises(ValueError, match="could not decode image bytes"):
        resolve(env, spec).transform_obs({"rgb": b"not an image"})


def test_bytes_are_rejected_for_non_image_adapter_inputs():
    env = Env(
        tags=adapt.EnvTags(
            observation={"state": adapt.StateTag(role=adapt.EEF_POS)},
            action=SMOLVLA.output,
        ),
        obs_space=gym.spaces.Dict({"state": box(3)}),
        action_space=ACTION7,
    )
    spec = adapt.ModelSpec(
        input={"state": adapt.State(adapt.EEF_POS)},
        output=SMOLVLA.output,
    )

    with pytest.raises(ValueError, match="bytes values are only valid"):
        resolve(env, spec).transform_obs({"state": b"not an image"})


def test_bilinear_resize_preserves_constant_images():
    env = image_env(10, 12)
    spec = adapt.ModelSpec(
        input={
            "image": adapt.Image(
                role=adapt.IMAGE_PRIMARY,
                height=4,
                width=5,
                resample="bilinear",
                fit="stretch",
            ),
        },
        output=SMOLVLA.output,
    )
    payload = resolve(env, spec).transform_obs(
        {"rgb": np.full((10, 12, 3), 117, dtype=np.uint8)}
    )
    assert payload["image"].shape == (4, 5, 3)
    np.testing.assert_array_equal(
        payload["image"], np.full((4, 5, 3), 117, dtype=np.uint8)
    )


def test_single_env_image_fallback_match():
    env = image_env(8, 8, role=adapt.IMAGE_SECONDARY)
    spec = adapt.ModelSpec(
        input={"image": adapt.Image(role=adapt.IMAGE_PRIMARY)},
        output=SMOLVLA.output,
    )
    adapter = resolve(env, spec)
    obs = {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}
    assert adapter.transform_obs(obs)["image"].shape == (8, 8, 3)


def single_state_env(
    key: str, obs_space: gym.spaces.Space[Any], role: str = adapt.EEF_POS
) -> Env:
    """An env with one state (``EEF_POS`` by default) under ``key``."""
    return Env(
        tags=adapt.EnvTags(
            observation={key: adapt.StateTag(role=role)},
            action=LIBERO_ACTION,
        ),
        obs_space=obs_space,
        action_space=ACTION7,
    )


STATE_ONLY_MODEL = adapt.ModelSpec(
    input={"state": adapt.State(adapt.EEF_POS)},
    output=SMOLVLA.output,
)


def test_nested_observation_keys():
    # Nesting is now a structural tag tree (a nested dict), not a dotted key.
    env = Env(
        tags=adapt.EnvTags(
            observation={"agent": {"eef_pos": adapt.StateTag(role=adapt.EEF_POS)}},
            action=LIBERO_ACTION,
        ),
        obs_space=gym.spaces.Dict({"agent": gym.spaces.Dict({"eef_pos": box(3)})}),
        action_space=ACTION7,
    )
    adapter = resolve(env, STATE_ONLY_MODEL)
    obs = {"agent": {"eef_pos": [1.0, 2.0, 3.0]}}
    np.testing.assert_allclose(adapter.transform_obs(obs)["state"], [1.0, 2.0, 3.0])


def test_numeric_payload_data_mapping():
    env = single_state_env("pos", gym.spaces.Dict({"pos": box(3)}))
    adapter = resolve(env, STATE_ONLY_MODEL)
    obs = {"pos": {"data": [4.0, 5.0, 6.0]}}
    np.testing.assert_allclose(adapter.transform_obs(obs)["state"], [4.0, 5.0, 6.0])


def test_custom_callable_transform():
    spec = adapt.ModelSpec(
        input={
            "engineered": adapt.Custom(
                transform=lambda obs: float(obs["robot0_eef_pos"][0])
            ),
        },
        output=SMOLVLA.output,
    )
    adapter = resolve(LIBERO_ENV, spec)
    obs = make_obs()
    assert adapter.transform_obs(obs)["engineered"] == pytest.approx(
        float(np.asarray(obs["robot0_eef_pos"])[0])
    )


def test_custom_entrypoint_requires_trust():
    spec = adapt.ModelSpec(
        input={"count": adapt.Custom(entrypoint="builtins:len")},
        output=SMOLVLA.output,
    )
    with pytest.raises(adapt.AdapterResolutionError, match="trust_entrypoints"):
        resolve(LIBERO_ENV, spec)

    adapter = resolve(LIBERO_ENV, spec, trust_entrypoints=True)
    assert adapter.transform_obs(make_obs())["count"] == len(make_obs())


def test_spec_normalize_door_roundtrips_validates_and_gates_custom():
    # P1.1: the single Rust normalize/validate door the Python codec will route
    # through in P1.2. Here it is exercised in isolation.
    import json

    from rlmesh._rlmesh import adapters_spec_normalize

    # Round-trips a model spec and env tags through the Rust serde codec.
    spec = adapt.ModelSpec(
        input={"state": adapt.Concat(adapt.EEF_POS)},
        output=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )
    assert (
        adapt.ModelSpec.from_json(
            adapters_spec_normalize("model", json.dumps(spec.to_dict()), True)
        )
        == spec
    )
    tags = LIBERO_ENV.tags
    assert (
        adapt.EnvTags.from_json(
            adapters_spec_normalize("env", json.dumps(tags.to_dict()), True)
        )
        == tags
    )

    # Validation: an unknown field on a plain struct is rejected by the codec.
    with pytest.raises(ValueError):
        adapters_spec_normalize(
            "model", '{"input": {}, "output": {"components": [], "bogus": 1}}', True
        )

    # Custom gate: an entrypoint custom is rejected at publish (allow_custom=
    # False) but passes through for resolve (allow_custom=True).
    custom_wire = json.dumps(
        {
            "input": {"x": {"type": "custom", "transform": "builtins:len"}},
            "output": {"components": []},
        }
    )
    with pytest.raises(ValueError, match="entrypoint"):
        adapters_spec_normalize("model", custom_wire, False)
    assert "builtins:len" in adapters_spec_normalize("model", custom_wire, True)

    # Unknown side is rejected.
    with pytest.raises(ValueError, match="side"):
        adapters_spec_normalize("bogus", "{}", True)


def test_spec_normalize_rejects_trailing_tokens():
    # The native door must reject a valid document followed by trailing junk;
    # serde_path_to_error does not check for EOF, so de_spec calls .end()
    # explicitly. Without it, malformed wire input would normalize green.
    from rlmesh._rlmesh import adapters_spec_normalize

    valid = '{"input": {}, "output": {"components": []}}'
    assert adapters_spec_normalize("model", valid, True)  # sanity: valid passes
    with pytest.raises(ValueError):
        adapters_spec_normalize("model", valid + " junk", True)
    with pytest.raises(ValueError):
        adapters_spec_normalize(
            "env", '{"observation": {}, "action": {"components": []}} 1', True
        )


def test_action_component_missing_dim_is_rejected() -> None:
    # `dim` is required at the codec boundary (no serde default), so an absent
    # dim is a missing-field error, not a silent 0 caught later as a width
    # mismatch.
    with pytest.raises(ValueError):
        adapt.ModelSpec.from_dict(
            {"input": {}, "output": {"components": [{"role": adapt.ACTION_GRIPPER}]}}
        )


def test_empty_state_layout_rejected_by_codec() -> None:
    # Rust accepts what Python can read back: a zero-field layout is rejected by
    # the authoritative codec (so from_dict fails cleanly, not in Python's own
    # StateLayout constructor on input the codec already called valid).
    with pytest.raises(ValueError):
        adapt.EnvTags.from_dict(
            {
                "observation": {"type": "split", "fields": []},
                "action": {"components": []},
            }
        )


def test_normalize_door_rejects_structurally_unconsumable_specs() -> None:
    # The normalize/publish door must reject what the read path (from_dict) or
    # resolve reject, so it never blesses a doc another RLMesh engine cannot
    # consume. Each malformation below was accepted by the codec before the
    # cross-engine parity guards (StateLayout dup role, ModelSpec dup key,
    # StateInput empty components, ActionLayout dup role).
    import json

    from rlmesh._rlmesh import adapters_spec_normalize

    bad = [
        # (side, doc) -- each is structurally valid JSON the codec must reject.
        (
            "env",
            '{"observation":{"s":{"type":"split","fields":[{"role":"r","dim":1},'
            '{"role":"r","dim":1}]}},"action":{"components":[]}}',
        ),
        (
            "model",
            '{"input":{"s":{"type":"state","components":[]}},'
            '"output":{"components":[]}}',
        ),
        (
            "model",
            '{"input":{},"output":{"components":[{"role":"g","dim":1},'
            '{"role":"g","dim":1}]}}',
        ),
    ]
    for side, doc in bad:
        with pytest.raises(ValueError):
            adapters_spec_normalize(side, doc, True)
        # And the Python read path (which normalizes first) rejects it too.
        cls = adapt.EnvTags if side == "env" else adapt.ModelSpec
        with pytest.raises(ValueError):
            cls.from_dict(json.loads(doc))


def test_wrap_predict_round_trip():
    adapter = resolve(LIBERO_ENV, SMOLVLA)

    def predict(payload):
        assert "observation.state" in payload
        return np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0], dtype=np.float32)

    action = adapter.wrap_predict(predict)(make_obs())
    np.testing.assert_allclose(action, [1.0, 0, 0, 0, 0, 0, -1.0])


def test_unreferenced_unencodable_obs_key_is_ignored():
    """An unused, unencodable observation key must not abort a step (#8)."""
    adapter = resolve(LIBERO_ENV, OPENVLA)
    obs = make_obs()
    obs["debug_handle"] = object()  # not bridge-encodable, but OpenVLA never reads it
    payload = adapter.transform_obs(obs)
    assert set(payload) == {"image", "instruction"}


def test_observation_roles_groups_a_dict_tree_in_declaration_order():
    tags = adapt.EnvTags(
        observation={
            "cam": adapt.ImageTag(role=adapt.IMAGE_PRIMARY),
            "instr": adapt.TextTag(role=adapt.INSTRUCTION),
            "wrist": adapt.ImageTag(role=adapt.IMAGE_WRIST),
            "eef": adapt.StateTag(role=adapt.EEF_POS),
        },
        action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )
    roles = tags.observation_roles
    assert roles == adapt.ObservationRoles(
        images=(adapt.IMAGE_PRIMARY, adapt.IMAGE_WRIST),
        states=(adapt.EEF_POS,),
        texts=(adapt.INSTRUCTION,),
    )


def test_observation_roles_split_fields_are_states_and_skips_are_excluded():
    tags = adapt.EnvTags(
        observation={
            "state": adapt.Split(
                adapt.Field(adapt.EEF_POS, 3),
                adapt.Field(dim=2),
                adapt.Field(adapt.GRIPPER_POS, 1),
            )
        },
        action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )
    assert tags.observation_roles.states == (adapt.EEF_POS, adapt.GRIPPER_POS)


def test_observation_roles_handles_bare_leaf_and_nested_containers():
    bare = adapt.EnvTags(
        observation=adapt.ImageTag(role=adapt.IMAGE_PRIMARY),
        action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )
    assert bare.observation_roles == adapt.ObservationRoles(
        images=(adapt.IMAGE_PRIMARY,)
    )

    nested = adapt.EnvTags(
        observation={
            "outer": {"eef": adapt.StateTag(role=adapt.EEF_POS)},
            "pair": (
                adapt.TextTag(role=adapt.INSTRUCTION),
                adapt.StateTag(role=adapt.GRIPPER_POS),
            ),
        },
        action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )
    assert nested.observation_roles == adapt.ObservationRoles(
        states=(adapt.EEF_POS, adapt.GRIPPER_POS),
        texts=(adapt.INSTRUCTION,),
    )


def test_env_tags_json_round_trip():
    tags = LIBERO_ENV.tags
    assert adapt.EnvTags.from_json(tags.to_json()) == tags


def test_model_spec_json_round_trip():
    for spec in (SMOLVLA, OPENVLA, XVLA, GR00T):
        assert adapt.ModelSpec.from_json(spec.to_json()) == spec


def test_env_tags_metadata_round_trip():
    tags = LIBERO_ENV.tags
    metadata = {"render_fps": 20, **tags.to_metadata()}
    assert adapt.EnvTags.from_metadata(metadata) == tags
    assert adapt.EnvTags.from_metadata({"render_fps": 20}) is None


def test_model_spec_metadata_round_trip():
    for spec in (SMOLVLA, OPENVLA, XVLA, GR00T):
        metadata = {"max_batch": 8, **spec.to_metadata()}
        assert adapt.ModelSpec.from_metadata(metadata) == spec
    assert adapt.ModelSpec.from_metadata({"max_batch": 8}) is None


def test_metadata_keys_are_side_specific():
    tags = LIBERO_ENV.tags
    assert adapt.ENV_METADATA_KEY != adapt.MODEL_METADATA_KEY
    merged = {**tags.to_metadata(), **SMOLVLA.to_metadata()}
    assert adapt.EnvTags.from_metadata(merged) == tags
    assert adapt.ModelSpec.from_metadata(merged) == SMOLVLA
    assert adapt.EnvTags.from_metadata(SMOLVLA.to_metadata()) is None
    assert adapt.ModelSpec.from_metadata(tags.to_metadata()) is None


@pytest.mark.parametrize("method", ["to_dict", "to_metadata"])
def test_custom_callable_spec_is_not_serializable(method: str):
    spec = adapt.ModelSpec(
        input={"x": adapt.Custom(transform=lambda obs: 0)},
        output=SMOLVLA.output,
    )
    with pytest.raises(ValueError, match="cannot be serialized"):
        getattr(spec, method)()


@pytest.mark.parametrize("method", ["to_dict", "to_metadata", "to_json"])
def test_custom_entrypoint_is_not_serializable(method: str):
    # P0.8: publishing an importable entrypoint custom is refused (it would ship
    # code in a contract a consumer might import). Resolve it locally instead
    # (the resolve path is covered by test_custom_entrypoint_requires_trust).
    spec = adapt.ModelSpec(
        input={"x": adapt.Custom(entrypoint="builtins:len")},
        output=SMOLVLA.output,
    )
    with pytest.raises(ValueError, match="entrypoint"):
        getattr(spec, method)()


def test_host_placeholder_custom_cannot_be_reconstructed_from_wire():
    # The resolver builds an internal `transform: "host:<key>"` wire placeholder
    # for an in-process custom (resolver._model_wire); it is NOT an importable
    # entrypoint, so feeding it back through the from-dict reader must error
    # rather than mint a bogus `host:` entrypoint a later resolve would import.
    from rlmesh.adapters.specs.model_serialization import model_input_from_dict

    with pytest.raises(ValueError, match="host-placeholder"):
        model_input_from_dict({"type": "custom", "transform": "host:x"})

    # A genuine entrypoint custom still reconstructs unchanged.
    rebuilt = model_input_from_dict({"type": "custom", "transform": "builtins:len"})
    assert isinstance(rebuilt, adapt.Custom)
    assert rebuilt.entrypoint == "builtins:len"


def test_missing_state_role_is_an_error():
    spec = adapt.ModelSpec(
        input={"state": adapt.State(adapt.JOINT_VEL)},
        output=SMOLVLA.output,
    )
    with pytest.raises(adapt.AdapterResolutionError, match="proprio/joint_vel"):
        resolve(LIBERO_ENV, spec)


def test_env_rotation_width_law_is_enforced():
    """An env tagging quat_xyzw on a non-4-wide state is rejected at join,
    regardless of what the model wants -- the rotation-width law is
    unconditional."""
    env = Env(
        tags=adapt.EnvTags(
            observation={
                "agentview_image": adapt.ImageTag(role=adapt.IMAGE_PRIMARY),
                "robot0_eef_quat": adapt.StateTag(
                    role=adapt.EEF_ROT, encoding="quat_xyzw"
                ),
                "instruction": adapt.TextTag(role=adapt.INSTRUCTION),
            },
            action=LIBERO_ACTION,
        ),
        obs_space=gym.spaces.Dict(
            {
                "agentview_image": image_space(),
                "robot0_eef_quat": box(5),
                "instruction": text_space(),
            }
        ),
        action_space=ACTION7,
    )
    spec = adapt.ModelSpec(
        input={"state": adapt.State(adapt.EEF_ROT, encoding="rot6d")},
        output=SMOLVLA.output,
    )
    with pytest.raises(adapt.AdapterResolutionError, match="quat_xyzw"):
        resolve(env, spec)


def test_unknown_rotation_encoding_pairing_is_an_error():
    spec = adapt.ModelSpec(
        input={"state": adapt.State(adapt.GRIPPER_POS, encoding="axis_angle")},
        output=SMOLVLA.output,
    )
    with pytest.raises(adapt.AdapterResolutionError, match="encoding"):
        resolve(LIBERO_ENV, spec)


def test_missing_action_role_is_an_error():
    spec = adapt.ModelSpec(
        input={"image": adapt.Image(role=adapt.IMAGE_PRIMARY)},
        output=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
        ),
    )
    with pytest.raises(adapt.AdapterResolutionError, match="action/delta_eef_rot"):
        resolve(LIBERO_ENV, spec)


def test_action_dim_mismatch_is_an_error():
    spec = adapt.ModelSpec(
        input={"image": adapt.Image(role=adapt.IMAGE_PRIMARY)},
        output=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_POS, dim=2),
            adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1),
        ),
    )
    with pytest.raises(adapt.AdapterResolutionError, match="dims"):
        resolve(LIBERO_ENV, spec)


def test_action_range_derived_from_bounded_space_maps_model_output():
    # Env action is Box(0, 1) with no range tag; model emits in [-1, 1]. The
    # env range is derived from the space, so transform_action maps the model
    # output into [0, 1] instead of passing it through (and out of bounds).
    env = Env(
        tags=adapt.EnvTags(
            observation={"image": adapt.ImageTag(role=adapt.IMAGE_PRIMARY)},
            action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
        ),
        obs_space=gym.spaces.Dict({"image": image_space()}),
        action_space=box(1, low=0.0, high=1.0),
    )
    spec = adapt.ModelSpec(
        input={"image": adapt.Image(role=adapt.IMAGE_PRIMARY)},
        output=adapt.Action(
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0))
        ),
    )
    adapter = resolve(env, spec)
    out = adapter.transform_action(np.array([0.0], dtype=np.float32))
    np.testing.assert_allclose(out, [0.5], atol=1e-6)  # [-1,1] -> [0,1]


def test_action_range_disagreeing_with_bounded_space_is_an_error():
    env = Env(
        tags=adapt.EnvTags(
            observation={"image": adapt.ImageTag(role=adapt.IMAGE_PRIMARY)},
            action=adapt.Action(
                adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0))
            ),
        ),
        obs_space=gym.spaces.Dict({"image": image_space()}),
        action_space=box(1, low=0.0, high=1.0),
    )
    spec = adapt.ModelSpec(
        input={"image": adapt.Image(role=adapt.IMAGE_PRIMARY)},
        output=adapt.Action(
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0))
        ),
    )
    with pytest.raises(adapt.AdapterResolutionError, match="disagrees"):
        resolve(env, spec)


def test_float_image_in_byte_range_is_not_saturated():
    # A float32 camera whose space bounds are [0, 255] must pass through, not be
    # scaled by 255 (which would clip every pixel > 1 to white).
    env = Env(
        tags=adapt.EnvTags(
            observation={"image": adapt.ImageTag(role=adapt.IMAGE_PRIMARY)},
            action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
        ),
        obs_space=gym.spaces.Dict(
            {"image": gym.spaces.Box(0.0, 255.0, (2, 2, 3), np.float32)}
        ),
        action_space=box(1, low=-1.0, high=1.0),
    )
    spec = adapt.ModelSpec(
        input={"image": adapt.Image(role=adapt.IMAGE_PRIMARY)},
        output=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )
    adapter = resolve(env, spec)
    pixels = np.full((2, 2, 3), 200.0, dtype=np.float32)
    out = adapter.transform_obs({"image": pixels})["image"]
    # Saturating-as-normalized would give 255; byte-range passthrough keeps 200.
    assert int(np.asarray(out).max()) == 200


def test_duplicate_env_action_role_is_an_error():
    # Two env action components with the same role would each resolve against
    # the single model slice, building the env action by repetition. Reject it
    # (mirroring the model-side dedup and the env-side StateLayout role check).
    env = Env(
        tags=adapt.EnvTags(
            observation={"pos": adapt.StateTag(role=adapt.EEF_POS)},
            action=adapt.Action(
                adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
                adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),  # duplicate role
                adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
                adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
            ),
        ),
        obs_space=gym.spaces.Dict({"pos": box(3)}),
        action_space=box(10),
    )
    # The duplicate role is now rejected by the authoritative Rust codec when
    # the env tags are normalized (resolve serializes env_tags through to_dict),
    # before plan_action runs -- still surfaced as AdapterResolutionError.
    with pytest.raises(adapt.AdapterResolutionError, match="more than once"):
        resolve(env, STATE_ONLY_MODEL)


def test_wrong_model_action_length_is_an_error():
    adapter = resolve(LIBERO_ENV, SMOLVLA)
    with pytest.raises(ValueError, match="7-dim"):
        adapter.transform_action(np.zeros(5, dtype=np.float32))


def test_describe_mentions_each_model_key():
    text = resolve(LIBERO_ENV, XVLA).explain()
    assert '"state"' in text
    # X-VLA proprio and action both use row-major rot6d.
    assert "quat_xyzw->rot6d_rowmajor" in text
    assert "rot6d_rowmajor->axis_angle" in text


def _fake_env(obs_space: gym.spaces.Space[Any]) -> Any:
    return SimpleNamespace(
        observation_space=obs_space, action_space=ACTION7, metadata={"render_fps": 30}
    )


def test_tag_publishes_and_validates() -> None:
    env = _fake_env(gym.spaces.Dict({"robot0_eef_pos": box(3)}))
    tags = adapt.EnvTags(
        observation={"robot0_eef_pos": adapt.StateTag(role=adapt.EEF_POS)},
        action=LIBERO_ACTION,
    )
    returned = adapt.tag(env, tags)
    assert returned is env
    assert env.metadata["render_fps"] == 30  # existing metadata preserved
    assert adapt.EnvTags.from_metadata(env.metadata) == tags


def test_tag_rejects_mismatched_tags() -> None:
    # The space is 3-wide but quat_xyzw requires 4 -> join fails fast.
    env = _fake_env(gym.spaces.Dict({"robot0_eef_quat": box(3)}))
    bad = adapt.EnvTags(
        observation={
            "robot0_eef_quat": adapt.StateTag(role=adapt.EEF_ROT, encoding="quat_xyzw")
        },
        action=LIBERO_ACTION,
    )
    with pytest.raises(adapt.AdapterResolutionError, match="quat_xyzw"):
        adapt.tag(env, bad)
    assert adapt.ENV_METADATA_KEY not in env.metadata  # nothing published on failure


def test_tag_without_validation_skips_the_check() -> None:
    env = _fake_env(gym.spaces.Dict({"robot0_eef_quat": box(3)}))
    bad = adapt.EnvTags(
        observation={
            "robot0_eef_quat": adapt.StateTag(role=adapt.EEF_ROT, encoding="quat_xyzw")
        },
        action=LIBERO_ACTION,
    )
    adapt.tag(env, bad, validate=False)
    assert adapt.EnvTags.from_metadata(env.metadata) == bad


def test_tag_warns_on_mislabeled_image_layout() -> None:
    # A CHW image [3,64,64] left at the default hwc layout derives channels=64;
    # tag() surfaces a non-fatal hint to declare layout="chw" — not an error.
    env = _fake_env(
        gym.spaces.Dict(
            {"cam": gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)}
        )
    )
    tags = adapt.EnvTags(
        observation={"cam": adapt.ImageTag(role=adapt.IMAGE_PRIMARY)},  # default hwc
        action=LIBERO_ACTION,
    )
    with pytest.warns(UserWarning, match="looks like chw"):
        adapt.tag(env, tags)
    # The hint is advisory: the tags still publish.
    assert adapt.EnvTags.from_metadata(env.metadata) == tags


def test_tag_silent_on_correctly_labeled_image() -> None:
    # A normal [64,64,3] hwc image is unambiguous -> no advisory.
    env = _fake_env(gym.spaces.Dict({"cam": image_space()}))
    tags = adapt.EnvTags(
        observation={"cam": adapt.ImageTag(role=adapt.IMAGE_PRIMARY)},
        action=LIBERO_ACTION,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        adapt.tag(env, tags)


def test_unregistered_role_nudges_at_tag_but_escape_does_not() -> None:
    env = _fake_env(gym.spaces.Dict({"thing": box(3)}))
    adhoc = adapt.EnvTags(
        observation={"thing": adapt.StateTag(role="proprio/made_up")},
        action=LIBERO_ACTION,
    )
    with pytest.warns(UserWarning, match="role registry"):
        adapt.tag(env, adhoc)
    clean = adapt.EnvTags(
        observation={"thing": adapt.StateTag(role="x/custom")},
        action=LIBERO_ACTION,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        adapt.tag(env, clean)


def test_role_policy_levels() -> None:
    from rlmesh._rlmesh import adapters_spec_normalize

    adhoc = '{"observation": {}, "action": {"components": [{"role": "action/wiggle", "dim": 1}]}}'
    escape = '{"observation": {}, "action": {"components": [{"role": "x/custom", "dim": 1}]}}'
    registered = '{"observation": {}, "action": {"components": [{"role": "action/gripper", "dim": 1}]}}'

    adapters_spec_normalize("env", adhoc, False)

    with pytest.raises(ValueError, match="unregistered"):
        adapters_spec_normalize("env", adhoc, False, "strict")
    adapters_spec_normalize("env", escape, False, "strict")
    adapters_spec_normalize("env", registered, False, "strict")

    with pytest.raises(ValueError, match="including the `x/`"):
        adapters_spec_normalize("env", escape, False, "forbid")
    adapters_spec_normalize("env", registered, False, "forbid")


def test_negative_u32_fields_rejected_by_rust_codec() -> None:
    # Negatives are rejected by the authoritative Rust codec (u32) at
    # serialize/normalize with a field-path-annotated message; model-input
    # payload fields resolve to `inputs[0]` (the ModelInput tag boundary) plus
    # the u32 constraint. An Actuator's dim never reaches the codec: its
    # construction guard enforces dim >= 1 first (matching Field).
    gripper = adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1))

    def to_dict(*, output: adapt.Action = gripper, **inputs: adapt.ModelLeaf) -> None:
        adapt.ModelSpec(input=inputs, output=output).to_dict()

    with pytest.raises(ValueError, match=r"dim must be >= 1, got -1"):
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=-1)
    with pytest.raises(ValueError, match=r"non-negative integer"):
        to_dict(image=adapt.Image(role=adapt.IMAGE_PRIMARY, width=-1))
    with pytest.raises(ValueError, match=r"non-negative integer"):
        to_dict(image=adapt.Image(role=adapt.IMAGE_PRIMARY, lead_dims=-2))
    with pytest.raises(ValueError, match=r"non-negative integer"):
        to_dict(s=adapt.State(adapt.EEF_POS, dim=-1))
    with pytest.raises(ValueError, match=r"non-negative integer"):
        to_dict(s=adapt.State(adapt.EEF_POS, index=-1))
    with pytest.raises(ValueError, match=r"non-negative integer"):
        to_dict(s=adapt.State(adapt.EEF_POS, pad_to=-1))


def test_value_bridge_encodes_numpy_bool_scalar_as_python_bool() -> None:
    from rlmesh._value_conversion import to_value
    from rlmesh.numpy import _numpy_bridge

    assert to_value(np.bool_(True), _numpy_bridge) is True
    assert to_value(np.bool_(False), _numpy_bridge) is False


def test_image_input_size_shorthand() -> None:
    assert adapt.Image(role=adapt.IMAGE_PRIMARY, size=224) == adapt.Image(
        role=adapt.IMAGE_PRIMARY, height=224, width=224
    )
    with pytest.raises(ValueError, match="size=, or height"):
        adapt.Image(role=adapt.IMAGE_PRIMARY, size=224, height=10)


def test_state_input_single_component_shorthand() -> None:
    # State takes a role plus part-fields directly; equal instances compare equal.
    s = adapt.State(adapt.EEF_POS, encoding="axis_angle")
    assert s == adapt.State(adapt.EEF_POS, encoding="axis_angle")
    # Concat assembles a multi-part state from bare roles and parameterized parts.
    c = adapt.Concat(adapt.EEF_POS, adapt.State(adapt.GRIPPER_POS, dim=1))
    assert len(c.parts) == 2


def test_state_input_sugar_resolves_like_explicit() -> None:
    spec = adapt.ModelSpec(
        input={"state": adapt.State(adapt.EEF_POS)},
        output=SMOLVLA.output,
    )
    adapter = resolve(LIBERO_ENV, spec)
    np.testing.assert_allclose(
        adapter.transform_obs(make_obs())["state"],
        np.asarray(make_obs()["robot0_eef_pos"], dtype=np.float32),
    )


def test_image_frame_stacking_buffers_and_pads() -> None:
    env = image_env(4, 4)
    spec = adapt.ModelSpec(
        input={"img": adapt.Image(role=adapt.IMAGE_PRIMARY, stack=3)},
        output=SMOLVLA.output,
    )
    adapter = resolve(env, spec)
    f1 = np.full((4, 4, 3), 10, dtype=np.uint8)
    f2 = np.full((4, 4, 3), 20, dtype=np.uint8)
    f3 = np.full((4, 4, 3), 30, dtype=np.uint8)

    p1 = adapter.transform_obs({"rgb": f1})["img"]
    assert p1.shape == (3, 4, 4, 3)
    np.testing.assert_array_equal(p1, np.stack([f1, f1, f1]))  # padded episode start
    np.testing.assert_array_equal(
        adapter.transform_obs({"rgb": f2})["img"], np.stack([f1, f1, f2])
    )
    np.testing.assert_array_equal(
        adapter.transform_obs({"rgb": f3})["img"], np.stack([f1, f2, f3])
    )

    adapter.reset()  # episode boundary clears history
    np.testing.assert_array_equal(
        adapter.transform_obs({"rgb": f2})["img"], np.stack([f2, f2, f2])
    )


def test_per_lane_reset_only_clears_on_whole_vector_or_lane_zero() -> None:
    env = image_env(4, 4)
    spec = adapt.ModelSpec(
        input={"img": adapt.Image(role=adapt.IMAGE_PRIMARY, stack=3)},
        output=SMOLVLA.output,
    )
    adapter = resolve(env, spec)
    f1 = np.full((4, 4, 3), 10, dtype=np.uint8)
    f2 = np.full((4, 4, 3), 20, dtype=np.uint8)

    adapter.transform_obs({"rgb": f1})
    np.testing.assert_array_equal(
        adapter.transform_obs({"rgb": f2})["img"], np.stack([f1, f1, f2])
    )

    # reset() clears the host-side buffer at an episode boundary, so the next
    # frame pads from itself again.
    adapter.reset()
    np.testing.assert_array_equal(
        adapter.transform_obs({"rgb": f2})["img"], np.stack([f2, f2, f2])
    )


def test_image_input_stack_round_trips_and_omits_default() -> None:
    from rlmesh.adapters.specs.model_serialization import model_input_to_dict

    spec = adapt.ModelSpec(
        input={"img": adapt.Image(role=adapt.IMAGE_PRIMARY, stack=4)},
        output=SMOLVLA.output,
    )
    assert adapt.ModelSpec.from_json(spec.to_json()) == spec
    assert "stack" not in model_input_to_dict(
        adapt.Image(role=adapt.IMAGE_PRIMARY)
    )  # default omitted
    assert (
        model_input_to_dict(adapt.Image(role=adapt.IMAGE_PRIMARY, stack=4))["stack"]
        == 4
    )
    # stack bounds (1..64) are enforced by the Rust codec at serialize/normalize.
    with pytest.raises(ValueError, match="stack must be between 1 and 64"):
        adapt.ModelSpec(
            input={"img": adapt.Image(role=adapt.IMAGE_PRIMARY, stack=0)},
            output=SMOLVLA.output,
        ).to_dict()
    # An untrusted spec cannot demand an unbounded buffer.
    with pytest.raises(ValueError, match="stack must be between 1 and 64"):
        adapt.ModelSpec(
            input={"img": adapt.Image(role=adapt.IMAGE_PRIMARY, stack=10_000)},
            output=SMOLVLA.output,
        ).to_dict()


def test_vector_env_rejected_by_single_env_eval_loop() -> None:
    # The SESSION's per-episode loop reads scalar reward/termination, so it rejects
    # any vector env (num_envs>1) up front instead of crashing on array truthiness
    # deep in the step loop. (Model.run drives vector envs through the native
    # runtime loop instead, so only the interactive session rejects them.)
    from rlmesh.numpy import Model

    env = image_env(4, 4)
    contract: Any = SimpleNamespace(
        metadata=env.tags.to_metadata(),
        observation_space=env.obs_space,
        action_space=env.action_space,
        num_envs=2,
    )
    # A steppable env-like object; the vector env is rejected before any step.
    fake_env: Any = SimpleNamespace(
        env_contract=contract,
        reset=lambda **_kw: ({}, {}),
        step=lambda _action: ({}, 0.0, True, False, {}),
    )

    def predict(payload: dict[str, Any]) -> Any:
        return np.zeros(SMOLVLA.output.dim, dtype=np.float32)

    model = Model(
        predict,
        spec=adapt.ModelSpec(
            input={"img": adapt.Image(role=adapt.IMAGE_PRIMARY, stack=2)},
            output=SMOLVLA.output,
        ),
    )
    import rlmesh

    with pytest.raises(ValueError, match="num_envs=2"):
        rlmesh.session(model, fake_env).run(max_episodes=1)


def test_stateful_adapter_allowed_on_vector_route() -> None:
    # Frame-stacking state is now episode-keyed in the native serving engine, so
    # a served stateful adapter resolves against a vectorized route -- the old
    # single-lane rejection is lifted.
    from rlmesh._models._eval import resolve_adapter

    env = image_env(4, 4)
    stateful_spec = adapt.ModelSpec(
        input={"img": adapt.Image(role=adapt.IMAGE_PRIMARY, stack=2)},
        output=SMOLVLA.output,
    )
    stateless_spec = adapt.ModelSpec(
        input={"img": adapt.Image(role=adapt.IMAGE_PRIMARY)},
        output=SMOLVLA.output,
    )

    def contract(num_envs: int) -> Any:
        return SimpleNamespace(
            metadata=env.tags.to_metadata(),
            observation_space=env.obs_space,
            action_space=env.action_space,
            num_envs=num_envs,
        )

    # A stateful (frame-stacking) adapter now resolves against a vector route.
    vector_stateful = resolve_adapter(
        stateful_spec, contract(2), trust_entrypoints=False
    )
    assert vector_stateful is not None
    # And still against a single lane.
    single_lane = resolve_adapter(stateful_spec, contract(1), trust_entrypoints=False)
    assert single_lane is not None
    # A stateless adapter on a vector route is harmless.
    stateless = resolve_adapter(stateless_spec, contract(2), trust_entrypoints=False)
    assert stateless is not None
    # spec=None / no tags also resolves to None without over-rejecting.
    untagged: Any = SimpleNamespace(metadata={}, num_envs=2)
    assert resolve_adapter(None, untagged, trust_entrypoints=False) is None


def test_euler_xyz_encoding_converts_end_to_end() -> None:
    # An env reporting orientation as roll-pitch-yaw, a model wanting axis_angle.
    env = Env(
        tags=adapt.EnvTags(
            observation={
                "rpy": adapt.StateTag(role=adapt.EEF_ROT, encoding="euler_xyz")
            },
            action=LIBERO_ACTION,
        ),
        obs_space=gym.spaces.Dict({"rpy": box(3)}),
        action_space=ACTION7,
    )
    spec = adapt.ModelSpec(
        input={"rot": adapt.State(adapt.EEF_ROT, encoding="axis_angle")},
        output=SMOLVLA.output,
    )
    adapter = resolve(env, spec)
    # Pure yaw of 90 degrees -> axis-angle about z.
    out = adapter.transform_obs(
        {"rpy": np.array([0.0, 0.0, np.pi / 2], dtype=np.float32)}
    )["rot"]
    np.testing.assert_allclose(out, [0.0, 0.0, np.pi / 2], atol=1e-4)


def test_bare_layout_observation_stored_as_bare_leaf():
    tags = adapt.EnvTags(
        observation=adapt.Split(
            adapt.Field(adapt.EEF_POS, 3),
            adapt.Field(adapt.GRIPPER_POS, 1),
        ),
        action=LIBERO_ACTION,
    )
    # A bare leaf observation is stored as-is, not normalized to {".": tag}.
    assert isinstance(tags.observation, adapt.Split)


def test_bare_obs_tag_observation_stored_as_bare_leaf():
    tags = adapt.EnvTags(
        observation=adapt.ImageTag(role=adapt.IMAGE_PRIMARY),
        action=LIBERO_ACTION,
    )
    assert isinstance(tags.observation, adapt.ImageTag)


def test_action_and_state_layout_take_varargs():
    action = adapt.Action(
        adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1),
        clip=(-1.0, 1.0),
    )
    assert action.dim == 4
    assert action.clip == (-1.0, 1.0)
    layout = adapt.Split(
        adapt.Field(adapt.EEF_POS, 3),
        adapt.Field(dim=1),
    )
    assert len(layout.fields) == 2
    assert layout.fields[1].role is None


def test_state_layout_round_trips_through_dict():
    tags = adapt.EnvTags(
        observation=adapt.Split(
            adapt.Field(adapt.EEF_POS, 3),
            adapt.Field(adapt.EEF_ROT, 4, encoding="quat_xyzw"),
            adapt.Field(dim=1),  # skip
            adapt.Field(adapt.GRIPPER_POS, 1, range=(0.0, 1.0)),
        ),
        action=LIBERO_ACTION,
    )
    restored = adapt.EnvTags.from_dict(tags.to_dict())
    assert restored == tags
    layout = restored.observation
    assert isinstance(layout, adapt.Split)
    assert [f.role for f in layout.fields] == [
        adapt.EEF_POS,
        adapt.EEF_ROT,
        None,
        adapt.GRIPPER_POS,
    ]


def _metaworld_env(width_box):
    """A flat-Box env whose 8-wide state splits into role fields."""
    return Env(
        tags=adapt.EnvTags(
            observation=adapt.Split(
                adapt.Field(adapt.EEF_POS, 3),
                adapt.Field(adapt.GRIPPER_POS, 1),
                adapt.Field(dim=1),  # skip an ignored element
                adapt.Field(adapt.JOINT_POS, 3),
            ),
            action=LIBERO_ACTION,
        ),
        obs_space=width_box,
        action_space=ACTION7,
    )


def test_flat_layout_resolves_by_role_and_slices():
    env = _metaworld_env(box(8))
    spec = adapt.ModelSpec(
        input={
            "state": adapt.Concat(
                adapt.EEF_POS,
                adapt.GRIPPER_POS,
                adapt.JOINT_POS,
            ),
        },
        output=SMOLVLA.output,
    )
    adapter = resolve(env, spec)
    flat = np.array([0.0, 1.0, 2.0, 9.0, -7.0, 3.0, 4.0, 5.0], dtype=np.float32)
    # The env returns a bare flat array (no dict); the adapter reads it as root.
    out = adapter.transform_obs(flat)["state"]
    # eef_pos [0:3] + gripper [3] + joint_pos [5:8]; index 4 is skipped.
    np.testing.assert_allclose(out, [0.0, 1.0, 2.0, 9.0, 3.0, 4.0, 5.0], atol=1e-6)
    # describe() shows each field's env-side slice (the skipped index is the
    # gap between [3:4] and [5:8]).
    text = adapter.explain()
    assert "<root>[0:3]" in text and "<root>[3:4]" in text and "<root>[5:8]" in text


def test_flat_layout_rotation_field_converts():
    env = Env(
        tags=adapt.EnvTags(
            observation=adapt.Split(
                adapt.Field(adapt.EEF_POS, 3),
                adapt.Field(adapt.EEF_ROT, 4, encoding="quat_xyzw"),
            ),
            action=LIBERO_ACTION,
        ),
        obs_space=box(7),
        action_space=ACTION7,
    )
    spec = adapt.ModelSpec(
        input={
            "state": adapt.Concat(
                adapt.EEF_POS,
                adapt.State(adapt.EEF_ROT, encoding="axis_angle"),
            ),
        },
        output=SMOLVLA.output,
    )
    adapter = resolve(env, spec)
    quat = np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32)
    quat /= np.linalg.norm(quat)
    flat = np.concatenate([[1.0, 2.0, 3.0], quat]).astype(np.float32)
    out = adapter.transform_obs(flat)["state"]
    expected = np.concatenate([[1.0, 2.0, 3.0], ref_quat2axisangle(quat)])
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_flat_layout_width_mismatch_is_an_error():
    env = _metaworld_env(box(5))  # layout sums to 8, space is 5
    spec = adapt.ModelSpec(
        input={"state": adapt.State(adapt.EEF_POS)},
        output=SMOLVLA.output,
    )
    with pytest.raises(adapt.AdapterResolutionError, match="sums to 8"):
        resolve(env, spec)


def test_state_layout_rejects_duplicate_role_at_construction():
    with pytest.raises(ValueError, match="role more than once"):
        adapt.Split(
            adapt.Field(adapt.EEF_POS, 3),
            adapt.Field(adapt.EEF_POS, 3),
        )


def test_state_field_skip_cannot_carry_encoding():
    # A role-less skip carrying an encoding is rejected by the Rust codec at
    # serialize/normalize (Field's TryFrom guard).
    tags = adapt.EnvTags(
        observation={"state": adapt.Split(adapt.Field(dim=4, encoding="quat_xyzw"))},
        action=LIBERO_ACTION,
    )
    with pytest.raises(ValueError, match="role-less"):
        tags.to_dict()


def test_state_field_requires_positive_dim():
    # dim=0 is rejected at construction (the 0 default only satisfies dataclass
    # field ordering), matching the Rust Field codec's dim >= 1 guard.
    with pytest.raises(ValueError, match="dim must be >= 1"):
        adapt.Field(adapt.EEF_POS, 0)


# A width-preserving rot6d repacking whose inverse is itself (reverse the
# 6-vector). Used to exercise the host-side shims against a native baseline.
ROT6D_REV = adapt.CustomEncoding(
    base="rot6d",
    from_base=lambda v: np.asarray(v)[::-1],
    to_base=lambda v: np.asarray(v)[::-1],
    name="rot6d_rev",
)

# The same reversal expressed as `module:callable` entrypoint arms. This is the
# portable-*schema* form: it publishes as a `{base, ...}` object and validates
# against an env, but its execution is deferred -- a custom encoding runs only
# via an in-process callable, so the entrypoint arm is never imported here.
ROT6D_FLIP_ENTRYPOINT = adapt.CustomEncoding(
    base="rot6d",
    from_base="numpy:flip",
    to_base="numpy:flip",
    name="rot6d_flip",
)


def _rot_obs_env() -> Env:
    """An env with a quaternion EEF_ROT and a trivial gripper action."""
    return Env(
        tags=adapt.EnvTags(
            observation={"q": adapt.StateTag(role=adapt.EEF_ROT, encoding="quat_xyzw")},
            action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
        ),
        obs_space=gym.spaces.Dict({"q": box(4)}),
        action_space=box(1, low=-1.0, high=1.0),
    )


def _gripper_action() -> adapt.Action:
    return adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1))


def test_custom_obs_encoding_repacks_after_base_conversion():
    env = _rot_obs_env()
    quat = np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32)
    quat /= np.linalg.norm(quat)
    base_spec = adapt.ModelSpec(
        input={"rot": adapt.State(adapt.EEF_ROT, encoding="rot6d")},
        output=_gripper_action(),
    )
    custom_spec = adapt.ModelSpec(
        input={"rot": adapt.State(adapt.EEF_ROT, encoding=ROT6D_REV)},
        output=_gripper_action(),
    )
    base_out = resolve(env, base_spec).transform_obs({"q": quat})["rot"]
    custom_out = resolve(env, custom_spec).transform_obs({"q": quat})["rot"]
    # The custom output is the native rot6d, repacked by from_base (reversed).
    np.testing.assert_allclose(custom_out, np.asarray(base_out)[::-1], atol=1e-6)


def test_custom_action_encoding_repacks_before_base_conversion():
    env = Env(
        tags=adapt.EnvTags(
            observation={"q": adapt.StateTag(role=adapt.EEF_ROT, encoding="quat_xyzw")},
            action=adapt.Action(
                adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle")
            ),
        ),
        obs_space=gym.spaces.Dict({"q": box(4)}),
        action_space=box(3),
    )
    model_input = {"rot": adapt.State(adapt.EEF_ROT, encoding="rot6d")}
    base_spec = adapt.ModelSpec(
        input=model_input,
        output=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=6, encoding="rot6d")
        ),
    )
    custom_spec = adapt.ModelSpec(
        input=model_input,
        output=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=6, encoding=ROT6D_REV)
        ),
    )
    model_action = np.arange(6, dtype=np.float32)
    # Feeding the custom action to the custom adapter equals feeding the
    # reversed (base) action to the base adapter: the action shim applied to_base.
    base_out = resolve(env, base_spec).transform_action(model_action[::-1].copy())
    custom_out = resolve(env, custom_spec).transform_action(model_action.copy())
    np.testing.assert_allclose(custom_out, base_out, atol=1e-6)


def test_custom_encoding_describe_shows_host_layer():
    env = _rot_obs_env()
    spec = adapt.ModelSpec(
        input={"rot": adapt.State(adapt.EEF_ROT, encoding=ROT6D_REV)},
        output=_gripper_action(),
    )
    text = resolve(env, spec).explain()
    assert "host-side encodings:" in text
    assert "rot6d -> rot6d_rev" in text


# RoboTwin-style bimanual env: each arm's end-effector pose arrives as one flat
# 7-wide leaf (xyz + a wxyz quaternion), plus a gripper scalar.
BIMANUAL_EEF_ENV = Env(
    tags=adapt.EnvTags(
        observation={
            "left_endpose": adapt.Split(
                adapt.Field(adapt.EEF_POS, 3),
                adapt.Field(adapt.EEF_ROT, 4, encoding="quat_wxyz"),
            ),
            "right_endpose": adapt.Split(
                adapt.Field(adapt.EEF_POS_2, 3),
                adapt.Field(adapt.EEF_ROT_2, 4, encoding="quat_wxyz"),
            ),
            "left_gripper": adapt.StateTag(role=adapt.GRIPPER_POS),
            "right_gripper": adapt.StateTag(role=adapt.GRIPPER_POS_2),
        },
        action=adapt.Action(
            adapt.Actuator(adapt.ACTION_EEF_POS, dim=3),
            adapt.Actuator(adapt.ACTION_EEF_ROT, dim=4, encoding="quat_wxyz"),
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1),
            adapt.Actuator(adapt.ACTION_EEF_POS_2, dim=3),
            adapt.Actuator(adapt.ACTION_EEF_ROT_2, dim=4, encoding="quat_wxyz"),
            adapt.Actuator(adapt.ACTION_GRIPPER_2, dim=1),
        ),
    ),
    obs_space=gym.spaces.Dict(
        {
            "left_endpose": box(7),
            "right_endpose": box(7),
            "left_gripper": box(1),
            "right_gripper": box(1),
        }
    ),
    action_space=box(16),
)

# A checkpoint whose rot6d convention lists the two columns the other way round:
# a width-preserving repack of the base encoding, self-inverse. (X-VLA's real
# RoboTwin2 quirk is a quaternion-order repack; what is under test here is that
# a part-level repack finds its own slice of a 20-wide state.)
ROT6D_COLS_SWAPPED = adapt.CustomEncoding(
    base="rot6d_rowmajor",
    from_base=lambda v: np.asarray(v)[[1, 0, 3, 2, 5, 4]],
    to_base=lambda v: np.asarray(v)[[1, 0, 3, 2, 5, 4]],
    name="rot6d_cols_swapped",
)


def _bimanual_proprio(encoding: Any) -> adapt.ModelSpec:
    """The xvla/robotwin2 proprio layout: 6 parts, 20 wide, rot at 3 and 13."""
    return adapt.ModelSpec(
        input={
            "proprio": adapt.Concat(
                adapt.State(adapt.EEF_POS, dim=3),
                adapt.State(adapt.EEF_ROT, dim=6, encoding=encoding),
                adapt.State(adapt.GRIPPER_POS, dim=1),
                adapt.State(adapt.EEF_POS_2, dim=3),
                adapt.State(adapt.EEF_ROT_2, dim=6, encoding=encoding),
                adapt.State(adapt.GRIPPER_POS_2, dim=1),
                container="array",
            )
        },
        output=adapt.Action(
            adapt.Actuator(adapt.ACTION_EEF_POS, dim=3),
            adapt.Actuator(adapt.ACTION_EEF_ROT, dim=6, encoding=encoding),
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1),
            adapt.Actuator(adapt.ACTION_EEF_POS_2, dim=3),
            adapt.Actuator(adapt.ACTION_EEF_ROT_2, dim=6, encoding=encoding),
            adapt.Actuator(adapt.ACTION_GRIPPER_2, dim=1),
        ),
    )


def _quat_wxyz_to_rot6d_rowmajor(quat: Any) -> Any:
    """The first two columns of R(quat), read row-major -- hand-rolled here so
    the expectation is independent of the core's own conversion."""
    w, x, y, z = (float(v) for v in np.asarray(quat) / np.linalg.norm(quat))
    matrix = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return matrix[:, :2].reshape(6)


def _bimanual_obs() -> dict[str, Any]:
    return {
        "left_endpose": np.array(
            [0.21, -0.13, 0.94, 0.8, 0.2, -0.1, 0.55], dtype=np.float32
        ),
        "right_endpose": np.array(
            [-0.31, 0.07, 0.88, 0.1, -0.7, 0.3, 0.64], dtype=np.float32
        ),
        "left_gripper": np.array([0.35], dtype=np.float32),
        "right_gripper": np.array([0.9], dtype=np.float32),
    }


def test_custom_obs_encoding_addresses_its_slice_of_a_multipart_concat():
    """Two custom-encoded parts of one 20-wide state each repack exactly their
    own slice; every other part is byte-identical to the base-encoding plan."""
    obs = _bimanual_obs()
    base = resolve(BIMANUAL_EEF_ENV, _bimanual_proprio("rot6d_rowmajor"))
    custom = resolve(BIMANUAL_EEF_ENV, _bimanual_proprio(ROT6D_COLS_SWAPPED))
    base_state = np.asarray(base.transform_obs(obs)["proprio"])
    custom_state = np.asarray(custom.transform_obs(obs)["proprio"])
    assert base_state.shape == (20,)
    swap = [1, 0, 3, 2, 5, 4]
    for offset in (3, 13):
        np.testing.assert_allclose(
            custom_state[offset : offset + 6],
            base_state[offset : offset + 6][swap],
            atol=1e-6,
        )
    # Positions and grippers are untouched: the shim wrote only its own slice.
    kept = [0, 1, 2, 9, 10, 11, 12, 19]
    np.testing.assert_allclose(custom_state[kept], base_state[kept], atol=1e-6)
    assert "'proprio'[3:9]" in custom.explain()
    assert "'proprio'[13:19]" in custom.explain()


def test_xvla_robotwin2_style_state_and_action_round_trip():
    """The pairing the offset addressing exists for: a 20-wide bimanual proprio
    whose rotation parts carry a host-side repack, checked against a
    hand-computed vector, with the action converted back to the env's 16."""
    obs = _bimanual_obs()
    adapter = resolve(BIMANUAL_EEF_ENV, _bimanual_proprio(ROT6D_COLS_SWAPPED))
    state = np.asarray(adapter.transform_obs(obs)["proprio"])
    swap = [1, 0, 3, 2, 5, 4]
    expected = np.concatenate(
        [
            obs["left_endpose"][:3],
            _quat_wxyz_to_rot6d_rowmajor(obs["left_endpose"][3:])[swap],
            obs["left_gripper"],
            obs["right_endpose"][:3],
            _quat_wxyz_to_rot6d_rowmajor(obs["right_endpose"][3:])[swap],
            obs["right_gripper"],
        ]
    )
    assert state.shape == (20,)  # in_dim 20
    np.testing.assert_allclose(state, expected, atol=0.002)
    # Echoing the proprio back as the action returns the observed pose: the
    # action shim undoes the repack before the core converts rot6d -> quaternion.
    action = adapter.transform_action(state)
    assert action.shape == (16,)
    for env_slice, obs_key in (
        (slice(0, 8), "left_endpose"),
        (slice(8, 16), "right_endpose"),
    ):
        pose = np.asarray(obs[obs_key])
        np.testing.assert_allclose(action[env_slice][:3], pose[:3], atol=0.002)
        quat = pose[3:] / np.linalg.norm(pose[3:])
        got = action[env_slice][3:7]
        # A rotation has two quaternion representations; either is correct.
        assert min(np.abs(got - quat).max(), np.abs(got + quat).max()) < 0.002


def test_custom_obs_encoding_pads_and_addresses_within_the_padded_state():
    """pad_to is compatible with a repack: the shim addresses its slice of the
    padded vector (the single-arm catalog variants' `pad_to=20` shape)."""
    env = _rot_obs_env()
    spec = adapt.ModelSpec(
        input={"rot": adapt.State(adapt.EEF_ROT, encoding=ROT6D_REV, pad_to=8)},
        output=_gripper_action(),
    )
    quat = np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32)
    quat /= np.linalg.norm(quat)
    out = np.asarray(resolve(env, spec).transform_obs({"q": quat})["rot"])
    base_spec = adapt.ModelSpec(
        input={"rot": adapt.State(adapt.EEF_ROT, encoding="rot6d", pad_to=8)},
        output=_gripper_action(),
    )
    base = np.asarray(resolve(env, base_spec).transform_obs({"q": quat})["rot"])
    assert out.shape == (8,)
    np.testing.assert_allclose(out[:6], base[:6][::-1], atol=1e-6)
    np.testing.assert_allclose(out[6:], 0.0, atol=1e-6)


def test_custom_obs_encoding_rejects_a_width_changing_dim():
    env = _rot_obs_env()
    spec = adapt.ModelSpec(
        input={"rot": adapt.State(adapt.EEF_ROT, dim=3, encoding=ROT6D_REV)},
        output=_gripper_action(),
    )
    with pytest.raises(adapt.AdapterResolutionError, match="keeps its base width"):
        resolve(env, spec)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"reshape": (1, 6)}, "reshape"),
        ({"container": "list"}, "container='array'"),
    ],
)
def test_custom_obs_encoding_rejects_assembly_options(kwargs, match):
    env = _rot_obs_env()
    spec = adapt.ModelSpec(
        input={"rot": adapt.State(adapt.EEF_ROT, encoding=ROT6D_REV, **kwargs)},
        output=_gripper_action(),
    )
    with pytest.raises(adapt.AdapterResolutionError, match=match):
        resolve(env, spec)


def test_unknown_encoding_string_rejected_at_resolve():
    env = _rot_obs_env()
    spec = adapt.ModelSpec(
        input={"rot": adapt.State(adapt.EEF_ROT, encoding="rot6d_typo")},  # type: ignore[arg-type]
        output=_gripper_action(),
    )
    with pytest.raises(adapt.AdapterResolutionError, match="unknown rotation encoding"):
        resolve(env, spec)


def test_obs_custom_encoding_without_from_base_is_rejected():
    action_only = adapt.CustomEncoding(base="rot6d", to_base=lambda v: v)
    env = _rot_obs_env()
    spec = adapt.ModelSpec(
        input={"rot": adapt.State(adapt.EEF_ROT, encoding=action_only)},
        output=_gripper_action(),
    )
    with pytest.raises(adapt.AdapterResolutionError, match="needs from_base"):
        resolve(env, spec)


def test_non_inverse_custom_encoding_caught_by_self_check():
    bad = adapt.CustomEncoding(
        base="rot6d",
        from_base=lambda v: np.asarray(v)[::-1],
        to_base=lambda v: np.asarray(v),  # identity is not the inverse of reverse
        name="broken",
    )
    env = _rot_obs_env()
    spec = adapt.ModelSpec(
        input={"rot": adapt.State(adapt.EEF_ROT, encoding=bad)},
        output=_gripper_action(),
    )
    with pytest.raises(adapt.AdapterResolutionError, match="not inverses"):
        resolve(env, spec)
    # The self-check is opt-out for intentionally non-invertible encodings.
    resolve(env, spec, check_inverse=False)


def test_custom_action_encoding_requires_flat_action():
    env = Env(
        tags=adapt.EnvTags(
            observation={"q": adapt.StateTag(role=adapt.EEF_ROT, encoding="quat_xyzw")},
            action=adapt.Action(
                adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle")
            ),
        ),
        obs_space=gym.spaces.Dict({"q": box(4)}),
        action_space=box(3),
    )
    spec = adapt.ModelSpec(
        input={"rot": adapt.State(adapt.EEF_ROT, encoding="rot6d")},
        output=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=6, encoding=ROT6D_REV)
        ),
    )
    adapter = resolve(env, spec)
    with pytest.raises(TypeError, match="flat array action"):
        adapter.transform_action({"not": "flat"})


def test_action_custom_encoding_must_preserve_base_width():
    with pytest.raises(ValueError, match="must be"):
        adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding=ROT6D_REV)


def test_custom_obs_encoding_preserves_declared_dtype():
    # from_base that upcasts to float64 (here by returning a Python list) must
    # not change the model input's declared dtype; the native core cast to it.
    upcast = adapt.CustomEncoding(
        base="rot6d",
        from_base=lambda v: [float(x) for x in np.asarray(v)],
        to_base=lambda v: np.asarray(v),
        name="upcast",
    )
    env = _rot_obs_env()
    spec = adapt.ModelSpec(
        input={"rot": adapt.State(adapt.EEF_ROT, encoding=upcast)},
        output=_gripper_action(),
    )
    quat = np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32)
    quat /= np.linalg.norm(quat)
    out = resolve(env, spec).transform_obs({"q": quat})["rot"]
    assert np.asarray(out).dtype == np.float32  # not float64 from the list


def test_custom_obs_encoding_rejects_non_1d_transform_output():
    # from_base returning a (2, 3) array has 6 elements but the wrong shape; it
    # must be rejected, not silently flattened (which could reorder the field).
    twod = adapt.CustomEncoding(
        base="rot6d",
        from_base=lambda v: np.asarray(v).reshape(2, 3),
        to_base=lambda v: np.asarray(v).reshape(-1),
        name="twod",
    )
    env = _rot_obs_env()
    spec = adapt.ModelSpec(
        input={"rot": adapt.State(adapt.EEF_ROT, encoding=twod)},
        output=_gripper_action(),
    )
    adapter = resolve(env, spec)
    quat = np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32)
    quat /= np.linalg.norm(quat)
    with pytest.raises(ValueError, match="flat width-6"):
        adapter.transform_obs({"q": quat})


def test_inprocess_custom_encoding_serializes_as_local_marker():
    # An in-process callable arm has no wire form, so the spec still serializes
    # (showable + validatable) but records a non-portable <local> marker. It runs
    # locally where the callable lives; the reconstructed stub does not.
    spec = adapt.ModelSpec(
        input={"rot": adapt.State(adapt.EEF_ROT, encoding=ROT6D_REV)},
        output=_gripper_action(),
    )
    data = spec.to_dict()
    enc = data["input"]["rot"]["components"][0]["encoding"]
    assert enc == {
        "base": "rot6d",
        "name": "rot6d_rev",
        "from_base": "<local>",
        "to_base": "<local>",
    }
    assert spec.to_metadata()  # publishable as a schema, no raise
    # Reading it back gives a describe-only stub that cannot be run.
    env = _rot_obs_env()
    stub = adapt.ModelSpec.from_dict(data)
    with pytest.raises(adapt.AdapterResolutionError, match="did not travel"):
        resolve(env, stub)


def test_entrypoint_custom_encoding_is_publishable_and_round_trips():
    # The entrypoint form serializes to a `{base, ...}` object, round-trips
    # through the Rust codec, and is publishable in contract metadata -- the
    # spec travels and validates even though the platform never runs the arm.
    spec = adapt.ModelSpec(
        input={"rot": adapt.State(adapt.EEF_ROT, encoding=ROT6D_FLIP_ENTRYPOINT)},
        output=adapt.Action(
            adapt.Actuator(
                adapt.ACTION_DELTA_ROT, dim=6, encoding=ROT6D_FLIP_ENTRYPOINT
            )
        ),
    )
    data = spec.to_dict()
    obs_leaf = data["input"]["rot"]["components"][0]["encoding"]
    assert obs_leaf == {
        "base": "rot6d",
        "name": "rot6d_flip",
        "from_base": "numpy:flip",
        "to_base": "numpy:flip",
    }
    action_enc = data["output"]["components"][0]["encoding"]
    assert action_enc["base"] == "rot6d" and action_enc["to_base"] == "numpy:flip"
    assert adapt.ModelSpec.from_dict(data) == spec
    assert spec.to_metadata()  # publishable, no raise


def test_entrypoint_custom_encoding_execution_is_unsupported():
    # An entrypoint custom encoding is a validation/describe schema: it serializes
    # and the CP resolves it structurally, but executing it is deferred. Execution
    # is pinned to an in-process callable (the only runnable form), so even
    # trust_entrypoints does not run the arm.
    env = _rot_obs_env()
    spec = adapt.ModelSpec(
        input={"rot": adapt.State(adapt.EEF_ROT, encoding=ROT6D_FLIP_ENTRYPOINT)},
        output=_gripper_action(),
    )
    with pytest.raises(adapt.AdapterResolutionError, match="not supported yet"):
        resolve(env, spec)
    with pytest.raises(adapt.AdapterResolutionError, match="not supported yet"):
        resolve(env, spec, trust_entrypoints=True)  # no bypass


def test_platform_resolve_validates_custom_encoding_object_without_running_it():
    # The Rust resolver (the platform's adapters-resolve door) validates a
    # published custom-encoding object against an env by shadowing it to its
    # base, and never imports the arm -- an unimportable arm still resolves.
    import json

    from rlmesh._rlmesh import adapters_resolve

    env = _rot_obs_env()
    model_spec_json = json.dumps(
        {
            "input": {
                "rot": {
                    "type": "state",
                    "components": [
                        {
                            "role": adapt.EEF_ROT,
                            "encoding": {
                                "base": "rot6d",
                                "from_base": "does.not:exist",
                            },
                        }
                    ],
                }
            },
            "output": {"components": [{"role": adapt.ACTION_GRIPPER, "dim": 1}]},
        }
    )
    plan = adapters_resolve(
        json.dumps(env.tags.to_dict()),
        env.obs_space,
        env.action_space,
        model_spec_json,
    )
    assert plan is not None  # validated structurally as rot6d; arm never touched


def test_platform_resolve_rejects_optional_custom_encoding():
    # A custom encoding cannot be optional: its host-side repack has no zero form,
    # so an absent part could never be filled. The platform door rejects it up
    # front rather than admit a spec the model can never resolve.
    import json

    from rlmesh._rlmesh import adapters_resolve

    env = _rot_obs_env()
    model_spec_json = json.dumps(
        {
            "input": {
                "rot": {
                    "type": "state",
                    "components": [
                        {
                            "role": adapt.EEF_ROT,
                            "encoding": {"base": "rot6d", "from_base": "m:f"},
                            "optional": True,
                        }
                    ],
                }
            },
            "output": {"components": [{"role": adapt.ACTION_GRIPPER, "dim": 1}]},
        }
    )
    with pytest.raises(ValueError, match="cannot be optional"):
        adapters_resolve(
            json.dumps(env.tags.to_dict()),
            env.obs_space,
            env.action_space,
            model_spec_json,
        )


def test_platform_resolve_reports_the_widths_of_a_multipart_concat():
    # A custom encoding may sit anywhere in a multi-part concat: the platform
    # resolves it and reports the resolved part widths, which is what a host
    # binding addresses the repack's slice by.
    import json

    from rlmesh._rlmesh import adapters_resolve

    env = _rot_obs_env()
    model_spec_json = json.dumps(
        {
            "input": {
                "proprio": {
                    "type": "state",
                    "components": [
                        {"role": adapt.EEF_ROT, "dim": 6, "encoding": "rot6d"},
                        {
                            "role": adapt.EEF_ROT,
                            "encoding": {"base": "rot6d", "from_base": "m:f"},
                        },
                    ],
                }
            },
            "output": {"components": [{"role": adapt.ACTION_GRIPPER, "dim": 1}]},
        }
    )
    plan = adapters_resolve(
        json.dumps(env.tags.to_dict()),
        env.obs_space,
        env.action_space,
        model_spec_json,
    )
    assert plan.state_layouts() == [(["proprio"], [6, 6], 12)]


def test_platform_resolve_rejects_a_width_changing_custom_dim():
    # dim restates the base width or is omitted; anything else would resize a
    # repack that is defined to preserve it.
    import json

    from rlmesh._rlmesh import adapters_resolve

    env = _rot_obs_env()
    model_spec_json = json.dumps(
        {
            "input": {
                "proprio": {
                    "type": "state",
                    "components": [
                        {
                            "role": adapt.EEF_ROT,
                            "dim": 3,
                            "encoding": {"base": "rot6d", "from_base": "m:f"},
                        }
                    ],
                }
            },
            "output": {"components": [{"role": adapt.ACTION_GRIPPER, "dim": 1}]},
        }
    )
    with pytest.raises(ValueError, match="keeps its base width"):
        adapters_resolve(
            json.dumps(env.tags.to_dict()),
            env.obs_space,
            env.action_space,
            model_spec_json,
        )


def test_encoding_free_spec_still_serializes_unchanged():
    # A spec with no CustomEncoding must round-trip byte-identically.
    assert adapt.ModelSpec.from_dict(SMOLVLA.to_dict()) == SMOLVLA


def test_scalar_reshape_survives_serialization_and_resolve():
    # reshape=() targets a 0-D scalar; it must not be dropped as falsy. Use a
    # 1-D gripper state (Variable dim law) -- eef_pos would correctly fail the
    # 3-D law over this width-1 space.
    env = single_state_env("g", gym.spaces.Dict({"g": box(1)}), role=adapt.GRIPPER_POS)
    spec = adapt.ModelSpec(
        input={"state": adapt.Concat(adapt.GRIPPER_POS, reshape=())},
        output=SMOLVLA.output,
    )
    restored = adapt.ModelSpec.from_dict(spec.to_dict())
    assert isinstance(restored.input, dict)
    restored_input = restored.input["state"]
    assert isinstance(restored_input, adapt.Concat)
    assert restored_input.reshape == ()
    out = resolve(env, spec).transform_obs({"g": np.array([0.5], dtype=np.float32)})
    assert np.asarray(out["state"]).ndim == 0


def test_resolve_from_contract_passes_check_inverse():
    # A non-invertible CustomEncoding must be skippable through both the contract
    # path and resolve().
    env = _rot_obs_env()
    contract: Any = SimpleNamespace(
        metadata=env.tags.to_metadata(),
        observation_space=env.obs_space,
        action_space=env.action_space,
    )
    bad = adapt.CustomEncoding(
        base="rot6d",
        from_base=lambda v: np.asarray(v)[::-1],
        to_base=lambda v: np.asarray(v),  # not the inverse of reverse
        name="broken",
    )
    spec = adapt.ModelSpec(
        input={"rot": adapt.State(adapt.EEF_ROT, encoding=bad)},
        output=_gripper_action(),
    )
    with pytest.raises(adapt.AdapterResolutionError, match="not inverses"):
        adapt.resolve_from_contract(contract, spec)
    adapt.resolve_from_contract(contract, spec, check_inverse=False)  # no raise


def _contract_with_unknown_kind() -> Any:
    # A peer's raw contract declaring an `audio` observation an old core cannot
    # build. The frozen Python dataclasses cannot represent this kind, so the
    # contract is built as a raw metadata dict — exactly the wire shape the
    # consume path must tolerate.
    metadata = {
        adapt.ENV_METADATA_KEY: {
            "observation": {
                "cam": {"type": "image", "role": adapt.IMAGE_PRIMARY},
                "mic": {"type": "audio", "role": "audio/mic", "sample_rate": 16000},
            },
            "action": {"components": [{"role": adapt.ACTION_DELTA_POS, "dim": 3}]},
        }
    }
    return SimpleNamespace(
        metadata=metadata,
        observation_space=gym.spaces.Dict({"cam": image_space(), "mic": box(16)}),
        action_space=box(3, low=-1.0, high=1.0),
    )


def test_unreferenced_unknown_obs_kind_resolves_through_contract():
    # The tolerant consume path: a model that references only the camera resolves
    # against a contract carrying an unrecognized `audio` kind, which is ignored
    # with an advisory. The frozen Python EnvTags reader is bypassed entirely.
    contract = _contract_with_unknown_kind()
    spec = adapt.ModelSpec(
        input={"pixels": adapt.Image(adapt.IMAGE_PRIMARY)},
        output=adapt.Action(adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3)),
    )
    adapter = adapt.resolve_from_contract(contract, spec)
    assert any(
        "audio" in note.message and "mic" in note.message
        for note in adapter.advisories()
    ), adapter.advisories()


def test_referenced_unknown_obs_kind_is_unsupported_through_contract():
    # Referencing the role the env offers only as an unrecognized kind fails with
    # a typed "upgrade the runtime" error, not a misdirecting missing-role one.
    contract = _contract_with_unknown_kind()
    spec = adapt.ModelSpec(
        input={"sound": adapt.State("audio/mic")},
        output=adapt.Action(adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3)),
    )
    with pytest.raises(adapt.AdapterResolutionError, match="upgrade the runtime"):
        adapt.resolve_from_contract(contract, spec)


def test_bare_unknown_field_on_contract_taints_resolve():
    # §8 central asymmetry through the consume path: a bare additive field on a
    # recognized kind fails closed (must-understand), while the same field marked
    # `x-` is tolerated. A peer's raw contract is the only way to inject it (the
    # frozen authoring dataclasses reject the kwarg outright).
    cam: dict[str, Any] = {
        "type": "image",
        "role": adapt.IMAGE_PRIMARY,
        "normalize": False,
    }
    metadata = {
        adapt.ENV_METADATA_KEY: {
            "observation": {"cam": cam},
            "action": {"components": [{"role": adapt.ACTION_DELTA_POS, "dim": 3}]},
        }
    }
    contract: Any = SimpleNamespace(
        metadata=metadata,
        observation_space=gym.spaces.Dict({"cam": image_space()}),
        action_space=box(3, low=-1.0, high=1.0),
    )
    spec = adapt.ModelSpec(
        input={"pixels": adapt.Image(adapt.IMAGE_PRIMARY)},
        output=adapt.Action(adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3)),
    )
    with pytest.raises(adapt.AdapterResolutionError, match="normalize"):
        adapt.resolve_from_contract(contract, spec)

    # Mark it `x-` and the same contract resolves.
    del cam["normalize"]
    cam["x-normalize"] = False
    adapt.resolve_from_contract(contract, spec)


def test_unknown_kind_is_rejected_at_the_publish_door():
    """Python mirror of crates/rlmesh-adapters/tests/tolerance_roundtrip.rs.

    The tolerant READ half (parse-total, hard-error only when a model input
    references the unknown leaf) is pinned above through resolve_from_contract.
    The Rust relay-fidelity round-trip has no Python hop to pin: the frozen
    dataclasses cannot represent an unknown leaf, so resolve_from_contract
    forwards a peer's raw tags verbatim and from_dict stays the strict PUBLISH
    door. This pins that complement: an unknown kind dies at from_dict on both
    sides, so a spec this core cannot build is never published from Python.
    """
    with pytest.raises(ValueError, match="unrecognized kind"):
        adapt.EnvTags.from_dict(
            {
                "observation": {
                    "mic": {"type": "audio", "role": "audio/mic", "sample_rate": 16000}
                },
                "action": {"components": [{"role": adapt.ACTION_DELTA_POS, "dim": 3}]},
            }
        )
    with pytest.raises(ValueError, match="unrecognized kind"):
        adapt.ModelSpec.from_dict(
            {
                "input": {"vibe": {"type": "haptics", "role": "touch", "channels": 12}},
                "output": {"components": [{"role": adapt.ACTION_GRIPPER, "dim": 1}]},
            }
        )


def test_non_serializable_contract_metadata_is_a_clean_error():
    # A peer's contract whose adapter-tag metadata holds a non-JSON value (here a
    # set) must surface a clean AdapterResolutionError, not a raw TypeError that
    # escapes the json.dumps guard (which only catches ValueError = NaN/inf).
    metadata = {
        adapt.ENV_METADATA_KEY: {
            "observation": {
                "cam": {
                    "type": "image",
                    "role": adapt.IMAGE_PRIMARY,
                    "x-vendor": {1, 2, 3},  # a set is not JSON-serializable
                }
            },
            "action": {"components": [{"role": adapt.ACTION_DELTA_POS, "dim": 3}]},
        }
    }
    contract: Any = SimpleNamespace(
        metadata=metadata,
        observation_space=gym.spaces.Dict({"cam": image_space()}),
        action_space=box(3, low=-1.0, high=1.0),
    )
    spec = adapt.ModelSpec(
        input={"pixels": adapt.Image(adapt.IMAGE_PRIMARY)},
        output=adapt.Action(adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3)),
    )
    with pytest.raises(adapt.AdapterResolutionError, match="not serializable JSON"):
        adapt.resolve_from_contract(contract, spec)


def test_fill_only_text_is_not_a_referenced_obs_key():
    # A TextInput satisfied only by its fill has env_key ""; that empty key
    # must not be reported as an observation key to decode.
    env = single_state_env("pos", gym.spaces.Dict({"pos": box(3)}))
    spec = adapt.ModelSpec(
        input={
            "state": adapt.State(adapt.EEF_POS),
            "instruction": adapt.Text(role=adapt.INSTRUCTION, fill="do the task"),
        },
        output=SMOLVLA.output,
    )
    adapter = resolve(env, spec)
    referenced = adapter._plan.referenced_obs_keys()
    assert "" not in referenced
    assert "pos" in referenced


def test_nonfinite_range_rejected_at_from_dict():
    # from_dict routes through the Rust codec, whose json.dumps(allow_nan=False)
    # rejects the non-finite value at the boundary (the dedicated Python check is
    # now redundant). The contract is "rejected", not the exact message.
    with pytest.raises(ValueError):
        adapt.ModelSpec.from_dict(
            {
                "input": {},
                "output": {
                    "components": [
                        {
                            "role": adapt.ACTION_GRIPPER,
                            "dim": 1,
                            "range": [float("inf"), 1.0],
                        }
                    ]
                },
            }
        )


def test_nonfinite_scale_rejected_at_from_dict():
    with pytest.raises(ValueError):
        adapt.ModelSpec.from_dict(
            {
                "input": {},
                "output": {
                    "components": [
                        {"role": adapt.ACTION_GRIPPER, "dim": 1, "scale": float("inf")}
                    ]
                },
            }
        )


def test_nonfinite_rejected_on_emit_to_json():
    # A directly-constructed dataclass bypasses the from_dict guards; allow_nan
    # =False on to_json is the backstop that refuses to emit the Infinity token.
    spec = adapt.ModelSpec(
        input={},
        output=adapt.Action(
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(float("inf"), 1.0)),
        ),
    )
    with pytest.raises(ValueError):
        spec.to_json()


def test_from_metadata_reads_v1_and_returns_none_when_absent():
    spec = adapt.ModelSpec(
        input={},
        output=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )
    assert adapt.ModelSpec.from_metadata(spec.to_metadata()) == spec
    assert adapt.ModelSpec.from_metadata({}) is None

    tags = adapt.EnvTags(
        observation={}, action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1))
    )
    assert adapt.EnvTags.from_metadata(tags.to_metadata()) == tags
    assert adapt.EnvTags.from_metadata({}) is None


def _rot_model(encoding) -> adapt.ModelSpec:
    return adapt.ModelSpec(
        input={
            "state": adapt.State(adapt.EEF_ROT, encoding=encoding, container="list"),
        },
        output=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )


def test_accept_set_authoring_round_trips():
    multi = _rot_model(("rot6d", "quat_xyzw"))
    doc = multi.to_dict()
    # A sequence serializes as a JSON list...
    assert doc["input"]["state"]["components"][0]["encoding"] == ["rot6d", "quat_xyzw"]
    back = adapt.ModelSpec.from_dict(doc)
    # ...and normalizes to a tuple so the frozen spec stays hashable and equal.
    assert isinstance(back.input, dict)
    back_input = back.input["state"]
    assert isinstance(back_input, adapt.State)
    assert back_input.encoding == ("rot6d", "quat_xyzw")
    assert back == multi

    # Byte-parity: a single encoding stays a bare string, not a one-element list.
    single = _rot_model("quat_xyzw")
    assert (
        single.to_dict()["input"]["state"]["components"][0]["encoding"] == "quat_xyzw"
    )


def test_accept_set_prefers_native_then_converts():
    env = Env(
        tags=adapt.EnvTags(
            observation={
                "eef_quat": adapt.StateTag(role=adapt.EEF_ROT, encoding="quat_xyzw")
            },
            action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
        ),
        obs_space=gym.spaces.Dict({"eef_quat": box(4)}),
        action_space=box(1),
    )
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    obs = {"eef_quat": quat}

    # The model accepts the env's native quat among its preferences -> no
    # conversion: the raw 4-element quaternion passes through untouched.
    native_ok = resolve(env, _rot_model(("rot6d", "quat_xyzw"))).transform_obs(obs)
    assert len(native_ok["state"]) == 4
    assert native_ok["state"] == pytest.approx(quat.tolist())

    # The model wants only rot6d -> the env's quat is converted (6 dims).
    converted = resolve(env, _rot_model("rot6d")).transform_obs(obs)
    assert len(converted["state"]) == 6


def test_image_fit_list_authoring_round_trips():
    spec = adapt.ModelSpec(
        input={
            "image": adapt.Image(
                role=adapt.IMAGE_PRIMARY, height=64, width=64, fit=("crop", "pad")
            )
        },
        output=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )
    doc = spec.to_dict()
    assert doc["input"]["image"]["fit"] == ["crop", "pad"]  # a sequence -> JSON list
    back = adapt.ModelSpec.from_dict(doc)
    assert isinstance(back.input, dict)
    back_input = back.input["image"]
    assert isinstance(back_input, adapt.Image)
    assert back_input.fit == ("crop", "pad")  # normalizes to a tuple
    assert back == spec

    # Byte-parity: a single fit stays a bare string, not a one-element list.
    single = adapt.ModelSpec(
        input={"image": adapt.Image(role=adapt.IMAGE_PRIMARY, fit="crop")},
        output=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )
    assert single.to_dict()["input"]["image"]["fit"] == "crop"


def test_image_fit_list_selects_per_env():
    # fit=[crop, pad] against an aspect-mismatched env: crop downscales a large
    # camera fine, so it resolves to the model's target shape.
    model = adapt.ModelSpec(
        input={
            "image": adapt.Image(
                role=adapt.IMAGE_PRIMARY, height=4, width=4, fit=("crop", "pad")
            )
        },
        output=SMOLVLA.output,
    )
    payload = resolve(image_env(8, 16), model).transform_obs(
        {"rgb": np.zeros((8, 16, 3), dtype=np.uint8), "instruction": "go"}
    )
    assert payload["image"].shape == (4, 4, 3)


def test_image_channel_mismatch_is_rejected():
    # A grayscale (1-channel) env image with a model declaring 3 channels is a
    # loud resolve error, not a silent wrong-channel feed.
    env = Env(
        tags=adapt.EnvTags(
            observation={"rgb": adapt.ImageTag(role=adapt.IMAGE_PRIMARY)},
            action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
        ),
        obs_space=gym.spaces.Dict(
            {"rgb": gym.spaces.Box(low=0, high=255, shape=(8, 8, 1), dtype=np.uint8)}
        ),
        action_space=box(1),
    )
    model = adapt.ModelSpec(
        input={"image": adapt.Image(role=adapt.IMAGE_PRIMARY, channels=3)},
        output=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )
    with pytest.raises(adapt.AdapterResolutionError, match="channel"):
        resolve(env, model)


def test_image_normalize_range_maps_into_declared_bounds():
    env = image_env(2, 2)
    model = adapt.ModelSpec(
        input={
            "image": adapt.Image(
                role=adapt.IMAGE_PRIMARY,
                dtype="float32",
                normalize=(-1.0, 1.0),
            ),
        },
        output=SMOLVLA.output,
    )
    adapter = resolve(env, model)
    black = adapter.transform_obs(
        {"rgb": np.zeros((2, 2, 3), dtype=np.uint8), "instruction": "go"}
    )
    white = adapter.transform_obs(
        {"rgb": np.full((2, 2, 3), 255, dtype=np.uint8), "instruction": "go"}
    )
    # 0 -> -1, 255 -> 1 (instead of the default [0, 1]).
    np.testing.assert_allclose(black["image"], -1.0, atol=1e-6)
    np.testing.assert_allclose(white["image"], 1.0, atol=1e-6)


def test_image_normalize_overloads_bool_and_range() -> None:
    # bool passes through; False is the off default; a pair coerces to a
    # validated float tuple; a reversed range is rejected at construction.
    assert adapt.Image(adapt.IMAGE_PRIMARY).normalize is False
    assert adapt.Image(adapt.IMAGE_PRIMARY, normalize=True).normalize is True
    assert adapt.Image(adapt.IMAGE_PRIMARY, normalize=(-1, 1)).normalize == (-1.0, 1.0)
    with pytest.raises(ValueError, match="min must be <= max"):
        adapt.Image(adapt.IMAGE_PRIMARY, normalize=(1.0, 0.0))


def test_concat_state_part_rejects_non_default_container_fields() -> None:
    # A State used as a Concat part contributes only its part fields; a non-default
    # container field (dtype/pad_to/reshape/container) is caught at construction.
    with pytest.raises(ValueError, match="container fields"):
        adapt.Concat(adapt.State(adapt.EEF_POS, dtype="int32"))
    # role-only and part-field States are valid parts.
    adapt.Concat(adapt.EEF_POS, adapt.State(adapt.EEF_ROT, encoding="rot6d"))


def test_image_optional_camera_zero_fills_when_absent():
    # A two-camera env (so the single-image fallback does not fire); the model
    # wants a third, absent camera but marks it optional -> a black frame.
    env = Env(
        tags=adapt.EnvTags(
            observation={
                "cam0": adapt.ImageTag(role=adapt.IMAGE_PRIMARY),
                "cam1": adapt.ImageTag(role=adapt.IMAGE_WRIST),
            },
            action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
        ),
        obs_space=gym.spaces.Dict(
            {"cam0": image_space(8, 8), "cam1": image_space(8, 8)}
        ),
        action_space=box(1),
    )
    model = adapt.ModelSpec(
        input={
            "primary": adapt.Image(role=adapt.IMAGE_PRIMARY),
            "overhead": adapt.Image(
                role="image/overhead",
                height=8,
                width=8,
                channels=3,
                optional=True,
            ),
        },
        output=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )
    adapter = resolve(env, model)
    payload = adapter.transform_obs(
        {
            "cam0": np.full((8, 8, 3), 7, dtype=np.uint8),
            "cam1": np.full((8, 8, 3), 9, dtype=np.uint8),
        }
    )
    assert payload["overhead"].shape == (8, 8, 3)
    np.testing.assert_array_equal(payload["overhead"], 0)
    # The zero-filled camera surfaces as a non-fatal advisory, on the caution
    # tier (the model consumes fabricated frames).
    assert any(
        "blank" in note.message
        and "overhead" in note.message
        and note.severity == "caution"
        for note in adapter.advisories()
    )


def test_image_fill_requires_optional():
    """fill without optional could never take effect; it fails at construction
    (naming the role), mirroring Actuator's fill validation."""
    with pytest.raises(ValueError, match=r"'image/primary'.*fill only applies"):
        adapt.Image(adapt.IMAGE_PRIMARY, fill=200)
    filled = adapt.Image(
        adapt.IMAGE_PRIMARY, size=8, channels=3, optional=True, fill=200
    )
    assert filled.fill == 200


def test_image_and_text_fill_travel_under_the_fill_wire_key():
    """The pre-1.0 wire rename: Image.fill and Text.fill serialize as "fill"
    (formerly "absent_fill" / "default"), with no back-compat alias."""
    spec = adapt.ModelSpec(
        input={
            "cam": adapt.Image(
                adapt.IMAGE_PRIMARY, size=8, channels=3, optional=True, fill=128
            ),
            "instruction": adapt.Text(role=adapt.INSTRUCTION, fill="do the task"),
        },
        output=SMOLVLA.output,
    )
    wire = spec.to_dict()
    assert wire["input"]["cam"]["fill"] == 128
    assert "absent_fill" not in wire["input"]["cam"]
    assert wire["input"]["instruction"]["fill"] == "do the task"
    assert "default" not in wire["input"]["instruction"]
    assert adapt.ModelSpec.from_dict(wire) == spec


def test_optional_actuator_round_trips_through_the_wire():
    """action_from_dict reads optional back, so an optional actuator (+fill)
    survives the round-trip instead of dropping the flag."""
    tags = adapt.EnvTags(
        observation={"pos": adapt.StateTag(role=adapt.EEF_POS)},
        action=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, optional=True, fill=0.5),
        ),
    )
    back = adapt.EnvTags.from_dict(tags.to_dict())
    assert back == tags
    assert back.action.components[1].optional is True
    assert back.action.components[1].fill == 0.5


def test_declarative_spec_resolves_without_numpy(monkeypatch):
    """A fully declarative spec resolves on a numpy-less install: the inverse
    self-check imports numpy only when a CustomEncoding is present. Uses
    rlmesh.spaces (native SpaceSpec) spaces; gymnasium spaces are numpy-backed
    and unavailable on such an install anyway."""
    import sys

    from rlmesh import spaces

    monkeypatch.setitem(sys.modules, "numpy", None)
    env_tags = adapt.EnvTags(
        observation={"pos": adapt.StateTag(role=adapt.EEF_POS)},
        action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )
    spec = adapt.ModelSpec(
        input={"state": adapt.State(adapt.EEF_POS)},
        output=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )
    adapter = adapt.resolve(
        env_tags,
        spaces.Dict({"pos": spaces.Box(-1.0, 1.0, (3,))}),
        spaces.Box(-1.0, 1.0, (1,)),
        spec,
    )
    assert adapter.explain()


def test_env_tags_reject_duplicate_observation_roles():
    with pytest.raises(ValueError, match=r"'image/primary' more than once"):
        adapt.EnvTags(
            observation={
                "a": adapt.ImageTag(role=adapt.IMAGE_PRIMARY),
                "b": adapt.ImageTag(role=adapt.IMAGE_PRIMARY),
            },
            action=LIBERO_ACTION,
        )
    with pytest.raises(ValueError, match=r"'proprio/eef_pos' more than once"):
        adapt.EnvTags(
            observation={
                "pos": adapt.StateTag(role=adapt.EEF_POS),
                "flat": adapt.Split(adapt.Field(adapt.EEF_POS, 3)),
            },
            action=LIBERO_ACTION,
        )


def test_model_spec_hash_is_key_order_insensitive():
    """Equal specs whose Dict inputs were authored in different key orders must
    hash equal (eq is order-insensitive); a Custom spec stays hashable."""
    a = adapt.ModelSpec(
        input={
            "state": adapt.State(adapt.EEF_POS),
            "instruction": adapt.Text(role=adapt.INSTRUCTION),
        },
        output=SMOLVLA.output,
    )
    b = adapt.ModelSpec(
        input={
            "instruction": adapt.Text(role=adapt.INSTRUCTION),
            "state": adapt.State(adapt.EEF_POS),
        },
        output=SMOLVLA.output,
    )
    assert a == b
    assert hash(a) == hash(b)
    custom = adapt.ModelSpec(
        input={"extra": adapt.Custom(transform=lambda obs: 0)},
        output=SMOLVLA.output,
    )
    assert isinstance(hash(custom), int)


def test_env_tags_hash_is_key_order_insensitive():
    forward = {
        "pos": adapt.StateTag(role=adapt.EEF_POS),
        "cam": adapt.ImageTag(role=adapt.IMAGE_PRIMARY),
    }
    backward = dict(reversed(list(forward.items())))
    a = adapt.EnvTags(observation=forward, action=LIBERO_ACTION)
    b = adapt.EnvTags(observation=backward, action=LIBERO_ACTION)
    assert a == b
    assert hash(a) == hash(b)


def test_one_element_accept_set_canonicalizes_to_the_bare_form():
    """A 1-element accept-set unwraps at construction (the Rust codec emits the
    bare string on the wire), so from_json(to_json()) == spec."""
    assert adapt.State(adapt.EEF_ROT, encoding=["rot6d"]) == adapt.State(
        adapt.EEF_ROT, encoding="rot6d"
    )
    spec = adapt.ModelSpec(
        input={
            "img": adapt.Image(adapt.IMAGE_PRIMARY, size=8, fit=["crop"]),
            "rot": adapt.State(adapt.EEF_ROT, encoding=["rot6d"]),
        },
        output=SMOLVLA.output,
    )
    assert adapt.ModelSpec.from_json(spec.to_json()) == spec


def test_custom_encoding_in_accept_set_is_rejected_at_construction():
    with pytest.raises(ValueError, match=r"State 'proprio/eef_rot'.*accept-set"):
        adapt.State(adapt.EEF_ROT, encoding=cast("Any", ["rot6d", ROT6D_REV]))
    with pytest.raises(ValueError, match=r"StateTag 'proprio/eef_rot'.*accept-set"):
        adapt.StateTag(role=adapt.EEF_ROT, encoding=cast("Any", ["rot6d", ROT6D_REV]))
    with pytest.raises(ValueError, match=r"Field 'proprio/eef_rot'.*accept-set"):
        adapt.Field(adapt.EEF_ROT, 6, encoding=cast("Any", ["rot6d", ROT6D_REV]))


def test_from_metadata_non_mapping_is_a_resolution_error():
    """The malformed-metadata error converges on AdapterResolutionError, the
    same type resolve_from_contract raises for the same payload."""
    with pytest.raises(adapt.AdapterResolutionError, match="must hold a mapping"):
        adapt.EnvTags.from_metadata({adapt.ENV_METADATA_KEY: "not-a-mapping"})
    with pytest.raises(adapt.AdapterResolutionError, match="must hold a mapping"):
        adapt.ModelSpec.from_metadata({adapt.MODEL_METADATA_KEY: "not-a-mapping"})


def test_observation_role_walk_rejects_a_list_container():
    """The role walker raises the same TypeError the wire encoder raises, so a
    list container fails at construction instead of first at to_dict."""
    with pytest.raises(TypeError, match="must be a leaf, a dict, or a tuple"):
        adapt.EnvTags(
            observation=cast("Any", [adapt.ImageTag(role=adapt.IMAGE_PRIMARY)]),
            action=LIBERO_ACTION,
        )


def test_actuator_rejects_non_positive_dim():
    with pytest.raises(ValueError, match=r"dim must be >= 1, got -2"):
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=-2)


def test_image_size_conflict_error_names_the_role():
    with pytest.raises(ValueError, match=r"'image/primary'.*not both"):
        adapt.Image(adapt.IMAGE_PRIMARY, height=8, size=8)


def test_serve_route_rejects_colliding_custom_placements():
    """Two custom inputs whose structured placements render to the same
    NodePath string cannot share a served-route customs map."""
    from rlmesh.numpy import _numpy_bridge

    env = single_state_env("pos", gym.spaces.Dict({"pos": box(3)}))
    spec = adapt.ModelSpec(
        input={
            "state": adapt.State(adapt.EEF_POS),
            "a": {"b": adapt.Custom(transform=lambda obs: 0.0)},
            "a.b": adapt.Custom(transform=lambda obs: 1.0),
        },
        output=SMOLVLA.output,
    )
    adapter = resolve(env, spec)
    with pytest.raises(ValueError, match="colliding route keys"):
        adapter.serve_route(_numpy_bridge)


ABS_LIBERO_ENV = Env(
    tags=adapt.EnvTags(
        observation=LIBERO_ENV.tags.observation,
        action=adapt.Action(
            adapt.Actuator(adapt.ACTION_EEF_POS, dim=3),
            adapt.Actuator(adapt.ACTION_EEF_ROT, dim=3, encoding="axis_angle"),
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
        ),
    ),
    obs_space=LIBERO_ENV.obs_space,
    action_space=box(7),
)

ABS_TARGET_MODEL = adapt.ModelSpec(
    input={"image": adapt.Image(role=adapt.IMAGE_PRIMARY, height=64, width=64)},
    output=adapt.Action(
        adapt.Actuator(adapt.ACTION_EEF_POS, dim=3),
        adapt.Actuator(adapt.ACTION_EEF_ROT, dim=6, encoding="rot6d"),
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, binary=True, threshold=0.5),
    ),
)


def test_absolute_eef_target_passes_through_with_rotation_conversion():
    adapter = resolve(ABS_LIBERO_ENV, ABS_TARGET_MODEL)
    # Column-concat rot6d of R = [[1,0,0],[0,0,-1],[0,1,0]], a +90deg turn about x.
    r6d = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    out = adapter.transform_action(
        np.array([0.3, -0.1, 1.2, *r6d, 0.9], dtype=np.float32)
    )
    assert out.shape == (7,)
    np.testing.assert_allclose(
        out[:3], [0.3, -0.1, 1.2], atol=1e-6
    )  # no clip, no scale
    np.testing.assert_allclose(out[3:6], [np.pi / 2, 0.0, 0.0], atol=1e-5)
    assert out[6] == 1.0


def test_absolute_eef_roles_are_registered_with_fixed_position_width():
    bad = adapt.ModelSpec(
        input={"image": adapt.Image(role=adapt.IMAGE_PRIMARY, height=64, width=64)},
        output=adapt.Action(adapt.Actuator(adapt.ACTION_EEF_POS, dim=2)),
    )
    with pytest.raises(Exception, match="3-D by convention"):
        resolve(ABS_LIBERO_ENV, bad)


JOINT_BIMANUAL_ENV = Env(
    tags=adapt.EnvTags(
        observation={
            "head": adapt.ImageTag(role=adapt.IMAGE_PRIMARY),
            "left_wrist": adapt.ImageTag(role=adapt.IMAGE_WRIST),
            "right_wrist": adapt.ImageTag(role=adapt.IMAGE_WRIST_2),
        },
        action=adapt.Action(
            adapt.Actuator(adapt.ACTION_JOINT_POS, dim=6),
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1),
            adapt.Actuator(adapt.ACTION_JOINT_POS_2, dim=6),
            adapt.Actuator(adapt.ACTION_GRIPPER_2, dim=1),
        ),
    ),
    obs_space=gym.spaces.Dict(
        {
            "head": image_space(),
            "left_wrist": image_space(),
            "right_wrist": image_space(),
        }
    ),
    action_space=box(14),
)

# The model emits both arms' joints first and both grippers last; the env
# interleaves them arm by arm. Only the `_2` roles can express the difference.
JOINT_BIMANUAL_MODEL = adapt.ModelSpec(
    input={
        "image": adapt.Image(role=adapt.IMAGE_PRIMARY, height=64, width=64),
        "wrist": adapt.Image(role=adapt.IMAGE_WRIST, height=64, width=64),
        "wrist_2": adapt.Image(role=adapt.IMAGE_WRIST_2, height=64, width=64),
    },
    output=adapt.Action(
        adapt.Actuator(adapt.ACTION_JOINT_POS, dim=6),
        adapt.Actuator(adapt.ACTION_JOINT_POS_2, dim=6),
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1),
        adapt.Actuator(adapt.ACTION_GRIPPER_2, dim=1),
    ),
)


def test_bimanual_joint_split_permutes_the_model_vector():
    adapter = resolve(JOINT_BIMANUAL_ENV, JOINT_BIMANUAL_MODEL)
    out = adapter.transform_action(np.arange(14, dtype=np.float32))
    np.testing.assert_allclose(out, [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13])
    payload = adapter.transform_obs(
        {
            "head": np.zeros((64, 64, 3), dtype=np.uint8),
            "left_wrist": np.zeros((64, 64, 3), dtype=np.uint8),
            "right_wrist": np.full((64, 64, 3), 7, dtype=np.uint8),
        }
    )
    # The second wrist camera lands in its own slot, not aliased onto the first.
    assert payload["wrist_2"].max() > payload["wrist"].max()


# --- Declarative state parts: constant / post_rotate / scale / offset ---------
#
# The three model-side rebuilds these fields replace, each pinned against the
# catalog's own arithmetic (rlmesh-catalog/xvla-chunk, rc.8).

# `models/gr00t-n1.7/gr00t_model.py` WidowXBridgeState._DEFAULT_ROT.
GR00T_DEFAULT_ROT = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

# `models/xvla/libero/main.py` XVLALibero.HAND_TO_GRIP.
XVLA_HAND_TO_GRIP = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)

BRIDGE_ENV = Env(
    tags=adapt.EnvTags(
        observation={
            "image": adapt.ImageTag(adapt.IMAGE_PRIMARY),
            "eef_pos": adapt.StateTag(adapt.EEF_POS),
            "eef_quat": adapt.StateTag(adapt.EEF_ROT, encoding="quat_wxyz"),
            "gripper": adapt.StateTag(adapt.GRIPPER_POS),
            "instruction": adapt.TextTag(adapt.INSTRUCTION),
        },
        action=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3, range=(-1.0, 1.0)),
            adapt.Actuator(
                adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle", range=(-1.0, 1.0)
            ),
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
        ),
    ),
    obs_space=gym.spaces.Dict(
        {
            "image": image_space(),
            "eef_pos": box(3),
            "eef_quat": box(4),
            "gripper": box(1),
            "instruction": text_space(),
        }
    ),
    action_space=ACTION7,
)

# A recorded widowx observation: wxyz quaternion straight off `env.tcp.pose.q`.
BRIDGE_OBS: dict[str, Any] = {
    "image": np.zeros((64, 64, 3), dtype=np.uint8),
    "eef_pos": np.array([0.281_25, -0.041_5, 0.137_75], dtype=np.float32),
    "eef_quat": np.array([0.137_84, 0.694_21, -0.135_47, 0.693_03], dtype=np.float32),
    "gripper": np.array([0.812_5], dtype=np.float32),
    "instruction": "put the eggplant in the basket",
}


def test_gr00t_bridge_state_is_expressible_declaratively():
    """The gr00t widowx embodiment wants ``[pos, euler, pad, gripper]``.

    ``models/gr00t-n1.7/gr00t_model.py`` builds it by hand in
    ``WidowXBridgeState._obs``::

        mat = R.from_quat(quat_xyzw).as_matrix()
        euler = R.from_matrix(mat @ self._DEFAULT_ROT.T).as_euler("xyz")
        state = np.concatenate([pos, euler, [0.0], grip])

    The env declares ``quat_wxyz``, so the old spec's ``encoding="quat_xyzw"``
    was doing the wxyz->xyzw permutation the hand-rolled code depended on.
    Declared instead: ``euler_xyz`` plus a ``post_rotate`` of
    ``_DEFAULT_ROT.T``, and the pad channel as a ``Constant``.
    """
    scipy_rotation = pytest.importorskip("scipy.spatial.transform").Rotation

    spec = adapt.ModelSpec(
        input={
            "state": adapt.Concat(
                adapt.EEF_POS,
                adapt.State(
                    adapt.EEF_ROT,
                    encoding="euler_xyz",
                    post_rotate=adapt.Rotation.from_matrix(GR00T_DEFAULT_ROT.T),
                ),
                adapt.Constant(dim=1),
                adapt.State(adapt.GRIPPER_POS, dim=1),
            )
        },
        output=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
            adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="euler_xyz"),
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(0.0, 1.0)),
        ),
    )

    wxyz = np.asarray(BRIDGE_OBS["eef_quat"], dtype=np.float64)
    quat_xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    matrix = scipy_rotation.from_quat(quat_xyzw).as_matrix()
    euler = scipy_rotation.from_matrix(matrix @ GR00T_DEFAULT_ROT.T).as_euler("xyz")
    expected = np.concatenate(
        [
            np.asarray(BRIDGE_OBS["eef_pos"], dtype=np.float64),
            euler,
            [0.0],
            np.asarray(BRIDGE_OBS["gripper"], dtype=np.float64),
        ]
    )

    state = resolve(BRIDGE_ENV, spec).transform_obs(BRIDGE_OBS)["state"]
    assert state.shape == (8,)
    np.testing.assert_allclose(state, expected, atol=1e-5)


def test_xvla_libero_hand_to_grip_is_a_post_rotation():
    """``models/xvla/libero/main.py`` rebuilds the rot6d block model-side::

        grip = _rot6d_to_matrix(state[3:9]) @ self.HAND_TO_GRIP
        state[3:9] = _matrix_to_rot6d(grip)
        state[9] = 0.0  # upstream proprio carries a constant 0 gripper slot

    Both halves are declarable: a rigid right-multiplication is ``post_rotate``
    and the pinned slot is a ``Constant``. (Binding the constant deliberately
    drops the env's ``proprio/gripper`` -- the checkpoint's slot 9 is not a
    function of the env gripper.)
    """
    scipy_rotation = pytest.importorskip("scipy.spatial.transform").Rotation

    spec = adapt.ModelSpec(
        input={
            "state": adapt.Concat(
                adapt.EEF_POS,
                adapt.State(
                    adapt.EEF_ROT,
                    encoding="rot6d",
                    post_rotate=adapt.Rotation.from_matrix(XVLA_HAND_TO_GRIP),
                ),
                adapt.Constant(dim=1),
            )
        },
        output=XVLA.output,
    )

    obs = make_obs()
    matrix = scipy_rotation.from_quat(
        np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
    ).as_matrix()
    # `_matrix_to_rot6d(rot) = rot[:, :2].T.reshape(-1)` -- the two leading
    # columns concatenated, which is exactly the `rot6d` encoding.
    grip = matrix @ XVLA_HAND_TO_GRIP
    expected = np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
            grip[:, :2].T.reshape(-1),
            [0.0],
        ]
    )

    state = resolve(LIBERO_ENV, spec).transform_obs(obs)["state"]
    assert state.shape == (10,)
    np.testing.assert_allclose(state, expected, atol=1e-5)


def test_state_scale_and_offset_express_the_robotwin_gripper():
    """``models/xvla/robotwin2/main.py`` maps RoboTwin's gripper into the
    model's training convention with ``proprio[9] = 1.0 - proprio[9] * 2.0``;
    declared, that is ``scale=-2, offset=1``."""
    spec = adapt.ModelSpec(
        input={
            "state": adapt.Concat(
                adapt.EEF_POS,
                adapt.State(adapt.GRIPPER_POS, dim=1, scale=-2.0, offset=1.0),
            )
        },
        output=XVLA.output,
    )
    obs = make_obs()
    gripper = float(np.asarray(obs["robot0_gripper_qpos"])[0])

    state = resolve(LIBERO_ENV, spec).transform_obs(obs)["state"]
    assert state.shape == (4,)
    np.testing.assert_allclose(state[3], 1.0 - 2.0 * gripper, atol=1e-6)


def test_constant_part_is_not_reported_as_a_zero_filled_role():
    """C14: a declared constant is authored data, so it must not read as
    fabricated the way an absent optional role does."""
    constant = adapt.ModelSpec(
        input={"state": adapt.Concat(adapt.EEF_POS, adapt.Constant(dim=2))},
        output=XVLA.output,
    )
    absent = adapt.ModelSpec(
        input={
            "state": adapt.Concat(
                adapt.EEF_POS,
                adapt.State(adapt.EEF_POS_2, dim=2, optional=True),
            )
        },
        output=XVLA.output,
    )
    constant_text = resolve(LIBERO_ENV, constant).explain()
    assert "const(2)=0.0" in constant_text
    assert "zeros(2)" not in constant_text
    assert not [
        note
        for note in resolve(LIBERO_ENV, constant).advisories()
        if "zero-filled" in note.message
    ]
    assert "zeros(2)" in resolve(LIBERO_ENV, absent).explain()
    assert [
        note
        for note in resolve(LIBERO_ENV, absent).advisories()
        if "zero-filled" in note.message
    ]


def test_state_of_only_constants_is_refused():
    with pytest.raises(ValueError, match="reads nothing from the env"):
        adapt.Concat(adapt.Constant(dim=3))


def test_non_zero_fill_needs_optional_and_folds_scale_and_offset():
    with pytest.raises(ValueError, match="only to an optional part"):
        adapt.State(adapt.EEF_POS_2, dim=1, fill=1.0)
    spec = adapt.ModelSpec(
        input={
            "state": adapt.Concat(
                adapt.EEF_POS,
                adapt.State(
                    adapt.EEF_POS_2,
                    dim=2,
                    optional=True,
                    fill=0.5,
                    scale=2.0,
                    offset=1.0,
                ),
            )
        },
        output=XVLA.output,
    )
    state = resolve(LIBERO_ENV, spec).transform_obs(make_obs())["state"]
    # fill * scale + offset, folded once at resolve.
    np.testing.assert_allclose(state[3:], [2.0, 2.0], atol=1e-6)


def test_post_rotate_needs_a_rotation_encoding():
    identity = adapt.Rotation.from_matrix(np.eye(3))
    with pytest.raises(ValueError, match="post_rotate needs a rotation encoding"):
        adapt.State(adapt.EEF_ROT, post_rotate=identity)
    assert identity.encoding == "rot6d"
    assert identity.value == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def test_rotation_literal_must_be_a_rotation():
    sheared = adapt.ModelSpec(
        input={
            "state": adapt.Concat(
                adapt.EEF_POS,
                adapt.State(
                    adapt.EEF_ROT,
                    encoding="rot6d",
                    post_rotate=adapt.Rotation(
                        encoding="rot6d", value=(1.0, 0.0, 0.0, 0.5, 1.0, 0.0)
                    ),
                ),
            )
        },
        output=XVLA.output,
    )
    with pytest.raises(ValueError, match="orthonormal"):
        sheared.to_dict()


def test_constant_part_shifts_a_later_custom_encoding_slice():
    """A constant contributes width like any other part, so the host-side
    repack that follows it must be addressed past it."""
    swap = adapt.CustomEncoding(
        base="quat_xyzw",
        from_base=lambda v: np.asarray(v)[[3, 0, 1, 2]],
        to_base=lambda v: np.asarray(v)[[1, 2, 3, 0]],
    )
    spec = adapt.ModelSpec(
        input={
            "state": adapt.Concat(
                adapt.EEF_POS,
                adapt.Constant(dim=2, fill=1.0),
                adapt.State(adapt.EEF_ROT, encoding=swap),
            )
        },
        output=XVLA.output,
    )
    adapter = resolve(LIBERO_ENV, spec)
    assert "'state'[5:9]" in adapter.explain()
    obs = make_obs()
    state = adapter.transform_obs(obs)["state"]
    np.testing.assert_allclose(state[3:5], [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(
        state[5:], np.asarray(obs["robot0_eef_quat"])[[3, 0, 1, 2]], atol=1e-6
    )


def test_frame_and_reference_are_keyword_only_and_omitted_when_unset() -> None:
    # Appended last and keyword-only, so every existing positional call site
    # keeps its meaning, and a spec that declares neither is byte-identical on
    # the wire to one written before the attributes existed.
    positional = adapt.Actuator(adapt.ACTION_GRIPPER, 1, None, None, True)
    assert positional.binary is True
    assert positional.frame is None and positional.reference is None
    assert adapt.StateTag(adapt.EEF_POS, "quat_xyzw").frame is None
    assert adapt.Field(adapt.EEF_POS, 3).frame is None
    assert adapt.State(adapt.EEF_POS, "quat_xyzw", 3).frame is None

    bare = adapt.EnvTags(
        observation={"p": adapt.StateTag(adapt.EEF_POS)},
        action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    ).to_dict()
    assert "frame" not in bare["observation"]["p"]
    assert "reference" not in bare["action"]["components"][0]

    # Declared, they round-trip by value through the Rust codec.
    tags = adapt.EnvTags(
        observation={"p": adapt.StateTag(adapt.EEF_POS, frame="robot_base")},
        action=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3, reference="current")
        ),
    )
    assert adapt.EnvTags.from_dict(tags.to_dict()) == tags
    spec = adapt.ModelSpec(
        input={"s": adapt.State(adapt.EEF_POS, dim=3, frame="world")},
        output=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3, reference="target")
        ),
    )
    assert adapt.ModelSpec.from_dict(spec.to_dict()) == spec


def test_a_role_less_leaf_may_not_carry_a_frame() -> None:
    # A skip advances the offset and a constant emits a fixed block; neither has
    # a pose to express in a frame.
    with pytest.raises(ValueError, match="role-less"):
        adapt.Actuator(dim=2, frame="world")
    with pytest.raises(ValueError, match="role-less"):
        adapt.Actuator(dim=2, reference="current")
    with pytest.raises(ValueError, match="role-less field"):
        adapt.EnvTags.from_dict(
            {
                "observation": {
                    "s": {
                        "type": "split",
                        "fields": [{"dim": 1, "frame": "world"}],
                    }
                },
                "action": {"components": [{"role": adapt.ACTION_GRIPPER, "dim": 1}]},
            }
        )


def test_frame_and_reference_disagreement_is_a_hard_resolve_error() -> None:
    # The xvla/widowx class of bug: an absolute base-frame head bound to a
    # delta controller. Both halves now have a name and fail loudly.
    env = LIBERO_ENV._replace(
        tags=adapt.EnvTags(
            observation={
                **LIBERO_ENV.tags.observation,
                "robot0_eef_pos": adapt.StateTag(adapt.EEF_POS, frame="world"),
            },
            action=LIBERO_ACTION,
        )
    )
    spec = adapt.ModelSpec(
        input={"state": adapt.State(adapt.EEF_POS, frame="robot_base")},
        output=LIBERO_MODEL_ACTION,
    )
    with pytest.raises(
        adapt.AdapterResolutionError,
        match='the model expects frame "robot_base" but the env declares "world"',
    ):
        resolve(env, spec)

    delta_target = adapt.ModelSpec(
        input={"state": adapt.Concat(adapt.EEF_POS)},
        output=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3, reference="target"),
            adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
        ),
    )
    env_current = LIBERO_ENV._replace(
        tags=adapt.EnvTags(
            observation=LIBERO_ENV.tags.observation,
            action=adapt.Action(
                adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3, reference="current"),
                adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
                adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
                clip=(-1.0, 1.0),
            ),
        )
    )
    with pytest.raises(
        adapt.AdapterResolutionError,
        match='the model expects reference "target" but the env declares "current"',
    ):
        resolve(env_current, delta_target)


def test_a_model_only_frame_is_a_caution_and_shows_in_the_summary() -> None:
    # C14 case 6: the model states a requirement the env cannot confirm. The
    # run proceeds; the pairing carries a caution and the summary the frame.
    spec = adapt.ModelSpec(
        input={"state": adapt.State(adapt.EEF_POS, frame="robot_base")},
        output=LIBERO_MODEL_ACTION,
    )
    adapter = resolve(LIBERO_ENV, spec)
    cautions = [
        note
        for note in adapter.advisories()
        if note.severity == "caution" and "frame" in note.message
    ]
    assert len(cautions) == 1
    assert "cannot be verified" in cautions[0].message
    assert "@robot_base" in adapter.explain()
    # Not a drop: the caution must stay out of the `dropped:` section.
    assert "dropped:" not in adapter.explain()


def test_require_frames_is_an_opt_in_publish_gate() -> None:
    import json

    from rlmesh._rlmesh import adapters_spec_normalize

    bare = json.dumps(
        adapt.EnvTags(
            observation={"p": adapt.StateTag(adapt.EEF_POS)},
            action=adapt.Action(adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3)),
        ).to_dict()
    )
    # Default tier: an absent frame is legal v1.
    adapters_spec_normalize("env", bare, True)
    with pytest.raises(ValueError, match="without a frame"):
        adapters_spec_normalize("env", bare, True, "passthrough", True)

    declared = json.dumps(
        adapt.EnvTags(
            observation={"p": adapt.StateTag(adapt.EEF_POS, frame="robot_base")},
            action=adapt.Action(
                adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3, reference="current")
            ),
        ).to_dict()
    )
    adapters_spec_normalize("env", declared, True, "passthrough", True)
