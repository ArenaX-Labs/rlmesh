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

from types import SimpleNamespace
from typing import Any, NamedTuple

import gymnasium as gym
import numpy as np
import pytest
import rlmesh.adapters as adapt

# ---------------------------------------------------------------------------
# Reference math ported from the bespoke adapter base class.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Space + tag helpers.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Shared LIBERO-style env and a synthetic observation.
# ---------------------------------------------------------------------------

LIBERO_ACTION = adapt.Action(
    adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
    adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
    adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
    clip=(-1.0, 1.0),
)

LIBERO_ENV = Env(
    tags=adapt.EnvTags(
        observation={
            "agentview_image": adapt.ImageTag(role=adapt.IMAGE_PRIMARY),
            "robot0_eye_in_hand_image": adapt.ImageTag(role=adapt.IMAGE_WRIST),
            "robot0_eef_pos": adapt.StateTag(role=adapt.EEF_POS),
            "robot0_eef_quat": adapt.StateTag(role=adapt.EEF_ROT, encoding="quat_xyzw"),
            "robot0_gripper_qpos": adapt.StateTag(role=adapt.GRIPPER_POS),
            "instruction": adapt.TextTag(),
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


# ---------------------------------------------------------------------------
# SmolVLA
# ---------------------------------------------------------------------------

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
        "instruction": adapt.Text(),
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
            observation={"instruction": adapt.TextTag()},
            action=adapt.Action(
                adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, invert=True, binary=True),
            ),
        ),
        obs_space=gym.spaces.Dict({"instruction": text_space()}),
        action_space=box(1, low=-1.0, high=1.0),
    )
    model = adapt.ModelSpec(
        input={"instruction": adapt.Text()},
        output=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )
    adapter = resolve(env, model)
    # The model says "close" (+0.8); the env's opposite sign + binary snap to -1.
    result = adapter.transform_action(np.array([0.8], dtype=np.float32))
    np.testing.assert_allclose(result, [-1.0], rtol=1e-6)


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


# ---------------------------------------------------------------------------
# OpenVLA
# ---------------------------------------------------------------------------

OPENVLA = adapt.ModelSpec(
    input={
        "image": adapt.Image(role=adapt.IMAGE_PRIMARY, height=64, width=64),
        "instruction": adapt.Text(),
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


# ---------------------------------------------------------------------------
# X-VLA (rot6d proprio, unified 20-dim single/bimanual state and ee6d action:
# dims 1-10 are the first arm, dims 11-20 the second; second-arm components
# are optional so single-arm envs resolve them to zero fill / dropped dims)
# ---------------------------------------------------------------------------

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
        "instruction": adapt.Text(),
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


# ---------------------------------------------------------------------------
# Bimanual env: the same X-VLA spec consumes dims 11-20 for real.
# ---------------------------------------------------------------------------

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
            "instruction": adapt.TextTag(),
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
    text = resolve(LIBERO_ENV, XVLA).describe()
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
    assert "JointSpaceAdapter" in adapter.describe()
    assert isinstance(resolve(LIBERO_ENV, SMOLVLA), adapt.AdapterBase)


def test_custom_adapter_reset_is_a_no_op_by_default():
    adapter = resolve(LIBERO_ENV, SMOLVLA)
    # Resolved adapters are stateless; reset must exist and do nothing.
    adapter.reset()
    payload = adapter.transform_obs(make_obs())
    assert "observation.state" in payload


# ---------------------------------------------------------------------------
# GR00T-style (split state keys, flipped images, lead dims, binary gripper)
# ---------------------------------------------------------------------------

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
            container="list",
            default="",
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


# ---------------------------------------------------------------------------
# Image pipeline details
# ---------------------------------------------------------------------------


def image_env(height: int, width: int, *, role: str = adapt.IMAGE_PRIMARY) -> Env:
    """A minimal single-image env (plus instruction) over a given image size."""
    return Env(
        tags=adapt.EnvTags(
            observation={
                "rgb": adapt.ImageTag(role=role),
                "instruction": adapt.TextTag(),
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


def test_bilinear_aa_resize_matches_pillow_within_one_step():
    pil = pytest.importorskip("PIL.Image")
    env = image_env(6, 8)
    image = (
        (np.arange(6 * 8 * 3, dtype=np.int64) * 7 % 251)
        .astype(np.uint8)
        .reshape(6, 8, 3)
    )
    for height, width in ((3, 4), (12, 16)):
        spec = adapt.ModelSpec(
            # (12, 16) upscales the 6x8 env image, which the resolver gates
            # behind allow_upscale; this test deliberately exercises both
            # directions of the bilinear-AA resize.
            input={
                "image": adapt.Image(height=height, width=width, allow_upscale=True)
            },
            output=SMOLVLA.output,
        )
        ours = (
            resolve(env, spec).transform_obs({"rgb": image})["image"].astype(np.int16)
        )
        theirs = np.asarray(
            pil.fromarray(image).resize((width, height), pil.Resampling.BILINEAR),
            dtype=np.int16,
        )
        assert int(np.abs(ours - theirs).max()) <= 1


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
        input={"image": adapt.Image()},
        output=SMOLVLA.output,
    )
    payload = resolve(env, spec).transform_obs({"rgb": make_png(pixels)})
    np.testing.assert_array_equal(payload["image"], pixels)


def test_undecodable_image_bytes_is_an_error():
    env = image_env(2, 2)
    spec = adapt.ModelSpec(
        input={"image": adapt.Image()},
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
            "image": adapt.Image(height=4, width=5, resample="bilinear", fit="stretch"),
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


# ---------------------------------------------------------------------------
# Lookup, custom transforms, wrap_predict, nested keys
# ---------------------------------------------------------------------------


def single_state_env(key: str, obs_space: gym.spaces.Space[Any]) -> Env:
    """An env with one EEF_POS state under ``key`` and the given obs space."""
    return Env(
        tags=adapt.EnvTags(
            observation={key: adapt.StateTag(role=adapt.EEF_POS)},
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


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Resolution errors
# ---------------------------------------------------------------------------


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
                "agentview_image": adapt.ImageTag(),
                "robot0_eef_quat": adapt.StateTag(
                    role=adapt.EEF_ROT, encoding="quat_xyzw"
                ),
                "instruction": adapt.TextTag(),
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
        input={"image": adapt.Image()},
        output=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
        ),
    )
    with pytest.raises(adapt.AdapterResolutionError, match="action/delta_eef_rot"):
        resolve(LIBERO_ENV, spec)


def test_action_dim_mismatch_is_an_error():
    spec = adapt.ModelSpec(
        input={"image": adapt.Image()},
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
    text = resolve(LIBERO_ENV, XVLA).describe()
    assert '"state"' in text
    # X-VLA proprio and action both use row-major rot6d.
    assert "quat_xyzw->rot6d_rowmajor" in text
    assert "rot6d_rowmajor->axis_angle" in text


# ---------------------------------------------------------------------------
# tag verb + Model(spec=) entry-point guards
# ---------------------------------------------------------------------------


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


def test_negative_u32_fields_rejected_by_rust_codec() -> None:
    # Negatives are rejected by the authoritative Rust codec (u32) at
    # serialize/normalize with a field-path-annotated message. Action components
    # (not behind a tag) get a deep path; model-input payload fields resolve to
    # `inputs[0]` (the ModelInput tag boundary) plus the u32 constraint.
    gripper = adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1))

    def to_dict(*, output: adapt.Action = gripper, **inputs: adapt.ModelLeaf) -> None:
        adapt.ModelSpec(input=inputs, output=output).to_dict()

    with pytest.raises(
        ValueError, match=r"at output\.components\[0\]\.dim.*non-negative integer"
    ):
        to_dict(output=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=-1)))
    with pytest.raises(ValueError, match=r"non-negative integer"):
        to_dict(image=adapt.Image(width=-1))
    with pytest.raises(ValueError, match=r"non-negative integer"):
        to_dict(image=adapt.Image(lead_dims=-2))
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


# ---------------------------------------------------------------------------
# Spec ergonomics: size=, single-component StateInput, eager validation
# ---------------------------------------------------------------------------


def test_image_input_size_shorthand() -> None:
    assert adapt.Image(size=224) == adapt.Image(height=224, width=224)
    with pytest.raises(ValueError, match="size=, or height"):
        adapt.Image(size=224, height=10)


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


# ---------------------------------------------------------------------------
# Frame stacking (host-side, stateful)
# ---------------------------------------------------------------------------


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

    # A non-zero lane's autoreset must not wipe the shared, not-yet-lane-indexed
    # buffers, or it would corrupt the still-running lane-0 episode.
    adapter.reset(env_index=1)
    np.testing.assert_array_equal(
        adapter.transform_obs({"rgb": f2})["img"], np.stack([f1, f2, f2])
    )

    # Lane 0 (and a whole-vector reset, env_index=None) does clear.
    adapter.reset(env_index=0)
    np.testing.assert_array_equal(
        adapter.transform_obs({"rgb": f2})["img"], np.stack([f2, f2, f2])
    )


def test_image_input_stack_round_trips_and_omits_default() -> None:
    from rlmesh.adapters.specs.model_serialization import model_input_to_dict

    spec = adapt.ModelSpec(input={"img": adapt.Image(stack=4)}, output=SMOLVLA.output)
    assert adapt.ModelSpec.from_json(spec.to_json()) == spec
    assert "stack" not in model_input_to_dict(adapt.Image())  # default omitted
    assert model_input_to_dict(adapt.Image(stack=4))["stack"] == 4
    # stack bounds (1..64) are enforced by the Rust codec at serialize/normalize.
    with pytest.raises(ValueError, match="stack must be between 1 and 64"):
        adapt.ModelSpec(
            input={"img": adapt.Image(stack=0)}, output=SMOLVLA.output
        ).to_dict()
    # An untrusted spec cannot demand an unbounded buffer.
    with pytest.raises(ValueError, match="stack must be between 1 and 64"):
        adapt.ModelSpec(
            input={"img": adapt.Image(stack=10_000)}, output=SMOLVLA.output
        ).to_dict()


# ---------------------------------------------------------------------------
# Vector-env interim guard (pending the per-lane affinity manager)
# ---------------------------------------------------------------------------


def test_io_adapter_is_stateful_only_with_frame_history() -> None:
    env = image_env(4, 4)
    stateless = resolve(
        env,
        adapt.ModelSpec(
            input={"img": adapt.Image(role=adapt.IMAGE_PRIMARY)},
            output=SMOLVLA.output,
        ),
    )
    stateful = resolve(
        env,
        adapt.ModelSpec(
            input={"img": adapt.Image(role=adapt.IMAGE_PRIMARY, stack=2)},
            output=SMOLVLA.output,
        ),
    )
    assert stateless.is_stateful is False
    assert stateful.is_stateful is True


def test_vector_env_rejected_by_single_env_eval_loop() -> None:
    # The per-episode eval loop reads scalar reward/termination, so it rejects any
    # vector env (num_envs>1) up front instead of crashing on array truthiness deep
    # in the step loop -- regardless of whether the adapter is stateful.
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
    with pytest.raises(ValueError, match="num_envs=2"):
        model.run(fake_env, max_episodes=1)


def test_stateful_adapter_allowed_on_vector_route() -> None:
    # Frame-stacking state is now episode-keyed in the native serving engine, so
    # a served stateful adapter resolves against a vectorized route -- the old
    # single-lane rejection is lifted.
    from rlmesh._models._eval import resolve_route_adapter

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
    vector_stateful = resolve_route_adapter(
        stateful_spec, contract(2), trust_entrypoints=False
    )
    assert vector_stateful is not None and vector_stateful.is_stateful
    # And still against a single lane.
    single_lane = resolve_route_adapter(
        stateful_spec, contract(1), trust_entrypoints=False
    )
    assert single_lane is not None and single_lane.is_stateful
    # A stateless adapter on a vector route is harmless.
    stateless = resolve_route_adapter(
        stateless_spec, contract(2), trust_entrypoints=False
    )
    assert stateless is not None and stateless.is_stateful is False
    # spec=None / no tags also resolves to None without over-rejecting.
    untagged: Any = SimpleNamespace(metadata={}, num_envs=2)
    assert resolve_route_adapter(None, untagged, trust_entrypoints=False) is None


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


# ---------------------------------------------------------------------------
# Flat (non-Dict) observations: env-side StateLayout
# ---------------------------------------------------------------------------


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
    text = adapter.describe()
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


# ---------------------------------------------------------------------------
# Host-side custom rotation encodings (CustomEncoding)
# ---------------------------------------------------------------------------

# A width-preserving rot6d repacking whose inverse is itself (reverse the
# 6-vector). Used to exercise the host-side shims against a native baseline.
ROT6D_REV = adapt.CustomEncoding(
    base="rot6d",
    from_base=lambda v: np.asarray(v)[::-1],
    to_base=lambda v: np.asarray(v)[::-1],
    name="rot6d_rev",
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
    text = resolve(env, spec).describe()
    assert "host-side encodings:" in text
    assert "rot6d -> rot6d_rev" in text


def test_custom_obs_encoding_must_be_sole_component():
    env = _rot_obs_env()
    spec = adapt.ModelSpec(
        input={
            "state": adapt.Concat(
                adapt.EEF_POS,
                adapt.State(adapt.EEF_ROT, encoding=ROT6D_REV),
            ),
        },
        output=_gripper_action(),
    )
    with pytest.raises(adapt.AdapterResolutionError, match="sole part"):
        resolve(env, spec)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"pad_to": 8}, "pad_to/reshape"),
        ({"reshape": (1, 6)}, "pad_to/reshape"),
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


def test_custom_encoding_spec_is_not_serializable():
    spec = adapt.ModelSpec(
        input={"rot": adapt.State(adapt.EEF_ROT, encoding=ROT6D_REV)},
        output=_gripper_action(),
    )
    with pytest.raises(ValueError, match="CustomEncoding"):
        spec.to_dict()


def test_encoding_free_spec_still_serializes_unchanged():
    # A spec with no CustomEncoding must round-trip byte-identically.
    assert adapt.ModelSpec.from_dict(SMOLVLA.to_dict()) == SMOLVLA


def test_scalar_reshape_survives_serialization_and_resolve():
    # reshape=() targets a 0-D scalar; it must not be dropped as falsy.
    env = single_state_env("g", gym.spaces.Dict({"g": box(1)}))
    spec = adapt.ModelSpec(
        input={"state": adapt.Concat(adapt.EEF_POS, reshape=())},
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


def test_default_only_text_is_not_a_referenced_obs_key():
    # A TextInput satisfied only by its default has env_key ""; that empty key
    # must not be reported as an observation key to decode.
    env = single_state_env("pos", gym.spaces.Dict({"pos": box(3)}))
    spec = adapt.ModelSpec(
        input={
            "state": adapt.State(adapt.EEF_POS),
            "instruction": adapt.Text(default="do the task"),
        },
        output=SMOLVLA.output,
    )
    adapter = resolve(env, spec)
    referenced = adapter._plan.referenced_obs_keys()
    assert "" not in referenced
    assert "pos" in referenced


# ---------------------------------------------------------------------------
# Finiteness contract (P0.3): range/clip/scale/threshold bounds are finite on
# the wire (an unbounded bound is omitted, not +/-inf). Python is the *produce*
# side: it must reject non-finite at construction and refuse to emit Infinity,
# so it never ships a spec the Rust serde codec rejects.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Metadata dual-read dispatch (P0.6): from_metadata iterates known keys
# newest-format-first, so a v2 reader still reads v1. Today only v1 exists.
# ---------------------------------------------------------------------------


def test_from_metadata_reads_v1_and_returns_none_when_absent():
    spec = adapt.ModelSpec(
        input={},
        output=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )
    assert adapt.ModelSpec.from_metadata(spec.to_metadata()) == spec
    assert adapt.ModelSpec.from_metadata({}) is None

    tags = adapt.EnvTags(observation={}, action=adapt.Action())
    assert adapt.EnvTags.from_metadata(tags.to_metadata()) == tags
    assert adapt.EnvTags.from_metadata({}) is None


# ---------------------------------------------------------------------------
# Rotation-encoding accept-sets: a side declares a preference list; the resolver
# prefers the env's native encoding when the model accepts it (no conversion),
# else converts into the first preference. A single encoding stays a bare string
# on the wire (byte-parity with pre-accept-set specs).
# ---------------------------------------------------------------------------


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
        input={"image": adapt.Image(height=64, width=64, fit=("crop", "pad"))},
        output=adapt.Action(),
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
        input={"image": adapt.Image(fit="crop")},
        output=adapt.Action(),
    )
    assert single.to_dict()["input"]["image"]["fit"] == "crop"


def test_image_fit_list_selects_per_env():
    # fit=[crop, pad] against an aspect-mismatched env: crop downscales a large
    # camera fine, so it resolves to the model's target shape.
    model = adapt.ModelSpec(
        input={"image": adapt.Image(height=4, width=4, fit=("crop", "pad"))},
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
        input={"image": adapt.Image(channels=3)},
        output=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )
    with pytest.raises(adapt.AdapterResolutionError, match="channel"):
        resolve(env, model)


def test_image_normalize_range_maps_into_declared_bounds():
    env = image_env(2, 2)
    model = adapt.ModelSpec(
        input={
            "image": adapt.Image(
                dtype="float32", normalize=True, normalize_range=(-1.0, 1.0)
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
    # The zero-filled camera surfaces as a non-fatal advisory.
    assert any("blank" in note and "overhead" in note for note in adapter.advisories())
