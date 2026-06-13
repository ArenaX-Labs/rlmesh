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
    a1, a2 = r6d[:3], r6d[3:]
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


def resolve(env: Env, model: adapt.ModelSpec, **kwargs: Any) -> adapt.IOAdapter:
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

LIBERO_ACTION = adapt.ActionLayout(
    components=(
        adapt.ActionComponent(adapt.ACTION_DELTA_POS, dim=3),
        adapt.ActionComponent(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
        adapt.ActionComponent(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
    ),
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
    inputs=(
        adapt.ImageInput(
            "observation.images.image",
            role=adapt.IMAGE_PRIMARY,
            height=64,
            width=64,
        ),
        adapt.ImageInput(
            "observation.images.image2",
            role=adapt.IMAGE_WRIST,
            height=64,
            width=64,
        ),
        adapt.StateInput(
            "observation.state",
            components=(
                adapt.StateComponent(adapt.EEF_POS),
                adapt.StateComponent(adapt.EEF_ROT, encoding="axis_angle"),
                adapt.StateComponent(adapt.GRIPPER_POS),
            ),
            container="list",
        ),
        adapt.TextInput("instruction"),
    ),
    action=adapt.ActionLayout(
        components=(
            adapt.ActionComponent(adapt.ACTION_DELTA_POS, dim=3),
            adapt.ActionComponent(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
            adapt.ActionComponent(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
        ),
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


# ---------------------------------------------------------------------------
# OpenVLA
# ---------------------------------------------------------------------------

OPENVLA = adapt.ModelSpec(
    inputs=(
        adapt.ImageInput("image", role=adapt.IMAGE_PRIMARY, height=64, width=64),
        adapt.TextInput("instruction"),
    ),
    action=adapt.ActionLayout(
        components=(
            adapt.ActionComponent(adapt.ACTION_DELTA_POS, dim=3),
            adapt.ActionComponent(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
            adapt.ActionComponent(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
        ),
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
    inputs=(
        adapt.ImageInput("image", role=adapt.IMAGE_PRIMARY, height=64, width=64),
        adapt.ImageInput("image2", role=adapt.IMAGE_WRIST, height=64, width=64),
        adapt.StateInput(
            "state",
            components=(
                adapt.StateComponent(adapt.EEF_POS, dim=3),
                adapt.StateComponent(adapt.EEF_ROT, encoding="rot6d"),
                adapt.StateComponent(adapt.GRIPPER_POS, dim=1),
                adapt.StateComponent(adapt.EEF_POS_2, dim=3, optional=True),
                adapt.StateComponent(adapt.EEF_ROT_2, encoding="rot6d", optional=True),
                adapt.StateComponent(adapt.GRIPPER_POS_2, dim=1, optional=True),
            ),
            pad_to=20,
            container="list",
        ),
        adapt.TextInput("instruction"),
    ),
    action=adapt.ActionLayout(
        components=(
            adapt.ActionComponent(adapt.ACTION_DELTA_POS, dim=3),
            adapt.ActionComponent(adapt.ACTION_DELTA_ROT, dim=6, encoding="rot6d"),
            adapt.ActionComponent(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
            adapt.ActionComponent(adapt.ACTION_DELTA_POS_2, dim=3),
            adapt.ActionComponent(adapt.ACTION_DELTA_ROT_2, dim=6, encoding="rot6d"),
            adapt.ActionComponent(adapt.ACTION_GRIPPER_2, dim=1, range=(-1.0, 1.0)),
        ),
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
        action=adapt.ActionLayout(
            components=(
                adapt.ActionComponent(adapt.ACTION_DELTA_POS, dim=3),
                adapt.ActionComponent(
                    adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"
                ),
                adapt.ActionComponent(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
                adapt.ActionComponent(adapt.ACTION_DELTA_POS_2, dim=3),
                adapt.ActionComponent(
                    adapt.ACTION_DELTA_ROT_2, dim=3, encoding="axis_angle"
                ),
                adapt.ActionComponent(adapt.ACTION_GRIPPER_2, dim=1, range=(-1.0, 1.0)),
            ),
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
        inputs=(
            adapt.StateInput(
                "state",
                components=(adapt.StateComponent("proprio/extra", optional=True),),
            ),
        ),
        action=SMOLVLA.action,
    )
    with pytest.raises(adapt.AdapterResolutionError, match="zero fill"):
        resolve(LIBERO_ENV, spec)


def test_describe_mentions_zero_fill_for_absent_optional_roles():
    text = resolve(LIBERO_ENV, XVLA).describe()
    assert "zeros(3)" in text and "zeros(6)" in text and "zeros(1)" in text


def test_role_constants_match_rust_crate():
    """Roles are single-sourced from the crate; this catches a role added
    to ``v1/roles/*.rs`` but not exposed through the binding's table."""
    import re
    from pathlib import Path

    roles_dir = (
        Path(__file__).resolve().parents[4]
        / "crates"
        / "rlmesh-adapters"
        / "src"
        / "v1"
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
    inputs=(
        adapt.ImageInput(
            "video.image",
            role=adapt.IMAGE_PRIMARY,
            height=64,
            width=64,
            lead_dims=2,
            upside_down=True,
        ),
        adapt.StateInput(
            "state.x",
            components=(adapt.StateComponent(adapt.EEF_POS, index=0),),
            reshape=(1, 1, 1),
        ),
        adapt.StateInput(
            "state.roll",
            components=(
                adapt.StateComponent(adapt.EEF_ROT, encoding="axis_angle", index=0),
            ),
            reshape=(1, 1, 1),
        ),
        adapt.StateInput(
            "state.gripper",
            components=(adapt.StateComponent(adapt.GRIPPER_POS, index=0),),
            reshape=(1, 1, 1),
        ),
        adapt.TextInput(
            "tag.human.action.task_description",
            container="list",
            default="",
        ),
    ),
    action=adapt.ActionLayout(
        components=(
            adapt.ActionComponent(adapt.ACTION_DELTA_POS, dim=3),
            adapt.ActionComponent(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
            adapt.ActionComponent(
                adapt.ACTION_GRIPPER, dim=1, range=(0.0, 1.0), binary=True
            ),
        ),
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
        inputs=(
            adapt.ImageInput(
                "pixels",
                role=adapt.IMAGE_PRIMARY,
                height=16,
                width=16,
                layout="chw",
                dtype="float32",
                normalize=True,
            ),
        ),
        action=SMOLVLA.action,
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
            inputs=(adapt.ImageInput("image", height=height, width=width),),
            action=SMOLVLA.action,
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
        inputs=(adapt.ImageInput("image"),),
        action=SMOLVLA.action,
    )
    payload = resolve(env, spec).transform_obs({"rgb": make_png(pixels)})
    np.testing.assert_array_equal(payload["image"], pixels)


def test_undecodable_image_bytes_is_an_error():
    env = image_env(2, 2)
    spec = adapt.ModelSpec(
        inputs=(adapt.ImageInput("image"),),
        action=SMOLVLA.action,
    )
    with pytest.raises(ValueError, match="could not decode image bytes"):
        resolve(env, spec).transform_obs({"rgb": b"not an image"})


def test_bilinear_resize_preserves_constant_images():
    env = image_env(10, 12)
    spec = adapt.ModelSpec(
        inputs=(adapt.ImageInput("image", height=4, width=5, resample="bilinear"),),
        action=SMOLVLA.action,
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
        inputs=(adapt.ImageInput("image", role=adapt.IMAGE_PRIMARY),),
        action=SMOLVLA.action,
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
    inputs=(
        adapt.StateInput("state", components=(adapt.StateComponent(adapt.EEF_POS),)),
    ),
    action=SMOLVLA.action,
)


def test_nested_observation_keys():
    env = single_state_env(
        "agent.eef_pos",
        gym.spaces.Dict({"agent": gym.spaces.Dict({"eef_pos": box(3)})}),
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
        inputs=(
            adapt.InlineCustomInput(
                "engineered", lambda obs: float(obs["robot0_eef_pos"][0])
            ),
        ),
        action=SMOLVLA.action,
    )
    adapter = resolve(LIBERO_ENV, spec)
    obs = make_obs()
    assert adapter.transform_obs(obs)["engineered"] == pytest.approx(
        float(np.asarray(obs["robot0_eef_pos"])[0])
    )


def test_custom_entrypoint_requires_trust():
    spec = adapt.ModelSpec(
        inputs=(adapt.EntrypointCustomInput("count", "builtins:len"),),
        action=SMOLVLA.action,
    )
    with pytest.raises(adapt.AdapterResolutionError, match="trust_entrypoints"):
        resolve(LIBERO_ENV, spec)

    adapter = resolve(LIBERO_ENV, spec, trust_entrypoints=True)
    assert adapter.transform_obs(make_obs())["count"] == len(make_obs())


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
    metadata = {"max_batch": 8, **SMOLVLA.to_metadata()}
    assert adapt.ModelSpec.from_metadata(metadata) == SMOLVLA
    assert adapt.ModelSpec.from_metadata({"max_batch": 8}) is None


def test_metadata_keys_are_side_specific():
    tags = LIBERO_ENV.tags
    assert adapt.ENV_METADATA_KEY != adapt.MODEL_METADATA_KEY
    merged = {**tags.to_metadata(), **SMOLVLA.to_metadata()}
    assert adapt.EnvTags.from_metadata(merged) == tags
    assert adapt.ModelSpec.from_metadata(merged) == SMOLVLA
    assert adapt.EnvTags.from_metadata(SMOLVLA.to_metadata()) is None
    assert adapt.ModelSpec.from_metadata(tags.to_metadata()) is None


def test_custom_callable_spec_is_not_publishable():
    spec = adapt.ModelSpec(
        inputs=(adapt.InlineCustomInput("x", lambda obs: 0),),
        action=SMOLVLA.action,
    )
    with pytest.raises(ValueError, match="cannot be serialized"):
        spec.to_metadata()


def test_custom_callable_is_not_serializable():
    spec = adapt.ModelSpec(
        inputs=(adapt.InlineCustomInput("x", lambda obs: 0),),
        action=SMOLVLA.action,
    )
    with pytest.raises(ValueError, match="cannot be serialized"):
        spec.to_dict()


def test_custom_entrypoint_is_serializable():
    spec = adapt.ModelSpec(
        inputs=(adapt.EntrypointCustomInput("x", "builtins:len"),),
        action=SMOLVLA.action,
    )
    assert adapt.ModelSpec.from_json(spec.to_json()) == spec


# ---------------------------------------------------------------------------
# Resolution errors
# ---------------------------------------------------------------------------


def test_missing_state_role_is_an_error():
    spec = adapt.ModelSpec(
        inputs=(
            adapt.StateInput(
                "state", components=(adapt.StateComponent(adapt.JOINT_VEL),)
            ),
        ),
        action=SMOLVLA.action,
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
        inputs=(
            adapt.StateInput(
                "state",
                components=(adapt.StateComponent(adapt.EEF_ROT, encoding="rot6d"),),
            ),
        ),
        action=SMOLVLA.action,
    )
    with pytest.raises(adapt.AdapterResolutionError, match="quat_xyzw"):
        resolve(env, spec)


def test_unknown_rotation_encoding_pairing_is_an_error():
    spec = adapt.ModelSpec(
        inputs=(
            adapt.StateInput(
                "state",
                components=(
                    adapt.StateComponent(adapt.GRIPPER_POS, encoding="axis_angle"),
                ),
            ),
        ),
        action=SMOLVLA.action,
    )
    with pytest.raises(adapt.AdapterResolutionError, match="encoding"):
        resolve(LIBERO_ENV, spec)


def test_missing_action_role_is_an_error():
    spec = adapt.ModelSpec(
        inputs=(adapt.ImageInput("image"),),
        action=adapt.ActionLayout(
            components=(adapt.ActionComponent(adapt.ACTION_DELTA_POS, dim=3),)
        ),
    )
    with pytest.raises(adapt.AdapterResolutionError, match="action/delta_eef_rot"):
        resolve(LIBERO_ENV, spec)


def test_action_dim_mismatch_is_an_error():
    spec = adapt.ModelSpec(
        inputs=(adapt.ImageInput("image"),),
        action=adapt.ActionLayout(
            components=(
                adapt.ActionComponent(adapt.ACTION_DELTA_POS, dim=2),
                adapt.ActionComponent(
                    adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"
                ),
                adapt.ActionComponent(adapt.ACTION_GRIPPER, dim=1),
            )
        ),
    )
    with pytest.raises(adapt.AdapterResolutionError, match="dims"):
        resolve(LIBERO_ENV, spec)


def test_wrong_model_action_length_is_an_error():
    adapter = resolve(LIBERO_ENV, SMOLVLA)
    with pytest.raises(ValueError, match="7-dim"):
        adapter.transform_action(np.zeros(5, dtype=np.float32))


def test_describe_mentions_each_model_key():
    text = resolve(LIBERO_ENV, XVLA).describe()
    assert '"state"' in text
    assert "quat_xyzw->rot6d" in text
    assert "rot6d->axis_angle" in text


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


def test_model_spec_run_requires_an_env_object() -> None:
    from rlmesh.numpy import Model

    def predict(payload: dict[str, Any]) -> Any:
        return np.zeros(SMOLVLA.action.dim, dtype=np.float32)

    model = Model(predict, spec=SMOLVLA)
    # A bare address carries no contract to resolve the adapter from.
    with pytest.raises(TypeError, match="env_contract"):
        model.run("127.0.0.1:5555", max_episodes=1)


def test_negative_u32_fields_are_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="width must be non-negative"):
        adapt.ImageInput("image", width=-1)
    with pytest.raises(ValueError, match="lead_dims must be non-negative"):
        adapt.ImageInput("image", lead_dims=-2)
    with pytest.raises(ValueError, match="dim must be non-negative"):
        adapt.StateComponent(adapt.EEF_POS, dim=-1)
    with pytest.raises(ValueError, match="index must be non-negative"):
        adapt.StateComponent(adapt.EEF_POS, index=-1)
    with pytest.raises(ValueError, match="pad_to must be non-negative"):
        adapt.StateInput(
            "s", components=(adapt.StateComponent(adapt.EEF_POS),), pad_to=-1
        )
    with pytest.raises(ValueError, match="dim must be non-negative"):
        adapt.ActionComponent(adapt.ACTION_GRIPPER, dim=-1)


def test_bridge_encodes_numpy_bool_scalar_as_number() -> None:
    from rlmesh.adapters.helpers.bridge import encode_value

    assert encode_value(np.bool_(True)) == ("n", 1.0)
    assert encode_value(np.bool_(False)) == ("n", 0.0)


# ---------------------------------------------------------------------------
# Spec ergonomics: size=, single-component StateInput, eager validation
# ---------------------------------------------------------------------------


def test_image_input_size_shorthand() -> None:
    assert adapt.ImageInput("img", size=224) == adapt.ImageInput(
        "img", height=224, width=224
    )
    with pytest.raises(ValueError, match="size=, or height"):
        adapt.ImageInput("img", size=224, height=10)


def test_state_input_single_component_shorthand() -> None:
    assert adapt.StateInput(
        "s", role=adapt.EEF_POS, encoding="axis_angle"
    ) == adapt.StateInput(
        "s", components=(adapt.StateComponent(adapt.EEF_POS, encoding="axis_angle"),)
    )
    with pytest.raises(ValueError, match="components=, or a single"):
        adapt.StateInput(
            "s", components=(adapt.StateComponent(adapt.EEF_POS),), role=adapt.EEF_ROT
        )
    with pytest.raises(ValueError, match="needs components"):
        adapt.StateInput("s")


def test_state_input_sugar_resolves_like_explicit() -> None:
    spec = adapt.ModelSpec(
        inputs=(adapt.StateInput("state", role=adapt.EEF_POS),),
        action=SMOLVLA.action,
    )
    adapter = resolve(LIBERO_ENV, spec)
    np.testing.assert_allclose(
        adapter.transform_obs(make_obs())["state"],
        np.asarray(make_obs()["robot0_eef_pos"], dtype=np.float32),
    )


def test_model_spec_rejects_duplicate_input_keys() -> None:
    with pytest.raises(ValueError, match="duplicate input keys"):
        adapt.ModelSpec(
            inputs=(adapt.ImageInput("dup"), adapt.TextInput("dup")),
            action=SMOLVLA.action,
        )


# ---------------------------------------------------------------------------
# Frame stacking (host-side, stateful)
# ---------------------------------------------------------------------------


def test_image_frame_stacking_buffers_and_pads() -> None:
    env = image_env(4, 4)
    spec = adapt.ModelSpec(
        inputs=(adapt.ImageInput("img", role=adapt.IMAGE_PRIMARY, stack=3),),
        action=SMOLVLA.action,
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


def test_image_input_stack_round_trips_and_omits_default() -> None:
    from rlmesh.adapters.specs.model_serialization import model_input_to_dict

    spec = adapt.ModelSpec(
        inputs=(adapt.ImageInput("img", stack=4),), action=SMOLVLA.action
    )
    assert adapt.ModelSpec.from_json(spec.to_json()) == spec
    assert "stack" not in model_input_to_dict(
        adapt.ImageInput("img")
    )  # default omitted
    assert model_input_to_dict(adapt.ImageInput("img", stack=4))["stack"] == 4
    with pytest.raises(ValueError, match="stack must be >= 1"):
        adapt.ImageInput("img", stack=0)


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
        inputs=(adapt.StateInput("rot", role=adapt.EEF_ROT, encoding="axis_angle"),),
        action=SMOLVLA.action,
    )
    adapter = resolve(env, spec)
    # Pure yaw of 90 degrees -> axis-angle about z.
    out = adapter.transform_obs(
        {"rpy": np.array([0.0, 0.0, np.pi / 2], dtype=np.float32)}
    )["rot"]
    np.testing.assert_allclose(out, [0.0, 0.0, np.pi / 2], atol=1e-4)
