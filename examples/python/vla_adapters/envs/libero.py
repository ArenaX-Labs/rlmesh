"""LIBERO: what its observations and actions look like, declared once.

This file is written from the environment's point of view and knows nothing
about any model. In a real deployment the env server would publish these
tags in its contract metadata (``rlmesh.EnvServer(env, tags=...)``
or ``rlmesh.adapters.tag(env, ...)``) so clients never need a local copy.

The env *tags* its spaces: it names the semantic role of each
observation entry and the action components, plus the few facts the spaces
cannot carry (image layout, rotation encoding, ranges). Widths, dtypes and
keys come from the gymnasium spaces.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import rlmesh.adapters as adapt

TAGS = adapt.EnvTags(
    observation={
        "agentview_image": adapt.ImageTag(adapt.IMAGE_PRIMARY),
        "robot0_eye_in_hand_image": adapt.ImageTag(adapt.IMAGE_WRIST),
        "robot0_eef_pos": adapt.StateTag(adapt.EEF_POS),
        "robot0_eef_quat": adapt.StateTag(adapt.EEF_ROT, encoding="quat_xyzw"),
        "robot0_gripper_qpos": adapt.StateTag(adapt.GRIPPER_POS),
        "instruction": adapt.TextTag(adapt.INSTRUCTION),
    },
    action=adapt.Action(
        adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
        adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
        clip=(-1.0, 1.0),
    ),
)

OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "agentview_image": gym.spaces.Box(0, 255, (256, 256, 3), np.uint8),
        "robot0_eye_in_hand_image": gym.spaces.Box(0, 255, (256, 256, 3), np.uint8),
        "robot0_eef_pos": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
        "robot0_eef_quat": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
        "robot0_gripper_qpos": gym.spaces.Box(-np.inf, np.inf, (2,), np.float32),
        "instruction": gym.spaces.Text(max_length=256),
    }
)

ACTION_SPACE = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)


def sample_obs() -> dict[str, Any]:
    """Synthetic observation shaped like a real LIBERO obs, for dry runs."""
    rng = np.random.default_rng(0)
    quat = rng.normal(size=4).astype(np.float32)
    quat /= np.linalg.norm(quat)
    return {
        "agentview_image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "robot0_eef_pos": rng.normal(size=3).astype(np.float32),
        "robot0_eef_quat": quat,
        "robot0_gripper_qpos": np.array([0.03, -0.03], dtype=np.float32),
        "instruction": "put the bowl on the plate",
    }
