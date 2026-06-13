"""SimplerEnv (Bridge): a second env with deliberately different conventions.

Compared to LIBERO this env has a single camera, nested observation keys
(``agent.eef_pos``), a wxyz quaternion, and a different instruction key.
None of that needs model-side code: the resolver reads it from the
annotations and the spaces.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import rlmesh.adapters as adapt

ANNOTATIONS = adapt.EnvAnnotations(
    observation={
        "rgb": adapt.ImageAnnotation(role=adapt.IMAGE_PRIMARY),
        "agent.eef_pos": adapt.StateAnnotation(role=adapt.EEF_POS),
        "agent.eef_quat": adapt.StateAnnotation(
            role=adapt.EEF_ROT, encoding="quat_wxyz"
        ),
        "agent.gripper_width": adapt.StateAnnotation(
            role=adapt.GRIPPER_POS, range=(0.0, 0.08)
        ),
        "task_instruction": adapt.TextAnnotation(),
    },
    action=adapt.ActionLayout(
        components=(
            adapt.ActionComponent(adapt.ACTION_DELTA_POS, dim=3),
            adapt.ActionComponent(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
            adapt.ActionComponent(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
        ),
        clip=(-1.0, 1.0),
    ),
)

OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "rgb": gym.spaces.Box(0, 255, (480, 640, 3), np.uint8),
        "agent": gym.spaces.Dict(
            {
                "eef_pos": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
                "eef_quat": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
                # Unbounded in the space; the meaningful range is declared on
                # the annotation (StateAnnotation.range) instead, the usual
                # split when a sim reports raw proprio without tight bounds.
                "gripper_width": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
            }
        ),
        "task_instruction": gym.spaces.Text(max_length=256),
    }
)

ACTION_SPACE = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)


def sample_obs() -> dict[str, Any]:
    """Synthetic observation shaped like a real Bridge obs, for dry runs."""
    rng = np.random.default_rng(1)
    quat = rng.normal(size=4).astype(np.float32)
    quat /= np.linalg.norm(quat)
    return {
        "rgb": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
        "agent": {
            "eef_pos": rng.normal(size=3).astype(np.float32),
            "eef_quat": quat,
            "gripper_width": np.array([0.05], dtype=np.float32),
        },
        "task_instruction": "move the spoon to the towel",
    }
