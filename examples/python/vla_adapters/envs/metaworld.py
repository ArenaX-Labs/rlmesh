"""Metaworld: a flat proprioception leaf, split by a Split.

Unlike LIBERO and SimplerEnv, Metaworld's proprioception is a single flat
``Box`` vector rather than one key per quantity: fixed index ranges carry
distinct meaning, and a task-specific tail holds object and goal positions the
arm policy reads from pixels, not proprio. The env tags that leaf with a
``Split`` (the observation-side mirror of ``Action``): the vector is split into
role fields in order, and the indices the model does not consume are skipped
with a role-less field.

The layout here is representative rather than byte-exact: it carries the
end-effector pose and gripper that the VLA specs expect, so the *same* model
spec that resolves against the Dict envs resolves against this one with no
change. The fixed indices live on the env side, where they belong.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import rlmesh.adapters as adapt

TAGS = adapt.EnvTags(
    observation={
        "corner_image": adapt.ImageTag(adapt.IMAGE_PRIMARY),
        "gripper_image": adapt.ImageTag(adapt.IMAGE_WRIST),
        # One flat leaf, split by index range. Field widths sum to the leaf
        # width (3 + 4 + 1 + 10 = 18); offsets are implied by order.
        "proprio": adapt.Split(
            adapt.Field(adapt.EEF_POS, 3),
            adapt.Field(adapt.EEF_ROT, 4, encoding="quat_xyzw"),
            adapt.Field(adapt.GRIPPER_POS, 1),
            adapt.Field(dim=10),  # object + goal positions: not consumed here
        ),
        "task": adapt.TextTag(adapt.INSTRUCTION),
    },
    # Metaworld is position-controlled: the model's rotation output has no env
    # counterpart and is dropped, the same way unused obs features are.
    action=adapt.Action(
        adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
        clip=(-1.0, 1.0),
    ),
)

OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "corner_image": gym.spaces.Box(0, 255, (256, 256, 3), np.uint8),
        "gripper_image": gym.spaces.Box(0, 255, (256, 256, 3), np.uint8),
        "proprio": gym.spaces.Box(-np.inf, np.inf, (18,), np.float32),
        "task": gym.spaces.Text(max_length=256),
    }
)

ACTION_SPACE = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)


def sample_obs() -> dict[str, Any]:
    """Synthetic observation; ``proprio`` is one flat width-18 vector."""
    rng = np.random.default_rng(0)
    quat = rng.normal(size=4).astype(np.float32)
    quat /= np.linalg.norm(quat)
    proprio = np.empty(18, dtype=np.float32)
    proprio[0:3] = rng.normal(size=3)  # eef_pos
    proprio[3:7] = quat  # eef_quat
    proprio[7] = 0.04  # gripper
    proprio[8:18] = rng.normal(size=10)  # object + goal (skipped by the layout)
    return {
        "corner_image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "gripper_image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "proprio": proprio,
        "task": "pick up the block and place it on the shelf",
    }
