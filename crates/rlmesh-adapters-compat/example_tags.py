"""Env tags of three VLA benchmarks, as their envs in rlmesh-examples publish them."""

from __future__ import annotations

import rlmesh.adapters as adapt

LIBERO = adapt.EnvTags(
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

METAWORLD = adapt.EnvTags(
    observation={
        "corner_image": adapt.ImageTag(adapt.IMAGE_PRIMARY),
        "gripper_image": adapt.ImageTag(adapt.IMAGE_WRIST),
        "proprio": adapt.Split(
            adapt.Field(adapt.EEF_POS, 3),
            adapt.Field(adapt.EEF_ROT, 4, encoding="quat_xyzw"),
            adapt.Field(adapt.GRIPPER_POS, 1),
            adapt.Field(dim=10),
        ),
        "task": adapt.TextTag(adapt.INSTRUCTION),
    },
    action=adapt.Action(
        adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
        clip=(-1.0, 1.0),
    ),
)

SIMPLER_BRIDGE = adapt.EnvTags(
    observation={
        "rgb": adapt.ImageTag(adapt.IMAGE_PRIMARY),
        "agent": {
            "eef_pos": adapt.StateTag(adapt.EEF_POS),
            "eef_quat": adapt.StateTag(adapt.EEF_ROT, encoding="quat_wxyz"),
            "gripper_width": adapt.StateTag(adapt.GRIPPER_POS, range=(0.0, 0.08)),
        },
        "task_instruction": adapt.TextTag(adapt.INSTRUCTION),
    },
    action=adapt.Action(
        adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
        adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
        clip=(-1.0, 1.0),
    ),
)
