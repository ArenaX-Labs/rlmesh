"""SimplerEnv (Bridge): a second env with deliberately different conventions.

Compared to LIBERO this env has a single camera, nested observation keys
(``agent.eef_pos``), a wxyz quaternion, and a different instruction key.
None of that needs model-side code: the resolver reads it from this spec.
"""

from __future__ import annotations

from typing import Any

import rlmesh.adapters as adapt

SPEC = adapt.EnvIOSpec(
    observation=(
        adapt.EnvImage("rgb", role=adapt.IMAGE_PRIMARY),
        adapt.EnvState("agent.eef_pos", role=adapt.EEF_POS, dim=3),
        adapt.EnvState("agent.eef_quat", role=adapt.EEF_ROT, encoding="quat_wxyz"),
        adapt.EnvState(
            "agent.gripper_width", role=adapt.GRIPPER_POS, dim=1, range=(0.0, 0.08)
        ),
        adapt.EnvText("task_instruction"),
    ),
    action=adapt.ActionLayout(
        components=(
            adapt.ActionComponent(adapt.ACTION_DELTA_POS, dim=3),
            adapt.ActionComponent(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
            adapt.ActionComponent(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
        ),
        clip=(-1.0, 1.0),
    ),
)


def sample_obs() -> dict[str, Any]:
    """Synthetic observation shaped like a real Bridge obs, for dry runs."""
    import numpy as np

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
