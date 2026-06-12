"""LIBERO: what its observations and actions look like, declared once.

This file is written from the environment's point of view and knows nothing
about any model. In a real deployment the env server would publish this spec
in its contract metadata (``rlmesh.EnvServer`` metadata + ``SPEC.to_metadata()``)
so clients never need a local copy.
"""

from __future__ import annotations

from typing import Any

import rlmesh.adapters as adapt

SPEC = adapt.EnvIOSpec(
    observation=(
        adapt.EnvImage("agentview_image", role=adapt.IMAGE_PRIMARY),
        adapt.EnvImage("robot0_eye_in_hand_image", role=adapt.IMAGE_WRIST),
        adapt.EnvState("robot0_eef_pos", role=adapt.EEF_POS, dim=3),
        adapt.EnvState("robot0_eef_quat", role=adapt.EEF_ROT, encoding="quat_xyzw"),
        adapt.EnvState("robot0_gripper_qpos", role=adapt.GRIPPER_POS, dim=2),
        adapt.EnvText("instruction"),
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
    """Synthetic observation shaped like a real LIBERO obs, for dry runs."""
    import numpy as np

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
