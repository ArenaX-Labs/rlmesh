"""SmolVLA: what this checkpoint ingests and emits, declared once.

This file is written from the model's point of view and knows nothing about
any environment. A spec describes one checkpoint's interface; a SmolVLA
fine-tune with different cameras or state layout would get its own module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import rlmesh.adapters as adapt

SPEC = adapt.ModelSpec(
    inputs=(
        adapt.ImageInput(
            "observation.images.image",
            role=adapt.IMAGE_PRIMARY,
            height=224,
            width=224,
        ),
        adapt.ImageInput(
            "observation.images.image2",
            role=adapt.IMAGE_WRIST,
            height=224,
            width=224,
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
        adapt.ActionComponent(adapt.ACTION_DELTA_POS, dim=3),
        adapt.ActionComponent(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
        adapt.ActionComponent(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
    ),
)


def load_predict_fn() -> Callable[[Mapping[str, Any]], Any]:
    """Return the checkpoint's raw predict callable.

    A real integration loads the policy here, e.g.::

        from lerobot.policies.smolvla import SmolVLAPolicy

        policy = SmolVLAPolicy.from_pretrained("HuggingFaceVLA/smolvla_libero")


        def predict(payload):
            return policy.select_action(payload)

    Note there is no env-specific code: ``payload`` already arrives in the
    format declared by ``SPEC``. The stub below returns a zero action so the
    example runs without GPU dependencies.
    """
    import numpy as np

    def predict(payload: Mapping[str, Any]) -> Any:
        return np.zeros(SPEC.action.dim, dtype=np.float32)

    return predict
