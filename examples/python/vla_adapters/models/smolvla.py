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
    input={
        "observation.images.image": adapt.Image(
            role=adapt.IMAGE_PRIMARY,
            height=224,
            width=224,
        ),
        "observation.images.image2": adapt.Image(
            role=adapt.IMAGE_WRIST,
            height=224,
            width=224,
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
        return np.zeros(SPEC.output.dim, dtype=np.float32)

    return predict
