"""X-VLA: a model with very different conventions from SmolVLA.

X-VLA wants 256x256 images, rot6d proprio in a 20-dim state, and it emits a
20-dim EE6D action that envs cannot consume directly. Both its proprio and its
action rotations use ``rot6d_rowmajor``, the row-major flattening of the
matrix's first two columns (``m[:, :2].reshape(6)``) that this checkpoint was
trained on, distinct from the standard column-concatenated ``rot6d``. The
20-dim layout is a unified single/bimanual convention: dims 1-10 are the first
arm, dims 11-20 the second. Rather than hardcoding dims 11-20 as zero padding
(which would bake in a single-arm assumption), the spec declares their real
meaning -- second-arm components, ``optional`` on the state side -- and the
resolver derives the per-env behavior: against a single-arm env the
second-arm proprio resolves to zero fill and the second-arm action dims are
dropped; against a bimanual env declaring the ``_2`` roles, the same spec
consumes and emits them for real.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import rlmesh.adapters as adapt

SPEC = adapt.ModelSpec(
    input={
        "image": adapt.Image(role=adapt.IMAGE_PRIMARY, height=256, width=256),
        "image2": adapt.Image(role=adapt.IMAGE_WRIST, height=256, width=256),
        "state": adapt.Concat(
            adapt.State(adapt.EEF_POS, dim=3),
            adapt.State(adapt.EEF_ROT, encoding="rot6d_rowmajor"),
            adapt.State(adapt.GRIPPER_POS, dim=1),
            adapt.State(adapt.EEF_POS_2, dim=3, optional=True),
            adapt.State(adapt.EEF_ROT_2, encoding="rot6d_rowmajor", optional=True),
            adapt.State(adapt.GRIPPER_POS_2, dim=1, optional=True),
            pad_to=20,
            container="list",
        ),
        "instruction": adapt.Text(role=adapt.INSTRUCTION),
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


def load_predict_fn() -> Callable[[Mapping[str, Any]], Any]:
    """Return the checkpoint's raw predict callable (stubbed; see smolvla.py)."""
    import numpy as np

    def predict(payload: Mapping[str, Any]) -> Any:
        return np.zeros(SPEC.output.dim, dtype=np.float32)

    return predict
