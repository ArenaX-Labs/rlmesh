"""GeoVLA: a model whose 6D rotation packing RLMesh does not ship as a built-in.

This checkpoint was trained with a rotation library that orders the two basis
vectors second-column-first (``[a2 | a1]``) rather than the standard
``[a1 | a2]``. Rather than reach for a stateful ``AdapterBase`` wrapper, it
declares a :class:`~rlmesh.adapters.CustomEncoding` on the known ``rot6d``
base and supplies the repacking. ``resolve`` lowers the field to ``rot6d`` for
the native core -- so role matching, range mapping, and the env<->rot6d
conversion are unchanged -- and applies the repacking host-side at the field
boundary (base->custom on the way in, custom->base on the way out).

The same ``ROT6D_COLSWAP`` constant is referenced from both the proprio input
and the action actuator: define the encoding once, use it on both sides. The
rotation gets its own single-part ``State`` input because the offset of a
custom field interior to a multi-part ``Concat`` is env-dependent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import rlmesh.adapters as adapt


def _swap_halves(vector: Any) -> Any:
    """Swap the two 3-vectors of a 6D rotation; its own inverse."""
    import numpy as np

    flat = np.asarray(vector)
    return np.concatenate([flat[3:], flat[:3]])


ROT6D_COLSWAP = adapt.CustomEncoding(
    base="rot6d",
    from_base=_swap_halves,
    to_base=_swap_halves,
    name="rot6d_colswap",
)

SPEC = adapt.ModelSpec(
    input={
        "image": adapt.Image(role=adapt.IMAGE_PRIMARY, size=224),
        "eef_rot": adapt.State(adapt.EEF_ROT, encoding=ROT6D_COLSWAP),
        "proprio": adapt.Concat(
            adapt.EEF_POS,
            adapt.GRIPPER_POS,
            container="list",
        ),
        "instruction": adapt.Text(),
    },
    output=adapt.Action(
        adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
        adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=6, encoding=ROT6D_COLSWAP),
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
    ),
)


def load_predict_fn() -> Callable[[Mapping[str, Any]], Any]:
    """Return the checkpoint's raw predict callable (stubbed: zero action).

    A real integration loads the policy here; ``payload`` already arrives in
    the format declared by ``SPEC`` (with ``eef_rot`` in the model's own 6D
    packing). The stub returns a zero action so the example runs with no GPU.
    """
    import numpy as np

    def predict(payload: Mapping[str, Any]) -> Any:
        return np.zeros(SPEC.output.dim, dtype=np.float32)

    return predict
