"""Complete overwrite for the pairings below.

Nothing below calls ``resolve()`` or reads a tag. The math is
stubbed; the point is the shape of the mechanism: when a single pairing
needs logic no spec can express, drop in a from-scratch
:class:`rlmesh.adapters.AdapterBase` and the harness uses it verbatim.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import rlmesh.adapters as adapt


class XVLABridgeAdapter(adapt.AdapterBase[Any]):
    """From-scratch stateful adapter for X-VLA on the Bridge WidowX."""

    _WORKSPACE_ORIGIN = (0.30, 0.0, 0.10)

    def __init__(self) -> None:
        self._step = 0

    def transform_obs(self, raw_obs: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np

        rgb = np.asarray(raw_obs["rgb"])
        crop = rgb[112:368, 192:448]  # calibrated 256x256 center crop
        agent = raw_obs["agent"]
        pos = np.asarray(agent["eef_pos"], np.float32) - np.asarray(
            self._WORKSPACE_ORIGIN, np.float32
        )
        quat = np.asarray(agent["eef_quat"], np.float32)
        width = float(agent["gripper_width"][0])
        stroke = 1.0 / (1.0 + np.exp(-80.0 * (width - 0.04)))
        phase = min(self._step / 200.0, 1.0)
        self._step += 1
        state = np.zeros(20, dtype=np.float32)
        state[:3] = pos
        state[3:7] = quat
        state[7] = stroke
        state[8] = phase
        return {
            "image": crop,
            "image2": crop,
            "state": state.tolist(),
            "instruction": str(raw_obs["task_instruction"]),
        }

    def transform_action(self, raw_action: object) -> Any:
        import numpy as np

        out = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        return np.clip(np.concatenate([out[:3] * 0.05, out[3:6], out[9:10]]), -1.0, 1.0)

    def reset(self) -> None:
        """Forget episode state (call between episodes)."""
        self._step = 0

    def describe(self) -> str:
        return (
            "bespoke xvla x simpler-bridge adapter "
            "(workspace-relative state, episode phase, IK-stub action)"
        )
