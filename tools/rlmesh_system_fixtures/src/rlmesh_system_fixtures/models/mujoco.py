from __future__ import annotations

from rlmesh_system_fixtures.registry import model_fixture


@model_fixture("mujoco.halfcheetah_zero_numpy")
def halfcheetah_zero_numpy(observation: object) -> object:
    _ = observation
    import numpy as np

    return np.zeros((6,), dtype=np.float32)
