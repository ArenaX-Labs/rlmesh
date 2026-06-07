from __future__ import annotations

from rlmesh_system_fixtures.registry import model_fixture


@model_fixture("gymnasium.pendulum_zero_numpy")
def pendulum_zero_numpy(observation: object) -> object:
    _ = observation
    import numpy as np

    return np.zeros((1,), dtype=np.float32)
