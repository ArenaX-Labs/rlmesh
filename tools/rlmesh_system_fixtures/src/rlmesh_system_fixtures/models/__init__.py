from __future__ import annotations

from rlmesh_system_fixtures.models.discrete import discrete_one, discrete_zero
from rlmesh_system_fixtures.models.gymnasium import pendulum_zero_numpy
from rlmesh_system_fixtures.models.image_grid import (
    image_grid_numpy_action,
    image_grid_torch_action,
)
from rlmesh_system_fixtures.models.mujoco import halfcheetah_zero_numpy

__all__ = [
    "discrete_one",
    "discrete_zero",
    "halfcheetah_zero_numpy",
    "image_grid_numpy_action",
    "image_grid_torch_action",
    "pendulum_zero_numpy",
]
