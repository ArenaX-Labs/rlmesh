from __future__ import annotations

from typing import Any

from rlmesh_system_fixtures.registry import model_fixture


@model_fixture("image_grid.torch_action")
def image_grid_torch_action(observation: Any) -> object:
    _ = observation
    import torch

    return torch.tensor(0, dtype=torch.int64)


@model_fixture("image_grid.numpy_action")
def image_grid_numpy_action(observation: Any) -> object:
    _ = observation
    import numpy as np

    return np.array(0, dtype=np.int64)
