from __future__ import annotations

from typing import assert_type

import torch
from rlmesh import Tensor
from rlmesh import torch as rlmesh_torch

tensor = Tensor(bytes(range(16)), [4], "int32")
view = rlmesh_torch.as_tensor(tensor)
assert_type(view, torch.Tensor)

source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
restored = rlmesh_torch.as_tensor(rlmesh_torch.from_tensor(source))
assert_type(restored, torch.Tensor)

scalar = rlmesh_torch.from_tensor(torch.tensor(1, dtype=torch.int32))
scalar_view = rlmesh_torch.as_tensor(scalar)
assert_type(scalar_view, torch.Tensor)
