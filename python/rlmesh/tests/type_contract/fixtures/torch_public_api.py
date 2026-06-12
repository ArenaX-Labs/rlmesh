from __future__ import annotations

from typing import cast

import torch
from rlmesh import Tensor, spaces
from rlmesh import torch as rlmesh_torch
from typing_extensions import assert_type

tensor = Tensor(bytes(range(16)), [4], "int32")
view = rlmesh_torch.as_tensor(tensor)
assert_type(view, torch.Tensor)

source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
restored = rlmesh_torch.as_tensor(rlmesh_torch.from_tensor(source))
assert_type(restored, torch.Tensor)

scalar = rlmesh_torch.from_tensor(torch.tensor(1, dtype=torch.int32))
scalar_view = rlmesh_torch.as_tensor(scalar)
assert_type(scalar_view, torch.Tensor)

space = rlmesh_torch.space_from_spec(spaces.Box(-1.0, 1.0, shape=[2]).spec)
assert_type(space.sample(), rlmesh_torch.TorchValue)

remote = cast(rlmesh_torch.RemoteEnv, object())
assert_type(remote.address, str)
action = remote.action_space.sample()
assert_type(action, rlmesh_torch.TorchValue)
assert_type(remote.action_space.contains(action), bool)
