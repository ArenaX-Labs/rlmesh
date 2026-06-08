from __future__ import annotations

from typing import Any, assert_type, cast

import numpy as np
from rlmesh import Tensor, spaces
from rlmesh import numpy as rlmesh_numpy
from rlmesh.types import PrimitiveValue

array = np.zeros((2, 3), dtype=np.float32)
value: rlmesh_numpy.NumpyValue = {"observation": array}

tensor_or_scalar: Tensor | PrimitiveValue = rlmesh_numpy.from_array(array)
if isinstance(tensor_or_scalar, Tensor):
    restored = rlmesh_numpy.asarray(tensor_or_scalar)
    assert_type(restored, np.ndarray[Any, Any])


def predict(observation: rlmesh_numpy.NumpyValue) -> rlmesh_numpy.NumpyValue:
    return observation


model = rlmesh_numpy.Model(predict)
assert_type(model, rlmesh_numpy.Model)

space = rlmesh_numpy.space_from_spec(spaces.Box(-1.0, 1.0, shape=[2]).spec)
assert_type(space.sample(), rlmesh_numpy.NumpyValue)

remote = cast(rlmesh_numpy.RemoteEnv, object())
assert_type(remote.address, str)
action = remote.action_space.sample()
assert_type(action, rlmesh_numpy.NumpyValue)
assert_type(remote.action_space.contains(action), bool)
