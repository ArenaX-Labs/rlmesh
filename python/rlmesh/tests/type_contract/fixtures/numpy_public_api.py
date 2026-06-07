from __future__ import annotations

from typing import Any, assert_type

import numpy as np
from rlmesh import Tensor
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
