from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
from rlmesh import Tensor, spaces
from rlmesh import jax as rlmesh_jax
from rlmesh.types import PrimitiveValue
from typing_extensions import assert_type

tensor = Tensor(bytes(range(16)), [4], "int32")
view = rlmesh_jax.asarray(tensor)
assert_type(view, jax.Array)

array = jnp.zeros((2, 3), dtype=jnp.float32)
value: rlmesh_jax.JaxValue = {"observation": array}

tensor_or_scalar: Tensor | PrimitiveValue = rlmesh_jax.from_array(array)
if isinstance(tensor_or_scalar, Tensor):
    restored = rlmesh_jax.asarray(tensor_or_scalar)
    assert_type(restored, jax.Array)


def predict(observation: rlmesh_jax.JaxValue) -> rlmesh_jax.JaxValue:
    return observation


model = rlmesh_jax.Model(predict)
assert_type(model, rlmesh_jax.Model)

space = rlmesh_jax.space_from_spec(spaces.Box(-1.0, 1.0, shape=[2]).spec)
assert_type(space.sample(), rlmesh_jax.JaxValue)

remote = cast(rlmesh_jax.RemoteEnv, object())
assert_type(remote.address, str)
action = remote.action_space.sample()
assert_type(action, rlmesh_jax.JaxValue)
assert_type(remote.action_space.contains(action), bool)
