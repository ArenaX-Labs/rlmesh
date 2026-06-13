from __future__ import annotations

from collections.abc import Mapping

from rlmesh import spaces
from rlmesh.types import Value
from typing_extensions import assert_type

box = spaces.Box(-1.0, 1.0, shape=[2], dtype="float32")
assert_type(box.kind, str)
assert_type(box.shape, list[int])
assert_type(box.dtype, str)
assert_type(box.spec, spaces.SpaceSpec)
assert_type(box.sample(), Value)
assert_type(box.contains(box.sample()), bool)

discrete = spaces.Discrete(3)
assert_type(discrete.n, int)
assert_type(discrete.start, int)

multi_binary = spaces.MultiBinary([2, 2])
assert_type(multi_binary.dims, list[int] | None)

multi_discrete = spaces.MultiDiscrete([2, 3])
assert_type(multi_discrete.nvec, list[int] | None)

text = spaces.Text(8, min_length=1)
assert_type(text.max_length, int)
assert_type(text.charset, str)

mapping = spaces.Dict({"box": box, "action": discrete})
assert_type(mapping.spaces, Mapping[str, spaces.Space[Value]])

tuple_space = spaces.Tuple([box, discrete])
assert_type(tuple_space.spaces, tuple[spaces.Space[Value], ...])

roundtripped = spaces.space_from_spec(box.spec)
assert_type(roundtripped, spaces.Space[Value])
