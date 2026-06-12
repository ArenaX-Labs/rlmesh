from __future__ import annotations

import importlib
import string
from collections.abc import Mapping
from typing import Any, cast

from ..specs import SpaceSpec
from ..types import Value
from ._base import Space
from ._registry import space_from_spec
from ._utils import space_shape, spec_details
from .box import Box
from .dict import Dict
from .discrete import Discrete
from .multi_binary import MultiBinary
from .multi_discrete import MultiDiscrete
from .text import Text
from .tuple import Tuple

_GYMNASIUM_DEFAULT_TEXT_CHARSET = frozenset(string.ascii_letters + string.digits)


def from_gymnasium_space(space: object) -> Space[Value]:
    """Convert a Gymnasium space into an RLMesh space wrapper.

    Args:
        space: Gymnasium or legacy Gym space.

    Returns:
        Matching RLMesh space wrapper.
    """
    gym_spaces = _gym_spaces()
    gym_space = cast(Any, space)

    if isinstance(space, gym_spaces.Box):
        return _box_from_gymnasium(gym_space)
    if isinstance(space, gym_spaces.Discrete):
        return Discrete(
            int(gym_space.n),
            start=int(getattr(gym_space, "start", 0)),
            dtype=str(getattr(gym_space, "dtype", "int64")),
        )
    if isinstance(space, gym_spaces.MultiBinary):
        return MultiBinary(space_shape(gym_space.n))
    if isinstance(space, gym_spaces.MultiDiscrete):
        return _multi_discrete_from_gymnasium(gym_space)
    if hasattr(gym_spaces, "Text") and isinstance(space, gym_spaces.Text):
        return Text(
            int(gym_space.max_length),
            min_length=int(getattr(gym_space, "min_length", 1)),
            charset=_rlmesh_text_charset_from_gymnasium(gym_space),
        )
    if isinstance(space, gym_spaces.Dict):
        return Dict(
            {
                str(key): from_gymnasium_space(child)
                for key, child in gym_space.spaces.items()
            }
        )
    if isinstance(space, gym_spaces.Tuple):
        return Tuple(from_gymnasium_space(child) for child in gym_space.spaces)

    from .._rlmesh import space_spec_from_gym_space

    return space_from_spec(space_spec_from_gym_space(space))


def to_gymnasium_space(space: Space[Any] | SpaceSpec) -> object:
    """Convert an RLMesh space or native spec into a Gymnasium space.

    Args:
        space: RLMesh space wrapper or native space specification.

    Returns:
        Gymnasium space object.
    """
    spec = space.spec if isinstance(space, Space) else space
    gym_spaces = _gym_spaces()

    if spec.kind == "box":
        return _box_to_gymnasium(gym_spaces, spec)
    if spec.kind == "discrete":
        details = spec_details(spec)
        n = cast(int, details["n"])
        start = cast(int, details.get("start", 0))
        return gym_spaces.Discrete(
            int(n),
            start=int(start),
        )
    if spec.kind == "multi_binary":
        details = spec_details(spec)
        shape = details.get("dims", details.get("size", spec.shape))
        return gym_spaces.MultiBinary(shape)
    if spec.kind == "multi_discrete":
        details = spec_details(spec)
        return gym_spaces.MultiDiscrete(details["nvec"])
    if spec.kind == "text":
        details = spec_details(spec)
        max_length = cast(int, details["max_length"])
        min_length = cast(int, details.get("min_length", 1))
        charset = cast(str, details.get("charset", ""))
        return gym_spaces.Text(
            int(max_length),
            min_length=int(min_length),
            charset=charset or string.printable,
        )
    if spec.kind == "dict":
        details = spec_details(spec)
        raw_spaces = cast(Mapping[str, SpaceSpec], details["spaces"])
        return gym_spaces.Dict(
            {key: to_gymnasium_space(child) for key, child in raw_spaces.items()}
        )
    if spec.kind == "tuple":
        details = spec_details(spec)
        raw_spaces = cast(list[SpaceSpec], details["spaces"])
        return gym_spaces.Tuple(
            tuple(to_gymnasium_space(child) for child in raw_spaces)
        )
    return spec.to_gym_space()


def _gym_spaces() -> Any:
    last_error: ImportError | None = None
    for module_name in ("gymnasium", "gym"):
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            last_error = exc
            continue
        return module.spaces
    raise ImportError(
        "Gymnasium space conversion requires gymnasium or gym. "
        "Install rlmesh[gymnasium]."
    ) from last_error


def _gymnasium_text_charset(space: Any) -> str:
    characters = getattr(space, "characters", None)
    if isinstance(characters, str):
        return characters

    character_list = getattr(space, "character_list", None)
    if character_list is not None:
        return "".join(str(character) for character in character_list)

    character_set = getattr(space, "character_set", None)
    if character_set is not None:
        return "".join(sorted(str(character) for character in character_set))

    charset = getattr(space, "charset", None)
    if charset is None:
        return ""
    if isinstance(charset, str):
        return charset
    return "".join(sorted(str(character) for character in charset))


def _rlmesh_text_charset_from_gymnasium(space: Any) -> str | None:
    charset = _gymnasium_text_charset(space)
    if not charset:
        return None
    # Gymnasium's default Text charset is alphanumeric-only; RLMesh treats it as generic text.
    if frozenset(charset) == _GYMNASIUM_DEFAULT_TEXT_CHARSET:
        return None
    return charset


def _box_from_gymnasium(space: Any) -> Box:
    import numpy as np

    # Annotated Any: numpy's stubs degrade under the 3.10 typecheck floor.
    low: Any = np.asarray(space.low)
    high: Any = np.asarray(space.high)
    if (
        low.size > 0
        and high.size > 0
        and np.all(low == low.flat[0])
        and np.all(high == high.flat[0])
    ):
        return Box(
            float(low.flat[0]),
            float(high.flat[0]),
            shape=space_shape(space.shape),
            dtype=str(getattr(space, "dtype", "float32")),
        )

    from .._rlmesh import space_spec_from_gym_space

    return cast(Box, space_from_spec(space_spec_from_gym_space(space)))


def _box_to_gymnasium(gym_spaces: Any, spec: SpaceSpec) -> object:
    import numpy as np

    details = spec_details(spec)
    shape = tuple(spec.shape)
    dtype = np.dtype(spec.dtype)
    bounds_kind = details.get("bounds_kind")
    low: Any
    high: Any
    if bounds_kind == "unbounded":
        low = np.full(shape, -np.inf, dtype=dtype)
        high = np.full(shape, np.inf, dtype=dtype)
    elif bounds_kind == "uniform":
        low = np.full(shape, details["low"], dtype=dtype)
        high = np.full(shape, details["high"], dtype=dtype)
    else:
        low_flat: Any = np.asarray(details["low"], dtype=dtype)
        high_flat: Any = np.asarray(details["high"], dtype=dtype)
        low = low_flat.reshape(shape)
        high = high_flat.reshape(shape)
    return gym_spaces.Box(low=low, high=high, shape=shape, dtype=dtype)


def _multi_discrete_from_gymnasium(space: Any) -> MultiDiscrete:
    import numpy as np

    nvec = np.asarray(space.nvec)
    if nvec.ndim == 1:
        return MultiDiscrete([int(value) for value in nvec.tolist()])

    from .._rlmesh import space_spec_from_gym_space

    return cast(MultiDiscrete, space_from_spec(space_spec_from_gym_space(space)))
