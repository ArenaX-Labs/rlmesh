"""The Gymnasium face of the NumPy remote clients.

With gymnasium installed, :class:`rlmesh.numpy.RemoteEnv` is a
``gymnasium.Env`` and :class:`rlmesh.numpy.RemoteVectorEnv` a
``gymnasium.vector.VectorEnv``, over gRPC and ``lockstep://`` alike. This
module rebuilds the env's own spaces from its contract and converts values
at the boundary, so code written against Gymnasium cannot tell the
difference:

- observations: a ``Discrete`` leaf is its space's NumPy integer, a
  ``MultiBinary`` leaf ``int8``, as Gymnasium's own spaces sample them;
- a vector env's ``info``: each per-lane list (length ``num_envs``) is an
  array again, as ``gymnasium.vector`` builds it (the wire's info values
  have no array type, so NumPy picks the dtype);
- actions: a plain array, list or scalar is cast to the space's dtype when
  the cast keeps the kind (``float64`` for a ``float32`` Box, ``np.int32``
  for a ``Discrete``), and refused with the leaf named when it does not
  (a float for an integer space) or when its shape does not fit.

Not part of the public API.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

from ..spaces import to_gymnasium_space
from ..spaces._internals import spec_details
from ..specs import SpaceSpec

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - optional
    gym = None

if TYPE_CHECKING:
    from typing import TypeAlias

    from gymnasium import Env
    from gymnasium.vector import VectorEnv

    GymEnv: TypeAlias = Env[Any, Any]
    GymVectorEnv: TypeAlias = VectorEnv[Any, Any, Any]
elif gym is not None:
    GymEnv = gym.Env
    GymVectorEnv = gym.vector.VectorEnv
else:  # pragma: no cover - without gymnasium the clients are plain classes
    GymEnv = GymVectorEnv = object

__all__ = [
    "Convert",
    "GymEnv",
    "GymVectorEnv",
    "action_converter",
    "gym_space",
    "observation_converter",
    "vector_info",
]

Convert = Callable[[Any], Any]


def gym_space(spec: SpaceSpec) -> Any:
    """The env's original Gymnasium space, rebuilt from its contract."""
    return to_gymnasium_space(spec)


def _identity(value: Any) -> Any:
    return value


def observation_converter(space: Any, batched: bool = False) -> Convert | None:
    """Wire observations to Gymnasium's leaf types; ``None`` when none change."""
    if gym is None:
        return None
    import numpy as np

    spaces = gym.spaces
    if isinstance(space, spaces.Dict):
        children = {
            key: observation_converter(child, batched)
            for key, child in space.spaces.items()
        }
        if all(c is None for c in children.values()):
            return None
        return lambda value: {
            key: (c(value[key]) if c is not None else value[key])
            for key, c in children.items()
        }
    if isinstance(space, spaces.Tuple):
        items = [observation_converter(child, batched) for child in space.spaces]
        if all(c is None for c in items):
            return None
        return lambda value: tuple(
            c(v) if c is not None else v for c, v in zip(items, value, strict=True)
        )
    if isinstance(space, spaces.Discrete):
        scalar = space.dtype.type
        if batched:
            return lambda value: np.asarray(value, dtype=space.dtype)
        return lambda value: scalar(value)
    if isinstance(space, spaces.MultiBinary):
        return lambda value: np.asarray(value).astype(np.int8, copy=False)
    return None


def vector_info(info: dict[str, Any], num_envs: int) -> dict[str, Any]:
    """A vector env's info with its per-lane lists as arrays again."""
    import numpy as np

    out: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, dict):
            out[key] = vector_info(cast("dict[str, Any]", value), num_envs)
        elif isinstance(value, list) and len(cast("list[Any]", value)) == num_envs:
            try:
                out[key] = np.asarray(cast("list[Any]", value))
            except ValueError:
                out[key] = value
        else:
            out[key] = value
    return out


def _refuse(where: str, what: str) -> TypeError:
    return TypeError(f"action{where}: {what}")


def _cast(value: Any, dtype: Any, where: str) -> Any:
    import numpy as np

    array = np.asarray(value)
    if array.dtype == dtype:
        return array
    if array.dtype.kind in "biuf" and np.can_cast(array.dtype, dtype, "same_kind"):
        return array.astype(dtype)
    raise _refuse(where, f"a {array.dtype} value does not cast to the space's {dtype}")


def action_converter(
    space: Any, spec: SpaceSpec, batch: int | None = None, where: str = ""
) -> Convert:
    """Gymnasium actions to the wire's leaf dtypes and shapes.

    ``batch`` is the leading lane count of a vector env's actions.
    """
    if gym is None:
        return _identity
    import numpy as np

    spaces = gym.spaces
    lead = () if batch is None else (batch,)
    if isinstance(space, spaces.Dict):
        dict_specs: Any = spec_details(spec)["spaces"]
        children = {
            key: action_converter(child, dict_specs[key], batch, f"{where}[{key!r}]")
            for key, child in space.spaces.items()
        }

        def dict_action(value: Any) -> Any:
            if not isinstance(value, Mapping):
                raise _refuse(where, f"expected a dict, got {type(value).__name__}")
            return {key: convert(value[key]) for key, convert in children.items()}

        return dict_action
    if isinstance(space, spaces.Tuple):
        child_specs: Any = spec_details(spec)["spaces"]
        items = [
            action_converter(child, child_specs[i], batch, f"{where}[{i}]")
            for i, child in enumerate(space.spaces)
        ]
        return lambda value: tuple(
            convert(v) for convert, v in zip(items, value, strict=True)
        )
    if isinstance(space, spaces.Text):
        return _identity
    dtype = np.dtype(spec.dtype)
    shape = lead + tuple(space.shape)
    multi_binary = isinstance(space, spaces.MultiBinary)

    def leaf(value: Any) -> Any:
        if multi_binary:
            array = np.asarray(value)
            if array.dtype.kind not in "biu":
                raise _refuse(where, f"a {array.dtype} value is not binary")
            array = array.astype(dtype)
        else:
            array = _cast(value, dtype, where)
        if array.shape != shape:
            raise _refuse(where, f"shape {array.shape} does not fit {shape}")
        if not shape:
            return array.item()
        return array

    return leaf
