"""NumPy-backed RLMesh clients and tensor helpers."""

from __future__ import annotations

import importlib
from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias, cast, final

from ._authoring import EnvFactory as _EnvFactory
from ._client import RemoteEnvBase, RemoteModelBase, RemoteVectorEnvBase
from ._models.base import ModelBase
from ._rlmesh import Tensor
from ._sandbox import (
    SandboxEnvBase,
    SandboxInfo,
    SandboxOptions,
    SandboxVectorEnvBase,
)
from ._sandbox._model import SandboxModel
from ._value_conversion import UNHANDLED, FrameworkBridge, ValueBridge
from .spaces import Space
from .spaces import space_from_spec as _space_from_spec
from .specs import SpaceSpec
from .types import PrimitiveValue

if TYPE_CHECKING:
    import numpy as np

    NumpyArray: TypeAlias = np.ndarray[Any, Any]
    NumpyValue: TypeAlias = (
        PrimitiveValue
        | NumpyArray
        | list["NumpyValue"]
        | tuple["NumpyValue", ...]
        | dict[str, "NumpyValue"]
    )
else:
    NumpyArray: TypeAlias = object
    NumpyValue: TypeAlias = (
        PrimitiveValue
        | NumpyArray
        | list["NumpyValue"]
        | tuple["NumpyValue", ...]
        | dict[str, "NumpyValue"]
    )


def ensure_available() -> None:
    """Raise if NumPy is not installed."""
    try:
        _ = importlib.import_module("numpy")
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "rlmesh.numpy requires numpy. Install rlmesh[numpy]."
        ) from exc


def asarray(tensor: Tensor) -> NumpyArray:
    """Return a writable NumPy array containing an RLMesh tensor's data.

    The returned array owns a fresh copy of the tensor bytes, so it is writable
    and matches Gymnasium, where ``reset``/``step`` observations are writable
    (idioms such as ``obs /= 255.0`` work). For an opt-in zero-copy view that
    shares the tensor buffer, use the buffer protocol or DLPack directly
    (for example ``numpy.from_dlpack(tensor)``), treating the result as
    read-only.

    Args:
        tensor: RLMesh tensor value to convert.

    Returns:
        A writable NumPy array with a copy of the tensor data.
    """
    ensure_available()
    import numpy as np

    shape = tuple(tensor.shape)
    dtype = np.dtype(tensor.dtype)
    # ``bytearray`` yields a writable buffer, so the resulting array is writable
    # (np.frombuffer over immutable ``bytes`` would be read-only).
    array = np.frombuffer(bytearray(tensor.tobytes()), dtype=dtype)
    return cast(NumpyArray, np.reshape(array, shape if shape else ()))


def from_array(array: object) -> Tensor | PrimitiveValue:
    """Encode a NumPy array or scalar as an RLMesh value.

    Args:
        array: NumPy array or scalar to encode.

    Returns:
        Tensor for non-scalar arrays, or a primitive for scalar values.
    """
    ensure_available()
    import numpy as np

    if isinstance(array, np.generic):
        return cast(PrimitiveValue, array.item())
    if not isinstance(array, np.ndarray):
        raise TypeError("from_array() expects a numpy.ndarray or numpy scalar")
    ndarray = cast(NumpyArray, array)
    if ndarray.ndim == 0:
        return cast(PrimitiveValue, ndarray.item())
    contiguous = cast(NumpyArray, np.ascontiguousarray(ndarray))
    return Tensor(
        contiguous.tobytes(order="C"),
        list(contiguous.shape),
        str(contiguous.dtype),
    )


def _encode_leaf(value: object) -> object:
    import numpy as np

    if isinstance(value, np.generic | np.ndarray):
        return from_array(cast(NumpyArray, value))
    return UNHANDLED


def _stack_leaf(values: list[object]) -> object:
    import numpy as np

    # Text leaves stay a per-lane list; arrays and numeric primitives stack to
    # [N, ...]. A ragged leaf cannot fuse -- raise rather than silently returning a
    # list for this leaf while siblings stack, which hands the model a structurally
    # inconsistent batch ({stacked leaves} + {one list leaf}).
    if isinstance(values[0], (str, bytes)):
        return list(values)
    try:
        return np.stack([np.asarray(v) for v in values])
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"cannot fuse a ragged observation leaf across {len(values)} lanes "
            "(per-lane shapes differ); a batched predict needs every non-text leaf "
            "to stack into [N, ...]"
        ) from exc


def _unstack_leaf(value: object, n: int) -> list[object]:
    import numpy as np

    if isinstance(value, np.ndarray):
        arr = cast(Any, value)
        if arr.ndim >= 1 and arr.shape[0] == n:
            return list(arr)
        raise ValueError(
            f"a batched predict corner must return leaves with leading batch axis "
            f"{n}; got a numpy array of shape {arr.shape}"
        )
    seq = cast("list[object] | tuple[object, ...]", value)
    if isinstance(value, (list, tuple)) and len(seq) == n:
        return list(seq)
    raise ValueError(
        f"cannot split a batched action leaf of type {type(cast(object, value)).__name__} into "
        f"{n} lanes; return one batched value (leaves [{n}, ...])"
    )


_numpy_bridge: ValueBridge = FrameworkBridge(
    name="numpy",
    ensure_available=ensure_available,
    decode_leaf=asarray,
    encode_leaf=_encode_leaf,
    stack_leaf=_stack_leaf,
    unstack_leaf=_unstack_leaf,
)


def space_from_spec(spec: SpaceSpec) -> Space[NumpyValue]:
    """Create a NumPy-adapted space wrapper for a native space spec."""
    return _space_from_spec(spec, bridge=_numpy_bridge)


@final
class RemoteEnv(RemoteEnvBase[NumpyValue, NumpyValue]):
    """NumPy-backed remote client for a single RLMesh environment.

    Observations, rewards, and actions are decoded into Python primitives,
    NumPy arrays, or nested containers of those values. Use this client when a
    model or notebook expects NumPy values at the environment boundary.

    Args:
        address: Endpoint address such as ``"tcp://127.0.0.1:5555"``,
            ``"127.0.0.1:5555"``, or ``"unix:///tmp/env.sock"``.
        host: TCP host helper used when ``address`` is omitted.
        port: TCP port helper used when ``address`` is omitted.
        path: Unix socket path helper used when ``address`` is omitted.
        transport: Explicit transport selector.

    Examples:
        >>> from rlmesh.numpy import RemoteEnv
        >>> env = RemoteEnv("127.0.0.1:5555")
        >>> observation, info = env.reset(seed=42)
        >>> observation, reward, terminated, truncated, info = env.step(0)
        >>> env.close()
    """

    _bridge: ClassVar[ValueBridge] = _numpy_bridge


@final
class RemoteModel(RemoteModelBase[NumpyValue, NumpyValue]):
    """NumPy-backed handle to a served model (policy).

    Bind it to an env with ``rlmesh.session(model, env)`` to get a :class:`rlmesh.Session`
    whose ``predict`` accepts and returns NumPy values, driven symmetrically with the env.

    Examples:
        >>> import rlmesh
        >>> from rlmesh.numpy import RemoteEnv, RemoteModel
        >>> env = RemoteEnv("127.0.0.1:5555")
        >>> sess = rlmesh.session(RemoteModel("127.0.0.1:5556"), env)
        >>> obs, _ = sess.reset(seed=0)
        >>> action = sess.predict(obs)
        >>> obs, reward, terminated, truncated, _ = sess.step(action)
    """

    _bridge: ClassVar[ValueBridge] = _numpy_bridge


@final
class RemoteVectorEnv(RemoteVectorEnvBase[NumpyValue, NumpyValue]):
    """NumPy-backed remote client for a vectorized RLMesh environment.

    A vector client connects one model process to an endpoint that owns
    multiple environment instances. Batched observations, rewards,
    terminations, and truncations decode into NumPy-compatible values.

    Args:
        address: Endpoint address such as ``"tcp://127.0.0.1:5555"``.
        host: TCP host helper used when ``address`` is omitted.
        port: TCP port helper used when ``address`` is omitted.
        path: Unix socket path helper used when ``address`` is omitted.
        transport: Explicit transport selector.

    Examples:
        >>> from rlmesh.numpy import RemoteVectorEnv
        >>> envs = RemoteVectorEnv("127.0.0.1:5555")
        >>> observations, infos = envs.reset(seed=42)
        >>> actions = [envs.single_action_space.sample() for _ in range(envs.num_envs)]
        >>> observations, rewards, terminations, truncations, infos = envs.step(actions)
        >>> envs.close()
    """

    _bridge: ClassVar[ValueBridge] = _numpy_bridge


class Model(ModelBase[NumpyValue, NumpyValue]):
    """NumPy-backed model: ``predict`` works in NumPy values.

    The NumPy-typed :class:`~rlmesh._models.base.ModelBase`: wrap a predict callable
    (``Model(fn, spec=...)``) or subclass and override ``predict``; ``run(env,
    seeds=[...])`` returns a typed ``RunResult``. See :class:`~rlmesh._models.base.ModelBase`.

    Examples:
        >>> from rlmesh.numpy import Model
        >>> Model(lambda observation: 0).run("127.0.0.1:5555", seeds=[0]).mean_reward
        0.0
    """

    _bridge: ClassVar[ValueBridge] = _numpy_bridge
    _remote_env_cls: ClassVar[type | None] = RemoteEnv


@final
class SandboxEnv(SandboxEnvBase[NumpyValue, NumpyValue]):
    """Owned NumPy-backed sandbox session for one environment.

    The sandbox starts an isolated environment process, connects a NumPy remote
    client to it, and stops the owned container when closed.

    Args:
        source: A gym id / ``gym://`` / ``hf://`` source built from source, or a
            prebuilt rlmesh-serving image (``docker://img`` / bare ``img:tag``).
        options: Optional :class:`SandboxOptions` build/run infrastructure (base
            image, packages, rlmesh pin, ...); the single reserved keyword.
        **params: Environment construction params -- the binding forwarded to the
            factory's ``make`` (validated in the container before construction).

    Examples:
        >>> from rlmesh.numpy import SandboxEnv, SandboxOptions
        >>> env = SandboxEnv(
        ...     "CartPole-v1", options=SandboxOptions(packages=["gymnasium==1.3.0"])
        ... )
        >>> observation, info = env.reset(seed=42)
        >>> env.close()
    """

    _bridge: ClassVar[ValueBridge] = _numpy_bridge


@final
class SandboxVectorEnv(SandboxVectorEnvBase[NumpyValue, NumpyValue]):
    """Owned NumPy-backed sandbox session for vectorized environments.

    The sandbox starts multiple isolated environment instances and exposes them
    through the same vector client interface as a separately served endpoint.

    Args:
        source: A gym id / ``gym://`` / ``hf://`` source built from source, or a
            prebuilt rlmesh-serving image (``docker://img`` / bare ``img:tag``).
        num_envs: Number of environment instances to create.
        vectorization_mode: Vectorization mode requested inside the sandbox.
        options: Optional :class:`SandboxOptions` build/run infrastructure; the
            single reserved keyword.
        **params: Environment construction params -- the binding forwarded to the
            factory's ``make`` (validated in the container before construction).

    Examples:
        >>> from rlmesh.numpy import SandboxVectorEnv
        >>> envs = SandboxVectorEnv("CartPole-v1", num_envs=2)
        >>> observations, infos = envs.reset(seed=42)
        >>> envs.close()
    """

    _bridge: ClassVar[ValueBridge] = _numpy_bridge


class EnvFactory(_EnvFactory, ABC):
    """NumPy-backed :class:`~rlmesh.EnvFactory` -- the default, named for symmetry.

    Equivalent to subclassing :class:`rlmesh.EnvFactory`; the served env's
    obs/action seam stays on the default numpy/Auto path. Provided so ``rlmesh.numpy``
    carries the env-author class next to :class:`Model`, matching ``rlmesh.torch`` /
    ``rlmesh.jax``.
    """

    _bridge: ClassVar[ValueBridge | None] = _numpy_bridge


__all__ = [
    "EnvFactory",
    "Model",
    "NumpyValue",
    "RemoteEnv",
    "RemoteModel",
    "RemoteVectorEnv",
    "SandboxEnv",
    "SandboxInfo",
    "SandboxModel",
    "SandboxOptions",
    "SandboxVectorEnv",
    "asarray",
    "ensure_available",
    "from_array",
    "space_from_spec",
]
