"""Experimental JAX-backed RLMesh clients and tensor helpers."""

from __future__ import annotations

import importlib
from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias, cast, final

from ._authoring import EnvFactory as _EnvFactory
from ._client import RemoteEnvBase, RemoteModelBase, RemoteVectorEnvBase
from ._models.base import ModelBase
from ._rlmesh import Tensor
from ._sandbox import (
    SandboxBuild,
    SandboxEnvBase,
    SandboxInfo,
    SandboxRuntime,
    SandboxVectorEnvBase,
)
from ._sandbox._model import SandboxModel
from ._value_conversion import UNHANDLED, FrameworkBridge, ValueBridge
from .spaces import Space
from .spaces import space_from_spec as _space_from_spec
from .specs import SpaceSpec
from .types import PrimitiveValue

if TYPE_CHECKING:
    import jax

    JaxArray: TypeAlias = jax.Array
    JaxValue: TypeAlias = (
        PrimitiveValue
        | JaxArray
        | list["JaxValue"]
        | tuple["JaxValue", ...]
        | dict[str, "JaxValue"]
    )
else:
    JaxArray: TypeAlias = object
    JaxValue: TypeAlias = (
        PrimitiveValue
        | JaxArray
        | list["JaxValue"]
        | tuple["JaxValue", ...]
        | dict[str, "JaxValue"]
    )

_MINIMUM_JAX = (0, 4, 24)


def ensure_available() -> None:
    """Raise if JAX is not installed or is older than the supported floor."""
    try:
        jax = importlib.import_module("jax")
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError("rlmesh.jax requires jax. Install rlmesh[jax].") from exc
    if _version_tuple(cast(str, jax.__version__)) < _MINIMUM_JAX:
        raise ImportError(
            f"rlmesh.jax requires jax >= 0.4.24 for DLPack bool support; "
            f"found jax {jax.__version__}. Install rlmesh[jax]."
        )


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in version.split(".")[:3]:
        digits = ""
        for char in part:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def asarray(tensor: Tensor) -> JaxArray:
    """Return a JAX array for an RLMesh tensor.

    Args:
        tensor: RLMesh tensor value to convert.

    Returns:
        JAX array imported over DLPack. XLA shares 64-byte-aligned buffers
        and copies otherwise; either way the result is immutable.
    """
    ensure_available()
    import jax.numpy as jnp

    return cast(JaxArray, cast(Any, jnp).from_dlpack(tensor))


def from_array(array: object) -> Tensor | PrimitiveValue:
    """Encode a JAX array as an RLMesh value.

    Args:
        array: JAX array to encode.

    Returns:
        Tensor for non-scalar arrays, or a primitive for scalar values.
    """
    ensure_available()
    import jax

    if not isinstance(array, jax.Array):
        raise TypeError("from_array() expects a jax.Array")
    # jax's own annotations are partially untyped; treat it as Any locally.
    jax_any = cast(Any, jax)
    jax_array = cast(Any, array)
    if jax_array.ndim == 0:
        return cast(PrimitiveValue, jax_array.item())
    device = next(iter(jax_array.devices()))
    if device.platform != "cpu":
        jax_array = jax_any.device_put(jax_array, jax_any.devices("cpu")[0])
    jax_array = jax_array.block_until_ready()
    return Tensor.from_dlpack(cast(object, jax_array))


def _encode_leaf(value: object) -> object:
    import jax

    if isinstance(value, jax.Array):
        return from_array(value)
    return UNHANDLED


def _stack_leaf(values: list[object]) -> object:
    import jax.numpy as jnp

    # Array/numeric leaves stack to [N, ...]; text leaves stay a per-lane list. A
    # ragged leaf cannot fuse -- raise rather than silently returning a list for this
    # leaf while siblings stack, which hands the model a structurally inconsistent
    # batch ({stacked leaves} + {one list leaf}).
    if isinstance(values[0], (str, bytes)):
        return list(values)
    try:
        return cast(Any, jnp).stack([cast(Any, jnp).asarray(v) for v in values])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"cannot fuse a ragged observation leaf across {len(values)} lanes "
            "(per-lane shapes differ); a batched predict needs every non-text leaf "
            "to stack into [N, ...]"
        ) from exc


def _unstack_leaf(value: object, n: int) -> list[object]:
    import jax

    if isinstance(value, jax.Array):
        shape = cast(Any, value).shape
        if len(shape) >= 1 and shape[0] == n:
            return [cast(Any, value)[i] for i in range(n)]
        raise ValueError(
            f"a batched predict corner must return leaves with leading batch axis "
            f"{n}; got a jax array of shape {tuple(shape)}"
        )
    if isinstance(value, (list, tuple)) and len(cast(Any, value)) == n:
        return list(cast(Any, value))
    raise ValueError(
        f"cannot split a batched action leaf of type {type(cast(Any, value)).__name__} into "
        f"{n} lanes; return one batched value (leaves [{n}, ...])"
    )


def _as_jax_device(device: object) -> object:
    """Resolve a device string to a ``jax.Device`` (``device_put`` rejects strings).

    The CLI/docstring tell users to pass a torch-style string like ``"cpu"`` or
    ``"cuda:0"``; ``jax.device_put`` only accepts a ``jax.Device``/Sharding/None, so
    map ``"platform[:index]"`` to ``jax.devices(platform)[index]`` (``"cuda"`` is
    jax's ``"gpu"``). A ``jax.Device`` / ``None`` passes through untouched.
    """
    if not isinstance(device, str):
        return device
    import jax

    platform, _, index = device.partition(":")
    platform = "gpu" if platform == "cuda" else platform
    return cast(Any, jax).devices(platform)[int(index) if index else 0]


def _to_device_leaf(value: object, device: object) -> object:
    import jax

    if isinstance(value, jax.Array):
        return cast(Any, jax).device_put(value, _as_jax_device(device))
    return value


def _to_host_leaf(value: object) -> object:
    import jax

    # A reward/terminated/truncated leaf: pull a device array back to host as a
    # Python scalar/list; a plain Python scalar passes through.
    if isinstance(value, jax.Array):
        return cast(Any, jax).device_get(value).tolist()
    return value


_jax_bridge: ValueBridge = FrameworkBridge(
    name="jax",
    ensure_available=ensure_available,
    decode_leaf=asarray,
    encode_leaf=_encode_leaf,
    stack_leaf=_stack_leaf,
    unstack_leaf=_unstack_leaf,
    to_device_leaf=_to_device_leaf,
    to_host_leaf=_to_host_leaf,
)


def space_from_spec(spec: SpaceSpec) -> Space[JaxValue]:
    """Create a JAX-adapted space wrapper for a native space spec."""
    return _space_from_spec(spec, bridge=_jax_bridge)


@final
class RemoteEnv(RemoteEnvBase[JaxValue, JaxValue]):
    """Experimental JAX-backed remote client for one environment.

    Tensor leaves decode to JAX arrays while Python primitives and nested
    containers are preserved.

    Args:
        address: Endpoint address such as ``"tcp://127.0.0.1:5555"``.
        host: TCP host helper used when ``address`` is omitted.
        port: TCP port helper used when ``address`` is omitted.
        path: Unix socket path helper used when ``address`` is omitted.
        transport: Explicit transport selector.
    """

    _bridge: ClassVar[ValueBridge] = _jax_bridge


@final
class RemoteModel(RemoteModelBase[JaxValue, JaxValue]):
    """Experimental JAX-backed handle to a model (policy) server.

    Bind it to an env with ``rlmesh.session(model, env)`` to get a
    :class:`rlmesh.Session` whose ``predict`` accepts and returns JAX values,
    symmetric with :class:`RemoteEnv`.
    """

    _bridge: ClassVar[ValueBridge] = _jax_bridge


@final
class RemoteVectorEnv(RemoteVectorEnvBase[JaxValue, JaxValue]):
    """Experimental JAX-backed remote client for vectorized environments.

    Args:
        address: Endpoint address such as ``"tcp://127.0.0.1:5555"``.
        host: TCP host helper used when ``address`` is omitted.
        port: TCP port helper used when ``address`` is omitted.
        path: Unix socket path helper used when ``address`` is omitted.
        transport: Explicit transport selector.
    """

    _bridge: ClassVar[ValueBridge] = _jax_bridge


class Model(ModelBase[JaxValue, JaxValue]):
    """Experimental JAX-backed model: ``predict`` works in JAX values.

    The JAX-typed :class:`~rlmesh._models.base.ModelBase`; see it for the
    wrap-a-callable / subclass-and-override-``predict`` construction and
    ``run(env, seeds=[...]) -> RunResult`` eval.
    """

    _bridge: ClassVar[ValueBridge] = _jax_bridge
    # Without this, run(address) falls back to the numpy RemoteEnv and decodes
    # observations as ndarrays instead of JAX arrays.
    _remote_env_cls = RemoteEnv


@final
class SandboxEnv(SandboxEnvBase[JaxValue, JaxValue]):
    """Experimental JAX-backed owned sandbox session for one environment.

    Args:
        source: A gym id / ``gym://`` / ``hf://`` source built from source, or a
            prebuilt rlmesh-serving image (``docker://img`` / bare ``img:tag``).
        build: Optional :class:`SandboxBuild` -- build-from-source infrastructure;
            ignored for a prebuilt image.
        runtime: Optional :class:`SandboxRuntime` -- ``docker run`` settings
            (``gpus`` / ``devices`` / ``volumes``); prebuilt-image source only.
        **params: Environment construction params -- the binding forwarded to the
            factory's ``make`` (validated in the container before construction).
    """

    _bridge: ClassVar[ValueBridge] = _jax_bridge


@final
class SandboxVectorEnv(SandboxVectorEnvBase[JaxValue, JaxValue]):
    """Experimental JAX-backed owned sandbox session for vectorized environments.

    Args:
        source: A gym id / ``gym://`` / ``hf://`` source built from source, or a
            prebuilt rlmesh-serving image (``docker://img`` / bare ``img:tag``).
        num_envs: Number of environment instances to create.
        vectorization_mode: Vectorization mode requested inside the sandbox.
        build: Optional :class:`SandboxBuild` -- build-from-source infrastructure;
            ignored for a prebuilt image.
        runtime: Optional :class:`SandboxRuntime` -- ``docker run`` settings
            (``gpus`` / ``devices`` / ``volumes``); prebuilt-image source only.
        **params: Environment construction params -- the binding forwarded to the
            factory's ``make`` (validated in the container before construction).
    """

    _bridge: ClassVar[ValueBridge] = _jax_bridge


class EnvFactory(_EnvFactory, ABC):
    """JAX-backed :class:`~rlmesh.EnvFactory`: served envs speak JAX arrays.

    The producer-side mirror of :class:`Model` (the author's own class). Subclass
    and implement ``make`` as for :class:`rlmesh.EnvFactory`; the JAX framework
    rides this class, so every serve route types the obs/action seam as JAX without
    a per-entrypoint flag. To serve a plain (already-built) env, hand it to the
    neutral ``rlmesh.EnvServer(env, framework="jax")`` instead.
    """

    _bridge: ClassVar[ValueBridge | None] = _jax_bridge


__all__ = [
    "EnvFactory",
    "JaxValue",
    "Model",
    "RemoteEnv",
    "RemoteModel",
    "RemoteVectorEnv",
    "SandboxBuild",
    "SandboxEnv",
    "SandboxInfo",
    "SandboxModel",
    "SandboxRuntime",
    "SandboxVectorEnv",
    "asarray",
    "ensure_available",
    "from_array",
    "space_from_spec",
]
