"""Experimental JAX-backed RLMesh clients and tensor helpers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias, cast, final

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


_jax_bridge: ValueBridge = FrameworkBridge(
    name="jax",
    ensure_available=ensure_available,
    decode_leaf=asarray,
    encode_leaf=_encode_leaf,
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

    The JAX-typed :class:`~rlmesh.model.ModelBase`; see it for the
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
        options: Optional :class:`SandboxOptions` build/run infrastructure; the
            single reserved keyword.
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
        options: Optional :class:`SandboxOptions` build/run infrastructure; the
            single reserved keyword.
        **params: Environment construction params -- the binding forwarded to the
            factory's ``make`` (validated in the container before construction).
    """

    _bridge: ClassVar[ValueBridge] = _jax_bridge


__all__ = [
    "JaxValue",
    "Model",
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
