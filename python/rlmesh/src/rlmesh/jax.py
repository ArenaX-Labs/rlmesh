"""Experimental JAX-backed RLMesh clients and tensor helpers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias, cast, final

from ._frameworks import FrameworkBridge
from ._rlmesh import Tensor
from ._values import UNHANDLED, ValueBridge
from .client import RemoteEnvBase, RemoteVectorEnvBase
from .model import ModelBase
from .sandbox import SandboxEnvBase, SandboxInfo, SandboxVectorEnvBase
from .spaces import Space, SpaceBridge
from .spaces import space_from_spec as _space_from_spec
from .spaces._sample import space_bridge_from_value_bridge
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
_jax_space_bridge: SpaceBridge[JaxValue] = cast(
    SpaceBridge[JaxValue],
    space_bridge_from_value_bridge(_jax_bridge),
)


def space_from_spec(spec: SpaceSpec) -> Space[JaxValue]:
    """Create a JAX-adapted space wrapper for a native space spec."""
    return _space_from_spec(spec, bridge=_jax_space_bridge)


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
    _space_bridge: ClassVar[SpaceBridge[Any] | None] = _jax_space_bridge


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
    _space_bridge: ClassVar[SpaceBridge[Any] | None] = _jax_space_bridge


@final
class Model(ModelBase[JaxValue, JaxValue]):
    """Experimental JAX-backed model: ``predict`` works in JAX values.

    The JAX-typed :class:`~rlmesh.model.ModelBase`; see it for the source/spec
    construction and ``run(env, seeds=[...]) -> RunResult`` eval.
    """

    _bridge: ClassVar[ValueBridge] = _jax_bridge


@final
class SandboxEnv(SandboxEnvBase[JaxValue, JaxValue]):
    """Experimental JAX-backed owned sandbox session for one environment.

    Args:
        source: Gymnasium id, explicit ``gym://`` source, or pinned environment
            source.
        base_image: Optional Docker base image override.
        rlmesh_package: Optional RLMesh package, wheel, or ``"local"`` installed
            in the sandbox.
        packages: Extra environment packages installed in the sandbox.
        imports: Import names checked during sandbox startup.
        trust_remote_code: Allow remote environment code to execute.
        allow_unpinned_hf: Allow Hugging Face sources without a pinned revision.
        **gym_make_kwargs: Keyword arguments forwarded to environment creation.
    """

    _remote_env_cls = RemoteEnv


@final
class SandboxVectorEnv(SandboxVectorEnvBase[JaxValue, JaxValue]):
    """Experimental JAX-backed owned sandbox session for vectorized environments.

    Args:
        source: Gymnasium id, explicit ``gym://`` source, or pinned environment
            source.
        num_envs: Number of environment instances to create.
        vectorization_mode: Vectorization mode requested inside the sandbox.
        base_image: Optional Docker base image override.
        rlmesh_package: Optional RLMesh package, wheel, or ``"local"`` installed
            in the sandbox.
        packages: Extra environment packages installed in the sandbox.
        imports: Import names checked during sandbox startup.
        trust_remote_code: Allow remote environment code to execute.
        allow_unpinned_hf: Allow Hugging Face sources without a pinned revision.
        **env_make_kwargs: Keyword arguments forwarded to environment creation.
    """

    _remote_env_cls = RemoteVectorEnv


__all__ = [
    "JaxValue",
    "Model",
    "RemoteEnv",
    "RemoteVectorEnv",
    "SandboxEnv",
    "SandboxInfo",
    "SandboxVectorEnv",
    "asarray",
    "from_array",
    "space_from_spec",
]
