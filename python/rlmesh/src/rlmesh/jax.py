"""Experimental JAX-backed RLMesh clients and tensor helpers.

Experimental: the classes below are functional and covered by tests, but their
surface may still change in a minor release -- the stable reference is
``rlmesh.numpy``. Every class labelled "Experimental" carries the label in
exactly this sense.
"""

from __future__ import annotations

import importlib
from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias, TypeVar, cast, final

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
    from typing_extensions import TypeVar as _DefaultTypeVar

    JaxArray: TypeAlias = jax.Array
    JaxValue: TypeAlias = (
        PrimitiveValue
        | JaxArray
        | list["JaxValue"]
        | tuple["JaxValue", ...]
        | dict[str, "JaxValue"]
    )
    _ObsT = _DefaultTypeVar("_ObsT", default=JaxValue)
    _ActT = _DefaultTypeVar("_ActT", default=JaxValue)
else:
    JaxArray: TypeAlias = object
    JaxValue: TypeAlias = (
        PrimitiveValue
        | JaxArray
        | list["JaxValue"]
        | tuple["JaxValue", ...]
        | dict[str, "JaxValue"]
    )
    _ObsT = TypeVar("_ObsT")
    _ActT = TypeVar("_ActT")

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


def asarray(tensor: Tensor | bool | int | float) -> JaxArray:
    """Return an immutable JAX array for an RLMesh tensor.

    Imports over DLPack: XLA shares 64-byte-aligned buffers (zero copy, the
    result may alias the wire buffer) and copies otherwise. Either way JAX
    arrays are immutable, so there is no mutation hazard and no ``copy=``
    knob. Scalar primitives (``bool``/``int``/``float``) are accepted for
    symmetry with :func:`rlmesh.torch.as_tensor` and become 0-d arrays.

    Args:
        tensor: RLMesh tensor or scalar primitive to convert.

    Returns:
        JAX array (immutable); shares the tensor buffer when XLA can, and
        copies otherwise.
    """
    ensure_available()
    import jax.numpy as jnp

    if not isinstance(tensor, Tensor):
        return cast(JaxArray, cast(Any, jnp).asarray(tensor))
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
    if str(jax_array.dtype) == "bfloat16":
        raise ValueError("bfloat16 is not supported on the wire; cast to float32 first")
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
    # batch ({stacked leaves} + {one list leaf}). A None leaf is rejected
    # explicitly, matching the numpy and torch bridges.
    if any(v is None for v in values):
        raise ValueError(
            f"cannot fuse a None observation leaf across {len(values)} lanes; a "
            "batched predict needs every lane to return a value for every leaf"
        )
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
    seq = cast("list[object] | tuple[object, ...]", value)
    if isinstance(value, (list, tuple)) and len(seq) == n:
        return list(seq)
    raise ValueError(
        f"cannot split a batched action leaf of type "
        f"{type(cast(object, value)).__name__} into "
        f"{n} lanes; return one batched value (leaves [{n}, ...]) or a "
        f"per-lane list of {n} actions"
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
    containers are preserved. Decoded arrays are immutable (see
    :func:`asarray`), so there is no wire-buffer mutation hazard.

    Args:
        address: Endpoint address such as ``"tcp://127.0.0.1:5555"``,
            ``"127.0.0.1:5555"``, or ``"unix:///tmp/env.sock"``.
        host: TCP host helper used when ``address`` is omitted.
        port: TCP port helper used when ``address`` is omitted.
        path: Unix socket path helper used when ``address`` is omitted.
        transport: Explicit transport selector.
        connect_timeout_seconds: Optional dial timeout in seconds; ``None``
            uses the native default.
        request_timeout_seconds: Optional per-request (reset/step/render)
            timeout in seconds; ``None`` waits indefinitely.

    Examples:
        >>> from rlmesh.jax import RemoteEnv
        >>> env = RemoteEnv("127.0.0.1:5555")  # doctest: +SKIP
        >>> observation, info = env.reset(seed=42)  # doctest: +SKIP
        >>> observation, reward, terminated, truncated, info = env.step(
        ...     0
        ... )  # doctest: +SKIP
        >>> env.close()  # doctest: +SKIP
    """

    _bridge: ClassVar[ValueBridge] = _jax_bridge


@final
class RemoteModel(RemoteModelBase[JaxValue, JaxValue]):
    """Experimental JAX-backed handle to a served model (policy).

    Bind it to an env with ``rlmesh.session(model, env)`` to get a
    :class:`rlmesh.Session` whose ``predict`` accepts and returns JAX values,
    driven symmetrically with the env.

    Args:
        address: Model endpoint address such as ``"tcp://127.0.0.1:5556"``.
        host: TCP host helper used when ``address`` is omitted.
        port: TCP port helper used when ``address`` is omitted.
        path: Unix socket path helper used when ``address`` is omitted.
        transport: Explicit transport selector.

    Examples:
        >>> import rlmesh
        >>> from rlmesh.jax import RemoteEnv, RemoteModel
        >>> env = RemoteEnv("127.0.0.1:5555")  # doctest: +SKIP
        >>> sess = rlmesh.session(RemoteModel("127.0.0.1:5556"), env)  # doctest: +SKIP
        >>> obs, _ = sess.reset(seed=0)  # doctest: +SKIP
        >>> action = sess.predict(obs)  # doctest: +SKIP
        >>> obs, reward, terminated, truncated, _ = sess.step(action)  # doctest: +SKIP
    """

    _bridge: ClassVar[ValueBridge] = _jax_bridge


@final
class RemoteVectorEnv(RemoteVectorEnvBase[JaxValue, JaxValue]):
    """Experimental JAX-backed remote client for vectorized environments.

    A vector client connects one model process to an endpoint that owns
    multiple environment instances. Batched observations, rewards,
    terminations, and truncations decode into JAX values.

    Args:
        address: Endpoint address such as ``"tcp://127.0.0.1:5555"``.
        host: TCP host helper used when ``address`` is omitted.
        port: TCP port helper used when ``address`` is omitted.
        path: Unix socket path helper used when ``address`` is omitted.
        transport: Explicit transport selector.
        connect_timeout_seconds: Optional dial timeout in seconds; ``None``
            uses the native default.
        request_timeout_seconds: Optional per-request (reset/step/render)
            timeout in seconds; ``None`` waits indefinitely.

    Examples:
        >>> from rlmesh.jax import RemoteVectorEnv
        >>> envs = RemoteVectorEnv("127.0.0.1:5555")  # doctest: +SKIP
        >>> observations, infos = envs.reset(seed=42)  # doctest: +SKIP
        >>> envs.close()  # doctest: +SKIP
    """

    _bridge: ClassVar[ValueBridge] = _jax_bridge


class Model(ModelBase[_ObsT, _ActT]):
    """Experimental JAX-backed model: ``predict`` works in JAX values.

    The JAX-typed :class:`~rlmesh._models.base.ModelBase`: wrap a predict
    callable (``Model(fn, spec=...)``) or subclass and override ``predict``;
    ``run(env, seeds=[...])`` returns a typed ``RunResult``. Observations
    arrive as immutable JAX arrays. See
    :class:`~rlmesh._models.base.ModelBase`.

    Generic over the observation/action types, defaulting to ``JaxValue``:
    wrapping an annotated predict callable infers them (``Model(predict)`` with
    ``def predict(obs: X) -> Y`` is a ``Model[X, Y]``, and its ``session`` a
    ``Session[X, Y]``); subclasses and unannotated sources bind ``JaxValue``.

    Examples:
        >>> from rlmesh.jax import Model
        >>> Model(lambda observation: 0).run(
        ...     "127.0.0.1:5555", seeds=[0]
        ... ).mean_reward  # doctest: +SKIP
        0.0
    """

    _bridge: ClassVar[ValueBridge] = _jax_bridge
    # Without this, run(address) falls back to the numpy RemoteEnv and decodes
    # observations as ndarrays instead of JAX arrays.
    _remote_env_cls: ClassVar[type | None] = RemoteEnv


@final
class SandboxEnv(SandboxEnvBase[JaxValue, JaxValue]):
    """Experimental JAX-backed owned sandbox session for one environment.

    The sandbox starts an isolated environment process, connects a JAX remote
    client to it, and stops the owned container when closed.

    Args:
        source: A gym id / ``gym://`` / ``hf://`` source built from source, or a
            prebuilt rlmesh-serving image (``docker://img`` / bare ``img:tag``).
        build: Optional :class:`SandboxBuild` -- build-from-source infrastructure;
            ignored for a prebuilt image.
        runtime: Optional :class:`SandboxRuntime` -- ``docker run`` settings
            (``gpus`` / ``devices`` / ``volumes``); prebuilt-image source only.
        **params: Environment construction params -- the binding forwarded to the
            factory's ``make`` (validated in the container before construction).

    Examples:
        >>> from rlmesh.jax import SandboxEnv, SandboxBuild
        >>> env = SandboxEnv(
        ...     "CartPole-v1", build=SandboxBuild(packages=["gymnasium==1.3.0"])
        ... )  # doctest: +SKIP
        >>> observation, info = env.reset(seed=42)  # doctest: +SKIP
        >>> env.close()  # doctest: +SKIP
    """

    _bridge: ClassVar[ValueBridge] = _jax_bridge


@final
class SandboxVectorEnv(SandboxVectorEnvBase[JaxValue, JaxValue]):
    """Experimental JAX-backed owned sandbox session for vectorized environments.

    The sandbox starts multiple isolated environment instances and exposes them
    through the same vector client interface as a separately served endpoint.

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

    Examples:
        >>> from rlmesh.jax import SandboxVectorEnv
        >>> envs = SandboxVectorEnv("CartPole-v1", num_envs=2)  # doctest: +SKIP
        >>> observations, infos = envs.reset(seed=42)  # doctest: +SKIP
        >>> envs.close()  # doctest: +SKIP
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
