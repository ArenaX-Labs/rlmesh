"""Experimental Torch-backed RLMesh clients and tensor helpers.

Experimental: the classes below are functional and covered by tests, but their
surface may still change in a minor release -- the stable reference is
``rlmesh.numpy``. Every class labelled "Experimental" carries the label in
exactly this sense.
"""

from __future__ import annotations

import importlib
import warnings
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
    import torch
    from typing_extensions import TypeVar as _DefaultTypeVar

    TorchTensor: TypeAlias = torch.Tensor
    TorchValue: TypeAlias = (
        PrimitiveValue
        | TorchTensor
        | list["TorchValue"]
        | tuple["TorchValue", ...]
        | dict[str, "TorchValue"]
    )
    _ObsT = _DefaultTypeVar("_ObsT", default=TorchValue)
    _ActT = _DefaultTypeVar("_ActT", default=TorchValue)
else:
    TorchTensor: TypeAlias = object
    TorchValue: TypeAlias = (
        PrimitiveValue
        | TorchTensor
        | list["TorchValue"]
        | tuple["TorchValue", ...]
        | dict[str, "TorchValue"]
    )
    _ObsT = TypeVar("_ObsT")
    _ActT = TypeVar("_ActT")


def ensure_available() -> None:
    """Raise if Torch is not installed."""
    try:
        _ = importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "rlmesh.torch requires torch. Install rlmesh[torch]."
        ) from exc


def as_tensor(
    tensor: Tensor | bool | int | float, *, copy: bool = False
) -> TorchTensor:
    """Return a Torch tensor view or copy of an RLMesh tensor.

    Views by default: without ``copy=True`` the result shares the tensor's
    memory (zero copy, via DLPack); with ``copy=True`` it owns a fresh,
    writable buffer (exactly one copy). Scalar primitives
    (``bool``/``int``/``float``) are accepted and become 0-d tensors (always
    freshly constructed, never aliased).

    Warning:
        Without ``copy=True`` the returned tensor shares memory with the
        RLMesh tensor, and Torch will let you write through it even though
        the export is flagged read-only (DLPack consumers may ignore that
        flag). Mutating the view corrupts the RLMesh tensor -- the wire
        buffer -- for every other view of the same data. Treat shared views
        as read-only, and pass ``copy=True`` for any tensor you intend to
        modify.

    Args:
        tensor: RLMesh tensor or scalar primitive to convert.
        copy: If ``True``, copy tensor data before creating the Torch tensor.

    Returns:
        Torch tensor sharing the RLMesh tensor's memory via DLPack, or an
        independent copy when ``copy=True``.
    """
    ensure_available()
    import torch
    import torch.utils.dlpack

    if not isinstance(tensor, Tensor):
        return torch.tensor(tensor) if copy else torch.as_tensor(tensor)

    if copy:
        return _frombuffer_view(tensor, writable_copy=True)
    if tensor.dtype == "bool" and not _bool_dlpack_supported():
        return _frombuffer_view(tensor, writable_copy=False)
    return torch.utils.dlpack.from_dlpack(tensor)


def from_tensor(tensor: object) -> Tensor | bool | int | float:
    """Encode a Torch tensor as an RLMesh value.

    Args:
        tensor: Torch tensor to encode.

    Returns:
        Tensor for non-scalar tensors, or a primitive for scalar values.
    """
    ensure_available()
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("from_tensor() expects a torch.Tensor")
    cpu_tensor = tensor.detach().cpu().contiguous()
    if cpu_tensor.ndim == 0:
        return cpu_tensor.item()
    if cpu_tensor.dtype == torch.bfloat16:
        raise ValueError("bfloat16 is not supported on the wire; cast to float32 first")
    if cpu_tensor.dtype == torch.bool and not _bool_dlpack_supported():
        raw = Tensor.from_dlpack(cpu_tensor.to(torch.uint8))
        return Tensor(raw, list(raw.shape), "bool")
    return Tensor.from_dlpack(cpu_tensor)


def _frombuffer_view(tensor: Tensor, *, writable_copy: bool) -> TorchTensor:
    """Buffer-protocol fallback used for copies and pre-2.2 bool tensors.

    Builds a zero-copy ``torch.frombuffer`` view over the tensor's buffer;
    ``writable_copy`` clones it -- exactly one copy, owned and writable.
    """
    import torch

    dtype = cast("torch.dtype", _torch_dtype(tensor.dtype))
    shape = tuple(tensor.shape)
    # torch.frombuffer rejects a zero-length buffer, so a zero-size leaf (an empty
    # mask, point cloud, or variable-length buffer) would crash this decode path.
    # Build the empty tensor directly instead.
    if any(dim == 0 for dim in shape):
        return torch.empty(shape, dtype=dtype)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given buffer is not writable.*",
            category=UserWarning,
        )
        view = torch.frombuffer(tensor, dtype=dtype)
    reshaped = view.reshape(shape if shape else ())
    return reshaped.clone() if writable_copy else reshaped


_bool_dlpack: bool | None = None


def _bool_dlpack_supported() -> bool:
    """Whether torch's DLPack path handles bool tensors (torch >= 2.2)."""
    global _bool_dlpack
    if _bool_dlpack is None:
        import torch.utils.dlpack

        try:
            _ = cast(
                object, torch.utils.dlpack.from_dlpack(Tensor(b"\x00", [1], "bool"))
            )
            _bool_dlpack = True
        except (RuntimeError, TypeError, BufferError, ValueError):
            _bool_dlpack = False
    return _bool_dlpack


def _torch_dtype(dtype: str) -> object:
    import torch

    mapping: dict[str, object] = {
        "bool": torch.bool,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if dtype in mapping:
        return mapping[dtype]
    if dtype in ("uint16", "uint32", "uint64"):
        torch_dtype = getattr(torch, dtype, None)
        if torch_dtype is None:
            raise ValueError(
                f"dtype {dtype!r} requires torch >= 2.3 "
                f"(running torch {torch.__version__})"
            )
        return torch_dtype
    raise ValueError(f"unsupported tensor dtype {dtype!r}")


def _encode_leaf(value: object) -> object:
    import torch

    if isinstance(value, torch.Tensor):
        return from_tensor(value)
    return UNHANDLED


def _stack_leaf(values: list[object]) -> object:
    import torch

    # Tensor leaves stack to [N, ...]; numeric primitives become a 1-D tensor. Text
    # leaves stay a per-lane list. A ragged leaf cannot fuse -- raise rather than
    # silently returning a list for this leaf while siblings stack, which hands the
    # model a structurally inconsistent batch ({stacked leaves} + {one list leaf}).
    # A None leaf is rejected explicitly, matching the numpy and jax bridges.
    if any(v is None for v in values):
        raise ValueError(
            f"cannot fuse a None observation leaf across {len(values)} lanes; a "
            "batched predict needs every lane to return a value for every leaf"
        )
    if isinstance(values[0], (str, bytes)):
        return list(values)
    try:
        if isinstance(values[0], torch.Tensor):
            return torch.stack(cast("list[TorchTensor]", values))
        return torch.as_tensor(values)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"cannot fuse a ragged observation leaf across {len(values)} lanes "
            "(per-lane shapes differ); a batched predict needs every non-text leaf "
            "to stack into [N, ...]"
        ) from exc


def _unstack_leaf(value: object, n: int) -> list[object]:
    import torch

    if isinstance(value, torch.Tensor):
        if value.dim() >= 1 and value.shape[0] == n:
            return list(torch.unbind(value, dim=0))
        raise ValueError(
            f"a batched predict corner must return leaves with leading batch axis "
            f"{n}; got a torch tensor of shape {tuple(value.shape)}"
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


def _decode_owned(tensor: Tensor) -> TorchTensor:
    """Owned, writable decode for the value-bridge (predict/step) path.

    The public ``as_tensor`` still shares memory by default; the decode path
    copies so an in-place op in ``predict`` (e.g. ``img.div_(255)``) cannot
    write through into the shared, read-only wire buffer. Opt back into the
    zero-copy view with ``as_tensor(tensor, copy=False)``.
    """
    return as_tensor(tensor, copy=True)


def _to_device_leaf(value: object, device: object) -> object:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(cast("torch.device | str", device))
    return value


def _to_host_leaf(value: object) -> object:
    import torch

    # A reward/terminated/truncated leaf: move a (possibly GPU) tensor to host as a
    # Python scalar/list in one transfer; a plain Python scalar passes through.
    if isinstance(value, torch.Tensor):
        return cast("object", cast("Any", value.detach().cpu()).tolist())
    return value


_torch_bridge: ValueBridge = FrameworkBridge(
    name="torch",
    ensure_available=ensure_available,
    decode_leaf=_decode_owned,
    encode_leaf=_encode_leaf,
    stack_leaf=_stack_leaf,
    unstack_leaf=_unstack_leaf,
    to_device_leaf=_to_device_leaf,
    to_host_leaf=_to_host_leaf,
)


def space_from_spec(spec: SpaceSpec) -> Space[TorchValue]:
    """Create a Torch-adapted space wrapper for a native space spec."""
    return _space_from_spec(spec, bridge=_torch_bridge)


@final
class RemoteEnv(RemoteEnvBase[TorchValue, TorchValue]):
    """Experimental Torch-backed remote client for one environment.

    Tensor leaves decode to Torch tensors while Python primitives and nested
    containers are preserved. Decoded observations are owned, writable copies
    (see :func:`as_tensor`), so in-place ops never corrupt the wire buffer.

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
        >>> from rlmesh.torch import RemoteEnv
        >>> env = RemoteEnv("127.0.0.1:5555")  # doctest: +SKIP
        >>> observation, info = env.reset(seed=42)  # doctest: +SKIP
        >>> observation, reward, terminated, truncated, info = env.step(
        ...     0
        ... )  # doctest: +SKIP
        >>> env.close()  # doctest: +SKIP
    """

    _bridge: ClassVar[ValueBridge] = _torch_bridge


@final
class RemoteModel(RemoteModelBase[TorchValue, TorchValue]):
    """Experimental Torch-backed handle to a served model (policy).

    Bind it to an env with ``rlmesh.session(model, env)`` to get a
    :class:`rlmesh.Session` whose ``predict`` accepts and returns Torch values,
    driven symmetrically with the env.

    Args:
        address: Model endpoint address such as ``"tcp://127.0.0.1:5556"``.
        host: TCP host helper used when ``address`` is omitted.
        port: TCP port helper used when ``address`` is omitted.
        path: Unix socket path helper used when ``address`` is omitted.
        transport: Explicit transport selector.

    Examples:
        >>> import rlmesh
        >>> from rlmesh.torch import RemoteEnv, RemoteModel
        >>> env = RemoteEnv("127.0.0.1:5555")  # doctest: +SKIP
        >>> sess = rlmesh.session(RemoteModel("127.0.0.1:5556"), env)  # doctest: +SKIP
        >>> obs, _ = sess.reset(seed=0)  # doctest: +SKIP
        >>> action = sess.predict(obs)  # doctest: +SKIP
        >>> obs, reward, terminated, truncated, _ = sess.step(action)  # doctest: +SKIP
    """

    _bridge: ClassVar[ValueBridge] = _torch_bridge


@final
class RemoteVectorEnv(RemoteVectorEnvBase[TorchValue, TorchValue]):
    """Experimental Torch-backed remote client for vectorized environments.

    A vector client connects one model process to an endpoint that owns
    multiple environment instances. Batched observations, rewards,
    terminations, and truncations decode into Torch values.

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
        >>> from rlmesh.torch import RemoteVectorEnv
        >>> envs = RemoteVectorEnv("127.0.0.1:5555")  # doctest: +SKIP
        >>> observations, infos = envs.reset(seed=42)  # doctest: +SKIP
        >>> actions = [
        ...     envs.single_action_space.sample() for _ in range(envs.num_envs)
        ... ]  # doctest: +SKIP
        >>> observations, rewards, terminations, truncations, infos = envs.step(
        ...     actions
        ... )  # doctest: +SKIP
        >>> envs.close()  # doctest: +SKIP
    """

    _bridge: ClassVar[ValueBridge] = _torch_bridge


class Model(ModelBase[_ObsT, _ActT]):
    """Experimental Torch-backed model: ``predict`` works in Torch values.

    The Torch-typed :class:`~rlmesh._models.base.ModelBase`: wrap a predict
    callable (``Model(fn, spec=...)``) or subclass and override ``predict``;
    ``run(env, seeds=[...])`` returns a typed ``RunResult``. Observations
    arrive as owned, writable Torch tensors. See
    :class:`~rlmesh._models.base.ModelBase`.

    Generic over the observation/action types, defaulting to ``TorchValue``:
    wrapping an annotated predict callable infers them (``Model(predict)`` with
    ``def predict(obs: X) -> Y`` is a ``Model[X, Y]``, and its ``session`` a
    ``Session[X, Y]``); subclasses and unannotated sources bind ``TorchValue``.

    Examples:
        >>> from rlmesh.torch import Model
        >>> Model(lambda observation: 0).run(
        ...     "127.0.0.1:5555", seeds=[0]
        ... ).mean_reward  # doctest: +SKIP
        0.0
    """

    _bridge: ClassVar[ValueBridge] = _torch_bridge
    _remote_env_cls: ClassVar[type | None] = RemoteEnv


@final
class SandboxEnv(SandboxEnvBase[TorchValue, TorchValue]):
    """Experimental Torch-backed owned sandbox session for one environment.

    The sandbox starts an isolated environment process, connects a Torch remote
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
        >>> from rlmesh.torch import SandboxEnv, SandboxBuild
        >>> env = SandboxEnv(
        ...     "CartPole-v1", build=SandboxBuild(packages=["gymnasium==1.3.0"])
        ... )  # doctest: +SKIP
        >>> observation, info = env.reset(seed=42)  # doctest: +SKIP
        >>> env.close()  # doctest: +SKIP
    """

    _bridge: ClassVar[ValueBridge] = _torch_bridge


@final
class SandboxVectorEnv(SandboxVectorEnvBase[TorchValue, TorchValue]):
    """Experimental Torch-backed owned sandbox session for vectorized environments.

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
        >>> from rlmesh.torch import SandboxVectorEnv
        >>> envs = SandboxVectorEnv("CartPole-v1", num_envs=2)  # doctest: +SKIP
        >>> observations, infos = envs.reset(seed=42)  # doctest: +SKIP
        >>> envs.close()  # doctest: +SKIP
    """

    _bridge: ClassVar[ValueBridge] = _torch_bridge


class EnvFactory(_EnvFactory, ABC):
    """Torch-backed :class:`~rlmesh.EnvFactory`: served envs speak torch.

    The producer-side mirror of :class:`Model` (the author's own class). Subclass
    and implement ``make`` exactly as for :class:`rlmesh.EnvFactory`; the torch
    framework rides this class, so every serve route (``serve``, the ``python -m
    rlmesh.serve`` CLI, a prebuilt/sandbox image) types the obs/action seam as torch
    without a per-entrypoint flag. To serve a plain (already-built) env, hand it to
    the neutral ``rlmesh.EnvServer(env, framework="torch")`` instead -- the server
    stays framework-neutral; the framework is a value you set on the env side.
    """

    _bridge: ClassVar[ValueBridge | None] = _torch_bridge


__all__ = [
    "EnvFactory",
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
    "TorchValue",
    "as_tensor",
    "ensure_available",
    "from_tensor",
    "space_from_spec",
]
