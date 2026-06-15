"""Experimental Torch-backed RLMesh clients and tensor helpers."""

from __future__ import annotations

import importlib
import warnings
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
    import torch

    TorchTensor: TypeAlias = torch.Tensor
    TorchValue: TypeAlias = (
        PrimitiveValue
        | TorchTensor
        | list["TorchValue"]
        | tuple["TorchValue", ...]
        | dict[str, "TorchValue"]
    )
else:
    TorchTensor: TypeAlias = object
    TorchValue: TypeAlias = (
        PrimitiveValue
        | TorchTensor
        | list["TorchValue"]
        | tuple["TorchValue", ...]
        | dict[str, "TorchValue"]
    )


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

    Warning:
        Without ``copy=True`` the returned tensor shares memory with the
        RLMesh tensor, and Torch will let you write through it even though
        the export is flagged read-only (DLPack consumers may ignore that
        flag). Mutating the view corrupts the RLMesh tensor for every other
        view of the same data. Treat shared views as read-only, and pass
        ``copy=True`` for any tensor you intend to modify.

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
    if cpu_tensor.dtype == torch.bool and not _bool_dlpack_supported():
        raw = Tensor.from_dlpack(cpu_tensor.to(torch.uint8))
        return Tensor(raw, list(raw.shape), "bool")
    return Tensor.from_dlpack(cpu_tensor)


def _frombuffer_view(tensor: Tensor, *, writable_copy: bool) -> TorchTensor:
    """Buffer-protocol fallback used for copies and pre-2.2 bool tensors."""
    import torch

    dtype = cast("torch.dtype", _torch_dtype(tensor.dtype))
    buffer: object = bytearray(tensor.tobytes()) if writable_copy else tensor
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given buffer is not writable.*",
            category=UserWarning,
        )
        view = torch.frombuffer(buffer, dtype=dtype)
    shape = tuple(tensor.shape)
    return view.reshape(shape if shape else ())


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
        "bfloat16": torch.bfloat16,
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


def _decode_owned(tensor: Tensor) -> TorchTensor:
    """Owned, writable decode for the value-bridge (predict/step) path.

    The public ``as_tensor`` still shares memory by default; the decode path
    copies so an in-place op in ``predict`` (e.g. ``img.div_(255)``) cannot
    write through into the shared, read-only wire buffer. Opt back into the
    zero-copy view with ``as_tensor(tensor, copy=False)``.
    """
    return as_tensor(tensor, copy=True)


_torch_bridge: ValueBridge = FrameworkBridge(
    name="torch",
    ensure_available=ensure_available,
    decode_leaf=_decode_owned,
    encode_leaf=_encode_leaf,
)
_torch_space_bridge: SpaceBridge[TorchValue] = cast(
    SpaceBridge[TorchValue],
    space_bridge_from_value_bridge(_torch_bridge),
)


def space_from_spec(spec: SpaceSpec) -> Space[TorchValue]:
    """Create a Torch-adapted space wrapper for a native space spec."""
    return _space_from_spec(spec, bridge=_torch_space_bridge)


@final
class RemoteEnv(RemoteEnvBase[TorchValue, TorchValue]):
    """Experimental Torch-backed remote client for one environment.

    Tensor leaves decode to Torch tensors while Python primitives and nested
    containers are preserved.

    Args:
        address: Endpoint address such as ``"tcp://127.0.0.1:5555"``.
        host: TCP host helper used when ``address`` is omitted.
        port: TCP port helper used when ``address`` is omitted.
        path: Unix socket path helper used when ``address`` is omitted.
        transport: Explicit transport selector.
    """

    _bridge: ClassVar[ValueBridge] = _torch_bridge
    _space_bridge: ClassVar[SpaceBridge[Any] | None] = _torch_space_bridge


@final
class RemoteVectorEnv(RemoteVectorEnvBase[TorchValue, TorchValue]):
    """Experimental Torch-backed remote client for vectorized environments.

    Args:
        address: Endpoint address such as ``"tcp://127.0.0.1:5555"``.
        host: TCP host helper used when ``address`` is omitted.
        port: TCP port helper used when ``address`` is omitted.
        path: Unix socket path helper used when ``address`` is omitted.
        transport: Explicit transport selector.
    """

    _bridge: ClassVar[ValueBridge] = _torch_bridge
    _space_bridge: ClassVar[SpaceBridge[Any] | None] = _torch_space_bridge


@final
class Model(ModelBase[TorchValue, TorchValue]):
    """Experimental Torch-backed model: ``predict`` works in Torch values.

    The Torch-typed :class:`~rlmesh.model.ModelBase`; see it for the source/spec
    construction and ``run(env, seeds=[...]) -> RunResult`` eval.
    """

    _bridge: ClassVar[ValueBridge] = _torch_bridge
    _remote_env_cls: ClassVar[type | None] = RemoteEnv


@final
class SandboxEnv(SandboxEnvBase[TorchValue, TorchValue]):
    """Experimental Torch-backed owned sandbox session for one environment.

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
class SandboxVectorEnv(SandboxVectorEnvBase[TorchValue, TorchValue]):
    """Experimental Torch-backed owned sandbox session for vectorized environments.

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
    "Model",
    "RemoteEnv",
    "RemoteVectorEnv",
    "SandboxEnv",
    "SandboxInfo",
    "SandboxVectorEnv",
    "TorchValue",
    "as_tensor",
    "from_tensor",
    "space_from_spec",
]
