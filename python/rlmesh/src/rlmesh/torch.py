"""Experimental Torch-backed RLMesh clients and tensor helpers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias, cast, final

from ._frameworks import FrameworkBridge
from ._rlmesh import Tensor
from ._values import UNHANDLED, ValueAdapter
from .client import RemoteEnvBase, RemoteVectorEnvBase
from .model import ModelBase
from .sandbox import SandboxEnvBase, SandboxInfo, SandboxVectorEnvBase
from .spaces import Space, SpaceAdapter
from .spaces import space_from_spec as _space_from_spec
from .spaces._sample import space_adapter_from_value_adapter
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

    Args:
        tensor: RLMesh tensor or scalar primitive to convert.
        copy: If ``True``, copy tensor data before creating the Torch tensor.

    Returns:
        Torch tensor view or copy.
    """
    ensure_available()
    import torch

    if not isinstance(tensor, Tensor):
        return torch.tensor(tensor) if copy else torch.as_tensor(tensor)

    dtype = cast(torch.dtype, _torch_dtype(tensor.dtype))
    buffer: object = bytearray(tensor.buffer) if copy else tensor
    view = torch.frombuffer(buffer, dtype=dtype)
    shape = tuple(tensor.shape)
    return view.reshape(shape if shape else ())


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
    try:
        array = cpu_tensor.numpy()
    except RuntimeError as exc:
        if "Numpy is not available" not in str(exc):
            raise
        raise ImportError(
            "rlmesh.torch.from_tensor requires numpy for Torch tensor export. "
            "Install rlmesh[torch]."
        ) from exc
    return Tensor(array, list(array.shape), str(array.dtype))


def _torch_dtype(dtype: str) -> object:
    import torch

    mapping: dict[str, object] = {
        "bool": torch.bool,
        "uint8": torch.uint8,
        "int32": torch.int32,
        "int64": torch.int64,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported tensor dtype {dtype!r}") from exc


def _encode_leaf(value: object) -> object:
    import torch

    if isinstance(value, torch.Tensor):
        return from_tensor(value)
    return UNHANDLED


_torch_bridge: ValueAdapter = FrameworkBridge(
    name="torch",
    ensure_available=ensure_available,
    decode_leaf=as_tensor,
    encode_leaf=_encode_leaf,
)
_torch_space_adapter: SpaceAdapter[TorchValue] = cast(
    SpaceAdapter[TorchValue],
    space_adapter_from_value_adapter(_torch_bridge),
)


def space_from_spec(spec: SpaceSpec) -> Space[TorchValue]:
    """Create a Torch-adapted space wrapper for a native space spec."""
    return _space_from_spec(spec, adapter=_torch_space_adapter)


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

    _adapter: ClassVar[ValueAdapter] = _torch_bridge
    _space_adapter: ClassVar[SpaceAdapter[Any] | None] = _torch_space_adapter


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

    _adapter: ClassVar[ValueAdapter] = _torch_bridge
    _space_adapter: ClassVar[SpaceAdapter[Any] | None] = _torch_space_adapter


@final
class Model(ModelBase[TorchValue, TorchValue]):
    """Experimental Torch-backed model worker.

    Args:
        predict_fn: Callable that maps one observation to one action.
        on_reset: Optional callback invoked when the environment resets.
        on_episode_end: Optional callback invoked when an episode ends.
        on_close: Optional callback invoked when the model worker closes.
    """

    _adapter: ClassVar[ValueAdapter] = _torch_bridge


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
