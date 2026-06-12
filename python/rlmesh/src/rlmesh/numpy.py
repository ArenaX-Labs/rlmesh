"""NumPy-backed RLMesh clients and tensor helpers."""

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
    """Return a NumPy array view over an RLMesh tensor.

    Args:
        tensor: RLMesh tensor value to view.

    Returns:
        Read-only array view over the tensor buffer.
    """
    ensure_available()
    import numpy as np

    array = np.frombuffer(tensor.buffer, dtype=np.dtype(tensor.dtype))
    shape = tuple(tensor.shape)
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


_numpy_bridge: ValueAdapter = FrameworkBridge(
    name="numpy",
    ensure_available=ensure_available,
    decode_leaf=asarray,
    encode_leaf=_encode_leaf,
)
_numpy_space_adapter: SpaceAdapter[NumpyValue] = cast(
    SpaceAdapter[NumpyValue],
    space_adapter_from_value_adapter(_numpy_bridge),
)


def space_from_spec(spec: SpaceSpec) -> Space[NumpyValue]:
    """Create a NumPy-adapted space wrapper for a native space spec."""
    return _space_from_spec(spec, adapter=_numpy_space_adapter)


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

    _adapter: ClassVar[ValueAdapter] = _numpy_bridge
    _space_adapter: ClassVar[SpaceAdapter[Any] | None] = _numpy_space_adapter


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

    _adapter: ClassVar[ValueAdapter] = _numpy_bridge
    _space_adapter: ClassVar[SpaceAdapter[Any] | None] = _numpy_space_adapter


@final
class Model(ModelBase[NumpyValue, NumpyValue]):
    """NumPy-backed model worker.

    The wrapped prediction function receives one decoded observation and returns
    one action. RLMesh handles value encoding at the runtime boundary.

    Args:
        predict_fn: Callable that maps one observation to one action.
        on_reset: Optional callback invoked when the environment resets.
        on_episode_end: Optional callback invoked when an episode ends.
        on_close: Optional callback invoked when the model worker closes.

    Examples:
        >>> from rlmesh.numpy import Model
        >>> model = Model(lambda observation: 0)
        >>> model.run("127.0.0.1:5555", max_episodes=1)
    """

    _adapter: ClassVar[ValueAdapter] = _numpy_bridge


@final
class SandboxEnv(SandboxEnvBase[NumpyValue, NumpyValue]):
    """Owned NumPy-backed sandbox session for one environment.

    The sandbox starts an isolated environment process, connects a NumPy remote
    client to it, and stops the owned container when closed.

    Args:
        source: Gymnasium id, explicit ``gym://`` source, or pinned environment
            source such as an EnvHub/Hugging Face reference.
        base_image: Optional Docker base image override.
        rlmesh_package: Optional RLMesh package, wheel, or ``"local"`` installed
            in the sandbox.
        packages: Extra environment packages installed in the sandbox.
        imports: Import names checked during sandbox startup.
        trust_remote_code: Allow remote environment code to execute.
        allow_unpinned_hf: Allow Hugging Face sources without a pinned revision.
        **gym_make_kwargs: Keyword arguments forwarded to environment creation.

    Examples:
        >>> from rlmesh.numpy import SandboxEnv
        >>> env = SandboxEnv("CartPole-v1", packages=["gymnasium==1.3.0"])
        >>> observation, info = env.reset(seed=42)
        >>> env.close()
    """

    _remote_env_cls = RemoteEnv


@final
class SandboxVectorEnv(SandboxVectorEnvBase[NumpyValue, NumpyValue]):
    """Owned NumPy-backed sandbox session for vectorized environments.

    The sandbox starts multiple isolated environment instances and exposes them
    through the same vector client interface as a separately served endpoint.

    Args:
        source: Gymnasium id, explicit ``gym://`` source, or pinned environment
            source such as an EnvHub/Hugging Face reference.
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

    Examples:
        >>> from rlmesh.numpy import SandboxVectorEnv
        >>> envs = SandboxVectorEnv("CartPole-v1", num_envs=2)
        >>> observations, infos = envs.reset(seed=42)
        >>> envs.close()
    """

    _remote_env_cls = RemoteVectorEnv


__all__ = [
    "Model",
    "NumpyValue",
    "RemoteEnv",
    "RemoteVectorEnv",
    "SandboxEnv",
    "SandboxInfo",
    "SandboxVectorEnv",
    "asarray",
    "from_array",
    "space_from_spec",
]
