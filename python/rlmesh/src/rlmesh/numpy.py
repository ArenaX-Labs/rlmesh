"""NumPy-backed RLMesh clients and tensor helpers."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias, cast, final

from ._frameworks import FrameworkBridge
from ._rlmesh import Tensor
from ._values import UNHANDLED, ValueBridge
from .client import RemoteEnvBase, RemoteVectorEnvBase
from .model import LifecycleCallback, ModelBase, PredictFn
from .sandbox import SandboxEnvBase, SandboxInfo, SandboxVectorEnvBase
from .spaces import Space, SpaceBridge
from .spaces import space_from_spec as _space_from_spec
from .spaces._sample import space_bridge_from_value_bridge
from .specs import SpaceSpec
from .types import PrimitiveValue

if TYPE_CHECKING:
    import numpy as np

    from .adapters import ModelSpec
    from .specs import EnvContract

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
        A writable NumPy array with a copy of the tensor data. ``bfloat16``
        tensors require the ``ml_dtypes`` package (``rlmesh[bfloat16]``).
    """
    ensure_available()
    import numpy as np

    shape = tuple(tensor.shape)
    if tensor.dtype == "bfloat16":
        dtype = _bfloat16_dtype()
    else:
        dtype = np.dtype(tensor.dtype)
    # ``bytearray`` yields a writable buffer, so the resulting array is writable
    # (np.frombuffer over immutable ``bytes`` would be read-only).
    array = np.frombuffer(bytearray(tensor.tobytes()), dtype=dtype)
    return cast(NumpyArray, np.reshape(array, shape if shape else ()))


def _bfloat16_dtype() -> Any:
    try:
        import ml_dtypes
    except ImportError as exc:
        raise ImportError(
            "bfloat16 tensors require ml_dtypes for NumPy conversion. "
            "Install rlmesh[bfloat16]."
        ) from exc
    return ml_dtypes.bfloat16


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


_numpy_bridge: ValueBridge = FrameworkBridge(
    name="numpy",
    ensure_available=ensure_available,
    decode_leaf=asarray,
    encode_leaf=_encode_leaf,
)
_numpy_space_bridge: SpaceBridge[NumpyValue] = cast(
    SpaceBridge[NumpyValue],
    space_bridge_from_value_bridge(_numpy_bridge),
)


def space_from_spec(spec: SpaceSpec) -> Space[NumpyValue]:
    """Create a NumPy-adapted space wrapper for a native space spec."""
    return _space_from_spec(spec, bridge=_numpy_space_bridge)


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
    _space_bridge: ClassVar[SpaceBridge[Any] | None] = _numpy_space_bridge


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
    _space_bridge: ClassVar[SpaceBridge[Any] | None] = _numpy_space_bridge


@final
class Model(ModelBase[NumpyValue, NumpyValue]):
    """NumPy-backed model worker.

    The wrapped prediction function receives one decoded observation and returns
    one action. RLMesh handles value encoding at the runtime boundary.

    Pass ``spec=`` to make this an *adapted* model: at :meth:`run` time the
    environment's published IO annotations (from its contract) and the model
    spec are resolved into an adapter, which preprocesses observations into the
    model's input format and postprocesses its actions back into the env's
    format. ``predict_fn`` then works in the model's own format, with no
    per-environment glue. ``run`` must be given an environment object (not a
    bare address) so the contract is available.

    Args:
        predict_fn: Callable that maps one observation to one action. With
            ``spec=``, it works in the model's declared input/output format.
        spec: Optional model IO spec (:class:`rlmesh.adapters.ModelSpec`) that
            turns this into an adapted model.
        on_reset: Optional callback invoked when the environment resets. With
            ``spec=``, the resolved adapter's ``reset`` is chained in.
        on_episode_end: Optional callback invoked when an episode ends.
        on_close: Optional callback invoked when the model worker closes.
        trust_entrypoints: Allow ``module:callable`` custom-input entrypoints
            in ``spec`` to be imported during resolution.

    Examples:
        >>> from rlmesh.numpy import Model
        >>> model = Model(lambda observation: 0)
        >>> model.run("127.0.0.1:5555", max_episodes=1)
    """

    _bridge: ClassVar[ValueBridge] = _numpy_bridge

    def __init__(
        self,
        predict_fn: PredictFn[NumpyValue, NumpyValue],
        *,
        spec: ModelSpec | None = None,
        on_reset: LifecycleCallback | None = None,
        on_episode_end: LifecycleCallback | None = None,
        on_close: LifecycleCallback | None = None,
        trust_entrypoints: bool = False,
    ) -> None:
        super().__init__(
            predict_fn,
            on_reset=on_reset,
            on_episode_end=on_episode_end,
            on_close=on_close,
        )
        self._spec = spec
        self._raw_predict = predict_fn
        self._user_on_reset = on_reset
        self._trust_entrypoints = trust_entrypoints

    def run(
        self,
        env_or_address: object,
        *,
        token: str = "",
        max_episodes: int | None = None,
        close_env: bool = False,
    ) -> None:
        """Run the model against an environment object or address.

        When this model was built with ``spec=``, ``env_or_address`` must be an
        environment object exposing ``env_contract`` (e.g.
        :class:`rlmesh.numpy.RemoteEnv`): the adapter is resolved from the env's
        published annotations and the model spec, then wired around the
        prediction function before the run begins.

        Args:
            env_or_address: Remote environment object or endpoint address.
            token: Optional endpoint token.
            max_episodes: Optional number of episodes to run before returning.
            close_env: If ``True``, request environment shutdown after the run.
        """
        if self._spec is not None:
            self._wire_adapter(env_or_address)
        super().run(
            env_or_address,
            token=token,
            max_episodes=max_episodes,
            close_env=close_env,
        )

    def _wire_adapter(self, env_or_address: object) -> None:
        from .adapters import resolve_from_contract

        spec = self._spec
        assert spec is not None  # guarded by the caller
        adapter = resolve_from_contract(
            _env_contract(env_or_address),
            spec,
            trust_entrypoints=self._trust_entrypoints,
        )
        wrapped = cast(
            "PredictFn[NumpyValue, NumpyValue]",
            adapter.wrap_predict(self._raw_predict),
        )
        self._install_worker(wrapped, _chain_reset(self._user_on_reset, adapter.reset))
        # Wired for this run; clear so a stray re-run does not silently re-resolve.
        self._spec = None


def _env_contract(env_or_address: object) -> EnvContract:
    contract = getattr(env_or_address, "env_contract", None)
    if contract is None:
        raise TypeError(
            "Model(spec=...).run() needs an environment object exposing "
            "'env_contract' (e.g. rlmesh.numpy.RemoteEnv); a bare address "
            "string carries no contract to resolve the adapter from"
        )
    return cast("EnvContract", contract)


def _chain_reset(
    user_on_reset: LifecycleCallback | None, adapter_reset: Callable[[], None]
) -> LifecycleCallback:
    def on_reset() -> None:
        adapter_reset()
        if user_on_reset is not None:
            user_on_reset()

    return on_reset


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
