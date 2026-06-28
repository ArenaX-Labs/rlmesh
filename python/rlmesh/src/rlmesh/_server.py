"""High-level EnvServer wrapper for serving gymnasium environments."""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

from ._client import Transport, normalize_bind_address
from ._value_conversion import resolve_bridge
from .specs import EnvContract
from .types import EnvLike, VectorEnvLike

try:
    from rlmesh._rlmesh import PyEnvServer, PyVectorEnvServer
except ImportError as e:
    raise ImportError("Failed to import _rlmesh native module.") from e

if TYPE_CHECKING:
    from rlmesh._rlmesh import ServeOptions

    from ._value_conversion import ValueBridge
    from .adapters import EnvTags

VectorServerEnvLike = VectorEnvLike[Any, Any, Any]


def _bridge_wraps(bridge: ValueBridge | None) -> bool:
    """Whether ``bridge`` needs the native ``BridgedEnv`` wrapper.

    numpy (and the identity ``rlmesh`` bridge) are served by the default
    ``ValueBackend::Auto`` path, which already reads numpy/dlpack leaves -- only a
    non-numpy framework (torch, jax) needs the Python bridge + ``Native`` backend.
    """
    return bridge is not None and bridge.name not in ("numpy", "rlmesh")


def _is_vector_env(env: object) -> bool:
    """Whether ``env`` has the vectorized (VectorEnvLike) shape.

    A vectorized env exposes ``num_envs`` and per-lane ``single_*`` spaces; a single
    env has none of these. The native vector server then enforces ``num_envs >= 2``.
    """
    return (
        hasattr(env, "num_envs")
        or hasattr(env, "single_observation_space")
        or hasattr(env, "single_action_space")
    )


class EnvServer:
    """Serves an RLMesh-compatible environment.

    Args:
        env: Environment satisfying the RLMesh protocols.
        address: Optional bind address. Supports ``"tcp://host:port"``,
            ``"host:port"``, ``"port"``, and ``"unix:///path/to/socket.sock"``.
            Defaults to ``"tcp://127.0.0.1:0"`` when omitted.
        host: TCP host helper used when ``address`` is omitted.
        port: TCP port helper used when ``address`` is omitted.
        path: Unix socket path helper used when ``address`` is omitted.
        transport: Explicit transport selector.
        options: Optional serve lifecycle options controlling remote shutdown,
            idle shutdown, drain timeout, and close timeout.
        tags: Optional adapter env tags
            (:class:`rlmesh.adapters.EnvTags`) to publish for this env.
            They are validated against the env's spaces and merged into its
            metadata, so a model client can resolve an adapter from the
            contract alone (see :func:`rlmesh.adapters.resolve_from_contract`).
        framework: The framework the env's ``step`` requires its *action* as --
            ``"torch"``, ``"jax"``, ``"numpy"`` (default), or a
            :class:`~rlmesh._value_conversion.ValueBridge`. Only needed for a
            framework-strict env (one whose ``step`` does e.g. ``action.to(...)``);
            a tolerant env can omit it. *Observations* need no declaration -- a
            torch/jax obs (GPU included) is auto-detected and encoded either way.
            The wire stays framework-neutral, so the env's action framework is
            independent of any consuming model's framework.
        device: Device to place the incoming action on (torch/jax only), e.g.
            ``"cuda:0"`` or a ``torch.device``. Requires ``framework=``; rejected
            for numpy/the default.

    Examples:
        >>> from rlmesh import EnvServer, spaces
        >>>
        >>> class TinyEnv:
        ...     observation_space = spaces.from_gymnasium_space(
        ...         __import__("gymnasium").spaces.Discrete(4)
        ...     )
        ...     action_space = spaces.from_gymnasium_space(
        ...         __import__("gymnasium").spaces.Discrete(2)
        ...     )
        ...
        ...     def reset(self, *, seed=None, options=None):
        ...         return 0, {}
        ...
        ...     def step(self, action):
        ...         return 0, 0.0, False, False, {}
        ...
        ...     def close(self):
        ...         return None
        >>> server = EnvServer(TinyEnv(), "localhost:5555")
        >>> server.serve()
    """

    def __init__(
        self,
        env: EnvLike[Any, Any] | VectorServerEnvLike,
        address: str | None = None,
        *,
        host: str | None = None,
        port: int | None = None,
        path: str | None = None,
        transport: Transport | None = None,
        options: ServeOptions | None = None,
        tags: EnvTags | None = None,
        framework: str | ValueBridge | None = None,
        device: object | None = None,
    ) -> None:
        # The env is self-describing: a vectorized env (the VectorEnvLike shape) is
        # served by the native vector server, a single env by the scalar server.
        # Detect on the RAW env, before any wrapping.
        is_vector = _is_vector_env(env)
        if tags is not None:
            # Imported lazily so the common (un-tagged) serve path does not
            # pull in the adapters/numpy stack.
            from .adapters import tag

            env = tag(env, tags)

        # The framework is a value the author sets on the env side -- here, the
        # framework= kwarg (an EnvFactory passes its declared framework through it).
        # torch/jax wrap the env in a Python bridge + native value backend; numpy
        # (and the default) keep the Auto backend unchanged. The wire stays neutral
        # rlmesh-native either way, so the env's framework is independent of any
        # consuming model's framework.
        bridge = resolve_bridge(framework) if framework is not None else None
        native_values = _bridge_wraps(bridge)
        if device is not None and not (
            native_values and bridge is not None and bridge.supports_device()
        ):
            raise ValueError(
                "device=... requires a framework with a device (framework='torch' "
                "or 'jax'); numpy envs and the default backend have no device."
            )
        if native_values:
            # Lazy import keeps the un-bridged serve path light (mirrors tag()).
            from ._server_bridge import BridgedEnv

            assert bridge is not None
            # BridgedEnv duck-types as the env (delegates every other attribute);
            # cast so the static env type is preserved for the server constructors.
            env = cast(
                "EnvLike[Any, Any] | VectorServerEnvLike",
                BridgedEnv(env, bridge, device),
            )

        normalized_address = normalize_bind_address(
            address,
            host=host,
            port=port,
            path=path,
            transport=transport,
        )
        self._server: PyEnvServer | PyVectorEnvServer = (
            PyVectorEnvServer(
                env=env,
                address=normalized_address,
                options=options,
                native_values=native_values,
            )
            if is_vector
            else PyEnvServer(
                env=env,
                address=normalized_address,
                options=options,
                native_values=native_values,
            )
        )

    @property
    def address(self) -> str:
        """Get the bound server address."""
        return self._server.address()

    @property
    def env_contract(self) -> EnvContract:
        """Environment contract served by this endpoint."""
        return self._server.env_contract

    @property
    def spec(self) -> EnvContract:
        """Alias for `env_contract`."""
        return self._server.spec

    def serve(self) -> None:
        """Start serving the environment (blocking)."""
        self._server.serve()

    def start(self) -> None:
        """Start serving the environment on a background thread."""
        self._server.start()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for a background server to stop.

        Args:
            timeout: Optional timeout in seconds. ``None`` waits indefinitely.

        Returns:
            ``True`` if the server has stopped, or ``False`` if the timeout elapsed.
        """
        return self._server.wait(timeout)

    def shutdown(self) -> None:
        """Stop the server if it is running."""
        self._server.shutdown()

    def __repr__(self) -> str:
        return f"EnvServer(address={self.address!r})"

    def __enter__(self) -> EnvServer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        _ = exc_type, exc_val, exc_tb
        self.shutdown()


__all__ = ["EnvLike", "EnvServer", "VectorServerEnvLike"]
