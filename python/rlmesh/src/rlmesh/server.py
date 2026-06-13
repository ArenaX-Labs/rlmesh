"""High-level EnvServer wrapper for serving gymnasium environments."""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

from .client import Transport, normalize_bind_address
from .specs import EnvContract
from .types import EnvLike as BaseEnvLike
from .types import VectorEnvLike

try:
    from rlmesh._rlmesh import PyEnvServer
except ImportError as e:
    raise ImportError("Failed to import _rlmesh native module.") from e

if TYPE_CHECKING:
    from rlmesh._rlmesh import ServeOptions
    from rlmesh.recipes import EnvRecipe, Recipe

EnvLike = BaseEnvLike[Any, Any] | VectorEnvLike[Any, Any, Any]


def _coerce_env(env: EnvLike | Recipe | type[EnvRecipe]) -> EnvLike:
    """Build a recipe (or project + build an EnvRecipe) into an env, else pass through."""
    from rlmesh.recipes import Recipe as _Recipe
    from rlmesh.recipes import build
    from rlmesh.recipes._authoring import EnvRecipe, construct_authored, is_env_recipe

    if is_env_recipe(env):
        return construct_authored(env)
    if isinstance(env, EnvRecipe):
        raise TypeError(
            "pass the EnvRecipe subclass (EnvServer(MyEnv)), not an instance "
            "(EnvServer(MyEnv()))"
        )
    if isinstance(env, _Recipe):
        return build(env)
    return cast("EnvLike", env)


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
        env: EnvLike | Recipe | type[EnvRecipe],
        address: str | None = None,
        *,
        host: str | None = None,
        port: int | None = None,
        path: str | None = None,
        transport: Transport | None = None,
        options: ServeOptions | None = None,
    ) -> None:
        env = _coerce_env(env)
        normalized_address = normalize_bind_address(
            address,
            host=host,
            port=port,
            path=path,
            transport=transport,
        )
        self._server: PyEnvServer = PyEnvServer(
            env=env,
            address=normalized_address,
            options=options,
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


__all__ = ["EnvLike", "EnvServer"]
