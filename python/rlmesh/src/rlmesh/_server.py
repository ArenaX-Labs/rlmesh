from __future__ import annotations

import contextvars
import json
import os
import signal
import threading
import weakref
from collections.abc import Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

from ._client import normalize_bind_address
from ._client._endpoint import redact_address
from ._editions import serve_options_declaring
from ._load_native import load_native
from ._upgrade import refuse_v1_address
from ._value_conversion import resolve_bridge
from .specs import EnvContract
from .types import EnvLike, VectorEnvLike

if TYPE_CHECKING:
    from rlmesh._rlmesh import PyEnvServer, PyVectorEnvServer, ServeOptions

    from ._value_conversion import ValueBridge
    from .adapters import EnvTags

VectorServerEnvLike = VectorEnvLike[Any, Any, Any]


def _bridge_wraps(bridge: ValueBridge | None) -> bool:
    """Whether ``bridge`` needs the native ``BridgedEnv`` wrapper."""
    return bridge is not None and bridge.name not in ("numpy", "rlmesh")


def _is_vector_env(env: object) -> bool:
    return (
        hasattr(env, "num_envs")
        or hasattr(env, "single_observation_space")
        or hasattr(env, "single_action_space")
    )


#: Seconds a finalizer waits for a running server to stop; must outlast the serve loop's 5 s drain and close.
_FINALIZE_JOIN_SECONDS = 15.0


def _stop_server(server: PyEnvServer | PyVectorEnvServer) -> None:
    server.shutdown()
    server.wait(_FINALIZE_JOIN_SECONDS)


#: Set while a run stands up its own server in front of a local env; that server announces nothing.
loopback: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "rlmesh_env_server_loopback", default=False
)

#: Env var naming the contract branch the platform expects, as JSON discriminant bindings.
EXPECTED_ENV_BRANCH_VAR = "RLMESH_EXPECTED_ENV_BRANCH"


def _check_spaces(env: object, *, vector: bool) -> None:
    names = (
        ("single_observation_space", "single_action_space")
        if vector
        else ("observation_space", "action_space")
    )
    for name in names:
        if not hasattr(env, name):
            raise TypeError(f"env must have '{name}'")


def _check_expected_branch(env: object) -> None:
    """Refuse to serve a contract branch the caller did not ask for."""
    expected = os.environ.get(EXPECTED_ENV_BRANCH_VAR)
    if not expected:
        return
    from collections.abc import Mapping

    from .adapters.constants import ENV_BRANCH_METADATA_KEY

    metadata = getattr(env, "metadata", None)
    stamped = (
        cast("Mapping[str, Any]", metadata).get(ENV_BRANCH_METADATA_KEY)
        if isinstance(metadata, Mapping)
        else None
    )
    if stamped != json.loads(expected):
        raise ValueError(
            f"{EXPECTED_ENV_BRANCH_VAR} expects the env to serve contract branch "
            f"{expected}, but it published {json.dumps(stamped)}; the image and "
            "the pairing that launched it declare different contracts"
        )


class EnvServer:
    """Serves an RLMesh-compatible environment.

    Args:
        env: Environment satisfying the RLMesh protocols, or a list of scalar
            environments to serve as the lanes of one endpoint. Lanes are
            stepped independently (a runtime keeps one request per lane in
            flight), so a slow lane never holds the others; a single env is the
            one-lane case of the same server. A natively vectorized env (the
            ``VectorEnvLike`` shape) is served by the vector server instead.
        address: The env's participant address in a session:
            ``"host:port/name"``, ``":0/name"`` (an OS-assigned local port) or
            ``"session/name"``; a bare ``"host:port"`` is short for
            ``"host:port/env"``. Defaults to ``":0/env"``. A v1 endpoint
            (``"tcp://..."`` or ``"unix://..."``) raises a ``ValueError``
            naming the session form.
        options: Optional serve lifecycle options: the close timeout and the
            workflow edition this endpoint declares (see
            :doc:`/editions/index`). An edition set here is declared verbatim;
            options without one take ``RLMESH_WORKFLOW_EDITION``, then
            ``[tool.rlmesh]``.
        tags: Optional adapter env tags
            (:class:`rlmesh.adapters.EnvTags`) to publish for this env.
            They are validated against the env's spaces and merged into its
            metadata, so a model client can resolve an adapter from the
            contract alone (see :func:`rlmesh.adapters.resolve_from_contract`).
        framework: The framework the env's ``step`` requires its *action* as --
            ``"torch"``, ``"jax"``, ``"numpy"`` (default), or a pre-resolved
            value-bridge object (advanced). Only needed for a
            framework-strict env (one whose ``step`` does e.g. ``action.to(...)``);
            a tolerant env can omit it. *Observations* need no declaration -- a
            torch/jax obs (GPU included) is auto-detected and encoded either way.
            The wire stays framework-neutral, so the env's action framework is
            independent of any consuming model's framework. ``render()`` is not
            bridged: frames pass through as-is and the native layer imports CPU
            arrays only, so a framework env should return numpy/CPU frames.
        device: Device to place the incoming action on (torch/jax only), e.g.
            ``"cuda:0"`` or a ``torch.device``. Requires ``framework=``; rejected
            for numpy/the default.

    Examples:
        >>> from rlmesh import EnvServer, spaces
        >>>
        >>> class TinyEnv:
        ...     observation_space = spaces.Discrete(4)
        ...     action_space = spaces.Discrete(2)
        ...
        ...     def reset(self, *, seed=None, options=None):
        ...         return 0, {}
        ...
        ...     def step(self, action):
        ...         return 0, 0.0, False, False, {}
        ...
        ...     def close(self):
        ...         return None
        >>> server = EnvServer(TinyEnv(), "localhost:5555/tiny")  # doctest: +SKIP
        >>> server.serve()  # doctest: +SKIP

    With ``RLMESH_READY_FILE`` set, the server writes that file once the env
    can answer and removes it when it stops (see ``python -m rlmesh.serve
    --help``).

    ``close_env_on_shutdown`` (default ``True``) calls the env's ``close()`` when
    the server stops; pass ``False`` for a server standing in front of an env
    its caller owns, whose ``close()`` stays the caller's.
    """

    def __init__(
        self,
        env: EnvLike[Any, Any] | VectorServerEnvLike | Sequence[EnvLike[Any, Any]],
        address: str | None = None,
        *,
        options: ServeOptions | None = None,
        tags: EnvTags | None = None,
        framework: str | ValueBridge | None = None,
        device: object | None = None,
        close_env_on_shutdown: bool = True,
    ) -> None:
        # Detect vectorization on the raw env, before any wrapping.
        lanes: list[Any] | None = (
            list(cast("Sequence[Any]", env)) if isinstance(env, (list, tuple)) else None
        )
        if lanes is not None:
            if not lanes:
                raise ValueError("EnvServer needs at least one environment")
            if any(_is_vector_env(lane) for lane in lanes):
                raise TypeError(
                    "lanes must be scalar environments; serve a vectorized env "
                    "on its own instead of inside a list"
                )
            for lane in lanes:
                _check_spaces(lane, vector=False)
            is_vector = False
        else:
            is_vector = _is_vector_env(env)
            _check_spaces(env, vector=is_vector)
            if is_vector and getattr(env, "num_envs", None) == 1:
                from ._single_lane import SingleLaneEnv

                env, is_vector = SingleLaneEnv(env), False

        # torch/jax wrap the env in a Python bridge; the wire stays framework-neutral.
        bridge = resolve_bridge(framework) if framework is not None else None
        native_values = _bridge_wraps(bridge)
        if device is not None and not (
            native_values and bridge is not None and bridge.supports_device()
        ):
            raise ValueError(
                "device=... requires a framework with a device (framework='torch' "
                "or 'jax'); numpy envs and the default backend have no device."
            )

        def prepare(one: Any) -> Any:
            if tags is not None:
                # Per-lane space validation is deferred for vector envs: served spaces are batched while tags describe one lane.
                from .adapters import tag

                one = tag(one, tags, validate=not is_vector)
            elif not is_vector:
                # Factory stamps use validate=False; validate a scalar env's published tags now to fail at startup.
                from collections.abc import Mapping

                metadata = getattr(one, "metadata", None)
                if isinstance(metadata, Mapping):
                    from .adapters import EnvTags, tag

                    published = EnvTags.from_metadata(
                        cast("Mapping[str, Any]", metadata)
                    )
                    if published is not None:
                        one = tag(one, published)
            _check_expected_branch(one)
            if native_values:
                from ._server_bridge import BridgedEnv

                assert bridge is not None
                one = BridgedEnv(one, bridge, device)
            return one

        if lanes is not None:
            env = cast("Any", [prepare(lane) for lane in lanes])
        else:
            env = prepare(env)

        normalized_address = normalize_bind_address(address)
        refuse_v1_address("EnvServer", normalized_address)
        server_cls = load_native("PyVectorEnvServer" if is_vector else "PyEnvServer")
        options = serve_options_declaring(options)
        self._server: PyEnvServer | PyVectorEnvServer = server_cls(
            env=env,
            address=normalized_address,
            options=options,
            native_values=native_values,
            close_env_on_shutdown=close_env_on_shutdown,
            announce=not loopback.get(),
        )
        # A server still running at CPython finalization is fatal (its threads call into a tearing-down
        # interpreter); stop it from a finalizer.
        self._finalizer = weakref.finalize(self, _stop_server, self._server)

    @property
    def address(self) -> str:
        """Get the bound server address."""
        return self._server.address()

    @property
    def env_contract(self) -> EnvContract:
        """Environment contract served by this endpoint."""
        return self._server.env_contract

    def serve(self) -> None:
        """Start serving the environment (blocking).

        The env keeps this thread: every ``reset``/``step``/``render``/``close``
        runs here, so an env built on the main thread (a simulator that only
        works from the thread that created it) is driven from it. On the main
        thread, Ctrl-C asks the server to stop, which drains and closes the env,
        and :class:`KeyboardInterrupt` is raised here afterwards rather than
        inside an env call; a second Ctrl-C interrupts an env call that will not
        return (Python code, not a call stuck inside a native library).
        """
        if threading.current_thread() is not threading.main_thread():
            self._server.serve()
            return
        interrupts = 0

        def on_sigint(_signum: int, _frame: object) -> None:
            nonlocal interrupts
            interrupts += 1
            if interrupts > 1:
                raise KeyboardInterrupt
            self._server.shutdown()

        previous = signal.signal(signal.SIGINT, on_sigint)
        try:
            self._server.serve()
        finally:
            signal.signal(signal.SIGINT, previous)
        if interrupts:
            raise KeyboardInterrupt

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
        self._finalizer.detach()
        self._server.shutdown()

    def __repr__(self) -> str:
        return f"EnvServer(address={redact_address(self.address)!r})"

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
