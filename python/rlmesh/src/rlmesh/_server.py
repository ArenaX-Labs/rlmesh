"""High-level EnvServer wrapper for serving gymnasium environments."""

from __future__ import annotations

import json
import os
import signal
import threading
import weakref
from collections.abc import Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

from ._client import Transport, normalize_bind_address
from ._editions import serve_options_declaring
from ._load_native import load_native
from ._value_conversion import resolve_bridge
from .specs import EnvContract
from .types import EnvLike, VectorEnvLike

if TYPE_CHECKING:
    from rlmesh._rlmesh import PyEnvServer, PyVectorEnvServer, ServeOptions

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
    env has none of these. A one-lane vector is served as a scalar env
    (:class:`~rlmesh._single_lane.SingleLaneEnv`); the native vector server
    takes ``num_envs >= 2``.
    """
    return (
        hasattr(env, "num_envs")
        or hasattr(env, "single_observation_space")
        or hasattr(env, "single_action_space")
    )


#: Seconds a finalizer waits for a still-running server to stop. The serve loop
#: bounds its own drain and close at 5 s each, so this only has to outlast them.
_FINALIZE_JOIN_SECONDS = 15.0


def _stop_server(server: PyEnvServer | PyVectorEnvServer) -> None:
    """Stop a server whose wrapper died without an explicit ``shutdown()``.

    Registered with :func:`weakref.finalize`, which runs pending finalizers from
    its own ``atexit`` hook -- i.e. while the interpreter is still alive. Both
    calls matter: ``shutdown()`` triggers the drain and ``env.close()``, and
    ``wait()`` joins the serve and lane threads, so no thread is left calling
    into a finalizing interpreter (which kills the process with a signal).
    """
    server.shutdown()
    server.wait(_FINALIZE_JOIN_SECONDS)


#: Env var the platform sets to the contract branch it expects this env to serve,
#: as a JSON object of discriminant bindings. Unset asserts nothing.
EXPECTED_ENV_BRANCH_VAR = "RLMESH_EXPECTED_ENV_BRANCH"


def _check_expected_branch(env: object) -> None:
    """Refuse to serve a contract branch the caller did not ask for.

    The tag validation above runs ``join``, which reconciles widths, dim laws and
    ranges but never role *names* against bounds -- so a same-width contract swap
    (end-effector deltas for absolute targets) passes it silently. When the
    platform pins the branch it resolved the pairing against, compare it with the
    one ``make()`` actually stamped, here, before a model can connect.
    """
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
        address: Optional bind address. Supports ``"tcp://host:port"``,
            ``"host:port"``, ``"port"``, and ``"unix:///path/to/socket.sock"``.
            Defaults to ``"tcp://127.0.0.1:0"`` when omitted.
        host: TCP host helper used when ``address`` is omitted.
        port: TCP port helper used when ``address`` is omitted.
        path: Unix socket path helper used when ``address`` is omitted.
        transport: Explicit transport selector.
        options: Optional serve lifecycle options controlling remote shutdown,
            idle shutdown, drain timeout, close timeout, and the workflow
            edition this endpoint declares (see :doc:`/editions/index`). An
            edition set here is declared verbatim; options without one take
            ``RLMESH_WORKFLOW_EDITION``, then ``[tool.rlmesh]``.
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
        >>> server = EnvServer(TinyEnv(), "localhost:5555")  # doctest: +SKIP
        >>> server.serve()  # doctest: +SKIP

    ``close_env_on_shutdown`` (default ``True``) calls the env's ``close()`` when
    the server stops; pass ``False`` for a server standing in front of an env
    its caller owns, whose ``close()`` stays the caller's.
    """

    def __init__(
        self,
        env: EnvLike[Any, Any] | VectorServerEnvLike | Sequence[EnvLike[Any, Any]],
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
        close_env_on_shutdown: bool = True,
    ) -> None:
        # A list is the lanes of one endpoint (each a scalar env); anything else
        # is one env. The env is self-describing: a vectorized env (the
        # VectorEnvLike shape) is served by the native vector server, a scalar
        # env by the lane server as its single lane. Detect on the RAW env,
        # before any wrapping.
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
            is_vector = False
        else:
            is_vector = _is_vector_env(env)
            if is_vector and getattr(env, "num_envs", None) == 1:
                from ._single_lane import SingleLaneEnv

                env, is_vector = SingleLaneEnv(env), False

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

        def prepare(one: Any) -> Any:
            """Stamp tags and the framework bridge onto one served env."""
            if tags is not None:
                # Imported lazily so the common (un-tagged) serve path does not
                # pull in the adapters/numpy stack. A vector env's served spaces
                # are batched while tags describe one lane, so per-lane space
                # validation is deferred to resolve time there (mirroring the
                # factory stamp).
                from .adapters import tag

                one = tag(one, tags, validate=not is_vector)
            elif not is_vector:
                # A prebuilt or EnvFactory-stamped env can carry tags in its
                # metadata that were never validated against its spaces (the
                # factory stamp uses validate=False, because a vectorized
                # make()'s per-lane spaces differ from the served shape). For a
                # scalar env the spaces are real, so validate the published tags
                # now -- surfacing a bad tag at startup instead of when a model
                # first connects. (Vector envs keep the deferred check.)
                from collections.abc import Mapping

                metadata = getattr(one, "metadata", None)
                if isinstance(metadata, Mapping):
                    from .adapters import EnvTags, tag

                    published = EnvTags.from_metadata(
                        cast("Mapping[str, Any]", metadata)
                    )
                    if published is not None:
                        one = tag(one, published)  # idempotent re-stamp + validate
            _check_expected_branch(one)
            if native_values:
                # Lazy import keeps the un-bridged serve path light.
                from ._server_bridge import BridgedEnv

                assert bridge is not None
                # BridgedEnv duck-types as the env (delegates every other
                # attribute).
                one = BridgedEnv(one, bridge, device)
            return one

        if lanes is not None:
            env = cast("Any", [prepare(lane) for lane in lanes])
        else:
            env = prepare(env)

        normalized_address = normalize_bind_address(
            address,
            host=host,
            port=port,
            path=path,
            transport=transport,
        )
        server_cls = load_native("PyVectorEnvServer" if is_vector else "PyEnvServer")
        if options is None or options.workflow_edition is None:
            # A served env declares an edition like any other participant.
            # There is no class here to read one off -- the env is an object --
            # so this picks up the process and project surfaces. Options that
            # already carry a declaration are the caller's resolved answer
            # (`serve_env`, the loopback server `run()` stands up) and are
            # adopted verbatim.
            options = serve_options_declaring(options)
        self._server: PyEnvServer | PyVectorEnvServer = server_cls(
            env=env,
            address=normalized_address,
            options=options,
            native_values=native_values,
            close_env_on_shutdown=close_env_on_shutdown,
        )
        # A server still running when CPython finalizes is fatal: its serve and
        # lane threads call into the interpreter as it tears down. Stop it from
        # a finalizer (detached by an explicit shutdown()) instead.
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
