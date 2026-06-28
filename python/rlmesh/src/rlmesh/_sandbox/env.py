"""Docker-backed single- and vector-environment sandbox sessions.

A sandbox env *is* a remote env -- ``reset`` / ``step`` / spaces / contract are inherited
from the ``Remote*EnvBase`` -- that also owns an isolated container started on
construction and stopped on close.
"""

from __future__ import annotations

import subprocess
import time
from typing import TypeVar

from .._client import RemoteEnvBase, RemoteVectorEnvBase
from .._rlmesh import sandbox_stop_env as _sandbox_stop_env
from .session import (
    SANDBOX_REMOTE_CONNECT_TIMEOUT_SECONDS,
    SandboxInfo,
    SandboxLifecycle,
    SandboxOptions,
    reject_sandbox_option_params,
    reject_single_env_vector_option,
    start_sandbox_container,
)

ValueT = TypeVar("ValueT")
ActionT = TypeVar("ActionT")

__all__ = [
    "SandboxEnvBase",
    "SandboxInfo",
    "SandboxOptions",
    "SandboxVectorEnvBase",
]

# Only these are retried while the container boots (mirrors ``SandboxModel``): a
# refused/unavailable connection or a connect timeout. The native env client dials
# once and fails fast against a not-yet-listening port, so the wait-for-ready loop
# lives in Python, not in the timeout handed to the native client.
_TRANSIENT_DIAL_ERRORS = (ConnectionError, TimeoutError, OSError)


def _container_running(container_id: str) -> bool:
    """Whether the sandbox container is still running (best effort)."""
    proc = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
        capture_output=True,
        text=True,
        check=False,
    )
    # On any inspect failure (gone / daemon error), treat as not running so the
    # caller fails fast rather than spinning the whole timeout.
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def _container_logs(container_id: str, tail: int = 50) -> str:
    proc = subprocess.run(
        ["docker", "logs", "--tail", str(tail), container_id],
        capture_output=True,
        text=True,
        check=False,
    )
    return (proc.stdout + proc.stderr).strip()


def _attach_sandbox(
    session: SandboxEnvBase | SandboxVectorEnvBase, connect_timeout_seconds: float
) -> None:
    """Attach the client to the started container, retrying while it boots.

    The rlmesh server only binds its port after the env factory's ``make()`` runs in
    the container, and the native client dials once and fails fast against a
    not-yet-listening port -- so the wait-for-ready loop lives here (mirroring
    ``SandboxModel._dial_with_retry``), not in the timeout handed to the native
    client. Stops the container on any failure so it is never leaked, and surfaces
    the container's recent logs when it exits or never becomes ready, instead of a
    bare ``transport error``.
    """
    sandbox = session.sandbox
    deadline = time.monotonic() + connect_timeout_seconds
    try:
        while True:
            try:
                session._initialize(
                    sandbox.address, connect_timeout_seconds=connect_timeout_seconds
                )
                return
            except _TRANSIENT_DIAL_ERRORS as exc:  # the container may still be starting
                short_id = sandbox.container_id[:12]
                # A dial error against an already-exited container is terminal: fail
                # fast with its logs rather than retrying for the whole timeout.
                if not _container_running(sandbox.container_id):
                    raise RuntimeError(
                        f"sandbox container {short_id} for {session._source!r} exited "
                        f"before becoming ready; recent logs:\n"
                        f"{_container_logs(sandbox.container_id)}"
                    ) from exc
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"sandbox container {short_id} for {session._source!r} did not "
                        f"become ready within {connect_timeout_seconds:.0f}s; recent "
                        f"logs:\n{_container_logs(sandbox.container_id)}"
                    ) from exc
                time.sleep(0.1)
    except BaseException:
        try:
            _sandbox_stop_env(container_id=sandbox.container_id)
        except BaseException:
            pass
        session._closed = True
        raise


class SandboxEnvBase(SandboxLifecycle, RemoteEnvBase[ValueT, ActionT]):
    """Experimental Docker-backed single-environment session.

    A remote env (reset/step/spaces/contract inherited) that also owns an isolated
    container; closing the session detaches the client and stops the container.

    Args:
        source: A gym id / ``gym://`` / ``hf://`` source built from source, or a
            prebuilt rlmesh-serving image (``docker://img`` / bare ``img:tag``) run
            directly (see :func:`rlmesh._sandbox.session.resolve_source_kind`).
        options: Optional :class:`~rlmesh._sandbox.session.SandboxOptions` carrying
            build/run infrastructure (base image, packages, rlmesh pin, ...); the
            single reserved keyword, ignored for a prebuilt image.
        connect_timeout_seconds: Seconds to wait for the container to start serving
            before giving up (default 30s). The server only binds its port after the
            env factory's ``make()`` runs, so an env that loads large sims/assets
            (e.g. a LIBERO task suite) needs headroom; raise it for slower startups.
        **params: Environment construction params -- the binding forwarded to the
            factory's ``make`` (validated against its declared ``params`` in the
            container, before construction). ``source`` is positional-only so a
            param named ``source`` flows here cleanly.
    """

    def __init__(
        self,
        source: str,
        /,
        *,
        options: SandboxOptions | None = None,
        connect_timeout_seconds: float = SANDBOX_REMOTE_CONNECT_TIMEOUT_SECONDS,
        **params: object,
    ) -> None:
        reject_single_env_vector_option(params)
        reject_sandbox_option_params(params)
        self._source = source
        self._closed = False
        self.sandbox = start_sandbox_container(
            source,
            options=options,
            num_envs=1,
            vectorization_mode=None,
            binding=params,
        )
        # Attach this client to the started container, waiting for it to become
        # ready; stops the container on any failure so it is never leaked.
        _attach_sandbox(self, connect_timeout_seconds)

    def _detach(self) -> None:
        # Skip the lifecycle mixin's close() override; detach via the remote base.
        super(SandboxLifecycle, self).close()


class SandboxVectorEnvBase(SandboxLifecycle, RemoteVectorEnvBase[ValueT, ActionT]):
    """Experimental Docker-backed vector-environment session.

    A remote vector env (reset/step, ``single_*`` spaces, contract inherited) that also
    owns an isolated container; closing the session detaches the client and stops the
    container.

    Args:
        source: A gym id / ``gym://`` / ``hf://`` source built from source, or a
            prebuilt rlmesh-serving image (``docker://img`` / bare ``img:tag``).
        num_envs: Number of environment instances to create (must be >= 2).
        vectorization_mode: Vectorization mode requested in the sandbox.
        options: Optional :class:`~rlmesh._sandbox.session.SandboxOptions` carrying
            build/run infrastructure; the single reserved keyword.
        connect_timeout_seconds: Seconds to wait for the container to start serving
            before giving up (default 30s); raise it for envs with slow ``make()``.
        **params: Environment construction params -- the binding forwarded to the
            factory's ``make`` (validated in the container before construction).
    """

    def __init__(
        self,
        source: str,
        /,
        num_envs: int,
        *,
        vectorization_mode: str = "sync",
        options: SandboxOptions | None = None,
        connect_timeout_seconds: float = SANDBOX_REMOTE_CONNECT_TIMEOUT_SECONDS,
        **params: object,
    ) -> None:
        if num_envs < 2:
            raise ValueError("SandboxVectorEnv requires num_envs >= 2")
        reject_sandbox_option_params(params)
        self._source = source
        self._closed = False
        self.sandbox = start_sandbox_container(
            source,
            options=options,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            binding=params,
        )
        _attach_sandbox(self, connect_timeout_seconds)

    def _detach(self) -> None:
        # Skip the lifecycle mixin's close() override; detach via the remote base.
        super(SandboxLifecycle, self).close()
