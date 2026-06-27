"""Docker-backed single- and vector-environment sandbox sessions.

A sandbox env *is* a remote env -- ``reset`` / ``step`` / spaces / contract are inherited
from the ``Remote*EnvBase`` -- that also owns an isolated container started on
construction and stopped on close.
"""

from __future__ import annotations

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
        # Attach *this* client to the started container; stop it on any failure so the
        # original error propagates instead of leaking the container.
        try:
            self._initialize(
                self.sandbox.address,
                connect_timeout_seconds=SANDBOX_REMOTE_CONNECT_TIMEOUT_SECONDS,
            )
        except BaseException:
            try:
                _sandbox_stop_env(container_id=self.sandbox.container_id)
            except BaseException:
                pass
            self._closed = True
            raise

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
        try:
            self._initialize(
                self.sandbox.address,
                connect_timeout_seconds=SANDBOX_REMOTE_CONNECT_TIMEOUT_SECONDS,
            )
        except BaseException:
            try:
                _sandbox_stop_env(container_id=self.sandbox.container_id)
            except BaseException:
                pass
            self._closed = True
            raise

    def _detach(self) -> None:
        # Skip the lifecycle mixin's close() override; detach via the remote base.
        super(SandboxLifecycle, self).close()
