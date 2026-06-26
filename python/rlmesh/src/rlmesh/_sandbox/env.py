"""Docker-backed single- and vector-environment sandbox sessions.

A sandbox env *is* a remote env -- ``reset`` / ``step`` / spaces / contract are inherited
from the ``Remote*EnvBase`` -- that also owns an isolated container started on
construction and stopped on close.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import TypeVar

from .._client import RemoteEnvBase, RemoteVectorEnvBase
from .._rlmesh import sandbox_stop_env as _sandbox_stop_env
from .session import (
    SANDBOX_REMOTE_CONNECT_TIMEOUT_SECONDS,
    SandboxInfo,
    SandboxLifecycle,
    reject_single_env_vector_option,
    start_sandbox_container,
)

ValueT = TypeVar("ValueT")
ActionT = TypeVar("ActionT")

__all__ = [
    "SandboxEnvBase",
    "SandboxInfo",
    "SandboxVectorEnvBase",
]


class SandboxEnvBase(SandboxLifecycle, RemoteEnvBase[ValueT, ActionT]):
    """Experimental Docker-backed single-environment session.

    A remote env (reset/step/spaces/contract inherited) that also owns an isolated
    container; closing the session detaches the client and stops the container.

    Args:
        source: Gymnasium id, explicit ``gym://`` source, or pinned environment source
            such as an EnvHub/Hugging Face reference.
        base_image: Optional Docker base image override.
        rlmesh_package: Optional RLMesh package, wheel, or ``"local"`` installed in the
            sandbox.
        packages: Extra environment packages installed in the sandbox.
        imports: Import names checked during sandbox startup.
        trust_remote_code: Allow remote environment code to execute.
        allow_unpinned_hf: Allow Hugging Face sources without a pinned revision.
        **gym_make_kwargs: Keyword arguments forwarded to environment creation.
    """

    def __init__(
        self,
        source: str,
        *,
        base_image: str | None = None,
        rlmesh_package: str | PathLike[str] | None = None,
        packages: Sequence[str] | None = None,
        imports: Sequence[str] | None = None,
        trust_remote_code: bool = False,
        allow_unpinned_hf: bool = False,
        build_memory: str | None = None,
        task: str | None = None,
        config: Mapping[str, object] | str | PathLike[str] | None = None,
        capabilities: Sequence[str] | None = None,
        override: str | PathLike[str] | None = None,
        cwd: str | PathLike[str] | None = None,
        repo_root: str | PathLike[str] | None = None,
        **gym_make_kwargs: object,
    ) -> None:
        reject_single_env_vector_option(gym_make_kwargs)
        self._source = source
        self._closed = False
        self.sandbox = start_sandbox_container(
            source,
            base_image=base_image,
            rlmesh_package=rlmesh_package,
            packages=packages,
            imports=imports,
            trust_remote_code=trust_remote_code,
            allow_unpinned_hf=allow_unpinned_hf,
            num_envs=1,
            vectorization_mode=None,
            build_memory=build_memory,
            task=task,
            config=config,
            capabilities=capabilities,
            override=override,
            cwd=cwd,
            repo_root=repo_root,
            gym_make_kwargs=gym_make_kwargs,
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
        source: Gymnasium id, explicit ``gym://`` source, or pinned environment source.
        num_envs: Number of environment instances to create (must be >= 2).
        vectorization_mode: Vectorization mode requested in the sandbox.
        base_image: Optional Docker base image override.
        rlmesh_package: Optional RLMesh package, wheel, or ``"local"`` installed in the
            sandbox.
        packages: Extra environment packages installed in the sandbox.
        imports: Import names checked during sandbox startup.
        trust_remote_code: Allow remote environment code to execute.
        allow_unpinned_hf: Allow Hugging Face sources without a pinned revision.
        **env_make_kwargs: Keyword arguments forwarded to environment creation.
    """

    def __init__(
        self,
        source: str,
        num_envs: int,
        *,
        vectorization_mode: str = "sync",
        base_image: str | None = None,
        rlmesh_package: str | PathLike[str] | None = None,
        packages: Sequence[str] | None = None,
        imports: Sequence[str] | None = None,
        trust_remote_code: bool = False,
        allow_unpinned_hf: bool = False,
        build_memory: str | None = None,
        task: str | None = None,
        config: Mapping[str, object] | str | PathLike[str] | None = None,
        capabilities: Sequence[str] | None = None,
        override: str | PathLike[str] | None = None,
        cwd: str | PathLike[str] | None = None,
        repo_root: str | PathLike[str] | None = None,
        **env_make_kwargs: object,
    ) -> None:
        if num_envs < 2:
            raise ValueError("SandboxVectorEnv requires num_envs >= 2")
        self._source = source
        self._closed = False
        self.sandbox = start_sandbox_container(
            source,
            base_image=base_image,
            rlmesh_package=rlmesh_package,
            packages=packages,
            imports=imports,
            trust_remote_code=trust_remote_code,
            allow_unpinned_hf=allow_unpinned_hf,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            build_memory=build_memory,
            task=task,
            config=config,
            capabilities=capabilities,
            override=override,
            cwd=cwd,
            repo_root=repo_root,
            gym_make_kwargs=env_make_kwargs,
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
