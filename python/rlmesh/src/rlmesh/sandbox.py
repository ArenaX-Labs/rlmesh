"""Experimental Python-first Docker sandbox wrappers for RLMesh environments."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Generic,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeVar,
    cast,
)

from ._rlmesh import sandbox_start_env as _sandbox_start_env
from ._rlmesh import sandbox_stop_env as _sandbox_stop_env
from .specs import EnvContract, SpaceSpec
from .types import Metadata

ResetInfo: TypeAlias = dict[str, object]
StepInfo: TypeAlias = dict[str, object]

if TYPE_CHECKING:
    from .spaces import Space

ValueT = TypeVar("ValueT")
ActionT = TypeVar("ActionT")
RemoteT = TypeVar("RemoteT")


def _missing_remote_env_cls(_address: str) -> object:
    raise NotImplementedError("sandbox remote env class must be configured")


class _SandboxStartInfo(TypedDict):
    requested_source: str
    resolved_source: str
    address: str
    container_id: str


class RemoteEnvHandle(Protocol):
    """Remote client shape required by single-environment sandbox sessions."""

    @property
    def env_contract(self) -> EnvContract:
        """Environment contract reported by the remote endpoint."""
        ...

    @property
    def spec(self) -> EnvContract:
        """Alias for ``env_contract``."""
        ...

    @property
    def render_mode(self) -> str | None:
        """Configured render mode reported by the endpoint."""
        ...

    @property
    def metadata(self) -> Metadata:
        """Endpoint metadata."""
        ...

    @property
    def observation_space(self) -> Space[object]:
        """Observation space reported by the endpoint."""
        ...

    @property
    def action_space(self) -> Space[object]:
        """Action space reported by the endpoint."""
        ...

    @property
    def observation_space_spec(self) -> SpaceSpec:
        """Native observation space spec reported by the endpoint."""
        ...

    @property
    def action_space_spec(self) -> SpaceSpec:
        """Native action space spec reported by the endpoint."""
        ...

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[object, ResetInfo]:
        """Reset the remote environment."""
        ...

    def step(self, action: object) -> tuple[object, float, bool, bool, StepInfo]:
        """Step the remote environment with one action."""
        ...

    def render(self, *, env_index: int = 0) -> object | None:
        """Render a frame from the remote environment."""
        ...

    def open_viewer(
        self, *, env_index: int = 0, fps: float | None | str = "env"
    ) -> None:
        """Open a local render viewer for the remote environment."""
        ...

    def close_viewer(self) -> None:
        """Close the local render viewer if one is open."""
        ...

    def close(self) -> None:
        """Detach from the remote endpoint."""
        ...


class RemoteVectorEnvHandle(Protocol):
    """Remote client shape required by vector sandbox sessions."""

    @property
    def env_contract(self) -> EnvContract:
        """Environment contract reported by the remote endpoint."""
        ...

    @property
    def spec(self) -> EnvContract:
        """Alias for ``env_contract``."""
        ...

    @property
    def render_mode(self) -> str | None:
        """Configured render mode reported by the endpoint."""
        ...

    @property
    def metadata(self) -> Metadata:
        """Endpoint metadata."""
        ...

    @property
    def observation_space(self) -> Space[object]:
        """Alias for ``single_observation_space``."""
        ...

    @property
    def action_space(self) -> Space[object]:
        """Alias for ``single_action_space``."""
        ...

    @property
    def single_observation_space(self) -> Space[object]:
        """Observation space for one environment in the vector."""
        ...

    @property
    def single_action_space(self) -> Space[object]:
        """Action space for one environment in the vector."""
        ...

    @property
    def observation_space_spec(self) -> SpaceSpec:
        """Native observation space spec reported by the endpoint."""
        ...

    @property
    def action_space_spec(self) -> SpaceSpec:
        """Native action space spec reported by the endpoint."""
        ...

    @property
    def num_envs(self) -> int:
        """Number of environments in the vector endpoint."""
        ...

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[object, ResetInfo]:
        """Reset all remote environments."""
        ...

    def step(self, actions: object) -> tuple[object, object, object, object, StepInfo]:
        """Step all remote environments with a batch of actions."""
        ...

    def render(self, *, env_index: int = 0) -> object | None:
        """Render a frame from one remote environment."""
        ...

    def open_viewer(
        self, *, env_index: int = 0, fps: float | None | str = "env"
    ) -> None:
        """Open a local render viewer for one remote environment."""
        ...

    def close_viewer(self) -> None:
        """Close the local render viewer if one is open."""
        ...

    def close(self) -> None:
        """Detach from the remote endpoint."""
        ...


@dataclass(frozen=True)
class SandboxInfo:
    """Information about a running RLMesh sandbox container."""

    requested_source: str
    resolved_source: str
    address: str
    container_id: str


class SandboxSessionBase(Generic[RemoteT]):
    """Experimental base for Docker-backed sandbox sessions.

    A sandbox session owns the isolated environment container and the remote
    client connected to it. Closing the session closes the client and stops the
    container.

    Args:
        source: Gymnasium id, explicit ``gym://`` source, or pinned environment
            source such as an EnvHub/Hugging Face reference.
        base_image: Optional Docker base image override.
        package_spec: Optional RLMesh package or wheel installed in the sandbox.
        packages: Extra environment packages installed in the sandbox.
        imports: Import names checked during sandbox startup.
        trust_remote_code: Allow remote environment code to execute.
        allow_unpinned_hf: Allow Hugging Face sources without a pinned revision.
        num_envs: Number of environment instances to create.
        vectorization_mode: Optional vectorization mode requested in the sandbox.
        **gym_make_kwargs: Keyword arguments forwarded to environment creation.
    """

    _remote_env_cls: ClassVar[Callable[[str], object]] = _missing_remote_env_cls
    _source: str
    _closed: bool
    sandbox: SandboxInfo
    _remote_env: RemoteT

    def __init__(
        self,
        source: str,
        *,
        base_image: str | None = None,
        package_spec: str | None = None,
        packages: Sequence[str] | None = None,
        imports: Sequence[str] | None = None,
        trust_remote_code: bool = False,
        allow_unpinned_hf: bool = False,
        num_envs: int = 1,
        vectorization_mode: str | None = None,
        task: str | None = None,
        config: Mapping[str, object] | str | PathLike[str] | None = None,
        capabilities: Sequence[str] | None = None,
        override: str | PathLike[str] | None = None,
        cwd: str | PathLike[str] | None = None,
        repo_root: str | PathLike[str] | None = None,
        **gym_make_kwargs: object,
    ) -> None:
        _reject_removed_option("task", task)
        _reject_removed_option("config", config)
        _reject_removed_option("capabilities", capabilities)
        _reject_removed_option("override", override)
        _reject_removed_option("cwd", cwd)
        _reject_removed_option("repo_root", repo_root)

        self._source = source
        self._closed = False
        self.sandbox = _start_sandbox(
            source,
            base_image=base_image,
            package_spec=package_spec,
            packages=packages,
            imports=imports,
            trust_remote_code=trust_remote_code,
            allow_unpinned_hf=allow_unpinned_hf,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            gym_make_kwargs=gym_make_kwargs,
        )
        try:
            self._remote_env = self._create_remote_env(self.sandbox.address)
        except Exception:
            self._shutdown()
            raise

    def _create_remote_env(self, address: str) -> RemoteT:
        return cast(RemoteT, type(self)._remote_env_cls(address))

    @property
    def source(self) -> str:
        """Original sandbox source string requested by the caller."""
        return self._source

    def close(self) -> None:
        """Close the remote client and stop the owned sandbox container."""
        self._shutdown()

    def __enter__(self) -> SandboxSessionBase[RemoteT]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        _ = exc_type, exc_val, exc_tb
        self._shutdown()

    def __getattr__(self, name: str) -> object:
        remote_env = cast(object, object.__getattribute__(self, "_remote_env"))
        return cast(object, getattr(remote_env, name))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"source={self._source!r}, "
            f"address={self.sandbox.address!r}, "
            f"container_id={self.sandbox.container_id!r})"
        )

    def __del__(self) -> None:
        try:
            self._shutdown()
        except Exception:
            pass

    def _shutdown(self) -> None:
        if self._closed:
            return

        remote_error: BaseException | None = None
        try:
            remote_env = self.__dict__.get("_remote_env")
            if remote_env is not None:
                cast(RemoteEnvHandle | RemoteVectorEnvHandle, remote_env).close()
        except BaseException as exc:  # pragma: no cover - best effort cleanup path
            remote_error = exc

        try:
            _sandbox_stop_env(container_id=self.sandbox.container_id)
        finally:
            self._closed = True

        if remote_error is not None:
            raise remote_error


class SandboxEnvBase(SandboxSessionBase[RemoteEnvHandle], Generic[ValueT, ActionT]):
    """Experimental Docker-backed single-environment session.

    Closing the session stops the owned sandbox container.

    Args:
        source: Gymnasium id, explicit ``gym://`` source, or pinned environment
            source such as an EnvHub/Hugging Face reference.
        base_image: Optional Docker base image override.
        package_spec: Optional RLMesh package or wheel installed in the sandbox.
        packages: Extra environment packages installed in the sandbox.
        imports: Import names checked during sandbox startup.
        trust_remote_code: Allow remote environment code to execute.
        allow_unpinned_hf: Allow Hugging Face sources without a pinned revision.
        **gym_make_kwargs: Keyword arguments forwarded to environment creation.
    """

    _remote_env_cls: ClassVar[Callable[[str], object]] = _missing_remote_env_cls

    def __init__(
        self,
        source: str,
        *,
        base_image: str | None = None,
        package_spec: str | None = None,
        packages: Sequence[str] | None = None,
        imports: Sequence[str] | None = None,
        trust_remote_code: bool = False,
        allow_unpinned_hf: bool = False,
        task: str | None = None,
        config: Mapping[str, object] | str | PathLike[str] | None = None,
        capabilities: Sequence[str] | None = None,
        override: str | PathLike[str] | None = None,
        cwd: str | PathLike[str] | None = None,
        repo_root: str | PathLike[str] | None = None,
        **gym_make_kwargs: object,
    ) -> None:
        _reject_single_env_vector_option(gym_make_kwargs)
        super().__init__(
            source,
            base_image=base_image,
            package_spec=package_spec,
            packages=packages,
            imports=imports,
            trust_remote_code=trust_remote_code,
            allow_unpinned_hf=allow_unpinned_hf,
            num_envs=1,
            vectorization_mode=None,
            task=task,
            config=config,
            capabilities=capabilities,
            override=override,
            cwd=cwd,
            repo_root=repo_root,
            **gym_make_kwargs,
        )

    @property
    def env_contract(self) -> EnvContract:
        """Environment contract reported by the sandboxed endpoint."""
        return self._remote_env.env_contract

    @property
    def spec(self) -> EnvContract:
        """Alias for `env_contract`."""
        return self._remote_env.spec

    @property
    def render_mode(self) -> str | None:
        """Configured render mode reported by the sandboxed endpoint."""
        return self._remote_env.render_mode

    @property
    def metadata(self) -> Metadata:
        """Metadata reported by the sandboxed endpoint."""
        return self._remote_env.metadata

    @property
    def observation_space(self) -> Space[ValueT]:
        """Observation space reported by the sandboxed endpoint."""
        return cast(Space[ValueT], self._remote_env.observation_space)

    @property
    def action_space(self) -> Space[ActionT]:
        """Action space reported by the sandboxed endpoint."""
        return cast(Space[ActionT], self._remote_env.action_space)

    @property
    def observation_space_spec(self) -> SpaceSpec:
        """Native observation space spec reported by the sandboxed endpoint."""
        return self._remote_env.observation_space_spec

    @property
    def action_space_spec(self) -> SpaceSpec:
        """Native action space spec reported by the sandboxed endpoint."""
        return self._remote_env.action_space_spec

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[ValueT, ResetInfo]:
        """Reset the sandboxed environment."""
        return cast(
            tuple[ValueT, ResetInfo], self._remote_env.reset(seed=seed, options=options)
        )

    def step(self, action: ActionT) -> tuple[ValueT, float, bool, bool, StepInfo]:
        """Step the sandboxed environment with one action."""
        return cast(
            tuple[ValueT, float, bool, bool, StepInfo], self._remote_env.step(action)
        )

    def render(self, *, env_index: int = 0) -> ValueT | None:
        """Render a frame from the sandboxed environment."""
        return cast(ValueT | None, self._remote_env.render(env_index=env_index))

    def open_viewer(
        self, *, env_index: int = 0, fps: float | None | str = "env"
    ) -> None:
        """Open a local render viewer for the sandboxed environment."""
        self._remote_env.open_viewer(env_index=env_index, fps=fps)

    def close_viewer(self) -> None:
        """Close the local render viewer if one is open."""
        self._remote_env.close_viewer()


class SandboxVectorEnvBase(
    SandboxSessionBase[RemoteVectorEnvHandle],
    Generic[ValueT, ActionT],
):
    """Experimental Docker-backed vector-environment session.

    Closing the session stops the owned sandbox container.

    Args:
        source: Gymnasium id, explicit ``gym://`` source, or pinned environment
            source such as an EnvHub/Hugging Face reference.
        num_envs: Number of environment instances to create.
        vectorization_mode: Vectorization mode requested in the sandbox.
        base_image: Optional Docker base image override.
        package_spec: Optional RLMesh package or wheel installed in the sandbox.
        packages: Extra environment packages installed in the sandbox.
        imports: Import names checked during sandbox startup.
        trust_remote_code: Allow remote environment code to execute.
        allow_unpinned_hf: Allow Hugging Face sources without a pinned revision.
        **env_make_kwargs: Keyword arguments forwarded to environment creation.
    """

    _remote_env_cls: ClassVar[Callable[[str], object]] = _missing_remote_env_cls

    def __init__(
        self,
        source: str,
        num_envs: int,
        *,
        vectorization_mode: str = "sync",
        base_image: str | None = None,
        package_spec: str | None = None,
        packages: Sequence[str] | None = None,
        imports: Sequence[str] | None = None,
        trust_remote_code: bool = False,
        allow_unpinned_hf: bool = False,
        task: str | None = None,
        config: Mapping[str, object] | str | PathLike[str] | None = None,
        capabilities: Sequence[str] | None = None,
        override: str | PathLike[str] | None = None,
        cwd: str | PathLike[str] | None = None,
        repo_root: str | PathLike[str] | None = None,
        **env_make_kwargs: object,
    ) -> None:
        super().__init__(
            source,
            base_image=base_image,
            package_spec=package_spec,
            packages=packages,
            imports=imports,
            trust_remote_code=trust_remote_code,
            allow_unpinned_hf=allow_unpinned_hf,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            task=task,
            config=config,
            capabilities=capabilities,
            override=override,
            cwd=cwd,
            repo_root=repo_root,
            **env_make_kwargs,
        )

    @property
    def env_contract(self) -> EnvContract:
        """Environment contract reported by the sandboxed vector endpoint."""
        return self._remote_env.env_contract

    @property
    def spec(self) -> EnvContract:
        """Alias for `env_contract`."""
        return self._remote_env.spec

    @property
    def render_mode(self) -> str | None:
        """Configured render mode reported by the sandboxed vector endpoint."""
        return self._remote_env.render_mode

    @property
    def metadata(self) -> Metadata:
        """Metadata reported by the sandboxed vector endpoint."""
        return self._remote_env.metadata

    @property
    def num_envs(self) -> int:
        """Number of environments in the sandboxed vector endpoint."""
        return self._remote_env.num_envs

    @property
    def observation_space(self) -> Space[ValueT]:
        """Alias for ``single_observation_space``."""
        return self.single_observation_space

    @property
    def action_space(self) -> Space[ActionT]:
        """Alias for ``single_action_space``."""
        return self.single_action_space

    @property
    def single_observation_space(self) -> Space[ValueT]:
        """Observation space for one sandboxed environment."""
        return cast(Space[ValueT], self._remote_env.single_observation_space)

    @property
    def single_action_space(self) -> Space[ActionT]:
        """Action space for one sandboxed environment."""
        return cast(Space[ActionT], self._remote_env.single_action_space)

    @property
    def observation_space_spec(self) -> SpaceSpec:
        """Native observation space spec reported by the sandboxed endpoint."""
        return self._remote_env.observation_space_spec

    @property
    def action_space_spec(self) -> SpaceSpec:
        """Native action space spec reported by the sandboxed endpoint."""
        return self._remote_env.action_space_spec

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[ValueT, ResetInfo]:
        """Reset all environments in the sandboxed vector endpoint."""
        return cast(
            tuple[ValueT, ResetInfo], self._remote_env.reset(seed=seed, options=options)
        )

    def step(self, actions: ActionT) -> tuple[ValueT, ValueT, ValueT, ValueT, StepInfo]:
        """Step all sandboxed environments with a batch of actions."""
        return cast(
            tuple[ValueT, ValueT, ValueT, ValueT, StepInfo],
            self._remote_env.step(actions),
        )

    def render(self, *, env_index: int = 0) -> ValueT | None:
        """Render a frame from one sandboxed environment."""
        return cast(ValueT | None, self._remote_env.render(env_index=env_index))

    def open_viewer(
        self, *, env_index: int = 0, fps: float | None | str = "env"
    ) -> None:
        """Open a local render viewer for one sandboxed environment."""
        self._remote_env.open_viewer(env_index=env_index, fps=fps)

    def close_viewer(self) -> None:
        """Close the local render viewer if one is open."""
        self._remote_env.close_viewer()


def _start_sandbox(
    source: str,
    *,
    base_image: str | None,
    package_spec: str | None,
    packages: Sequence[str] | None,
    imports: Sequence[str] | None,
    trust_remote_code: bool,
    allow_unpinned_hf: bool,
    num_envs: int,
    vectorization_mode: str | None,
    gym_make_kwargs: Mapping[str, object],
) -> SandboxInfo:
    kwargs_json = json.dumps(gym_make_kwargs) if gym_make_kwargs else None
    started = cast(
        _SandboxStartInfo,
        _sandbox_start_env(
            source,
            base_image=base_image,
            package_spec=package_spec,
            packages=list(packages or []),
            imports=list(imports or []),
            kwargs_json=kwargs_json,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            trust_remote_code=trust_remote_code,
            allow_unpinned_hf=allow_unpinned_hf,
        ),
    )
    return SandboxInfo(
        requested_source=started["requested_source"],
        resolved_source=started["resolved_source"],
        address=started["address"],
        container_id=started["container_id"],
    )


def _reject_removed_option(name: str, value: object) -> None:
    if value is not None:
        raise TypeError(
            f"SandboxEnv no longer supports {name}=...; use base_image=, "
            "package_spec=, packages=, imports=, and environment make kwargs"
        )


def _reject_single_env_vector_option(kwargs: Mapping[str, object]) -> None:
    for name in ("num_envs", "vectorization_mode"):
        if name in kwargs:
            raise TypeError(
                f"SandboxEnv is single-env only; use SandboxVectorEnv for {name}=..."
            )


__all__ = [
    "RemoteEnvHandle",
    "RemoteVectorEnvHandle",
    "SandboxEnvBase",
    "SandboxInfo",
    "SandboxSessionBase",
    "SandboxVectorEnvBase",
]
