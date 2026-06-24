"""Docker-backed single- and vector-environment sandbox sessions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from typing import ClassVar, Generic, TypeVar, cast

from ..spaces import Space
from ..specs import EnvContract, SpaceSpec
from ..types import Metadata
from .session import (
    RemoteEnvHandle,
    RemoteVectorEnvHandle,
    ResetInfo,
    SandboxInfo,
    SandboxSessionBase,
    StepInfo,
    missing_remote_env_cls,
    reject_single_env_vector_option,
)

ValueT = TypeVar("ValueT")
ActionT = TypeVar("ActionT")

__all__ = [
    "SandboxEnvBase",
    "SandboxInfo",
    "SandboxVectorEnvBase",
]


class SandboxEnvBase(SandboxSessionBase[RemoteEnvHandle], Generic[ValueT, ActionT]):
    """Experimental Docker-backed single-environment session.

    Closing the session stops the owned sandbox container.

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
    """

    _remote_env_cls: ClassVar[Callable[[str], object]] = missing_remote_env_cls

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
        super().__init__(
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
        rlmesh_package: Optional RLMesh package, wheel, or ``"local"`` installed
            in the sandbox.
        packages: Extra environment packages installed in the sandbox.
        imports: Import names checked during sandbox startup.
        trust_remote_code: Allow remote environment code to execute.
        allow_unpinned_hf: Allow Hugging Face sources without a pinned revision.
        **env_make_kwargs: Keyword arguments forwarded to environment creation.
    """

    _remote_env_cls: ClassVar[Callable[[str], object]] = missing_remote_env_cls

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
        super().__init__(
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
