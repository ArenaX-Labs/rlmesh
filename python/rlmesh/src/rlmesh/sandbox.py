"""Experimental Python-first Docker sandbox wrappers for RLMesh environments."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike, fspath
from types import MappingProxyType
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

from ._rlmesh import sandbox_build_image as _sandbox_build_image
from ._rlmesh import sandbox_start_env as _sandbox_start_env
from ._rlmesh import sandbox_stop_env as _sandbox_stop_env
from .spaces import Space
from .specs import EnvContract, SpaceSpec
from .types import Metadata

if TYPE_CHECKING:
    from .recipes import EnvRecipe, Recipe

ResetInfo: TypeAlias = dict[str, object]
StepInfo: TypeAlias = dict[str, object]

ValueT = TypeVar("ValueT")
ActionT = TypeVar("ActionT")
RemoteT = TypeVar("RemoteT")
SANDBOX_REMOTE_CONNECT_TIMEOUT_SECONDS = 10.0
# Immutable empty mapping used as the default for callers that pass no make kwargs.
_NO_MAKE_KWARGS: Mapping[str, object] = MappingProxyType({})


def _missing_remote_env_cls(_address: str, **_kwargs: object) -> object:
    raise NotImplementedError("sandbox remote env class must be configured")


class _SandboxStartInfo(TypedDict):
    requested_source: str
    resolved_source: str
    address: str
    container_id: str


class _SandboxBuildInfo(TypedDict):
    requested_source: str
    resolved_source: str
    image: str
    alias: str | None
    image_id: str


class _RemoteHandleBase(Protocol):
    """Surface shared by single- and vector-environment remote handles."""

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
    def observation_space_spec(self) -> SpaceSpec:
        """Native observation space spec reported by the endpoint."""
        ...

    @property
    def action_space_spec(self) -> SpaceSpec:
        """Native action space spec reported by the endpoint."""
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


class RemoteEnvHandle(_RemoteHandleBase, Protocol):
    """Remote client shape required by single-environment sandbox sessions."""

    @property
    def observation_space(self) -> Space[object]:
        """Observation space reported by the endpoint."""
        ...

    @property
    def action_space(self) -> Space[object]:
        """Action space reported by the endpoint."""
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


class RemoteVectorEnvHandle(_RemoteHandleBase, Protocol):
    """Remote client shape required by vector sandbox sessions."""

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


@dataclass(frozen=True)
class SandboxInfo:
    """Information about a running RLMesh sandbox container."""

    requested_source: str
    resolved_source: str
    address: str
    container_id: str


@dataclass(frozen=True)
class ExportResult:
    """The image produced by :func:`export`.

    ``image`` is the deterministic, content-addressed reference; it is stable
    for a given build, so the managed platform pins it. ``alias`` is the human
    tag passed as ``tag=``, if any. Both name the same image.
    """

    requested_source: str
    resolved_source: str
    image: str
    alias: str | None
    image_id: str


def export(
    source: str | Recipe | type[EnvRecipe] | type,
    *,
    tag: str | None = None,
    base_image: str | None = None,
    rlmesh_package: str | PathLike[str] | None = None,
    packages: Sequence[str] = (),
    trust_remote_code: bool = False,
    allow_unpinned_hf: bool = False,
    build_memory: str | None = None,
) -> ExportResult:
    """Build a recipe into a Docker image and return its reference.

    No container is started. The image is self-describing -- it bakes the recipe
    document and the kind-aware entrypoint, so ``docker run`` with no arguments
    serves the env or model on port 50051. It is the same image the RLMesh Managed
    platform runs; ``docker push`` the returned reference to a registry it can reach.

    Works for both env recipes (``EnvRecipe``, a ``Recipe``, or a registered env
    name) and model recipes (``ModelRecipe``, a ``kind='model'`` Recipe, or a
    registered model name); the kind selects the baked entrypoint. ``tag`` adds a
    human alias alongside the always-applied content-addressed
    ``rlmesh-sandbox-<slug>:<hash>`` tag.
    """
    if _is_model_source(source):
        from ._sandbox_model import resolve_model_recipe

        recipe, context_root = resolve_model_recipe(source)
        display, recipe_json, provenance = recipe.name, recipe.to_json(), "installed"
    else:
        display, recipe_json, provenance, context_root = _resolve_recipe_source(
            source, {}, ()
        )
    info = cast(
        _SandboxBuildInfo,
        _sandbox_build_image(
            display,
            tag=tag,
            base_image=base_image,
            rlmesh_package=normalize_rlmesh_package(rlmesh_package),
            packages=_string_sequence("packages", packages),
            trust_remote_code=trust_remote_code,
            allow_unpinned_hf=allow_unpinned_hf,
            recipe_json=recipe_json,
            recipe_provenance=provenance,
            context_root=context_root,
            build_memory=build_memory,
        ),
    )
    return ExportResult(
        requested_source=info["requested_source"],
        resolved_source=info["resolved_source"],
        image=info["image"],
        alias=info["alias"],
        image_id=info["image_id"],
    )


def _is_model_source(source: object) -> bool:
    """Whether ``source`` names a model recipe (vs an env recipe)."""
    from .recipes import Recipe, resolve
    from .recipes._authoring_model import is_model_recipe
    from .recipes._registry import RecipeNotFoundError

    if is_model_recipe(source):
        return True
    if isinstance(source, Recipe):
        return source.kind == "model"
    if isinstance(source, str):
        # An unregistered name is not a model here; let the env path raise.
        try:
            return resolve(source).kind == "model"
        except RecipeNotFoundError:
            return False
    return False


class SandboxSessionBase(Generic[RemoteT]):
    """Experimental base for Docker-backed sandbox sessions.

    A sandbox session owns the isolated environment container and the remote
    client connected to it. Closing the session closes the client and stops the
    container.

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
        num_envs: Number of environment instances to create.
        vectorization_mode: Optional vectorization mode requested in the sandbox.
        **gym_make_kwargs: Keyword arguments forwarded to environment creation.
    """

    _remote_env_cls: ClassVar[Callable[..., object]] = _missing_remote_env_cls
    _source: str | Recipe | type[EnvRecipe]
    _closed: bool
    sandbox: SandboxInfo
    _remote_env: RemoteT

    def __init__(
        self,
        source: str | Recipe | type[EnvRecipe],
        *,
        base_image: str | None = None,
        rlmesh_package: str | PathLike[str] | None = None,
        packages: Sequence[str] | None = None,
        imports: Sequence[str] | None = None,
        trust_remote_code: bool = False,
        allow_unpinned_hf: bool = False,
        num_envs: int = 1,
        vectorization_mode: str | None = None,
        build_memory: str | None = None,
        task: str | None = None,
        config: Mapping[str, object] | str | PathLike[str] | None = None,
        capabilities: Sequence[str] | None = None,
        override: str | PathLike[str] | None = None,
        cwd: str | PathLike[str] | None = None,
        repo_root: str | PathLike[str] | None = None,
        **gym_make_kwargs: object,
    ) -> None:
        resolved_rlmesh_package = _resolve_rlmesh_package(
            rlmesh_package,
            gym_make_kwargs,
        )
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
            rlmesh_package=resolved_rlmesh_package,
            packages=packages,
            imports=imports,
            trust_remote_code=trust_remote_code,
            allow_unpinned_hf=allow_unpinned_hf,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            build_memory=build_memory,
            gym_make_kwargs=gym_make_kwargs,
        )
        try:
            self._remote_env = self._create_remote_env(self.sandbox.address)
        except BaseException:
            try:
                self._shutdown()
            except BaseException:
                pass
            raise

    def _create_remote_env(self, address: str) -> RemoteT:
        remote_env_cls = type(self)._remote_env_cls
        sandbox_factory = getattr(remote_env_cls, "_connect_for_sandbox", None)
        if callable(sandbox_factory):
            return cast(
                RemoteT,
                sandbox_factory(
                    address,
                    connect_timeout_seconds=SANDBOX_REMOTE_CONNECT_TIMEOUT_SECONDS,
                ),
            )
        return cast(
            RemoteT,
            remote_env_cls(address),
        )

    @property
    def source(self) -> str | Recipe | type[EnvRecipe]:
        """Original sandbox source (string or ``Recipe``) requested by the caller."""
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

        # Only mark the session closed once the container is actually stopped.
        # If stopping fails (e.g. a transient Docker daemon error) leave
        # ``_closed`` False so close()/__exit__/__del__ can retry the teardown
        # instead of permanently leaking the container.
        _sandbox_stop_env(container_id=self.sandbox.container_id)
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
        rlmesh_package: Optional RLMesh package, wheel, or ``"local"`` installed
            in the sandbox.
        packages: Extra environment packages installed in the sandbox.
        imports: Import names checked during sandbox startup.
        trust_remote_code: Allow remote environment code to execute.
        allow_unpinned_hf: Allow Hugging Face sources without a pinned revision.
        **gym_make_kwargs: Keyword arguments forwarded to environment creation.
    """

    _remote_env_cls: ClassVar[Callable[[str], object]] = _missing_remote_env_cls

    def __init__(
        self,
        source: str | Recipe | type[EnvRecipe],
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
        _reject_single_env_vector_option(gym_make_kwargs)
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
        rlmesh_package: Optional RLMesh package, wheel, or ``"local"`` installed
            in the sandbox.
        packages: Extra environment packages installed in the sandbox.
        imports: Import names checked during sandbox startup.
        trust_remote_code: Allow remote environment code to execute.
        allow_unpinned_hf: Allow Hugging Face sources without a pinned revision.
        **env_make_kwargs: Keyword arguments forwarded to environment creation.
    """

    _remote_env_cls: ClassVar[Callable[[str], object]] = _missing_remote_env_cls

    def __init__(
        self,
        source: str | Recipe | type[EnvRecipe],
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

    def open_viewer(
        self, *, env_index: int = 0, fps: float | None | str = "env"
    ) -> None:
        """Open a local render viewer for one sandboxed environment."""
        self._remote_env.open_viewer(env_index=env_index, fps=fps)

    def close_viewer(self) -> None:
        """Close the local render viewer if one is open."""
        self._remote_env.close_viewer()


def _resolve_recipe_source(
    source: str | Recipe | type[EnvRecipe],
    gym_make_kwargs: Mapping[str, object] = _NO_MAKE_KWARGS,
    imports: Sequence[str] | None = None,
) -> tuple[str, str | None, str | None, str | None]:
    """Resolve a sandbox source into (display, recipe_json, provenance, context_root).

    An ``EnvRecipe`` subclass or an in-process ``Recipe`` is ``Installed`` -- it
    came from your installed/loaded code (pip-install-is-consent), so its build
    (including ``ProjectInstall``) is trusted. A registered name is ``Installed``
    too. ``Remote`` is reserved for a document handed in from an untrusted external
    source (the future catalog/wire path). A plain id/name that is not a recipe is
    an ordinary gym/hf source string, unchanged. When the recipe stages a project
    tree, ``context_root`` is the recipe's defining-package directory when that can
    be determined, falling back to the current directory.

    ``gym_make_kwargs`` are baked into ``recipe.make.kwargs`` so the in-container
    ``build()`` forwards them to the factory, matching local
    ``rlmesh.make(recipe, **kwargs)``. They are *not* also forwarded via
    ``kwargs_json`` on the recipe path (the recipe bootstrap payload carries only
    ``make.kwargs``), so nothing is applied twice.

    ``imports`` are merged into ``recipe.requires.imports`` for the same reason: the
    recipe bootstrap reads ``requires.imports`` in-container, never the caller's
    ``imports=`` (which only the gym/hf path forwards). Merging keeps a caller's
    registration import (e.g. ``ale_py``) from being silently dropped on the recipe
    path. ``requires.imports`` is meaningless for a ``PyMake``/build-only recipe (the
    py factory owns its own imports), so caller imports on those raise.
    """
    import dataclasses
    import os

    from rlmesh.recipes import (
        HfMake,
        PyMake,
        RecipeNotFoundError,
        UnsupportedRecipeError,
        resolve,
        resolve_from_recipe,
    )
    from rlmesh.recipes._authoring import as_authored_recipe, is_env_recipe
    from rlmesh.recipes._registry import (
        class_origin_dir,
        from_recipe_origin,
        recipe_origin_dir,
    )

    # ``origin`` is the filesystem directory of the code that *defined* the recipe,
    # used to stage a ProjectInstall from the package's own source tree rather than
    # the caller's cwd. ``None`` means we could not determine it.
    origin: str | None = None
    authored = as_authored_recipe(source)
    if authored is not None:
        recipe = authored
        provenance = "installed"
        if is_env_recipe(source):
            # An authored EnvRecipe knows its defining module; stage from there.
            origin = class_origin_dir(source)
    elif isinstance(source, str):
        try:
            recipe = resolve(source)
        except RecipeNotFoundError:
            return source, None, None, None
        provenance = "installed"
        # A name resolves to a recipe registered by some package; prefer that
        # registrant's module directory when the registry recorded it.
        origin = recipe_origin_dir(source)
    else:
        raise TypeError(
            f"sandbox source must be a str, Recipe, or EnvRecipe, got "
            f"{type(source).__name__}"
        )
    # Capture the from_recipe base origin BEFORE inlining: from_recipe is exclusive
    # with other build fields, so when it is set the inlined ProjectInstall always
    # comes from the TERMINAL base (the first ancestor whose build.from_recipe is
    # None), and its `src` is relative to that base's source tree -- not the child's
    # or an intermediate base's. from_recipe_origin walks the chain to that terminal
    # base; we use it below to stage from the right tree.
    from_recipe_base_origin = from_recipe_origin(recipe)
    # Inline any `from_recipe` base build before the wire so a task family shares
    # one image.
    recipe = resolve_from_recipe(recipe)
    # An HfMake recipe materializes its source only via the sandbox HF path, never
    # the recipe path; reject it here so we fail fast instead of after a full image
    # build (the in-container build() would otherwise raise UnsupportedRecipeError).
    if isinstance(recipe.make, HfMake):
        raise UnsupportedRecipeError(
            f"recipe {recipe.name!r} uses HfMake, which materializes its source only "
            "via the sandbox HF path, not the recipe path; pass the HF source string "
            "to SandboxEnv instead"
        )
    # setup.files is not applied anywhere yet (the tempdir-gated file writer has not
    # landed; the in-container build() -> apply_setup raises on it too). Reject it
    # here so we fail fast instead of after a full image build.
    if recipe.setup.files:
        raise UnsupportedRecipeError(
            "setup.files is not applied yet (local or sandbox); remove it and stage "
            "files via the build phase (build.project / build.fetch) instead"
        )
    # Bake any make kwargs into the recipe document so the in-container factory
    # receives them (the recipe bootstrap payload carries make.kwargs, not the
    # gym/hf kwargs_json). A build-only base has no make to carry them.
    if gym_make_kwargs:
        if recipe.make is None:
            raise TypeError(
                f"recipe {recipe.name!r} is a build-only base (make is None); it takes "
                "no environment make kwargs"
            )
        merged_make = dataclasses.replace(
            recipe.make, kwargs={**recipe.make.kwargs, **gym_make_kwargs}
        )
        recipe = dataclasses.replace(recipe, make=merged_make)
    # Merge any caller imports into the recipe document's requires.imports so the
    # in-container bootstrap runs them (it reads requires.imports, never the caller's
    # imports=). requires.imports is forbidden/meaningless for a PyMake or build-only
    # base -- the py factory owns its own imports -- so reject caller imports there
    # instead of silently dropping them.
    if imports:
        if recipe.make is None or isinstance(recipe.make, PyMake):
            raise TypeError(
                f"recipe {recipe.name!r} is a PyMake/build-only recipe; imports= does "
                "not apply -- a py factory performs its own imports"
            )
        # De-duplicate while preserving order: the recipe's own imports first, then
        # the caller's new entries.
        merged_imports = list(recipe.requires.imports)
        for name in imports:
            if name not in merged_imports:
                merged_imports.append(name)
        merged_requires = dataclasses.replace(recipe.requires, imports=merged_imports)
        recipe = dataclasses.replace(recipe, requires=merged_requires)
    if recipe.build.project is None:
        context_root = None
    else:
        # When the build came from a `from_recipe` chain, the inlined project.src is
        # relative to the TERMINAL base's source tree, not the child's or an
        # intermediate base's -- so prefer that terminal base's recorded origin.
        # (from_recipe is exclusive with other build fields, so the ProjectInstall is
        # always the terminal base's.) Fall back to the child's origin only when the
        # terminal base has no recorded origin.
        if from_recipe_base_origin is not None:
            origin = from_recipe_base_origin
        # Stage a ProjectInstall from the recipe's defining package when we know it,
        # falling back to the caller's cwd otherwise (e.g. a plain in-process Recipe
        # instance assembled at the call site has no determinable origin -- the cwd
        # is then the only host tree we can reasonably stage from).
        context_root = origin if origin is not None else os.getcwd()
    return recipe.name, recipe.to_json(), provenance, context_root


def _start_sandbox(
    source: str | Recipe | type[EnvRecipe],
    *,
    base_image: str | None,
    rlmesh_package: str | None,
    packages: Sequence[str] | None,
    imports: Sequence[str] | None,
    trust_remote_code: bool,
    allow_unpinned_hf: bool,
    num_envs: int,
    vectorization_mode: str | None,
    build_memory: str | None = None,
    gym_make_kwargs: Mapping[str, object],
) -> SandboxInfo:
    display, recipe_json, provenance, context_root = _resolve_recipe_source(
        source, gym_make_kwargs, imports
    )
    # Only forward the recipe arguments when this is actually a recipe source, so
    # the gym/hf path stays byte-identical to before.
    recipe_kwargs: dict[str, str] = {}
    # On the recipe path the caller imports are merged into the document's
    # requires.imports by _resolve_recipe_source (the bootstrap reads requires.imports,
    # not the imports= channel), so do not also forward them via _sandbox_start_env --
    # the merged document is their sole carrier. The gym/hf path keeps forwarding
    # imports exactly as before.
    forwarded_imports = imports
    if recipe_json is not None and provenance is not None:
        recipe_kwargs["recipe_json"] = recipe_json
        recipe_kwargs["recipe_provenance"] = provenance
        if context_root is not None:
            recipe_kwargs["context_root"] = context_root
        # On the recipe path the make kwargs are baked into the document's
        # make.kwargs by _resolve_recipe_source (the recipe bootstrap payload
        # carries make.kwargs, not kwargs_json), so do not also ship kwargs_json --
        # the document is their sole carrier. The gym/hf path below keeps shipping
        # kwargs_json exactly as before.
        kwargs_json: str | None = None
        forwarded_imports = None
    else:
        kwargs_json = json.dumps(gym_make_kwargs) if gym_make_kwargs else None
    started = cast(
        _SandboxStartInfo,
        _sandbox_start_env(
            display,
            base_image=base_image,
            rlmesh_package=rlmesh_package,
            packages=_string_sequence("packages", packages),
            imports=_string_sequence("imports", forwarded_imports),
            kwargs_json=kwargs_json,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            trust_remote_code=trust_remote_code,
            allow_unpinned_hf=allow_unpinned_hf,
            build_memory=build_memory,
            **recipe_kwargs,
        ),
    )
    return SandboxInfo(
        requested_source=started["requested_source"],
        resolved_source=started["resolved_source"],
        address=started["address"],
        container_id=started["container_id"],
    )


def _string_sequence(name: str, value: Sequence[str] | None) -> list[str]:
    """Normalize a package/import sequence, rejecting a bare ``str``.

    A bare ``str`` satisfies ``Sequence[str]`` but iterating it yields single
    characters, which would silently forward one-letter package or import names
    to the sandbox. Require an explicit list/tuple of names instead.
    """
    if value is None:
        return []
    if isinstance(value, str):
        raise TypeError(
            f"{name}= expects a sequence of strings, not a bare str; "
            f"pass [{value!r}] for a single entry"
        )
    return list(value)


def _reject_removed_option(name: str, value: object) -> None:
    if value is not None:
        raise TypeError(
            f"SandboxEnv no longer supports {name}=...; use base_image=, "
            "rlmesh_package=, packages=, imports=, and environment make kwargs"
        )


def normalize_rlmesh_package(value: str | PathLike[str] | None) -> str | None:
    if value is None:
        return None
    return fspath(value)


def _resolve_rlmesh_package(
    rlmesh_package: str | PathLike[str] | None,
    gym_make_kwargs: dict[str, object],
) -> str | None:
    package_spec = gym_make_kwargs.pop("package_spec", None)
    if package_spec is not None:
        if rlmesh_package is not None:
            raise TypeError(
                "SandboxEnv got both rlmesh_package=... and package_spec=...; "
                "use rlmesh_package=..."
            )
        rlmesh_package = cast(str | PathLike[str], package_spec)
    return normalize_rlmesh_package(rlmesh_package)


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
