"""Docker-backed sandbox container lifecycle, shared by the env sessions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from typing import TypedDict, cast

from .._rlmesh import sandbox_start_env as _sandbox_start_env
from .._rlmesh import sandbox_stop_env as _sandbox_stop_env
from ._package import normalize_rlmesh_package

SANDBOX_REMOTE_CONNECT_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class SandboxInfo:
    """Information about a running RLMesh sandbox container."""

    requested_source: str
    resolved_source: str
    address: str
    container_id: str


class _SandboxStartInfo(TypedDict):
    requested_source: str
    resolved_source: str
    address: str
    container_id: str


class SandboxLifecycle:
    """Container-lifecycle mixin for Docker-backed env sessions.

    Combined with a ``Remote*EnvBase`` (which provides the reset/step/contract surface
    and the client this session attaches to its container): a sandbox env *is* a remote
    env that also owns an isolated container. Closing it detaches the client and stops
    the container. The concrete subclass supplies :meth:`_detach` (its remote base's
    ``close``) and starts the container in ``__init__`` before attaching ``self``.
    """

    _source: str
    _closed: bool
    sandbox: SandboxInfo

    def _detach(self) -> None:
        """Detach the remote client; supplied by the ``Remote*EnvBase`` mixed in."""
        raise NotImplementedError

    @property
    def source(self) -> str:
        """Original sandbox source string requested by the caller."""
        return self._source

    def close(self) -> None:
        """Detach the remote client and stop the owned sandbox container."""
        self._stop()

    def _stop(self) -> None:
        if self._closed:
            return
        sandbox = getattr(self, "sandbox", None)
        remote_error: BaseException | None = None
        try:
            self._detach()
        except BaseException as exc:  # best effort: still stop the container
            remote_error = exc
        # Only mark the session closed once the container is actually stopped. If
        # stopping fails (e.g. a transient Docker daemon error) leave ``_closed``
        # False so close()/__exit__/__del__ can retry instead of leaking the container.
        if sandbox is not None:
            _sandbox_stop_env(container_id=sandbox.container_id)
        self._closed = True
        if remote_error is not None:
            raise remote_error

    def __del__(self) -> None:
        try:
            self._stop()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"source={self._source!r}, "
            f"address={self.sandbox.address!r}, "
            f"container_id={self.sandbox.container_id!r})"
        )


def start_sandbox_container(
    source: str,
    *,
    base_image: str | None,
    rlmesh_package: str | PathLike[str] | None,
    packages: Sequence[str] | None,
    imports: Sequence[str] | None,
    trust_remote_code: bool,
    allow_unpinned_hf: bool,
    num_envs: int,
    vectorization_mode: str | None,
    build_memory: str | None,
    task: str | None,
    config: Mapping[str, object] | str | PathLike[str] | None,
    capabilities: Sequence[str] | None,
    override: str | PathLike[str] | None,
    cwd: str | PathLike[str] | None,
    repo_root: str | PathLike[str] | None,
    gym_make_kwargs: dict[str, object],
) -> SandboxInfo:
    """Resolve the package spec, reject removed options, and start the container.

    The shared ``__init__`` prelude for the sandbox env sessions.
    """
    resolved_rlmesh_package = _resolve_rlmesh_package(rlmesh_package, gym_make_kwargs)
    _reject_removed_option("task", task)
    _reject_removed_option("config", config)
    _reject_removed_option("capabilities", capabilities)
    _reject_removed_option("override", override)
    _reject_removed_option("cwd", cwd)
    _reject_removed_option("repo_root", repo_root)
    return _start_sandbox(
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


def _start_sandbox(
    source: str,
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
    kwargs_json = json.dumps(gym_make_kwargs) if gym_make_kwargs else None
    started = cast(
        _SandboxStartInfo,
        _sandbox_start_env(
            source,
            base_image=base_image,
            rlmesh_package=rlmesh_package,
            packages=string_sequence("packages", packages),
            imports=string_sequence("imports", imports),
            kwargs_json=kwargs_json,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            trust_remote_code=trust_remote_code,
            allow_unpinned_hf=allow_unpinned_hf,
            build_memory=build_memory,
        ),
    )
    return SandboxInfo(**started)


def string_sequence(name: str, value: Sequence[str] | None) -> list[str]:
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


def reject_single_env_vector_option(kwargs: Mapping[str, object]) -> None:
    for name in ("num_envs", "vectorization_mode"):
        if name in kwargs:
            raise TypeError(
                f"SandboxEnv is single-env only; use SandboxVectorEnv for {name}=..."
            )


__all__ = [
    "SANDBOX_REMOTE_CONNECT_TIMEOUT_SECONDS",
    "SandboxInfo",
    "SandboxLifecycle",
    "reject_single_env_vector_option",
    "start_sandbox_container",
    "string_sequence",
]
