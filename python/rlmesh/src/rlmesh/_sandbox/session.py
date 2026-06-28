"""Docker-backed sandbox container lifecycle, shared by the env sessions."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from os import PathLike, fspath
from typing import TypedDict, cast

from .._rlmesh import sandbox_start_env as _sandbox_start_env
from .._rlmesh import sandbox_stop_env as _sandbox_stop_env
from ._sources import resolve_source_kind

#: Default seconds a sandbox env client waits for its container to become
#: reachable. The server only binds its port after the env factory's ``make()``
#: runs, so a heavy env (large sim/asset load -- e.g. a LIBERO task suite takes
#: ~12s) needs headroom; matches ``SandboxModel``'s 30s default. Overridable per
#: construction via ``SandboxEnv(..., connect_timeout_seconds=...)``.
SANDBOX_REMOTE_CONNECT_TIMEOUT_SECONDS = 30.0

#: Port an rlmesh-serving container binds inside the container (the templated
#: entrypoint default, ``RLMESH_ADDRESS=0.0.0.0:50051``).
CONTAINER_SERVE_PORT = 50051


@dataclass(frozen=True)
class SandboxOptions:
    """Build/run infrastructure for a sandbox container -- the single reserved knob.

    Everything here configures how the container is *built or run*, never the
    environment/model itself: env/model construction params are the sandbox's
    ``**params`` (the binding forwarded to ``make``/``load``). For a prebuilt
    image (run, not built) the build-only fields are meaningless and ignored with
    a warning.
    """

    base_image: str | None = None
    rlmesh_package: str | PathLike[str] | None = None
    packages: Sequence[str] | None = None
    imports: Sequence[str] | None = None
    trust_remote_code: bool = False
    allow_unpinned_hf: bool = False
    build_memory: str | None = None


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
    options: SandboxOptions | None,
    num_envs: int,
    vectorization_mode: str | None,
    binding: Mapping[str, object],
) -> SandboxInfo:
    """Resolve the source, then build-from-source or run a prebuilt image.

    The shared ``__init__`` prelude for the sandbox env sessions. A ``gym://``/
    ``hf://`` (or bare gym id) source is built from source via the native sandbox
    builder; a ``docker://``/``image://`` or bare image-shaped source is run
    directly as a prebuilt rlmesh-serving image, with the binding injected as
    ``RLMESH_MAKE_KWARGS`` (no build, no rlmesh pin).
    """
    opts = options or SandboxOptions()
    kind, resolved = resolve_source_kind(source)
    if kind == "prebuilt":
        _warn_ignored_build_options(opts)
        return start_prebuilt_container(
            resolved,
            requested_source=source,
            binding=binding,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
        )
    return _start_build(
        source,
        options=opts,
        num_envs=num_envs,
        vectorization_mode=vectorization_mode,
        binding=binding,
    )


def start_prebuilt_container(
    image: str,
    *,
    requested_source: str,
    binding: Mapping[str, object],
    num_envs: int | None = None,
    vectorization_mode: str | None = None,
    gpus: str | None = None,
) -> SandboxInfo:
    """Run a prebuilt rlmesh-serving image and return its connection info.

    The image runs as-is (its own ``CMD``); the binding and eval shape are injected
    as environment variables (``RLMESH_MAKE_KWARGS`` etc.), the serve port is
    published on a Docker-assigned host port, and the host port is read back. No
    build and no rlmesh pin -- the image is whatever the publisher baked.

    ``num_envs`` / ``vectorization_mode`` are the vectorized-env eval shape: when
    left ``None`` (the model path -- a model is never vectorized) the corresponding
    ``RLMESH_NUM_ENVS`` / ``RLMESH_VECTORIZATION_MODE`` env vars are not injected, so
    the container serves in its default (single, model) mode.
    """
    if shutil.which("docker") is None:
        raise RuntimeError(
            "Docker CLI not found on PATH; install Docker to run a prebuilt "
            "sandbox image, or use gym://... to build from source"
        )
    reap_orphans()
    env_vars: dict[str, str] = {}
    if binding:
        env_vars["RLMESH_MAKE_KWARGS"] = json.dumps(dict(binding), sort_keys=True)
    if num_envs and num_envs != 1:
        env_vars["RLMESH_NUM_ENVS"] = str(num_envs)
    if vectorization_mode:
        env_vars["RLMESH_VECTORIZATION_MODE"] = vectorization_mode

    cmd = prebuilt_run_cmd(
        image,
        env_vars=env_vars,
        gpus=gpus,
        container_port=CONTAINER_SERVE_PORT,
        owner_pid=os.getpid(),
        owner_pid_ns=pid_namespace_id(),
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"failed to start prebuilt container ({proc.returncode}):\n{proc.stderr}"
        )
    container_id = proc.stdout.strip()
    try:
        port = _resolve_published_port(container_id)
    except BaseException:
        # Port discovery failed after the container started: stop it before
        # re-raising so the container is not leaked.
        try:
            _sandbox_stop_env(container_id=container_id)
        except Exception:
            pass
        raise
    return SandboxInfo(
        requested_source=requested_source,
        resolved_source=f"docker://{image}",
        address=f"127.0.0.1:{port}",
        container_id=container_id,
    )


def prebuilt_run_cmd(
    image: str,
    *,
    env_vars: Mapping[str, str],
    gpus: str | None,
    container_port: int,
    owner_pid: int,
    owner_pid_ns: str | None,
) -> list[str]:
    """Build the ``docker run`` argv for a prebuilt sandbox container (pure).

    The one place container hardening lives (``--cap-drop ALL``,
    ``--security-opt no-new-privileges``, owner labels for orphan reaping), shared
    by the env and model prebuilt paths so the flags never drift apart. The image
    is always last and env vars precede it, so callers/tests can rely on argv shape.
    """
    cmd = [
        "docker",
        "run",
        "-d",
        "-p",
        f"127.0.0.1:0:{container_port}",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--label",
        "rlmesh.sandbox=1",
        "--label",
        f"rlmesh.sandbox.owner-pid={owner_pid}",
    ]
    if owner_pid_ns is not None:
        cmd += ["--label", f"rlmesh.sandbox.owner-pid-ns={owner_pid_ns}"]
    if gpus is not None:
        cmd += ["--gpus", gpus]
    for key, value in env_vars.items():
        cmd += ["-e", f"{key}={value}"]
    cmd.append(image)
    return cmd


def parse_published_port(stdout: str, container_port: int) -> int:
    """Parse the host port from ``docker port`` output, or raise.

    The host port is the last ``:``-separated field of any mapping line, so an
    IPv6 binding like ``[::]:51000`` parses, not just IPv4 prefixes.
    """
    for line in stdout.splitlines():
        _, sep, port = line.strip().rpartition(":")
        if sep and port.isdigit():
            return int(port)
    raise RuntimeError(
        f"container published no host port for {container_port}; "
        f"docker port output:\n{stdout}"
    )


def _resolve_published_port(container_id: str) -> int:
    proc = subprocess.run(
        ["docker", "port", container_id, str(CONTAINER_SERVE_PORT)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"failed to read published port for container "
            f"({proc.returncode}):\n{proc.stderr}"
        )
    return parse_published_port(proc.stdout, CONTAINER_SERVE_PORT)


def reap_orphans() -> None:
    """Best-effort sweep of containers orphaned by a prior hard kill.

    Delegates to the native ``sandbox_reap_orphans`` reaper. The getattr guard
    degrades to a no-op against a version-skewed native extension that predates
    the reaper; failures are swallowed -- reaping is hygiene, never a reason to
    fail a start.
    """
    import rlmesh._rlmesh as _rlmesh

    reap = getattr(_rlmesh, "sandbox_reap_orphans", None)
    if reap is None:
        return
    try:
        reap()
    except Exception:
        pass


def pid_namespace_id() -> str | None:
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return None


def _warn_ignored_build_options(options: SandboxOptions) -> None:
    """Warn when build-only options are set on a run-prebuilt (not built) source."""
    ignored = [
        name
        for name in (
            "base_image",
            "rlmesh_package",
            "packages",
            "imports",
            "build_memory",
        )
        if getattr(options, name) is not None
    ]
    if ignored:
        import warnings

        warnings.warn(
            "prebuilt image source ignores build options "
            f"{', '.join(ignored)} (the image is already built)",
            stacklevel=3,
        )


def _start_build(
    source: str,
    *,
    options: SandboxOptions,
    num_envs: int,
    vectorization_mode: str | None,
    binding: Mapping[str, object],
) -> SandboxInfo:
    rlmesh_package = (
        fspath(options.rlmesh_package) if options.rlmesh_package is not None else None
    )
    kwargs_json = json.dumps(dict(binding)) if binding else None
    started = cast(
        _SandboxStartInfo,
        _sandbox_start_env(
            source,
            base_image=options.base_image,
            rlmesh_package=rlmesh_package,
            packages=string_sequence("packages", options.packages),
            imports=string_sequence("imports", options.imports),
            kwargs_json=kwargs_json,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            trust_remote_code=options.trust_remote_code,
            allow_unpinned_hf=options.allow_unpinned_hf,
            build_memory=options.build_memory,
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


def reject_single_env_vector_option(params: Mapping[str, object]) -> None:
    for name in ("num_envs", "vectorization_mode"):
        if name in params:
            raise TypeError(
                f"SandboxEnv is single-env only; use SandboxVectorEnv for {name}=..."
            )


def reject_sandbox_option_params(params: Mapping[str, object]) -> None:
    """Reject construction params that collide with a :class:`SandboxOptions` field.

    These names (``trust_remote_code``, ``base_image``, ``packages``, ...) configure
    how the container is built/run, not the env/model. Letting them fall into
    ``**params`` (the make/load binding) would silently drop a security flag or a
    build setting and forward it as a bogus construction kwarg, so fail loud and
    point at ``options=``. Derived from the dataclass so the two never drift.
    """
    collisions = sorted({f.name for f in fields(SandboxOptions)} & set(params))
    if collisions:
        raise TypeError(
            f"{', '.join(collisions)} configure the sandbox container build/run, "
            "not the env/model construction params; pass them via "
            f"options=SandboxOptions({collisions[0]}=...)"
        )


__all__ = [
    "CONTAINER_SERVE_PORT",
    "SANDBOX_REMOTE_CONNECT_TIMEOUT_SECONDS",
    "SandboxInfo",
    "SandboxLifecycle",
    "SandboxOptions",
    "parse_published_port",
    "prebuilt_run_cmd",
    "reap_orphans",
    "reject_sandbox_option_params",
    "reject_single_env_vector_option",
    "start_prebuilt_container",
    "start_sandbox_container",
    "string_sequence",
]
