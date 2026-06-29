"""Docker-backed sandbox container lifecycle, shared by the env sessions."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from os import PathLike, fspath
from typing import TYPE_CHECKING, TypedDict, cast

from .._rlmesh import sandbox_start_env as _sandbox_start_env
from .._rlmesh import sandbox_stop_env as _sandbox_stop_env
from ._sources import resolve_source_kind

if TYPE_CHECKING:
    from .._client._endpoint import Transport

#: Default seconds a sandbox env client waits for its container to become
#: reachable. The server only binds its port after the env factory's ``make()``
#: runs, so a heavy env (large sim/asset load -- e.g. a LIBERO task suite takes
#: ~12s) needs headroom; matches ``SandboxModel``'s 30s default. Overridable per
#: construction via ``SandboxEnv(..., connect_timeout_seconds=...)``.
SANDBOX_REMOTE_CONNECT_TIMEOUT_SECONDS = 30.0

#: Port an rlmesh-serving container binds inside the container (the templated
#: entrypoint default, ``RLMESH_ADDRESS=0.0.0.0:50051``).
CONTAINER_SERVE_PORT = 50051


def normalize_gpus(gpus: str | int | None) -> str | None:
    """Normalize the ``gpus`` knob into a docker ``--gpus`` value, or None.

    Passes straight through to ``docker run --gpus``: ``"all"``, a count
    (``2`` / ``"2"``), or a device selector (``"device=0,1"``). Shared by the
    model and env prebuilt paths so the two never drift.
    """
    if gpus is None:
        return None
    value = str(gpus).strip()
    if not value:
        raise ValueError("gpus= must be non-empty, e.g. 'all', 1, or 'device=0,1'")
    return value


@dataclass(frozen=True)
class SandboxBuild:
    """Build-from-source infrastructure for a sandbox image.

    Configures how the image is *built* from a ``gym://``/``hf://`` source (base
    image, extra packages, the rlmesh pin, ...). Meaningless for a prebuilt image
    (run, not built) -- those fields are ignored with a warning. Pair with
    :class:`SandboxRuntime` for ``docker run`` settings. Env/model construction
    params stay in the sandbox's ``**params`` (the ``make``/``load`` binding).
    """

    base_image: str | None = None
    rlmesh_package: str | PathLike[str] | None = None
    packages: Sequence[str] | None = None
    imports: Sequence[str] | None = None
    trust_remote_code: bool = False
    allow_unpinned_hf: bool = False
    build_memory: str | None = None


@dataclass(frozen=True)
class SandboxRuntime:
    """Container run-time settings for a sandbox -- ``docker run`` flags.

    Applies whenever the container is *run* (prebuilt images and build-then-run),
    unlike :class:`SandboxBuild` which only configures the from-source build. The
    host-resource knobs sim envs/models need:

    - ``gpus``: ``docker run --gpus`` (``"all"`` / a count / ``"device=0,1"``) --
      legacy GPU injection (CUDA compute only). NOTE: this does *not* inject the
      graphics/Vulkan driver; for SAPIEN/Vulkan use ``devices`` with a CDI ref.
    - ``devices``: ``docker run --device`` entries, e.g. ``["nvidia.com/gpu=all"]``
      (a CDI device ref -- the full graphics+compute driver) or a ``/dev`` node.
    - ``volumes``: ``docker run -v`` mounts, e.g.
      ``["/host/assets:/ctr/assets", "/host/ro:/ctr/ro:ro"]``.

    Build-from-source attaches none of these (it has no ``docker run`` step); set
    them only for a prebuilt image source.
    """

    gpus: str | int | None = None
    devices: Sequence[str] | None = None
    volumes: Sequence[str] | None = None


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

    if TYPE_CHECKING:
        # Supplied by the ``Remote*EnvBase`` mixed in (declared here so :meth:`_attach`
        # can dial through it); no runtime body, so the real one is not shadowed.
        def _initialize(
            self,
            address: str | None = None,
            *,
            host: str | None = None,
            port: int | None = None,
            path: str | None = None,
            transport: Transport | None = None,
            connect_timeout_seconds: float | None,
        ) -> None: ...

    def _detach(self) -> None:
        """Detach the remote client; supplied by the ``Remote*EnvBase`` mixed in."""
        raise NotImplementedError

    def _attach(self, connect_timeout_seconds: float) -> None:
        """Attach the client to the started container, retrying while it boots.

        The rlmesh server only binds its port after the env factory's ``make()`` runs in
        the container, and the native client dials once and fails fast against a
        not-yet-listening port -- so the wait-for-ready loop lives here (mirroring
        ``SandboxModel._dial_with_retry``), not in the timeout handed to the native
        client. Stops the container on any failure so it is never leaked, and surfaces
        the container's recent logs when it exits or never becomes ready, instead of a
        bare ``transport error``.
        """
        sandbox = self.sandbox
        deadline = time.monotonic() + connect_timeout_seconds
        try:
            while True:
                try:
                    self._initialize(
                        sandbox.address, connect_timeout_seconds=connect_timeout_seconds
                    )
                    return
                except (
                    _TRANSIENT_DIAL_ERRORS
                ) as exc:  # the container may still be starting
                    short_id = sandbox.container_id[:12]
                    # A dial error against an already-exited container is terminal: fail
                    # fast with its logs rather than retrying for the whole timeout.
                    if not _container_running(sandbox.container_id):
                        raise RuntimeError(
                            f"sandbox container {short_id} for {self._source!r} exited "
                            f"before becoming ready; recent logs:\n"
                            f"{_container_logs(sandbox.container_id)}"
                        ) from exc
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"sandbox container {short_id} for {self._source!r} did not "
                            f"become ready within {connect_timeout_seconds:.0f}s; recent "
                            f"logs:\n{_container_logs(sandbox.container_id)}"
                        ) from exc
                    time.sleep(0.1)
        except BaseException:
            try:
                _sandbox_stop_env(container_id=sandbox.container_id)
            except BaseException:
                pass
            self._closed = True
            raise

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
    build: SandboxBuild | None,
    runtime: SandboxRuntime | None,
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

    :class:`SandboxBuild` configures the from-source build; :class:`SandboxRuntime`
    (``gpus`` / ``devices`` / ``volumes``) configures ``docker run`` and applies to
    the prebuilt path only -- the native build-from-source path has no ``docker run``
    step and rejects a set runtime flag.
    """
    build = build or SandboxBuild()
    run = runtime or SandboxRuntime()
    gpus = normalize_gpus(run.gpus)
    devices = string_sequence("devices", run.devices)
    volumes = string_sequence("volumes", run.volumes)
    kind, resolved = resolve_source_kind(source)
    if kind == "prebuilt":
        _warn_ignored_build_options(build)
        return start_prebuilt_container(
            resolved,
            requested_source=source,
            binding=binding,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            gpus=gpus,
            devices=devices,
            volumes=volumes,
        )
    if gpus is not None or devices or volumes:
        raise ValueError(
            "SandboxRuntime (gpus/devices/volumes) is only supported for a prebuilt "
            f"sandbox image; building an env from source ({source!r}) has no docker-run "
            "step -- pass a prebuilt image (docker://img / bare img:tag) instead"
        )
    return _start_build(
        source,
        build=build,
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
    devices: Sequence[str] | None = None,
    volumes: Sequence[str] | None = None,
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
        devices=devices,
        volumes=volumes,
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
    devices: Sequence[str] | None = None,
    volumes: Sequence[str] | None = None,
) -> list[str]:
    """Build the ``docker run`` argv for a prebuilt sandbox container (pure).

    The one place container hardening lives (``--cap-drop ALL``,
    ``--security-opt no-new-privileges``, owner labels for orphan reaping), shared
    by the env and model prebuilt paths so the flags never drift apart. The image
    is always last and env vars precede it, so callers/tests can rely on argv shape.
    ``devices`` -> ``--device`` (incl. CDI refs like ``nvidia.com/gpu=all``) and
    ``volumes`` -> ``-v`` mounts, both before the env vars.
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
    for device in devices or []:
        cmd += ["--device", device]
    for volume in volumes or []:
        cmd += ["-v", volume]
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


def _warn_ignored_build_options(options: SandboxBuild) -> None:
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
    build: SandboxBuild,
    num_envs: int,
    vectorization_mode: str | None,
    binding: Mapping[str, object],
) -> SandboxInfo:
    rlmesh_package = (
        fspath(build.rlmesh_package) if build.rlmesh_package is not None else None
    )
    kwargs_json = json.dumps(dict(binding)) if binding else None
    started = cast(
        _SandboxStartInfo,
        _sandbox_start_env(
            source,
            base_image=build.base_image,
            rlmesh_package=rlmesh_package,
            packages=string_sequence("packages", build.packages),
            imports=string_sequence("imports", build.imports),
            kwargs_json=kwargs_json,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            trust_remote_code=build.trust_remote_code,
            allow_unpinned_hf=build.allow_unpinned_hf,
            build_memory=build.build_memory,
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
    """Reject construction params that collide with a sandbox build/runtime field.

    These names (``base_image``, ``packages``, ``gpus``, ``devices``, ``volumes``,
    ...) configure how the container is built or run, not the env/model. Letting
    them fall into ``**params`` (the make/load binding) would silently drop a
    setting and forward it as a bogus construction kwarg, so fail loud and point at
    ``build=``/``runtime=``. Derived from the dataclasses so the two never drift.
    """
    reserved = {f.name for f in fields(SandboxBuild)} | {
        f.name for f in fields(SandboxRuntime)
    }
    collisions = sorted(reserved & set(params))
    if collisions:
        raise TypeError(
            f"{', '.join(collisions)} configure the sandbox container build/run, not "
            "the env/model construction params; pass them via build=SandboxBuild(...) "
            "or runtime=SandboxRuntime(...)"
        )


__all__ = [
    "CONTAINER_SERVE_PORT",
    "SANDBOX_REMOTE_CONNECT_TIMEOUT_SECONDS",
    "SandboxBuild",
    "SandboxInfo",
    "SandboxLifecycle",
    "SandboxRuntime",
    "normalize_gpus",
    "parse_published_port",
    "prebuilt_run_cmd",
    "reap_orphans",
    "reject_sandbox_option_params",
    "reject_single_env_vector_option",
    "start_prebuilt_container",
    "start_sandbox_container",
    "string_sequence",
]
