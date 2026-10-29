"""Docker-backed sandbox container lifecycle, shared by the env sessions."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from os import PathLike, fspath
from typing import TYPE_CHECKING, TypedDict, cast

from .._client._endpoint import redact_address
from .._rlmesh import sandbox_host_identity as _sandbox_host_identity
from .._rlmesh import sandbox_session_ports as _sandbox_session_ports
from .._rlmesh import sandbox_start_env as _sandbox_start_env
from .._rlmesh import sandbox_stop_env as _sandbox_stop_env
from ._sources import (
    docker_image_exists,
    docker_pull,
    resolve_source_kind,
    run_docker,
)

if TYPE_CHECKING:
    from .._client._endpoint import Transport

#: Default seconds a sandbox waits for its container to start serving. The
#: server only joins its session after the env factory's ``make()`` (or the
#: model's ``load()``) runs, so a heavy env (large sim/asset load -- e.g. a
#: LIBERO task suite takes ~12s) needs headroom. Overridable per construction
#: via ``SandboxEnv(..., connect_timeout_seconds=...)``.
SANDBOX_REMOTE_CONNECT_TIMEOUT_SECONDS = 30.0

#: The ports a session host listens on, one per traffic class, consecutive.
SESSION_PORTS = 4
#: The variable a container's session host takes its keys from
#: (``rlmesh_session::address::HOST_IDENTITY_VAR``).
HOST_IDENTITY_VAR = "RLMESH_HOST_IDENTITY"

#: The longest one readiness attempt waits, so an image that will never
#: answer is told apart between attempts.
_READY_ATTEMPT_SECONDS = 1.0
#: Held from picking a container's free ports until ``docker run`` binds
#: them, so sandboxes started from several threads never pick the same ones.
_PORTS_LOCK = threading.Lock()

#: An HTTP/2 client preface and an empty SETTINGS frame: what a v1 (gRPC)
#: server answers with a SETTINGS frame of its own.
_H2_PREFACE = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n" + bytes([0, 0, 0, 4, 0, 0, 0, 0, 0])
_H2_SETTINGS = 0x04


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


def normalize_user(user: str | int | None) -> str | None:
    """Normalize the ``user`` knob into a docker ``--user`` value, or None.

    Passes straight through to ``docker run --user``: an int uid (``1000``) or a
    ``"uid[:gid]"`` string (``"1000:1000"``). Shared by the model and env
    prebuilt paths so the two never drift.
    """
    if user is None:
        return None
    if isinstance(user, int):
        if user < 0:
            raise ValueError(f"user= must be a uid >= 0, got {user}")
        return str(user)
    value = user.strip()
    if not value:
        raise ValueError("user= must be non-empty, e.g. 1000 or '1000:1000'")
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
    - ``user``: ``docker run --user`` passthrough -- an int uid or a
      ``"uid[:gid]"`` string. Set it to the owner of writable volume mounts
      (typically ``os.getuid()``); the container runs ``--cap-drop ALL``, so
      even root inside it obeys the mount's permission bits.

    Build-from-source attaches none of these (it has no ``docker run`` step); set
    them only for a prebuilt image source.
    """

    gpus: str | int | None = None
    devices: Sequence[str] | None = None
    volumes: Sequence[str] | None = None
    user: str | int | None = None


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


# Only these are retried while the container boots: a refused/unavailable
# connection or a connect timeout.
_TRANSIENT_DIAL_ERRORS = (ConnectionError, TimeoutError, OSError)


def _base(address: str) -> tuple[str, int]:
    """The host and base port of a sandbox's ``host:port[#key]/name`` address."""
    host, _, port = address.split("/", 1)[0].split("#", 1)[0].rpartition(":")
    return host, int(port)


def _accepts(address: str) -> bool:
    """Whether something listens on the base port behind ``address``.

    Until then, joining would host the session here instead of reaching the
    container's.
    """
    try:
        with socket.create_connection(_base(address), timeout=0.5):
            return True
    except OSError:
        return False


def _answers_v1(address: str) -> bool:
    """Whether the container behind ``address`` answers the v1 gRPC wire.

    An image built with rlmesh 0.1 serves gRPC (HTTP/2) where a 0.2 one hosts
    its session: it answers an HTTP/2 preface with a SETTINGS frame.
    """
    try:
        with socket.create_connection(_base(address), timeout=0.5) as conn:
            conn.sendall(_H2_PREFACE)
            head = conn.recv(9)
    except OSError:
        return False
    return len(head) == 9 and head[3] == _H2_SETTINGS


def wait_serving(
    sandbox: SandboxInfo, timeout: float, ready: Callable[[float], None]
) -> None:
    """Wait up to ``timeout`` seconds for the container to serve, or raise.

    ``ready(attempt_timeout)`` connects to what it serves, raising a dial
    error until it answers. An image built with rlmesh 0.1 (it answers the
    v1 wire) is refused naming it, and a dial error against an exited
    container fails fast with its recent logs; an inspect failure is reported
    as such instead of being misread as an exit.
    """
    deadline = time.monotonic() + timeout
    short_id = sandbox.container_id[:12]
    source = sandbox.requested_source
    while True:
        attempt = min(_READY_ATTEMPT_SECONDS, max(deadline - time.monotonic(), 0.1))
        try:
            if not _accepts(sandbox.address):
                raise ConnectionError(
                    f"nothing listens at {redact_address(sandbox.address)} yet"
                )
            ready(attempt)
            return
        except _TRANSIENT_DIAL_ERRORS as exc:
            if _answers_v1(sandbox.address):
                raise ValueError(
                    f"{source!r} was built with rlmesh 0.1; rebuild it with rlmesh "
                    "0.2: a 0.2 sandbox joins the session its container hosts, and "
                    "this image serves the v1 wire instead"
                ) from exc
            running = _container_running(sandbox.container_id)
            if running is None:
                raise RuntimeError(
                    f"could not inspect sandbox container {short_id} for "
                    f"{source!r} (docker daemon busy?); last dial error: {exc}"
                ) from exc
            if not running:
                raise RuntimeError(
                    f"sandbox container {short_id} for {source!r} exited before "
                    f"becoming ready; recent logs:\n"
                    f"{_container_logs(sandbox.container_id)}"
                ) from exc
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"sandbox container {short_id} for {source!r} did not become "
                    f"ready within {timeout:.0f}s; recent logs:\n"
                    f"{_container_logs(sandbox.container_id)}"
                ) from exc
            time.sleep(0.1)


def peer_session(sandbox: SandboxInfo) -> str:
    """The sandbox's session as another sandbox container reaches it.

    Its ports publish on the host's loopback, which no other container
    reaches, so a container joins at this one's address on their shared
    Docker network, on the same port numbers (they match inside).
    """
    proc = run_docker(
        [
            "docker",
            "inspect",
            "-f",
            "{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}",
            sandbox.container_id,
        ]
    )
    ips = proc.stdout.split() if proc.returncode == 0 else []
    if not ips:
        raise RuntimeError(
            f"sandbox container {sandbox.container_id[:12]} has no network "
            f"address another container can reach: {proc.stderr.strip()}"
        )
    key = sandbox.address.split("/", 1)[0].partition("#")[2]
    return f"{ips[0]}:{_base(sandbox.address)[1]}" + (f"#{key}" if key else "")


def _container_running(container_id: str) -> bool | None:
    """Whether the sandbox container is still running, or None if unknowable.

    ``True``/``False`` report the container's actual state; ``None`` means the
    inspect itself failed (daemon busy/erroring), which callers must surface as
    "could not inspect" rather than misreporting the container as exited. Any
    outcome returns promptly, so callers fail fast instead of spinning the
    whole connect timeout.
    """
    proc = run_docker(["docker", "inspect", "-f", "{{.State.Running}}", container_id])
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() == "true"


def _container_logs(container_id: str, tail: int = 50) -> str:
    """Recent container logs for error messages, degrading instead of raising.

    Only ever called while building a failure message, so a stalled daemon
    yields a placeholder rather than masking the original error.
    """
    try:
        proc = run_docker(["docker", "logs", "--tail", str(tail), container_id])
    except RuntimeError as exc:
        return f"<no logs: {exc}>"
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
    _connect_timeout: float
    sandbox: SandboxInfo
    # Lent to a run or binding, which holds the env's lease until it ends.
    _on_loan: bool = False

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
            request_timeout_seconds: float | None = None,
            workflow_edition: str | None = None,
            api_key: str | None = None,
            tls_ca_file: str | os.PathLike[str] | None = None,
        ) -> None: ...

    def _detach(self) -> None:
        """Detach the remote client; supplied by the ``Remote*EnvBase`` mixed in."""
        raise NotImplementedError

    @contextlib.contextmanager
    def _lent(self) -> Iterator[str]:
        """Lend the container to a run on its address, then attach again.

        A run drives the env on a lease of its own, so this handle lets go of
        its lease (and episode) for the run and takes it back after, unless
        the run closed the sandbox. A second loan is refused before it lets
        go of anything: its attach would wait out the first's lease, then stop
        the container under it.
        """
        if self._on_loan:
            raise RuntimeError(
                "the env is held by another driver (a binding, a run or a "
                "RemoteEnv): close that one first"
            )
        self._detach()
        self._on_loan = True
        try:
            yield self.sandbox.address
        finally:
            self._on_loan = False
            if not self._closed:
                self._attach(self._connect_timeout)

    def _attach(self, connect_timeout_seconds: float) -> None:
        """Attach the client to the started container once it serves.

        Stops the container on any failure so it is never leaked; ``_closed``
        is only set once that stop succeeds, so ``close()``/``__del__`` can
        retry a failed stop instead of leaking the container for the process
        lifetime (mirroring :meth:`_stop`).
        """
        sandbox = self.sandbox
        self._connect_timeout = connect_timeout_seconds
        try:
            wait_serving(
                sandbox,
                connect_timeout_seconds,
                lambda attempt: self._initialize(
                    sandbox.address, connect_timeout_seconds=attempt
                ),
            )
        except BaseException:
            try:
                _sandbox_stop_env(container_id=sandbox.container_id)
            except BaseException:
                pass
            else:
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
            f"address={redact_address(self.sandbox.address)!r}, "
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
    render_size: tuple[int, int] | None = None,
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
    step and rejects a set runtime flag. ``render_size`` is the model's camera
    size, which a prebuilt image binds on its env's declared camera dial.
    """
    build = build or SandboxBuild()
    run = runtime or SandboxRuntime()
    gpus = normalize_gpus(run.gpus)
    devices = string_sequence("devices", run.devices)
    volumes = string_sequence("volumes", run.volumes)
    user = normalize_user(run.user)
    kind, resolved = resolve_source_kind(source)
    if kind == "prebuilt":
        _warn_ignored_build_options(build)
        return start_prebuilt_container(
            resolved,
            requested_source=source,
            name="env",
            binding=binding,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            render_size=render_size,
            gpus=gpus,
            devices=devices,
            volumes=volumes,
            user=user,
        )
    if gpus is not None or devices or volumes or user is not None:
        raise ValueError(
            "SandboxRuntime (gpus/devices/volumes/user) is only supported for a "
            f"prebuilt sandbox image; building an env from source ({source!r}) has no "
            "docker-run step -- pass a prebuilt image (docker://img / bare img:tag) "
            "instead"
        )
    return _start_build(
        source,
        build=build,
        num_envs=num_envs,
        vectorization_mode=vectorization_mode,
        binding=binding,
    )


def model_render_size(model: object, timeout_seconds: float) -> tuple[int, int] | None:
    """The camera size ``model``'s ``Image(render=...)`` inputs assert, or ``None``.

    ``model`` is a :class:`~rlmesh.adapters.ModelSpec`, a model carrying one as
    its ``spec``, a served model (``Model("host:port/name")``) or a
    :class:`~rlmesh.SandboxModel` -- served first, so its own spec is read off
    its describe envelope. A model with no spec asserts nothing.
    """
    from .._rlmesh import adapters_render_size, model_describe
    from ..adapters import ModelSpec
    from ._model import SandboxModel

    spec_json: str | None = None
    address: str | None = None
    if isinstance(model, ModelSpec):
        spec_json = model.to_json()
    elif isinstance(model, SandboxModel):
        address = model.serve().address
    elif isinstance(served := getattr(model, "_served_address", None), str):
        address = served
    elif isinstance(spec := getattr(model, "spec", None), ModelSpec):
        spec_json = spec.to_json()
    if address is not None:
        envelope = model_describe(address, timeout_seconds)
        described = json.loads(envelope).get("model_spec") if envelope else None
        if isinstance(described, dict) and "input" in described:
            spec_json = json.dumps(described)
    return None if spec_json is None else adapters_render_size(spec_json)


def start_prebuilt_container(
    image: str,
    *,
    requested_source: str,
    name: str,
    binding: Mapping[str, object],
    num_envs: int | None = None,
    vectorization_mode: str | None = None,
    render_size: tuple[int, int] | None = None,
    gpus: str | None = None,
    devices: Sequence[str] | None = None,
    volumes: Sequence[str] | None = None,
    user: str | None = None,
) -> SandboxInfo:
    """Run a prebuilt rlmesh-serving image and return its connection info.

    The image runs as-is (its own ``CMD``), hosting a session on
    ``RLMESH_ADDRESS`` and serving in it as ``RLMESH_NAME`` (``name``); the
    binding and eval shape are injected as environment variables
    (``RLMESH_MAKE_KWARGS`` etc.), and the session's ports are published on
    consecutive loopback ports, which its participants dial. No build and no
    rlmesh pin -- the image is whatever the publisher baked.

    ``num_envs`` / ``vectorization_mode`` are the vectorized-env eval shape: when
    left ``None`` (the model path -- a model is never vectorized) the corresponding
    ``RLMESH_NUM_ENVS`` / ``RLMESH_VECTORIZATION_MODE`` env vars are not injected, so
    the container serves in its default (single, model) mode.

    A missing image is pulled here, announced on stderr, rather than invisibly
    inside ``docker run`` -- a multi-GB pull with no progress line reads as a
    hang.
    """
    if shutil.which("docker") is None:
        raise RuntimeError(
            "Docker CLI not found on PATH; install Docker to run a prebuilt "
            "sandbox image, or use gym://... to build from source"
        )
    reap_orphans()
    if not docker_image_exists(image):
        pulled, pull_stderr = docker_pull(image)
        if not pulled:
            detail = f":\n{pull_stderr}" if pull_stderr else ""
            raise RuntimeError(f"could not pull prebuilt image {image!r}{detail}")
    # The container hosts its session under keys made here, so its address
    # carries their key (design §10: the workload spec); the secret reaches
    # docker through its environment, never argv.
    identity, key = _sandbox_host_identity()
    with _PORTS_LOCK:
        host_port = _sandbox_session_ports()
        # The container hosts its session on the host's own port numbers, so a
        # session address means the same inside it and out.
        env_vars = {
            "RLMESH_ADDRESS": f"0.0.0.0:{host_port}",
            "RLMESH_NAME": name,
        }
        if binding:
            env_vars["RLMESH_MAKE_KWARGS"] = binding_json(binding)
        if num_envs and num_envs != 1:
            env_vars["RLMESH_NUM_ENVS"] = str(num_envs)
        if vectorization_mode:
            env_vars["RLMESH_VECTORIZATION_MODE"] = vectorization_mode
        if render_size is not None:
            env_vars["RLMESH_RENDER_SIZE"] = f"{render_size[0]}x{render_size[1]}"

        cmd = prebuilt_run_cmd(
            image,
            env_vars=env_vars,
            gpus=gpus,
            devices=devices,
            volumes=volumes,
            user=user,
            host_port=host_port,
            owner_pid=os.getpid(),
            owner_pid_ns=pid_namespace_id(),
            pass_env=[HOST_IDENTITY_VAR],
        )
        proc = run_docker(cmd, env={HOST_IDENTITY_VAR: identity})
    if proc.returncode != 0:
        raise RuntimeError(
            f"failed to start prebuilt container ({proc.returncode}):\n{proc.stderr}"
        )
    container_id = proc.stdout.strip()
    return SandboxInfo(
        requested_source=requested_source,
        resolved_source=f"docker://{image}",
        address=f"127.0.0.1:{host_port}#{key}/{name}",
        container_id=container_id,
    )


def prebuilt_run_cmd(
    image: str,
    *,
    env_vars: Mapping[str, str],
    gpus: str | None,
    host_port: int,
    owner_pid: int,
    owner_pid_ns: str | None,
    devices: Sequence[str] | None = None,
    volumes: Sequence[str] | None = None,
    user: str | None = None,
    pass_env: Sequence[str] = (),
) -> list[str]:
    """Build the ``docker run`` argv for a prebuilt sandbox container (pure).

    The one place container hardening lives (``--cap-drop ALL``,
    ``--security-opt no-new-privileges``, owner labels for orphan reaping), shared
    by the env and model prebuilt paths so the flags never drift apart. The
    session's ports publish on loopback from ``host_port``, in order, to the
    same numbers inside. The image
    is always last and env vars precede it, so callers/tests can rely on argv shape.
    ``devices`` -> ``--device`` (incl. CDI refs like ``nvidia.com/gpu=all``),
    ``volumes`` -> ``-v`` mounts, and ``user`` -> ``--user``, all before the env
    vars. ``pass_env`` names variables passed from docker's own environment
    (``-e NAME``), so a secret stays out of argv.
    """
    cmd = [
        "docker",
        "run",
        "-d",
        "-p",
        f"127.0.0.1:{host_port}-{host_port + SESSION_PORTS - 1}:"
        f"{host_port}-{host_port + SESSION_PORTS - 1}",
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
    if user is not None:
        cmd += ["--user", user]
    for device in devices or []:
        cmd += ["--device", device]
    for volume in volumes or []:
        cmd += ["-v", volume]
    for key, value in env_vars.items():
        cmd += ["-e", f"{key}={value}"]
    for name in pass_env:
        cmd += ["-e", name]
    cmd.append(image)
    return cmd


def reap_orphans() -> None:
    """Best-effort sweep of containers orphaned by a prior hard kill.

    Delegates to the native ``sandbox_reap_orphans`` reaper; failures are
    swallowed -- reaping is hygiene, never a reason to fail a start.
    """
    from .._rlmesh import sandbox_reap_orphans

    try:
        sandbox_reap_orphans()
    except Exception:
        pass


def pid_namespace_id() -> str | None:
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return None


def _warn_ignored_build_options(options: SandboxBuild) -> None:
    """Reject or warn on build-only options set on a run-prebuilt (not built) source.

    ``trust_remote_code`` / ``allow_unpinned_hf`` are security grants: silently
    ignoring one would leave the caller believing a trust decision took effect,
    so setting either on a prebuilt source is a hard error. The remaining build
    knobs merely cannot apply (the image is already built) and keep their
    documented warning behavior.
    """
    granted = [
        name
        for name in ("trust_remote_code", "allow_unpinned_hf")
        if getattr(options, name)
    ]
    if granted:
        raise ValueError(
            f"{', '.join(granted)} configure the from-source build and have no "
            "effect on a prebuilt image source (the image is already built); "
            "remove them, or build from a gym://.../hf://... source"
        )
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
    kwargs_json = binding_json(binding) if binding else None
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


def binding_json(binding: Mapping[str, object]) -> str:
    """Serialize the construction-param binding to canonical JSON, or raise.

    The one encoder for ``RLMESH_MAKE_KWARGS`` (prebuilt) and the native build's
    ``kwargs_json``, so both emit identical, key-sorted JSON. A value JSON cannot
    represent is named by its param -- a raw ``TypeError`` from ``json.dumps``
    points at nothing the caller typed.
    """
    try:
        return json.dumps(dict(binding), sort_keys=True)
    except TypeError:
        for key, value in binding.items():
            try:
                _ = json.dumps(value)
            except TypeError as exc:
                raise TypeError(
                    f"construction param {key!r} is not JSON-serializable "
                    f"({type(value).__name__}); pass JSON-encodable values"
                ) from exc
        raise


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
    "SANDBOX_REMOTE_CONNECT_TIMEOUT_SECONDS",
    "SESSION_PORTS",
    "SandboxBuild",
    "SandboxInfo",
    "SandboxLifecycle",
    "SandboxRuntime",
    "binding_json",
    "normalize_gpus",
    "normalize_user",
    "prebuilt_run_cmd",
    "reap_orphans",
    "reject_sandbox_option_params",
    "reject_single_env_vector_option",
    "start_prebuilt_container",
    "start_sandbox_container",
    "string_sequence",
    "wait_serving",
]
