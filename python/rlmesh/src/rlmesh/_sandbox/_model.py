"""``SandboxModel``: a prebuilt model container running in its own container.

The model-side sibling of ``SandboxEnv``. The source is a prebuilt
``image://<tag>`` container you built yourself (BYO); construction is inert -- it
records the tag but starts nothing. ``serve()`` (or use as a context manager)
starts a long-lived container serving the policy as a model endpoint;
``rlmesh.session(model, env)`` (or ``model.session(env)``) serves it and binds it to an
env for the drive loop.
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlmesh._rlmesh import PyModelClient

    from .._models._eval import Session
    from ..specs import EnvContract

__all__ = ["SandboxModel"]

# The port a BYO model container serves on inside the container (the
# byo_container/model serve-mode default, RLMESH_ADDRESS=0.0.0.0:50051).
_CONTAINER_SERVE_PORT = 50051


def _pid_namespace_id() -> str | None:
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return None


_IMAGE_SCHEME = "image://"

# Only these are retried while the container boots: a refused/unavailable
# connection (ConnectionError/OSError) or a connect timeout. Deterministic
# config errors -- a missing/invalid contract space (RuntimeError/TypeError,
# raised before the client even dials) -- are NOT here, so they fail fast. Each
# retry first re-checks the container is still running, so a dial error against
# an already-exited container also fails fast (with logs) rather than retrying.
_TRANSIENT_DIAL_ERRORS = (ConnectionError, TimeoutError, OSError)


def _parse_image_source(source: object) -> str | None:
    """Return the tag from an ``image://<tag>`` source, or None.

    A prebuilt (BYO) container image is run directly: recipe resolution and the
    image build are bypassed, and the container's own baked ``runtime.json``
    drives it.
    """
    if isinstance(source, str) and source.startswith(_IMAGE_SCHEME):
        tag = source[len(_IMAGE_SCHEME) :].strip()
        if not tag:
            raise ValueError(
                "image:// source must include a tag, e.g. image://my-model:latest"
            )
        return tag
    return None


def _reap_orphans() -> None:
    """Best-effort sweep of containers orphaned by a prior hard kill.

    Delegates to the native ``sandbox_reap_orphans`` reaper, which sweeps any
    rlmesh-owned container (env or model) whose owner process is gone. The
    getattr guard degrades to a no-op against a version-skewed native extension
    that predates the reaper; failures are swallowed -- reaping is hygiene, never
    a reason to fail a serve.
    """
    import rlmesh._rlmesh as _rlmesh

    reap = getattr(_rlmesh, "sandbox_reap_orphans", None)
    if reap is None:
        return
    try:
        reap()
    except Exception:
        pass


def _normalize_gpus(gpus: str | int | None) -> str | None:
    """Normalize the ``gpus`` knob into a docker ``--gpus`` value, or None.

    Passes straight through to ``docker run --gpus``: ``"all"``, a count
    (``2`` / ``"2"``), or a device selector (``"device=0,1"``).
    """
    if gpus is None:
        return None
    value = str(gpus).strip()
    if not value:
        raise ValueError("gpus= must be non-empty, e.g. 'all', 1, or 'device=0,1'")
    return value


class SandboxModel:
    """A model served from an isolated container.

    The source is a prebuilt ``image://<tag>`` container you built yourself (BYO):
    the tag is run directly and its baked ``runtime.json`` drives it.
    """

    def __init__(
        self,
        source: object,
        *,
        gpus: str | int | None = None,
    ) -> None:
        # BYO prebuilt container: ``image://<tag>`` is run directly, and the
        # container's own baked runtime.json drives it.
        prebuilt = _parse_image_source(source)
        if prebuilt is None:
            raise TypeError(
                "SandboxModel requires a prebuilt image source, e.g. "
                f"image://my-model:latest; got {type(source).__name__}"
            )
        self._image = prebuilt
        self._gpus = _normalize_gpus(gpus)
        self._address: str | None = None
        self._container_id: str | None = None
        self._closed = False

    def _gpu_args(self) -> list[str]:
        return ["--gpus", self._gpus] if self._gpus is not None else []

    def _serve_image(self) -> SandboxModel:
        """Start a long-lived container for a BYO ``image://`` tag (serve mode).

        Publishes the container's serve port on a Docker-assigned host port (no
        host-side bind/close, so no TOCTOU race), then reads the real port back
        with ``docker port``. The container enters serve mode because no
        ``RLMESH_DRIVE_ENV_ADDRESS`` is set (the BYO model protocol). Readiness is
        the caller's connect-with-retry (see :meth:`against`).
        """
        assert self._image is not None
        _reap_orphans()
        cmd = [
            "docker",
            "run",
            "-d",
            "-p",
            f"127.0.0.1:0:{_CONTAINER_SERVE_PORT}",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--label",
            "rlmesh.sandbox=1",
            "--label",
            f"rlmesh.sandbox.owner-pid={os.getpid()}",
        ]
        pid_ns = _pid_namespace_id()
        if pid_ns is not None:
            cmd += ["--label", f"rlmesh.sandbox.owner-pid-ns={pid_ns}"]
        cmd += self._gpu_args()
        cmd.append(self._image)
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"failed to start model container ({proc.returncode}):\n{proc.stderr}"
            )
        container_id = proc.stdout.strip()
        self._container_id = container_id
        try:
            port = self._resolve_published_port(container_id)
        except BaseException:
            # Port discovery failed after the container started: stop it before
            # re-raising so a retry doesn't overwrite _container_id and leak it.
            from .._rlmesh import sandbox_stop_env

            try:
                sandbox_stop_env(container_id=container_id)
            except Exception:
                pass
            self._container_id = None
            raise
        self._address = f"127.0.0.1:{port}"
        self._closed = False
        return self

    def _resolve_published_port(self, container_id: str) -> int:
        """Read the host port Docker assigned for the container's serve port."""
        proc = subprocess.run(
            ["docker", "port", container_id, str(_CONTAINER_SERVE_PORT)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"failed to read published port for model container "
                f"({proc.returncode}):\n{proc.stderr}"
            )
        for line in proc.stdout.splitlines():
            # Mirror the Rust sibling (docker::parse_published_port): the host
            # port is the last `:`-separated field of any mapping line, so an
            # IPv6 binding like `[::]:51000` parses, not just IPv4 prefixes.
            _, sep, port = line.strip().rpartition(":")
            if sep and port.isdigit():
                return int(port)
        raise RuntimeError(
            "model container published no host port for "
            f"{_CONTAINER_SERVE_PORT}; docker port output:\n{proc.stdout}"
        )

    def serve(self) -> SandboxModel:
        """Start a long-lived container serving the policy as a model endpoint.

        The prebuilt ``image://`` image is run in serve mode.

        Idempotent: a second call returns the already-running handle. The endpoint
        is reachable at :attr:`address` until :meth:`shutdown`.
        """
        if self._address is not None:
            return self
        return self._serve_image()

    def session(
        self,
        env: object,
        *,
        instruction: str | None = None,
        close_env: bool = False,
        token: str | None = None,
        trust_entrypoints: bool | None = None,
        connect_timeout_seconds: float = 30.0,
    ) -> Session[object, object]:
        """Serve this model and bind it to ``env``, returning a neutral :class:`rlmesh.Session`.

        The managed sibling of :meth:`rlmesh.RemoteModel.session`: starts the model
        container (idempotent), then opens a route configured from the env's contract so
        the *same* drive loop works for both pairs::

            with rlmesh.SandboxEnv("gym://CartPole-v1") as env:
                sess = rlmesh.session(
                    rlmesh.SandboxModel("image://my-model:latest"), env
                )
                obs, _ = sess.reset()
                while not sess.done:
                    obs, reward, terminated, truncated, _ = sess.step(sess.predict(obs))

        Closing the session stops the container it started. ``instruction`` / ``token`` /
        ``trust_entrypoints`` apply to local models and are ignored here. Retries the
        connection while the container starts, up to ``connect_timeout_seconds``.
        """
        _ = instruction, token, trust_entrypoints
        try:
            from rlmesh._rlmesh import PyModelClient
        except ImportError as e:  # pragma: no cover - import guard
            raise ImportError("Failed to import _rlmesh native module.") from e

        from .._client._remote_model import env_contract_of, remote_session

        # Only tear down on failure if THIS call started the container; a handle the
        # caller is managing (context manager / reuse) must survive a failed bind
        # (serve() is idempotent and returns early when already serving).
        started_here = self._address is None
        self.serve()
        try:
            contract = env_contract_of(env)
            client = self._dial_with_retry(
                PyModelClient, contract, connect_timeout_seconds
            )
            # Hand ownership (and so container teardown on session close) only to a
            # session that actually started the container. A caller-managed handle
            # (serve()/context manager) must survive its sessions closing.
            return remote_session(
                client, env, owner=self if started_here else None, close_env=close_env
            )
        except BaseException:
            if started_here:
                self.shutdown()
            raise

    def _dial_with_retry(
        self,
        client_cls: type[PyModelClient],
        contract: EnvContract,
        connect_timeout_seconds: float,
    ) -> PyModelClient:
        """Dial the serving container, retrying while it is still starting.

        Retries only genuine transient startup failures (see
        :data:`_TRANSIENT_DIAL_ERRORS`): a deterministic contract/config error
        (e.g. a missing observation space) fails fast instead of being retried
        for the whole timeout and then masked as "did not become ready". If the
        container has exited/crashed, fail fast with its recent logs.
        """
        deadline = time.monotonic() + connect_timeout_seconds
        last_error: BaseException | None = None
        while True:
            try:
                return client_cls(self.address, contract)
            except _TRANSIENT_DIAL_ERRORS as exc:  # the container may still be starting
                last_error = exc
                if not self._container_running():
                    raise RuntimeError(
                        f"model container at {self.address} exited before becoming "
                        f"ready; recent logs:\n{self._container_logs()}"
                    ) from last_error
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"model container at {self.address} did not become ready "
                        f"within {connect_timeout_seconds:.0f}s"
                    ) from last_error
                time.sleep(0.1)

    def _container_running(self) -> bool:
        """Whether the served container is still running (best effort)."""
        if self._container_id is None:
            return False
        proc = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", self._container_id],
            capture_output=True,
            text=True,
            check=False,
        )
        # On any inspect failure (gone/daemon error), treat as not running so the
        # caller fails fast rather than spinning the full timeout.
        return proc.returncode == 0 and proc.stdout.strip() == "true"

    def _container_logs(self, tail: int = 50) -> str:
        if self._container_id is None:
            return ""
        proc = subprocess.run(
            ["docker", "logs", "--tail", str(tail), self._container_id],
            capture_output=True,
            text=True,
            check=False,
        )
        return (proc.stdout + proc.stderr).strip()

    @property
    def address(self) -> str:
        if self._address is None:
            raise RuntimeError(
                "SandboxModel is not serving; call serve() (or use it as a context "
                "manager) before reading address"
            )
        return self._address

    @property
    def container_id(self) -> str:
        if self._container_id is None:
            raise RuntimeError(
                "SandboxModel is not serving; call serve() before reading container_id"
            )
        return self._container_id

    def shutdown(self) -> None:
        """Stop the served container, if any. Idempotent; safe to call repeatedly."""
        if self._closed or self._container_id is None:
            return
        from .._rlmesh import sandbox_stop_env

        # Mark closed only once the stop succeeds, so a transient failure can be
        # retried by a later shutdown()/__exit__/__del__ instead of leaking the
        # container (mirrors the sandbox env session teardown).
        sandbox_stop_env(container_id=self._container_id)
        self._closed = True
        self._address = None
        self._container_id = None

    def __enter__(self) -> SandboxModel:
        return self.serve()

    def __exit__(self, *exc: object) -> bool:
        self.shutdown()
        return False

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
