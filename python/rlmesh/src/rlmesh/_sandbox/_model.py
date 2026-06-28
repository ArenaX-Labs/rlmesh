"""``SandboxModel``: a prebuilt model container running in its own container.

The model-side sibling of ``SandboxEnv``. The source is a prebuilt
``image://<tag>`` container you built yourself (BYO); construction is inert -- it
records the tag but starts nothing. ``serve()`` (or use as a context manager)
starts a long-lived container serving the policy as a model endpoint;
``rlmesh.session(model, env)`` (or ``model.session(env)``) serves it and binds it to an
env for the drive loop.
"""

from __future__ import annotations

import subprocess
import time
from typing import TYPE_CHECKING

from ._sources import looks_like_gym_id
from .session import start_prebuilt_container

if TYPE_CHECKING:
    from rlmesh._rlmesh import PyModelClient

    from .._models._eval import Session
    from ..specs import EnvContract
    from .session import SandboxOptions

__all__ = ["SandboxModel"]

_IMAGE_SCHEMES = ("image://", "docker://")

# Only these are retried while the container boots: a refused/unavailable
# connection (ConnectionError/OSError) or a connect timeout. Deterministic
# config errors -- a missing/invalid contract space (RuntimeError/TypeError,
# raised before the client even dials) -- are NOT here, so they fail fast. Each
# retry first re-checks the container is still running, so a dial error against
# an already-exited container also fails fast (with logs) rather than retrying.
_TRANSIENT_DIAL_ERRORS = (ConnectionError, TimeoutError, OSError)


def _resolve_model_source(source: object) -> str:
    """Resolve a model source to a prebuilt image tag.

    A model is always a prebuilt rlmesh-serving image (BYO) -- there is no
    build-from-source for models. An explicit ``image://``/``docker://`` scheme
    is stripped; a bare string is taken as the image tag (an untagged repo ref
    like ``policy/run-test`` is valid), but an obvious gym env id (``CartPole-v1``,
    ``pkg:Env-v0``) is rejected up front rather than failing opaquely at
    ``docker run`` -- a model is never a gym source.
    """
    if not isinstance(source, str):
        raise TypeError(
            "SandboxModel requires a prebuilt image source string (e.g. "
            f"'smolvla:latest' or 'docker://smolvla:latest'); got "
            f"{type(source).__name__}"
        )
    raw = source.strip()
    tag = raw
    had_scheme = False
    for scheme in _IMAGE_SCHEMES:
        if tag.startswith(scheme):
            tag = tag[len(scheme) :].strip()
            had_scheme = True
            break
    if not tag or "://" in tag:
        raise ValueError(
            "SandboxModel source must be an image tag (e.g. 'smolvla:latest') or "
            f"image://<tag> / docker://<tag>; got {source!r}"
        )
    # A bare gym env id (version-suffixed) is never a model image; an explicit
    # scheme is trusted, and an untagged repo ref is a valid image.
    if not had_scheme and looks_like_gym_id(tag):
        raise ValueError(
            f"SandboxModel source {source!r} looks like a gym env id, not a model "
            "image; pass a prebuilt image (e.g. 'smolvla:latest') or image://<tag>"
        )
    return tag


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

    The source is a prebuilt rlmesh-serving image you built yourself (BYO), given
    as a bare ``smolvla:latest`` or an explicit ``image://``/``docker://`` tag: the
    image is run directly and its baked ``CMD`` drives it. ``**params`` are the
    model's construction params -- forwarded into the container as the
    ``load(**binding)`` binding (``RLMESH_MAKE_KWARGS``), validated against the
    model's declared ``params`` before weights load.
    """

    def __init__(
        self,
        source: object,
        /,
        *,
        gpus: str | int | None = None,
        options: SandboxOptions | None = None,
        **params: object,
    ) -> None:
        self._image = _resolve_model_source(source)
        self._gpus = _normalize_gpus(gpus)
        self._binding = dict(params)
        if options is not None:
            import warnings

            # A model is always a prebuilt image (run, not built), so build infra
            # has nothing to act on.
            warnings.warn(
                "SandboxModel runs a prebuilt image; SandboxOptions build "
                "settings are ignored",
                stacklevel=2,
            )
        self._address: str | None = None
        self._container_id: str | None = None
        self._closed = False

    def _serve_image(self) -> SandboxModel:
        """Start a long-lived container for a prebuilt image tag (serve mode).

        Delegates to the shared :func:`start_prebuilt_container` (the same path the
        env sandbox uses), passing ``num_envs``/``vectorization_mode`` as ``None`` so
        the container serves in single (model) mode -- a model is never vectorized.
        The container enters serve mode because no ``RLMESH_DRIVE_ENV_ADDRESS`` is set
        (the BYO model protocol); the model's construction params, if any, ride in as
        ``RLMESH_MAKE_KWARGS``. The port is published on a Docker-assigned host port
        and read back (no host-side bind/close, so no TOCTOU race); on port-discovery
        failure the helper stops the container before re-raising, so nothing leaks.
        Readiness is the caller's connect-with-retry (see :meth:`session`).
        """
        assert self._image is not None
        info = start_prebuilt_container(
            self._image,
            requested_source=self._image,
            binding=self._binding,
            gpus=self._gpus,
        )
        self._container_id = info.container_id
        self._address = info.address
        self._closed = False
        return self

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
        execution_horizon: int = 1,
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
        ``trust_entrypoints`` apply to local models and are ignored here. ``execution_horizon``
        requests open-loop action chunking: the runtime executes that many actions of each
        predicted chunk before re-planning (1 = re-plan every step; only engages if the served
        policy defines a chunk corner). Retries the connection while the container starts, up to
        ``connect_timeout_seconds``.
        """
        _ = instruction, token, trust_entrypoints
        from .._client._remote_model import env_contract_of, remote_session
        from .._load_native import load_native

        # Only tear down on failure if THIS call started the container; a handle the
        # caller is managing (context manager / reuse) must survive a failed bind
        # (serve() is idempotent and returns early when already serving).
        started_here = self._address is None
        self.serve()
        try:
            contract = env_contract_of(env)
            client = self._dial_with_retry(
                load_native("PyModelClient"),
                contract,
                connect_timeout_seconds,
                execution_horizon,
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
        execution_horizon: int = 1,
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
                return client_cls(
                    self.address, contract, execution_horizon=execution_horizon
                )
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
