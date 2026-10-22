"""``SandboxModel``: a prebuilt model container running in its own container.

The model-side sibling of ``SandboxEnv``. The source is a prebuilt
``image://<tag>`` container you built yourself (BYO); construction is inert -- it
records the tag but starts nothing. ``serve()`` (or use as a context manager)
starts a long-lived container serving the policy in the session it hosts;
``run(env)`` and ``bind(env)`` drive it there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

from .._upgrade import removed
from ._sources import looks_like_gym_id
from .session import (
    SANDBOX_REMOTE_CONNECT_TIMEOUT_SECONDS,
    SandboxRuntime,
    normalize_gpus,
    normalize_user,
    reject_sandbox_option_params,
    start_prebuilt_container,
    string_sequence,
    wait_serving,
)

if TYPE_CHECKING:
    from .._models import Binding, RunResult
    from .._models.base import ModelBase
    from ..types import LocalEnvTarget

__all__ = ["SandboxModel"]

_IMAGE_SCHEMES = ("image://", "docker://")


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


class SandboxModel:
    """A model served from an isolated container.

    The source is a prebuilt rlmesh-serving image you built yourself (BYO), given
    as a bare ``smolvla:latest`` or an explicit ``image://``/``docker://`` tag: the
    image is run directly and its baked ``CMD`` drives it. ``**params`` are the
    model's construction params -- forwarded into the container as the
    ``load(**binding)`` binding (``RLMESH_MAKE_KWARGS``), validated against the
    model's declared ``params`` before weights load.

    A :class:`~rlmesh._sandbox.session.SandboxRuntime` (``runtime=``) sets ``docker
    run`` flags -- ``gpus`` for CUDA compute, or ``devices=["nvidia.com/gpu=all"]``
    (a CDI ref, full graphics+compute) and ``volumes=[...]`` for large local
    checkpoints/assets. A model is always a prebuilt image, so there is no build
    config (no ``SandboxBuild``). The image must be built with rlmesh 0.2; one
    built with 0.1 is refused, naming it.

    :meth:`run` and :meth:`bind` drive the served model as
    ``rlmesh.Model(address)`` does, serving a local env in the session the
    container hosts, which it reaches. :meth:`run` also takes a
    :class:`~rlmesh.SandboxEnv`, whose session the container joins over their
    shared Docker network.
    """

    def __init__(
        self,
        source: object,
        /,
        *,
        runtime: SandboxRuntime | None = None,
        connect_timeout_seconds: float = SANDBOX_REMOTE_CONNECT_TIMEOUT_SECONDS,
        **params: object,
    ) -> None:
        self._image = _resolve_model_source(source)
        reject_sandbox_option_params(params)
        run = runtime or SandboxRuntime()
        self._gpus = normalize_gpus(run.gpus)
        self._devices = string_sequence("devices", run.devices)
        self._volumes = string_sequence("volumes", run.volumes)
        self._user = normalize_user(run.user)
        self._binding = dict(params)
        self._connect_timeout = connect_timeout_seconds
        self._address: str | None = None
        self._container_id: str | None = None
        self._model: ModelBase[Any, Any] | None = None
        self._closed = False

    def serve(self) -> SandboxModel:
        """Start a long-lived container serving the policy, once it answers.

        Delegates to the shared :func:`start_prebuilt_container` (the same path
        the env sandbox uses) without an eval shape -- a model is never
        vectorized; the model's construction params, if any, ride in as
        ``RLMESH_MAKE_KWARGS``. A container that does not answer is stopped
        before the error is raised, so nothing leaks.

        Idempotent: a second call returns the already-running handle. The model
        is reachable at :attr:`address` until :meth:`shutdown`.
        """
        if self._address is not None:
            return self
        from .._rlmesh import model_answers, sandbox_stop_env

        info = start_prebuilt_container(
            self._image,
            requested_source=self._image,
            name="model",
            binding=self._binding,
            gpus=self._gpus,
            devices=self._devices,
            volumes=self._volumes,
            user=self._user,
        )
        try:
            wait_serving(
                info,
                self._connect_timeout,
                lambda attempt: model_answers(info.address, attempt),
            )
        except BaseException:
            sandbox_stop_env(container_id=info.container_id)
            raise
        self._container_id = info.container_id
        self._address = info.address
        self._closed = False
        return self

    def run(self, env: LocalEnvTarget, **options: Any) -> RunResult:
        """Run the served model against ``env``, as :meth:`rlmesh.Model.run` does.

        Serves the container first; a local env is served in its session, and
        a :class:`~rlmesh.SandboxEnv`'s session is joined from the container.
        """
        return self._served().run(env, **options)

    def bind(self, env: LocalEnvTarget, **options: Any) -> Binding[Any, Any]:
        """Bind the served model to ``env``, as :meth:`rlmesh.Model.bind` does.

        Serves the container first; a local env is served in its session.
        """
        return self._served().bind(env, **options)

    def _served(self) -> ModelBase[Any, Any]:
        """The served model, reached at :attr:`address`."""
        self.serve()
        if self._model is None:
            from .._native import Model

            model = Model(self.address)
            model._env_session = self.address.rsplit("/", 1)[0]  # pyright: ignore[reportPrivateUsage]
            self._model = model
        return self._model

    def session(self, *args: object, **kwargs: object) -> NoReturn:
        """Removed in 0.2: a model is bound to an env with ``bind(target)``."""
        _ = args, kwargs
        removed(
            "SandboxModel.session(env)",
            "SandboxModel(...).bind(target) returns a Binding you step by hand",
        )

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
        if self._model is not None:
            self._model.close()
            self._model = None
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
