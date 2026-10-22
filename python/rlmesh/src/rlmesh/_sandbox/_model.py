"""``SandboxModel``: a prebuilt model container running in its own container.

The model-side sibling of ``SandboxEnv``. The source is a prebuilt
``image://<tag>`` container you built yourself (BYO); construction is inert -- it
records the tag but starts nothing. ``serve()`` (or use as a context manager)
starts a long-lived container serving the policy as a model endpoint.
"""

from __future__ import annotations

from typing import NoReturn

from .._upgrade import removed
from ._sources import looks_like_gym_id
from .session import (
    SandboxRuntime,
    normalize_gpus,
    normalize_user,
    reject_sandbox_option_params,
    start_prebuilt_container,
    string_sequence,
)

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
    config (no ``SandboxBuild``).
    """

    def __init__(
        self,
        source: object,
        /,
        *,
        runtime: SandboxRuntime | None = None,
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
        """
        info = start_prebuilt_container(
            self._image,
            requested_source=self._image,
            binding=self._binding,
            gpus=self._gpus,
            devices=self._devices,
            volumes=self._volumes,
            user=self._user,
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
