"""Sandbox source classification: gym id vs Docker image, and image probing.

Pure, lifecycle-free helpers used by :mod:`rlmesh._sandbox.session` (build vs
prebuilt routing) and :mod:`rlmesh._sandbox._model`. Kept separate from container
lifecycle so the source/Docker heuristics can be tested and reasoned about on
their own.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from typing import Literal

SourceKind = Literal["build", "prebuilt"]

#: Gymnasium's mandatory version suffix (``-v<int>``). A registered id ends with
#: it, including the module-import form ``pkg:Env-v0`` and ``ALE/Pong-v5``.
_GYM_VERSION_RE = re.compile(r"-v\d+$")


def resolve_source_kind(source: str) -> tuple[SourceKind, str]:
    """Classify a sandbox source and return ``(kind, resolved_ref)``.

    * ``gym://`` / ``hf://`` / bare gym id (no tag) -> build from source.
    * ``docker://img`` / ``image://img`` -> prebuilt (explicit).
    * bare image-shaped (``:tag`` / ``@sha256:``), local image -> prebuilt.
    * bare image-shaped, not local -> ``docker pull`` then prebuilt, else error.

    The resolved kind is always logged -- silent autodetect is the trap. Use an
    explicit ``gym://`` / ``docker://`` scheme to override the bare-source guess.
    """
    value = source.strip()
    if not value:
        raise ValueError("sandbox source must not be empty")
    if value.startswith(("gym://", "hf://")):
        return "build", value
    for scheme in ("docker://", "image://"):
        if value.startswith(scheme):
            ref = value[len(scheme) :].strip()
            if not ref:
                raise ValueError(f"{scheme} source must include an image tag")
            _log_resolution(source, "prebuilt", f"explicit {scheme}{ref}")
            return "prebuilt", ref
    if "://" in value:
        raise ValueError(f"unsupported sandbox source scheme: {source!r}")
    if not _is_image_shaped(value):
        _log_resolution(source, "build", "gym id")
        return "build", value
    if shutil.which("docker") is None:
        raise RuntimeError(
            f"{source!r} looks like a Docker image but the Docker CLI is not on "
            "PATH; install Docker, or use gym://... to build from source / "
            "docker://... to force a prebuilt image"
        )
    if docker_image_exists(value):
        _log_resolution(source, "prebuilt", "local Docker image")
        return "prebuilt", value
    if docker_pull(value):
        _log_resolution(source, "prebuilt", "pulled Docker image")
        return "prebuilt", value
    raise ValueError(
        f"{source!r} looks like a Docker image but was not found locally or "
        "pullable; use gym://... to build from source or docker://... to force a "
        "prebuilt image"
    )


def looks_like_gym_id(value: str) -> bool:
    """Whether a bare source carries Gymnasium's mandatory ``-v<int>`` suffix.

    The reliable signal that a colon-bearing source is a gym env id (``pkg:Env-v0``,
    ``ALE/Pong-v5``) rather than a Docker image ref -- so it routes to the build
    path / is rejected as a model image instead of being probed as an image.
    """
    return _GYM_VERSION_RE.search(value) is not None


def _is_image_shaped(value: str) -> bool:
    """Whether a bare source looks like a Docker image ref rather than a gym id.

    An image ref carries a ``:tag`` (a colon in the final path segment) or an
    ``@sha256:`` digest; a gym id like ``CartPole-v1`` or ``ALE/Pong-v5`` has
    neither, so it never triggers a Docker probe. The module-import gym id form
    ``pkg:Env-v0`` *does* carry a colon, so the gym version suffix short-circuits
    first -- otherwise it would be misrouted to the Docker path.
    """
    if looks_like_gym_id(value):
        return False
    return "@sha256:" in value or ":" in value.rsplit("/", 1)[-1]


def docker_image_exists(image: str) -> bool:
    """Whether a Docker image is present locally (``docker image inspect``)."""
    proc = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def docker_pull(image: str) -> bool:
    """Attempt ``docker pull``; return whether it succeeded."""
    proc = subprocess.run(
        ["docker", "pull", image], capture_output=True, text=True, check=False
    )
    return proc.returncode == 0


def _log_resolution(source: str, kind: SourceKind, detail: str) -> None:
    # Autodetect must never be silent: always announce the resolved kind.
    print(
        f"rlmesh: resolved {source!r} -> {kind} ({detail})", file=sys.stderr, flush=True
    )


__all__ = [
    "SourceKind",
    "docker_image_exists",
    "docker_pull",
    "looks_like_gym_id",
    "resolve_source_kind",
]
