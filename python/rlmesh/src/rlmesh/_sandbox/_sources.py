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

#: Ceiling for a quick Docker CLI call (inspect/port/logs/run/stop). Generous --
#: these normally finish in well under a second -- but bounded, so a wedged
#: Docker Desktop surfaces as an error instead of hanging the session forever.
DOCKER_TIMEOUT_SECONDS = 60.0

#: Ceiling for ``docker pull``: a multi-GB image over a slow link is legitimate,
#: so the guard only catches a daemon that has stopped making progress entirely.
DOCKER_PULL_TIMEOUT_SECONDS = 1800.0


def run_docker(
    cmd: list[str], *, timeout: float = DOCKER_TIMEOUT_SECONDS
) -> subprocess.CompletedProcess[str]:
    """Run a Docker CLI command with a hang guard.

    Every sandbox ``docker ...`` subprocess goes through here: a wedged Docker
    daemon otherwise blocks a plain ``subprocess.run`` forever. A timeout is
    converted into a directive ``RuntimeError`` naming the stalled command.
    """
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Docker daemon not responding: '{' '.join(cmd[:2])}' did not "
            f"complete within {timeout:.0f}s; check that Docker is running "
            "and responsive, then retry"
        ) from exc


#: Gymnasium's mandatory version suffix (``-v<int>``). A registered id ends with
#: it, including the module-import form ``pkg:Env-v0`` and ``ALE/Pong-v5``.
_GYM_VERSION_RE = re.compile(r"-v\d+$")


def resolve_source_kind(source: str) -> tuple[SourceKind, str]:
    """Classify a sandbox source and return ``(kind, resolved_ref)``.

    * ``gym://`` / ``hf://`` / gym-versioned bare id (``-v<int>``) -> build from
      source, never probed as a Docker image.
    * ``docker://img`` / ``image://img`` -> prebuilt (explicit).
    * bare image-shaped (``:tag`` / ``@sha256:``), local image -> prebuilt.
    * bare image-shaped, not local -> ``docker pull`` then prebuilt, else error.
    * bare tagless name (no scheme, no tag/digest, no gym version suffix) ->
      prebuilt only when the image exists locally (``docker image inspect``, no
      auto-pull); otherwise an error naming both ``docker://`` and ``gym://``.

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
    if looks_like_gym_id(value):
        _log_resolution(source, "build", "gym id")
        return "build", value
    if not _is_image_shaped(value):
        return _resolve_tagless(source, value)
    if shutil.which("docker") is None:
        raise RuntimeError(
            f"{source!r} looks like a Docker image but the Docker CLI is not on "
            "PATH; install Docker, or use gym://... to build from source / "
            "docker://... to force a prebuilt image"
        )
    if docker_image_exists(value):
        _log_resolution(source, "prebuilt", "local Docker image")
        return "prebuilt", value
    pulled, pull_stderr = docker_pull(value)
    if pulled:
        _log_resolution(source, "prebuilt", "pulled Docker image")
        return "prebuilt", value
    detail = f"; docker pull failed with:\n{pull_stderr}" if pull_stderr else ""
    raise ValueError(
        f"{source!r} looks like a Docker image but was not found locally or "
        "pullable; use gym://... to build from source or docker://... to force a "
        f"prebuilt image{detail}"
    )


def _resolve_tagless(source: str, value: str) -> tuple[SourceKind, str]:
    """Classify a bare tagless name by probing local Docker images only.

    A name with no scheme, no ``:tag``/digest, and no gym version suffix is
    ambiguous. A local ``docker image inspect`` hit resolves it to prebuilt;
    anything else (no hit, or no Docker CLI) is an error naming both explicit
    spellings -- never an auto-pull.
    """
    if shutil.which("docker") is None:
        raise ValueError(
            f"cannot classify sandbox source {source!r} (Docker CLI not on PATH "
            "to probe local images); use docker://"
            f"{value} to run it as a prebuilt image (pulls if needed) or gym://"
            f"{value} to build it from source"
        )
    if docker_image_exists(value):
        _log_resolution(source, "prebuilt", "local Docker image")
        return "prebuilt", value
    raise ValueError(
        f"{source!r} matches no local Docker image and has no gym version "
        "suffix; use docker://"
        f"{value} to run it as a prebuilt image (pulls if needed) or gym://"
        f"{value} to build it from source"
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
    proc = run_docker(["docker", "image", "inspect", image])
    return proc.returncode == 0


def docker_pull(image: str) -> tuple[bool, str]:
    """Attempt ``docker pull``; return ``(ok, stderr_tail)``.

    Announces the pull on stderr first -- a multi-GB pull is otherwise invisible
    behind the captured output -- and returns the pull's trailing stderr so a
    failure (auth, missing tag, rate limit) can be surfaced verbatim instead of
    collapsing into a generic "not found or pullable".
    """
    print(f"rlmesh: pulling image {image!r}...", file=sys.stderr, flush=True)
    proc = run_docker(["docker", "pull", image], timeout=DOCKER_PULL_TIMEOUT_SECONDS)
    stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
    return proc.returncode == 0, stderr_tail


def _log_resolution(source: str, kind: SourceKind, detail: str) -> None:
    # Autodetect must never be silent: always announce the resolved kind.
    print(
        f"rlmesh: resolved {source!r} -> {kind} ({detail})", file=sys.stderr, flush=True
    )


__all__ = [
    "DOCKER_PULL_TIMEOUT_SECONDS",
    "DOCKER_TIMEOUT_SECONDS",
    "SourceKind",
    "docker_image_exists",
    "docker_pull",
    "looks_like_gym_id",
    "resolve_source_kind",
    "run_docker",
]
