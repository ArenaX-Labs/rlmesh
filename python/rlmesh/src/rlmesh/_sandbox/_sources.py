"""Sandbox source classification: gym id vs Docker image, and image probing.

Lifecycle-free helpers used by :mod:`rlmesh._sandbox.session` and
:mod:`rlmesh._sandbox._model`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import warnings
from collections.abc import Mapping
from typing import Literal

SourceKind = Literal["build", "prebuilt"]

#: Upper bound for a quick Docker CLI call, so a wedged daemon errors instead of hanging.
DOCKER_TIMEOUT_SECONDS = 60.0

#: Upper bound for ``docker pull``; a large image over a slow link is legitimate.
DOCKER_PULL_TIMEOUT_SECONDS = 1800.0


def run_docker(
    cmd: list[str],
    *,
    timeout: float = DOCKER_TIMEOUT_SECONDS,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a Docker CLI command with a hang guard.

    ``env`` adds to the CLI's environment: a secret a ``-e NAME`` passes on
    without putting it in argv.

    A timeout becomes a ``RuntimeError`` naming the stalled command.
    """
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=None if env is None else {**os.environ, **env},
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Docker daemon not responding: '{' '.join(cmd[:2])}' did not "
            f"complete within {timeout:.0f}s; check that Docker is running "
            "and responsive, then retry"
        ) from exc


_GYM_VERSION_RE = re.compile(r"-v\d+$")

#: Docker-valid tagless name. Gymnasium ids are CamelCase or ``module:Env-v0``, which Docker refuses.
_COMPONENT = r"[a-z0-9]+(?:(?:\.|_|__|-+)[a-z0-9]+)*"
_DOCKER_NAME_RE = re.compile(rf"^{_COMPONENT}(?:/{_COMPONENT})*$")


def resolve_source_kind(
    source: str, *, prefer_build: bool = False
) -> tuple[SourceKind, str]:
    """Classify a sandbox source and return ``(kind, resolved_ref)``.

    * ``gym://`` / ``hf://`` -> build from source, never probed as a Docker image.
    * gym-versioned bare id (``-v<int>``) -> build from source, unless it is
      also a valid tagless Docker image name and a local image of it exists:
      then a registered gym id (or ``prefer_build``, set when build options
      were given) still builds, with a warning naming ``docker://``, and an id
      no installed gym package registers runs the image.
    * ``docker://img`` / ``image://img`` -> prebuilt (explicit).
    * bare image-shaped (``:tag`` / ``@sha256:``), local image -> prebuilt.
    * bare image-shaped, not local -> ``docker pull`` then prebuilt, else error.
    * bare tagless name (no scheme, no tag/digest, no gym version suffix) ->
      prebuilt only when the image exists locally (``docker image inspect``, no
      auto-pull); otherwise an error naming both ``docker://`` and ``gym://``.

    The resolved kind is always logged. An explicit ``gym://`` / ``docker://``
    scheme overrides the bare-source guess.
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
        return _resolve_gym_shaped(source, value, prefer_build=prefer_build)
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

    A local ``docker image inspect`` hit resolves it to prebuilt; anything else
    is an error naming both explicit spellings. Never auto-pulls.
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


def _resolve_gym_shaped(
    source: str, value: str, *, prefer_build: bool
) -> tuple[SourceKind, str]:
    """Classify a bare id with Gymnasium's ``-v<int>`` suffix.

    Builds from source unless a local image of the same tagless name exists.
    Then a registered gym id, or a build requested via build options, still
    builds and warns naming ``docker://``; any other id runs the image.
    """
    if (
        _DOCKER_NAME_RE.match(value) is None
        or shutil.which("docker") is None
        or not docker_image_exists(value)
    ):
        _log_resolution(source, "build", "gym id")
        return "build", value
    if prefer_build or gym_id_registered(value):
        why = (
            "build options were given" if prefer_build else "it is a registered gym id"
        )
        warnings.warn(
            f"sandbox source {source!r} names both a gym id and a local Docker "
            f"image; building the gym env from source because {why}. Use "
            f"docker://{value} to run the image, or gym://{value} to build "
            "without this warning",
            stacklevel=3,
        )
        _log_resolution(source, "build", "gym id; a local image has the same name")
        return "build", value
    _log_resolution(
        source, "prebuilt", "local Docker image; no installed gym package registers it"
    )
    return "prebuilt", value


def gym_id_registered(value: str) -> bool:
    """Whether a gym package installed here registers ``value``.

    Reads Gymnasium's registry without importing any other module, so an id
    a package registers only when imported reads as unregistered. ``False``
    when Gymnasium is not installed.
    """
    try:
        from gymnasium.envs.registration import registry
    except ImportError:
        return False
    return value in registry


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
    ``@sha256:`` digest. The gym id form ``pkg:Env-v0`` also carries a colon, so
    the gym version suffix is checked first.
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

    Announces the pull on stderr and returns its trailing stderr so a failure
    can be surfaced verbatim.
    """
    print(f"rlmesh: pulling image {image!r}...", file=sys.stderr, flush=True)
    proc = run_docker(["docker", "pull", image], timeout=DOCKER_PULL_TIMEOUT_SECONDS)
    stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
    return proc.returncode == 0, stderr_tail


def _log_resolution(source: str, kind: SourceKind, detail: str) -> None:
    print(
        f"rlmesh: resolved {source!r} -> {kind} ({detail})", file=sys.stderr, flush=True
    )


__all__ = [
    "DOCKER_PULL_TIMEOUT_SECONDS",
    "DOCKER_TIMEOUT_SECONDS",
    "SourceKind",
    "docker_image_exists",
    "docker_pull",
    "gym_id_registered",
    "looks_like_gym_id",
    "resolve_source_kind",
    "run_docker",
]
