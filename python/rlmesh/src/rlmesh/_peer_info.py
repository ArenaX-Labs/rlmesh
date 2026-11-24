"""Best-effort collection of this Python runtime's identity for the handshake."""

from __future__ import annotations

import platform
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from typing import TypedDict

# Read from distribution metadata only; these are deliberately not imported.
_FRAMEWORK_DISTRIBUTIONS: tuple[str, ...] = (
    "numpy",
    "torch",
    "jax",
    "jaxlib",
    "gymnasium",
)

_COMPONENT = "rlmesh-python"


def _safe(fn: Callable[[], object]) -> str:
    try:
        value = fn()
    except Exception:
        return ""
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def _normalize_os(system: str) -> str:
    mapping = {
        "darwin": "macos",
        "linux": "linux",
        "windows": "windows",
    }
    lowered = system.strip().lower()
    return mapping.get(lowered, lowered)


def _framework_versions() -> dict[str, str]:
    """Versions of installed frameworks, from distribution metadata only."""
    versions: dict[str, str] = {}
    for name in _FRAMEWORK_DISTRIBUTIONS:
        try:
            versions[name] = _dist_version(name)
        except PackageNotFoundError:
            continue
        except Exception:
            continue
    return versions


def _package_version() -> str:
    try:
        return _dist_version("rlmesh")
    except Exception:
        return ""


class PeerInfoDict(TypedDict):
    """Advisory handshake identity collected from the Python runtime."""

    component: str
    language: str
    language_version: str
    package_version: str
    os: str
    os_version: str
    arch: str
    framework_versions: dict[str, str]


def collect_peer_info() -> PeerInfoDict:
    """Gather this Python runtime's advisory handshake identity.

    Returns a dict suitable for ``rlmesh._rlmesh._set_python_peer_info(**info)``.
    Every value is best-effort; this function never raises.
    """
    return PeerInfoDict(
        component=_COMPONENT,
        language="python",
        language_version=_safe(platform.python_version),
        package_version=_package_version(),
        os=_normalize_os(_safe(platform.system)),
        os_version=_safe(platform.release),
        arch=_safe(platform.machine),
        framework_versions=_framework_versions(),
    )


def register_python_peer_info(extra: dict[str, str] | None = None) -> None:
    """Install the collected identity into the native handshake builder.

    ``extra`` adds advisory key/value facts to the handshake ``PeerInfo.extra``
    map beside the build keys (the serve entrypoint stamps its startup phase
    marks this way). Each call replaces the whole override, so pass everything
    the peer should report.

    Best-effort and idempotent: a missing native symbol or any collection error
    is swallowed so importing the package never fails over advisory diagnostics.
    """
    try:
        from ._rlmesh import _set_python_peer_info  # type: ignore[attr-defined]
    except Exception:
        return

    try:
        info = collect_peer_info()
        _set_python_peer_info(
            language=info["language"],
            language_version=info["language_version"],
            package_version=info["package_version"],
            os=info["os"],
            os_version=info["os_version"],
            arch=info["arch"],
            framework_versions=info["framework_versions"],
            extra=dict(extra or {}),
        )
    except Exception:
        return


__all__ = [
    "collect_peer_info",
    "register_python_peer_info",
]
