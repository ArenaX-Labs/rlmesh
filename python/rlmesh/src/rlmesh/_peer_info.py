"""Best-effort collection of this Python runtime's identity for the handshake.

The native handshake stamps every peer with an advisory ``PeerInfo`` (language,
runtime version, OS, arch, framework versions) for operator debugging. The Rust
side fills Rust defaults; this module gathers the *Python* host's real runtime
and hands it to the native module (``rlmesh._rlmesh._set_python_peer_info``),
which installs a process-wide override the Rust handshake builder merges in.

Design notes:

* Every field is best-effort. Collection never raises: a failure to read any
  one field yields an empty value (which the Rust merge replaces with its
  detected fallback for ``os``/``arch``/``package_version``).
* Framework versions are read from *distribution metadata*
  (:func:`importlib.metadata.version`), never by importing the framework. This
  keeps import-time light and avoids forcing an optional heavy dependency
  (numpy/torch/jax/jaxlib/gymnasium) to load.

This is advisory diagnostics only; ``PeerInfo`` never gates compatibility.
"""

from __future__ import annotations

import platform
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from typing import TypedDict

# Frameworks whose installed version is worth reporting for debugging. Read from
# distribution metadata only — these are deliberately not imported here.
_FRAMEWORK_DISTRIBUTIONS: tuple[str, ...] = (
    "numpy",
    "torch",
    "jax",
    "jaxlib",
    "gymnasium",
)

# Component label for a Python-hosted env/model peer.
_COMPONENT = "rlmesh-python"


def _safe(fn: Callable[[], object]) -> str:
    """Call ``fn`` and return its string result, or ``""`` on any failure."""
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
    """Map :func:`platform.system` to the Rust ``std::env::consts::OS`` vocab.

    Rust reports ``"macos"``/``"windows"``/``"linux"``; Python's
    ``platform.system()`` reports ``"Darwin"``/``"Windows"``/``"Linux"``. We
    align the common ones and otherwise pass the lowercased value through (the
    Rust merge keeps its own detected ``os`` when this is empty).
    """
    mapping = {
        "darwin": "macos",
        "linux": "linux",
        "windows": "windows",
    }
    lowered = system.strip().lower()
    return mapping.get(lowered, lowered)


def _framework_versions() -> dict[str, str]:
    """Versions of installed frameworks, from distribution metadata only.

    Skips any framework that is not installed (``PackageNotFoundError``) and any
    that raises for any other reason. Never imports the framework itself.
    """
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
    """This SDK's installed distribution version, best-effort."""
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


def register_python_peer_info() -> None:
    """Install the collected identity into the native handshake builder.

    Best-effort and idempotent: a missing native symbol or any collection error
    is swallowed so importing the package never fails over advisory diagnostics.
    """
    try:
        from ._rlmesh import _set_python_peer_info  # type: ignore[attr-defined]
    except Exception:
        return

    try:
        info = collect_peer_info()
        # `component` is fixed Rust-side per call site, so it is not forwarded.
        _set_python_peer_info(
            language=info["language"],
            language_version=info["language_version"],
            package_version=info["package_version"],
            os=info["os"],
            os_version=info["os_version"],
            arch=info["arch"],
            framework_versions=info["framework_versions"],
        )
    except Exception:
        return


__all__ = [
    "collect_peer_info",
    "register_python_peer_info",
]
