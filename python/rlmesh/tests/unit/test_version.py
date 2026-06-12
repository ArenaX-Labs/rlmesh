from __future__ import annotations

import importlib

import pytest


def test_version_fallback_is_explicit_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When distribution metadata is missing, report an honest unknown marker.

    The compiled extension never defines __version__, so the old
    getattr(_rlmesh, "__version__", "0.0.0") fallback always produced a
    misleading concrete "0.0.0". Reloading the package with a missing
    distribution must instead surface "0+unknown".
    """
    import importlib.metadata as importlib_metadata

    def raise_not_found(name: str) -> str:
        raise importlib_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib_metadata, "version", raise_not_found)

    import rlmesh

    reloaded = importlib.reload(rlmesh)
    try:
        assert reloaded.__version__ == "0+unknown"
    finally:
        # Restore the real version for other tests sharing the module.
        importlib.reload(reloaded)
