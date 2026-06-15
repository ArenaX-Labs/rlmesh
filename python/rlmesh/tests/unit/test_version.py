from __future__ import annotations

import importlib

import pytest


def test_version_fallback_uses_native_extension_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When distribution metadata is missing, fall back to the native version.

    The compiled extension now defines __version__ (CARGO_PKG_VERSION), so a
    missing distribution (PyInstaller/zipapp bundle, vendored copy without
    dist-info) surfaces that real version instead of a misleading placeholder.
    """
    import importlib.metadata as importlib_metadata

    def raise_not_found(name: str) -> str:
        raise importlib_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib_metadata, "version", raise_not_found)

    import rlmesh

    native_version = rlmesh._rlmesh.__version__

    reloaded = importlib.reload(rlmesh)
    try:
        assert reloaded.__version__ == native_version
    finally:
        # Restore the real version for other tests sharing the module.
        importlib.reload(reloaded)
