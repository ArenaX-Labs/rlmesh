from __future__ import annotations

import importlib.metadata

import pytest
from rlmesh._peer_info import collect_peer_info, register_python_peer_info


def test_collect_peer_info_reports_python_runtime() -> None:
    info = collect_peer_info()

    assert info["component"] == "rlmesh-python"
    assert info["language"] == "python"
    # Best-effort but always available on a real interpreter.
    assert isinstance(info["language_version"], str)
    assert info["language_version"]
    assert isinstance(info["arch"], str)
    assert info["arch"]


def test_collect_peer_info_includes_installed_framework_versions() -> None:
    info = collect_peer_info()
    frameworks = info["framework_versions"]

    assert isinstance(frameworks, dict)
    # numpy is a test/dev dependency, so its version must be reported.
    assert "numpy" in frameworks
    assert frameworks["numpy"]


def test_collect_peer_info_normalizes_os_to_rust_vocabulary() -> None:
    import platform

    info = collect_peer_info()
    system = platform.system().strip().lower()
    expected = {"darwin": "macos", "linux": "linux", "windows": "windows"}.get(
        system, system
    )
    assert info["os"] == expected


def test_framework_versions_skip_uninstalled_without_importing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frameworks come from dist metadata; uninstalled ones are skipped cleanly."""
    real_version = importlib.metadata.version

    def fake_version(name: str) -> str:
        if name == "torch":
            raise importlib.metadata.PackageNotFoundError(name)
        return real_version(name)

    # Patch the symbol the helper imported at module load.
    import rlmesh._peer_info as peer_info_mod

    monkeypatch.setattr(peer_info_mod, "_dist_version", fake_version)

    frameworks = collect_peer_info()["framework_versions"]
    assert "torch" not in frameworks
    # numpy is still present; collection did not raise.
    assert "numpy" in frameworks


def test_register_python_peer_info_is_robust_to_missing_native_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registration never raises even if the native entry point is unavailable."""
    import rlmesh._rlmesh as native

    monkeypatch.delattr(native, "_set_python_peer_info", raising=False)
    # Should swallow the missing symbol rather than raise on import paths.
    register_python_peer_info()


def test_native_set_python_peer_info_accepts_collected_info() -> None:
    """The collected dict round-trips through the native entry point."""
    from rlmesh._rlmesh import _set_python_peer_info

    info = collect_peer_info()
    # Smoke test: installing the override must not raise; component is fixed
    # Rust-side per call site, so it is not forwarded.
    _set_python_peer_info(
        language=str(info["language"]),
        language_version=str(info["language_version"]),
        package_version=str(info["package_version"]),
        os=str(info["os"]),
        os_version=str(info["os_version"]),
        arch=str(info["arch"]),
        framework_versions=dict(info["framework_versions"]),  # type: ignore[arg-type]
    )
