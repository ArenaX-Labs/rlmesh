"""Helpers for RLMesh checkout-only CLI fallback."""

from __future__ import annotations

from pathlib import Path


def find_repo_root() -> Path | None:
    """Resolve the workspace root when running from an editable checkout."""
    current_file = safe_resolve(Path(__file__))
    if current_file is None:
        return None

    for parent in current_file.parents:
        if not (parent / "Cargo.toml").exists():
            continue
        if not (parent / "python" / "rlmesh" / "pyproject.toml").exists():
            continue
        if not (parent / "crates" / "rlmesh-cli" / "Cargo.toml").exists():
            continue
        return parent

    return None


def safe_resolve(path: Path) -> Path | None:
    """Resolve a path if possible."""
    try:
        return path.resolve()
    except OSError:
        return None


__all__ = ["find_repo_root", "safe_resolve"]
