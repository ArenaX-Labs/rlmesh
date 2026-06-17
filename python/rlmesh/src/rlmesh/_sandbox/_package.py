"""Normalize the ``rlmesh_package`` argument shared by the sandbox helpers."""

from __future__ import annotations

from os import PathLike, fspath


def normalize_rlmesh_package(value: str | PathLike[str] | None) -> str | None:
    if value is None:
        return None
    return fspath(value)


__all__ = ["normalize_rlmesh_package"]
