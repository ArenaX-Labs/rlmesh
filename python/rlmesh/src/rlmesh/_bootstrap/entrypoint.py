"""Shared module:callable entrypoint loading helpers."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, cast


def parse_entrypoint(value: str, *, label: str = "entrypoint") -> tuple[str, str]:
    """Parse ``module:callable`` entrypoint syntax."""
    module_name, sep, callable_path = value.partition(":")
    segments = callable_path.split(".")
    if (
        not sep
        or not module_name
        or not callable_path
        or any(not segment for segment in segments)
    ):
        raise ValueError(f"{label} must be in module:callable form")
    return module_name, callable_path


def resolve_entrypoint(value: str, *, label: str = "entrypoint") -> Callable[..., Any]:
    """Resolve a ``module:callable`` entrypoint to a callable object."""
    module_name, callable_path = parse_entrypoint(value, label=label)
    module = importlib.import_module(module_name)

    resolved: object = module
    for segment in callable_path.split("."):
        try:
            resolved = getattr(resolved, segment)
        except AttributeError as exc:
            raise AttributeError(
                f"{label} {value!r} could not resolve {segment!r}"
            ) from exc

    if not callable(resolved):
        raise TypeError(f"{label} {value!r} did not resolve to a callable")

    return cast(Callable[..., Any], resolved)


__all__ = ["parse_entrypoint", "resolve_entrypoint"]
