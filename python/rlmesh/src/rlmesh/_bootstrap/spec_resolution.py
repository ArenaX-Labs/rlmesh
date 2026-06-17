"""Bootstrap spec resolution and JSON value coercion."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast


class BootstrapUsageError(ValueError):
    """Bad sandbox-bootstrap CLI usage (argv vs env-var source mismatch)."""


def resolve_bootstrap_spec(argv: Sequence[str], *, prog: str) -> Mapping[str, object]:
    """Resolve the bootstrap spec from its source.

    ``RLMESH_BOOTSTRAP_JSON`` (the inline run-time payload, wins) > a
    ``bootstrap.json`` path passed as the sole argument. Both carry a
    ``{"spec": ...}`` payload (a gym or hf bootstrap spec).
    """
    inline = os.environ.get("RLMESH_BOOTSTRAP_JSON")
    if inline is not None:
        if argv:
            raise BootstrapUsageError(
                f"usage: {prog} (set RLMESH_BOOTSTRAP_JSON, no arguments)"
            )
        payload = expect_mapping(cast(object, json.loads(inline)), "bootstrap payload")
        return expect_mapping(payload.get("spec"), "bootstrap spec")

    if len(argv) != 1:
        raise BootstrapUsageError(
            f"usage: {prog} <bootstrap.json> (or set RLMESH_BOOTSTRAP_JSON)"
        )
    payload = expect_mapping(
        cast(object, json.loads(Path(argv[0]).read_text(encoding="utf-8"))),
        "bootstrap payload",
    )
    return expect_mapping(payload.get("spec"), "bootstrap spec")


def select_mapping_item(
    mapping: Mapping[str, object], key: str | None, label: str
) -> tuple[str, object]:
    """Select an explicit or sole item from a mapping."""
    if key is not None:
        if key not in mapping:
            raise KeyError(
                f"{label} {key!r} was not found; "
                f"available {label}s: {_format_mapping_keys(mapping)}"
            )
        return key, mapping[key]

    if not mapping:
        raise ValueError(f"no {label}s found")
    if len(mapping) != 1:
        raise ValueError(
            f"multiple {label}s found; specify one in the source URI: "
            f"{_format_mapping_keys(mapping)}"
        )
    selected_key = next(iter(mapping))
    return selected_key, mapping[selected_key]


def select_task_item(
    mapping: Mapping[object, object], key: str | None, suite: str
) -> tuple[object, object]:
    """Select an explicit or sole task from a nested EnvHub mapping."""
    if key is not None:
        matches = [
            (task_key, value)
            for task_key, value in mapping.items()
            if str(task_key) == key
        ]
        if not matches:
            raise KeyError(
                f"task {key!r} was not found in suite {suite!r}; "
                f"available tasks: {_format_mapping_keys(mapping)}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"multiple tasks in suite {suite!r} match {key!r}; "
                f"available tasks: {_format_mapping_keys(mapping)}"
            )
        return matches[0]

    if not mapping:
        raise ValueError(f"no tasks found in suite {suite!r}")
    if len(mapping) != 1:
        raise ValueError(
            f"multiple tasks found in suite {suite!r}; specify one in the source URI: "
            f"{_format_mapping_keys(mapping)}"
        )
    selected_key = next(iter(mapping))
    return selected_key, mapping[selected_key]


def expect_mapping(value: object, label: str) -> Mapping[str, object]:
    """Validate a bootstrap mapping with string keys."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    raw_mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in raw_mapping.keys()):
        raise TypeError(f"{label} keys must be strings")
    return cast(Mapping[str, object], raw_mapping)


def optional_mapping(value: object, label: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    return expect_mapping(value, label)


def optional_any_mapping(value: object, label: str) -> Mapping[object, object] | None:
    _ = label
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return None
    return cast(Mapping[object, object], value)


def _format_mapping_keys(mapping: Mapping[Any, object]) -> str:
    return ", ".join(sorted(str(key) for key in mapping.keys())) or "<none>"


def expect_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def optional_str(value: object, label: str) -> str | None:
    if value is None:
        return None
    return expect_str(value, label)


def expect_num_envs(value: object, label: str) -> int:
    if value is None:
        return 1
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    if value < 1:
        raise ValueError(f"{label} must be at least 1")
    return value


def expect_vectorization_mode(value: object, label: str) -> str:
    if value is None:
        return "sync"
    value = expect_str(value, label)
    if value not in {"sync", "async"}:
        raise ValueError(f"{label} must be 'sync' or 'async'")
    return value


def expect_str_list(value: object, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list[str]")
    items = cast(list[object], value)
    if not all(isinstance(item, str) for item in items):
        raise TypeError(f"{label} must be a list[str]")
    return cast(list[str], items)


def mapping_to_kwargs(value: object, label: str) -> dict[str, object]:
    mapping = optional_mapping(value, label)
    if mapping is None:
        return {}
    return dict(mapping)
