"""Bootstrap spec resolution, member-param layering, and JSON value coercion."""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from rlmesh.recipes import Recipe


class BootstrapUsageError(ValueError):
    """Bad sandbox-bootstrap CLI usage (argv vs env-var source mismatch)."""


def resolve_bootstrap_spec(argv: Sequence[str], *, prog: str) -> Mapping[str, object]:
    """Resolve the bootstrap spec from its source, in precedence order.

    ``RLMESH_BOOTSTRAP_JSON`` (inline payload, wins) > ``RLMESH_RECIPE_PATH`` (the
    baked self-describing recipe.json) > a ``bootstrap.json`` path passed as the
    sole argument. The first and last carry a ``{"spec": ...}`` payload; the baked
    recipe.json is a bare recipe document, wrapped here into a ``kind="recipe"``
    spec (its ``num_envs``/``vectorization_mode`` come from the flat
    ``RLMESH_NUM_ENVS``/``RLMESH_VECTORIZATION_MODE`` vars, applied by the caller).
    """
    inline = os.environ.get("RLMESH_BOOTSTRAP_JSON")
    if inline is not None:
        if argv:
            raise BootstrapUsageError(
                f"usage: {prog} (set RLMESH_BOOTSTRAP_JSON, no arguments)"
            )
        payload = expect_mapping(cast(object, json.loads(inline)), "bootstrap payload")
        return expect_mapping(payload.get("spec"), "bootstrap spec")

    recipe_path = os.environ.get("RLMESH_RECIPE_PATH")
    if recipe_path is not None:
        if argv:
            raise BootstrapUsageError(
                f"usage: {prog} (set RLMESH_RECIPE_PATH, no arguments)"
            )
        document = cast(
            object, json.loads(Path(recipe_path).read_text(encoding="utf-8"))
        )
        return {
            "kind": "recipe",
            "document": expect_mapping(document, "recipe document"),
        }

    if len(argv) != 1:
        raise BootstrapUsageError(
            f"usage: {prog} <bootstrap.json> "
            "(or set RLMESH_BOOTSTRAP_JSON / RLMESH_RECIPE_PATH)"
        )
    payload = expect_mapping(
        cast(object, json.loads(Path(argv[0]).read_text(encoding="utf-8"))),
        "bootstrap payload",
    )
    return expect_mapping(payload.get("spec"), "bootstrap spec")


def member_params_from_env() -> tuple[dict[str, object], dict[str, object]]:
    """Parse ``RLMESH_PARAMS_JSON`` into ``(setup_env, kwargs)`` (empty when unset).

    The member selector ``{"setup_env": {...}, "kwargs": {...}}`` is layered over a
    recipe's ``setup.env`` (declared keys only) and ``make.kwargs`` at startup, so
    one image serves any declared member.
    """
    raw = os.environ.get("RLMESH_PARAMS_JSON")
    if not raw:
        return {}, {}
    data = expect_mapping(cast(object, json.loads(raw)), "RLMESH_PARAMS_JSON")
    return (
        mapping_to_kwargs(data.get("setup_env"), "RLMESH_PARAMS_JSON.setup_env"),
        mapping_to_kwargs(data.get("kwargs"), "RLMESH_PARAMS_JSON.kwargs"),
    )


def apply_member_params(
    recipe: Recipe,
    *,
    setup_env: Mapping[str, object] | None = None,
    kwargs: Mapping[str, object] | None = None,
) -> Recipe:
    """Layer the member selector over a recipe, returning a new frozen recipe.

    ``setup_env`` overrides ``setup.env`` only for keys the recipe DECLARES in
    ``setup.params``; undeclared keys are silently ignored -- the managed shim
    deliberately emits extra selector keys that stay as flat env. ``kwargs`` merges
    over ``make.kwargs``.
    """
    new = recipe
    if setup_env:
        setup = new.setup
        declared = set(setup.params)
        applied = {
            key: str(value) for key, value in setup_env.items() if key in declared
        }
        if applied:
            merged = {**setup.env, **applied}
            new = dataclasses.replace(new, setup=dataclasses.replace(setup, env=merged))
    if kwargs and new.make is not None:
        merged_kwargs = {**dict(new.make.kwargs), **kwargs}
        new = dataclasses.replace(
            new, make=dataclasses.replace(new.make, kwargs=merged_kwargs)
        )
    return new


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
