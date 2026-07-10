from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import cast

from rlmesh_system_fixtures.envs.counter import make_counter
from rlmesh_system_fixtures.envs.image_grid import make_image_grid
from rlmesh_system_fixtures.models.discrete import discrete_zero
from rlmesh_system_fixtures.models.gymnasium import pendulum_zero_numpy
from rlmesh_system_fixtures.models.image_grid import (
    image_grid_numpy_action,
    image_grid_torch_action,
)
from rlmesh_system_fixtures.models.mujoco import halfcheetah_zero_numpy

EnvFactory = Callable[..., object]
ModelFactory = Callable[[object], object]

_ENV_FIXTURES: dict[str, EnvFactory] = {
    "counter": make_counter,
    "image-grid": make_image_grid,
}

_MODEL_FIXTURES: dict[str, ModelFactory] = {
    "discrete.zero": discrete_zero,
    "gymnasium.pendulum_zero_numpy": pendulum_zero_numpy,
    "image_grid.numpy_action": image_grid_numpy_action,
    "image_grid.torch_action": image_grid_torch_action,
    "mujoco.halfcheetah_zero_numpy": halfcheetah_zero_numpy,
}


def make_env(fixture: str, kwargs: dict[str, object] | None = None) -> object:
    try:
        factory = _ENV_FIXTURES[fixture]
    except KeyError as exc:
        raise ValueError(
            unknown_fixture_message("env", fixture, _ENV_FIXTURES)
        ) from exc
    return factory(**(kwargs or {}))


def resolve_model(name_or_entrypoint: str) -> ModelFactory:
    if ":" in name_or_entrypoint:
        return cast(ModelFactory, resolve_dotted_entrypoint(name_or_entrypoint))

    try:
        return _MODEL_FIXTURES[name_or_entrypoint]
    except KeyError as exc:
        raise ValueError(
            unknown_fixture_message("model", name_or_entrypoint, _MODEL_FIXTURES)
        ) from exc


def list_env_fixtures() -> tuple[str, ...]:
    return tuple(sorted(_ENV_FIXTURES))


def list_model_fixtures() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_FIXTURES))


def resolve_dotted_entrypoint(entrypoint: str) -> object:
    module_name, _, attribute_path = entrypoint.partition(":")
    if not module_name or not attribute_path:
        raise ValueError(f"entrypoint must use 'module:attribute', got {entrypoint!r}")

    value: object = import_module(module_name)
    for attribute in attribute_path.split("."):
        value = getattr(value, attribute)
    return value


def unknown_fixture_message(
    fixture_type: str,
    fixture: str,
    registry: dict[str, Callable[..., object]],
) -> str:
    available = ", ".join(sorted(registry)) or "<none>"
    return f"unknown {fixture_type} fixture {fixture!r}; available: {available}"
