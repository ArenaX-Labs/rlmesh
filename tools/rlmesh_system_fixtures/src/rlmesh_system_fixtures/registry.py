from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pkgutil import iter_modules
from typing import TypeVar, cast

EnvFactory = Callable[..., object]
ModelFactory = Callable[[object], object]
FixtureT = TypeVar("FixtureT", bound=Callable[..., object])

_ENV_FIXTURES: dict[str, EnvFactory] = {}
_MODEL_FIXTURES: dict[str, ModelFactory] = {}
_DISCOVERED = False


def env_fixture(name: str) -> Callable[[FixtureT], FixtureT]:
    return _register(name, _ENV_FIXTURES)


def model_fixture(name: str) -> Callable[[FixtureT], FixtureT]:
    return _register(name, _MODEL_FIXTURES)


def make_env(fixture: str, kwargs: dict[str, object] | None = None) -> object:
    discover_fixtures()
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

    discover_fixtures()
    try:
        return _MODEL_FIXTURES[name_or_entrypoint]
    except KeyError as exc:
        raise ValueError(
            unknown_fixture_message("model", name_or_entrypoint, _MODEL_FIXTURES)
        ) from exc


def list_env_fixtures() -> tuple[str, ...]:
    discover_fixtures()
    return tuple(sorted(_ENV_FIXTURES))


def list_model_fixtures() -> tuple[str, ...]:
    discover_fixtures()
    return tuple(sorted(_MODEL_FIXTURES))


def discover_fixtures() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    import_fixture_modules("rlmesh_system_fixtures.envs")
    import_fixture_modules("rlmesh_system_fixtures.models")


def import_fixture_modules(package_name: str) -> None:
    package = import_module(package_name)
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return
    for module in iter_modules(package_path, f"{package_name}."):
        if not module.ispkg:
            import_module(module.name)


def resolve_dotted_entrypoint(entrypoint: str) -> object:
    module_name, _, attribute_path = entrypoint.partition(":")
    if not module_name or not attribute_path:
        raise ValueError(f"entrypoint must use 'module:attribute', got {entrypoint!r}")

    value: object = import_module(module_name)
    for attribute in attribute_path.split("."):
        value = getattr(value, attribute)
    return value


def _register(
    name: str, registry: dict[str, Callable[..., object]]
) -> Callable[[FixtureT], FixtureT]:
    def decorator(func: FixtureT) -> FixtureT:
        existing = registry.get(name)
        if existing is not None and existing is not func:
            raise ValueError(f"duplicate RLMesh system fixture {name!r}")
        registry[name] = func
        return func

    return decorator


def unknown_fixture_message(
    fixture_type: str,
    fixture: str,
    registry: dict[str, Callable[..., object]],
) -> str:
    available = ", ".join(sorted(registry)) or "<none>"
    return f"unknown {fixture_type} fixture {fixture!r}; available: {available}"
