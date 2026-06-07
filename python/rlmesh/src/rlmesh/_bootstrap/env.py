"""Private environment loaders shared by CLI and sandbox entrypoints."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

from rlmesh._bootstrap.entrypoint import resolve_entrypoint

if TYPE_CHECKING:
    from rlmesh.server import EnvLike


def load_environment(
    env_id: str,
    package_names: Sequence[str],
    num_envs: int,
    vectorization_mode: str | None = None,
    kwargs: Mapping[str, object] | None = None,
) -> EnvLike:
    """Load a Gymnasium/Gym environment for the interactive CLI."""
    import_errors: list[str] = []
    imports = list(package_names)
    import_packages(imports)

    for module_name in ("gymnasium", "gym"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            import_errors.append(module_name)
            continue

        try:
            return cast(
                "EnvLike",
                make_gym_environment(
                    module,
                    env_id=env_id,
                    kwargs=dict(kwargs or {}),
                    num_envs=num_envs,
                    vectorization_mode=vectorization_mode,
                ),
            )
        except Exception as exc:
            if is_env_lookup_error(exc):
                continue
            raise

    msg = f"Unable to load '{env_id}'."
    if import_errors:
        msg = f"{msg} Missing modules: {', '.join(import_errors)}."
    if not imports:
        msg = f"{msg} If the env is registered by another package, pass --package."
    raise ImportError(msg)


def load_env_from_spec(spec: Mapping[str, object]) -> object:
    """Load an environment from a sandbox bootstrap spec."""
    kind = _expect_str(spec.get("kind"), "bootstrap spec.kind")
    if kind == "gym":
        return load_gym_env(spec)
    if kind == "hf":
        return load_hf_env(spec)
    raise ValueError(f"unsupported bootstrap kind: {kind}")


def load_env_entrypoint(
    entrypoint: str,
    package_names: Sequence[str] = (),
    kwargs: Mapping[str, object] | None = None,
) -> EnvLike:
    """Load an environment from a ``module:callable`` factory entrypoint."""
    import_packages(package_names)
    factory = resolve_entrypoint(entrypoint, label="env entrypoint")
    env = factory(**dict(kwargs or {}))
    if not looks_like_env(env):
        raise TypeError(
            f"env entrypoint {entrypoint!r} did not return an environment "
            "with reset(...) and step(...)"
        )
    return cast("EnvLike", env)


def load_gym_env(spec: Mapping[str, object]) -> object:
    """Load a Gymnasium/Gym environment from a sandbox bootstrap spec."""
    import_packages(_expect_str_list(spec.get("imports"), "bootstrap imports"))

    env_id = _expect_str(spec.get("env_id"), "bootstrap spec.env_id")
    kwargs = _mapping_to_kwargs(spec.get("kwargs"), "bootstrap spec.kwargs")
    num_envs = _expect_num_envs(spec.get("num_envs"), "bootstrap spec.num_envs")
    vectorization_mode = _expect_vectorization_mode(
        spec.get("vectorization_mode"), "bootstrap spec.vectorization_mode"
    )

    errors: list[tuple[str, Exception]] = []
    for gym_module in import_gym_modules():
        try:
            return make_gym_environment(
                gym_module,
                env_id=env_id,
                kwargs=kwargs,
                num_envs=num_envs,
                vectorization_mode=vectorization_mode,
            )
        except Exception as exc:
            errors.append((gym_module.__name__, exc))

    if errors:
        names = ", ".join(name for name, _ in errors)
        first_error = errors[0][1]
        raise RuntimeError(
            f"failed to create gym environment {env_id!r} with {names}"
        ) from first_error

    raise ImportError("gymnasium or gym must be installed in the sandbox container")


def make_gym_environment(
    gym_module: object,
    *,
    env_id: str,
    kwargs: Mapping[str, object],
    num_envs: int,
    vectorization_mode: str | None,
) -> object:
    """Construct a single or vectorized Gymnasium/Gym environment."""
    env_kwargs = dict(kwargs)
    make = _load_callable(gym_module, "make")
    if num_envs <= 1:
        return make(env_id, **env_kwargs)

    make_vec = getattr(gym_module, "make_vec", None)
    if callable(make_vec):
        make_vec_kwargs: dict[str, object] = {"num_envs": num_envs, **env_kwargs}
        if vectorization_mode is not None:
            make_vec_kwargs["vectorization_mode"] = vectorization_mode
        return make_vec(env_id, **make_vec_kwargs)

    vector_module = getattr(gym_module, "vector", None)
    if vector_module is None:
        raise ValueError(
            f"module '{getattr(gym_module, '__name__', '<unknown>')}' does not expose vector env helpers"
        )

    vector_cls_name = (
        "AsyncVectorEnv" if vectorization_mode == "async" else "SyncVectorEnv"
    )
    vector_cls = getattr(vector_module, vector_cls_name, None)
    if not callable(vector_cls):
        raise ValueError(
            f"module '{getattr(gym_module, '__name__', '<unknown>')}' does not expose {vector_cls_name}"
        )

    factory = cast(Callable[[list[Callable[[], object]]], object], vector_cls)

    def make_one() -> object:
        return make(env_id, **env_kwargs)

    return factory([make_one for _ in range(num_envs)])


def load_hf_env(spec: Mapping[str, object]) -> object:
    """Load an HF-materialized environment from a sandbox bootstrap spec."""
    import_packages(_expect_str_list(spec.get("imports"), "bootstrap imports"))

    source_subdir = _expect_str(
        spec.get("source_subdir"), "bootstrap spec.source_subdir"
    )
    source_root = Path("/opt/rlmesh") / source_subdir
    env_py = source_root / "env.py"
    if not env_py.exists():
        raise FileNotFoundError(f"missing env.py in {source_root}")

    source_root_str = str(source_root)
    if source_root_str not in sys.path:
        sys.path.insert(0, source_root_str)

    module = load_module_from_path("rlmesh_hf_env", env_py)
    make_env = _load_callable(module, "make_env")

    kwargs = _mapping_to_kwargs(spec.get("kwargs"), "bootstrap spec.kwargs")
    num_envs = _expect_num_envs(spec.get("num_envs"), "bootstrap spec.num_envs")
    vectorization_mode = _expect_vectorization_mode(
        spec.get("vectorization_mode"), "bootstrap spec.vectorization_mode"
    )

    envs = _call_hf_make_env(
        make_env,
        kwargs,
        num_envs=num_envs,
        vectorization_mode=vectorization_mode,
    )
    suite = _optional_str(spec.get("suite"), "bootstrap spec.suite")
    return normalize_hf_env(envs, suite=suite)


def import_packages(package_names: Sequence[str]) -> None:
    """Import packages that register environments on import."""
    for module_name in package_names:
        try:
            _ = importlib.import_module(module_name)
        except ImportError:
            msg = f"Unable to import package '{module_name}'."
            raise ImportError(msg) from None


def import_gym_modules() -> list[ModuleType]:
    """Import supported Gym modules in preference order."""
    modules: list[ModuleType] = []
    for module_name in ("gymnasium", "gym"):
        try:
            modules.append(importlib.import_module(module_name))
        except ImportError:
            continue
    return modules


def is_env_lookup_error(exc: Exception) -> bool:
    """Return whether an exception means the env id was not registered."""
    return exc.__class__.__name__ in {
        "NameNotFound",
        "NamespaceNotFound",
        "UnregisteredEnv",
    }


def _call_hf_make_env(
    make_env: Callable[..., object],
    kwargs: dict[str, object],
    *,
    num_envs: int,
    vectorization_mode: str,
) -> object:
    call_kwargs = dict(kwargs)
    accepts_kwargs, keyword_names = _callable_keyword_parameters(make_env)

    if "n_envs" not in call_kwargs:
        if accepts_kwargs or "n_envs" in keyword_names:
            call_kwargs["n_envs"] = num_envs
        elif num_envs != 1:
            raise TypeError(
                "HF sandbox source requested num_envs="
                f"{num_envs}, but env.py make_env(...) does not accept n_envs"
            )

    if "use_async_envs" not in call_kwargs:
        if accepts_kwargs or "use_async_envs" in keyword_names:
            call_kwargs["use_async_envs"] = vectorization_mode == "async"
        elif vectorization_mode == "async":
            raise TypeError(
                "HF sandbox source requested async vectorization, but env.py "
                "make_env(...) does not accept use_async_envs"
            )

    return make_env(**call_kwargs)


def _callable_keyword_parameters(
    value: Callable[..., object],
) -> tuple[bool, set[str]]:
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return True, set()

    accepts_kwargs = False
    keyword_names: set[str] = set()
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_kwargs = True
        elif parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            keyword_names.add(parameter.name)
    return accepts_kwargs, keyword_names


def normalize_hf_env(envs: object, *, suite: str | None) -> object:
    """Select the served environment from an HF make_env result."""
    if looks_like_env(envs):
        return envs

    env_mapping = _optional_mapping(envs, "hf env mapping")
    if env_mapping is None:
        raise TypeError("env.py make_env(...) returned an unsupported value")

    _, suite_value = select_mapping_item(env_mapping, suite, "suite")
    if looks_like_env(suite_value):
        return suite_value

    raise TypeError(
        "selected suite did not resolve to an env; minimal HF sandbox support "
        "does not include nested task selection"
    )


def select_mapping_item(
    mapping: Mapping[str, object], key: str | None, label: str
) -> tuple[str, object]:
    """Select an explicit or sole item from a mapping."""
    if key is not None:
        if key not in mapping:
            raise KeyError(f"{label} {key!r} was not found")
        return key, mapping[key]

    if len(mapping) != 1:
        raise ValueError(f"multiple {label}s found; specify one in the source URI")
    selected_key = next(iter(mapping))
    return selected_key, mapping[selected_key]


def load_module_from_path(name: str, path: Path) -> ModuleType:
    """Import a module from a concrete Python source path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def looks_like_env(value: object) -> bool:
    """Return whether a value has the minimum env methods RLMesh serves."""
    return hasattr(value, "reset") and hasattr(value, "step")


def _load_callable(module: object, name: str) -> Callable[..., object]:
    value = getattr(module, name, None)
    module_name = getattr(module, "__name__", "<unknown>")
    if not callable(value):
        raise RuntimeError(f"module {module_name!r} must define {name}(...)")
    return value


def expect_mapping(value: object, label: str) -> Mapping[str, object]:
    """Validate a bootstrap mapping with string keys."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    raw_mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in raw_mapping.keys()):
        raise TypeError(f"{label} keys must be strings")
    return cast(Mapping[str, object], raw_mapping)


def _optional_mapping(value: object, label: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    return expect_mapping(value, label)


def _expect_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def _optional_str(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _expect_str(value, label)


def _expect_num_envs(value: object, label: str) -> int:
    if value is None:
        return 1
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    if value < 1:
        raise ValueError(f"{label} must be at least 1")
    return value


def _expect_vectorization_mode(value: object, label: str) -> str:
    if value is None:
        return "sync"
    value = _expect_str(value, label)
    if value not in {"sync", "async"}:
        raise ValueError(f"{label} must be 'sync' or 'async'")
    return value


def _expect_str_list(value: object, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list[str]")
    items = cast(list[object], value)
    if not all(isinstance(item, str) for item in items):
        raise TypeError(f"{label} must be a list[str]")
    return cast(list[str], items)


def _mapping_to_kwargs(value: object, label: str) -> dict[str, object]:
    mapping = _optional_mapping(value, label)
    if mapping is None:
        return {}
    return dict(mapping)


__all__ = [
    "expect_mapping",
    "import_gym_modules",
    "import_packages",
    "is_env_lookup_error",
    "load_env_entrypoint",
    "load_env_from_spec",
    "load_environment",
    "load_gym_env",
    "load_hf_env",
    "load_module_from_path",
    "looks_like_env",
    "make_gym_environment",
    "normalize_hf_env",
    "select_mapping_item",
]
