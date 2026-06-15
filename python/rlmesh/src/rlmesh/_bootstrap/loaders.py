"""Environment loaders shared by CLI and sandbox entrypoints."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

from rlmesh._entrypoint import resolve_entrypoint

from .gym_support import (
    _call_hf_make_env,
    _load_callable,
    import_gym_modules,
    make_gym_environment,
)
from .spec_resolution import (
    _expect_num_envs,
    _expect_str,
    _expect_str_list,
    _expect_vectorization_mode,
    _mapping_to_kwargs,
    _optional_any_mapping,
    _optional_mapping,
    _optional_str,
    apply_member_params,
    expect_mapping,
    select_mapping_item,
    select_task_item,
)

if TYPE_CHECKING:
    from rlmesh.numpy import NumpyValue
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


def load_env_from_spec(
    spec: Mapping[str, object],
    *,
    setup_env: Mapping[str, object] | None = None,
    kwargs: Mapping[str, object] | None = None,
) -> object:
    """Load an environment from a sandbox bootstrap spec.

    ``setup_env``/``kwargs`` are the parsed ``RLMESH_PARAMS_JSON`` member selector;
    they apply to recipe sources only (gym/hf sources predate it and ignore them).
    """
    kind = _expect_str(spec.get("kind"), "bootstrap spec.kind")
    if kind == "gym":
        return load_gym_env(spec)
    if kind == "hf":
        return load_hf_env(spec)
    if kind == "recipe":
        return load_recipe_env(spec, setup_env=setup_env, kwargs=kwargs)
    raise ValueError(f"unsupported bootstrap kind: {kind}")


def load_recipe_env(
    spec: Mapping[str, object],
    *,
    setup_env: Mapping[str, object] | None = None,
    kwargs: Mapping[str, object] | None = None,
) -> object:
    """Construct an environment from a recipe bootstrap spec.

    The build phase already shaped the image; this runs the recipe's runtime half
    (setup + make) in-container via ``rlmesh.recipes.build``. Imported lazily to
    avoid a recipes <-> bootstrap import cycle.
    """
    from rlmesh.recipes import Recipe, build

    document = expect_mapping(spec.get("document"), "bootstrap spec.document")
    num_envs = _expect_num_envs(spec.get("num_envs"), "bootstrap spec.num_envs")
    vectorization_mode = _expect_vectorization_mode(
        spec.get("vectorization_mode"), "bootstrap spec.vectorization_mode"
    )
    recipe = apply_member_params(
        Recipe.from_dict(document), setup_env=setup_env, kwargs=kwargs
    )
    return build(recipe, num_envs=num_envs, vectorization_mode=vectorization_mode)


class RecipeConstructionError(RuntimeError, ImportError):
    """Raised when a recipe's factory entrypoint cannot be loaded.

    Wraps the import/attribute/not-callable boundary of resolving a
    ``module:callable`` factory, naming the entrypoint and pointing at
    ``rlmesh.recipes.check``. Errors raised *inside* a successfully-loaded factory
    are not wrapped.

    It subclasses both ``RuntimeError`` and ``ImportError`` so the nicer message is
    raised while existing ``except ImportError`` callers of the public
    ``rlmesh.serving.load_env_entrypoint`` still catch the common bad-entrypoint
    case (which previously raised a raw ``ImportError``/``AttributeError``/
    ``TypeError``/``ValueError``). The MRO is well-defined -- both bases derive from
    ``Exception`` -- so ``raise``/``isinstance`` behave normally.
    """


def load_env_entrypoint(
    entrypoint: str,
    package_names: Sequence[str] = (),
    kwargs: Mapping[str, object] | None = None,
) -> EnvLike:
    """Load an environment from a ``module:callable`` factory entrypoint."""
    try:
        import_packages(package_names)
        factory = resolve_entrypoint(entrypoint, label="env entrypoint")
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise RecipeConstructionError(
            f"could not load env entrypoint {entrypoint!r}: {exc}. Run "
            "rlmesh.recipes.check(recipe) to validate the entrypoint shape without "
            "importing dependencies."
        ) from exc
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
    task = _optional_str(spec.get("task"), "bootstrap spec.task")
    return normalize_hf_env(envs, suite=suite, task=task)


def import_packages(package_names: Sequence[str]) -> None:
    """Import packages that register environments on import."""
    for module_name in package_names:
        try:
            _ = importlib.import_module(module_name)
        except ImportError:
            msg = f"Unable to import package '{module_name}'."
            raise ImportError(msg) from None


def is_env_lookup_error(exc: Exception) -> bool:
    """Return whether an exception means the env id was not registered."""
    return exc.__class__.__name__ in {
        "NameNotFound",
        "NamespaceNotFound",
        "UnregisteredEnv",
    }


def normalize_hf_env(envs: object, *, suite: str | None, task: str | None) -> object:
    """Select the served environment from an HF make_env result."""
    if looks_like_env(envs):
        return envs

    env_mapping = _optional_mapping(envs, "hf env mapping")
    if env_mapping is None:
        raise TypeError("env.py make_env(...) returned an unsupported value")

    suite_key, suite_value = select_mapping_item(env_mapping, suite, "suite")
    if looks_like_env(suite_value):
        if task is not None:
            raise ValueError(
                f"task {task!r} was specified, but suite {suite_key!r} "
                "resolved directly to an env"
            )
        return suite_value

    task_mapping = _optional_any_mapping(
        suite_value, f"task mapping for suite {suite_key!r}"
    )
    if task_mapping is not None:
        task_key, task_value = select_task_item(task_mapping, task, suite_key)
        if looks_like_env(task_value):
            return task_value
        raise TypeError(
            f"selected task {task_key!r} in suite {suite_key!r} "
            "did not resolve to an env"
        )

    raise TypeError(
        f"selected suite {suite_key!r} did not resolve to an env or task mapping"
    )


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


def load_predict(entrypoint: str) -> Callable[[NumpyValue], NumpyValue]:
    """Load a model prediction callable from ``module:callable`` syntax."""
    value = resolve_entrypoint(entrypoint, label="model entrypoint")
    return cast(Callable[[Any], Any], value)
