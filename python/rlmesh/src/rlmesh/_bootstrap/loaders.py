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
    call_hf_make_env,
    import_gym_modules,
    load_callable,
    make_gym_environment,
)
from .spec_resolution import (
    expect_num_envs,
    expect_str,
    expect_str_list,
    expect_vectorization_mode,
    mapping_to_kwargs,
    optional_any_mapping,
    optional_mapping,
    optional_str,
    select_mapping_item,
    select_task_item,
)

if TYPE_CHECKING:
    from rlmesh._server import EnvLike, VectorServerEnvLike
    from rlmesh.numpy import NumpyValue
    from rlmesh.params import ParamSpec

    ServedEnv = EnvLike[Any, Any] | VectorServerEnvLike


def load_environment(
    env_id: str,
    package_names: Sequence[str],
    num_envs: int,
    vectorization_mode: str | None = None,
    kwargs: Mapping[str, object] | None = None,
) -> ServedEnv:
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
                "ServedEnv",
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
    """Load an environment from a sandbox bootstrap spec (gym, hf, or factory)."""
    kind = expect_str(spec.get("kind"), "bootstrap spec.kind")
    if kind == "gym":
        return load_gym_env(spec)
    if kind == "hf":
        return load_hf_env(spec)
    if kind == "factory":
        return load_factory_env(spec)
    raise ValueError(f"unsupported bootstrap kind: {kind}")


def load_factory_env(spec: Mapping[str, object]) -> object:
    """Construct an :class:`~rlmesh.EnvFactory` env from a bootstrap spec.

    Lets a prebuilt EnvFactory image boot any binding: the ``entrypoint``
    (``module:Class``) is resolved and constructed with the spec's ``kwargs`` as
    the ``make`` binding. The binding is validated against the factory's declared
    ``params`` (if any) inside :func:`construct_authored_env`, before construction.
    ``num_envs``/``vectorization_mode`` fan the factory out into a vector env, the
    same as the gym/hf paths.
    """
    import_packages(expect_str_list(spec.get("imports"), "bootstrap imports"))
    entrypoint = expect_str(spec.get("entrypoint"), "bootstrap spec.entrypoint")
    kwargs = mapping_to_kwargs(spec.get("kwargs"), "bootstrap spec.kwargs")
    num_envs = expect_num_envs(spec.get("num_envs"), "bootstrap spec.num_envs")
    vectorization_mode = expect_vectorization_mode(
        spec.get("vectorization_mode"), "bootstrap spec.vectorization_mode"
    )
    factory = resolve_entrypoint(entrypoint, label="env entrypoint")
    return construct_authored_env(
        factory,
        num_envs=num_envs,
        vectorization_mode=vectorization_mode,
        **kwargs,
    )


class EntrypointConstructionError(RuntimeError, ImportError):
    """Raised when a ``module:callable`` entrypoint factory cannot be loaded.

    Wraps the import/attribute/not-callable boundary of resolving a
    ``module:callable`` factory; errors raised *inside* a loaded factory are not
    wrapped. Subclasses ``ImportError`` so existing ``except ImportError`` callers of
    ``rlmesh._serving.load_env_entrypoint`` still catch the bad-entrypoint case.
    """


def load_env_entrypoint(
    entrypoint: str,
    package_names: Sequence[str] = (),
    kwargs: Mapping[str, object] | None = None,
) -> ServedEnv:
    """Load an environment from a ``module:callable`` factory entrypoint."""
    try:
        import_packages(package_names)
        factory = resolve_entrypoint(entrypoint, label="env entrypoint")
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise EntrypointConstructionError(
            f"could not load env entrypoint {entrypoint!r}: {exc}."
        ) from exc
    env = factory(**dict(kwargs or {}))
    if not looks_like_env(env):
        raise TypeError(
            f"env entrypoint {entrypoint!r} did not return an environment "
            "with reset(...) and step(...)"
        )
    return cast("ServedEnv", env)


def load_gym_env(spec: Mapping[str, object]) -> object:
    """Load a Gymnasium/Gym environment from a sandbox bootstrap spec."""
    import_packages(expect_str_list(spec.get("imports"), "bootstrap imports"))

    env_id = expect_str(spec.get("env_id"), "bootstrap spec.env_id")
    kwargs = mapping_to_kwargs(spec.get("kwargs"), "bootstrap spec.kwargs")
    num_envs = expect_num_envs(spec.get("num_envs"), "bootstrap spec.num_envs")
    vectorization_mode = expect_vectorization_mode(
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
    import_packages(expect_str_list(spec.get("imports"), "bootstrap imports"))

    source_subdir = expect_str(
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
    make_env = load_callable(module, "make_env")

    kwargs = mapping_to_kwargs(spec.get("kwargs"), "bootstrap spec.kwargs")
    num_envs = expect_num_envs(spec.get("num_envs"), "bootstrap spec.num_envs")
    vectorization_mode = expect_vectorization_mode(
        spec.get("vectorization_mode"), "bootstrap spec.vectorization_mode"
    )

    envs = call_hf_make_env(
        make_env,
        kwargs,
        num_envs=num_envs,
        vectorization_mode=vectorization_mode,
    )
    suite = optional_str(spec.get("suite"), "bootstrap spec.suite")
    task = optional_str(spec.get("task"), "bootstrap spec.task")
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

    env_mapping = optional_mapping(envs, "hf env mapping")
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

    task_mapping = optional_any_mapping(
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


def looks_like_policy(value: object) -> bool:
    """Return whether a value exposes a predict callable (a duck-typed policy object).

    Matches both a policy *class* (its ``predict`` is an unbound function) and an
    instance (a bound method); a bare predict callable has no ``predict`` attribute.
    """
    return callable(getattr(value, "predict", None))


def construct_authored_model(source: Any, /, **kwargs: object) -> Any:
    """Instantiate a model/policy (class or instance), ``load(**binding)``, return it.

    The bootstrap-authoritative model path: a class source is instantiated with
    the eager ``__init__`` auto-load suppressed (a ``Model`` subclass builds its
    worker but does not load weights), then the binding is validated against the
    declared ``params`` and ``load(**resolved)`` runs exactly once. With no
    ``kwargs`` and no ``params`` this is the prior behavior -- a single ``load()``.
    """
    from rlmesh._models.base import ModelBase, suppress_autoload
    from rlmesh.params import resolve

    if isinstance(source, type):
        with suppress_autoload():
            inst = source()
    else:
        inst = source

    spec = cast("ParamSpec | None", getattr(inst, "params", None))
    load = getattr(inst, "load", None)
    if not callable(load):
        if kwargs:
            raise TypeError(
                f"{type(inst).__name__} has no load(...) to receive construction "
                f"params: {', '.join(sorted(kwargs))}"
            )
        return inst
    resolved = resolve(spec, load, kwargs)
    # The default ModelBase.load is a no-op: a non-empty binding would be silently
    # swallowed (and the eager auto-load was already suppressed). Fail loud so the
    # operator's requested variation is never silently dropped. (getattr identity,
    # mirroring base.py's overridden-method probe.)
    inst_type = cast("type[object]", type(inst))
    if resolved and getattr(inst_type, "load", None) is getattr(
        ModelBase, "load", None
    ):
        raise TypeError(
            f"{type(inst).__name__} received construction params "
            f"({', '.join(sorted(resolved))}) but does not override load(...) to "
            "apply them; implement load(**kwargs) to consume the binding"
        )
    load(**resolved)
    return inst


def construct_authored_env(
    source: Any,
    /,
    *,
    num_envs: int = 1,
    vectorization_mode: str | None = None,
    **kwargs: object,
) -> Any:
    """Instantiate an EnvFactory class (or accept an instance), ``prepare()``, ``make()``.

    The binding is validated against the factory's declared ``params`` (a
    :class:`~rlmesh.ParamSpec`, or ``None`` for blind passthrough) *before*
    ``make`` runs, so a typo'd or out-of-range key fails pre-construction. The
    resolved binding is then published into the env's ``metadata`` (under
    :data:`rlmesh.params.PARAM_METADATA_KEY`) via the same merge rail as tags, so
    the operator can read back exactly what was sent.

    ``num_envs > 1`` builds a self-describing vector env: the binding is validated
    once, then ``make`` is fanned out into a gym Sync/Async vector wrapper (see
    :func:`rlmesh._bootstrap.gym_support.vectorize`) so a prebuilt EnvFactory image
    honors a ``SandboxVectorEnv`` request instead of serving a lone env.
    """
    from rlmesh.params import resolve, to_metadata

    inst = source() if isinstance(source, type) else source
    prepare = getattr(inst, "prepare", None)
    if callable(prepare):
        prepare()

    spec = cast("ParamSpec | None", getattr(inst, "params", None))
    resolved = resolve(spec, inst.make, kwargs)
    if num_envs > 1:
        from .gym_support import vectorize

        make = inst.make
        env = vectorize(lambda: make(**resolved), num_envs, vectorization_mode)
    else:
        env = inst.make(**resolved)
    if spec is not None:
        _merge_metadata(env, to_metadata(spec, inst.make, resolved))
    return env


def _merge_metadata(env: object, fragment: Mapping[str, object]) -> None:
    """Merge a metadata fragment into ``env.metadata`` (mirrors ``adapters.tag``)."""
    existing = getattr(env, "metadata", None)
    merged: dict[str, object] = (
        dict(cast("Mapping[str, object]", existing))
        if isinstance(existing, Mapping)
        else {}
    )
    merged.update(fragment)
    env.metadata = merged  # type: ignore[attr-defined]
