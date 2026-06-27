"""Gymnasium/Gym factory construction and HF make_env adaptation helpers."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import cast


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
    make = load_callable(gym_module, "make")
    if num_envs <= 1:
        return make(env_id, **env_kwargs)

    make_vec = getattr(gym_module, "make_vec", None)
    if callable(make_vec):
        make_vec_kwargs: dict[str, object] = {"num_envs": num_envs, **env_kwargs}
        if vectorization_mode is not None:
            make_vec_kwargs["vectorization_mode"] = vectorization_mode
        return make_vec(env_id, **make_vec_kwargs)

    return vectorize(
        lambda: make(env_id, **env_kwargs),
        num_envs,
        vectorization_mode,
        gym_module=gym_module,
    )


def vectorize(
    make_one: Callable[[], object],
    num_envs: int,
    vectorization_mode: str | None,
    *,
    gym_module: object | None = None,
) -> object:
    """Wrap ``num_envs`` copies of ``make_one()`` in a gym Sync/Async vector env.

    The one fan-out used to vectorize *any* env factory -- a gym ``make`` thunk or
    an :class:`~rlmesh.EnvFactory`'s ``make`` -- into a self-describing vector env
    (``num_envs`` + ``single_*`` spaces) the native vector server serves. The
    sub-envs must be gym-compatible (the gym vector wrappers build the batched
    space from each sub-env's spaces).
    """
    modules = [gym_module] if gym_module is not None else import_gym_modules()
    cls_name = "AsyncVectorEnv" if vectorization_mode == "async" else "SyncVectorEnv"
    for module in modules:
        vector_module = getattr(module, "vector", None)
        vector_cls = getattr(vector_module, cls_name, None) if vector_module else None
        if callable(vector_cls):
            factory = cast("Callable[[list[Callable[[], object]]], object]", vector_cls)
            return factory([make_one for _ in range(num_envs)])
    raise ValueError(
        f"no gym vector env support available for {cls_name}; install gymnasium/gym"
    )


def import_gym_modules() -> list[ModuleType]:
    """Import supported Gym modules in preference order."""
    modules: list[ModuleType] = []
    for module_name in ("gymnasium", "gym"):
        try:
            modules.append(importlib.import_module(module_name))
        except ImportError:
            continue
    return modules


def call_hf_make_env(
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


def load_callable(module: object, name: str) -> Callable[..., object]:
    value = getattr(module, name, None)
    module_name = getattr(module, "__name__", "<unknown>")
    if not callable(value):
        raise RuntimeError(f"module {module_name!r} must define {name}(...)")
    return value
