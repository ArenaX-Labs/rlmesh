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


def import_gym_modules() -> list[ModuleType]:
    """Import supported Gym modules in preference order."""
    modules: list[ModuleType] = []
    for module_name in ("gymnasium", "gym"):
        try:
            modules.append(importlib.import_module(module_name))
        except ImportError:
            continue
    return modules


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


def _load_callable(module: object, name: str) -> Callable[..., object]:
    value = getattr(module, name, None)
    module_name = getattr(module, "__name__", "<unknown>")
    if not callable(value):
        raise RuntimeError(f"module {module_name!r} must define {name}(...)")
    return value
