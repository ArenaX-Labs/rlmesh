"""Gymnasium/Gym factory construction and HF make_env adaptation helpers."""

from __future__ import annotations

import functools
import importlib
import inspect
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any, cast


class EpisodeSeedEnv:
    """Seed autoreset rolls and report every episode's seed as ``info["seed"]``.

    A gym vector env rolls a finished lane with a bare ``reset()``, so the seed of
    every episode after the first is unknowable to the caller. This wrapper (one
    per lane) derives it as ``base + ordinal`` from the lane's last explicit seed,
    keeping the result in the non-negative ``i64`` range the wire carries.

    Duck-typed proxy for any env with ``reset``; :func:`episode_seed_env` picks a
    real ``gym.Wrapper`` subclass instead when the lane is a ``gym.Env``.
    """

    def __init__(self, env: Any) -> None:
        self._env = env
        self._seeds = _EpisodeSeeds()

    def reset(self, *, seed: int | None = None, **kwargs: Any) -> object:
        return _seeded_reset(self._seeds, self._env.reset, seed, kwargs)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._env, name)


def episode_seed_env(env: Any) -> Any:
    """Wrap one lane in the seed wrapper matching its kind (see ``EpisodeSeedEnv``)."""
    for module in import_gym_modules():
        env_cls = getattr(module, "Env", None)
        wrapper_cls = getattr(module, "Wrapper", None)
        if isinstance(wrapper_cls, type) and isinstance(env_cls, type):
            if isinstance(env, env_cls):
                return _gym_episode_seed_env(wrapper_cls)(env)
    return EpisodeSeedEnv(env)


@functools.cache
def _gym_episode_seed_env(wrapper_cls: type[Any]) -> type[Any]:
    class GymEpisodeSeedEnv(wrapper_cls):
        def __init__(self, env: Any) -> None:
            super().__init__(env)  # pyright: ignore[reportUnknownMemberType]
            self._seeds = _EpisodeSeeds()

        def reset(self, *, seed: int | None = None, **kwargs: Any) -> Any:
            return _seeded_reset(self._seeds, self.env.reset, seed, kwargs)

    return GymEpisodeSeedEnv


class _EpisodeSeeds:
    def __init__(self) -> None:
        self._base: int | None = None
        self._ordinal = 0

    def resolve(self, seed: int | None) -> int | None:
        if seed is not None:
            self._base, self._ordinal = seed, 0
            return seed
        if self._base is None:
            return None
        self._ordinal += 1
        return (self._base + self._ordinal) % (1 << 63)


def _seeded_reset(
    seeds: _EpisodeSeeds,
    reset: Callable[..., object],
    seed: int | None,
    kwargs: Mapping[str, Any],
) -> object:
    seed = seeds.resolve(seed)
    result = reset(seed=seed, **kwargs)
    info = _reset_info(result)
    if seed is not None and info is not None:
        info.setdefault("seed", seed)
    return result


def _reset_info(result: object) -> dict[str, Any] | None:
    """The info dict of a gym ``(obs, info)`` reset result, else ``None``."""
    if isinstance(result, tuple):
        items = cast("tuple[Any, ...]", result)
        if len(items) == 2 and isinstance(items[1], dict):
            return cast("dict[str, Any]", items[1])
    return None


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
        # gymnasium rejects per-env wrappers when its auto mode picks a native
        # vector entry point; those envs roll lanes internally, seed unreported.
        if vectorization_mode is not None or not _has_vector_entry_point(
            gym_module, env_id
        ):
            make_vec_kwargs["wrappers"] = [episode_seed_env]
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
    space from each sub-env's spaces). ``vectorization_mode=None`` is auto and
    resolves to the sync fan-out here (there is no registry entry to consult);
    any other unknown mode is an error rather than a silent collapse to sync.
    """
    if vectorization_mode not in (None, "sync", "async"):
        raise ValueError(
            "vectorization_mode must be 'sync' or 'async' (or None for auto); "
            f"got {vectorization_mode!r}"
        )
    modules = [gym_module] if gym_module is not None else import_gym_modules()
    cls_name = "AsyncVectorEnv" if vectorization_mode == "async" else "SyncVectorEnv"
    for module in modules:
        vector_module = getattr(module, "vector", None)
        vector_cls = getattr(vector_module, cls_name, None) if vector_module else None
        if callable(vector_cls):
            factory = cast("Callable[[list[Callable[[], object]]], object]", vector_cls)
            return factory(
                [lambda: episode_seed_env(make_one()) for _ in range(num_envs)]
            )
    raise ValueError(
        f"no gym vector env support available for {cls_name}; install gymnasium/gym"
    )


def _has_vector_entry_point(gym_module: object, env_id: str) -> bool:
    spec = getattr(gym_module, "spec", None)
    if not callable(spec):
        return False
    return getattr(spec(env_id), "vector_entry_point", None) is not None


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
    vectorization_mode: str | None,
) -> object:
    """Call an HF ``make_env`` with the eval shape mapped onto its signature.

    ``n_envs``/``use_async_envs`` are injected into a signature that names them;
    a bare ``**kwargs`` signature only receives them when vectorization is
    actually requested (``num_envs != 1`` or async mode), so the default
    single-env case never crashes a natural passthrough ``make_env(**kwargs)``
    that forwards to a constructor without those names.
    """
    call_kwargs = dict(kwargs)
    accepts_kwargs, keyword_names = _callable_keyword_parameters(make_env)
    vector_requested = num_envs != 1 or vectorization_mode == "async"

    if "n_envs" not in call_kwargs:
        if "n_envs" in keyword_names or (accepts_kwargs and vector_requested):
            call_kwargs["n_envs"] = num_envs
        elif num_envs != 1:
            raise TypeError(
                "HF sandbox source requested num_envs="
                f"{num_envs}, but env.py make_env(...) does not accept n_envs"
            )

    if "use_async_envs" not in call_kwargs:
        if "use_async_envs" in keyword_names or (accepts_kwargs and vector_requested):
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
