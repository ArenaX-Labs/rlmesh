"""The model-side eval loop and source coercion, shared by every framework ``Model``.

A model consumes the env contract rather than publishing its own: ``evaluate``
dials an env, pulls its contract, resolves the adapter from the env's tags and the
model's spec, and runs a per-episode loop that returns a typed :class:`RunResult`.
``coerce_model`` turns any model source -- a predict callable, a ``ModelRecipe``
(class or instance), a ``kind='model'`` Recipe, or a registered name -- into the
``(predict, spec, on_reset, on_close, policy)`` the loop needs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from ..recipes._schema import ArtifactInput, Recipe
from ..recipes.authoring.model import (
    DELEGATED,
    ModelRecipe,
    construct_authored_model,
    is_model_recipe,
)

if TYPE_CHECKING:
    from ..adapters import Adapter
    from ..specs import EnvContract

__all__ = ["EpisodeResult", "RunResult", "coerce_model", "evaluate"]

# Bound the loop so a non-terminating env cannot hang it forever.
_MAX_STEPS_PER_EPISODE = 100_000


@dataclass(frozen=True)
class EpisodeResult:
    """The outcome of one evaluation episode."""

    index: int
    seed: int | None
    steps: int
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    """The result of a :meth:`Model.run` eval."""

    episodes: tuple[EpisodeResult, ...] = ()

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def total_steps(self) -> int:
        return sum(e.steps for e in self.episodes)

    @property
    def mean_reward(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.reward for e in self.episodes) / len(self.episodes)

    @property
    def success_rate(self) -> float:
        """Fraction of episodes that terminated rather than truncated."""
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.terminated) / len(self.episodes)

    def __repr__(self) -> str:
        return (
            f"RunResult(episodes={self.num_episodes}, mean_reward={self.mean_reward:.3f}, "
            f"total_steps={self.total_steps})"
        )


def evaluate(
    predict: Callable[[Any], Any],
    spec: object | None,
    env_or_address: object,
    *,
    seeds: Sequence[int] | None = None,
    max_episodes: int | None = None,
    instruction: str | None = None,
    close_env: bool = False,
    token: str = "",
    on_reset: Callable[[], None] | None = None,
    on_episode_end: Callable[[], None] | None = None,
    on_close: Callable[[], None] | None = None,
    trust_entrypoints: bool = False,
    remote_env_cls: type | None = None,
) -> RunResult:
    """Drive ``predict`` against an env and return a :class:`RunResult`.

    Resolves the adapter from the env's tags and ``spec``, then runs a per-episode
    loop: reset env, adapter, and policy; step until the episode ends; collect the
    result. ``seeds`` gives a per-episode seed, and its length sets the episode
    count unless ``max_episodes`` is given. ``instruction`` is written into the
    model's text inputs each episode.
    """
    client, contract, owns_client = _connect(env_or_address, token, remote_env_cls)
    adapter = _resolve_adapter(spec, contract, trust_entrypoints)
    text_keys = _text_input_keys(spec)
    n_episodes = _episode_count(seeds, max_episodes)
    episodes: list[EpisodeResult] = []
    try:
        for i in range(n_episodes):
            seed = seeds[i] if seeds is not None and i < len(seeds) else None
            episodes.append(
                _run_episode(
                    client, predict, adapter, on_reset, i, seed, instruction, text_keys
                )
            )
            if on_episode_end is not None:
                on_episode_end()
    finally:
        if on_close is not None:
            on_close()
        # Stop the remote env before closing the local connection. When we own
        # the client, shutdown rides it (a bare address string has no shutdown).
        if close_env:
            _shutdown(client if owns_client else env_or_address)
        if owns_client:
            _close(client)
    return RunResult(episodes=tuple(episodes))


def _run_episode(
    client: Any,
    predict: Callable[[Any], Any],
    adapter: Any,
    on_reset: Callable[[], None] | None,
    index: int,
    seed: int | None,
    instruction: str | None,
    text_keys: tuple[str, ...],
) -> EpisodeResult:
    obs, _info = _reset(client, seed)
    if adapter is not None:
        adapter.reset()
    if on_reset is not None:
        on_reset()
    total = 0.0
    steps = 0
    terminated = truncated = False
    info: Mapping[str, Any] = {}
    while not (terminated or truncated) and steps < _MAX_STEPS_PER_EPISODE:
        payload = adapter.transform_obs(obs) if adapter is not None else obs
        if instruction is not None and isinstance(payload, dict):
            for key in text_keys:
                payload[key] = instruction
        action = predict(payload)
        if adapter is not None:
            action = adapter.transform_action(action)
        obs, reward, terminated, truncated, info = _step(client, action)
        total += float(reward)
        steps += 1
    return EpisodeResult(
        index=index,
        seed=seed,
        steps=steps,
        reward=total,
        terminated=bool(terminated),
        truncated=bool(truncated),
        info=dict(info),
    )


def _resolve_adapter(
    spec: object | None, contract: EnvContract | None, trust_entrypoints: bool
) -> Adapter | None:
    from ..adapters import (
        AdapterResolutionError,
        EnvTags,
        ModelSpec,
        resolve_from_contract,
    )

    if spec is DELEGATED:
        return None
    metadata = contract.metadata if contract is not None else None
    tagged = EnvTags.from_metadata(metadata or {}) is not None
    if spec is None:
        if tagged:
            raise AdapterResolutionError(
                "the env publishes adapter tags but this model has spec=None; "
                "pass spec=<ModelSpec> to adapt, or spec=DELEGATED if the model "
                "adapts its own observations"
            )
        return None
    if not isinstance(spec, ModelSpec):
        raise AdapterResolutionError(
            f"a model spec must be a ModelSpec or DELEGATED; got {type(spec).__name__}"
        )
    if contract is None:
        raise AdapterResolutionError(
            "resolving a spec'd adapter requires an env contract, but the env exposes none"
        )
    adapter = resolve_from_contract(contract, spec, trust_entrypoints=trust_entrypoints)
    if contract.num_envs > 1 and adapter.is_stateful:
        raise AdapterResolutionError(
            "a stateful adapter cannot run on a vector env yet "
            f"(num_envs={contract.num_envs}); per-lane affinity is not implemented. "
            "Use num_envs=1 or a stateless adapter."
        )
    return adapter


def coerce_model(
    source: Any,
    *,
    spec: object | None,
    artifacts: tuple[ArtifactInput, ...],
    load_kwargs: dict[str, object] | None,
) -> tuple[
    Callable[[Any], Any],
    object | None,
    Callable[[], None] | None,
    Callable[[], None] | None,
    Any,
]:
    """Resolve a model source into ``(predict, spec, on_reset, on_close, policy)``."""
    if isinstance(source, ModelRecipe):
        return _bind_policy(source, spec)
    if is_model_recipe(source):
        policy = construct_authored_model(
            source, in_container=False, load_kwargs=load_kwargs, artifacts=artifacts
        )
        return _bind_policy(policy, spec)
    recipe: Recipe | None = None
    if isinstance(source, str):
        from ._registry import lookup_model_class

        cls = lookup_model_class(source)
        if cls is not None:
            policy = construct_authored_model(
                cls, in_container=False, load_kwargs=load_kwargs, artifacts=artifacts
            )
            return _bind_policy(policy, spec)
        from ..recipes import resolve as resolve_recipe

        recipe = resolve_recipe(source)
    elif isinstance(source, Recipe):
        recipe = source
    if recipe is not None:
        if recipe.kind != "model":
            raise TypeError(
                f"Model source {recipe.name!r} is a kind={recipe.kind!r} recipe, "
                "not a model recipe"
            )
        policy = _construct_from_recipe(
            recipe, load_kwargs=load_kwargs, artifacts=artifacts
        )
        return _bind_policy(policy, spec)
    if callable(source):
        return source, spec, None, None, None
    raise TypeError(
        "Model source must be a predict callable, a ModelRecipe (class or instance), "
        f"a kind='model' Recipe, or a registered name; got {type(source).__name__}"
    )


def _bind_policy(
    policy: ModelRecipe, spec_override: object | None
) -> tuple[
    Callable[[Any], Any],
    object | None,
    Callable[[], None],
    Callable[[], None],
    ModelRecipe,
]:
    spec = spec_override if spec_override is not None else type(policy).spec
    return policy.predict, spec, policy.reset, policy.close, policy


def _construct_from_recipe(
    recipe: Recipe,
    *,
    load_kwargs: dict[str, object] | None,
    artifacts: tuple[ArtifactInput, ...],
) -> ModelRecipe:
    """Construct a policy from an inert model recipe via its ``module:Class`` entrypoint."""
    from .._entrypoint import resolve_entrypoint
    from ..recipes._schema import PyMake

    make = recipe.make
    if not isinstance(make, PyMake):
        raise TypeError(
            f"model recipe {recipe.name!r} has no PyMake entrypoint to construct from"
        )
    cls_path = make.entrypoint.rsplit(".", 1)[0]
    cls = resolve_entrypoint(cls_path, label="model recipe class")
    if not is_model_recipe(cls):
        raise TypeError(f"{cls_path} is not a ModelRecipe subclass")
    return construct_authored_model(
        cls, in_container=False, load_kwargs=load_kwargs, artifacts=artifacts
    )


def _text_input_keys(spec: object | None) -> tuple[str, ...]:
    if spec is None or spec is DELEGATED:
        return ()
    from ..adapters import TextInput

    return tuple(
        inp.key for inp in getattr(spec, "inputs", ()) if isinstance(inp, TextInput)
    )


def _episode_count(seeds: Sequence[int] | None, max_episodes: int | None) -> int:
    if max_episodes is not None:
        return max_episodes
    if seeds is not None:
        return len(seeds)
    return 1


def _connect(
    target: object, token: str, remote_env_cls: type | None
) -> tuple[Any, Any, bool]:
    """Return ``(client, contract, owns_client)`` for an env object, address, or EnvServer."""
    if isinstance(target, str):
        client = _remote_env(target, remote_env_cls)
        return client, client.env_contract, True
    if hasattr(target, "reset") and hasattr(target, "step"):
        return target, getattr(target, "env_contract", None), False
    address = getattr(target, "address", None)
    if isinstance(address, str):
        client = _remote_env(address, remote_env_cls)
        return client, client.env_contract, True
    raise TypeError(
        "Model.run() expects an env object, a remote-env object, or an address "
        f"string; got {type(target).__name__}"
    )


def _remote_env(address: str, remote_env_cls: type | None) -> Any:
    if remote_env_cls is None:
        from ..numpy import RemoteEnv

        remote_env_cls = RemoteEnv
    return remote_env_cls(address)


def _reset(client: Any, seed: int | None) -> tuple[Any, Mapping[str, Any]]:
    result: Any = client.reset(seed=seed) if seed is not None else client.reset()
    if isinstance(result, tuple):
        pair = cast("tuple[Any, ...]", result)
        if len(pair) == 2:
            return pair[0], pair[1]
        return pair, {}
    return result, {}


def _step(client: Any, action: Any) -> tuple[Any, float, bool, bool, Mapping[str, Any]]:
    obs, reward, terminated, truncated, info = client.step(action)
    return obs, reward, terminated, truncated, info


def _close(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _shutdown(target: object) -> None:
    """Stop the driven env via ``shutdown()`` or ``close()``.

    Whether a reason argument is passed is decided by binding the signature, so an
    unrelated ``TypeError`` raised inside the callable is never swallowed.
    """
    import inspect

    for name in ("shutdown", "close"):
        fn = getattr(target, name, None)
        if not callable(fn):
            continue
        try:
            inspect.signature(fn).bind("model run complete")
            accepts_reason = True
        except TypeError:
            accepts_reason = False
        except (ValueError, KeyError):  # un-introspectable builtin
            accepts_reason = False
        fn("model run complete") if accepts_reason else fn()
        return
