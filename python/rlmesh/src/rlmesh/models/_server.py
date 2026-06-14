"""``ModelServer`` -- serve a policy, or drive it against an env.

The model-side mirror of ``EnvServer``, with one asymmetry: a model consumes the
env contract rather than publishing its own. ``run(env)`` dials an env, pulls its
contract, resolves the adapter from the env's tags and this model's spec, and runs
a per-episode eval loop that returns a typed :class:`RunResult`. ``serve()`` hosts
a model endpoint for the runtime to dial.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..recipes._authoring_model import (
    DELEGATED,
    ModelRecipe,
    construct_authored_model,
    is_model_recipe,
)
from ..recipes._schema import ArtifactInput, Recipe

__all__ = ["EpisodeResult", "ModelServer", "RunResult"]

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
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    """The result of a ``ModelServer.run`` eval."""

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


class ModelServer:
    """Serve a policy, or drive it against an env."""

    def __init__(
        self,
        source: Callable[..., Any] | type[ModelRecipe] | ModelRecipe | Recipe | str,
        address: str | None = None,
        *,
        host: str | None = None,
        port: int | None = None,
        path: str | None = None,
        transport: object | None = None,
        options: object | None = None,
        spec: object | None = None,
        artifacts: Sequence[ArtifactInput] = (),
        load_kwargs: Mapping[str, object] | None = None,
        trust_entrypoints: bool = False,
    ) -> None:
        self._address = address
        self._host = host
        self._port = port
        self._path = path
        self._transport = transport
        self._options = options
        self._trust_entrypoints = trust_entrypoints
        (
            self._predict,
            self._spec,
            self._on_reset,
            self._on_close,
            self._policy,
        ) = _coerce_model(
            source,
            spec=spec,
            artifacts=tuple(artifacts),
            load_kwargs=dict(load_kwargs) if load_kwargs else None,
        )

    @property
    def spec(self) -> object | None:
        """The model's content: a ``ModelSpec``, ``DELEGATED``, or ``None``."""
        return self._spec

    def run(
        self,
        env_or_address: object,
        *,
        seeds: Sequence[int] | None = None,
        max_episodes: int | None = None,
        instruction: str | None = None,
        close_env: bool = False,
        token: str = "",
    ) -> RunResult:
        """Drive this policy against an env and return a :class:`RunResult`.

        Resolves the adapter from the env's tags and this model's spec, then runs a
        per-episode loop: reset env, adapter, and policy; step until the episode
        ends; collect the result. ``seeds`` gives a per-episode seed, and its length
        sets the episode count unless ``max_episodes`` is given. ``instruction`` is
        written into the model's text inputs each episode.
        """
        client, contract, owns_client = _connect(env_or_address, token)
        adapter = self._resolve_adapter(contract)
        text_keys = _text_input_keys(self._spec)
        n_episodes = _episode_count(seeds, max_episodes)
        episodes: list[EpisodeResult] = []
        try:
            for i in range(n_episodes):
                seed = seeds[i] if seeds is not None and i < len(seeds) else None
                episodes.append(
                    self._run_episode(client, adapter, i, seed, instruction, text_keys)
                )
        finally:
            if self._on_close is not None:
                self._on_close()
            # Stop the remote env before closing the local connection. When we own
            # the client, shutdown rides it (a bare address string has no shutdown).
            if close_env:
                _shutdown(client if owns_client else env_or_address)
            if owns_client:
                _close(client)
        return RunResult(episodes=tuple(episodes))

    def _run_episode(
        self,
        client: Any,
        adapter: Any,
        index: int,
        seed: int | None,
        instruction: str | None,
        text_keys: tuple[str, ...],
    ) -> EpisodeResult:
        obs, _info = _reset(client, seed)
        if adapter is not None:
            adapter.reset()
        if self._on_reset is not None:
            self._on_reset()
        total = 0.0
        steps = 0
        terminated = truncated = False
        info: Mapping[str, Any] = {}
        while not (terminated or truncated) and steps < _MAX_STEPS_PER_EPISODE:
            payload = adapter.transform_obs(obs) if adapter is not None else obs
            if instruction is not None and isinstance(payload, dict):
                for key in text_keys:
                    payload[key] = instruction
            action = self._predict(payload)
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
            info=dict(info) if isinstance(info, Mapping) else {},
        )

    def _resolve_adapter(self, contract: object) -> Any:
        from ..adapters import AdapterResolutionError, EnvTags, resolve_from_contract

        spec = self._spec
        if spec is DELEGATED:
            return None
        metadata = getattr(contract, "metadata", None) or {}
        tagged = EnvTags.from_metadata(metadata) is not None
        if spec is None:
            if tagged:
                raise AdapterResolutionError(
                    "the env publishes adapter tags but this model has spec=None; "
                    "pass spec=<ModelSpec> to adapt, or spec=DELEGATED if the model "
                    "adapts its own observations"
                )
            return None
        adapter = resolve_from_contract(
            contract, spec, trust_entrypoints=self._trust_entrypoints
        )
        num_envs = int(getattr(contract, "num_envs", 1) or 1)
        if num_envs > 1 and adapter.is_stateful:
            raise AdapterResolutionError(
                "a stateful adapter cannot run on a vector env yet "
                f"(num_envs={num_envs}); per-lane affinity is not implemented. "
                "Use num_envs=1 or a stateless adapter."
            )
        return adapter

    def serve(self, address: str | None = None, *, token: str = "") -> None:
        """Host this policy as a model endpoint (blocking).

        Binds at ``address``, else the constructor's ``host``/``port``/``path``,
        and forwards ``options``. A spec-less model (``spec=None`` or ``DELEGATED``)
        serves directly. A model carrying a ``ModelSpec`` must be driven with
        ``run(env)`` instead: its adapter resolves from an env contract, which a
        passive endpoint only sees when a client dials in, and serve-side
        resolution is not implemented. Serving the raw predict would silently skip
        the spec's observation/action transforms, so it is refused rather than run
        wrong.
        """
        from ..adapters import ModelSpec
        from ..numpy import Model

        if isinstance(self._spec, ModelSpec):
            raise NotImplementedError(
                "serve() cannot host a spec'd model yet: its adapter resolves from "
                "an env contract, which a served endpoint only sees on dial-in "
                "(not implemented). Drive it with ModelServer(model).run(env), or "
                "use spec=DELEGATED if the model adapts its own observations."
            )
        worker = Model(self._predict, on_reset=self._on_reset, on_close=self._on_close)
        worker.serve(self._bind_address(address), token=token, options=self._options)

    def _bind_address(self, override: str | None) -> str:
        if override is not None:
            return override
        if self._address is not None:
            return self._address
        if self._path is not None:
            return f"unix://{self._path}"
        if self._host is not None or self._port is not None:
            return f"{self._host or '127.0.0.1'}:{self._port or 0}"
        return "127.0.0.1:0"


def _coerce_model(
    source: Any,
    *,
    spec: object | None,
    artifacts: tuple[ArtifactInput, ...],
    load_kwargs: dict[str, object] | None,
) -> tuple[Callable[[Any], Any], object | None, Callable[[], None] | None, Callable[[], None] | None, Any]:
    """Resolve a source into (predict_fn, spec, on_reset, on_close, policy)."""
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
                f"ModelServer source {recipe.name!r} is a kind={recipe.kind!r} recipe, "
                "not a model recipe"
            )
        policy = _construct_from_recipe(recipe, load_kwargs=load_kwargs, artifacts=artifacts)
        return _bind_policy(policy, spec)
    if callable(source):
        return source, spec, None, None, None
    raise TypeError(
        "ModelServer source must be a predict callable, a ModelRecipe (class or "
        f"instance), a kind='model' Recipe, or a registered name; got {type(source).__name__}"
    )


def _bind_policy(
    policy: ModelRecipe, spec_override: object | None
) -> tuple[Callable[[Any], Any], object | None, Callable[[], None], Callable[[], None], ModelRecipe]:
    spec = spec_override if spec_override is not None else type(policy).spec
    return policy.predict, spec, policy.reset, policy.close, policy


def _construct_from_recipe(
    recipe: Recipe, *, load_kwargs: dict[str, object] | None, artifacts: tuple[ArtifactInput, ...]
) -> ModelRecipe:
    """Construct a policy from an inert model recipe via its ``module:Class`` entrypoint.

    Per-run ``artifacts`` and ``load_kwargs`` apply because construction runs the
    in-process path, with the class's own ``inputs`` providing the declared mounts.
    """
    from .._bootstrap.entrypoint import resolve_entrypoint
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


def _connect(target: object, token: str) -> tuple[Any, Any, bool]:
    """Return (client, contract, owns_client) for an EnvServer, address, or env-like."""
    if isinstance(target, str):
        client = _remote_env(target)
        return client, client.env_contract, True
    if hasattr(target, "reset") and hasattr(target, "step"):
        return target, getattr(target, "env_contract", None), False
    address = getattr(target, "address", None)
    if isinstance(address, str):
        client = _remote_env(address)
        return client, client.env_contract, True
    raise TypeError(
        "ModelServer.run() expects an EnvServer, a remote-env object, or an "
        f"address string; got {type(target).__name__}"
    )


def _remote_env(address: str) -> Any:
    from ..numpy import RemoteEnv

    return RemoteEnv(address)


def _reset(client: Any, seed: int | None) -> tuple[Any, Mapping[str, Any]]:
    result = client.reset(seed=seed) if seed is not None else client.reset()
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1]
    return result, {}


def _step(client: Any, action: Any) -> tuple[Any, float, bool, bool, Mapping[str, Any]]:
    out = client.step(action)
    obs, reward, terminated, truncated, info = out
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
