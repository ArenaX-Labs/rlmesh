"""The model-side eval loop and source coercion, shared by every framework ``Model``.

A model consumes the env contract rather than publishing its own: ``evaluate``
dials an env, pulls its contract, resolves the adapter from the env's tags and the
model's spec, and runs a per-episode loop that returns a typed :class:`RunResult`.
``coerce_model`` turns any model source into a :class:`CoercedModel`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from .._spec._core import DELEGATED, ArtifactInput
from .._value_conversion import from_value

if TYPE_CHECKING:
    from .._framework_bridge import ValueBridge
    from ..adapters import Adapter
    from ..specs import EnvContract

__all__ = [
    "EpisodeResult",
    "RunResult",
    "adapted_predict",
    "coerce_model",
    "evaluate",
    "resolve_route_adapter",
]

# bound the loop so a non-terminating env cannot hang it forever.
_MAX_STEPS_PER_EPISODE = 100_000


class CoercedModel(NamedTuple):
    predict: Callable[[Any], Any]
    spec: object | None
    on_reset: Callable[[], None] | None
    on_close: Callable[[], None] | None
    policy: Any


@dataclass(frozen=True)
class EpisodeResult:
    """The outcome of one evaluation episode."""

    index: int
    seed: int | None
    steps: int
    reward: float
    terminated: bool
    truncated: bool


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
    bridge: ValueBridge | None = None,
) -> RunResult:
    """Drive ``predict`` against an env and return a :class:`RunResult` (see ``Model.run``)."""
    client, contract, owns_client = _connect(env_or_address, token, remote_env_cls)
    _reject_vector_env(contract)
    adapter = _resolve_adapter(spec, contract, trust_entrypoints)
    text_keys = _text_input_keys(spec)
    if max_episodes is not None:
        n_episodes = max_episodes
    elif seeds is not None:
        n_episodes = len(seeds)
    else:
        n_episodes = 1
    episodes: list[EpisodeResult] = []
    on_close_error: BaseException | None = None
    try:
        for i in range(n_episodes):
            seed = seeds[i] if seeds is not None and i < len(seeds) else None
            episodes.append(
                _run_episode(
                    client,
                    predict,
                    adapter,
                    on_reset,
                    i,
                    seed,
                    instruction,
                    text_keys,
                    bridge,
                )
            )
            if on_episode_end is not None:
                on_episode_end()
    finally:
        if on_close is not None:
            try:
                on_close()
            except BaseException as exc:
                on_close_error = exc
        if close_env:
            # close_env is the caller's explicit opt-in to stop the env. When we
            # dialed the address ourselves the owner-level target is the client we
            # opened; otherwise it is the caller-supplied env/address. Without
            # close_env we never shut down a possibly-shared remote env.
            _shutdown(client if owns_client else env_or_address)
        if owns_client:
            _close(client)
    if on_close_error is not None:
        raise on_close_error
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
    bridge: ValueBridge | None = None,
) -> EpisodeResult:
    obs, _info = _reset(client, seed)
    if adapter is not None:
        adapter.reset()
    if on_reset is not None:
        on_reset()
    total = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated) and steps < _MAX_STEPS_PER_EPISODE:
        if adapter is not None:
            payload = from_value(
                adapter.transform_obs_value(
                    obs, input_bridge=bridge, custom_bridge=bridge
                ),
                bridge,
            )
        else:
            payload = obs
        if instruction is not None and isinstance(payload, dict):
            # Inject into a shallow copy; don't mutate the obs the env returned.
            payload = cast("dict[str, Any]", payload).copy()
            for key in text_keys:
                payload[key] = instruction
        action = predict(payload)
        if adapter is not None:
            action = from_value(
                adapter.transform_action_value(action, action_bridge=bridge),
                bridge,
            )
        obs, reward, terminated, truncated, _info = client.step(action)
        total += float(reward)
        steps += 1
    # Hitting the step cap is a truncation, not a silent non-outcome: without this
    # the episode would count as neither success nor truncation in success_rate.
    if not terminated and not truncated:
        truncated = True
    return EpisodeResult(
        index=index,
        seed=seed,
        steps=steps,
        reward=total,
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


def resolve_route_adapter(
    spec: object | None, contract: EnvContract, trust_entrypoints: bool
) -> Adapter | None:
    """Resolve a served route's adapter from its configure-time env contract.

    The serve-path counterpart of the run(env) resolution: a served model
    receives the env contract once per route (the ``ConfigureRoute`` RPC), so it
    resolves the adapter there rather than at connect. Returns ``None`` for a
    spec-less / ``DELEGATED`` model (no transform). Raises on a spec/env mismatch
    so route configuration fails loudly instead of predicting wrongly.
    """
    from ..adapters import AdapterResolutionError

    adapter = _resolve_adapter(spec, contract, trust_entrypoints)
    num_envs = getattr(contract, "num_envs", 1) or 1
    # A stateful adapter's frame-history buffers are not lane-indexed, so on a
    # vectorized route one lane's autoreset would clear or stale the others.
    # Reject at configure rather than corrupt; the resolved adapter is then
    # always a single lane.
    if adapter is not None and adapter.is_stateful and num_envs > 1:
        raise AdapterResolutionError(
            f"a stateful adapter (frame-stacking etc.) cannot be served against a "
            f"vectorized route (num_envs={num_envs}): its frame-history buffers are "
            "not lane-indexed, so one lane's autoreset would corrupt the others. "
            "Serve it against num_envs=1, or use a stateless adapter."
        )
    return adapter


def adapted_predict(
    predict: Callable[[Any], Any],
    adapter: Adapter | None,
    observation: Any,
    bridge: ValueBridge | None,
) -> Any:
    """Run ``predict`` with the route's adapter (if any) applied around it.

    Mirrors the per-step transform in :func:`_run_episode` (obs in, action out)
    for the serve path, where the adapter is resolved per route at configure time
    rather than once per run.
    """
    if adapter is None:
        return predict(observation)
    payload = from_value(
        adapter.transform_obs_value(
            observation, input_bridge=bridge, custom_bridge=bridge
        ),
        bridge,
    )
    action = predict(payload)
    return from_value(adapter.transform_action_value(action, action_bridge=bridge), bridge)


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
    return resolve_from_contract(contract, spec, trust_entrypoints=trust_entrypoints)


def _reject_vector_env(contract: EnvContract | None) -> None:
    # The per-episode loop is single-env: it reads scalar reward/termination. A
    # vector env (num_envs>1) would crash on array truthiness, so reject it up
    # front rather than deep in the step loop.
    num_envs = getattr(contract, "num_envs", 1) if contract is not None else 1
    if num_envs and num_envs > 1:
        raise ValueError(
            f"Model.run() drives a single env, but the env reports num_envs={num_envs}; "
            "use num_envs=1 (the per-episode loop reads scalar reward/termination)."
        )


def coerce_model(
    source: Any,
    *,
    spec: object | None,
    artifacts: tuple[ArtifactInput, ...],
    load_kwargs: dict[str, object] | None,
) -> CoercedModel:
    """Resolve a model source into a :class:`CoercedModel`.

    The model source is a predict callable; ``artifacts``/``load_kwargs`` are
    accepted for forward compatibility but are unused on the callable path.
    """
    _ = artifacts, load_kwargs
    if callable(source):
        return CoercedModel(source, spec, None, None, None)
    raise TypeError(
        f"Model source must be a predict callable; got {type(source).__name__}"
    )


def _text_input_keys(spec: object | None) -> tuple[str, ...]:
    if spec is None or spec is DELEGATED:
        return ()
    from ..adapters import TextInput

    return tuple(
        inp.key for inp in getattr(spec, "inputs", ()) if isinstance(inp, TextInput)
    )


def _connect(
    target: object, token: str, remote_env_cls: type | None
) -> tuple[Any, Any, bool]:
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


def _close(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _shutdown(target: object) -> None:
    # A SandboxSession's `shutdown` forwards to the remote handle; only close() stops
    # the container.
    from .._sandbox.session import SandboxSessionBase

    if isinstance(target, SandboxSessionBase):
        target.close()
        return
    # decide reason-passing by binding the signature, not by try/except
    # around the call, so a TypeError raised *inside* the callable is never swallowed.
    import inspect

    for name in ("shutdown", "close"):
        fn = getattr(target, name, None)
        if not callable(fn):
            continue
        try:
            inspect.signature(fn).bind("model run complete")
            accepts_reason = True
        except (TypeError, ValueError, KeyError):  # also un-introspectable builtins
            accepts_reason = False
        fn("model run complete") if accepts_reason else fn()
        return
