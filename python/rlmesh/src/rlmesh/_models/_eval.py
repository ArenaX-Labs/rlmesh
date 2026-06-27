"""The model-side eval loop and source coercion, shared by every framework ``Model``.

A model consumes the env contract rather than publishing its own: a :class:`Session`
dials an env, pulls its contract, resolves the adapter from the env's tags and the
model's spec, and runs a per-episode loop that returns a typed :class:`RunResult`.
``coerce_model`` turns any model source into a :class:`CoercedModel`.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar, cast

from .._value_conversion import from_value, identity_bridge
from ._adapter_mode import NO_ADAPTER
from ._chunk import ChunkReplay

if TYPE_CHECKING:
    from rlmesh._rlmesh import PyModelClient

    from .._value_conversion import ValueBridge
    from ..adapters import Adapter
    from ..specs import EnvContract

__all__ = [
    "RANDOM_SAMPLE",
    "EpisodeResult",
    "RunResult",
    "Session",
    "coerce_model",
    "resolve_route_adapter",
]

# bound the loop so a non-terminating env cannot hang it forever.
_MAX_STEPS_PER_EPISODE = 100_000

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


class _RandomSample:
    """Sentinel policy: act by sampling the env's action space (a random baseline)."""

    def __repr__(self) -> str:
        return "RANDOM_SAMPLE"


RANDOM_SAMPLE = _RandomSample()
"""Pass as the model to :func:`rlmesh.session`/:func:`rlmesh.run` to sample actions."""


class CoercedModel(NamedTuple):
    predict: Callable[[Any], Any]
    spec: object | None
    # A duck-typed policy's ``reset()`` is wired here, to the episode-END edge: it
    # is the only per-episode boundary both the local loop and the served wire path
    # signal, so a stateful policy clears its state identically either way.
    on_episode_end: Callable[[], None] | None
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


def _predict_step(
    predict: Callable[[Any], Any],
    obs: Any,
    adapter: Any,
    instruction: str | None,
    text_placements: tuple[_TextPlacement, ...],
    env_bridge: ValueBridge | None,
    model_bridge: ValueBridge | None,
) -> Any:
    """Assemble one observation into the model payload and call ``predict``.

    The re-plan half of the chunk-replay loop (skipped while a chunk is replaying):
    the declarative obs transform (or the raw obs for a spec-less model),
    instruction injection (one rebuilt copy, never the env's obs), then the model
    forward.
    """
    if adapter is not None:
        payload = from_value(
            adapter.transform_obs_value(
                obs, input_bridge=env_bridge, custom_bridge=env_bridge
            ),
            model_bridge,
        )
    else:
        payload = obs
    if instruction is not None:
        # Inject into every text leaf the spec declares, at its tree placement and
        # in its declared shape: ``[instruction]`` for container='list', a bare
        # ``str`` otherwise. ``_tree_set`` rebuilds the path it touches (the empty
        # path replaces the whole payload, for a bare-root text input), so the obs
        # the env returned is never mutated.
        for placement in text_placements:
            value: Any = [instruction] if placement.as_list else instruction
            payload = _tree_set(payload, placement.segments, value)
    return predict(payload)


class Session(Generic[ObsT, ActT]):
    """A model bound to one env: drive it by hand, or pump whole episodes.

    The neutral pair-driver returned by :func:`rlmesh.session`. ``reset`` / ``predict`` /
    ``step`` drive one step at a time -- ``predict`` applies the model's adapter (resolved
    from the env's published contract) around the model's own predict, replaying an action
    chunk one action per step when the spec declares an execute horizon > 1. ``run`` pumps
    whole episodes and returns a typed :class:`RunResult`.

    The env connection is opened lazily on first ``reset`` (manual driving); ``run`` drives
    whole episodes through the same primitives. Use it as a context manager to close a
    manually-opened connection.
    """

    def __init__(
        self,
        *,
        env: object,
        predict: Callable[[Any], Any] | None = None,
        predict_chunk: Callable[..., Any] | None = None,
        spec: object | None = None,
        on_episode_end: Callable[[], None] | None = None,
        on_close: Callable[[], None] | None = None,
        trust_entrypoints: bool = False,
        bridge: ValueBridge | None = None,
        remote_env_cls: type | None = None,
        instruction: str | None = None,
        close_env: bool = False,
        token: str = "",
        action_horizon: int = 1,
        model_client: PyModelClient | None = None,
        owner: Any = None,
    ) -> None:
        # Two modes: a local model (``predict`` + client-side ``spec`` adapter) or a
        # served model (``model_client`` -- the server applies its adapter). ``owner``
        # is a managed source (e.g. a SandboxModel container) to shut down on close.
        self._predict = predict
        self._predict_chunk = predict_chunk
        self._action_horizon = action_horizon
        self._spec = spec
        self._env = env
        self._on_episode_end = on_episode_end
        self._on_close = on_close
        self._trust = trust_entrypoints
        self._bridge = bridge
        self._remote_env_cls = remote_env_cls
        self._instruction = instruction
        self._close_env = close_env
        self._token = token
        self._model_client = model_client
        self._owner = owner
        self._connected = False
        self._client: Any = None
        self._owns_client = False
        self._adapter: Any = None
        self._env_bridge: ValueBridge | None = None
        self._text_placements: tuple[_TextPlacement, ...] = ()
        self._horizon = 1
        self._replay = ChunkReplay(1)
        self._terminated = False
        self._truncated = False
        self._steps = 0
        self._reward = 0.0
        #: Whether an episode has been reset and not yet ended. The local episode
        #: boundary (a stateful model's `on_episode_end`) fires when the next reset()
        #: begins or the session closes — see `_end_episode`.
        self._episode_open = False

    def _ensure_connected(self) -> None:
        if self._connected:
            return
        client, contract, owns = _connect(self._env, self._token, self._remote_env_cls)
        _reject_vector_env(contract)
        self._client = client
        self._owns_client = owns
        # A served model resolves its adapter server-side (from the contract sent at
        # bind); only a local model resolves it here, client-side.
        if self._model_client is None:
            self._adapter = _resolve_adapter(self._spec, contract, self._trust)
            self._env_bridge = (
                _adapter_env_bridge(client) if self._adapter is not None else None
            )
            self._text_placements = _text_placements(self._spec)
            # The replay horizon is a caller decision (action_horizon), not the
            # spec. It engages only when the model exposes a predict_chunk corner;
            # without one, fall back to single-step predict.
            if self._action_horizon > 1 and self._predict_chunk is None:
                warnings.warn(
                    f"action_horizon={self._action_horizon} was requested but the "
                    "model defines no predict_chunk(); running un-chunked.",
                    stacklevel=2,
                )
            self._horizon = (
                self._action_horizon if self._predict_chunk is not None else 1
            )
            # Seed the replay with the resolved horizon so a hand-driven predict()
            # before the first reset() already replays the right chunk length.
            self._replay = ChunkReplay(self._horizon)
        self._connected = True

    @property
    def done(self) -> bool:
        """Whether the current episode has terminated or truncated."""
        return self._terminated or self._truncated

    def _end_episode(self) -> None:
        """Fire the local model's `on_episode_end` once for the currently-open episode.

        The episode boundary on the local drive path: an open episode ends when the
        next reset() begins, or when the session closes. This fires the same
        `on_episode_end` the served path drives via `ResetAdapter`, so a stateful
        local model (a subclass/duck-typed policy's `reset()` is wired here) clears
        its per-episode state identically whether driven by hand or via `run()`. For
        a served model `_on_episode_end` is None (the remote engine owns the hook),
        so this is a no-op there. Idempotent: safe to call from both reset() and
        close().
        """
        if self._episode_open:
            self._episode_open = False
            if self._on_episode_end is not None:
                self._on_episode_end()

    def reset(self, *, seed: int | None = None) -> tuple[ObsT, Mapping[str, Any]]:
        """Begin a new episode: end the previous one, then reset the env and adapter.

        Ending the previous episode fires the model's `on_episode_end` (the local
        per-episode boundary), so a stateful model clears its state between episodes
        on the hand-driven path too, not only via `run()`.
        """
        self._ensure_connected()
        self._end_episode()
        obs, info = _reset(self._client, seed)
        if self._model_client is not None:
            self._model_client.reset()  # mark a reset boundary on the served route
        else:
            if self._adapter is not None:
                self._adapter.reset()
            self._replay = ChunkReplay(self._horizon)
        self._terminated = self._truncated = False
        self._steps = 0
        self._reward = 0.0
        self._episode_open = True
        return cast("ObsT", obs), info

    def predict(self, observation: ObsT) -> ActT:
        """Map one env observation to an env-ready action (the model's adapter applied)."""
        self._ensure_connected()
        if self._predict is RANDOM_SAMPLE:
            return cast("ActT", self._client.action_space.sample())
        if self._model_client is not None:
            # Served model: the server applies the adapter (and any chunk replay);
            # we only bridge the obs out and the env-ready action back.
            bridge = self._bridge if self._bridge is not None else identity_bridge
            action = self._model_client.predict(bridge.encode(observation))
            return cast("ActT", bridge.decode(action))
        model_bridge = self._bridge if self._bridge is not None else self._env_bridge
        # Local mode always has a predict (only the served-model branch above lacks
        # one). When chunking (horizon > 1) the replay re-plans through
        # predict_chunk(payload, horizon) -- which returns a chunk the queue splits
        # and replays one step at a time -- otherwise through single-step predict.
        if self._horizon > 1 and self._predict_chunk is not None:
            replay_fn = cast(
                "Callable[[Any], Any]",
                functools.partial(self._predict_chunk, horizon=self._horizon),
            )
        else:
            replay_fn = cast("Callable[[Any], Any]", self._predict)
        raw_action = self._replay.next_action(
            lambda: _predict_step(
                replay_fn,
                observation,
                self._adapter,
                self._instruction,
                self._text_placements,
                self._env_bridge,
                model_bridge,
            )
        )
        if self._adapter is not None:
            return cast(
                "ActT",
                from_value(
                    self._adapter.transform_action_value(
                        raw_action, action_bridge=model_bridge
                    ),
                    self._env_bridge,
                ),
            )
        return cast("ActT", raw_action)

    def step(self, action: ActT) -> tuple[ObsT, float, bool, bool, Mapping[str, Any]]:
        """Apply one action to the env; record reward and termination."""
        self._ensure_connected()
        obs, reward, terminated, truncated, info = self._client.step(action)
        self._reward += float(reward)
        self._steps += 1
        self._terminated = bool(terminated)
        self._truncated = bool(truncated)
        return (
            cast("ObsT", obs),
            float(reward),
            bool(terminated),
            bool(truncated),
            cast("Mapping[str, Any]", info),
        )

    def run(
        self,
        *,
        seeds: Sequence[int] | None = None,
        max_episodes: int | None = None,
    ) -> RunResult:
        """Drive whole episodes to completion and return a typed :class:`RunResult`.

        The single drive loop: pumps this session's own ``reset`` / ``predict`` /
        ``step`` primitives, so ``Model.run`` routes through here.
        ``seeds`` gives a per-episode seed and sets the episode count unless
        ``max_episodes`` is given.
        """
        self._ensure_connected()
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
                obs, _info = self.reset(seed=seed)
                while not self.done and self._steps < _MAX_STEPS_PER_EPISODE:
                    obs, _r, _t, _tr, _info = self.step(self.predict(obs))
                # Hitting the step cap is a truncation, not a silent non-outcome.
                if not self._terminated and not self._truncated:
                    self._truncated = True
                episodes.append(
                    EpisodeResult(
                        index=i,
                        seed=seed,
                        steps=self._steps,
                        reward=self._reward,
                        terminated=self._terminated,
                        truncated=self._truncated,
                    )
                )
        finally:
            # End the final episode (earlier ones end at the next reset()), then the
            # close hook — `["episode_end", ..., "close"]`.
            self._end_episode()
            if self._on_close is not None:
                try:
                    self._on_close()
                except BaseException as exc:
                    on_close_error = exc
            self.close()
        if on_close_error is not None:
            raise on_close_error
        return RunResult(episodes=tuple(episodes))

    def close(self) -> None:
        """Close this session: the served route (and any owned source), then the env.

        For a served model, closes the model client and shuts down a managed source it
        started (e.g. a ``SandboxModel`` container). For the env, shuts it down only on
        the ``close_env`` opt-in and closes a connection this session dialed.
        """
        # End an episode left open by a hand-driven loop, so a stateful model's
        # `on_episode_end` fires for the last episode even without a following reset().
        self._end_episode()
        model_client = self._model_client
        if model_client is not None:
            self._model_client = None
            try:
                model_client.close()
            finally:
                owner = self._owner
                self._owner = None
                if owner is not None:
                    owner.shutdown()
        if self._connected:
            try:
                if self._close_env:
                    # Explicit opt-in to stop the env: the dialed client if we opened
                    # it, else the caller-supplied env/address.
                    _shutdown(self._client if self._owns_client else self._env)
            finally:
                # Always release the dialed connection and clear state, even if the
                # shutdown raised (the error still propagates after cleanup).
                if self._owns_client and self._client is not None:
                    _close(self._client)
                self._connected = False
                self._client = None

    def __enter__(self) -> Session[ObsT, ActT]:
        return self

    def __exit__(self, *exc: object) -> None:
        _ = exc
        self.close()


def resolve_route_adapter(
    spec: object | None, contract: EnvContract, trust_entrypoints: bool
) -> Adapter | None:
    """Resolve a served route's adapter from its configure-time env contract.

    The serve-path counterpart of the run(env) resolution: a served model
    receives the env contract once per route (the ``ConfigureRoute`` RPC), so it
    resolves the adapter there rather than at connect. Returns ``None`` for a
    spec-less / ``NO_ADAPTER`` model (no transform). Raises on a spec/env mismatch
    so route configuration fails loudly instead of predicting wrongly.

    Frame-stacking state is now episode-keyed in the native serving engine, so a
    stateful (frame-stacking) adapter serves correctly against a vectorized route
    -- the old single-lane rejection is lifted. (Model-*internal* state, which the
    engine cannot key by episode, is gated to single-lane by a registration-time
    probe instead.)
    """
    return _resolve_adapter(spec, contract, trust_entrypoints)


def _resolve_adapter(
    spec: object | None, contract: EnvContract | None, trust_entrypoints: bool
) -> Adapter | None:
    from ..adapters import (
        AdapterResolutionError,
        EnvTags,
        ModelSpec,
        resolve_from_contract,
    )

    if spec is NO_ADAPTER:
        return None
    metadata = contract.metadata if contract is not None else None
    tagged = EnvTags.from_metadata(metadata or {}) is not None
    if spec is None:
        if tagged:
            raise AdapterResolutionError(
                "the env publishes adapter tags but this model has spec=None; "
                "pass spec=<ModelSpec> to adapt, or spec=NO_ADAPTER if the model "
                "adapts its own observations"
            )
        return None
    if not isinstance(spec, ModelSpec):
        raise AdapterResolutionError(
            f"a model spec must be a ModelSpec or NO_ADAPTER; got {type(spec).__name__}"
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


def _is_model_source(source: object) -> bool:
    """Whether ``source`` is a ``Model`` (instance or subclass) -- rejected by coerce_model.

    Kept as a helper so the ``isinstance`` narrowing stays on this local ``object``
    parameter rather than leaking onto coerce_model's ``Any`` source.
    """
    from .base import ModelBase

    return isinstance(source, ModelBase) or (
        isinstance(source, type) and issubclass(source, ModelBase)
    )


def coerce_model(
    source: Any,
    *,
    spec: object | None,
) -> CoercedModel:
    """Resolve a model source into a :class:`CoercedModel`.

    The source is either a bare predict callable or a duck-typed policy object
    (class or instance) exposing ``predict`` plus optional ``spec``/``reset``/``close``.
    A :class:`~rlmesh._models.base.ModelBase` is rejected: a ``Model`` builds its own
    worker, so instantiate the subclass directly rather than wrapping it again.
    """
    if _is_model_source(source):
        raise TypeError(
            "coerce_model received a Model. Instantiate your Model subclass directly "
            "(it builds its own worker), or pass a predict callable."
        )

    from .._bootstrap.loaders import construct_authored_model, looks_like_policy

    # A policy *class* is also callable, so check the policy shape first.
    if looks_like_policy(source):
        inst = construct_authored_model(source)
        return CoercedModel(
            inst.predict,
            spec if spec is not None else getattr(inst, "spec", None),
            getattr(inst, "reset", None),
            getattr(inst, "close", None),
            inst,
        )
    if callable(source):
        return CoercedModel(source, spec, None, None, None)
    raise TypeError(
        "Model source must be a predict callable or a policy object with predict(); "
        f"got {type(source).__name__}"
    )


class _TextPlacement(NamedTuple):
    """Where (and how) the ``instruction=`` override lands in the model payload.

    ``segments`` is the text leaf's position in the model input tree (str for a
    Dict key, int for a Tuple index; the empty tuple is a bare-root text input,
    whose payload *is* the text leaf). ``as_list`` is True when the leaf declares
    ``container='list'`` (inject ``[instruction]``, not a bare ``str``, to keep
    the model's declared shape).
    """

    segments: tuple[str | int, ...]
    as_list: bool


def _text_placements(spec: object | None) -> tuple[_TextPlacement, ...]:
    """Find every text leaf the ``instruction=`` override should be written into.

    Walks the model spec's input tree locally (a public structure: a leaf
    dataclass, a ``dict`` Dict node, or a ``tuple`` Tuple node), so the override
    reaches *every* text leaf -- bare-root, top-level, and nested -- and carries
    each leaf's ``container`` so a list-shaped leaf gets ``[instruction]``. A
    spec-less / ``NO_ADAPTER`` model declares no text inputs, so none.
    """
    if spec is None or spec is NO_ADAPTER:
        return ()
    from ..adapters import Text

    input_tree = getattr(spec, "input", None)
    if input_tree is None:
        return ()
    placements: list[_TextPlacement] = []

    def walk(node: Any, segments: tuple[str | int, ...]) -> None:
        if isinstance(node, Text):
            placements.append(_TextPlacement(segments, node.container == "list"))
        elif isinstance(node, Mapping):
            for key, child in cast("Mapping[str, Any]", node).items():
                walk(child, (*segments, key))
        elif isinstance(node, tuple):
            for index, child in enumerate(cast("tuple[Any, ...]", node)):
                walk(child, (*segments, index))

    walk(input_tree, ())
    return tuple(placements)


def _tree_set(tree: Any, segments: tuple[str | int, ...], value: Any) -> Any:
    """Return ``tree`` with the value at ``segments`` replaced by ``value``.

    A small structured set over the payload tree (dict for str segments, list for
    int segments). Rebuilds only the path it touches, so the env's observation is
    never mutated; the empty path replaces the whole payload (a bare-root leaf).
    """
    if not segments:
        return value
    head, rest = segments[0], segments[1:]
    if isinstance(head, int):
        items: list[Any] = list(cast("Sequence[Any]", tree))
        items[head] = _tree_set(items[head], rest, value)
        return items
    node: dict[str, Any] = (
        dict(cast("Mapping[str, Any]", tree)) if isinstance(tree, Mapping) else {}
    )
    # A subtree may not exist yet (e.g. injecting into a missing nested key);
    # descend into an empty dict rather than indexing a missing key.
    node[head] = _tree_set(node.get(head, {}), rest, value)
    return node


def _connect(
    target: object, token: str, remote_env_cls: type | None
) -> tuple[Any, Any, bool]:
    if isinstance(target, str):
        client = _remote_env(target, remote_env_cls)
        return client, client.env_contract, True
    if hasattr(target, "reset") and hasattr(target, "step"):
        # A live env: a remote/served handle exposes a native env_contract; a local
        # env exposes its spaces + metadata directly, so synthesize the contract from
        # the env (tags ride in env.metadata via tag() / EnvFactory.make).
        native = _native_contract(target)
        contract = native if native is not None else _local_contract(target)
        return target, contract, False
    if hasattr(target, "make"):
        # An EnvFactory: prepare()+make() its env (which carries the factory's tags)
        # and drive it locally -- no serving needed to resolve a spec'd adapter.
        env = _factory_env(target)
        return env, _local_contract(env), False
    address = getattr(target, "address", None)
    if isinstance(address, str):
        client = _remote_env(address, remote_env_cls)
        return client, client.env_contract, True
    raise TypeError(
        "session()/run() expect an env object, an EnvFactory, a remote-env object, "
        f"or an address string; got {type(target).__name__}"
    )


@dataclass(frozen=True)
class _LocalEnvContract:
    """Client-side stand-in for an env contract when driving a *local* env.

    A served env publishes a native ``EnvContract`` (spaces + metadata) over the
    handshake; a local env object exposes the same pieces directly. Bundling them
    here lets the adapter-resolution path be identical for local and remote envs --
    the env's tags ride in ``env.metadata`` (attached by :func:`rlmesh.adapters.tag`
    or :meth:`rlmesh.EnvFactory.make`).
    """

    metadata: Mapping[str, Any] | None
    observation_space: object
    action_space: object
    num_envs: int


def _native_contract(env: object) -> object | None:
    """Return a real ``env_contract`` (a remote/served handle) or ``None`` for a local env.

    Reads the attribute off the type or instance ``__dict__`` rather than via
    ``getattr``, so a gymnasium env does not trigger its deprecated
    wrapper-attribute forwarding warning just because we probed for a contract.
    """
    if getattr(type(env), "env_contract", None) is not None or (
        "env_contract" in getattr(env, "__dict__", {})
    ):
        try:
            return cast("Any", env).env_contract
        except AttributeError:
            return None
    return None


def _local_contract(env: object) -> Any:
    return _LocalEnvContract(
        metadata=getattr(env, "metadata", None),
        observation_space=getattr(env, "observation_space", None),
        action_space=getattr(env, "action_space", None),
        num_envs=_num_envs(env),
    )


def _num_envs(env: object) -> int:
    # A single env has no ``num_envs``; only a vector env does. Probe the type /
    # instance __dict__ (not a plain getattr) so a gymnasium env does not emit its
    # deprecated wrapper-attribute forwarding warning, same as _native_contract.
    if getattr(type(env), "num_envs", None) is not None or (
        "num_envs" in getattr(env, "__dict__", {})
    ):
        try:
            return int(cast("Any", env).num_envs or 1)
        except (AttributeError, TypeError, ValueError):
            return 1
    return 1


def _factory_env(factory: object) -> Any:
    """Build a local env from an EnvFactory: ``prepare()`` + ``make()``.

    ``EnvFactory.make`` stamps the factory's ``tags`` onto the env it returns, so a
    spec'd model can resolve its adapter from the local env alone -- no serving.
    """
    from .._bootstrap.loaders import construct_authored_env

    return construct_authored_env(factory)


def _remote_env(address: str, remote_env_cls: type | None) -> Any:
    if remote_env_cls is None:
        from ..numpy import RemoteEnv

        remote_env_cls = RemoteEnv
    return remote_env_cls(address)


def _adapter_env_bridge(client: Any) -> ValueBridge:
    """The bridge for the framework the env hands its observations *in*.

    The env-side encoder/decoder must match the env's own value type, never the
    model's. A remote/served handle decodes the wire payload into its framework
    before returning it (a torch ``RemoteEnv`` hands the loop torch tensors), so
    its ``_bridge`` is the right re-encoder for the native plan -- and is why the
    served cross-framework path works. A raw local env returns observations in its
    native array type, which for a gym/gymnasium env is numpy, so default to the
    numpy bridge: the model's framework bridge would reject numpy (the
    cross-framework local-driving bug). A custom local env that emits another
    framework's tensors can expose ``_bridge`` on its class to override.
    """
    bridge = getattr(client, "_bridge", None)
    if bridge is not None:
        return cast("ValueBridge", bridge)
    from ..numpy import _numpy_bridge  # pyright: ignore[reportPrivateUsage]

    return _numpy_bridge


def _reset(client: Any, seed: int | None) -> tuple[Any, Mapping[str, Any]]:
    result: Any = client.reset(seed=seed) if seed is not None else client.reset()
    if isinstance(result, tuple):
        pair = cast("tuple[Any, ...]", result)
        # Only a (obs, info) pair where the second element is a Mapping is a
        # gymnasium reset return; any other tuple is itself the observation.
        if len(pair) == 2 and isinstance(pair[1], Mapping):
            return pair[0], cast("Mapping[str, Any]", pair[1])
        return pair, {}
    return result, {}


def _close(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _shutdown(target: object) -> None:
    # A sandbox session's close() stops its container; its inherited shutdown() is the
    # remote owner-shutdown, so close() is the right teardown for an owned sandbox.
    from .._sandbox.session import SandboxLifecycle

    if isinstance(target, SandboxLifecycle):
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
