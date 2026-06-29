"""The model-side eval loop, shared by every framework ``Model``.

A model consumes the env contract rather than publishing its own: a :class:`Session`
dials an env, pulls its contract, resolves the adapter from the env's tags and the
model's spec, and runs a per-episode loop that returns a typed :class:`RunResult`.

The supporting machinery lives in sibling modules and is re-exported here for the
names :class:`Session` resolves as module globals: connection/contract synthesis
(:mod:`._connect`), role-addressed reads (:mod:`._read`), adapter resolution
(:mod:`._resolve`), source coercion (:mod:`._coerce`), and instruction injection
(:mod:`._instruction`).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from .._value_conversion import from_value, identity_bridge
from ._chunk import ChunkReplay
from ._coerce import RANDOM_SAMPLE
from ._connect import (
    adapter_env_bridge,
    close_client,
    connect_env,
    reset_env,
    shutdown_env,
)
from ._instruction import TextPlacement, text_placements, tree_set
from ._read import Reader, resolve_read_adapter
from ._resolve import reject_vector_env, resolve_adapter, resolve_route_adapter
from ._view import ViewerDriver, resolve_view

if TYPE_CHECKING:
    from rlmesh._rlmesh import PyModelClient

    from .._value_conversion import ValueBridge

__all__ = [
    "RANDOM_SAMPLE",
    "EpisodeResult",
    "RunResult",
    "Session",
    "resolve_route_adapter",
]

# bound the loop so a non-terminating env cannot hang it forever.
_MAX_STEPS_PER_EPISODE = 100_000

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


def _episode_success(info: Mapping[str, Any]) -> bool | None:
    """Read an env-reported task outcome from a step ``info`` (Gymnasium convention).

    Returns the ``is_success`` / ``success`` flag when the env emits one, else
    ``None`` -- callers then fall back to ``terminated``.
    """
    for key in ("is_success", "success"):
        if key in info:
            return bool(info[key])
    return None


@dataclass(frozen=True)
class EpisodeResult:
    """The outcome of one evaluation episode."""

    index: int
    seed: int | None
    steps: int
    reward: float
    terminated: bool
    truncated: bool
    #: The env-reported task outcome from the final step's ``info``
    #: (Gymnasium's ``is_success`` / ``success`` key), or ``None`` when the env
    #: emits no such signal. Distinct from ``terminated`` (which only says the
    #: episode reached a terminal state, not whether it succeeded).
    success: bool | None = None


@dataclass(frozen=True)
class RunResult:
    """The result of a :meth:`Model.run` eval."""

    episodes: tuple[EpisodeResult, ...] = ()

    @property
    def num_episodes(self) -> int:
        """Number of episodes in this result."""
        return len(self.episodes)

    @property
    def total_steps(self) -> int:
        """Total env steps across all episodes."""
        return sum(e.steps for e in self.episodes)

    @property
    def mean_reward(self) -> float:
        """Mean total reward per episode (``0.0`` when empty)."""
        if not self.episodes:
            return 0.0
        return sum(e.reward for e in self.episodes) / len(self.episodes)

    @property
    def success_rate(self) -> float:
        """Fraction of episodes that succeeded.

        Prefers the env-reported task outcome (Gymnasium ``info["is_success"]`` /
        ``["success"]``, captured per episode in :attr:`EpisodeResult.success`).
        For an env that emits no such signal, falls back to ``terminated`` for
        that episode -- so a time-limit env whose success *is* the truncation
        cap should report success via ``info`` rather than rely on this.
        """
        if not self.episodes:
            return 0.0
        succeeded = sum(
            1
            for e in self.episodes
            if (e.terminated if e.success is None else e.success)
        )
        return succeeded / len(self.episodes)

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
    text_placements: tuple[TextPlacement, ...],
    env_bridge: ValueBridge | None,
    model_bridge: ValueBridge | None,
    device: object | None,
) -> Any:
    """Assemble one observation into the model payload and call ``predict``.

    The re-plan half of the chunk-replay loop (skipped while a chunk is replaying):
    the declarative obs transform (or the raw obs for a spec-less model),
    instruction injection (one rebuilt copy, never the env's obs), then the model
    forward. ``device`` (the model's, torch/jax) moves every obs tensor leaf onto it
    before predict -- the local dual of the served worker -- so the author never
    calls ``.to(device)``; a no-op for None or a non-device framework.
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
    if device is not None and model_bridge is not None:
        payload = model_bridge.to_device(payload, device)
    if instruction is not None:
        # Inject into every text leaf the spec declares, at its tree placement and
        # in its declared shape: ``[instruction]`` for container='list', a bare
        # ``str`` otherwise. ``tree_set`` rebuilds the path it touches (the empty
        # path replaces the whole payload, for a bare-root text input), so the obs
        # the env returned is never mutated.
        for placement in text_placements:
            value: Any = [instruction] if placement.as_list else instruction
            payload = tree_set(payload, placement.segments, value)
    return predict(payload)


class Session(Generic[ObsT, ActT]):
    """A model bound to one env: drive it by hand, or pump whole episodes.

    The neutral pair-driver returned by :func:`rlmesh.session`. ``reset`` / ``predict`` /
    ``step`` drive one step at a time -- ``predict`` applies the model's adapter (resolved
    from the env's published contract) around the model's own predict, replaying an action
    chunk one action per step when ``execution_horizon`` > 1 and the model defines
    ``predict_chunk``. ``run`` pumps whole episodes and returns a typed :class:`RunResult`.

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
        execution_horizon: int = 1,
        model_client: PyModelClient | None = None,
        owner: Any = None,
        device: object | None = None,
        view: object = None,
    ) -> None:
        # Two modes: a local model (``predict`` + client-side ``spec`` adapter) or a
        # served model (``model_client`` -- the server applies its adapter). ``owner``
        # is a managed source (e.g. a SandboxModel container) to shut down on close.
        self._predict = predict
        self._predict_chunk = predict_chunk
        self._execution_horizon = execution_horizon
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
        #: Compute device for the local model's inputs (torch/jax), from the model's
        #: ``device``; obs tensor leaves are moved onto it before predict. None / a
        #: served model (the server worker places obs) leaves it unset.
        self._device = device
        self._connected = False
        self._client: Any = None
        self._owns_client = False
        self._adapter: Any = None
        self._contract: Any = None
        self._env_bridge: ValueBridge | None = None
        self._text_placements: tuple[TextPlacement, ...] = ()
        self._horizon = 1
        self._replay = ChunkReplay(1)
        self._terminated = False
        self._truncated = False
        self._steps = 0
        self._reward = 0.0
        self._last_info: Mapping[str, Any] = {}
        #: Whether an episode has been reset and not yet ended. The local episode
        #: boundary (a stateful model's `on_episode_end`) fires when the next reset()
        #: begins or the session closes -- see `_end_episode`.
        self._episode_open = False
        #: Resolved Readers (sess.read), cached per read item so a per-step read
        #: does not re-resolve.
        self._read_cache: dict[Any, Reader] = {}
        #: Optional built-in debug viewer (``view=`` on run/session). Lazily builds
        #: a native ``PyViewer`` on the first fed frame; best-effort, never fatal.
        _view = resolve_view(view)
        self._view_driver = ViewerDriver(_view) if _view is not None else None

    def _ensure_connected(self) -> None:
        if self._connected:
            return
        client, contract, owns = connect_env(
            self._env, self._token, self._remote_env_cls
        )
        reject_vector_env(contract)
        self._client = client
        self._contract = contract
        self._owns_client = owns
        # A served model resolves its adapter server-side (from the contract sent at
        # bind); only a local model resolves it here, client-side.
        if self._model_client is None:
            self._adapter = resolve_adapter(self._spec, contract, self._trust)
            self._env_bridge = (
                adapter_env_bridge(client) if self._adapter is not None else None
            )
            self._text_placements = text_placements(self._spec)
            # The execution horizon is a caller decision (execution_horizon), not the
            # spec. It engages only when the model exposes a predict_chunk corner;
            # without one, fall back to single-step predict.
            if self._execution_horizon > 1 and self._predict_chunk is None:
                warnings.warn(
                    f"execution_horizon={self._execution_horizon} was requested but "
                    "the model defines no predict_chunk(); running un-chunked.",
                    stacklevel=2,
                )
            self._horizon = (
                self._execution_horizon if self._predict_chunk is not None else 1
            )
            # Seed the replay with the resolved horizon so a hand-driven predict()
            # before the first reset() already replays the right chunk length.
            self._replay = ChunkReplay(self._horizon)
        self._connected = True

    @property
    def done(self) -> bool:
        """Whether the current episode has terminated or truncated."""
        return self._terminated or self._truncated

    def _feed_view(self, obs: object) -> None:
        """Push the current obs + HUD to the debug viewer, if one is attached."""
        if self._view_driver is not None:
            self._view_driver.feed(
                contract=self._contract,
                client=self._client,
                obs=obs,
                read=self.read,
                steps=self._steps,
                reward=self._reward,
                outcome=self._view_outcome(),
            )

    def _view_outcome(self) -> str:
        """The viewer HUD's outcome label for the current step.

        Prefers the env-reported task result (:func:`_episode_success` over the last
        step's ``info``); only when the env emits no such signal does it fall back to
        ``terminated`` -- matching :attr:`RunResult.success_rate`, and never reading a
        plain terminal state as a success.
        """
        if not (self._terminated or self._truncated):
            return ""
        success = _episode_success(self._last_info)
        if success is None:
            success = self._terminated
        if success:
            return "success"
        return "failure" if self._terminated else "timeout"

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
        obs, info = reset_env(self._client, seed)
        if self._model_client is not None:
            self._model_client.reset()  # mark a reset boundary on the served route
        else:
            if self._adapter is not None:
                self._adapter.reset()
            self._replay = ChunkReplay(self._horizon)
        self._terminated = self._truncated = False
        self._steps = 0
        self._reward = 0.0
        self._last_info = info
        self._episode_open = True
        self._feed_view(obs)
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
        # The horizon goes in positionally: the corner was normalized to the internal
        # (obs, horizon) contract, so a model that ignores it still binds cleanly.
        if self._horizon > 1 and self._predict_chunk is not None:
            chunk_fn = self._predict_chunk
            horizon = self._horizon

            def _replay(obs: Any) -> Any:
                return chunk_fn(obs, horizon)

            replay_fn = cast("Callable[[Any], Any]", _replay)
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
                self._device,
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
        self._last_info = info
        self._feed_view(obs)
        return (
            cast("ObsT", obs),
            float(reward),
            bool(terminated),
            bool(truncated),
            cast("Mapping[str, Any]", info),
        )

    def reader(self, *items: object) -> Reader:
        """Build a read-only, role-addressed view over this env's observations.

        Each item is a role constant -- kept in the env's native encoding -- or a
        model-input leaf declaring the encoding you want
        (``Image(IMAGE_PRIMARY, layout="hwc")``, ``State(EEF_POS)``). The returned
        :class:`Reader` maps a raw observation to ``{role: value}`` through the same
        adapter pipeline a model uses, so it is encoding-agnostic across envs and
        runs identically in the native core. Resolved once here, reused each step::

            read = sess.reader(Image(IMAGE_PRIMARY, layout="hwc"), EEF_POS)
            obs, _ = sess.reset()
            while not sess.done:
                screen.show(read(obs)[IMAGE_PRIMARY])
                obs, *_ = sess.step(sess.predict(obs))

        A bare role is desugared to the env-native leaf for that role (by the env's
        own tag); pass an explicit leaf to override the encoding.
        """
        if not items:
            raise TypeError(
                "reader() needs at least one role or model-input leaf to read"
            )
        self._ensure_connected()
        adapter, roles = resolve_read_adapter(self._contract, items, self._trust)
        return Reader(adapter, roles, adapter_env_bridge(self._client))

    def read(self, observation: object, item: object) -> object:
        """One-shot read of a single role from one observation.

        The single-value convenience for :meth:`reader` -- ``item`` is a role
        constant or a model-input leaf. The reader is resolved once and cached per
        item, so calling this every step does not re-resolve::

            ee = sess.read(obs, EEF_POS)
            img = sess.read(obs, Image(IMAGE_PRIMARY, layout="hwc"))
        """
        reader = self._read_cache.get(item)
        if reader is None:
            reader = self.reader(item)
            self._read_cache[item] = reader
        return reader(observation)[reader.roles[0]]

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
                obs, last_info = self.reset(seed=seed)
                while not self.done and self._steps < _MAX_STEPS_PER_EPISODE:
                    obs, _r, _t, _tr, last_info = self.step(self.predict(obs))
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
                        success=_episode_success(last_info),
                    )
                )
        finally:
            # End the final episode (earlier ones end at the next reset()), then the
            # close hook -- `["episode_end", ..., "close"]`.
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
        if self._view_driver is not None:
            self._view_driver.close()
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
                    shutdown_env(self._client if self._owns_client else self._env)
            finally:
                # Always release the dialed connection and clear state, even if the
                # shutdown raised (the error still propagates after cleanup).
                if self._owns_client and self._client is not None:
                    close_client(self._client)
                self._connected = False
                self._client = None

    def __enter__(self) -> Session[ObsT, ActT]:
        return self

    def __exit__(self, *exc: object) -> None:
        _ = exc
        self.close()
