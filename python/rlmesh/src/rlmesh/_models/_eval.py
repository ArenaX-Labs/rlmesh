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

import inspect
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
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
from ._resolve import reject_vector_env, resolve_adapter
from ._view import ViewerDriver, resolve_view

if TYPE_CHECKING:
    from rlmesh._rlmesh import PyModelClient

    from .._value_conversion import ValueBridge
    from ..adapters import ObservationRoles

__all__ = [
    "RANDOM_SAMPLE",
    "EpisodeResult",
    "RunHooks",
    "RunResult",
    "Session",
    "StepEvent",
    "resolve_adapter",
]

# bound the loop so a non-terminating env cannot hang it forever.
_MAX_STEPS_PER_EPISODE = 100_000

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


def _ema(prev: float, sample: float) -> float:
    """Smooth a per-step timing sample, matching the viewer's fps smoothing (0.85/0.15).

    Seeds on the first real sample (``prev <= 0``) so the displayed value isn't dragged
    up from a cold zero.
    """
    return sample if prev <= 0.0 else 0.85 * prev + 0.15 * sample


def _episode_success(info: Mapping[str, Any]) -> bool | None:
    """Read an env-reported task outcome from a step ``info`` (Gymnasium convention).

    Returns the ``is_success`` / ``success`` flag when the env emits one, else
    ``None`` -- callers then fall back to ``terminated``.
    """
    for key in ("is_success", "success"):
        if key in info:
            return bool(info[key])
    return None


def episode_succeeded(*, success: bool | None, terminated: bool) -> bool:
    """The SDK success doctrine, as a bare function.

    Single source for :attr:`EpisodeResult.succeeded` and the recorder's
    :class:`~rlmesh.recorder.schema.EpisodeRecord`, which snapshots the same
    fields into a bundle document and must count success identically.
    """
    return terminated if success is None else success


@dataclass(frozen=True)
class EpisodeResult:
    """The outcome of one evaluation episode.

    Attributes:
        index: 0-based episode index within the run.
        seed: The seed the episode was reset with, or ``None``.
        steps: Number of env steps taken.
        reward: Total reward accumulated over the episode.
        terminated: Whether the env reported a terminal state.
        truncated: Whether the episode was cut short (env truncation, a
            ``max_episode_steps`` / ``max_episode_seconds`` cap, or the built-in
            step bound).
        success: The env-reported task outcome from the final step's ``info``
            (Gymnasium's ``is_success`` / ``success`` key), or ``None`` when the
            env emits no such signal. Distinct from ``terminated`` (which only
            says the episode reached a terminal state, not whether it succeeded).
        duration_s: Wall time from reset-return to episode end, in seconds.
        predict_ms: Mean per-step wall time of ``predict``, in milliseconds.
        step_ms: Mean per-step wall time of the env ``step`` round trip, in
            milliseconds.
    """

    index: int
    seed: int | None
    steps: int
    reward: float
    terminated: bool
    truncated: bool
    success: bool | None = None
    duration_s: float = 0.0
    predict_ms: float = 0.0
    step_ms: float = 0.0

    @property
    def succeeded(self) -> bool:
        """Whether this episode counts as a success.

        The single success doctrine: the env-reported :attr:`success` signal
        when present, falling back to :attr:`terminated` for an env that emits
        none. :attr:`RunResult.success_rate` and the recorder's exported
        ``successRate`` both count episodes through this property, so an SDK
        metric and an uploaded metric always agree.
        """
        return episode_succeeded(success=self.success, terminated=self.terminated)


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
        cap should report success via ``info`` rather than rely on this. When
        *no* episode in the run reported a signal, warns once: the whole rate is
        then the terminal-state fallback, a different definition of success.
        """
        if not self.episodes:
            return 0.0
        if all(e.success is None for e in self.episodes):
            warnings.warn(
                "no episode reported a task-outcome signal (Gymnasium "
                "info['is_success'] / info['success']); success_rate fell back "
                "to `terminated`, counting any terminal state as a success. Have "
                "the env emit success through step info to measure it directly.",
                stacklevel=2,
            )
        succeeded = sum(1 for e in self.episodes if e.succeeded)
        return succeeded / len(self.episodes)

    def __repr__(self) -> str:
        return (
            f"RunResult(episodes={self.num_episodes}, mean_reward={self.mean_reward:.3f}, "
            f"total_steps={self.total_steps})"
        )


@dataclass(frozen=True)
class StepEvent:
    """One env step of a :meth:`Session.run` eval, passed to :meth:`RunHooks.on_step`.

    Attributes:
        episode: 0-based episode index (equals :attr:`EpisodeResult.index`).
        seed: The episode's reset seed, or ``None``.
        step: 0-based step index within the episode.
        observation: The raw observation the action was predicted from (pre-step).
        action: The env-ready action applied to the env.
        reward: The step's reward.
        terminated: Whether the env reported a terminal state on this step.
        truncated: Whether this step truncated the episode.
        info: The step's ``info`` mapping.
        predict_ms: Raw wall time of this step's ``predict``, in milliseconds;
            near zero on chunk-replay steps, where no model forward runs.
        step_ms: Raw wall time of the env ``step`` round trip, in milliseconds.
        read: Lazy role reader bound to ``observation`` -- ``event.read(item)``
            delegates to :meth:`Session.read`, so resolution is cached per item
            and never triggered unless called.
    """

    episode: int
    seed: int | None
    step: int
    observation: Any
    action: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    predict_ms: float
    step_ms: float
    read: Callable[[object], object]


class RunHooks:
    """Observer callbacks for :meth:`Session.run`; every default is a no-op.

    Subclass and override any subset, then pass an instance as ``hooks=`` to
    :func:`rlmesh.run`, :meth:`Model.run <rlmesh.Model.run>`, or
    :meth:`Session.run`. Hook exceptions propagate and abort the run;
    :meth:`on_run_end` still fires exactly once with the completed episodes.
    """

    def on_run_start(self, session: Session[Any, Any]) -> None:
        """Called once per ``run``, after the session connects (no-op by default).

        Receives the running :class:`Session` so a hook can inspect the
        connected env (e.g. :meth:`Session.read` items, declared image roles)
        without the caller wiring the session into the hook by hand. Fires
        before the first episode's reset.
        """

    def on_episode_start(self, *, episode: int, seed: int | None) -> None:
        """Called after each episode's reset returns (no-op by default).

        Args:
            episode: 0-based episode index (equals :attr:`EpisodeResult.index`).
            seed: The seed the episode was reset with, or ``None``.
        """

    def on_step(self, event: StepEvent) -> None:
        """Called after each env step with its :class:`StepEvent` (no-op by default)."""

    def on_episode_end(self, result: EpisodeResult) -> None:
        """Called with each completed episode's :class:`EpisodeResult` (no-op by default)."""

    def on_run_end(self, result: RunResult) -> None:
        """Called exactly once when the run ends (no-op by default).

        Fires even on an exception or interrupt, with the possibly-partial
        :class:`RunResult` of the episodes that completed.
        """


def _summarize_payload(payload: Any) -> str:
    """One value's dtype/shape signature for error context.

    Mirrors the served engine's wording: ``float32[8]``,
    ``{image: uint8[8, 8, 3], state: float32[7]}``.
    """
    shape = getattr(payload, "shape", None)
    dtype = getattr(payload, "dtype", None)
    if shape is not None and dtype is not None:
        return f"{dtype}{list(shape)}"
    if isinstance(payload, Mapping):
        items = cast("Mapping[Any, Any]", payload).items()
        entries = ", ".join(
            f"{key}: {_summarize_payload(value)}" for key, value in items
        )
        return "{" + entries + "}"
    if isinstance(payload, str):
        return f"text(len={len(payload)})"
    if isinstance(payload, bytes):
        return f"bytes(len={len(payload)})"
    if isinstance(payload, (list, tuple)):
        seq = cast("Sequence[Any]", payload)
        if not seq:
            return "list(len=0)"
        return f"list(len={len(seq)}, first={_summarize_payload(seq[0])})"
    return type(payload).__name__


def _predict_step(
    predict: Callable[..., Any],
    obs: Any,
    adapter: Any,
    instruction: str | None,
    text_placements: tuple[TextPlacement, ...],
    env_bridge: ValueBridge | None,
    model_bridge: ValueBridge | None,
    device: object | None,
    context: Mapping[str, Any] | None = None,
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
    try:
        return _call_predict_with_optional_context(predict, payload, context)
    except Exception as exc:
        kind = "adapter-assembled" if adapter is not None else "spec-less (raw obs)"
        note = f"{kind} model input: {_summarize_payload(payload)}"
        add_note = getattr(exc, "add_note", None)
        if add_note is not None:
            add_note(note)
        raise


def _accepts_predict_context(predict: Callable[..., Any]) -> bool:
    """Whether ``predict`` can accept an optional second positional context."""
    try:
        signature = inspect.signature(predict)
    except (TypeError, ValueError):
        return False

    positional = 0
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return True
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional += 1
    return positional >= 2


def _call_predict_with_optional_context(
    predict: Callable[..., Any],
    payload: Any,
    context: Mapping[str, Any] | None,
) -> Any:
    """Call ``predict(payload)`` or ``predict(payload, context)`` compatibly."""
    if context is not None and _accepts_predict_context(predict):
        return predict(payload, context)
    return predict(payload)


class Session(Generic[ObsT, ActT]):
    """A model bound to one env: drive it by hand, or pump whole episodes.

    The neutral pair-driver returned by :func:`rlmesh.session`. ``reset`` / ``predict`` /
    ``step`` drive one step at a time -- ``predict`` applies the model's adapter (resolved
    from the env's published contract) around the model's own predict, replaying an action
    chunk one action per step when ``execution_horizon`` > 1 and the model defines
    ``predict_chunk``. ``run`` pumps whole episodes and returns a typed :class:`RunResult`.

    The env connection is opened lazily on first ``reset`` (manual driving); ``run``
    drives whole episodes through the same primitives and leaves the session open, so
    a caller-held session runs as often as you like. Close it yourself -- ``close()``
    or the ``with`` block; after that, any further use raises. (The one-shot
    :func:`rlmesh.run` / :meth:`Model.run <rlmesh.Model.run>` create their session
    internally and close it when the run ends.)
    """

    #: Instance attributes, declared here because :meth:`_create` (a classmethod)
    #: populates a bare ``object.__new__`` instance -- the public ``__init__`` only
    #: rejects direct construction. ``_device`` is the compute device for the local
    #: model's inputs (torch/jax); obs tensor leaves are moved onto it before
    #: predict (a served model leaves it unset -- the server worker places obs).
    _predict: Callable[[Any], Any] | None
    _predict_chunk: Callable[..., Any] | None
    _execution_horizon: int
    _spec: object | None
    _env: object
    _on_episode_end: Callable[[], None] | None
    _on_close: Callable[[], None] | None
    _trust: bool
    _bridge: ValueBridge | None
    _remote_env_cls: type | None
    _instruction: str | None
    _close_env: bool
    _model_client: PyModelClient | None
    _owner: Any
    _device: object | None
    _closed: bool
    _connected: bool
    _client: Any
    _owns_client: bool
    _adapter: Any
    _contract: Any
    _env_bridge: ValueBridge | None
    _text_placements: tuple[TextPlacement, ...]
    _horizon: int
    _replay: ChunkReplay
    _terminated: bool
    _truncated: bool
    _steps: int
    _reward: float
    _last_info: Mapping[str, Any]
    _model_ms: float
    _env_ms: float
    _sps: float
    _ep_start: float | None
    _last_step_t: float | None
    _ep_index: int
    _ep_total: int
    _seed: int | None
    _chunk_pos: int
    _chunk_len: int
    _episode_open: bool
    _read_cache: dict[Any, Reader]
    _view_driver: ViewerDriver | None

    def __init__(self) -> None:
        raise TypeError(
            "Session is not constructed directly; use rlmesh.session(model, env) "
            "or model.session(env)"
        )

    @classmethod
    def _create(
        cls,
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
        execution_horizon: int = 1,
        model_client: PyModelClient | None = None,
        owner: Any = None,
        device: object | None = None,
        view: object = None,
    ) -> Session[Any, Any]:
        """Build and populate a session (two modes: local predict+spec, or served model_client).

        The single construction seam behind :func:`rlmesh.session`,
        :meth:`Model.session`, and the served handles' ``session``; the public
        ``__init__`` only rejects direct construction.
        """
        if execution_horizon < 1:
            raise ValueError(f"execution_horizon must be >= 1, got {execution_horizon}")
        self = object.__new__(cls)
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
        self._model_client = model_client
        self._owner = owner
        self._device = device
        self._closed = False
        self._connected = False
        self._client = None
        self._owns_client = False
        self._adapter = None
        self._contract = None
        self._env_bridge = None
        self._text_placements = ()
        self._horizon = 1
        self._replay = ChunkReplay(1)
        self._terminated = False
        self._truncated = False
        self._steps = 0
        self._reward = 0.0
        self._last_info = {}
        #: Debug-viewer telemetry (smoothed), fed to the HUD each step. ``model_ms``
        #: tracks the forward cost only when the model actually runs (so chunk-replay
        #: steps don't read as 0ms); ``env_ms`` the simulator ``step``; ``sps`` the
        #: realized throughput. Episode index/total/seed are set by ``run``; chunk
        #: pos/len mirror the local replay queue. All inert when no viewer is attached.
        self._model_ms = 0.0
        self._env_ms = 0.0
        self._sps = 0.0
        self._ep_start = None
        self._last_step_t = None
        self._ep_index = 0
        self._ep_total = 0
        self._seed = None
        self._chunk_pos = 0
        self._chunk_len = 0
        #: Whether an episode has been reset and not yet ended. The local episode
        #: boundary (a stateful model's `on_episode_end`) fires when the next reset()
        #: begins or the session closes -- see `_end_episode`.
        self._episode_open = False
        #: Resolved Readers (sess.read), cached per read item so a per-step read
        #: does not re-resolve.
        self._read_cache = {}
        #: Optional built-in debug viewer (``view=`` on run/session). Lazily builds
        #: a native ``PyViewer`` on the first fed frame; best-effort, never fatal.
        _view = resolve_view(view)
        self._view_driver = ViewerDriver(_view) if _view is not None else None
        return self

    def _require_open(self) -> None:
        """Reject any use of an explicitly closed session (no silent reconnect)."""
        if self._closed:
            raise RuntimeError("session is closed")

    def _ensure_connected(self) -> None:
        self._require_open()
        if self._connected:
            return
        client, contract, owns = connect_env(self._env, self._remote_env_cls)
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
            # spec. It engages only when the model exposes a predict_chunk corner
            # (the entry points warn about the un-chunked fallback at creation);
            # without one, fall back to single-step predict.
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
            elapsed = (
                time.perf_counter() - self._ep_start
                if self._ep_start is not None
                else 0.0
            )
            self._view_driver.feed(
                contract=self._contract,
                client=self._client,
                obs=obs,
                read=self.read,
                steps=self._steps,
                reward=self._reward,
                outcome=self._view_outcome(),
                model_ms=self._model_ms,
                env_ms=self._env_ms,
                sps=self._sps,
                elapsed_s=elapsed,
                episode=self._ep_index,
                episodes=self._ep_total,
                seed=self._seed,
                chunk_pos=self._chunk_pos,
                chunk_len=self._chunk_len,
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
        # Viewer telemetry: start the episode clock, drop the inter-step timer (so the
        # first step's sps isn't computed off the reset gap), and record the seed.
        self._ep_start = time.perf_counter()
        self._last_step_t = None
        self._seed = seed
        self._feed_view(obs)
        return cast("ObsT", obs), info

    def predict(self, observation: ObsT) -> ActT:
        """Map one env observation to an env-ready action (the model's adapter applied)."""
        self._ensure_connected()
        if self._predict is RANDOM_SAMPLE:
            t0 = time.perf_counter()
            action = self._client.action_space.sample()
            self._model_ms = _ema(self._model_ms, (time.perf_counter() - t0) * 1000.0)
            return cast("ActT", action)
        if self._model_client is not None:
            # Served model: the server applies the adapter (and any chunk replay);
            # we only bridge the obs out and the env-ready action back. The timed span
            # is the whole served round-trip -- the meaningful "model" cost host-side.
            bridge = self._bridge if self._bridge is not None else identity_bridge
            t0 = time.perf_counter()
            action = self._model_client.predict(bridge.encode(observation))
            self._model_ms = _ema(self._model_ms, (time.perf_counter() - t0) * 1000.0)
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

            def _replay(obs: Any, context: Mapping[str, Any] | None = None) -> Any:
                replay_context = (
                    {"execution_horizon": horizon, **context}
                    if context is not None
                    else {"execution_horizon": horizon}
                )
                return _call_predict_with_optional_context(
                    chunk_fn, obs, replay_context
                )

            replay_fn = cast("Callable[[Any], Any]", _replay)
        else:
            replay_fn = cast("Callable[[Any], Any]", self._predict)

        # Time the forward inside the replay thunk so model_ms records only on steps
        # that re-plan (the thunk is skipped while a chunk replays), keeping it a true
        # forward cost rather than a near-zero queue pop.
        def _forward() -> Any:
            t0 = time.perf_counter()
            predict_context = {
                "episode_index": self._ep_index,
                "episode_id": (
                    self._last_info.get("episode_ids", [""])[0]
                    if isinstance(self._last_info, Mapping)
                    and self._last_info.get("episode_ids")
                    else ""
                ),
                "episode_ids": (
                    list(self._last_info.get("episode_ids", []))
                    if isinstance(self._last_info, Mapping)
                    else []
                ),
                "episode_seed": self._seed,
                "episode_seeds": [self._seed],
                "step": self._steps,
                "num_envs": 1,
            }
            out = _predict_step(
                replay_fn,
                observation,
                self._adapter,
                self._instruction,
                self._text_placements,
                self._env_bridge,
                model_bridge,
                self._device,
                predict_context,
            )
            self._model_ms = _ema(self._model_ms, (time.perf_counter() - t0) * 1000.0)
            return out

        raw_action = self._replay.next_action(_forward)
        # Mirror the local chunk-replay position for the HUD (1-based; 0/0 = not
        # chunking). The length is the chunk the model actually returned (post-cap),
        # not the requested horizon, so a short native chunk displays truthfully.
        if self._horizon > 1:
            self._chunk_len = self._replay.last_chunk_len
            self._chunk_pos = self._chunk_len - self._replay.pending
        else:
            self._chunk_len = self._chunk_pos = 0
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
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = self._client.step(action)
        now = time.perf_counter()
        # env_ms = the simulator step alone; sps = realized throughput from the gap
        # between consecutive steps (model + env + overhead), skipping the first step
        # of an episode (no prior timestamp).
        self._env_ms = _ema(self._env_ms, (now - t0) * 1000.0)
        if self._last_step_t is not None:
            dt = now - self._last_step_t
            if dt > 0.0:
                self._sps = _ema(self._sps, 1.0 / dt)
        self._last_step_t = now
        self._reward += float(reward)
        self._steps += 1
        self._terminated = bool(terminated)
        self._truncated = bool(truncated)
        self._last_info = info
        self._feed_view(obs)
        # Viewer "skip episode" (the `n` key / `/skip` button): a soft, non-failure end
        # -- like a time-limit truncation -- that advances to the next episode, distinct
        # from quit (which stops the whole run early with a partial result). Forcing the
        # returned `truncated` true ends the episode in any loop that reads this tuple
        # (the catalog's record.run) as well as ones reading `self.done` (Session.run).
        # Quit (`q` / Esc) also truncates the current episode so a hand-driven loop
        # ends; Session.run additionally stops iterating episodes (see run()).
        if self._view_driver is not None and (
            self._view_driver.consume_skip() or self._view_driver.quit_requested()
        ):
            truncated = True
            self._truncated = True
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

    def read(self, observation: object, item: object) -> Any:
        """One-shot read of a single role from one observation.

        The single-value convenience for :meth:`reader` -- ``item`` is a role
        constant or a model-input leaf. The reader is resolved once and cached per
        item, so calling this every step does not re-resolve::

            ee = sess.read(obs, EEF_POS)
            img = sess.read(obs, Image(IMAGE_PRIMARY, layout="hwc"))

        The value is typed ``Any`` (like :class:`Reader`'s): its concrete shape is
        the leaf's declared encoding, which the caller owns.
        """
        self._require_open()
        reader = self._read_cache.get(item)
        if reader is None:
            reader = self.reader(item)
            self._read_cache[item] = reader
        return reader(observation)[reader.roles[0]]

    def observation_roles(self) -> ObservationRoles:
        """The observation roles this session's env declares, grouped by kind.

        Connects if needed, reads the env contract's published tags, and returns
        their :attr:`~rlmesh.adapters.EnvTags.observation_roles`. An env that
        publishes no tags yields empty groups -- "none declared" is an answer,
        not an error.
        """
        from ..adapters import EnvTags
        from ..adapters import ObservationRoles as _ObservationRoles

        self._ensure_connected()
        env_tags = EnvTags.from_metadata(
            getattr(self._contract, "metadata", None) or {}
        )
        if env_tags is None:
            return _ObservationRoles()
        return env_tags.observation_roles

    def run(
        self,
        *,
        seeds: Sequence[int] | None = None,
        max_episodes: int | None = None,
        max_episode_steps: int | None = None,
        max_episode_seconds: float | None = None,
        hooks: RunHooks | None = None,
    ) -> RunResult:
        """Drive whole episodes to completion and return a typed :class:`RunResult`.

        The single drive loop: pumps this session's own ``reset`` / ``predict`` /
        ``step`` primitives, so ``Model.run`` routes through here.
        ``seeds`` gives a per-episode seed and sets the episode count unless
        ``max_episodes`` is given. ``max_episode_steps`` / ``max_episode_seconds``
        cap each episode -- hitting a cap marks it ``truncated``, exactly like the
        built-in step bound (the wall-clock cap is checked at the top of the step
        loop). ``hooks`` (:class:`RunHooks`) observes the loop; hook exceptions
        propagate and abort the run, and :meth:`RunHooks.on_run_end` always fires
        exactly once with the completed episodes, even on an error or interrupt.

        Does **not** close the session: a caller-held session (from
        :func:`rlmesh.session` / :meth:`Model.session`) stays connected -- viewer
        and hooks included -- so ``run`` can be called again or mixed with manual
        driving; close it via :meth:`close` or the ``with`` block. (The one-shot
        :func:`rlmesh.run` / :meth:`Model.run` own their internal session and
        close it for you.)
        """
        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError(f"max_episode_steps must be >= 1, got {max_episode_steps}")
        if max_episode_seconds is not None and max_episode_seconds <= 0:
            raise ValueError(
                f"max_episode_seconds must be > 0, got {max_episode_seconds}"
            )
        self._ensure_connected()
        if max_episodes is not None:
            n_episodes = max_episodes
        elif seeds is not None:
            n_episodes = len(seeds)
        else:
            n_episodes = 1
        step_cap = (
            max_episode_steps
            if max_episode_steps is not None
            else _MAX_STEPS_PER_EPISODE
        )
        episodes: list[EpisodeResult] = []
        run_end_error: BaseException | None = None
        self._ep_total = n_episodes
        try:
            if hooks is not None:
                hooks.on_run_start(self)
            for i in range(n_episodes):
                self._ep_index = i + 1
                seed = seeds[i] if seeds is not None and i < len(seeds) else None
                obs, last_info = self.reset(seed=seed)
                ep_start = time.perf_counter()
                if hooks is not None:
                    hooks.on_episode_start(episode=i, seed=seed)
                predict_total_ms = 0.0
                step_total_ms = 0.0
                while not self.done and self._steps < step_cap:
                    if (
                        max_episode_seconds is not None
                        and time.perf_counter() - ep_start >= max_episode_seconds
                    ):
                        break
                    step_index = self._steps
                    prev_obs = obs
                    t0 = time.perf_counter()
                    action = self.predict(prev_obs)
                    t1 = time.perf_counter()
                    obs, reward, terminated, truncated, last_info = self.step(action)
                    t2 = time.perf_counter()
                    predict_ms = (t1 - t0) * 1000.0
                    step_ms = (t2 - t1) * 1000.0
                    predict_total_ms += predict_ms
                    step_total_ms += step_ms
                    if hooks is not None:
                        hooks.on_step(
                            StepEvent(
                                episode=i,
                                seed=seed,
                                step=step_index,
                                observation=prev_obs,
                                action=action,
                                reward=reward,
                                terminated=terminated,
                                truncated=truncated,
                                info=last_info,
                                predict_ms=predict_ms,
                                step_ms=step_ms,
                                read=partial(self.read, prev_obs),
                            )
                        )
                if not self._terminated and not self._truncated:
                    self._truncated = True
                steps = self._steps
                episode = EpisodeResult(
                    index=i,
                    seed=seed,
                    steps=steps,
                    reward=self._reward,
                    terminated=self._terminated,
                    truncated=self._truncated,
                    success=_episode_success(last_info),
                    duration_s=time.perf_counter() - ep_start,
                    predict_ms=predict_total_ms / steps if steps else 0.0,
                    step_ms=step_total_ms / steps if steps else 0.0,
                )
                episodes.append(episode)
                if hooks is not None:
                    hooks.on_episode_end(episode)
                # Viewer quit (`q` / Esc): stop early and return the partial result
                # -- the just-recorded (truncated) episode is the last one. A real
                # Ctrl-C still raises KeyboardInterrupt out of the loop above.
                if self._view_driver is not None and self._view_driver.quit_requested():
                    break
        finally:
            if hooks is not None:
                try:
                    hooks.on_run_end(RunResult(episodes=tuple(episodes)))
                except BaseException as exc:
                    run_end_error = exc
            self._end_episode()
        if run_end_error is not None:
            raise run_end_error
        return RunResult(episodes=tuple(episodes))

    def close(self) -> None:
        """Close this session: model close hook, served route (and owned source), env.

        For a served model, closes the model client and shuts down a managed source it
        started (e.g. a ``SandboxModel`` container). For the env, shuts it down only on
        the ``close_env`` opt-in and closes a connection this session dialed.

        Idempotent: the first call tears everything down (firing a local model's
        ``on_close`` exactly once, whether the session was pumped via ``run`` or
        driven by hand); later calls are no-ops, and any other use of a closed
        session raises ``RuntimeError``.
        """
        if self._closed:
            return
        self._closed = True
        # End an episode left open by a hand-driven loop, so a stateful model's
        # `on_episode_end` fires for the last episode even without a following reset().
        if self._view_driver is not None:
            self._view_driver.close()
        self._end_episode()
        on_close_error: BaseException | None = None
        if self._on_close is not None:
            try:
                self._on_close()
            except BaseException as exc:
                on_close_error = exc
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
        if on_close_error is not None:
            raise on_close_error

    def __enter__(self) -> Session[ObsT, ActT]:
        return self

    def __exit__(self, *exc: object) -> None:
        _ = exc
        self.close()
