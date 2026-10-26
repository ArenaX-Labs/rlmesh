"""A run's results and hooks, shared by every framework ``Model``.

The Rust facade drives both runs and manual bindings. Python converts its
callbacks and values into :class:`RunHooks` and typed :class:`RunResult`.
"""

from __future__ import annotations

import contextlib
import numbers
import os
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

from ._coerce import RANDOM_SAMPLE
from ._connect import (
    adapter_env_bridge,
)
from ._read import Reader, resolve_read_adapter
from ._resolve import resolve_adapter
from ._view import FrameSources, discover_frame_sources

if TYPE_CHECKING:
    from rlmesh._rlmesh import Advisory


__all__ = [
    "RANDOM_SAMPLE",
    "EpisodeResult",
    "EvalReport",
    "Interrupted",
    "RunHooks",
    "RunResult",
    "StepEvent",
    "TelemetryRow",
    "resolve_adapter",
]

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


# Mirrors `EditionDefaults::success_info_keys` in crates/rlmesh-proto; keep in sync.
_SUCCESS_INFO_KEYS = ("is_success", "success", "task_success")


def _success_flag(value: object) -> bool | None:
    """One info value as a task outcome, or ``None`` when its kind carries none.

    Matches the native runtime (`success_from_final_info` in
    crates/rlmesh/src/run/runtime `driver.rs`) over the value as it crosses the wire:
    NumPy scalars and 0-d arrays unwrap to Python scalars first (as metadata
    encoding does), then a bool is taken as-is and an integer or float is
    true when non-zero. Any other kind -- a string, ``None``, a list -- is
    skipped, so the next key decides.
    """
    for unwrap in ("tolist", "item"):
        method: object = getattr(value, unwrap, None)
        if callable(method):
            value = method()
            break
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value) != 0
    if isinstance(value, numbers.Real):
        return float(value) != 0.0
    return None


def episode_success(info: Mapping[str, Any]) -> bool | None:
    """Read an env-reported task outcome from a step ``info`` (Gymnasium convention).

    Checks ``is_success``, ``success``, then ``task_success``; the first one
    holding a bool, integer, or float decides (numbers by non-zero). Returns
    ``None`` when none does: the outcome is unknown, never inferred from
    ``terminated``.
    """
    for key in _SUCCESS_INFO_KEYS:
        if key in info and (flag := _success_flag(info[key])) is not None:
            return flag
    return None


def episode_outcome(terminated: bool, truncated: bool, info: Mapping[str, Any]) -> str:
    """The viewer's word for how a step ended its episode ("" while it runs)."""
    if not (terminated or truncated):
        return ""
    success = episode_success(info)
    if success is None:
        return "done" if terminated else "timeout"
    if success:
        return "success"
    return "failure" if terminated else "timeout"


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
            (``is_success``, ``success``, then ``task_success``; the first
            holding a bool or number decides), or ``None`` when the env emits
            no such signal. Distinct from ``terminated`` (which only
            says the episode reached a terminal state, not how it ended); an
            unknown outcome is never inferred from it.
        duration_s: Wall time from reset-return to episode end, in seconds.
        predict_ms: This episode's mean per-step wall time of ``predict``, in
            milliseconds -- the mean of its :attr:`StepEvent.predict_ms`
            values, so on the native loop a chunk's predict is spread over the
            steps it served. ``None`` when no step was timed (an episode that
            ended before its first step); never a run-wide average.
        step_ms: This episode's mean per-step wall time of the env ``step``
            round trip, in milliseconds, or ``None`` on the same terms.
        trial: The trial ordinal the episode walked, ``trial_index_base + index``
            on ``run()``. Delivered as ``reset(options={"trial_index": ...})``
            only to an env that declared the option, but recorded either way so
            the sweep can be read off the result. ``None`` for a hand-driven
            :meth:`Session.reset` that passed no ``trial_index``, and for an env
            that owns its resets (``NEXT_STEP`` autoreset on the native loop).
    """

    index: int
    seed: int | None
    steps: int
    reward: float
    terminated: bool
    truncated: bool
    success: bool | None = None
    duration_s: float = 0.0
    predict_ms: float | None = None
    step_ms: float | None = None
    trial: int | None = None


@dataclass(frozen=True)
class TelemetryRow:
    """One aggregated metric series from a :meth:`Model.run` eval.

    The runtime aggregates every measurement per ``(op, component, metric)``
    over the whole run. ``rpc.total`` is the client-observed round trip of an
    op; ``endpoint.total`` the peer's own handling wall, split by
    ``endpoint.decode`` / ``endpoint.user`` / ``endpoint.encode``;
    ``endpoint.queue`` the wait before the op's handler ran (the model's
    permit, gate, and handler lock; the env's lock) and ``predict.in_flight``
    the model endpoint's slot occupancy at dispatch; ``predict.adapter`` the RLMesh
    adapter work (observation assembly + action apply) inside the handler's
    own time, so the model's own forward is ``endpoint.user`` minus it;
    ``held.episodes`` / ``held.bytes`` the per-episode frame-stack state the
    adapter engine was holding when the predict was stamped (what grows with
    concurrent episodes and shrinks on episode-end GC; a zero is a real sample);
    ``lane.skew`` a vector env's straggler dispersion, only from an env that
    times its own lanes; ``runner.round`` one full
    predict -> step -> transform loop iteration (subtract the per-op rows for
    the driver's own residual). A metric a peer never stamps produces no row.

    Attributes:
        op: The measured operation (e.g. ``model.predict``, ``env.step``).
        component: Who produced the sample (``model``, ``env``, ``runner``).
        metric: The metric name (e.g. ``rpc.total``, ``endpoint.user``).
        unit: What the values measure: ``ms``, ``bytes``, or ``count``.
        count: Samples aggregated into this row.
        avg: Mean value, in :attr:`unit`.
        p50: Median value, in :attr:`unit`.
        p95: 95th-percentile value, in :attr:`unit`.
        p99: 99th-percentile value, in :attr:`unit`.
    """

    op: str
    component: str
    metric: str
    unit: str
    count: int
    avg: float
    p50: float
    p95: float
    p99: float


@dataclass(frozen=True)
class RunResult:
    """The result of a :meth:`Model.run` eval."""

    episodes: tuple[EpisodeResult, ...] = ()
    #: The run's aggregated timing/size telemetry, one :class:`TelemetryRow`
    #: per measured series: every series the run's peers stamp. A binding's
    #: result carries none.
    telemetry: tuple[TelemetryRow, ...] = ()
    #: Each :class:`~rlmesh.adapters.Advisory` the model's adapter resolve
    #: raised for the env: what it tolerated or applied, such as a zero-filled
    #: camera or an allowed projection (``select_state``). A served model sends
    #: them to a driver that reads them. Empty for a spec-less model.
    advisories: tuple[Advisory, ...] = ()

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
    def success_rate(self) -> float | None:
        """Fraction of episodes the env reported as a success, or ``None``.

        Counts the env-reported task outcome only (the final step's
        ``info["is_success"]`` / ``["success"]`` / ``["task_success"]``,
        captured per episode in :attr:`EpisodeResult.success`).
        ``None`` when the run is empty or any episode lacks that signal: an
        unknown outcome is never inferred from ``terminated``. Read
        :attr:`EpisodeResult.terminated` yourself if a terminal state is the
        metric you want.
        """
        if not self.episodes or any(e.success is None for e in self.episodes):
            return None
        return sum(1 for e in self.episodes if e.success) / len(self.episodes)

    def format_telemetry(self) -> str:
        """The :attr:`telemetry` rows as an aligned text table.

        One line per row -- op, metric, unit, count, avg/p50/p95/p99 -- so
        ``print(result.format_telemetry())`` answers where a run's time went
        (model forward, env step, serialization, queueing) without a profiler.
        """
        if not self.telemetry:
            return "(no telemetry recorded)"
        header = ("op", "metric", "unit", "count", "avg", "p50", "p95", "p99")
        rows = [
            (
                r.op,
                r.metric,
                r.unit,
                str(r.count),
                f"{r.avg:.2f}",
                f"{r.p50:.2f}",
                f"{r.p95:.2f}",
                f"{r.p99:.2f}",
            )
            for r in self.telemetry
        ]
        widths = [
            max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(header)
        ]

        def line(cells: tuple[str, ...]) -> str:
            return "  ".join(
                cell.ljust(width) if i < 3 else cell.rjust(width)
                for i, (cell, width) in enumerate(zip(cells, widths, strict=True))
            )

        return "\n".join([line(header), *(line(row) for row in rows)])

    def save(
        self,
        path: str | os.PathLike[str],
        *,
        model: str = "model",
        env: str = "env",
        config: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write this result as an ``rlmesh.result.v1`` bundle and return its path.

        One workload of ``model`` on ``env``, with ``config`` recorded beside
        it: a folder, or a zip when ``path`` ends ``.zip`` (as
        :meth:`Recorder.export <rlmesh.Recorder.export>` writes). The bundle
        keeps each episode's outcome, not its trial ordinal or latency; the
        run's telemetry rows are kept beside the workload. For media, record
        the run with :meth:`Recorder.capture <rlmesh.Recorder.capture>` instead.
        """
        from ..recorder import Recorder

        with Recorder() as recorder:
            recorder.add(
                self, model=model, env=env, config=dict(config) if config else None
            )
            return recorder.export(Path(path))

    @classmethod
    def load(
        cls, path: str | os.PathLike[str] | Sequence[str | os.PathLike[str]]
    ) -> RunResult:
        """The result the ``rlmesh.result.v1`` bundles at ``path`` hold.

        ``path`` is a bundle (folder or zip) of one or several workloads, a
        folder of bundles (read in name order), or a list of these. The
        workloads' episodes come back in order with the fields the bundle
        keeps, indexed afresh from 0, and their telemetry merged per series
        (counts add; ``avg`` and percentiles are count-weighted means).
        Workloads of several models raise ``ValueError``; report them with
        :class:`EvalReport` instead.
        """
        from .._rlmesh import load_run_result
        from ._facade import result_from_native

        return result_from_native(load_run_result(_bundle_paths(path)))

    def __repr__(self) -> str:
        return (
            f"RunResult(episodes={self.num_episodes}, mean_reward={self.mean_reward:.3f}, "
            f"total_steps={self.total_steps})"
        )


def _bundle_paths(
    path: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
) -> list[str]:
    paths = [path] if isinstance(path, (str, os.PathLike)) else path
    return [os.fspath(p) for p in paths]


class EvalReport:
    """Success and telemetry of saved ``rlmesh.result.v1`` bundles, per model.

    Experimental. Per task, per suite (a workload's ``config["suite"]``) and
    overall: counted episodes, successes, the success rate with its Wilson
    95% interval, steps, mean episode length and wall time; and the runs'
    telemetry merged per series. An episode with no success signal counts as
    a failure and is reported as ``unknown``, as on the managed platform.
    """

    def __init__(self, document: dict[str, Any], table: str) -> None:
        self._document = document
        self._table = table

    @classmethod
    def load(
        cls, path: str | os.PathLike[str] | Sequence[str | os.PathLike[str]]
    ) -> EvalReport:
        """The report of the bundles at ``path`` (as :meth:`RunResult.load` reads them)."""
        import json

        from .._rlmesh import eval_report

        document, table = eval_report(_bundle_paths(path))
        return cls(json.loads(document), table)

    def to_dict(self) -> dict[str, Any]:
        """The report as a JSON-native dict: ``{"models": [...]}``."""
        import copy

        return copy.deepcopy(self._document)

    def format_table(self) -> str:
        """The report as aligned text tables, one per model."""
        return self._table

    def __str__(self) -> str:
        return self._table


class Interrupted(KeyboardInterrupt):
    """Ctrl-C during :meth:`Model.run <rlmesh.Model.run>`, with what it finished.

    A ``KeyboardInterrupt``, so code that stops on Ctrl-C still stops; catch it
    to keep :attr:`result`, the :class:`RunResult` of the episodes that
    completed before the interrupt (the one in progress is not in it).
    """

    def __init__(self, result: RunResult) -> None:
        super().__init__(
            f"run interrupted after {result.num_episodes} completed episode(s)"
        )
        #: The episodes that completed before the interrupt.
        self.result = result


@dataclass(frozen=True)
class StepEvent:
    """One env step of a run, passed to :meth:`RunHooks.on_step`.

    The same event on both loops, with one timing definition: ``step_ms`` is
    the wall time of this step's env round trip, and ``predict_ms`` the wall
    time of the model predict(s) that produced an action since the previous
    step -- the session loop's synchronous ``predict``; on the native loop the
    runtime's own measurement, which is zero for a step served from chunk
    replay (``execution_horizon > 1``: the chunk's predict lands on the step
    that first consumed it) and, with ``prefetch_lead > 0``, the next chunk's
    predict on the step it landed on. On a vectorized env every lane of a
    group shares both numbers. The per-episode means of these values are
    :attr:`EpisodeResult.predict_ms` / :attr:`EpisodeResult.step_ms`. A
    ``NEXT_STEP`` autoreset roll is not a step of any episode and never
    surfaces here.

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
        predict_ms: Wall time of the predict(s) charged to this step, in
            milliseconds (see above); zero on a chunk-replay step.
        step_ms: Wall time of the env ``step`` round trip, in milliseconds.
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


class RunContext(Protocol):
    """What a run exposes to its hooks: the connected env.

    :meth:`RunHooks.on_run_start` receives one: :meth:`Model.run
    <rlmesh.Model.run>` passes its own, and a :class:`~rlmesh.Binding` is one.
    ``contract`` and ``num_envs`` are known once the env is connected (before
    the first :meth:`RunHooks.on_episode_start`); ``read`` resolves a role over
    one observation, and
    ``frame_sources`` lists the env's image roles plus its ``render()`` thunk
    when one is reachable (a native run reaches it only for a local env object).
    """

    @property
    def contract(self) -> Any: ...

    @property
    def num_envs(self) -> int: ...

    def read(self, observation: object, item: object) -> Any: ...

    def frame_sources(self) -> FrameSources: ...


class RunHooks:
    """Observer callbacks for a run; every default is a no-op.

    Subclass and override any subset, then pass an instance (or a list of them,
    each called in order) as ``hooks=`` to :func:`rlmesh.run`,
    :meth:`Model.run <rlmesh.Model.run>`, or
    :meth:`Model.bind <rlmesh.Model.bind>`. A run and a binding fire the same
    callbacks in the same per-episode order. Hook exceptions propagate and abort the run with their own
    type; :meth:`on_run_end` still fires exactly once with the completed
    episodes. Hooks run inline, so a slow hook slows the loop.
    """

    def on_run_start(self, context: RunContext) -> None:
        """Called once per ``run``, before the first episode (no-op by default).

        Receives the run's :class:`RunContext` (the :class:`~rlmesh.Binding`
        itself on a binding) so a hook can read roles off observations and
        discover the env's frame sources without the caller wiring anything in.
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


HooksArg = RunHooks | Sequence[RunHooks] | None


class _HookList(RunHooks):
    """Several hooks as one: every callback reaches each hook, in order.

    ``on_run_end`` reaches every hook even when one raises; the first exception
    is raised after the last hook ran, so each still sees the run end once.
    """

    def __init__(self, hooks: Sequence[RunHooks]) -> None:
        self._hooks = tuple(hooks)

    def on_run_start(self, context: RunContext) -> None:
        for hook in self._hooks:
            hook.on_run_start(context)

    def on_episode_start(self, *, episode: int, seed: int | None) -> None:
        for hook in self._hooks:
            hook.on_episode_start(episode=episode, seed=seed)

    def on_step(self, event: StepEvent) -> None:
        for hook in self._hooks:
            hook.on_step(event)

    def on_episode_end(self, result: EpisodeResult) -> None:
        for hook in self._hooks:
            hook.on_episode_end(result)

    def on_run_end(self, result: RunResult) -> None:
        error: BaseException | None = None
        for hook in self._hooks:
            try:
                hook.on_run_end(result)
            except BaseException as exc:  # re-raised after the last hook
                error = error or exc
        if error is not None:
            raise error


def as_hooks(hooks: HooksArg) -> RunHooks | None:
    """``hooks=`` as one :class:`RunHooks`: a list fans out, in order."""
    if hooks is None or isinstance(hooks, RunHooks):
        return hooks
    listed = tuple(hooks)
    if not listed:
        return None
    return listed[0] if len(listed) == 1 else _HookList(listed)


class NativeRunContext:
    """The :class:`RunContext` of a native :meth:`Model.run`.

    ``contract`` is filled when the runtime connects to the env (a local model's
    route resolve may refine it); ``client`` is the locally served env object
    (so ``render()`` is reachable while a hook runs, the env being idle between
    runtime events) or ``None`` for an address / handle target.
    """

    def __init__(self, *, trust: bool) -> None:
        self._trust = trust
        self._contract: Any = None
        self._core: Any = None
        self._client: object | None = None
        self._bridge = adapter_env_bridge(None)
        self._read_cache: dict[Any, Reader] = {}
        self._hook_thread: int | None = None
        self._rendering = False

    @property
    def contract(self) -> Any:
        return self._core.env_contract if self._core is not None else self._contract

    @contract.setter
    def contract(self, value: Any) -> None:
        self._contract = value

    @contextlib.contextmanager
    def _dispatch(self) -> Iterator[None]:
        previous = self._hook_thread
        self._hook_thread = threading.get_ident()
        try:
            yield
        finally:
            self._hook_thread = previous

    @property
    def client(self) -> object | None:
        return self._client

    def bind_client(self, client: object | None, *, local: bool = True) -> None:
        """Decode values in ``client``'s framework.

        ``local`` also reaches its ``render()``; a remote handle lent to the run
        is not reachable.
        """
        self._client = client if local else None
        self._bridge = adapter_env_bridge(client)

    @property
    def num_envs(self) -> int:
        return int(getattr(self.contract, "num_envs", 1) or 1)

    def _decode(self, value: object) -> Any:
        """A neutral runtime value in the env's own framework."""
        return self._bridge.decode(cast("Any", value))

    def read(self, observation: object, item: object) -> Any:
        reader = self._read_cache.get(item)
        if reader is None:
            adapter, roles = resolve_read_adapter(self.contract, (item,), self._trust)
            reader = Reader(adapter, roles, self._bridge)
            self._read_cache[item] = reader
        return reader(observation)[reader.roles[0]]

    def frame_sources(self) -> FrameSources:
        sources = discover_frame_sources(self.contract, self._client)
        raw = sources.render
        if raw is None or self._hook_thread != threading.get_ident() or self._rendering:
            return FrameSources(sources.roles, None, None)

        def render() -> object:
            if self._hook_thread != threading.get_ident() or self._rendering:
                raise RuntimeError("render is unreachable outside its hook dispatch")
            self._rendering = True
            try:
                return raw()
            finally:
                self._rendering = False

        return FrameSources(sources.roles, sources.render_label, render)
