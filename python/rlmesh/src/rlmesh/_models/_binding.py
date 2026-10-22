"""A model bound to one env, stepped by hand: what :meth:`Model.bind` returns."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Generic, cast

from .._value_conversion import ValueBridge
from ._eval import (
    ActT,
    EpisodeResult,
    ObsT,
    RunHooks,
    RunResult,
    StepEvent,
    episode_success,
)
from ._read import Reader, resolve_read_adapter
from ._view import FrameSources, ViewerDriver, discover_frame_sources, resolve_view

if TYPE_CHECKING:
    from rlmesh._rlmesh import PyBinding

__all__ = ["Binding"]


class Binding(Generic[ObsT, ActT]):
    """A model bound to one env, stepped by hand: :meth:`Model.bind` returns one.

    ``reset`` starts an episode; ``step()`` steps the env under the model's
    action (the model predicts, replaying a chunk across steps when the binding
    was made with ``execution_horizon`` > 1), and ``step(action)`` under yours.
    An episode's end evicts its state on the model, as a run does, and ``done``
    says it ended. A model's own prediction is ``model.predict(obs)``.

    Hooks and the viewer see each reset and step as a run's would: the binding
    is their :class:`RunContext`, an episode reset over before it ended counts
    as truncated, and ``on_run_end`` fires at close. Close the binding
    (``close()`` or the ``with`` block) to release the env; the model stays
    yours.
    """

    _native: PyBinding
    _bridge: ValueBridge
    _hooks: RunHooks | None
    _view: ViewerDriver | None
    _release: Callable[[], None]
    _closed: bool
    _failed: bool
    _episodes: list[EpisodeResult]
    _index: int
    _seed: int | None
    _steps: int
    _reward: float
    _done: bool
    _started: float
    _predict_ms: list[float]
    _step_ms: list[float]
    _info: Mapping[str, Any]
    _last_obs: Any
    _read_cache: dict[Any, Reader]

    def __init__(self) -> None:
        raise TypeError("Binding is not constructed directly; use model.bind(target)")

    @classmethod
    def _create(
        cls,
        *,
        native: PyBinding,
        bridge: ValueBridge,
        hooks: RunHooks | None,
        view: object,
        release: Callable[[], None],
    ) -> Binding[Any, Any]:
        self = object.__new__(cls)
        self._native = native
        self._bridge = bridge
        self._hooks = hooks
        resolved = resolve_view(view)
        self._view = ViewerDriver(resolved) if resolved is not None else None
        self._release = release
        self._closed = False
        #: A reset or step raised: closing skips the viewer's hold.
        self._failed = False
        self._episodes = []
        self._index = -1
        self._seed = None
        self._steps = 0
        self._reward = 0.0
        self._done = True
        self._started = 0.0
        self._predict_ms = []
        self._step_ms = []
        self._info = {}
        self._last_obs = None
        self._read_cache = {}
        if hooks is not None:
            hooks.on_run_start(self)
        return self

    # -- the run context --------------------------------------------------------

    @property
    def contract(self) -> Any:
        """The env's contract."""
        return self._native.env_contract

    @property
    def num_envs(self) -> int:
        """Lanes a binding drives: one."""
        return 1

    @property
    def render_mode(self) -> str:
        """The env's render mode."""
        return str(getattr(self.contract, "render_mode", "") or "")

    def render(self) -> object:
        """The env's rendered frame, or ``None``."""
        self._require_open()
        return self._native.render()

    def read(self, observation: object, item: object) -> Any:
        """One role (or model-input leaf) read off ``observation``."""
        reader = self._read_cache.get(item)
        if reader is None:
            adapter, roles = resolve_read_adapter(self.contract, (item,), False)
            reader = Reader(adapter, roles, self._bridge)
            self._read_cache[item] = reader
        return reader(observation)[reader.roles[0]]

    def frame_sources(self) -> FrameSources:
        """The env's image roles and ``render()`` source."""
        return discover_frame_sources(self.contract, self)

    # -- stepping ---------------------------------------------------------------

    @property
    def done(self) -> bool:
        """Whether the episode ended (or none started yet)."""
        return self._done

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> tuple[ObsT, Mapping[str, Any]]:
        """Start the next episode: its first observation and the reset's info.

        ``options`` reach the env's ``reset(options=...)`` over the model's own
        :meth:`~rlmesh.Model.reset_options`.
        """
        self._require_open()
        self._end_episode(truncated=True)
        with self._failing():
            observation, info = self._native.reset(
                seed, dict(options) if options else None
            )
        self._index += 1
        self._seed = seed
        self._steps = 0
        self._reward = 0.0
        self._done = False
        self._started = time.perf_counter()
        self._predict_ms = []
        self._step_ms = []
        self._info = info
        obs = cast(ObsT, self._bridge.decode(observation))
        self._last_obs = obs
        if self._hooks is not None:
            self._hooks.on_episode_start(episode=self._index, seed=seed)
        self._feed_view(obs)
        return obs, info

    def step(
        self, action: ActT | None = None
    ) -> tuple[ObsT, float, bool, bool, Mapping[str, Any]]:
        """Step the env under ``action``, or the model's when ``None``."""
        self._require_open()
        if self._done:
            raise RuntimeError("the episode ended; reset to start the next")
        given = None if action is None else self._bridge.encode(action)
        with self._failing():
            step = self._native.step(given)
        observation, applied, reward, terminated, truncated, info = step[:6]
        predict_ms, step_ms = step[6:]
        before = self._last_obs
        obs = cast(ObsT, self._bridge.decode(observation))
        self._last_obs = obs
        self._steps += 1
        self._reward += reward
        self._predict_ms.append(predict_ms)
        self._step_ms.append(step_ms)
        self._info = info
        self._feed_view(obs, outcome=self._outcome(terminated, truncated))
        # The viewer's skip (or quit) ends the episode as a truncation.
        if self._view is not None and (
            self._view.consume_skip() or self._view.quit_requested()
        ):
            truncated = True
        if self._hooks is not None:
            self._hooks.on_step(
                StepEvent(
                    episode=self._index,
                    seed=self._seed,
                    step=self._steps - 1,
                    observation=before,
                    action=self._bridge.decode(applied),
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    predict_ms=predict_ms,
                    step_ms=step_ms,
                    read=lambda item: self.read(before, item),
                )
            )
        if terminated or truncated:
            self._end_episode(terminated=terminated, truncated=truncated)
        if self._view is not None:
            self._view.pace()
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """End the episode, release the env, and fire ``on_run_end`` (idempotent)."""
        if self._closed:
            return
        self._closed = True
        try:
            self._end_episode(truncated=True)
            if self._view is not None:
                self._view.close(hold=not self._failed)
        finally:
            try:
                self._native.close()
            finally:
                try:
                    self._release()
                finally:
                    if self._hooks is not None:
                        self._hooks.on_run_end(
                            RunResult(episodes=tuple(self._episodes))
                        )

    def __enter__(self) -> Binding[ObsT, ActT]:
        return self

    def __exit__(self, *exc: object) -> None:
        self._failed = self._failed or exc[0] is not None
        self.close()

    # -- internals --------------------------------------------------------------

    @contextlib.contextmanager
    def _failing(self) -> Iterator[None]:
        try:
            yield
        except BaseException:
            self._failed = True
            raise

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("the binding is closed")

    def _end_episode(
        self, *, terminated: bool = False, truncated: bool = False
    ) -> None:
        """Record the episode in flight as ended, once."""
        if self._done:
            return
        self._done = True

        def mean(values: list[float]) -> float | None:
            return sum(values) / len(values) if values else None

        result = EpisodeResult(
            index=self._index,
            seed=self._seed,
            steps=self._steps,
            reward=self._reward,
            terminated=terminated,
            truncated=truncated,
            success=episode_success(self._info),
            duration_s=time.perf_counter() - self._started,
            predict_ms=mean(self._predict_ms),
            step_ms=mean(self._step_ms),
        )
        self._episodes.append(result)
        if self._hooks is not None:
            self._hooks.on_episode_end(result)

    def _outcome(self, terminated: bool, truncated: bool) -> str:
        if not (terminated or truncated):
            return ""
        success = episode_success(self._info)
        if success is None:
            return "done" if terminated else "timeout"
        if success:
            return "success"
        return "failure" if terminated else "timeout"

    def _feed_view(self, obs: object, outcome: str = "") -> None:
        if self._view is None:
            return
        self._view.feed(
            contract=self.contract,
            client=self,
            obs=obs,
            read=self.read,
            steps=self._steps,
            reward=self._reward,
            outcome=outcome,
            model_ms=self._predict_ms[-1] if self._predict_ms else 0.0,
            env_ms=self._step_ms[-1] if self._step_ms else 0.0,
            elapsed_s=time.perf_counter() - self._started,
            episode=self._index,
            seed=self._seed,
        )
