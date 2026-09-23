"""The unified native ``Model.run``: seeds, caps, and the hook contract.

``run()`` drives every env shape through the native runtime loop. These pin
the Session-parity features the runtime enforces itself: explicit per-episode
seeds (``episode_seeds``, echoed on the ``RunResult``), the step cap
(runtime-truncated episodes), and ``hooks`` firing the same callbacks in the
same per-episode order as ``Session.run`` -- with the terminal flags the
runtime knows at each step, exceptions aborting the run with their own type,
and ``on_run_end`` firing exactly once. Only the live viewer stays session-only.
"""

from __future__ import annotations

import time
from typing import Any, cast

import pytest
import rlmesh
import rlmesh.adapters as adapt
import rlmesh.numpy

np = pytest.importorskip("numpy")
pytest.importorskip("gymnasium")


class CountEnv:
    """Single env: Box obs, terminates after ``episode_len`` steps (0 = never).

    ``final_info``, when given, is emitted as the terminal step's info dict --
    the shape a Gymnasium env uses to report ``is_success`` / ``success``.
    """

    def __init__(
        self, episode_len: int = 3, final_info: dict[str, Any] | None = None
    ) -> None:
        import gymnasium as gym

        self.observation_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.episode_len = episode_len
        self.final_info = final_info or {}
        self.seen_seeds: list[int | None] = []
        self._t = 0

    def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, Any]:
        _ = options
        self.seen_seeds.append(seed)
        self._t = 0
        return np.zeros(2, np.float32), {}

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
        _ = action
        self._t += 1
        done = self.episode_len > 0 and self._t >= self.episode_len
        info = dict(self.final_info) if done else {}
        return np.zeros(2, np.float32), 1.0, done, False, info

    def close(self) -> None:
        return None


def _model() -> Any:
    from rlmesh.numpy import Model

    return Model(lambda obs: np.zeros(2, np.float32))


def test_run_seeds_set_episode_count_and_echo_on_result() -> None:
    env = CountEnv(episode_len=3)
    try:
        result = _model().run(env, seeds=[7, 8])
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise

    assert result.num_episodes == 2
    assert [e.seed for e in result.episodes] == [7, 8]
    assert env.seen_seeds[:2] == [7, 8]
    assert all(e.steps == 3 and e.reward == 3.0 for e in result.episodes)
    assert all(e.terminated and not e.truncated for e in result.episodes)
    assert all(e.predict_ms > 0.0 and e.step_ms > 0.0 for e in result.episodes)


def test_run_surfaces_the_session_telemetry_aggregate() -> None:
    env = CountEnv(episode_len=3)
    try:
        result = _model().run(env, seeds=[7])
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise

    series = {(row.op, row.metric) for row in result.telemetry}
    assert ("model.predict", "rpc.total") in series
    assert ("env.step", "rpc.total") in series
    assert ("runner.round", "rpc.total") in series
    for row in result.telemetry:
        assert row.count > 0
        assert row.unit in ("ms", "bytes", "count")
        assert row.avg >= 0.0 and row.p50 <= row.p95 <= row.p99
    # The formatter renders one aligned line per row plus a header.
    table = result.format_telemetry()
    assert len(table.splitlines()) == len(result.telemetry) + 1
    assert table.splitlines()[0].split()[:2] == ["op", "metric"]


def test_run_empty_seeds_returns_an_empty_result() -> None:
    result = _model().run(CountEnv(), seeds=[])
    assert result.num_episodes == 0
    assert result.telemetry == ()
    assert "no telemetry" in result.format_telemetry()


def test_run_reports_the_envs_success_signal() -> None:
    failed = CountEnv(episode_len=2, final_info={"is_success": False})
    succeeded = CountEnv(episode_len=2, final_info={"success": 1})
    try:
        failed_result = _model().run(failed, episodes=1)
        succeeded_result = _model().run(succeeded, episodes=1)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise

    assert failed_result.episodes[0].success is False
    assert failed_result.success_rate == 0.0
    assert succeeded_result.episodes[0].success is True
    assert succeeded_result.success_rate == 1.0


def test_run_trust_entrypoints_is_scoped_to_the_call() -> None:
    model = _model()
    try:
        model.run(CountEnv(), episodes=1, trust_entrypoints=True)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise
    assert model._trust_entrypoints is False


def test_run_truncates_an_over_produced_dict_chunk_to_the_horizon() -> None:
    import gymnasium as gym
    from rlmesh.numpy import Model

    class DictActionEnv(CountEnv):
        def __init__(self) -> None:
            super().__init__(episode_len=3)
            self.action_space = gym.spaces.Dict(
                {"arm": gym.spaces.Box(-1.0, 1.0, (2,), np.float32)}
            )
            self.seen: list[Any] = []

        def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
            self.seen.append(action)
            return super().step(action)

    class ChunkPolicy(Model):
        def predict_chunk(self, obs: Any) -> Any:
            chunk = np.zeros((5, 2), np.float32)
            chunk[:, 0] = np.arange(5, dtype=np.float32) / 10.0
            return {"arm": chunk}

    env = DictActionEnv()
    try:
        result = ChunkPolicy().run(env, episodes=1, execution_horizon=3)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise

    assert result.episodes[0].steps == 3
    assert len(env.seen) == 3
    np.testing.assert_allclose(
        [action["arm"][0] for action in env.seen], [0.0, 0.1, 0.2], atol=1e-6
    )


def test_run_prefetch_lead_predicts_the_next_chunk_from_a_stale_observation() -> None:
    """``prefetch_lead`` reaches the runtime driver: at chunk 3 / lead 1 the
    second chunk is asked for while one replay frame is still queued, from the
    observation after step 1 rather than step 3, and the chunk prefetched at
    each episode's tail is discarded so the next episode re-plans from its
    reset observation. Every episode still completes and scores."""
    import gymnasium as gym
    from rlmesh.numpy import Model

    class StepEnv(CountEnv):
        """Observation ``[t, 0]``: the step count, so a chunk corner can
        report which observation it was conditioned on."""

        def __init__(self) -> None:
            super().__init__(episode_len=6, final_info={"is_success": True})
            self.observation_space = gym.spaces.Box(0.0, 100.0, (2,), np.float32)

        def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, Any]:
            super().reset(seed=seed, options=options)
            return np.array([0.0, 0.0], np.float32), {}

        def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
            _, reward, done, truncated, info = super().step(action)
            return np.array([self._t, 0.0], np.float32), reward, done, truncated, info

    def run(lead: int) -> tuple[Any, list[int]]:
        seen: list[int] = []

        class ChunkPolicy(Model):
            native_chunk = 3

            def predict_chunk(self, obs: Any) -> Any:
                seen.append(int(obs[0]))
                return np.zeros((3, 2), np.float32)

        try:
            result = ChunkPolicy().run(
                StepEnv(), episodes=2, execution_horizon=3, prefetch_lead=lead
            )
        except ConnectionError as exc:
            if "Operation not permitted" in str(exc):
                pytest.skip("local tcp bind is not permitted in this environment")
            raise
        return result, seen

    result, seen = run(1)
    assert [e.steps for e in result.episodes] == [6, 6]
    assert all(e.success is True for e in result.episodes)
    assert result.success_rate == 1.0
    # The prefetch at each episode's tail (from step 4, with the last frame
    # still queued) may or may not reach the model before the terminal step
    # lands; every other chunk is pinned: the reset observation opens each
    # episode (the tail prefetch was discarded) and the second chunk was
    # conditioned on step 1, two steps before the synchronous loop's step 3.
    assert [obs for obs in seen if obs != 4] == [0, 1, 0, 1]

    _, synchronous = run(0)
    assert synchronous == [0, 3, 0, 3]


def test_run_holds_a_declared_native_chunk_to_its_length() -> None:
    """The spec-less native path measures the chunk against a declared K the
    way a Session does: a short chunk is a model error, not a silent early
    re-plan that only the adapted route would have caught."""
    from rlmesh.numpy import Model

    class ShortChunk(Model):
        native_chunk = 8

        def predict_chunk(self, obs: Any) -> Any:
            return np.zeros((2, 2), np.float32)

    try:
        with pytest.raises(
            Exception, match="native_chunk=8 but its chunk corner returned 2"
        ):
            ShortChunk().run(CountEnv(), episodes=1, execution_horizon=4)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise


def test_run_delivers_every_step_to_a_stacked_model_at_any_horizon() -> None:
    """The native path carries replayed steps to the model as observation
    history, so a frame-stacked model sees the same windows under ``run`` as
    under a Session, at execution_horizon 1 and above."""
    import rlmesh.adapters as adapt
    from rlmesh.numpy import Model

    class CameraEnv:
        observation_space = rlmesh.spaces.Box(0, 255, shape=(2, 2, 3), dtype="uint8")
        action_space = rlmesh.spaces.Box(-1.0, 1.0, shape=(2,), dtype="float32")

        def __init__(self) -> None:
            self.n = 0

        def _frame(self) -> Any:
            return np.full((2, 2, 3), self.n, np.uint8)

        def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, Any]:
            self.n = 0
            return self._frame(), {}

        def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
            self.n += 1
            return self._frame(), 1.0, self.n >= 7, False, {}

        def close(self) -> None:
            return None

    out = adapt.Action(adapt.Actuator("x/action", dim=2))
    tags = adapt.EnvTags(observation=adapt.ImageTag(adapt.IMAGE_PRIMARY), action=out)

    class Stacked(Model):
        native_chunk = 3
        spec = adapt.ModelSpec(
            input=adapt.Image(adapt.IMAGE_PRIMARY, stack=3), output=out
        )

        def load(self) -> None:
            self.windows: list[tuple[int, ...]] = []

        def predict_chunk(self, observation: Any) -> Any:
            # The stacked axis leads: (3, H, W, C). Record which frames it holds.
            self.windows.append(tuple(int(frame[0, 0, 0]) for frame in observation))
            return np.zeros((3, 2), np.float32)

    def windows(path: str, horizon: int) -> list[tuple[int, ...]]:
        model = Stacked()
        env = adapt.tag(CameraEnv(), tags)
        try:
            if path == "session":
                model.session(env, execution_horizon=horizon).run(episodes=1)
            else:
                model.run(env, episodes=1, execution_horizon=horizon)
        except ConnectionError as exc:
            if "Operation not permitted" in str(exc):
                pytest.skip("local tcp bind is not permitted in this environment")
            raise
        return model.windows

    every_step = windows("native", 1)
    assert every_step[:3] == [(0, 0, 0), (0, 0, 1), (0, 1, 2)]
    # Re-plans at steps 0, 3 and 6: the window at each holds the consecutive
    # frames, not the decision points, on both paths.
    assert windows("native", 3) == [every_step[0], every_step[3], every_step[6]]
    assert windows("session", 3) == windows("native", 3)


def test_run_max_episode_steps_truncates_via_the_runtime() -> None:
    env = CountEnv(episode_len=0)
    try:
        result = _model().run(env, episodes=2, max_episode_steps=4)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise

    assert result.num_episodes == 2
    assert len(result.episodes) == 2
    assert all(e.steps == 4 for e in result.episodes)
    assert all(e.truncated and not e.terminated for e in result.episodes)
    assert all(e.reward == 4.0 for e in result.episodes)
    # A truncation echoed back by the env must not be counted twice or cut the
    # next episode short: two distinct episodes, each driven for all four steps.
    assert [e.index for e in result.episodes] == [0, 1]
    assert len(env.seen_seeds) == 2


def test_run_on_a_driven_handle_explains_the_session_conflict() -> None:
    """A RemoteEnv handle that was driven holds the env's single session slot;
    run() (which dials the handle's address) surfaces a clear pointer instead
    of the raw wire error."""
    try:
        server = rlmesh.EnvServer(CountEnv(), "127.0.0.1:0")
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise
    server.start()
    try:
        handle = rlmesh.numpy.RemoteEnv(server.address)
        handle.reset()
        with pytest.raises(RuntimeError, match=r"close\(\) the handle"):
            _model().run(handle, episodes=1)
        handle.close()
    finally:
        server.shutdown()


def test_the_viewer_stays_a_session_option() -> None:
    import inspect

    assert "view" not in inspect.signature(rlmesh.Model.run).parameters
    with pytest.raises(TypeError, match=r"session\(\) option"):
        rlmesh.run(_model(), CountEnv(), view="terminal")


# ---------------------------------------------------------------------------
# hooks: the same contract on both loops


class _BoomError(Exception):
    pass


class _Recorder(rlmesh.RunHooks):
    """Every hook call in order; flags any call that lands after the run returned."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.events: list[rlmesh.StepEvent] = []
        self.episode_results: list[rlmesh.EpisodeResult] = []
        self.run_results: list[rlmesh.RunResult] = []
        self.contexts: list[Any] = []
        self.closed = False
        self.late: list[tuple[Any, ...]] = []

    def _note(self, *call: Any) -> None:
        if self.closed:
            self.late.append(call)
        self.calls.append(call)

    def on_run_start(self, context: rlmesh.RunContext) -> None:
        self.contexts.append(context)
        self._note("run_start")

    def on_episode_start(self, *, episode: int, seed: int | None) -> None:
        self._note("start", episode, seed)

    def on_step(self, event: rlmesh.StepEvent) -> None:
        self._note("step", event.episode, event.step)
        self.events.append(event)

    def on_episode_end(self, result: rlmesh.EpisodeResult) -> None:
        self._note("end", result.index)
        self.episode_results.append(result)

    def on_run_end(self, result: rlmesh.RunResult) -> None:
        self._note("run_end", result.num_episodes)
        self.run_results.append(result)


def _raising(hook: str, error: BaseException) -> _Recorder:
    recorder = _Recorder()
    original = getattr(recorder, hook)

    def raise_after(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        raise error

    setattr(recorder, hook, raise_after)
    return recorder


def _drive(model: Any, env: Any, path: str, **kwargs: Any) -> rlmesh.RunResult:
    try:
        if path == "session":
            with model.session(env) as sess:
                return sess.run(**kwargs)
        return model.run(env, **kwargs)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise


PATHS = ["session", "native"]


@pytest.mark.parametrize("path", PATHS)
def test_hooks_fire_in_order_with_indices_and_seeds(path: str) -> None:
    recorder = _Recorder()
    result = _drive(
        _model(), CountEnv(episode_len=1), path, seeds=[7, 8], hooks=recorder
    )
    recorder.closed = True

    assert recorder.calls == [
        ("run_start",),
        ("start", 0, 7),
        ("step", 0, 0),
        ("end", 0),
        ("start", 1, 8),
        ("step", 1, 0),
        ("end", 1),
        ("run_end", 2),
    ]
    first = recorder.events[0]
    assert first.seed == 7 and first.reward == 1.0
    assert first.terminated and not first.truncated
    assert first.observation.shape == (2,) and first.action.shape == (2,)
    assert first.predict_ms >= 0.0 and first.step_ms >= 0.0
    assert recorder.run_results == [result]
    # Hooks never change the result, and the record delivered to on_episode_end
    # IS the result's: one timing definition, timings included.
    assert recorder.episode_results == list(result.episodes)
    assert all(
        e.predict_ms is not None and e.step_ms is not None and e.step_ms > 0.0
        for e in result.episodes
    )
    assert recorder.late == []


@pytest.mark.parametrize("path", PATHS)
def test_hooks_carry_the_terminal_flag_on_the_terminal_step_only(path: str) -> None:
    recorder = _Recorder()
    _drive(_model(), CountEnv(episode_len=3), path, episodes=1, hooks=recorder)
    flags = [(e.step, e.terminated, e.truncated) for e in recorder.events]
    assert flags == [(0, False, False), (1, False, False), (2, True, False)]


@pytest.mark.parametrize("path", PATHS)
def test_hooks_carry_the_truncation_on_the_capped_step(path: str) -> None:
    recorder = _Recorder()
    _drive(
        _model(),
        CountEnv(episode_len=0),
        path,
        episodes=1,
        max_episode_steps=3,
        hooks=recorder,
    )
    flags = [(e.step, e.terminated, e.truncated) for e in recorder.events]
    assert flags == [(0, False, False), (1, False, False), (2, False, True)]
    assert recorder.episode_results[0].truncated


@pytest.mark.parametrize("path", PATHS)
def test_hooks_read_a_role_off_the_observation(path: str) -> None:
    gym = pytest.importorskip("gymnasium")

    class _ArmEnv:
        def __init__(self) -> None:
            self.metadata: dict[str, Any] = {}
            self.observation_space = gym.spaces.Dict(
                {"eef_pos": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32)}
            )
            self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

        def reset(self, *, seed: object = None, options: object = None) -> Any:
            return {"eef_pos": np.array([0.1, 0.2, 0.3], np.float32)}, {}

        def step(self, action: object) -> Any:
            return (
                {"eef_pos": np.array([0.1, 0.2, 0.3], np.float32)},
                1.0,
                True,
                False,
                {},
            )

        def close(self) -> None:
            pass

    tags = adapt.EnvTags(
        observation={"eef_pos": adapt.StateTag(role=adapt.EEF_POS)},
        action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )
    from rlmesh.numpy import Model

    class _Reading(_Recorder):
        def on_step(self, event: rlmesh.StepEvent) -> None:
            super().on_step(event)
            self.values = [event.read(adapt.EEF_POS)]
            context = self.contexts[0]
            self.seen = (context.num_envs, context.frame_sources().roles)

    recorder = _Reading()
    _drive(
        Model(lambda obs: np.zeros(1, np.float32), spec=rlmesh.NO_ADAPTER),
        adapt.tag(_ArmEnv(), tags),
        path,
        episodes=1,
        hooks=recorder,
    )
    assert recorder.values[0].shape == (3,)
    assert recorder.seen == (1, ())


@pytest.mark.parametrize(
    "hook", ["on_run_start", "on_episode_start", "on_step", "on_episode_end"]
)
def test_a_hook_exception_aborts_the_native_run_with_its_own_type(hook: str) -> None:
    recorder = _raising(hook, _BoomError(hook))
    model = _model()
    with pytest.raises(_BoomError, match=hook):
        _drive(model, CountEnv(episode_len=2), "native", seeds=[1, 2], hooks=recorder)
    recorder.closed = True
    time.sleep(0.05)

    run_ends = [call for call in recorder.calls if call[0] == "run_end"]
    assert len(run_ends) == 1 and recorder.calls[-1] == run_ends[0]
    assert (
        recorder.calls[-2][0]
        == {
            "on_run_start": "run_start",
            "on_episode_start": "start",
            "on_step": "step",
            "on_episode_end": "end",
        }[hook]
    )
    assert run_ends[0][1] == len(recorder.episode_results)
    assert run_ends[0][1] <= 1, "the run stopped at the raising episode"
    assert recorder.late == []
    assert model._instruction is None  # pyright: ignore[reportPrivateUsage]
    assert model._native_run is None  # pyright: ignore[reportPrivateUsage]


def test_a_keyboard_interrupt_in_a_hook_propagates_from_the_native_run() -> None:
    recorder = _raising("on_step", KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        _drive(
            _model(), CountEnv(episode_len=2), "native", seeds=[1, 2], hooks=recorder
        )
    assert [c for c in recorder.calls if c[0] == "run_end"] == [("run_end", 0)]


def test_the_original_hook_exception_wins_over_a_raising_on_run_end() -> None:
    recorder = _raising("on_step", _BoomError("original"))
    recorder.on_run_end = _raising("on_run_end", _BoomError("run_end")).on_run_end  # type: ignore[method-assign]
    with pytest.raises(_BoomError, match="original"):
        _drive(_model(), CountEnv(episode_len=2), "native", episodes=1, hooks=recorder)

    lone = _raising("on_run_end", _BoomError("lone"))
    with pytest.raises(_BoomError, match="lone"):
        _drive(_model(), CountEnv(episode_len=2), "native", episodes=1, hooks=lone)


def test_hooks_fire_on_an_empty_native_run() -> None:
    recorder = _Recorder()
    result = _drive(_model(), CountEnv(), "native", episodes=0, hooks=recorder)
    assert result.num_episodes == 0
    assert recorder.calls == [("run_start",), ("run_end", 0)]


class _NextStepVectorEnv:
    """Lanes under NEXT_STEP autoreset, each with its own episode length.

    A lane that ended resets on its next ``step`` call (reward 0, the reset
    observation), as Gymnasium's vector envs do, while the other lanes step on.
    """

    def __init__(self, lengths: tuple[int, ...] = (1, 1)) -> None:
        from rlmesh import spaces

        self.num_envs = len(lengths)
        self.single_observation_space = spaces.Box(
            0.0, 1.0, shape=(2,), dtype="float32"
        )
        self.single_action_space = spaces.Box(0.0, 1.0, shape=(2,), dtype="float32")
        self.metadata = {"autoreset_mode": "NextStep"}
        self._lengths = lengths
        self._t = [0] * self.num_envs
        self._pending = [False] * self.num_envs

    def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, Any]:
        self._t = [0] * self.num_envs
        self._pending = [False] * self.num_envs
        return np.zeros((self.num_envs, 2), dtype=np.float32), {}

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
        rewards: list[float] = []
        terminated: list[bool] = []
        for lane in range(self.num_envs):
            if self._pending[lane]:
                self._pending[lane] = False
                self._t[lane] = 0
                rewards.append(0.0)
                terminated.append(False)
            else:
                self._t[lane] += 1
                done = self._t[lane] >= self._lengths[lane]
                self._pending[lane] = done
                rewards.append(1.0)
                terminated.append(done)
        obs = np.zeros((self.num_envs, 2), dtype=np.float32)
        return obs, rewards, terminated, [False] * self.num_envs, {}

    def close(self) -> None:
        return None


def _episodes_by_index(recorder: _Recorder) -> dict[int, list[tuple[Any, ...]]]:
    by_episode: dict[int, list[tuple[Any, ...]]] = {}
    for call in recorder.calls:
        if call[0] in ("start", "step", "end"):
            by_episode.setdefault(call[1], []).append(call)
    return by_episode


def _assert_interleaved_episodes_are_well_formed(
    recorder: _Recorder, result: rlmesh.RunResult
) -> None:
    assert len(recorder.episode_results) == result.num_episodes
    assert recorder.run_results == [result]
    steps_of = {r.index: r.steps for r in recorder.episode_results}
    last_events = {e.episode: e for e in recorder.events}
    for index, calls in _episodes_by_index(recorder).items():
        # The budget bounds starts: every episode a hook sees start completes.
        assert index in steps_of, index
        kinds = [c[0] for c in calls]
        assert kinds == ["start"] + ["step"] * steps_of[index] + ["end"], (index, kinds)
        assert [c[2] for c in calls if c[0] == "step"] == list(range(steps_of[index]))
        assert last_events[index].terminated, index
    assert all(e.reward == 1.0 for e in recorder.events), "no autoreset roll surfaces"


def test_hooks_on_a_next_step_vector_env() -> None:
    from rlmesh.numpy import Model

    recorder = _Recorder()
    # A spec-less policy on a vector env predicts on the fused (N, ...) batch.
    stacked = Model(lambda obs: np.zeros((2, 2), np.float32))
    result = _drive(stacked, _NextStepVectorEnv(), "native", episodes=2, hooks=recorder)
    assert result.num_episodes == 2
    assert recorder.contexts[0].num_envs == 2
    _assert_interleaved_episodes_are_well_formed(recorder, result)


@pytest.mark.parametrize("lengths", [(2, 2), (1, 3), (3, 1)])
def test_an_uneven_budget_is_exact_on_a_lockstep_vector_env(
    lengths: tuple[int, int],
) -> None:
    # Two lanes, three episodes: equal lengths complete in pairs (the second
    # pair overshoots the budget by one), unequal ones roll apart. Either way
    # exactly three are started, scored, and reported; the fourth lane-episode
    # the env rolls into is never announced.
    from rlmesh.numpy import Model

    recorder = _Recorder()
    stacked = Model(lambda obs: np.zeros((2, 2), np.float32))
    result = _drive(
        stacked, _NextStepVectorEnv(lengths), "native", episodes=3, hooks=recorder
    )
    assert result.num_episodes == 3
    assert sorted(e.index for e in result.episodes) == [0, 1, 2]
    assert [c[1] for c in recorder.calls if c[0] == "start"] == [0, 1, 2]
    _assert_interleaved_episodes_are_well_formed(recorder, result)
    assert sum(e.steps for e in result.episodes) == sum(
        e.steps for e in recorder.episode_results
    )


def test_an_uneven_budget_is_exact_on_driver_reset_lanes() -> None:
    lanes = [CountEnv(episode_len=1), CountEnv(episode_len=3)]
    server = rlmesh.EnvServer(cast("Any", lanes), host="127.0.0.1", port=0)
    try:
        server.start()
    except (OSError, ConnectionError) as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise
    recorder = _Recorder()
    try:
        result = _model().run(server.address, seeds=[3, 4, 5], hooks=recorder)
    finally:
        server.shutdown()
    assert result.num_episodes == 3
    assert sorted(e.seed for e in result.episodes) == [3, 4, 5]
    assert sorted(e.trial for e in result.episodes) == [0, 1, 2]
    _assert_interleaved_episodes_are_well_formed(recorder, result)


def test_the_episode_budget_is_validated_before_any_episode() -> None:
    model = _model()
    with pytest.raises(ValueError, match="episodes must be >= 0"):
        model.run(CountEnv(episode_len=1), episodes=-1)
    with pytest.raises(ValueError, match="does not match len\\(seeds\\)"):
        model.run(CountEnv(episode_len=1), episodes=2, seeds=[1, 2, 3])
    assert model.run(CountEnv(episode_len=1), episodes=0).num_episodes == 0
    assert (
        model.run(CountEnv(episode_len=1), episodes=2, seeds=[1, 2]).num_episodes == 2
    )


def test_seeds_on_an_autoresetting_vector_env_are_refused_before_execution() -> None:
    from rlmesh.numpy import Model

    recorder = _Recorder()
    stacked = Model(lambda obs: np.zeros((2, 2), np.float32))
    with pytest.raises(Exception, match="autoreset"):
        _drive(stacked, _NextStepVectorEnv(), "native", seeds=[1, 2], hooks=recorder)
    assert [c[0] for c in recorder.calls] == ["run_start", "run_end"]


def test_hooks_on_a_next_step_vector_env_whose_lanes_roll_apart() -> None:
    # Lane 0 rolls while lane 1 is mid-episode: the roll is skipped for lane 0
    # only, and lane 1's step on that response is delivered.
    from rlmesh.numpy import Model

    recorder = _Recorder()
    stacked = Model(lambda obs: np.zeros((2, 2), np.float32))
    result = _drive(
        stacked, _NextStepVectorEnv((1, 3)), "native", episodes=3, hooks=recorder
    )
    _assert_interleaved_episodes_are_well_formed(recorder, result)
    assert {r.steps for r in recorder.episode_results} >= {1, 3}


def test_hooks_on_driver_reset_lanes() -> None:
    lanes = [CountEnv(episode_len=2), CountEnv(episode_len=2)]
    server = rlmesh.EnvServer(cast("Any", lanes), host="127.0.0.1", port=0)
    try:
        server.start()
    except (OSError, ConnectionError) as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise
    recorder = _Recorder()
    try:
        result = _model().run(server.address, episodes=2, seeds=[3, 4], hooks=recorder)
    finally:
        server.shutdown()
    assert result.num_episodes == 2
    assert recorder.contexts[0].num_envs == 2
    _assert_interleaved_episodes_are_well_formed(recorder, result)
    assert sorted(c[2] for c in recorder.calls if c[0] == "start") == [3, 4]


def test_session_served_env_context_carries_stable_episode_identity() -> None:
    from rlmesh.numpy import Model

    seen: list[dict[str, Any]] = []

    def predict(observation: Any, context: dict[str, Any]) -> Any:
        seen.append(dict(context))
        return np.zeros(2, np.float32)

    try:
        server = rlmesh.EnvServer(CountEnv(episode_len=3), host="127.0.0.1", port=0)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise
    server.start()
    try:
        with Model(predict).session(server.address) as sess:
            sess.run(seeds=[7], episodes=1)
    finally:
        server.shutdown()

    assert len(seen) == 3
    episode_ids = {str(context["episode_id"]) for context in seen}
    assert len(episode_ids) == 1, f"episode_id changed mid-episode: {seen}"
    assert all(episode_ids), f"episode_id blank on some step: {seen}"
    assert [context["episode_seed"] for context in seen] == [7, 7, 7]


def test_run_feeds_a_previous_action_part_the_frame_executed_the_step_before() -> None:
    """A Go2-style policy reading its own last command sees, at every re-plan,
    the raw chunk frame the runtime executed at the step before: the last frame
    of the previous chunk at execution_horizon 7, the only one at 1, and the
    fill at step 0, on the native and the session paths alike."""
    import gymnasium as gym
    import rlmesh.adapters as adapt
    from rlmesh.numpy import Model

    sdk = adapt.GO2.joints
    default_pose = (0.0, 0.8, -1.5) * 4

    class Go2Env:
        observation_space = gym.spaces.Dict(
            {
                "ang_vel": gym.spaces.Box(-20.0, 20.0, (3,), np.float32),
                "base_quat": gym.spaces.Box(-1.0, 1.0, (4,), np.float32),
                "command": gym.spaces.Box(-3.0, 3.0, (3,), np.float32),
                "joint_pos": gym.spaces.Box(-10.0, 10.0, (12,), np.float32),
                "joint_vel": gym.spaces.Box(-30.0, 30.0, (12,), np.float32),
            }
        )
        action_space = gym.spaces.Box(-10.0, 10.0, (12,), np.float32)

        def __init__(self) -> None:
            self.n = 0

        def _obs(self) -> dict[str, Any]:
            return {
                "ang_vel": np.zeros(3, np.float32),
                "base_quat": np.array([1.0, 0.0, 0.0, 0.0], np.float32),
                "command": np.zeros(3, np.float32),
                "joint_pos": np.full(12, self.n, np.float32),
                "joint_vel": np.zeros(12, np.float32),
            }

        def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, Any]:
            self.n = 0
            return self._obs(), {}

        def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
            self.n += 1
            return self._obs(), 1.0, self.n >= 15, False, {}

        def close(self) -> None:
            return None

    tags = adapt.EnvTags(
        observation={
            "ang_vel": adapt.StateTag(adapt.BASE_ANG_VEL, frame="robot_base"),
            "base_quat": adapt.StateTag(
                adapt.BASE_ROT, encoding="quat_wxyz", frame="world"
            ),
            "command": adapt.StateTag(adapt.COMMAND_BASE_VEL, frame="robot_base"),
            "joint_pos": adapt.StateTag(adapt.JOINT_POS, labels=sdk),
            "joint_vel": adapt.StateTag(adapt.JOINT_VEL, labels=sdk),
        },
        action=adapt.Action(adapt.Actuator(adapt.ACTION_JOINT_POS, dim=12, labels=sdk)),
    )

    class Walk(Model):
        native_chunk = 7
        spec = adapt.ModelSpec(
            input={
                "obs": adapt.Concat(
                    adapt.State(adapt.BASE_ANG_VEL, frame="robot_base", scale=0.25),
                    adapt.State(adapt.BASE_ROT, encoding="gravity_xyz", frame="world"),
                    adapt.State(adapt.COMMAND_BASE_VEL, frame="robot_base"),
                    adapt.State(
                        adapt.JOINT_POS,
                        labels=sdk,
                        offset=tuple(-q for q in default_pose),
                    ),
                    adapt.State(adapt.JOINT_VEL, labels=sdk, scale=0.05),
                    adapt.Previous(adapt.ACTION_JOINT_POS),
                    clip=(-100.0, 100.0),
                )
            },
            output=adapt.Action(
                adapt.Actuator(
                    adapt.ACTION_JOINT_POS,
                    dim=12,
                    labels=sdk,
                    scale=(0.125, 0.25, 0.25) * 4,
                    offset=default_pose,
                )
            ),
        )

        def load(self) -> None:
            self.previous: list[float] = []
            self.calls = 0

        def predict_chunk(self, observation: Any) -> Any:
            obs = observation["obs"]
            assert obs.shape == (45,)
            # The previous-action slot is uniform by construction; record it.
            assert len(set(obs[33:].tolist())) == 1
            self.previous.append(float(obs[33]))
            # Frame k of call c is 5c + k + 1 on every joint (inside the
            # container clip): the scaled, offset env command is a different
            # number, so a slot holding anything but the raw frame is visible.
            chunk = np.full((7, 12), 5 * self.calls + 1, np.float32)
            chunk += np.arange(7, dtype=np.float32)[:, None]
            self.calls += 1
            return chunk

    def previous(path: str, horizon: int) -> list[float]:
        model = Walk()
        env = adapt.tag(Go2Env(), tags)
        try:
            if path == "session":
                model.session(env, execution_horizon=horizon).run(episodes=1)
            else:
                model.run(env, episodes=1, execution_horizon=horizon)
        except ConnectionError as exc:
            if "Operation not permitted" in str(exc):
                pytest.skip("local tcp bind is not permitted in this environment")
            raise
        return model.previous

    # Horizon 1: predict k+1 sees frame 0 of call k (5k + 1).
    every_step = [0.0] + [5.0 * call + 1 for call in range(14)]
    assert previous("native", 1) == every_step
    assert previous("session", 1) == every_step
    # Horizon 7: re-plans at 0, 7 and 14 see the fill, then frame 6 of the
    # chunk executed before (7 and 12), never frame 0 of it (1 and 6).
    assert previous("native", 7) == [0.0, 7.0, 12.0]
    assert previous("session", 7) == [0.0, 7.0, 12.0]


# ---------------------------------------------------------------------------
# Ownership: a borrowed model and env survive a run, a failure, an interrupt
# ---------------------------------------------------------------------------


def test_a_model_is_reusable_after_a_predict_failure_and_an_interrupt() -> None:
    from rlmesh.numpy import Model

    closes: list[int] = []
    calls = {"n": 0}

    def predict(observation: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _BoomError("first predict fails")
        return np.zeros(2, np.float32)

    model = Model(predict, on_close=lambda: closes.append(1))
    env = CountEnv(episode_len=2)
    with pytest.raises(Exception, match="first predict fails"):
        _drive(model, env, "native", episodes=1)
    assert _drive(model, env, "native", episodes=1).num_episodes == 1
    with pytest.raises(KeyboardInterrupt):
        _drive(
            model,
            env,
            "native",
            episodes=1,
            hooks=_raising("on_step", KeyboardInterrupt()),
        )
    assert _drive(model, env, "native", episodes=2).num_episodes == 2
    # A session on the same model, then a run on it again: nothing was closed.
    with model.session(env) as sess:
        assert sess.run(episodes=1).num_episodes == 1
    assert _drive(model, env, "native", episodes=1).num_episodes == 1
    assert closes == []
    model.close()
    assert closes == [1]


def test_a_borrowed_env_is_left_open_by_the_native_run() -> None:
    closed: list[int] = []

    class _ClosingEnv(CountEnv):
        def close(self) -> None:
            closed.append(1)

    env = _ClosingEnv(episode_len=1)
    _drive(_model(), env, "native", episodes=2)
    _drive(_model(), env, "native", episodes=1)
    assert closed == []
    _drive(_model(), env, "native", episodes=1, close_env=True)
    assert closed == [1]


def test_a_predict_exception_keeps_its_own_type_on_the_native_run() -> None:
    def predict(obs: Any) -> Any:
        raise KeyError("missing_key")

    model = rlmesh.numpy.Model(predict, spec=rlmesh.NO_ADAPTER)
    with pytest.raises(KeyError, match="missing_key"):
        _drive(model, CountEnv(episode_len=2), "native", seeds=[1])


def test_a_spec_less_predict_batch_takes_the_vector_batch() -> None:
    from rlmesh.numpy import Model

    seen: list[tuple[str, tuple[int, ...]]] = []

    class Batched(Model):
        def predict(self, observation: Any) -> Any:
            seen.append(("predict", np.shape(observation)))
            return np.zeros(2, np.float32)

        def predict_batch(self, observations: Any) -> Any:
            seen.append(("batch", np.shape(observations)))
            return np.zeros((len(observations), 2), np.float32)

    result = _drive(Batched(), _NextStepVectorEnv(), "native", episodes=2)
    assert result.num_episodes == 2
    assert seen and all(kind == "batch" for kind, _ in seen)
    assert seen[0][1] == (2, 2)
