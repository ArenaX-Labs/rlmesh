"""The neutral pair-driver: rlmesh.run / rlmesh.session / Session.

These pin the Session seam against a tiny in-process env (no server): how a model is
bound to an env and driven, both auto (run) and by hand (reset/predict/step). The
full adapter/remote loop is exercised in the integration suite.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import rlmesh
import rlmesh.numpy
from rlmesh._models._view import ViewerDriver


class _TinyEnv:
    """A minimal local env: one step, then terminates with reward 1.0."""

    def __init__(self) -> None:
        from rlmesh import spaces

        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(1)

    def reset(
        self, *, seed: object = None, options: object = None
    ) -> tuple[int, dict[str, object]]:
        return 0, {"seed": seed}

    def step(self, action: object) -> tuple[int, float, bool, bool, dict[str, object]]:
        return 0, 1.0, True, False, {"action": action}

    def close(self) -> None:
        pass


def test_run_drives_a_model_against_a_local_env() -> None:
    result = rlmesh.run(rlmesh.Model(lambda obs: 0), _TinyEnv(), seeds=[0, 1])
    assert isinstance(result, rlmesh.RunResult)
    assert result.num_episodes == 2
    assert result.mean_reward == 1.0  # one step, reward 1.0, then terminates


def test_run_refuses_a_bare_callable_and_takes_a_framework_model() -> None:
    with pytest.raises(TypeError, match=r"take a Model.*rlmesh\.numpy\.Model"):
        rlmesh.run(lambda obs: 0, _TinyEnv())
    with pytest.raises(TypeError, match="take a Model"):
        rlmesh.session(lambda obs: 0, _TinyEnv())
    result = rlmesh.run(rlmesh.numpy.Model(lambda obs: 0), _TinyEnv())
    assert result.num_episodes == 1


def test_session_manual_drive() -> None:
    sess = rlmesh.session(rlmesh.Model(lambda obs: 0), _TinyEnv())
    assert isinstance(sess, rlmesh.Session)
    obs, _info = sess.reset(seed=0)
    assert sess.done is False
    action = sess.predict(obs)
    _obs, reward, terminated, _trunc, _info = sess.step(action)
    assert reward == 1.0
    assert terminated is True
    assert sess.done is True  # the env terminated after one step
    sess.close()


def test_session_run_matches_top_level_run() -> None:
    result = rlmesh.session(rlmesh.Model(lambda obs: 0), _TinyEnv()).run(seeds=[0])
    assert result.num_episodes == 1


class _ForeverEnv:
    """A local env that never terminates on its own (only a cap/skip can end it)."""

    def __init__(self) -> None:
        from rlmesh import spaces

        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(1)

    def reset(
        self, *, seed: object = None, options: object = None
    ) -> tuple[int, dict[str, object]]:
        return 0, {}

    def step(self, action: object) -> tuple[int, float, bool, bool, dict[str, object]]:
        return 0, 0.0, False, False, {}

    def close(self) -> None:
        pass


class _SkipDriver:
    """A stand-in ViewerDriver that asks to end each episode after its first step."""

    def __init__(self) -> None:
        self._steps = 0

    def feed(self, *, steps: int, **_: Any) -> None:
        self._steps = steps

    def consume_skip(self) -> bool:
        return self._steps >= 1

    def quit_requested(self) -> bool:
        return False

    def close(self) -> None:
        pass


class _QuitDriver(_SkipDriver):
    """A stand-in ViewerDriver that requests quit after the first fed step."""

    def consume_skip(self) -> bool:
        return False

    def quit_requested(self) -> bool:
        return self._steps >= 1


def test_viewer_skip_truncates_episode_without_failing() -> None:
    sess = rlmesh.session(rlmesh.Model(lambda obs: 0), _ForeverEnv())
    sess._view_driver = cast(ViewerDriver, _SkipDriver())
    obs, _info = sess.reset(seed=0)
    _obs, _reward, terminated, truncated, _info = sess.step(sess.predict(obs))
    assert truncated is True
    assert terminated is False
    assert sess.done is True
    sess.close()


def test_viewer_skip_advances_run_through_all_seeds() -> None:
    sess = rlmesh.session(rlmesh.Model(lambda obs: 0), _ForeverEnv())
    sess._view_driver = cast(ViewerDriver, _SkipDriver())
    result = sess.run(seeds=[0, 1, 2])
    assert result.num_episodes == 3
    assert all(e.truncated and not e.terminated for e in result.episodes)
    assert all(e.steps == 1 for e in result.episodes)


def test_viewer_quit_stops_early_and_returns_the_partial_result() -> None:
    # Quit ('q') is stop-early, not an interrupt: the current episode is truncated,
    # the loop stops iterating seeds, and the partial RunResult is RETURNED (a real
    # Ctrl-C still raises KeyboardInterrupt).
    sess = rlmesh.session(rlmesh.Model(lambda obs: 0), _ForeverEnv())
    sess._view_driver = cast(ViewerDriver, _QuitDriver())
    result = sess.run(seeds=[0, 1, 2])
    assert result.num_episodes == 1
    assert result.episodes[0].truncated is True
    assert result.episodes[0].steps == 1
    sess.close()


def test_session_is_a_context_manager() -> None:
    with rlmesh.session(rlmesh.Model(lambda obs: 0), _TinyEnv()) as sess:
        obs, _info = sess.reset()
        sess.step(sess.predict(obs))
        assert sess.done is True


def test_caller_held_session_survives_run_and_runs_again() -> None:
    # Session.run() must NOT close a caller-held session: the connection (and any
    # viewer/hooks state) stays alive, so run() composes -- run twice, or mix runs
    # with manual driving -- until the caller closes it.
    sess = rlmesh.session(rlmesh.Model(lambda obs: 0), _TinyEnv())
    first = sess.run(seeds=[0])
    assert first.num_episodes == 1
    assert sess._connected is True
    second = sess.run(seeds=[1, 2])
    assert second.num_episodes == 2
    obs, _ = sess.reset()
    sess.step(sess.predict(obs))
    sess.close()


def test_closed_session_rejects_any_further_use() -> None:
    # After an explicit close() (or `with` exit) every primitive raises instead of
    # silently reconnecting through a broken path.
    sess = rlmesh.session(rlmesh.Model(lambda obs: 0), _TinyEnv())
    obs, _ = sess.reset()
    sess.close()
    sess.close()  # idempotent
    with pytest.raises(RuntimeError, match="session is closed"):
        sess.run(episodes=1)
    with pytest.raises(RuntimeError, match="session is closed"):
        sess.reset()
    with pytest.raises(RuntimeError, match="session is closed"):
        sess.predict(obs)
    with pytest.raises(RuntimeError, match="session is closed"):
        sess.step(0)
    with pytest.raises(RuntimeError, match="session is closed"):
        sess.read(obs, "state/eef_pos")


def test_session_close_leaves_a_borrowed_model_open() -> None:
    # A model instance is the caller's: closing the session releases the
    # connection and episode state, never the model. close() is explicit.
    closes: list[int] = []
    model = rlmesh.Model(lambda obs: 0, on_close=lambda: closes.append(1))
    with rlmesh.session(model, _TinyEnv()) as sess:
        sess.run(seeds=[0])
    assert closes == []
    with rlmesh.session(model, _TinyEnv()) as sess:  # still usable
        assert sess.run(seeds=[1]).num_episodes == 1
    model.close()
    assert closes == [1]


def test_session_owns_a_model_it_built_from_a_class() -> None:
    closes: list[int] = []

    class _Owned(rlmesh.Model):
        def predict(self, observation: object) -> int:
            return 0

        def close(self) -> None:
            closes.append(1)

    with rlmesh.session(_Owned, _TinyEnv()) as sess:
        sess.run(seeds=[0])
        assert closes == []
    assert closes == [1]
    sess.close()
    assert closes == [1]  # idempotent
    rlmesh.run(_Owned, _TinyEnv())
    assert closes == [1, 1]  # one-shot run closes the model it built


def test_one_shot_run_borrows_a_model_instance() -> None:
    closes: list[int] = []
    model = rlmesh.Model(lambda obs: 0, on_close=lambda: closes.append(1))
    rlmesh.run(model, _TinyEnv())
    model.run(_TinyEnv())
    assert closes == []
    model.close()
    assert closes == [1]


def test_run_and_session_release_a_factory_built_env() -> None:
    closed: list[int] = []

    class _Env(_TinyEnv):
        def close(self) -> None:
            closed.append(1)

    class _Factory(rlmesh.EnvFactory):
        def make(self) -> object:
            return _Env()

    model = rlmesh.Model(lambda obs: 0)
    model.run(_Factory())
    assert closed == [1]
    with rlmesh.session(model, _Factory()) as sess:
        sess.run(seeds=[0])
    assert closed == [1, 1]
    # A caller's env is borrowed: closed only on the opt-in.
    env = _Env()
    model.run(env)
    with rlmesh.session(model, env) as sess:
        sess.run(seeds=[0])
    assert closed == [1, 1]
    model.run(env, close_env=True)
    assert closed == [1, 1, 1]
    # Opting a factory-built env into shutdown closes it once, not twice.
    with rlmesh.session(model, _Factory(), close_env=True) as sess:
        sess.run(seeds=[0])
    assert closed == [1, 1, 1, 1]
    model.run(_Factory(), close_env=True)
    assert closed == [1, 1, 1, 1, 1]


def test_session_run_episode_budget_is_exact_and_validated() -> None:
    model = rlmesh.Model(lambda obs: 0)
    with rlmesh.session(model, _TinyEnv()) as sess:
        assert sess.run().num_episodes == 1
        assert sess.run(seeds=[1, 2]).num_episodes == 2
        assert sess.run(episodes=3).num_episodes == 3
        assert sess.run(episodes=0).num_episodes == 0
        assert sess.run(episodes=2, seeds=[1, 2]).num_episodes == 2
        with pytest.raises(ValueError, match="episodes must be >= 0"):
            sess.run(episodes=-1)
        with pytest.raises(ValueError, match="does not match"):
            sess.run(episodes=1, seeds=[1, 2])
        episode = sess.run(seeds=[5]).episodes[0]
        assert episode.predict_ms is not None and episode.step_ms is not None


def test_as_model_rejects_a_non_model_source() -> None:
    from rlmesh._models.base import as_model

    with pytest.raises(TypeError, match="take a Model"):
        as_model(object())
    with pytest.raises(TypeError, match="take a Model"):
        as_model(lambda obs: 0)


# ---------------------------------------------------------------------------
# instruction= override injection (placement + container shape)
# ---------------------------------------------------------------------------


def _spec(input_tree: Any) -> object:
    import rlmesh.adapters as adapt

    return adapt.ModelSpec(
        input=input_tree,
        output=adapt.Action(
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0))
        ),
    )


def test_text_placements_covers_every_placement_and_container() -> None:
    import rlmesh.adapters as adapt
    from rlmesh._models._eval import TextPlacement, text_placements

    # bare-root: the whole payload IS the text leaf (empty placement)
    assert text_placements(_spec(adapt.Text(role=adapt.INSTRUCTION))) == (
        TextPlacement((), False),
    )
    # top-level dict key, both container shapes
    assert text_placements(
        _spec({"prompt": adapt.Text(role=adapt.INSTRUCTION, container="str")})
    ) == (TextPlacement(("prompt",), False),)
    assert text_placements(
        _spec({"prompt": adapt.Text(role=adapt.INSTRUCTION, container="list")})
    ) == (TextPlacement(("prompt",), True),)
    # nested dict placement
    assert text_placements(
        _spec({"lang": {"instr": adapt.Text(role=adapt.INSTRUCTION)}})
    ) == (TextPlacement(("lang", "instr"), False),)
    # tuple placement (positional)
    assert text_placements(_spec((adapt.Text(role=adapt.INSTRUCTION),))) == (
        TextPlacement((0,), False),
    )


def test_text_placements_empty_for_specless_models() -> None:
    from rlmesh import NO_ADAPTER
    from rlmesh._models._eval import text_placements

    assert text_placements(None) == ()
    assert text_placements(NO_ADAPTER) == ()


def _inject(placements: tuple[Any, ...], payload: Any) -> Any:
    """Run _predict_step's injection (adapter=None hands the payload through)."""
    from rlmesh._models._eval import _predict_step

    captured: dict[str, Any] = {}

    def predict(p: Any) -> int:
        captured["payload"] = p
        return 0

    _predict_step(predict, payload, None, "do the task", placements, None, None, None)
    return captured["payload"]


def test_instruction_injects_into_a_bare_root_text_input() -> None:
    from rlmesh._models._eval import TextPlacement

    # The whole payload is the text leaf; the override replaces it outright.
    assert _inject((TextPlacement((), False),), "old") == "do the task"


def test_instruction_injects_into_a_nested_text_input() -> None:
    from rlmesh._models._eval import TextPlacement

    out = _inject((TextPlacement(("lang", "instr"), False),), {"lang": {"instr": "x"}})
    assert out == {"lang": {"instr": "do the task"}}


def test_instruction_injects_list_for_list_container() -> None:
    from rlmesh._models._eval import TextPlacement

    out = _inject((TextPlacement(("prompt",), True),), {"prompt": ["x"]})
    assert out == {"prompt": ["do the task"]}


def test_instruction_injection_does_not_mutate_the_source_payload() -> None:
    from rlmesh._models._eval import TextPlacement

    source = {"lang": {"instr": "x"}}
    _inject((TextPlacement(("lang", "instr"), False),), source)
    assert source == {"lang": {"instr": "x"}}  # injected into a rebuilt copy


def test_predict_failure_is_annotated_with_the_payload_signature() -> None:
    """A predict that raises gains a note naming the shapes it was handed."""
    import numpy as np
    from rlmesh._models._eval import _predict_step

    def predict(payload: Any) -> Any:
        raise RuntimeError("size mismatch")

    payload = {"image": np.zeros((8, 8, 3), dtype=np.uint8), "instruction": "pick"}
    with pytest.raises(RuntimeError, match="size mismatch") as excinfo:
        _predict_step(predict, payload, None, None, (), None, None, None)
    notes = getattr(excinfo.value, "__notes__", [])
    assert any("model input" in note and "uint8[8, 8, 3]" in note for note in notes), (
        notes
    )


# ---------------------------------------------------------------------------
# run() observability (hooks) + per-episode caps
# ---------------------------------------------------------------------------


class _Recorder(rlmesh.RunHooks):
    """Records every hook invocation in order, plus events and run results."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.events: list[rlmesh.StepEvent] = []
        self.episode_results: list[rlmesh.EpisodeResult] = []
        self.run_results: list[rlmesh.RunResult] = []

    def on_episode_start(self, *, episode: int, seed: int | None) -> None:
        self.calls.append(("start", episode, seed))

    def on_step(self, event: rlmesh.StepEvent) -> None:
        self.calls.append(("step", event.episode, event.step))
        self.events.append(event)

    def on_episode_end(self, result: rlmesh.EpisodeResult) -> None:
        self.calls.append(("end", result.index))
        self.episode_results.append(result)

    def on_run_end(self, result: rlmesh.RunResult) -> None:
        self.calls.append(("run_end", result.num_episodes))
        self.run_results.append(result)


def test_run_hooks_fire_in_order_with_indices_and_seeds() -> None:
    recorder = _Recorder()
    result = rlmesh.session(rlmesh.Model(lambda obs: 0), _TinyEnv()).run(
        seeds=[7, 8], hooks=recorder
    )

    assert recorder.calls == [
        ("start", 0, 7),
        ("step", 0, 0),
        ("end", 0),
        ("start", 1, 8),
        ("step", 1, 0),
        ("end", 1),
        ("run_end", 2),
    ]
    event = recorder.events[0]
    assert event.seed == 7
    assert event.reward == 1.0
    assert event.terminated is True
    assert event.truncated is False
    assert event.observation == 0
    assert event.info["action"] == 0
    assert event.predict_ms >= 0.0
    assert event.step_ms >= 0.0
    assert recorder.run_results == [result]
    assert recorder.episode_results[0] == result.episodes[0]
    assert result.episodes[0].duration_s > 0.0
    assert result.episodes[0].predict_ms is not None
    assert result.episodes[0].step_ms is not None


def test_max_episode_steps_caps_and_truncates_each_episode() -> None:
    result = rlmesh.Model(lambda obs: 0).run(
        _ForeverEnv(), seeds=[0, 1], max_episode_steps=3
    )

    assert result.num_episodes == 2
    assert all(e.steps == 3 for e in result.episodes)
    assert all(e.truncated and not e.terminated for e in result.episodes)


class _FakeTime:
    """A perf_counter that jumps a fixed amount per call."""

    def __init__(self, tick: float) -> None:
        self._now = 0.0
        self._tick = tick

    def perf_counter(self) -> float:
        self._now += self._tick
        return self._now


def test_max_episode_seconds_truncates_and_records_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlmesh._models._eval as eval_mod

    monkeypatch.setattr(eval_mod, "time", _FakeTime(1.0))
    result = rlmesh.session(rlmesh.Model(lambda obs: 0), _ForeverEnv()).run(
        max_episode_seconds=4.0
    )

    episode = result.episodes[0]
    assert episode.truncated is True
    assert episode.terminated is False
    assert episode.steps < 5
    assert episode.duration_s > 0.0


def test_keyboard_interrupt_propagates_and_run_end_sees_completed_episodes() -> None:
    recorder = _Recorder()
    predictions = {"count": 0}

    def predict(obs: object) -> int:
        predictions["count"] += 1
        if predictions["count"] > 1:
            raise KeyboardInterrupt
        return 0

    with pytest.raises(KeyboardInterrupt):
        rlmesh.session(rlmesh.Model(predict), _TinyEnv()).run(
            seeds=[0, 1], hooks=recorder
        )

    assert [c for c in recorder.calls if c[0] == "run_end"] == [("run_end", 1)]
    assert [c for c in recorder.calls if c[0] == "end"] == [("end", 0)]
    assert recorder.run_results[0].num_episodes == 1


def test_raising_on_step_aborts_but_on_run_end_still_fires() -> None:
    class _BoomStep(_Recorder):
        def on_step(self, event: rlmesh.StepEvent) -> None:
            raise RuntimeError("boom-step")

    recorder = _BoomStep()
    with pytest.raises(RuntimeError, match="boom-step"):
        rlmesh.session(rlmesh.Model(lambda obs: 0), _TinyEnv()).run(hooks=recorder)

    assert [c for c in recorder.calls if c[0] == "run_end"] == [("run_end", 0)]
    assert not [c for c in recorder.calls if c[0] == "end"]


def test_original_exception_wins_over_a_raising_on_run_end() -> None:
    class _BoomBoth(_Recorder):
        def on_step(self, event: rlmesh.StepEvent) -> None:
            raise RuntimeError("original")

        def on_run_end(self, result: rlmesh.RunResult) -> None:
            super().on_run_end(result)
            raise RuntimeError("masker")

    recorder = _BoomBoth()
    with pytest.raises(RuntimeError, match="original"):
        rlmesh.session(rlmesh.Model(lambda obs: 0), _TinyEnv()).run(hooks=recorder)
    assert [c for c in recorder.calls if c[0] == "run_end"] == [("run_end", 0)]


def test_a_lone_raising_on_run_end_propagates() -> None:
    class _BoomRunEnd(rlmesh.RunHooks):
        def on_run_end(self, result: rlmesh.RunResult) -> None:
            raise RuntimeError("boom-run-end")

    with pytest.raises(RuntimeError, match="boom-run-end"):
        rlmesh.session(rlmesh.Model(lambda obs: 0), _TinyEnv()).run(hooks=_BoomRunEnd())


def test_step_event_read_resolves_roles_against_a_tagged_env() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("gymnasium")
    import gymnasium as gym
    import numpy as np
    import rlmesh.adapters as adapt

    class _ArmEnv:
        def __init__(self) -> None:
            self.metadata: dict[str, Any] = {}
            self.observation_space = gym.spaces.Dict(
                {"eef_pos": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32)}
            )
            self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

        def reset(
            self, *, seed: object = None, options: object = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            return {"eef_pos": np.array([0.1, 0.2, 0.3], np.float32)}, {}

        def step(
            self, action: object
        ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
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

    class _ReadHook(rlmesh.RunHooks):
        def __init__(self) -> None:
            self.values: list[Any] = []

        def on_step(self, event: rlmesh.StepEvent) -> None:
            self.values.append(event.read(adapt.EEF_POS))

    hook = _ReadHook()
    rlmesh.run(rlmesh.RANDOM_SAMPLE, adapt.tag(_ArmEnv(), tags), hooks=hook)
    assert len(hook.values) == 1
    assert hook.values[0].shape == (3,)


def test_step_event_read_is_lazy_and_never_resolves_unless_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlmesh._models._eval as eval_mod

    monkeypatch.setattr(
        eval_mod,
        "resolve_read_adapter",
        lambda *_a, **_kw: pytest.fail("read resolution must stay lazy"),
    )
    recorder = _Recorder()
    result = rlmesh.session(rlmesh.Model(lambda obs: 0), _TinyEnv()).run(hooks=recorder)
    assert result.num_episodes == 1
    assert len(recorder.events) == 1


def test_invalid_caps_raise_value_error() -> None:
    sess = rlmesh.session(rlmesh.Model(lambda obs: 0), _TinyEnv())
    with pytest.raises(ValueError, match="max_episode_steps"):
        sess.run(max_episode_steps=0)
    with pytest.raises(ValueError, match="max_episode_seconds"):
        sess.run(max_episode_seconds=0.0)
    with pytest.raises(ValueError, match="max_episode_seconds"):
        sess.run(max_episode_seconds=-1.0)
    sess.close()


def test_invalid_execution_horizon_raises_value_error_at_entry() -> None:
    # 0 / negative used to silently run un-chunked via max(1, h); now the session
    # entry rejects them outright.
    with pytest.raises(ValueError, match="execution_horizon must be >= 1"):
        rlmesh.session(rlmesh.Model(lambda obs: 0), _TinyEnv(), execution_horizon=0)
    with pytest.raises(ValueError, match="execution_horizon must be >= 1"):
        rlmesh.run(rlmesh.Model(lambda obs: 0), _TinyEnv(), execution_horizon=-2)


def test_negative_prefetch_lead_is_rejected_at_entry() -> None:
    # The lead is a frame count; the native loop treats 0 as synchronous and
    # anything else as async, so a negative one is a mis-set knob.
    with pytest.raises(ValueError, match="prefetch_lead must be >= 0"):
        rlmesh.run(rlmesh.Model(lambda obs: 0), _TinyEnv(), prefetch_lead=-1)
    with pytest.raises(ValueError, match="prefetch_lead must be >= 0"):
        rlmesh.Model(lambda obs: 0).run(_TinyEnv(), prefetch_lead=-1)


def test_prefetch_lead_is_refused_on_the_session_loop() -> None:
    # RANDOM_SAMPLE (like a served handle) runs through the Python session loop,
    # which steps in Python and has no chunk replay to prefetch over.
    with pytest.raises(ValueError, match="prefetch_lead=1 drives the native runtime"):
        rlmesh.run(rlmesh.RANDOM_SAMPLE, _TinyEnv(), prefetch_lead=1)


def test_execution_horizon_over_the_bound_is_rejected() -> None:
    # The ceiling is the SDK twin of the engine's MAX_EXECUTION_HORIZON: above it
    # a horizon is always a mis-set knob, not a real open-loop plan.
    with pytest.raises(ValueError, match="execution_horizon must be <= 1024"):
        rlmesh.session(rlmesh.Model(lambda obs: 0), _TinyEnv(), execution_horizon=1025)


def test_execution_horizon_over_the_declared_chunk_is_rejected() -> None:
    # Declaring K is a promise the session holds the caller to: the model cannot
    # produce more than K actions per predict, so a larger horizon is refused up
    # front rather than silently short-replaying every step.
    class _Chunky(rlmesh.Model):
        native_chunk = 4

        def predict(self, observation: object) -> int:
            return 0

        def predict_chunk(self, observation: object) -> list[int]:
            return [0, 1, 2, 3]

    with pytest.raises(ValueError, match="native_chunk=4"):
        _Chunky().session(_ForeverEnv(), execution_horizon=8)


def test_a_declared_model_that_slices_its_own_chunk_fails_the_replay() -> None:
    # The glue this contract removes: returning chunk[:horizon] from a model that
    # declares K short-replays silently. Declared, it raises instead.
    from rlmesh._models._chunk import ChunkReplay

    replay = ChunkReplay(4, native_chunk=10)
    with pytest.raises(ValueError, match="native_chunk=10"):
        replay.next_action(lambda: [0, 1, 2, 3])
    # The whole native chunk is accepted and the horizon prefix replayed.
    assert ChunkReplay(4, native_chunk=10).next_action(lambda: list(range(10))) == 0


def test_an_undeclared_short_chunk_warns_once() -> None:
    import warnings as warnings_mod

    from rlmesh._models._chunk import ChunkReplay

    replay = ChunkReplay(6)
    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter("always")
        assert replay.next_action(lambda: [0, 1]) == 0
        assert replay.next_action(lambda: [0, 1]) == 1  # replayed, no predict
        assert replay.next_action(lambda: [0, 1]) == 0  # re-plans, still short
    shorts = [w for w in caught if "execution_horizon=6" in str(w.message)]
    assert len(shorts) == 1
    assert issubclass(shorts[0].category, RuntimeWarning)


def test_unchunked_warning_points_at_the_caller() -> None:
    import warnings as warnings_mod

    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter("always")
        sess = rlmesh.Model(lambda obs: 0).session(_TinyEnv(), execution_horizon=4)
    (warning,) = [w for w in caught if "running un-chunked" in str(w.message)]
    assert warning.filename == __file__  # stacklevel reaches the caller's frame
    sess.close()


def test_hud_chunk_length_tracks_the_actual_chunk_not_the_horizon() -> None:
    # A model whose native chunk (4) is shorter than execution_horizon (8) must
    # display 1/4..4/4, not 5/8: the HUD reads the queued chunk's real length.
    class _Chunky:
        def predict(self, observation: object) -> int:
            return 0

        def predict_chunk(self, observation: object) -> list[int]:
            return [0, 1, 2, 3]

    sess = rlmesh.session(
        rlmesh.numpy.Model(_Chunky()), _ForeverEnv(), execution_horizon=8
    )
    obs, _ = sess.reset()
    positions: list[tuple[int, int]] = []
    for _step in range(5):
        action = sess.predict(obs)
        positions.append((sess._chunk_pos, sess._chunk_len))
        obs, *_rest = sess.step(action)
    assert positions == [(1, 4), (2, 4), (3, 4), (4, 4), (1, 4)]
    sess.close()


def test_tree_set_preserves_tuple_payloads() -> None:
    from rlmesh._models._instruction import tree_set

    out = tree_set(("keep", {"instr": "x"}), (1, "instr"), "do the task")
    assert out == ("keep", {"instr": "do the task"})
    assert isinstance(out, tuple)


class _ScalarActionEnv:
    """A local env with a scalar (Discrete) action space, recording what it got."""

    def __init__(self) -> None:
        from rlmesh import spaces

        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(2)
        self.actions: list[object] = []

    def reset(
        self, *, seed: object = None, options: object = None
    ) -> tuple[int, dict[str, object]]:
        self.actions = []
        return 0, {}

    def step(self, action: object) -> tuple[int, float, bool, bool, dict[str, object]]:
        self.actions.append(action)
        return 0, 1.0, len(self.actions) >= 8, False, {}

    def close(self) -> None:
        pass


def test_numpy_chunk_runs_over_a_scalar_action_space() -> None:
    # A chunk corner returning a 1-D numpy chunk unstacks into rank-0 rows; a
    # Discrete space takes numbers, so those rows must arrive as scalars rather
    # than rank-0 tensors (which used to raise "must be real number" at
    # execution_horizon >= 2).
    import numpy as np

    class _NumpyChunky:
        def predict(self, observation: object) -> int:
            return 0

        def predict_chunk(self, observation: object, execution_horizon: int = 1) -> Any:
            return np.zeros(execution_horizon, dtype=np.int64)

    env = _ScalarActionEnv()
    result = rlmesh.run(
        rlmesh.numpy.Model(_NumpyChunky()), env, episodes=1, execution_horizon=4
    )
    assert result.num_episodes == 1
    assert len(env.actions) == 8
    assert all(isinstance(action, int) for action in env.actions)


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        ({"is_success": 0, "success": True}, False),
        ({"success": 1}, True),
        ({"task_success": 1.0}, True),
        ({"task_success": 0}, False),
        ({"other": True}, None),
    ],
)
def test_episode_success_reads_the_same_keys_as_the_runtime(
    info: dict[str, Any], expected: bool | None
) -> None:
    """The Python loop and the Rust runtime must agree on which final-step
    info keys carry the task outcome (is_success, success, task_success)."""
    from rlmesh._models._eval import _episode_success

    assert _episode_success(info) is expected


def test_closed_model_is_collectable_after_a_native_run() -> None:
    import gc
    import weakref

    model = rlmesh.Model(lambda obs: 0)
    model.run(_TinyEnv(), episodes=1)
    ref = weakref.ref(model)
    model.close()
    del model
    gc.collect()

    assert ref() is None


class _ClosingEnv(_TinyEnv):
    def __init__(self, closed: list[int]) -> None:
        super().__init__()
        self._closed = closed

    def close(self) -> None:
        self._closed.append(1)


def _closing_factory(closed: list[int]) -> rlmesh.EnvFactory:
    class Factory(rlmesh.EnvFactory):
        def make(self) -> _ClosingEnv:
            return _ClosingEnv(closed)

    return Factory()


def test_run_closes_a_factory_env_when_adapter_resolution_fails() -> None:
    from rlmesh.adapters import AdapterResolutionError

    closed: list[int] = []
    model = rlmesh.Model(lambda obs: 0, spec=cast(Any, object()))
    with pytest.raises(AdapterResolutionError):
        model.run(_closing_factory(closed), episodes=1)

    assert closed == [1]


def test_session_closes_a_factory_env_when_adapter_resolution_fails() -> None:
    from rlmesh.adapters import AdapterResolutionError

    closed: list[int] = []
    model = rlmesh.Model(lambda obs: 0, spec=cast(Any, object()))
    with (
        pytest.raises(AdapterResolutionError),
        model.session(_closing_factory(closed)) as session,
    ):
        session.reset()

    assert closed == [1]


def test_session_close_releases_the_env_when_episode_end_raises() -> None:
    def fail(*_: object) -> None:
        raise ValueError("end hook failed")

    closed: list[int] = []
    model = rlmesh.Model(lambda obs: 0, on_episode_end=fail)
    session = model.session(_closing_factory(closed))
    session.reset()
    with pytest.raises(ValueError, match="end hook failed"):
        session.close()
    session.close()

    assert closed == [1]


@pytest.mark.parametrize(("shape", "dtype"), [((2, 3), "float32"), ((6,), "float64")])
def test_adapted_session_and_run_hand_the_env_its_box_action(
    shape: tuple[int, ...], dtype: str
) -> None:
    np = pytest.importorskip("numpy")
    import rlmesh.adapters as adapt
    from rlmesh import spaces

    class Env:
        def __init__(self) -> None:
            self.observation_space = spaces.Dict(
                {"q": spaces.Box(-np.inf, np.inf, (6,))}
            )
            self.action_space = spaces.Box(-1, 1, shape, dtype=dtype)
            self.seen: list[tuple[tuple[int, ...], str]] = []

        def reset(self, *, seed: object = None, options: object = None) -> object:
            return {"q": np.zeros(6, np.float32)}, {}

        def step(self, action: Any) -> object:
            self.seen.append((tuple(action.shape), str(action.dtype)))
            return {"q": np.zeros(6, np.float32)}, 0.0, True, False, {}

        def close(self) -> None:
            pass

    action = adapt.Action(
        adapt.Actuator(adapt.ACTION_JOINT_POS, dim=6, labels=adapt.UR5E.joints)
    )
    spec = adapt.ModelSpec(
        input={"q": adapt.State(adapt.JOINT_POS, labels=adapt.UR5E.joints)},
        output=action,
    )
    tags = adapt.EnvTags(
        observation={"q": adapt.StateTag(adapt.JOINT_POS, labels=adapt.UR5E.joints)},
        action=action,
    )

    def model() -> rlmesh.numpy.Model[Any, Any]:
        return rlmesh.numpy.Model(
            lambda payload: np.arange(6, dtype=np.float32) / 10.0, spec=spec
        )

    session_env = adapt.tag(Env(), tags)
    with model().session(session_env) as session:
        session.run(episodes=1)
    run_env = adapt.tag(Env(), tags)
    model().run(run_env, episodes=1)

    assert session_env.seen == run_env.seen == [(shape, dtype)]


def test_an_authored_close_fires_once_and_drops_the_worker() -> None:
    import gc
    import weakref

    calls: list[str] = []

    class WithSuper(rlmesh.Model[Any, Any]):
        def predict(self, obs: Any) -> Any:
            return 0

        def close(self) -> None:
            calls.append("with")
            super().close()

    class WithoutSuper(rlmesh.Model[Any, Any]):
        def predict(self, obs: Any) -> Any:
            return 0

        def close(self) -> None:
            calls.append("without")

    for cls, expected in ((WithSuper, ["with"]), (WithoutSuper, ["without"])):
        calls.clear()
        model = cls()
        model.run(_TinyEnv(), episodes=1)
        ref = weakref.ref(model)
        model.close()
        assert calls == expected
        del model
        gc.collect()
        assert ref() is None, cls.__name__


@pytest.mark.parametrize("value", [0.75, 130.0])
def test_adapted_session_refuses_an_action_the_env_int_box_cannot_hold(
    value: float,
) -> None:
    np = pytest.importorskip("numpy")
    import rlmesh.adapters as adapt
    from rlmesh import spaces

    class Env:
        def __init__(self) -> None:
            self.observation_space = spaces.Dict(
                {"q": spaces.Box(-np.inf, np.inf, (6,))}
            )
            self.action_space = spaces.Box(-128, 127, (6,), dtype="int8")

        def reset(self, *, seed: object = None, options: object = None) -> object:
            return {"q": np.zeros(6, np.float32)}, {}

        def step(self, action: Any) -> object:
            raise AssertionError(f"env stepped with {action!r}")

        def close(self) -> None:
            pass

    action = adapt.Action(
        adapt.Actuator(adapt.ACTION_JOINT_POS, dim=6, labels=adapt.UR5E.joints)
    )
    spec = adapt.ModelSpec(
        input={"q": adapt.State(adapt.JOINT_POS, labels=adapt.UR5E.joints)},
        output=action,
    )
    tags = adapt.EnvTags(
        observation={"q": adapt.StateTag(adapt.JOINT_POS, labels=adapt.UR5E.joints)},
        action=action,
    )
    model = rlmesh.numpy.Model(
        lambda payload: np.full(6, value, dtype=np.float32), spec=spec
    )
    with model.session(adapt.tag(Env(), tags)) as session:
        obs, _ = session.reset()
        with pytest.raises(ValueError):
            session.predict(obs)
    with pytest.raises(Exception):
        model.run(adapt.tag(Env(), tags), episodes=1)
