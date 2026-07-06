"""The neutral pair-driver: rlmesh.run / rlmesh.session / Session.

These pin the Session seam against a tiny in-process env (no server): how a model is
bound to an env and driven, both auto (run) and by hand (reset/predict/step). The
full adapter/remote loop is exercised in the integration suite.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import rlmesh
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


def test_run_accepts_a_bare_callable() -> None:
    result = rlmesh.run(lambda obs: 0, _TinyEnv())
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
        sess.run(max_episodes=1)
    with pytest.raises(RuntimeError, match="session is closed"):
        sess.reset()
    with pytest.raises(RuntimeError, match="session is closed"):
        sess.predict(obs)
    with pytest.raises(RuntimeError, match="session is closed"):
        sess.step(0)
    with pytest.raises(RuntimeError, match="session is closed"):
        sess.read(obs, "state/eef_pos")


def test_session_close_fires_model_on_close_exactly_once() -> None:
    # The model's close() hook fires from Session.close() -- so a hand-driven
    # `with model.session(env)` block fires it too -- and idempotently.
    closes: list[int] = []
    model = rlmesh.Model(lambda obs: 0, on_close=lambda: closes.append(1))
    with rlmesh.session(model, _TinyEnv()) as sess:
        sess.run(seeds=[0])
        assert closes == []  # run() on a caller-held session does not close
    assert closes == [1]
    sess.close()
    assert closes == [1]  # idempotent


def test_one_shot_run_still_closes_everything() -> None:
    # rlmesh.run / Model.run create their session internally and close it when the
    # run ends, firing the model's close() once per one-shot run.
    closes: list[int] = []
    model = rlmesh.Model(lambda obs: 0, on_close=lambda: closes.append(1))
    rlmesh.run(model, _TinyEnv())
    assert closes == [1]
    model.run(_TinyEnv())
    assert closes == [1, 1]


def test_as_model_rejects_a_non_model_source() -> None:
    from rlmesh._models.base import as_model

    with pytest.raises(TypeError, match="predict callable or a policy object"):
        as_model(object())


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
    assert result.episodes[0].predict_ms >= 0.0
    assert result.episodes[0].step_ms >= 0.0


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

    sess = rlmesh.session(_Chunky(), _ForeverEnv(), execution_horizon=8)
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
