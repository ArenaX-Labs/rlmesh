"""A model driven on an env: rlmesh.run / Model.run, and Model.bind by hand.

These pin both against a tiny local env: how a model is bound to an env and
driven, whole runs and by hand (reset/step). The served and adapted loops are
exercised in the integration suite.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
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
    with pytest.raises(TypeError, match=r"takes a Model.*rlmesh\.numpy\.Model"):
        rlmesh.run(lambda obs: 0, _TinyEnv())
    result = rlmesh.run(rlmesh.numpy.Model(lambda obs: 0), _TinyEnv())
    assert result.num_episodes == 1


def test_a_binding_steps_the_env_by_hand() -> None:
    binding = rlmesh.Model(lambda obs: 0).bind(_TinyEnv())
    assert isinstance(binding, rlmesh.Binding)
    assert binding.done is True  # no episode yet
    _obs, info = binding.reset(seed=0)
    assert info["seed"] == 0
    assert binding.done is False
    _obs, reward, terminated, _trunc, info = binding.step()
    assert (reward, terminated, info["action"]) == (1.0, True, 0)
    assert binding.done is True  # the env terminated after one step
    with pytest.raises(RuntimeError, match="reset to start the next"):
        binding.step()
    binding.reset()
    assert binding.step(0)[2] is True  # the caller's action
    binding.close()


def test_ctrl_c_interrupts_a_step_the_env_is_stuck_in() -> None:
    import _thread
    import threading

    release = threading.Event()

    class _StuckEnv(_TinyEnv):
        def step(
            self, action: object
        ) -> tuple[int, float, bool, bool, dict[str, object]]:
            release.wait(10)
            return super().step(action)

    with rlmesh.Model(lambda obs: 0).bind(_StuckEnv()) as binding:
        binding.reset()
        timer = threading.Timer(0.2, _thread.interrupt_main)
        timer.start()
        try:
            with pytest.raises(KeyboardInterrupt):
                binding.step()
        finally:
            release.set()
            timer.join()


def test_a_predict_exception_keeps_its_own_type() -> None:
    def predict(obs: object) -> int:
        raise KeyError("missing_key")

    with rlmesh.Model(predict).bind(_TinyEnv()) as binding:
        binding.reset()
        with pytest.raises(KeyError, match="missing_key"):
            binding.step()


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

    def pace(self) -> float:
        return 0.0

    def close(self, *, hold: bool = True) -> None:
        pass


class _QuitDriver(_SkipDriver):
    """A stand-in ViewerDriver that requests quit after the first fed step."""

    def consume_skip(self) -> bool:
        return False

    def quit_requested(self) -> bool:
        return self._steps >= 1


def test_viewer_skip_truncates_episode_without_failing() -> None:
    binding = rlmesh.Model(lambda obs: 0).bind(_ForeverEnv())
    binding._view = cast(ViewerDriver, _SkipDriver())
    binding.reset(seed=0)
    _obs, _reward, terminated, truncated, _info = binding.step()
    assert truncated is True
    assert terminated is False
    assert binding.done is True
    binding.close()


def test_viewer_quit_truncates_the_episode() -> None:
    binding = rlmesh.Model(lambda obs: 0).bind(_ForeverEnv())
    binding._view = cast(ViewerDriver, _QuitDriver())
    binding.reset()
    assert binding.step()[3] is True
    assert binding.done is True
    binding.close()


def test_a_binding_is_a_context_manager() -> None:
    with rlmesh.Model(lambda obs: 0).bind(_TinyEnv()) as binding:
        binding.reset()
        binding.step()
        assert binding.done is True


def test_a_closed_binding_rejects_any_further_use() -> None:
    binding = rlmesh.Model(lambda obs: 0).bind(_TinyEnv())
    binding.reset()
    binding.close()
    binding.close()  # idempotent
    with pytest.raises(RuntimeError, match="binding is closed"):
        binding.reset()
    with pytest.raises(RuntimeError, match="binding is closed"):
        binding.step(0)


def test_closing_a_binding_leaves_its_model_open() -> None:
    closes: list[int] = []
    model = rlmesh.Model(lambda obs: 0, on_close=lambda: closes.append(1))
    with model.bind(_TinyEnv()) as binding:
        binding.reset()
        binding.step()
    assert closes == []
    with model.bind(_TinyEnv()) as binding:  # still usable
        binding.reset()
    model.close()
    assert closes == [1]


def test_one_shot_run_borrows_a_model_instance() -> None:
    closes: list[int] = []
    model = rlmesh.Model(lambda obs: 0, on_close=lambda: closes.append(1))
    rlmesh.run(model, _TinyEnv())
    model.run(_TinyEnv())
    assert closes == []
    model.close()
    assert closes == [1]


def test_run_and_bind_release_a_factory_built_env() -> None:
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
    with model.bind(_Factory()) as binding:
        binding.reset()
    assert closed == [1, 1]
    # A caller's env is borrowed: closed only on the opt-in.
    env = _Env()
    model.run(env)
    with model.bind(env) as binding:
        binding.reset()
    assert closed == [1, 1]
    model.run(env, close_env=True)
    with model.bind(env, close_env=True) as binding:
        binding.reset()
    assert closed == [1, 1, 1, 1]
    # Opting a factory-built env into shutdown closes it once, not twice.
    with model.bind(_Factory(), close_env=True) as binding:
        binding.reset()
    model.run(_Factory(), close_env=True)
    assert closed == [1, 1, 1, 1, 1, 1]


def test_as_model_rejects_a_non_model_source() -> None:
    from rlmesh._models.base import as_model

    with pytest.raises(TypeError, match="takes a Model"):
        as_model(object())
    with pytest.raises(TypeError, match="takes a Model"):
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
    from rlmesh._models._instruction import TextPlacement, text_placements

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
    from rlmesh._models._instruction import text_placements

    assert text_placements(None) == ()
    assert text_placements(NO_ADAPTER) == ()


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


def test_binding_hooks_fire_in_order_with_indices_and_seeds() -> None:
    recorder = _Recorder()
    with rlmesh.Model(lambda obs: 0).bind(_TinyEnv(), hooks=recorder) as binding:
        for seed in (7, 8):
            binding.reset(seed=seed)
            binding.step()
        binding.reset(seed=9)  # closed before it ends: truncated

    assert recorder.calls == [
        ("start", 0, 7),
        ("step", 0, 0),
        ("end", 0),
        ("start", 1, 8),
        ("step", 1, 0),
        ("end", 1),
        ("start", 2, 9),
        ("end", 2),
        ("run_end", 3),
    ]
    event = recorder.events[0]
    assert event.seed == 7
    assert event.reward == 1.0
    assert event.terminated is True
    assert event.truncated is False
    assert event.observation == 0
    assert event.action == 0
    assert event.info["action"] == 0
    assert event.predict_ms >= 0.0
    assert event.step_ms >= 0.0
    first, _, last = recorder.episode_results
    assert recorder.run_results[0].episodes == tuple(recorder.episode_results)
    assert first.duration_s > 0.0 and first.predict_ms is not None
    assert last.truncated is True and last.steps == 0


def test_max_episode_steps_caps_and_truncates_each_episode() -> None:
    result = rlmesh.Model(lambda obs: 0).run(
        _ForeverEnv(), seeds=[0, 1], max_episode_steps=3
    )

    assert result.num_episodes == 2
    assert all(e.steps == 3 for e in result.episodes)
    assert all(e.truncated and not e.terminated for e in result.episodes)


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
    import rlmesh._models._binding as binding_mod

    monkeypatch.setattr(
        binding_mod,
        "resolve_read_adapter",
        lambda *_a, **_kw: pytest.fail("read resolution must stay lazy"),
    )
    recorder = _Recorder()
    with rlmesh.Model(lambda obs: 0).bind(_TinyEnv(), hooks=recorder) as binding:
        binding.reset()
        binding.step()
    assert len(recorder.events) == 1


def test_invalid_execution_horizon_raises_value_error_at_entry() -> None:
    with pytest.raises(ValueError, match="execution_horizon must be >= 1"):
        rlmesh.Model(lambda obs: 0).bind(_TinyEnv(), execution_horizon=0)
    with pytest.raises(ValueError, match="execution_horizon must be >= 1"):
        rlmesh.run(rlmesh.Model(lambda obs: 0), _TinyEnv(), execution_horizon=-2)


def test_negative_prefetch_lead_is_rejected_at_entry() -> None:
    # The lead is a frame count; the native loop treats 0 as synchronous and
    # anything else as async, so a negative one is a mis-set knob.
    with pytest.raises(ValueError, match="prefetch_lead must be >= 0"):
        rlmesh.run(rlmesh.Model(lambda obs: 0), _TinyEnv(), prefetch_lead=-1)
    with pytest.raises(ValueError, match="prefetch_lead must be >= 0"):
        rlmesh.Model(lambda obs: 0).run(_TinyEnv(), prefetch_lead=-1)


def test_execution_horizon_over_the_bound_is_rejected() -> None:
    # The ceiling is the SDK twin of the engine's MAX_EXECUTION_HORIZON: above it
    # a horizon is always a mis-set knob, not a real open-loop plan.
    with pytest.raises(ValueError, match="execution_horizon must be <= 1024"):
        rlmesh.Model(lambda obs: 0).bind(_TinyEnv(), execution_horizon=1025)


def test_execution_horizon_over_the_declared_chunk_is_rejected() -> None:
    # Declaring K is a promise the binding holds the caller to: the model cannot
    # produce more than K actions per predict, so a larger horizon is refused up
    # front rather than silently short-replaying every step.
    class _Chunky(rlmesh.Model):
        native_chunk = 4

        def predict(self, observation: object) -> int:
            return 0

        def predict_chunk(self, observation: object) -> list[int]:
            return [0, 1, 2, 3]

    with pytest.raises(ValueError, match="native_chunk=4"):
        _Chunky().bind(_ForeverEnv(), execution_horizon=8)


def test_unchunked_warning_points_at_the_caller() -> None:
    import warnings as warnings_mod

    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter("always")
        binding = rlmesh.Model(lambda obs: 0).bind(_TinyEnv(), execution_horizon=4)
    (warning,) = [w for w in caught if "running un-chunked" in str(w.message)]
    assert warning.filename == __file__  # stacklevel reaches the caller's frame
    binding.close()


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
        # Kinds the runtime skips never decide; the next key does.
        ({"is_success": "False", "success": True}, True),
        ({"is_success": "True"}, None),
        ({"is_success": None, "success": False}, False),
        ({"is_success": [1], "task_success": 0.0}, False),
        ({"success": b"\x01"}, None),
        # NumPy scalars and 0-d arrays unwrap as they do on the wire.
        ({"is_success": np.bool_(True)}, True),
        ({"is_success": np.int64(0)}, False),
        ({"success": np.float32(0.5)}, True),
        ({"success": np.array(True)}, True),
        ({"success": np.array([True]), "task_success": 1}, True),
    ],
)
def test_episode_success_reads_the_same_keys_as_the_runtime(
    info: dict[str, Any], expected: bool | None
) -> None:
    """The Python loop and the Rust runtime must agree on which final-step
    info keys carry the task outcome (is_success, success, task_success)."""
    from rlmesh._models._eval import episode_success

    assert episode_success(info) is expected


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


def test_bind_closes_a_factory_env_when_adapter_resolution_fails() -> None:
    from rlmesh.adapters import AdapterResolutionError

    closed: list[int] = []
    model = rlmesh.Model(lambda obs: 0, spec=cast(Any, object()))
    with pytest.raises(AdapterResolutionError):
        model.bind(_closing_factory(closed))

    assert closed == [1]


def test_closing_a_binding_releases_the_env_when_episode_end_raises() -> None:
    def fail(*_: object) -> None:
        raise ValueError("end hook failed")

    closed: list[int] = []
    model = rlmesh.Model(lambda obs: 0, on_episode_end=fail)
    binding = model.bind(_closing_factory(closed))
    binding.reset()
    with pytest.raises(Exception, match="end hook failed"):
        binding.close()
    binding.close()

    assert closed == [1]


_UR5E_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


@pytest.mark.parametrize(("shape", "dtype"), [((2, 3), "float32"), ((6,), "float64")])
def test_adapted_bind_and_run_hand_the_env_its_box_action(
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
        adapt.Actuator(adapt.ACTION_JOINT_POS, dim=6, labels=_UR5E_JOINTS)
    )
    spec = adapt.ModelSpec(
        input={"q": adapt.State(adapt.JOINT_POS, labels=_UR5E_JOINTS)},
        output=action,
    )
    tags = adapt.EnvTags(
        observation={"q": adapt.StateTag(adapt.JOINT_POS, labels=_UR5E_JOINTS)},
        action=action,
    )

    def model() -> rlmesh.numpy.Model[Any, Any]:
        return rlmesh.numpy.Model(
            lambda payload: np.arange(6, dtype=np.float32) / 10.0, spec=spec
        )

    bound_env = adapt.tag(Env(), tags)
    with model().bind(bound_env) as binding:
        binding.reset()
        binding.step()
    run_env = adapt.tag(Env(), tags)
    model().run(run_env, episodes=1)

    assert bound_env.seen == run_env.seen == [(shape, dtype)]


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
def test_adapted_bind_and_run_refuse_an_action_the_env_int_box_cannot_hold(
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
        adapt.Actuator(adapt.ACTION_JOINT_POS, dim=6, labels=_UR5E_JOINTS)
    )
    spec = adapt.ModelSpec(
        input={"q": adapt.State(adapt.JOINT_POS, labels=_UR5E_JOINTS)},
        output=action,
    )
    tags = adapt.EnvTags(
        observation={"q": adapt.StateTag(adapt.JOINT_POS, labels=_UR5E_JOINTS)},
        action=action,
    )
    model = rlmesh.numpy.Model(
        lambda payload: np.full(6, value, dtype=np.float32), spec=spec
    )
    with model.bind(adapt.tag(Env(), tags)) as binding:
        binding.reset()
        with pytest.raises(Exception):
            binding.step()
    with pytest.raises(Exception):
        model.run(adapt.tag(Env(), tags), episodes=1)


def test_random_sample_seeds_its_sampler_with_each_episode_seed() -> None:
    # SPC-P8: a seeded episode seeds RANDOM_SAMPLE's action sampler, so the same
    # seed reproduces the same actions.
    from rlmesh import spaces

    class _WideEnv(_ForeverEnv):
        def __init__(self) -> None:
            super().__init__()
            self.action_space = spaces.Discrete(1_000_000)
            self.actions: list[object] = []

        def step(
            self, action: object
        ) -> tuple[int, float, bool, bool, dict[str, object]]:
            self.actions.append(action)
            return super().step(action)

    def actions(seed: int) -> list[object]:
        env = _WideEnv()
        with rlmesh.Model(rlmesh.RANDOM_SAMPLE).bind(env) as binding:
            binding.reset(seed=seed)
            for _ in range(4):
                binding.step()
        return env.actions

    assert actions(11) == actions(11)
    assert actions(11) != actions(12)


@pytest.mark.parametrize("operation", ["run", "model_run", "model_bind"])
@pytest.mark.parametrize(
    ("entrypoint", "cause_type"),
    [
        ("_rlmesh_missing_entrypoint_module:transform", ImportError),
        ("builtins:_rlmesh_missing_transform", AttributeError),
    ],
)
def test_model_run_and_bind_wrap_custom_entrypoint_lookup_failure(
    operation: str, entrypoint: str, cause_type: type[Exception]
) -> None:
    import rlmesh.adapters as adapt

    class TaggedEnv(_TinyEnv):
        def __init__(self) -> None:
            self.observation_space = rlmesh.spaces.Box(-1, 1, [1])
            self.action_space = rlmesh.spaces.Box(-1, 1, [1])

    action = adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1))
    env = adapt.tag(
        TaggedEnv(),
        adapt.EnvTags(observation=adapt.StateTag(adapt.JOINT_POS), action=action),
    )
    spec = adapt.ModelSpec(input=adapt.Custom(entrypoint=entrypoint), output=action)
    model = rlmesh.Model(lambda obs: 0, spec=spec, trust_entrypoints=True)
    try:
        with pytest.raises(adapt.AdapterResolutionError, match=entrypoint) as caught:
            if operation == "run":
                rlmesh.run(model, env)
            elif operation == "model_run":
                model.run(env)
            else:
                model.bind(env).close()
        assert isinstance(caught.value.__cause__, cause_type)
    finally:
        model.close()


def test_a_binding_replays_a_chunk_and_ends_each_episode_on_the_model() -> None:
    calls: list[str] = []

    class Chunky(rlmesh.Model):
        def predict(self, observation: object) -> int:
            return 0

        def predict_chunk(self, observation: object) -> list[int]:
            calls.append("chunk")
            return [0, 0, 0]

        def on_episode_end(self, episode_id: str = "") -> None:
            calls.append("end")

    with Chunky().bind(_ForeverEnv(), execution_horizon=3) as binding:
        binding.reset()
        for _ in range(4):
            binding.step()
        binding.step(0)  # the caller's action drops the rest of the chunk
        binding.step()
        binding.reset()  # ends the first episode on the model
    assert calls == ["chunk", "chunk", "chunk", "end", "end"]


def test_reset_options_reach_the_env_under_the_callers() -> None:
    received: list[object] = []

    class OptionsEnv(_TinyEnv):
        def reset(
            self, *, seed: object = None, options: object = None
        ) -> tuple[int, dict[str, object]]:
            received.append(options)
            return 0, {}

    class Perturb(rlmesh.Model):
        def predict(self, observation: object) -> int:
            return 0

        def reset_options(self, episode: Any) -> dict[str, Any]:
            return {"friction": 0.5, "mass": 1.0}

    with Perturb().bind(OptionsEnv()) as binding:
        binding.reset(options={"mass": 2.0})
    assert received == [{"friction": 0.5, "mass": 2.0}]


def test_a_binding_drives_one_lane() -> None:
    class VectorEnv:
        num_envs = 2

        def __init__(self) -> None:
            from rlmesh import spaces

            self.single_observation_space = spaces.Discrete(1)
            self.single_action_space = spaces.Discrete(1)

        def reset(self, *, seed: object = None, options: object = None) -> object:
            return [0, 0], {}

        def step(self, actions: object) -> object:
            return [0, 0], [0.0, 0.0], [True, True], [False, False], {}

        def close(self) -> None:
            pass

    with pytest.raises(ValueError, match="drives one env lane"):
        rlmesh.Model(lambda obs: 0).bind(VectorEnv())


def test_the_0_1_session_shapes_name_bind() -> None:
    model = rlmesh.Model(lambda obs: 0)
    for call in (
        lambda: model.session(_TinyEnv()),
        lambda: rlmesh.session(model, _TinyEnv()),
        lambda: rlmesh.Session(),
        lambda: rlmesh.SandboxModel("image://m:latest").session(_TinyEnv()),
    ):
        with pytest.raises(TypeError, match=r"bind\(target\)"):
            call()
