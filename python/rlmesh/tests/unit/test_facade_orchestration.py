"""Language glue preserves caller context around the shared Rust run/bind core."""

from __future__ import annotations

import threading
from typing import Any, cast

import numpy as np
import pytest
import rlmesh
from rlmesh import spaces
from rlmesh._models._view import ViewerDriver


class Env:
    render_mode = "rgb_array"

    def __init__(self) -> None:
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(1)
        self.calls: list[tuple[str, int]] = []
        self.frame = np.zeros((2, 2, 3), np.uint8)

    def reset(self, **kwargs: object) -> tuple[int, dict[str, object]]:
        self.calls.append(("reset", threading.get_ident()))
        # A sync call on another model must be legal from a caller-side env
        # callback (outside the facade's async driving context).
        assert rlmesh.Model(lambda obs: 0).run(self, episodes=0).num_episodes == 0
        return 0, {"nested": {"flag": True, "large": 2**60, "raw": b"\x00x"}}

    def step(self, action: object) -> tuple[int, float, bool, bool, dict[str, object]]:
        self.calls.append(("step", threading.get_ident()))
        return 0, 1.0, True, False, {"nested": {"flag": True, "large": 2**60}}

    def render(self) -> object:
        self.calls.append(("render", threading.get_ident()))
        return self.frame

    def close(self) -> None:
        self.calls.append(("close", threading.get_ident()))


@pytest.mark.parametrize("path", ["run", "bind"])
def test_local_env_and_hook_callbacks_keep_the_callers_thread(path: str) -> None:
    env = Env()
    caller = threading.get_ident()
    contexts: list[Any] = []

    class Hooks(rlmesh.RunHooks):
        def on_run_start(self, context: rlmesh.RunContext) -> None:
            contexts.append(context)

        def on_episode_start(self, **kwargs: Any) -> None:
            render = contexts[0].frame_sources().render
            assert render is not None
            assert render() is env.frame

        def on_step(self, event: rlmesh.StepEvent) -> None:
            assert threading.get_ident() == caller
            assert event.info["nested"] == {"flag": True, "large": 2**60}
            assert type(event.info["nested"]["flag"]) is bool
            assert type(event.info["nested"]["large"]) is int

    model = rlmesh.Model(lambda obs: 0)
    if path == "run":
        model.run(env, hooks=Hooks(), close_env=True)
    else:
        with model.bind(env, hooks=Hooks(), close_env=True) as binding:
            _, info = binding.reset()
            assert info["nested"] == {"flag": True, "large": 2**60, "raw": b"\x00x"}
            binding.step()
    assert {name for name, _ in env.calls} == {"reset", "step", "render", "close"}
    assert {thread for _, thread in env.calls} == {caller}


def test_binding_hooks_can_render_a_remote_env_before_the_first_step() -> None:
    env = Env()
    frames: list[object] = []

    class Hooks(rlmesh.RunHooks):
        def on_run_start(self, context: rlmesh.RunContext) -> None:
            render = context.frame_sources().render
            assert render is not None
            frames.append(render())

    server = rlmesh.EnvServer(env, ":0/render")
    server.start()
    try:
        with rlmesh.Model(lambda obs: 0).bind(server.address, hooks=Hooks()):
            pass
    finally:
        server.shutdown()
    assert len(frames) == 1
    assert cast(Any, frames[0]).shape == [2, 2, 3]


def test_viewer_truncation_evicts_model_state_before_the_next_reset() -> None:
    ended: list[str] = []

    class NeverDone(Env):
        def step(
            self, action: object
        ) -> tuple[int, float, bool, bool, dict[str, object]]:
            return 0, 0.0, False, False, {}

    class Skip:
        def feed(self, *, steps: int, **kwargs: Any) -> None:
            self.steps = steps

        def consume_skip(self) -> bool:
            return self.steps >= 1

        def quit_requested(self) -> bool:
            return False

        def pace(self) -> None:
            pass

        def close(self, **kwargs: Any) -> None:
            pass

    model = rlmesh.Model(
        lambda obs: 0, on_episode_end=lambda episode_id: ended.append(episode_id)
    )
    with model.bind(NeverDone()) as binding:
        binding._view = cast(ViewerDriver, Skip())
        binding.reset()
        assert binding.step()[3] is True
        assert len(ended) == 1
        assert binding.done
    assert len(ended) == 1


def test_failed_run_keeps_its_exception_when_end_hook_attempts_reentry() -> None:
    original = KeyError("original predict failure")

    def predict(observation: object) -> int:
        raise original

    model = rlmesh.Model(predict)

    class Hooks(rlmesh.RunHooks):
        def on_run_end(self, result: rlmesh.RunResult) -> None:
            with pytest.raises(RuntimeError, match="already in progress"):
                model.run(Env())

    with pytest.raises(KeyError) as raised:
        model.run(Env(), hooks=Hooks())
    assert raised.value is original


def test_binding_close_does_not_swallow_a_hook_with_closed_error_text() -> None:
    original = RuntimeError("the binding is closed")

    class Hooks(rlmesh.RunHooks):
        def on_run_end(self, result: rlmesh.RunResult) -> None:
            raise original

    binding = rlmesh.Model(lambda obs: 0).bind(Env(), hooks=Hooks())
    binding.reset()
    binding.step()
    with pytest.raises(RuntimeError) as raised:
        binding.close()
    assert raised.value is original
    binding.close()


def test_dropped_binding_releases_its_factory_without_holding_the_gil() -> None:
    import subprocess
    import sys
    import textwrap

    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import gc
            import rlmesh
            from rlmesh import spaces
            closed = []
            class Env:
                observation_space = spaces.Discrete(1)
                action_space = spaces.Discrete(1)
                def reset(self, **kwargs): return 0, {}
                def step(self, action): return 0, 0.0, False, False, {}
                def close(self): closed.append(1)
            class Factory(rlmesh.EnvFactory):
                def make(self): return Env()
            model = rlmesh.Model(lambda obs: 0)
            binding = model.bind(Factory())
            binding.reset()
            del binding
            gc.collect()
            assert closed == [1], closed
            model.close()
        """),
        ],
        check=True,
        timeout=15,
    )


def test_binding_viewer_receives_the_shared_terminal_outcome() -> None:
    outcomes: list[str] = []

    class Success(Env):
        def step(
            self, action: object
        ) -> tuple[int, float, bool, bool, dict[str, object]]:
            return 0, 1.0, True, False, {"is_success": True}

    class View:
        def feed(self, *, outcome: str, **kwargs: Any) -> None:
            outcomes.append(outcome)

        def consume_skip(self) -> bool:
            return False

        def quit_requested(self) -> bool:
            return False

        def pace(self) -> None:
            pass

        def close(self, **kwargs: Any) -> None:
            pass

    with rlmesh.Model(lambda obs: 0).bind(Success()) as binding:
        binding._view = cast(ViewerDriver, View())
        binding.reset()
        binding.step()
    assert outcomes == ["", "success"]


@pytest.mark.parametrize("path", ["run", "bind"])
def test_viewer_receives_shared_step_timings(
    path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames: list[dict[str, Any]] = []

    class NeverDone(Env):
        def step(
            self, action: object
        ) -> tuple[int, float, bool, bool, dict[str, object]]:
            return 0, 1.0, False, False, {}

    class View:
        def __init__(self, config: object) -> None:
            pass

        def feed(self, **kwargs: Any) -> None:
            frames.append(kwargs)

        def consume_skip(self) -> bool:
            return False

        def quit_requested(self) -> bool:
            return bool(frames and frames[-1]["steps"] >= 1)

        def pace(self) -> None:
            pass

        def close(self, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr("rlmesh._models._view.ViewerDriver", View)
    monkeypatch.setattr("rlmesh._models._binding.ViewerDriver", View)
    model = rlmesh.Model(lambda obs: 0)
    if path == "run":
        model.run(NeverDone(), view="terminal")
    else:
        with model.bind(NeverDone(), view="terminal") as binding:
            binding.reset()
            binding.step()
    frame = next(frame for frame in frames if frame["steps"] == 1)
    assert frame["model_ms"] > 0
    assert frame["env_ms"] > 0
    assert frame["elapsed_s"] > 0
    assert frame["reward"] == 1.0


def test_ctrl_c_interrupts_reset_over_a_live_episode() -> None:
    import _thread
    import time

    release = threading.Event()

    class SlowReset(Env):
        resets = 0

        def reset(self, **kwargs: object) -> tuple[int, dict[str, object]]:
            self.resets += 1
            if self.resets > 1:
                release.wait(5)
            return 0, {}

    server = rlmesh.EnvServer(SlowReset(), ":0/cancel-reset")
    server.start()
    interrupt = threading.Timer(0.1, _thread.interrupt_main)
    watchdog = threading.Timer(2, release.set)
    try:
        with rlmesh.Model(lambda obs: 0).bind(server.address) as binding:
            binding.reset()
            started = time.monotonic()
            interrupt.start()
            watchdog.start()
            try:
                with pytest.raises(KeyboardInterrupt):
                    binding.reset()
                assert time.monotonic() - started < 1.5
            finally:
                release.set()
                interrupt.join()
                watchdog.cancel()
                watchdog.join()
    finally:
        server.shutdown()


class Counter:
    """Observes its step count; three steps an episode; records each action."""

    render_mode = None

    def __init__(self) -> None:
        self.observation_space = spaces.Discrete(10)
        self.action_space = spaces.Discrete(10)
        self.actions: list[int] = []
        self.t = 0

    def reset(self, **kwargs: object) -> tuple[int, dict[str, object]]:
        self.t = 0
        return 0, {}

    def step(self, action: object) -> tuple[int, float, bool, bool, dict[str, object]]:
        self.actions.append(int(cast("int", action)))
        self.t += 1
        return self.t, 0.0, self.t >= 3, False, {}

    def close(self) -> None:
        pass


def test_run_hooks_transform_each_observation_and_action_on_the_way() -> None:
    env, seen, calls, stepped = Counter(), [], [], []

    def predict(observation: object) -> int:
        seen.append(int(cast("int", observation)))
        return 9

    class Shift(rlmesh.RunHooks):
        def transform_observation(
            self, observation: Any, *, episode: int, step: int
        ) -> Any:
            calls.append(("observation", episode, step))
            return int(observation) + 5

        def transform_action(self, action: Any, *, episode: int, step: int) -> Any:
            calls.append(("action", episode, step))
            return step

        def on_step(self, event: rlmesh.StepEvent) -> None:
            stepped.append((int(event.observation), int(event.action)))

    rlmesh.Model(predict).run(env, episodes=1, hooks=[rlmesh.RunHooks(), Shift()])

    assert seen == [5, 6, 7], "the model sees each transformed observation"
    assert env.actions == [0, 1, 2], "the env receives each transformed action"
    assert stepped == [(5, 0), (6, 1), (7, 2)], "on_step reports what was sent"
    assert calls == [
        (kind, 0, step) for step in range(3) for kind in ("observation", "action")
    ]


def test_a_transform_returns_an_image_in_the_env_framework() -> None:
    class Camera:
        render_mode = None

        def __init__(self) -> None:
            self.observation_space = spaces.Box(0, 255, (2, 2, 3), np.uint8)
            self.action_space = spaces.Discrete(2)

        def reset(self, **kwargs: object) -> tuple[object, dict[str, object]]:
            return np.zeros((2, 2, 3), np.uint8), {}

        def step(
            self, action: object
        ) -> tuple[object, float, bool, bool, dict[str, object]]:
            return np.zeros((2, 2, 3), np.uint8), 0.0, True, False, {}

        def close(self) -> None:
            pass

    seen: list[np.ndarray] = []

    class Invert(rlmesh.RunHooks):
        def transform_observation(
            self, observation: Any, *, episode: int, step: int
        ) -> Any:
            return 255 - observation

    def predict(observation: object) -> int:
        seen.append(np.asarray(observation))
        return 0

    rlmesh.Model(predict).run(Camera(), episodes=1, hooks=Invert())

    assert len(seen) == 1 and seen[0].dtype == np.uint8
    assert (seen[0] == 255).all(), "the model sees the transformed image"


def test_a_raising_transform_aborts_the_run_with_its_own_exception() -> None:
    class Broken(rlmesh.RunHooks):
        def transform_action(self, action: Any, *, episode: int, step: int) -> Any:
            raise ZeroDivisionError("perturbation failed")

    with pytest.raises(ZeroDivisionError, match="perturbation failed"):
        rlmesh.Model(lambda obs: 0).run(Counter(), episodes=1, hooks=Broken())


def test_run_routes_split_the_trials_and_a_failed_route_fails_alone() -> None:
    class Dies(Counter):
        def step(
            self, action: object
        ) -> tuple[int, float, bool, bool, dict[str, object]]:
            raise RuntimeError("env gone")

    envs = [Counter(), Dies(), Counter()]
    servers = [rlmesh.EnvServer(env, f":0/route-{i}") for i, env in enumerate(envs)]
    for server in servers:
        server.start()
    try:
        results = rlmesh.Model(rlmesh.RANDOM_SAMPLE).run_routes(
            servers, episodes=5, base_seed=3
        )
        with pytest.raises(ValueError, match="one route at a time"):
            rlmesh.Model(lambda obs: 0).run_routes(servers)

        class Nests(Counter):
            def reset(self, **kwargs: Any) -> tuple[int, dict[str, object]]:
                rlmesh.Model(rlmesh.RANDOM_SAMPLE).run_routes(servers[:1])
                return super().reset(**kwargs)

        (nested,) = rlmesh.Model(rlmesh.RANDOM_SAMPLE).run_routes([Nests()])
    finally:
        for server in servers:
            server.shutdown()

    assert isinstance(results[1], Exception) and "env gone" in str(results[1])
    # A route cannot run routes of its own: the outer Ctrl-C would never reach them.
    assert isinstance(nested, RuntimeError) and "inside a running route" in str(nested)
    merged = rlmesh.RunResult.merge(
        r for r in results if isinstance(r, rlmesh.RunResult)
    )
    # Five episodes over three routes: trials 0-1, 2-3 (failed) and 4.
    assert [e.trial for e in merged.episodes] == [0, 1, 4]
    assert [e.index for e in merged.episodes] == [0, 1, 2]
    assert all(e.seed is not None for e in merged.episodes)
