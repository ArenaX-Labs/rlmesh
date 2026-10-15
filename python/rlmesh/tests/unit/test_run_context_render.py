"""RUN-7/RUN-30 render access is scoped independently of retained read access."""

from __future__ import annotations

import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import rlmesh
from rlmesh import spaces


class RenderEnv:
    render_mode = "rgb_array"

    def __init__(self) -> None:
        self.metadata: dict[str, object] = {}
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(1)
        self.frame = np.zeros((2, 2, 3), dtype=np.uint8)
        self.render_again: Any = None

    def reset(self, **kwargs: object) -> tuple[int, dict[str, object]]:
        return 0, {}

    def step(self, action: object) -> tuple[int, float, bool, bool, dict[str, object]]:
        return 0, 1.0, True, False, {}

    def render(self) -> object:
        if self.render_again is not None:
            return self.render_again()
        return self.frame

    def close(self) -> None:
        pass


@pytest.mark.parametrize("native", [False, True])
def test_local_hook_render_scope_and_retained_context(native: bool) -> None:
    env = RenderEnv()
    model = rlmesh.Model(lambda obs: 0)
    session = rlmesh.session(model, env)
    retained: list[Any] = []
    renders: list[Any] = []
    stages: list[str] = []

    class Hooks(rlmesh.RunHooks):
        def on_run_start(self, context: Any) -> None:
            retained.append(context)
            if native:
                assert context.contract is None
                assert context.frame_sources().render is None

        def check(self, stage: str) -> None:
            stages.append(stage)
            render = retained[0].frame_sources().render
            assert render is not None
            assert render() is env.frame
            renders.append(render)
            with ThreadPoolExecutor(max_workers=1) as pool:
                assert pool.submit(retained[0].frame_sources).result().render is None
                with pytest.raises(RuntimeError):
                    pool.submit(render).result()

        def on_episode_start(self, **kwargs: Any) -> None:
            self.check("start")

        def on_step(self, event: rlmesh.StepEvent) -> None:
            self.check("step")

        def on_episode_end(self, result: rlmesh.EpisodeResult) -> None:
            self.check("episode_end")

        def on_run_end(self, result: rlmesh.RunResult) -> None:
            self.check("run_end")

    if native:
        model.run(env, hooks=Hooks())
    else:
        session.run(hooks=Hooks())
    assert stages == ["start", "step", "episode_end", "run_end"]
    context = retained[0]
    assert context.contract is not None and context.num_envs == 1
    assert context.frame_sources().render is None
    for render in renders:
        with pytest.raises(RuntimeError):
            render()
    direct = session.frame_sources().render
    assert direct is not None and direct() is env.frame
    env.render_again = direct
    with pytest.raises(RuntimeError):
        direct()
    env.render_again = None
    assert direct() is env.frame
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        direct()
    if native:
        assert context.contract is not None and context.num_envs == 1
    else:
        with pytest.raises(RuntimeError, match="closed"):
            context.frame_sources()
        with pytest.raises(RuntimeError, match="closed"):
            _ = context.contract


def test_remote_session_context_is_idle_only_and_does_not_own_session() -> None:
    env = RenderEnv()
    # A handle uses its existing connection, represented here by an in-process
    # protocol double. Native remote integration tests exercise the transport.
    env.env_contract = SimpleNamespace(  # type: ignore[attr-defined]
        observation_space=env.observation_space,
        action_space=env.action_space,
        metadata={},
        num_envs=1,
    )
    entered = threading.Event()
    release = threading.Event()

    def predict(obs: object) -> int:
        entered.set()
        assert release.wait(10)
        return 0

    session = rlmesh.session(rlmesh.Model(predict), env)
    contexts: list[Any] = []

    class Hooks(rlmesh.RunHooks):
        def on_run_start(self, context: Any) -> None:
            contexts.append(context)

    session.run(episodes=0, hooks=Hooks())
    context = contexts[0]
    render = context.frame_sources().render
    assert render is not None and render() is env.frame
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(session.predict, 0)
        try:
            assert entered.wait(10)
            assert context.frame_sources().render is None
            with pytest.raises(RuntimeError, match="in flight"):
                render()
        finally:
            release.set()
        assert future.result() == 0
    assert render() is env.frame
    session_ref = weakref.ref(session)
    del session
    gc.collect()
    assert session_ref() is None
    with pytest.raises(RuntimeError, match="dropped"):
        context.frame_sources()
    with pytest.raises(RuntimeError, match="dropped"):
        render()
