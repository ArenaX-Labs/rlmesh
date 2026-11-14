"""RUN-7/RUN-30 render access is scoped independently of retained read access."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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


def test_local_hook_render_scope_and_retained_context() -> None:
    env = RenderEnv()
    model = rlmesh.Model(lambda obs: 0)
    retained: list[Any] = []
    renders: list[Any] = []
    stages: list[str] = []

    class Hooks(rlmesh.RunHooks):
        def on_run_start(self, context: Any) -> None:
            retained.append(context)
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

    model.run(env, hooks=Hooks())
    assert stages == ["start", "step", "episode_end", "run_end"]
    context = retained[0]
    assert context.contract is not None and context.num_envs == 1
    assert context.frame_sources().render is None
    for render in renders:
        with pytest.raises(RuntimeError):
            render()
