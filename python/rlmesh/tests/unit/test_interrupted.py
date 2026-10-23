"""Ctrl-C during `Model.run` raises `rlmesh.Interrupted` with what finished."""

from __future__ import annotations

import _thread
import time
from typing import Any, ClassVar

import pytest
import rlmesh


class _TwoStepEnv:
    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = rlmesh.spaces.Discrete(1)
        self.action_space = rlmesh.spaces.Discrete(1)
        self.steps = 0

    def reset(self, *, seed: object = None, options: Any = None) -> tuple[int, dict]:
        self.steps = 0
        return 0, {}

    def step(self, action: object) -> tuple[int, float, bool, bool, dict]:
        self.steps += 1
        return 0, 1.0, self.steps >= 2, False, {}

    def close(self) -> None:
        pass


def test_ctrl_c_keeps_the_episodes_that_finished() -> None:
    calls = {"n": 0}

    def predict(obs: object) -> int:
        calls["n"] += 1
        if calls["n"] == 7:  # the first step of the fourth episode
            _thread.interrupt_main()
        if calls["n"] >= 7:
            time.sleep(0.01)  # the run's signal poll lands within a few steps
        return 0

    with pytest.raises(rlmesh.Interrupted) as caught:
        rlmesh.Model(predict).run(_TwoStepEnv(), episodes=1000)

    assert isinstance(caught.value, KeyboardInterrupt)
    done = caught.value.result.episodes
    assert 3 <= len(done) < 1000
    assert all(e.steps == 2 for e in done)
