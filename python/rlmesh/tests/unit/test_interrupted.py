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


def test_ctrl_c_in_run_routes_keeps_the_routes_that_finished() -> None:
    class Interrupts(_TwoStepEnv):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def step(self, action: object) -> tuple[int, float, bool, bool, dict]:
            self.calls += 1
            if self.calls == 20:
                _thread.interrupt_main()
            if self.calls >= 20:
                time.sleep(0.01)
            return 0, 1.0, False, False, {}

    interrupts = Interrupts()
    servers = [
        rlmesh.EnvServer(env, f":0/interrupted-{i}")
        for i, env in enumerate([_TwoStepEnv(), interrupts])
    ]
    for server in servers:
        server.start()
    model = rlmesh.Model(rlmesh.RANDOM_SAMPLE)
    try:
        finished, cut = model._run_routes(  # pyright: ignore[reportPrivateUsage]
            [(server, {"episodes": 1}) for server in servers]
        )
        interrupts.calls = 0
        with pytest.raises(KeyboardInterrupt) as caught:
            model.run_routes(servers[1:])
    finally:
        for server in servers:
            server.shutdown()

    assert isinstance(finished, rlmesh.RunResult) and finished.num_episodes == 1
    assert type(cut) is KeyboardInterrupt
    assert type(caught.value) is KeyboardInterrupt
