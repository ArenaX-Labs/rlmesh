"""Ctrl-C during `Model.run` raises `rlmesh.Interrupted` with what finished."""

from __future__ import annotations

import _thread
import threading
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
    ended = threading.Event()

    class Ends(_TwoStepEnv):
        def step(self, action: object) -> tuple[int, float, bool, bool, dict]:
            result = super().step(action)
            if result[2]:
                ended.set()
            return result

    class Interrupts(_TwoStepEnv):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def step(self, action: object) -> tuple[int, float, bool, bool, dict]:
            self.calls += 1
            if self.calls == 20:
                # The other route's episode ends first, however slow the host.
                ended.wait(30)
                _thread.interrupt_main()
            if self.calls >= 20:
                time.sleep(0.01)
            return 0, 1.0, False, False, {}

    interrupts = Interrupts()
    servers = [
        rlmesh.EnvServer(env, f":0/interrupted-{i}")
        for i, env in enumerate([Ends(), interrupts])
    ]
    for server in servers:
        server.start()
    model = rlmesh.Model(rlmesh.RANDOM_SAMPLE)
    try:
        (finished, cut), interrupt = model._run_routes(  # pyright: ignore[reportPrivateUsage]
            [(server, {"episodes": 1}) for server in servers]
        )
        interrupts.calls = 0
        with pytest.raises(KeyboardInterrupt) as caught:
            model.run_routes(servers[1:])
    finally:
        for server in servers:
            server.shutdown()

    assert isinstance(finished, rlmesh.RunResult) and finished.num_episodes == 1
    assert type(cut) is KeyboardInterrupt and type(interrupt) is KeyboardInterrupt
    assert type(caught.value) is KeyboardInterrupt


class _RaisesOnReset(_TwoStepEnv):
    def reset(self, *, seed: object = None, options: Any = None) -> tuple[int, dict]:
        raise KeyboardInterrupt


def _raising_predict(obs: object) -> int:
    raise KeyboardInterrupt


@pytest.mark.parametrize(
    ("model", "env"),
    [
        (lambda: rlmesh.Model(rlmesh.RANDOM_SAMPLE), _RaisesOnReset),
        (lambda: rlmesh.Model(_raising_predict), _TwoStepEnv),
    ],
    ids=["env", "in-process-model"],
)
def test_a_routes_own_ctrl_c_is_raised_as_the_runs(model: Any, env: Any) -> None:
    model = model()
    (cut,), interrupt = model._run_routes([(env(), {"episodes": 1})])  # pyright: ignore[reportPrivateUsage]
    assert type(cut) is KeyboardInterrupt and interrupt is cut
    with pytest.raises(KeyboardInterrupt):
        model.run_routes([env()])


def test_a_routes_own_ctrl_c_stops_its_siblings() -> None:
    class Endless(_TwoStepEnv):
        def step(self, action: object) -> tuple[int, float, bool, bool, dict]:
            time.sleep(0.001)
            # Ends only past the test's bound, so a regression fails, not hangs.
            return 0, 1.0, time.monotonic() > deadline, False, {}

    deadline = time.monotonic() + 30
    model = rlmesh.Model(rlmesh.RANDOM_SAMPLE)
    (cut, stopped), interrupt = model._run_routes(  # pyright: ignore[reportPrivateUsage]
        [(_RaisesOnReset(), {"episodes": 1}), (Endless(), {"episodes": 1})]
    )
    assert time.monotonic() < deadline
    assert type(cut) is KeyboardInterrupt and interrupt is cut
    assert stopped is cut
