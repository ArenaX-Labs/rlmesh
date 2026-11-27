from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest
import rlmesh


def test_concurrency_must_be_a_positive_int() -> None:
    for bad in (0, -1, True, 2.0, "4"):
        with pytest.raises(TypeError, match="concurrency must be a positive int"):
            rlmesh.Model(lambda observation: observation, concurrency=bad)  # type: ignore[arg-type]


def test_a_batching_model_cannot_also_overlap_predicts() -> None:
    class Both(rlmesh.Model):
        concurrency = 4

        def predict(self, observation: Any) -> Any:
            return observation

        def predict_batch(self, observations: Any) -> Any:
            return observations

    with pytest.raises(TypeError, match="batches them instead"):
        Both()


def test_only_predict_can_be_async() -> None:
    class Chunky(rlmesh.Model):
        def predict(self, observation: Any) -> Any:
            return observation

        async def predict_chunk(self, observation: Any) -> Any:
            return [observation]

    with pytest.raises(TypeError, match="predict_chunk cannot be async"):
        Chunky()


def test_an_async_predict_runs_on_the_local_loop_and_close_stops_it() -> None:
    from rlmesh import spaces

    class Counter:
        observation_space = spaces.Discrete(4)
        action_space = spaces.Discrete(4)

        def reset(self, *, seed: int | None = None, options: Any = None) -> Any:
            self.steps = 0
            return 0, {}

        def step(self, action: Any) -> Any:
            self.steps += 1
            return int(action), 0.0, self.steps >= 3, False, {}

        def close(self) -> None:
            return None

    loops: set[int] = set()

    class Agent(rlmesh.Model):
        async def predict(self, observation: Any) -> Any:
            loops.add(id(asyncio.get_running_loop()))
            await asyncio.sleep(0)
            return (int(observation) + 1) % 4

    agent = Agent()
    result = agent.run(Counter(), seeds=[0])
    assert result.episodes[0].steps == 3
    assert len(loops) == 1
    before = {thread.name for thread in threading.enumerate()}
    assert "Agent-predict" in before
    agent.close()
    assert "Agent-predict" not in {thread.name for thread in threading.enumerate()}
