from __future__ import annotations

import asyncio
import functools
from typing import Any

import pytest
import rlmesh
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

    class AsyncBoth(rlmesh.Model):
        async def predict(self, observation: Any) -> Any:
            return observation

        def predict_batch(self, observations: Any) -> Any:
            return observations

    with pytest.raises(
        TypeError, match=r"concurrency overlaps .* batches them instead"
    ):
        Both()
    with pytest.raises(TypeError, match="an async predict overlaps"):
        AsyncBoth()


def test_only_predict_can_be_async() -> None:
    class Chunky(rlmesh.Model):
        def predict(self, observation: Any) -> Any:
            return observation

        async def predict_chunk(self, observation: Any) -> Any:
            return [observation]

    with pytest.raises(TypeError, match="predict_chunk cannot be async"):
        Chunky()


def test_an_async_predict_runs_on_the_local_loop_and_close_stops_it() -> None:
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
    loop_thread = agent._concurrency._thread  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]
    assert loop_thread is not None and loop_thread.is_alive()
    agent.close()
    assert not loop_thread.is_alive()


def test_an_async_callable_object_is_an_async_predict() -> None:
    class Policy:
        async def __call__(self, observation: Any) -> Any:
            await asyncio.sleep(0)
            return (int(observation) + 1) % 4

    model = rlmesh.Model(Policy())
    assert model.run(Counter(), seeds=[0]).episodes[0].steps == 3
    model.close()


def test_a_sync_wrapper_around_an_async_predict_is_refused_on_call() -> None:
    def sync_wrapper(fn: Any) -> Any:
        @functools.wraps(fn)
        def call(*args: Any) -> Any:
            return fn(*args)

        return call

    class Hidden(rlmesh.Model):
        @sync_wrapper
        async def predict(self, observation: Any) -> Any:
            return observation

    with pytest.raises(TypeError, match="returned a coroutine"):
        Hidden().run(Counter(), seeds=[0])


def test_close_from_inside_an_async_predict_raises_instead_of_hanging() -> None:
    class Closer(rlmesh.Model):
        async def predict(self, observation: Any) -> Any:
            self.close()
            return observation

    with pytest.raises(RuntimeError, match="re-entered its own Model"):
        Closer().run(Counter(), seeds=[0])


def test_a_system_exit_in_an_async_predict_reaches_the_caller_and_spares_the_loop() -> (
    None
):
    raised: list[bool] = []

    class Exiting(rlmesh.Model):
        async def predict(self, observation: Any) -> Any:
            if not raised:
                raised.append(True)
                raise SystemExit(3)
            return (int(observation) + 1) % 4

    model = Exiting()
    with pytest.raises(SystemExit):
        model.run(Counter(), seeds=[0])
    assert model.run(Counter(), seeds=[0]).episodes[0].steps == 3
    model.close()


def test_an_async_predict_cannot_reach_inner_models() -> None:
    class Outer(rlmesh.Model):
        def __init__(self) -> None:
            super().__init__()
            self.inner = rlmesh.Model(lambda observation: observation)

        async def predict(self, observation: Any) -> Any:
            return self.inner.predict(observation)

    with pytest.raises(TypeError, match="calls inner models"):
        Outer().run(Counter(), seeds=[0])


def test_concurrency_set_on_the_instance_or_a_wrapped_policy_overlaps() -> None:
    class OnInstance(rlmesh.Model):
        def __init__(self) -> None:
            super().__init__()
            self.concurrency = 3

        def predict(self, observation: Any) -> Any:
            return (int(observation) + 1) % 4

    class Policy:
        concurrency = 5

        def predict(self, observation: Any) -> Any:
            return (int(observation) + 1) % 4

    for model, limit in ((OnInstance(), 3), (rlmesh.Model(Policy()), 5)):
        assert model.run(Counter(), seeds=[0]).episodes[0].steps == 3
        concurrency = model._concurrency  # pyright: ignore[reportPrivateUsage]
        assert concurrency is not None and concurrency.limit == limit
        model.close()
