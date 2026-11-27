from __future__ import annotations

from typing import Any

import rlmesh
from rlmesh import spaces


class Chat:
    def __init__(self, turns: int = 3) -> None:
        self.observation_space = spaces.Text(64, min_length=0)
        self.action_space = spaces.Text(64, min_length=0)
        self.turns, self.t, self.seed = turns, 0, 0

    def reset(self, *, seed: Any = None, options: Any = None) -> tuple[str, dict]:
        self.t, self.seed = 0, seed or 0
        return f"{self.seed}:prompt", {}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
        self.t += 1
        done = self.t >= self.turns
        return f"{self.seed}:{'verdict' if done else self.t}", 1.0, done, False, {}

    def close(self) -> None:
        return None


class VecChat:
    def __init__(self, lanes: int, autoreset: str) -> None:
        self.num_envs = lanes
        self.single_observation_space = spaces.Text(64, min_length=0)
        self.single_action_space = spaces.Text(64, min_length=0)
        self.metadata = {"autoreset_mode": autoreset}
        self.t = [0] * lanes

    def reset(self, *, seed: Any = None, options: Any = None) -> tuple[list, dict]:
        self.t = [0] * self.num_envs
        return [f"{lane}:prompt" for lane in range(self.num_envs)], {}

    def step(self, actions: Any) -> tuple[list, list, list, list, dict]:
        obs, done = [], []
        for lane in range(self.num_envs):
            if self.t[lane] >= lane + 2:
                self.t[lane] = 0
                obs.append(f"{lane}:prompt")
                done.append(False)
                continue
            self.t[lane] += 1
            ended = self.t[lane] >= lane + 2
            obs.append(f"{lane}:{'verdict' if ended else self.t[lane]}")
            done.append(ended)
        return obs, [1.0] * self.num_envs, done, [False] * self.num_envs, {}

    def close(self) -> None:
        return None


class Steps(rlmesh.RunHooks):
    def __init__(self) -> None:
        self.events: list[Any] = []

    def on_step(self, event: Any) -> None:
        self.events.append(event)


def echo(observation: str) -> str:
    return f"re:{observation}"


def test_the_last_step_carries_the_final_observation() -> None:
    hooks = Steps()
    rlmesh.Model(echo).run(Chat(), seeds=[4, 5], hooks=hooks)
    finals = [
        (e.seed, e.step, e.final_observation)
        for e in hooks.events
        if e.final_observation is not None
    ]
    assert finals == [(4, 2, "4:verdict"), (5, 2, "5:verdict")]
    assert all(
        (e.final_observation is None) == (not (e.terminated or e.truncated))
        for e in hooks.events
    )
    assert "verdict" not in "".join(str(e.observation) for e in hooks.events)


def test_a_step_cap_truncation_carries_the_final_observation() -> None:
    hooks = Steps()
    rlmesh.Model(echo).run(Chat(turns=9), seeds=[1], max_episode_steps=2, hooks=hooks)
    last = hooks.events[-1]
    assert last.truncated and last.final_observation == "1:2"


class Batch(rlmesh.Model):
    def predict_batch(self, observations: list[str], context: Any) -> list[str]:
        return [echo(o) for o in observations]


def test_each_vector_lane_carries_its_own_final_observation() -> None:
    hooks = Steps()
    Batch().run(VecChat(2, "NextStep"), episodes=2, hooks=hooks)
    finals = sorted(
        str(e.final_observation)
        for e in hooks.events
        if e.final_observation is not None
    )
    assert finals == ["0:verdict", "1:verdict"]


def test_each_served_lane_carries_its_own_final_observation() -> None:
    hooks = Steps()
    server = rlmesh.EnvServer([Chat(turns=2), Chat(turns=3)], ":0/env")
    server.start()
    try:
        Batch().run(server.address, seeds=[7, 8, 9], hooks=hooks)
    finally:
        server.shutdown()
    finals = sorted(
        (e.seed, str(e.final_observation))
        for e in hooks.events
        if e.final_observation is not None
    )
    assert [seed for seed, _ in finals] == [7, 8, 9]
    assert all(final.endswith(":verdict") for _, final in finals)


def test_a_transformed_episode_still_reports_the_envs_final_observation() -> None:
    class Shout(Steps):
        def transform_observation(
            self, observation: Any, *, episode: int, step: int
        ) -> Any:
            return str(observation).upper()

    hooks = Shout()
    rlmesh.Model(echo).run(Chat(turns=1), seeds=[3], hooks=hooks)
    (event,) = hooks.events
    assert event.observation == "3:PROMPT"
    assert event.final_observation == "3:verdict"
