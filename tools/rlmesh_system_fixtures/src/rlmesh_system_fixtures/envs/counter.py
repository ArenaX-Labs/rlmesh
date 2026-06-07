from __future__ import annotations

from dataclasses import dataclass

from rlmesh_system_fixtures.registry import env_fixture


@dataclass
class CounterEnv:
    limit: int = 3
    value: int = 0

    def __post_init__(self) -> None:
        from rlmesh import spaces

        self.observation_space = spaces.Discrete(self.limit + 1)
        self.action_space = spaces.Discrete(2)

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[int, dict[str, object]]:
        self.value = 0
        return self.value, {"seed": seed, "options": options}

    def step(self, action: object) -> tuple[int, float, bool, bool, dict[str, object]]:
        self.value += 1
        terminated = self.value >= self.limit
        reward = float(self.value)
        return self.value, reward, terminated, False, {"action": action}

    def close(self) -> None:
        pass


@env_fixture("counter")
def make_counter(limit: int = 3) -> CounterEnv:
    return CounterEnv(limit=limit)
