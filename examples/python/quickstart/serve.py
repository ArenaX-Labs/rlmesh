"""Serve a tiny RLMesh environment for the quickstart."""

from __future__ import annotations

import rlmesh

ADDRESS = "127.0.0.1:5555"


class CounterEnv:
    """Tiny single-agent environment used by the quickstart."""

    observation_space = rlmesh.spaces.Discrete(5)
    action_space = rlmesh.spaces.Discrete(2)

    def __init__(self) -> None:
        self.step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[int, dict[str, object]]:
        _ = seed, options
        self.step_count = 0
        return 0, {}

    def step(self, action: object) -> tuple[int, float, bool, bool, dict[str, object]]:
        self.step_count += 1
        observation = self.step_count % 5
        terminated = self.step_count >= 3
        return observation, 1.0, terminated, False, {"action": action}

    def close(self) -> None:
        return None


def main() -> None:
    """Serve the environment until interrupted."""
    server = rlmesh.EnvServer(CounterEnv(), ADDRESS)
    print(f"serving CounterEnv on {ADDRESS}")
    server.serve()


if __name__ == "__main__":
    main()
