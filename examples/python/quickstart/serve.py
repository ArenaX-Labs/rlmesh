"""Serve a tiny RLMesh environment for the quickstart."""

import rlmesh


class CounterEnv:
    """Tiny single-agent environment used by the quickstart."""

    observation_space = rlmesh.spaces.Discrete(5)
    action_space = rlmesh.spaces.Discrete(2)

    def __init__(self):
        self.step_count = 0

    def reset(self, seed=None, options=None):
        self.step_count = 0
        return 0, {}

    def step(self, action):
        self.step_count += 1
        observation = self.step_count % 5
        terminated = self.step_count >= 3
        return observation, 1.0, terminated, False, {"action": action}

    def close(self):
        pass


server = rlmesh.EnvServer(CounterEnv(), "127.0.0.1:5555")
print(f"serving CounterEnv on {server.address}")
server.serve()
