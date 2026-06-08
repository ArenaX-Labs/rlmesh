"""Serve the SAI MuJoCo color-sort environment through RLMesh."""

import gymnasium as gym
from rlmesh import EnvServer

ENV_ID = "sai_mujoco:So101IkColorSortPickPlace-v0"

env = gym.make(ENV_ID, render_mode="rgb_array")
server = EnvServer(env, "127.0.0.1:5555")
server.start()
print(f"serving {ENV_ID} on {server.address}")
print(f"observation_space={env.observation_space!r}")
print(f"action_space={env.action_space!r}")
server.wait()
