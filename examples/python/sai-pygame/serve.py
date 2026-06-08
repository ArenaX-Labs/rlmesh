"""Serve the SAI Pygame SquidHunt environment through RLMesh."""

import gymnasium as gym
from rlmesh import EnvServer

ADDRESS = "127.0.0.1:5555"
ENV_ID = "sai_pygame:SquidHunt-v0"

env = gym.make(ENV_ID)
print(f"serving {ENV_ID} on {ADDRESS}")
print(f"observation_space={env.observation_space!r}")
print(f"action_space={env.action_space!r}")

server = EnvServer(env, ADDRESS)
server.serve()
