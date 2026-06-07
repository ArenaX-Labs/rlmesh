import gymnasium as gym
from rlmesh import EnvServer

ADDRESS = "127.0.0.1:5555"

env = gym.make("sai_pygame:SquidHunt-v0")
print(env.spec)
print(env.action_space)
print(env.observation_space)
print(env.metadata)

server = EnvServer(env, ADDRESS)
server.serve()
