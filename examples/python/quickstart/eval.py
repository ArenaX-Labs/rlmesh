"""Run a sampled-action eval against any example environment server."""

from rlmesh.numpy import RemoteEnv

MAX_STEPS = 64

env = RemoteEnv("127.0.0.1:5555")

print(f"connected to {env.address}")
print(f"observation_space={env.observation_space!r}")
print(f"action_space={env.action_space!r}")

obs, info = env.reset(seed=0)
for step in range(1, MAX_STEPS + 1):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"step={step} reward={reward:.3f}")
    if terminated or truncated:
        print("episode complete")
        break
else:
    print(f"stopped after {MAX_STEPS} steps")

env.close()
