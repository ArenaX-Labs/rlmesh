from rlmesh.numpy import SandboxEnv

MAX_STEPS = 45

env = SandboxEnv(
    "CartPole-v1",
    packages=["gymnasium==1.3.0"],
    imports=["gymnasium"],
)

try:
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
finally:
    env.close()
