from rlmesh.numpy import SandboxBuild, SandboxEnv

MAX_STEPS = 45


env = SandboxEnv(
    "hf://lerobot/cartpole-env:cartpole_suite/0",
    build=SandboxBuild(trust_remote_code=True, allow_unpinned_hf=True),
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
