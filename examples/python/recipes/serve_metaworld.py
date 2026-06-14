"""Build and roll out the MetaworldReach recipe in a Docker-backed sandbox.

Needs Docker. The recipe's build phase installs MetaWorld and pins the render
backend, then the env serves with the camera frame folded into the observation.

    uv run python examples/python/recipes/serve_metaworld.py
"""

from metaworld_reach import MetaworldReach
from rlmesh.numpy import SandboxEnv

# Pass the class (not the name string) so it works whether or not the registry
# was populated -- SandboxEnv projects it to an inert recipe and builds the image.
env = SandboxEnv(MetaworldReach)
try:
    # obs is a dict here -- {"state": <array>, "pixels": <frame>} -- because the
    # recipe wrapped the env with AddRenderObservation. See render_into_obs.py.
    obs, info = env.reset(seed=0)
    step = 0
    terminated = truncated = False
    while not terminated and not truncated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        print(f"step={step} reward={reward:.3f}")
    print("episode complete")
finally:
    env.close()
