"""Validate the MetaworldReach recipe and show the render->obs mechanic.

Runs WITHOUT installing MetaWorld or starting Docker:

1. importing ``metaworld_reach`` auto-registers the recipe (``@rlmesh.register``);
2. ``check()`` validates it dependency-free (authoring != running -- it never
   imports MetaWorld); and
3. a tiny render-stub env shows the observation space change locally.

    uv run python examples/python/recipes/render_into_obs.py
"""

from metaworld_reach import MetaworldReach
from rlmesh import recipes


def demo_wrapper_locally():
    """Show the obs change on a render-stub env -- no MetaWorld, no Docker."""
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces
    from gymnasium.wrappers import AddRenderObservation

    class ArrayStateRenderEnv(gym.Env):
        """Stand-in for MetaWorld: flat-array state, image only via render()."""

        def __init__(self):
            self.render_mode = "rgb_array"
            self.metadata = {"render_modes": ["rgb_array"]}
            self.observation_space = spaces.Box(-np.inf, np.inf, (39,), np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(39, np.float32), {}

        def step(self, action):
            return np.zeros(39, np.float32), 0.0, False, False, {}

        def render(self):
            return np.zeros((84, 84, 3), np.uint8)

    base = ArrayStateRenderEnv()
    wrapped = AddRenderObservation(base, render_only=False)
    obs, _ = wrapped.reset(seed=0)

    print(f"  before:  {base.observation_space}")
    print(f"  after :  {wrapped.observation_space}")
    print(
        f"  obs   :  keys={list(obs)}  state={obs['state'].shape}  pixels={obs['pixels'].shape}"
    )


def main():
    print("1. registered by importing the module:")
    print(
        f"   {recipes.resolve('metaworld/reach').name!r} ->",
        recipes.resolve("metaworld/reach").make,
    )

    print("\n2. validated dependency-free (MetaWorld is not imported):")
    MetaworldReach.check()
    print("   ok\n")
    recipes.pprint_registry()

    print("\n3. the render->obs mechanic, locally (no MetaWorld, no Docker):")
    demo_wrapper_locally()

    print("\nto run it for real, see serve_metaworld.py (builds a container).")


if __name__ == "__main__":
    main()
