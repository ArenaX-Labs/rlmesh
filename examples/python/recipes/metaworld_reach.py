"""A recipe that folds a render-only env's frame into its observation.

Some envs (MetaWorld, many MuJoCo tasks) return a flat state array and only
expose the camera image through ``env.render()`` -- the image is not in the
observation. A recipe fixes this in one place: ``make()`` returns the env
already wrapped with Gymnasium's ``AddRenderObservation``, and ``build`` pins
the render backend (``MUJOCO_GL``) so off-screen rendering works in the
container instead of being rediscovered per machine.

The recipe lives in an importable module (not the run script) on purpose: the
factory travels *by reference* (``metaworld_reach:MetaworldReach...``), so the
container can import and construct it. ``@rlmesh.register`` makes it resolvable
by name; see ``render_into_obs.py`` and ``serve_metaworld.py`` to use it.
"""

from __future__ import annotations

import rlmesh
from rlmesh.recipes import Build, PipInstall


@rlmesh.register
class MetaworldReach(rlmesh.EnvRecipe):
    name = "metaworld/reach"

    # The build phase pins the deps and the render backend: env.render() needs an
    # off-screen GL context, and that missing context is the real reason "the
    # image isn't in the obs" only bites at runtime.
    build = Build(
        pip=[PipInstall(["metaworld", "gymnasium>=1.3"])],
        env={"MUJOCO_GL": "egl"},  # off-screen render without a display
    )

    def make(self, task="reach-v2", camera_name="corner", **kwargs):
        import gymnasium as gym
        from gymnasium.wrappers import AddRenderObservation

        # The exact env id/args are MetaWorld-version specific -- swap to match
        # your install; only the two calls below are the recipe's real job.
        env = gym.make(
            f"Meta-World/{task}",
            render_mode="rgb_array",  # makes env.render() produce frames at all
            camera_name=camera_name,
            **kwargs,
        )

        # Fold the rendered frame into the obs space, in one line:
        #   Box(state) -> Dict(state=Box(...), pixels=Box(H, W, 3))
        # Use render_only=True for pixels-only.
        return AddRenderObservation(env, render_only=False)
