"""A recipe that folds a render-only env's frame into its observation.

Some envs (MetaWorld, many MuJoCo tasks) return a flat state array and only
expose the camera image through ``env.render()`` -- the image is not in the
observation. A recipe fixes this in one place: ``make()`` returns the env
already wrapped with Gymnasium's ``AddRenderObservation``, and ``build`` pins a
headless render backend (``MUJOCO_GL=osmesa`` plus its system libs) so off-screen
rendering works in the container with no GPU or display.

The recipe lives in an importable module (not the run script) on purpose: the
factory travels *by reference* (``metaworld_reach:MetaworldReach...``), and the
container imports and constructs it from there. ``build.project`` stages this
folder (see the sibling ``pyproject.toml``) so the import resolves inside the
image. ``@rlmesh.register`` makes it resolvable by name; see ``render_into_obs.py``
and ``serve_metaworld.py`` to use it.
"""

from __future__ import annotations

import rlmesh
from rlmesh.recipes import Build, PipInstall, ProjectInstall


@rlmesh.register
class MetaworldReach(rlmesh.EnvRecipe):
    name = "metaworld/reach"
    build = Build(
        pip=[PipInstall(["metaworld", "gymnasium>=1.3", "packaging"])],
        system=["libosmesa6", "libgl1", "libglfw3"],
        env={"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"},
        project=ProjectInstall(src="."),  # stage this folder so the class imports
    )

    def make(self, task="reach-v3", seed=0, **kwargs):
        import numpy as np
        from gymnasium import spaces
        from gymnasium.wrappers import AddRenderObservation
        from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE

        # Meta-World v3 registers benchmark ids (Meta-World/MT1, .../goal_observable),
        # not per-task gym ids, and its goal_observable entry point does not forward
        # render_mode -- so construct the single-task class directly to get frames.
        env_cls = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[f"{task}-goal-observable"]
        env = env_cls(seed=seed, render_mode="rgb_array", **kwargs)

        # Meta-World's last 3 obs dims (the goal) declare a [0, 0] bound but hold
        # real goal values, so every obs falls outside its own space. Widen to an
        # unbounded Box so a strict validator (rlmesh's wire check) accepts it.
        obs_space = env.observation_space
        env.observation_space = spaces.Box(
            -np.inf, np.inf, obs_space.shape, obs_space.dtype
        )

        # Fold the rendered frame into the obs space, in one line:
        #   Box(state) -> Dict(state=Box(...), pixels=Box(H, W, 3))
        # Use render_only=True for pixels-only.
        return AddRenderObservation(env, render_only=False)
