"""An IsaacSim recipe authored on a machine that can't import IsaacSim.

This is the headline *authoring != running* case. You write this on a Mac (or any
box without the GPU sim stack), where ``import isaaclab`` would fail -- and that's
fine, because the recipe never imports it. The heavy imports live inside
``make()``/``prepare()``, which only run in the built GPU container. Projection
(``@rlmesh.register`` / ``to_recipe`` / ``check``) is pure data, so authoring and
validation work anywhere.

Two things this shows that the gym/flat recipes don't:
  * ``prepare()`` -- a construct-time hook that runs before the env exists (here,
    launching the headless simulator app);
  * vectorize *inside the factory* -- IsaacSim batches envs itself, so num_envs is
    a make() kwarg the factory owns, not rlmesh's num_envs.
"""

from __future__ import annotations

# The IsaacSim packages are deliberately absent here -- that's the point of the
# example (you author where they can't be imported), so don't flag them.
# pyright: reportMissingImports=false
import rlmesh
from rlmesh.recipes import Build, PipInstall, ProjectInstall


@rlmesh.register
class IsaacFrankaStack(rlmesh.EnvRecipe):
    name = "isaac/franka-stack"

    build = Build(
        base="nvcr.io/nvidia/isaac-lab:2.0.0",  # GPU base image, pinned by you
        pip=[PipInstall(["rl-games"])],
        project=ProjectInstall(src=".", dest="/opt/task"),  # your task code rides along
        gpu=True,
    )

    def prepare(self):
        # Construct-time setup that must happen before the env exists: launch the
        # headless simulator. Heavy and GPU-only, so it never runs at authoring time.
        from isaaclab.app import AppLauncher

        self._app = AppLauncher(headless=True).app

    def make(self, task="Isaac-Stack-Cube-Franka-v0", sim_envs=1, **kwargs):
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401  -- registers the Isaac tasks on import

        # IsaacSim vectorizes internally: sim_envs is the factory's own batch knob,
        # distinct from rlmesh's num_envs (a py factory returns one env object).
        return gym.make(task, num_envs=sim_envs, **kwargs)
