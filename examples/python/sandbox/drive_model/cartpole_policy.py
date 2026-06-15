"""A trivial model recipe used to demonstrate ``SandboxModel.run``.

Like the env recipes, a model recipe lives in an importable module so its policy
travels *by reference* into the container. ``build.project`` stages this folder
(see ``pyproject.toml``) so the container can ``import cartpole_policy`` and run
``load``/``predict`` next to the weights and deps -- here just a constant action.
"""

from __future__ import annotations

import rlmesh
from rlmesh.models import ModelRecipe
from rlmesh.recipes import Build, PipInstall, ProjectInstall


@rlmesh.register
class CartpolePolicy(ModelRecipe):
    name = "cartpole/always-left"
    build = Build(
        pip=[PipInstall(["gymnasium", "numpy"])],
        project=ProjectInstall(src="."),  # stage this folder so the class imports
    )

    def load(self) -> None:
        pass

    def predict(self, observation: object) -> int:
        return 0  # always push the cart left
