"""Model recipes: author a policy and its construction document in one class.

The model-side sibling of ``rlmesh.recipes``. :class:`ModelRecipe` is the policy:
set ``name``/``build``/``spec``/``inputs``, write ``load()`` and ``predict()``,
and optionally ``reset()``/``close()``. ``Model(recipe).run(env)`` connects it to
an env.

* :class:`ModelRecipe` -- author a policy; run it with ``rlmesh.numpy.Model``.
* :class:`RunResult` -- the typed result of a ``Model.run`` eval.
* :class:`ArtifactInput` / :func:`input_path` -- runtime weight mounts.
* :func:`hf_load` -- load a HuggingFace policy inside ``load()``.
* :func:`register` -- register a model by class or flat ``hf=``/``load=``.
* :data:`DELEGATED` -- the self-adapting-model sentinel.
"""

from __future__ import annotations

from ..recipes._artifacts import hf_load, input_path
from ..recipes._schema import ArtifactInput
from ..recipes.authoring.model import (
    DELEGATED,
    ModelRecipe,
    construct_authored_model,
    is_model_recipe,
)
from ._eval import EpisodeResult, RunResult
from ._registry import register

__all__ = [
    "DELEGATED",
    "ArtifactInput",
    "EpisodeResult",
    "ModelRecipe",
    "RunResult",
    "construct_authored_model",
    "hf_load",
    "input_path",
    "is_model_recipe",
    "register",
]
