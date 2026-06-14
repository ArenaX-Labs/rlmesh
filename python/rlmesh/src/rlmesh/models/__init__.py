"""Model recipes: author a policy and its construction document in one class.

The model-side sibling of ``rlmesh.recipes`` (FINAL_API_SPEC §3). The headline
authoring surface is :class:`ModelRecipe` -- one class IS the policy: set
``name``/``build``/``spec``/``inputs``, write ``load()`` (build the model into
``self``) and ``predict()``, and optionally ``reset()``/``close()``. Connect it to
an env with :class:`ModelServer`, the consumer-side mirror of ``EnvServer``.

Surface (~mirrors ``rlmesh.recipes`` / ``EnvServer`` one-to-one):

* :class:`ModelRecipe` -- author a policy.
* :class:`ModelServer` -- serve it / drive it against an env (returns a
  :class:`RunResult`).
* :class:`ArtifactInput` / :func:`input_path` -- runtime weight mounts.
* :func:`hf_load` -- the HuggingFace one-liner you call inside ``load()``.
* :func:`register` -- register a model by class or flat ``hf=``/``load=`` sugar.
* :data:`DELEGATED` -- the "this model self-adapts, do not resolve" sentinel.
"""

from __future__ import annotations

from ..recipes._artifacts import hf_load, input_path
from ..recipes._authoring_model import (
    DELEGATED,
    ModelRecipe,
    construct_authored_model,
    is_model_recipe,
)
from ..recipes._schema import ArtifactInput
from ._registry import register
from ._server import EpisodeResult, ModelServer, RunResult

__all__ = [
    "DELEGATED",
    "ArtifactInput",
    "EpisodeResult",
    "ModelRecipe",
    "ModelServer",
    "RunResult",
    "construct_authored_model",
    "hf_load",
    "input_path",
    "is_model_recipe",
    "register",
]
