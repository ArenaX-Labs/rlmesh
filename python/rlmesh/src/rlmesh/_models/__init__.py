"""Model-side runtime: the eval loop, result types, and weight-mount helpers.

The kept neutral core a framework ``Model`` runs against. ``Model(predict).run(env)``
connects a predict callable to an env.

* :class:`RunResult` -- the typed result of a ``Model.run`` eval.
* :class:`ArtifactInput` -- a runtime weight mount.
* :func:`hf_load` -- load a HuggingFace policy.
* :data:`DELEGATED` -- the self-adapting-model sentinel.
"""

from __future__ import annotations

from .._spec._artifacts import hf_load
from .._spec._core import DELEGATED, ArtifactInput
from ._eval import EpisodeResult, RunResult

__all__ = [
    "DELEGATED",
    "ArtifactInput",
    "EpisodeResult",
    "RunResult",
    "hf_load",
]
