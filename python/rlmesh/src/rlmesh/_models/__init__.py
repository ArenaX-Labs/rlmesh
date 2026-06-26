"""Model-side runtime: the eval loop and result types.

The kept neutral core a framework ``Model`` runs against. ``Model(predict).run(env)``
connects a predict callable to an env.

* :class:`RunResult` -- the typed result of a ``Model.run`` eval.
* :data:`NO_ADAPTER` -- explicitly skip RLMesh adapter resolution.
"""

from __future__ import annotations

from ._adapter_mode import NO_ADAPTER
from ._eval import RANDOM_SAMPLE, EpisodeResult, RunResult, Session
from .base import run, session

__all__ = [
    "NO_ADAPTER",
    "RANDOM_SAMPLE",
    "EpisodeResult",
    "RunResult",
    "Session",
    "run",
    "session",
]
