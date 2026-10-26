"""Model-side runtime: the eval loop and result types.

The kept neutral core a framework ``Model`` runs against. ``Model(predict).run(env)``
connects a predict callable to an env.

* :class:`RunResult` -- the typed result of a ``Model.run`` eval.
* :data:`NO_ADAPTER` -- explicitly skip RLMesh adapter resolution.
"""

from __future__ import annotations

from .._upgrade import session
from ._adapter_mode import NO_ADAPTER
from ._binding import Binding
from ._eval import (
    RANDOM_SAMPLE,
    EpisodeResult,
    EvalReport,
    Interrupted,
    RunContext,
    RunHooks,
    RunResult,
    StepEvent,
    TelemetryRow,
)
from ._read import Reader
from ._view import FrameSources, View
from .base import run

__all__ = [
    "NO_ADAPTER",
    "RANDOM_SAMPLE",
    "Binding",
    "EpisodeResult",
    "EvalReport",
    "FrameSources",
    "Interrupted",
    "Reader",
    "RunContext",
    "RunHooks",
    "RunResult",
    "StepEvent",
    "TelemetryRow",
    "View",
    "run",
    "session",
]
