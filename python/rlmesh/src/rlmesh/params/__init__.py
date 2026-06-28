"""Declared construction parameters for envs and models.

Declare the parameters that *mint* an env/model so a managed dashboard can
present, validate (before GPU cost), and sweep variations of it. Attach a
:class:`ParamSpec` of :class:`Param` objects as the ``params`` class attribute on
an :class:`~rlmesh.EnvFactory` (validated against ``make``) or a ``Model``
(validated against ``load``). ``Param`` and ``ParamSpec`` are re-exported at the
top level as :data:`rlmesh.Param` / :data:`rlmesh.ParamSpec`.
"""

from __future__ import annotations

from ._resolve import (
    PARAM_METADATA_KEY,
    MissingParamError,
    ParamError,
    UnknownParamError,
)
from ._spec import Param, ParamSpec

__all__ = [
    "PARAM_METADATA_KEY",
    "MissingParamError",
    "Param",
    "ParamError",
    "ParamSpec",
    "UnknownParamError",
]
