"""The declared construction-parameter primitive: ``Param`` and ``ParamSpec``.

A factory/model declares the *parameters that mint it* -- the introspectable,
validatable surface a managed dashboard presents, validates before paying GPU
cost, and sweeps. Construction (this) is orthogonal to adaptation
(:mod:`rlmesh.adapters`): params describe how to *build* an env/model; adapters
map the obs/action contract of an already-built one.

The leaf is named ``Param`` (not ``Field``): :mod:`rlmesh.adapters` already
exports ``Field`` as a contiguous obs-leaf slice, a different concept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

# Sentinel default meaning "required". Distinct from ``None``, which is a valid
# declared default; identity (``is``) is the only correct test for it.
_UNSET: Any = object()

ExtraPolicy = Literal["forbid", "passthrough"]


@dataclass(frozen=True)
class Vector:
    """A fixed-length float-vector ``Param`` type, passed as ``Param(type=Vector(3))``.

    The *value* stays a plain tuple of ``dim`` finite floats; a JSON-bound
    ``list`` is canonicalized to a tuple. ``Vector`` is only a schema descriptor
    sitting in :attr:`Param.type` next to ``"float"``/``"enum"`` -- it is not a
    container the runtime carries. ``unit`` requires unit L2 norm (the
    quaternion/direction case). Sweeps come from ``choices`` (a list of whole
    vectors): there is no continuous vector domain.
    """

    dim: int
    unit: bool = False

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError(f"Vector dim must be >= 1, got {self.dim}")


@dataclass(frozen=True)
class Param:
    """One declared construction parameter -- validated, presentable, sweepable.

    Declaring a ``Param`` *is* the act of marking a knob primary: it is presented
    as a first-class widget, validated (type/choices/required) before
    construction, and offered as a sweep axis. Undeclared keyword args of
    ``make``/``load`` are still presented and type-checked for free (the
    signature-derived tier); ``Param`` enriches one with domains, choices,
    grouping, and sweepability.

    Args:
        name: The keyword name, matching the ``make``/``load`` parameter.
        type: A Python type (``int``/``float``/``str``/``bool``) or its string
            name, ``"enum"`` (domain defined entirely by ``choices``), or a
            :class:`Vector` (a fixed-length float vector).
        default: The default value; omit for a required param.
        choices: Allowed values -- the enumeration/sweep axis; a supplied value
            outside them is rejected.
        description: Human-facing help text for the dashboard widget.
        group: Optional UI grouping label (advisory; the core never reads it).
    """

    name: str
    type: type | str | Vector = "str"
    default: Any = _UNSET
    choices: tuple[Any, ...] | None = None
    description: str = ""
    group: str | None = None

    @property
    def required(self) -> bool:
        """Whether the param has no default and must be supplied."""
        return self.default is _UNSET


@dataclass(frozen=True)
class ParamSpec:
    """A factory's/model's declared construction-parameter surface.

    The validated ceiling over the free signature-derived floor. ``extra``
    governs the single boundary door for undeclared keys:

    * ``"forbid"`` (default): an undeclared key raises before construction, so a
      typo (``robtos=``) fails pre-GPU instead of vanishing.
    * ``"passthrough"``: undeclared keys forward verbatim through the author's
      own ``**kwargs`` into a third-party constructor (the escape hatch for a
      wrapper author). Bounded by that ``**kwargs`` -- never by any downstream
      target -- so it cannot collide with a body-computed argument.
    """

    params: tuple[Param, ...] = ()
    extra: ExtraPolicy = "forbid"

    def __init__(
        self,
        *params: Param,
        extra: ExtraPolicy = "forbid",
    ) -> None:
        # Variadic ``*params`` reads at the call site like a literal list of
        # knobs; a frozen dataclass forbids normal assignment, so set via the
        # base ``__setattr__``.
        object.__setattr__(self, "params", tuple(params))
        object.__setattr__(self, "extra", extra)


__all__ = ["Param", "ParamSpec", "Vector"]
