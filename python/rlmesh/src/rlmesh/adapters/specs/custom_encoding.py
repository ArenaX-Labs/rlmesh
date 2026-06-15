"""A host-side custom rotation encoding layered on a known base encoding.

A :class:`CustomEncoding` lets a model spec use a rotation packing RLMesh does
not ship as a built-in, without changing the closed native vocabulary. The
field resolves natively as its ``base`` encoding -- so role matching, range
mapping, and the env->base conversion are unchanged -- and the adapter applies
host-side transforms at the field boundary: ``from_base`` on the observation
side (base -> custom) and ``to_base`` on the action side (custom -> base).

See :doc:`the adapters guide </user-guide/adapters>`. For a general, stable
convention (a published model's), add it to the native ``RotationEncoding``
instead; this host-side path is the one-off / proprietary escape hatch that
keeps the declarative pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from ..._rlmesh import ROTATION_DIMS
from .vocabularies import RotationEncoding

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ...numpy import NumpyArray

# One arm of a custom encoding: an in-process callable, or a ``module:callable``
# entrypoint string (imported only under ``resolve(..., trust_entrypoints=True)``).
RotationTransform: TypeAlias = Callable[["NumpyArray"], "ArrayLike"]


@dataclass(frozen=True)
class CustomEncoding:
    """A rotation packing layered on a known native ``base`` encoding.

    The two arms mirror the native encode/decode pair; each is required only on
    the side it is used (validated at resolve time). The packing must preserve
    the base width (``ROTATION_DIMS[base]``).

    Attributes:
        base: The native encoding the field resolves as.
        from_base: ``base -> custom``, used when the encoding tags an
            observation (model) state component. An in-process callable, or a
            ``"module:callable"`` entrypoint string. None when action-only.
        to_base: ``custom -> base``, used when the encoding tags an action
            component. A callable or entrypoint string. None when obs-only.
        name: Display name surfaced in :meth:`Adapter.describe`.
    """

    base: RotationEncoding
    from_base: RotationTransform | str | None = None
    to_base: RotationTransform | str | None = None
    name: str = "custom"

    def __post_init__(self) -> None:
        if self.base not in ROTATION_DIMS:
            raise ValueError(
                "CustomEncoding.base must be a native rotation encoding "
                f"{sorted(ROTATION_DIMS)}, got {self.base!r}"
            )
        if self.from_base is None and self.to_base is None:
            raise ValueError(
                "CustomEncoding needs at least one of from_base (observation "
                "side) or to_base (action side)"
            )
        arms = [arm for arm in (self.from_base, self.to_base) if arm is not None]
        if any(isinstance(arm, str) for arm in arms) and any(
            not isinstance(arm, str) for arm in arms
        ):
            raise ValueError(
                "CustomEncoding arms must agree: both in-process callables or "
                "both 'module:callable' entrypoint strings, not a mix"
            )

    @property
    def is_entrypoint(self) -> bool:
        """Whether the arms are ``module:callable`` strings (serializable)."""
        return isinstance(self.from_base, str) or isinstance(self.to_base, str)

    @property
    def width(self) -> int:
        """Element width of the base (and therefore custom) encoding."""
        return int(ROTATION_DIMS[self.base])


__all__ = ["CustomEncoding", "RotationTransform"]
