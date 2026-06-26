"""Action layout dataclasses shared by env and model declarations."""

from __future__ import annotations

from dataclasses import dataclass

from .custom_encoding import CustomEncoding
from .vocabularies import RotationEncoding


@dataclass(frozen=True)
class Actuator:
    """One contiguous slice of an action vector.

    Attributes:
        role: Semantic role used for matching, e.g. ``action/gripper``.
        dim: Number of action dimensions occupied by this component.
        encoding: Rotation encoding when the component is a rotation.
        range: Optional ``(low, high)`` range of the component values.
        scale: Optional multiplier applied to the model value for this role.
        invert: Negate the model value for this role (equivalent to
            ``scale=-1`` but explicit; the common gripper-sign correction).
        threshold: Subtract this from the value, recentering the decision
            boundary -- typically paired with ``binary`` so the snap splits at
            ``threshold`` instead of zero.
        binary: Whether the component encodes a binary decision (resolved
            adapters snap the value to a definite side after range mapping:
            ``>= 0`` opens (``+1``), below closes (``-1``); a value exactly on the
            boundary opens rather than emitting an undefined ``0``).

    ``scale``, ``invert``, and ``threshold`` are env-side corrections: they
    declare the env actuator's convention and are applied to the incoming model
    value after the declared formats (rotation, range) are bridged, in the order
    scale, invert, threshold, then ``binary``. Declared once on the env, every
    model evaluated against it inherits the correction.
    """

    role: str
    dim: int
    encoding: RotationEncoding | CustomEncoding | None = None
    range: tuple[float, float] | None = None
    binary: bool = False
    # Appended after binary so the existing positional layout (..., range, binary)
    # is unchanged; these are keyword-only in practice.
    scale: float | None = None
    invert: bool = False
    threshold: float | None = None

    def __post_init__(self) -> None:
        # dim >= 0 is enforced by the Rust codec (u32) at serialize/normalize.
        # The CustomEncoding width law is host-only: CustomEncoding never crosses
        # the wire, so Rust cannot own it -- it stays here as construction sugar.
        if (
            isinstance(self.encoding, CustomEncoding)
            and self.dim != self.encoding.width
        ):
            raise ValueError(
                f"Actuator {self.role!r} with a CustomEncoding on base "
                f"{self.encoding.base!r} must keep its width: dim must be "
                f"{self.encoding.width}, got {self.dim}"
            )


@dataclass(frozen=True, init=False)
class Action:
    """Ordered action actuators plus optional clipping bounds.

    Actuators are passed positionally, mirroring the observation-side
    :class:`~rlmesh.adapters.Split`::

        Action(Actuator(DELTA_POS, 3), Actuator(GRIPPER, 1))

    Attributes:
        components: Action actuators in vector order.
        clip: Optional ``(low, high)`` clip applied to the final vector.
        execute_horizon: How many actions ``predict`` returns as a chunk and the
            engine replays before predicting again; ``1`` (the default) predicts
            every step. When ``> 1`` the model output's leading axis is the chunk
            axis -- the engine replays up to ``execute_horizon`` of them per chunk,
            re-planning from a fresh observation when the queue drains. A model-side
            knob; the env declaration leaves it ``1``.
    """

    components: tuple[Actuator, ...]
    clip: tuple[float, float] | None = None
    execute_horizon: int = 1

    def __init__(
        self,
        *components: Actuator,
        clip: tuple[float, float] | None = None,
        execute_horizon: int = 1,
    ) -> None:
        object.__setattr__(self, "components", tuple(components))
        object.__setattr__(self, "clip", clip)
        object.__setattr__(self, "execute_horizon", execute_horizon)

    @property
    def dim(self) -> int:
        """Total action vector length."""
        return sum(component.dim for component in self.components)


__all__ = ["Action", "Actuator"]
