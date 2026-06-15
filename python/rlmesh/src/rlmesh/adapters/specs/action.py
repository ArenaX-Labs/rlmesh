"""Action layout dataclasses shared by env and model declarations."""

from __future__ import annotations

from dataclasses import dataclass

from .custom_encoding import CustomEncoding
from .validation import check_non_negative
from .vocabularies import RotationEncoding


@dataclass(frozen=True)
class ActionComponent:
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
            adapters snap the value to ``sign`` after range mapping).

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
        check_non_negative(self.dim, "ActionComponent.dim")
        if (
            isinstance(self.encoding, CustomEncoding)
            and self.dim != self.encoding.width
        ):
            raise ValueError(
                f"ActionComponent {self.role!r} with a CustomEncoding on base "
                f"{self.encoding.base!r} must keep its width: dim must be "
                f"{self.encoding.width}, got {self.dim}"
            )


@dataclass(frozen=True, init=False)
class ActionLayout:
    """Ordered action components plus optional clipping bounds.

    Components are passed positionally, mirroring the observation-side
    :class:`~rlmesh.adapters.StateLayout`::

        ActionLayout(ActionComponent(DELTA_POS, 3), ActionComponent(GRIPPER, 1))

    Attributes:
        components: Action components in vector order.
        clip: Optional ``(low, high)`` clip applied to the final vector.
    """

    components: tuple[ActionComponent, ...]
    clip: tuple[float, float] | None = None

    def __init__(
        self,
        *components: ActionComponent,
        clip: tuple[float, float] | None = None,
    ) -> None:
        object.__setattr__(self, "components", tuple(components))
        object.__setattr__(self, "clip", clip)

    @property
    def dim(self) -> int:
        """Total action vector length."""
        return sum(component.dim for component in self.components)


__all__ = ["ActionComponent", "ActionLayout"]
