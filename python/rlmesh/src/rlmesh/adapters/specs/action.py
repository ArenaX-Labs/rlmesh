"""Action layout dataclasses shared by env and model declarations."""

from __future__ import annotations

from dataclasses import dataclass

from .custom_encoding import CustomEncoding
from .vocabularies import RotationEncoding


@dataclass(frozen=True)
class Actuator:
    """One contiguous slice of an action vector.

    Attributes:
        role: Semantic role used for matching, e.g. ``action/gripper``. ``None``
            makes the actuator *opaque*: it occupies ``dim`` dims of the action
            with the constant ``fill``, matched by no model output -- the
            action-side mirror of a role-less :class:`~rlmesh.adapters.Field`. Use
            it for dims the env requires but no model produces (e.g. a control-mode
            selector). An opaque actuator carries only ``dim`` and ``fill``.
        dim: Number of action dimensions occupied by this component.
        fill: Constant emitted for each dim of an opaque (role-less) actuator.
            Defaults to ``0.0``; inert (must stay ``0.0``) on a roled actuator.
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
        clip: Clamp this actuator's mapped value to its declared ``range``. The
            per-component safety clamp the global ``Action.clip`` cannot give a
            mixed-range action: a global clip applies one bound to every dim, so
            it is wrong when dims have different ranges (e.g. delta-pos in
            ``[-1, 1]`` but rotation in ``[-pi/2, pi/2]``). ``clip=True`` requires
            ``range``. (ponytail: a bool clamps to ``range``; widen to a
            ``(low, high)`` overload, like ``Image.normalize``, only if an
            asymmetric clamp is ever needed.)

    ``scale``, ``invert``, ``threshold``, and ``clip`` are env-side corrections:
    they declare the env actuator's convention and are applied to the incoming
    model value after the declared formats (rotation, range) are bridged, in the
    order scale, invert, threshold, ``binary``, then ``clip``. Declared once on
    the env, every model evaluated against it inherits the correction.
    """

    # `role` is optional (None => opaque); `dim` then needs a default for field
    # ordering, like Field. dim=0 is never valid and is rejected below.
    role: str | None = None
    dim: int = 0
    encoding: RotationEncoding | CustomEncoding | None = None
    range: tuple[float, float] | None = None
    binary: bool = False
    # Appended after binary so the existing positional layout (..., range, binary)
    # is unchanged; these are keyword-only in practice.
    scale: float | None = None
    invert: bool = False
    threshold: float | None = None
    clip: bool = False
    fill: float = 0.0

    def __post_init__(self) -> None:
        # dim defaults to 0 only to satisfy field ordering (the optional role
        # precedes it); 0 means it was omitted. A negative dim is left to the Rust
        # codec, which rejects it with a field-path-annotated message.
        if self.dim == 0:
            raise ValueError(f"Actuator {self.role!r}: dim is required (>= 1)")
        if self.role is None:
            # Opaque: emits a constant, so the model-mapping fields are meaningless
            # (mirrors a role-less Field's skip rule). Only dim + fill are allowed.
            if (
                self.encoding is not None
                or self.range is not None
                or self.binary
                or self.scale is not None
                or self.invert
                or self.threshold is not None
                or self.clip
            ):
                raise ValueError(
                    "a role-less (opaque) Actuator carries only dim and fill; drop "
                    "encoding/range/scale/invert/threshold/binary/clip"
                )
            return
        if self.fill != 0.0:
            raise ValueError(
                f"Actuator {self.role!r}: fill applies only to a role-less (opaque) "
                "actuator; a roled actuator takes its values from the model"
            )
        # clip-to-range needs a range to clamp to; catch the contradiction at
        # authoring rather than at resolve (the resolver enforces it too).
        if self.clip and self.range is None:
            raise ValueError(
                f"Actuator {self.role!r}: clip=True clamps to range, so range "
                "must be set"
            )
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

    Action chunking is no longer a spec knob: the execution horizon is chosen by the
    runtime (``execution_horizon`` on ``ConfigureRoute``), and a chunked policy
    declares a ``predict_chunk`` corner rather than an ``execute_horizon``.
    """

    components: tuple[Actuator, ...]
    clip: tuple[float, float] | None = None

    def __init__(
        self,
        *components: Actuator,
        clip: tuple[float, float] | None = None,
    ) -> None:
        object.__setattr__(self, "components", tuple(components))
        object.__setattr__(self, "clip", clip)

    @property
    def dim(self) -> int:
        """Total action vector length."""
        return sum(component.dim for component in self.components)


__all__ = ["Action", "Actuator"]
