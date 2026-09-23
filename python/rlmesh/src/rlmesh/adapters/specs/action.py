"""Action layout dataclasses shared by env and model declarations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ._codec import axis_or_scalar, check_labels
from .custom_encoding import CustomEncoding
from .vocabularies import Frame, Reference, RotationEncoding


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
        fill: Constant emitted for each dim of an opaque (role-less) actuator,
            and the fallback for an ``optional`` roled actuator. Defaults to
            ``0.0``; inert (must stay ``0.0``) on a roled, non-optional actuator.
            On an ``optional`` env actuator that carries ``labels`` it may be a
            sequence with one value per axis (serialized as ``axis_fill``): the
            value each axis a model's label subset leaves undriven is held at,
            and the whole-actuator fallback when no model output drives it.
        optional: On a roled actuator, make the role optional -- if no model
            output declares it, fill the actuator's ``dim`` dims with ``fill``
            instead of failing resolution (the action-side mirror of a model
            input's ``optional`` zero-fill). A model that does output the role
            drives it normally. Meaningless on a role-less actuator (already
            always filled).
        encoding: Rotation encoding when the component is a rotation.
        range: Optional ``(low, high)`` range of the component values.
        scale: Optional multiplier applied to the model value for this role:
            one float, or a sequence with one value per axis in this side's own
            ``labels`` order (serialized as ``axis_scale``).
        offset: Optional addend applied after ``scale`` (``value * scale +
            offset``), a float or a per-axis sequence like ``scale``: the stand
            pose a joint-position target is expressed around. Keyword-only.
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
            ``range``.
        frame: Coordinate frame a Cartesian command's axes are expressed in.
            Required of an absolute pose (``action/eef_*``) under the
            require-frames tier; a delta (``action/delta_eef_*``) may declare it
            too (a tool-frame delta and a base-frame delta are different
            commands), and when both sides do they must agree. Keyword-only,
            omitted from the wire when unset.
        reference: What a *delta* command is integrated against
            (``action/delta_eef_*``): an env declares what its Cartesian
            controller adds the delta to -- the measured pose (``"current"``) or
            the last commanded target (``"target"``) -- and a model declares what
            it was trained against. A disagreement is a hard resolve error, which
            is what stops an absolute-pose head from binding cleanly to a
            delta controller. ``reference`` says what a delta is added to;
            ``frame`` says which axes it is expressed in. Declare both unless
            the controller fixes the frame.
        part: The body part this actuator drives, when the role repeats across
            a body (``"left_arm"``, ``"right_arm"``, ...): an identity key the
            resolver matches on, never a value it checks. Keyword-only and
            omitted from the wire when unset; an opaque actuator may not carry
            one.
        labels: The axis names this actuator drives, ``dim`` of them in this
            side's own order (``embodiments.GO2.joints``). When both sides carry
            them the model's output is scattered onto the env's axes by name:
            equal sets in another order are a permutation; a model that names a
            strict subset drives those axes and, if the env actuator is
            ``optional``, the rest take its ``fill``. Keyword-only and omitted
            from the wire when unset; an opaque actuator may not carry them.

    ``scale``, ``offset``, ``invert``, and ``threshold`` declare a side's actuator
    convention. They can be set on either side and compose as literal transforms
    applied after the declared formats (rotation, range) are bridged -- model-side
    first (the model's own output convention, in the model's own axis order),
    then the scatter by labels, then env-side -- each in the order scale, offset,
    invert, threshold, then ``binary``. So an env declares its quirk once for every
    model to inherit, and a model whose output differs from a shared env it cannot
    edit declares the bridge on its own actuator (e.g. a gripper-sign flip as
    ``invert=True``). ``clip`` is the exception: it stays env-side only, clamping to
    the env actuator's ``range``.
    """

    role: str | None = None
    dim: int = 0
    encoding: RotationEncoding | CustomEncoding | None = None
    range: tuple[float, float] | None = None
    binary: bool = False
    scale: float | Sequence[float] | None = None
    invert: bool = False
    threshold: float | None = None
    clip: bool = False
    fill: float | Sequence[float] = 0.0
    optional: bool = False
    frame: Frame | None = field(default=None, kw_only=True)
    reference: Reference | None = field(default=None, kw_only=True)
    part: str | None = field(default=None, kw_only=True)
    offset: float | Sequence[float] | None = field(default=None, kw_only=True)
    labels: Sequence[str] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError(
                f"Actuator {self.role!r}: dim must be >= 1, got {self.dim}"
            )
        for name in ("scale", "offset", "fill"):
            object.__setattr__(
                self,
                name,
                axis_or_scalar("Actuator", self.role, name, getattr(self, name)),
            )
        object.__setattr__(
            self, "labels", check_labels("Actuator", self.role, self.labels)
        )
        if self.role is None:
            if (
                self.encoding is not None
                or self.range is not None
                or self.binary
                or self.scale is not None
                or self.offset is not None
                or self.invert
                or self.threshold is not None
                or self.clip
                or self.optional
                or self.frame is not None
                or self.reference is not None
                or self.part is not None
                or self.labels is not None
                or isinstance(self.fill, tuple)
            ):
                raise ValueError(
                    "a role-less (opaque) Actuator carries only dim and a scalar fill; "
                    "drop encoding/range/scale/offset/invert/threshold/binary/clip/"
                    "optional/frame/reference/part/labels"
                )
            return
        if self.labels is not None and len(self.labels) != self.dim:
            raise ValueError(
                f"Actuator {self.role!r}: dim {self.dim} disagrees with the "
                f"{len(self.labels)} labels; one label per axis"
            )
        if isinstance(self.fill, tuple):
            if not self.optional or self.labels is None:
                raise ValueError(
                    f"Actuator {self.role!r}: a per-axis fill applies only to an "
                    "optional, labeled actuator (set optional=True and labels=)"
                )
            if len(self.fill) != self.dim:
                raise ValueError(
                    f"Actuator {self.role!r}: fill has {len(self.fill)} values but "
                    f"dim is {self.dim}; one fill per axis"
                )
        elif self.fill != 0.0 and not self.optional:
            raise ValueError(
                f"Actuator {self.role!r}: fill applies only to a role-less (opaque) "
                "or optional actuator; a roled, non-optional actuator takes its "
                "values from the model"
            )
        if self.clip and self.range is None:
            raise ValueError(
                f"Actuator {self.role!r}: clip=True clamps to range, so range "
                "must be set"
            )
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

    The execution horizon is chosen by the runtime (``execution_horizon`` on
    ``ResolveAdapter``), and a chunked policy implements ``predict_chunk``.
    """

    components: tuple[Actuator, ...]
    clip: tuple[float, float] | None = None

    def __init__(
        self,
        *components: Actuator,
        clip: tuple[float, float] | None = None,
    ) -> None:
        if not components:
            raise ValueError("Action needs at least one actuator")
        object.__setattr__(self, "components", tuple(components))
        object.__setattr__(self, "clip", clip)

    @property
    def dim(self) -> int:
        """Total action vector length."""
        return sum(component.dim for component in self.components)


__all__ = ["Action", "Actuator"]
