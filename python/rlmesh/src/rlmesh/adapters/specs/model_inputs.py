"""Input leaf dataclasses declared by models.

The model leaves are bare (no ``key``): placement in the input tree *is* the
payload position. Only image/state/text/custom exist; a :class:`Concat` is the
multi-part state leaf (a single tensor concatenated from several role parts).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from typing import Any, Literal, TypeAlias

from ._codec import axis_or_scalar, check_accept_set, check_labels, one_or_many
from .custom_encoding import CustomEncoding
from .vocabularies import (
    ChannelOrder,
    CropMode,
    FitMode,
    Frame,
    ImageLayout,
    Resample,
    RotationEncoding,
    StackPad,
    StackSpec,
)

ObsTransform: TypeAlias = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class Image:
    """An image input expected by a model.

    There is no ``key`` -- placement in the input tree *is* the payload position.

    Attributes:
        role: Semantic role matched against env image features.
        height: Target image height, or None to keep the env height.
        width: Target image width, or None to keep the env width.
        layout: Axis layout the model expects (``"hwc"`` the default, or
            ``"chw"``). This is the model author's declaration: when it differs
            from the env's layout the adapter transposes silently to match, so a
            model that wants ``"chw"`` must say so -- an omitted ``layout`` means
            ``"hwc"`` and the env's frame is fed through unchanged. (``channels``
            guards the channel *count*, not the axis order.)
        channels: Channel count the model expects (e.g. 3 for RGB, 1 for
            grayscale). When set, a resolve error if the env image differs;
            the adapter does not convert between channel counts.
        dtype: NumPy dtype name the model expects.
        normalize: Whether (and into what range) to map 8-bit pixel values
            before casting: ``False`` (off, the default), ``True`` (the
            conventional ``[0, 1]``), or a ``(low, high)`` pair for a model
            trained on a different range (e.g. ``(-1.0, 1.0)``). One field, so an
            on/off flag can never disagree with a range; ``False`` is an
            authoritative off-switch.
        lead_dims: Number of leading singleton axes to add (batch/time).
        upside_down: Whether the model was trained on images rotated 180
            degrees from the canonical upright orientation (a true rotation --
            rows and columns reversed -- not a vertical flip). Declared on both
            ends: the adapter rotates only when the env and model disagree, so an
            env that already renders upside-down pairs with a model that also
            sets ``upside_down`` and no rotation happens.
        resample: Resize algorithm the model's training pipeline used. An
            un-suffixed name has OpenCV/torch semantics, an ``_aa`` suffix has
            PIL's (an antialiased filter whose support widens with the
            downscale factor): ``"bilinear"`` (4-tap half-pixel-center
            bilinear; the default, which most trained policies match),
            ``"bilinear_aa"`` / ``"bicubic_aa"`` / ``"lanczos3_aa"`` (PIL's
            ``BILINEAR`` / ``BICUBIC`` / ``LANCZOS``), or ``"area"``
            (OpenCV ``INTER_AREA``). Bare ``"bicubic"``/``"lanczos3"`` are not
            accepted -- name the library whose kernel you trained against.
        allow_upscale: Permit a target larger than the env's native resolution
            (interpolating detail that is not there). Off by default: an
            upscaling target is a resolve error unless this is set.
        fit: How to reconcile a target whose aspect ratio differs from the env
            image: ``"stretch"`` (distort), ``"crop"`` (cover + center-crop), or
            ``"pad"`` (letterbox) -- or a sequence of them in preference order.
            The resolver picks, per env, the first that does not need a
            disallowed upscale, so one spec can crop a large camera and
            letterbox a small one. Required only on an aspect mismatch; absent
            it, an aspect-changing resize is a resolve error.
        optional: Zero-fill a black frame when the env does not provide this
            camera, instead of failing resolution. Needs ``height``, ``width``,
            and ``channels`` so the blank can be sized.
        fill: Fill value (0-255) for the blank frame an ``optional`` camera
            synthesizes when the env offers no matching image; default 0
            (black). Requires ``optional=True`` (without it the fill could
            never take effect, so setting it alone is a construction error);
            named to match ``Actuator.fill``.
        stack: Number of frames to stack on a new leading axis (frame
            history). ``1`` (default) means no stacking. Stacking is applied by
            the adapter core from an episode-keyed rolling window, padded at the
            start of an episode (see ``stack_pad``) and cleared on ``reset``.
            Which frames it gathers is ``offsets`` (or ``stride``); by default
            they are the last ``stack`` consecutive ones. The env still sends
            one frame per step either way.
        size: Convenience for square targets -- sets both ``height`` and
            ``width``. Pass ``size`` or ``height``/``width``, not both.
        crop: Side fraction of the frame a center crop keeps, in ``(0, 1]``
            (``2/3`` keeps the middle two thirds of each axis). Pass ``crop``
            or ``crop_area``, not both.
        crop_area: The same center crop stated as an *area* fraction, in
            ``(0, 1]`` -- the form training pipelines usually quote ("a 90%
            center crop"). The side fraction is its square root
            (``0.9`` -> ``0.949``).
        crop_mode: How the crop box is taken. ``"zoom"`` (the default)
            resamples the fractional box straight to the target -- PIL's
            ``Image.resize(size, box=...)``, one pass, no intermediate
            rounding. ``"slice"`` cuts an integer center box out first and
            resizes that, which is what a ``numpy`` slice in a training
            pipeline does.
        jpeg_quality: Quality of a JPEG round-trip applied to the upright
            frame *before* the crop and resize, on the IJG 1-100 scale (``95``
            is the usual training value). A *declaration*, not a request: a
            pipeline that stored its frames as JPEG fed the model the codec's
            artifacts, so the adapter reproduces them rather than handing the
            model a cleaner frame than it was trained on. Baseline sequential,
            4:2:0 box-averaged chroma, standard IJG tables; needs a 3-channel
            image.
        channel_order: Channel order the model was trained on. ``"bgr"`` swaps
            red and blue after the spatial ops and before the dtype cast, and
            needs a 3-channel image.
        offsets: The frame window ``stack`` gathers, as non-positive offsets
            from the current step -- oldest first, ending at ``0``.
            ``(-6, -4, -2, 0)`` is "every second frame of the last seven".
            ``None`` (the default) is the contiguous window ``stack`` already
            describes. ``offsets`` and ``stack`` are never inferred from each
            other: both are declared and they must agree (``len(offsets) ==
            stack``), and the window may span at most 128 consecutive frames.
        stride: Convenience for an evenly spaced window -- ``stack=4,
            stride=2`` is ``offsets=(-6, -4, -2, 0)``. Pass ``stride`` or
            ``offsets``, not both.
        stack_pad: What fills the window before the episode has produced
            enough frames: ``"first"`` (the default) replicates the first
            observed frame, ``"black"`` a raw 8-bit ``0`` frame pushed through
            this input's pipeline -- so under ``normalize=(-1.0, 1.0)`` a black
            pad frame is ``-1.0``, not ``0.0``. Requires ``stack > 1``.
        render: The camera resolution this model was trained against, as a
            square ``int`` or an ``(height, width)`` pair (each axis 1-4096).
            An *assertion*, not a request: the adapter never resizes a camera
            to reach it. The platform binds the environment's camera dial from
            it before the run; resolution then fails with ``RenderMismatch`` if
            the camera it bound does not actually render at this size. Declare
            it instead of pinning the env's camera width/height by hand.
        part: The body part the wanted camera sits on, when the role repeats
            across a body (``"head"``, ``"left_arm"``, ...): an identity key
            the resolver matches on. Naming one binds only that camera; naming
            none binds the env's only camera of the role under any part (with
            an ``info``) and fails when there are several. Keyword-only and
            omitted from the wire when unset.
    """

    role: str
    height: int | None = None
    width: int | None = None
    layout: ImageLayout = "hwc"
    channels: int | None = None
    dtype: str = "uint8"
    normalize: bool | tuple[float, float] = False
    lead_dims: int = 0
    upside_down: bool = False
    resample: Resample = "bilinear"
    allow_upscale: bool = False
    fit: FitMode | Sequence[FitMode] | None = None
    optional: bool = False
    fill: int | None = None
    stack: int = 1
    # Keyword-only and appended after `stack`: the positional prefix through
    # `stack` is frozen, so specs written against it keep constructing.
    crop: float | None = field(default=None, kw_only=True)
    crop_area: float | None = field(default=None, kw_only=True)
    crop_mode: CropMode = field(default="zoom", kw_only=True)
    jpeg_quality: int | None = field(default=None, kw_only=True)
    channel_order: ChannelOrder = field(default="rgb", kw_only=True)
    render: int | tuple[int, int] | None = field(default=None, kw_only=True)
    offsets: StackSpec | None = field(default=None, kw_only=True)
    stack_pad: StackPad = field(default="first", kw_only=True)
    part: str | None = field(default=None, kw_only=True)
    size: InitVar[int | None] = None
    # Kw-only like the fields it sugars, so the positional signature stays
    # exactly what specs were written against.
    stride: InitVar[int | None] = field(default=None, kw_only=True)

    def __post_init__(self, size: int | None, stride: int | None) -> None:
        # size= is construction sugar (sets both height and width). The
        # non-negative bounds (height/width/lead_dims) and the stack 1..=64 bound
        # are enforced by the Rust codec at serialize/normalize (u32 + de_stack).
        if size is not None:
            if self.height is not None or self.width is not None:
                raise ValueError(
                    f"Image {self.role!r}: pass size=, or height=/width=, not both"
                )
            object.__setattr__(self, "height", size)
            object.__setattr__(self, "width", size)
        # crop and crop_area are one box said two ways; declaring both leaves
        # no honest precedence rule, so it is a construction error rather than
        # a silent winner. The bound matches the Rust codec's wire guard.
        if self.crop is not None and self.crop_area is not None:
            raise ValueError(
                f"Image {self.role!r}: pass crop (a side fraction) or crop_area "
                "(an area fraction), not both"
            )
        for name in ("crop", "crop_area"):
            fraction = getattr(self, name)
            if fraction is None:
                continue
            fraction = float(fraction)
            if not 0.0 < fraction <= 1.0:
                raise ValueError(
                    f"Image {self.role!r}: {name} must be a fraction in (0, 1], "
                    f"got {fraction}"
                )
            object.__setattr__(self, name, fraction)
        # The IJG scale the encoder is written on; matches the Rust codec's
        # wire guard, so a spec rejected here is rejected there too.
        if self.jpeg_quality is not None and not 1 <= self.jpeg_quality <= 100:
            raise ValueError(
                f"Image {self.role!r}: jpeg_quality must be between 1 and 100, "
                f"got {self.jpeg_quality}"
            )
        # A square int is shorthand for the (height, width) pair the wire
        # carries; normalizing here means the field has one shape everywhere
        # (mirrors `normalize`'s pair coercion). The bound matches the Rust
        # codec's wire guard.
        if self.render is not None:
            render = self.render
            if isinstance(render, int):
                render = (render, render)
            else:
                try:
                    height, width = render
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Image {self.role!r}: render must be an int or a "
                        f"(height, width) pair, got {self.render!r}"
                    ) from None
                render = (int(height), int(width))
            for axis, value in zip(("height", "width"), render, strict=True):
                if not 1 <= value <= 4096:
                    raise ValueError(
                        f"Image {self.role!r}: render {axis} must be between 1 "
                        f"and 4096, got {value}"
                    )
            object.__setattr__(self, "render", render)
        # stride= is construction sugar for an evenly spaced window; offsets is
        # the general form. Declaring both leaves no honest precedence rule.
        # The window LAW (non-positive, increasing, ending at 0, agreeing with
        # stack, within the span ceiling) is enforced once, by the Rust resolver,
        # so both engines can only ever say the same thing about it.
        if stride is not None:
            if self.offsets is not None:
                raise ValueError(
                    f"Image {self.role!r}: pass stride=, or offsets=, not both"
                )
            if stride < 1:
                raise ValueError(
                    f"Image {self.role!r}: stride must be >= 1, got {stride}"
                )
            object.__setattr__(
                self,
                "offsets",
                tuple(-(self.stack - 1 - step) * stride for step in range(self.stack)),
            )
        elif self.offsets is not None:
            object.__setattr__(self, "offsets", tuple(int(v) for v in self.offsets))
        if self.fill is not None and not self.optional:
            raise ValueError(
                f"Image {self.role!r}: fill only applies to an optional camera; "
                "set optional=True or drop fill"
            )
        # A single fit stays a string; a preference list normalizes to a tuple
        # (hashable, round-trips by value) -- mirrors the rotation accept-set.
        object.__setattr__(self, "fit", one_or_many(self.fit))
        # normalize overloads a bool with a (low, high) range. A pair coerces to a
        # validated float tuple (finite, low <= high), mirroring the Rust codec's
        # range guard; a bool passes through, and False stays the off-switch.
        norm = self.normalize
        if norm is not True and norm is not False:
            try:
                low, high = norm
            except (TypeError, ValueError):
                raise ValueError(
                    "Image.normalize must be a bool or a (low, high) range, "
                    f"got {norm!r}"
                ) from None
            low, high = float(low), float(high)
            if not (math.isfinite(low) and math.isfinite(high)):
                raise ValueError("Image.normalize range must be finite")
            if low > high:
                raise ValueError(
                    f"Image.normalize range min must be <= max, got ({low}, {high})"
                )
            object.__setattr__(self, "normalize", (low, high))


@dataclass(frozen=True, kw_only=True)
class Rotation:
    """A literal rotation, authored as a model state part's ``post_rotate``.

    ``R_out = R_in @ R(post_rotate)``: the rigid re-frame a checkpoint was
    trained against (a tool/grip offset), right-multiplied onto the env's
    rotation after it is decoded and before it is re-encoded into the model's
    encoding. Build one with :meth:`from_matrix`; the value must already *be* a
    rotation (orthonormal, ``|det - 1| <= 1e-4``) or the codec rejects it.

    Attributes:
        encoding: Rotation encoding ``value`` is written in.
        value: The rotation's components in that encoding.
    """

    encoding: RotationEncoding
    value: tuple[float, ...]

    @classmethod
    def from_matrix(cls, rows: Iterable[Iterable[float]]) -> Rotation:
        """Build a ``rot6d`` literal from a 3x3 matrix given as three rows.

        ``rot6d`` is the matrix's first two columns concatenated, so the
        round-trip is exact -- no Euler recovery, no gimbal case.
        """
        matrix = [[float(value) for value in row] for row in rows]
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
            raise ValueError(
                f"Rotation.from_matrix needs a 3x3 matrix, got "
                f"{len(matrix)} rows of {[len(row) for row in matrix]}"
            )
        return cls(
            encoding="rot6d",
            value=tuple(matrix[row][col] for col in (0, 1) for row in (0, 1, 2)),
        )


@dataclass(frozen=True, kw_only=True)
class Constant:
    """A constant part of a :class:`Concat`: ``dim`` copies of ``fill``.

    The declarative form of a slot the checkpoint was trained to see but the env
    does not produce -- a pad channel, or a proprio slot the training pipeline
    held at a fixed value. Unlike an absent ``optional`` part this is authored
    data, not fabricated: it never reads as a zero-filled role.

    Attributes:
        dim: Width of the constant block.
        fill: The value every element carries.
    """

    dim: int = 1
    fill: float = 0.0


@dataclass(frozen=True)
class State:
    """A single-part numeric state input expected by a model.

    The 1-part case: one role packed into the value, sourced from an env state
    feature. Use :class:`Concat` to pack several roles into one tensor. A
    ``State`` is also a valid :class:`Concat` part (its part fields -- ``role``,
    ``encoding``, ``dim``, ``index``, ``optional``, ``range``, ``fill``,
    ``post_rotate``, ``scale``, ``offset``, ``frame``, ``part``, ``labels`` -- are
    taken; its container fields must stay default when used as a part).

    There is no ``key`` -- placement in the input tree *is* the payload position.

    Attributes:
        role: Semantic role matched against env state features.
        encoding: Rotation encoding the model expects for this part. A single
            encoding, or a sequence of them in preference order (most-preferred
            first) — the resolver picks the env's native encoding when it
            appears here (no conversion), else converts into the first entry.
            A ``CustomEncoding`` is a single host-side packing (not a set).
        dim: Optional number of leading elements to keep from the source.
        index: Optional single element to select after any conversion.
        optional: Zero-fill this part when the env does not declare the role,
            instead of failing resolution. The fill width comes from ``index``
            (one), ``dim``, or ``encoding``; one of them must be set so the
            width is known without an env feature.
        range: Optional ``(low, high)`` the model expects this part in; when the
            env declares its own (derived or tagged) range, the value is affinely
            mapped from the env range to this one (symmetric to action ranges).
            With no env source range there is nothing to map from, so it is a
            no-op -- it does not clamp or rescale on its own.
        fill: Value this part contributes when ``optional`` is set and the env
            lacks the role (zeros by default). A non-zero ``fill`` requires
            ``optional``, matching ``Actuator.fill``.
        post_rotate: A fixed :class:`Rotation` right-multiplied onto the env's
            rotation (``R_out = R_in @ R(post_rotate)``) before it is re-encoded
            into ``encoding``. Needs a rotation ``encoding``; not combinable
            with a ``CustomEncoding``.
        scale: Model-side multiplier applied after the range map: one float for
            every axis, or a sequence with one value per axis in this part's
            own ``labels`` order (its resolved width otherwise), serialized as
            ``axis_scale``.
        offset: Model-side addend applied after ``scale`` (``value * scale +
            offset``) -- e.g. a ``1 - 2g`` gripper is ``scale=-2, offset=1``, a
            stand pose is ``offset=tuple(-q for q in DEFAULT_POSE)``. A float or
            a per-axis sequence like ``scale``.
        frame: Coordinate frame the checkpoint was trained to read this part in,
            when the role is an absolute pose (``proprio/eef_*``). Keyword-only
            and omitted from the wire when unset. A frame the env contradicts
            fails resolution; a frame the env does not declare draws a caution
            (the model states a requirement nothing can confirm).
        part: The body part this part reads, when the role repeats across a
            body (``"left_arm"``, ``"right_arm"``, ...): an identity key the
            resolver matches on. Naming one binds only that env leaf (a missing
            one fills if ``optional``, else fails); naming none binds the env's
            only leaf of the role under any part (with an ``info``) and fails
            when there are several, naming them. Keyword-only and omitted from
            the wire when unset.
        labels: The axis names this part reads, in the order the checkpoint
            was trained on (``embodiments.GO2.joints``, or a subset). Fixes the
            width. Against an env leaf that labels its axes the values are
            gathered by name, so a differing order is a permutation and a
            subset a selection; an env leaf that declares no labels is a
            resolve error. Keyword-only and omitted from the wire when unset;
            not combinable with ``index``.
        pad_to: Zero-pad the resulting vector to this length. Padding is the
            last step: every part is converted, ranged and scaled, the parts are
            concatenated in order, and only then is the result padded.
        dtype: NumPy dtype name of the resulting value.
        reshape: Optional target shape for the resulting value.
        container: Emit a NumPy array or a plain Python list.
    """

    role: str
    encoding: RotationEncoding | Sequence[RotationEncoding] | CustomEncoding | None = (
        None
    )
    dim: int | None = None
    index: int | None = None
    optional: bool = False
    range: tuple[float, float] | None = None
    fill: float = field(default=0.0, kw_only=True)
    post_rotate: Rotation | None = field(default=None, kw_only=True)
    scale: float | Sequence[float] | None = field(default=None, kw_only=True)
    offset: float | Sequence[float] | None = field(default=None, kw_only=True)
    frame: Frame | None = field(default=None, kw_only=True)
    part: str | None = field(default=None, kw_only=True)
    labels: Sequence[str] | None = field(default=None, kw_only=True)
    pad_to: int | None = None
    dtype: str = "float32"
    reshape: tuple[int, ...] | None = None
    container: Literal["array", "list"] = "array"
    # dim/index/pad_to >= 0 is enforced by the Rust codec (u32) at serialize.

    def __post_init__(self) -> None:
        # index selects one element, dim truncates to the leading N; the resolver
        # applies index and ignores dim when both are set, so reject the ambiguous
        # pairing here (matching the Rust codec guard).
        if self.dim is not None and self.index is not None:
            raise ValueError(
                f"State {self.role!r}: set dim or index, not both "
                "(index selects one element, dim truncates to the leading N)"
            )
        # `fill` is what an absent part contributes; a non-optional part always
        # has an env source, so a non-zero fill there could never fire.
        if self.fill != 0.0 and not self.optional:
            raise ValueError(
                f"State {self.role!r}: fill applies only to an optional part; a "
                "non-optional part takes its values from the env"
            )
        if self.post_rotate is not None:
            if self.encoding is None:
                raise ValueError(
                    f"State {self.role!r}: post_rotate needs a rotation encoding"
                )
            if isinstance(self.encoding, CustomEncoding):
                raise ValueError(
                    f"State {self.role!r}: post_rotate cannot combine with a custom "
                    "encoding; fold the rotation into the repack"
                )
        object.__setattr__(self, "encoding", one_or_many(self.encoding))
        check_accept_set("State", self.role, self.encoding)
        for name in ("scale", "offset"):
            object.__setattr__(
                self,
                name,
                axis_or_scalar("State", self.role, name, getattr(self, name)),
            )
        object.__setattr__(
            self, "labels", check_labels("State", self.role, self.labels)
        )
        if self.labels is not None:
            if self.index is not None:
                raise ValueError(
                    f"State {self.role!r}: set labels or index, not both (labels name "
                    "every axis the part reads; to pick one, name just that label)"
                )
            if self.dim is not None and self.dim != len(self.labels):
                raise ValueError(
                    f"State {self.role!r}: dim {self.dim} disagrees with the "
                    f"{len(self.labels)} labels; labels fix the width, so drop dim"
                )


# A part of a :class:`Concat`: a bare role string, a :class:`State` whose part
# fields are taken, or a :class:`Constant` block.
ConcatPart: TypeAlias = "str | State | Constant"


@dataclass(frozen=True, init=False)
class Concat:
    """A multi-part numeric state input: several roles packed into one tensor.

    The multi-part state leaf. Parts are concatenated in order; each part is a
    bare role string (sugar for a role-only :class:`State`), a :class:`State`
    carrying part fields, or a :class:`Constant` block. A single-role state is
    :class:`State` directly; this is the >1-part case. Both serialize to the same
    ``{"type": "state", ...}`` wire form.

    There is no ``key`` -- placement in the input tree *is* the payload position.

    Attributes:
        parts: Roles (or :class:`State` / :class:`Constant` parts) concatenated
            in order. At least one part must carry a role -- a state of only
            constants reads nothing from the env.
        pad_to: Zero-pad the concatenated vector to this length. Padding is the
            last step: every part is converted, ranged and scaled, the parts are
            concatenated in order, and only then is the result padded.
        dtype: NumPy dtype name of the resulting value.
        reshape: Optional target shape for the resulting value.
        container: Emit a NumPy array or a plain Python list.
    """

    parts: tuple[ConcatPart, ...]
    pad_to: int | None = None
    dtype: str = "float32"
    reshape: tuple[int, ...] | None = None
    container: Literal["array", "list"] = "array"

    def __init__(
        self,
        *parts: ConcatPart,
        pad_to: int | None = None,
        dtype: str = "float32",
        reshape: tuple[int, ...] | None = None,
        container: Literal["array", "list"] = "array",
    ) -> None:
        if not parts:
            raise ValueError("Concat needs at least one part")
        for part in parts:
            if not isinstance(part, (str, State, Constant)):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError(
                    f"Concat parts must be role strings, State leaves or Constant "
                    f"blocks; got {type(part).__name__!r}. Pass a role constant "
                    "(e.g. Concat(EEF_POS, GRIPPER_POS)) or wrap it in State(...)"
                )
        # A state of only constants is a fabricated tensor pretending to be an
        # observation (the Rust codec refuses it too).
        if all(isinstance(part, Constant) for part in parts):
            raise ValueError(
                "a state input of only constant parts reads nothing from the env; "
                "give it at least one roled part"
            )
        # A State used as a part contributes only its part fields; its container
        # fields (pad_to/dtype/reshape/container) belong on the Concat. Catch a
        # non-default one at construction rather than letting it be silently
        # dropped (the wire codec also rejects it, but later and less obviously).
        for part in parts:
            if isinstance(part, State) and (
                part.pad_to is not None
                or part.dtype != "float32"
                or part.reshape is not None
                or part.container != "array"
            ):
                raise ValueError(
                    f"Concat part {part.role!r}: a State used as a part must keep "
                    "its container fields (pad_to, dtype, reshape, container) at "
                    "their defaults; set them on the Concat instead"
                )
        object.__setattr__(self, "parts", tuple(parts))
        object.__setattr__(self, "pad_to", pad_to)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "reshape", reshape)
        object.__setattr__(self, "container", container)


@dataclass(frozen=True)
class Text:
    """A text input expected by a model.

    There is no ``key`` -- placement in the input tree *is* the payload position.

    Attributes:
        role: Semantic role matched against env text features.
        container: Emit a plain string or a single-element list.
        fill: Fallback value used when the observation omits the feature; when
            None the input is omitted from the payload instead. Named to match
            ``Actuator.fill``.
    """

    role: str
    container: Literal["str", "list"] = "str"
    fill: str | None = None


@dataclass(frozen=True)
class Custom:
    """A custom input computed by host-language code.

    Exactly one of ``transform`` (an in-process callable) or ``entrypoint``
    (a ``module:callable`` string) is set:

    - ``transform``: local only -- the callable cannot be serialized, so a model
      spec carrying it cannot be published in contract metadata.
    - ``entrypoint``: serializable and publishable as wire form, but imported
      only when ``resolve(..., trust_entrypoints=True)``; otherwise resolution
      refuses to import it. Travels on the wire under the key ``transform``.

    There is no ``key`` -- placement in the input tree *is* the payload position.

    The callable receives the raw observation as a mapping: the observation
    itself for a Dict env, or the single-leaf envelope ``{"<obs>": leaf}`` for a
    flat (non-Dict) env.

    Attributes:
        transform: An in-process callable taking the raw observation mapping.
        entrypoint: A ``module:callable`` string.
    """

    transform: ObsTransform | None = None
    entrypoint: str | None = None

    def __post_init__(self) -> None:
        if (self.transform is None) == (self.entrypoint is None):
            raise ValueError(
                "Custom: set exactly one of transform= (an in-process callable) "
                "or entrypoint= (a 'module:callable' string)"
            )


# A model input leaf.
ModelLeaf: TypeAlias = Image | State | Concat | Text | Custom
# A recursive model input tree: a leaf, a Dict, or a Tuple. The container type
# *is* the payload container the model's predict receives.
InputNode: TypeAlias = "ModelLeaf | Mapping[str, InputNode] | tuple[InputNode, ...]"

__all__ = [
    "Concat",
    "ConcatPart",
    "Custom",
    "Image",
    "InputNode",
    "ModelLeaf",
    "ObsTransform",
    "State",
    "Text",
]
