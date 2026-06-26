"""Input leaf dataclasses declared by models.

The model leaves are bare (no ``key``): placement in the input tree *is* the
payload position. Only image/state/text/custom exist; a :class:`Concat` is the
multi-part state leaf (a single tensor concatenated from several role parts).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import InitVar, dataclass
from typing import Any, Literal, TypeAlias

from ..constants import IMAGE_PRIMARY, INSTRUCTION
from ._codec import one_or_many
from .custom_encoding import CustomEncoding
from .vocabularies import FitMode, ImageLayout, RotationEncoding

ObsTransform: TypeAlias = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class Image:
    """An image input expected by a model.

    There is no ``key`` -- placement in the input tree *is* the payload position.

    Attributes:
        role: Semantic role matched against env image features.
        height: Target image height, or None to keep the env height.
        width: Target image width, or None to keep the env width.
        layout: Axis layout the model expects.
        channels: Channel count the model expects (e.g. 3 for RGB, 1 for
            grayscale). When set, a resolve error if the env image differs;
            the adapter does not convert between channel counts.
        dtype: NumPy dtype name the model expects.
        normalize: Map 8-bit pixel values into ``normalize_range`` (default
            ``[0, 1]``) before casting. Setting ``normalize_range`` alone also
            turns normalization on, so this flag is only needed to request the
            default range without spelling it out (not an off-switch).
        normalize_range: Target range, mapped from ``[0, 255]``; implies
            normalization even when ``normalize`` is False. Defaults to
            ``[0, 1]``; set e.g. ``(-1.0, 1.0)`` for a model trained on signed
            inputs.
        lead_dims: Number of leading singleton axes to add (batch/time).
        upside_down: Whether the model was trained on images rotated 180
            degrees relative to the canonical upright orientation.
        resample: Resize algorithm the model's training pipeline used:
            ``"bilinear_aa"`` (antialiased triangle filter, PIL-compatible)
            or ``"bilinear"`` (4-tap half-pixel-center bilinear,
            OpenCV/torch-compatible).
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
        stack: Number of consecutive observations to stack on a new leading
            axis (frame history). ``1`` (default) means no stacking. Stacking
            is applied by the adapter from an episode-keyed rolling buffer
            (padding with the first frame at the start of an episode, cleared on
            ``reset``) -- host-side on the local path, natively in the core on
            the served path. Either way the env still sends one frame per step.
        size: Convenience for square targets -- sets both ``height`` and
            ``width``. Pass ``size`` or ``height``/``width``, not both.
    """

    role: str = IMAGE_PRIMARY
    height: int | None = None
    width: int | None = None
    layout: ImageLayout = "hwc"
    channels: int | None = None
    dtype: str = "uint8"
    normalize: bool = False
    normalize_range: tuple[float, float] | None = None
    lead_dims: int = 0
    upside_down: bool = False
    resample: str = "bilinear_aa"
    allow_upscale: bool = False
    fit: FitMode | Sequence[FitMode] | None = None
    optional: bool = False
    absent_fill: int | None = None
    stack: int = 1
    size: InitVar[int | None] = None

    def __post_init__(self, size: int | None) -> None:
        # size= is construction sugar (sets both height and width). The
        # non-negative bounds (height/width/lead_dims) and the stack 1..=64 bound
        # are enforced by the Rust codec at serialize/normalize (u32 + de_stack).
        if size is not None:
            if self.height is not None or self.width is not None:
                raise ValueError("Image: pass size=, or height=/width=, not both")
            object.__setattr__(self, "height", size)
            object.__setattr__(self, "width", size)
        # A single fit stays a string; a preference list normalizes to a tuple
        # (hashable, round-trips by value) -- mirrors the rotation accept-set.
        object.__setattr__(self, "fit", one_or_many(self.fit))


@dataclass(frozen=True)
class State:
    """A single-part numeric state input expected by a model.

    The 1-part case: one role packed into the value, sourced from an env state
    feature. Use :class:`Concat` to pack several roles into one tensor. A
    ``State`` is also a valid :class:`Concat` part (its part fields -- ``role``,
    ``encoding``, ``dim``, ``index``, ``optional``, ``range`` -- are taken; its
    container fields must stay default when used as a part).

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
        pad_to: Zero-pad the resulting vector to this length.
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
        object.__setattr__(self, "encoding", one_or_many(self.encoding))


# A part of a :class:`Concat`: a bare role string, or a :class:`State` whose part
# fields are taken.
ConcatPart: TypeAlias = "str | State"


@dataclass(frozen=True, init=False)
class Concat:
    """A multi-part numeric state input: several roles packed into one tensor.

    The multi-part state leaf. Parts are concatenated in order; each part is a
    bare role string (sugar for a role-only :class:`State`) or a :class:`State`
    carrying part fields. A single-role state is :class:`State` directly; this is
    the >1-part case. Both serialize to the same ``{"type": "state", ...}`` wire
    form.

    There is no ``key`` -- placement in the input tree *is* the payload position.

    Attributes:
        parts: Roles (or :class:`State` parts) concatenated in order.
        pad_to: Zero-pad the concatenated vector to this length.
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
        default: Value used when the observation omits the feature; when
            None the input is omitted from the payload instead.
    """

    role: str = INSTRUCTION
    container: Literal["str", "list"] = "str"
    default: str | None = None


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

# Backwards-readable alias retained.
ModelInput: TypeAlias = ModelLeaf

__all__ = [
    "Concat",
    "ConcatPart",
    "Custom",
    "Image",
    "InputNode",
    "ModelInput",
    "ModelLeaf",
    "ObsTransform",
    "State",
    "Text",
]
