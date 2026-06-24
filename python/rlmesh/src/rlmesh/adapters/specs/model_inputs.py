"""Input feature dataclasses declared by models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import InitVar, dataclass
from typing import Any, Literal, TypeAlias

from ..constants import IMAGE_PRIMARY, INSTRUCTION
from .custom_encoding import CustomEncoding
from .vocabularies import ImageLayout, RotationEncoding

ObsTransform: TypeAlias = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class ImageInput:
    """An image input expected by a model.

    Attributes:
        key: Key of the entry in the model input payload.
        role: Semantic role matched against env image features.
        height: Target image height, or None to keep the env height.
        width: Target image width, or None to keep the env width.
        layout: Axis layout the model expects.
        dtype: NumPy dtype name the model expects.
        normalize: Scale 8-bit pixel values into ``[0, 1]`` before casting.
        lead_dims: Number of leading singleton axes to add (batch/time).
        upside_down: Whether the model was trained on images rotated 180
            degrees relative to the canonical upright orientation.
        resample: Resize algorithm the model's training pipeline used:
            ``"bilinear_aa"`` (antialiased triangle filter, PIL-compatible)
            or ``"bilinear"`` (4-tap half-pixel-center bilinear,
            OpenCV/torch-compatible).
        stack: Number of consecutive observations to stack on a new leading
            axis (frame history). ``1`` (default) means no stacking. Stacking
            is applied host-side by the adapter, which buffers processed
            frames (padding with the first frame at the start of an episode)
            and clears them on ``reset`` -- the env still sends one frame per
            step.
        size: Convenience for square targets -- sets both ``height`` and
            ``width``. Pass ``size`` or ``height``/``width``, not both.
    """

    key: str
    role: str = IMAGE_PRIMARY
    height: int | None = None
    width: int | None = None
    layout: ImageLayout = "hwc"
    dtype: str = "uint8"
    normalize: bool = False
    lead_dims: int = 0
    upside_down: bool = False
    resample: str = "bilinear_aa"
    stack: int = 1
    size: InitVar[int | None] = None

    def __post_init__(self, size: int | None) -> None:
        # size= is construction sugar (sets both height and width). The
        # non-negative bounds (height/width/lead_dims) and the stack 1..=64 bound
        # are enforced by the Rust codec at serialize/normalize (u32 + de_stack).
        if size is not None:
            if self.height is not None or self.width is not None:
                raise ValueError("ImageInput: pass size=, or height=/width=, not both")
            object.__setattr__(self, "height", size)
            object.__setattr__(self, "width", size)


@dataclass(frozen=True)
class StateComponent:
    """One piece of a model state vector, sourced from an env state feature.

    Attributes:
        role: Semantic role matched against env state features.
        encoding: Rotation encoding the model expects for this piece.
        dim: Optional number of leading elements to keep from the source.
        index: Optional single element to select after any conversion.
        optional: Zero-fill this piece when the env does not declare the
            role, instead of failing resolution. The fill width comes from
            ``index`` (one), ``dim``, or ``encoding``; one of them must be
            set so the width is known without an env feature.
        range: Optional ``(low, high)`` the model expects this piece in;
            when the env declares its own range, the value is affinely
            mapped from the env range to this one (symmetric to action
            ranges).
    """

    role: str
    encoding: RotationEncoding | CustomEncoding | None = None
    dim: int | None = None
    index: int | None = None
    optional: bool = False
    range: tuple[float, float] | None = None
    # dim/index >= 0 is enforced by the Rust codec (u32) at serialize/normalize.


@dataclass(frozen=True)
class StateInput:
    """A numeric state input expected by a model.

    Attributes:
        key: Key of the entry in the model input payload.
        components: Pieces concatenated (in order) to form the value.
        pad_to: Zero-pad the concatenated vector to this length.
        dtype: NumPy dtype name of the resulting value.
        reshape: Optional target shape for the resulting value.
        container: Emit a NumPy array or a plain Python list.

    For a single-piece state, pass ``role`` (and optionally ``encoding`` /
    ``dim`` / ``index``) instead of ``components`` -- e.g.
    ``StateInput("state", role=EEF_POS)`` is shorthand for
    ``StateInput("state", components=(StateComponent(EEF_POS),))``.
    """

    key: str
    components: tuple[StateComponent, ...] = ()
    pad_to: int | None = None
    dtype: str = "float32"
    reshape: tuple[int, ...] | None = None
    container: Literal["array", "list"] = "array"
    role: InitVar[str | None] = None
    encoding: InitVar[RotationEncoding | CustomEncoding | None] = None
    dim: InitVar[int | None] = None
    index: InitVar[int | None] = None

    def __post_init__(
        self,
        role: str | None,
        encoding: RotationEncoding | CustomEncoding | None,
        dim: int | None,
        index: int | None,
    ) -> None:
        single = (
            role is not None
            or encoding is not None
            or dim is not None
            or index is not None
        )
        if self.components and single:
            raise ValueError(
                "StateInput: pass components=, or a single role=/encoding=/"
                "dim=/index=, not both"
            )
        if not self.components:
            if role is None:
                raise ValueError("StateInput needs components=(...) or a single role=")
            object.__setattr__(
                self,
                "components",
                (StateComponent(role, encoding, dim, index),),
            )
        # pad_to >= 0 is enforced by the Rust codec (u32) at serialize/normalize.


@dataclass(frozen=True)
class TextInput:
    """A text input expected by a model.

    Attributes:
        key: Key of the entry in the model input payload.
        role: Semantic role matched against env text features.
        container: Emit a plain string or a single-element list.
        default: Value used when the observation omits the feature; when
            None the key is omitted from the payload instead.
    """

    key: str
    role: str = INSTRUCTION
    container: Literal["str", "list"] = "str"
    default: str | None = None


@dataclass(frozen=True)
class InlineCustomInput:
    """A custom input computed in-process by a user callable.

    Local only: the callable cannot be serialized, so a model spec carrying
    an :class:`InlineCustomInput` cannot be published in contract metadata.
    Use :class:`EntrypointCustomInput` for a spec that must travel.

    Attributes:
        key: Key of the entry in the model input payload.
        transform: Callable taking the raw observation mapping.
    """

    key: str
    transform: ObsTransform


@dataclass(frozen=True)
class EntrypointCustomInput:
    """A custom input computed by a ``module:callable`` entrypoint.

    Serializable and publishable. The entrypoint is imported only when
    ``resolve(..., trust_entrypoints=True)``; otherwise resolution refuses
    to import it.

    Attributes:
        key: Key of the entry in the model input payload.
        entrypoint: A ``module:callable`` string.
    """

    key: str
    entrypoint: str


ModelInput: TypeAlias = (
    ImageInput | StateInput | TextInput | InlineCustomInput | EntrypointCustomInput
)

__all__ = [
    "EntrypointCustomInput",
    "ImageInput",
    "InlineCustomInput",
    "ModelInput",
    "ObsTransform",
    "StateComponent",
    "StateInput",
    "TextInput",
]
