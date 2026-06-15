"""Dict round-trip for model input features."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from .custom_encoding import CustomEncoding
from .model_inputs import (
    EntrypointCustomInput,
    ImageInput,
    ModelInput,
    StateComponent,
    StateInput,
    TextInput,
)
from .validation import (
    as_mapping,
    opt_encoding,
    opt_layout,
    opt_range,
    require_sequence,
    require_str,
)


def model_input_to_dict(item: ModelInput) -> dict[str, Any]:
    """Return the JSON-compatible dict form of a model input feature."""
    if isinstance(item, ImageInput):
        image: dict[str, Any] = {
            "type": "image",
            "key": item.key,
            "role": item.role,
            "height": item.height,
            "width": item.width,
            "layout": item.layout,
            "dtype": item.dtype,
            "normalize": item.normalize,
            "lead_dims": item.lead_dims,
            "upside_down": item.upside_down,
            "resample": item.resample,
        }
        # Host-side field, emitted only when set so it is additive over the
        # pinned wire format (the native core ignores it).
        if item.stack != 1:
            image["stack"] = item.stack
        return image
    if isinstance(item, StateInput):
        for component in item.components:
            if isinstance(component.encoding, CustomEncoding):
                raise ValueError(
                    f"state input {item.key!r} uses a CustomEncoding, whose "
                    "host-side transforms cannot be serialized; resolve it "
                    "locally (the model spec need not travel), or add the "
                    "encoding to the native vocabulary for a shared convention"
                )
        return {
            "type": "state",
            "key": item.key,
            "components": [
                {
                    "role": component.role,
                    "encoding": component.encoding,
                    "dim": component.dim,
                    "index": component.index,
                    "optional": component.optional,
                    "range": list(component.range) if component.range else None,
                }
                for component in item.components
            ],
            "pad_to": item.pad_to,
            "dtype": item.dtype,
            # `()` is a valid 0-D scalar target; only None means "no reshape".
            "reshape": list(item.reshape) if item.reshape is not None else None,
            "container": item.container,
        }
    if isinstance(item, TextInput):
        return {
            "type": "text",
            "key": item.key,
            "role": item.role,
            "container": item.container,
            "default": item.default,
        }
    if isinstance(item, EntrypointCustomInput):
        return {"type": "custom", "key": item.key, "transform": item.entrypoint}
    raise ValueError(
        f"custom input {item.key!r} holds an in-process callable and cannot be "
        "serialized; use an EntrypointCustomInput ('module:callable') instead"
    )


def model_input_from_dict(item: object) -> ModelInput:
    """Build a model input feature from :func:`model_input_to_dict` output."""
    data = as_mapping(item, "model input")
    kind = data.get("type")
    if kind == "image":
        height = data.get("height")
        width = data.get("width")
        return ImageInput(
            key=require_str(data, "key", "image input"),
            role=require_str(data, "role", "image input"),
            height=None if height is None else int(height),
            width=None if width is None else int(width),
            layout=opt_layout(data.get("layout"), "image input"),
            dtype=str(data.get("dtype", "uint8")),
            normalize=bool(data.get("normalize", False)),
            lead_dims=int(data.get("lead_dims", 0)),
            upside_down=bool(data.get("upside_down", False)),
            resample=str(data.get("resample", "bilinear_aa")),
            stack=int(data.get("stack", 1)),
        )
    if kind == "state":
        components: list[StateComponent] = []
        for entry in require_sequence(data, "components"):
            component = as_mapping(entry, "state component")
            dim = component.get("dim")
            index = component.get("index")
            components.append(
                StateComponent(
                    role=require_str(component, "role", "state component"),
                    encoding=opt_encoding(component.get("encoding"), "state component"),
                    dim=None if dim is None else int(dim),
                    index=None if index is None else int(index),
                    optional=bool(component.get("optional", False)),
                    range=opt_range(component.get("range"), "state component"),
                )
            )
        pad_to = data.get("pad_to")
        reshape = data.get("reshape")
        container = data.get("container", "array")
        if container not in ("array", "list"):
            raise ValueError(
                f"state input container must be 'array' or 'list', got {container!r}"
            )
        return StateInput(
            key=require_str(data, "key", "state input"),
            components=tuple(components),
            pad_to=None if pad_to is None else int(pad_to),
            dtype=str(data.get("dtype", "float32")),
            reshape=None
            if reshape is None
            else tuple(int(n) for n in cast(Sequence[Any], reshape)),
            container=container,
        )
    if kind == "text":
        container = data.get("container", "str")
        if container not in ("str", "list"):
            raise ValueError(
                f"text input container must be 'str' or 'list', got {container!r}"
            )
        default = data.get("default")
        return TextInput(
            key=require_str(data, "key", "text input"),
            role=require_str(data, "role", "text input"),
            container=container,
            default=None if default is None else str(default),
        )
    if kind == "custom":
        # Only entrypoint-form customs survive serialization; an in-process
        # callable cannot be on the wire.
        return EntrypointCustomInput(
            key=require_str(data, "key", "custom input"),
            entrypoint=require_str(data, "transform", "custom input"),
        )
    raise ValueError(f"unknown model input type {kind!r}")


__all__ = ["model_input_from_dict", "model_input_to_dict"]
