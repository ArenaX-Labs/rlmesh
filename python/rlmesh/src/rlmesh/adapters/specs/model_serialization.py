"""Dict round-trip for model input features.

The dataclass<->dict *shape* lives here; validation and canonicalization are done
by the authoritative Rust codec (see :mod:`._codec`), so the from-dict reader
operates on already-valid canonical data.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from ._codec import one_or_many, to_pair
from .custom_encoding import CustomEncoding
from .model_inputs import (
    EntrypointCustomInput,
    ImageInput,
    ModelInput,
    StateComponent,
    StateInput,
    TextInput,
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
        # Additive over the pinned v1 wire (the env still sends one frame/step),
        # emitted only when set. It IS consumed by the native adapter engine
        # (episode-keyed FrameBuffers), not host-only -- see ImageInput.stack.
        if item.stack != 1:
            image["stack"] = item.stack
        # Opt-in, emitted only when set (byte-parity with the pinned wire format).
        if item.allow_upscale:
            image["allow_upscale"] = True
        if item.fit is not None:
            image["fit"] = item.fit
        if item.channels is not None:
            image["channels"] = item.channels
        if item.normalize_range is not None:
            image["normalize_range"] = list(item.normalize_range)
        if item.optional:
            image["optional"] = True
        if item.absent_fill is not None:
            image["absent_fill"] = item.absent_fill
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
        # Publish boundary: an entrypoint custom carries an importable
        # `module:callable`, so publishing it would ship executable code in a
        # contract a remote consumer might import. Until an allowlist-based
        # trust model exists, refuse to serialize it here. Local resolve still
        # runs it (gated by trust_entrypoints) via the resolver's _model_wire,
        # which handles customs itself and never routes them through this
        # serializer -- so this gate is publish-only.
        raise ValueError(
            f"custom input {item.key!r} carries an entrypoint "
            f"({item.entrypoint!r}) and cannot be published in v1 contract "
            "metadata; resolve the spec locally (the model spec need not "
            "travel), or wait for the allowlist-gated trust system"
        )
    raise ValueError(
        f"custom input {item.key!r} holds an in-process callable and cannot be "
        "serialized; use an EntrypointCustomInput ('module:callable') instead"
    )


def model_input_from_dict(item: object) -> ModelInput:
    """Build a model input feature from canonical (Rust-validated) dict form."""
    data = cast(Mapping[str, Any], item)
    kind = data["type"]
    if kind == "image":
        return ImageInput(
            key=data["key"],
            role=data["role"],
            height=data.get("height"),
            width=data.get("width"),
            layout=data.get("layout", "hwc"),
            channels=data.get("channels"),
            dtype=data.get("dtype", "uint8"),
            normalize=bool(data.get("normalize", False)),
            normalize_range=to_pair(data.get("normalize_range")),
            optional=bool(data.get("optional", False)),
            absent_fill=data.get("absent_fill"),
            lead_dims=int(data.get("lead_dims", 0)),
            upside_down=bool(data.get("upside_down", False)),
            resample=data.get("resample", "bilinear_aa"),
            allow_upscale=bool(data.get("allow_upscale", False)),
            fit=one_or_many(data.get("fit")),
            stack=int(data.get("stack", 1)),
        )
    if kind == "state":
        components = tuple(
            StateComponent(
                role=component["role"],
                encoding=one_or_many(component.get("encoding")),
                dim=component.get("dim"),
                index=component.get("index"),
                optional=bool(component.get("optional", False)),
                range=to_pair(component.get("range")),
            )
            for component in data["components"]
        )
        reshape = data.get("reshape")
        return StateInput(
            key=data["key"],
            components=components,
            pad_to=data.get("pad_to"),
            dtype=data.get("dtype", "float32"),
            reshape=None if reshape is None else tuple(int(n) for n in reshape),
            container=data.get("container", "array"),
        )
    if kind == "text":
        return TextInput(
            key=data["key"],
            role=data["role"],
            container=data.get("container", "str"),
            default=data.get("default"),
        )
    if kind == "custom":
        # Only entrypoint-form customs survive serialization; an in-process
        # callable cannot be on the wire. The resolver's internal
        # `transform: "host:<key>"` placeholder (see resolver._model_wire)
        # references a host-side callable that stayed local during resolve; it
        # is NOT an importable entrypoint and is never meant to round-trip back
        # through from_dict. Reject it rather than minting a custom with a bogus
        # `host:` entrypoint that a later trust_entrypoints resolve would try to
        # `import host`.
        transform = data["transform"]
        if isinstance(transform, str) and transform.startswith("host:"):
            raise ValueError(
                f"custom input {data['key']!r} carries a host-placeholder "
                f"transform ({transform!r}); it references a host-side callable "
                "that stayed local during resolve and cannot be reconstructed "
                "from the wire form -- resolve the spec locally instead of "
                "routing its internal wire form back through from_dict"
            )
        return EntrypointCustomInput(key=data["key"], entrypoint=transform)
    raise ValueError(f"unknown model input type {kind!r}")


__all__ = ["model_input_from_dict", "model_input_to_dict"]
