"""The model-side IO spec: expected input payload plus the action output."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from ..constants import MODEL_METADATA_KEY
from .action import ActionLayout
from .action_serialization import action_layout_from_dict, action_layout_to_dict
from .model_inputs import ModelInput
from .model_serialization import model_input_from_dict, model_input_to_dict
from .serialization import (
    load_json_mapping,
    require_mapping,
    require_sequence,
)


@dataclass(frozen=True)
class ModelSpec:
    """Declarative description of a model's input payload and action output.

    Attributes:
        inputs: Input features keyed into the model payload dict.
        action: Layout of the action vector produced by the model.
    """

    inputs: tuple[ModelInput, ...]
    action: ActionLayout

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict form of this spec.

        Raises:
            ValueError: If a custom input holds an in-process callable,
                which cannot be serialized.
        """
        return {
            "inputs": [model_input_to_dict(item) for item in self.inputs],
            "action": action_layout_to_dict(self.action),
        }

    def to_json(self) -> str:
        """Return this spec serialized as a JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)

    def to_metadata(self) -> dict[str, Any]:
        """Return a metadata mapping fragment carrying this spec.

        Merge the result into model contract metadata so remote consumers
        can recover the spec via :meth:`from_metadata`. A remotely published
        spec must be fully declarative: custom transforms must use
        ``module:callable`` entrypoint strings, not in-process callables.

        Raises:
            ValueError: If a custom input holds an in-process callable,
                which cannot be serialized.
        """
        return {MODEL_METADATA_KEY: self.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelSpec:
        """Build a spec from :meth:`to_dict` output."""
        inputs = tuple(
            model_input_from_dict(item) for item in require_sequence(data, "inputs")
        )
        return cls(
            inputs=inputs,
            action=action_layout_from_dict(require_mapping(data, "action")),
        )

    @classmethod
    def from_json(cls, payload: str) -> ModelSpec:
        """Build a spec from :meth:`to_json` output."""
        return cls.from_dict(load_json_mapping(payload))

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> ModelSpec | None:
        """Extract a spec from model contract metadata, if one is present."""
        payload = metadata.get(MODEL_METADATA_KEY)
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise TypeError(f"metadata key {MODEL_METADATA_KEY!r} must hold a mapping")
        return cls.from_dict(cast(Mapping[str, Any], payload))


__all__ = ["ModelSpec"]
