"""The environment-side IO spec: observation features plus the action layout."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from ..constants import ENV_METADATA_KEY
from .action import ActionLayout
from .action_serialization import action_layout_from_dict, action_layout_to_dict
from .env_features import EnvFeature
from .env_serialization import env_feature_from_dict, env_feature_to_dict
from .serialization import (
    load_json_mapping,
    require_mapping,
    require_sequence,
)


@dataclass(frozen=True)
class EnvIOSpec:
    """Declarative description of an environment's observation and action.

    Attributes:
        observation: Observation features keyed into the raw obs dict.
        action: Layout of the action vector accepted by ``step``.
    """

    observation: tuple[EnvFeature, ...]
    action: ActionLayout

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict form of this spec."""
        return {
            "observation": [env_feature_to_dict(f) for f in self.observation],
            "action": action_layout_to_dict(self.action),
        }

    def to_json(self) -> str:
        """Return this spec serialized as a JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)

    def to_metadata(self) -> dict[str, Any]:
        """Return a metadata mapping fragment carrying this spec.

        Merge the result into env contract metadata so remote clients can
        recover the spec via :meth:`from_metadata`.
        """
        return {ENV_METADATA_KEY: self.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EnvIOSpec:
        """Build a spec from :meth:`to_dict` output."""
        observation = tuple(
            env_feature_from_dict(item)
            for item in require_sequence(data, "observation")
        )
        return cls(
            observation=observation,
            action=action_layout_from_dict(require_mapping(data, "action")),
        )

    @classmethod
    def from_json(cls, payload: str) -> EnvIOSpec:
        """Build a spec from :meth:`to_json` output."""
        return cls.from_dict(load_json_mapping(payload))

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> EnvIOSpec | None:
        """Extract a spec from env contract metadata, if one is present."""
        payload = metadata.get(ENV_METADATA_KEY)
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise TypeError(f"metadata key {ENV_METADATA_KEY!r} must hold a mapping")
        return cls.from_dict(cast(Mapping[str, Any], payload))


__all__ = ["EnvIOSpec"]
