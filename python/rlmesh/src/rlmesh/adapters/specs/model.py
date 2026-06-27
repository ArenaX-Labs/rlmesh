"""The model-side spec: expected input payload tree plus the action output."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from ..constants import MODEL_METADATA_KEY
from ._codec import normalize_spec
from .action import Action
from .action_serialization import action_from_dict, action_to_dict
from .model_inputs import InputNode
from .model_serialization import model_input_from_dict, model_input_to_dict


@dataclass(frozen=True)
class ModelSpec:
    """Declarative description of a model's input payload tree and action output.

    ``input`` is a recursive tree whose container type *is* the payload container
    the model's ``predict`` receives: a bare leaf (a single tensor/string), a
    ``dict[str, subtree]``, or a ``tuple`` of subtrees. A leaf is an
    :class:`~rlmesh.adapters.Image`, :class:`~rlmesh.adapters.State`,
    :class:`~rlmesh.adapters.Concat`, :class:`~rlmesh.adapters.Text`, or
    :class:`~rlmesh.adapters.Custom`. Placement (tree position) is the payload
    position -- model leaves carry no ``key``, and a role may be reused across
    leaves (one env camera can feed several input slots).

    Attributes:
        input: The model input tree.
        output: Layout of the action vector produced by the model.
    """

    input: InputNode
    output: Action

    def __hash__(self) -> int:
        # `input` can be a Dict node (an unhashable Python ``dict``), so the
        # dataclass-default field hash would fail even though the spec is frozen
        # and compares by value. Hash the dataclass ``repr`` instead: it is a
        # stable, field-by-field rendering, so equal specs (equal fields) render
        # identically and hash equal -- consistent with the generated ``__eq__``.
        # ``repr`` (unlike ``to_dict``) never raises, so a Custom-input spec stays
        # hashable, and it applies no canonicalization, so distinct authored forms
        # cannot collide. (frozen=True keeps the generated __eq__.)
        return hash(repr(self))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict form of this spec.

        Raises:
            ValueError: If a custom input cannot be serialized (an in-process
                callable, or an entrypoint custom at the publish boundary).
        """
        raw = {
            "input": model_input_to_dict(self.input),
            "output": action_to_dict(self.output),
        }
        return normalize_spec("model", raw, allow_custom=True)

    def to_json(self) -> str:
        """Return this spec serialized as a JSON string."""
        # allow_nan=False: refuse to emit the non-RFC-8259 `Infinity`/`NaN`
        # tokens the Rust serde codec rejects (a directly-constructed dataclass
        # bypasses the from_dict finiteness guards).
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    def to_metadata(self) -> dict[str, Any]:
        """Return a metadata mapping fragment carrying this spec.

        Merge the result into model contract metadata so remote consumers
        can recover the spec via :meth:`from_metadata`. A published spec must
        be fully declarative: custom inputs (whether in-process callables or
        ``module:callable`` entrypoint strings) cannot be published, because a
        consumer would have to import code from the contract. Resolve such a
        spec locally instead (the model spec need not travel).

        Raises:
            ValueError: If any input is a custom transform (in-process callable
                or entrypoint); neither can be safely published in v1.
        """
        return {MODEL_METADATA_KEY: self.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelSpec:
        """Build a spec from :meth:`to_dict` output.

        The input is validated and canonicalized by the Rust codec first, so the
        Python shape readers below operate on already-valid data.
        """
        canonical = normalize_spec("model", data, allow_custom=True)
        return cls(
            input=model_input_from_dict(canonical["input"]),
            output=action_from_dict(canonical["output"]),
        )

    @classmethod
    def from_json(cls, payload: str) -> ModelSpec:
        """Build a spec from :meth:`to_json` output."""
        return cls.from_dict(json.loads(payload))

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> ModelSpec | None:
        """Extract a spec from model contract metadata, or None when absent.

        Reads the single v1 metadata key (``rlmesh.adapters.v1.model_spec``).
        When a future v2 format lands it ships a new key and reader, restoring a
        newest-format-first dual read so a newer build still reads an older
        peer's v1 spec; that dispatch moves into the Rust codec (the single
        source of truth) once the PyO3 normalize door lands.
        """
        payload = metadata.get(MODEL_METADATA_KEY)
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise TypeError(f"metadata key {MODEL_METADATA_KEY!r} must hold a mapping")
        return cls.from_dict(cast(Mapping[str, Any], payload))


__all__ = ["ModelSpec"]
