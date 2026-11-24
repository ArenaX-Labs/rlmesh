"""The model-side spec: expected input payload tree plus the action output."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from ..constants import MODEL_METADATA_KEY
from ._codec import hashable_node, normalize_spec
from .action import Action
from .action_serialization import action_from_dict, action_to_dict
from .model_inputs import InputNode
from .model_serialization import model_input_from_dict, model_input_to_dict


@dataclass(frozen=True)
class Timing:
    """Action pacing used unless load() or a run overrides it.

    Attributes:
        execution: Actions executed before re-planning (1..=1024), or None.
    """

    execution: int | None = None


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
        timing: Action pacing used when the run and load() name no horizon.
        allow: Named projections (lossy fits) the model accepts. Without one,
            the pairing it fits is refused; with it, the fit is applied and
            recorded in ``describe()`` and the run's advisories:

            - ``"drop_action"``: discard a model output the env has no
              actuator for.
            - ``"select_state"``: read the leading ``dim`` elements of a
              wider env state field (one finger of a two-finger width).
            - ``"rebind_camera"``: feed a camera input from the env's only
              camera under another role (a wrist input from a third-person
              view).
            - ``"flip_gripper"``: reverse a gripper command or reading whose
              ``polarity`` the two sides declare differently.
            - ``"convert_units"``: rescale a delta command whose ``unit`` the
              two sides declare differently.
    """

    input: InputNode
    output: Action
    allow: tuple[str, ...] = ()
    timing: Timing = Timing()

    def __hash__(self) -> int:
        """Hash consistently with the generated order-insensitive ``__eq__``.

        Hashes a key-order-canonical rendering (see :func:`hashable_node`),
        since a Dict node is an unhashable ``dict`` and ``__eq__`` ignores key
        order. A Custom-input spec stays hashable.
        """
        return hash((hashable_node(self.input), self.output, self.allow, self.timing))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict form of this spec.

        A ``CustomEncoding`` field serializes to a ``{base, ...}`` schema; an
        entrypoint arm travels as its ``module:callable`` string and an
        in-process callable as a ``<local>`` marker.

        Raises:
            ValueError: If a custom *input* cannot be serialized (an in-process
                callable, or an entrypoint custom at the publish boundary).
        """
        raw: dict[str, Any] = {
            "input": model_input_to_dict(self.input),
            "output": action_to_dict(self.output),
        }
        if self.allow:
            raw["allow"] = list(self.allow)
        if self.timing.execution is not None:
            raw["timing"] = {"execution": self.timing.execution}
        return normalize_spec("model", raw, allow_custom=False)

    def to_json(self) -> str:
        """Return this spec serialized as a JSON string."""
        # allow_nan=False: the Rust serde codec rejects Infinity/NaN tokens.
        from ..._rlmesh import adapters_spec_normalize

        return adapters_spec_normalize(
            "model", json.dumps(self.to_dict(), allow_nan=False), False
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return a metadata mapping fragment carrying this spec.

        Merge the result into model contract metadata so remote consumers
        can recover the spec via :meth:`from_metadata`. Custom *inputs*
        (in-process callables or ``module:callable`` entrypoints) cannot be
        published, because a consumer would have to import code from the
        contract; resolve such a spec locally. A custom *encoding* publishes as
        a ``{base, ...}`` schema and never runs remotely.

        Raises:
            ValueError: If any input is a custom transform (in-process callable
                or entrypoint); neither can be safely published in v1.
        """
        return {MODEL_METADATA_KEY: self.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelSpec:
        """Build a spec from :meth:`to_dict` output.

        The input is validated and canonicalized by the Rust codec first.

        A :class:`~rlmesh.adapters.Concat` with a single *parameterized* part
        serializes identically to the equivalent :class:`~rlmesh.adapters.State`
        and reads back as that ``State``, so ``from_dict(spec.to_dict())`` can
        differ from ``spec`` by that leaf class alone.
        """
        canonical = normalize_spec("model", data, allow_custom=True)
        return cls(
            input=model_input_from_dict(canonical["input"]),
            output=action_from_dict(canonical["output"]),
            allow=tuple(canonical.get("allow", ())),
            timing=Timing(execution=canonical.get("timing", {}).get("execution")),
        )

    @classmethod
    def from_json(cls, payload: str) -> ModelSpec:
        """Build a spec from :meth:`to_json` output."""
        return cls.from_dict(json.loads(payload))

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> ModelSpec | None:
        """Extract a spec from model contract metadata, or None when absent.

        Reads the v1 metadata key (``rlmesh.adapters.v1.model_spec``).
        """
        payload = metadata.get(MODEL_METADATA_KEY)
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            from ..resolver import AdapterResolutionError

            raise AdapterResolutionError(
                f"metadata key {MODEL_METADATA_KEY!r} must hold a mapping"
            )
        return cls.from_dict(cast(Mapping[str, Any], payload))


__all__ = ["ModelSpec"]
