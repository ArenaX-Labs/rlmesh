"""``rlmesh.describe`` CLI: schema reflection without running __init__/__new__."""

from __future__ import annotations

from typing import cast

from rlmesh.describe import _describe, _variations


def test_variations_treats_bare_str_axis_as_one_value() -> None:
    # list("pick-place") would explode into characters; a bare str is one value.
    assert _variations({"task": "pick-place"}) == {"task": ["pick-place"]}
    assert _variations({"task": ["a", "b"]}) == {"task": ["a", "b"]}


def test_describe_class_with_unsafe_bare_new() -> None:
    # Subclassing a C type makes object.__new__(cls) raise; describe must still
    # reflect the load signature (via the partial fallback) instead of crashing.
    class _CModel(int):
        params = None

        def load(self, *, checkpoint: str = "ck") -> None: ...

        def predict(self, observation: object) -> int:
            return 0

    payload = _describe(_CModel, "load")
    tier = cast("list[dict[str, object]]", payload["signature_tier"])
    assert "checkpoint" in [p["name"] for p in tier]
