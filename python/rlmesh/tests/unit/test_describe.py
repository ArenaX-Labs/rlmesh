"""``rlmesh.describe`` CLI: schema reflection without running __init__/__new__."""

from __future__ import annotations

from typing import cast

import pytest
import rlmesh
from rlmesh.describe import _catalog, _describe, _variations


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


# --- enumerate_variants() catalog --------------------------------------------


def _noop(**_kwargs: object) -> None: ...


def test_catalog_variant_nested_shape() -> None:
    # Display info rides nested in metadata, never flattened onto the entry.
    cat = _catalog([rlmesh.Variant("s/0", {"a": 1}, name="Zero")], None, _noop)
    assert cat == [{"id": "s/0", "params": {"a": 1}, "metadata": {"name": "Zero"}}]


def test_catalog_tolerant_plain_dict_entry() -> None:
    # A plain mapping (no Variant) yields the same nested shape; non id/params keys
    # collapse into metadata.
    cat = _catalog([{"id": "s/0", "params": {"a": 1}, "name": "Zero"}], None, _noop)
    assert cat == [{"id": "s/0", "params": {"a": 1}, "metadata": {"name": "Zero"}}]


def test_catalog_rejects_duplicate_id() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _catalog([rlmesh.Variant("x", {}), rlmesh.Variant("x", {})], None, _noop)


def test_catalog_rejects_empty_or_non_str_id() -> None:
    with pytest.raises(ValueError):
        _catalog([rlmesh.Variant("", {})], None, _noop)
    with pytest.raises(ValueError):
        _catalog([{"id": 3, "params": {}}], None, _noop)


def test_catalog_rejects_non_variant_entry() -> None:
    with pytest.raises(TypeError):
        _catalog([42], None, _noop)


def test_catalog_badges_unbuildable_variant() -> None:
    spec = rlmesh.ParamSpec(
        rlmesh.Param("task", type=str, default="a", choices=("a", "b")),
    )

    def make(*, task: str = "a", n: int = 0) -> None: ...

    good = _catalog([rlmesh.Variant("ok", {"task": "a"})], spec, make)
    assert "error" not in good[0]

    bad = _catalog([rlmesh.Variant("bad", {"task": "zzz"})], spec, make)
    assert "error" in bad[0]
    # params kept verbatim despite the badge -- catalog never rewrites an entry.
    assert bad[0]["params"] == {"task": "zzz"}


def test_variant_defensively_copies_params() -> None:
    # The reuse-one-dict generator idiom must not alias the whole catalog.
    src = {"task_id": 0}
    variant = rlmesh.Variant("s/0", src)
    src["task_id"] = 99
    assert variant.params == {"task_id": 0}


def test_describe_omits_catalog_without_enumerate_variants() -> None:
    class _Factory:
        params = None

        def make(self, *, x: int = 0) -> None: ...

    payload = _describe(_Factory, "make")
    assert "catalog" not in payload
    assert "catalog_error" not in payload


def test_describe_emits_catalog() -> None:
    class _Factory:
        params = None

        @classmethod
        def enumerate_variants(cls):
            yield rlmesh.Variant("only/0", {"x": 1}, name="Only")

        def make(self, *, x: int = 0) -> None: ...

    payload = _describe(_Factory, "make")
    cat = cast("list[dict[str, object]]", payload["catalog"])
    assert cat == [{"id": "only/0", "params": {"x": 1}, "metadata": {"name": "Only"}}]


def test_describe_reports_broken_catalog_as_error() -> None:
    class _Factory:
        params = None

        @classmethod
        def enumerate_variants(cls):
            raise RuntimeError("boom")

        def make(self, *, x: int = 0) -> None: ...

    payload = _describe(_Factory, "make")
    assert "catalog" not in payload
    assert "boom" in cast("str", payload["catalog_error"])
