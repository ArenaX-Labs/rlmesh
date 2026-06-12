"""Run the v1 adapter conformance vectors against this implementation.

The vectors in crates/rlmesh-adapters/conformance/v1/ are the contract
every adapter implementation (the Rust core, this binding stack, future
bindings) must pass. See the README next to the vectors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rlmesh.adapters as adapt

CASES_DIR = (
    Path(__file__).resolve().parents[4]
    / "crates"
    / "rlmesh-adapters"
    / "conformance"
    / "v1"
    / "cases"
)
CASE_PATHS = sorted(CASES_DIR.glob("*.json"))


def dec(value: dict[str, Any]) -> Any:
    if value["kind"] == "text":
        return value["data"]
    if value["kind"] == "list":
        return value["data"]
    if value["kind"] == "map":
        return {key: dec(item) for key, item in value["data"].items()}
    return np.asarray(value["data"], dtype=value["dtype"]).reshape(value["shape"])


def assert_value(actual: Any, expected: dict[str, Any], atol: float) -> None:
    if expected["kind"] == "text":
        assert actual == expected["data"]
        return
    if expected["kind"] == "list":
        assert isinstance(actual, list)
        np.testing.assert_allclose(actual, expected["data"], atol=atol)
        return
    assert expected["kind"] == "array"
    arr = np.asarray(actual)
    assert str(arr.dtype) == expected["dtype"]
    assert list(arr.shape) == expected["shape"]
    np.testing.assert_allclose(
        arr.reshape(-1).astype(np.float64), expected["data"], atol=atol
    )


def test_vectors_exist() -> None:
    assert CASE_PATHS, f"no conformance vectors found in {CASES_DIR}"


@pytest.mark.parametrize("path", CASE_PATHS, ids=lambda p: p.stem)
def test_vector(path: Path) -> None:
    case = json.loads(path.read_text())

    if case["kind"] == "serialization":
        cls = adapt.EnvIOSpec if case["side"] == "env" else adapt.ModelIOSpec
        assert cls.from_dict(case["doc"]).to_dict() == case["doc"]
        return

    env_spec = adapt.EnvIOSpec.from_dict(case["env_spec"])
    model_spec = adapt.ModelIOSpec.from_dict(case["model_spec"])

    if case["kind"] == "resolve":
        expect = case["expect"]
        if "error_contains" in expect:
            with pytest.raises(adapt.AdapterResolutionError) as excinfo:
                adapt.resolve(env_spec, model_spec)
            assert expect["error_contains"] in str(excinfo.value)
        else:
            assert adapt.resolve(env_spec, model_spec).describe() == expect["describe"]
        return

    assert case["kind"] == "apply"
    adapter = adapt.resolve(env_spec, model_spec)
    atol = case["expect"]["atol"]
    payload = adapter.transform_obs(dec(case["observation"]))
    expected_payload = case["expect"]["payload"]
    assert sorted(payload) == sorted(expected_payload)
    for key, expected in expected_payload.items():
        assert_value(payload[key], expected, atol)
    action = adapter.transform_action(dec(case["model_output"]))
    assert_value(action, case["expect"]["action"], atol)
