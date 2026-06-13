"""Run the v1 adapter conformance vectors against this implementation.

The vectors in crates/rlmesh-adapters/conformance/v1/ are the contract
every adapter implementation (the Rust core, this binding stack, future
bindings) must pass. See the README next to the vectors.

Each resolve/apply vector carries the env *tags*, the observation
and action *spaces* (as the adapters ``SpaceView`` projection), and the
model spec. We rebuild gymnasium spaces from those views and drive the
public Python API, exercising the full tag -> join -> resolve -> apply
path the native core runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
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


def gym_space_from_view(view: dict[str, Any]) -> gym.spaces.Space[Any]:
    """Rebuild a gymnasium space from an adapters ``SpaceView`` projection.

    The view is what ``SpaceView::from(&SpaceSpec)`` serializes; rebuilding a
    gymnasium space that re-projects to the same view lets the vectors drive
    the real ``parse_space`` -> ``SpaceView`` path through the binding.
    """
    kind = view["kind"]
    if kind == "dict":
        return gym.spaces.Dict(
            {
                key: gym_space_from_view(child)
                for key, child in zip(view["keys"], view["children"], strict=True)
            }
        )
    if kind == "tuple":
        return gym.spaces.Tuple(
            tuple(gym_space_from_view(child) for child in view["children"])
        )
    if kind == "text":
        return gym.spaces.Text(max_length=1024)
    if kind == "box":
        dtype = np.dtype(view["dtype"])
        shape = tuple(int(dim) for dim in view.get("shape", []))
        low, high = _box_bounds(view, dtype, shape)
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    raise ValueError(f"conformance space view kind {kind!r} is not supported")


def _box_bounds(
    view: dict[str, Any], dtype: np.dtype[Any], shape: tuple[int, ...]
) -> tuple[Any, Any]:
    numel = int(np.prod(shape)) if shape else 1
    if "low" in view and "high" in view:
        low = np.asarray(view["low"], dtype=np.float64)
        high = np.asarray(view["high"], dtype=np.float64)
        low_out = low.reshape(shape) if low.size == numel else float(low.reshape(-1)[0])
        high_out = (
            high.reshape(shape) if high.size == numel else float(high.reshape(-1)[0])
        )
        return low_out, high_out
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return info.min, info.max
    return -np.inf, np.inf


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


def resolve_case(case: dict[str, Any]) -> adapt.IOAdapter:
    tags = adapt.EnvTags.from_dict(case["env_tags"])
    model_spec = adapt.ModelSpec.from_dict(case["model_spec"])
    obs_space = gym_space_from_view(case["observation_space"])
    action_space = gym_space_from_view(case["action_space"])
    return adapt.resolve(tags, obs_space, action_space, model_spec)


def test_vectors_exist() -> None:
    assert CASE_PATHS, f"no conformance vectors found in {CASES_DIR}"


@pytest.mark.parametrize("path", CASE_PATHS, ids=lambda p: p.stem)
def test_vector(path: Path) -> None:
    case = json.loads(path.read_text())

    if case["kind"] == "serialization":
        if case["side"] == "env":
            assert adapt.EnvTags.from_dict(case["doc"]).to_dict() == case["doc"]
        else:
            assert adapt.ModelSpec.from_dict(case["doc"]).to_dict() == case["doc"]
        return

    if case["kind"] == "resolve":
        expect = case["expect"]
        if "error_contains" in expect:
            with pytest.raises(adapt.AdapterResolutionError) as excinfo:
                resolve_case(case)
            assert expect["error_contains"] in str(excinfo.value)
        else:
            assert resolve_case(case).describe() == expect["describe"]
        return

    assert case["kind"] == "apply"
    adapter = resolve_case(case)
    atol = case["expect"]["atol"]
    payload = adapter.transform_obs(dec(case["observation"]))
    expected_payload = case["expect"]["payload"]
    assert sorted(payload) == sorted(expected_payload)
    for key, expected in expected_payload.items():
        assert_value(payload[key], expected, atol)
    action = adapter.transform_action(dec(case["model_output"]))
    assert_value(action, case["expect"]["action"], atol)
