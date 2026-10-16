from __future__ import annotations

import json
from pathlib import Path

from rlmesh_api_surface import collect_python_api_surface


def test_api_surface_contract() -> None:
    contract_path = Path(__file__).parent / "snapshots" / "api_surface.json"
    expected = json.loads(contract_path.read_text(encoding="utf-8"))

    surface = collect_python_api_surface()

    assert surface.to_contract() == expected


def test_stable_public_api_has_documentation() -> None:
    surface = collect_python_api_surface()

    assert surface.missing_stable_documentation() == []


def test_bridge_and_deferred_parameter_labels() -> None:
    objects = collect_python_api_surface().to_contract()["rlmesh"]["objects"]
    assert objects["sanitize_metadata"]["stability"] == "Bridge"
    assert objects["params"]["stability"] == "Experimental"
    for name in ("Model", "EnvFactory"):
        assert objects[name]["members"]["params"]["stability"] == "Experimental"


def test_adapter_function_and_plumbing_labels() -> None:
    objects = collect_python_api_surface().to_contract()["rlmesh.adapters"]["objects"]
    assert objects["tag"]["kind"] == "function"
    assert "validate: bool = True" in objects["tag"]["signature"]
    assert "signature" not in objects["Adapter"]
    members = objects["Adapter"]["members"]
    assert members["__init__"]["stability"] == "Internal"
    assert members["serve_route"]["stability"] == "Internal"
    assert members["transform_obs_value"]["stability"] == "Bridge"
    assert members["transform_action_value"]["stability"] == "Bridge"


def test_recording_types_are_stable_with_internal_helpers() -> None:
    contract = collect_python_api_surface().to_contract()
    assert contract["rlmesh"]["objects"]["FrameSources"]["stability"] == "Stable"
    objects = contract["rlmesh.recorder"]["objects"]
    for name in ("SCHEMA", "MediaRef", "EpisodeRecord", "WorkloadRecord", "Recorder"):
        assert objects[name]["stability"] == "Stable"
    assert objects["ResultSet"]["stability"] == "Internal"
    assert objects["EpisodeRecord"]["members"]["from_result"]["stability"] == "Internal"
