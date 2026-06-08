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
