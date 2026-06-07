from __future__ import annotations

import json
from pathlib import Path

from rlmesh_api_surface import collect_python_api_surface


def test_public_api_snapshot() -> None:
    snapshot_path = Path(__file__).parent / "snapshots" / "public_api.json"
    expected = json.loads(snapshot_path.read_text(encoding="utf-8"))

    surface = collect_python_api_surface()

    assert surface.to_snapshot() == expected


def test_stable_public_api_has_documentation() -> None:
    surface = collect_python_api_surface()

    assert surface.missing_stable_documentation() == []
