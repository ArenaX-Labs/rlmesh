from __future__ import annotations

import pytest


def test_normalize_autoreset_mode_restores_enum() -> None:
    autoreset = pytest.importorskip("gymnasium.vector").AutoresetMode
    from rlmesh.client.remote_vector_env import _normalize_autoreset_mode

    normalized = _normalize_autoreset_mode({"autoreset_mode": "NextStep"})

    assert isinstance(normalized["autoreset_mode"], autoreset)
    assert normalized["autoreset_mode"] is autoreset.NEXT_STEP


def test_normalize_autoreset_mode_passes_through_other_keys() -> None:
    from rlmesh.client.remote_vector_env import _normalize_autoreset_mode

    metadata = {"render_fps": 30}
    normalized = _normalize_autoreset_mode(metadata)

    assert normalized == metadata


def test_normalize_autoreset_mode_leaves_unknown_string() -> None:
    from rlmesh.client.remote_vector_env import _normalize_autoreset_mode

    normalized = _normalize_autoreset_mode({"autoreset_mode": "bogus"})

    assert normalized["autoreset_mode"] == "bogus"


def test_normalize_autoreset_mode_idempotent_on_enum() -> None:
    autoreset = pytest.importorskip("gymnasium.vector").AutoresetMode
    from rlmesh.client.remote_vector_env import _normalize_autoreset_mode

    normalized = _normalize_autoreset_mode({"autoreset_mode": autoreset.SAME_STEP})

    assert normalized["autoreset_mode"] is autoreset.SAME_STEP
