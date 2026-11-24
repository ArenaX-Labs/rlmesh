from __future__ import annotations

import json
from pathlib import Path

from rlmesh._rlmesh import episode_seed

FIXTURE = (
    Path(__file__).resolve().parents[4]
    / "crates"
    / "rlmesh"
    / "conformance"
    / "episode_seeds.v1.json"
)


def test_episode_seeds_match_the_cross_language_fixture():
    cases = json.loads(FIXTURE.read_text())["cases"]
    assert cases
    for case in cases:
        got = episode_seed(case["base_seed"], case["seed_key"], case["slot"])
        assert got == case["seed"], case
