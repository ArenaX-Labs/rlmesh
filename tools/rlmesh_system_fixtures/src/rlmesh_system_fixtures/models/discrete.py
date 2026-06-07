from __future__ import annotations

from rlmesh_system_fixtures.registry import model_fixture


@model_fixture("discrete.zero")
def discrete_zero(observation: object) -> int:
    _ = observation
    return 0


@model_fixture("discrete.one")
def discrete_one(observation: object) -> int:
    _ = observation
    return 1
