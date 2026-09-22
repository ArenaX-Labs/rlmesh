"""`RunResult.advisories`: the runtime relay's advisories reach the result."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
import rlmesh


class _TinyEnv:
    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(self) -> None:
        from rlmesh import spaces

        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(1)

    def reset(
        self, *, seed: object = None, options: Any = None
    ) -> tuple[int, dict[str, object]]:
        return 0, {}

    def step(self, action: object) -> tuple[int, float, bool, bool, dict[str, object]]:
        return 0, 1.0, True, False, {}

    def close(self) -> None:
        pass


def test_the_open_source_relay_raises_no_advisories() -> None:
    from rlmesh.numpy import Model

    try:
        result = Model(lambda obs: 0).run(_TinyEnv(), max_episodes=1)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise

    assert result.num_episodes == 1
    assert result.advisories == ()


def test_report_advisories_reach_the_run_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh.numpy import Model

    caution = object()

    def converted_run(*_args: object, **_kwargs: object) -> dict[str, Any]:
        return {"episodes": [], "telemetry": [], "advisories": [caution]}

    model = Model(lambda obs: 0)
    monkeypatch.setattr(model, "_run_native", converted_run)

    result = model.run("tcp://127.0.0.1:1", max_episodes=1)

    assert result.advisories == (caution,)
    assert rlmesh.RunResult().advisories == ()
