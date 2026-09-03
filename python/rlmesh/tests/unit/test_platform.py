from __future__ import annotations

import io
import json
import urllib.request

import pytest
from rlmesh import platform


def test_scenario_tasks_expand_a_group_and_ask_for_workers() -> None:
    tasks = platform.scenario_tasks(
        "prebuilt/libero-mujoco330", "libero_90", tasks=2, episodes=7, workers=2
    )

    assert tasks == [
        {
            "env": "prebuilt/libero-mujoco330",
            "taskId": f"libero_90/{i}",
            "episodes": 7,
            "seed": 0,
            "workers": 2,
            "metadata": {"environmentId": f"prebuilt/libero-mujoco330:libero_90/{i}"},
        }
        for i in range(2)
    ]

    (single,) = platform.scenario_tasks(
        "env", "g", task_ids=["g/7"], episodes=2, seed=5
    )
    assert single == {
        "env": "env",
        "taskId": "g/7",
        "episodes": 2,
        "seed": 5,
        "metadata": {"environmentId": "env:g/7"},
    }


def test_client_sends_bearer_and_repeatable_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[urllib.request.Request] = []
    pages = iter(
        [
            {"items": [{"id": "eval_1"}], "nextCursor": "c2"},
            {"items": [{"id": "eval_2"}], "nextCursor": None},
        ]
    )

    class Response(io.BytesIO):
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_: object) -> None:
            self.close()

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> Response:
        seen.append(request)
        return Response(json.dumps(next(pages)).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = platform.Client(url="https://example.test/", token="secret")

    ids = [e["id"] for e in client.evaluations(tags={"experiment": "x"}, limit=1)]

    assert ids == ["eval_1", "eval_2"]
    assert seen[0].get_header("Authorization") == "Bearer secret"
    assert (
        seen[0].full_url
        == "https://example.test/v1/evaluations?limit=1&tag=experiment%3Ax"
    )
    assert "cursor=c2" in seen[1].full_url
