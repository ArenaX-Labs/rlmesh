"""``rlmesh eval run``'s sandbox launcher, with fake containers."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
import rlmesh
from rlmesh import _eval_run
from rlmesh._models._eval import EpisodeResult


def test_eval_run_keeps_at_most_max_envs_containers_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live, peak, waves = [0], [0], [0]

    class Env:
        def __init__(self, *args: object, **kwargs: object) -> None:
            live[0] += 1
            peak[0] = max(peak[0], live[0])

        def close(self) -> None:
            live[0] -= 1

    class Served:
        def _run_routes(self, routes: list[Any], **_: object) -> tuple[list, Any]:
            waves[0] += 1
            episode = EpisodeResult(
                0, None, 5, 1.0, terminated=True, truncated=False, success=True
            )
            results = [rlmesh.RunResult((episode,)) for _ in routes]
            # Ctrl-C caught as the second wave's last route finished.
            return results, KeyboardInterrupt() if waves[0] == 2 else None

    class Model:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> Model:
            return self

        def __exit__(self, *exc: object) -> None:
            pass

        def serve(self) -> None:
            pass

        def _served(self) -> Served:
            return Served()

    monkeypatch.setattr(rlmesh, "SandboxModel", Model)
    monkeypatch.setattr(rlmesh, "SandboxEnv", Env)
    workloads = [
        {
            "id": f"w{i}",
            "model": 0,
            "env": "env",
            "task": "task",
            "envImage": "env:latest",
            "envGpu": False,
            "envParams": {},
            "episodes": 1,
            "seeds": None,
            "seed": 0,
            "seedKey": "env",
            "seedSlotOffset": i,
            "trialIndexBase": i,
            "maxEpisodeSteps": None,
            "suite": None,
            "scenario": None,
            "shard": None,
        }
        for i in range(5)
    ]
    plan = {
        "models": [
            {
                "label": "m",
                "image": "m:latest",
                "gpu": False,
                "params": {},
                "checkpoint": None,
            }
        ],
        "workloads": workloads,
        "prefetchLead": None,
        "executionHorizon": None,
    }
    status = tmp_path / "status.jsonl"
    job = {
        "plan": plan,
        "pending": [w["id"] for w in workloads],
        "out": str(tmp_path),
        "gpus": None,
        "volumes": [],
        "maxEnvs": 2,
        "status": str(status),
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(job)))

    with pytest.raises(KeyboardInterrupt):
        _eval_run.main()
    assert peak[0] == 2 and live[0] == 0
    # The interrupted wave's finished workloads are kept; the third never starts.
    assert waves[0] == 2
    lines = [json.loads(line) for line in status.read_text().splitlines()]
    assert [line["status"] for line in lines] == ["done"] * 4
    assert all((tmp_path / "workloads" / f"w{i}").is_dir() for i in range(4))
