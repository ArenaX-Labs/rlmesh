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


@pytest.mark.parametrize("cut", [False, True])
def test_eval_run_waves_of_max_envs_keep_finished_bundles_past_a_failure_and_ctrl_c(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cut: bool
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
            results: list[Any] = [rlmesh.RunResult((episode,)) for _ in routes]
            if waves[0] == 1:
                results[1] = ValueError("the env broke")
                return results, None
            # Ctrl-C in the second wave: it cut the last route short, or was
            # caught as that route finished.
            interrupt = KeyboardInterrupt()
            if cut:
                results[1] = interrupt
            return results, interrupt

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
    # Finished workloads keep their bundles, a failed or cut one fails alone,
    # and the wave after the interrupt never starts.
    assert waves[0] == 2
    verdicts = {
        line["workload"]: (line["status"], line.get("error"))
        for line in map(json.loads, status.read_text().splitlines())
    }
    assert verdicts == {
        "w0": ("done", None),
        "w1": ("failed", "ValueError: the env broke"),
        "w2": ("done", None),
        "w3": ("failed", "interrupted") if cut else ("done", None),
    }
    bundles = {p.name for p in (tmp_path / "workloads").iterdir()}
    assert bundles == ({"w0", "w2"} if cut else {"w0", "w2", "w3"})
