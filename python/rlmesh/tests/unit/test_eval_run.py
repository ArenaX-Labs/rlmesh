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


@pytest.mark.parametrize("ctrl_c", [None, "late", "cut", "all cut", "export fails"])
def test_eval_run_waves_of_max_envs_keep_finished_bundles_past_a_failure_and_ctrl_c(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ctrl_c: str | None
) -> None:
    live, peak, waves = [0], [0], [0]
    launched: set[object] = set()

    class Env:
        def __init__(self, image: object, **kwargs: object) -> None:
            launched.add(image)
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
            if waves[0] != 2 or ctrl_c is None:
                return results, None
            # Ctrl-C in the second wave: caught as its routes finished, or it
            # cut the last route or both short.
            interrupt = KeyboardInterrupt()
            if ctrl_c in ("cut", "all cut"):
                results[1] = interrupt
            if ctrl_c == "all cut":
                results[0] = interrupt
            return results, interrupt

    class Model:
        def __init__(self, image: object, **kwargs: object) -> None:
            launched.add(image)

        def __enter__(self) -> Model:
            return self

        def __exit__(self, *exc: object) -> None:
            pass

        def serve(self) -> None:
            pass

        def _served(self) -> Served:
            return Served()

    if ctrl_c == "export fails":
        from rlmesh.recorder import Recorder

        export = Recorder.export

        def full_disk(self: Recorder, path: Path, **kwargs: Any) -> Path:
            if path.name == ".w3.partial":
                raise OSError("no space left on device")
            return export(self, path, **kwargs)

        monkeypatch.setattr(Recorder, "export", full_disk)
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
        "images": {"m:latest": "sha256:m", "env:latest": "sha256:e"},
        "pending": [w["id"] for w in workloads],
        "out": str(tmp_path),
        "gpus": None,
        "volumes": [],
        "maxEnvs": 2,
        "status": str(status),
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(job)))

    if ctrl_c is None:
        assert _eval_run.main() == 0
    else:
        with pytest.raises(KeyboardInterrupt):
            _eval_run.main()
    assert peak[0] == 2 and live[0] == 0
    # Every wave runs the images the CLI resolved, by id.
    assert launched == {"docker://sha256:m", "docker://sha256:e"}
    # Finished workloads keep their bundles, a failed or cut one fails alone,
    # and the wave after the interrupt never starts; without one the last,
    # partial wave runs too.
    assert waves[0] == (3 if ctrl_c is None else 2)
    verdicts = {
        line["workload"]: (line["status"], line.get("error"))
        for line in map(json.loads, status.read_text().splitlines())
    }
    cut = ("failed", "interrupted")
    want = {
        None: {"w2": ("done", None), "w3": ("done", None), "w4": ("done", None)},
        "late": {"w2": ("done", None), "w3": ("done", None)},
        "cut": {"w2": ("done", None), "w3": cut},
        "all cut": {"w2": cut, "w3": cut},
        "export fails": {"w2": ("done", None)},
    }[ctrl_c]
    assert verdicts == {
        "w0": ("done", None),
        "w1": ("failed", "ValueError: the env broke"),
        **want,
    }
    # A bundle is in place only once written whole.
    bundles = {p.name for p in (tmp_path / "workloads").iterdir()}
    done = {w for w, (state, _) in verdicts.items() if state == "done"}
    assert bundles == done
