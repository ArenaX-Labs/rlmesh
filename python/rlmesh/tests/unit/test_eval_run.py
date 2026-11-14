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
                path.mkdir()
                (path / "result.json").write_text('{"schema": "rlm')
                raise OSError("no space left on device")
            return export(self, path, **kwargs)

        monkeypatch.setattr(Recorder, "export", full_disk)
    monkeypatch.setattr(rlmesh, "SandboxModel", Model)
    monkeypatch.setattr(rlmesh, "SandboxEnv", Env)
    status = tmp_path / "status.jsonl"
    # A staging folder an earlier run left behind is written over.
    stale = tmp_path / "workloads" / ".w0.partial" / "stale"
    stale.parent.mkdir(parents=True)
    stale.touch()
    job = _job(tmp_path, [f"w{i}" for i in range(5)], max_envs=2)
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
    # A bundle is in place only once written whole; one cut short stays
    # in its staging folder, which the report skips and a resume rewrites.
    staged = {p.name for p in (tmp_path / "workloads").iterdir()}
    done = {w for w, (state, _) in verdicts.items() if state == "done"}
    assert {name for name in staged if not name.startswith(".")} == done
    assert (".w3.partial" in staged) == (ctrl_c == "export fails")
    assert not (tmp_path / "workloads" / "w0" / "stale").exists()
    # The staging folder an export left half written is not reported.
    rlmesh.EvalReport.load(tmp_path / "workloads")


def test_eval_run_resume_keeps_a_bundle_renamed_into_place_before_its_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Served:
        def _run_routes(self, routes: list[Any], **_: object) -> tuple[list, Any]:
            episode = EpisodeResult(
                0, None, 5, 1.0, terminated=True, truncated=False, success=True
            )
            return [rlmesh.RunResult((episode,)) for _ in routes], None

    class Model:
        def __init__(self, image: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> Model:
            return self

        def __exit__(self, *exc: object) -> None:
            pass

        def serve(self) -> None:
            pass

        def _served(self) -> Served:
            return Served()

    class Env:
        def __init__(self, image: object, **kwargs: object) -> None:
            pass

        def close(self) -> None:
            pass

    rename = Path.rename

    def ctrl_c_after_rename(self: Path, target: Path) -> Path:
        renamed = rename(self, target)
        if Path(target).name == "w0":
            raise KeyboardInterrupt
        return renamed

    monkeypatch.setattr(rlmesh, "SandboxModel", Model)
    monkeypatch.setattr(rlmesh, "SandboxEnv", Env)
    workloads = tmp_path / "workloads"
    status = tmp_path / "status.jsonl"
    with monkeypatch.context() as interrupted:
        interrupted.setattr(Path, "rename", ctrl_c_after_rename)
        interrupted.setattr(
            "sys.stdin", io.StringIO(json.dumps(_job(tmp_path, ["w0", "w1"])))
        )
        with pytest.raises(KeyboardInterrupt):
            _eval_run.main()
    # w0 is in place with no verdict, w1 never written.
    assert not status.exists()
    assert {p.name for p in workloads.iterdir()} == {"w0"}

    # The resume keeps w0, as `recorded` does for a bundle holding its
    # planned episodes, and runs only w1.
    kept = rlmesh.RunResult.load(workloads / "w0")
    assert kept.num_episodes == 1
    before = (workloads / "w0" / "result.json").read_bytes()
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(_job(tmp_path, ["w1"]))))
    assert _eval_run.main() == 0
    assert (workloads / "w0" / "result.json").read_bytes() == before
    assert [
        json.loads(line)["workload"] for line in status.read_text().splitlines()
    ] == ["w1"]
    assert {p.name for p in workloads.iterdir()} == {"w0", "w1"}
    report = rlmesh.EvalReport.load(workloads).to_dict()
    assert report["models"][0]["overall"]["episodes"] == 2


def _job(tmp_path: Path, pending: list[str], max_envs: int = 2) -> dict[str, Any]:
    """An ``eval run`` job of five one-episode workloads on one model."""
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
    return {
        "plan": plan,
        "images": {"m:latest": "sha256:m", "env:latest": "sha256:e"},
        "pending": pending,
        "out": str(tmp_path),
        "gpus": None,
        "volumes": [],
        "maxEnvs": max_envs,
        "status": str(tmp_path / "status.jsonl"),
    }
