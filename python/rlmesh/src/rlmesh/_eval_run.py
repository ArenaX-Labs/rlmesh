"""Run the workloads of an ``rlmesh eval run`` plan in Docker sandboxes."""

from __future__ import annotations

import contextlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

MODEL_CONNECT_TIMEOUT_SECONDS = 900.0
ENV_CONNECT_TIMEOUT_SECONDS = 300.0
ENV_STARTS = 8
MAX_ENVS = 16


def main() -> int:
    job = json.load(sys.stdin)
    plan = job["plan"]
    pending = set(job["pending"])
    out = Path(job["out"])
    status_path = Path(job["status"])

    settled: set[str] = set()

    def report(workload: dict[str, Any], **fields: object) -> None:
        settled.add(workload["id"])
        with status_path.open("a") as status:
            status.write(json.dumps({"workload": workload["id"], **fields}) + "\n")

    for index, model in enumerate(plan["models"]):
        mine = [
            w for w in plan["workloads"] if w["model"] == index and w["id"] in pending
        ]
        if not mine:
            continue
        print(
            f"eval run: {model['label']}: {len(mine)} workloads on {model['image']}",
            file=sys.stderr,
            flush=True,
        )
        try:
            _run_model(plan, model, mine, job, out, report)
        except Exception as error:
            for workload in mine:
                if workload["id"] in settled:
                    continue
                report(
                    workload, status="failed", error=f"{type(error).__name__}: {error}"
                )
    return 0


def _run_model(
    plan: dict[str, Any],
    model: dict[str, Any],
    workloads: list[dict[str, Any]],
    job: dict[str, Any],
    out: Path,
    report: Any,
) -> None:
    import rlmesh

    def runtime(gpu: bool) -> rlmesh.SandboxRuntime:
        return rlmesh.SandboxRuntime(
            gpus=job["gpus"] if gpu else None, volumes=job["volumes"] or None
        )

    with rlmesh.SandboxModel(
        _pinned(job["images"], model["image"]),
        runtime=runtime(model["gpu"]),
        connect_timeout_seconds=MODEL_CONNECT_TIMEOUT_SECONDS,
        **model["params"],
    ) as served:
        served.serve()
        cap = max(1, job.get("maxEnvs") or MAX_ENVS)
        for first in range(0, len(workloads), cap):
            _run_wave(
                plan,
                model,
                workloads[first : first + cap],
                served,
                runtime,
                out,
                report,
                job["images"],
            )


def _run_wave(
    plan: dict[str, Any],
    model: dict[str, Any],
    workloads: list[dict[str, Any]],
    served: Any,
    runtime: Any,
    out: Path,
    report: Any,
    images: dict[str, str],
) -> None:
    import rlmesh

    envs: list[tuple[dict[str, Any], Any]] = []
    results: list[rlmesh.RunResult | BaseException] = []
    interrupt: BaseException | None = None
    try:

        def start(workload: dict[str, Any]) -> object:
            return rlmesh.SandboxEnv(
                _pinned(images, workload["envImage"]),
                runtime=runtime(workload["envGpu"]),
                connect_timeout_seconds=ENV_CONNECT_TIMEOUT_SECONDS,
                model=served,
                **workload["envParams"],
            )

        with ThreadPoolExecutor(ENV_STARTS) as pool:
            starts = [pool.submit(start, w) for w in workloads]
        for workload, started in zip(workloads, starts, strict=True):
            try:
                envs.append((workload, started.result()))
            except Exception as error:
                report(
                    workload, status="failed", error=f"{type(error).__name__}: {error}"
                )
        if not envs:
            return
        results, interrupt = served._served()._run_routes(  # pyright: ignore[reportPrivateUsage]
            [
                (
                    env,
                    dict(
                        episodes=w["episodes"],
                        seeds=w["seeds"] or None,
                        base_seed=w["seed"],
                        seed_key=w["seedKey"],
                        max_episode_steps=w["maxEpisodeSteps"],
                        max_episode_seconds=None,
                        prefetch_lead=plan["prefetchLead"] or 0,
                        # Walking ordinals from the workload's first global slot draws the plan's seeds
                        # (Workload::episode_seeds); trialIndexBase is absent without a global ordinal.
                        trial_index_base=w["seedSlotOffset"],
                        close_env=False,
                    ),
                )
                for w, env in envs
            ],
            execution_horizon=plan["executionHorizon"],
        )
    finally:
        for _, env in envs:
            with contextlib.suppress(Exception):
                env.close()
    try:
        _persist(model, envs, results, out, report)
    finally:
        # A pending Ctrl-C wins over an export that failed beside it.
        if interrupt is not None:
            raise interrupt


def _persist(
    model: dict[str, Any],
    envs: list[tuple[dict[str, Any], Any]],
    results: list[Any],
    out: Path,
    report: Any,
) -> None:
    from rlmesh.recorder import Recorder

    for (workload, _), result in zip(envs, results, strict=True):
        if isinstance(result, KeyboardInterrupt):
            report(workload, status="failed", error="interrupted")
            continue
        if isinstance(result, BaseException):
            report(
                workload, status="failed", error=f"{type(result).__name__}: {result}"
            )
            continue
        config = {
            key: workload[name]
            for key, name in (
                ("suite", "suite"),
                ("workload", "id"),
                ("scenario", "scenario"),
                ("seedKey", "seedKey"),
                ("seed", "seed"),
                ("seedSlotOffset", "seedSlotOffset"),
                ("trialIndexBase", "trialIndexBase"),
                ("maxEpisodeSteps", "maxEpisodeSteps"),
            )
            if workload[name] is not None
        }
        if model["checkpoint"] is not None:
            config["checkpoint"] = model["checkpoint"]
        if workload["shard"] is not None:
            config["shard"] = workload["shard"]
        bundle = out / "workloads" / workload["id"]
        partial = bundle.with_name(f".{bundle.name}.partial")
        shutil.rmtree(partial, ignore_errors=True)
        with Recorder() as recorder:
            recorder.add(
                result,
                model=model["label"],
                env=workload["env"],
                task=workload["task"],
                config=config,
            )
            recorder.export(partial)
        partial.rename(bundle)
        report(workload, status="done", episodes=result.num_episodes)


def _pinned(images: dict[str, str], image: str) -> str:
    return f"docker://{images[image]}"


if __name__ == "__main__":
    sys.exit(main())
