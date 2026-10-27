"""Run the workloads of an ``rlmesh eval run`` plan in Docker sandboxes.

Private glue for the CLI, which compiles the plan (``rlmesh::binding::plan``),
skips what ``--resume`` finds recorded, and reports on the bundles written
here. It reads one job on stdin::

    {"plan": {...}, "pending": [workload ids], "out": dir,
     "gpus": "all" | null, "volumes": ["host:ctr", ...], "status": path}

Models run one at a time, each as one :class:`~rlmesh.SandboxModel` that
every workload of it reaches as a route of one
:meth:`~rlmesh.Model.run_routes` call, its env a :class:`~rlmesh.SandboxEnv`
of its own. Each finished workload is written as an ``rlmesh.result.v1``
bundle under ``<out>/workloads/<id>`` and gets one JSON line in the status
file; a failed one gets its error there instead.
"""

from __future__ import annotations

import contextlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

#: A model loads its weights before it answers; an env such as a LIBERO
#: suite loads its sim assets before it serves.
MODEL_CONNECT_TIMEOUT_SECONDS = 900.0
ENV_CONNECT_TIMEOUT_SECONDS = 300.0
#: Env containers started at once.
ENV_STARTS = 8


def main() -> int:
    job = json.load(sys.stdin)
    plan = job["plan"]
    pending = set(job["pending"])
    out = Path(job["out"])
    status_path = Path(job["status"])

    def report(workload: dict[str, Any], **fields: object) -> None:
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
            # The model failed every workload it had left.
            for workload in mine:
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
    from rlmesh.recorder import Recorder

    def runtime(gpu: bool) -> rlmesh.SandboxRuntime:
        return rlmesh.SandboxRuntime(
            gpus=job["gpus"] if gpu else None, volumes=job["volumes"] or None
        )

    envs: list[tuple[dict[str, Any], Any]] = []
    results: list[rlmesh.RunResult | BaseException] = []
    with rlmesh.SandboxModel(
        model["image"],
        runtime=runtime(model["gpu"]),
        connect_timeout_seconds=MODEL_CONNECT_TIMEOUT_SECONDS,
        **model["params"],
    ) as served:
        served.serve()

        def start(workload: dict[str, Any]) -> object:
            return rlmesh.SandboxEnv(
                workload["envImage"],
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
                # One env failing fails only its workload.
                report(
                    workload, status="failed", error=f"{type(error).__name__}: {error}"
                )
        if not envs:
            return
        try:
            results = served._served()._run_routes(  # pyright: ignore[reportPrivateUsage]
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
                # A killed env has nothing left to close.
                with contextlib.suppress(Exception):
                    env.close()
    interrupt = None
    for (workload, _), result in zip(envs, results, strict=True):
        if isinstance(result, KeyboardInterrupt):
            # Ctrl-C cut it short; the workloads that finished are still recorded.
            interrupt = result
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
        with Recorder() as recorder:
            recorder.add(
                result,
                model=model["label"],
                env=workload["env"],
                task=workload["task"],
                config=config,
            )
            recorder.export(out / "workloads" / workload["id"])
        report(workload, status="done", episodes=result.num_episodes)
    if interrupt is not None:
        raise interrupt


if __name__ == "__main__":
    sys.exit(main())
