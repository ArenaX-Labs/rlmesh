"""The :class:`Recorder` -- capture eval runs, export a portable bundle.

Path-agnostic by design: every OSS execution path (``rlmesh.run`` or a
:class:`~rlmesh.Binding` stepped by hand) converges on the same
:class:`~rlmesh.RunResult`, so :meth:`Recorder.add` records from that return value
with no runtime changes. :meth:`Recorder.capture` returns a
:class:`~rlmesh.RunHooks` for live per-episode capture (and media) on the
Python-driven paths. Either way, :meth:`Recorder.export` writes an
``rlmesh.result.v1`` bundle the managed platform's upload path can ingest.
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .constants import (
    DEFAULT_CODEC,
    DEFAULT_FPS,
    DEFAULT_QUALITY,
    DEFAULT_VIDEO_INFO_KEYS,
)
from .export import write_bundle
from .frames import sanitize_part
from .hooks import CaptureHooks
from .media import MediaStager
from .schema import EpisodeRecord, ResultSet, WorkloadRecord

if TYPE_CHECKING:
    from .._models import RunContext, RunHooks, RunResult, TelemetryRow
    from .._rlmesh import PyEpisodeArtifacts


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Recorder:
    """Accumulates recorded workloads and exports them as one bundle.

    Each :meth:`add` / :meth:`capture` starts a new workload (one model x env x
    task cell). A single ``capture`` hooks object may drive several ``run`` calls;
    their episodes accumulate into that one workload, uniquely indexed.

    Example -- metrics only, path-agnostic::

        rec = rlmesh.Recorder()
        result = rlmesh.run(model, env, episodes=50)
        rec.add(result, model="smolvla", env="libero", task="libero-spatial-0")
        rec.export("results/run.zip")

    Example -- with media::

        rec = rlmesh.Recorder()
        rlmesh.run(
            model,
            env,
            episodes=50,
            hooks=rec.capture(model="smolvla", env="libero", task="libero-spatial-0"),
        )
        rec.export("results/run.zip")
    """

    def __init__(
        self,
        *,
        result_set_id: str | None = None,
        fps: int = DEFAULT_FPS,
        quality: int = DEFAULT_QUALITY,
        codec: str = DEFAULT_CODEC,
        ffmpeg: str | None = None,
    ) -> None:
        """Create a recorder.

        Captured frames are encoded to mp4. ``codec="av1"`` (the default)
        encodes in process (pure Rust, no ffmpeg); ``codec="h264"`` plays in
        every browser and runs the ``ffmpeg`` you installed (``ffmpeg`` on
        ``PATH``, or this path), which must have libx264: rlmesh ships no
        H.264 encoder. A missing one raises :class:`RuntimeError` here. ``fps``
        sets the recorded playback rate and ``quality`` (1..=100, higher is
        better/larger, AV1 only) trades file size against fidelity.
        """
        if (
            not isinstance(fps, int)  # pyright: ignore[reportUnnecessaryIsInstance]
            or fps < 1
        ):
            raise ValueError(f"Recorder.fps must be an integer >= 1; got {fps!r}")
        if (
            not isinstance(quality, int)  # pyright: ignore[reportUnnecessaryIsInstance]
            or not 1 <= quality <= 100
        ):
            raise ValueError(
                f"Recorder.quality must be an integer in 1..=100; got {quality!r}"
            )
        from .._rlmesh import PyVideoWriter

        PyVideoWriter.check_codec(codec, ffmpeg)
        self._result_set = ResultSet(result_set_id=result_set_id or uuid.uuid4().hex)
        self._stager = MediaStager(fps=fps, quality=quality, codec=codec, ffmpeg=ffmpeg)
        #: Each workload's telemetry rows (keyed by ``id(workload)``), one
        #: batch per recorded run, written beside the workload.
        self._telemetry: dict[int, list[TelemetryRow]] = {}
        #: Each recorded workload's artifact tree: its writer and bundle prefix.
        self._trees: dict[int, tuple[PyEpisodeArtifacts, str]] = {}
        #: This machine's hardware stamp, taken with the first tree.
        self._stamp: dict[str, Any] | None = None

    @property
    def result_set_id(self) -> str:
        """The locally minted id anchoring this bundle (re-upload dedupe key)."""
        return self._result_set.result_set_id

    @property
    def workloads(self) -> tuple[WorkloadRecord, ...]:
        """The workloads recorded so far, in the order they were started."""
        return tuple(self._result_set.workloads)

    def _new_workload(
        self,
        model: str,
        env: str,
        task: str | None,
        config: dict[str, Any] | None,
    ) -> WorkloadRecord:
        workload = WorkloadRecord(
            model=model,
            env=env,
            task=task if task is not None else env,
            config=dict(config or {}),
        )
        self._result_set.workloads.append(workload)
        return workload

    def add(
        self,
        result: RunResult,
        *,
        model: str,
        env: str,
        task: str | None = None,
        config: dict[str, Any] | None = None,
        included_in_metrics: bool = True,
    ) -> WorkloadRecord:
        """Record a completed :class:`~rlmesh.RunResult` as a new workload (no media).

        The post-hoc path -- works for any run's return value, including pure
        ``rlmesh.run``. For video/frames, use :meth:`capture` instead.
        """
        workload = self._new_workload(model, env, task, config)
        self._telemetry[id(workload)] = list(result.telemetry)
        for episode in result.episodes:
            workload.episodes.append(
                EpisodeRecord.from_result(
                    episode,
                    index=len(workload.episodes),
                    included_in_metrics=included_in_metrics,
                )
            )
        return workload

    def capture(
        self,
        *,
        model: str,
        env: str,
        task: str | None = None,
        config: dict[str, Any] | None = None,
        cameras: list[str] | None = None,
        binding: RunContext | None = None,
        video_info_keys: tuple[str, ...] = DEFAULT_VIDEO_INFO_KEYS,
        included_in_metrics: bool = True,
        session: object = None,
    ) -> RunHooks:
        """A :class:`~rlmesh.RunHooks` that records a live run into a new workload.

        Frame sourcing:

        * By default the recorded sources are auto-discovered on the first step
          from the run context driving the hooks (received via
          :meth:`~rlmesh.RunHooks.on_run_start`): the env's ``render()`` frame
          (when it exposes an rgb render mode) plus every declared image role,
          each to its own video -- exactly the sources the live viewer offers,
          under the same labels.
        * ``cameras`` -- explicit sources to record each step: env image roles
          (read as HWC) and/or ``"render"`` for the env's ``render()`` frame.
          ``cameras=[]`` opts out of frames entirely (metrics only).
        * ``binding`` -- a :class:`~rlmesh.RunContext`, only needed when
          driving the hooks by hand (outside a run or a binding); an explicitly
          passed one also wins over the running loop's. The ``render()`` frame
          is reachable on a binding and, on ``Model.run``, for a local env
          object; a vector env is refused unless ``cameras=[]``.

        ``video_info_keys`` name the step-``info`` keys checked for an env-produced
        video file path (the env renders its own video); the file is copied into the
        bundle. Recorded frames are encoded to AV1 mp4 in process. ``session``
        is 0.1's name for ``binding`` and is refused.
        """
        if session is not None:
            from .._upgrade import removed

            removed(
                "Recorder.capture(session=...)",
                "the run context is binding=: Recorder.capture(..., binding=...)",
            )
        workload = self._new_workload(model, env, task, config)
        ordinal = len(self._result_set.workloads) - 1
        prefix = f"w{ordinal:03d}-{sanitize_part(model)}/{sanitize_part(workload.task)}"
        return CaptureHooks(
            workload=workload,
            stager=self._stager,
            prefix=prefix,
            cameras=cameras,
            binding=binding,
            video_keys=tuple(video_info_keys),
            included_in_metrics=included_in_metrics,
            telemetry=self._telemetry.setdefault(id(workload), []),
        )

    def record(
        self,
        *,
        model: str,
        env: str,
        spec: dict[str, Any] | None = None,
        task: str | None = None,
        config: dict[str, Any] | None = None,
        included_in_metrics: bool = True,
    ) -> RunHooks:
        """A :class:`~rlmesh.RunHooks` that records a run's artifact tree (Experimental).

        Writes the managed platform's layout under
        ``workloads/cNNNN/attempts/a001/`` in the bundle: per episode
        ``episodes/epNNNNNN/episode.json`` (seed, trial, verdict, timing,
        invariant flags), ``trajectory.parquet`` and
        ``video/{camera}.{stream}.mp4``, and the run's
        ``telemetry/windows.jsonl``; the document carries the machine's
        hardware stamp. ``spec`` is the platform's ``record`` block
        (``{"video": {...}, "trajectory": {...}}``); the default records no
        video, and a trajectory of reward, info and step timing (no
        observations or actions) for every episode. Raw and perturbed streams
        differ only under a ``transform_observation`` / ``transform_action``
        hook. Runs only, not a manual :meth:`Model.bind <rlmesh.Model.bind>`.
        """
        from .._rlmesh import PyEpisodeArtifacts

        workload = self._new_workload(model, env, task, config)
        ordinal = len(self._result_set.workloads)
        workload_id = f"c{ordinal:04d}"
        prefix = f"workloads/{workload_id}/attempts/a001"
        root = self._stager.tree_dir()
        tap = PyEpisodeArtifacts(
            root,
            json.dumps(spec or {}),
            workload.task,
            workload_id,
            self._stager.quality,
            self._stager.codec,
            self._stager.ffmpeg,
        )
        self._trees[id(workload)] = (tap, prefix)
        if self._stamp is None:
            self._stamp = json.loads(PyEpisodeArtifacts.stamp())
        return CaptureHooks(
            workload=workload,
            stager=self._stager,
            prefix=f"w{ordinal - 1:03d}-{sanitize_part(model)}/{sanitize_part(workload.task)}",
            cameras=[],
            binding=None,
            video_keys=DEFAULT_VIDEO_INFO_KEYS,
            included_in_metrics=included_in_metrics,
            telemetry=self._telemetry.setdefault(id(workload), []),
            tree=(tap, root, prefix),
        )

    def to_dict(self, *, recorded_at: str | None = None) -> dict[str, Any]:
        """The ``rlmesh.result.v1`` document as a JSON-native dict."""
        document = self._result_set.to_dict(recorded_at=recorded_at or _now_iso())
        for workload, out in zip(
            self._result_set.workloads, document["workloads"], strict=True
        ):
            if rows := self._telemetry.get(id(workload)):
                out["telemetry"] = [dataclasses.asdict(row) for row in rows]
            if tree := self._trees.get(id(workload)):
                out["artifactRoot"] = tree[1]
                if phases := json.loads(tree[0].phases()):
                    out["phases"] = phases
        if self._stamp is not None:
            document["stamp"] = self._stamp
        return document

    def export(
        self,
        path: str | Path,
        *,
        archive: bool | str | None = None,
        recorded_at: str | None = None,
    ) -> Path:
        """Write the bundle to ``path`` and return it.

        Writes a folder by default, or a zip when ``path`` ends ``.zip`` or
        ``archive`` is ``True`` / ``"zip"``. ``archive=False`` forces a folder.
        """
        if self._stager.closed and self._stager.assets:
            raise RuntimeError(
                "Recorder is closed (its staged media was removed); "
                "export before close()"
            )
        return write_bundle(
            self.to_dict(recorded_at=recorded_at),
            Path(path),
            assets=self._stager.assets,
            archive=archive,
        )

    def close(self) -> None:
        """Remove the temp staging dir holding captured frame stacks. Idempotent.

        Call after :meth:`export`. Staged files are also under the OS temp dir, so a
        missed ``close`` is cleaned by the OS eventually.
        """
        self._stager.cleanup()

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
