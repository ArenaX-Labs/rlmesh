"""The ``rlmesh.result.v1`` document model.

Plain dataclasses that mirror the SDK's own :class:`~rlmesh.EpisodeResult`
vocabulary (``reward``/``steps``/``duration_s``/``success``), *not* the managed
platform's camelCase ``episode.v1`` names. The closed-side mapper renames on
upload (``reward -> cumulativeReward``, ``steps -> stepCount``,
``duration_s * 1000 -> durationMs``) and mints the platform join ids; keeping this
document in SDK vocabulary is what lets the open-source repo stay decoupled.

Every ``to_dict`` emits JSON-native values only (str/int/float/bool/None/list/dict)
so :func:`json.dump` needs no custom encoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .constants import SCHEMA

if TYPE_CHECKING:
    from .._models import EpisodeResult


@dataclass(frozen=True)
class MediaRef:
    """A per-episode media asset referenced by a bundle-relative ``path``.

    ``kind="video"`` is an env-produced container (e.g. mp4) carried verbatim;
    ``kind="frames"`` is a captured per-step stack (an uint8 ``(T, H, W, C)`` array
    saved as compressed ``.npz``) that the closed-side mapper stitches into mp4.
    The actual bytes live in the bundle at ``path``; this is just the manifest row.
    """

    camera: str
    kind: str
    path: str
    format: str
    frame_count: int | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-native manifest row for this asset."""
        out: dict[str, Any] = {
            "camera": self.camera,
            "kind": self.kind,
            "path": self.path,
            "format": self.format,
        }
        if self.frame_count is not None:
            out["frameCount"] = self.frame_count
        if self.width is not None:
            out["width"] = self.width
        if self.height is not None:
            out["height"] = self.height
        if self.fps is not None:
            out["fps"] = self.fps
        return out


@dataclass(frozen=True)
class EpisodeRecord:
    """One episode's outcome plus any media captured for it.

    Fields mirror :class:`rlmesh.EpisodeResult`. ``included_in_metrics`` (default
    ``True``) lets a caller mark warmup/discarded episodes so aggregates and the
    dashboard skip them -- it maps to the platform's ``includedInMetrics``.
    """

    index: int
    seed: int | None
    steps: int
    reward: float
    terminated: bool
    truncated: bool
    success: bool | None = None
    duration_s: float = 0.0
    included_in_metrics: bool = True
    media: tuple[MediaRef, ...] = ()

    @classmethod
    def from_result(
        cls,
        result: EpisodeResult,
        *,
        index: int | None = None,
        included_in_metrics: bool = True,
        media: tuple[MediaRef, ...] = (),
    ) -> EpisodeRecord:
        """Build a record from a :class:`~rlmesh.EpisodeResult` (drops op-latency).

        ``index`` overrides the per-run episode index with a workload-stable one, so
        episodes from several ``run`` calls into one workload stay uniquely keyed
        (the platform dedupes on ``resultSetId`` + task + index). Defaults to the
        result's own index.
        """
        return cls(
            index=result.index if index is None else index,
            seed=result.seed,
            steps=result.steps,
            reward=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
            success=result.success,
            duration_s=result.duration_s,
            included_in_metrics=included_in_metrics,
            media=media,
        )

    @property
    def succeeded(self) -> bool:
        """Env-reported success, falling back to ``terminated`` when unreported.

        Matches :attr:`rlmesh.RunResult.success_rate` semantics so an SDK metric and
        an uploaded metric agree.
        """
        return self.terminated if self.success is None else self.success

    def to_dict(self) -> dict[str, Any]:
        """JSON-native episode record (SDK vocabulary; media only when present)."""
        out: dict[str, Any] = {
            "index": self.index,
            "seed": self.seed,
            "steps": self.steps,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "success": self.success,
            "durationS": self.duration_s,
            "includedInMetrics": self.included_in_metrics,
        }
        if self.media:
            out["media"] = [m.to_dict() for m in self.media]
        return out


@dataclass
class WorkloadRecord:
    """One model x env x task cell: the episodes from one or more ``run`` calls.

    ``task`` is the natural workload key the dashboard groups by; it defaults to
    ``env`` when the caller does not distinguish tasks within an env. ``config`` is
    free-form caller metadata (seeds, horizon, notes) carried verbatim.
    """

    model: str
    env: str
    task: str
    config: dict[str, Any] = field(default_factory=dict)
    episodes: list[EpisodeRecord] = field(default_factory=list)

    def _metric_episodes(self) -> list[EpisodeRecord]:
        return [e for e in self.episodes if e.included_in_metrics]

    @property
    def mean_reward(self) -> float:
        """Mean reward over metric-counted episodes (``0.0`` when none)."""
        counted = self._metric_episodes()
        if not counted:
            return 0.0
        return sum(e.reward for e in counted) / len(counted)

    @property
    def success_rate(self) -> float:
        """Fraction of metric-counted episodes that succeeded (``0.0`` when none)."""
        counted = self._metric_episodes()
        if not counted:
            return 0.0
        return sum(1 for e in counted if e.succeeded) / len(counted)

    @property
    def total_steps(self) -> int:
        """Total env steps across all episodes (counted or not)."""
        return sum(e.steps for e in self.episodes)

    def to_dict(self) -> dict[str, Any]:
        """JSON-native workload record with computed metrics and its episodes."""
        return {
            "model": self.model,
            "env": self.env,
            "task": self.task,
            "config": self.config,
            "numEpisodes": len(self.episodes),
            "metrics": {
                "meanReward": self.mean_reward,
                "successRate": self.success_rate,
                "totalSteps": self.total_steps,
            },
            "episodes": [e.to_dict() for e in self.episodes],
        }


@dataclass
class ResultSet:
    """The whole exported bundle: an id plus every recorded workload.

    ``result_set_id`` is a locally minted anchor so a re-upload of the same bundle
    is idempotent on the platform (the mapper derives deterministic episode ids from
    it). ``recorded_at`` is stamped at export time, not construction.
    """

    result_set_id: str
    workloads: list[WorkloadRecord] = field(default_factory=list)

    def to_dict(self, *, recorded_at: str) -> dict[str, Any]:
        """The full ``rlmesh.result.v1`` document as a JSON-native dict."""
        return {
            "schema": SCHEMA,
            "resultSetId": self.result_set_id,
            "recordedAt": recorded_at,
            "workloads": [w.to_dict() for w in self.workloads],
        }
