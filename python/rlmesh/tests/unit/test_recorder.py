"""Recorder: capture eval runs and export an ``rlmesh.result.v1`` bundle.

Pins the OSS-native record shape and the two capture modes -- post-hoc ``add`` from
a :class:`~rlmesh.RunResult` (path-agnostic) and live ``capture`` hooks (media on the
session path) -- plus folder/zip export. The capture hooks are driven directly with
fabricated :class:`~rlmesh.StepEvent` / :class:`~rlmesh.EpisodeResult` values, the
same objects ``Session.run`` feeds them, so the tests need no env.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
import rlmesh
from rlmesh import EpisodeResult, Recorder, RunResult, StepEvent
from rlmesh.recorder import SCHEMA


def _episode(
    index: int, *, reward: float, success: bool | None, steps: int = 3
) -> EpisodeResult:
    return EpisodeResult(
        index=index,
        seed=100 + index,
        steps=steps,
        reward=reward,
        terminated=success is True,
        truncated=success is not True,
        success=success,
        duration_s=0.5,
    )


def _run(*episodes: EpisodeResult) -> RunResult:
    return RunResult(episodes=tuple(episodes))


def _step_event(info: dict[str, Any] | None = None, read: Any = None) -> StepEvent:
    return StepEvent(
        episode=0,
        seed=7,
        step=0,
        observation=object(),
        action=object(),
        reward=1.0,
        terminated=False,
        truncated=False,
        info=info or {},
        predict_ms=0.0,
        step_ms=0.0,
        read=read if read is not None else (lambda _item: None),
    )


def test_recorder_exports_recorder_symbol() -> None:
    assert rlmesh.Recorder is Recorder


def test_add_records_schema_and_metrics() -> None:
    rec = Recorder(result_set_id="fixed-id")
    rec.add(
        _run(
            _episode(0, reward=2.0, success=True),
            _episode(1, reward=0.0, success=False),
        ),
        model="smolvla",
        env="libero",
        task="libero-spatial-0",
        config={"horizon": 8},
    )
    doc = rec.to_dict(recorded_at="2026-07-07T00:00:00+00:00")

    assert doc["schema"] == SCHEMA
    assert doc["resultSetId"] == "fixed-id"
    assert doc["recordedAt"] == "2026-07-07T00:00:00+00:00"
    assert len(doc["workloads"]) == 1
    wl = doc["workloads"][0]
    assert (wl["model"], wl["env"], wl["task"]) == (
        "smolvla",
        "libero",
        "libero-spatial-0",
    )
    assert wl["config"] == {"horizon": 8}
    assert wl["numEpisodes"] == 2
    assert wl["metrics"]["meanReward"] == 1.0
    assert wl["metrics"]["successRate"] == 0.5
    assert wl["metrics"]["totalSteps"] == 6
    # SDK vocabulary, not the platform's camelCase episode.v1 names.
    ep = wl["episodes"][0]
    assert (
        ep["reward"] == 2.0
        and ep["durationS"] == 0.5
        and ep["includedInMetrics"] is True
    )
    assert "cumulativeReward" not in ep and "media" not in ep


def test_task_defaults_to_env() -> None:
    rec = Recorder()
    wl = rec.add(_run(_episode(0, reward=1.0, success=True)), model="m", env="cartpole")
    assert wl.task == "cartpole"


def test_included_in_metrics_excludes_from_aggregates() -> None:
    rec = Recorder()
    rec.add(
        _run(_episode(0, reward=10.0, success=True)),
        model="m",
        env="e",
        task="warmup",
        included_in_metrics=False,
    )
    wl = rec.workloads[0]
    # Excluded episode still recorded, but not counted.
    assert wl.total_steps == 3
    assert wl.mean_reward == 0.0 and wl.success_rate == 0.0
    assert wl.episodes[0].included_in_metrics is False


def test_success_rate_falls_back_to_terminated_when_unreported() -> None:
    rec = Recorder()
    # success=None -> use terminated; here terminated is False -> not a success.
    rec.add(_run(_episode(0, reward=1.0, success=None)), model="m", env="e")
    assert rec.workloads[0].success_rate == 0.0


def test_export_folder(tmp_path: Path) -> None:
    rec = Recorder(result_set_id="rs")
    rec.add(_run(_episode(0, reward=1.0, success=True)), model="m", env="e", task="t")
    out = rec.export(tmp_path / "bundle")

    assert out == tmp_path / "bundle"
    doc = json.loads((out / "result.json").read_text())
    assert doc["resultSetId"] == "rs"
    assert doc["workloads"][0]["episodes"][0]["reward"] == 1.0


def test_export_zip_by_suffix(tmp_path: Path) -> None:
    rec = Recorder()
    rec.add(_run(_episode(0, reward=1.0, success=True)), model="m", env="e", task="t")
    out = rec.export(tmp_path / "bundle.zip")

    assert out.suffix == ".zip"
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "result.json" in names
        doc = json.loads(zf.read("result.json"))
    assert doc["workloads"][0]["numEpisodes"] == 1


def test_export_folder_clears_stale_media(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    dest = tmp_path / "bundle"

    # First export captures a frame stack -> media/.../*.npz on disk.
    rec1 = Recorder()
    hooks = rec1.capture(
        model="m",
        env="e",
        task="t",
        frame_fn=lambda _e: np.zeros((4, 4, 3), dtype=np.uint8),
    )
    hooks.on_episode_start(episode=0, seed=0)
    hooks.on_step(_step_event())
    hooks.on_episode_end(_episode(0, reward=1.0, success=True))
    rec1.export(dest)
    assert (dest / rec1.workloads[0].episodes[0].media[0].path).is_file()
    rec1.close()

    # Re-exporting a media-less bundle to the same folder drops the stale media/.
    rec2 = Recorder()
    rec2.add(_run(_episode(0, reward=1.0, success=True)), model="m", env="e", task="t")
    rec2.export(dest)
    assert not (dest / "media").exists()
    assert (dest / "result.json").is_file()


def test_capture_env_video_path_is_copied(tmp_path: Path) -> None:
    # An env that renders its own video (case 1): it leaves the path in step info.
    video = tmp_path / "ep.mp4"
    video.write_bytes(b"\x00\x01fake-mp4")
    rec = Recorder()
    hooks = rec.capture(model="m", env="e", task="t")

    hooks.on_episode_start(episode=0, seed=7)
    hooks.on_step(_step_event(info={"video_artifact_path": str(video)}))
    hooks.on_episode_end(_episode(0, reward=1.0, success=True))

    ep = rec.workloads[0].episodes[0]
    assert len(ep.media) == 1
    ref = ep.media[0]
    assert ref.kind == "video" and ref.format == "mp4"

    out = rec.export(tmp_path / "bundle")
    copied = out / ref.path
    assert copied.is_file() and copied.read_bytes() == b"\x00\x01fake-mp4"
    rec.close()


def test_capture_frame_fn_stages_npz(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    rec = Recorder()
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    hooks = rec.capture(model="m", env="e", task="t", frame_fn=lambda _e: frame)

    hooks.on_episode_start(episode=0, seed=7)
    hooks.on_step(_step_event())
    hooks.on_step(_step_event())
    hooks.on_episode_end(_episode(0, reward=1.0, success=True))

    ref = rec.workloads[0].episodes[0].media[0]
    assert ref.kind == "frames" and ref.format == "npz"
    assert ref.frame_count == 2 and ref.height == 4 and ref.width == 5

    out = rec.export(tmp_path / "bundle")
    npz = out / ref.path
    assert npz.is_file()
    with np.load(npz) as data:
        assert data["frames"].shape == (2, 4, 5, 3)
    rec.close()


def test_capture_video_encodes_mp4(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    from rlmesh.recorder.encode import ffmpeg_available

    if not ffmpeg_available():
        pytest.skip("ffmpeg not available (install rlmesh[recorder])")
    rec = Recorder(video=True, fps=10)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    hooks = rec.capture(model="m", env="e", task="t", frame_fn=lambda _e: frame)
    hooks.on_episode_start(episode=0, seed=0)
    for _ in range(6):
        hooks.on_step(_step_event())
    hooks.on_episode_end(_episode(0, reward=1.0, success=True))

    ref = rec.workloads[0].episodes[0].media[0]
    assert ref.kind == "video" and ref.format == "mp4"
    assert ref.frame_count == 6 and ref.fps == 10.0
    out = rec.export(tmp_path / "bundle")
    mp4 = out / ref.path
    assert mp4.is_file() and mp4.stat().st_size > 0
    assert mp4.read_bytes()[4:8] == b"ftyp"  # mp4 container signature
    rec.close()


def test_capture_cameras_read_through_event(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    rec = Recorder()
    img = np.full((2, 2, 3), 128, dtype=np.uint8)
    hooks = rec.capture(model="m", env="e", task="t", cameras=["cam0"])

    # event.read(item) ignores the item here and returns the HWC frame.
    hooks.on_episode_start(episode=0, seed=7)
    hooks.on_step(_step_event(read=lambda _item: img))
    hooks.on_episode_end(_episode(0, reward=1.0, success=True))

    media = rec.workloads[0].episodes[0].media
    assert [m.camera for m in media] == ["cam0"]
    assert media[0].frame_count == 1
    rec.close()


def test_capture_metrics_only_when_no_media() -> None:
    rec = Recorder()
    hooks = rec.capture(model="m", env="e", task="t")
    hooks.on_episode_start(episode=0, seed=7)
    hooks.on_step(_step_event())
    hooks.on_episode_end(_episode(0, reward=3.0, success=True))

    ep = rec.workloads[0].episodes[0]
    assert ep.media == () and ep.reward == 3.0


def test_multiple_episodes_into_one_capture_are_uniquely_indexed() -> None:
    rec = Recorder()
    hooks = rec.capture(model="m", env="e", task="t")
    for i in range(3):
        hooks.on_episode_start(episode=i, seed=i)
        hooks.on_episode_end(
            _episode(0, reward=float(i), success=True)
        )  # per-run index all 0

    indices = [e.index for e in rec.workloads[0].episodes]
    assert indices == [0, 1, 2]


def test_context_manager_cleans_staging(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    with Recorder() as rec:
        hooks = rec.capture(
            model="m",
            env="e",
            task="t",
            frame_fn=lambda _e: np.zeros((2, 2, 3), dtype=np.uint8),
        )
        hooks.on_episode_start(episode=0, seed=0)
        hooks.on_step(_step_event())
        hooks.on_episode_end(_episode(0, reward=1.0, success=True))
        staging = rec._stager._dir
        rec.export(tmp_path / "bundle")
    assert staging is None or not Path(staging).exists()
