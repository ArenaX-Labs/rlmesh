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
    """An excluded episode is still recorded, but not counted in aggregates."""
    rec = Recorder()
    rec.add(
        _run(_episode(0, reward=10.0, success=True)),
        model="m",
        env="e",
        task="warmup",
        included_in_metrics=False,
    )
    wl = rec.workloads[0]
    assert wl.total_steps == 3
    assert wl.mean_reward == 0.0 and wl.success_rate == 0.0
    assert wl.episodes[0].included_in_metrics is False


def test_success_rate_falls_back_to_terminated_when_unreported() -> None:
    """success=None falls back to terminated; terminated False -> not a success."""
    rec = Recorder()
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
    """Re-exporting a media-less bundle over a folder drops the earlier media/."""
    np = pytest.importorskip("numpy")
    dest = tmp_path / "bundle"
    img = np.full((32, 32, 3), 100, dtype=np.uint8)

    rec1 = Recorder()
    hooks = rec1.capture(model="m", env="e", task="t", cameras=["cam0"])
    hooks.on_episode_start(episode=0, seed=0)
    hooks.on_step(_step_event(read=lambda _item: img))
    hooks.on_episode_end(_episode(0, reward=1.0, success=True))
    rec1.export(dest)
    assert (dest / rec1.workloads[0].episodes[0].media[0].path).is_file()
    rec1.close()

    rec2 = Recorder()
    rec2.add(_run(_episode(0, reward=1.0, success=True)), model="m", env="e", task="t")
    rec2.export(dest)
    assert not (dest / "media").exists()
    assert (dest / "result.json").is_file()


def test_capture_records_av1_mp4(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    rec = Recorder(fps=10)
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    hooks = rec.capture(model="m", env="e", task="t", cameras=["cam0"])

    hooks.on_episode_start(episode=0, seed=7)
    for _ in range(5):
        hooks.on_step(_step_event(read=lambda _item: img))
    hooks.on_episode_end(_episode(0, reward=1.0, success=True))

    ref = rec.workloads[0].episodes[0].media[0]
    assert ref.kind == "video" and ref.format == "mp4"
    assert ref.frame_count == 5 and ref.fps == 10.0 and ref.camera == "cam0"
    assert ref.width == 32 and ref.height == 32

    out = rec.export(tmp_path / "bundle")
    mp4 = out / ref.path
    assert mp4.is_file() and mp4.read_bytes()[4:8] == b"ftyp"
    rec.close()


def test_quality_controls_file_size(tmp_path: Path) -> None:
    """Lower quality (higher quantizer) yields a smaller file for the same footage."""
    np = pytest.importorskip("numpy")
    frames = []
    for t in range(12):
        col = ((np.arange(128) + t) % 256).astype(np.uint8)
        frames.append(np.repeat(np.tile(col, (128, 1))[:, :, None], 3, axis=2))

    def recorded_size(quality: int) -> int:
        rec = Recorder(quality=quality)
        hooks = rec.capture(model="m", env="e", task="t", cameras=["cam"])
        hooks.on_episode_start(episode=0, seed=0)
        for f in frames:
            hooks.on_step(_step_event(read=lambda _i, _f=f: _f))
        hooks.on_episode_end(_episode(0, reward=1.0, success=True))
        out = rec.export(tmp_path / f"b{quality}")
        size = (out / rec.workloads[0].episodes[0].media[0].path).stat().st_size
        rec.close()
        return size

    assert recorded_size(5) < recorded_size(95)


def test_capture_multi_camera_records_each(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    rec = Recorder()
    img = np.full((32, 32, 3), 64, dtype=np.uint8)
    hooks = rec.capture(model="m", env="e", task="t", cameras=["front", "wrist"])

    hooks.on_episode_start(episode=0, seed=0)
    hooks.on_step(_step_event(read=lambda _item: img))
    hooks.on_episode_end(_episode(0, reward=1.0, success=True))

    media = rec.workloads[0].episodes[0].media
    assert sorted(m.camera for m in media) == ["front", "wrist"]
    assert len({m.path for m in media}) == 2
    rec.close()


def test_capture_records_render_source(tmp_path: Path) -> None:
    """An env with an rgb render mode but no image obs roles still records render()."""
    np = pytest.importorskip("numpy")
    img = np.full((48, 64, 3), 120, dtype=np.uint8)

    class _Client:
        render_mode = "rgb_array"

        def render(self) -> object:
            return img

    class _Sess:
        _client = _Client()
        _contract = None

    rec = Recorder(fps=20)
    session: Any = _Sess()
    hooks = rec.capture(model="m", env="e", task="t", session=session)
    hooks.on_episode_start(episode=0, seed=0)
    for _ in range(4):
        hooks.on_step(_step_event())
    hooks.on_episode_end(_episode(0, reward=1.0, success=True))

    media = rec.workloads[0].episodes[0].media
    assert [m.camera for m in media] == ["render"]
    assert media[0].frame_count == 4
    assert (media[0].width, media[0].height) == (64, 48)
    out = rec.export(tmp_path / "bundle")
    assert (out / media[0].path).read_bytes()[4:8] == b"ftyp"
    rec.close()


def test_media_paths_unique_across_workloads(tmp_path: Path) -> None:
    """Two models on the same task must not overwrite each other's media."""
    np = pytest.importorskip("numpy")
    rec = Recorder()
    img = np.full((32, 32, 3), 200, dtype=np.uint8)
    paths = []
    for model in ("a", "b"):
        hooks = rec.capture(model=model, env="e", task="t", cameras=["cam0"])
        hooks.on_episode_start(episode=0, seed=0)
        hooks.on_step(_step_event(read=lambda _item: img))
        hooks.on_episode_end(_episode(0, reward=1.0, success=True))
        paths.append(rec.workloads[-1].episodes[0].media[0].path)

    assert paths[0] != paths[1]
    out = rec.export(tmp_path / "bundle")
    assert all((out / p).is_file() for p in paths)
    rec.close()


def test_colliding_camera_names_get_distinct_paths(tmp_path: Path) -> None:
    """Two roles that sanitize to the same filename must not share a media path."""
    np = pytest.importorskip("numpy")
    rec = Recorder()
    img = np.full((32, 32, 3), 90, dtype=np.uint8)
    hooks = rec.capture(model="m", env="e", task="t", cameras=["cam/left", "cam left"])

    hooks.on_episode_start(episode=0, seed=0)
    hooks.on_step(_step_event(read=lambda _item: img))
    hooks.on_episode_end(_episode(0, reward=1.0, success=True))

    media = rec.workloads[0].episodes[0].media
    assert len({m.path for m in media}) == 2
    out = rec.export(tmp_path / "bundle")
    assert all((out / m.path).is_file() for m in media)
    rec.close()


def test_env_video_eager_copy_survives_overwrite(tmp_path: Path) -> None:
    """An env reusing one output path per episode must not clobber earlier ones."""
    src = tmp_path / "rollout.mp4"
    rec = Recorder()
    hooks = rec.capture(model="m", env="e", task="t")

    src.write_bytes(b"AAAA-ep0")
    hooks.on_episode_start(episode=0, seed=0)
    hooks.on_step(_step_event(info={"video_artifact_path": str(src)}))
    hooks.on_episode_end(_episode(0, reward=1.0, success=True))

    src.write_bytes(b"BBBB-ep1")
    hooks.on_episode_start(episode=1, seed=1)
    hooks.on_step(_step_event(info={"video_artifact_path": str(src)}))
    hooks.on_episode_end(_episode(0, reward=1.0, success=True))

    eps = rec.workloads[0].episodes
    assert eps[0].media[0].format == "mp4" and eps[0].media[0].kind == "video"
    out = rec.export(tmp_path / "bundle")
    assert (out / eps[0].media[0].path).read_bytes() == b"AAAA-ep0"
    assert (out / eps[1].media[0].path).read_bytes() == b"BBBB-ep1"
    rec.close()


def test_capture_defers_camera_discovery_to_first_step() -> None:
    """capture(session=) must not read the contract before the session connects."""
    rec = Recorder()

    class _Spy:
        reads = 0

        @property
        def _contract(self) -> None:
            type(self).reads += 1
            return None

    spy: Any = _Spy()
    hooks = rec.capture(model="m", env="e", task="t", session=spy)
    assert _Spy.reads == 0

    hooks.on_episode_start(episode=0, seed=0)
    with pytest.warns(UserWarning, match="no render.* or image roles"):
        hooks.on_step(_step_event())
    assert _Spy.reads >= 1


def test_explicit_empty_cameras_skips_discovery() -> None:
    """cameras=[] is an explicit opt-out, distinct from cameras=None (auto-discover)."""
    rec = Recorder()

    class _Spy:
        reads = 0

        @property
        def _contract(self) -> None:
            type(self).reads += 1
            return None

    spy: Any = _Spy()
    hooks = rec.capture(model="m", env="e", task="t", cameras=[], session=spy)
    hooks.on_episode_start(episode=0, seed=0)
    hooks.on_step(_step_event())
    hooks.on_episode_end(_episode(0, reward=1.0, success=True))

    assert _Spy.reads == 0
    assert rec.workloads[0].episodes[0].media == ()


def test_transient_read_error_skips_frame_keeps_camera() -> None:
    """A one-off read failure drops that frame only; the camera keeps recording."""
    np = pytest.importorskip("numpy")
    rec = Recorder()
    img = np.full((32, 32, 3), 7, dtype=np.uint8)
    calls = {"n": 0}

    def read(_item: object) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient read blip")
        return img

    hooks = rec.capture(model="m", env="e", task="t", cameras=["cam0"])
    hooks.on_episode_start(episode=0, seed=0)
    for _ in range(3):
        hooks.on_step(_step_event(read=read))
    hooks.on_episode_end(_episode(0, reward=1.0, success=True))

    ref = rec.workloads[0].episodes[0].media[0]
    assert ref.camera == "cam0" and ref.frame_count == 2
    rec.close()


def test_capture_hook_guards_encode_errors() -> None:
    """A staging/encode failure warns and drops the camera, never aborts the run."""
    np = pytest.importorskip("numpy")
    rec = Recorder()
    img = np.full((32, 32, 3), 10, dtype=np.uint8)
    hooks = rec.capture(model="m", env="e", task="t", cameras=["cam0"])

    def boom(**_: Any) -> Any:
        raise RuntimeError("no encoder")

    rec._stager.open_video = boom  # type: ignore[method-assign]

    hooks.on_episode_start(episode=0, seed=0)
    with pytest.warns(UserWarning, match="cam0"):
        hooks.on_step(_step_event(read=lambda _item: img))
    hooks.on_episode_end(_episode(0, reward=2.0, success=True))

    ep = rec.workloads[0].episodes[0]
    assert ep.media == () and ep.reward == 2.0
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
    """Episodes all built with per-run index 0 are reindexed 0,1,2 in the workload."""
    rec = Recorder()
    hooks = rec.capture(model="m", env="e", task="t")
    for i in range(3):
        hooks.on_episode_start(episode=i, seed=i)
        hooks.on_episode_end(_episode(0, reward=float(i), success=True))

    indices = [e.index for e in rec.workloads[0].episodes]
    assert indices == [0, 1, 2]


def test_context_manager_cleans_staging(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    img = np.full((32, 32, 3), 50, dtype=np.uint8)
    with Recorder() as rec:
        hooks = rec.capture(model="m", env="e", task="t", cameras=["cam0"])
        hooks.on_episode_start(episode=0, seed=0)
        hooks.on_step(_step_event(read=lambda _item: img))
        hooks.on_episode_end(_episode(0, reward=1.0, success=True))
        staging = rec._stager._dir
        rec.export(tmp_path / "bundle")
    assert staging is None or not Path(staging).exists()
