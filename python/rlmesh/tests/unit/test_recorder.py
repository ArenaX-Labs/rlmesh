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
from types import SimpleNamespace
from typing import Any

import pytest
import rlmesh
import rlmesh.adapters as adapt
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


def test_recorder_validates_fps() -> None:
    with pytest.raises(ValueError, match="fps"):
        Recorder(fps=0)
    Recorder(fps=1)


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


def test_capture_smoke_with_real_session_run(tmp_path: Path) -> None:
    """Recorder.capture works through a real session().run hook path.

    Deliberately omits ``session=``: the hooks must adopt the running session
    via ``RunHooks.on_run_start`` and auto-discover the image role from it.
    """
    np = pytest.importorskip("numpy")
    gym = pytest.importorskip("gymnasium")

    class _ImageEnv:
        def __init__(self) -> None:
            self.metadata: dict[str, object] = {}
            self.observation_space = gym.spaces.Dict(
                {"cam": gym.spaces.Box(0, 255, shape=(16, 16, 3), dtype=np.uint8)}
            )
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
            self._step = 0

        def _obs(self) -> dict[str, object]:
            return {"cam": np.full((16, 16, 3), 40 + self._step * 20, dtype=np.uint8)}

        def reset(
            self, *, seed: object = None, options: object = None
        ) -> tuple[dict[str, object], dict[str, object]]:
            self._step = 0
            return self._obs(), {"seed": seed}

        def step(
            self, action: object
        ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
            self._step += 1
            return self._obs(), 1.0, self._step >= 2, False, {"action": action}

        def close(self) -> None:
            pass

    tags = adapt.EnvTags(
        observation={"cam": adapt.ImageTag(role=adapt.IMAGE_PRIMARY)},
        action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )
    env = adapt.tag(_ImageEnv(), tags)

    rec = Recorder(fps=12)
    with rlmesh.session(rlmesh.RANDOM_SAMPLE, env) as sess:
        result = sess.run(
            seeds=[123],
            hooks=rec.capture(model="random", env="image-env", task="smoke"),
        )

    assert result.num_episodes == 1
    assert result.episodes[0].steps == 2
    media = rec.workloads[0].episodes[0].media
    assert len(media) == 1
    ref = media[0]
    assert ref.camera == adapt.IMAGE_PRIMARY
    assert ref.frame_count == 2 and ref.fps == 12.0
    assert (ref.width, ref.height) == (16, 16)
    out = rec.export(tmp_path / "bundle")
    assert (out / ref.path).read_bytes()[4:8] == b"ftyp"
    rec.close()


def test_auto_discovery_records_image_role_named_render() -> None:
    """A role named ``render`` must not be hidden by the env render() source."""
    np = pytest.importorskip("numpy")
    render_img = np.full((32, 32, 3), 200, dtype=np.uint8)
    role_img = np.full((32, 32, 3), 50, dtype=np.uint8)
    reads: list[str] = []
    tags = adapt.EnvTags(
        observation={"rgb": adapt.ImageTag(role="render")},
        action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )

    class _Client:
        render_mode = "rgb_array"

        def render(self) -> object:
            return render_img

    session: Any = SimpleNamespace(
        _client=_Client(),
        _contract=SimpleNamespace(metadata=tags.to_metadata()),
    )

    def read(item: object) -> object:
        role = getattr(item, "role", None)
        assert role == "render"
        reads.append(role)
        return role_img

    rec = Recorder()
    hooks = rec.capture(model="m", env="e", task="t", session=session)
    hooks.on_episode_start(episode=0, seed=0)
    hooks.on_step(_step_event(read=read))
    hooks.on_episode_end(_episode(0, reward=1.0, success=True))

    media = rec.workloads[0].episodes[0].media
    assert [m.camera for m in media] == ["render()", "render"]
    assert len({m.path for m in media}) == 2
    assert reads == ["render"]
    rec.close()


def test_auto_discovery_reads_render_role_without_render_source() -> None:
    """A ``render`` image role is recordable when there is no env render() source."""
    np = pytest.importorskip("numpy")
    role_img = np.full((32, 32, 3), 75, dtype=np.uint8)
    reads: list[str] = []
    tags = adapt.EnvTags(
        observation={"rgb": adapt.ImageTag(role="render")},
        action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
    )
    session: Any = SimpleNamespace(
        _client=SimpleNamespace(render_mode=None),
        _contract=SimpleNamespace(metadata=tags.to_metadata()),
    )

    def read(item: object) -> object:
        role = getattr(item, "role", None)
        assert role == "render"
        reads.append(role)
        return role_img

    rec = Recorder()
    hooks = rec.capture(model="m", env="e", task="t", session=session)
    hooks.on_episode_start(episode=0, seed=0)
    hooks.on_step(_step_event(read=read))
    hooks.on_episode_end(_episode(0, reward=1.0, success=True))

    media = rec.workloads[0].episodes[0].media
    assert [m.camera for m in media] == ["render"]
    assert reads == ["render"]
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


def test_recorder_validates_quality_and_integer_fps() -> None:
    """Bad fps/quality fail at construction, not as a swallowed first-frame error."""
    with pytest.raises(ValueError, match="quality"):
        _ = Recorder(quality=300)
    with pytest.raises(ValueError, match="quality"):
        _ = Recorder(quality=0)
    with pytest.raises(ValueError, match="fps"):
        _ = Recorder(fps=29.97)  # pyright: ignore[reportArgumentType]
    _ = Recorder(quality=1)
    _ = Recorder(quality=100)


def test_export_rejects_unknown_archive_string(tmp_path: Path) -> None:
    rec = Recorder()
    rec.add(_run(_episode(0, reward=1.0, success=True)), model="m", env="e", task="t")
    with pytest.raises(ValueError, match="archive"):
        rec.export(tmp_path / "bundle", archive="tar")


def test_export_refuses_to_clobber_foreign_media_dir(tmp_path: Path) -> None:
    """Folder export into a non-bundle directory must not delete its media/."""
    dest = tmp_path / "project"
    keep = dest / "media" / "precious.txt"
    keep.parent.mkdir(parents=True)
    keep.write_text("mine")

    rec = Recorder()
    rec.add(_run(_episode(0, reward=1.0, success=True)), model="m", env="e", task="t")
    with pytest.raises(FileExistsError, match="not an rlmesh bundle"):
        rec.export(dest)
    assert keep.read_text() == "mine"


def test_export_folder_over_zip_file_errors(tmp_path: Path) -> None:
    rec = Recorder()
    rec.add(_run(_episode(0, reward=1.0, success=True)), model="m", env="e", task="t")
    out = rec.export(tmp_path / "bundle", archive="zip")
    with pytest.raises(FileExistsError, match="not a directory"):
        rec.export(out, archive="folder")


def test_export_after_close_with_media_errors(tmp_path: Path) -> None:
    """close() removes staged media; a later export must fail, not ship a bundle
    whose manifest references files absent from the archive."""
    np = pytest.importorskip("numpy")
    img = np.full((32, 32, 3), 50, dtype=np.uint8)
    rec = Recorder()
    hooks = rec.capture(model="m", env="e", task="t", cameras=["cam0"])
    hooks.on_episode_start(episode=0, seed=0)
    hooks.on_step(_step_event(read=lambda _item: img))
    hooks.on_episode_end(_episode(0, reward=1.0, success=True))
    rec.export(tmp_path / "a.zip")
    rec.close()
    with pytest.raises(RuntimeError, match="closed"):
        rec.export(tmp_path / "b.zip")


def test_export_after_close_without_media_is_fine(tmp_path: Path) -> None:
    rec = Recorder()
    rec.add(_run(_episode(0, reward=1.0, success=True)), model="m", env="e", task="t")
    rec.close()
    assert rec.export(tmp_path / "bundle.zip").is_file()


def test_zip_export_stores_media_uncompressed(tmp_path: Path) -> None:
    """Already-AV1-compressed mp4s are STORED; result.json stays deflated."""
    np = pytest.importorskip("numpy")
    img = np.full((32, 32, 3), 90, dtype=np.uint8)
    rec = Recorder()
    hooks = rec.capture(model="m", env="e", task="t", cameras=["cam0"])
    hooks.on_episode_start(episode=0, seed=0)
    hooks.on_step(_step_event(read=lambda _item: img))
    hooks.on_episode_end(_episode(0, reward=1.0, success=True))
    out = rec.export(tmp_path / "bundle.zip")
    with zipfile.ZipFile(out) as zf:
        by_name = {info.filename: info.compress_type for info in zf.infolist()}
    assert by_name["result.json"] == zipfile.ZIP_DEFLATED
    mp4s = [ct for name, ct in by_name.items() if name.endswith(".mp4")]
    assert mp4s == [zipfile.ZIP_STORED]
    rec.close()


def test_explicit_render_camera_prefers_declared_role() -> None:
    """cameras=["render"] must reach an image role literally named render."""
    np = pytest.importorskip("numpy")
    role_img = np.full((16, 16, 3), 30, dtype=np.uint8)
    reads: list[str] = []

    def read(item: Any) -> object:
        reads.append(item.role)
        return role_img

    rec = Recorder()
    hooks = rec.capture(model="m", env="e", task="t", cameras=["render"])
    hooks.on_episode_start(episode=0, seed=0)
    hooks.on_step(_step_event(read=read))
    hooks.on_episode_end(_episode(0, reward=1.0, success=True))

    assert reads == ["render"]
    media = rec.workloads[0].episodes[0].media
    assert [m.camera for m in media] == ["render"] and media[0].frame_count == 1
    rec.close()


def test_explicit_render_camera_falls_back_to_render_source() -> None:
    """cameras=["render"] with no such role still records the env render()."""
    np = pytest.importorskip("numpy")
    img = np.full((16, 16, 3), 60, dtype=np.uint8)

    class _Client:
        render_mode = "rgb_array"

        def render(self) -> object:
            return img

    class _Sess:
        _client = _Client()
        _contract = None

    session: Any = _Sess()
    rec = Recorder()
    hooks = rec.capture(
        model="m", env="e", task="t", cameras=["render"], session=session
    )
    hooks.on_episode_start(episode=0, seed=0)
    hooks.on_step(_step_event())
    hooks.on_episode_end(_episode(0, reward=1.0, success=True))

    media = rec.workloads[0].episodes[0].media
    assert [m.camera for m in media] == ["render"] and media[0].frame_count == 1
    rec.close()


def test_unreadable_env_video_path_warns(tmp_path: Path) -> None:
    """A video_url that is not a local file is skipped loudly, not silently."""
    rec = Recorder()
    hooks = rec.capture(model="m", env="e", task="t")
    hooks.on_episode_start(episode=0, seed=0)
    hooks.on_step(_step_event(info={"video_url": "https://bucket/rollout.mp4"}))
    with pytest.warns(UserWarning, match="not a readable local file"):
        hooks.on_episode_end(_episode(0, reward=1.0, success=True))
    assert rec.workloads[0].episodes[0].media == ()


def test_on_run_start_prefers_explicit_session() -> None:
    """A session passed to capture(session=...) wins over the running one."""
    from rlmesh.recorder.hooks import CaptureHooks

    rec = Recorder()
    explicit: Any = SimpleNamespace(_client=None, _contract=None)
    other: Any = SimpleNamespace(_client=None, _contract=None)
    hooks = rec.capture(model="m", env="e", task="t", session=explicit)
    assert isinstance(hooks, CaptureHooks)
    hooks.on_run_start(other)
    assert hooks._session is explicit  # pyright: ignore[reportPrivateUsage]

    adopted = rec.capture(model="m", env="e", task="t")
    assert isinstance(adopted, CaptureHooks)
    adopted.on_run_start(other)
    assert adopted._session is other  # pyright: ignore[reportPrivateUsage]


def test_capture_warnings_never_raise_under_warnings_as_errors() -> None:
    """The capture contract holds under -W error: degradation never aborts the run."""
    import warnings as _warnings

    np = pytest.importorskip("numpy")
    img = np.full((32, 32, 3), 10, dtype=np.uint8)
    rec = Recorder()
    hooks = rec.capture(model="m", env="e", task="t", cameras=["cam0"])

    def boom(**_: Any) -> Any:
        raise RuntimeError("no encoder")

    rec._stager.open_video = boom  # type: ignore[method-assign]
    hooks.on_episode_start(episode=0, seed=0)
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        hooks.on_step(_step_event(read=lambda _item: img))
        hooks.on_episode_end(_episode(0, reward=2.0, success=True))
    ep = rec.workloads[0].episodes[0]
    assert ep.media == () and ep.reward == 2.0
    rec.close()
