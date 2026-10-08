from __future__ import annotations

import io
import threading
import time

import numpy as np
import pytest
import rlmesh._rlmesh as native
from rlmesh._models import _view
from rlmesh._models._eval import Session
from rlmesh._models._view import (
    FrameSources,
    View,
    ViewerDriver,
    _resolve_render,
    _to_hwc_u8,
    resolve_view,
)


def _resolved(spec: object) -> View:
    view = resolve_view(spec)
    assert view is not None
    return view


def _u8(frame: object) -> tuple[bytes, int, int, int]:
    out = _to_hwc_u8(frame)
    assert out is not None
    return out


def test_resolve_view_shorthands() -> None:
    assert resolve_view(None) is None
    assert resolve_view(False) is None
    assert resolve_view(True) == View()
    assert _resolved("terminal").backend == "terminal"
    assert _resolved("both").backend == "both"
    http = _resolved("http")
    assert http.backend == "http" and http.port == 8008
    assert _resolved("http:9000").port == 9000


def test_resolve_view_bad_port_raises() -> None:
    with pytest.raises(ValueError):
        resolve_view("http:nine")


def test_resolve_view_unknown_backend_raises() -> None:
    with pytest.raises(ValueError):
        resolve_view("web")


def test_view_validates_backend_and_port() -> None:
    with pytest.raises(ValueError):
        View(backend="termnal")
    with pytest.raises(ValueError):
        View(backend="http", port=99999)
    with pytest.raises(ValueError):
        View(backend="http", port=0)


def test_view_validates_fps_format_and_quality() -> None:
    # Every documented field range is enforced, not just backend/port.
    with pytest.raises(ValueError, match="fps"):
        View(fps=0)
    with pytest.raises(ValueError, match="format"):
        View(format="gif")
    with pytest.raises(ValueError, match="quality"):
        View(quality=0)
    with pytest.raises(ValueError, match="quality"):
        View(quality=101)
    View(fps=1, format="png", quality=100)  # bounds are inclusive


def test_viewer_driver_feed_failure_warns_once_and_disables() -> None:
    # A per-step feed failure must not be swallowed silently forever: the first
    # failure warns and disables the viewer; later feeds are quiet no-ops.
    from rlmesh._models._view import ViewerDriver

    class _BoomPV:
        def wants_frame(self) -> bool:
            raise RuntimeError("boom-feed")

        def close(self) -> None:
            pass

    driver = ViewerDriver(View())
    driver._pv = _BoomPV()
    with pytest.warns(UserWarning, match="disabled after feed error: boom-feed"):
        driver.feed(
            contract=None,
            client=None,
            obs=None,
            read=lambda o, i: None,
            steps=1,
            reward=0.0,
            outcome="",
        )
    assert driver._pv is None
    driver.feed(  # disabled: no second warning, no crash
        contract=None,
        client=None,
        obs=None,
        read=lambda o, i: None,
        steps=2,
        reward=0.0,
        outcome="",
    )


def test_to_hwc_u8_uint8_passthrough() -> None:
    arr = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    data, h, w, c = _u8(arr)
    assert (h, w, c) == (2, 2, 3)
    assert data == arr.tobytes()


def test_to_hwc_u8_unit_float_scales() -> None:
    arr = np.zeros((1, 2, 3), dtype=np.float32)
    arr[0, 1, :] = 1.0
    data, h, w, c = _u8(arr)
    px = np.frombuffer(data, dtype=np.uint8).reshape(h, w, c)
    assert px[0, 0, 0] == 0
    assert px[0, 1, 0] == 255


def test_to_hwc_u8_signed_normalized_is_not_crushed() -> None:
    arr = np.array([[[-1.0, 0.0, 1.0]]], dtype=np.float32)
    data, h, w, c = _u8(arr)
    px = np.frombuffer(data, dtype=np.uint8).reshape(h, w, c)
    assert px[0, 0, 0] == 0
    assert 126 <= px[0, 0, 1] <= 129
    assert px[0, 0, 2] == 255


def test_to_hwc_u8_high_range_stretches() -> None:
    arr = np.array([[[0, 2000, 4000]]], dtype=np.float32)
    data, h, w, c = _u8(arr)
    px = np.frombuffer(data, dtype=np.uint8).reshape(h, w, c)
    assert px[0, 0, 0] == 0
    assert px[0, 0, 2] == 255
    assert 100 <= px[0, 0, 1] <= 160


def test_to_hwc_u8_rejects_non_image() -> None:
    assert _to_hwc_u8(np.zeros((4, 4))) is None
    assert _to_hwc_u8(np.zeros((2, 2, 5))) is None


def test_resolve_render_picks_convention_by_signature() -> None:
    class RemoteLike:
        def render(self, *, env_index: int) -> object:
            return ("remote", env_index)

    class GymLike:
        def render(self) -> object:
            return "gym"

    class NoRender:
        pass

    assert _resolve_render(RemoteLike())() == ("remote", 0)
    assert _resolve_render(GymLike())() == "gym"
    assert _resolve_render(NoRender())() is None


def test_view_outcome_prefers_info_over_terminated() -> None:
    sess: Session[object, object] = Session._create(  # pyright: ignore[reportPrivateUsage]
        env=object()
    )
    assert sess._view_outcome() == ""

    sess._terminated = True
    sess._last_info = {"is_success": False}
    assert sess._view_outcome() == "failure"

    sess._last_info = {"is_success": True}
    assert sess._view_outcome() == "success"

    sess._last_info = {}
    assert sess._view_outcome() == "done"  # terminal, outcome unknown: never "success"

    sess._terminated = False
    sess._truncated = True
    assert sess._view_outcome() == "timeout"


def test_pyviewer_api_smoke() -> None:
    from rlmesh._rlmesh import PyViewer

    pv = PyViewer(terminal=False, http_port=None, fps=30, format="jpeg", quality=75)
    pv.set_sources(["a", "b"], 0)
    assert pv.selected_source() == "a"
    pv.feed_hud(1, 0.5, "success")
    assert pv.should_quit() is False
    assert pv.warnings() == []
    pv.close()


def test_view_validates_hold_and_step_hz() -> None:
    View(hold=True)
    View(hold=2.5)
    View(hold=0)
    View(step_hz=20.0)
    with pytest.raises(ValueError, match="hold"):
        View(hold=-1.0)
    with pytest.raises(ValueError, match="hold"):
        View(hold=float("inf"))
    with pytest.raises(ValueError, match="step_hz"):
        View(step_hz=0)
    with pytest.raises(ValueError, match="step_hz"):
        View(step_hz=float("nan"))


def test_frame_sources_default_to_the_first_image_role() -> None:
    def render() -> object:
        return None

    both = FrameSources(
        roles=("wrist", "agentview"), render_label="render", render=render
    )
    assert both.labels == ("render", "wrist", "agentview")
    assert both.default_label == "wrist"  # read off the obs: no render() RPC
    only_render = FrameSources(roles=(), render_label="render", render=render)
    assert only_render.default_label == "render"
    assert FrameSources(roles=(), render_label=None, render=None).default_label is None


class _FakePV:
    """A PyViewer double: records draws and HUD outcomes, quits on request."""

    def __init__(self) -> None:
        self.sources: list[str] = []
        self.selected = 0
        self.frames: list[tuple[int, int]] = []
        self.outcomes: list[str] = []
        self.quit = False
        self.closed = False

    def warnings(self) -> list[str]:
        return []

    def set_sources(self, sources: list[str], default: int) -> None:
        self.sources, self.selected = sources, default

    def selected_source(self) -> str | None:
        return self.sources[self.selected] if self.sources else None

    def wants_frame(self) -> bool:
        return True

    def feed_frame(self, data: bytes, width: int, height: int, channels: int) -> None:
        self.frames.append((width, height))

    def feed_hud(self, steps: int, reward: float, outcome: str, **_: object) -> None:
        self.outcomes.append(outcome)

    def should_quit(self) -> bool:
        return self.quit

    def take_skip(self) -> bool:
        return False

    def close(self) -> None:
        self.closed = True


def _driver(
    monkeypatch: pytest.MonkeyPatch, view: View
) -> tuple[ViewerDriver, _FakePV]:
    """A driver over a fake viewer and one ``wrist`` role, fed one final step."""
    pv = _FakePV()
    monkeypatch.setattr(native, "PyViewer", lambda *a, **k: pv)
    monkeypatch.setattr(
        _view,
        "discover_frame_sources",
        lambda contract, client: FrameSources(
            roles=("wrist",),
            render_label="render",
            render=lambda: np.zeros((4, 6, 3), dtype=np.uint8),
        ),
    )
    driver = ViewerDriver(view)
    driver.feed(
        contract=None,
        client=None,
        obs=np.zeros((2, 3, 3), dtype=np.uint8),
        read=lambda obs, item: obs,
        steps=7,
        reward=1.0,
        outcome="success",
    )
    return driver, pv


def test_viewer_selects_the_image_role_not_render_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, pv = _driver(monkeypatch, View())
    assert pv.sources == ["render", "wrist"]
    assert pv.selected_source() == "wrist"
    assert pv.frames == [(3, 2)]  # the role's frame, not the render() one
    _, pv = _driver(monkeypatch, View(source="render"))
    assert pv.selected_source() == "render"


def test_viewer_closes_immediately_without_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver, pv = _driver(monkeypatch, View(backend="http"))
    start = time.monotonic()
    driver.close()
    assert time.monotonic() - start < 0.5
    assert pv.closed and pv.outcomes == ["success"]


def test_viewer_hold_serves_the_final_frame_until_it_elapses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver, pv = _driver(monkeypatch, View(backend="http", hold=0.2))
    start = time.monotonic()
    driver.close()
    assert time.monotonic() - start >= 0.2
    assert pv.closed
    assert pv.outcomes[-1] == "success [held]"
    assert len(pv.frames) == 2  # the final frame is redrawn for the hold


def test_viewer_hold_redraws_on_source_switch_and_ends_on_quit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver, pv = _driver(monkeypatch, View(backend="http", hold=True))

    def user() -> None:
        time.sleep(0.15)
        pv.selected = 0  # switch to render() while held
        time.sleep(0.15)
        pv.quit = True

    thread = threading.Thread(target=user)
    thread.start()
    driver.close()
    thread.join()
    assert pv.closed
    assert (6, 4) in pv.frames  # the render() frame, drawn after the switch


def test_unbounded_hold_without_a_way_to_quit_does_not_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver, pv = _driver(monkeypatch, View(backend="terminal", hold=True))
    monkeypatch.setattr("sys.stdin", io.StringIO())
    with pytest.warns(UserWarning, match="hold=True needs an interactive terminal"):
        driver.close()
    assert pv.closed


def test_hold_is_skipped_after_the_user_quit(monkeypatch: pytest.MonkeyPatch) -> None:
    driver, pv = _driver(monkeypatch, View(backend="http", hold=30.0))
    pv.quit = True
    driver.feed(
        contract=None,
        client=None,
        obs=np.zeros((2, 3, 3), dtype=np.uint8),
        read=lambda obs, item: obs,
        steps=8,
        reward=1.0,
        outcome="",
    )
    assert driver.quit_requested()
    start = time.monotonic()
    driver.close()
    assert time.monotonic() - start < 0.5


def test_step_hz_paces_steps_without_bursting() -> None:
    driver = ViewerDriver(View(step_hz=50.0))
    start = time.perf_counter()
    slept = [driver.pace() for _ in range(6)]
    elapsed = time.perf_counter() - start
    assert slept[0] == 0.0  # the first step only starts the schedule
    assert elapsed >= 5 * 0.02 * 0.9
    time.sleep(0.1)  # a slow step: the schedule restarts instead of bursting
    assert driver.pace() == 0.0
    assert ViewerDriver(View()).pace() == 0.0
