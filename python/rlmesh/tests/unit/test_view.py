from __future__ import annotations

import numpy as np
import pytest
from rlmesh._models._eval import Session
from rlmesh._models._view import View, _resolve_render, _to_hwc_u8, resolve_view


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
    sess: Session[object, object] = Session(env=object())
    assert sess._view_outcome() == ""

    sess._terminated = True
    sess._last_info = {"is_success": False}
    assert sess._view_outcome() == "failure"

    sess._last_info = {"is_success": True}
    assert sess._view_outcome() == "success"

    sess._last_info = {}
    assert sess._view_outcome() == "success"

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
