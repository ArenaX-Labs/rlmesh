from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import rlmesh
import rlmesh.numpy
from rlmesh._models.base import _synthesize_corners
from rlmesh._value_conversion import identity_bridge
from rlmesh.numpy import _numpy_bridge as bridge


def _pcb(observations, horizon):
    n = observations["x"].shape[0]
    return np.stack(
        [np.stack([[lane, frame] for frame in range(horizon)]) for lane in range(n)]
    )


def _pc(observations, horizon):
    return np.stack([[0, frame] for frame in range(horizon)])


def _pb(observations):
    n = observations["x"].shape[0]
    return np.stack([[lane, -1] for lane in range(n)])


def _p(observations):
    return np.array([0, -1])


SINGLE = {"x": np.zeros(3)}
FUSED = {"x": np.zeros((4, 3))}


def test_chunk_batch_alone_yields_all_four() -> None:
    p, pc, pb, pcb = _synthesize_corners(bridge, None, None, None, _pcb)
    assert pcb is _pcb and pc is not None and pb is not None and p is not None
    assert cast("Any", pc(SINGLE, 5)).shape == (5, 2)
    assert cast("Any", pc(SINGLE, 5))[:, 1].tolist() == [0, 1, 2, 3, 4]
    assert cast("Any", pb(FUSED)).shape == (4, 2)
    assert cast("Any", pb(FUSED))[:, 0].tolist() == [0, 1, 2, 3]
    assert cast("Any", pb(FUSED))[:, 1].tolist() == [0, 0, 0, 0]
    assert cast("Any", p(SINGLE)).tolist() == [0, 0]


def test_chunk_alone_yields_predict_only() -> None:
    p, pc, pb, pcb = _synthesize_corners(bridge, None, _pc, None, None)
    assert pc is _pc and p is not None
    assert pb is None and pcb is None
    assert cast("Any", p(SINGLE)).tolist() == [0, 0]


def test_batch_alone_yields_predict_no_chunk() -> None:
    p, pc, pb, pcb = _synthesize_corners(bridge, None, None, _pb, None)
    assert pb is _pb and p is not None
    assert pc is None and pcb is None
    assert cast("Any", p(SINGLE)).tolist() == [0, -1]


def test_predict_alone_synthesizes_nothing() -> None:
    p, pc, pb, pcb = _synthesize_corners(bridge, _p, None, None, None)
    assert (p, pc, pb, pcb) == (_p, None, None, None)


def test_tie_predict_prefers_batch_over_chunk() -> None:
    p, pc, pb, pcb = _synthesize_corners(bridge, None, _pc, _pb, None)
    assert p is not None
    assert cast("Any", p(SINGLE)).tolist() == [
        0,
        -1,
    ]


def test_explicit_corners_are_never_overwritten() -> None:
    p, pc, pb, pcb = _synthesize_corners(bridge, _p, _pc, _pb, _pcb)
    assert (p, pc, pb, pcb) == (_p, _pc, _pb, _pcb)


def test_identity_bridge_cannot_dechunk() -> None:
    p, pc, pb, pcb = _synthesize_corners(identity_bridge, None, None, None, _pcb)
    assert pc is not None
    assert pb is None and p is None


def test_subclass_with_only_chunk_batch_constructs_and_runs() -> None:
    class M(rlmesh.numpy.Model):
        spec = rlmesh.NO_ADAPTER

        def predict_chunk_batch(self, observations, horizon):
            return _pcb(observations, horizon)

    m = M()
    assert m._raw_predict is not None
    assert m._raw_predict_chunk is not None
    assert m._raw_predict_batch is not None
    assert m._raw_predict_chunk_batch is not None
    assert cast("Any", m._raw_predict)(SINGLE).tolist() == [0, 0]
    assert cast("Any", m._raw_predict_batch)(FUSED)[:, 0].tolist() == [0, 1, 2, 3]
    assert cast("Any", m._raw_predict_chunk)(SINGLE, 3).shape == (3, 2)


def test_subclass_with_no_corner_still_errors() -> None:
    with pytest.raises(TypeError):

        class M(rlmesh.numpy.Model):
            spec = rlmesh.NO_ADAPTER

        M()


def test_synthesized_chunk_corner_accepts_keyword_horizon() -> None:
    _, pc, _, _ = _synthesize_corners(bridge, None, None, None, _pcb)
    assert cast("Any", pc)(SINGLE, horizon=4).shape == (4, 2)


def test_constructed_model_chunk_corner_keyword_path() -> None:
    class M(rlmesh.numpy.Model):
        spec = rlmesh.NO_ADAPTER

        def predict_chunk_batch(self, observations, horizon):
            return _pcb(observations, horizon)

    m = M()
    assert cast("Any", m._raw_predict_chunk)(SINGLE, horizon=3).shape == (3, 2)


def test_horizon_mode_detects_positional_and_keyword_spellings() -> None:
    from rlmesh._models.base import _horizon_mode

    assert _horizon_mode(lambda obs: obs) is None
    assert _horizon_mode(lambda obs, h: obs) == "positional"
    assert _horizon_mode(lambda obs, execution_horizon=1: obs) == "positional"
    assert _horizon_mode(lambda obs, *args: obs) == "positional"
    assert _horizon_mode(lambda obs, *, execution_horizon=1: obs) == "keyword"
    assert _horizon_mode(lambda obs, *, other=1: obs) is None


def test_predict_chunk_without_horizon_constructs_and_swallows_horizon() -> None:
    ran: list[int] = []

    class M(rlmesh.numpy.Model):
        spec = rlmesh.NO_ADAPTER

        def predict_chunk(self, observation):
            ran.append(1)
            return np.stack([[0, frame] for frame in range(4)])

    m = M()
    chunk = cast("Any", m._raw_predict_chunk)(SINGLE, 2)
    assert chunk.shape == (4, 2)
    assert ran


def test_predict_chunk_with_execution_horizon_receives_runtime_value() -> None:
    seen: list[int] = []

    class M(rlmesh.numpy.Model):
        spec = rlmesh.NO_ADAPTER

        def predict_chunk(self, observation, execution_horizon: int = 1):
            seen.append(execution_horizon)
            return np.stack([[0, frame] for frame in range(execution_horizon)])

    m = M()
    chunk = cast("Any", m._raw_predict_chunk)(SINGLE, 3)
    assert chunk.shape == (3, 2)
    assert seen == [3]


def test_predict_chunk_keyword_only_execution_horizon_receives_runtime_value() -> None:
    seen: list[int] = []

    class M(rlmesh.numpy.Model):
        spec = rlmesh.NO_ADAPTER

        def predict_chunk(self, observation, *, execution_horizon: int = 1):
            seen.append(execution_horizon)
            return np.stack([[0, frame] for frame in range(execution_horizon)])

    m = M()
    chunk = cast("Any", m._raw_predict_chunk)(SINGLE, 3)
    assert chunk.shape == (3, 2)
    assert seen == [3]


def _pc_list(observations, horizon):
    return [10 + frame for frame in range(horizon)]


def test_dechunk_handles_python_list_chunk() -> None:
    p, _, _, _ = _synthesize_corners(bridge, None, _pc_list, None, None)
    assert cast("Any", p)(SINGLE) == 10


def _pc_dict_frames(observations, horizon):
    return [{"arm": np.full(2, frame)} for frame in range(horizon)]


def test_dechunk_handles_list_of_dict_frames() -> None:
    p, _, _, _ = _synthesize_corners(bridge, None, _pc_dict_frames, None, None)
    assert cast("Any", p)(SINGLE)["arm"].tolist() == [0, 0]


def test_dechunk_holds_a_declared_native_chunk_at_horizon_one() -> None:
    def _short(obs: Any, execution_horizon: int = 1) -> Any:
        return np.zeros((execution_horizon, 2), np.float32)

    def _whole(obs: Any, execution_horizon: int = 1) -> Any:
        return np.zeros((8, 2), np.float32)

    obs = np.zeros(2, np.float32)
    p, _, _, _ = _synthesize_corners(bridge, None, _short, None, None, lambda: 8)
    assert p is not None
    with pytest.raises(
        ValueError, match="native_chunk=8 but its chunk corner returned 1"
    ):
        p(obs)
    p, _, _, _ = _synthesize_corners(bridge, None, _whole, None, None, lambda: 8)
    assert p is not None
    assert cast("Any", p(obs)).shape == (2,)


def test_first_frame_recurses_into_dict_action() -> None:
    from rlmesh._models.base import _first_frame

    chunk = {"arm": np.arange(6).reshape(3, 2), "grip": np.arange(3).reshape(3, 1)}
    frame = cast("Any", _first_frame(chunk))
    assert frame["arm"].tolist() == [0, 1]
    assert frame["grip"].tolist() == [0]


def test_first_frame_rejects_empty_chunk() -> None:
    from rlmesh._models.base import _first_frame

    with pytest.raises(ValueError, match="empty chunk"):
        _first_frame([])
    with pytest.raises(ValueError, match="empty chunk"):
        _first_frame(np.zeros((0, 2)))
