"""Predict-corner synthesis: defining one corner derives the rest (down the
batch/chunk lattice); chunking can't be synthesized; the tie prefers predict_batch."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import rlmesh
import rlmesh.numpy
from rlmesh._models.base import _synthesize_corners
from rlmesh._value_conversion import identity_bridge
from rlmesh.numpy import _numpy_bridge as bridge

# Reference corners. The chunk corners encode (lane, frame) into the action so the
# derivations' axis/lane/frame selection is verifiable: action[..., 0]=lane,
# action[..., 1]=frame.


def _pcb(observations, horizon):  # fused {x:[N,F]} -> chunk [N, horizon, 2]
    n = observations["x"].shape[0]
    return np.stack(
        [np.stack([[lane, frame] for frame in range(horizon)]) for lane in range(n)]
    )


def _pc(observations, horizon):  # single {x:[F]} -> chunk [horizon, 2]
    return np.stack([[0, frame] for frame in range(horizon)])


def _pb(observations):  # fused {x:[N,F]} -> [N, 2]
    n = observations["x"].shape[0]
    return np.stack([[lane, -1] for lane in range(n)])


def _p(observations):  # single {x:[F]} -> [2]
    return np.array([0, -1])


SINGLE = {"x": np.zeros(3)}
FUSED = {"x": np.zeros((4, 3))}


def test_chunk_batch_alone_yields_all_four() -> None:
    p, pc, pb, pcb = _synthesize_corners(bridge, None, None, None, _pcb)
    assert pcb is _pcb and pc is not None and pb is not None and p is not None
    # predict_chunk = debatch(pcb): [horizon, 2], frame axis preserved
    assert cast("Any", pc(SINGLE, 5)).shape == (5, 2)
    assert cast("Any", pc(SINGLE, 5))[:, 1].tolist() == [0, 1, 2, 3, 4]
    # predict_batch = de-chunk(pcb): [N, 2], lane axis preserved, frame 0 taken
    assert cast("Any", pb(FUSED)).shape == (4, 2)
    assert cast("Any", pb(FUSED))[:, 0].tolist() == [0, 1, 2, 3]
    assert cast("Any", pb(FUSED))[:, 1].tolist() == [0, 0, 0, 0]
    # predict = debatch(predict_batch): single action
    assert cast("Any", p(SINGLE)).tolist() == [0, 0]


def test_chunk_alone_yields_predict_only() -> None:
    p, pc, pb, pcb = _synthesize_corners(bridge, None, _pc, None, None)
    assert pc is _pc and p is not None
    assert pb is None and pcb is None  # batched left to the engine's per-lane loop
    assert cast("Any", p(SINGLE)).tolist() == [0, 0]  # de-chunk single: frame 0


def test_batch_alone_yields_predict_no_chunk() -> None:
    p, pc, pb, pcb = _synthesize_corners(bridge, None, None, _pb, None)
    assert pb is _pb and p is not None
    assert pc is None and pcb is None  # no chunk capability
    assert cast("Any", p(SINGLE)).tolist() == [0, -1]  # debatch


def test_predict_alone_synthesizes_nothing() -> None:
    p, pc, pb, pcb = _synthesize_corners(bridge, _p, None, None, None)
    assert (p, pc, pb, pcb) == (_p, None, None, None)


def test_tie_predict_prefers_batch_over_chunk() -> None:
    # predict_batch AND predict_chunk defined, predict missing -> derive from
    # predict_batch (stays un-chunked), not predict_chunk.
    p, pc, pb, pcb = _synthesize_corners(bridge, None, _pc, _pb, None)
    assert p is not None
    assert cast("Any", p(SINGLE)).tolist() == [
        0,
        -1,
    ]  # _pb's marker (-1), not _pc's frame


def test_explicit_corners_are_never_overwritten() -> None:
    p, pc, pb, pcb = _synthesize_corners(bridge, _p, _pc, _pb, _pcb)
    assert (p, pc, pb, pcb) == (_p, _pc, _pb, _pcb)


def test_identity_bridge_cannot_dechunk() -> None:
    # The raw Value bridge can debatch but not de-chunk (no array leaves), so a
    # chunk-only model yields no predict -> caller raises.
    p, pc, pb, pcb = _synthesize_corners(identity_bridge, None, None, None, _pcb)
    assert pc is not None  # debatch works (list-based)
    assert pb is None and p is None  # de-chunk does not


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
    # Production now calls chunk corners positionally (obs, horizon); a synthesized
    # corner forwards *args/**kwargs, so it stays tolerant of a keyword horizon too.
    _, pc, _, _ = _synthesize_corners(bridge, None, None, None, _pcb)
    assert cast("Any", pc)(SINGLE, horizon=4).shape == (4, 2)


def test_constructed_model_chunk_corner_keyword_path() -> None:
    class M(rlmesh.numpy.Model):
        spec = rlmesh.NO_ADAPTER

        def predict_chunk_batch(self, observations, horizon):
            return _pcb(observations, horizon)

    m = M()
    assert cast("Any", m._raw_predict_chunk)(SINGLE, horizon=3).shape == (3, 2)


def test_accepts_horizon_detects_optional_second_param() -> None:
    from rlmesh._models.base import _accepts_horizon

    assert not _accepts_horizon(lambda obs: obs)
    assert _accepts_horizon(lambda obs, h: obs)
    assert _accepts_horizon(lambda obs, execution_horizon=1: obs)
    assert _accepts_horizon(lambda obs, *args: obs)


def test_predict_chunk_without_horizon_constructs_and_swallows_horizon() -> None:
    # The common case: a fixed-chunk policy writes predict_chunk(obs) with no horizon.
    # Construction normalizes it to the internal (obs, horizon) contract, so the
    # runtime still calls it positionally with a horizon the corner swallows, and the
    # model returns its NATIVE chunk length regardless.
    ran: list[int] = []

    class M(rlmesh.numpy.Model):
        spec = rlmesh.NO_ADAPTER

        def predict_chunk(self, observation):
            ran.append(1)
            return np.stack([[0, frame] for frame in range(4)])  # native length 4

    m = M()
    chunk = cast("Any", m._raw_predict_chunk)(SINGLE, 2)  # runtime passes a horizon
    assert chunk.shape == (4, 2)  # native length, the horizon=2 is ignored
    assert ran  # the no-horizon corner ran


def test_predict_chunk_with_execution_horizon_receives_runtime_value() -> None:
    # An autoregressive decoder opts in with an optional second param; the runtime
    # hands it how many actions it will execute, so it can decode exactly that many.
    seen: list[int] = []

    class M(rlmesh.numpy.Model):
        spec = rlmesh.NO_ADAPTER

        def predict_chunk(self, observation, execution_horizon: int = 1):
            seen.append(execution_horizon)
            return np.stack([[0, frame] for frame in range(execution_horizon)])

    m = M()
    chunk = cast("Any", m._raw_predict_chunk)(SINGLE, 3)
    assert chunk.shape == (3, 2)  # decoded exactly execution_horizon actions
    assert seen == [3]


def _pc_list(observations, horizon):  # Discrete-style: the chunk is a Python list
    return [10 + frame for frame in range(horizon)]


def test_dechunk_handles_python_list_chunk() -> None:
    # split_chunk treats a List as the frames; de-chunk takes element 0, it must
    # not recurse into the list (which would index a scalar and raise).
    p, _, _, _ = _synthesize_corners(bridge, None, _pc_list, None, None)
    assert cast("Any", p)(SINGLE) == 10


def _pc_dict_frames(observations, horizon):  # dict action chunked as a list of frames
    return [{"arm": np.full(2, frame)} for frame in range(horizon)]


def test_dechunk_handles_list_of_dict_frames() -> None:
    p, _, _, _ = _synthesize_corners(bridge, None, _pc_dict_frames, None, None)
    assert cast("Any", p)(SINGLE)["arm"].tolist() == [0, 0]


def test_first_frame_recurses_into_dict_action() -> None:
    from rlmesh._models.base import _first_frame

    # A Dict-action chunk carries the horizon INSIDE each leaf ({k: [H, ...]}); the
    # first frame is each leaf[0], not the whole dict (which would drop the horizon,
    # matching the Rust split_chunk Map arm).
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
