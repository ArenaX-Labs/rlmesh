"""Batched predict-corner fusion: the runtime stacks N lanes into one batched
value (keys get batched) and splits the batched action back per lane."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from rlmesh.numpy import _numpy_bridge as bridge


def test_dict_obs_fuses_keys_to_leading_batch_axis() -> None:
    lanes = [
        {"img": np.full((2, 2), i, np.float32), "state": np.arange(4) + i}
        for i in range(3)
    ]
    fused = cast(Any, bridge.tree_stack(lanes))
    assert fused["img"].shape == (3, 2, 2)
    assert fused["state"].shape == (3, 4)
    assert (fused["img"][1] == 1).all()


def test_bare_array_obs_fuses() -> None:
    fused = cast(Any, bridge.tree_stack([np.arange(3) + i for i in range(4)]))
    assert fused.shape == (4, 3)


def test_batched_action_splits_back_per_lane() -> None:
    action = np.stack([np.arange(5) + 10 * i for i in range(3)])  # [3, 5]
    parts = cast(Any, bridge.tree_unstack(action, 3))
    assert len(parts) == 3
    assert parts[2].tolist() == [20, 21, 22, 23, 24]


def test_chunk_splits_batch_axis_only_keeping_horizon() -> None:
    chunk = np.zeros((3, 4, 5))  # [N, horizon, action_dim]
    chunk[1] = 7
    parts = cast(Any, bridge.tree_unstack(chunk, 3))
    assert [p.shape for p in parts] == [(4, 5), (4, 5), (4, 5)]
    assert (parts[1] == 7).all()


def test_dict_action_round_trips_structurally() -> None:
    action = {
        "move": np.stack([np.ones(2) * i for i in range(3)]),
        "grip": np.array([0, 1, 0]),
    }
    parts = cast(Any, bridge.tree_unstack(action, 3))
    assert len(parts) == 3
    assert parts[2]["move"].tolist() == [2, 2]
    assert parts[1]["grip"] == 1


def test_text_leaf_falls_back_to_per_lane_list() -> None:
    fused = cast(Any, bridge.tree_stack([{"instr": "go"}, {"instr": "stop"}]))
    assert fused["instr"] == ["go", "stop"]


def test_unstack_rejects_wrong_leading_axis() -> None:
    with pytest.raises(ValueError, match="leading batch axis 3"):
        bridge.tree_unstack(np.zeros((2, 5)), 3)


def test_ragged_leaf_raises_instead_of_silent_list() -> None:
    # Per-lane leaves with different shapes cannot fuse; raise rather than returning
    # a list for this leaf while siblings stack into [N, ...] (a structurally
    # inconsistent batch the model's single forward can't consume).
    with pytest.raises(ValueError, match="ragged"):
        bridge.tree_stack([{"x": np.zeros(2)}, {"x": np.zeros(3)}])
