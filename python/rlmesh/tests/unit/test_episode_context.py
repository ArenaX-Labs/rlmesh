"""Per-episode context: the bounded store, the corner-synthesis matrix, the end edge."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import rlmesh
from rlmesh._models._episodes import EpisodeStore
from rlmesh._models.base import accepts_episode_id
from rlmesh.numpy import Model


def raw(episode_id: str = "ep", seed: int | None = 7) -> dict[str, Any]:
    """One raw native context, as the engine hands a corner."""
    return {"episode_id": episode_id, "episode_seed": seed}


# ---------------------------------------------------------------- store semantics


def test_store_counts_predicts_and_derives_a_seed_per_predict() -> None:
    store = EpisodeStore()

    first = store.context(raw())
    second = store.context(raw())

    assert first["predict_index"] == 0
    assert second["predict_index"] == 1
    assert first["predict_seed"] == rlmesh.predict_seed(7, 0)
    assert second["predict_seed"] == rlmesh.predict_seed(7, 1)
    assert first["predict_seed"] != second["predict_seed"]


def test_store_keys_state_by_episode_and_hands_the_same_dict_back() -> None:
    store = EpisodeStore()

    store.context(raw("a"))["state"]["plan"] = [1, 2]
    other = store.context(raw("b"))["state"]

    assert store.context(raw("a"))["state"] == {"plan": [1, 2]}
    assert other == {}
    assert len(store) == 2


def test_store_without_a_seed_leaves_predict_seed_none() -> None:
    assert EpisodeStore().context(raw(seed=None))["predict_seed"] is None


def test_store_gives_an_anonymous_row_a_throwaway_context() -> None:
    store = EpisodeStore()

    context = store.context(raw(episode_id="", seed=None))

    assert context == {
        "episode_id": "",
        "episode_seed": None,
        "predict_index": 0,
        "predict_seed": None,
        "state": {},
    }
    assert len(store) == 0


def test_store_end_drops_the_entry_and_fires_the_hook() -> None:
    ended: list[str] = []
    store = EpisodeStore(ended.append)

    store.context(raw("a"))["state"]["plan"] = 1
    store.end("a")

    assert ended == ["a"]
    assert len(store) == 0
    # A fresh episode under the same id starts clean.
    assert store.context(raw("a"))["state"] == {}


def test_store_eviction_fires_the_same_end_hook_and_warns() -> None:
    ended: list[str] = []
    store = EpisodeStore(ended.append, capacity=2)

    store.context(raw("a"))
    store.context(raw("b"))
    with pytest.warns(RuntimeWarning, match="live episodes"):
        store.context(raw("c"))

    # The least-recently-used episode is dropped through the end edge, so a model
    # that mirrors the store elsewhere is told either way.
    assert ended == ["a"]
    assert len(store) == 2


def test_store_eviction_is_least_recently_used() -> None:
    ended: list[str] = []
    store = EpisodeStore(ended.append, capacity=2)

    store.context(raw("a"))
    store.context(raw("b"))
    store.context(raw("a"))  # touch a
    with pytest.warns(RuntimeWarning):
        store.context(raw("c"))

    assert ended == ["b"]


# ------------------------------------------------------------------- arity sniff


def test_accepts_episode_id_reads_the_hook_signature() -> None:
    assert not accepts_episode_id(lambda: None)
    assert accepts_episode_id(lambda _episode_id: None)
    assert accepts_episode_id(lambda *_args: None)
    assert not accepts_episode_id(lambda *, keyword=None: None)


def test_zero_argument_reset_still_fires() -> None:
    calls: list[int] = []

    class Policy(Model):
        def predict(self, observation: Any) -> Any:
            return 0

        def reset(self) -> None:
            calls.append(1)

    Policy()._on_episode_end("ep-1")

    assert calls == [1]


def test_reset_taking_an_episode_id_receives_it() -> None:
    ended: list[str] = []

    class Policy(Model):
        def predict(self, observation: Any) -> Any:
            return 0

        def reset(self, episode_id: str = "") -> None:
            ended.append(episode_id)

    Policy()._on_episode_end("ep-1")

    assert ended == ["ep-1"]


def test_episode_end_drops_the_store_entry_for_that_episode() -> None:
    class Policy(Model):
        def predict(self, observation: Any, context: Any) -> Any:
            context["state"]["seen"] = context["predict_index"]
            return 0

        def reset(self, episode_id: str = "") -> None:
            pass

    model = Policy()
    cast("Any", model._raw_predict)(np.zeros(2), raw("ep-1"))
    assert len(model._episodes) == 1
    model._on_episode_end("ep-1")
    assert len(model._episodes) == 0


# ------------------------------------------------- corner-synthesis context matrix


def test_chunk_batch_corner_receives_a_context_per_row() -> None:
    rows: list[Any] = []

    class Policy(Model):
        def predict_chunk_batch(
            self, observations: Any, horizon: int, context: Any
        ) -> Any:
            rows.append(context)
            n = observations["x"].shape[0]
            return np.zeros((n, horizon, 2), dtype=np.float32)

    model = Policy()
    contexts = [raw("a", 1), raw("b", 2)]
    cast("Any", model._raw_predict_chunk_batch)({"x": np.zeros((2, 3))}, 4, contexts)

    assert [row["episode_id"] for row in rows[0]] == ["a", "b"]
    assert [row["predict_seed"] for row in rows[0]] == [
        rlmesh.predict_seed(1, 0),
        rlmesh.predict_seed(2, 0),
    ]


def test_derived_single_lane_corner_hands_down_a_one_element_list() -> None:
    rows: list[Any] = []

    class Policy(Model):
        def predict_chunk_batch(
            self, observations: Any, horizon: int, context: Any
        ) -> Any:
            rows.append(context)
            n = observations["x"].shape[0]
            return np.zeros((n, horizon, 2), dtype=np.float32)

    model = Policy()
    # predict_chunk and predict are both derived from the batched chunk corner;
    # each runs it as a batch of one, so the context list has exactly one row.
    cast("Any", model._raw_predict_chunk)({"x": np.zeros(3)}, 4, raw("a", 1))
    cast("Any", model._raw_predict)({"x": np.zeros(3)}, raw("a", 1))

    assert [len(batch) for batch in rows] == [1, 1]
    assert [batch[0]["episode_id"] for batch in rows] == ["a", "a"]
    # Both calls are the same episode, so the ordinal keeps counting.
    assert [batch[0]["predict_index"] for batch in rows] == [0, 1]


def test_a_corner_that_never_asked_for_a_context_is_left_alone() -> None:
    class Policy(Model):
        def predict(self, observation: Any) -> Any:
            return np.zeros(2, dtype=np.float32)

    model = Policy()
    model._raw_predict(np.zeros(3))

    assert len(model._episodes) == 0


def test_chunk_corner_without_a_horizon_keeps_its_context() -> None:
    seen: list[Any] = []

    class Policy(Model):
        def predict_chunk(self, observation: Any, context: Any) -> Any:
            seen.append(context)
            return np.zeros((3, 2), dtype=np.float32)

    model = Policy()
    # The internal contract is always (obs, horizon[, context]); a corner that
    # declared a context but no horizon must not read the horizon as its context.
    cast("Any", model._raw_predict_chunk)(np.zeros(3), 4, raw("a", 5))

    assert seen[0]["episode_id"] == "a"
    assert seen[0]["predict_seed"] == rlmesh.predict_seed(5, 0)
