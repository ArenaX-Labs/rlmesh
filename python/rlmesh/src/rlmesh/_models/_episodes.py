"""The model's bounded per-episode store: predict ordinal, seed, and free state.

One model server fans several env workers in, so consecutive predicts alternate
between episodes A, B, C... Anything a model keeps per episode -- a plan, a
history buffer, a sampler seed -- has to be keyed by ``episode_id`` rather than
held in one slot on the model object. This is that keying, once, in the SDK,
instead of once per model image.

Entries are dropped at the episode-end edge (:meth:`EpisodeStore.end`, driven by
the explicit ``ResetAdapter`` on the served path and by the session's episode
boundary locally). Live state is never evicted: a run that never signals an end
still cannot grow without bound, because past the capacity a NEW episode's
predict fails explicitly instead of silently dropping another episode's state.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import PredictContext

EPISODE_STATE_CAPACITY = 65_536
"""Live episodes a model tracks before a NEW episode is refused.

A ceiling, not an eviction threshold: an entry only leaves at its end edge. Far
above the lanes any one model server legitimately admits at once, so reaching
it means episode ends are not arriving; a deployment that really runs more
concurrent context-aware episodes raises :data:`EPISODE_STATE_CAPACITY_ENV`.
"""

EPISODE_STATE_CAPACITY_ENV = "RLMESH_EPISODE_CAPACITY"
"""Env var overriding :data:`EPISODE_STATE_CAPACITY` for one model process."""


def episode_capacity() -> int:
    """The live-episode ceiling: the env override when set, else the default."""
    raw = os.environ.get(EPISODE_STATE_CAPACITY_ENV, "").strip()
    return max(1, int(raw)) if raw.isdigit() else EPISODE_STATE_CAPACITY


def _native_predict_seed(episode_seed: int, predict_index: int) -> int:
    from .._load_native import load_native

    return int(load_native("predict_seed")(episode_seed, predict_index))


class _Episode:
    """One episode's bookkeeping: how many predicts it has had, its seed, its state."""

    __slots__ = ("index", "seed", "state")

    def __init__(self, seed: int | None) -> None:
        self.index = 0
        self.seed = seed
        self.state: dict[str, Any] = {}


class EpisodeStore:
    """Bounded per-episode state, keyed by ``episode_id``.

    ``on_end`` is the model's own episode-end hook (a ``Model`` subclass's
    ``reset``, or the ``on_episode_end=`` callback), already normalized to take
    the ended episode's id.
    """

    def __init__(
        self,
        on_end: Callable[[str], None] | None = None,
        *,
        capacity: int | None = None,
    ) -> None:
        self._on_end = on_end
        self._capacity = (
            episode_capacity() if capacity is None else max(1, int(capacity))
        )
        self._episodes: dict[str, _Episode] = {}

    def __len__(self) -> int:
        return len(self._episodes)

    def context(self, raw: Any) -> PredictContext:
        """Enrich one raw native context into the full :class:`PredictContext`.

        ``raw`` is what the engine hands a corner: ``{"episode_id",
        "episode_seed"}``. This adds the episode's re-plan ordinal
        (``predict_index``), the seed derived from it (``predict_seed``, ``None``
        when the episode was not explicitly seeded), and the episode's own
        ``state`` dict -- the model's free slot, alive until the episode ends.

        A row with no identity (an anonymous spec-less lane) gets a throwaway
        context that is never stored: there is nothing to key it by. A new
        episode past the capacity is refused (``RuntimeError``) rather than
        admitted at another live episode's expense.
        """
        episode_id = str(raw.get("episode_id") or "") if raw is not None else ""
        seed = raw.get("episode_seed") if raw is not None else None
        if not episode_id:
            return {
                "episode_id": "",
                "episode_seed": seed,
                "predict_index": 0,
                "predict_seed": None,
                "state": {},
            }
        episode = self._episodes.get(episode_id)
        if episode is None:
            if len(self._episodes) >= self._capacity:
                raise RuntimeError(
                    f"model already tracks {self._capacity} live episodes; refusing "
                    f"episode {episode_id} rather than evicting another episode's "
                    "live state. Either episode ends are not reaching this model, "
                    "or the deployment runs more concurrent episodes than "
                    f"{EPISODE_STATE_CAPACITY_ENV} allows"
                )
            episode = self._episodes[episode_id] = _Episode(seed)
        index = episode.index
        episode.index += 1
        return {
            "episode_id": episode_id,
            "episode_seed": episode.seed,
            "predict_index": index,
            "predict_seed": (
                None
                if episode.seed is None
                else _native_predict_seed(episode.seed, index)
            ),
            "state": episode.state,
        }

    def end(self, episode_id: str = "") -> None:
        """Drop ``episode_id``'s entry and fire the model's own end hook."""
        self._episodes.pop(episode_id, None)
        if self._on_end is not None:
            self._on_end(episode_id)
