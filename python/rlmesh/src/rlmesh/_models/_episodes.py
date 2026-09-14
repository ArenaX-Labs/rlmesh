"""The model's bounded per-episode store: predict ordinal, seed, and free state.

One model server fans several env workers in, so consecutive predicts alternate
between episodes A, B, C... Anything a model keeps per episode -- a plan, a
history buffer, a sampler seed -- has to be keyed by ``episode_id`` rather than
held in one slot on the model object. This is that keying, once, in the SDK,
instead of once per model image.

Entries are dropped at the episode-end edge (:meth:`EpisodeStore.end`, driven by
the explicit ``ResetAdapter`` on the served path and by the session's episode
boundary locally). A run that never signals an end still cannot grow without
bound: past :data:`EPISODE_STATE_CAPACITY` the least-recently-used entry is
evicted through the *same* end callback, so a model that mirrors the store
elsewhere gets the drop edge either way, and warns.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import PredictContext

EPISODE_STATE_CAPACITY = 4096
"""Live episodes a model tracks before the least-recently-used one is evicted.

Comfortably above the env workers any one model server admits at once, so a real
eviction means episode ends are being missed, not that the fleet is large.
"""


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
        capacity: int = EPISODE_STATE_CAPACITY,
    ) -> None:
        self._on_end = on_end
        self._capacity = max(1, int(capacity))
        self._episodes: OrderedDict[str, _Episode] = OrderedDict()

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
        context that is never stored: there is nothing to key it by.
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
            episode = self._episodes[episode_id] = _Episode(seed)
            self._evict_overflow()
        else:
            self._episodes.move_to_end(episode_id)
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

    def _evict_overflow(self) -> None:
        while len(self._episodes) > self._capacity:
            evicted, _ = self._episodes.popitem(last=False)
            if self._on_end is not None:
                self._on_end(evicted)
            warnings.warn(
                f"tracking more than {self._capacity} live episodes; evicted "
                f"episode {evicted} and fired its end hook. Episode ends are "
                "probably not reaching this model.",
                RuntimeWarning,
                stacklevel=4,
            )
