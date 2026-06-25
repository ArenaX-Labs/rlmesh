"""Action-chunk replay: split a chunked action and replay it one step at a time.

Shared by the in-process run(env) loop (:mod:`rlmesh._models._eval`) and the
explicit ``adapter.wrap_predict`` path so the two stay byte-for-byte identical,
and aligned with the native ``split_chunk`` the served Rust engine uses. A model
that declares ``execute_horizon > 1`` returns a *chunk* of actions; the queue
replays them one per step, predicting again only when it drains.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from typing import Any


def split_chunk(raw_action: Any) -> list[Any]:
    """Split a chunked model action along its leading (chunk) axis.

    A ``[chunk, dim]`` output becomes ``chunk`` per-step ``[dim]`` actions. Splits
    with ``list()`` so each per-step leaf stays in the model's own framework (a
    torch/jax *device* tensor is NOT force-converted to numpy here) -- it is then
    bridged identically to the non-chunked path and the serve path. A string/bytes
    leaf, a non-iterable (scalar), or a structured (mapping) output is a degenerate
    single-step "chunk", matching the native ``split_chunk`` (which treats a
    text / scalar / map leaf as one step). Called only when ``execute_horizon > 1``;
    a mis-shaped output fails the action conversion downstream rather than being
    silently mis-sliced (except a flat dim-1 action, the inference blind spot the
    native ``split_chunk`` shares).
    """
    # A str/bytes is iterable but a single action leaf, not a per-character chunk;
    # match the native side, which keeps a text action as one step.
    if isinstance(raw_action, (str, bytes, bytearray, Mapping)):
        return [raw_action]
    try:
        return list(raw_action)
    except TypeError:
        # A 0-d array / scalar tensor / bare number is not iterable: one step.
        return [raw_action]


class ChunkReplay:
    """A per-episode action-chunk replay queue.

    ``horizon == 1`` is a passthrough (it never queues); ``horizon > 1`` splits a
    predicted chunk, returns its first row now, and replays the rest (capped to
    ``horizon`` -- a receding-horizon model may emit a longer chunk than it
    re-plans) one per subsequent step before predicting again. :meth:`reset` (an
    episode boundary) drops any un-replayed tail.
    """

    def __init__(self, horizon: int) -> None:
        # Coerce defensively: a duck-typed adapter may expose a non-int horizon.
        self.horizon = max(1, int(horizon))
        self._queue: deque[Any] = deque()

    def reset(self) -> None:
        """Drop any un-replayed actions at an episode boundary."""
        self._queue.clear()

    def next_action(self, predict: Callable[[], Any]) -> Any:
        """Return the next raw model action.

        Calls ``predict`` (a thunk doing obs assembly + the model forward) only
        when the queue drains; while a chunk is replaying it pops the next queued
        action and ``predict`` is never invoked.
        """
        if self.horizon > 1 and self._queue:
            return self._queue.popleft()
        predicted = predict()
        if self.horizon == 1:
            return predicted
        chunk = split_chunk(predicted)[: self.horizon]
        if not chunk:
            raise ValueError(
                "a chunked model (execute_horizon>1) returned an empty action chunk"
            )
        self._queue.extend(chunk[1:])
        return chunk[0]
