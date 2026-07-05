"""Action-chunk replay: split a chunked action and replay it one step at a time.

Shared by the in-process run(env) loop (:mod:`rlmesh._models._eval`) and the
explicit ``adapter.wrap_predict`` path so the two stay byte-for-byte identical,
and aligned with the native ``split_chunk`` the served Rust engine uses. A
chunk-capable model (one defining ``predict_chunk``) returns a *chunk* of actions
when the replay horizon is > 1; the queue replays them one per step, predicting
again only when it drains.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from typing import Any, cast


def split_chunk(raw_action: Any) -> list[Any]:
    """Split a chunked model action along its leading (chunk) axis.

    A ``[chunk, dim]`` output becomes ``chunk`` per-step ``[dim]`` actions. Splits
    with ``list()`` so each per-step leaf stays in the model's own framework (a
    torch/jax *device* tensor is NOT force-converted to numpy here) -- it is then
    bridged identically to the non-chunked path and the serve path. A Dict-space
    action carries the horizon as each leaf's leading axis (e.g. ``{"arm":
    tensor[H, ...]}``): every ``Mapping`` leaf is split recursively, leaves must
    agree on the horizon, and the frames are zipped into H per-step mappings --
    mirroring the native ``split_chunk`` Map arm and the ``_first_frame`` Mapping
    recursion. A string/bytes leaf or a non-iterable (scalar) is a degenerate
    single-step "chunk", matching the native ``split_chunk`` (which treats a
    text / scalar leaf as one step). Called only when the replay ``horizon > 1``;
    a mis-shaped output fails the action conversion downstream rather than being
    silently mis-sliced (except a flat dim-1 action, the inference blind spot the
    native ``split_chunk`` shares).
    """
    if isinstance(raw_action, (str, bytes, bytearray)):
        return [raw_action]
    if isinstance(raw_action, Mapping):
        items = cast("Mapping[Any, Any]", raw_action)
        fields: list[tuple[Any, list[Any]]] = []
        horizon: int | None = None
        for key, value in items.items():
            frames = split_chunk(value)
            if horizon is not None and horizon != len(frames):
                raise ValueError(
                    f"action chunk fields disagree on horizon: {horizon} vs "
                    f"{len(frames)}"
                )
            horizon = len(frames)
            fields.append((key, frames))
        return [{key: frames[i] for key, frames in fields} for i in range(horizon or 0)]
    try:
        return list(raw_action)
    except TypeError:
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
        self.horizon = max(1, int(horizon))
        self._queue: deque[Any] = deque()
        self.last_chunk_len = 0
        """Length of the chunk the model last returned (post-cap), 0 before any.

        The HUD reads this instead of the requested horizon so a model whose
        native chunk is shorter than ``execution_horizon`` displays its real
        replay position (e.g. 1/4, not 5/8).
        """

    def reset(self) -> None:
        """Drop any un-replayed actions at an episode boundary."""
        self._queue.clear()
        self.last_chunk_len = 0

    @property
    def pending(self) -> int:
        """Actions still queued for replay before the model re-plans.

        ``0`` on a step that re-planned (the model just ran); the debug viewer uses
        this to show the action-chunk replay position without reaching into internals.
        """
        return len(self._queue)

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
        self.last_chunk_len = len(chunk)
        self._queue.extend(chunk[1:])
        return chunk[0]
