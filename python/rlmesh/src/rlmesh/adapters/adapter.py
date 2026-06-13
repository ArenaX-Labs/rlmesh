"""The runtime adapter that applies a resolved plan to observations and actions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast, final

from ..numpy import NumpyArray, ensure_available
from .helpers.bridge import decode_value, encode_value
from .specs import ObsTransform

if TYPE_CHECKING:
    from .._rlmesh import AdapterPlan

ActionT = TypeVar("ActionT")


class AdapterBase(ABC, Generic[ActionT]):
    """Base class for env-to-model adapters.

    :func:`rlmesh.adapters.resolve` derives the spec-driven implementation
    (:class:`IOAdapter`); subclass this directly to plug a fully custom
    pairing into anything built around adapters instead. Implement the two
    transforms; ``wrap_predict`` comes for free. Custom adapters may hold
    state across steps (e.g. cache proprio from ``transform_obs`` for use
    in ``transform_action``) -- that is their power over declarative specs.
    Override :meth:`reset` to clear any such state at episode boundaries.
    """

    @abstractmethod
    def transform_obs(self, raw_obs: Mapping[str, Any]) -> dict[str, Any]:
        """Convert a raw env observation into the model input payload."""

    @abstractmethod
    def transform_action(self, raw_action: object) -> ActionT:
        """Convert a model action output into the env action."""

    def reset(self) -> None:
        """Clear any episode-scoped state.

        The default does nothing (resolved adapters are stateless). Stateful
        custom adapters override this and wire it to the model worker's
        ``on_reset`` callback so state never leaks across episodes.
        """

    def describe(self) -> str:
        """Return a human-readable summary of the adapter."""
        return f"{type(self).__name__} (custom adapter)"

    def wrap_predict(
        self, predict_fn: Callable[[dict[str, Any]], object]
    ) -> Callable[[Any], ActionT]:
        """Wrap a model predict function with both transforms.

        The returned callable takes a raw env observation (a mapping) and
        returns an env-ready action, suitable for :class:`rlmesh.numpy.Model`.
        """

        def predict(raw_obs: Any) -> ActionT:
            if not isinstance(raw_obs, Mapping):
                raise TypeError(
                    f"expected a mapping observation, got {type(raw_obs)!r}"
                )
            return self.transform_action(
                predict_fn(self.transform_obs(cast(Mapping[str, Any], raw_obs)))
            )

        return predict


@final
class IOAdapter(AdapterBase[NumpyArray]):
    """A resolved env-to-model adapter; build instances with ``resolve()``.

    Declarative plans run in the native ``rlmesh-adapters`` core; custom
    inputs run their host-language transforms on the raw Python
    observation, exactly as before.
    """

    def __init__(
        self,
        plan: AdapterPlan,
        customs: Mapping[str, ObsTransform],
        stacks: Mapping[str, int] | None = None,
    ) -> None:
        self._plan = plan
        self._customs = dict(customs)
        # Per-key frame-history depth (>1 only) and the rolling buffers that
        # back it. Stacking happens host-side, after the native transform.
        self._stacks = {key: n for key, n in (stacks or {}).items() if n > 1}
        self._buffers: dict[str, deque[Any]] = {}

    def transform_obs(self, raw_obs: Mapping[str, Any]) -> dict[str, Any]:
        """Convert a raw env observation into the model input payload.

        Only the observation keys the plan actually reads are encoded and
        sent across the native boundary, so an unused -- possibly
        unencodable -- observation key never aborts a step. Custom inputs
        still see the full raw observation. Inputs that request frame history
        are stacked here from a rolling buffer, cleared by :meth:`reset`.
        """
        ensure_available()
        selected = {
            key: raw_obs[key]
            for key in self._plan.referenced_obs_keys()
            if key in raw_obs
        }
        encoded = self._plan.transform_obs(encode_value(selected))
        payload: dict[str, Any] = {
            key: decode_value(value) for key, value in encoded.items()
        }
        for key, depth in self._stacks.items():
            if key in payload:
                payload[key] = self._stack_frames(key, payload[key], depth)
        for key, transform in self._customs.items():
            payload[key] = transform(raw_obs)
        return payload

    def _stack_frames(self, key: str, frame: Any, depth: int) -> NumpyArray:
        import numpy as np

        frames = self._buffers.get(key)
        if frames is None:
            frames = deque[Any](maxlen=depth)
            self._buffers[key] = frames
        if not frames:
            # Pad the start of an episode with copies of the first frame so
            # the stack is full from step zero.
            for _ in range(depth - 1):
                frames.append(frame)
        frames.append(frame)
        return cast(
            "NumpyArray", np.stack(cast("list[NumpyArray]", list(frames)), axis=0)
        )

    def reset(self) -> None:
        """Clear the frame-history buffers at an episode boundary."""
        self._buffers.clear()

    def transform_action(self, raw_action: object) -> NumpyArray:
        """Convert a model action vector into the env action vector."""
        ensure_available()
        return cast(
            "NumpyArray",
            decode_value(self._plan.transform_action(encode_value(raw_action))),
        )

    def describe(self) -> str:
        """Return a human-readable summary of the resolved transformations."""
        return self._plan.describe()


__all__ = ["AdapterBase", "IOAdapter"]
