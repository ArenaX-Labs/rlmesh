"""ACT-style chunking policy: a model that needs a stateful custom adapter.

The checkpoint emits a *chunk* of the next ``CHUNK`` actions per inference
(ACT/ALOHA-style), and deployment quality depends on temporal ensembling:
each step's command averages the predictions that past chunks made for the
current step. That is inherently stateful -- it remembers chunks across
steps -- so no declarative spec can express it.

This module shows the sanctioned escape hatch: ``SPEC`` still declares
everything declarable (inputs and the per-step action layout), and
:class:`ChunkEnsembleAdapter` subclasses :class:`rlmesh.adapters.AdapterBase`,
wrapping the resolved adapter to add only the stateful part. Observation
handling and per-step action conversion (encodings, ranges, clipping) stay
spec-driven; the custom code is the ensemble alone.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

import gymnasium as gym
import rlmesh.adapters as adapt

CHUNK = 8

SPEC = adapt.ModelSpec(
    input={
        "observation.images.image": adapt.Image(
            role=adapt.IMAGE_PRIMARY,
            height=224,
            width=224,
        ),
        "observation.state": adapt.Concat(
            adapt.State(adapt.EEF_POS, dim=3),
            adapt.State(adapt.EEF_ROT, encoding="axis_angle"),
            adapt.State(adapt.GRIPPER_POS, dim=1),
        ),
        "instruction": adapt.Text(),
    },
    # The layout of ONE action; the chunk dimension is adapter business.
    output=adapt.Action(
        adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
        adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
    ),
)


class ChunkEnsembleAdapter(adapt.AdapterBase[Any]):
    """Temporal ensembling over action chunks, layered on a resolved adapter.

    ``transform_action`` receives the model's raw ``(CHUNK, action_dim)``
    output, remembers it, ensembles every live chunk's prediction for the
    current step (newer chunks weighted higher), and hands the resulting
    single action to the resolved adapter for spec-driven conversion.

    Call :meth:`reset` at episode boundaries (wire it to the model worker's
    ``on_reset`` callback) so chunks never ensemble across episodes.
    """

    def __init__(
        self,
        inner: adapt.Adapter,
        horizon: int = CHUNK,
        temperature: float = 0.25,
    ):
        self._inner = inner
        self._horizon = horizon
        self._temperature = temperature
        self._chunks: deque[Any] = deque(maxlen=horizon)

    def transform_obs(self, raw_obs: Mapping[str, Any]) -> dict[str, Any]:
        return self._inner.transform_obs(raw_obs)

    def transform_action(self, raw_action: object) -> Any:
        import numpy as np

        chunk = np.asarray(raw_action, dtype=np.float32).reshape(self._horizon, -1)
        self._chunks.append(chunk)
        # A chunk that is `age` steps old predicted the current step at row
        # `age`; ensemble all live predictions, newest weighted highest.
        rows = [c[age] for age, c in enumerate(reversed(self._chunks))]
        weights = np.exp(-self._temperature * np.arange(len(rows)))
        ensembled = np.average(np.stack(rows), axis=0, weights=weights)
        return self._inner.transform_action(ensembled)

    def reset(self) -> None:
        """Forget pending chunks (call between episodes)."""
        self._chunks.clear()

    def describe(self) -> str:
        return (
            f"temporal ensemble over {self._horizon}-step action chunks, then:\n"
            + self._inner.describe()
        )


def make_adapter(
    tags: adapt.EnvTags,
    observation_space: gym.spaces.Space[Any],
    action_space: gym.spaces.Space[Any],
) -> ChunkEnsembleAdapter:
    """Build this model's adapter for an env: resolve, then add state."""
    return ChunkEnsembleAdapter(
        adapt.resolve(tags, observation_space, action_space, SPEC)
    )


def load_predict_fn() -> Callable[[Mapping[str, Any]], Any]:
    """Return the checkpoint's raw predict callable (stubbed; see smolvla.py).

    A real ACT policy returns its full action chunk here -- the adapter
    consumes the chunk, so no per-step slicing leaks into model code.
    """
    import numpy as np

    def predict(payload: Mapping[str, Any]) -> Any:
        return np.zeros((CHUNK, SPEC.output.dim), dtype=np.float32)

    return predict
