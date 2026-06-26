"""End-to-end: a vectorized (num_envs>1) served model whose policy returns one
N-stacked array action.

The numpy/torch/jax bridges encode that array into a single native ``rlmesh.Tensor``
of shape ``(N, *act_shape)`` (``from_array`` -> ``Tensor``). The model worker must
split that native tensor into N per-lane wire values. A native ``Tensor`` is not
Python-iterable, so the batch converter has to slice it along the lane axis rather
than fall back to ``try_iter`` — the path the Rust unit tests can't reach because
they never go through the real bridge. Regression for the non-iterable-PyTensor
batch split.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest


class BoxVectorEnv:
    """Two-lane vector env with Box observation/action spaces.

    A Box action is what makes the policy's numpy return cross the bridge as a
    native ``Tensor`` (a Discrete action would come back as plain Python ints and
    never exercise the tensor-split path).
    """

    def __init__(self) -> None:
        from rlmesh import spaces

        self.num_envs = 2
        self.single_observation_space = spaces.Box(
            0.0, 1.0, shape=(2,), dtype="float32"
        )
        self.single_action_space = spaces.Box(0.0, 1.0, shape=(2,), dtype="float32")
        # Vectorized runtime sessions require NEXT_STEP autoreset (the gymnasium
        # vector default); without it the runtime refuses the num_envs>1 session.
        self.metadata = {"autoreset_mode": "NextStep"}
        self.seen_actions: list[Any] = []

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[Any, dict[str, object]]:
        _ = seed, options
        return np.zeros((self.num_envs, 2), dtype=np.float32), {}

    def step(
        self, actions: Any
    ) -> tuple[Any, list[float], list[bool], list[bool], dict[str, object]]:
        self.seen_actions.append(np.asarray(actions))
        obs = np.zeros((self.num_envs, 2), dtype=np.float32)
        # Terminate both lanes so the single bounded episode ends after one step.
        return obs, [1.0, 1.0], [True, True], [False, False], {}

    def close(self) -> None:
        return None


def _serve_env(env: object) -> Any:
    import rlmesh
    from rlmesh._server import VectorServerEnvLike as ServedEnv

    try:
        server = rlmesh.EnvServer(cast("ServedEnv", env), host="127.0.0.1", port=0)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise
    server.start()
    return server


def test_vectorized_numpy_model_splits_stacked_action() -> None:
    from rlmesh import numpy as rlmesh_numpy

    env = BoxVectorEnv()
    server = _serve_env(env)

    predict_calls: list[Any] = []

    def predict(observation: Any) -> Any:
        predict_calls.append(np.asarray(observation))
        # The natural vectorized return: ONE (N, *act_shape) array — not a list of
        # N per-lane arrays. The bridge turns this into a single native Tensor.
        return np.full((env.num_envs, 2), 0.5, dtype=np.float32)

    try:
        rlmesh_numpy.Model(predict).run_local_for_episodes(
            server.address, max_episodes=1
        )
    finally:
        server.shutdown()

    # The policy saw a 2-lane observation...
    assert predict_calls, "served policy predict was never called"
    assert predict_calls[0].shape == (2, 2)
    # ...and the stacked action was split + decoded back to a 2-lane env action.
    assert env.seen_actions, "env never received the decoded action"
    np.testing.assert_allclose(
        env.seen_actions[0], np.full((2, 2), 0.5, dtype=np.float32)
    )
