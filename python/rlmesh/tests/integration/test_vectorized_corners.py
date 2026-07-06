"""End-to-end: the batched predict corners on a vectorized spec'd route.

A spec'd model driven by the native in-process loop against a vectorized
(``num_envs=2``) tagged env resolves its route at connect (the same
``resolve_adapter`` the served path runs at ``ResolveAdapter``), so the engine
dispatches the most-specific predict corner: ``predict_batch`` for an
un-chunked route, ``predict_chunk_batch`` when an ``execution_horizon > 1`` is
pinned. Regression for the route never resolving on ``run_local`` (every
predict fell into the spec-less branch and the batched corners were
unreachable).
"""

from __future__ import annotations

from typing import Any

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("gymnasium")

EPISODE_STEPS = 4
NUM_ENVS = 2


def _tags() -> Any:
    import rlmesh.adapters as adapt

    return adapt.EnvTags(
        observation={"eef_pos": adapt.StateTag(role=adapt.EEF_POS)},
        action=adapt.Action(adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3)),
    )


def _spec() -> Any:
    import rlmesh.adapters as adapt

    return adapt.ModelSpec(
        input={"state": adapt.State(role=adapt.EEF_POS)},
        output=adapt.Action(adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3)),
    )


class VecEnv:
    """Two-lane vector env: Dict obs with one tagged state, 3-dim Box action."""

    def __init__(self) -> None:
        import gymnasium as gym

        self.num_envs = NUM_ENVS
        self.single_observation_space = gym.spaces.Dict(
            {"eef_pos": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32)}
        )
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.metadata = {"autoreset_mode": "NextStep"}
        self._t = 0
        self.seen_actions: list[Any] = []

    def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, Any]:
        _ = seed, options
        self._t = 0
        return {"eef_pos": np.zeros((NUM_ENVS, 3), np.float32)}, {}

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any, Any]:
        self.seen_actions.append(np.asarray(actions))
        self._t += 1
        done = self._t >= EPISODE_STEPS
        return (
            {"eef_pos": np.zeros((NUM_ENVS, 3), np.float32)},
            [1.0] * NUM_ENVS,
            [done] * NUM_ENVS,
            [False] * NUM_ENVS,
            {},
        )

    def close(self) -> None:
        return None


def _serve_env(env: VecEnv) -> Any:
    import rlmesh

    try:
        server = rlmesh.EnvServer(env, "127.0.0.1:0", tags=_tags())
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise
    server.start()
    return server


def _policy_cls(calls: dict[str, int], batch_shapes: list[Any]) -> Any:
    from rlmesh.numpy import Model

    class Policy(Model):
        spec = _spec()

        def predict(self, obs: Any) -> Any:
            calls["predict"] += 1
            return np.zeros(3, np.float32)

        def predict_batch(self, obs: Any) -> Any:
            calls["predict_batch"] += 1
            batch_shapes.append(obs["state"].shape)
            return np.full((obs["state"].shape[0], 3), 0.25, np.float32)

        def predict_chunk(self, obs: Any) -> Any:
            calls["predict_chunk"] += 1
            return np.zeros((2, 3), np.float32)

        def predict_chunk_batch(self, obs: Any) -> Any:
            calls["predict_chunk_batch"] += 1
            batch_shapes.append(obs["state"].shape)
            n = obs["state"].shape[0]
            chunk = np.zeros((n, 2, 3), np.float32)
            chunk[:, 0, :] = 0.25
            chunk[:, 1, :] = 0.75
            return chunk

    return Policy


def test_vectorized_route_dispatches_predict_batch() -> None:
    """One batched forward per step with the fused obs carrying the batch axis;
    the single-lane corners never drive the route (the resolve-time stateless
    probe accounts for every ``predict`` call), and the adapter's action path
    hands the env the batched 2-lane action."""
    calls = {
        k: 0
        for k in ("predict", "predict_batch", "predict_chunk", "predict_chunk_batch")
    }
    batch_shapes: list[Any] = []
    env = VecEnv()
    server = _serve_env(env)
    try:
        _policy_cls(calls, batch_shapes)()._run_local_for_episodes(
            server.address, max_episodes=1
        )
    finally:
        server.shutdown()

    assert calls["predict_batch"] == EPISODE_STEPS
    assert calls["predict_chunk"] == 0
    assert calls["predict_chunk_batch"] == 0
    assert all(shape == (NUM_ENVS, 3) for shape in batch_shapes)
    assert env.seen_actions[0].shape == (NUM_ENVS, 3)
    np.testing.assert_allclose(env.seen_actions[0], 0.25)


def test_run_public_api_drives_vectorized_env_end_to_end() -> None:
    """The public ``Model.run`` on a tagged local vector env: served on loopback,
    driven by the native loop, batched chunk corner dispatched, and the
    RunResult carries the runtime's per-episode report."""
    import rlmesh.adapters as adapt

    calls = {
        k: 0
        for k in ("predict", "predict_batch", "predict_chunk", "predict_chunk_batch")
    }
    batch_shapes: list[Any] = []
    env = adapt.tag(VecEnv(), _tags(), validate=False)
    result = _policy_cls(calls, batch_shapes)().run(
        env, max_episodes=2, execution_horizon=2
    )

    assert calls["predict_chunk_batch"] == EPISODE_STEPS // 2
    assert result.num_episodes == NUM_ENVS
    assert result.mean_reward == float(EPISODE_STEPS)
    assert all(e.steps == EPISODE_STEPS for e in result.episodes)
    assert all(e.terminated and not e.truncated for e in result.episodes)


def test_chunk_only_model_collapses_to_batched_corner_at_horizon_1() -> None:
    """A model defining ONLY the chunk corners still drives an un-chunked
    (execution_horizon=1) vectorized route batched: corner synthesis derives
    ``predict_batch`` from ``predict_chunk_batch`` (one-step decode, frame 0),
    so the model's own chunk-batch corner runs once per step. The derived
    corner also serves the resolve-time stateless probe, so the assertions are
    behavioral (frames on the env) rather than an exact call count."""
    from rlmesh.numpy import Model

    calls = {"predict_chunk": 0, "predict_chunk_batch": 0}

    class ChunkOnly(Model):
        spec = _spec()

        def predict_chunk(self, obs: Any) -> Any:
            calls["predict_chunk"] += 1
            return np.full((2, 3), 0.25, np.float32)

        def predict_chunk_batch(self, obs: Any) -> Any:
            calls["predict_chunk_batch"] += 1
            n = obs["state"].shape[0]
            chunk = np.zeros((n, 2, 3), np.float32)
            chunk[:, 0, :] = 0.25
            chunk[:, 1, :] = 0.75
            return chunk

    env = VecEnv()
    server = _serve_env(env)
    try:
        ChunkOnly()._run_local_for_episodes(server.address, max_episodes=1)
    finally:
        server.shutdown()

    assert calls["predict_chunk_batch"] >= EPISODE_STEPS
    assert calls["predict_chunk"] == 0
    assert len(env.seen_actions) == EPISODE_STEPS
    for action in env.seen_actions:
        np.testing.assert_allclose(action, 0.25)


def test_vectorized_chunked_route_dispatches_predict_chunk_batch() -> None:
    """One batched forward per 2-step chunk (open-loop replay in between), the
    un-chunked corners idle, and the runtime replaying each chunk in model
    order: frame 0 then frame 1."""
    calls = {
        k: 0
        for k in ("predict", "predict_batch", "predict_chunk", "predict_chunk_batch")
    }
    batch_shapes: list[Any] = []
    env = VecEnv()
    server = _serve_env(env)
    try:
        _policy_cls(calls, batch_shapes)()._run_local_for_episodes(
            server.address, max_episodes=1, execution_horizon=2
        )
    finally:
        server.shutdown()

    assert calls["predict_chunk_batch"] == EPISODE_STEPS // 2
    assert calls["predict_batch"] == 0
    assert calls["predict_chunk"] == 0
    assert all(shape == (NUM_ENVS, 3) for shape in batch_shapes)
    assert len(env.seen_actions) == EPISODE_STEPS
    np.testing.assert_allclose(env.seen_actions[0], 0.25)
    np.testing.assert_allclose(env.seen_actions[1], 0.75)
    np.testing.assert_allclose(env.seen_actions[2], 0.25)
    np.testing.assert_allclose(env.seen_actions[3], 0.75)
