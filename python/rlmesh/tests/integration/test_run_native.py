"""The unified native ``Model.run``: seeds, caps, and Session-only rejections.

``run()`` drives every env shape through the native runtime loop. These pin
the Session-parity features the runtime enforces itself: explicit per-episode
seeds (``episode_seeds``, echoed on the ``RunResult``), the step cap
(runtime-truncated episodes), and the loud rejection of the Session-only
knobs (``hooks`` / ``instruction`` / ``view``).
"""

from __future__ import annotations

from typing import Any

import pytest
import rlmesh

np = pytest.importorskip("numpy")
pytest.importorskip("gymnasium")


class CountEnv:
    """Single env: Box obs, terminates after ``episode_len`` steps (0 = never).

    ``final_info``, when given, is emitted as the terminal step's info dict --
    the shape a Gymnasium env uses to report ``is_success`` / ``success``.
    """

    def __init__(
        self, episode_len: int = 3, final_info: dict[str, Any] | None = None
    ) -> None:
        import gymnasium as gym

        self.observation_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.episode_len = episode_len
        self.final_info = final_info or {}
        self.seen_seeds: list[int | None] = []
        self._t = 0

    def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, Any]:
        _ = options
        self.seen_seeds.append(seed)
        self._t = 0
        return np.zeros(2, np.float32), {}

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
        _ = action
        self._t += 1
        done = self.episode_len > 0 and self._t >= self.episode_len
        info = dict(self.final_info) if done else {}
        return np.zeros(2, np.float32), 1.0, done, False, info

    def close(self) -> None:
        return None


def _model() -> Any:
    from rlmesh.numpy import Model

    return Model(lambda obs: np.zeros(2, np.float32))


def test_run_seeds_set_episode_count_and_echo_on_result() -> None:
    env = CountEnv(episode_len=3)
    try:
        result = _model().run(env, seeds=[7, 8])
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise

    assert result.num_episodes == 2
    assert [e.seed for e in result.episodes] == [7, 8]
    assert env.seen_seeds[:2] == [7, 8]
    assert all(e.steps == 3 and e.reward == 3.0 for e in result.episodes)
    assert all(e.terminated and not e.truncated for e in result.episodes)
    assert all(e.predict_ms > 0.0 and e.step_ms > 0.0 for e in result.episodes)


def test_run_surfaces_the_session_telemetry_aggregate() -> None:
    env = CountEnv(episode_len=3)
    try:
        result = _model().run(env, seeds=[7])
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise

    series = {(row.op, row.metric) for row in result.telemetry}
    assert ("model.predict", "rpc.total") in series
    assert ("env.step", "rpc.total") in series
    assert ("runner.round", "rpc.total") in series
    for row in result.telemetry:
        assert row.count > 0
        assert row.unit in ("ms", "bytes", "count")
        assert row.avg >= 0.0 and row.p50 <= row.p95 <= row.p99
    # The formatter renders one aligned line per row plus a header.
    table = result.format_telemetry()
    assert len(table.splitlines()) == len(result.telemetry) + 1
    assert table.splitlines()[0].split()[:2] == ["op", "metric"]


def test_run_empty_seeds_returns_an_empty_result() -> None:
    result = _model().run(CountEnv(), seeds=[])
    assert result.num_episodes == 0
    assert result.telemetry == ()
    assert "no telemetry" in result.format_telemetry()


def test_run_reports_the_envs_success_signal() -> None:
    failed = CountEnv(episode_len=2, final_info={"is_success": False})
    succeeded = CountEnv(episode_len=2, final_info={"success": 1})
    try:
        failed_result = _model().run(failed, max_episodes=1)
        succeeded_result = _model().run(succeeded, max_episodes=1)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise

    assert failed_result.episodes[0].success is False
    assert failed_result.success_rate == 0.0
    assert succeeded_result.episodes[0].success is True
    assert succeeded_result.success_rate == 1.0


def test_run_trust_entrypoints_is_scoped_to_the_call() -> None:
    model = _model()
    try:
        model.run(CountEnv(), max_episodes=1, trust_entrypoints=True)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise
    assert model._trust_entrypoints is False


def test_run_truncates_an_over_produced_dict_chunk_to_the_horizon() -> None:
    import gymnasium as gym
    from rlmesh.numpy import Model

    class DictActionEnv(CountEnv):
        def __init__(self) -> None:
            super().__init__(episode_len=3)
            self.action_space = gym.spaces.Dict(
                {"arm": gym.spaces.Box(-1.0, 1.0, (2,), np.float32)}
            )
            self.seen: list[Any] = []

        def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
            self.seen.append(action)
            return super().step(action)

    class ChunkPolicy(Model):
        def predict_chunk(self, obs: Any) -> Any:
            chunk = np.zeros((5, 2), np.float32)
            chunk[:, 0] = np.arange(5, dtype=np.float32) / 10.0
            return {"arm": chunk}

    env = DictActionEnv()
    try:
        result = ChunkPolicy().run(env, max_episodes=1, execution_horizon=3)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise

    assert result.episodes[0].steps == 3
    assert len(env.seen) == 3
    np.testing.assert_allclose(
        [action["arm"][0] for action in env.seen], [0.0, 0.1, 0.2], atol=1e-6
    )


def test_run_max_episode_steps_truncates_via_the_runtime() -> None:
    env = CountEnv(episode_len=0)
    try:
        result = _model().run(env, max_episodes=2, max_episode_steps=4)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise

    assert result.num_episodes == 2
    assert all(e.steps == 4 for e in result.episodes)
    assert all(e.truncated and not e.terminated for e in result.episodes)
    assert all(e.reward == 4.0 for e in result.episodes)


def test_run_on_a_driven_handle_explains_the_session_conflict() -> None:
    """A RemoteEnv handle that was driven holds the env's single session slot;
    run() (which dials the handle's address) surfaces a clear pointer instead
    of the raw wire error."""
    try:
        server = rlmesh.EnvServer(CountEnv(), "127.0.0.1:0")
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise
    server.start()
    try:
        handle = rlmesh.numpy.RemoteEnv(server.address)
        handle.reset()
        with pytest.raises(RuntimeError, match=r"close\(\) the handle"):
            _model().run(handle, max_episodes=1)
        handle.close()
    finally:
        server.shutdown()


def test_run_rejects_session_only_knobs() -> None:
    with pytest.raises(ValueError, match=r"session\(\)\.run"):
        _model().run(CountEnv(), hooks=rlmesh.RunHooks())
    with pytest.raises(ValueError, match=r"session\(\)\.run"):
        _model().run(CountEnv(), instruction="pick up the cube")


def test_session_served_env_context_carries_stable_episode_identity() -> None:
    from rlmesh.numpy import Model

    seen: list[dict[str, Any]] = []

    def predict(observation: Any, context: dict[str, Any]) -> Any:
        seen.append(dict(context))
        return np.zeros(2, np.float32)

    try:
        server = rlmesh.EnvServer(CountEnv(episode_len=3), host="127.0.0.1", port=0)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise
    server.start()
    try:
        with Model(predict).session(server.address) as sess:
            sess.run(seeds=[7], max_episodes=1)
    finally:
        server.shutdown()

    assert len(seen) == 3
    episode_ids = {str(context["episode_id"]) for context in seen}
    assert len(episode_ids) == 1, f"episode_id changed mid-episode: {seen}"
    assert all(episode_ids), f"episode_id blank on some step: {seen}"
    assert [context["episode_seed"] for context in seen] == [7, 7, 7]
