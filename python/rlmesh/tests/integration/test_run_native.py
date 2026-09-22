"""The unified native ``Model.run``: seeds, caps, and the honest signature.

``run()`` drives every env shape through the native runtime loop. These pin
the Session-parity features the runtime enforces itself: explicit per-episode
seeds (``episode_seeds``, echoed on the ``RunResult``), the step cap
(runtime-truncated episodes), and that the Session-only knobs (``hooks`` /
``instruction`` / ``view``) are not ``Model.run`` parameters at all -- the
module-level ``rlmesh.run`` refuses them for a local model.
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


def test_run_prefetch_lead_predicts_the_next_chunk_from_a_stale_observation() -> None:
    """``prefetch_lead`` reaches the runtime driver: at chunk 3 / lead 1 the
    second chunk is asked for while one replay frame is still queued, from the
    observation after step 1 rather than step 3, and the chunk prefetched at
    each episode's tail is discarded so the next episode re-plans from its
    reset observation. Every episode still completes and scores."""
    import gymnasium as gym
    from rlmesh.numpy import Model

    class StepEnv(CountEnv):
        """Observation ``[t, 0]``: the step count, so a chunk corner can
        report which observation it was conditioned on."""

        def __init__(self) -> None:
            super().__init__(episode_len=6, final_info={"is_success": True})
            self.observation_space = gym.spaces.Box(0.0, 100.0, (2,), np.float32)

        def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, Any]:
            super().reset(seed=seed, options=options)
            return np.array([0.0, 0.0], np.float32), {}

        def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
            _, reward, done, truncated, info = super().step(action)
            return np.array([self._t, 0.0], np.float32), reward, done, truncated, info

    def run(lead: int) -> tuple[Any, list[int]]:
        seen: list[int] = []

        class ChunkPolicy(Model):
            native_chunk = 3

            def predict_chunk(self, obs: Any) -> Any:
                seen.append(int(obs[0]))
                return np.zeros((3, 2), np.float32)

        try:
            result = ChunkPolicy().run(
                StepEnv(), max_episodes=2, execution_horizon=3, prefetch_lead=lead
            )
        except ConnectionError as exc:
            if "Operation not permitted" in str(exc):
                pytest.skip("local tcp bind is not permitted in this environment")
            raise
        return result, seen

    result, seen = run(1)
    assert [e.steps for e in result.episodes] == [6, 6]
    assert all(e.success is True for e in result.episodes)
    assert result.success_rate == 1.0
    # The prefetch at each episode's tail (from step 4, with the last frame
    # still queued) may or may not reach the model before the terminal step
    # lands; every other chunk is pinned: the reset observation opens each
    # episode (the tail prefetch was discarded) and the second chunk was
    # conditioned on step 1, two steps before the synchronous loop's step 3.
    assert [obs for obs in seen if obs != 4] == [0, 1, 0, 1]

    _, synchronous = run(0)
    assert synchronous == [0, 3, 0, 3]


def test_run_holds_a_declared_native_chunk_to_its_length() -> None:
    """The spec-less native path measures the chunk against a declared K the
    way a Session does: a short chunk is a model error, not a silent early
    re-plan that only the adapted route would have caught."""
    from rlmesh.numpy import Model

    class ShortChunk(Model):
        native_chunk = 8

        def predict_chunk(self, obs: Any) -> Any:
            return np.zeros((2, 2), np.float32)

    try:
        with pytest.raises(
            Exception, match="native_chunk=8 but its chunk corner returned 2"
        ):
            ShortChunk().run(CountEnv(), max_episodes=1, execution_horizon=4)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise


def test_run_delivers_every_step_to_a_stacked_model_at_any_horizon() -> None:
    """The native path carries replayed steps to the model as observation
    history, so a frame-stacked model sees the same windows under ``run`` as
    under a Session, at execution_horizon 1 and above."""
    import rlmesh.adapters as adapt
    from rlmesh.numpy import Model

    class CameraEnv:
        observation_space = rlmesh.spaces.Box(0, 255, shape=(2, 2, 3), dtype="uint8")
        action_space = rlmesh.spaces.Box(-1.0, 1.0, shape=(2,), dtype="float32")

        def __init__(self) -> None:
            self.n = 0

        def _frame(self) -> Any:
            return np.full((2, 2, 3), self.n, np.uint8)

        def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, Any]:
            self.n = 0
            return self._frame(), {}

        def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
            self.n += 1
            return self._frame(), 1.0, self.n >= 7, False, {}

        def close(self) -> None:
            return None

    out = adapt.Action(adapt.Actuator("x/action", dim=2))
    tags = adapt.EnvTags(observation=adapt.ImageTag(adapt.IMAGE_PRIMARY), action=out)

    class Stacked(Model):
        native_chunk = 3
        spec = adapt.ModelSpec(
            input=adapt.Image(adapt.IMAGE_PRIMARY, stack=3), output=out
        )

        def load(self) -> None:
            self.windows: list[tuple[int, ...]] = []

        def predict_chunk(self, observation: Any) -> Any:
            # The stacked axis leads: (3, H, W, C). Record which frames it holds.
            self.windows.append(tuple(int(frame[0, 0, 0]) for frame in observation))
            return np.zeros((3, 2), np.float32)

    def windows(path: str, horizon: int) -> list[tuple[int, ...]]:
        model = Stacked()
        env = adapt.tag(CameraEnv(), tags)
        try:
            if path == "session":
                model.session(env, execution_horizon=horizon).run(max_episodes=1)
            else:
                model.run(env, max_episodes=1, execution_horizon=horizon)
        except ConnectionError as exc:
            if "Operation not permitted" in str(exc):
                pytest.skip("local tcp bind is not permitted in this environment")
            raise
        return model.windows

    every_step = windows("native", 1)
    assert every_step[:3] == [(0, 0, 0), (0, 0, 1), (0, 1, 2)]
    # Re-plans at steps 0, 3 and 6: the window at each holds the consecutive
    # frames, not the decision points, on both paths.
    assert windows("native", 3) == [every_step[0], every_step[3], every_step[6]]
    assert windows("session", 3) == windows("native", 3)


def test_run_max_episode_steps_truncates_via_the_runtime() -> None:
    env = CountEnv(episode_len=0)
    try:
        result = _model().run(env, max_episodes=2, max_episode_steps=4)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise

    assert result.num_episodes == 2
    assert len(result.episodes) == 2
    assert all(e.steps == 4 for e in result.episodes)
    assert all(e.truncated and not e.terminated for e in result.episodes)
    assert all(e.reward == 4.0 for e in result.episodes)
    # A truncation echoed back by the env must not be counted twice or cut the
    # next episode short: two distinct episodes, each driven for all four steps.
    assert [e.index for e in result.episodes] == [0, 1]
    assert len(env.seen_seeds) == 2


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


def test_session_only_knobs_are_not_run_parameters() -> None:
    import inspect

    params = inspect.signature(rlmesh.Model.run).parameters
    assert not {"hooks", "instruction", "view"} & params.keys()
    for knob in ({"hooks": rlmesh.RunHooks()}, {"instruction": "pick up the cube"}):
        with pytest.raises(TypeError, match=r"session\(\) option"):
            rlmesh.run(_model(), CountEnv(), **knob)
    with pytest.raises(TypeError, match=r"session\(\) option"):
        rlmesh.run(_model(), CountEnv(), view="terminal")


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


def test_run_feeds_a_previous_action_part_the_frame_executed_the_step_before() -> None:
    """A Go2-style policy reading its own last command sees, at every re-plan,
    the raw chunk frame the runtime executed at the step before: the last frame
    of the previous chunk at execution_horizon 7, the only one at 1, and the
    fill at step 0, on the native and the session paths alike."""
    import gymnasium as gym
    import rlmesh.adapters as adapt
    from rlmesh.numpy import Model

    sdk = adapt.GO2.joints
    default_pose = (0.0, 0.8, -1.5) * 4

    class Go2Env:
        observation_space = gym.spaces.Dict(
            {
                "ang_vel": gym.spaces.Box(-20.0, 20.0, (3,), np.float32),
                "base_quat": gym.spaces.Box(-1.0, 1.0, (4,), np.float32),
                "command": gym.spaces.Box(-3.0, 3.0, (3,), np.float32),
                "joint_pos": gym.spaces.Box(-10.0, 10.0, (12,), np.float32),
                "joint_vel": gym.spaces.Box(-30.0, 30.0, (12,), np.float32),
            }
        )
        action_space = gym.spaces.Box(-10.0, 10.0, (12,), np.float32)

        def __init__(self) -> None:
            self.n = 0

        def _obs(self) -> dict[str, Any]:
            return {
                "ang_vel": np.zeros(3, np.float32),
                "base_quat": np.array([1.0, 0.0, 0.0, 0.0], np.float32),
                "command": np.zeros(3, np.float32),
                "joint_pos": np.full(12, self.n, np.float32),
                "joint_vel": np.zeros(12, np.float32),
            }

        def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, Any]:
            self.n = 0
            return self._obs(), {}

        def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
            self.n += 1
            return self._obs(), 1.0, self.n >= 15, False, {}

        def close(self) -> None:
            return None

    tags = adapt.EnvTags(
        observation={
            "ang_vel": adapt.StateTag(adapt.BASE_ANG_VEL, frame="robot_base"),
            "base_quat": adapt.StateTag(
                adapt.BASE_ROT, encoding="quat_wxyz", frame="world"
            ),
            "command": adapt.StateTag(adapt.COMMAND_BASE_VEL, frame="robot_base"),
            "joint_pos": adapt.StateTag(adapt.JOINT_POS, labels=sdk),
            "joint_vel": adapt.StateTag(adapt.JOINT_VEL, labels=sdk),
        },
        action=adapt.Action(adapt.Actuator(adapt.ACTION_JOINT_POS, dim=12, labels=sdk)),
    )

    class Walk(Model):
        native_chunk = 7
        spec = adapt.ModelSpec(
            input={
                "obs": adapt.Concat(
                    adapt.State(adapt.BASE_ANG_VEL, frame="robot_base", scale=0.25),
                    adapt.State(adapt.BASE_ROT, encoding="gravity_xyz", frame="world"),
                    adapt.State(adapt.COMMAND_BASE_VEL, frame="robot_base"),
                    adapt.State(
                        adapt.JOINT_POS,
                        labels=sdk,
                        offset=tuple(-q for q in default_pose),
                    ),
                    adapt.State(adapt.JOINT_VEL, labels=sdk, scale=0.05),
                    adapt.Previous(adapt.ACTION_JOINT_POS),
                    clip=(-100.0, 100.0),
                )
            },
            output=adapt.Action(
                adapt.Actuator(
                    adapt.ACTION_JOINT_POS,
                    dim=12,
                    labels=sdk,
                    scale=(0.125, 0.25, 0.25) * 4,
                    offset=default_pose,
                )
            ),
        )

        def load(self) -> None:
            self.previous: list[float] = []
            self.calls = 0

        def predict_chunk(self, observation: Any) -> Any:
            obs = observation["obs"]
            assert obs.shape == (45,)
            # The previous-action slot is uniform by construction; record it.
            assert len(set(obs[33:].tolist())) == 1
            self.previous.append(float(obs[33]))
            # Frame k of call c is 5c + k + 1 on every joint (inside the
            # container clip): the scaled, offset env command is a different
            # number, so a slot holding anything but the raw frame is visible.
            chunk = np.full((7, 12), 5 * self.calls + 1, np.float32)
            chunk += np.arange(7, dtype=np.float32)[:, None]
            self.calls += 1
            return chunk

    def previous(path: str, horizon: int) -> list[float]:
        model = Walk()
        env = adapt.tag(Go2Env(), tags)
        try:
            if path == "session":
                model.session(env, execution_horizon=horizon).run(max_episodes=1)
            else:
                model.run(env, max_episodes=1, execution_horizon=horizon)
        except ConnectionError as exc:
            if "Operation not permitted" in str(exc):
                pytest.skip("local tcp bind is not permitted in this environment")
            raise
        return model.previous

    # Horizon 1: predict k+1 sees frame 0 of call k (5k + 1).
    every_step = [0.0] + [5.0 * call + 1 for call in range(14)]
    assert previous("native", 1) == every_step
    assert previous("session", 1) == every_step
    # Horizon 7: re-plans at 0, 7 and 14 see the fill, then frame 6 of the
    # chunk executed before (7 and 12), never frame 0 of it (1 and 6).
    assert previous("native", 7) == [0.0, 7.0, 12.0]
    assert previous("session", 7) == [0.0, 7.0, 12.0]
