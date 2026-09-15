"""RemoteModel: drive a served policy against a served env in the symmetric loop."""

from __future__ import annotations

import socket
import threading
import time
from typing import Any, cast

import pytest


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return cast(int, sock.getsockname()[1])
    finally:
        sock.close()


class TinyEnv:
    def __init__(self) -> None:
        from rlmesh import spaces

        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(2)

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[int, dict[str, object]]:
        _ = seed, options
        return 0, {}

    def step(self, action: object) -> tuple[int, float, bool, bool, dict[str, object]]:
        return 1, 1.0, True, False, {"action": action}

    def close(self) -> None:
        return None


def _serve_env(env: object) -> Any:
    import rlmesh
    from rlmesh._server import EnvLike as ServedEnv

    try:
        server = rlmesh.EnvServer(cast("ServedEnv", env), host="127.0.0.1", port=0)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise
    server.start()
    return server


def _serve_model(address: str, seen: list[object]) -> threading.Thread:
    import rlmesh

    def predict(observation: object) -> int:
        seen.append(observation)
        return 1

    def run() -> None:
        rlmesh.Model(predict).serve(
            address, options=rlmesh.ServeOptions(allow_remote_shutdown=True)
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _connect_model_with_retry(address: str, env: object) -> Any:
    import rlmesh

    deadline = time.monotonic() + 5.0
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            return rlmesh.session(rlmesh.RemoteModel(address), env)
        except Exception as exc:  # retry until the server is up
            last_error = exc
            time.sleep(0.05)
    raise AssertionError(f"model server at {address} never came up") from last_error


def test_remote_model_drives_a_served_env_in_the_symmetric_loop() -> None:
    import rlmesh

    seen: list[object] = []
    env_server = _serve_env(TinyEnv())
    model_address = f"127.0.0.1:{_free_port()}"
    _serve_model(model_address, seen)

    try:
        env = rlmesh.RemoteEnv(env_server.address)
        sess = _connect_model_with_retry(model_address, env)

        obs, _info = sess.reset(seed=0)
        steps = 0
        while not sess.done and steps < 5:
            action = sess.predict(obs)
            obs, _reward, _terminated, _truncated, _info = sess.step(action)
            steps += 1

        # The policy was consulted, returned its action (1), and the env stepped.
        assert seen, "served policy predict was never called"
        assert steps == 1
        sess.close()
        env.close()
    finally:
        env_server.shutdown()


def test_session_requires_an_env_contract() -> None:
    import rlmesh

    model = rlmesh.RemoteModel(f"127.0.0.1:{_free_port()}")
    with pytest.raises(TypeError, match="env_contract"):
        rlmesh.session(model, object())


class ThreeStepEnv(TinyEnv):
    """Terminates after three steps, so an episode spans several predicts."""

    def __init__(self) -> None:
        super().__init__()
        self.steps = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[int, dict[str, object]]:
        _ = seed, options
        self.steps = 0
        return 0, {}

    def step(self, action: object) -> tuple[int, float, bool, bool, dict[str, object]]:
        self.steps += 1
        return 1, 1.0, self.steps >= 3, False, {"action": action}


def _serve_policy(address: str, model: Any) -> threading.Thread:
    import rlmesh

    def run() -> None:
        model.serve(address, options=rlmesh.ServeOptions(allow_remote_shutdown=True))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _drive(session: Any, seeds: list[int]) -> None:
    for seed in seeds:
        obs, _info = session.reset(seed=seed)
        steps = 0
        while not session.done and steps < 10:
            obs, *_rest = session.step(session.predict(obs))
            steps += 1


def test_served_model_fires_on_episode_end_once_per_episode_id() -> None:
    """The served twin of the hand-driven `on_episode_end` case in test_lifecycle.

    A `RemoteModel` is the only client that can see an episode boundary (the
    server cannot), so `reset()` must carry that edge over the wire: the previous
    episode's id rides a `ResetAdapter`, which fires the served model's
    `on_episode_end` for exactly that id -- and the last episode's edge arrives at
    close, so two episodes yield two distinct ids.
    """
    import rlmesh

    ended: list[str] = []

    class Policy(rlmesh.Model):
        def predict(self, observation: object, context: dict[str, Any]) -> int:
            _ = observation, context
            return 1

        def reset(self, episode_id: str = "") -> None:
            ended.append(episode_id)

    env_server = _serve_env(ThreeStepEnv())
    model_address = f"127.0.0.1:{_free_port()}"
    _serve_policy(model_address, Policy())

    try:
        env = rlmesh.RemoteEnv(env_server.address)
        sess = _connect_model_with_retry(model_address, env)
        _drive(sess, [1, 2])
        sess.close()
        env.close()
    finally:
        env_server.shutdown()

    assert len(ended) == 2, ended
    assert all(ended) and len(set(ended)) == 2, ended


def test_predict_seed_is_the_same_local_and_served() -> None:
    """`predict_seed` is a pure function of (episode seed, re-plan ordinal), so a
    model driven locally and the same model driven over the wire must be handed
    the same sequence -- otherwise a seeded policy is not reproducible across the
    two paths the SDK offers.
    """
    import rlmesh

    def policy(seen: list[int | None]) -> Any:
        class Policy(rlmesh.Model):
            def predict(self, observation: object, context: dict[str, Any]) -> int:
                _ = observation
                seen.append(context["predict_seed"])
                return 1

        return Policy()

    local_seeds: list[int | None] = []
    with policy(local_seeds).session(ThreeStepEnv()) as sess:
        _drive(sess, [11])

    served_seeds: list[int | None] = []
    env_server = _serve_env(ThreeStepEnv())
    model_address = f"127.0.0.1:{_free_port()}"
    _serve_policy(model_address, policy(served_seeds))
    try:
        env = rlmesh.RemoteEnv(env_server.address)
        sess = _connect_model_with_retry(model_address, env)
        _drive(sess, [11])
        sess.close()
        env.close()
    finally:
        env_server.shutdown()

    assert local_seeds == [rlmesh.predict_seed(11, index) for index in range(3)]
    assert served_seeds == local_seeds


def test_served_model_reads_its_previous_action_through_remote_model_replay() -> None:
    """RemoteModel replays chunk frames client-side; a served policy reading
    its own last command still sees, at each re-plan, the raw frame executed
    at the step before (frame 3 of the last chunk at execution_horizon 4),
    because the route negotiates history and every replayed step rides as a
    numbered row."""
    import gymnasium as gym
    import numpy as np
    import rlmesh
    import rlmesh.adapters as adapt
    from rlmesh.numpy import Model

    class JointEnv:
        observation_space = gym.spaces.Dict(
            {"joint_pos": gym.spaces.Box(-10.0, 10.0, (2,), np.float32)}
        )
        action_space = gym.spaces.Box(-1000.0, 1000.0, (2,), np.float32)

        def __init__(self) -> None:
            self.n = 0

        def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, Any]:
            self.n = 0
            return {"joint_pos": np.zeros(2, np.float32)}, {}

        def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
            self.n += 1
            return (
                {"joint_pos": np.full(2, self.n, np.float32)},
                1.0,
                self.n >= 9,
                False,
                {},
            )

        def close(self) -> None:
            return None

    out = adapt.Action(adapt.Actuator(adapt.ACTION_JOINT_POS, dim=2))
    tags = adapt.EnvTags(
        observation={"joint_pos": adapt.StateTag(adapt.JOINT_POS)}, action=out
    )
    seen: list[float] = []

    class Walk(Model):
        native_chunk = 4
        spec = adapt.ModelSpec(
            input=adapt.Concat(
                adapt.State(adapt.JOINT_POS, dim=2),
                adapt.Previous(adapt.ACTION_JOINT_POS),
            ),
            output=out,
        )

        def load(self) -> None:
            self.calls = 0

        def predict_chunk(self, observation: Any) -> Any:
            seen.append(float(observation[2]))
            chunk = np.full((4, 2), 100 * self.calls + 1, np.float32)
            chunk += np.arange(4, dtype=np.float32)[:, None]
            self.calls += 1
            return chunk

    env_server = _serve_env(adapt.tag(JointEnv(), tags))
    model_address = f"127.0.0.1:{_free_port()}"

    def serve() -> None:
        Walk().serve(
            model_address, options=rlmesh.ServeOptions(allow_remote_shutdown=True)
        )

    threading.Thread(target=serve, daemon=True).start()
    try:
        env = rlmesh.RemoteEnv(env_server.address)
        deadline = time.monotonic() + 5.0
        last_error: BaseException | None = None
        while True:
            try:
                sess = rlmesh.session(
                    rlmesh.RemoteModel(model_address), env, execution_horizon=4
                )
                break
            except (ConnectionError, OSError) as exc:
                last_error = exc
                if time.monotonic() >= deadline:
                    raise AssertionError(
                        f"model server never came up: {last_error}"
                    ) from exc
                time.sleep(0.05)
        obs, _info = sess.reset(seed=0)
        while not sess.done:
            action = sess.predict(obs)
            obs, _reward, _terminated, _truncated, _info = sess.step(action)
        # Re-plans at steps 0, 4 and 8: the fill, then frame 3 of the chunk
        # executed before each (4 and 104), never that chunk's frame 0.
        assert seen == [0.0, 4.0, 104.0]
        sess.close()
        env.close()
    finally:
        env_server.shutdown()
