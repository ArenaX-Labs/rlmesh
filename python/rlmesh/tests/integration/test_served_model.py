"""A model served on a session, driven elsewhere through ``Model(address)``."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest


@contextmanager
def _served(model: Any, port: int) -> Iterator[str]:
    """Serve ``model`` on a session at ``port`` in a thread; yield the address
    that reaches it."""
    session = f"127.0.0.1:{port}"
    thread = threading.Thread(
        target=model.serve, args=(session,), kwargs={"name": "policy"}
    )
    thread.start()
    try:
        yield f"{session}/policy"
    finally:
        model.close()
        thread.join(10)
        assert not thread.is_alive()


class ThreeStepEnv:
    """Terminates after three steps, so an episode spans several predicts."""

    def __init__(self) -> None:
        from rlmesh import spaces

        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(2)
        self.steps = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[int, dict[str, object]]:
        _ = seed, options
        self.steps = 0
        return 0, {}

    def step(self, action: object) -> tuple[int, float, bool, bool, dict[str, object]]:
        self.steps += 1
        return 1, 1.0, self.steps >= 3, False, {}

    def close(self) -> None:
        return None


def test_a_served_model_drives_a_local_env_with_numpy_values(session_port: int) -> None:
    import gymnasium as gym
    import numpy as np
    import rlmesh

    class BoxEnv:
        observation_space = gym.spaces.Box(-10.0, 10.0, (3,), np.float32)
        action_space = gym.spaces.Box(-10.0, 10.0, (2,), np.float32)

        def __init__(self) -> None:
            self.n = 0
            self.actions: list[Any] = []

        def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, Any]:
            _ = seed, options
            self.n = 0
            return np.zeros(3, np.float32), {}

        def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
            self.n += 1
            self.actions.append(action)
            return np.full(3, self.n, np.float32), 1.0, self.n >= 3, False, {}

        def close(self) -> None:
            return None

    seen: list[Any] = []

    class Policy(rlmesh.numpy.Model):
        def predict(self, observation: Any) -> Any:
            seen.append(observation.copy())
            return np.asarray([observation[0], -observation[0]], np.float32)

    env = BoxEnv()
    with _served(Policy(), session_port) as address:
        result = rlmesh.numpy.Model(address).run(env, episodes=1)
    assert result.episodes[0].steps == 3
    assert [float(obs[0]) for obs in seen] == [0.0, 1.0, 2.0]
    assert np.asarray(env.actions[1]).tolist() == [1.0, -1.0]


def test_a_served_model_ends_each_episode_once_and_seeds_as_a_local_one(
    session_port: int,
) -> None:
    import rlmesh

    def policy(seeds: list[int | None], ended: list[str]) -> Any:
        class Policy(rlmesh.Model):
            def predict(self, observation: object, context: dict[str, Any]) -> int:
                _ = observation
                seeds.append(context["predict_seed"])
                return 1

            def on_episode_end(self, episode_id: str = "") -> None:
                ended.append(episode_id)

        return Policy()

    local_seeds: list[int | None] = []
    policy(local_seeds, []).run(ThreeStepEnv(), seeds=[11])

    served_seeds: list[int | None] = []
    ended: list[str] = []
    with _served(policy(served_seeds, ended), session_port) as address:
        rlmesh.Model(address).run(ThreeStepEnv(), seeds=[11, 12])

    assert local_seeds == [rlmesh.predict_seed(11, index) for index in range(3)]
    assert served_seeds[:3] == local_seeds
    assert len(ended) == 2 and len(set(ended)) == 2 and all(ended), ended


def test_a_served_model_reads_its_previous_action_through_replayed_history(
    session_port: int,
) -> None:
    """At each re-plan the served model sees the frame executed the step
    before (frame 3 of the last chunk at execution_horizon 4): the route
    negotiates history and every replayed step rides as a numbered row."""
    import gymnasium as gym
    import numpy as np
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
            done = self.n >= 9
            return {"joint_pos": np.full(2, self.n, np.float32)}, 1.0, done, False, {}

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

    with _served(Walk(), session_port) as address:
        Model(address).run(adapt.tag(JointEnv(), tags), episodes=1, execution_horizon=4)
    # Re-plans at steps 0, 4 and 8: the fill, then frame 3 of the chunk
    # executed before each (4 and 104), never that chunk's frame 0.
    assert seen == [0.0, 4.0, 104.0]


def test_old_model_shapes_say_what_replaced_them() -> None:
    import rlmesh

    with pytest.raises(ValueError, match="no longer dials a v1 model endpoint"):
        rlmesh.Model("127.0.0.1:5555")
    with pytest.raises(TypeError, match=r'Model\("host:port/name"\)'):
        rlmesh.RemoteModel("127.0.0.1:5555")
    with pytest.raises(ValueError, match="pass name="):
        rlmesh.Model(lambda obs: 0).serve("lab:7447/bench")
