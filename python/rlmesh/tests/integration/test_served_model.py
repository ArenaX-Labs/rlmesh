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
    with pytest.raises(TypeError, match="without a name is gone"):
        rlmesh.Model(lambda obs: 0).serve("lab:7447/bench")


def _perturbed_run(inner: Any) -> list[Any]:
    """Run a Perturb wrapper around ``inner`` for one episode on a Box env;
    return the actions the env received."""
    import gymnasium as gym
    import numpy as np
    import rlmesh

    class BoxEnv:
        observation_space = gym.spaces.Box(-10.0, 10.0, (2,), np.float32)
        action_space = gym.spaces.Box(-10.0, 10.0, (2,), np.float32)

        def __init__(self) -> None:
            self.n = 0
            self.actions: list[Any] = []

        def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, Any]:
            _ = seed, options
            self.n = 0
            return np.zeros(2, np.float32), {}

        def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
            self.n += 1
            self.actions.append(np.asarray(action).tolist())
            return np.full(2, self.n, np.float32), 1.0, self.n >= 3, False, {}

        def close(self) -> None:
            return None

    class Perturb(rlmesh.numpy.Model):
        def __init__(self, inner: Any) -> None:
            self.inner = inner

        def predict(self, observation: Any) -> Any:
            return self.inner.predict(observation + 0.5)

    env = BoxEnv()
    result = Perturb(inner).run(env, episodes=1)
    assert result.episodes[0].steps == 3
    return env.actions


def test_a_wrapper_predicts_through_a_served_inner_model(session_port: int) -> None:
    import numpy as np
    import rlmesh

    seen: list[list[float]] = []

    class Policy(rlmesh.numpy.Model):
        def predict(self, observation: Any) -> Any:
            seen.append(observation.tolist())
            return np.asarray(observation * 2, np.float32)

    with _served(Policy(), session_port) as address:
        inner = rlmesh.numpy.Model(address)
        actions = _perturbed_run(inner)
        # Outside a run of the model that holds it, a served model refuses.
        with pytest.raises(TypeError, match="served elsewhere"):
            inner.predict(np.zeros(2, np.float32))
    # The inner model saw the perturbed observations, per step of the episode.
    assert seen == [[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]]
    assert actions == [[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]]


def test_a_wrapper_predicts_through_a_local_inner_model() -> None:
    import numpy as np
    import rlmesh

    inner = rlmesh.numpy.Model(lambda observation: np.asarray(-observation, np.float32))
    actions = _perturbed_run(inner)
    assert actions == [[-0.5, -0.5], [-1.5, -1.5], [-2.5, -2.5]]


@pytest.mark.parametrize("served", [False, True])
def test_reset_options_reach_each_episode_reset(
    served: bool, session_port: int
) -> None:
    import rlmesh

    received: list[Any] = []

    class OptionsEnv(ThreeStepEnv):
        def reset(
            self, *, seed: int | None = None, options: dict[str, object] | None = None
        ) -> tuple[int, dict[str, object]]:
            received.append(options)
            return super().reset(seed=seed, options=options)

    asked: list[dict[str, Any]] = []

    class Perturb(rlmesh.Model):
        def predict(self, observation: Any) -> Any:
            return 0

        def reset_options(self, episode: Any) -> dict[str, Any]:
            asked.append(dict(episode))
            return {"friction": 0.1 * len(asked)}

    if served:
        with _served(Perturb(), session_port) as address:
            result = rlmesh.Model(address).run(OptionsEnv(), seeds=[3, 4])
    else:
        result = Perturb().run(OptionsEnv(), seeds=[3, 4])
    assert result.num_episodes == 2
    assert [o and round(float(o["friction"]), 3) for o in received] == [0.1, 0.2]
    assert [a["seed"] for a in asked] == [3, 4]
    assert [a["trial_index"] for a in asked] == [0, 1]
    assert all(a["episode_id"] for a in asked)


@contextmanager
def _sandbox_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """A SandboxEnv whose "container" is an env served in this process."""
    import rlmesh
    from rlmesh._sandbox import session as sandbox

    server = rlmesh.EnvServer(ThreeStepEnv(), ":0/env")
    server.start()
    monkeypatch.setattr(
        sandbox,
        "_sandbox_start_env",
        lambda *_a, **_kw: {
            "requested_source": "CartPole-v1",
            "resolved_source": "gym://CartPole-v1",
            "address": server.address,
            "container_id": "container-1",
        },
    )
    monkeypatch.setattr(sandbox, "_sandbox_stop_env", lambda *, container_id: None)
    try:
        with rlmesh.SandboxEnv("CartPole-v1") as env:
            yield env
    finally:
        server.shutdown()


def test_a_run_borrows_a_sandbox_env_and_hands_it_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlmesh

    with _sandbox_env(monkeypatch) as env:
        result = rlmesh.run(rlmesh.Model(lambda obs: 0), env, seeds=[0, 1])
        assert (result.num_episodes, result.total_steps) == (2, 6)
        env.reset(seed=0)
        assert env.step(0)[1] == 1.0


def test_a_sandboxed_model_joins_a_sandbox_env_where_its_container_reaches_it(
    monkeypatch: pytest.MonkeyPatch, session_port: int
) -> None:
    import rlmesh
    from rlmesh._sandbox import session as sandbox

    with (
        _served(rlmesh.Model(lambda obs: 0), session_port) as address,
        _sandbox_env(monkeypatch) as env,
    ):
        model = rlmesh.Model(address)
        # What SandboxModel sets: the model lives in a container.
        model._env_session = address.rsplit("/", 1)[0]  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setattr(
            sandbox, "peer_session", lambda info: info.address.rsplit("/", 1)[0]
        )
        assert model.run(env, seeds=[0, 1]).total_steps == 6
        # The invitation is the peer session, not the one this driver dialed.
        monkeypatch.setattr(
            sandbox, "peer_session", lambda _info: "nowhere.invalid:7447"
        )
        with pytest.raises(ConnectionError, match=r"nowhere\.invalid:7447"):
            model.run(env)
        model.close()
