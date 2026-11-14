"""A model's action or an env's observation that does not fit its space.

Either is the peer breaking its contract: a ``ValueError`` naming who (the
model's or env's class, or a gym id) and what (the space and the misfit),
the same in process and across a session, not an ``[INTERNAL]`` failure.
"""

from __future__ import annotations

import threading
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import rlmesh

ACTION_MISFIT = (
    'model "OneFloat" returned an action that doesn\'t fit the action space '
    "Box(-10.0, 10.0, (4,), float32): data is 4 bytes but shape and dtype require "
    "exactly 16"
)


class BoxEnv:
    observation_space = gym.spaces.Box(-10.0, 10.0, (3,), np.float32)
    action_space = gym.spaces.Box(-10.0, 10.0, (4,), np.float32)

    def __init__(self, observation_size: int = 3) -> None:
        self.observation_size = observation_size

    def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, Any]:
        _ = seed, options
        return np.zeros(self.observation_size, np.float32), {}

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
        _ = action
        return np.zeros(self.observation_size, np.float32), 0.0, True, False, {}

    def close(self) -> None:
        return None


class OneFloat(rlmesh.numpy.Model):
    """Returns one float where the env's Box(4,) wants four."""

    def predict(self, observation: Any) -> Any:
        _ = observation
        return 0.0


class Fits(rlmesh.numpy.Model):
    def predict(self, observation: Any) -> Any:
        _ = observation
        return np.zeros(4, np.float32)


def test_a_models_misfit_action_names_the_model_and_space_in_process() -> None:
    with pytest.raises(ValueError) as failed:
        OneFloat().run(BoxEnv(), episodes=1)
    assert str(failed.value) == ACTION_MISFIT
    assert not isinstance(failed.value, rlmesh.RLMeshException)


def test_a_served_models_misfit_action_is_its_contract_not_internal(
    session_port: int,
) -> None:
    model = OneFloat()
    session = f"127.0.0.1:{session_port}"
    thread = threading.Thread(
        target=model.serve, args=(session,), kwargs={"name": "policy"}
    )
    thread.start()
    try:
        with pytest.raises(ValueError) as failed:
            rlmesh.numpy.Model(f"{session}/policy").run(BoxEnv(), episodes=1)
    finally:
        model.close()
        thread.join(10)
    assert str(failed.value) == ACTION_MISFIT
    assert "[INTERNAL]" not in str(failed.value)


def test_an_envs_misfit_observation_names_the_env_and_space() -> None:
    server = rlmesh.EnvServer(BoxEnv(observation_size=1), ":0/env")
    server.start()
    try:
        with rlmesh.numpy.RemoteEnv(server.address) as env:
            with pytest.raises(ValueError) as failed:
                env.reset()
        text = (
            'env "BoxEnv" returned an observation that doesn\'t fit its declared '
            "observation space Box(-10.0, 10.0, (3,), float32): data is 4 bytes but "
            "shape and dtype require exactly 12"
        )
        assert str(failed.value) == text
        # The env is to blame: its class says so beside ValueError.
        assert isinstance(failed.value, rlmesh.EnvironmentException)

        with pytest.raises(ValueError) as failed:
            Fits().run(server.address, episodes=1)
        assert text in str(failed.value)
        # Only a closed connection gets the hint about one.
        assert "connection closed" not in str(failed.value)
    finally:
        server.shutdown()
