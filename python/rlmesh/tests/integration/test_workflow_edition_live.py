from __future__ import annotations

from typing import Any

import pytest
import rlmesh
from rlmesh._editions import WORKFLOW_EDITION_ENV_VAR, current_workflow_edition

np = pytest.importorskip("numpy")
pytest.importorskip("gymnasium")

UNKNOWN_EDITION = "2099.01"


class CountEnv:
    def __init__(self) -> None:
        import gymnasium as gym

        self.observation_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._t = 0

    def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, Any]:
        _ = seed, options
        self._t = 0
        return np.zeros(2, np.float32), {}

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
        _ = action
        self._t += 1
        return np.zeros(2, np.float32), 1.0, self._t >= 2, False, {}

    def close(self) -> None:
        return None


@pytest.fixture
def undeclared(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(WORKFLOW_EDITION_ENV_VAR, raising=False)


def model() -> Any:
    from rlmesh.numpy import Model

    return Model(lambda obs: np.zeros(2, np.float32))


def test_a_model_declaring_an_unknown_edition_is_refused_before_it_runs(
    undeclared: None,
) -> None:
    from rlmesh.numpy import Model

    class Future(Model):
        workflow_edition = UNKNOWN_EDITION

        def predict(self, obs: Any) -> Any:
            return np.zeros(2, np.float32)

    with pytest.raises(ValueError, match=UNKNOWN_EDITION):
        _ = Future().run(CountEnv(), seeds=[0])


RETIRED_EDITION = "2026.06"

REFUSAL = f'workflow edition "{RETIRED_EDITION}" is unknown to this rlmesh build'


class RetiredFactory(rlmesh.EnvFactory):
    workflow_edition = RETIRED_EDITION

    def __init__(self) -> None:
        self.made = 0

    def make(self) -> Any:
        self.made += 1
        return CountEnv()


class RetiredEnv(CountEnv):
    workflow_edition = RETIRED_EDITION


def retired_vector_env() -> Any:
    import gymnasium as gym

    class GymCountEnv(CountEnv, gym.Env[Any, Any]):
        def __init__(self) -> None:
            CountEnv.__init__(self)

    class RetiredVectorEnv(gym.vector.SyncVectorEnv):
        workflow_edition = RETIRED_EDITION

    return RetiredVectorEnv([GymCountEnv, GymCountEnv])


def drive(path: str, target: object, **kwargs: Any) -> None:
    if path == "run":
        _ = model().run(target, seeds=[0], **kwargs)
    elif path == "rlmesh.run":
        _ = rlmesh.run(model(), target, episodes=1, **kwargs)
    elif path == "random":
        _ = rlmesh.Model(rlmesh.RANDOM_SAMPLE).run(target, episodes=1, **kwargs)
    elif path == "bind":
        with model().bind(target, **kwargs) as binding:
            binding.reset()
    else:
        [outcome] = model().run_routes([target], episodes=1, **kwargs)
        if isinstance(outcome, BaseException):
            raise outcome


LOCAL_PATHS = ["run", "rlmesh.run", "random", "bind", "run_routes"]


@pytest.mark.parametrize("path", LOCAL_PATHS)
def test_a_local_run_refuses_a_factory_declaring_a_retired_edition(
    undeclared: None, path: str
) -> None:
    factory = RetiredFactory()
    with pytest.raises(ValueError, match=REFUSAL):
        drive(path, factory)
    assert factory.made == 0


@pytest.mark.parametrize("path", LOCAL_PATHS)
def test_a_local_run_refuses_an_env_class_declaring_a_retired_edition(
    undeclared: None, path: str
) -> None:
    with pytest.raises(ValueError, match=REFUSAL):
        drive(path, RetiredEnv())


@pytest.mark.parametrize("path", ["run", "random", "run_routes"])
def test_a_local_run_refuses_a_vector_env_declaring_a_retired_edition(
    undeclared: None, path: str
) -> None:
    env = retired_vector_env()
    try:
        with pytest.raises(ValueError, match=REFUSAL):
            drive(path, env)
    finally:
        env.close()


@pytest.mark.parametrize("path", LOCAL_PATHS)
def test_a_local_run_drives_a_factory_declaring_this_builds_edition(
    undeclared: None, path: str
) -> None:
    class CurrentFactory(rlmesh.EnvFactory):
        workflow_edition = current_workflow_edition()

        def make(self) -> Any:
            return CountEnv()

    drive(path, CurrentFactory())


def test_the_environment_variable_outranks_a_local_factory_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(WORKFLOW_EDITION_ENV_VAR, current_workflow_edition())
    factory = RetiredFactory()
    drive("run", factory)
    assert factory.made == 1
