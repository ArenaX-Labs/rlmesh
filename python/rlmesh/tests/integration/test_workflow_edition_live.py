from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import rlmesh
from rlmesh import _editions
from rlmesh._editions import WORKFLOW_EDITION_ENV_VAR, current_workflow_edition
from rlmesh._rlmesh import ServeOptions

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

np = pytest.importorskip("numpy")
pytest.importorskip("gymnasium")

UNKNOWN_EDITION = "2099.01"

COHORT = rlmesh.build_info().workflow_edition


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
def declared_edition() -> str:
    return current_workflow_edition()


@pytest.fixture
def undeclared(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(WORKFLOW_EDITION_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_editions, "_pyproject_cache", {})


@pytest.fixture
def served_env(declared_edition: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.delenv(WORKFLOW_EDITION_ENV_VAR, raising=False)
    server = rlmesh.EnvServer(
        CountEnv(),
        ":0/env",
        options=ServeOptions(workflow_edition=declared_edition),
    )
    server.start()
    try:
        yield server.address
    finally:
        server.shutdown()


def model() -> Any:
    from rlmesh.numpy import Model

    return Model(lambda obs: np.zeros(2, np.float32))


@pytest.mark.parametrize("random_baseline", [False, True])
def test_public_wrappers_forward_the_call_pin(
    served_env: str,
    declared_edition: str,
    monkeypatch: pytest.MonkeyPatch,
    random_baseline: bool,
) -> None:
    monkeypatch.setenv(WORKFLOW_EDITION_ENV_VAR, UNKNOWN_EDITION)
    policy = rlmesh.Model(rlmesh.RANDOM_SAMPLE) if random_baseline else model()
    with policy.bind(served_env, workflow_edition=declared_edition) as binding:
        binding.reset()
    result = rlmesh.run(
        policy, served_env, episodes=1, workflow_edition=declared_edition
    )
    assert len(result.episodes) == 1


def test_both_pins_negotiate_the_declared_edition(
    served_env: str, declared_edition: str
) -> None:
    from rlmesh.numpy import RemoteEnv

    client = RemoteEnv(served_env, workflow_edition=declared_edition)
    try:
        assert client.selected_workflow_edition == COHORT
    finally:
        client.close()


def test_a_cohort_declaration_pins_the_exact_build(served_env: str) -> None:
    from rlmesh.numpy import RemoteEnv

    client = RemoteEnv(served_env, workflow_edition=COHORT)
    try:
        assert client.selected_workflow_edition == COHORT
    finally:
        client.close()


def test_an_undeclared_client_still_negotiates_the_env_declaration(
    served_env: str, undeclared: None, recwarn: pytest.WarningsRecorder
) -> None:
    from rlmesh.numpy import RemoteEnv

    client = RemoteEnv(served_env)
    try:
        assert client.selected_workflow_edition == COHORT
    finally:
        client.close()
    assert [w for w in recwarn if "workflow edition" in str(w.message)] == []


def test_a_pyproject_bare_base_declaration_runs_a_local_model(
    undeclared: None, tmp_path: Path, recwarn: pytest.WarningsRecorder
) -> None:
    _ = (tmp_path / "pyproject.toml").write_text(
        f'[tool.rlmesh]\nworkflow_edition = "{current_workflow_edition()}"\n',
        encoding="utf-8",
    )
    result = model().run(CountEnv(), seeds=[0])
    assert len(result.episodes) == 1
    assert [
        w for w in recwarn if "no workflow edition declared" in str(w.message)
    ] == []


def test_run_with_a_matching_pin_completes(
    served_env: str, declared_edition: str
) -> None:
    result = model().run(served_env, seeds=[0], workflow_edition=declared_edition)
    assert len(result.episodes) == 1
    assert result.episodes[0].steps == 2


def test_run_refuses_an_edition_this_build_cannot_drive(served_env: str) -> None:
    with pytest.raises(ValueError, match=UNKNOWN_EDITION):
        _ = model().run(served_env, seeds=[0], workflow_edition=UNKNOWN_EDITION)


def test_serving_refuses_an_edition_this_build_cannot_drive() -> None:
    with pytest.raises(ValueError, match=UNKNOWN_EDITION):
        _ = ServeOptions(workflow_edition=UNKNOWN_EDITION)


def test_the_env_var_pins_a_run_that_declares_nothing(
    served_env: str, declared_edition: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(WORKFLOW_EDITION_ENV_VAR, declared_edition)
    from rlmesh.numpy import RemoteEnv

    client = RemoteEnv(served_env)
    try:
        assert client.selected_workflow_edition == COHORT
    finally:
        client.close()
    assert len(model().run(served_env, seeds=[0]).episodes) == 1


RETIRED_EDITION = "2026.06"

REFUSAL = (
    f'runtime cannot drive workflow edition "{RETIRED_EDITION}"; this build implements'
)


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


def test_the_call_keyword_outranks_a_local_factory_declaration(
    undeclared: None, declared_edition: str
) -> None:
    factory = RetiredFactory()
    drive("run", factory, workflow_edition=declared_edition)
    assert factory.made == 1
