"""A declared edition, end to end: served env, served model, dialed client, native run."""

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

#: A well-formed `YYYY.MM` no build implements.
UNKNOWN_EDITION = "2099.01"

#: The exact spelling this build offers -- what a bare-base declaration selects
#: on it: the sealed base on a release, the cohort on a prerelease or dev build.
COHORT = rlmesh.build_info().workflow_edition


class CountEnv:
    """Two-step env with a Box obs/action seam."""

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
    """The edition both peers declare: the bare base, the stateVersion value.

    A base-level ceiling: on a prerelease or dev build, whose only offer is its
    cohort spelling, it selects that cohort (``COHORT``) rather than refusing.
    """
    return current_workflow_edition()


@pytest.fixture
def undeclared(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Nothing declared anywhere: no env var, no manifest, warning re-armed."""
    monkeypatch.delenv(WORKFLOW_EDITION_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_editions, "_warned", False)
    monkeypatch.setattr(_editions, "_pyproject_cache", {})


@pytest.fixture
def served_env(declared_edition: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """An env server that declares ``declared_edition`` on every handshake."""
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
    # The process declares an edition no env accepts: the call's pin is what
    # lets these connect.
    with policy.bind(served_env, workflow_edition=declared_edition) as binding:
        binding.reset()
    result = rlmesh.run(
        policy, served_env, episodes=1, workflow_edition=declared_edition
    )
    assert len(result.episodes) == 1


def test_both_pins_negotiate_the_declared_edition(
    served_env: str, declared_edition: str
) -> None:
    # The env's ServeOptions declaration and the client's own pin are the two
    # WANTs of this leg; the session runs at what they agree on.
    from rlmesh.numpy import RemoteEnv

    client = RemoteEnv(served_env, workflow_edition=declared_edition)
    try:
        assert client.selected_workflow_edition == COHORT
    finally:
        client.close()


def test_a_cohort_declaration_pins_the_exact_build(served_env: str) -> None:
    # The other accepted spelling: this build's cohort, which pins to exactly
    # this moving build rather than to the contract.
    from rlmesh.numpy import RemoteEnv

    client = RemoteEnv(served_env, workflow_edition=COHORT)
    try:
        assert client.selected_workflow_edition == COHORT
    finally:
        client.close()


def test_an_undeclared_client_still_negotiates_the_env_declaration(
    served_env: str, undeclared: None, recwarn: pytest.WarningsRecorder
) -> None:
    # The no-op case: with nothing pinned on this side the session lands on the
    # same edition, so declaring changes nothing that was already working. A
    # client handle has no home for a declaration, so it floats quietly.
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
    # The docs' own snippet, on whatever build this is: a bare base in the
    # project manifest selects this build's spelling of it on the loopback env
    # server `run()` stands up, instead of refusing every session on a
    # prerelease or dev build.
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
    # The operator's process-wide override reaches the wire without the program
    # naming an edition anywhere.
    monkeypatch.setenv(WORKFLOW_EDITION_ENV_VAR, declared_edition)
    from rlmesh.numpy import RemoteEnv

    client = RemoteEnv(served_env)
    try:
        assert client.selected_workflow_edition == COHORT
    finally:
        client.close()
    assert len(model().run(served_env, seeds=[0]).episodes) == 1
