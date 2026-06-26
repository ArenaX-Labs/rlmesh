"""The neutral pair-driver: rlmesh.run / rlmesh.session / Session.

These pin the Session seam against a tiny in-process env (no server): how a model is
bound to an env and driven, both auto (run) and by hand (reset/predict/step). The
full adapter/remote loop is exercised in the integration suite.
"""

from __future__ import annotations

import pytest
import rlmesh


class _TinyEnv:
    """A minimal local env: one step, then terminates with reward 1.0."""

    def reset(
        self, *, seed: object = None, options: object = None
    ) -> tuple[int, dict[str, object]]:
        return 0, {"seed": seed}

    def step(self, action: object) -> tuple[int, float, bool, bool, dict[str, object]]:
        return 0, 1.0, True, False, {"action": action}

    def close(self) -> None:
        pass


def test_run_drives_a_model_against_a_local_env() -> None:
    result = rlmesh.run(rlmesh.Model(lambda obs: 0), _TinyEnv(), seeds=[0, 1])
    assert isinstance(result, rlmesh.RunResult)
    assert result.num_episodes == 2
    assert result.mean_reward == 1.0  # one step, reward 1.0, then terminates


def test_run_accepts_a_bare_callable() -> None:
    result = rlmesh.run(lambda obs: 0, _TinyEnv())
    assert result.num_episodes == 1


def test_session_manual_drive() -> None:
    sess = rlmesh.session(rlmesh.Model(lambda obs: 0), _TinyEnv())
    assert isinstance(sess, rlmesh.Session)
    obs, _info = sess.reset(seed=0)
    assert sess.done is False
    action = sess.predict(obs)
    _obs, reward, terminated, _trunc, _info = sess.step(action)
    assert reward == 1.0
    assert terminated is True
    assert sess.done is True  # the env terminated after one step
    sess.close()


def test_session_run_matches_top_level_run() -> None:
    result = rlmesh.session(rlmesh.Model(lambda obs: 0), _TinyEnv()).run(seeds=[0])
    assert result.num_episodes == 1


def test_session_is_a_context_manager() -> None:
    with rlmesh.session(rlmesh.Model(lambda obs: 0), _TinyEnv()) as sess:
        obs, _info = sess.reset()
        sess.step(sess.predict(obs))
        assert sess.done is True


def test_as_model_rejects_a_non_model_source() -> None:
    from rlmesh._models.base import _as_model

    with pytest.raises(TypeError, match="predict callable or a policy object"):
        _as_model(object())
