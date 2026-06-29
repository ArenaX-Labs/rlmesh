"""The neutral pair-driver: rlmesh.run / rlmesh.session / Session.

These pin the Session seam against a tiny in-process env (no server): how a model is
bound to an env and driven, both auto (run) and by hand (reset/predict/step). The
full adapter/remote loop is exercised in the integration suite.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import rlmesh
from rlmesh._models._view import ViewerDriver


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


class _ForeverEnv:
    """A local env that never terminates on its own (only a cap/skip can end it)."""

    def reset(
        self, *, seed: object = None, options: object = None
    ) -> tuple[int, dict[str, object]]:
        return 0, {}

    def step(self, action: object) -> tuple[int, float, bool, bool, dict[str, object]]:
        return 0, 0.0, False, False, {}

    def close(self) -> None:
        pass


class _SkipDriver:
    """A stand-in ViewerDriver that asks to end each episode after its first step."""

    def __init__(self) -> None:
        self._steps = 0

    def feed(self, *, steps: int, **_: Any) -> None:
        self._steps = steps

    def consume_skip(self) -> bool:
        return self._steps >= 1

    def close(self) -> None:
        pass


def test_viewer_skip_truncates_episode_without_failing() -> None:
    sess = rlmesh.session(rlmesh.Model(lambda obs: 0), _ForeverEnv())
    sess._view_driver = cast(ViewerDriver, _SkipDriver())
    obs, _info = sess.reset(seed=0)
    _obs, _reward, terminated, truncated, _info = sess.step(sess.predict(obs))
    assert truncated is True
    assert terminated is False
    assert sess.done is True
    sess.close()


def test_viewer_skip_advances_run_through_all_seeds() -> None:
    sess = rlmesh.session(rlmesh.Model(lambda obs: 0), _ForeverEnv())
    sess._view_driver = cast(ViewerDriver, _SkipDriver())
    result = sess.run(seeds=[0, 1, 2])
    assert result.num_episodes == 3
    assert all(e.truncated and not e.terminated for e in result.episodes)
    assert all(e.steps == 1 for e in result.episodes)


def test_session_is_a_context_manager() -> None:
    with rlmesh.session(rlmesh.Model(lambda obs: 0), _TinyEnv()) as sess:
        obs, _info = sess.reset()
        sess.step(sess.predict(obs))
        assert sess.done is True


def test_as_model_rejects_a_non_model_source() -> None:
    from rlmesh._models.base import as_model

    with pytest.raises(TypeError, match="predict callable or a policy object"):
        as_model(object())


# ---------------------------------------------------------------------------
# instruction= override injection (placement + container shape)
# ---------------------------------------------------------------------------


def _spec(input_tree: Any) -> object:
    import rlmesh.adapters as adapt

    return adapt.ModelSpec(
        input=input_tree,
        output=adapt.Action(
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0))
        ),
    )


def test_text_placements_covers_every_placement_and_container() -> None:
    import rlmesh.adapters as adapt
    from rlmesh._models._eval import TextPlacement, text_placements

    # bare-root: the whole payload IS the text leaf (empty placement)
    assert text_placements(_spec(adapt.Text(role=adapt.INSTRUCTION))) == (
        TextPlacement((), False),
    )
    # top-level dict key, both container shapes
    assert text_placements(
        _spec({"prompt": adapt.Text(role=adapt.INSTRUCTION, container="str")})
    ) == (TextPlacement(("prompt",), False),)
    assert text_placements(
        _spec({"prompt": adapt.Text(role=adapt.INSTRUCTION, container="list")})
    ) == (TextPlacement(("prompt",), True),)
    # nested dict placement
    assert text_placements(
        _spec({"lang": {"instr": adapt.Text(role=adapt.INSTRUCTION)}})
    ) == (TextPlacement(("lang", "instr"), False),)
    # tuple placement (positional)
    assert text_placements(_spec((adapt.Text(role=adapt.INSTRUCTION),))) == (
        TextPlacement((0,), False),
    )


def test_text_placements_empty_for_specless_models() -> None:
    from rlmesh import NO_ADAPTER
    from rlmesh._models._eval import text_placements

    assert text_placements(None) == ()
    assert text_placements(NO_ADAPTER) == ()


def _inject(placements: tuple[Any, ...], payload: Any) -> Any:
    """Run _predict_step's injection (adapter=None hands the payload through)."""
    from rlmesh._models._eval import _predict_step

    captured: dict[str, Any] = {}

    def predict(p: Any) -> int:
        captured["payload"] = p
        return 0

    _predict_step(predict, payload, None, "do the task", placements, None, None, None)
    return captured["payload"]


def test_instruction_injects_into_a_bare_root_text_input() -> None:
    from rlmesh._models._eval import TextPlacement

    # The whole payload is the text leaf; the override replaces it outright.
    assert _inject((TextPlacement((), False),), "old") == "do the task"


def test_instruction_injects_into_a_nested_text_input() -> None:
    from rlmesh._models._eval import TextPlacement

    out = _inject((TextPlacement(("lang", "instr"), False),), {"lang": {"instr": "x"}})
    assert out == {"lang": {"instr": "do the task"}}


def test_instruction_injects_list_for_list_container() -> None:
    from rlmesh._models._eval import TextPlacement

    out = _inject((TextPlacement(("prompt",), True),), {"prompt": ["x"]})
    assert out == {"prompt": ["do the task"]}


def test_instruction_injection_does_not_mutate_the_source_payload() -> None:
    from rlmesh._models._eval import TextPlacement

    source = {"lang": {"instr": "x"}}
    _inject((TextPlacement(("lang", "instr"), False),), source)
    assert source == {"lang": {"instr": "x"}}  # injected into a rebuilt copy
