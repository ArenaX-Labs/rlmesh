"""The neutral pair-driver: rlmesh.run / rlmesh.session / Session.

These pin the Session seam against a tiny in-process env (no server): how a model is
bound to an env and driven, both auto (run) and by hand (reset/predict/step). The
full adapter/remote loop is exercised in the integration suite.
"""

from __future__ import annotations

from typing import Any

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
    from rlmesh._models._eval import _text_placements, _TextPlacement

    # bare-root: the whole payload IS the text leaf (empty placement)
    assert _text_placements(_spec(adapt.Text(role=adapt.INSTRUCTION))) == (
        _TextPlacement((), False),
    )
    # top-level dict key, both container shapes
    assert _text_placements(
        _spec({"prompt": adapt.Text(role=adapt.INSTRUCTION, container="str")})
    ) == (_TextPlacement(("prompt",), False),)
    assert _text_placements(
        _spec({"prompt": adapt.Text(role=adapt.INSTRUCTION, container="list")})
    ) == (_TextPlacement(("prompt",), True),)
    # nested dict placement
    assert _text_placements(
        _spec({"lang": {"instr": adapt.Text(role=adapt.INSTRUCTION)}})
    ) == (_TextPlacement(("lang", "instr"), False),)
    # tuple placement (positional)
    assert _text_placements(_spec((adapt.Text(role=adapt.INSTRUCTION),))) == (
        _TextPlacement((0,), False),
    )


def test_text_placements_empty_for_specless_models() -> None:
    from rlmesh import NO_ADAPTER
    from rlmesh._models._eval import _text_placements

    assert _text_placements(None) == ()
    assert _text_placements(NO_ADAPTER) == ()


def _inject(placements: tuple[Any, ...], payload: Any) -> Any:
    """Run _predict_step's injection (adapter=None hands the payload through)."""
    from rlmesh._models._eval import _predict_step

    captured: dict[str, Any] = {}

    def predict(p: Any) -> int:
        captured["payload"] = p
        return 0

    _predict_step(predict, payload, None, "do the task", placements, None, None)
    return captured["payload"]


def test_instruction_injects_into_a_bare_root_text_input() -> None:
    from rlmesh._models._eval import _TextPlacement

    # The whole payload is the text leaf; the override replaces it outright.
    assert _inject((_TextPlacement((), False),), "old") == "do the task"


def test_instruction_injects_into_a_nested_text_input() -> None:
    from rlmesh._models._eval import _TextPlacement

    out = _inject((_TextPlacement(("lang", "instr"), False),), {"lang": {"instr": "x"}})
    assert out == {"lang": {"instr": "do the task"}}


def test_instruction_injects_list_for_list_container() -> None:
    from rlmesh._models._eval import _TextPlacement

    out = _inject((_TextPlacement(("prompt",), True),), {"prompt": ["x"]})
    assert out == {"prompt": ["do the task"]}


def test_instruction_injection_does_not_mutate_the_source_payload() -> None:
    from rlmesh._models._eval import _TextPlacement

    source = {"lang": {"instr": "x"}}
    _inject((_TextPlacement(("lang", "instr"), False),), source)
    assert source == {"lang": {"instr": "x"}}  # injected into a rebuilt copy
