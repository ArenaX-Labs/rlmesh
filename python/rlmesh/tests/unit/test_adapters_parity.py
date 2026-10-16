"""Regression cases for the adapter language-parity behavior changes."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import rlmesh.adapters as adapt
from rlmesh._models._resolve import resolve_adapter


def _action() -> adapt.Action:
    return adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1))


def _space(width: int = 1) -> gym.spaces.Box:
    return gym.spaces.Box(-1.0, 1.0, (width,), dtype=np.float32)


def test_model_path_uses_tolerant_contract_tags() -> None:
    contract: Any = SimpleNamespace(
        metadata={
            adapt.ENV_METADATA_KEY: {
                "observation": {
                    "state": {"type": "state", "role": adapt.GRIPPER_POS},
                    "future": {"type": "audio", "role": "audio/mic"},
                },
                "action": {"components": [{"role": adapt.ACTION_GRIPPER, "dim": 1}]},
            }
        },
        observation_space=gym.spaces.Dict({"state": _space(), "future": _space()}),
        action_space=_space(),
    )
    spec = adapt.ModelSpec(adapt.State(adapt.GRIPPER_POS), _action())
    assert resolve_adapter(spec, contract, False) is not None
    with pytest.raises(adapt.AdapterResolutionError, match="spec=None"):
        resolve_adapter(None, contract, False)


def test_vector_tag_checks_lane_spaces_and_keeps_metadata() -> None:
    env = SimpleNamespace(
        num_envs=3,
        single_observation_space=_space(),
        single_action_space=_space(),
        observation_space=_space(3),
        action_space=_space(3),
        metadata={"custom": 1},
    )
    tags = adapt.EnvTags(adapt.StateTag(adapt.GRIPPER_POS), _action())
    assert adapt.tag(env, tags) is env
    assert env.metadata["custom"] == 1
    assert adapt.ENV_METADATA_KEY in env.metadata


def test_tag_without_space_validation_wraps_codec_errors_atomically() -> None:
    env = SimpleNamespace(metadata={"custom": 1})
    tags = adapt.EnvTags(adapt.ImageTag(adapt.IMAGE_PRIMARY), _action())
    # Authored tolerance still stops at the codec's closed layout field.
    object.__setattr__(tags.observation, "layout", "invalid")
    with pytest.raises(adapt.AdapterResolutionError):
        adapt.tag(env, tags, validate=False)
    assert env.metadata == {"custom": 1}


def test_advisories_have_value_equality_and_hashes() -> None:
    tags = adapt.EnvTags(adapt.StateTag(adapt.GRIPPER_POS), _action())
    spec = adapt.ModelSpec(
        adapt.Image(adapt.IMAGE_PRIMARY, optional=True, height=2, width=2, channels=3),
        _action(),
    )
    first = adapt.resolve(tags, _space(), _space(), spec).advisories()
    second = adapt.resolve(tags, _space(), _space(), spec).advisories()
    assert first and first == second
    assert set(first) == set(second)
    assert first[0].severity in ("info", "caution")


@pytest.mark.parametrize(
    "leaf",
    [
        adapt.State(adapt.GRIPPER_POS, dtype="typo"),
        adapt.Concat(adapt.GRIPPER_POS, dtype="typo"),
    ],
)
def test_unknown_state_dtype_fails_at_resolution(
    leaf: adapt.State | adapt.Concat,
) -> None:
    tags = adapt.EnvTags(adapt.StateTag(adapt.GRIPPER_POS), _action())
    spec = adapt.ModelSpec(leaf, _action())
    with pytest.raises(adapt.AdapterResolutionError, match="unknown dtype"):
        adapt.resolve(tags, _space(), _space(), spec)


def test_json_is_compact_sorted_utf8_and_preserves_signed_zero() -> None:
    spec = adapt.ModelSpec(adapt.State("x/é", range=(-0.0, 1e-5)), _action())
    text = spec.to_json()
    assert "é" in text and "\\u00e9" not in text
    assert ": " not in text and ", " not in text
    assert "-0.0" in text
    assert list(json.loads(text)) == sorted(json.loads(text))
    assert adapt.ModelSpec.from_json(text) == spec
    tags = adapt.EnvTags(adapt.StateTag("x/é"), _action())
    assert "é" in tags.to_json()
    assert adapt.EnvTags.from_json(tags.to_json()) == tags


def test_equal_custom_encodings_are_probed_once() -> None:
    calls = []

    def identity(value):
        calls.append(1)
        return value

    first = adapt.CustomEncoding("quat_xyzw", identity, identity)
    second = adapt.CustomEncoding("quat_xyzw", identity, identity)
    assert first == second and first is not second
    tags = adapt.EnvTags(adapt.StateTag(adapt.EEF_ROT, encoding="quat_xyzw"), _action())
    spec = adapt.ModelSpec(
        (
            adapt.State(adapt.EEF_ROT, encoding=first),
            adapt.State(adapt.EEF_ROT, encoding=second),
        ),
        _action(),
    )
    adapt.resolve(tags, _space(4), _space(), spec)
    assert len(calls) == 2


def test_previous_action_source_survives_single_part_round_trip() -> None:
    spec = adapt.ModelSpec(adapt.Previous(adapt.ACTION_GRIPPER, fill=2.0), _action())
    assert adapt.ModelSpec.from_dict(spec.to_dict()) == spec
    assert adapt.ModelSpec.from_json(spec.to_json()) == spec
