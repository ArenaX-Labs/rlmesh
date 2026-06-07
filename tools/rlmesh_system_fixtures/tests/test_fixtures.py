from __future__ import annotations

from rlmesh_system_fixtures.registry import (
    list_env_fixtures,
    list_model_fixtures,
    make_env,
    resolve_model,
)
from rlmesh_system_fixtures.trace import fingerprint


def test_counter_env_is_deterministic() -> None:
    env = make_env("counter", {"limit": 2})

    observation, info = env.reset(seed=7)
    assert observation == 0
    assert info["seed"] == 7
    observation, reward, terminated, truncated, info = env.step(0)
    assert observation == 1
    assert reward == 1.0
    assert terminated is False
    assert truncated is False
    assert info["action"] == 0
    observation, reward, terminated, truncated, _info = env.step(0)
    assert observation == 2
    assert reward == 2.0
    assert terminated is True
    assert truncated is False


def test_fixture_registry_discovers_envs_and_models() -> None:
    assert set(list_env_fixtures()) == {"counter", "image-grid"}
    assert {
        "discrete.one",
        "discrete.zero",
        "gymnasium.pendulum_zero_numpy",
        "image_grid.numpy_action",
        "image_grid.torch_action",
        "mujoco.halfcheetah_zero_numpy",
    }.issubset(set(list_model_fixtures()))


def test_image_grid_fingerprint_is_stable() -> None:
    env = make_env("image-grid", {"width": 4, "height": 3, "channels": 3})

    observation, _info = env.reset(seed=11)

    assert fingerprint(observation) == {
        "pixels": {
            "dtype": "uint8",
            "shape": [3, 4, 3],
            "sha256": "5d7e2d9b1dcbc85e7c890036a2cf2f9fe7b66554f2df08cec6aa9c0a25c99c21",
            "type": "ndarray",
        },
        "state": {
            "dtype": "float32",
            "shape": [4],
            "sha256": "d97296dff8fc38f75b6feeaaf30940d74f886f7f8f66a72f2a8cf82b256604fd",
            "type": "ndarray",
        },
    }


def test_model_fixtures_return_supported_action_shapes() -> None:
    assert resolve_model("discrete.zero")(object()) == 0
    action = resolve_model("image_grid.numpy_action")(object())
    assert fingerprint(action) == {
        "dtype": "int64",
        "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc",
        "shape": [1],
        "type": "ndarray",
    }


def test_model_resolver_keeps_dotted_entrypoint_escape_hatch() -> None:
    model = resolve_model("rlmesh_system_fixtures.models.discrete:discrete_zero")

    assert model(object()) == 0
