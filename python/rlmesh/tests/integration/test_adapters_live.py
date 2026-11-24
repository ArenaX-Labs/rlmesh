from __future__ import annotations

import importlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import pytest
import rlmesh
import rlmesh.adapters as adapt
from rlmesh.numpy import Model, RemoteEnv

if TYPE_CHECKING:
    import numpy as np

    NumpyArray = np.ndarray[Any, Any]


def _tags() -> adapt.EnvTags:
    return adapt.EnvTags(
        observation={
            "cam": adapt.ImageTag(role=adapt.IMAGE_PRIMARY),
            "eef_pos": adapt.StateTag(role=adapt.EEF_POS),
            "eef_quat": adapt.StateTag(role=adapt.EEF_ROT, encoding="quat_xyzw"),
            "gripper": adapt.StateTag(role=adapt.GRIPPER_POS),
            "instruction": adapt.TextTag(role=adapt.INSTRUCTION),
        },
        action=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
            adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
            clip=(-1.0, 1.0),
        ),
    )


def _model_spec() -> adapt.ModelSpec:
    return adapt.ModelSpec(
        input={
            "image": adapt.Image(role=adapt.IMAGE_PRIMARY, height=8, width=8),
            "state": adapt.Concat(
                adapt.EEF_POS,
                adapt.State(adapt.EEF_ROT, encoding="axis_angle"),
                adapt.GRIPPER_POS,
                container="list",
            ),
            "instruction": adapt.Text(role=adapt.INSTRUCTION),
        },
        output=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
            adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
        ),
    )


class TinyArmEnv:
    def __init__(self) -> None:
        import gymnasium as gym
        import numpy as np

        self.metadata: dict[str, Any] = {"render_modes": []}
        self.observation_space = gym.spaces.Dict(
            {
                "cam": gym.spaces.Box(0, 255, (8, 8, 3), np.uint8),
                "eef_pos": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
                "eef_quat": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
                "gripper": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
                "instruction": gym.spaces.Text(max_length=64),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)
        self._t = 0
        self.last_action: NumpyArray | None = None
        self.actions: list[NumpyArray] = []

    def _obs(self) -> dict[str, Any]:
        import numpy as np

        rng = np.random.default_rng(self._t)
        quat = rng.normal(size=4).astype(np.float32)
        quat /= np.linalg.norm(quat)
        return {
            "cam": rng.integers(0, 256, (8, 8, 3), dtype=np.uint8),
            "eef_pos": rng.normal(size=3).astype(np.float32),
            "eef_quat": quat,
            "gripper": np.array([0.02], dtype=np.float32),
            "instruction": "pick up the cube",
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        _ = seed, options
        self._t = 0
        return self._obs(), {}

    def step(
        self, action: object
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        import numpy as np

        self.last_action = cast("NumpyArray", np.asarray(action, dtype=np.float32))
        self.actions.append(self.last_action)
        self._t += 1
        return self._obs(), 1.0, self._t >= 3, False, {}

    def close(self) -> None:
        return None


def test_adapted_model_runs_against_tagged_server() -> None:
    pytest.importorskip("numpy")

    tags = _tags()
    spec = _model_spec()
    env_obj = TinyArmEnv()

    seen: dict[str, Any] = {"episode_ends": 0, "payload_keys": None}

    def predict(payload: dict[str, Any]) -> Any:
        import numpy as np

        seen["payload_keys"] = sorted(payload)
        return np.zeros(spec.output.dim, dtype=np.float32)

    def on_episode_end() -> None:
        seen["episode_ends"] = cast(int, seen["episode_ends"]) + 1

    server = rlmesh.EnvServer(env_obj, ":0/env", tags=tags)
    server.start()
    try:
        with RemoteEnv(server.address) as client:
            recovered = adapt.EnvTags.from_metadata(client.env_contract.metadata or {})
        assert recovered == tags

        Model(predict, spec=spec, on_episode_end=on_episode_end).run(
            server.address, episodes=1
        )
    finally:
        server.shutdown()

    assert seen["payload_keys"] == ["image", "instruction", "state"]
    assert env_obj.last_action is not None
    assert tuple(env_obj.last_action.shape) == (7,)
    assert cast(int, seen["episode_ends"]) >= 1


class TinyArmFactory(rlmesh.EnvFactory):
    tags = _tags()

    def make(self, **kwargs: Any) -> Any:
        _ = kwargs
        return TinyArmEnv()


def _arm_predict(spec: adapt.ModelSpec, captured: dict[str, Any]) -> Any:
    def predict(payload: dict[str, Any]) -> Any:
        import numpy as np

        captured["keys"] = sorted(payload)
        return np.zeros(spec.output.dim, dtype=np.float32)

    return predict


def test_envfactory_make_stamps_its_tags_on_the_env() -> None:
    pytest.importorskip("numpy")
    env = TinyArmFactory().make()
    assert adapt.EnvTags.from_metadata(getattr(env, "metadata", {}) or {}) == _tags()


def test_adapted_model_runs_against_local_tagged_env() -> None:
    pytest.importorskip("numpy")
    spec = _model_spec()
    env_obj = TinyArmEnv()
    captured: dict[str, Any] = {}

    tagged = adapt.tag(env_obj, _tags())
    result = Model(_arm_predict(spec, captured), spec=spec).run(tagged, episodes=1)

    assert result.num_episodes == 1
    assert captured["keys"] == ["image", "instruction", "state"]
    assert env_obj.last_action is not None
    assert tuple(env_obj.last_action.shape) == (7,)
    adapter_rows = [r for r in result.telemetry if r.metric == "predict.adapter"]
    assert adapter_rows and adapter_rows[0].avg > 0.0


@pytest.mark.parametrize(
    ("container", "expected"),
    [("str", "follow the override"), ("list", ["follow the override"])],
)
def test_instruction_override_reaches_predict_in_declared_shape(
    container: str, expected: object
) -> None:
    pytest.importorskip("numpy")

    spec = adapt.ModelSpec(
        input={
            "image": adapt.Image(role=adapt.IMAGE_PRIMARY, height=8, width=8),
            "instruction": adapt.Text(role=adapt.INSTRUCTION, container=container),  # pyright: ignore[reportArgumentType]
        },
        output=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
            adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
        ),
    )
    env_obj = TinyArmEnv()
    seen: dict[str, Any] = {"instruction": None}

    def predict(payload: dict[str, Any]) -> Any:
        import numpy as np

        seen["instruction"] = payload["instruction"]
        return np.zeros(spec.output.dim, dtype=np.float32)

    tagged = adapt.tag(env_obj, _tags())
    model = Model(predict, spec=spec)
    model.run(tagged, episodes=1, instruction="follow the override")
    assert seen["instruction"] == expected

    model.run(tagged, episodes=1)
    assert seen["instruction"] != expected, "the override is scoped to its run"


_FRAMEWORK_LEAF = {
    "numpy": ("numpy", "ndarray"),
    "torch": ("torch", "Tensor"),
    "jax": ("jax", "Array"),
}


def _fw_zeros(framework: str, dim: int) -> Any:
    if framework == "torch":
        import torch

        return torch.zeros(dim)
    if framework == "jax":
        import jax.numpy as jnp

        return jnp.zeros(dim)
    import numpy as np

    return np.zeros(dim, dtype=np.float32)


@pytest.mark.parametrize("framework", list(_FRAMEWORK_LEAF))
def test_spec_model_drives_local_numpy_env_for_any_framework(framework: str) -> None:
    pytest.importorskip("numpy")
    mod_name, type_name = _FRAMEWORK_LEAF[framework]
    leaf_type = getattr(pytest.importorskip(mod_name), type_name)
    model_cls = importlib.import_module(f"rlmesh.{framework}").Model

    spec = _model_spec()
    env_obj = TinyArmEnv()
    captured: dict[str, Any] = {}

    def predict(payload: dict[str, Any]) -> Any:
        captured["image_type"] = type(payload["image"])
        return _fw_zeros(framework, spec.output.dim)

    tagged = adapt.tag(env_obj, _tags())
    result = model_cls(predict, spec=spec).run(tagged, episodes=1)

    assert result.num_episodes == 1
    assert issubclass(captured["image_type"], leaf_type), captured["image_type"]
    assert env_obj.last_action is not None
    assert tuple(env_obj.last_action.shape) == (7,)


def test_adapted_model_runs_against_env_factory() -> None:
    pytest.importorskip("numpy")
    spec = _model_spec()
    captured: dict[str, Any] = {}

    result = Model(_arm_predict(spec, captured), spec=spec).run(
        TinyArmFactory(), episodes=1
    )

    assert result.num_episodes == 1
    assert captured["keys"] == ["image", "instruction", "state"]


def test_adapted_model_against_untagged_local_env_errors_clearly() -> None:
    pytest.importorskip("numpy")
    model = Model(lambda payload: None, spec=_model_spec())
    with pytest.raises(adapt.AdapterResolutionError, match="no adapter tags"):
        model.run(TinyArmEnv(), episodes=1)


def test_resolve_from_contract_describes_the_pairing() -> None:
    tags = _tags()
    spec = _model_spec()
    env_obj = TinyArmEnv()

    server = rlmesh.EnvServer(env_obj, ":0/env", tags=tags)
    server.start()
    try:
        client = RemoteEnv(server.address)
        adapter = adapt.resolve_from_contract(client.env_contract, spec)
        text = adapter.explain()
        assert 'image "cam"' in text
        assert "quat_xyzw->axis_angle" in text
        client.close()
    finally:
        server.shutdown()


@contextmanager
def _served(model: Any, port: int) -> Iterator[str]:
    session = f"127.0.0.1:{port}"
    thread = threading.Thread(
        target=model.serve, args=(session,), kwargs={"name": "model"}
    )
    thread.start()
    try:
        yield f"{session}/model"
    finally:
        model.close()
        thread.join(10)


def test_served_spec_model_resolves_adapter_at_configure_route(
    session_port: int,
) -> None:
    pytest.importorskip("numpy")

    tags = _tags()
    spec = _model_spec()
    env_obj = TinyArmEnv()
    seen: dict[str, Any] = {"payload_keys": None}

    def predict(payload: dict[str, Any]) -> Any:
        import numpy as np

        seen["payload_keys"] = sorted(payload)
        return np.zeros(spec.output.dim, dtype=np.float32)

    with _served(Model(predict, spec=spec), session_port) as address:
        Model(address).run(adapt.tag(env_obj, tags), episodes=1)

    assert seen["payload_keys"] == ["image", "instruction", "state"]
    assert env_obj.last_action is not None
    assert tuple(env_obj.last_action.shape) == (7,)


def test_local_stacked_run_reports_held_state_telemetry() -> None:
    pytest.importorskip("numpy")

    spec = adapt.ModelSpec(
        input={
            "image": adapt.Image(role=adapt.IMAGE_PRIMARY, height=8, width=8, stack=2),
        },
        output=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
            adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
        ),
    )
    tagged = adapt.tag(TinyArmEnv(), _tags())

    def predict(payload: dict[str, Any]) -> Any:
        import numpy as np

        return np.zeros(spec.output.dim, dtype=np.float32)

    result = Model(predict, spec=spec).run(tagged, episodes=1)

    rows = {r.metric: r for r in result.telemetry if r.op == "model.predict"}
    assert rows["held.episodes"].avg >= 1.0
    assert rows["held.bytes"].avg > 0.0
    assert rows["held.bytes"].unit == "bytes"


def test_served_frame_stacking_adapter_stacks_per_episode(session_port: int) -> None:
    pytest.importorskip("numpy")

    tags = _tags()
    spec = adapt.ModelSpec(
        input={
            "image": adapt.Image(role=adapt.IMAGE_PRIMARY, height=8, width=8, stack=2),
        },
        output=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
            adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
        ),
    )
    env_obj = TinyArmEnv()
    images: list[Any] = []

    def predict(payload: dict[str, Any]) -> Any:
        import numpy as np

        images.append(np.asarray(payload["image"]))
        return np.zeros(spec.output.dim, dtype=np.float32)

    with _served(Model(predict, spec=spec), session_port) as address:
        Model(address).run(adapt.tag(env_obj, tags), episodes=1)

    import numpy as np

    assert len(images) >= 2, "policy was not called for at least two steps"
    assert images[0].shape[0] == 2, images[0].shape
    np.testing.assert_array_equal(images[0][0], images[0][1])
    assert not np.array_equal(images[1][0], images[1][1])
    np.testing.assert_array_equal(images[1][0], images[0][1])


def test_run_env_chunks_a_predict_chunk_model_in_process() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("gymnasium")
    import numpy as np

    calls = {"predict": 0, "predict_chunk": 0, "horizon": 0}
    single = np.zeros(7, dtype=np.float32)
    chunk = np.zeros((2, 7), dtype=np.float32)

    class ChunkModel(Model):
        spec = rlmesh.NO_ADAPTER

        def predict(self, obs: object) -> Any:
            calls["predict"] += 1
            return single

        def predict_chunk(self, obs: object, execution_horizon: int = 1) -> Any:
            calls["predict_chunk"] += 1
            calls["horizon"] = execution_horizon
            return chunk

    env_obj = TinyArmEnv()
    server = rlmesh.EnvServer(env_obj, ":0/env", tags=_tags())
    server.start()
    try:
        ChunkModel().run(server.address, episodes=1, execution_horizon=2)
    finally:
        server.shutdown()

    assert calls["predict_chunk"] == 2, calls
    assert calls["predict"] == 0, calls
    assert calls["horizon"] == 2, calls
    assert len(env_obj.actions) == 3, env_obj.actions


def test_run_env_without_predict_chunk_warns_and_runs_unchunked() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("gymnasium")
    import warnings

    import numpy as np

    calls = {"predict": 0}
    single = np.zeros(7, dtype=np.float32)

    class PlainModel(Model):
        spec = rlmesh.NO_ADAPTER

        def predict(self, obs: object) -> Any:
            calls["predict"] += 1
            return single

    env_obj = TinyArmEnv()
    server = rlmesh.EnvServer(env_obj, ":0/env", tags=_tags())
    server.start()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            PlainModel().run(server.address, episodes=1, execution_horizon=2)
    finally:
        server.shutdown()

    assert calls["predict"] == 3, calls
    assert any("predict_chunk" in str(w.message) for w in caught), [
        str(w.message) for w in caught
    ]


def test_served_model_chunks_through_a_run(session_port: int) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("gymnasium")
    import numpy as np

    calls = {"predict": 0, "predict_chunk": 0, "horizon": 0}
    single = np.zeros(7, dtype=np.float32)
    chunk = np.zeros((2, 7), dtype=np.float32)

    class ChunkModel(Model):
        spec = _model_spec()

        def predict(self, payload: object) -> Any:
            calls["predict"] += 1
            return single

        def predict_chunk(self, payload: object, execution_horizon: int = 1) -> Any:
            calls["predict_chunk"] += 1
            calls["horizon"] = execution_horizon
            return chunk

    env_obj = TinyArmEnv()
    with _served(ChunkModel(), session_port) as address:
        Model(address).run(adapt.tag(env_obj, _tags()), episodes=1, execution_horizon=2)

    assert calls["predict_chunk"] == 2, calls
    assert calls["predict"] == 0, calls
    assert calls["horizon"] == 2, calls
    assert len(env_obj.actions) == 3, env_obj.actions
