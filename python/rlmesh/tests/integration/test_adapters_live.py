"""End-to-end live path for rlmesh.adapters.

Serves a tagged env, then runs an adapted ``Model(spec=...)`` against it:
the adapter is resolved from the env's published tags in the contract,
the prediction function works in the model's own format, and the env receives
actions in its format. This exercises tag -> serve -> resolve_from_contract
-> Model(spec=).run() and the on_episode_end chaining, over a real transport.
"""

from __future__ import annotations

import importlib
import socket
import threading
import time
from typing import TYPE_CHECKING, Any, cast

import pytest
import rlmesh
import rlmesh.adapters as adapt
from rlmesh.numpy import Model, RemoteEnv, RemoteModel

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
    """A 3-step episodic env: camera image + eef state, 7-dim action."""

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
        # Every action the env received, in order (chunk-replay value assertions).
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

    server = rlmesh.EnvServer(env_obj, "127.0.0.1:0", tags=tags)
    server.start()
    try:
        client = RemoteEnv(server.address)
        # The tags published via EnvServer(tags=) survive the
        # round-trip through the contract metadata.
        recovered = adapt.EnvTags.from_metadata(client.env_contract.metadata or {})
        assert recovered == tags

        Model(predict, spec=spec, on_episode_end=on_episode_end).run(
            client, max_episodes=1
        )
        client.close()
    finally:
        server.shutdown()

    # The prediction function saw the model's declared payload, and the env
    # received a transformed 7-dim action; the episode-end hook chained.
    assert seen["payload_keys"] == ["image", "instruction", "state"]
    assert env_obj.last_action is not None
    assert tuple(env_obj.last_action.shape) == (7,)
    assert cast(int, seen["episode_ends"]) >= 1


class TinyArmFactory(rlmesh.EnvFactory):
    """EnvFactory whose tags are the env's contract -- make() stamps them on."""

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
    # The tag rides the environment: a locally-made env carries the factory's
    # contract in its metadata, with no server in the loop.
    assert adapt.EnvTags.from_metadata(getattr(env, "metadata", {}) or {}) == _tags()


def test_adapted_model_runs_against_local_tagged_env() -> None:
    pytest.importorskip("numpy")
    spec = _model_spec()
    env_obj = TinyArmEnv()
    captured: dict[str, Any] = {}

    # A locally tagged env -- no EnvServer, no transport -- resolves the adapter.
    tagged = adapt.tag(env_obj, _tags())
    result = Model(_arm_predict(spec, captured), spec=spec).run(tagged, max_episodes=1)

    assert result.num_episodes == 1
    assert captured["keys"] == ["image", "instruction", "state"]
    assert env_obj.last_action is not None
    assert tuple(env_obj.last_action.shape) == (7,)


@pytest.mark.parametrize(
    ("container", "expected"),
    [("str", "follow the override"), ("list", ["follow the override"])],
)
def test_instruction_override_reaches_predict_in_declared_shape(
    container: str, expected: object
) -> None:
    """``run(..., instruction=)`` overrides the text input in its declared shape.

    The env publishes its own instruction ("pick up the cube"); the override must
    win and land as a bare ``str`` for ``container='str'`` and as ``[instruction]``
    for ``container='list'``.
    """
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
    Model(predict, spec=spec).run(
        tagged, max_episodes=1, instruction="follow the override"
    )

    assert seen["instruction"] == expected


# (framework module, leaf tensor type, zeros constructor) for the cross-framework
# local-driving test. A torch/jax model driven against a local numpy gym env must
# encode the env's numpy obs with the *env's* (numpy) bridge, not the model's.
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
    """A spec'd model of any framework drives a *local* numpy gym env.

    The env hands the loop raw numpy obs, so the env-side bridge must be numpy
    (the env's native type) -- not the model's framework bridge, which rejects
    numpy at the native plan (the cross-framework local-driving bug). ``predict``
    still receives the model's own framework tensors, and the env still receives
    a numpy action of the declared shape.
    """
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
    result = model_cls(predict, spec=spec).run(tagged, max_episodes=1)

    assert result.num_episodes == 1
    # predict saw the model's own framework tensors (obs decoded into them)...
    assert issubclass(captured["image_type"], leaf_type), captured["image_type"]
    # ...and the local numpy env got a numpy action of the declared 7-dim shape.
    assert env_obj.last_action is not None
    assert tuple(env_obj.last_action.shape) == (7,)


def test_adapted_model_runs_against_env_factory() -> None:
    pytest.importorskip("numpy")
    spec = _model_spec()
    captured: dict[str, Any] = {}

    # Pass the EnvFactory straight in: session/run builds + tags + drives it locally.
    result = Model(_arm_predict(spec, captured), spec=spec).run(
        TinyArmFactory(), max_episodes=1
    )

    assert result.num_episodes == 1
    assert captured["keys"] == ["image", "instruction", "state"]


def test_adapted_model_against_untagged_local_env_errors_clearly() -> None:
    pytest.importorskip("numpy")
    model = Model(lambda payload: None, spec=_model_spec())
    with pytest.raises(adapt.AdapterResolutionError, match="no adapter tags"):
        model.run(TinyArmEnv(), max_episodes=1)


def test_resolve_from_contract_describes_the_pairing() -> None:
    tags = _tags()
    spec = _model_spec()
    env_obj = TinyArmEnv()

    server = rlmesh.EnvServer(env_obj, "127.0.0.1:0", tags=tags)
    server.start()
    try:
        client = RemoteEnv(server.address)
        adapter = adapt.resolve_from_contract(client.env_contract, spec)
        text = adapter.describe()
        assert 'image "cam"' in text
        assert "quat_xyzw->axis_angle" in text
        client.close()
    finally:
        server.shutdown()


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return cast(int, sock.getsockname()[1])
    finally:
        sock.close()


def test_served_spec_model_resolves_adapter_at_configure_route() -> None:
    """A spec'd model served over the wire resolves its adapter from the route's
    contract (configure_route) and applies transform_obs/transform_action, so the
    same RemoteEnv/RemoteModel loop drives an adapted model end-to-end."""
    pytest.importorskip("numpy")

    tags = _tags()
    spec = _model_spec()
    env_obj = TinyArmEnv()
    seen: dict[str, Any] = {"payload_keys": None}

    def predict(payload: dict[str, Any]) -> Any:
        import numpy as np

        seen["payload_keys"] = sorted(payload)
        return np.zeros(spec.output.dim, dtype=np.float32)

    env_server = rlmesh.EnvServer(env_obj, "127.0.0.1:0", tags=tags)
    env_server.start()
    model_address = f"127.0.0.1:{_free_port()}"

    def serve_model() -> None:
        Model(predict, spec=spec).serve(
            model_address, options=rlmesh.ServeOptions(allow_remote_shutdown=True)
        )

    threading.Thread(target=serve_model, daemon=True).start()

    try:
        env = RemoteEnv(env_server.address)
        deadline = time.monotonic() + 5.0
        sess: Any = None
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                sess = rlmesh.session(RemoteModel(model_address), env)
                break
            except Exception as exc:
                last_error = exc
                time.sleep(0.05)
        if sess is None:
            raise AssertionError("served model never came up") from last_error

        obs, _info = sess.reset(seed=0)
        steps = 0
        while not sess.done and steps < 5:
            action = sess.predict(obs)
            obs, _reward, _terminated, _truncated, _info = sess.step(action)
            steps += 1

        sess.close()
        env.close()
    finally:
        env_server.shutdown()

    # transform_obs ran server-side: the policy saw the model's declared payload.
    assert seen["payload_keys"] == ["image", "instruction", "state"]
    # transform_action ran and round-tripped: the env got its 7-dim action.
    assert env_obj.last_action is not None
    assert tuple(env_obj.last_action.shape) == (7,)


def test_served_frame_stacking_adapter_stacks_per_episode() -> None:
    """A served frame-stacking (stateful) spec'd model end-to-end.

    The native serving engine episode-keys the frame buffers and stacks
    server-side, so the policy sees a stacked image with first-frame padding at
    step 0 and a sliding window after -- the vectorized-stateful relocation the
    old single-lane rejection forbade.
    """
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

    env_server = rlmesh.EnvServer(env_obj, "127.0.0.1:0", tags=tags)
    env_server.start()
    model_address = f"127.0.0.1:{_free_port()}"

    def serve_model() -> None:
        Model(predict, spec=spec).serve(
            model_address, options=rlmesh.ServeOptions(allow_remote_shutdown=True)
        )

    threading.Thread(target=serve_model, daemon=True).start()
    try:
        env = RemoteEnv(env_server.address)
        deadline = time.monotonic() + 5.0
        sess: Any = None
        while time.monotonic() < deadline:
            try:
                sess = rlmesh.session(RemoteModel(model_address), env)
                break
            except Exception:  # retry until the server is up
                time.sleep(0.05)
        assert sess is not None, "served model never came up"

        obs, _info = sess.reset(seed=0)
        steps = 0
        while not sess.done and steps < 3:
            action = sess.predict(obs)
            obs, _reward, _terminated, _truncated, _info = sess.step(action)
            steps += 1
        sess.close()
        env.close()
    finally:
        env_server.shutdown()

    import numpy as np

    assert len(images) >= 2, "policy was not called for at least two steps"
    # Each obs carries a leading stack axis of depth 2.
    assert images[0].shape[0] == 2, images[0].shape
    # Step 0: window is first-frame padded, so both stacked frames are equal.
    np.testing.assert_array_equal(images[0][0], images[0][1])
    # Step 1: the window slid -- newest frame differs from the retained one...
    assert not np.array_equal(images[1][0], images[1][1])
    # ...and the retained (older) slot equals step 0's frame (episode continuity).
    np.testing.assert_array_equal(images[1][0], images[0][1])


def test_run_env_chunks_a_predict_chunk_model_in_process() -> None:
    """In-process run(env, execution_horizon=2): a predict_chunk model emits a 2-row
    chunk and the loop replays it one step per env step, re-planning every 2 steps.

    The runtime-owned chunking story for the local path: predict_chunk is the chunk
    corner and execution_horizon is a caller (not spec) decision. This model opts into
    the horizon (an optional second param), so it can also assert the value reaches it.
    """
    pytest.importorskip("numpy")
    pytest.importorskip("gymnasium")
    import numpy as np

    calls = {"predict": 0, "predict_chunk": 0, "horizon": 0}
    single = np.zeros(7, dtype=np.float32)
    chunk = np.zeros((2, 7), dtype=np.float32)  # leading axis is the chunk axis

    class ChunkModel(Model):
        spec = rlmesh.NO_ADAPTER  # model returns raw 7-dim env actions, no adapter

        def predict(self, obs: object) -> Any:
            calls["predict"] += 1
            return single

        def predict_chunk(self, obs: object, execution_horizon: int = 1) -> Any:
            calls["predict_chunk"] += 1
            calls["horizon"] = execution_horizon
            return chunk

    env_obj = TinyArmEnv()
    server = rlmesh.EnvServer(env_obj, "127.0.0.1:0", tags=_tags())
    server.start()
    try:
        client = RemoteEnv(server.address)
        ChunkModel().run(client, max_episodes=1, execution_horizon=2)
        client.close()
    finally:
        server.shutdown()

    # 3-step episode, chunk size 2: predict_chunk fires at steps 0 and 2 only; the
    # single-step predict is never used. The chosen horizon reaches the model.
    assert calls["predict_chunk"] == 2, calls
    assert calls["predict"] == 0, calls
    assert calls["horizon"] == 2, calls
    assert len(env_obj.actions) == 3, env_obj.actions


def test_run_env_without_predict_chunk_warns_and_runs_unchunked() -> None:
    """execution_horizon > 1 on a model with no predict_chunk warns and runs per-step."""
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
    server = rlmesh.EnvServer(env_obj, "127.0.0.1:0", tags=_tags())
    server.start()
    try:
        client = RemoteEnv(server.address)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            PlainModel().run(client, max_episodes=1, execution_horizon=2)
        client.close()
    finally:
        server.shutdown()

    # No chunk corner: predict every step (3-step episode), with a warning.
    assert calls["predict"] == 3, calls
    assert any("predict_chunk" in str(w.message) for w in caught), [
        str(w.message) for w in caught
    ]


def test_served_model_chunks_via_remote_model_mini_driver() -> None:
    """A served spec'd model with predict_chunk + RemoteModel.session(execution_horizon=2):
    the served engine emits the chunk; RemoteModel pins the horizon on ConfigureRoute,
    buffers the chunk's future frames, and replays them open-loop (a predict RPC only
    every 2 steps). The end-to-end served-OSS chunking path."""
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
    env_server = rlmesh.EnvServer(env_obj, "127.0.0.1:0", tags=_tags())
    env_server.start()
    model_address = f"127.0.0.1:{_free_port()}"

    def serve_model() -> None:
        ChunkModel().serve(
            model_address, options=rlmesh.ServeOptions(allow_remote_shutdown=True)
        )

    threading.Thread(target=serve_model, daemon=True).start()

    try:
        env = RemoteEnv(env_server.address)
        deadline = time.monotonic() + 5.0
        sess: Any = None
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                sess = RemoteModel(model_address).session(env, execution_horizon=2)
                break
            except Exception as exc:
                last_error = exc
                time.sleep(0.05)
        if sess is None:
            raise AssertionError("served model never came up") from last_error

        sess.run(max_episodes=1)
        sess.close()
        env.close()
    finally:
        env_server.shutdown()

    # 3-step episode, chunk size 2: the served model's predict_chunk fires once per
    # 2 steps (steps 0 and 2); RemoteModel replays the in-between step without an RPC.
    # The runtime-chosen horizon reaches the served model's predict_chunk.
    assert calls["predict_chunk"] == 2, calls
    assert calls["predict"] == 0, calls
    assert calls["horizon"] == 2, calls
    assert len(env_obj.actions) == 3, env_obj.actions
