"""End-to-end live path for rlmesh.adapters.

Serves a tagged env, then runs an adapted ``Model(spec=...)`` against it:
the adapter is resolved from the env's published tags in the contract,
the prediction function works in the model's own format, and the env receives
actions in its format. This exercises tag -> serve -> resolve_from_contract
-> Model(spec=).run() and the on_reset chaining, over a real transport.
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
            "instruction": adapt.TextTag(),
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
            "instruction": adapt.Text(),
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

    seen: dict[str, Any] = {"resets": 0, "payload_keys": None}

    def predict(payload: dict[str, Any]) -> Any:
        import numpy as np

        seen["payload_keys"] = sorted(payload)
        return np.zeros(spec.output.dim, dtype=np.float32)

    def on_reset() -> None:
        seen["resets"] = cast(int, seen["resets"]) + 1

    server = rlmesh.EnvServer(env_obj, "127.0.0.1:0", tags=tags)
    server.start()
    try:
        client = RemoteEnv(server.address)
        # The tags published via EnvServer(tags=) survive the
        # round-trip through the contract metadata.
        recovered = adapt.EnvTags.from_metadata(client.env_contract.metadata or {})
        assert recovered == tags

        Model(predict, spec=spec, on_reset=on_reset).run(client, max_episodes=1)
        client.close()
    finally:
        server.shutdown()

    # The prediction function saw the model's declared payload, and the env
    # received a transformed 7-dim action; reset chained to the adapter.
    assert seen["payload_keys"] == ["image", "instruction", "state"]
    assert env_obj.last_action is not None
    assert tuple(env_obj.last_action.shape) == (7,)
    assert cast(int, seen["resets"]) >= 1


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
            "instruction": adapt.Text(container=container),  # pyright: ignore[reportArgumentType]
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


def _chunk_spec(execute_horizon: int) -> adapt.ModelSpec:
    """A minimal image->7-dim-action spec with action-chunk replay."""
    return adapt.ModelSpec(
        input={
            "image": adapt.Image(role=adapt.IMAGE_PRIMARY, height=8, width=8),
        },
        output=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
            adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
            execute_horizon=execute_horizon,
        ),
    )


def _distinct_chunk(calls: dict[str, int], dim: int) -> Any:
    """A 2-row action chunk whose rows are distinct within and across predict
    calls, all inside the [-1, 1] clip so each row maps to a distinct env action.

    Call 1 -> rows [0.1.., 0.2..]; call 2 -> [0.3.., 0.4..]. The leading axis is
    the chunk axis. Used so the tests assert the per-step replay VALUE (FIFO
    order), not merely the predict call cadence.
    """
    import numpy as np

    calls["predict"] += 1
    c = calls["predict"]
    return np.stack(
        [
            np.full(dim, 0.1 * (2 * c - 1), dtype=np.float32),
            np.full(dim, 0.1 * (2 * c), dtype=np.float32),
        ]
    )


def _assert_replayed_in_order(actions: list[Any]) -> None:
    """Assert a 3-step episode replayed a horizon-2 chunk in FIFO order.

    Step 0 emits chunk row 0, step 1 replays row 1, step 2 is a fresh predict's
    row 0 -- so all three env actions differ. A bug that re-emits row 0, drops the
    queue, reverses the slice, or skips the re-plan collapses two of these to equal.
    """
    import numpy as np

    assert len(actions) == 3, actions
    assert not np.array_equal(actions[0], actions[1]), "step 1 did not replay row 1"
    assert not np.array_equal(actions[1], actions[2]), "step 2 did not re-plan"
    assert not np.array_equal(actions[0], actions[2]), actions


def test_run_env_chunk_replay_predicts_once_per_horizon() -> None:
    """run(env): a chunked model predicts a chunk and the loop replays it in order.

    TinyArmEnv runs a 3-step episode; with execute_horizon=2 the loop predicts at
    step 0 and step 2 only -- step 1 replays the queued (second) action. Assert both
    the predict cadence (2 calls, not 3) and the FIFO replay value order.
    """
    pytest.importorskip("numpy")

    spec = _chunk_spec(2)
    env_obj = TinyArmEnv()
    calls = {"predict": 0}

    def predict(payload: dict[str, Any]) -> Any:
        return _distinct_chunk(calls, spec.output.dim)

    server = rlmesh.EnvServer(env_obj, "127.0.0.1:0", tags=_tags())
    server.start()
    try:
        client = RemoteEnv(server.address)
        Model(predict, spec=spec).run(client, max_episodes=1)
        client.close()
    finally:
        server.shutdown()

    assert calls["predict"] == 2, calls
    _assert_replayed_in_order(env_obj.actions)


def test_served_chunk_replay_predicts_once_per_horizon() -> None:
    """Serve path: the native engine queues the chunk per episode and replays it.

    Across three separate predict RPCs (one per env step), the model server's
    ChunkBuffers replays the queued (second) action on the middle step, so the user
    predict callback fires twice and the env receives the chunk rows in FIFO order
    -- the relocation of chunk replay into the Rust engine.
    """
    pytest.importorskip("numpy")

    spec = _chunk_spec(2)
    env_obj = TinyArmEnv()
    calls = {"predict": 0}

    def predict(payload: dict[str, Any]) -> Any:
        return _distinct_chunk(calls, spec.output.dim)

    env_server = rlmesh.EnvServer(env_obj, "127.0.0.1:0", tags=_tags())
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

    assert calls["predict"] == 2, calls
    _assert_replayed_in_order(env_obj.actions)


def test_run_env_chunk_replay_drains_a_3_row_chunk_in_fifo_order() -> None:
    """A horizon-3 chunk replays in exact FIFO order across a 3-step episode.

    TinyArmEnv runs 3 steps; with execute_horizon=3 the loop predicts ONCE and
    replays rows 1 then 2 -- so a reversed/LIFO drain (which a 2-row chunk cannot
    distinguish, since its queue only ever holds one item) is caught here: the
    delta_pos[0] component of the three env actions must strictly increase,
    matching chunk rows 0.1 < 0.2 < 0.3 emitted in order.
    """
    pytest.importorskip("numpy")
    import numpy as np

    spec = _chunk_spec(3)
    env_obj = TinyArmEnv()
    dim = spec.output.dim
    chunk = np.stack([np.full(dim, v, dtype=np.float32) for v in (0.1, 0.2, 0.3)])
    calls = {"predict": 0}

    def predict(payload: dict[str, Any]) -> Any:
        calls["predict"] += 1
        return chunk

    server = rlmesh.EnvServer(env_obj, "127.0.0.1:0", tags=_tags())
    server.start()
    try:
        client = RemoteEnv(server.address)
        Model(predict, spec=spec).run(client, max_episodes=1)
        client.close()
    finally:
        server.shutdown()

    assert calls["predict"] == 1, calls
    assert len(env_obj.actions) == 3, env_obj.actions
    # delta_pos[0] is a passthrough component, so it tracks each row's value:
    # FIFO -> 0.1, 0.2, 0.3 (increasing); a LIFO/reversed drain breaks the order.
    firsts = [float(a[0]) for a in env_obj.actions]
    assert firsts[0] < firsts[1] < firsts[2], env_obj.actions


def test_run_env_chunk_replay_caps_a_chunk_longer_than_the_horizon() -> None:
    """A chunk longer than execute_horizon is capped; the extra rows are dropped.

    horizon=2 with a 3-row chunk: step 0 emits row 0, step 1 replays row 1, step 2
    RE-PLANS (row 2 is discarded, not replayed). So predict fires twice and the
    step-2 action comes from the second predict, not the dropped row 2.
    """
    pytest.importorskip("numpy")
    import numpy as np

    spec = _chunk_spec(2)
    env_obj = TinyArmEnv()
    dim = spec.output.dim
    # Two 3-row chunks (> horizon 2); rows distinct within and across predicts.
    chunks = [
        np.stack([np.full(dim, 0.3 * c + 0.01 * r, dtype=np.float32) for r in range(3)])
        for c in (1, 2)
    ]
    calls = {"predict": 0}

    def predict(payload: dict[str, Any]) -> Any:
        chunk = chunks[calls["predict"]]
        calls["predict"] += 1
        return chunk

    server = rlmesh.EnvServer(env_obj, "127.0.0.1:0", tags=_tags())
    server.start()
    try:
        client = RemoteEnv(server.address)
        Model(predict, spec=spec).run(client, max_episodes=1)
        client.close()
    finally:
        server.shutdown()

    # Capped to horizon=2: predict at step 0 and step 2 (call-1 row 2 never runs).
    assert calls["predict"] == 2, calls
    a1 = float(env_obj.actions[1][0])  # call-1 row 1 (~0.31)
    a2 = float(env_obj.actions[2][0])  # call-2 row 0 (~0.60), NOT call-1 row 2
    assert a2 > a1 + 0.1, env_obj.actions
