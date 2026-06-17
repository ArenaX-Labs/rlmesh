"""End-to-end live path for rlmesh.adapters.

Serves a tagged env, then runs an adapted ``Model(spec=...)`` against it:
the adapter is resolved from the env's published tags in the contract,
the prediction function works in the model's own format, and the env receives
actions in its format. This exercises tag -> serve -> resolve_from_contract
-> Model(spec=).run() and the on_reset chaining, over a real transport.
"""

from __future__ import annotations

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
        action=adapt.ActionLayout(
            adapt.ActionComponent(adapt.ACTION_DELTA_POS, dim=3),
            adapt.ActionComponent(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
            adapt.ActionComponent(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
            clip=(-1.0, 1.0),
        ),
    )


def _model_spec() -> adapt.ModelSpec:
    return adapt.ModelSpec(
        inputs=(
            adapt.ImageInput("image", role=adapt.IMAGE_PRIMARY, height=8, width=8),
            adapt.StateInput(
                "state",
                components=(
                    adapt.StateComponent(adapt.EEF_POS),
                    adapt.StateComponent(adapt.EEF_ROT, encoding="axis_angle"),
                    adapt.StateComponent(adapt.GRIPPER_POS),
                ),
                container="list",
            ),
            adapt.TextInput("instruction"),
        ),
        action=adapt.ActionLayout(
            adapt.ActionComponent(adapt.ACTION_DELTA_POS, dim=3),
            adapt.ActionComponent(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
            adapt.ActionComponent(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
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
        return np.zeros(spec.action.dim, dtype=np.float32)

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
        return np.zeros(spec.action.dim, dtype=np.float32)

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
        model: Any = None
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                model = RemoteModel(model_address).against(env)
                break
            except Exception as exc:
                last_error = exc
                time.sleep(0.05)
        if model is None:
            raise AssertionError("served model never came up") from last_error

        obs, _info = env.reset(seed=0)
        model.reset()
        done = False
        steps = 0
        while not done and steps < 5:
            action = model.predict(obs)
            obs, _reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            steps += 1

        model.close()
        env.close()
    finally:
        env_server.shutdown()

    # transform_obs ran server-side: the policy saw the model's declared payload.
    assert seen["payload_keys"] == ["image", "instruction", "state"]
    # transform_action ran and round-tripped: the env got its 7-dim action.
    assert env_obj.last_action is not None
    assert tuple(env_obj.last_action.shape) == (7,)
