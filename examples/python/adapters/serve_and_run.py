"""Adapters quickstart: tag an env, then run a model against it.

One process, end to end:

1. an environment tags its observation/action spaces with semantic roles,
2. it is served with those tags published in its contract,
3. a model declares its own input/output format once, and
4. ``Model(spec=...).run(env)`` resolves the adapter from the env's contract and
   runs -- the prediction function works purely in the model's format, with no
   per-environment glue.

Run it (no GPU, no simulator):

    uv run serve_and_run.py
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import rlmesh
import rlmesh.adapters as adapt
from rlmesh.numpy import Model, RemoteEnv

# --- 1. The environment, written without any knowledge of a model. ----------


class CubePickEnv:
    """A toy 5-step arm env: wrist camera + end-effector state, 7-dim action."""

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {"render_modes": []}
        self.observation_space = gym.spaces.Dict(
            {
                "wrist_rgb": gym.spaces.Box(0, 255, (16, 16, 3), np.uint8),
                "ee_pos": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
                "ee_quat": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
                "grip": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
                "goal": gym.spaces.Text(max_length=64),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)
        self._t = 0

    def _obs(self) -> dict[str, Any]:
        rng = np.random.default_rng(self._t)
        quat = rng.normal(size=4).astype(np.float32)
        quat /= np.linalg.norm(quat)
        return {
            "wrist_rgb": rng.integers(0, 256, (16, 16, 3), dtype=np.uint8),
            "ee_pos": rng.normal(size=3).astype(np.float32),
            "ee_quat": quat,
            "grip": np.array([0.02], dtype=np.float32),
            "goal": "pick up the red cube",
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
        cmd = np.asarray(action, dtype=np.float32)
        print(f"  step {self._t}: env received action {np.round(cmd, 3)}")
        self._t += 1
        return self._obs(), 1.0, self._t >= 5, False, {}

    def close(self) -> None:
        return None


# The env's tags: roles plus the facts the spaces cannot carry.
ENV_TAGS = adapt.EnvTags(
    observation={
        "wrist_rgb": adapt.ImageTag(role=adapt.IMAGE_PRIMARY),
        "ee_pos": adapt.StateTag(role=adapt.EEF_POS),
        "ee_quat": adapt.StateTag(role=adapt.EEF_ROT, encoding="quat_xyzw"),
        "grip": adapt.StateTag(role=adapt.GRIPPER_POS),
        "goal": adapt.TextTag(role=adapt.INSTRUCTION),
    },
    action=adapt.Action(
        adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
        adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
        clip=(-1.0, 1.0),
    ),
)


# --- 2. The model, written without any knowledge of an environment. ---------

MODEL_SPEC = adapt.ModelSpec(
    input={
        # This checkpoint wants a 224x224 image, a list state with rot6d rotation,
        # and the instruction under its own key. Each key is a payload slot.
        "image": adapt.Image(role=adapt.IMAGE_PRIMARY, height=224, width=224),
        "proprio": adapt.Concat(
            adapt.EEF_POS,
            adapt.State(adapt.EEF_ROT, encoding="rot6d"),
            adapt.GRIPPER_POS,
            container="list",
        ),
        "task": adapt.Text(role=adapt.INSTRUCTION),
    },
    output=adapt.Action(
        adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
        adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=6, encoding="rot6d"),
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
    ),
)


def predict(payload: dict[str, Any]) -> Any:
    """The policy. ``payload`` already arrives in MODEL_SPEC's format."""
    assert payload["image"].shape == (224, 224, 3)
    assert len(payload["proprio"]) == 10  # pos(3) + rot6d(6) + grip(1)
    # A real policy runs the network here; we return a zero 10-dim action.
    return np.zeros(MODEL_SPEC.output.dim, dtype=np.float32)


def main() -> None:
    env = CubePickEnv()
    # Publish the tags in the served contract (validated against the
    # env's spaces up front).
    server = rlmesh.EnvServer(env, "127.0.0.1:0", tags=ENV_TAGS)
    server.start()
    try:
        client = RemoteEnv(server.address)

        # The adapter the model would get against this env, for illustration:
        print("Resolved adapter:")
        print(adapt.resolve_from_contract(client.env_contract, MODEL_SPEC).describe())
        print("\nRunning one episode:")

        # No glue: the adapter is resolved from the env's published tags.
        Model(predict, spec=MODEL_SPEC).run(client, max_episodes=1)

        client.close()
        print("\nDone.")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
