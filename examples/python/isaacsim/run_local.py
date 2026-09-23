"""Drive the served Franka reach env with a scripted policy: move straight at the target."""

import numpy as np
import rlmesh
from rlmesh.numpy import Model

MAX_DELTA = 0.05


def predict(obs):
    return np.clip(obs["target_pos"] - obs["eef_pos"], -MAX_DELTA, MAX_DELTA).astype(np.float32)


if __name__ == "__main__":
    result = Model(predict, spec=rlmesh.NO_ADAPTER).run("127.0.0.1:50051", episodes=4)
    print(f"episodes={result.num_episodes} steps={result.total_steps} success_rate={result.success_rate}")
