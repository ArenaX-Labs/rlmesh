"""Serve one LIBERO task as an RLMesh env, with privileged object state (FAEA style).

Observations are the robot state plus every object's pose; there are no camera
images, so a policy works from geometry alone. The action is LIBERO's 7-D delta
end-effector command (xyz, axis-angle, gripper) in [-1, 1]. The task text comes
back in reset()'s info under "instruction".

Runs in a Python environment with LIBERO installed (see README). The agent does
not need LIBERO.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import rlmesh

# Nothing renders, so skip OpenGL; must be set before anything imports mujoco.
os.environ.setdefault("MUJOCO_GL", "disable")

DEFAULT_ADDRESS = "127.0.0.1:5555"
SETTLE_STEPS = 10
NOOP = [0.0] * 6 + [-1.0]


class Trace:
    def __init__(self, path: str, env):
        self._env = env  # robosuite rebuilds env.sim on reset, so read it each time
        Path(path).with_suffix(".xml").write_text(env.sim.model.get_xml())
        self._file = open(path, "w")

    def record(self, kind: str, **fields) -> None:
        qpos = np.round(self._env.sim.data.qpos, 5).tolist()
        row = {"t": time.time(), "kind": kind, **fields, "qpos": qpos}
        self._file.write(json.dumps(row) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class LiberoEnv:
    def __init__(self, suite: str, task_id: int, max_steps: int, trace: str | None):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs.env_wrapper import ControlEnv

        bench = benchmark.get_benchmark_dict()[suite]()
        task = bench.get_task(task_id)
        self.instruction = task.language
        self._init_states = bench.get_task_init_states(task_id)
        bddl = (
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )
        self._env = ControlEnv(
            bddl_file_name=str(bddl), use_camera_obs=False, has_offscreen_renderer=False
        )
        self._max_steps = max_steps
        self._steps = 0
        self._trace = Trace(trace, self._env.env) if trace else None

        obs = self._state(self._env.reset())
        self.observation_space = rlmesh.spaces.Dict(
            {
                k: rlmesh.spaces.Box(-np.inf, np.inf, v.shape, np.float32)
                for k, v in obs.items()
            }
        )
        self.action_space = rlmesh.spaces.Box(-1.0, 1.0, (7,), np.float32)

    @staticmethod
    def _state(raw: dict) -> dict:
        return {
            k: np.asarray(v, dtype=np.float32).reshape(-1)
            for k, v in raw.items()
            if np.ndim(v) <= 1
        }

    def reset(self, seed=None, options=None):
        index = (seed or 0) % len(self._init_states)
        self._env.reset()
        raw = self._env.set_init_state(self._init_states[index])
        for _ in range(SETTLE_STEPS):
            raw, _, _, _ = self._env.step(NOOP)
        self._steps = 0
        if self._trace:
            self._trace.record("reset", instruction=self.instruction)
        return self._state(raw), {"instruction": self.instruction, "init_state": index}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        raw, reward, done, _ = self._env.step(action.tolist())
        self._steps += 1
        if self._trace:
            self._trace.record("step", success=bool(done))
        truncated = self._steps >= self._max_steps and not done
        return self._state(raw), float(reward), bool(done), truncated, {"success": done}

    def close(self):
        if self._trace:
            self._trace.close()
        self._env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve one LIBERO task.")
    parser.add_argument("--address", default=DEFAULT_ADDRESS, help="bind address")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300, help="per episode")
    parser.add_argument("--trace", help="JSONL file of env states, for make_video.py")
    args = parser.parse_args()

    env = LiberoEnv(args.suite, args.task_id, args.max_steps, args.trace)
    server = rlmesh.EnvServer(env, args.address)
    print(
        f"serving {args.suite}[{args.task_id}] on {server.address}: {env.instruction}",
        flush=True,
    )
    server.serve()


if __name__ == "__main__":
    main()
