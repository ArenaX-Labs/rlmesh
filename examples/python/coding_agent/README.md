# Coding agent: Claude writes the robot's policy

Claude solves a LIBERO manipulation task the way RAX and FAEA do: it probes the environment by hand, writes a Python policy, evaluates it through RLMesh, and fixes it until it passes. Claude is never in the control loop; the finished `policy.py` runs at the env's own rate with no model calls.

| File            | Role                                                                                 |
| --------------- | ------------------------------------------------------------------------------------ |
| `libero_env.py` | Serves one LIBERO task as an RLMesh env, with object poses instead of camera images. |
| `agent.py`      | Claude with five tools: `describe`, `reset`, `step`, `write_policy`, `evaluate`.     |
| `make_video.py` | Renders a recorded run: the agent's activity on top, LIBERO below.                   |

How a run goes:

1. `describe`, `reset` and `step` let the agent explore the env through `rlmesh.RemoteEnv`.
2. `write_policy` saves `workdir/policy.py`, which must define `predict(obs) -> action`.
3. `evaluate` scores it with `Model(predict).run(address, seeds=...)` on layouts the agent never explored, and reports per-episode success.
4. A policy that passes every episode is copied to `workdir/skills/<skill>.py`.

The agent reads the task from the env (`info["instruction"]` on reset), so `agent.py` works against any RLMesh env that provides one.

## Setup

**Agent side** (the repository environment): a Claude API key.

```sh
export ANTHROPIC_API_KEY=sk-ant-...
```

**Env side**: a separate Python 3.10 environment with [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and `rlmesh`. LIBERO pins old robosuite and MuJoCo versions that must not share the repository environment. Tested on macOS (Apple silicon):

```sh
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ~/LIBERO
uv venv ~/libero-venv --python 3.10

# From outside this repository (its pyproject pins a different torch).
cd ~/LIBERO
uv pip install --python ~/libero-venv/bin/python -e . --config-settings editable_mode=compat
uv pip install --python ~/libero-venv/bin/python \
  robosuite==1.4.1 mujoco==2.3.7 torch==2.5.1 bddl==1.0.1 \
  easydict gym cloudpickle imageio future matplotlib h5py termcolor tqdm pyyaml

# From this repository, so mise supplies the pinned Rust toolchain.
cd <this repo>
mise exec -- uv pip install --python ~/libero-venv/bin/python "./python/rlmesh[numpy]"
```

Why these pins: `editable_mode=compat` because LIBERO's top-level package has no `__init__.py`; `mujoco==2.3.7` because robosuite 1.4.1 breaks on MuJoCo 3; `torch==2.5.1` because newer `torch.load` refuses LIBERO's pickled initial states.

On first import LIBERO asks about a dataset path: answer `N`. If you already have a `~/.libero/config.yaml` from another checkout, point this one at a fresh config with `LIBERO_CONFIG_PATH=~/.libero-rlmesh`.

## Run

From the repository root, in two terminals:

```sh
# 1. Serve a task
~/libero-venv/bin/python examples/python/coding_agent/libero_env.py --suite libero_spatial --task-id 0

# 2. Run the agent
uv run --with anthropic python examples/python/coding_agent/agent.py --skill libero_spatial_0
```

The agent prints its narration, tool calls and evaluation results as it goes; a passing policy lands in `workdir/skills/libero_spatial_0.py`.

| Flag                        | Default           | Meaning                                                         |
| --------------------------- | ----------------- | --------------------------------------------------------------- |
| `libero_env.py --suite`     | `libero_spatial`  | `libero_spatial`, `libero_object`, `libero_goal` or `libero_10` |
| `libero_env.py --task-id`   | `0`               | task within the suite (0-9)                                     |
| `libero_env.py --max-steps` | `300`             | steps per episode; use `520` for `libero_10`                    |
| `agent.py --model`          | `claude-opus-5-5` | any Claude model, e.g. `claude-sonnet-5-5` for a cheaper run    |
| `agent.py --max-iterations` | `60`              | agent turns before it stops                                     |
| `agent.py --skill`          | `policy`          | file name for a passing policy                                  |

Both scripts take `--address` (default `127.0.0.1:5555`).

## Record a video

Add `--trace` to the env and `--log` to the agent, then render. The video shows the agent's narration, each tool call, the policy code (in full the first time, as a diff after that) and evaluation results on top, with the matching LIBERO view below.

```sh
mkdir -p run
~/libero-venv/bin/python examples/python/coding_agent/libero_env.py --trace run/trace.jsonl
uv run --with anthropic python examples/python/coding_agent/agent.py --log run/agent.jsonl
uv run examples/python/coding_agent/make_video.py run/trace.jsonl run/agent.jsonl --out run/agent_run.mp4
```

`make_video.py` installs its own dependencies (MuJoCo 3.17, Pillow, imageio) and renders from the saved scene XML and joint positions, so it needs no OpenGL. Evaluations are long, so it shows only the first `--max-eval-episodes 2` episodes of each at `--eval-stride 2`; `--camera` picks `agentview` (default), `frontview` or `robot0_eye_in_hand`.

The env rewrites `trace.jsonl` and its `.xml` sibling every time it starts; the agent log is appended to. Use a fresh `--trace` path per task, or pass `--xml` to render an older run whose scene was overwritten.

## Safety

`evaluate` executes the code the model writes, in the agent's process. Run the agent in a container or VM. On real hardware, the env's `step()` must enforce workspace, joint and speed limits whatever action it receives, and the agent must not be able to edit the env or the drivers.

## Next steps

- **Vision:** turn rendering on in `libero_env.py` (Linux with `MUJOCO_GL=egl`), add a `look(camera)` tool that returns a frame to Claude, and drop the privileged object poses.
- **Another model:** the five tools are plain functions; replace the `tool_runner` loop in `run()` with another model's tool-calling loop.
- **Memory across tasks:** load `workdir/skills/` into the prompt so later tasks reuse earlier policies.
