# Project Chrono env in native C++

[Project Chrono](https://projectchrono.org/)'s six-axis industrial robot (`IndustrialRobot6dof`) reaches a target, simulated by Chrono in C++ and served as an RLMesh environment through the C ABI (`crates/rlmesh-capi`). There is no Python in the env. A Python model drives it like any other RLMesh env, and RLMesh adapts between the two from the tags the C++ env publishes.

- **Simulation.** Chrono integrates the robot's multibody dynamics at 200 Hz under gravity. Each env step runs 50 ms of it. The action moves the tool target, Chrono's analytic inverse kinematics turns that into joint setpoints, and the motors drive the arm there.
- **Observation.**
  - `image`: a 256×256 RGB camera, ray-traced on the CPU, so it works headless and in a container.
  - `joint_pos`, `joint_vel`, `joint_torque`: the motors' angles, rates, and Chrono's reaction torques.
  - `eef_pos`, `target_pos`: the tool and target positions.
  - `instruction`: the task text.
- **Action.** A 3-D tool delta in [-1, 1], up to 2 cm per step.
- **Episodes.** An episode succeeds (`info["is_success"]`) when the tool is within 2.5 cm of the target. It truncates at 120 steps.
- **Seeding.** `reset(seed=…)` and the `trial_index` reset option pick the target, so a seeded run is reproducible.
- **Rendering.** `render()` returns 480×480 frames for viewers.

The env tags its spaces with adapter roles (`image/primary`, `proprio/eef_pos`, `proprio/joint_pos`, `text/instruction`, `action/delta_eef_pos`, …). The tags are validated against the spaces when the env starts, exactly as Python's `rlmesh.EnvServer(tags=...)` does. `run_model.py` declares a model that wants a 224×224 image, one proprio vector, the goal, and the instruction under its own keys, and prints the adapter RLMesh resolves between them:

```text
observation:
  "goal" <- concat(target_pos)
  "image" <- image "image" (resize 224x224 (bilinear), uint8)
  "proprio" <- concat(eef_pos, joint_pos)
  "task" <- text "instruction"
action:
  "action/delta_eef_pos" <- model[0:3]
  clip to (-1.0, 1.0)
```

## Run it locally

You need a Chrono install. The core plus its bundled `ChronoModels_robot` library is enough; no visualization module is needed. Build Chrono 10 with `BUILD_DEMOS=OFF`, and pass `-DCMAKE_DISABLE_FIND_PACKAGE_Thrust=ON -DCMAKE_CUDA_COMPILER=` when a CUDA toolkit is installed: the CPU-only core does not need it, and the multicore math does not compile against a recent CCCL. The `Dockerfile` has the exact recipe.

```bash
CHRONO_DIR=~/opt/chrono EIGEN_DIR=~/opt/eigen examples/chrono/demo.sh --episodes 8
```

`demo.sh` builds `rlmesh-capi`, configures this CMake project against it, serves the env on `127.0.0.1:50051`, and runs `run_model.py`.

- **Live viewer.** Add `--view http:9000` and open <http://localhost:9000>.
- **Random policy.** `--policy random` swaps the scripted policy for a random one.

To run the pieces separately:

```bash
cmake -S examples/chrono -B target/chrono-env -DCMAKE_PREFIX_PATH="$HOME/opt/chrono;$HOME/opt/eigen"
cmake --build target/chrono-env
target/chrono-env/chrono_reach_env --address 127.0.0.1:50051     # Ctrl-C stops it cleanly
uv run python examples/chrono/run_model.py 127.0.0.1:50051 --view http:9000
```

Build the env and the Python package from the same checkout with `RLMESH_RELEASE_BUILD=1`, as `demo.sh` does. Otherwise each dev build pins its own per-commit workflow edition, and the handshake refuses the other.

## No Python at all

`chrono_reach_model` is the same scripted policy written in C++ against `rlmesh.hpp`'s `Model`, so both sides of the wire are native:

```bash
target/chrono-env/chrono_reach_env --address 127.0.0.1:50051 &
target/chrono-env/chrono_reach_model 127.0.0.1:50051 6
# episodes=6 steps=67 terminated=6 truncated=0 mean_reward=-0.38
```

It ignores the adapter tags and reads `eef_pos` and `target_pos` by key: the C++ model surface does not apply adapters per step yet.

## Run it as a container

```bash
docker build -f examples/chrono/Dockerfile -t chrono-reach:dev .     # from the repo root
docker run --rm -p 50051:50051 chrono-reach:dev
uv run python examples/chrono/run_model.py 127.0.0.1:50051
```

The image holds just the binary, `librlmesh_capi.so` and Chrono's shared libraries. It serves on `$RLMESH_ADDRESS` (default `0.0.0.0:50051`), honors `RLMESH_ENV_ENDPOINT_TOKEN`, and drains and closes on SIGTERM, so the same image runs on RLMesh Managed:

```bash
docker tag chrono-reach:dev registry.rlmesh.dev/<namespace>/chrono-reach:v1
docker push registry.rlmesh.dev/<namespace>/chrono-reach:v1
```

The env publishes its describe envelope (spaces, tags, and workflow editions) on the handshake when it binds, so the platform can read the image as it is. To describe the image before it runs, bake the envelope as its `dev.rlmesh.describe` label. A `LABEL` cannot run the binary, so print the envelope from the built image and rebuild with it; the second build reuses every cached layer:

```bash
describe="$(docker run --rm chrono-reach:dev --describe)"
docker build -f examples/chrono/Dockerfile -t chrono-reach:dev --label dev.rlmesh.describe="$describe" .
rlmesh check-image chrono-reach:dev
```

Generate it from the image rather than a local build: the envelope records the machine it ran on, and the platform fails a label whose `os` is not `linux`. Pass the same `--image-size`/`--max-steps` the image serves with, since they change the spaces.

## Layout

- `src/reach_env.{h,cpp}`: the Chrono scene, the `rlmesh::Environment`, and its spaces and tags.
- `src/raytracer.h`: the CPU camera.
- `src/main.cpp`: arguments, `rlmesh::EnvServer`, `--describe`, and signal handling.
- `src/scripted_model.cpp`: the scripted policy as a C++ model.
- `run_model.py`: the Python model, with its `ModelSpec` and a scripted or random policy.
