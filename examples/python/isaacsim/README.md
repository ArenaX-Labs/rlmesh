# Isaac Sim env (bring-your-own container)

A Franka Panda reaches a random target on Isaac Sim 6.1, with an RTX camera. It's an `EnvFactory` served by `python -m rlmesh.serve`, so the same image runs locally and on RLMesh Managed.

- `prepare()` starts Isaac Sim once per process. `make(lanes=N)` builds N Frankas in one stage. At `lanes=1` (the default) RLMesh serves it as a scalar env; at 2 or more the lanes step together on the GPU as a lockstep vector env.
- Observations are `image` (224×224 RGB), `joint_pos`, `eef_pos` and `target_pos`. The action is a 3-D hand delta of up to 5 cm; the env solves the arm IK.
- Lanes that finish reset on their next step (gymnasium `NEXT_STEP` autoreset), and the env truncates at `max_steps` itself. That's what lockstep requires, so a multi-lane run takes `episodes=` but not per-episode `seeds=` or `max_episode_steps=`. A one-lane run takes all three.
- Isaac Sim only works from the thread that created it. `python -m rlmesh.serve` runs `prepare()`, `make()` and every env call on the main thread, so the env calls Kit directly.

Needs an RTX GPU with a driver Isaac Sim 6.1 supports, and about 11 GB of RAM. Assets stream from NVIDIA's S3 at startup.

## Run locally

```bash
docker build -t franka-reach:dev examples/python/isaacsim
docker run --rm --gpus all --memory=16g -p 50051:50051 \
  -e 'RLMESH_MAKE_KWARGS={"lanes": 4}' franka-reach:dev   # ready after ~45 s
python examples/python/isaacsim/run_local.py   # scripted policy against every lane
```

`--memory` caps the container, so an out-of-memory kill takes down the container instead of your desktop.

## Coming from Isaac Lab

| In Isaac Lab                                                                         | Here                                                                                                                                         |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| The env is always batched, even at `num_envs=1`                                      | Same. Write one batched env; RLMesh serves one lane as a scalar env and N lanes in lockstep.                                                 |
| Finished envs reset inside `step()` (`SAME_STEP`); the terminal frame is in `extras` | Hold the reset for the lane's next step (`NEXT_STEP`) and return the real terminal frame, as `step()` does here. RLMesh refuses `SAME_STEP`. |
| The policy gets a batch of observations                                              | `predict()` gets one lane's observation. Override `predict_batch()` for one batched forward pass.                                            |
| `AppLauncher` starts the app in your script                                          | `prepare()` starts it; `make()` builds the scene.                                                                                            |
| Torch tensors on the GPU                                                             | Return numpy, or subclass `rlmesh.torch.EnvFactory` to serve torch values without converting.                                                |

## Run on RLMesh Managed

```bash
rlmesh login && rlmesh registry login
docker tag franka-reach:dev registry.rlmesh.dev/<namespace>/franka-reach:v1
docker push registry.rlmesh.dev/<namespace>/franka-reach:v1
```

Once the image shows **ready** under Registry, start a New evaluation with it. The `gpu` tag in the image's `dev.rlmesh.package` label schedules the env on a GPU node. Pin `RLMESH_VERSION` to the `rlmesh` release that drives it.
