# Sandbox Examples

These examples start an owned Docker-backed environment and connect to it like any other RLMesh remote environment. They do not need a separate `EnvServer` terminal.

Run commands from the repository root.

## Gymnasium Sandbox

Start here. It builds a sandbox for Gymnasium `CartPole-v1`, samples actions, and stops the container when the script exits.

```bash
uv run python examples/python/sandbox/gym_sandbox.py
```

The script passes `packages=["gymnasium==1.3.0"]` and `imports=["gymnasium"]` so the dependency is installed and checked inside the sandbox.

## Hugging Face Sandbox

`hf_sandbox.py` shows the single-env client loop for a Hugging Face EnvHub source:

```bash
uv run python examples/python/sandbox/hf_sandbox.py
```

It uses `hf://lerobot/cartpole-env:cartpole_suite/0`. The selector chooses suite `cartpole_suite`, task `0`.

The example uses `SandboxEnv` because it requests one environment. Use `SandboxVectorEnv` when you want more than one environment from the same source.

## Model Drives Env

`SandboxModel("image://<tag>").against(env)` is the managed sibling of `RemoteModel(address).against(env)`: it starts the policy in its own container and returns a session you drive with the same `reset`/`predict` loop as the env, so the policy executes in its own container, not in your process. In v0.1 the model container is a prebuilt image you build yourself -- see [`byo_container/model`](../byo_container/model) for the Dockerfile and entrypoint:

```python
import rlmesh

env = rlmesh.RemoteEnv("127.0.0.1:50051")
model = rlmesh.SandboxModel("image://my-model:latest").against(env)
obs, _ = env.reset()
model.reset()
done = False
while not done:
    action = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
```

## Local Development Notes

Sandbox runs need Docker access. The generated image installs RLMesh inside the container. By default that is the published release, which matches a pip-installed host. To run these examples against this checkout instead, pass `rlmesh_package="local"`, which installs a wheel from `python/rlmesh/dist`. Build that wheel first with `mise run build:python:docker`, which produces the manylinux wheel the container can load. `mise run build:python` builds a host-platform wheel that will not load in the container when the host glibc is newer than the base image. To test an exact artifact or published version, pass a wheel path or a pip spec such as `rlmesh==0.1.0rc1`.
