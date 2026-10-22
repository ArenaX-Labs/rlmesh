# Sandbox Examples

These examples start an owned Docker-backed environment and connect to it like any other RLMesh remote environment. They do not need a separate `EnvServer` terminal.

Run commands from the repository root.

## Gymnasium Sandbox

Start here. It builds a sandbox for Gymnasium `CartPole-v1`, samples actions, and stops the container when the script exits.

```bash
uv run python examples/python/sandbox/gym_sandbox.py
```

The script passes `build=SandboxBuild(packages=["gymnasium==1.3.0"], imports=["gymnasium"])` so the dependency is installed and checked inside the sandbox.

## Hugging Face Sandbox

`hf_sandbox.py` shows the single-env client loop for a Hugging Face EnvHub source:

```bash
uv run python examples/python/sandbox/hf_sandbox.py
```

It uses `hf://lerobot/cartpole-env:cartpole_suite/0`. The selector chooses suite `cartpole_suite`, task `0`.

The example uses `SandboxEnv` because it requests one environment. Use `SandboxVectorEnv` when you want more than one environment from the same source.

## Model Drives Env

`rlmesh.SandboxModel("image://<tag>")` is the managed sibling of `rlmesh.Model(address)`: it starts the policy in its own container, which hosts a session, and its `run(env)` and `bind(env)` drive it there, serving a local env in that session. The policy executes in its own container, not in your process. The model container is a prebuilt image you build yourself with rlmesh 0.2 -- see [`byo_container/model`](../byo_container/model) for the Dockerfile and entrypoint:

```python
import rlmesh

with rlmesh.SandboxModel("image://my-model:latest") as model:
    result = model.run(my_env, episodes=4)
```

## Local Development Notes

Sandbox runs need Docker access. The generated image installs RLMesh inside the container. By default that is the published release, which matches a pip-installed host. To run these examples against this checkout instead, pass `build=SandboxBuild(rlmesh_package="local")`, which installs a wheel from `python/rlmesh/dist`. Build that wheel first with `mise run build:python:docker`, which produces the manylinux wheel the container can load. `mise run build:python` builds a host-platform wheel that will not load in the container when the host glibc is newer than the base image. To test an exact artifact or published version, pass a wheel path or a pip spec such as `rlmesh==0.2.0a0`.
