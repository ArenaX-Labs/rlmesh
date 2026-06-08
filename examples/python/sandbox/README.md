# Sandbox Examples

These examples start an owned Docker-backed environment and connect to it like any other RLMesh
remote environment. They do not need a separate `EnvServer` terminal.

Run commands from the repository root.

## Gymnasium Sandbox

Use this first. It builds a sandbox for Gymnasium `CartPole-v1`, samples actions, and stops the
container when the script exits.

```bash
uv run python examples/python/sandbox/gym_sandbox.py
```

The script passes `packages=["gymnasium==1.3.0"]` and `imports=["gymnasium"]` so the dependency is
installed and checked inside the sandbox.

## Hugging Face Sandbox

`hf_sandbox.py` shows the single-env client loop for a Hugging Face EnvHub source:

```bash
uv run python examples/python/sandbox/hf_sandbox.py
```

It uses `hf://lerobot/cartpole-env:cartpole_suite/0`. The selector chooses suite `cartpole_suite`,
task `0`.

The example uses `SandboxEnv` because it requests one environment. Use `SandboxVectorEnv` when you
want more than one environment from the same source.

The demo opts into `trust_remote_code=True` and `allow_unpinned_hf=True`. For real evaluations, pin
the source to a full commit SHA and only enable remote code for repositories you have reviewed:

```text
hf://lerobot/cartpole-env@<full-commit-sha>:cartpole_suite/0
```

## Local Development Notes

Sandbox runs need Docker access. The generated image installs RLMesh inside the container. To test
an unreleased local wheel from this checkout, pass `rlmesh_package="local"`. To test an exact
artifact or published version, pass a wheel path or a pip spec such as `rlmesh==0.1.0b2`.
