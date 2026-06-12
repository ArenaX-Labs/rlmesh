# Sandbox Examples

Sandbox helpers are experimental. Use them when an environment needs an owned Docker-backed process
instead of a separate server terminal.

The runnable files live in `examples/python/sandbox`.

## Gymnasium Sandbox

Start with the Gymnasium example:

```bash
uv run python examples/python/sandbox/gym_sandbox.py
```

It starts `CartPole-v1` inside a sandbox image and connects with `rlmesh.numpy.SandboxEnv`.

```python
env = SandboxEnv(
    "CartPole-v1",
    packages=["gymnasium==1.3.0"],
    imports=["gymnasium"],
)
```

The client shape remains the same:

```python
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

## Hugging Face Sandbox

`hf_sandbox.py` shows the single-env client loop for a Hugging Face EnvHub source:

```bash
uv run python examples/python/sandbox/hf_sandbox.py
```

```python
env = SandboxEnv(
    "hf://lerobot/cartpole-env:cartpole_suite/0",
    trust_remote_code=True,
    allow_unpinned_hf=True,
)
```

The selector chooses suite `cartpole_suite`, task `0`. The example uses `SandboxEnv` because it
requests one environment. Use `SandboxVectorEnv` when serving more than one environment.

The demo is intentionally unpinned; for real evaluations, pin the repository to a full commit SHA
and keep `trust_remote_code=False` unless you have reviewed the source.
