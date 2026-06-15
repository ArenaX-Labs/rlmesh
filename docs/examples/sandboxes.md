# Sandbox Examples

Sandbox helpers are experimental. Use them when an environment needs an owned Docker-backed process
instead of a separate server terminal.

The runnable files live in `examples/python/sandbox`.

## Gymnasium Sandbox

Start with the Gymnasium example:

```bash
uv run python examples/python/sandbox/gym_sandbox.py
```

It starts `CartPole-v1` inside a sandbox image and connects with `rlmesh.numpy.SandboxEnv`:

```python
from rlmesh.numpy import SandboxEnv

env = SandboxEnv(
    "CartPole-v1",
    packages=["gymnasium==1.3.0"],
    imports=["gymnasium"],
)
```

`packages` are installed in the sandbox image and `imports` are checked during startup. The client
shape is the same as `RemoteEnv`, so a `try`/`finally` keeps the owned container from leaking:

```python
MAX_STEPS = 45

try:
    obs, info = env.reset(seed=0)
    for step in range(1, MAX_STEPS + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"step={step} reward={reward:.3f}")
        if terminated or truncated:
            print("episode complete")
            break
finally:
    env.close()
```

The runnable file is
{source}`examples/python/sandbox/gym_sandbox.py <examples/python/sandbox/gym_sandbox.py>`.

## Hugging Face Sandbox

`hf_sandbox.py` shows the same single-env loop against a Hugging Face EnvHub source:

```bash
uv run python examples/python/sandbox/hf_sandbox.py
```

Only the constructor changes; the source is an `hf://` reference instead of a Gymnasium id:

```python
from rlmesh.numpy import SandboxEnv

env = SandboxEnv(
    "hf://lerobot/cartpole-env:cartpole_suite/0",
    trust_remote_code=True,
    allow_unpinned_hf=True,
)
```

The selector chooses suite `cartpole_suite`, task `0`. The example uses `SandboxEnv` because it
requests one environment. Use `SandboxVectorEnv` when serving more than one:

```python
from rlmesh.numpy import SandboxVectorEnv

envs = SandboxVectorEnv("CartPole-v1", num_envs=2)
```

The demo is intentionally unpinned; for real evaluations, pin the repository to a full commit SHA
and keep `trust_remote_code=False` unless you have reviewed the source.

The runnable file is
{source}`examples/python/sandbox/hf_sandbox.py <examples/python/sandbox/hf_sandbox.py>`.
