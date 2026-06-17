# Sandbox Environments

Sandbox helpers are experimental. Use one when an environment needs its own dependencies and
process. Pin versions; see {doc}`/compatibility`. The client still uses the normal `reset`, `step`,
`render`, and `close` loop.

For runnable files, see {doc}`../examples/sandboxes`.

## Gymnasium

The simplest source is a Gymnasium id:

```python
from rlmesh.numpy import SandboxEnv

with SandboxEnv(
    "CartPole-v1",
    packages=["gymnasium==1.3.0"],
    imports=["gymnasium"],
    render_mode="rgb_array",
) as env:
    obs, info = env.reset(seed=42)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

Use `gym://CartPole-v1` when you want to make the source scheme explicit. The plain Gymnasium id and
the `gym://` form resolve to the same kind of source.

## Hugging Face EnvHub

Sandbox sources can point at Hugging Face EnvHub repositories with `hf://`. These repositories
expose an environment factory, as described in the
[Hugging Face EnvHub docs](https://huggingface.co/docs/lerobot/envhub).

The LeRobot CartPole demo returns suite `cartpole_suite`, task `0`, so the selector is explicit:

```python
from rlmesh.numpy import SandboxEnv

with SandboxEnv(
    "hf://lerobot/cartpole-env:cartpole_suite/0",
    trust_remote_code=True,
    allow_unpinned_hf=True,
) as env:
    observation, info = env.reset(seed=0)
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
```

Use `SandboxVectorEnv` when the selected source serves more than one environment.

The demo is unpinned for convenience. For real evaluations, pin to a full commit SHA:

```text
hf://lerobot/cartpole-env@<full-commit-sha>:cartpole_suite/0
```

## Runtime Options

| Option              | Use it for                                                               |
| ------------------- | ------------------------------------------------------------------------ |
| `packages`          | Python packages installed in the sandbox environment.                    |
| `imports`           | Import names checked before the sandbox is considered ready.             |
| `base_image`        | Docker base image override for environments with native dependencies.    |
| `rlmesh_package`    | RLMesh package, wheel path, or `"local"` wheel installed in the sandbox. |
| `trust_remote_code` | Required when a remote source needs to execute repository Python code.   |
| `allow_unpinned_hf` | Allows unpinned Hugging Face sources; keep this off for reproducibility. |
| `**gym_make_kwargs` | Keyword arguments forwarded to Gymnasium or EnvHub environment creation. |

Use `rlmesh_package="local"` from the RLMesh checkout to install a wheel from `python/rlmesh/dist`
into the sandbox image. You can also pass an exact wheel path or a pip package specifier such as
`rlmesh==0.1.0rc1`. For process-wide configuration, set `RLMESH_SANDBOX_RLMESH_PACKAGE`.

```{warning}
The container and your host must currently run the same rlmesh release: the protocol generation
holds a single version, so a mismatch fails the startup handshake (see {doc}`/compatibility`).
Cross-version interoperability is on the roadmap. With `rlmesh_package` unset the container installs
the published release, so a host on an unreleased or source build will mismatch.
Use `rlmesh_package="local"` and keep `python/rlmesh/dist` rebuilt from your checkout.
```

## Safety

```{warning}
Keep `trust_remote_code=False` unless the environment source is trusted. Untrusted environment code
should be pinned and reviewed before it is run.
```
