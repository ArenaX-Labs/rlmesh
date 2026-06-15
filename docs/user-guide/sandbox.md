# Sandbox Environments

Sandbox helpers are experimental in this beta. Use one when an environment needs its own
dependencies and process. The client still uses the normal `reset`, `step`, `render`, and `close`
loop.

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

## Authored recipe

`SandboxEnv` and `SandboxVectorEnv` also accept an authored {doc}`EnvRecipe <env-recipes>`, a
`Recipe`, or a registered env name as the source. When such a recipe declares `ArtifactInput` assets
with a host `local_dir`, those directories are bind-mounted read-only into the sandbox at the input's
target path (validated on the host before the container starts); an `hf://` uri input is fetched
in-container through the rlmesh cache instead.

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
`rlmesh==0.1.0b2`. For process-wide configuration, set `RLMESH_SANDBOX_RLMESH_PACKAGE`.

## Export a Docker image

`SandboxEnv` and `SandboxModel` build an image and run a container in one step. To build the image
and keep it — for example to push it to a registry the RLMesh Managed platform can pull from — call
`rlmesh.export` instead. It builds the image, applies your tag, and returns without starting a
container:

```python
import rlmesh

result = rlmesh.export(MyPolicy, tag="me/my-policy:v1", rlmesh_package="local")
print(result.image)  # rlmesh-sandbox-recipe:<hash> (content-addressed, always applied)
print(result.alias)  # me/my-policy:v1
```

```console
$ docker push me/my-policy:v1
```

`export` works for both env recipes (`EnvRecipe`, a `Recipe`, or a registered env name) and model
recipes (`ModelRecipe`, a `kind="model"` Recipe, or a registered model name). The image is
self-describing: it bakes the recipe document and a kind-aware entrypoint, so `docker run` with no
arguments serves the env or model on port 50051. A human `tag` is optional; the content-addressed
`rlmesh-sandbox-<slug>:<hash>` tag is always applied and is the stable handle to pin.

## Safety

```{warning}
Keep `trust_remote_code=False` unless the environment source is trusted. Untrusted environment code
should be pinned and reviewed before it is run.
```
