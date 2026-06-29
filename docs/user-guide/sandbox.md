# Sandbox Environments

A sandbox session runs an environment in its own container and connects a remote client to it, so an environment with its own dependencies never has to share your process. A sandbox env _is_ a remote env: it inherits the same `reset` / `step` / `render` / `close` loop and the same contract, and it also owns the container, which it starts on construction and stops on close.

Sandbox helpers are experimental. Pin versions for any real run; see {doc}`/compatibility`. For runnable files, see {doc}`../examples/sandboxes`.

```python
from rlmesh.numpy import SandboxEnv, SandboxBuild

with SandboxEnv(
    "CartPole-v1",
    build=SandboxBuild(packages=["gymnasium==1.3.0"], imports=["gymnasium"]),
    render_mode="rgb_array",
) as env:
    obs, info = env.reset(seed=42)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

The first argument is the source. Everything else splits into three places: `build=` configures how an image is built from source, `runtime=` configures how a prebuilt container is run, and any remaining keyword is forwarded to the environment's constructor. `render_mode` above is one such construction param, passed through to `gym.make`.

## Sources

The source string both names the environment and selects how RLMesh gets it. A gym id or a `gym://` / `hf://` scheme is built into an image from source; a Docker reference is run as-is. RLMesh always logs which kind it resolved, so the choice is never silent.

A Gymnasium id is the simplest source:

```python
SandboxEnv("CartPole-v1", build=SandboxBuild(packages=["gymnasium==1.3.0"]))
```

`gym://CartPole-v1` makes the scheme explicit. The plain id and the `gym://` form resolve to the same kind of source.

A `hf://` source points at a Hugging Face EnvHub repository, which exposes an environment factory as described in the [Hugging Face EnvHub docs](https://huggingface.co/docs/lerobot/envhub). The LeRobot CartPole demo returns suite `cartpole_suite`, task `0`, so the selector is explicit:

```python
from rlmesh.numpy import SandboxEnv, SandboxBuild

with SandboxEnv(
    "hf://lerobot/cartpole-env:cartpole_suite/0",
    build=SandboxBuild(trust_remote_code=True, allow_unpinned_hf=True),
) as env:
    observation, info = env.reset(seed=0)
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
```

A Docker reference (`docker://img`, `image://img`, or a bare `img:tag`) runs a prebuilt rlmesh-serving image directly, with no build step and no rlmesh pin. Construction params are injected into the running container as the environment's make binding.

Use `SandboxVectorEnv` when the selected source serves more than one environment. It takes `num_envs` (which must be at least two) and otherwise mirrors `SandboxEnv`:

```python
from rlmesh.numpy import SandboxVectorEnv

with SandboxVectorEnv("CartPole-v1", num_envs=2) as envs:
    observations, infos = envs.reset(seed=0)
```

## Where each option goes

Three groups of configuration sit on a sandbox session, and they apply at different moments. Passing a build or runtime field directly as a keyword is an error that points you at the right group, so a setting can never silently fall through into the construction binding.

| Pass it via                   | Configures                                   | Applies to                       |
| ----------------------------- | -------------------------------------------- | -------------------------------- |
| `build=SandboxBuild(...)`     | how an image is built from source            | `gym://` / `hf://` / bare gym id |
| `runtime=SandboxRuntime(...)` | `docker run` flags when the container starts | prebuilt image sources           |
| `**params`                    | the env's make/load construction binding     | every source                     |
| `connect_timeout_seconds`     | how long to wait for the container to serve  | every source                     |

### Build options

`SandboxBuild` describes how a from-source image is assembled. It is meaningless for a prebuilt image, which is already built; setting it there is ignored with a warning.

| Field               | Purpose                                                                   |
| ------------------- | ------------------------------------------------------------------------- |
| `packages`          | Python packages installed in the sandbox image                            |
| `imports`           | import names checked before the sandbox is considered ready               |
| `base_image`        | Docker base image override for environments with native dependencies      |
| `rlmesh_package`    | the rlmesh package, wheel path, or `"local"` wheel installed in the image |
| `trust_remote_code` | allow a remote source to execute repository Python code                   |
| `allow_unpinned_hf` | allow unpinned Hugging Face sources; keep it off for reproducibility      |
| `build_memory`      | memory ceiling for the image build                                        |

### Runtime options

`SandboxRuntime` carries `docker run` flags and applies only to prebuilt image sources; a from-source build has no run step and rejects these. Simulation environments that render through a GPU need them.

| Field     | Purpose                                                                                         |
| --------- | ----------------------------------------------------------------------------------------------- |
| `gpus`    | `docker run --gpus`: `"all"`, a count, or a selector like `"device=0,1"` (CUDA compute only)    |
| `devices` | `docker run --device` entries, such as `["nvidia.com/gpu=all"]` for SAPIEN/Vulkan via a CDI ref |
| `volumes` | `docker run -v` mounts for bind-mounting large assets                                           |

### The rlmesh package in the image

When building from source, `rlmesh_package="local"` installs a wheel from `python/rlmesh/dist` in your checkout, which is what you want while developing against an unreleased build. You can also pass an exact wheel path or a pip specifier such as `rlmesh==0.1.0rc2`. For process-wide configuration, set `RLMESH_SANDBOX_RLMESH_PACKAGE`.

```{warning}
The container and your host must currently run the same rlmesh release: the protocol generation
holds a single version, so a mismatch fails the startup handshake (see {doc}`/compatibility`).
Cross-version interoperability is on the roadmap. With `rlmesh_package` unset the container installs
the published release, so a host on an unreleased or source build will mismatch.
Use `rlmesh_package="local"` and keep `python/rlmesh/dist` rebuilt from your checkout.
```

## Startup timing

A sandbox client retries while the container boots. The server inside the container binds its port only after the environment factory's `make()` runs, so an environment that loads large simulations or assets, such as a LIBERO task suite, needs headroom. `connect_timeout_seconds` (default 30) sets how long the client waits before giving up; if the container exits or never becomes ready, the failure includes its recent logs instead of a bare transport error.

## Pinning and safety

The HF demo above is unpinned for convenience. For real evaluations, pin the repository to a full commit SHA:

```text
hf://lerobot/cartpole-env@<full-commit-sha>:cartpole_suite/0
```

```{warning}
Keep `trust_remote_code=False` unless the environment source is trusted. Untrusted environment code
should be pinned and reviewed before it is run.
```

## Where next

- {doc}`serving-environments`: serve an environment yourself instead of letting the sandbox own it.
- {doc}`remote-clients`: the client surface a sandbox session inherits.
- {doc}`backends`: choose the value backend the sandbox client decodes into.
- {doc}`evaluation`: run a model against a sandboxed environment.
- {doc}`environments`: author the `EnvFactory` a prebuilt image serves.
- {doc}`../api/sandbox`: the autodoc for the sandbox classes.
