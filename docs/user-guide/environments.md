# Environments

{class}`~rlmesh.EnvFactory` authors an environment's **runtime**: the `tags` that describe its observation and action contract, and the `make()` that constructs it. It is not a build DSL. Packaging (the base image, system libraries, and dependencies) stays in your Dockerfile.

## What it is and when to use it

There are two ways to serve a Gymnasium-style environment with RLMesh.

| Path                                 | Reach for it when                                                                                                                                                                                    |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| {class}`~rlmesh.EnvServer` directly  | An ad-hoc env you already have in hand. `rlmesh.EnvServer(gym.make("CartPole-v1"), "127.0.0.1:5555").serve()` and you are done. See {doc}`serving-environments`.                                     |
| {class}`~rlmesh.EnvFactory` subclass | Anything you ship, serve, sweep, or want a model to auto-adapt to. The factory owns the construction surface (`make` + declared `params`), the catalog of concrete variants, and the adapter `tags`. |

The dividing line is reuse. `EnvServer` serves one object once. `EnvFactory` is the thing you put in a container, hand to a managed dashboard, or sweep across tasks — it carries the metadata those paths need before anything is built.

Authoring an env is independent of authoring a model. Models subclass {class}`~rlmesh.Model` (see {doc}`models`); the two sides meet only through roles, resolved by {doc}`adapters`.

## The minimal factory

Subclass {class}`~rlmesh.EnvFactory`, set `tags`, and implement `make()`.

```python
import rlmesh
import rlmesh.adapters as adapt


class Libero(rlmesh.EnvFactory):
    tags = adapt.EnvTags(
        observation={
            "agentview_image": adapt.ImageTag(adapt.IMAGE_PRIMARY, upside_down=True),
            "robot0_eye_in_hand_image": adapt.ImageTag(adapt.IMAGE_WRIST, upside_down=True),
            "robot0_eef_pos": adapt.StateTag(adapt.EEF_POS),
            "robot0_eef_quat": adapt.StateTag(adapt.EEF_ROT, encoding="quat_xyzw"),
            "robot0_gripper_qpos": adapt.StateTag(adapt.GRIPPER_POS),
            "instruction": adapt.TextTag(adapt.INSTRUCTION),
        },
        action=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
            adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
            clip=(-1.0, 1.0),
        ),
    )

    def make(self, *, task_id: int = 0) -> rlmesh.types.EnvLike:
        from libero.libero import benchmark  # heavy import stays inside make()

        suite = benchmark.get_benchmark_dict()["libero_10"]()
        return LiberoGymEnv(suite, task_id)  # a Gymnasium-shape wrapper, see below
```

`tags` are the obs/action contract a spec'd model resolves against. The role is the first argument on every tag; everything else is the few facts the gymnasium spaces cannot carry. See {doc}`adapters` for what `EnvTags` declares and {doc}`adapters/reference` for the full registry.

`make()`'s return is **auto-stamped** with the factory's `tags` (into `env.metadata`), so the tag rides the environment. A locally-made env resolves the same adapter as a served one — `Libero().make()` driven through {func}`rlmesh.session` and the served endpoint behave identically. Setting `tags = None` (the default) means a generic, un-adapted env.

```{note}
The stamp is validated against the env's spaces lazily, at adapter-resolution time (serve or
session), not inside `make()`. A `make()` that returns a vectorized batch — whose per-lane spaces
differ — is therefore not rejected at construction.
```

## Lifecycle

A factory has three hooks plus the `make` body.

| Hook             | When it runs              | Use for                                                                      |
| ---------------- | ------------------------- | ---------------------------------------------------------------------------- |
| `prepare()`      | Once, before `make()`     | One-time setup: download assets, start a sim daemon, warm a cache. Optional. |
| `make(**kwargs)` | Once per env construction | Build and return the env. Required.                                          |
| `close()`        | On teardown               | Release resources. Optional.                                                 |

```python
class Libero(rlmesh.EnvFactory):
    def prepare(self) -> None:
        download_task_suite()  # one-time, before the first make()

    def make(self, *, task_id: int = 0):
        ...

    def close(self) -> None:
        ...
```

The final `serve(address, **kwargs)` method runs `prepare()`, then `make(**kwargs)`, publishes `tags`, and blocks.

```{caution}
Import heavy or optional dependencies **inside** `make()` and `enumerate_variants()`, not at module
top level. `python -m rlmesh.describe` reflects a factory's schema and catalog without running
`make()`, so a dashboard can list it off-GPU. A simulator or framework imported at module scope
defeats that and pulls the GPU stack into a describe-only call.
```

## Parameters

`make()`'s keyword arguments are its construction surface. Declaring a {class}`~rlmesh.ParamSpec` makes that surface introspectable, validatable, and sweepable: a managed dashboard presents each {class}`~rlmesh.Param` as a widget and **rejects a bad binding before paying GPU cost** — a typo (`task_idd=`) or an out-of-range choice fails pre-construction instead of mid-startup.

```python
class Libero(rlmesh.EnvFactory):
    params = rlmesh.ParamSpec(
        rlmesh.Param("suite", str, choices=("libero_10", "libero_90", "libero_spatial"), group="task"),
        rlmesh.Param("task_id", "int", default=0, description="task index within the suite"),
        rlmesh.Param("camera_size", "int", default=256),
    )

    def make(self, *, suite: str, task_id: int = 0, camera_size: int = 256):
        ...
```

`Param` fields:

| Field         | Meaning                                                                                               |
| ------------- | ----------------------------------------------------------------------------------------------------- |
| `name`        | The keyword, matching `make()`'s parameter.                                                           |
| `type`        | `int`, `float`, `str`, `bool` (the type or its string name), or `"enum"` (domain is `choices` alone). |
| `default`     | The default; **omit to make the param required** (`required` is true when there is no default).       |
| `choices`     | Allowed values — the enumeration / sweep axis; a value outside them is rejected.                      |
| `description` | Help text for the dashboard widget.                                                                   |
| `group`       | Optional UI grouping label (advisory; the core never reads it).                                       |

`ParamSpec(extra=...)` governs undeclared keys. `extra="forbid"` (the default) raises on an unknown key before construction. `extra="passthrough"` forwards undeclared keys verbatim through your own `**kwargs` into a third-party constructor — the escape hatch for a wrapper author. Undeclared keyword args of `make()` are still presented and type-checked from the signature; declaring a `Param` only enriches one with a domain, choices, grouping, and sweepability. `params = None` (the default) keeps a blind passthrough to `make()`.

## Variants

`enumerate_variants()` is the finite, named catalog of concrete sub-environments a factory contains — typically one per benchmark task. Where `ParamSpec` declares independent sweep _axes_, a catalog is a flat list of already-bound, human-named entries — the right shape when the dimensions are dependent (a task index whose range depends on the suite) and each entry has an identity.

```python
class Libero(rlmesh.EnvFactory):
    @classmethod
    def enumerate_variants(cls):
        from libero.libero import benchmark  # lazy, like make()

        suite = benchmark.get_benchmark_dict()["libero_10"]()
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            yield rlmesh.Variant(
                f"libero_10/{task.name}",
                {"suite": "libero_10", "task_id": task_id},
                name=task.name,
                instruction=task.language,
            )
```

{class}`~rlmesh.Variant` takes `id`, a `params` mapping, and free-form `**metadata`:

| Argument     | Meaning                                                                                                                                                                                                        |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`         | Author-explicit, non-empty, factory-unique handle. Prefer a stable upstream identity (`"libero_10/pick_up_the_black_bowl"`) over a positional index, which silently repoints if the upstream library reorders. |
| `params`     | The `make()` binding for **only the identity-defining** params.                                                                                                                                                |
| `**metadata` | Open display bag; `name` is the title a dashboard renders, every other key is domain metadata passed through untouched (e.g. `instruction`).                                                                   |

```{caution}
Identity params go in `Variant.params`; free dials stay in `ParamSpec`. Never describe the same
dimension in both. Above, `suite` and `task_id` define the variant's identity, while `camera_size`
remains a free dial the consumer composes — the variant's free dials are the `ParamSpec` names minus
the variant's `params` keys.
```

## Framework bridge

By default an env's `step`/`reset` seam speaks numpy. For an environment whose `step`/`reset` produce or consume **torch or jax tensors** (a GPU sim, a differentiable env), pin the framework so the seam is typed for it. The wire stays framework-neutral either way — this only types the obs/action seam at the env boundary; it is independent of any consuming model's framework.

There are two ways to pin it, mirroring the model side.

```python
# 1. The author's own class — the framework rides the class, so every serve route
#    (serve, the CLI, a prebuilt image) types the seam without a per-entrypoint flag.
class MyTorchEnv(rlmesh.torch.EnvFactory):
    def make(self, **kwargs):
        ...

# 2. A plain, already-built env — the framework is a value you set on the env side.
rlmesh.EnvServer(env, "127.0.0.1:5555", framework="torch", device="cuda:0")
```

`rlmesh.torch.EnvFactory` and `rlmesh.jax.EnvFactory` subclass {class}`~rlmesh.EnvFactory` and are written exactly the same way. `rlmesh.numpy.EnvFactory` (the default) needs no bridge. For a plain env, {class}`~rlmesh.EnvServer` takes `framework="torch"`/`"jax"`/`"numpy"` and an optional `device` (torch/jax only). See {doc}`backends` for the framework backends and {doc}`../api/torch` for signatures.

```{note}
A torch/jax env cannot be gym-vectorized (`num_envs > 1`): gym vectorization concatenates
observations with numpy, which discards the framework tensors. Serve it scalar — a natively batched
env that returns `[N, ...]` tensors works there — or use `framework="numpy"`.
```

## Common env quirks → which adapter feature

Most authoring effort is matching the environment's actual shape to a tag. Each row links into {doc}`adapters/reference`.

| Quirk                                                            | Reach for                                                                                                                             |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| One flat `Box` observation with fixed index ranges (Metaworld)   | {class}`~rlmesh.adapters.Split` of {class}`~rlmesh.adapters.Field` slices; a role-less `Field` skips indices the model does not read. |
| Upside-down simulator camera (robosuite / LIBERO render flipped) | `adapt.ImageTag(adapt.IMAGE_PRIMARY, upside_down=True)`.                                                                              |
| A `Dict` observation (one key per quantity)                      | A Python `dict` of tags — the container _is_ the space; real nesting, no dotted keys.                                                 |
| A `Tuple` observation                                            | A Python `tuple` of tags.                                                                                                             |
| A single space leaf                                              | A bare leaf, no dict wrapper: `EnvTags(observation=adapt.Split(...), action=...)`.                                                    |
| A second arm (bimanual)                                          | The unsuffixed roles for the first arm, the `_2` roles for the second (`EEF_POS_2`, `ACTION_GRIPPER_2`, ...).                         |

### Wrapping a raw env into Gymnasium shape

RLMesh works with anything that follows the Gymnasium shape: `observation_space`, `action_space`, `reset(seed=None, options=None) -> (obs, info)`, `step(action) -> (obs, reward, terminated, truncated, info)`, and `close()`. A benchmark library that does not expose that shape gets a thin wrapper, returned from `make()`. A common detail is injecting the task instruction as a `Text` observation key so a model can read it through the `adapt.INSTRUCTION` role:

```python
import gymnasium as gym
import numpy as np


class LiberoGymEnv:
    def __init__(self, suite, task_id):
        self._task = suite.get_task(task_id)
        self._env = suite.get_task_env(task_id)
        self.observation_space = gym.spaces.Dict({
            "agentview_image": gym.spaces.Box(0, 255, (256, 256, 3), np.uint8),
            "robot0_eye_in_hand_image": gym.spaces.Box(0, 255, (256, 256, 3), np.uint8),
            "robot0_eef_pos": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
            "robot0_eef_quat": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
            "robot0_gripper_qpos": gym.spaces.Box(-np.inf, np.inf, (2,), np.float32),
            "instruction": gym.spaces.Text(max_length=256),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)

    def reset(self, *, seed=None, options=None):
        obs = self._env.reset()
        return self._observe(obs), {}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return self._observe(obs), reward, done, False, info

    def _observe(self, obs):
        return {**obs, "instruction": self._task.language}  # inject the instruction key

    def close(self):
        self._env.close()
```

The keys in `observation_space` are exactly the keys the `tags` address. Widths, dtypes, and bounds are read from these spaces by the native join step; the tags only name roles and the facts the spaces cannot carry.

## Serve and run

The blocking serve is one call:

```python
Libero().serve("0.0.0.0:50051", suite="libero_10", task_id=0)
```

Or use {class}`~rlmesh.EnvServer` when the process must keep working — `start()` spawns a background thread, `wait()` joins it:

```python
server = rlmesh.EnvServer(Libero().make(suite="libero_10"), port=50051, tags=Libero.tags)
server.start()
print(server.address)
server.wait()
```

Addresses accept `"tcp://host:port"`, `"host:port"`, `"port"`, or `"unix:///path"`, or the `host=`/`port=`/`path=` helpers. See {doc}`serving-environments` for addresses, readiness signals, and health checks.

### The CLI

`python -m rlmesh.serve --env pkg.module:Factory` runs the factory with no hand-written loop — the **lazy path**. It is the entrypoint a container uses.

| Flag            | Env var              | Meaning                                                                                                                           |
| --------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `--address`     | `RLMESH_ADDRESS`     | Bind address (default `0.0.0.0:50051`).                                                                                           |
| `--framework`   | `RLMESH_FRAMEWORK`   | `torch` / `jax` / `numpy`. An `EnvFactory` pins it on the class; needed only for a classless `--env` (a make-callable or gym id). |
| `--device`      | `RLMESH_DEVICE`      | Device for the incoming action (torch/jax only), e.g. `cuda:0`.                                                                   |
| `--kwargs-json` | `RLMESH_MAKE_KWARGS` | JSON object bound to `make(**binding)` — the variation to serve. Validated against `params` before construction.                  |
| —               | `RLMESH_NUM_ENVS`    | Fan the factory out into a vector env (numpy only); the vector server is auto-detected.                                           |

```sh
python -m rlmesh.serve --env environments.libero:Libero \
  --address tcp://0.0.0.0:50051 \
  --kwargs-json '{"suite": "libero_10", "task_id": 3}'
```

### Connecting

A client dials the endpoint with {class}`~rlmesh.RemoteEnv`; it then drives the normal `reset` / `step` / `close` loop. For a managed prebuilt image, {class}`~rlmesh.SandboxEnv` runs the image and connects to it.

```python
env = rlmesh.RemoteEnv("127.0.0.1:50051")
obs, info = env.reset(seed=0)
```

A spec'd model resolves its adapter from the published tags and runs with no glue — see {doc}`models` for `model.run` / `model.session` and {doc}`evaluation` for full evaluations. See {doc}`remote-clients` for the client surface and {doc}`sandbox` for sandboxed environments.

## Build into a container

`EnvFactory` describes the runtime; the Dockerfile describes the package. The pattern: a slim base, the system libraries the simulator needs, your dependencies, a lazy asset/warmup step, and the CLI as the entrypoint.

```dockerfile
FROM python:3.11-slim

# 1. System libraries the simulator needs (here: an EGL/OpenGL stack for MuJoCo).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1-mesa-dev libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*
ENV MUJOCO_GL=egl

WORKDIR /app

# 2. Dependencies. Pin rlmesh to the build your host runs against — the handshake
#    checks compatibility — and add the env's own deps.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 3. Source.
COPY . .

# 4. Warmup step: run prepare() at build time so the task suite / weights it
#    downloads are baked into the image and the first request does not pay for it.
RUN uv run python -c "from environments.libero import Libero; Libero().prepare()"

EXPOSE 50051
# 5. The lazy path: no entrypoint.py, no hand-written serve loop.
CMD ["uv", "run", "python", "-m", "rlmesh.serve", "--env", "environments.libero:Libero"]
```

The same image runs locally (`docker run -p 50051:50051 my-env`, then dial it with `rlmesh.RemoteEnv("127.0.0.1:50051")`) and on RLMesh Managed, which runs the image and connects via {class}`~rlmesh.SandboxEnv`. The runtime variation (which suite, which task) is chosen at run time through `RLMESH_MAKE_KWARGS` / `--kwargs-json`, validated against the factory's `params`, so one image serves the whole catalog.

For a complete, runnable container (both the entrypoint-script form and the lazy CLI form), see {source}`examples/python/byo_container`.
