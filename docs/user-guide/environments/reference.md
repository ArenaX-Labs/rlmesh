# Environment reference

The complete field-by-field reference for authoring an environment factory. Use it to match your environment to a feature, then look up the exact behavior of every hook, parameter, and variant rule.

For the concepts (the two sides of an eval and how they meet), start with {doc}`/user-guide/environments`. For the obs/action contract a factory's `tags` declare, see {doc}`/user-guide/adapters` and {doc}`/user-guide/adapters/reference`. For the other side of the boundary, see {doc}`/user-guide/models`; for the serving mechanics, {doc}`/user-guide/serving-environments`. Exact signatures live in {doc}`/api/env-server` and {doc}`/api/core`. Runnable examples are at {source}`examples/python/byo_container`.

## EnvFactory

{class}`~rlmesh.EnvFactory` is an abstract base. A subclass describes one obs/action contract: set `tags` to that contract, implement `make()` to build the env, and optionally declare `params` and `enumerate_variants()`.

| Class member           | Kind             | Default | What it is                                                                                      |
| ---------------------- | ---------------- | ------- | ----------------------------------------------------------------------------------------------- |
| `tags`                 | class attribute  | `None`  | The {class}`~rlmesh.adapters.EnvTags` obs/action contract; `None` is a generic, un-adapted env. |
| `params`               | class attribute  | `None`  | A {class}`~rlmesh.ParamSpec` over `make()`'s keywords; `None` is a blind passthrough.           |
| `prepare()`            | method           | no-op   | One-time setup before the first `make()`.                                                       |
| `make(**kwargs)`       | method, required | --      | Build and return one env (or a vectorized batch).                                               |
| `close()`              | method           | no-op   | Release resources on teardown.                                                                  |
| `serve(address, **kw)` | method, final    | --      | Blocking host: `prepare()`, `make(**kw)`, publish `tags`.                                       |
| `enumerate_variants()` | classmethod      | absent  | Return (or `yield`) the {class}`~rlmesh.Variant` catalog of concrete sub-envs.                  |
| `describe()`           | classmethod      | --      | The factory's metadata envelope; see [Describe](#describe).                                     |

### Tag stamping

`make()`'s return is stamped with the factory's `tags` into `env.metadata`, so the contract rides the environment. A locally-made env resolves the same adapter as a served one: `Libero().make()` driven through {func}`rlmesh.session` and the served endpoint behave identically.

The stamp is validated against the env's spaces **lazily**, at adapter-resolution time (serve or session), not inside `make()`. This is deliberate: a `make()` that returns a vectorized batch, whose per-lane spaces differ from the served shape, is not rejected at construction. Serving a scalar env validates the published tags at startup instead, so a bad tag surfaces before a model first connects; a vector env keeps the deferred, resolve-time check.

Stamping is idempotent. Serving an already-stamped env re-stamps the same tags rather than duplicating them, so the local path and the served path agree.

## Lifecycle

```{mermaid}
flowchart LR
  A["serve(address, **kwargs)"] --> B["prepare()"]
  B --> C["make(**kwargs)"]
  C --> D["stamp tags into env.metadata"]
  D --> E["publish contract, block"]
```

| Hook             | When it runs              | Reach for it when                                                                                                           |
| ---------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `prepare()`      | Once, before `make()`     | A one-time cost the env should not pay per construction: download a task suite, start a sim daemon, warm a cache. Optional. |
| `make(**kwargs)` | Once per env construction | Always. Build and return the env; task selection and `num_envs` are its parameters, not separate subclasses. Required.      |
| `close()`        | On teardown               | The env holds a resource that outlives a single step loop: a subprocess, a GPU context, a file handle. Optional.            |

`serve(address, **kwargs)` is `final`: it runs `prepare()`, then `make(**kwargs)`, publishes `tags`, and blocks. The same sequence runs under `python -m rlmesh.serve` and inside a container.

```{caution}
Import heavy or optional dependencies **inside** `make()` and `enumerate_variants()`, not at module
top level. `rlmesh.describe` reflects a factory's `params` and catalog by inspecting signatures and
running the `enumerate_*` classmethods, without needing a working `make()`, so a dashboard can list
the factory off-GPU. A simulator or framework imported at module scope defeats that and pulls the GPU
stack into a describe-only call. (Describe still *attempts* one representative `make()` to capture the
env's spaces; that step is best-effort and degrades to an error badge when it cannot run, so the rest
of the envelope still ships.)
```

## Parameters

`make()`'s keyword arguments are its construction surface. Two tiers cover that surface:

- The **signature-derived floor**: every keyword of `make()` is presented and type-checked from the signature for free. The four scalar annotations (`int`, `float`, `str`, `bool`) are checked; anything else passes through verbatim.
- The **declared ceiling**: a {class}`~rlmesh.ParamSpec` of {class}`~rlmesh.Param` entries enriches chosen knobs with a domain, choices, grouping, and sweepability, and validates them before construction.

Declaring a `Param` is the act of marking a knob primary. A managed dashboard presents it as a first-class widget, validates it (type, choices, required) before paying GPU cost, and offers it as a sweep axis. A typo (`task_idd=`) or an out-of-range choice fails pre-construction instead of mid-startup.

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

### Param fields

| Field         | Default | What it declares                                                                                                                                |
| ------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`        | --      | The keyword, matching `make()`'s parameter.                                                                                                     |
| `type`        | `"str"` | `int`, `float`, `str`, `bool` (the type or its string name), `"enum"` (domain is `choices` alone), or a `Vector` (a fixed-length float vector). |
| `choices`     | `None`  | Allowed values: the enumeration / sweep axis. A supplied value outside them is rejected.                                                        |
| `description` | `""`    | Help text for the dashboard widget.                                                                                                             |
| `group`       | `None`  | Optional UI grouping label (advisory; the core never reads it).                                                                                 |

The default lives in the `make()` signature, never on the `Param`. A param with a signature default is optional and presents that default; one without is required. A `Param` only enriches a knob with choices, grouping, description, and sweepability.

`type` coercion is light and strict. `bool` is rejected where an `int` or `float` is expected (so `True` is never silently `1`), a non-integral `float` is rejected for an `int`, and a non-finite `float` (`NaN`/`inf`) is rejected outright since a construction param is never legitimately either. `"enum"` and any opaque custom type skip coercion and rely on `choices` for their domain.

### The Vector type

A `Vector` is a fixed-length float-vector `Param` type, passed as `Param("offset", type=Vector(3))`. The value stays a plain tuple of `dim` finite floats; a JSON-bound `list` (from an env var or recorded metadata) is canonicalized to a tuple. `Vector(3, unit=True)` additionally requires unit L2 norm (the quaternion or direction case), within a small tolerance. `Vector` is only a schema descriptor in `Param.type`; it is not a container the runtime carries. Sweeps come from `choices` as a list of whole vectors, since there is no continuous vector domain.

### ParamSpec and the extra boundary

`ParamSpec(*params, extra="forbid")` is the validated ceiling over the free signature-derived floor. `extra` governs the single door for undeclared keys.

| `extra`         | Behavior                                                                                                                                                         | Reach for it when                                            |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `"forbid"`      | Default. An undeclared key raises before construction, so a typo (`robtos=`) fails pre-GPU instead of vanishing.                                                 | The construction surface is fully known.                     |
| `"passthrough"` | Undeclared keys forward verbatim through the author's own `**kwargs` into a third-party constructor. Bounded by that `**kwargs`, never by any downstream target. | You wrap a constructor whose keyword surface you do not own. |

`params = None` (the default factory state) skips all of this and keeps a blind passthrough to `make()`, for full back-compat.

A declared `Param` that is not a keyword of `make()` and is not covered by `**kwargs` would silently fall through to passthrough, so the validation it promised never runs. That case warns rather than fails (it is the author's mistake, not the operator's). With `extra="passthrough"` set but no `**kwargs` on `make()` to forward to, an undeclared key still raises, since there is nowhere to forward it.

### Validation errors

The resolve step runs in the container, before construction, so a bad binding fails before GPU cost. It raises subclasses of `rlmesh.params.ParamError` (a `ValueError`):

- `rlmesh.params.ParamError` — a supplied parameter failed type, choices, or range validation.
- `rlmesh.params.MissingParamError` — a required declared parameter was not supplied.
- `rlmesh.params.UnknownParamError` — a key was neither declared nor a `make()` keyword, under `extra="forbid"` (or under `extra="passthrough"` with no `**kwargs`).

## Variants

`enumerate_variants()` returns (or `yield`s) the finite, named catalog of concrete sub-environments a factory contains, typically one per benchmark task. It is distinct from a sweep: `ParamSpec` declares independent _axes_, while a catalog is a flat list of already-bound, human-named entries. Reach for a catalog when the dimensions are dependent (a task index whose range depends on the suite) and each entry carries a human identity.

```python
class Libero(rlmesh.EnvFactory):
    @classmethod
    def enumerate_variants(cls):
        from libero.libero import benchmark  # lazy, like make()

        suite = benchmark.get_benchmark_dict()["libero_10"]()
        variants = []
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            variants.append(
                rlmesh.Variant(
                    f"libero_10/{task.name}",
                    {"suite": "libero_10", "task_id": task_id},
                    name=task.name,
                    instruction=task.language,
                )
            )
        return variants
```

Return a list, or `yield` lazily for a very large catalog. Both work. `python -m rlmesh.describe` emits the catalog off-GPU for a managed dashboard or env hub to list and spawn.

### Variant arguments

{class}`~rlmesh.Variant` takes `id` and a `params` mapping positionally, then free-form `**metadata`.

| Argument     | What it is                                                                                                                                                                                                                                                                 |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`         | Author-explicit, non-empty, factory-unique handle. Prefer a stable upstream identity (`"libero_10/pick_up_the_black_bowl"`) over a positional index, which silently repoints if the upstream library reorders. A hub composes a global handle as `(factory identity, id)`. |
| `params`     | The `make()` binding for **only the identity-defining** params. Copied defensively, so reusing one dict across entries in a loop is safe.                                                                                                                                  |
| `**metadata` | Open display bag (keyword-only). `name` is the title a dashboard renders; every other key is domain metadata passed through untouched (e.g. `instruction`).                                                                                                                |

```{caution}
Identity params go in `Variant.params`; free dials stay in `ParamSpec`. Never describe the same
dimension in both. Above, `suite` and `task_id` define the variant's identity, while `camera_size`
remains a free dial the consumer composes. A variant's free dials are the `ParamSpec` names minus
the variant's `params` keys.
```

A variant's `id` must be a unique, non-empty string. `describe` rejects a duplicate id (a duplicate would silently collapse a by-id spawn map) and validates each variant's `params` against the `ParamSpec` and `make()` signature off-GPU; an unbuildable variant gets an error badge but keeps its params verbatim, so the catalog never silently drops or rewrites an entry.

## Wrapping a raw env into Gymnasium shape

RLMesh works with anything that follows the Gymnasium shape:

- `observation_space`
- `action_space`
- `reset(seed=None, options=None) -> (obs, info)`
- `step(action) -> (obs, reward, terminated, truncated, info)`
- `close()`

A benchmark library that does not expose that shape gets a thin wrapper, returned from `make()`. A common detail is injecting the task instruction as a `Text` observation key so a model can read it through the `adapt.INSTRUCTION` role.

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

The keys in `observation_space` are exactly the keys the `tags` address. Widths, dtypes, and bounds are read from these spaces by the native join step; the tags only name roles and the facts the spaces cannot carry. Common Gymnasium wrappers can stay in place. A vectorized env exposes `num_envs`, `single_observation_space`, and `single_action_space` instead, and {class}`~rlmesh.EnvServer` detects that shape and serves a vector endpoint.

## Describe

`rlmesh.describe(MyEnv)` (or `MyEnv.describe()`) returns one self-contained JSON envelope describing the factory. It is generated once, at build or generate time, and is forward-compatible with an OCI image label baked later.

```python
schema = rlmesh.describe(MyEnv)        # parsed dict; same for a Model
```

```bash
python -m rlmesh.describe --env mypkg.envs:Libero            # prints the envelope
python -m rlmesh.describe --env mypkg.envs:Libero --out describe.json
```

The envelope (env kind) carries:

| Key              | Contents                                                                                                       |
| ---------------- | -------------------------------------------------------------------------------------------------------------- |
| `schema_version` | Rust-stamped format version.                                                                                   |
| `kind`           | `"env"` (a model envelope is `"model"`).                                                                       |
| `target`         | The entrypoint and qualname, so the artifact maps back to its source.                                          |
| `env_spec`       | The constructed env's `observation_space` / `action_space` (+ `num_envs` for a vector env), or an error badge. |
| `env_tags`       | The factory's `tags` serialized, or `null`.                                                                    |
| `params`         | The declared `param_spec` plus the free `signature_tier`.                                                      |
| `variants`       | The `catalog` from `enumerate_variants()` and the `variations` axes from `enumerate_params()`.                 |
| `runtime`        | The peer info it was generated under: Python and framework versions, OS, arch.                                 |

Every gathered piece is best-effort: a failure to build the env, read a spec, or run an enumeration becomes an `"error"` badge rather than a crash, so the artifact is always emitted (for example, a no-GPU OCI build still ships a valid envelope minus its `env_spec`). A model envelope drops `env_spec`/`env_tags` and carries `model_spec` instead.

The format (version, shape, serialization) is owned by the Rust layer, so the bytes are identical across Python versions and any future native producer. See [the contract](../../specs/describe.v1.md). The artifact is self-contained JSON, ready to bake into an image at build time:

```dockerfile
RUN python -m rlmesh.describe --env mypkg.envs:Libero --out /etc/rlmesh/describe.json
```

Use `rlmesh.describe_json(...)` when you need the exact byte-stable string (for an OCI label) rather than a parsed dict. Pass `generated_at=` for an RFC-3339 timestamp, or omit it for a content-addressable artifact.

## The serve CLI

`python -m rlmesh.serve --env pkg.module:Factory` runs the factory with no hand-written loop. The target may be an {class}`~rlmesh.EnvFactory`, a bare make-env callable, or a gym id. It is the entrypoint a container uses.

| Flag            | Env var                     | Meaning                                                                                                                                            |
| --------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--address`     | `RLMESH_ADDRESS`            | Bind address (default `0.0.0.0:50051`).                                                                                                            |
| `--framework`   | `RLMESH_FRAMEWORK`          | `torch` / `jax` / `numpy`. An `EnvFactory` pins it on the class; needed only for a classless `--env` (a make-callable or gym id).                  |
| `--device`      | `RLMESH_DEVICE`             | Device for the incoming action (torch/jax only), e.g. `cuda:0`. Ignored for numpy and the default backend.                                         |
| `--kwargs-json` | `RLMESH_MAKE_KWARGS`        | JSON object bound to `make(**binding)`: the variation to serve. Absent serves `make()`'s defaults. Validated against `params` before construction. |
| --              | `RLMESH_NUM_ENVS`           | Fan the factory out into a vector env (numpy only); the vector server is auto-detected.                                                            |
| --              | `RLMESH_VECTORIZATION_MODE` | Vectorization mode for the fan-out.                                                                                                                |

```sh
python -m rlmesh.serve --env environments.libero:Libero \
  --address tcp://0.0.0.0:50051 \
  --kwargs-json '{"suite": "libero_10", "task_id": 3}'
```

`num_envs` and `vectorization_mode` control vectorization, not env construction. Passing either inside `--kwargs-json` / `RLMESH_MAKE_KWARGS` is an error; set `RLMESH_NUM_ENVS` / `RLMESH_VECTORIZATION_MODE` instead. A torch/jax env cannot be fanned out this way: gym vectorization concatenates observations with numpy, which discards the framework tensors. Serve it scalar, or use `framework="numpy"`. Adapters resolve per single-env lane, so a vector serve publishes no tags; serve scalar to publish the contract.

Addresses accept `"tcp://host:port"`, `"host:port"`, `"port"`, or `"unix:///path"`, or the `host=`/`port=`/`path=` helpers on {class}`~rlmesh.EnvServer`. For readiness signals and health checks, see {doc}`/user-guide/serving-environments`.

## Build into a container

`EnvFactory` describes the runtime; the Dockerfile describes the package. The pattern: a slim base, the system libraries the simulator needs, your dependencies, a lazy asset or warmup step, and the CLI as the entrypoint.

```dockerfile
FROM python:3.11-slim

# 1. System libraries the simulator needs (here: an EGL/OpenGL stack for MuJoCo).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1-mesa-dev libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*
ENV MUJOCO_GL=egl

WORKDIR /app

# 2. Dependencies. Pin rlmesh to the build your host runs against -- the handshake
#    checks compatibility -- and add the env's own deps.
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

## Where next

- {doc}`/user-guide/environments` — the concept tour these fields back: the factory, tagging, params, variants, and serving in brief.
- {doc}`/user-guide/adapters/reference` — the role registry, every leaf field, and the conversion policy your `tags` resolve under.
- {doc}`/user-guide/models` — authoring and running the model that resolves against this env.
- {doc}`/user-guide/serving-environments` — addresses, readiness, and health for a served endpoint.
