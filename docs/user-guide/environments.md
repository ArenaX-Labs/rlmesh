# Environments

{class}`~rlmesh.EnvFactory` is the authoring base for an environment's _runtime_. A subclass sets the `tags` that describe its observation and action contract and implements `make()` to construct the env. RLMesh stamps the tags onto every env `make()` returns, serves the factory with one call, and reflects its schema off-GPU for a catalog. It is not a build DSL: packaging stays in your Dockerfile.

Authoring an environment is independent of authoring a model. Models subclass {class}`~rlmesh.Model` (see {doc}`models`); the two sides meet only through roles, resolved by {doc}`adapters`.

This page is the concept tour. Reach for {doc}`environments/reference` when you need every `Param` option, the full variant and lifecycle rules, the raw-env wrapper recipe, or the authoritative container build pattern.

## Two ways to serve

There are two ways to serve a Gymnasium-style environment with RLMesh.

| Path                                 | Reach for it when                                                                                                                                                                                    |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| {class}`~rlmesh.EnvServer` directly  | An ad-hoc env you already have in hand. `rlmesh.EnvServer(gym.make("CartPole-v1"), "127.0.0.1:5555").serve()` and you are done. See {doc}`serving-environments`.                                     |
| {class}`~rlmesh.EnvFactory` subclass | Anything you ship, serve, sweep, or want a model to auto-adapt to. The factory owns the construction surface (`make` + declared `params`), the catalog of concrete variants, and the adapter `tags`. |

The dividing line is reuse. `EnvServer` serves one object once. `EnvFactory` is the thing you put in a container, hand to a managed dashboard, or sweep across tasks, so it carries the metadata those paths need before anything is built.

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
        return LiberoGymEnv(suite, task_id)  # a Gymnasium-shape wrapper
```

`tags` are the obs/action contract a spec'd model resolves against. The role is the first argument on every tag; everything else is the few facts the gymnasium spaces cannot carry. See {doc}`adapters` for what `EnvTags` declares and {doc}`adapters/reference` for the full registry. The wrapper that gives a raw benchmark its Gymnasium shape is in {doc}`environments/reference`.

The env `make()` returns is auto-stamped with the factory's `tags` (into `env.metadata`), so the tags ride the environment. A locally-made env resolves the same adapter as a served one: `Libero().make()` driven through {func}`rlmesh.session` and the served endpoint behave identically. Setting `tags = None` (the default) means a generic, un-adapted env.

```{note}
The stamp is validated against the env's spaces lazily, at adapter-resolution time (serve or
session), not inside `make()`. A `make()` that returns a vectorized batch, whose per-lane spaces
differ, is therefore not rejected at construction.
```

## Lifecycle

A factory has three hooks plus the `make` body. The final `serve(address, **kwargs)` runs them in order, publishes `tags`, and blocks.

```{mermaid}
flowchart LR
  A["serve(address, **kwargs)"] --> B["prepare()"]
  B --> C["make(**kwargs)"]
  C --> D["stamp tags into env.metadata"]
  D --> E["publish contract, block"]
```

| Hook             | When it runs              | Use for                                                                      |
| ---------------- | ------------------------- | ---------------------------------------------------------------------------- |
| `prepare()`      | Once, before `make()`     | One-time setup: download assets, start a sim daemon, warm a cache. Optional. |
| `make(**kwargs)` | Once per env construction | Build and return the env. Required.                                          |
| `close()`        | On teardown               | Release resources. Optional.                                                 |

```{caution}
Import heavy or optional dependencies **inside** `make()` and `enumerate_variants()`, not at module
top level. `python -m rlmesh.describe` reflects a factory's schema and catalog without running
`make()`, so a dashboard can list it off-GPU. A simulator or framework imported at module scope
defeats that and pulls the GPU stack into a describe-only call.
```

The full hook contract and `serve()` sequence are in {doc}`environments/reference`.

## Parameters

`make()`'s keyword arguments are its construction surface. Every keyword is already presented and type-checked from the signature. Declaring a {class}`~rlmesh.ParamSpec` enriches one of those knobs with a domain, choices, grouping, and sweepability, so a managed dashboard presents it as a widget and rejects a bad binding before paying GPU cost: a typo (`task_idd=`) or an out-of-range choice fails pre-construction instead of mid-startup.

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

`params = None` (the default) keeps a blind passthrough to `make()`. The full `Param` field table, the `Vector` type, and the `extra="forbid"` / `extra="passthrough"` boundary are in {doc}`environments/reference`.

## Variants

`enumerate_variants()` is the finite, named catalog of concrete sub-environments a factory contains, typically one per benchmark task. Where `ParamSpec` declares independent sweep _axes_, a catalog is a flat list of already-bound, human-named entries: the right shape when the dimensions are dependent (a task index whose range depends on the suite) and each entry has an identity.

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

Return a list, or `yield` lazily for a very large catalog. {class}`~rlmesh.Variant` takes `id`, a `params` mapping of only the identity-defining params, and free-form `**metadata`. Identity params go in `Variant.params`; free dials stay in `ParamSpec`; never describe the same dimension in both. The argument-by-argument rules are in {doc}`environments/reference`.

## Describe

`rlmesh.describe(MyEnv)` (or `MyEnv.describe()`) returns one versioned JSON envelope describing the factory: its `params`, `variants`, `env_tags`, the constructed env's obs/action `env_spec`, and the `runtime` it was generated under. It is the in-process form of the CLI.

```python
schema = rlmesh.describe(MyEnv)        # or MyEnv.describe(); same for a Model
```

```bash
python -m rlmesh.describe --env mypkg.envs:Libero            # prints the envelope
```

The format is owned by the Rust layer, so the bytes are identical across Python versions and future language SDKs. The envelope shape, the `--out` flag, and baking the artifact into an image are in {doc}`environments/reference`.

## Framework bridge

By default an env's `step`/`reset` seam speaks numpy. For an environment whose `step`/`reset` produce or consume **torch or jax tensors** (a GPU sim, a differentiable env), pin the framework so the seam is typed for it. The wire stays framework-neutral either way; this types only the obs/action seam at the env boundary, independent of any consuming model's framework.

There are two ways to pin it, mirroring the model side.

```python
# 1. The author's own class: the framework rides the class, so every serve route
#    (serve, the CLI, a prebuilt image) types the seam without a per-entrypoint flag.
class MyTorchEnv(rlmesh.torch.EnvFactory):
    def make(self, **kwargs):
        ...

# 2. A plain, already-built env: the framework is a value you set on the env side.
rlmesh.EnvServer(env, "127.0.0.1:5555", framework="torch", device="cuda:0")
```

`rlmesh.torch.EnvFactory` and `rlmesh.jax.EnvFactory` subclass {class}`~rlmesh.EnvFactory` and are written exactly the same way. `rlmesh.numpy.EnvFactory` (the default) needs no bridge. For a plain env, {class}`~rlmesh.EnvServer` takes `framework="torch"`/`"jax"`/`"numpy"` and an optional `device` (torch/jax only). See {doc}`backends` for the framework backends and {doc}`../api/torch` for signatures.

```{note}
A torch/jax env cannot be gym-vectorized (`num_envs > 1`): gym vectorization concatenates
observations with numpy, which discards the framework tensors. Serve it scalar, where a natively
batched env that returns `[N, ...]` tensors works, or use `framework="numpy"`.
```

## Common env quirks, by adapter feature

Most authoring effort is matching the environment's actual shape to a tag. Each row links into {doc}`adapters/reference`.

| Quirk                                                            | Reach for                                                                                                                             |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| One flat `Box` observation with fixed index ranges (Metaworld)   | {class}`~rlmesh.adapters.Split` of {class}`~rlmesh.adapters.Field` slices; a role-less `Field` skips indices the model does not read. |
| Upside-down simulator camera (robosuite / LIBERO render flipped) | `adapt.ImageTag(adapt.IMAGE_PRIMARY, upside_down=True)`.                                                                              |
| A `Dict` observation (one key per quantity)                      | A Python `dict` of tags; the container _is_ the space, with real nesting and no dotted keys.                                          |
| A `Tuple` observation                                            | A Python `tuple` of tags.                                                                                                             |
| A single space leaf                                              | A bare leaf, no dict wrapper: `EnvTags(observation=adapt.Split(...), action=...)`.                                                    |
| A second arm (bimanual)                                          | The unsuffixed roles for the first arm, the `_2` roles for the second (`EEF_POS_2`, `ACTION_GRIPPER_2`, ...).                         |

## Serve and run

The blocking serve is one call.

```python
Libero().serve("0.0.0.0:50051", suite="libero_10", task_id=0)
```

Use {class}`~rlmesh.EnvServer` when the process must keep working: `start()` spawns a background thread, `wait()` joins it.

```python
server = rlmesh.EnvServer(Libero().make(suite="libero_10"), port=50051, tags=Libero.tags)
server.start()
print(server.address)
server.wait()
```

`python -m rlmesh.serve --env pkg.module:Factory` runs the factory with no hand-written loop. It is the entrypoint a container uses, and the variation to serve is chosen at run time through `RLMESH_MAKE_KWARGS` / `--kwargs-json`, validated against the factory's `params`. The full flag and environment-variable table is in {doc}`environments/reference`; addresses, readiness signals, and health checks are in {doc}`serving-environments`.

A client dials the endpoint with {class}`~rlmesh.RemoteEnv` and drives the normal `reset` / `step` / `close` loop. For a managed prebuilt image, {class}`~rlmesh.SandboxEnv` runs the image and connects to it.

```python
env = rlmesh.RemoteEnv("127.0.0.1:50051")
obs, info = env.reset(seed=0)
```

A spec'd model resolves its adapter from the published tags and runs with no glue. See {doc}`models` for `model.run` / `model.session`, {doc}`evaluation` for full evaluations, {doc}`remote-clients` for the client surface, and {doc}`sandbox` for sandboxed environments.

## Where next

- {doc}`environments/reference` — every `Param` and `ParamSpec` option, the `Variant` rules, the lifecycle hook contract, the raw-env wrapper recipe, the describe envelope, the serve CLI, and the authoritative container build pattern.
- {doc}`adapters` — how the `tags` you declare here meet a model spec, and {doc}`adapters/reference` for the role registry and every leaf field.
- {doc}`models` — the other side of the boundary: authoring and running a model against your env.
- {doc}`serving-environments` — addresses, readiness, and health for a served endpoint.
