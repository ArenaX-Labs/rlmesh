# Adapters

`rlmesh.adapters` derives a model-to-environment IO adapter at runtime from two declarations: an environment tags its observation and action spaces, a model specifies the payload it ingests, and {func}`~rlmesh.adapters.resolve` matches them by role. This replaces most of the per-(model, environment) glue you would otherwise write by hand. Cases the declarative specs do not cover fall back to an escape hatch (see {doc}`adapters/escape-hatches`).

It is opt-in. Nothing here is imported by the core Gymnasium loop. Direct adapter calls and the examples below use the NumPy backend (`pip install "rlmesh[numpy]"`); model runtime paths use the active RLMesh backend.

This page is the concept tour. Reach for {doc}`adapters/reference` when you need the full role registry, every field on every leaf, or the conversion policy; reach for {doc}`adapters/escape-hatches` when a declarative spec cannot express the pairing.

## The core idea

The two sides of an eval are declared independently and never import each other. An environment publishes tags, a model declares a spec, and `resolve` bridges them by matching semantic roles.

```{mermaid}
flowchart LR
  tags["Env tags<br/>(roles + a few facts)"] -- matches by role --> R{{"resolve()"}}
  spaces["obs / action spaces"] -- widths / dtypes --> R
  spec["Model spec<br/>(full payload + action layout)"] -- matches by role --> R
  R --> A(["Adapter"])
  A --> obs["transform_obs"]
  A --> act["transform_action"]
```

The asymmetry between the two sides is deliberate.

| Side                                            | What it declares                                                                                                                                                                                    |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Environment ({class}`~rlmesh.adapters.EnvTags`) | Each space entry's **role**, plus the few facts a gymnasium space cannot carry: image axis layout, rotation encoding, an explicit range. Keys, widths, dtypes, and bounds are read from the spaces. |
| Model ({class}`~rlmesh.adapters.ModelSpec`)     | The **full payload** it ingests and the action it emits, in its own conventions: sizes, encodings, container shapes.                                                                                |

Roles are an open vocabulary of strings matched verbatim between the two sides. RLMesh ships well-known conventions (`IMAGE_PRIMARY`, `EEF_POS`, `EEF_ROT`, ...), but any agreed string works. The native `join` step reads widths, dtypes, and bounds from the gymnasium spaces, so the tags stay sparse.

## Tag the environment

An environment tags its observation and action spaces once. The role is the first argument on every tag; everything else is the facts the spaces cannot carry.

```python
import rlmesh.adapters as adapt

tags = adapt.EnvTags(
    observation={
        "wrist_rgb": adapt.ImageTag(adapt.IMAGE_PRIMARY),
        "ee_pos": adapt.StateTag(adapt.EEF_POS),
        "ee_quat": adapt.StateTag(adapt.EEF_ROT, encoding="quat_xyzw"),
        "grip": adapt.StateTag(adapt.GRIPPER_POS),
        "goal": adapt.TextTag(adapt.INSTRUCTION),
    },
    action=adapt.Action(
        adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
        adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
        clip=(-1.0, 1.0),
    ),
)
```

The observation is a tree whose container _is_ the runtime container: a `dict` maps a `Dict` space, a `tuple` maps a `Tuple` space, and a bare leaf tags a single space leaf. Nesting is real `dict` nesting that mirrors a nested `Dict` space (`{"agent": {"eef_pos": adapt.StateTag(adapt.EEF_POS)}}`), not dotted keys.

When you author an environment with {doc}`environments`, the {class}`~rlmesh.EnvFactory` `tags` class attribute is stamped onto the env automatically, so the same tags ride a local env and a served one.

### Flat (non-Dict) observations

Some environments expose a single flat numeric vector with fixed index ranges instead of one key per quantity (Metaworld is the common case). A {class}`~rlmesh.adapters.Split` tags that vector. It is the observation-side mirror of `Action`: a sequence of {class}`~rlmesh.adapters.Field` slices in order, each naming its role with offsets implied by order. A field with no role is a skip that advances the offset over indices the model does not read.

```python
"proprio": adapt.Split(
    adapt.Field(adapt.EEF_POS, 3),
    adapt.Field(adapt.EEF_ROT, 4, encoding="quat_xyzw"),
    adapt.Field(adapt.GRIPPER_POS, 1),
    adapt.Field(dim=10),  # object/goal indices the policy reads from pixels
),
```

`Split` is a leaf, not a container. When the whole observation is one flat box, pass it directly:

```python
adapt.EnvTags(observation=adapt.Split(...), action=adapt.Action(...))
```

A model matches purely by role, so the same spec resolves against a flat env and a `Dict` env with no change. The full `Field` table is in {doc}`adapters/reference`.

## Specify the model

A model fully specifies the payload it ingests and the action it emits, in its own conventions. The role is again the first argument; `size=` sets a square image's height and width together.

```python
spec = adapt.ModelSpec(
    input={
        "image": adapt.Image(adapt.IMAGE_PRIMARY, size=224),
        "proprio": adapt.Concat(
            adapt.EEF_POS,
            adapt.State(adapt.EEF_ROT, encoding="rot6d"),
            adapt.GRIPPER_POS,
        ),
        "task": adapt.Text(adapt.INSTRUCTION),
    },
    output=adapt.Action(
        adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
        adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=6, encoding="rot6d"),
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
    ),
)
```

The `input` is a tree whose container _is_ the payload the prediction function receives: a `dict` (each key a payload slot), a `tuple`, or a bare single leaf. A leaf carries no key — its position in the tree is the payload position, and a role may be reused across leaves. {class}`~rlmesh.adapters.Concat` is the multi-part state leaf: each part is a bare role string (sugar for a role-only `State`) or a `State`, concatenated in order. Every leaf and its options are enumerated in {doc}`adapters/reference`.

## Resolve and apply

{func}`~rlmesh.adapters.resolve` matches the model spec against the tags and the spaces and returns an {class}`~rlmesh.adapters.Adapter`. The adapter preprocesses an observation into the model's input format and postprocesses the model's action back into the environment's.

```python
adapter = adapt.resolve(tags, env.observation_space, env.action_space, spec)
print(adapter.describe())               # the exact transforms chosen
payload = adapter.transform_obs(obs)    # env observation -> model input
action = adapter.transform_action(out)  # model output    -> env action
```

`describe()` prints what the resolver derived. For the pair above the image is resized, the rotation goes `quat_xyzw -> rot6d`, the instruction key is remapped (`goal -> task`), and the 6-d rotation in the model's action is converted back to the env's 3-d `axis_angle` and clipped. Resolution raises {exc}`~rlmesh.adapters.AdapterResolutionError` when a model input or action actuator has no usable counterpart, or when a declared conversion is impossible. The conversion policy in {doc}`adapters/reference` decides which conversions apply silently, warn, or fail.

```{warning}
Specs are pure data. Nothing in a tag or spec is ever evaluated as code. The one exception is
a {class}`~rlmesh.adapters.Custom` input built with `entrypoint=`, which imports a named
`module:callable` only when you pass `resolve(..., trust_entrypoints=True)`.
```

## Run a model with no glue

The shortest path publishes the tags on the served environment and lets the model resolve the adapter from the contract.

```python
server = rlmesh.EnvServer(env, "127.0.0.1:5555", tags=tags)
server.serve()
```

`EnvServer(tags=...)` validates the tags against the environment's spaces and merges them into the contract metadata (the {func}`~rlmesh.adapters.tag` verb does the same for an environment you serve yourself). A model then resolves from the handshake alone.

```python
from rlmesh.numpy import Model, RemoteEnv

env = RemoteEnv("127.0.0.1:5555")
model = Model(predict, spec=spec)  # predict works in the model's own format
model.run(env, max_episodes=10)
```

`run(env)` reads the environment's contract, resolves the adapter, and wraps `predict` so it only ever sees the model's declared payload. To resolve explicitly, use {func}`~rlmesh.adapters.resolve_from_contract` and `adapter.wrap_predict(predict)`. See {doc}`models` for the prediction corners a `predict` may implement, and {doc}`serving-environments` for addresses, readiness, and health. {source}`examples/python/adapters` is the smallest end-to-end serve-and-run loop.

## Frame history

A model that conditions on a short history of frames declares `stack=N` on an image input. The adapter buffers the last `N` processed frames host-side and emits them on a new leading axis (`(N, H, W, C)`), padding the start of an episode with the first frame and clearing the buffer on `reset`.

```python
"image": adapt.Image(adapt.IMAGE_PRIMARY, size=224, stack=4)
```

The environment still sends one frame per step; nothing extra crosses the wire.

```{caution}
Frame stacking is episode state held outside the model: host-side on the local path, in the core
on the served path. The spec's `stack` round-trips through `to_json`, the buffer clears on `reset`,
and the env still sends one frame per step, so nothing extra crosses the wire.
```

## Known limitations

The system targets the manipulation/VLA case: RGB cameras, proprioception, and an instruction. A few things are out of scope for now and fall back to an escape hatch.

| Area                                   | Status                                                                                                                                                                                                       |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Modalities beyond image / state / text | Depth, lidar, and point clouds are not first-class; carry them through a {class}`~rlmesh.adapters.Custom` input or a custom {class}`~rlmesh.adapters.AdapterBase`.                                           |
| Tokenization                           | Stays in the model. `Text` delivers the instruction as a string; tokenize it inside your prediction function. There is intentionally no `TokenizerInput`.                                                    |
| Rotation encodings                     | Fixed set: `quat_xyzw`, `quat_wxyz`, `axis_angle`, `rot6d`, `rot6d_rowmajor`, `euler_xyz`. For a one-off convention, declare a {class}`~rlmesh.adapters.CustomEncoding`; see {doc}`adapters/escape-hatches`. |

## Where next

- {doc}`adapters/reference` — the full role registry (including the bimanual `_2` variants), every field on every leaf, the rotation/layout/fit vocabularies, the conversion policy (silent / advisory / opt-in / error), and how to match your model's shape.
- {doc}`adapters/escape-hatches` — {class}`~rlmesh.adapters.Custom` inputs, {class}`~rlmesh.adapters.AdapterBase` subclasses, pair overrides, and {class}`~rlmesh.adapters.CustomEncoding`.
- {doc}`../api/adapters` — the autodoc signatures for every symbol above.
