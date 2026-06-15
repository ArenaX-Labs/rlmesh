# Adapters

`rlmesh.adapters` derives a model-to-environment IO adapter at runtime from two declarations: an
environment tags its observation and action spaces, a model specifies the payload it ingests, and
{func}`~rlmesh.adapters.resolve` matches them by role. It removes most of the per-(model,
environment) adapter code you would otherwise write by hand; cases the declarative specs do not
cover fall back to an escape hatch (see Known limitations).

Adapters connect the two halves of the recipe family: an {doc}`environment recipe <env-recipes>`
publishes tags, a {doc}`model recipe <model-recipes>` declares a spec, and `resolve` bridges them.

```{note}
`rlmesh.adapters` is **experimental** in this beta: it may change or disappear before the
stable release. Pin versions; see {doc}`/compatibility`.
```

It is fully opt-in. Nothing here is imported by the core Gymnasium loop, and it needs the NumPy
backend (`pip install --pre "rlmesh[numpy]"`).

## Tag the environment

An environment tags its observation and action spaces. Tags are sparse: they carry each entry's
semantic role plus the few facts the gymnasium spaces cannot, such as image axis layout or rotation
encoding. Keys, widths, dtypes, and bounds are read from the spaces.

```python
import rlmesh.adapters as adapt

tags = adapt.EnvTags(
    observation={
        "wrist_rgb": adapt.ImageTag(role=adapt.IMAGE_PRIMARY),
        "ee_pos": adapt.StateTag(role=adapt.EEF_POS),
        "ee_quat": adapt.StateTag(role=adapt.EEF_ROT, encoding="quat_xyzw"),
        "grip": adapt.StateTag(role=adapt.GRIPPER_POS),
        "goal": adapt.TextTag(),
    },
    action=adapt.ActionLayout(
        adapt.ActionComponent(adapt.ACTION_DELTA_POS, dim=3),
        adapt.ActionComponent(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
        adapt.ActionComponent(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
        clip=(-1.0, 1.0),
    ),
)
```

The observation map is keyed by observation path; dotted keys (`"agent.eef_pos"`) traverse nested
`Dict` spaces. Roles are an open vocabulary of strings matched verbatim between tags and specs.
RLMesh ships well-known conventions (`IMAGE_PRIMARY`, `EEF_POS`, `EEF_ROT`, ...), but any agreed
string works.

### Flat (non-Dict) observations

Some environments expose a single flat numeric vector with fixed index ranges instead of one key per
quantity (Metaworld is the common case). A {class}`~rlmesh.adapters.StateLayout` tags that vector.
It is the observation-side mirror of `ActionLayout`: a sequence of
{class}`~rlmesh.adapters.StateField` slices in order, each naming its role and offsets implied by
order. A field with no role is a skip that advances the offset over indices the model does not read.

```python
"proprio": adapt.StateLayout(
    adapt.StateField(adapt.EEF_POS, 3),
    adapt.StateField(adapt.EEF_ROT, 4, encoding="quat_xyzw"),
    adapt.StateField(adapt.GRIPPER_POS, 1),
    adapt.StateField(dim=10),  # object/goal indices the policy reads from pixels
),
```

When the whole observation is one leaf, pass the `StateLayout` directly as `observation`:

```python
adapt.EnvTags(observation=adapt.StateLayout(...), action=adapt.ActionLayout(...))
```

A model matches purely by role, so the same spec resolves against a flat env and a `Dict` env with
no change.

## Specify the model

A model fully specifies the payload it ingests and the action it emits, in its own conventions.

```python
spec = adapt.ModelSpec(
    inputs=(
        adapt.ImageInput("image", role=adapt.IMAGE_PRIMARY, height=224, width=224),
        adapt.StateInput(
            "proprio",
            components=(
                adapt.StateComponent(adapt.EEF_POS),
                adapt.StateComponent(adapt.EEF_ROT, encoding="rot6d"),
                adapt.StateComponent(adapt.GRIPPER_POS),
            ),
            container="list",
        ),
        adapt.TextInput("task"),
    ),
    action=adapt.ActionLayout(
        adapt.ActionComponent(adapt.ACTION_DELTA_POS, dim=3),
        adapt.ActionComponent(adapt.ACTION_DELTA_ROT, dim=6, encoding="rot6d"),
        adapt.ActionComponent(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
    ),
)
```

## Resolve and apply

{func}`~rlmesh.adapters.resolve` matches the model spec against the tags and the spaces and returns
an {class}`~rlmesh.adapters.Adapter`. The adapter preprocesses an observation into the model's input
format and postprocesses the model's action back into the environment's.

```python
adapter = adapt.resolve(tags, env.observation_space, env.action_space, spec)
print(adapter.describe())          # the exact transformations chosen
payload = adapter.transform_obs(obs)      # env observation -> model input
action = adapter.transform_action(output) # model output    -> env action
```

`describe()` prints what the resolver derived. Here the image is resized, the rotation goes
`quat_xyzw -> rot6d`, the instruction key is remapped (`goal -> task`), and the 10-dim action is
converted `rot6d -> axis_angle`, sliced, and clipped into the env's 7-dim action. Resolution fails
with an {exc}`~rlmesh.adapters.AdapterResolutionError` if a model input or action component has no
usable counterpart.

```{warning}
Specs are pure data. Nothing in a tag or spec is ever evaluated as code. The one exception is
{class}`~rlmesh.adapters.EntrypointCustomInput`, which imports a named `module:callable` only
when you pass `resolve(..., trust_entrypoints=True)`.
```

## Run a model with no glue

The ergonomic path publishes the tags on the served environment and lets the model resolve the
adapter from the contract.

```python
server = rlmesh.EnvServer(env, "127.0.0.1:5555", tags=tags)
server.serve()
```

`EnvServer(tags=...)` validates the tags against the environment's spaces and merges them into the
contract metadata (the {func}`~rlmesh.adapters.tag` verb does the same for an environment you serve
yourself). A model then resolves from the handshake alone.

```python
from rlmesh.numpy import Model, RemoteEnv

env = RemoteEnv("127.0.0.1:5555")
model = Model(predict_fn, spec=spec)   # predict_fn works in the model's own format
model.run(env, max_episodes=10)
```

`run(env)` reads the environment's contract, resolves the adapter, and wraps `predict_fn` so it only
ever sees the model's declared payload. To resolve explicitly, use
{func}`~rlmesh.adapters.resolve_from_contract` and `adapter.wrap_predict(predict_fn)`.

## Frame history

A model that conditions on a short history of frames declares `stack=N` on an image input. The
adapter buffers the last `N` processed frames host-side and emits them on a new leading axis
(`(N, H, W, C)`), padding the start of an episode with the first frame and clearing the buffer on
`reset`.

```python
ImageInput("image", role=IMAGE_PRIMARY, size=224, stack=4)
```

The environment still sends one frame per step; nothing extra crosses the wire.

```{caution}
Frame stacking is host-side state. A spec that sets `stack` round-trips through `to_json`, but
the native resolution ignores it; stacking happens in the adapter, not the core.
```

## Escape hatches

When a pairing needs logic a declarative spec cannot express, three mechanisms compose, most local
first.

| Mechanism                                                                                     | Use                                                                                                                                                         |
| --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| {class}`~rlmesh.adapters.InlineCustomInput` / {class}`~rlmesh.adapters.EntrypointCustomInput` | Compute one payload key from the raw observation; the rest stays spec-driven.                                                                               |
| {class}`~rlmesh.adapters.AdapterBase` subclass                                                | Add stateful behavior a spec cannot describe (for example temporal ensembling), usually by wrapping a resolved adapter.                                     |
| Pair override                                                                                 | Replace the adapter for one (model, environment) pairing entirely. No special machinery: keep a registry keyed by the pair and consult it before resolving. |

```python
OVERRIDES: dict[tuple[str, str], Callable[[], adapt.AdapterBase]] = {
    ("xvla", "simpler-bridge"): XVLABridgeAdapter,
}

def build_adapter(model_name, env_name, ...):
    if (factory := OVERRIDES.get((model_name, env_name))) is not None:
        return factory()
    return adapt.resolve(...)
```

The {source}`examples/python/vla_adapters <examples/python/vla_adapters>` example shows all three
over several VLA models and environments;
{source}`examples/python/adapters <examples/python/adapters>` is the smallest end-to-end
serve-and-run loop.

### Custom encodings

Rotation encodings are a closed vocabulary, because a spec must resolve on a remote client with no
code. For a general, stable convention (a published model's `rot6d_rowmajor`), add it first-party on
the native `RotationEncoding` enum so it serializes into the contract and is conformance-tested. For
a one-off, declare a {class}`~rlmesh.adapters.CustomEncoding` on the nearest base encoding and
supply host-side repacking; reach for first-party once you want it matched by role and reused. The
{doc}`../api/adapters` reference covers `CustomEncoding`, the `from_base`/`to_base` boundary, and
the resolve-time invariants.

## Known limitations

The system is tuned for the manipulation/VLA case: RGB cameras, proprioception, and an instruction.
A few things are out of scope for now and fall back to an escape hatch.

| Area                                   | Status                                                                                                                                                                 |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Modalities beyond image / state / text | Depth, lidar, and point clouds are not first-class; carry them through an {class}`~rlmesh.adapters.InlineCustomInput` or custom {class}`~rlmesh.adapters.AdapterBase`. |
| Tokenization                           | Stays in the model. `TextInput` delivers the instruction as a string; tokenize it inside your prediction function. There is intentionally no `TokenizerInput`.         |
| Rotation encodings                     | Fixed set: `quat_xyzw`, `quat_wxyz`, `axis_angle`, `rot6d`, `rot6d_rowmajor`, `euler_xyz`. Conventions and how to add one are in {doc}`../api/adapters`.               |
