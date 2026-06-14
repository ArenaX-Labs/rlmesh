# IO Adapters

A model and an environment rarely speak the same format. The model wants a 224x224 image, a flat
proprioception vector with a particular rotation encoding, and the instruction under its own key;
the environment emits a differently sized camera, a quaternion, nested observation keys, and a delta
action in its own units. The usual fix is a bespoke adapter per (model, environment) pair, which is
N x M classes to write and maintain.

`rlmesh.adapters` removes that glue. An environment describes its format once and a model describes
its format once; the pairing is derived at runtime. It is experimental in this beta and fully opt-in
— nothing here is imported by the core Gymnasium loop, and it needs the NumPy backend
(`pip install --pre "rlmesh[numpy]"`).

## Tag the environment

An environment **tags** its observation and action spaces. Tags are sparse: they carry the semantic
role of each entry and the few facts the gymnasium spaces cannot — image axis layout, rotation
encoding, an explicit value range. Everything else (keys, widths, dtypes, bounds) is read from the
spaces.

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
`Dict` spaces. Roles are an open vocabulary of strings matched verbatim between tags and specs;
RLMesh ships well-known conventions (`IMAGE_PRIMARY`, `EEF_POS`, `EEF_ROT`, ...), but any agreed
string works.

### Flat (non-Dict) observations

Some environments expose a single flat numeric vector instead of one key per quantity, with fixed
index ranges carrying distinct meaning (Metaworld is the common case). A
{class}`~rlmesh.adapters.StateLayout` tags that vector: it is the observation-side mirror of
`ActionLayout`, a sequence of {class}`~rlmesh.adapters.StateField` slices laid out in order. Each
field names the role (and rotation encoding, where relevant) of its slice; offsets are implied by
order, and the field widths must sum to the leaf width. A field with no role is a skip — it advances
the offset over indices the model does not read.

```python
"proprio": adapt.StateLayout(
    adapt.StateField(adapt.EEF_POS, 3),
    adapt.StateField(adapt.EEF_ROT, 4, encoding="quat_xyzw"),
    adapt.StateField(adapt.GRIPPER_POS, 1),
    adapt.StateField(dim=10),  # object/goal indices the policy reads from pixels
),
```

When the whole observation is one leaf (no `Dict` at all), pass the `StateLayout` directly as
`observation`, mirroring `action` being one `ActionLayout`:

```python
adapt.EnvTags(observation=adapt.StateLayout(...), action=adapt.ActionLayout(...))
```

A model still matches purely by role, so the same spec resolves against a flat env and a `Dict` env
with no change. The fixed indices stay on the env side, and `describe()` shows the slice each field
reads (`proprio[0:3]`, `proprio[3:7] (quat_xyzw->rot6d)`, ...).

## Specify the model

A model **fully specifies** the payload it ingests and the action it emits, in its own conventions.

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

{func}`~rlmesh.adapters.resolve` matches the model spec against the tags and the spaces, and returns
an {class}`~rlmesh.adapters.IOAdapter`. The adapter preprocesses an observation into the model's
input format and postprocesses the model's action back into the environment's format.

```python
adapter = adapt.resolve(tags, env.observation_space, env.action_space, spec)
print(adapter.describe())          # the exact transformations chosen
payload = adapter.transform_obs(obs)      # env observation -> model input
action = adapter.transform_action(output) # model output    -> env action
```

`describe()` prints what the resolver derived — here the image is resized, `quat_xyzw -> rot6d` is
applied to the rotation, the instruction key is remapped (`goal -> task`), and on the way back the
10-dim `rot6d` action is converted `rot6d -> axis_angle`, sliced, and clipped into the env's 7-dim
action. Resolution fails with an {exc}`~rlmesh.adapters.AdapterResolutionError` if a model input or
action component has no usable counterpart.

Specs are data: nothing in a tag or spec is ever evaluated as code.

## Run a model with no glue

The ergonomic path publishes the tags on the served environment and lets the model resolve the
adapter from the contract. Serve the environment with its tags:

```python
server = rlmesh.EnvServer(env, "127.0.0.1:5555", tags=tags)
server.serve()
```

`EnvServer(tags=...)` validates the tags against the environment's spaces up front and merges them
into the contract metadata (the {func}`~rlmesh.adapters.tag` verb does the same for an environment
object you serve yourself). A model then resolves from the handshake alone — pass `spec=` to
{class}`rlmesh.numpy.Model` and run it against the environment:

```python
from rlmesh.numpy import Model, RemoteEnv

env = RemoteEnv("127.0.0.1:5555")
model = Model(predict_fn, spec=spec)   # predict_fn works in the model's own format
model.run(env, max_episodes=10)
```

`run(env)` reads the environment's contract, resolves the adapter, and wraps `predict_fn` so it only
ever sees the model's declared payload — the environment only ever sees its own action format. To
resolve explicitly instead, use {func}`~rlmesh.adapters.resolve_from_contract` and
`adapter.wrap_predict(predict_fn)`.

## Frame history

Models that condition on a short history of frames declare `stack=N` on an image input:

```python
ImageInput("image", role=IMAGE_PRIMARY, size=224, stack=4)
```

The adapter buffers the last `N` processed frames host-side and emits them on a new leading axis
(`(N, H, W, C)`), padding the start of an episode with the first frame. The environment still sends
one frame per step — nothing extra crosses the wire — and the buffer is cleared on `reset` (wired
automatically when you use `Model(spec=...).run(env)`). `ImageInput` also takes a `size=` shorthand
for square targets, and `StateInput` accepts a single `role=` instead of a one-element `components=`
tuple.

## Escape hatches

When a pairing needs logic a declarative spec cannot express, three mechanisms compose, most local
first:

- **A custom input** computes one payload key from the raw observation while the rest stays
  spec-driven: {class}`~rlmesh.adapters.InlineCustomInput` runs an in-process callable (local only),
  or {class}`~rlmesh.adapters.EntrypointCustomInput` names a `module:callable` string that is
  imported only when you pass `resolve(..., trust_entrypoints=True)`.
- **A custom adapter** subclasses {class}`~rlmesh.adapters.AdapterBase` to add stateful behavior a
  spec cannot describe (for example temporal ensembling across action chunks), typically by wrapping
  a resolved adapter and overriding only the stateful part. Override
  {meth}`~rlmesh.adapters.AdapterBase.reset` to clear episode state and wire it to the model
  worker's `on_reset`.
- **A pair override** replaces the adapter for one specific (model, environment) pairing entirely,
  for cases like control-space conversion against a robot's kinematic model. There is no special
  machinery: keep a registry keyed by the pair and consult it before resolving, e.g.

  ```python
  OVERRIDES: dict[tuple[str, str], Callable[[], adapt.AdapterBase]] = {
      ("xvla", "bridge"): XVLABridgeAdapter,
  }

  def build_adapter(model_name, env_name, ...):
      if (factory := OVERRIDES.get((model_name, env_name))) is not None:
          return factory()
      return adapt.resolve(...)
  ```

The {source}`examples/python/vla_adapters <examples/python/vla_adapters>` example shows all three
over several VLA models and environments;
{source}`examples/python/adapters_quickstart <examples/python/adapters_quickstart>` is the smallest
end-to-end serve-and-run loop.

### When you need an encoding we don't ship

Rotation encodings are a closed vocabulary (see [Known limitations](#known-limitations)): you cannot
register one from Python, because a spec is data that travels in a contract and resolves on a remote
client with no code. For a convention that is general and stable, like a published model's
`rot6d_rowmajor`, the right move is to add it **first-party** (a few lines on the native
`RotationEncoding` enum plus the Python `Literal`). It then works on both the observation and action
sides, serializes into the contract, and is conformance-tested once.

For a bespoke or proprietary rotation convention, declare a {class}`~rlmesh.adapters.CustomEncoding`
on the nearest native **base** encoding (`rot6d` or a quaternion) and supply the host-side
repacking. `resolve` lowers the field to its base for the native core, so role-matching,
range-mapping, and the env-to-base conversion are unchanged, and the adapter applies your transforms
at the boundary: `from_base` after the native conversion on the observation side, `to_base` before
it on the action side. Define the encoding once and reference it from both sides:

```python
ROT6D_MINE = adapt.CustomEncoding(
    base="rot6d", from_base=rot6d_to_mine, to_base=mine_to_rot6d, name="rot6d_mine"
)

spec = adapt.ModelSpec(
    inputs=(
        adapt.ImageInput("image", role=adapt.IMAGE_PRIMARY, size=224),
        adapt.StateInput("eef_rot", role=adapt.EEF_ROT, encoding=ROT6D_MINE),  # its own key
        adapt.TextInput("instruction"),
    ),
    action=adapt.ActionLayout(
        adapt.ActionComponent(adapt.ACTION_DELTA_ROT, dim=6, encoding=ROT6D_MINE),
        ...,
    ),
)
```

The packing must preserve the base width, and an observation custom encoding must be the sole
component of a single-piece `StateInput` (the offset of a field interior to a multi-piece state is
env-dependent). At resolve time the two arms are round-tripped on a probe to catch a mispaired
encode/decode; pass `resolve(..., check_inverse=False)` to skip. The transforms are in-process
callables, so the spec is local, which the dominant resolve-from-contract flow already is; a
serializable `module:callable` form is planned. `examples/python/vla_adapters/models/geovla.py` is a
runnable end-to-end example.

When the constraints do not fit (a width-changing repack, a rotation interior to a multi-field
state, or non-rotation feature engineering), drop to a custom {class}`~rlmesh.adapters.AdapterBase`
that wraps a resolved adapter and repacks the field on the boundary (the `act.py` pattern), or
replace a whole payload key from the raw observation with an
{class}`~rlmesh.adapters.InlineCustomInput`. The custom input receives the env's own keys, not
roles, and returns the entire payload key, so it does no role-matching, `dim`/`index`, or
range-mapping, and is observation-side only:

```python
def encode_state(raw_obs):
    pos, quat = raw_obs["eef_pos"], raw_obs["eef_quat"]
    return np.concatenate([pos, my_rotation_encoding(quat)])

adapt.InlineCustomInput("state", transform=encode_state)  # one input among the spec's
```

What none of these do is attach a custom encoding _to a role_ in the spec itself: the vocabulary
stays closed so specs remain pure data that resolve on a remote client with no code. The boundary
wrapper is the local answer and keeps the machinery; first-party is the shared one, matched by role
and carried in the contract. Reach for the wrapper for a one-off, and upstream the encoding once you
want it attached to a role and reused.

## Known limitations

The system is tuned for the manipulation/VLA case (RGB cameras + proprioception + an instruction). A
few things are deliberately out of scope for now and fall back to an escape hatch:

- **Modalities beyond image / state / text** (depth, lidar, point clouds) are not first-class; carry
  them through an {class}`~rlmesh.adapters.InlineCustomInput` or a custom
  {class}`~rlmesh.adapters.AdapterBase`.
- **Tokenization is the model's job, not the adapter's.** `TextInput` delivers the instruction as a
  string; tokenize it inside your prediction function with the tokenizer you loaded alongside the
  checkpoint. Declaring a tokenizer here would couple the IO layer to model internals (the very
  thing adapters avoid), so there is intentionally no `TokenizerInput`.
- **Rotation encodings** are a fixed set (`quat_xyzw`, `quat_wxyz`, `axis_angle`, `rot6d`,
  `rot6d_rowmajor`, `euler_xyz`). `rot6d` is the standard 6D rotation: the matrix's first two
  columns concatenated. `rot6d_rowmajor` is the same two columns flattened row-major (an
  interleaving some checkpoints, e.g. X-VLA proprio, were trained on), named distinctly so `rot6d`
  stays the standard convention. `euler_xyz` is roll-pitch-yaw, extrinsic XYZ
  (`R = Rz(yaw) Ry(pitch) Rx(roll)`, the ROS/scipy-`'xyz'` convention); a _different_ Euler
  convention is a custom input, or a small addition to the native crate. See
  [When you need an encoding we don't ship](#when-you-need-an-encoding-we-dont-ship).
- **Frame stacking** is host-side state, so a model spec that sets `stack` only carries it when
  serialized from the Python object (it round-trips through `to_json`, but the native resolution
  ignores it — stacking happens in the adapter, not the core).
