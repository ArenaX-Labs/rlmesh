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
        components=(
            adapt.ActionComponent(adapt.ACTION_DELTA_POS, dim=3),
            adapt.ActionComponent(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
            adapt.ActionComponent(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
        ),
        clip=(-1.0, 1.0),
    ),
)
```

The observation map is keyed by observation path; dotted keys (`"agent.eef_pos"`) traverse nested
`Dict` spaces. Roles are an open vocabulary of strings matched verbatim between tags and specs;
RLMesh ships well-known conventions (`IMAGE_PRIMARY`, `EEF_POS`, `EEF_ROT`, ...), but any agreed
string works.

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
        components=(
            adapt.ActionComponent(adapt.ACTION_DELTA_POS, dim=3),
            adapt.ActionComponent(adapt.ACTION_DELTA_ROT, dim=6, encoding="rot6d"),
            adapt.ActionComponent(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
        ),
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

## Known limitations

The system is tuned for the manipulation/VLA case (RGB cameras + proprioception + an instruction). A
few things are deliberately out of scope for now and fall back to an escape hatch:

- **Modalities beyond image / state / text** (depth, lidar, point clouds) are not first-class; carry
  them through an {class}`~rlmesh.adapters.InlineCustomInput` or a custom
  {class}`~rlmesh.adapters.AdapterBase`.
- **Tokenized language** — `TextInput` passes the string through; tokenize in a custom input with
  the model's own tokenizer rather than declaring one here.
- **Rotation encodings** are a fixed set (`quat_xyzw`, `quat_wxyz`, `axis_angle`, `rot6d`); an
  exotic convention is a custom input, or a small addition to the native crate.
- **Frame stacking** is host-side state, so a model spec that sets `stack` only carries it when
  serialized from the Python object (it round-trips through `to_json`, but the native resolution
  ignores it — stacking happens in the adapter, not the core).
