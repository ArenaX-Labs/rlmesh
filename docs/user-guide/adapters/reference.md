# Adapter Reference

The complete field-by-field reference for the declarative adapter specs. Use it to match your environment or model shape to a feature, then look up the exact behavior of every field.

For the concepts (how the two sides connect and why), start with {doc}`/user-guide/adapters`. For logic a spec cannot express, see {doc}`/user-guide/adapters/escape-hatches`. For exact signatures and the autodoc, see {doc}`/api/adapters`. Examples live at {source}`examples/python/adapters`.

Every snippet uses `import rlmesh.adapters as adapt`.

## Role registry

A role is the string that matches an environment feature to a model input. Roles are an **open vocabulary**: any string works as long as the env tag and the model spec agree on it verbatim. The constants below are the well-known conventions RLMesh ships; reach for them so independently authored envs and models line up, and invent your own string for anything they do not cover.

Role strings carry a feature-kind prefix (`image/`, `proprio/`, `text/`, `action/`), not a domain prefix -- two domains sharing `proprio/joint_pos` is intentional.

| Constant           | Wire string            | Domain       | Kind       | Typical width / encoding            |
| ------------------ | ---------------------- | ------------ | ---------- | ----------------------------------- |
| `IMAGE_PRIMARY`    | `image/primary`        | core         | `image/`   | H×W×C frame (main/exterior camera)  |
| `IMAGE_SECONDARY`  | `image/secondary`      | core         | `image/`   | H×W×C frame (second fixed camera)   |
| `IMAGE_WRIST`      | `image/wrist`          | core         | `image/`   | H×W×C frame (wrist/hand camera)     |
| `INSTRUCTION`      | `text/instruction`     | core         | `text/`    | string (task instruction)           |
| `JOINT_POS`        | `proprio/joint_pos`    | core         | `proprio/` | N joints (embodiment-dependent)     |
| `JOINT_VEL`        | `proprio/joint_vel`    | core         | `proprio/` | N joints                            |
| `EEF_POS`          | `proprio/eef_pos`      | manipulation | `proprio/` | 3 (Cartesian xyz)                   |
| `EEF_ROT`          | `proprio/eef_rot`      | manipulation | `proprio/` | width follows the rotation encoding |
| `GRIPPER_POS`      | `proprio/gripper`      | manipulation | `proprio/` | 1+ (embodiment-dependent)           |
| `ACTION_DELTA_POS` | `action/delta_eef_pos` | manipulation | `action/`  | 3 (Cartesian delta)                 |
| `ACTION_DELTA_ROT` | `action/delta_eef_rot` | manipulation | `action/`  | width follows the rotation encoding |
| `ACTION_GRIPPER`   | `action/gripper`       | manipulation | `action/`  | 1                                   |

Roles do not imply widths mechanically -- specs pin widths explicitly where they matter (`dim`/`index` on a part, `dim` on an actuator). Rotation widths follow the declared encoding (see [Vocabularies](#vocabularies)).

### Bimanual roles

Every manipulation role has a `_2` variant for the second arm: `EEF_POS_2`, `EEF_ROT_2`, `GRIPPER_POS_2`, `ACTION_DELTA_POS_2`, `ACTION_DELTA_ROT_2`, `ACTION_GRIPPER_2`. The first (or only) arm uses the unsuffixed role; the second arm uses `_2`. A single-arm environment never declares `_2`, so a model part targeting it zero-fills on the observation side and drops the extra dims on the action side.

## Vocabularies

Rotation encodings are a closed set (a remote client must resolve a spec with no code). Each has a fixed native width:

| Encoding         | Width | Notes                             |
| ---------------- | ----- | --------------------------------- |
| `quat_xyzw`      | 4     | quaternion, scalar-last           |
| `quat_wxyz`      | 4     | quaternion, scalar-first          |
| `axis_angle`     | 3     | rotation vector                   |
| `rot6d`          | 6     | 6-D continuous (Zhou et al.)      |
| `rot6d_rowmajor` | 6     | 6-D continuous, row-major packing |
| `euler_xyz`      | 3     | Euler angles, XYZ                 |

Other vocabularies:

| Vocabulary   | Values                   | Default                             | Notes                               |
| ------------ | ------------------------ | ----------------------------------- | ----------------------------------- |
| Image layout | `hwc`, `chw`             | `hwc`                               | axis order of the stored image      |
| Fit mode     | `stretch`, `crop`, `pad` | (none)                              | how to reconcile an aspect mismatch |
| dtype        | any NumPy dtype name     | `uint8` (image) / `float32` (state) | string, e.g. `"float32"`            |

Normalization is one overloaded field, `normalize`: `False` (off, the default), `True` (the conventional `[0, 1]`), or a `(low, high)` pair (e.g. `(-1.0, 1.0)`) to map into a specific range. One field, so an on/off flag can never disagree with a range, and `False` is an authoritative off-switch.

## The environment side

An environment **tags** its observation and action spaces. Tags are sparse: they carry each entry's role plus the few facts the gymnasium spaces cannot express (image layout, rotation encoding, an explicit range). Keys, widths, dtypes, and bounds are read from the spaces by the native `join` step at resolve time.

```python
import rlmesh.adapters as adapt

tags = adapt.EnvTags(
    observation={
        "pixels": adapt.ImageTag(adapt.IMAGE_PRIMARY),
        "eef_pos": adapt.StateTag(adapt.EEF_POS),
        "eef_quat": adapt.StateTag(adapt.EEF_ROT, encoding="quat_xyzw"),
        "gripper": adapt.StateTag(adapt.GRIPPER_POS),
        "task": adapt.TextTag(adapt.INSTRUCTION),
    },
    action=adapt.Action(
        adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
        adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1),
    ),
)
```

{class}`~rlmesh.adapters.EnvTags` takes `observation` and `action`. The observation is a recursive tree whose container type **is** the runtime container type:

| Authored container | Maps a space        | Example                                           |
| ------------------ | ------------------- | ------------------------------------------------- |
| Python `dict`      | `Dict`              | `{"pixels": ImageTag(...), "eef": StateTag(...)}` |
| Python `tuple`     | `Tuple`             | `(ImageTag(...), StateTag(...))`                  |
| bare leaf          | a single space leaf | `Split(...)` or one `StateTag(...)`               |

Nesting is real `dict` nesting that mirrors a nested `Dict` space (`{"agent": {"eef_pos": StateTag(...)}}`) -- there are no dotted keys. A single-leaf observation is the bare leaf with no dict wrapper.

### ImageTag

{class}`~rlmesh.adapters.ImageTag` -- one camera image leaf.

| Field                   | Default | What it declares                             | When to use           |
| ----------------------- | ------- | -------------------------------------------- | --------------------- |
| `role` (1st positional) | --      | the image role to match                      | always                |
| `layout`                | `"hwc"` | axis order of the stored frame               | the env stores `chw`  |
| `upside_down`           | `False` | the camera renders 180° rotated from upright | a known flipped mount |

### StateTag

{class}`~rlmesh.adapters.StateTag` -- one numeric proprioception leaf.

| Field                   | Default | What it declares                                                  | When to use                          |
| ----------------------- | ------- | ----------------------------------------------------------------- | ------------------------------------ |
| `role` (1st positional) | --      | the state role to match                                           | always                               |
| `encoding`              | `None`  | rotation encoding (single, or a native-first preference sequence) | the role is a rotation               |
| `range`                 | `None`  | `(low, high)` bounds where the space is unbounded                 | the space leaves this leaf unbounded |

`range` only supplies bounds the space lacks. If the space declares finite bounds that disagree with it, resolution errors rather than silently overriding them.

### TextTag

{class}`~rlmesh.adapters.TextTag` -- a text leaf (typically the instruction). Single field: `role` (1st positional). Use it when the observation carries a string the model conditions on.

### Split + Field

Some environments expose one flat numeric `Box` with fixed index ranges instead of a key per quantity (Metaworld is the common case). {class}`~rlmesh.adapters.Split` tags that single vector -- it is a **leaf**, not a container, and the observation-side mirror of {class}`~rlmesh.adapters.Action`.

```python
adapt.EnvTags(
    observation=adapt.Split(
        adapt.Field(adapt.EEF_POS, dim=3),
        adapt.Field(adapt.EEF_ROT, dim=4, encoding="quat_xyzw"),
        adapt.Field(adapt.GRIPPER_POS, dim=1),
        adapt.Field(dim=10),  # skip the object/goal indices the policy reads from pixels
    ),
    action=adapt.Action(...),
)
```

`Split(*Field)` takes its fields positionally and needs at least one. Field widths must sum to the leaf width (checked at join). A {class}`~rlmesh.adapters.Field`:

| Field                   | Default            | What it declares                                  | When to use                         |
| ----------------------- | ------------------ | ------------------------------------------------- | ----------------------------------- |
| `role` (1st positional) | `None`             | the role for this slice; `None` is a **skip**     | name it, or skip with `None`        |
| `dim`                   | -- (required, ≥ 1) | element count of the slice                        | always                              |
| `encoding`              | `None`             | rotation encoding (single or preference sequence) | the slice is a rotation             |
| `range`                 | `None`             | `(low, high)` where the space is unbounded        | the slice is unbounded in the space |

A `role=None` field advances the offset without producing a feature -- use it to step over indices the model never reads. A skip carries no encoding or range.

## The model side

A model **fully specifies** the payload it ingests and the action it emits, in its own conventions. {class}`~rlmesh.adapters.ModelSpec` takes `input` and `output`. The `input` tree's container type **is** the payload container `predict` receives (a `dict`, a `tuple`, or a bare single leaf). A leaf carries no key (its position in the tree is the payload position), and a role may be reused across leaves.

```python
spec = adapt.ModelSpec(
    input={
        "image": adapt.Image(adapt.IMAGE_PRIMARY, size=256, normalize=True),
        "state": adapt.Concat(
            adapt.EEF_POS,
            adapt.State(adapt.EEF_ROT, encoding="rot6d"),
            adapt.GRIPPER_POS,
        ),
        "instruction": adapt.Text(adapt.INSTRUCTION),
    },
    output=adapt.Action(
        adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
        adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=6, encoding="rot6d"),
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, binary=True),
    ),
)
```

### Image

{class}`~rlmesh.adapters.Image` -- a camera input. Every field:

| Field                   | Default      | What it does                                                                  | When to use                                                    |
| ----------------------- | ------------ | ----------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `role` (1st positional) | --           | match an env image                                                            | always                                                         |
| `size`                  | `None`       | sugar that sets `height` **and** `width`                                      | square targets (pass `size` _or_ `height`/`width`, not both)   |
| `height`                | `None`       | target height (keep env height if `None`)                                     | non-square target                                              |
| `width`                 | `None`       | target width                                                                  | non-square target                                              |
| `layout`                | `"hwc"`      | axis order the model wants                                                    | the model wants `chw`                                          |
| `channels`              | `None`       | channel count the model wants (3 RGB, 1 gray)                                 | make a channel mismatch an error instead of silent             |
| `dtype`                 | `"uint8"`    | NumPy dtype of the result                                                     | the model wants floats                                         |
| `normalize`             | `False`      | map 8-bit pixels: `True` → `[0,1]`, or a `(low, high)` pair → that range      | scale `[0,255]`; a pair for signed inputs, e.g. `(-1.0, 1.0)`  |
| `lead_dims`             | `0`          | leading singleton axes to add                                                 | the model wants a batch/time axis                              |
| `upside_down`           | `False`      | the model was trained on 180°-rotated frames                                  | training-time flip                                             |
| `resample`              | `"bilinear"` | resize filter: `bilinear` (OpenCV/torch) or `bilinear_aa` (PIL)               | match the training pipeline                                    |
| `allow_upscale`         | `False`      | permit a target larger than the env resolution                                | the model needs more pixels than the camera has                |
| `fit`                   | `None`       | reconcile an aspect mismatch: `stretch`/`crop`/`pad` or a preference sequence | target aspect differs from the env                             |
| `optional`              | `False`      | zero-fill a black frame when the env lacks this camera                        | the camera may be absent (needs `height`, `width`, `channels`) |
| `absent_fill`           | `None`       | fill value for the blank frame                                                | non-black fill                                                 |
| `stack`                 | `1`          | buffer N frames on a new leading axis                                         | frame history (see [Frame history](#frame-history-stack))      |

`size` is the idiomatic square form. `fit` accepts a preference sequence (`("crop", "pad")`); the resolver picks, per env, the first that does not need a disallowed upscale, so one spec can crop a large camera and letterbox a small one.

### State

{class}`~rlmesh.adapters.State` -- the single-part numeric input. Every field:

| Field                   | Default     | What it does                                                        | When to use                      |
| ----------------------- | ----------- | ------------------------------------------------------------------- | -------------------------------- |
| `role` (1st positional) | --          | match an env state feature                                          | always                           |
| `encoding`              | `None`      | rotation encoding: single, preference sequence, or `CustomEncoding` | the part is a rotation           |
| `dim`                   | `None`      | keep the leading N elements                                         | truncate the source              |
| `index`                 | `None`      | select one element after conversion                                 | pick a single scalar             |
| `optional`              | `False`     | zero-fill when the env lacks the role                               | the role may be absent           |
| `range`                 | `None`      | `(low, high)` the model wants; affinely maps from the env range     | model and env disagree on scale  |
| `pad_to`                | `None`      | zero-pad the result to this length                                  | fixed-width input                |
| `dtype`                 | `"float32"` | NumPy dtype of the result                                           | non-default dtype                |
| `reshape`               | `None`      | target shape for the result                                         | the model wants a specific shape |
| `container`             | `"array"`   | emit a NumPy array or a plain `list`                                | the model wants a list           |

`dim` and `index` are mutually exclusive (`dim` keeps the leading N, `index` selects one). When `optional` is set the fill width must be known without an env feature, so set one of `index`, `dim`, or `encoding`. `range` is a no-op when the env has no source range to map from -- it does not clamp on its own.

A `State` is also a valid `Concat` part: its part fields (`role`, `encoding`, `dim`, `index`, `optional`, `range`) are taken, and its container fields (`pad_to`, `dtype`, `reshape`, `container`) must stay default when used as a part.

### Concat

{class}`~rlmesh.adapters.Concat` -- the **multi-part** state leaf: several roles packed into one tensor. `Concat(*parts, pad_to=None, dtype="float32", reshape=None, container="array")` needs at least one part. A part is a bare role string (sugar for a role-only `State`) or a `State` carrying part fields:

```python
adapt.Concat(
    adapt.EEF_POS,                              # bare role: no options needed
    adapt.State(adapt.EEF_ROT, encoding="rot6d"),  # State part: needs an encoding
    adapt.GRIPPER_POS,
)
```

Parts are concatenated in order. The container-level fields (`pad_to`, `dtype`, `reshape`, `container`) apply to the concatenated result and behave as in `State`. A single-role state is `State` directly; `Concat` is the >1-part case (both serialize to the same wire form).

### Text

{class}`~rlmesh.adapters.Text` -- a text input.

| Field                   | Default | What it does                                                 | When to use                   |
| ----------------------- | ------- | ------------------------------------------------------------ | ----------------------------- |
| `role` (1st positional) | --      | match an env text feature                                    | always                        |
| `container`             | `"str"` | emit a plain string or a single-element list                 | the model wants a list        |
| `default`               | `None`  | value when the obs omits the feature; `None` omits the input | supply a fallback instruction |

Tokenization stays in the model -- `Text` delivers the raw string.

### Custom

{class}`~rlmesh.adapters.Custom` -- a payload slot computed by host-language code. Set **exactly one** of `transform=` (an in-process callable, local only) or `entrypoint=` (a `"module:callable"` string, imported only under `resolve(..., trust_entrypoints=True)`). The rest of the spec stays declarative. See {doc}`/user-guide/adapters/escape-hatches` for the full pattern and the trust model.

## The action side

{class}`~rlmesh.adapters.Action` is shared by env tags and model specs. `Action(*Actuator, clip=None)` takes its actuators positionally and exposes a `.dim` property (the sum of component dims). `clip` is an optional `(low, high)` applied to the final vector.

{class}`~rlmesh.adapters.Actuator` -- one contiguous slice of the action vector:

| Field                   | Default       | What it does                                              | When to use                            |
| ----------------------- | ------------- | --------------------------------------------------------- | -------------------------------------- |
| `role` (1st positional) | `None`        | match the actuator across sides; `None` = opaque (below)  | usually                                |
| `dim`                   | -- (required) | dimensions this component occupies                        | always                                 |
| `encoding`              | `None`        | rotation encoding (or a `CustomEncoding`)                 | the component is a rotation            |
| `range`                 | `None`        | `(low, high)` of the component values                     | declare/convert the value range        |
| `binary`                | `False`       | the component is a binary decision (snap after range map) | a gripper open/close                   |
| `scale`                 | `None`        | multiply the model value                                  | env actuator is scaled                 |
| `invert`                | `False`       | negate the model value (explicit `scale=-1`)              | gripper sign correction                |
| `threshold`             | `None`        | subtract to recenter the decision boundary                | shift a `binary` split off zero        |
| `clip`                  | `False`       | clamp the mapped value to `range` (requires `range`)      | per-dim safety on a mixed-range action |
| `fill`                  | `0.0`         | constant per dim of an opaque (role-less) actuator        | env-required dims no model reads       |

`scale`, `invert`, and `threshold` declare a side's actuator convention. They can be set on **either side** and compose as literal transforms applied **after** the declared formats (rotation, range) are bridged -- **model-side first** (the model's own output convention), then **env-side** (the env's):

```
rotation/range bridged  →  model(scale → invert → threshold)  →  env(scale → invert → threshold)  →  binary  →  clip
```

So an env declares its quirk once and every model inherits it; _and_ a model whose own output differs from a **shared** env it cannot edit declares the bridge on its own actuator -- e.g. a sign-flipped gripper as `Actuator(ACTION_GRIPPER, dim=1, invert=True)`, or a sigmoid-probability gripper as `binary=True, threshold=0.5` -- instead of hardcoding the env's convention in `predict()`. `binary` snaps to a definite side after range mapping: `>= 0` opens (`+1`), below closes (`-1`); a value exactly on the boundary opens rather than emitting an undefined `0`.

`clip` is the exception -- it stays **env-side only**: it clamps to the env actuator's `range` (a final safety bound, not a convention), so declaring it on a model actuator is a resolve error.

A **role-less actuator** -- `Actuator(dim=N, fill=...)` with no `role` -- is _opaque_: it occupies `N` dims of the env action with the constant `fill`, matched by no model output (the action-side mirror of a role-less `Field`). Use it for dims the env requires but no model produces, such as a control-mode selector or base padding. A registered `role` with a fixed canonical dim (e.g. `eef_pos` is 3-D) also validates the declared `dim` -- a mismatch is a resolve error.

## Conversion semantics and policy

Each conversion the resolver can perform falls into one of four policies. **Silent** is always applied when declared; **opt-in** is off until you set the flag; **advisory-warn** succeeds but logs data loss; **resolve-error** fails resolution.

| Conversion                             | Policy        | Trigger                                                                             |
| -------------------------------------- | ------------- | ----------------------------------------------------------------------------------- |
| Image resize (target ≤ env resolution) | SILENT        | a smaller `size`/`height`/`width`                                                   |
| Layout transpose (`hwc` ↔ `chw`)       | SILENT        | model `layout` differs from the env's                                               |
| Normalize                              | SILENT        | `normalize` set (`True` or a `(low, high)` range)                                   |
| dtype cast                             | SILENT        | model `dtype` differs from the env's                                                |
| Rotation encoding conversion           | SILENT        | model encoding differs (both known)                                                 |
| Range map (affine)                     | SILENT        | model `range` set and env range known                                               |
| `binary` + `threshold` snap            | SILENT        | declared on the actuator                                                            |
| `fit` (aspect-changing resize)         | OPT-IN        | aspect mismatch; **absent `fit` → resolve error**                                   |
| `allow_upscale`                        | OPT-IN        | target > env resolution; **absent → resolve error**                                 |
| `channels` declared                    | OPT-IN        | declaring it turns a channel-count mismatch into a resolve error (silent otherwise) |
| `optional` / `absent_fill`             | OPT-IN        | env lacks the camera/role; **absent → resolve error**                               |
| Crop                                   | ADVISORY-WARN | `fit="crop"` chosen (pixels discarded)                                              |
| Pad                                    | ADVISORY-WARN | `fit="pad"` chosen (border added)                                                   |
| Zero-filled camera / state             | ADVISORY-WARN | an `optional` part filled because the env lacks the role                            |

### Two axes: parsing and resolve

Spec handling has two independent stages. **Parsing** is split: publishing a spec is strict and rejects unknown fields, while reading one back is tolerant and round-trips it, so a newer peer's spec does not break an older reader. **Resolve** is where a spec meets a concrete pair of spaces and the policy table above applies.

The **bare-field taint rule**: an unknown field on a known kind is a resolve error unless its name is prefixed `x-` or `ext-`. Prefixed extension fields are carried through untouched; an un-prefixed unknown field is treated as a typo and rejected.

**Join-time validation** is the final gate: when a tag and its gymnasium space disagree on class, width, encoding, or range, resolution errors rather than guessing.

## Frame history (stack)

A model that conditions on a short history sets `stack=N` on an `Image`. The adapter keeps an **episode-keyed rolling buffer** of the last N processed frames and emits them on a new leading axis, padding the start of an episode with the first frame and clearing on `reset`.

```python
adapt.Image(adapt.IMAGE_PRIMARY, size=256, stack=4)
```

Stacking is host-side on the local path and native in the core on the served path. Either way the environment still sends **one frame per step** -- nothing extra crosses the wire.

```{caution}
Frame stacking is episode state held outside the model: host-side on the local path, in the core
on the served path (an episode-keyed buffer per vector lane). The spec's `stack` round-trips through
`to_json`, the buffer clears on `reset`, and the env still sends one frame per step, so no frames
leak across episodes or lanes and nothing extra crosses the wire.
```

## Match your shape

Find the row that matches your environment, then tag it:

| My environment looks like...              | Tag it...                                       |
| ----------------------------------------- | ----------------------------------------------- |
| `Dict` of cameras + proprio + instruction | a `dict` of `ImageTag` / `StateTag` / `TextTag` |
| one flat `Box` with fixed index ranges    | a bare `Split(Field(...), ...)`                 |
| a `Tuple` of sub-spaces                   | a Python `tuple` of leaves                      |
| an upside-down camera                     | `ImageTag(role, upside_down=True)`              |
| quaternion proprioception                 | `StateTag(EEF_ROT, encoding="quat_xyzw")`       |
| two arms                                  | the role plus its `_2` variant per arm          |

Find the row that matches your model, then spec it:

| My model wants...                    | Spec it...                                                       |
| ------------------------------------ | ---------------------------------------------------------------- |
| a resized, normalized image          | `Image(IMAGE_PRIMARY, size=256, normalize=True)`                 |
| channels-first                       | `Image(IMAGE_PRIMARY, size=256, layout="chw")`                   |
| stacked frames                       | `Image(IMAGE_PRIMARY, size=256, stack=4)`                        |
| concatenated proprio with a rotation | `Concat(EEF_POS, State(EEF_ROT, encoding="rot6d"), GRIPPER_POS)` |
| a binary gripper command             | `Actuator(ACTION_GRIPPER, dim=1, binary=True)`                   |
| an optional second camera            | `Image(IMAGE_WRIST, size=256, channels=3, optional=True)`        |
| an instruction string                | `Text(INSTRUCTION)`                                              |

### Common pitfalls

| Symptom                           | Cause                                | Fix                                                |
| --------------------------------- | ------------------------------------ | -------------------------------------------------- |
| Wrong channel count slips through | RGB vs grayscale not declared        | set `channels` to make a mismatch an error         |
| Image axes scrambled              | HWC vs CHW mismatch                  | set `layout` to what the model wants               |
| Rotation looks rotated wrong      | `quat_xyzw` vs `quat_wxyz` confusion | match the env's exact encoding                     |
| Values out of range               | scale mismatch                       | set `range` on the model side to map it            |
| Resolve fails on a missing camera | env lacks the role                   | `optional=True` (with `height`/`width`/`channels`) |
| Resolve fails on upscale          | target larger than the camera        | `allow_upscale=True`, or lower the target          |

## Errors and `describe()`

Resolution raises {exc}`~rlmesh.adapters.AdapterResolutionError` when a spec cannot be bridged to the spaces: a required role with no `optional`/zero-fill, a declared channel mismatch, an upscale without `allow_upscale`, an aspect mismatch without `fit`, an unsupported `resample`/`dtype`, an impossible encoding conversion, a bare unknown field on a known kind, or a join-time class/width/encoding/range disagreement between a tag and its space. The message names the offending leaf and what it expected.

Once resolution succeeds, call `adapter.describe()` to print the exact transforms the resolver chose (each resize, layout transpose, encoding conversion, range map, key remap, slice, and clip) before you run a single step. It is the fastest way to confirm the bridge is what you intended.

```python
adapter = adapt.resolve(tags, env.observation_space, env.action_space, spec)
print(adapter.describe())
```
