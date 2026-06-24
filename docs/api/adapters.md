# Adapters

```{note}
`rlmesh.adapters` is **experimental**: it may change or disappear. Pin versions; see {doc}`/compatibility`.
```

`rlmesh.adapters` derives the preprocessing and postprocessing between an environment and a model from declarative descriptions, instead of a hand-written adapter per pair.

The split is asymmetric. An environment tags its observation and action spaces: it names the semantic role of each entry plus the few facts the spaces cannot carry (image layout, rotation encoding, an explicit value range). A model fully specifies the payload it ingests and the action it emits. {func}`~rlmesh.adapters.resolve` matches the two by role and produces an {class}`~rlmesh.adapters.Adapter`; widths, dtypes, and keys come from the gymnasium spaces. See {doc}`../user-guide/adapters` for a guided walkthrough.

Install it with the NumPy backend:

```bash
pip install "rlmesh[numpy]"
```

```{note}
Adapters are entirely opt-in: the core Gymnasium loop never imports this package. Resolution and
plan application run in the native `rlmesh-adapters` core; this package keeps the host-language half:
spec construction and serialization, entrypoint-trust gating, custom callables, and the
custom-adapter base class.
```

## Resolution

```{eval-rst}
.. autofunction:: rlmesh.adapters.resolve
```

```{eval-rst}
.. autofunction:: rlmesh.adapters.resolve_from_contract
```

```{eval-rst}
.. autofunction:: rlmesh.adapters.tag
```

## Environment Tags

An environment publishes {class}`~rlmesh.adapters.EnvTags` in its contract metadata (via {func}`~rlmesh.adapters.tag` or `EnvServer(env, tags=...)`), so a client can resolve an adapter from the handshake alone.

```{eval-rst}
.. autoclass:: rlmesh.adapters.EnvTags
   :class-doc-from: class
   :exclude-members: __init__, __new__
```

```{eval-rst}
.. autoclass:: rlmesh.adapters.ImageTag
   :class-doc-from: class
   :exclude-members: __init__, __new__
```

```{eval-rst}
.. autoclass:: rlmesh.adapters.StateTag
   :class-doc-from: class
   :exclude-members: __init__, __new__
```

```{tip}
For a flat numeric leaf whose fixed index ranges carry distinct meaning, tag it with a
{class}`~rlmesh.adapters.StateLayout` of {class}`~rlmesh.adapters.StateField` slices instead of a
`StateTag`.
```

```{eval-rst}
.. autoclass:: rlmesh.adapters.StateLayout
   :class-doc-from: class
   :exclude-members: __init__, __new__
```

```{eval-rst}
.. autoclass:: rlmesh.adapters.StateField
   :class-doc-from: class
   :exclude-members: __init__, __new__
```

```{eval-rst}
.. autoclass:: rlmesh.adapters.TextTag
   :class-doc-from: class
   :exclude-members: __init__, __new__
```

## Model Spec

```{eval-rst}
.. autoclass:: rlmesh.adapters.ModelSpec
   :class-doc-from: class
   :exclude-members: __init__, __new__
```

```{eval-rst}
.. autoclass:: rlmesh.adapters.ImageInput
   :class-doc-from: class
   :exclude-members: __init__, __new__
```

```{eval-rst}
.. autoclass:: rlmesh.adapters.StateInput
   :class-doc-from: class
   :exclude-members: __init__, __new__
```

```{eval-rst}
.. autoclass:: rlmesh.adapters.StateComponent
   :class-doc-from: class
   :exclude-members: __init__, __new__
```

```{eval-rst}
.. autoclass:: rlmesh.adapters.TextInput
   :class-doc-from: class
   :exclude-members: __init__, __new__
```

## Action Layout

The action layout is a shared vocabulary. An environment tags the action vector its `step` accepts; a model declares the action vector it emits. The resolver converts between them per component.

```{eval-rst}
.. autoclass:: rlmesh.adapters.ActionLayout
   :class-doc-from: class
   :exclude-members: __init__, __new__
```

```{eval-rst}
.. autoclass:: rlmesh.adapters.ActionComponent
   :class-doc-from: class
   :exclude-members: __init__, __new__
```

## Escape Hatches

When a pairing needs logic a declarative spec cannot express, three mechanisms compose, most local first. A custom input computes one payload key from the raw observation while the rest stays spec-driven. A custom encoding handles a rotation convention the native crate does not ship. A custom adapter subclasses {class}`~rlmesh.adapters.AdapterBase` to add stateful behavior, typically by wrapping a resolved adapter and overriding only the stateful part.

### Custom inputs

{class}`~rlmesh.adapters.InlineCustomInput` runs an in-process callable that maps the raw observation to one payload key; it is local only. {class}`~rlmesh.adapters.EntrypointCustomInput` names a `module:callable` string that is imported only when you pass `resolve(..., trust_entrypoints=True)`, so it can travel in a contract. A custom input receives the environment's own keys, not roles, and returns the entire payload key, so it does no role-matching, `dim`/`index`, or range-mapping, and is observation-side only.

```{eval-rst}
.. autoclass:: rlmesh.adapters.InlineCustomInput
   :class-doc-from: class
   :exclude-members: __init__, __new__
```

```{eval-rst}
.. autoclass:: rlmesh.adapters.EntrypointCustomInput
   :class-doc-from: class
   :exclude-members: __init__, __new__
```

### Custom encodings

Rotation encodings are a closed vocabulary (the `RotationEncoding` set listed under [Vocabulary](#vocabulary)). You cannot register one from Python, because a spec is data that travels in a contract and resolves on a remote client with no code. For a convention that is general and stable, like a published model's `rot6d_rowmajor`, add it first-party: a few lines on the native `RotationEncoding` enum plus the Python `Literal`. It then works on both the observation and action sides, serializes into the contract, and is conformance-tested once.

For a bespoke or proprietary convention, declare a {class}`~rlmesh.adapters.CustomEncoding` on the nearest native base encoding (`rot6d` or a quaternion) and supply the host-side repacking. `resolve` lowers the field to its base for the native core, so role-matching, range-mapping, and the env-to-base conversion are unchanged; the adapter applies your transforms at the boundary: `from_base` after the native conversion on the observation side, `to_base` before it on the action side. Define the encoding once and reference it from both arms:

```python
ROT6D_MINE = adapt.CustomEncoding(
    base="rot6d", from_base=rot6d_to_mine, to_base=mine_to_rot6d, name="rot6d_mine"
)
```

The packing must preserve the base width, and an observation custom encoding must be the sole component of a single-piece `StateInput` (the offset of a field interior to a multi-piece state is env-dependent). At resolve time the two arms are round-tripped on a probe to catch a mispaired encode/decode; pass `resolve(..., check_inverse=False)` to skip. The transforms are in-process callables, so the spec is local; a serializable `module:callable` form is planned.

When the constraints do not fit (a width-changing repack, a rotation interior to a multi-field state, or non-rotation feature engineering), drop to a custom `AdapterBase` or replace a whole payload key with an `InlineCustomInput`. What none of these do is attach a custom encoding to a role in the spec itself: the vocabulary stays closed so specs remain pure data that resolve on a remote client with no code. Reach for the boundary wrapper for a one-off; upstream the encoding once you want it attached to a role and reused.

### Custom adapters

Subclass {class}`~rlmesh.adapters.AdapterBase` for stateful behavior a spec cannot describe (for example temporal ensembling across action chunks, or a width-changing rotation repack interior to a multi-field state). The usual shape wraps a resolved adapter and overrides only the stateful part. Override {meth}`~rlmesh.adapters.AdapterBase.reset` to clear episode state and wire it to the model worker's `on_reset`.

A pair override replaces the adapter for one specific (model, environment) pairing entirely, for cases like control-space conversion against a robot's kinematic model. There is no special machinery: keep a registry keyed by the pair and consult it before resolving.

```{eval-rst}
.. autoclass:: rlmesh.adapters.AdapterBase
   :class-doc-from: class
   :members:
   :show-inheritance:
```

## Adapter Objects

```{eval-rst}
.. autoclass:: rlmesh.adapters.Adapter
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

## Errors

```{eval-rst}
.. autoexception:: rlmesh.adapters.AdapterResolutionError
   :show-inheritance:
```

## Vocabulary

Semantic roles are an open vocabulary of wire strings matched verbatim between independently authored tags and specs. The well-known conventions that ship with RLMesh are re-exported from the package (single-sourced from the native crate): the domain-agnostic roles `IMAGE_PRIMARY`, `IMAGE_SECONDARY`, `INSTRUCTION`, `JOINT_POS`, `JOINT_VEL`; the arm-manipulation observation roles `IMAGE_WRIST`, `EEF_POS`, `EEF_ROT`, `GRIPPER_POS`; and the action roles `ACTION_DELTA_POS`, `ACTION_DELTA_ROT`, `ACTION_GRIPPER`. Bimanual `_2` variants exist for the per-arm roles `EEF_POS`, `EEF_ROT`, `GRIPPER_POS`, `ACTION_DELTA_POS`, `ACTION_DELTA_ROT`, and `ACTION_GRIPPER`.

Rotation widths follow the declared encoding. `rlmesh.adapters.ROTATION_DIMS` maps each encoding to its dimension count:

| Encoding         | Dims | Convention                                             |
| ---------------- | ---- | ------------------------------------------------------ |
| `quat_xyzw`      | 4    | quaternion, scalar-last                                |
| `quat_wxyz`      | 4    | quaternion, scalar-first                               |
| `axis_angle`     | 3    | rotation vector                                        |
| `rot6d`          | 6    | first two columns of the rotation matrix, concatenated |
| `rot6d_rowmajor` | 6    | same two columns flattened row-major                   |
| `euler_xyz`      | 3    | roll-pitch-yaw, extrinsic XYZ                          |

`rot6d` is the standard 6D rotation; `rot6d_rowmajor` exists for checkpoints trained on the row-major interleaving. See {doc}`../user-guide/adapters` for when to add an encoding versus reach for a custom encoding.
