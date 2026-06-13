# Adapters

`rlmesh.adapters` derives the preprocessing and postprocessing between an environment and a model
from declarative descriptions, instead of a hand-written adapter per pair. It is experimental in
this beta and entirely opt-in: the core Gymnasium loop never imports it.

The split is asymmetric. An environment **tags** its observation and action spaces — it names the
semantic role of each entry plus the few facts the spaces cannot carry (image layout, rotation
encoding, an explicit value range). A model **fully specifies** the payload it ingests and the
action it emits. {func}`~rlmesh.adapters.resolve` matches the two by role and produces an
{class}`~rlmesh.adapters.IOAdapter`; widths, dtypes, and keys come from the gymnasium spaces.

Install it with the NumPy backend:

```bash
pip install --pre "rlmesh[numpy]"
```

Resolution and plan application run in the native `rlmesh-adapters` core — the same implementation
behind every language binding, pinned by conformance vectors. This package keeps the host-language
half: spec construction and serialization, entrypoint-trust gating, custom callables, and the
custom-adapter base class. See {doc}`../user-guide/adapters` for a guided walkthrough.

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

An environment publishes {class}`~rlmesh.adapters.EnvTags` in its contract metadata (via
{func}`~rlmesh.adapters.tag` or `EnvServer(env, tags=...)`), so a client can resolve an adapter from
the handshake alone.

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

```{eval-rst}
.. autoclass:: rlmesh.adapters.CustomInput
   :class-doc-from: class
   :exclude-members: __init__, __new__
```

## Action Layout

The action layout is shared vocabulary: an environment tags the action vector its `step` accepts,
and a model declares the action vector it emits. The resolver converts between them per component.

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

## Adapters

```{eval-rst}
.. autoclass:: rlmesh.adapters.IOAdapter
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.adapters.AdapterBase
   :class-doc-from: class
   :members:
   :show-inheritance:
```

## Errors

```{eval-rst}
.. autoexception:: rlmesh.adapters.AdapterResolutionError
   :show-inheritance:
```

## Vocabulary

Semantic roles are an open vocabulary of wire strings matched verbatim between independently
authored tags and specs. The well-known conventions that ship with RLMesh are re-exported from the
package (single-sourced from the native crate), including the domain-agnostic roles `IMAGE_PRIMARY`,
`IMAGE_SECONDARY`, `INSTRUCTION`, `JOINT_POS`, `JOINT_VEL` and the arm-manipulation roles
`IMAGE_WRIST`, `EEF_POS`, `EEF_ROT`, `GRIPPER_POS` (with bimanual `_2` variants) and their
`ACTION_*` counterparts. Rotation widths follow the declared encoding;
`rlmesh.adapters.ROTATION_DIMS` maps each encoding (`quat_xyzw`, `quat_wxyz`, `axis_angle`, `rot6d`)
to its dimension count.
