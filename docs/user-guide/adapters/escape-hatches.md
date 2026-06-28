# Escape Hatches

A declarative spec resolves at runtime with no code (see {doc}`/user-guide/adapters`). That covers the manipulation/VLA case (cameras, proprioception, an instruction, a per-step action), but some pairings need host-language logic the spec language cannot carry. Four escape hatches add exactly that, and no more.

They are ordered most-local first: each later hatch takes over more of the pipeline, so reach for the earliest one that fits. The first three keep the rest of the IO spec-driven; only the last replaces the adapter outright.

| Hatch                                          | Scope                         | Reach for it when                                                          |
| ---------------------------------------------- | ----------------------------- | -------------------------------------------------------------------------- |
| {class}`~rlmesh.adapters.Custom` input         | one payload slot              | one input is computed from the raw observation; the rest stays declarative |
| {class}`~rlmesh.adapters.CustomEncoding`       | one rotation field            | a model uses a rotation packing RLMesh does not ship as a built-in         |
| {class}`~rlmesh.adapters.AdapterBase` subclass | the whole adapter, model-wide | the model needs state across steps (e.g. temporal ensembling)              |
| Pair override                                  | one `(model, env)` pairing    | a single pairing is special enough to hand-write end to end                |

```{warning}
Specs are pure data; nothing in a tag or spec is evaluated as code. The two exceptions are
the entrypoint forms below: a {class}`~rlmesh.adapters.Custom` built with `entrypoint=` and a
{class}`~rlmesh.adapters.CustomEncoding` whose arms are `"module:callable"` strings. Both import
named code, and only when you pass `resolve(..., trust_entrypoints=True)`.
```

## Custom input -- one payload slot

A {class}`~rlmesh.adapters.Custom` leaf fills one slot of the model payload from the raw observation with host code. Everything else in the `input` tree stays spec-driven. Use it for a modality RLMesh does not model first-class (depth, lidar, point clouds) or any value that needs bespoke assembly.

The leaf takes a callable that receives the raw observation mapping and returns the slot's value:

```python
import rlmesh.adapters as adapt

def encode_depth(obs):
    return obs["depth"].astype("float32") / 1000.0

spec = adapt.ModelSpec(
    input={
        "image": adapt.Image(adapt.IMAGE_PRIMARY, size=224),
        "proprio": adapt.Concat(adapt.EEF_POS, adapt.GRIPPER_POS),
        "depth": adapt.Custom(transform=encode_depth),
    },
    output=adapt.Action(
        adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1),
    ),
)
```

`transform=` is local only -- a callable cannot be serialized, so a spec carrying one cannot be published in contract metadata and resolves only in the process that defined it. To ship the spec to a remote client, give an `entrypoint` instead, a `"module:callable"` string that travels on the wire:

```python
adapt.Custom(entrypoint="my_pkg.adapters:encode_depth")
```

The entrypoint is imported only under {func}`~rlmesh.adapters.resolve` with `trust_entrypoints=True`; otherwise resolution refuses to import it. Set exactly one of `transform=` or `entrypoint=`.

```{note}
A `Custom` input only touches the observation side -- it computes a payload slot. It cannot
post-process actions. For action-side logic, use a {class}`~rlmesh.adapters.CustomEncoding`
(one rotation field) or an {class}`~rlmesh.adapters.AdapterBase` subclass (the whole action).
```

## Custom encoding -- a host-side rotation packing

Rotation encodings are a closed vocabulary so a spec can resolve on a remote client with no code. When a model expects a packing outside that set, a {class}`~rlmesh.adapters.CustomEncoding` layers a host-side repack on top of the nearest built-in **base** encoding, keeping the rest of the rotation pipeline declarative.

You supply the two arms of an encode/decode pair and drop the result in as an `encoding=`:

```python
import rlmesh.adapters as adapt

rot6d_packed = adapt.CustomEncoding(
    "rot6d",
    from_base=unpack_rot6d,  # base -> custom, applied on the obs side
    to_base=pack_rot6d,      # custom -> base, applied on the action side
    name="rot6d_packed",
)

spec = adapt.ModelSpec(
    input={"proprio": adapt.Concat(adapt.EEF_POS, adapt.State(adapt.EEF_ROT, encoding=rot6d_packed))},
    output=adapt.Action(adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=6, encoding=rot6d_packed)),
)
```

The field resolves **natively as its base**: role matching, range mapping, and the env-to-base rotation conversion are unchanged. The adapter then repacks at the field boundary: `from_base` (base -> custom) on the observation side and `to_base` (custom -> base) on the action side, so the model sees its packing and the env sees the base.

| Rule              | Detail                                                                                                                             |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Width-preserving  | the packing must keep the base width (`ROTATION_DIMS[base]`); it repacks, it does not resize                                       |
| Arm per side      | `from_base` is needed only when the encoding tags an observation state; `to_base` only when it tags an action; supply at least one |
| Arms agree        | both in-process callables, or both `"module:callable"` entrypoint strings -- not a mix                                             |
| Entrypoints gated | string arms are serializable but import only under `resolve(..., trust_entrypoints=True)`                                          |

Reach for a `CustomEncoding` for a one-off or proprietary packing. When the convention is general, stable, and published (a checkpoint's documented rotation format), upstream it to the first-party `RotationEncoding` enum instead: it then serializes into the contract, is matched by role with no host code, and is conformance-tested.

## AdapterBase subclass -- stateful behavior

A declarative adapter is stateless step to step. When a model needs memory across steps (ACT-style temporal ensembling is the canonical case), subclass {class}`~rlmesh.adapters.AdapterBase` and wrap a resolved adapter, adding only the stateful part. Observation handling and per-step action conversion (encodings, ranges, clipping) stay spec-driven; the custom code is the state alone.

```python
import rlmesh.adapters as adapt
from collections import deque

class ChunkEnsembleAdapter(adapt.AdapterBase):
    def __init__(self, inner: adapt.Adapter, horizon: int = 8):
        self._inner = inner
        self._chunks: deque = deque(maxlen=horizon)

    def transform_obs(self, raw_obs):
        return self._inner.transform_obs(raw_obs)

    def transform_action(self, raw_action):
        ...  # remember the chunk, ensemble every live chunk's row for this step
        return self._inner.transform_action(ensembled)  # spec-driven conversion

    def reset(self, env_index=None):
        self._chunks.clear()  # never ensemble across episodes
```

Build it by resolving first, then wrapping:

```python
adapter = ChunkEnsembleAdapter(adapt.resolve(tags, env.observation_space, env.action_space, spec))
```

A custom adapter reports `is_stateful` as `True` by default (the safe assumption). That means it must keep affinity to its stream: override `reset` to clear episode-scoped state, and wire it to the per-episode boundary (the model worker's `on_episode_end`) so a finished episode never bleeds into the next. `reset`'s `env_index` names the vector lane whose episode rolled (`None` is a whole-vector reset); the per-lane affinity manager that makes a stateful adapter safe across the lanes of a vector env is not implemented yet, so run one instance per lane for now.

The full worked example (the spec, the ensemble math, and the factory that resolves then wraps) is {source}`examples/python/vla_adapters/models/act.py`.

## Pair override -- replace the adapter for one pairing

When a single `(model, env)` pairing is special enough that none of the above fit, replace its adapter outright. This needs no RLMesh machinery: keep a registry keyed by the pair and consult it before {func}`~rlmesh.adapters.resolve`.

```python
import rlmesh.adapters as adapt

OVERRIDES = {
    ("xvla", "simpler-bridge"): XVLABridgeAdapter,
}

def build_adapter(model_name, env_name, env, spec):
    if (override := OVERRIDES.get((model_name, env_name))) is not None:
        return override()
    return adapt.resolve(env.tags, env.observation_space, env.action_space, spec)
```

Because both the resolved adapter and any override are {class}`~rlmesh.adapters.AdapterBase` instances, the rest of the eval loop (`wrap_predict`, the served path via {func}`~rlmesh.adapters.resolve_from_contract`) is identical either way. Reach for an override only when the special-casing is pairing-wide; a per-slot or per-field need belongs in one of the earlier hatches.

## See also

- {source}`examples/python/vla_adapters` -- a runnable registry of models and envs exercising all four hatches.
- {doc}`/user-guide/adapters/reference` -- every leaf, field, and conversion-policy rung in one place.
- {doc}`/user-guide/adapters` -- the declarative pipeline these hatches extend.
- {doc}`/api/adapters` -- exact signatures for `Custom`, `CustomEncoding`, `AdapterBase`, and `resolve`.
