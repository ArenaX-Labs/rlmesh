# VLA adapters: tag each env, spec each model

This example lays out a project that uses `rlmesh.adapters` so models and environments can be added independently. There are no per-pair adapters: `rlmesh.adapters.resolve` derives every (model, env) combination at runtime by matching an env's **tags** against a model's **spec** by semantic role.

The split is deliberate. An environment only _tags_ its observation and action **spaces**: it names the role of each entry plus the few facts spaces cannot carry (image layout, rotation encoding, explicit ranges). Widths, dtypes and keys come from the gymnasium spaces. A model _fully specifies_ its payload.

```
vla_adapters/
├── eval.py               # generic harness: --model X --env Y, any pairing
├── overrides/
│   ├── __init__.py       # OVERRIDES registry (one line per special pairing)
│   └── xvla_simpler_bridge.py  # from-scratch adapter for one pairing
├── models/
│   ├── __init__.py       # MODELS registry (one line per checkpoint)
│   ├── act.py            # ACT-style chunking: stateful custom adapter (AdapterBase)
│   ├── geovla.py         # GeoVLA's input payload + action layout
│   ├── smolvla.py        # SmolVLA's input payload + action layout
│   └── xvla.py           # X-VLA: rot6d proprio, 20-dim single/bimanual EE6D action
└── envs/
    ├── __init__.py       # ENVS registry (one line per environment)
    ├── libero.py         # two cameras, xyzw quat, one obs key per quantity
    ├── metaworld.py      # cameras + a flat Box proprio leaf split by a StateLayout
    └── simpler_bridge.py # one camera, wxyz quat, nested obs keys
```

4 model files + 3 env files cover 12 combinations without touching any model code. With bespoke adapters the same coverage is N x M hand-written classes.

There are two custom-adapter escape hatches, used at different scopes:

- **Model-wide (compose resolution):** `act.py` is an ACT-style policy that emits 8-step action chunks; its temporal ensembling is stateful (it remembers past chunks), which no declarative spec can express. Its `ChunkEnsembleAdapter` subclasses `rlmesh.adapters.AdapterBase`, delegating observation handling and per-step action conversion to the resolved adapter and adding only the ensemble — so it still works against every registered env, including ones it has never seen.
- **Pair-level (complete overwrite):** `overrides/` maps one specific `(model, env)` pairing to a from-scratch adapter that touches neither side — for special conditions like state math outside the declarative vocabulary or an action head needing differential IK against that robot's kinematic chain. The harness checks overrides first; the same model and env resolve declaratively in all their other pairings.

## Run it

The dry run needs no simulator, GPU, or server — it pushes a synthetic observation through the resolved pipeline and prints the model payload and the env-ready action. Run it as a module from `examples/python`:

```sh
uv run python -m vla_adapters.eval                       # every model x env pair
uv run python -m vla_adapters.eval --model xvla --env simpler-bridge  # a single pairing
```

Each run starts by printing `adapter.describe()` — the exact transformations the resolver chose for that pairing, e.g. for `xvla` on `libero`:

- `agentview_image`/`robot0_eye_in_hand_image` are resized to 256x256,
- `robot0_eef_quat` is converted `quat_xyzw -> rot6d_rowmajor`, and the second-arm proprio components resolve to zero fill because this env declares no `_2` roles (the spec marks them `optional`),
- the 20-dim EE6D action is sliced, `rot6d_rowmajor -> axis_angle` converted, and the second-arm dims are dropped because the env does not consume them.

X-VLA's spec never hardcodes zero padding for dims 11-20: it declares them as second-arm components, and the padding/dropping above is _derived_ from the env at resolve time. A bimanual env declaring the `_2` roles would consume those same dims for real, with no model change.

Against a live endpoint, pass `--address`. Serve the env with its tags published (`rlmesh.EnvServer(env, tags=TAGS)`); the harness then resolves the adapter straight from the handshake — for a plain pairing it just hands `Model(spec=...)` the env and the adapter is built from the contract:

```sh
uv run python -m vla_adapters.eval --model smolvla --env libero --address 127.0.0.1:5555
```

## Adding a model

1. Create `models/<name>.py` declaring `SPEC` (a `ModelSpec`: what payload keys the checkpoint ingests, what action vector it emits) and `load_predict_fn()` (loads the checkpoint and returns its raw predict callable — no env-specific code in it).
2. Register it with one line in `models/__init__.py`.

It can now be evaluated on every registered env. A different fine-tune of the same architecture (different cameras, state layout, action dims) is just another spec module.

## Adding an environment

1. Create `envs/<name>.py` declaring `TAGS` (an `EnvTags`: obs paths with semantic roles and encodings, plus the action layout), the `OBSERVATION_SPACE`/`ACTION_SPACE` gymnasium spaces, and a `sample_obs()` for dry runs.
2. Register it with one line in `envs/__init__.py`.

Better still, serve the env with `rlmesh.EnvServer(env, tags=TAGS)` so remote clients discover the tags from the handshake (via `resolve_from_contract`) and need no registry entry at all.

### Flat (non-Dict) observations

Some envs return a single flat `Box` vector instead of a dict, with fixed index ranges carrying distinct meaning (e.g. Metaworld). Tag it with a `StateLayout` passed directly as `observation` (mirroring `action` being one `ActionLayout`), splitting the vector into role fields in order:

```python
TAGS = adapt.EnvTags(
    observation=adapt.StateLayout(
        adapt.StateField(adapt.EEF_POS, 3),
        adapt.StateField(adapt.GRIPPER_POS, 1),
        adapt.StateField(dim=2),               # role-less = skip these dims
        adapt.StateField(adapt.JOINT_POS, 4),
    ),
    action=adapt.ActionLayout(adapt.ActionComponent(adapt.ACTION_DELTA_POS, 4)),
)
```

Field widths must sum to the leaf width; offsets are implied by order. Every model still matches purely by role, so the same model spec resolves against this env and a dict env with no change. The fixed indices live on the env side, where they belong.

`envs/metaworld.py` is the runnable version. It is mixed rather than fully flat: a `StateLayout` tags a flat `proprio` leaf inside a `Dict` that also carries the cameras the VLA specs need, so the same `smolvla`/`act`/`xvla` specs pair with it. Run `uv run eval.py --env metaworld` and read `describe()` — each field prints the slice it reads, e.g. `proprio[3:7] (quat_xyzw->rot6d_rowmajor)` for X-VLA.

## When the built-in vocabulary is not enough

If a checkpoint needs feature engineering the declarative spec cannot express, add an `InlineCustomInput` (an in-process callable) or an `EntrypointCustomInput` (a `module:callable` string, which `resolve` only imports when called with `trust_entrypoints=True`) to its spec. Specs are data — nothing in them is ever eval'd.
