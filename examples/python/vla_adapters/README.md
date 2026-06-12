# VLA adapters: one spec per model, one spec per env

This example shows how a project using `rlmesh.adapters` is laid out so that models and environments
can be added independently. There are no per-pair adapters: every (model, env) combination is
derived at runtime by `rlmesh.adapters.resolve`, which matches the two specs by semantic role.

```
vla_adapters/
├── eval.py               # generic harness: --model X --env Y, any pairing
├── overrides/
│   ├── __init__.py       # OVERRIDES registry (one line per special pairing)
│   └── xvla_simpler_bridge.py  # from-scratch adapter for one pairing
├── models/
│   ├── __init__.py       # MODELS registry (one line per checkpoint)
│   ├── act.py            # ACT-style chunking: stateful custom adapter (AdapterBase)
│   ├── smolvla.py        # SmolVLA's input payload + action layout
│   └── xvla.py           # X-VLA: rot6d proprio, 20-dim single/bimanual EE6D action
└── envs/
    ├── __init__.py       # ENVS registry (one line per environment)
    ├── libero.py         # two cameras, xyzw quat, flat obs keys
    └── simpler_bridge.py # one camera, wxyz quat, nested obs keys
```

3 model files + 2 env files already cover 6 combinations; adding a third env makes that 9 without
touching any model code. With bespoke adapters the same coverage is N x M hand-written classes.

There are two custom-adapter escape hatches, used at different scopes:

- **Model-wide (compose resolution):** `act.py` is an ACT-style policy that emits 8-step action
  chunks; its temporal ensembling is stateful (it remembers past chunks), which no declarative spec
  can express. Its `ChunkEnsembleAdapter` subclasses `rlmesh.adapters.AdapterBase`, delegating
  observation handling and per-step action conversion to the resolved adapter and adding only the
  ensemble — so it still works against every registered env, including ones it has never seen.
- **Pair-level (complete overwrite):** `overrides/` maps one specific `(model, env)` pairing to a
  from-scratch adapter that touches neither spec — for special conditions like state math outside
  the declarative vocabulary or an action head needing differential IK against that robot's
  kinematic chain. The harness checks overrides first; the same model and env resolve declaratively
  in all their other pairings.

## Run it

The dry run needs no simulator, GPU, or server — it pushes a synthetic observation through the
resolved pipeline and prints the model payload and the env-ready action:

```sh
cd examples/python/vla_adapters
uv run eval.py                                  # every model x env pair
uv run eval.py --model xvla --env simpler-bridge  # a single pairing
```

(Equivalently: `python -m vla_adapters.eval ...` from `examples/python`.)

Each run starts by printing `adapter.describe()` — the exact transformations the resolver chose for
that pairing, e.g. for `xvla` on `simpler-bridge`:

- both X-VLA camera inputs fall back to the env's only camera,
- `agent.eef_quat` is converted `quat_wxyz -> rot6d`, and the second-arm proprio components resolve
  to zero fill because this env declares no `_2` roles (the spec marks them `optional`),
- the 20-dim EE6D action is sliced, `rot6d -> axis_angle` converted, and the second-arm dims are
  dropped because the env does not consume them.

Note that X-VLA's spec never hardcodes zero padding for dims 11-20: it declares them as second-arm
components, and the padding/dropping above is _derived_ from the env spec at resolve time. A
bimanual env declaring the `_2` roles would consume those same dims for real, with no model change.

Against a live endpoint (e.g. an env started with `rlmesh.EnvServer`), pass `--address`; the harness
then prefers the spec the env publishes in its contract metadata (`SPEC.to_metadata()` on the server
side) over the local registry copy:

```sh
uv run eval.py --model smolvla --env libero --address 127.0.0.1:5555
```

## Adding a model

1. Create `models/<name>.py` declaring `SPEC` (a `ModelIOSpec`: what payload keys the checkpoint
   ingests, what action vector it emits) and `load_predict_fn()` (loads the checkpoint and returns
   its raw predict callable — no env-specific code in it).
2. Register it with one line in `models/__init__.py`.

It can now be evaluated on every registered env. A different fine-tune of the same architecture
(different cameras, state layout, action dims) is just another spec module.

## Adding an environment

1. Create `envs/<name>.py` declaring `SPEC` (an `EnvIOSpec`: obs keys with semantic roles and
   encodings, plus the action layout) and a `sample_obs()` for dry runs.
2. Register it with one line in `envs/__init__.py`.

Better still, serve the env with `metadata={**SPEC.to_metadata()}` so remote clients discover the
spec from the handshake and need no registry entry at all.

## When the built-in vocabulary is not enough

If a checkpoint needs feature engineering the declarative spec cannot express, add a `CustomInput`
to its spec with an in-process callable (or a `module:callable` entrypoint string, which `resolve`
only imports when called with `trust_entrypoints=True`). Specs are data — nothing in them is ever
eval'd.
