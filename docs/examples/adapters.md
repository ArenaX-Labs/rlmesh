# Adapters Example

These examples show {doc}`../user-guide/adapters`: an environment tags its spaces, a model declares
its format, and the pairing is derived at runtime. Both need the NumPy backend and run with no GPU
or simulator.

## Smallest serve-and-run loop

One process serves an tagged environment and runs an adapted model against it:

```bash
uv run python examples/python/adapters_quickstart/serve_and_run.py
```

The environment is served with its tags published in the contract:

```python
server = rlmesh.EnvServer(env, "127.0.0.1:0", tags=ENV_TAGS)
server.start()
```

The model declares its own format and resolves the adapter from the handshake — `predict` only ever
sees the model's payload, the environment only ever sees its action format:

```python
model = Model(predict, spec=MODEL_SPEC)
model.run(client, max_episodes=1)
```

The script first prints `resolve_from_contract(...).describe()`, the exact transformations chosen:
the image is resized, `quat_xyzw -> rot6d` is applied to the rotation, the instruction key is
remapped, and the model's `rot6d` action is converted `rot6d -> axis_angle` and clipped into the
environment's action.

## A project with many models and environments

The VLA example lays out a project where models and environments are added independently. With no
per-pair adapters, every (model, environment) combination is derived by `resolve`:

```bash
cd examples/python/vla_adapters
uv run eval.py                                    # every model x env pair, offline
uv run eval.py --model xvla --env libero          # a single pairing
```

```
vla_adapters/
├── eval.py            # generic harness: --model X --env Y, any pairing
├── models/            # one ModelSpec per checkpoint (smolvla, xvla, act)
├── envs/              # one EnvTags + spaces per environment (libero, simpler-bridge)
└── overrides/         # complete adapter overwrites for special pairings
```

Three model files and two environment files already cover six combinations; adding a third
environment makes nine without touching any model code. The example also demonstrates the escape
hatches — a stateful `AdapterBase` subclass for an ACT-style chunking policy, and a pair override —
and the live path against a served endpoint:

```bash
uv run eval.py --model smolvla --env libero --address 127.0.0.1:5555
```
