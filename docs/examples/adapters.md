# Adapters Example

These examples show {doc}`../user-guide/adapters`: an environment tags its spaces, a model declares its format, and the pairing is derived at runtime. Both need the NumPy backend and run with no GPU or simulator.

## Smallest serve-and-run loop

One process serves a tagged environment and runs an adapted model against it. The runnable file is {source}`examples/python/adapters/serve_and_run.py <examples/python/adapters/serve_and_run.py>`.

```bash
uv run python examples/python/adapters/serve_and_run.py
```

The environment tags its spaces: each observation entry and action component gets a semantic role plus the few facts the spaces cannot carry (rotation encoding, ranges, clipping).

```python
import rlmesh.adapters as adapt

ENV_TAGS = adapt.EnvTags(
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

The model declares its own format, written without any knowledge of an environment: a 224x224 image, a `list` state whose rotation is `rot6d`, the instruction under its own key, and a `rot6d` action.

```python
MODEL_SPEC = adapt.ModelSpec(
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

The `predict` function works purely in the model's format; the payload already arrives shaped to `MODEL_SPEC`, so there is no per-environment glue inside it.

```python
def predict(payload: dict[str, Any]) -> Any:
    assert payload["image"].shape == (224, 224, 3)
    assert len(payload["proprio"]) == 10  # pos(3) + rot6d(6) + grip(1)
    return np.zeros(MODEL_SPEC.action.dim, dtype=np.float32)
```

The env is served with its tags published in the contract, then `Model(spec=...).run(env)` resolves the adapter from that contract and runs the episode.

```python
import rlmesh
from rlmesh.numpy import Model, RemoteEnv

server = rlmesh.EnvServer(env, "127.0.0.1:0", tags=ENV_TAGS)
server.start()
client = RemoteEnv(server.address)

print(adapt.resolve_from_contract(client.env_contract, MODEL_SPEC).describe())
Model(predict, spec=MODEL_SPEC).run(client, max_episodes=1)
```

The script first prints `resolve_from_contract(...).describe()`, the exact transformations chosen: the image is resized, `quat_xyzw -> rot6d` is applied to the rotation, the instruction key is remapped, and the model's `rot6d` action is converted `rot6d -> axis_angle` and clipped into the environment's action.

## A project with many models and environments

The VLA example lays out a project where models and environments are added independently. With no per-pair adapters, `resolve` derives every (model, environment) combination. The runnable harness is {source}`examples/python/vla_adapters/eval.py <examples/python/vla_adapters/eval.py>`.

```bash
cd examples/python
uv run python -m vla_adapters.eval                       # every model x env pair, offline
uv run python -m vla_adapters.eval --model xvla --env simpler-bridge   # a single pairing
```

```
vla_adapters/
├── eval.py            # generic harness: --model X --env Y, any pairing
├── models/            # one ModelSpec per checkpoint (act, geovla, smolvla, xvla)
├── envs/              # one EnvTags + spaces per environment (libero, metaworld, simpler-bridge)
└── overrides/         # complete adapter overwrites for special pairings
```

Each model is one spec module plus a loader; the registry is one line per checkpoint. The same goes for envs, so adding an environment pairs it with every model without touching model code.

```python
# models/smolvla.py
SPEC = adapt.ModelSpec(
    inputs=(
        adapt.ImageInput(
            "observation.images.image", role=adapt.IMAGE_PRIMARY, height=224, width=224
        ),
        adapt.ImageInput(
            "observation.images.image2", role=adapt.IMAGE_WRIST, height=224, width=224
        ),
        adapt.StateInput(
            "observation.state",
            components=(
                adapt.StateComponent(adapt.EEF_POS),
                adapt.StateComponent(adapt.EEF_ROT, encoding="axis_angle"),
                adapt.StateComponent(adapt.GRIPPER_POS),
            ),
            container="list",
        ),
        adapt.TextInput("instruction"),
    ),
    action=adapt.ActionLayout(...),
)
```

The harness picks the most specific mechanism per pairing: a pair-level override, the model's own adapter factory, or plain spec resolution from the env's tags and spaces.

```python
def build_adapter(model_name, env_name, env):
    override = OVERRIDES.get((model_name, env_name))
    if override is not None:
        return override()
    model_entry = MODELS[model_name]
    if model_entry.make_adapter is not None:
        return model_entry.make_adapter(
            env.tags, env.observation_space, env.action_space
        )
    return adapt.resolve(
        env.tags, env.observation_space, env.action_space, model_entry.spec
    )
```

`metaworld` is the flat-observation case: its proprioception is a single `Box` vector split by a `StateLayout`, so the same specs that pair with the Dict envs resolve against it unchanged.

```python
# envs/metaworld.py — one flat leaf split by index range
"proprio": adapt.StateLayout(
    adapt.StateField(adapt.EEF_POS, 3),
    adapt.StateField(adapt.EEF_ROT, 4, encoding="quat_xyzw"),
    adapt.StateField(adapt.GRIPPER_POS, 1),
    adapt.StateField(dim=10),  # object + goal positions: not consumed here
),
```

The example also demonstrates the escape hatches. `act.py` is an ACT-style policy whose temporal ensembling is stateful, so its `ChunkEnsembleAdapter` subclasses `AdapterBase` and wraps the resolved adapter, adding only the ensemble.

```python
class ChunkEnsembleAdapter(adapt.AdapterBase[Any]):
    def transform_obs(self, raw_obs):
        return self._inner.transform_obs(raw_obs)

    def transform_action(self, raw_action):
        chunk = np.asarray(raw_action, dtype=np.float32).reshape(self._horizon, -1)
        self._chunks.append(chunk)
        rows = [c[age] for age, c in enumerate(reversed(self._chunks))]
        weights = np.exp(-self._temperature * np.arange(len(rows)))
        ensembled = np.average(np.stack(rows), axis=0, weights=weights)
        return self._inner.transform_action(ensembled)
```

Against a served endpoint, pass `--address`. For a plain pairing the harness hands `Model(spec=...)` the env and the adapter is built from the handshake; an escape-hatch pairing builds the adapter explicitly and wraps the predict function.

```bash
uv run python -m vla_adapters.eval --model smolvla --env libero --address 127.0.0.1:5555
```

```python
if is_plain:
    print(adapt.resolve_from_contract(env.env_contract, model_entry.spec).describe())
    model = Model(model_entry.load_predict_fn(), spec=model_entry.spec)
    model.run(env, max_episodes=episodes)
else:
    adapter = build_adapter(model_name, env_name, ENVS[env_name])
    model = Model(
        adapter.wrap_predict(model_entry.load_predict_fn()), on_reset=adapter.reset
    )
    model.run(env, max_episodes=episodes)
```
