# Models

A model in RLMesh is a policy you serve or drive against an environment. You wrap a predict callable or subclass a backend `Model`, declare the payload your policy ingests with a {class}`~rlmesh.adapters.ModelSpec`, and RLMesh resolves the model-to-environment adapter for you. The model only ever sees its own declared input and output format; no per-environment glue lives in your code.

Authoring a model is independent of authoring an environment. The two sides meet only through roles, resolved by {doc}`adapters`. The environment side is in {doc}`environments`.

This page is the concept tour. Reach for {doc}`models/reference` when you need the full corner-synthesis rules, the batching and device mechanics, the serve-and-connect surface, or the model-quirk recipes.

## Construction styles

There are two ways to build a model, and the choice is about reuse.

| Style                          | What you write                                                             | Reach for it when               |
| ------------------------------ | -------------------------------------------------------------------------- | ------------------------------- |
| **Wrap** a predict callable    | `rlmesh.numpy.Model(lambda obs: ...)`                                      | a baseline or a one-file script |
| **Subclass** a backend `Model` | set the `spec` class attribute, implement `load()` plus one predict corner | the form you serve and ship     |

Pick the backend by the array type your policy speaks. The backend only changes how observation leaves are decoded before `predict` and how returned actions are encoded; the four predict corners and the lifecycle are identical across all of them.

| Class                | Import                      | Observation it receives      | Action it returns    |
| -------------------- | --------------------------- | ---------------------------- | -------------------- |
| `rlmesh.Model`       | native (no extras)          | RLMesh-native values         | RLMesh-native values |
| `rlmesh.numpy.Model` | `pip install rlmesh[numpy]` | NumPy arrays + primitives    | NumPy arrays         |
| `rlmesh.torch.Model` | `pip install rlmesh[torch]` | Torch tensors (device-aware) | Torch tensors        |
| `rlmesh.jax.Model`   | `pip install rlmesh[jax]`   | JAX arrays                   | JAX arrays           |

See {doc}`/api/numpy`, {doc}`/api/torch`, and {doc}`/api/jax` for the backend helpers, and {doc}`backends` for choosing one.

## The minimal model

A subclass loads its weights in `load()` and maps an observation to an action in `predict()`. This is the shape the bring-your-own-container example serves:

```python
import rlmesh


class MyPolicy(rlmesh.numpy.Model):
    def load(self):
        self.bias = 0  # load weights INTO self

    def predict(self, observation):
        return self.bias


model = MyPolicy()
```

Set the `spec` class attribute to a {class}`~rlmesh.adapters.ModelSpec` and the adapter resolves from the environment's published tags, so `predict` works in your model's own conventions regardless of which environment it runs against. Set it to `rlmesh.NO_ADAPTER` to skip resolution and have `predict` receive the raw observation.

```python
import rlmesh
import rlmesh.adapters as adapt


class MyPolicy(rlmesh.torch.Model):
    spec = adapt.ModelSpec(
        input={
            "image": adapt.Image(adapt.IMAGE_PRIMARY, size=224),
            "state": adapt.Concat(adapt.EEF_POS, adapt.GRIPPER_POS),
        },
        output=adapt.Action(adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3)),
    )

    def load(self):
        self.device = "cuda"
        self.net = load_weights().to(self.device)

    def predict(self, observation):
        return self.net(observation["image"], observation["state"])
```

`size=224` sets both height and width. The spec is the model side of the contract in {doc}`adapters`; every field on every spec leaf is in {doc}`adapters/reference`.

## Lifecycle

A model has four lifecycle seams. They fire identically on the local `run` / `session` loop and the served wire path, so a model behaves the same whether you drive it in-process or dial it over a socket.

| Seam                          | When it fires            | What to do in it                                                      |
| ----------------------------- | ------------------------ | --------------------------------------------------------------------- |
| `load(**binding)`             | once, at construction    | Load weights into `self`; keep heavy imports here, not at module top. |
| `reset()`                     | at each episode boundary | Clear per-episode state (RNN hidden state, chunk replay).             |
| `close()`                     | at the end of a run      | Release resources.                                                    |
| `on_episode_end` / `on_close` | constructor callbacks    | The same edges for a wrapped callable that cannot override methods.   |

There is no episode-_begin_ hook. Per-episode state is lazy-seeded on the first `predict`, so a stateful model clears its state at episode _end_ via `reset()` (a subclass's `reset()` is wired to the `on_episode_end` edge).

For torch and jax models, set `self.device` inside `load()` when you move your weights onto it. That is the one source of truth: RLMesh moves every observation tensor leaf onto `self.device` before `predict`, so you never call `.to(device)` yourself.

```python
def load(self):
    self.device = "cuda"
    self.policy = Policy.from_pretrained("org/checkpoint").to(self.device)
    self.stats = load_norm_stats()  # normalization stats load here too
```

The device and framework mechanics, and what happens when you set `device` on a numpy or native model, are in {doc}`models/reference`.

## The four corners

`predict` is one of four corners. They form a 2×2 over a **batch** axis (one lane vs. all vectorized lanes at once) and a **chunk** axis (one action vs. a chunk of future actions per forward pass). You implement the corners your policy supports; the runtime derives the rest.

| Corner                | Signature                                                | Returns                                               |
| --------------------- | -------------------------------------------------------- | ----------------------------------------------------- |
| `predict`             | `predict(observation)`                                   | one action                                            |
| `predict_chunk`       | `predict_chunk(observation[, execution_horizon])`        | action chunk, shape `[H, ...]` (leading axis = chunk) |
| `predict_batch`       | `predict_batch(observations)`                            | batched action `[N, ...]`                             |
| `predict_chunk_batch` | `predict_chunk_batch(observations[, execution_horizon])` | batched chunk `[N, H, ...]`                           |

Implement the most general corner you can and the runtime derives the others by deriving _downward_: it drops the batch axis by running a batch of one, and drops the chunk axis by running at horizon 1 and taking the first action. So a model that defines `predict_chunk_batch` alone gets all four for free.

```{mermaid}
graph TD
    PCB["predict_chunk_batch<br/>[N, H, ...]"] -->|de-batch| PC["predict_chunk<br/>[H, ...]"]
    PCB -->|de-chunk| PB["predict_batch<br/>[N, ...]"]
    PB -->|de-batch| P["predict<br/>one action"]
    PC -->|de-chunk| P
```

Going _up_ either axis is impossible: chunking is a model capability, not glue, and batching up is left to the engine's per-lane loop. The exact derivation rules, the ambiguous `predict` case, and the one restriction on the native backend are in {doc}`models/reference`.

### Which corner should I implement?

| Your policy                      | Implement             | Why                                                                    |
| -------------------------------- | --------------------- | ---------------------------------------------------------------------- |
| Batched VLA / action-head policy | `predict_chunk_batch` | One forward over all lanes returning a chunk; all four derive from it. |
| Simple per-step policy           | `predict`             | One action per observation; nothing to chunk or batch.                 |
| ACT / diffusion / flow chunker   | `predict_chunk`       | Emits a native chunk per forward; the runtime replays it.              |

A batched VLA is the common case. The SmolVLA-style pattern is a `rlmesh.torch.Model` with a two-camera, state, and instruction spec, `from_pretrained` in `load()`, and `predict_chunk_batch` returning the native chunk truncated to the horizon and moved to host:

```python
import rlmesh
import rlmesh.adapters as adapt


class SmolVLA(rlmesh.torch.Model):
    spec = adapt.ModelSpec(
        input={
            "observation.images.image": adapt.Image(adapt.IMAGE_PRIMARY, size=224),
            "observation.images.wrist": adapt.Image(adapt.IMAGE_WRIST, size=224),
            "observation.state": adapt.Concat(
                adapt.EEF_POS,
                adapt.State(adapt.EEF_ROT, encoding="axis_angle"),
                adapt.GRIPPER_POS,
            ),
            "instruction": adapt.Text(adapt.INSTRUCTION),
        },
        output=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
            adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
            adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0)),
        ),
    )

    def load(self):
        from lerobot.policies.smolvla import SmolVLAPolicy

        self.device = "cuda"
        self.policy = SmolVLAPolicy.from_pretrained("org/smolvla").to(self.device)

    def reset(self):
        self.policy.reset()

    def predict_chunk_batch(self, observations, execution_horizon=1):
        chunk = self.policy.predict_action_chunk(self._obs(observations))  # [N, H, A]
        return chunk[:, :execution_horizon].cpu()
```

The `_obs()` remap helper is the recipe for a checkpoint whose keys differ from the spec; see [the model-quirk recipes](models/reference.md#model-quirks). More worked VLA models live in {source}`examples/python/vla_adapters`.

### The execution horizon

`execution_horizon` is **optional** on the chunk corners. Most policies write `predict_chunk(obs)` and ignore it: a trained chunk length is fixed, and the runtime executes a prefix of the native chunk. A decoder that can stop early writes `predict_chunk(obs, execution_horizon=1)` to receive how many actions the runtime will execute before re-planning, and decodes exactly that many. Keep the `=1` default so it stays a compatible override.

```python
def predict_chunk(self, observation, execution_horizon=1):
    chunk = self.policy.decode(observation)        # native [H_native, A]
    return chunk[:execution_horizon]               # truncate to what runs
```

The runtime chooses the horizon at the call site (`run(..., execution_horizon=N)`); the spec output declares the layout of one action, not the chunk length. The full replay story is in {doc}`evaluation`.

## Batched observations

The batch corners do not receive a list of `N` observations. The runtime **fuses** the per-lane observations into one batched observation: every leaf gains a leading batch axis, so a Dict observation arrives as `{key: array[N, ...]}`. You return the batched action the same way, leaves carrying the leading batch axis, and the runtime splits it back per lane.

```python
def predict_batch(self, observations):
    # observations["image"] is array[N, H, W, C]; one forward over the batch.
    return self.net(observations["image"])          # return array[N, action_dim]
```

Text leaves stay per-lane lists rather than stacked arrays, and lanes must share an observation space so the leaves align. The fusion rules, the `tree_stack` / `tree_unstack` machinery, and the native-backend fallback are in {doc}`models/reference`.

## Run it

Drive the model against an environment with `run` to auto-pump whole episodes, or `session` to step `reset` / `predict` / `step` by hand:

```python
model = MyPolicy()

result = model.run(env, seeds=range(10), instruction="put the cup on the plate")
print(result.mean_reward, result.success_rate)
```

`run` returns a {class}`~rlmesh.RunResult` with `.episodes`, `.mean_reward`, and `.success_rate`. `env` may be a local env, an {class}`~rlmesh.EnvFactory`, a `RemoteEnv`, or a bare address string the loop dials. The module-level `rlmesh.run(model, env, ...)` and `rlmesh.session(model, env, ...)` accept a bare predict callable, a `Model` subclass or instance, or a served handle.

The full `run` / `session` / `read` story — seeds, instruction injection, the execution horizon end to end, and reading canonical roles off an observation — is in {doc}`evaluation`.

## Build into a container

A model ships as a container image that serves the policy on an endpoint. Write a `Dockerfile` and a small entrypoint, or skip the entrypoint with the lazy serve path.

The entrypoint subclasses a backend `Model` and calls `serve`:

```python
import os

import rlmesh


class MyPolicy(rlmesh.numpy.Model):
    def load(self):
        self.bias = 0

    def predict(self, observation):
        return self.bias


if __name__ == "__main__":
    MyPolicy().serve(os.environ.get("RLMESH_ADDRESS", "0.0.0.0:50051"))
```

```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir "rlmesh" "gymnasium" "numpy"
WORKDIR /app
COPY entrypoint.py .
ENV RLMESH_ADDRESS=0.0.0.0:50051
ENTRYPOINT ["python", "entrypoint.py"]
```

The lazy path skips the entrypoint file. Point the image (or your shell) at the recipe and RLMesh writes the serve loop:

```sh
python -m rlmesh.serve my_pkg:MyPolicy
```

On the served path, `load(**binding)` receives its parameters from `RLMESH_MAKE_KWARGS` once the worker is built. Connect to a served model with `rlmesh.RemoteModel(address)` for a server you started yourself, or `rlmesh.SandboxModel("image://my-model:latest")` for a prebuilt image RLMesh runs:

```python
sess = rlmesh.session(rlmesh.SandboxModel("image://my-model:latest"), env)
```

Both handles plug into `rlmesh.run` / `rlmesh.session` like a local model. The serve env vars and address forms, and the full bring-your-own-container example, are covered in {doc}`models/reference`. The sandbox mechanics are in {doc}`sandbox` and connecting clients in {doc}`remote-clients`.

## Where next

- {doc}`models/reference` — every predict corner with its synthesis rules, the batching and chunk-replay semantics, the device and framework handling, the model-quirk recipes, and the serve-and-connect surface.
- {doc}`evaluation` — `run` / `session` / `read`, seeds and instruction injection, and the execution horizon end to end.
- {doc}`adapters` — how the model spec and the environment tags match by role.
- {doc}`/api/models` — the autodoc signatures for every symbol above.
