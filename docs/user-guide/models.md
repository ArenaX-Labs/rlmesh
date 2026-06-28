# Models

A model in RLMesh is a policy you serve or drive against an environment. You wrap a predict callable or subclass a backend `Model`, declare the payload your policy ingests with a {class}`~rlmesh.adapters.ModelSpec`, and RLMesh resolves the model-to-environment adapter for you. The model only ever sees its own declared input and output format; no per-environment glue lives in your code.

This page is the guide to the `Model` class. The autodoc signatures live in {doc}`/api/models`.

## What it is and when

Reach for a `Model` when you have a policy to evaluate against one or more environments. There are two construction styles:

- **Wrap** a predict callable: `rlmesh.numpy.Model(lambda obs: ...)`. Good for a baseline or a one-file script.
- **Subclass** a backend `Model`, set the `spec` class attribute for automatic adapters, and implement `load()` plus one predict corner. This is the form you serve and ship.

Pick the backend by the array type your policy speaks:

| Class                | Import                      | Observation it receives      | Action it returns    |
| -------------------- | --------------------------- | ---------------------------- | -------------------- |
| `rlmesh.Model`       | native (no extras)          | RLMesh-native values         | RLMesh-native values |
| `rlmesh.numpy.Model` | `pip install rlmesh[numpy]` | NumPy arrays + primitives    | NumPy arrays         |
| `rlmesh.torch.Model` | `pip install rlmesh[torch]` | Torch tensors (device-aware) | Torch tensors        |
| `rlmesh.jax.Model`   | `pip install rlmesh[jax]`   | JAX arrays                   | JAX arrays           |

The backend only changes how observation leaves are decoded before `predict` and how returned actions are encoded; the four predict corners and the lifecycle are identical across backends. See {doc}`/api/numpy`, {doc}`/api/torch`, and {doc}`/api/jax` for the backend helpers, and {doc}`/user-guide/backends` for choosing one.

Set the `spec` class attribute to a {class}`~rlmesh.adapters.ModelSpec` to get automatic adapters resolved from the environment's published tags. Set it to `rlmesh.NO_ADAPTER` to skip adapter resolution and have your predict function receive the raw observation. The spec is the model side of the contract documented in {doc}`/user-guide/adapters`.

## The minimal model

A subclass sets up its weights in `load()` and maps an observation to an action in `predict()`. This is the shape the bring-your-own-container example serves:

```python
import rlmesh


class MyPolicy(rlmesh.numpy.Model):
    def load(self):
        self.bias = 0  # load weights INTO self

    def predict(self, observation):
        return self.bias


model = MyPolicy()
```

Add a `spec` to get adapters for free, so `predict` works in your model's own conventions regardless of the environment:

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

`size=224` sets both height and width. See {doc}`/user-guide/adapters/reference` for the full field set on every spec leaf.

## Lifecycle

A model has four lifecycle seams. All of them fire identically on the local `run`/`session` loop and the served wire path, so a model behaves the same whether you drive it in-process or dial it over a socket.

| Seam                          | When it fires            | What to do in it                                                      |
| ----------------------------- | ------------------------ | --------------------------------------------------------------------- |
| `load(**binding)`             | once, at construction    | Load weights into `self`; keep heavy imports here, not at module top. |
| `reset()`                     | at each episode boundary | Clear per-episode state (RNN hidden state, chunk replay).             |
| `close()`                     | at the end of a run      | Release resources.                                                    |
| `on_episode_end` / `on_close` | constructor callbacks    | The same edges for a wrapped callable that cannot override methods.   |

There is no episode-_begin_ hook. Per-episode state is lazy-seeded on the first `predict`, so a stateful model clears its state at episode _end_ via `reset()` (a subclass's `reset()` is wired to the `on_episode_end` edge).

For torch and jax models, set `self.device` inside `load()` alongside moving your weights onto it. That is the one source of truth: RLMesh moves every observation tensor leaf onto `self.device` before `predict`, so you never call `.to(device)` yourself. `device` is ignored by the numpy and native backends, which have no device concept; setting it on one raises.

```python
def load(self):
    self.device = "cuda"
    self.policy = Policy.from_pretrained("org/checkpoint").to(self.device)
    self.stats = load_norm_stats()  # normalization stats load here too
```

## The four corners

`predict` is one of four corners. They form a 2x2 over a **batch** axis (single lane vs. all vectorized lanes at once) and a **chunk** axis (one action vs. a chunk of future actions per forward pass). You implement the corners your policy supports; the runtime derives the rest.

| Corner                | Signature                                                | Returns                                               |
| --------------------- | -------------------------------------------------------- | ----------------------------------------------------- |
| `predict`             | `predict(observation)`                                   | one action                                            |
| `predict_chunk`       | `predict_chunk(observation[, execution_horizon])`        | action chunk, shape `[H, ...]` (leading axis = chunk) |
| `predict_batch`       | `predict_batch(observations)`                            | batched action `[N, ...]`                             |
| `predict_chunk_batch` | `predict_chunk_batch(observations[, execution_horizon])` | batched chunk `[N, H, ...]`                           |

The batch corners receive a **fused** observation, not a list (see [Batched observation fusion](#batched-observation-fusion)). The chunk corners return your model's _native_ chunk; the runtime owns the replay and executes a prefix of it.

### Corner synthesis

Implement the most general corner you can and the runtime derives the others. It only ever derives _downward_:

- **De-batch** a batched corner by stacking one lane into a batch of one, calling the batched corner, and unstacking lane 0. This turns `predict_batch` into `predict` and `predict_chunk_batch` into `predict_chunk`.
- **De-chunk** a chunk corner by running it at horizon 1 and taking the first action.

So a model that defines `predict_chunk_batch` alone gets all four corners for free. Going _up_ either axis is impossible: chunking is a model capability, not glue, and batching up is left to the engine's per-lane loop.

```{note}
The native/identity bridge (`rlmesh.Model` over raw values) cannot de-chunk,
because de-chunking needs array leaves to index. A chunk-only model on that
backend must define `predict()` explicitly, or construction raises a clear error.
The numpy/torch/jax backends have no such restriction.
```

### The execution horizon

`execution_horizon` is **optional** on the chunk corners and detected by arity:

- Write `predict_chunk(obs)` to ignore it. Most policies do: a trained chunk length is fixed, and the runtime executes a prefix of your native chunk.
- Write `predict_chunk(obs, execution_horizon=1)` to receive how many actions the runtime will execute before re-planning. An autoregressive decoder that can stop early uses this to decode exactly that many. Keep the `=1` default so it stays a compatible override.

```python
def predict_chunk(self, observation, execution_horizon=1):
    chunk = self.policy.decode(observation)        # native [H_native, A]
    return chunk[:execution_horizon]               # truncate to what runs
```

```{note}
`execution_horizon` was renamed from `action_horizon`. Action chunking is no
longer a spec knob; the *runtime* chooses the horizon (via `run(..., execution_horizon=N)`)
and you return your native chunk. The spec output declares the layout of one action.
```

### Which corner should I implement?

| Your policy                      | Implement             | Why                                                                    |
| -------------------------------- | --------------------- | ---------------------------------------------------------------------- |
| Batched VLA / action-head policy | `predict_chunk_batch` | One forward over all lanes returning a chunk; all four derive from it. |
| Simple per-step policy           | `predict`             | One action per observation; nothing to chunk or batch.                 |
| ACT / diffusion / flow chunker   | `predict_chunk`       | Emits a native chunk per forward; the runtime replays it.              |

A batched VLA is the common case. The SmolVLA-style pattern: a `rlmesh.torch.Model` with a two-camera + state + instruction spec, `from_pretrained` in `load()`, and `predict_chunk_batch` returning the native chunk truncated to the horizon and moved to host:

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

The `_obs()` remap helper is covered under [Model quirks](#model-quirks-and-how-to-handle-them). More worked VLA models live in {source}`examples/python/vla_adapters`.

## Batched observation fusion

The batch corners (`predict_batch`, `predict_chunk_batch`) do not receive a list of `N` observations. The runtime **fuses** the per-lane observations into one batched observation: every leaf gains a leading batch axis, so a Dict observation arrives as `{key: array[N, ...]}` — the shape every RL/VLA runtime hands a policy. You return the batched action the same way, leaves carrying the leading batch axis (`array[N, action_dim]` or `array[N, H, ...]` for a chunk), and the runtime splits it back per lane.

```python
def predict_batch(self, observations):
    # observations["image"] is array[N, H, W, C]; one forward over the batch.
    return self.net(observations["image"])          # return array[N, action_dim]
```

Rules:

- Return exactly `[N, ...]` (or `[N, H, ...]` for a chunk). The runtime splits the batch axis back per lane.
- **Text leaves stay per-lane lists**, not stacked arrays. An `instruction` leaf arrives as a list of `N` strings.
- **Ragged leaves error.** Lanes must share an observation space so the leaves align.

This is the `tree_stack` / `tree_unstack` machinery in `_value_conversion.py`: `tree_stack` recurses the container and stacks aligned leaves with the framework's stack op; `tree_unstack` is its inverse and splits only the batch axis, leaving each lane's chunk axis intact.

```{note}
The native `rlmesh.Model` over raw values cannot fuse opaque tensors, so its
batch corners receive the per-lane **list** and return a list. Use a numpy/torch/jax
backend when you want true batched fusion.
```

## Model quirks and how to handle them

Real checkpoints rarely speak the spec's keys and shapes directly. The seams above absorb the difference so model code stays clean.

| Quirk                                | Handling                                                                                                       |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Compute device                       | Set `self.device` in `load()`; requires a torch/jax `Model`. RLMesh moves obs leaves onto it before `predict`. |
| Stateful policy (RNN, chunk replay)  | Implement `reset()`; it fires at each episode boundary, local and served.                                      |
| Policy expects different keys        | Write an `_obs()` helper mapping spec keys to the policy's dict (the SmolVLA pattern).                         |
| Normalization stats                  | Load them in `load()` alongside the weights.                                                                   |
| Native chunk longer than the horizon | Return your native chunk; truncate to `execution_horizon` if you accept it, else the runtime uses a prefix.    |
| Instruction text                     | Arrives as a plain `str` (a per-lane list in batch corners). Tokenize inside `predict`, not in the spec.       |

The `_obs()` remap helper is the SmolVLA pattern: the adapter delivers the payload under the spec's keys, and a private method renames them to whatever dict the underlying policy expects.

```python
def _obs(self, observations):
    return {
        "image": observations["observation.images.image"],
        "wrist": observations["observation.images.wrist"],
        "state": observations["observation.state"],
        "task": observations["instruction"],  # a list of N strings in a batch corner
    }
```

Tokenization stays in the model on purpose. `Text` delivers the instruction as a string; tokenize it inside your prediction function.

## Run it

Drive the model against an environment with `run` (auto-pumps whole episodes) or `session` (you drive `reset` / `predict` / `step` by hand):

```python
model = MyPolicy()

result = model.run(env, seeds=range(10), instruction="put the cup on the plate")
print(result.mean_reward, result.success_rate)
```

`run` returns a `RunResult` with `.episodes` (a tuple of `EpisodeResult`), `.mean_reward`, and `.success_rate`. `env` may be a local env, an `EnvFactory`, a `RemoteEnv`, or a bare address string the loop dials. Pass `execution_horizon=N` to execute `N` actions of each predicted chunk before re-planning (only takes effect when the model defines a chunk corner).

```python
sess = model.session(env)
obs, info = sess.reset(seed=0)
action = sess.predict(obs)
obs, reward, terminated, truncated, info = sess.step(action)
sess.close()
```

The module-level `rlmesh.run(model, env, ...)` and `rlmesh.session(model, env, ...)` accept a bare predict callable, a `Model` subclass or instance, or a served `RemoteModel`:

```python
rlmesh.run(lambda obs: 0, env, max_episodes=5)
```

The full `run` / `session` / `read` story (seeds, instruction injection, reading canonical roles off an observation) is in {doc}`/user-guide/evaluation`. The environment side (`EnvFactory`, tags, serving) is in {doc}`/user-guide/environments`.

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

The lazy path skips the entrypoint file entirely. Point the image (or your shell) at the recipe and RLMesh writes the serve loop for you:

```sh
python -m rlmesh.serve my_pkg:MyPolicy
```

On the served path, `load(**binding)` receives its parameters from `RLMESH_MAKE_KWARGS` once the worker is built. The other serve env vars (`RLMESH_FRAMEWORK`, `RLMESH_DEVICE`) and the address forms match the environment serve CLI; see {doc}`/user-guide/serving-environments` for addresses, readiness, and health.

Connect to a served model from a client in one of two ways:

- `rlmesh.RemoteModel(address)` — un-managed; you started the server yourself.
- `rlmesh.SandboxModel("image://my-model:latest")` — managed; RLMesh runs the prebuilt image. `docker push` the tag to a registry it can reach first.

```python
sess = rlmesh.session(rlmesh.SandboxModel("image://my-model:latest"), env)
```

Both handles plug into `rlmesh.run` / `rlmesh.session` like a local model. The full bring-your-own-container example is {source}`examples/python/byo_container`; the sandbox mechanics are in {doc}`/user-guide/sandbox`, and connecting clients in {doc}`/user-guide/remote-clients`.
