# Model reference

The complete reference for the `Model` class: every predict corner and how the runtime derives the ones you leave out, the batching and chunk-replay semantics, the device and framework handling, the model-quirk recipes, and the serve-and-connect surface.

For the concepts (the two construction styles, the spec, the lifecycle), start with {doc}`/user-guide/models`. For how the model spec matches an environment, see {doc}`/user-guide/adapters` and {doc}`/user-guide/adapters/reference`. For the evaluation loop (`run` / `session` / `read`), see {doc}`/user-guide/evaluation`. For exact signatures, see {doc}`/api/models`. Examples live at {source}`examples/python/vla_adapters`.

## The four corners

A model implements one or more of four predict corners. They sit on a 2×2 lattice over a **batch** axis (one lane vs. all vectorized lanes fused into one forward) and a **chunk** axis (one action vs. a chunk of future actions per forward).

| Corner                | Signature                                                | Receives                                 | Returns                     | Reach for it when                                             |
| --------------------- | -------------------------------------------------------- | ---------------------------------------- | --------------------------- | ------------------------------------------------------------- |
| `predict`             | `predict(observation)`                                   | one observation                          | one action                  | a simple per-step policy with nothing to chunk or batch       |
| `predict_chunk`       | `predict_chunk(observation[, execution_horizon])`        | one observation                          | action chunk `[H, ...]`     | an ACT / diffusion / flow head that emits a chunk per forward |
| `predict_batch`       | `predict_batch(observations)`                            | one fused observation, leaves `[N, ...]` | batched action `[N, ...]`   | a batched policy with one forward over all lanes              |
| `predict_chunk_batch` | `predict_chunk_batch(observations[, execution_horizon])` | one fused observation, leaves `[N, ...]` | batched chunk `[N, H, ...]` | a batched VLA / action-head policy (all four derive from it)  |

On the chunk corners the leading axis of the return is the chunk axis. On the batch corners the leading axis is the batch axis; a chunk corner that is also batched returns the batch axis first, then the per-lane chunk axis (`[N, H, ...]`). A `predict` that subclasses `Model` and does not override one of these inherits a method that raises, so the runtime knows it is undefined.

## Corner synthesis

Implement the most general corner your policy supports and the runtime fills the rest. It only ever derives _downward_ along the lattice.

```{mermaid}
graph TD
    PCB["predict_chunk_batch<br/>[N, H, ...]"] -->|de-batch| PC["predict_chunk<br/>[H, ...]"]
    PCB -->|de-chunk| PB["predict_batch<br/>[N, ...]"]
    PB -->|de-batch| P["predict<br/>one action"]
    PC -->|de-chunk| P
```

| Derivation   | How it works                                                                                                                                                                        |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **De-batch** | Stack the lone observation into a batch of one, call the batched corner, peel lane 0 back off. Turns `predict_batch` into `predict` and `predict_chunk_batch` into `predict_chunk`. |
| **De-chunk** | Run the chunk corner at horizon 1 and take the first action. Turns `predict_chunk` into `predict` and `predict_chunk_batch` into `predict_batch`.                                   |

So `predict_chunk_batch` alone yields all four corners. Going _up_ either axis is impossible: chunking is a model capability rather than glue, and batching up is left to the engine's per-lane loop.

Two cases are worth pinning down:

- **The ambiguous `predict`.** When both `predict_batch` and `predict_chunk` exist but `predict` does not, the runtime derives `predict` from the un-chunked `predict_batch`, so a single-step call stays un-chunked instead of paying for a chunk decode it would discard.
- **The native backend cannot de-chunk.** De-chunking indexes the chunk axis on array leaves, which the native `rlmesh.Model` over raw values does not have. A chunk-only model on that backend must define `predict()` explicitly, or construction raises a clear `TypeError`. The numpy, torch, and jax backends have no such restriction.

```{note}
De-batch and de-chunk are byte-consistent with the served path. The local first-frame split matches
the engine's `split_chunk`, so a model derived down to `predict` behaves the same whether you drive it
in-process or over the wire.
```

## The execution horizon

`execution_horizon` is optional on the chunk corners and detected by arity. A corner with a single positional parameter (`predict_chunk(obs)`) is called without the horizon; a corner that declares a second positional parameter (`predict_chunk(obs, execution_horizon=1)`) receives it.

| Form                                      | Behavior                                                                                                                                                                                                                                               |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `predict_chunk(obs)`                      | The horizon is swallowed before the call. Return the native chunk; the runtime executes a prefix of it. Most policies use this — a trained chunk length is fixed by the weights.                                                                       |
| `predict_chunk(obs, execution_horizon=1)` | The runtime fills `execution_horizon` with how many actions it will execute before re-planning. An autoregressive decoder that can stop early decodes exactly that many. Keep the `=1` default so the override stays compatible with the one-arg base. |

```python
def predict_chunk(self, observation, execution_horizon=1):
    chunk = self.policy.decode(observation)        # native [H_native, A]
    return chunk[:execution_horizon]               # truncate to what runs
```

### Replay semantics

The runtime owns the replay. It calls the model once, splits the returned chunk along its leading axis, executes one action per environment `step`, and re-plans only when the queue drains. A horizon of 1 is a passthrough that never queues. A horizon above 1 caps the replay at that many actions, so a receding-horizon model may emit a longer chunk than it re-plans. The episode boundary (`reset`) drops any un-replayed tail, so a chunk never bleeds across episodes.

The chunk split treats a string, a bytes value, a mapping (a Dict-space action with the chunk axis inside each leaf), or a non-iterable scalar as a single-step chunk, matching the native `split_chunk`. An array's leading axis is the chunk axis. A returned empty chunk raises rather than running an empty step.

`execution_horizon` only engages on a chunk corner. Requesting it on a model with no chunk corner warns and runs un-chunked, so the default of 1 is always safe. The end-to-end story (where the horizon is passed and how the rollout applies it) is in {doc}`/user-guide/evaluation`.

## Batched observation fusion

The batch corners do not receive a list of `N` observations. The runtime fuses the per-lane observations into one batched observation: every leaf gains a leading batch axis, so a Dict observation arrives as `{key: array[N, ...]}` — the shape every RL/VLA runtime hands a policy. You return the batched action the same way, and the runtime splits the batch axis back per lane.

```python
def predict_batch(self, observations):
    # observations["image"] is array[N, H, W, C]; one forward over the batch.
    return self.net(observations["image"])          # return array[N, action_dim]
```

| Rule          | Detail                                                                                                    |
| ------------- | --------------------------------------------------------------------------------------------------------- |
| Return shape  | Exactly `[N, ...]`, or `[N, H, ...]` for a chunk corner. The runtime splits the batch axis back per lane. |
| Text leaves   | Stay per-lane lists, not stacked arrays. An `instruction` leaf arrives as a list of `N` strings.          |
| Ragged leaves | Error. Lanes must share an observation space so the leaves align.                                         |

This is the `tree_stack` / `tree_unstack` machinery in `_value_conversion.py`. `tree_stack` recurses the container and stacks aligned leaves with the framework's stack op; `tree_unstack` is its inverse and splits only the batch axis, leaving each lane's chunk axis intact.

```{note}
The native `rlmesh.Model` over raw values cannot fuse opaque tensors, so its batch corners receive the
per-lane **list** and return a list. Use a numpy, torch, or jax backend when you want true batched fusion.
```

## Device and framework handling

For torch and jax models, `self.device` is the single source of truth for placement. Set it inside `load()` when you move your weights:

```python
def load(self):
    self.device = "cuda"
    self.policy = Policy.from_pretrained("org/checkpoint").to(self.device)
```

| Behavior      | Detail                                                                                                                                                                 |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Obs placement | RLMesh moves every framework tensor leaf of an observation onto `self.device` before `predict`. Non-tensor leaves pass through. You never call `.to(device)` yourself. |
| Read time     | `device` is read at predict time, so a value set in `load()` is honored even though `load()` runs after the served worker is built.                                    |
| Default       | `None` leaves observations as decoded.                                                                                                                                 |
| Wrong backend | Setting `device` on a numpy or native model raises a `ValueError` — those frameworks have no device concept.                                                           |

The backend itself only changes value conversion at the predict seam. The four corners and the lifecycle are identical across native, numpy, torch, and jax. See {doc}`/user-guide/backends` for choosing one and {doc}`/api/torch` for the torch helpers.

## Model quirks

Real checkpoints rarely speak the spec's keys and shapes directly. The seams above absorb the difference so model code stays clean.

| Quirk                                | Handling                                                                                                       |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Compute device                       | Set `self.device` in `load()`; requires a torch/jax `Model`. RLMesh moves obs leaves onto it before `predict`. |
| Stateful policy (RNN, chunk replay)  | Implement `reset()`; it fires at each episode boundary, local and served.                                      |
| Policy expects different keys        | Write an `_obs()` helper mapping spec keys to the policy's dict (the SmolVLA pattern, below).                  |
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

Tokenization stays in the model on purpose. `Text` delivers the instruction as a string; tokenize it inside your prediction function. There is no `TokenizerInput` on the spec.

## Construction and lifecycle

The four lifecycle seams fire identically on the local loop and the served path.

| Seam                          | Default | When it fires                                                                | Notes                                                                                                                                   |
| ----------------------------- | ------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `load(**binding)`             | no-op   | once, during construction (subclass mode), before the native worker is built | Keep heavy imports here. On the served path the eager auto-load is suppressed and `load(**binding)` runs once with the resolved params. |
| `reset()`                     | no-op   | at each episode boundary                                                     | Wired to the `on_episode_end` edge. Clear per-episode state here.                                                                       |
| `close()`                     | no-op   | at the end of a run                                                          | Wired to the `on_close` edge.                                                                                                           |
| `on_episode_end` / `on_close` | none    | constructor callbacks                                                        | Override a subclass's `reset` / `close` (and a wrapped callable's), the only edges for a callable that cannot define methods.           |

There is no episode-_begin_ hook. Per-episode state is lazy-seeded on the first `predict`, so a stateful model clears its state at episode _end_.

Other construction inputs:

| Attribute / argument           | Meaning                                                                                                                                                                                                                          |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `spec` (class attr or `spec=`) | A {class}`~rlmesh.adapters.ModelSpec`, `rlmesh.NO_ADAPTER`, or `None`. The `spec=` kwarg overrides the class attribute per instance. `NO_ADAPTER` skips resolution and `predict` receives the raw observation.                   |
| `params` (class attr)          | A `ParamSpec`, the model counterpart of {attr}`rlmesh.EnvFactory.params`, presented and swept the same way. Advisory today: a dashboard reads it via `rlmesh.describe`; binding it into `load` is gated on the served-load seam. |
| `trust_entrypoints=`           | Allow `module:callable` custom-input entrypoints in a spec to be imported during adapter resolution.                                                                                                                             |
| `describe()`                   | Classmethod returning the model's full metadata envelope (the same envelope `rlmesh.describe(MyModel)` returns).                                                                                                                 |

For a baseline that ignores observations, pass `rlmesh.RANDOM_SAMPLE` as the model to `rlmesh.run` / `rlmesh.session`. It samples the environment's action space and resolves no adapter, even on a tagged env.

## Serve and connect

`serve(address)` hosts the model as a blocking endpoint. A spec'd model resolves its adapter per env from the contract the runtime delivers in the handshake, then applies it around `predict`; a spec-less or `NO_ADAPTER` model serves its own predict directly.

```python
MyPolicy().serve("0.0.0.0:50051")
```

The lazy path skips a hand-written entrypoint. Point `rlmesh.serve` at the recipe and it writes the serve loop:

```sh
python -m rlmesh.serve my_pkg:MyPolicy
```

The serve env vars match the environment serve CLI:

| Env var              | Meaning                                                           |
| -------------------- | ----------------------------------------------------------------- |
| `RLMESH_ADDRESS`     | Bind address (e.g. `0.0.0.0:50051`).                              |
| `RLMESH_MAKE_KWARGS` | JSON object bound to `load(**binding)` once the worker is built.  |
| `RLMESH_FRAMEWORK`   | `torch` / `jax` / `numpy`.                                        |
| `RLMESH_DEVICE`      | Device for incoming observations (torch/jax only), e.g. `cuda:0`. |

Connect to a served model in one of two ways:

- `rlmesh.RemoteModel(address)` — un-managed; you started the server yourself.
- `rlmesh.SandboxModel("image://my-model:latest")` — managed; RLMesh runs the prebuilt image. `docker push` the tag to a registry it can reach first.

```python
sess = rlmesh.session(rlmesh.SandboxModel("image://my-model:latest"), env)
```

Both handles plug into `rlmesh.run` / `rlmesh.session` like a local model. The full bring-your-own-container example is {source}`examples/python/byo_container`; the sandbox mechanics are in {doc}`/user-guide/sandbox`, and connecting clients in {doc}`/user-guide/remote-clients`.

## Match your model's shape

Find the row that matches your policy, then build it.

| My policy is...                              | Build it as...                                          |
| -------------------------------------------- | ------------------------------------------------------- |
| a one-file baseline                          | `rlmesh.numpy.Model(lambda obs: ...)`                   |
| a per-step policy with weights               | a `Model` subclass with `load()` + `predict()`          |
| a batched VLA with an action head            | a `rlmesh.torch.Model` with `predict_chunk_batch`       |
| an ACT / diffusion / flow chunker            | a `Model` subclass with `predict_chunk`                 |
| a checkpoint whose keys differ from the spec | the same, plus an `_obs()` remap helper                 |
| stateful across steps (RNN, ensembling)      | the same, plus `reset()`                                |
| GPU-resident                                 | a torch/jax `Model` that sets `self.device` in `load()` |

### Common pitfalls

| Symptom                                       | Cause                                      | Fix                                                     |
| --------------------------------------------- | ------------------------------------------ | ------------------------------------------------------- |
| `TypeError` on construction for a chunk model | chunk-only on the native `rlmesh.Model`    | define `predict()`, or use a numpy/torch/jax backend    |
| `execution_horizon` seems ignored             | the model has no chunk corner              | implement `predict_chunk` or `predict_chunk_batch`      |
| Batch corner gets a list, not a fused obs     | the model is the native `rlmesh.Model`     | use a numpy/torch/jax backend for true fusion           |
| `ValueError` on `device=`                     | `device` set on a numpy/native model       | use a torch/jax `Model`, or drop `device`               |
| Obs not on the GPU                            | `device` set somewhere other than `load()` | set `self.device` in `load()` — the one source of truth |
| Stateful policy leaks across episodes         | per-episode state never cleared            | implement `reset()` (wired to the episode boundary)     |

## Where next

- {doc}`/user-guide/models` — the concept tour: construction styles, the spec, the lifecycle, and the worked SmolVLA model.
- {doc}`/user-guide/evaluation` — `run` / `session` / `read`, seeds and instruction injection, and the execution horizon end to end.
- {doc}`/user-guide/adapters` and {doc}`/user-guide/adapters/reference` — how the model spec matches an environment by role.
- {doc}`/user-guide/environments` — the environment side of the contract.
- {doc}`/api/models` — the autodoc signatures for every symbol above.
