# Performance

RLMesh keeps the hot path narrow on purpose: the wire is framework-neutral bytes, the adapter encodes only the observation keys a model reads, and frame history never crosses the network. Most of what makes an eval fast is structural, chosen when you write the model and serve the env, not tuned afterward. This page covers the knobs that actually exist in the code, what each one changes, and the things that look like knobs but are not.

No benchmark numbers appear here. The mechanisms are real; the right setting depends on your model, your env, and your hardware, so measure your own pair.

## Batch the forward pass

A vectorized route runs many env lanes against one model. By default the runtime calls the model once per lane. Define {meth}`predict_batch <rlmesh._models.base.ModelBase.predict_batch>` (alongside `predict`) and the runtime instead fuses the N per-lane observations into one batched observation and calls the model once for the whole vector.

```python
from rlmesh.torch import Model

class Policy(Model):
    def predict(self, obs):
        return self._forward(obs[None])[0]      # one lane

    def predict_batch(self, obs):
        return self._forward(obs)               # N lanes in one forward
```

The fused observation gives every leaf a leading batch axis, so a Dict observation arrives as `{key: array[N, ...]}`, the shape an RL or VLA stack already hands a policy. Return the batched action the same way; the runtime splits it back per lane. The engine prefers this corner for a vectorized route and falls back to per-lane `predict` when it is absent.

For a policy that emits an action chunk, {meth}`predict_chunk_batch <rlmesh._models.base.ModelBase.predict_chunk_batch>` is the batched chunk corner; the runtime splits the batch axis per lane and replays each chunk. The four corners and how the runtime derives the ones you skip are in {doc}`models`.

```{note}
The dependency-free `rlmesh.Model` over raw RLMesh values cannot fuse opaque
tensors, so it receives the per-lane list and returns one. Batched fusion is a
NumPy / Torch / JAX feature; pick a framework backend to use it.
```

## Fewer model calls per episode

A model that predicts an action chunk can execute several actions before re-planning. Pass `execution_horizon` to `run` or `session`, and the runtime calls the model once, executes that many actions of the returned chunk one per env step, then calls again.

```python
result = model.run(env, seeds=range(50), execution_horizon=8)
```

This cuts the number of model forwards per episode by the horizon, at the cost of acting open-loop between re-plans. It engages only when the model defines {meth}`predict_chunk <rlmesh._models.base.ModelBase.predict_chunk>`; requested on a model without one, it warns and runs un-chunked, so the default of `1` is always safe. The replay lives in the runtime, so one action still reaches the env per step. See {doc}`evaluation` for the end-to-end horizon behavior.

## Frame-stack overhead

A model that conditions on a short history declares `stack=N` on an image input. The env still sends one frame per step; RLMesh buffers the last `N` processed frames and emits them on a new leading axis.

```python
"image": adapt.Image(adapt.IMAGE_PRIMARY, size=224, stack=4)
```

The cost is local, not network. On the in-process run path the buffer is a host-side rolling deque of `N` processed frames per stacked input, cleared on reset; on the served path the core keeps the same buffer, keyed per episode. Either way nothing extra crosses the wire, so stacking trades a little host memory and a stack copy for keeping the network payload at one frame per step. The buffer holds the processed (resized, converted) frame, so its size follows the model's target resolution, not the camera's.

## Payload encoding

The adapter encodes only the observation keys the plan actually reads. An env that returns extra keys, or one unencodable key, does not pay for them and does not abort a step over them. This is automatic; there is no flag, and `describe()` shows which keys the plan touches (see {doc}`debugging`).

Values travel as framework-neutral bytes directed by the spec, so the env's framework and the model's framework are independent and neither forces a conversion on the other. The smaller you make the model's declared input (a single primary camera instead of three, a target resolution the policy needs rather than the camera's native one), the less there is to encode and move. That is a spec decision, made once.

## Device and framework placement

For a torch or JAX model, set `device` in {meth}`load <rlmesh._models.base.ModelBase.load>` alongside moving your weights. RLMesh moves every observation tensor leaf onto that device before `predict`, so you never call `.to(device)` in the hot path and there is one source of truth for placement.

```python
class Policy(Model):
    device = "cuda:0"

    def load(self):
        self.weights = load_checkpoint().to(self.device)
```

On the env side, {class}`~rlmesh.EnvServer` takes `framework=` (`"torch"` / `"jax"` / `"numpy"`) to type the obs/action seam, and `device=` to place the incoming action for a torch/jax env. `device=` requires a framework with a device; a numpy env or the default backend rejects it. Observations need no declaration, a torch/jax obs (GPU included) is auto-detected and encoded either way. See {doc}`serving-environments`.

## Vector endpoints

One endpoint can serve many env instances. {class}`~rlmesh.EnvServer` detects a vectorized env (one exposing `num_envs` and `single_*` spaces) and serves a vector endpoint automatically; connect to it with `RemoteVectorEnv`.

```python
import gymnasium as gym
import rlmesh

envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(4)])
rlmesh.EnvServer(envs, "127.0.0.1:5555").serve()
```

A vector endpoint plus a batched model corner is the fast combination: N lanes step together and one model forward covers them. The client side is in {doc}`remote-clients`.

```{caution}
A torch/jax env cannot be fanned out with gym vectorization (`num_envs > 1`):
that path concatenates observations with NumPy and discards the framework tensors,
so RLMesh rejects it. Serve scalar, or have a natively batched env return
`[N, ...]` tensors at `num_envs=1`.
```

## What is not a knob

Some things that look tunable are fixed, automatic, or not exposed in the Python API. Knowing which is which saves you looking for a setting that is not there.

| Looks like a knob                                 | Reality                                                                                                                      |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Per-step request timeout                          | Exists on the native client but not exposed through `RemoteEnv` / `RemoteVectorEnv` or `run`/`session`. Bound it at the env. |
| Connect timeout on the high-level client          | The native client accepts one; the public `RemoteEnv` does not pass it through today.                                        |
| Client-side retry / reconnect                     | Not done for you. A dropped session raises; re-dial a fresh client (see {doc}`error-handling`).                              |
| `predict_concurrency` (server pipelining)         | Present in the Rust serve options but not in the Python `ServeOptions` constructor.                                          |
| Adapter conversions (resize, normalize, encoding) | Correctness transforms the resolver chooses, not performance dials. Shrink the spec to do less work.                         |
| Frame-stack depth                                 | A model capability set by `stack=N`, sized by the model's needs, not a tuning parameter.                                     |

The lifecycle options that `ServeOptions` does expose, `idle_timeout_seconds`, `drain_timeout_seconds`, `close_timeout_seconds`, and `allow_remote_shutdown`, control shutdown behavior rather than throughput; they belong to the session lifecycle in {doc}`streaming`.

## Where next

- {doc}`models` — the four prediction corners and how the runtime fuses, splits, and derives them.
- {doc}`evaluation` — `execution_horizon` and the chunk-replay loop end to end.
- {doc}`serving-environments` — `framework=`, `device=`, and the vector serve path.
- {doc}`remote-clients` — vector clients and the connection surface.
- {doc}`adapters` — what the spec encodes and why a leaner spec moves less data.
